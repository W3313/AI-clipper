"""Reframing: turn a wide source into an animated vertical crop window.

The engine's shorts are vertical, the sources usually are not.  This module
works out *where* to put the crop window over time:

* :func:`center_crop` -- the deterministic, dependency-free baseline: the
  largest window of the requested aspect ratio that fits the source, centred.
* :func:`track` -- samples the media with OpenCV, finds a point of interest per
  sample (face first, motion energy second, frame centre last), smooths the
  resulting series and emits :class:`~aiclipper.models.CropKeyframe` objects.
* :func:`simplify` -- a Ramer-Douglas-Peucker pass that throws away keyframes
  the renderer can reconstruct by interpolation.

Contract with the renderer (hard requirement, do not "optimise" it away)
-----------------------------------------------------------------------
Every keyframe emitted by this module carries the **same** ``w``/``h``, and
both are **even**.  ``render.py`` turns a non-static :class:`CropPath` into a
``sendcmd`` script driving a single ``crop`` filter: ffmpeg's ``crop`` allocates
its output frame once, so an animated path may only pan (``x``/``y``), never
resize.  Odd dimensions additionally break ``yuv420p`` chroma subsampling.
``x``/``y`` are kept even too, for chroma alignment, and are always clamped so
the window lies entirely inside the source frame.

OpenCV is an optional extra.  It is imported lazily inside :func:`track`, and
when it is missing (or cannot open the file) :func:`track` transparently falls
back to :func:`center_crop` instead of raising -- reframing must never be the
reason a render fails.  Nothing here is random: the same media always yields the
same path.
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from .config import Settings
from .ffmpeg import probe
from .models import CropKeyframe, CropPath, MediaInfo

log = logging.getLogger(__name__)

__all__ = [
    "CASCADE_NAME",
    "DEADZONE_FRACTION",
    "DETECT_WIDTH",
    "center_crop",
    "simplify",
    "track",
]

#: Sampled frames are downscaled to at most this width before detection.
DETECT_WIDTH = 320

#: Bundled Haar cascade used for face detection.
CASCADE_NAME = "haarcascade_frontalface_default.xml"

#: Point-of-interest moves smaller than this fraction of the crop window are
#: ignored, so a nearly-still subject does not make the frame drift.
DEADZONE_FRACTION = 0.02

#: Per-pixel absolute difference below this is treated as sensor/codec noise.
MOTION_FLOOR = 12.0

#: Mean (post-floor) motion energy per pixel below this counts as "no motion".
MOTION_MIN_MEAN = 0.15

#: How strongly a face near the frame centre is preferred over a bigger one.
FACE_CENTRE_BIAS = 0.55

#: Hard cap on sampled frames, so a feature-length input cannot blow up memory.
MAX_SAMPLES = 4000

_CASCADE_CACHE: dict[str, Any] = {}


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #

def _even(value: float, maximum: int) -> int:
    """Round ``value`` to the nearest even integer within ``2..maximum``."""
    n = int(round(value / 2.0)) * 2
    return max(2, min(n, maximum - maximum % 2))


def _even_floor(value: float, maximum: int) -> int:
    """Largest even integer <= ``value``, clamped into ``0..maximum``.

    ``maximum`` is itself rounded down to even first: an odd ceiling (a crop
    window inside an odd-sized source) must never leak an odd offset out, or the
    even-``x``/``y`` guarantee this module makes to the renderer breaks.
    """
    ceiling = max(0, int(maximum))
    ceiling -= ceiling % 2
    n = int(math.floor(max(0.0, value)))
    n -= n % 2
    return max(0, min(n, ceiling))


def _copy(k: CropKeyframe) -> CropKeyframe:
    return CropKeyframe(float(k.t), int(k.x), int(k.y), int(k.w), int(k.h))


def _window_size(src_w: int, src_h: int, target_aspect: float) -> tuple[int, int]:
    """Largest even-sided ``w, h`` of ``target_aspect`` that fits the source."""
    if src_w / src_h >= target_aspect:
        # Source is wider than the target: keep full height, crop the sides.
        w, h = src_h * target_aspect, float(src_h)
    else:
        # Source is *narrower* than the target: keep full width, crop top/bottom.
        w, h = float(src_w), src_w / target_aspect
    return _even(w, src_w), _even(h, src_h)


# --------------------------------------------------------------------------- #
# baseline
# --------------------------------------------------------------------------- #

def _centred(src_w: int, src_h: int, target_aspect: float) -> CropPath:
    """The centred window for a frame of ``src_w`` x ``src_h`` actual pixels."""
    if not math.isfinite(target_aspect) or target_aspect <= 0:
        raise ValueError(f"target_aspect must be a positive, finite ratio, got {target_aspect!r}")

    src_w, src_h = max(0, int(src_w)), max(0, int(src_h))
    if src_w < 2 or src_h < 2:
        return CropPath(keyframes=[], source_width=src_w, source_height=src_h)

    w, h = _window_size(src_w, src_h, target_aspect)
    x = _even_floor((src_w - w) / 2.0, src_w - w)
    y = _even_floor((src_h - h) / 2.0, src_h - h)
    return CropPath.static(x, y, w, h, src_w, src_h)


def center_crop(info: MediaInfo, target_aspect: float = 9 / 16) -> CropPath:
    """Static, centred crop of ``target_aspect`` -- the always-available path.

    The window is the largest one of the requested ratio that fits inside the
    source, with even width/height (and even offsets).  A source that is already
    *narrower* than the target keeps its full width and loses height instead.

    A source with no usable video dimensions yields a keyframe-less
    :class:`CropPath`, which the renderer reads as "no crop".

    The dimensions come from :func:`aiclipper.ffmpeg.probe`, i.e. the *coded*
    ones: a clip carrying a 90 degree display matrix reports them un-rotated,
    while every decoder downstream hands out rotated frames.  :func:`track`
    decodes, so it corrects for that; this function cannot.
    """
    return _centred(int(info.width), int(info.height), target_aspect)


# --------------------------------------------------------------------------- #
# simplification
# --------------------------------------------------------------------------- #

def simplify(path: CropPath, tolerance: float = 8.0) -> CropPath:
    """Drop keyframes the renderer can reconstruct by linear interpolation.

    A Ramer-Douglas-Peucker pass over the ``(t, x, y)`` polyline: a keyframe
    survives only if dropping it would move the interpolated window by more than
    ``tolerance`` pixels in x or y.  The first and last keyframes are always
    kept, and ``w``/``h`` are untouched (they are constant by contract).
    """
    kfs = list(path.keyframes)
    if len(kfs) <= 2 or tolerance <= 0:
        return CropPath([_copy(k) for k in kfs], path.source_width, path.source_height)

    keep = [False] * len(kfs)
    keep[0] = keep[-1] = True
    stack: list[tuple[int, int]] = [(0, len(kfs) - 1)]
    while stack:
        lo, hi = stack.pop()
        if hi - lo < 2:
            continue
        a, b = kfs[lo], kfs[hi]
        span = b.t - a.t
        worst, worst_i = -1.0, -1
        for i in range(lo + 1, hi):
            k = kfs[i]
            f = min(1.0, max(0.0, (k.t - a.t) / span)) if abs(span) > 1e-9 else 0.0
            dev = max(abs(k.x - (a.x + (b.x - a.x) * f)), abs(k.y - (a.y + (b.y - a.y) * f)))
            if dev > worst:
                worst, worst_i = dev, i
        if worst > tolerance and worst_i > lo:
            keep[worst_i] = True
            stack.append((lo, worst_i))
            stack.append((worst_i, hi))

    return CropPath(
        [_copy(k) for k, wanted in zip(kfs, keep, strict=True) if wanted],
        path.source_width,
        path.source_height,
    )


# --------------------------------------------------------------------------- #
# signal helpers
# --------------------------------------------------------------------------- #

def _smooth_series(values: Sequence[float], window: int) -> list[float]:
    """Centred moving average; the window shrinks at both ends."""
    n = len(values)
    if n == 0 or window <= 1:
        return [float(v) for v in values]
    half = window // 2
    out: list[float] = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        out.append(sum(float(v) for v in values[lo:hi]) / (hi - lo))
    return out


def _apply_deadzone(values: Sequence[float], deadzone: float) -> list[float]:
    """Hold the current value until the input moves more than ``deadzone``.

    Once the threshold is crossed the held value follows the input, offset by
    exactly ``deadzone``, so the output keeps the direction of travel instead of
    snapping.
    """
    if not values:
        return []
    if deadzone <= 0:
        return [float(v) for v in values]
    held = float(values[0])
    out = [held]
    for value in values[1:]:
        delta = float(value) - held
        if abs(delta) > deadzone:
            held = float(value) - math.copysign(deadzone, delta)
        out.append(held)
    return out


# --------------------------------------------------------------------------- #
# OpenCV plumbing (all lazy)
# --------------------------------------------------------------------------- #

def _load_cv2() -> Any | None:
    """Import OpenCV, or return ``None`` when it is unavailable."""
    try:
        import cv2  # noqa: PLC0415 - optional extra, imported at call time
    except Exception as exc:  # pragma: no cover - depends on the environment
        log.debug("opencv unavailable (%s); falling back to a centred crop", exc)
        return None
    return cv2


def _face_detector(cv2mod: Any) -> Any | None:
    """The bundled Haar cascade classifier, or ``None`` if it is not shipped.

    OpenCV >= 5.0 dropped ``CascadeClassifier`` and the ``cv2.data`` cascade
    XMLs, so this legitimately returns ``None`` on new builds; tracking then
    relies on motion energy.
    """
    key = getattr(cv2mod, "__version__", "?")
    if key in _CASCADE_CACHE:
        return _CASCADE_CACHE[key]

    detector = None
    factory = getattr(cv2mod, "CascadeClassifier", None)
    data_dir = getattr(getattr(cv2mod, "data", None), "haarcascades", "")
    if factory is not None and data_dir:
        xml = os.path.join(data_dir, CASCADE_NAME)
        if os.path.exists(xml):
            try:
                candidate = factory(xml)
                if not candidate.empty():
                    detector = candidate
            except Exception as exc:  # pragma: no cover - corrupt install
                log.debug("could not load %s: %s", xml, exc)
    if detector is None:
        log.debug("no Haar cascade available; face tracking disabled")
    _CASCADE_CACHE[key] = detector
    return detector


def _face_boxes(detector: Any, gray: Any, min_size: int) -> list[tuple[int, int, int, int]]:
    try:
        found = detector.detectMultiScale(
            gray, scaleFactor=1.15, minNeighbors=5, minSize=(min_size, min_size)
        )
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("face detection failed: %s", exc)
        return []
    return [(int(b[0]), int(b[1]), int(b[2]), int(b[3])) for b in found]


def _pick_face(
    boxes: Sequence[tuple[int, int, int, int]], frame_w: int, frame_h: int
) -> tuple[float, float] | None:
    """Centre of the best face: big wins, and near the frame centre wins more."""
    best_score, best_point = 0.0, None
    half_w, half_h = frame_w / 2.0, frame_h / 2.0
    for x, y, w, h in boxes:
        if w <= 0 or h <= 0:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        dist = math.hypot((cx - half_w) / max(1.0, half_w), (cy - half_h) / max(1.0, half_h))
        centrality = max(0.0, 1.0 - FACE_CENTRE_BIAS * min(1.0, dist / math.sqrt(2.0)))
        score = float(w * h) * centrality
        if score > best_score:
            best_score, best_point = score, (cx, cy)
    return best_point


def _motion_point(cv2mod: Any, gray: Any, prev_gray: Any) -> tuple[float, float] | None:
    """Centre of mass of the frame-to-frame difference, or ``None`` if still."""
    import numpy as np  # noqa: PLC0415 - base dependency, kept out of import time

    if prev_gray is None or prev_gray.shape != gray.shape:
        return None
    diff = cv2mod.absdiff(gray, prev_gray)
    diff = cv2mod.GaussianBlur(diff, (0, 0), 3.0)
    energy = diff.astype(np.float32)
    energy[energy < MOTION_FLOOR] = 0.0
    total = float(energy.sum())
    if total <= 0.0 or total / float(energy.size) < MOTION_MIN_MEAN:
        return None
    cols = energy.sum(axis=0)
    rows = energy.sum(axis=1)
    cx = float((cols * np.arange(cols.size, dtype=np.float32)).sum() / total)
    cy = float((rows * np.arange(rows.size, dtype=np.float32)).sum() / total)
    return cx, cy


def _to_gray(cv2mod: Any, frame: Any) -> tuple[Any, float]:
    """Downscale + greyscale one sampled frame.  Returns ``(gray, scale)``."""
    height, width = int(frame.shape[0]), int(frame.shape[1])
    scale = min(1.0, DETECT_WIDTH / float(width)) if width > 0 else 1.0
    small = frame
    if scale < 1.0:
        small = cv2mod.resize(
            frame,
            (max(2, int(round(width * scale))), max(2, int(round(height * scale)))),
            interpolation=cv2mod.INTER_AREA,
        )
    gray = small if getattr(small, "ndim", 2) == 2 else cv2mod.cvtColor(small, cv2mod.COLOR_BGR2GRAY)
    return gray, (float(gray.shape[1]) / float(width) if width else 1.0)


def _open_capture(cv2mod: Any, media: Path) -> Any | None:
    try:
        cap = cv2mod.VideoCapture(str(media))
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("VideoCapture(%s) raised %s", media, exc)
        return None
    if cap is None or not cap.isOpened():
        if cap is not None:
            cap.release()
        return None
    return cap


def _iter_samples(
    cv2mod: Any, cap: Any, *, start: float, end: float, sample_fps: float, fallback_fps: float
) -> Iterator[tuple[float, Any]]:
    """Yield ``(t, frame)`` roughly every ``1 / sample_fps`` seconds.

    Frames between samples are ``grab``-ed but never decoded, which keeps long
    sources cheap.
    """
    fps = float(cap.get(cv2mod.CAP_PROP_FPS) or 0.0)
    if not math.isfinite(fps) or fps <= 0.0:
        fps = fallback_fps if fallback_fps > 0 else 25.0
    step = 1.0 / max(0.05, sample_fps)

    if start > 1e-3:
        cap.set(cv2mod.CAP_PROP_POS_MSEC, start * 1000.0)

    next_t = start
    index = 0
    emitted = 0
    while emitted < MAX_SAMPLES:
        if not cap.grab():
            return
        # POS_MSEC is the presentation time of the frame just grabbed (verified
        # against known content); it is 0.0 both for the very first frame and
        # for backends that do not track position, hence the frame counter.
        raw = float(cap.get(cv2mod.CAP_PROP_POS_MSEC) or 0.0)
        t = max(0.0, raw / 1000.0) if raw > 0.0 else start + index / fps
        index += 1
        if t > end + 1e-6:
            return
        if t + 1e-6 < next_t:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            return
        yield t, frame
        emitted += 1
        next_t = max(next_t + step, t + step * 0.5)
    log.debug("stopped sampling at %d frames (MAX_SAMPLES)", MAX_SAMPLES)


# --------------------------------------------------------------------------- #
# tracking
# --------------------------------------------------------------------------- #

def track(
    media: str | Path,
    *,
    target_aspect: float = 9 / 16,
    settings: Settings | None = None,
    sample_fps: float = 4.0,
    smooth_seconds: float = 1.2,
    start: float = 0.0,
    end: float | None = None,
) -> CropPath:
    """Follow the action and return an animated :class:`CropPath`.

    Frames are sampled at ``sample_fps`` between ``start`` and ``end`` (source
    time, seconds).  Each sample is downscaled to :data:`DETECT_WIDTH` and gives
    one point of interest:

    1. the largest / most central face found by the bundled Haar cascade,
    2. otherwise the centre of mass of the motion energy against the previous
       sampled frame,
    3. otherwise the frame centre.

    The series is smoothed with a ``smooth_seconds`` moving average, passed
    through a deadzone (:data:`DEADZONE_FRACTION` of the window width) so a
    nearly-still subject does not cause drift, then converted into crop windows
    that are clamped inside the frame.  Keyframe ``t`` values are in *source*
    time.  All windows share the size returned by :func:`center_crop`.

    Falls back to :func:`center_crop` -- without raising -- when OpenCV is
    missing, when the file cannot be opened, or when no frame could be sampled.
    """
    path = Path(media)
    info = probe(path, settings=settings)
    base = center_crop(info, target_aspect)
    if not base.keyframes:
        return base

    win = base.keyframes[0]
    src_w, src_h = base.source_width, base.source_height
    max_x, max_y = src_w - win.w, src_h - win.h

    duration = info.duration if info.duration > 0 else float("inf")
    lo = max(0.0, float(start))
    hi = duration if end is None else min(float(end), duration)
    if not (hi > lo):
        return base

    cv2mod = _load_cv2()
    if cv2mod is None:
        return base
    cap = _open_capture(cv2mod, path)
    if cap is None:
        log.debug("could not open %s with OpenCV; using a centred crop", path)
        return base

    times: list[float] = []
    xs: list[float] = []
    ys: list[float] = []
    frame_size: tuple[int, int] | None = None
    detector = _face_detector(cv2mod)
    prev_gray = None
    try:
        for t, frame in _iter_samples(
            cv2mod, cap, start=lo, end=hi, sample_fps=sample_fps, fallback_fps=info.fps
        ):
            frame_size = (int(frame.shape[1]), int(frame.shape[0]))
            gray, scale = _to_gray(cv2mod, frame)
            height, width = int(gray.shape[0]), int(gray.shape[1])
            point = None
            if detector is not None:
                boxes = _face_boxes(detector, gray, max(12, width // 12))
                point = _pick_face(boxes, width, height)
            if point is None:
                point = _motion_point(cv2mod, gray, prev_gray)
            if point is None:
                point = (width / 2.0, height / 2.0)
            prev_gray = gray
            times.append(t)
            xs.append(point[0] / scale if scale else point[0])
            ys.append(point[1] / scale if scale else point[1])
    except Exception:
        # A decode/detection failure part-way through must not fail a render:
        # keep the samples collected so far (logged, never swallowed silently).
        log.warning("sampling %s stopped early after %d frames", path, len(times), exc_info=True)
    finally:
        cap.release()

    if not times:
        log.debug("no frames sampled from %s; using a centred crop", path)
        return base

    if frame_size is not None and frame_size != (src_w, src_h):
        # ffprobe reports *coded* dimensions; a display matrix (phone footage
        # shot sideways) means the decoder -- OpenCV here, and equally ffmpeg in
        # render.py -- hands out rotated frames.  The points below are in decoded
        # coordinates, so the window has to be too, or the emitted crop falls
        # outside the frame the renderer will actually see.
        log.debug("%s decodes as %dx%d, not the probed %dx%d", path, *frame_size, src_w, src_h)
        base = _centred(frame_size[0], frame_size[1], target_aspect)
        if not base.keyframes:
            return base
        win = base.keyframes[0]
        src_w, src_h = base.source_width, base.source_height
        max_x, max_y = src_w - win.w, src_h - win.h

    window = max(1, int(round(max(0.0, smooth_seconds) * max(0.05, sample_fps))))
    if window % 2 == 0:
        window += 1
    xs = _apply_deadzone(_smooth_series(xs, window), DEADZONE_FRACTION * win.w)
    ys = _apply_deadzone(_smooth_series(ys, window), DEADZONE_FRACTION * win.h)
    # Anchor the path at the start of the requested window so the renderer has a
    # command at t=start -- but never at the cost of monotonic keyframe times.
    if len(times) < 2 or lo < times[1]:
        times[0] = lo

    keyframes = [
        CropKeyframe(
            t=t,
            x=_even_floor(round(cx - win.w / 2.0), max_x),
            y=_even_floor(round(cy - win.h / 2.0), max_y),
            w=win.w,
            h=win.h,
        )
        for t, cx, cy in zip(times, xs, ys, strict=True)
    ]
    return CropPath(keyframes=keyframes, source_width=src_w, source_height=src_h)
