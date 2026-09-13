"""Tests for :mod:`aiclipper.crop`.

The moving-subject fixtures are built on the fly with ffmpeg's lavfi sources: a
bright box translating left-to-right over a dark background, so the "correct"
answer for the tracker is known in closed form.  ``drawbox`` evaluates its
geometry once at init in ffmpeg 6.x, so the box is composited with ``overlay``
(``eval=frame``), which really does re-evaluate the ``t`` expression per frame.
"""

from __future__ import annotations

import statistics
import sys
from itertools import pairwise
from pathlib import Path

import pytest

from aiclipper import crop
from aiclipper import ffmpeg as ff
from aiclipper.models import CropKeyframe, CropPath, MediaInfo

VERTICAL = 9 / 16
LANDSCAPE = 16 / 9

CLIP_W, CLIP_H = 320, 180
CLIP_FPS = 25
CLIP_SECONDS = 3.0
BOX = 40

#: The bright box's left edge, in ffmpeg expression form and as Python.
BOX_X_EXPR = "30+70*t"
BOX_Y = 70


def _box_center_x(t: float) -> float:
    return 30.0 + 70.0 * t + BOX / 2.0


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

def _render_clip(dest: Path, x_expr: str, *, seconds: float = CLIP_SECONDS) -> Path:
    ff.run_ffmpeg(
        [
            "-y",
            "-f", "lavfi", "-i", f"color=c=black:s={CLIP_W}x{CLIP_H}:r={CLIP_FPS}:d={seconds}",
            "-f", "lavfi", "-i", f"color=c=white:s={BOX}x{BOX}:r={CLIP_FPS}:d={seconds}",
            "-filter_complex", f"[0][1]overlay=x='{x_expr}':y={BOX_Y}:eval=frame",
            "-t", str(seconds), "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(dest),
        ]
    )
    return dest


@pytest.fixture(scope="session")
def clip_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    if not ff.have_ffmpeg():
        pytest.skip("ffmpeg is not installed")
    return tmp_path_factory.mktemp("crop-clips")


@pytest.fixture(scope="session")
def box_clip(clip_dir: Path) -> Path:
    """A white box sliding steadily from x=30 to x=240 over three seconds."""
    return _render_clip(clip_dir / "box.mp4", BOX_X_EXPR)


@pytest.fixture(scope="session")
def jitter_clip(clip_dir: Path) -> Path:
    """The same slide, plus a 5 Hz wobble that aliases badly at 4 samples/s."""
    return _render_clip(clip_dir / "jitter.mp4", "30+55*t+18*sin(2*PI*t*5)")


@pytest.fixture(scope="session")
def flat_clip(clip_dir: Path) -> Path:
    """A completely static frame: no face, no motion, nothing to follow."""
    dest = clip_dir / "flat.mp4"
    ff.run_ffmpeg(
        ["-y", "-f", "lavfi", "-i", f"color=c=0x203040:s={CLIP_W}x{CLIP_H}:r={CLIP_FPS}:d=2",
         "-t", "2", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p", str(dest)]
    )
    return dest


class _BrightBoxDetector:
    """Stand-in for the Haar cascade: reports the bright box as a "face".

    OpenCV >= 5 ships neither ``CascadeClassifier`` nor the cascade XMLs, so the
    real detector cannot be exercised everywhere; this keeps the face branch of
    :func:`aiclipper.crop.track` under test with the same call signature.
    """

    def __init__(self) -> None:
        self.calls = 0

    def detectMultiScale(self, gray, **kwargs):  # noqa: N802 - mirrors the cv2 API
        import numpy as np

        self.calls += 1
        ys, xs = np.nonzero(np.asarray(gray) > 160)
        if xs.size == 0:
            return []
        return [(int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))]


def _info(width: int, height: int, *, duration: float = 3.0) -> MediaInfo:
    return MediaInfo(path=Path("memory.mp4"), duration=duration, width=width, height=height,
                     fps=25.0, has_video=width > 0)


def _linear_path(n: int = 41, *, noise: tuple[int, ...] = (0, 1, -1, 2, -2)) -> CropPath:
    """A straight pan from x=0 to x=200 with sub-tolerance wobble on top."""
    kfs = []
    for i in range(n):
        x = round(i * 200 / (n - 1)) + noise[i % len(noise)]
        kfs.append(CropKeyframe(t=i * 0.25, x=max(0, x), y=0, w=102, h=180))
    return CropPath(kfs, CLIP_W, CLIP_H)


# --------------------------------------------------------------------------- #
# center_crop
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("width", "height", "aspect", "expect_w", "expect_h"),
    [
        (1920, 1080, VERTICAL, 608, 1080),    # wide source: full height, crop sides
        (1080, 1920, VERTICAL, 1080, 1920),   # already vertical: untouched
        (400, 400, VERTICAL, 224, 400),       # square -> vertical
        (400, 400, LANDSCAPE, 400, 224),      # source narrower than target: crop height
        (321, 181, VERTICAL, 102, 180),       # odd dimensions round down to even
        (1921, 1081, LANDSCAPE, 1920, 1080),
    ],
)
def test_center_crop_geometry(width, height, aspect, expect_w, expect_h):
    path = crop.center_crop(_info(width, height), aspect)
    assert path.is_static
    (window,) = path.keyframes
    assert (window.w, window.h) == (expect_w, expect_h)
    # Even, inside the frame, centred to within a pixel, and the right shape.
    assert window.w % 2 == 0 and window.h % 2 == 0
    assert window.x % 2 == 0 and window.y % 2 == 0
    assert 0 <= window.x and window.x + window.w <= width
    assert 0 <= window.y and window.y + window.h <= height
    # Even offsets can sit up to two pixels off an odd centre.
    assert abs(window.x - (width - window.w) / 2) <= 2
    assert abs(window.y - (height - window.h) / 2) <= 2
    # Whichever side was cropped lands within a pixel of the exact ratio.
    assert min(abs(window.w - window.h * aspect), abs(window.h - window.w / aspect)) <= 1.0
    assert (path.source_width, path.source_height) == (width, height)


def test_center_crop_is_the_largest_window_that_fits():
    window = crop.center_crop(_info(1920, 1080), VERTICAL).keyframes[0]
    # One step bigger in either direction would no longer fit the target ratio.
    assert window.h == 1080
    assert (window.w + 2) / window.h > VERTICAL


def test_center_crop_without_video_dimensions_is_a_no_op():
    path = crop.center_crop(_info(0, 0), VERTICAL)
    assert path.keyframes == []
    assert path.size == (0, 0)
    assert path.is_static
    assert crop.center_crop(_info(1, 1), VERTICAL).keyframes == []


@pytest.mark.parametrize("aspect", [0.0, -1.0, float("nan"), float("inf")])
def test_center_crop_rejects_impossible_aspect(aspect):
    with pytest.raises(ValueError):
        crop.center_crop(_info(1920, 1080), aspect)


# --------------------------------------------------------------------------- #
# simplify
# --------------------------------------------------------------------------- #

def test_simplify_drops_redundant_keyframes_and_keeps_endpoints():
    path = _linear_path()
    result = crop.simplify(path, tolerance=8.0)

    assert len(result.keyframes) < len(path.keyframes) / 4
    assert result.keyframes[0] == path.keyframes[0]
    assert result.keyframes[-1] == path.keyframes[-1]
    assert (result.source_width, result.source_height) == (path.source_width, path.source_height)
    assert {(k.w, k.h) for k in result.keyframes} == {(102, 180)}


def test_simplify_stays_within_tolerance_of_the_original():
    path = _linear_path()
    result = crop.simplify(path, tolerance=8.0)
    # +1 px of slack: CropPath.at() rounds its interpolation to whole pixels.
    worst = max(abs(result.at(k.t).x - k.x) for k in path.keyframes)
    assert worst <= 8.0 + 1.0


def test_simplify_keeps_a_real_turn():
    kfs = [CropKeyframe(t=i * 0.25, x=0, y=0, w=102, h=180) for i in range(5)]
    kfs += [CropKeyframe(t=1.25, x=120, y=0, w=102, h=180)]
    kfs += [CropKeyframe(t=1.5 + i * 0.25, x=0, y=0, w=102, h=180) for i in range(5)]
    result = crop.simplify(CropPath(kfs, CLIP_W, CLIP_H), tolerance=8.0)
    assert 120 in [k.x for k in result.keyframes]
    assert len(result.keyframes) < len(kfs)


def test_simplify_tracks_vertical_movement_too():
    kfs = [CropKeyframe(t=i * 0.25, x=10, y=0 if i != 3 else 90, w=102, h=120) for i in range(7)]
    result = crop.simplify(CropPath(kfs, CLIP_W, CLIP_H), tolerance=8.0)
    assert 90 in [k.y for k in result.keyframes]


@pytest.mark.parametrize("tolerance", [0.0, -4.0])
def test_simplify_with_no_tolerance_keeps_everything(tolerance):
    path = _linear_path(9)
    result = crop.simplify(path, tolerance=tolerance)
    assert [(k.t, k.x) for k in result.keyframes] == [(k.t, k.x) for k in path.keyframes]


def test_simplify_copies_rather_than_aliases():
    path = _linear_path(2)
    result = crop.simplify(path, tolerance=8.0)
    assert len(result.keyframes) == 2
    result.keyframes[0].x = 999
    assert path.keyframes[0].x != 999


def test_simplify_handles_empty_and_single_keyframe_paths():
    assert crop.simplify(CropPath([], 320, 180)).keyframes == []
    one = CropPath([CropKeyframe(0.0, 4, 6, 102, 180)], 320, 180)
    assert crop.simplify(one).keyframes == one.keyframes


# --------------------------------------------------------------------------- #
# signal helpers
# --------------------------------------------------------------------------- #

def test_smooth_series_reduces_jitter_and_preserves_length():
    noisy = [0.0, 20.0] * 12
    smoothed = crop._smooth_series(noisy, 5)
    assert len(smoothed) == len(noisy)
    assert statistics.pvariance(smoothed) < statistics.pvariance(noisy) / 4
    # The mean survives; only the wobble goes away.
    assert abs(statistics.fmean(smoothed) - statistics.fmean(noisy)) < 1.5


def test_smooth_series_tracks_a_ramp_without_lagging_much():
    ramp = [float(i) for i in range(20)]
    smoothed = crop._smooth_series(ramp, 5)
    assert max(abs(a - b) for a, b in zip(ramp, smoothed, strict=True)) <= 1.5


@pytest.mark.parametrize("window", [0, 1])
def test_smooth_series_passes_through_for_tiny_windows(window):
    assert crop._smooth_series([1.0, 9.0, 2.0], window) == [1.0, 9.0, 2.0]


def test_deadzone_ignores_small_moves_but_follows_real_ones():
    held = crop._apply_deadzone([100.0, 102.0, 98.0, 101.0], 5.0)
    assert held == [100.0] * 4

    moved = crop._apply_deadzone([100.0, 140.0], 5.0)
    assert moved[1] == pytest.approx(135.0)  # follows, trailing by exactly the deadzone

    assert crop._apply_deadzone([1.0, 50.0], 0.0) == [1.0, 50.0]
    assert crop._apply_deadzone([], 4.0) == []


def test_pick_face_prefers_the_big_central_one():
    boxes = [(0, 0, 20, 20), (140, 70, 60, 60), (300, 160, 18, 18)]
    point = crop._pick_face(boxes, CLIP_W, CLIP_H)
    assert point == pytest.approx((170.0, 100.0))

    # A slightly smaller but far more central face beats a big one in the corner.
    point = crop._pick_face([(0, 0, 70, 70), (140, 60, 62, 62)], CLIP_W, CLIP_H)
    assert point[0] == pytest.approx(171.0)

    assert crop._pick_face([], CLIP_W, CLIP_H) is None
    assert crop._pick_face([(0, 0, 0, 0)], CLIP_W, CLIP_H) is None


# --------------------------------------------------------------------------- #
# track
# --------------------------------------------------------------------------- #

def _assert_renderable(path: CropPath, *, source=(CLIP_W, CLIP_H)):
    """The renderer's hard requirements: one even size, always inside frame."""
    assert path.keyframes
    sizes = {(k.w, k.h) for k in path.keyframes}
    assert len(sizes) == 1, "an animated crop may pan but never resize"
    (w, h) = sizes.pop()
    assert w % 2 == 0 and h % 2 == 0
    assert (path.source_width, path.source_height) == source
    for k in path.keyframes:
        assert 0 <= k.x <= source[0] - w
        assert 0 <= k.y <= source[1] - h
        assert k.x % 2 == 0 and k.y % 2 == 0
    assert [k.t for k in path.keyframes] == sorted(k.t for k in path.keyframes)


@pytest.mark.needs_ffmpeg
def test_track_follows_the_moving_box(box_clip: Path):
    path = crop.track(box_clip, target_aspect=VERTICAL, sample_fps=8.0)
    _assert_renderable(path)

    xs = [k.x for k in path.keyframes]
    assert len(xs) >= 8
    assert xs[-1] > xs[0] + 60, "the window should end well right of where it started"

    # Monotonic-ish: the tracker may hesitate, but it must not wander backwards.
    steps = [b - a for a, b in pairwise(xs)]
    assert sum(1 for s in steps if s >= 0) >= 0.8 * len(steps)
    assert min(steps) > -8


@pytest.mark.needs_ffmpeg
def test_track_window_matches_center_crop_size(box_clip: Path):
    info = ff.probe(box_clip)
    still = crop.center_crop(info, VERTICAL)
    path = crop.track(box_clip, target_aspect=VERTICAL, sample_fps=6.0)
    assert path.size == still.size
    assert not path.is_static


@pytest.mark.needs_ffmpeg
def test_track_smoothing_reduces_jitter(jitter_clip: Path, monkeypatch: pytest.MonkeyPatch):
    # Isolate smoothing: the deadzone would suppress wobble on its own.
    monkeypatch.setattr(crop, "DEADZONE_FRACTION", 0.0)
    raw = crop.track(jitter_clip, sample_fps=4.0, smooth_seconds=0.0)
    smoothed = crop.track(jitter_clip, sample_fps=4.0, smooth_seconds=1.2)

    assert len(raw.keyframes) == len(smoothed.keyframes)

    def wobble(path: CropPath) -> float:
        # Skip sample 0: it has no previous frame, so it is the centre fallback.
        xs = [k.x for k in path.keyframes][1:]
        return statistics.pvariance([b - a for a, b in pairwise(xs)])

    assert wobble(smoothed) < wobble(raw) * 0.9
    _assert_renderable(smoothed)


@pytest.mark.needs_ffmpeg
def test_track_deadzone_suppresses_small_drift(box_clip: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(crop, "DEADZONE_FRACTION", 0.0)
    loose = crop.track(box_clip, sample_fps=8.0, smooth_seconds=0.4)
    monkeypatch.setattr(crop, "DEADZONE_FRACTION", 0.5)
    tight = crop.track(box_clip, sample_fps=8.0, smooth_seconds=0.4)
    assert len({k.x for k in tight.keyframes}) < len({k.x for k in loose.keyframes})


@pytest.mark.needs_ffmpeg
def test_track_on_a_static_scene_stays_centred(flat_clip: Path):
    path = crop.track(flat_clip, sample_fps=6.0)
    still = crop.center_crop(ff.probe(flat_clip), VERTICAL)
    assert path.is_static
    assert (path.keyframes[0].x, path.keyframes[0].y) == (still.keyframes[0].x, still.keyframes[0].y)


@pytest.mark.needs_ffmpeg
def test_track_honours_the_time_window(box_clip: Path):
    path = crop.track(box_clip, sample_fps=8.0, start=1.0, end=2.0)
    times = [k.t for k in path.keyframes]
    assert times[0] == pytest.approx(1.0)
    assert 1.0 <= min(times) and max(times) <= 2.0 + 1e-6
    assert 4 <= len(times) <= 12

    inverted = crop.track(box_clip, sample_fps=8.0, start=2.0, end=1.0)
    assert inverted.is_static and inverted.keyframes == crop.center_crop(ff.probe(box_clip)).keyframes


@pytest.mark.needs_ffmpeg
def test_track_uses_faces_when_a_detector_is_available(
    box_clip: Path, monkeypatch: pytest.MonkeyPatch
):
    detector = _BrightBoxDetector()
    monkeypatch.setattr(crop, "_face_detector", lambda cv2mod: detector)
    monkeypatch.setattr(crop, "DEADZONE_FRACTION", 0.0)

    path = crop.track(box_clip, sample_fps=8.0, smooth_seconds=0.0)
    _assert_renderable(path)
    assert detector.calls >= len(path.keyframes)

    w, _ = path.size
    checked = 0
    for k in path.keyframes:
        want = _box_center_x(k.t)
        if not (w / 2 + 4 <= want <= CLIP_W - w / 2 - 4):
            continue  # the window is clamped against the frame edge here
        assert abs((k.x + w / 2) - want) <= 10, f"t={k.t}"
        checked += 1
    assert checked >= 6


@pytest.mark.needs_ffmpeg
def test_track_without_opencv_falls_back_to_center_crop(box_clip: Path, monkeypatch: pytest.MonkeyPatch):
    # Poisoning sys.modules makes `import cv2` raise, exercising the real guard.
    monkeypatch.setitem(sys.modules, "cv2", None)
    assert crop._load_cv2() is None

    path = crop.track(box_clip, target_aspect=VERTICAL)
    assert path.is_static
    assert path.keyframes == crop.center_crop(ff.probe(box_clip), VERTICAL).keyframes


@pytest.mark.needs_ffmpeg
def test_track_falls_back_when_the_file_cannot_be_opened(box_clip: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(crop, "_open_capture", lambda cv2mod, media: None)
    path = crop.track(box_clip, target_aspect=VERTICAL)
    assert path.is_static
    assert path.keyframes == crop.center_crop(ff.probe(box_clip), VERTICAL).keyframes


@pytest.mark.needs_ffmpeg
def test_track_on_audio_only_media_returns_an_empty_path(tmp_path: Path):
    tone = ff.make_tone(0.5, tmp_path / "tone.wav")
    assert crop.track(tone).keyframes == []


@pytest.mark.needs_ffmpeg
def test_track_output_survives_simplification(box_clip: Path):
    path = crop.track(box_clip, sample_fps=8.0)
    small = crop.simplify(path, tolerance=6.0)
    assert len(small.keyframes) < len(path.keyframes)
    _assert_renderable(small)
    assert max(abs(small.at(k.t).x - k.x) for k in path.keyframes) <= 7.0


# --------------------------------------------------------------------------- #
# invariants the renderer depends on
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("value", "maximum"),
    [(300.0, 219), (218.0, 219), (7.0, 7), (1.0, 1), (5.0, 0), (-4.0, 9), (3.9, 100)],
)
def test_even_floor_never_leaks_an_odd_offset(value, maximum):
    """An odd ceiling (an odd-sized source) must not produce an odd x/y."""
    got = crop._even_floor(value, maximum)
    assert got % 2 == 0
    assert 0 <= got <= max(0, maximum)
    assert got <= max(0.0, value) + 1e-9


@pytest.mark.parametrize(
    ("width", "height"),
    [(321, 181), (641, 361), (101, 57), (1919, 1079), (3, 3), (5, 2), (2, 3), (33, 1081)],
)
@pytest.mark.parametrize("aspect", [VERTICAL, LANDSCAPE, 1.0])
def test_center_crop_invariants_hold_for_awkward_sizes(width, height, aspect):
    path = crop.center_crop(_info(width, height), aspect)
    (window,) = path.keyframes
    assert window.w % 2 == 0 and window.h % 2 == 0
    assert window.x % 2 == 0 and window.y % 2 == 0
    assert window.w >= 2 and window.h >= 2
    assert 0 <= window.x and window.x + window.w <= width
    assert 0 <= window.y and window.y + window.h <= height


@pytest.mark.needs_ffmpeg
def test_track_keyframe_times_match_the_frame_content(box_clip: Path):
    """Sample timestamps must be the *grabbed* frame's time, not a frame off.

    The box's left edge encodes the true time, so a systematic +-1/fps error in
    the sampler shows up here as a consistent bias.
    """
    cv2 = pytest.importorskip("cv2")
    numpy = pytest.importorskip("numpy")

    cap = cv2.VideoCapture(str(box_clip))
    assert cap.isOpened()
    errors = []
    try:
        for t, frame in crop._iter_samples(
            cv2, cap, start=0.0, end=CLIP_SECONDS, sample_fps=8.0, fallback_fps=CLIP_FPS
        ):
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            _, xs = numpy.nonzero(gray > 160)
            # left = floor(30 + 70t), so this underestimates t by < 1/70 s.
            errors.append(t - (int(xs.min()) - 30) / 70.0)
    finally:
        cap.release()

    assert len(errors) >= 20
    frame_period = 1.0 / CLIP_FPS
    assert max(abs(e) for e in errors) < frame_period, errors[:5]
    # And no systematic lead/lag: the mean error is pure pixel quantisation.
    assert abs(statistics.fmean(errors)) < 0.5 * frame_period


@pytest.mark.needs_ffmpeg
def test_track_is_deterministic(box_clip: Path):
    first = crop.track(box_clip, sample_fps=8.0)
    second = crop.track(box_clip, sample_fps=8.0)
    assert first.keyframes == second.keyframes


@pytest.mark.needs_ffmpeg
def test_track_survives_a_failure_part_way_through_sampling(box_clip: Path, monkeypatch):
    """A decode/detect blow-up must degrade, not propagate: renders come first."""
    real = crop._to_gray
    state = {"n": 0, "released": False}

    def flaky(cv2mod, frame):
        state["n"] += 1
        if state["n"] > 4:
            raise RuntimeError("simulated decode failure")
        return real(cv2mod, frame)

    real_open = crop._open_capture

    class _ReleaseWatcher:
        """Forwards every call to the real capture; only notes ``release()``."""

        def __init__(self, cap):
            self._cap = cap

        def __getattr__(self, name):
            return getattr(self._cap, name)

        def release(self):
            state["released"] = True
            self._cap.release()

    def watched_open(cv2mod, media):
        cap = real_open(cv2mod, media)
        return None if cap is None else _ReleaseWatcher(cap)

    monkeypatch.setattr(crop, "_to_gray", flaky)
    monkeypatch.setattr(crop, "_open_capture", watched_open)

    path = crop.track(box_clip, sample_fps=8.0)
    assert state["n"] > 4, "the fault was never reached"
    assert state["released"], "the VideoCapture must be released on the error path"
    assert len(path.keyframes) == 4
    _assert_renderable(path)


@pytest.mark.needs_ffmpeg
def test_tracked_path_is_accepted_by_the_renderer(box_clip: Path, tmp_path: Path):
    """The cross-module contract: render.py must swallow a tracked path whole."""
    from aiclipper.models import Timeline, VisualLayer
    from aiclipper.render import build_command

    path = crop.simplify(crop.track(box_clip, sample_fps=8.0), tolerance=4.0)
    assert not path.is_static, "this fixture should produce a moving path"

    timeline = Timeline(width=180, height=320, fps=25, duration=CLIP_SECONDS)
    timeline.add_visual(VisualLayer(kind="video", src=str(box_clip), crop=path, label="main"))
    workdir = tmp_path / "graph"
    workdir.mkdir()
    cmd = build_command(timeline, tmp_path / "out.mp4", workdir=workdir)
    assert any("sendcmd" in part for part in cmd)

    (script,) = list(workdir.glob("*.cmd"))
    lines = [ln for ln in script.read_text().splitlines() if ln.strip()]
    assert len(lines) == len(path.keyframes)
    w, h = path.size
    for line in lines:
        x = int(line.split(" x ")[1].split(",")[0])
        y = int(line.rsplit(" y ", 1)[1].rstrip(";"))
        assert x % 2 == 0 and y % 2 == 0
        assert 0 <= x <= CLIP_W - w and 0 <= y <= CLIP_H - h


@pytest.fixture(scope="session")
def rotated_clip(clip_dir: Path, box_clip: Path) -> Path:
    """The box clip re-muxed with a 90 degree display matrix.

    ``ffprobe`` keeps reporting the *coded* 320x180, while every decoder --
    OpenCV and the ffmpeg filtergraph alike -- hands out rotated 180x320 frames.
    """
    dest = clip_dir / "rotated.mp4"
    ff.run_ffmpeg(["-y", "-display_rotation", "90", "-i", str(box_clip), "-c", "copy", str(dest)])
    return dest


@pytest.mark.needs_ffmpeg
def test_track_uses_the_decoded_frame_size_not_the_coded_one(rotated_clip: Path):
    """A rotated source must not yield a window that sits outside the frame."""
    cv2 = pytest.importorskip("cv2")
    probed = ff.probe(rotated_clip)

    cap = cv2.VideoCapture(str(rotated_clip))
    try:
        ok, frame = cap.read()
        assert ok
        decoded = (int(frame.shape[1]), int(frame.shape[0]))
    finally:
        cap.release()
    if decoded == (probed.width, probed.height):
        pytest.skip("this ffmpeg/OpenCV pair does not apply the display matrix")

    path = crop.track(rotated_clip, target_aspect=VERTICAL, sample_fps=4.0)
    # The window belongs to the frames the renderer will see, not to ffprobe's.
    assert (path.source_width, path.source_height) == decoded
    _assert_renderable(path, source=decoded)
    w, h = path.size
    assert w <= decoded[0] and h <= decoded[1]
    # A 180x320 source is already vertical: the whole frame is the crop.
    assert (w, h) == (180, 320)
