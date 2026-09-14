"""The auto-clipper: one long video (or URL) becomes N captioned vertical shorts.

The flow is deliberately linear::

    ingest.resolve -> transcribe.transcribe -> highlight.select
        -> per candidate: crop (tracked or centred)
                          captions from ``transcript.slice(start, end)``
                          Timeline(one video layer + subtitles)
                          render.render

The source is transcribed **once** -- :mod:`aiclipper.transcribe` caches next to
the media, and every candidate reads its captions out of that one transcript
with :meth:`~aiclipper.models.Transcript.slice`, rebased so the clip's own clock
starts at zero.

Nothing here is fatal but a source we cannot decode.  A file with no speech (or
an ASR backend that is not installed) falls back to evenly spaced windows, a
transcript that yields fewer good moments than were asked for simply produces
fewer shorts, and a source shorter than ``min_duration`` becomes a single clip
of whatever length it has.  The source's own audio rides along on the video
layer, so a clip sounds like the moment it was cut from.
"""

from __future__ import annotations

import dataclasses
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import Settings, get_settings
from ..errors import AiclipperError, IngestError
from ..models import (
    CaptionCue,
    CaptionStyle,
    ClipCandidate,
    CropPath,
    MediaInfo,
    ProjectResult,
    RenderOptions,
    SubtitleTrack,
    Timeline,
    Transcript,
    VisualLayer,
    Word,
)
from . import common

log = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_COUNT",
    "DEFAULT_MIN_DURATION",
    "DEFAULT_MAX_DURATION",
    "MIN_RENDERABLE",
    "even_windows",
    "run",
]

#: How many shorts a bare :func:`run` produces.
DEFAULT_COUNT = 3

#: Default clip length bounds, in seconds.
DEFAULT_MIN_DURATION = 15.0
DEFAULT_MAX_DURATION = 60.0

#: A window shorter than this is not worth an encode.
MIN_RENDERABLE = 0.2

#: How much of the source stem / title slug survives into the output name.
_STEM_CHARS = 24
_TITLE_CHARS = 28

#: Timing differences below this are float noise, not a word cut in half.
_EPSILON = 1e-6


# --------------------------------------------------------------------------- #
# candidate selection
# --------------------------------------------------------------------------- #

def even_windows(
    duration: float,
    *,
    count: int = DEFAULT_COUNT,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
) -> list[ClipCandidate]:
    """Evenly spaced windows across ``duration`` -- the no-speech fallback.

    As many windows as fit at ``min_duration`` are laid out (never more than
    ``count``), each one ``max_duration`` long at the most and never overlapping
    its neighbour.  A source too short to host even one window of the requested
    length yields a single window covering the whole thing, because one short
    clip beats no clip at all.
    """
    span = max(0.0, float(duration))
    wanted = max(0, int(count))
    if span <= MIN_RENDERABLE or wanted == 0:
        return []

    low = max(MIN_RENDERABLE, float(min_duration))
    high = max(low, float(max_duration))

    slots = int(span // low) if span >= low else 1
    slots = max(1, min(wanted, slots))
    stride = span / slots
    length = min(high, stride)

    out: list[ClipCandidate] = []
    for index in range(slots):
        start = round(index * stride, 3)
        end = round(min(start + length, span), 3)
        if end - start <= MIN_RENDERABLE:
            continue
        out.append(
            ClipCandidate(
                start=start,
                end=end,
                title=f"Part {index + 1}",
                hook="",
                reason="evenly spaced window (no speech to rank)",
                score=0.0,
                tags=["even"],
            )
        )
    return out


def _clamped(candidates: list[ClipCandidate], limit: float) -> list[ClipCandidate]:
    """Keep every candidate inside ``[0, limit]`` and drop the degenerate ones."""
    out: list[ClipCandidate] = []
    for candidate in candidates:
        start = max(0.0, float(candidate.start))
        end = min(float(candidate.end), limit) if limit > 0 else float(candidate.end)
        if end - start <= MIN_RENDERABLE:
            continue
        out.append(
            ClipCandidate(
                start=round(start, 3),
                end=round(end, 3),
                title=candidate.title,
                hook=candidate.hook,
                reason=candidate.reason,
                score=candidate.score,
                tags=list(candidate.tags),
            )
        )
    return out


def _whole_word_window(words: Sequence[Word], start: float, end: float) -> tuple[float, float]:
    """Pull ``[start, end)`` back onto whole-word edges, never outwards.

    :func:`aiclipper.highlight.select` already snaps to word edges, but a window
    does not necessarily survive the trip: clamping it to the probed media
    duration (:func:`_clamped`) can land inside the final word, and the evenly
    spaced fallback windows know nothing about speech at all.  Either way
    :meth:`~aiclipper.models.Transcript.slice` would hand the caption builder a
    word truncated to the boundary, and the render would cut it mid-read.

    So both edges move *inwards* to the nearest whole word: a word straddling
    the head is dropped (the clip starts after it), and the tail ends on the
    last word that fits entirely.  Shrinking rather than growing keeps the
    window inside the media, inside ``max_duration`` and clear of its
    neighbours; the price is at most one part-heard word per edge.  A window
    holding no whole word at all is left exactly as it was -- a truncated word
    still beats no clip.
    """
    inside = [w for w in words if w.end > start + _EPSILON and w.start < end - _EPSILON]
    whole = [w for w in inside if w.start >= start - _EPSILON and w.end <= end + _EPSILON]
    if not inside or not whole:
        return start, end
    new_start = start if inside[0] is whole[0] else whole[0].start
    new_end = end if inside[-1] is whole[-1] else whole[-1].end
    if new_end - new_start <= MIN_RENDERABLE:
        return start, end
    return round(max(0.0, new_start), 6), round(new_end, 6)


def _snapped(candidates: list[ClipCandidate], transcript: Transcript) -> list[ClipCandidate]:
    """Apply :func:`_whole_word_window` to every candidate, in place of nothing."""
    words = transcript.words
    if not words:
        return candidates
    out: list[ClipCandidate] = []
    for candidate in candidates:
        start, end = _whole_word_window(words, float(candidate.start), float(candidate.end))
        if (start, end) != (candidate.start, candidate.end):
            log.debug(
                "snapped clip window %.3f-%.3f to whole words %.3f-%.3f",
                candidate.start,
                candidate.end,
                start,
                end,
            )
        out.append(dataclasses.replace(candidate, start=start, end=end))
    return out


def _transcribe(media: MediaInfo, settings: Settings) -> tuple[Transcript, str]:
    """Transcribe the source once, degrading to an empty transcript.

    A missing ASR backend is not a reason to refuse the job: the caller still
    gets clips, just chosen by the clock instead of by what was said.
    """
    from .. import transcribe as transcribe_module

    try:
        return transcribe_module.transcribe(media.path, settings=settings), "asr"
    except AiclipperError as exc:
        log.warning("transcription of %s unavailable (%s); clipping without speech", media.path, exc)
        return Transcript(segments=[], language="en", duration=media.duration), "unavailable"


def _select(
    transcript: Transcript,
    *,
    media: MediaInfo,
    count: int,
    min_duration: float,
    max_duration: float,
    settings: Settings,
    provider: Any,
) -> tuple[list[ClipCandidate], str]:
    """Rank moments, falling back to evenly spaced windows when nothing ranks."""
    from .. import highlight

    picked: list[ClipCandidate] = []
    if not transcript.is_empty:
        picked = _snapped(
            _clamped(
                list(
                    highlight.select(
                        transcript,
                        count=count,
                        min_duration=min_duration,
                        max_duration=max_duration,
                        settings=settings,
                        provider=provider,
                    )
                ),
                media.duration,
            ),
            transcript,
        )
    if picked:
        return picked, "highlight"

    log.info("no ranked moments in %s; falling back to evenly spaced windows", media.path.name)
    windows = even_windows(
        media.duration, count=count, min_duration=min_duration, max_duration=max_duration
    )
    return _snapped(_clamped(windows, media.duration), transcript), "even"


# --------------------------------------------------------------------------- #
# per-clip pieces
# --------------------------------------------------------------------------- #

def _crop_for(
    media: MediaInfo,
    candidate: ClipCandidate,
    *,
    reframe: bool,
    aspect: float,
    settings: Settings,
) -> tuple[CropPath | None, str]:
    """The crop window for one candidate, plus the name of the path that made it.

    Tracking only earns its decode when the source is wider than the canvas: a
    vertical source is already framed, so it takes the centred crop (which is a
    no-op when the aspect already matches) and skips reframing entirely.
    """
    from .. import crop as crop_module

    if not reframe or media.is_vertical:
        mode = "vertical" if media.is_vertical else "center"
        path = crop_module.center_crop(media, aspect)
    else:
        mode = "track"
        path = crop_module.simplify(
            crop_module.track(
                media.path,
                target_aspect=aspect,
                settings=settings,
                start=candidate.start,
                end=candidate.end,
            )
        )
    if not path.keyframes:
        return None, "none"
    if path.is_static and mode == "track":
        mode = "track-static"
    return path, mode


def _fit_cues(cues: Sequence[CaptionCue], duration: float, *, minimum: float) -> list[CaptionCue]:
    """Trim a cue list to ``[0, duration]``, dropping what is left of nothing.

    Cues are built from word timings alone, and the last one deliberately
    lingers past its final word (``captions.HOLD_SECONDS``).  On a clip that
    ends on a word edge that linger has nowhere to go: the cue outlives the
    video.  Trimming here keeps the caption clock and the clip's own clock
    honest -- no cue starts at or past the end, none ends after it.
    """
    out: list[CaptionCue] = []
    for cue in cues:
        start = max(0.0, float(cue.start))
        end = min(float(cue.end), duration)
        if end - start < minimum - _EPSILON:
            continue
        words = [w for w in cue.words if w.start < end - _EPSILON]
        if not words:
            continue
        out.append(CaptionCue(start=start, end=end, words=list(words)))
    return out


def _caption_track(
    words: Sequence[Word],
    out_dir: Path,
    *,
    style: CaptionStyle | None,
    width: int,
    height: int,
    duration: float,
    name: str,
    settings: Settings,
) -> SubtitleTrack | None:
    """Like :func:`common.caption_track`, but clamped to this clip's timeline.

    Same grouping, same preset, same reference resolution -- the cue list is
    simply cut to ``duration`` first (see :func:`_fit_cues`), because a clip is
    a window out of a longer talk and has no room past its own last frame.
    """
    if style is None or not words or duration <= 0.0:
        return None

    from .. import captions as captions_module

    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    play_w, play_h = common.caption_resolution(width, height)
    resolved = captions_module.get_style(style)
    cues = _fit_cues(
        captions_module.group_words(words, resolved), duration, minimum=captions_module.MIN_CUE
    )
    if not cues:
        return None
    ass_path = captions_module.write_ass(
        cues, target / name, style=resolved, width=play_w, height=play_h
    )
    fonts = settings.fonts_dir
    return SubtitleTrack(ass_path=ass_path, fonts_dir=fonts if fonts.is_dir() else None)


def _out_directory(out_dir: str | Path | None) -> str | None:
    """Normalise ``out_dir`` so :func:`common.resolve_output` always sees a *directory*.

    ``clip`` writes one file per candidate, so the caller's ``out_dir`` can only
    ever be a folder.  Without the trailing separator a path that happens to
    carry an extension (``--out shorts.mp4``) would be read as an explicit file
    name and every short would overwrite the one before it.
    """
    if out_dir is None:
        return None
    raw = str(out_dir)
    return raw if raw.endswith(("/", "\\")) else f"{raw}/"


def _clip_stem(source_stem: str, index: int, title: str) -> str:
    """``<source stem>-01-<slug of title>`` -- short enough to survive intact."""
    head = common.safe_stem(source_stem, fallback="source")[:_STEM_CHARS].strip("-")
    tail = common.safe_stem(title or f"clip {index}", fallback=f"clip-{index}")[:_TITLE_CHARS].strip("-")
    return f"{head or 'source'}-{index:02d}-{tail or f'clip-{index}'}"


def _timeline(
    media: MediaInfo,
    candidate: ClipCandidate,
    *,
    index: int,
    width: int,
    height: int,
    fps: int,
    crop_path: CropPath | None,
    subtitles: SubtitleTrack | None,
) -> Timeline:
    """One video layer carrying its own audio, plus the burned caption track."""
    span = round(candidate.duration, 3)
    timeline = Timeline(
        width=width,
        height=height,
        fps=fps,
        duration=span,
        background="#000000",
        title=candidate.title,
    )
    timeline.add_visual(
        VisualLayer(
            kind="video",
            src=str(media.path),
            start=0.0,
            end=span,
            src_start=candidate.start,
            x=0,
            y=0,
            w=width,
            h=height,
            fit="cover",
            loop=False,
            crop=crop_path,
            take_audio=bool(media.has_audio),
            volume=1.0,
            z=0,
            label=f"clip{index:02d}",
        )
    )
    timeline.subtitles = subtitles
    return timeline


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def run(
    source: str | Path,
    *,
    count: int = DEFAULT_COUNT,
    min_duration: float = DEFAULT_MIN_DURATION,
    max_duration: float = DEFAULT_MAX_DURATION,
    style: str = "clean",
    reframe: bool = True,
    captions: bool = True,
    out_dir: Path | None = None,
    settings: Settings | None = None,
    provider: Any = None,
) -> list[ProjectResult]:
    """Cut ``source`` into up to ``count`` captioned vertical shorts.

    ``source`` is anything :func:`aiclipper.ingest.resolve` accepts: a local
    file or an http(s) URL.  ``style`` names a caption preset; ``captions=False``
    burns none.  ``reframe=False`` (or a source that is already vertical) uses a
    static centred crop instead of tracking the action.  Outputs land in
    ``out_dir`` -- or ``settings.output_dir`` -- named
    ``<source stem>-<rank>-<slug of title>.mp4`` in rank order.

    Returns one :class:`~aiclipper.models.ProjectResult` per rendered short,
    best first, each carrying its source window, score and the reason it was
    chosen in ``metadata``.  The list is empty only when the source has no
    window worth rendering.

    Raises :class:`~aiclipper.errors.IngestError` when the source cannot be
    resolved or carries no usable video.
    """
    from .. import ingest, render

    s = settings or get_settings()
    wanted = max(0, int(count))
    low = max(MIN_RENDERABLE, float(min_duration))
    high = max(low, float(max_duration))
    width, height = common.canvas_size(s)
    aspect = width / height
    # Resolve the preset before ingesting, transcribing and tracking: an unknown
    # name should fail in milliseconds, not after a source has been decoded.
    caption_style = common.caption_style(style, captions)
    target_dir = _out_directory(out_dir)

    work = common.workspace(f"clip-{Path(str(source)).stem}", s)
    media = ingest.resolve(source, settings=s, workspace=work / "source")
    if not media.has_video or media.width <= 0 or media.height <= 0:
        raise IngestError(f"clip needs a video source; {media.path} has no usable video stream")
    if media.duration <= MIN_RENDERABLE:
        raise IngestError(f"source is too short to clip: {media.path} is {media.duration:.2f}s")
    if wanted == 0:
        return []

    transcript, asr = _transcribe(media, s)
    candidates, selection = _select(
        transcript,
        media=media,
        count=wanted,
        min_duration=low,
        max_duration=high,
        settings=s,
        provider=provider,
    )
    if not candidates:
        log.warning("no clip windows found in %s", media.path)
        return []
    if len(candidates) < wanted:
        log.info("%s supported %d of the %d clips requested", media.path.name, len(candidates), wanted)

    provider_name = common.provider_name(provider) or _settings_provider_name(s)
    source_stem = media.path.stem
    options = RenderOptions()
    results: list[ProjectResult] = []

    for offset, candidate in enumerate(candidates):
        index = offset + 1
        stem = _clip_stem(source_stem, index, candidate.title)
        out_path = common.resolve_output(target_dir, stem, s)

        crop_path, crop_mode = _crop_for(
            media, candidate, reframe=reframe, aspect=aspect, settings=s
        )
        window = transcript.slice(candidate.start, candidate.end, rebase=True)
        words = window.words
        subtitles = _caption_track(
            words,
            work / "captions",
            style=caption_style,
            width=width,
            height=height,
            duration=round(candidate.duration, 3),
            name=f"{stem}.ass",
            settings=s,
        )
        timeline = _timeline(
            media,
            candidate,
            index=index,
            width=width,
            height=height,
            fps=int(s.fps),
            crop_path=crop_path,
            subtitles=subtitles,
        )
        rendered = render.render(timeline, out_path, options=options, settings=s)
        log.info("clip %02d: %.2f-%.2fs -> %s", index, candidate.start, candidate.end, rendered.path)

        results.append(
            ProjectResult(
                output=rendered.path,
                kind="clip",
                title=candidate.title or stem,
                duration=rendered.duration or timeline.duration,
                transcript=window,
                metadata={
                    "rank": index,
                    "score": round(float(candidate.score), 6),
                    "title": candidate.title,
                    "hook": candidate.hook,
                    "reason": candidate.reason,
                    "tags": list(candidate.tags),
                    "source": str(media.path),
                    "source_title": media.title,
                    "source_url": media.source_url,
                    "source_duration": round(float(media.duration), 3),
                    "source_size": [int(media.width), int(media.height)],
                    "source_start": candidate.start,
                    "source_end": candidate.end,
                    "window": [candidate.start, candidate.end],
                    "clip_duration": round(float(candidate.duration), 3),
                    "canvas": [width, height],
                    "fps": int(s.fps),
                    "reframed": crop_mode.startswith("track"),
                    "crop": crop_mode,
                    "crop_keyframes": len(crop_path.keyframes) if crop_path else 0,
                    "captions": subtitles is not None,
                    "style": common.style_name(caption_style) if subtitles is not None else None,
                    "words": len(words),
                    "source_audio": bool(media.has_audio),
                    "selection": selection,
                    "transcription": asr,
                    "provider": provider_name,
                    "requested": wanted,
                    "produced": len(candidates),
                },
            )
        )

    paths = [r.output for r in results]
    for result in results:
        result.siblings = [p for p in paths if p != result.output]
    return results


def _settings_provider_name(settings: Settings) -> str:
    """Which LLM backend the highlight step would have reached for."""
    from ..llm import get_provider

    try:
        return str(getattr(get_provider(settings.llm_provider or None, settings=settings), "name", ""))
    except AiclipperError as exc:  # an unknown provider name in the settings
        log.debug("could not resolve an llm provider (%s)", exc)
        return str(settings.llm_provider or "")
