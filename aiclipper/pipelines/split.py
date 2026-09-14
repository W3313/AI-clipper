"""The ``split`` pipeline: two clips stacked into one vertical short.

The format everybody recognises from a phone screen: something to watch on top,
something to keep the eye busy underneath.  The canvas is cut into two panes --
exact halves, no gap, no overlap -- and each pane is filled edge to edge by its
own source::

    ingest.resolve(top)                      -> the upper pane
    ingest.resolve(bottom) / assets.pick_background -> the lower pane
    crop.track / crop.center_crop            -> reframe each source to its pane
    tts.synthesize_lines -> concat_audio     -> optional narration
    captions.build                           -> the narration, burned in
    Timeline(two visual layers + audio)      -> render.render

Three rules carry most of the design:

**The panes tile the canvas.**  :func:`pane_rects` splits an even canvas height
into two *even* pane heights that sum back to it exactly.  ``1920`` gives two
``960`` panes; an awkward ``962`` gives ``480`` over ``482`` rather than a
half-pixel seam.  Nothing is ever letterboxed: each pane is ``fit="cover"``.

**A pane's aspect ratio is not the canvas's.**  A 9:16 canvas cut in half gives
two 9:8 panes -- *landscape* rectangles.  Reframing therefore targets
``pane.aspect``, never ``width / height``, or every pane would be cropped to a
tall slot and then squeezed sideways into a wide one.

**One voice at a time.**  Without narration the top pane's own audio is the
soundtrack.  With narration the top pane is muted and the spoken track takes
over, with captions cut from the same word timings.  The bottom pane is filler
and never contributes audio, because two soundtracks at once is noise.

Everything here runs offline: the offline speech backend, a procedurally
generated background library, and no import of an optional dependency at module
scope.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .. import ingest, tts
from .. import render as render_module
from ..config import Settings, get_settings
from ..errors import IngestError
from ..models import (
    CaptionStyle,
    CropPath,
    MediaInfo,
    ProjectResult,
    Timeline,
    Transcript,
    VisualLayer,
    Word,
)
from . import common

if TYPE_CHECKING:  # pragma: no cover - typing only, the import stays lazy at runtime
    from ..assets import Asset

log = logging.getLogger(__name__)

__all__ = [
    "KIND",
    "LINE_GAP",
    "TAIL_SECONDS",
    "MUSIC_GAIN_DB",
    "MIN_RENDERABLE",
    "CAPTION_LINES",
    "CAPTION_CLEARANCE",
    "CAPTION_MIN_MARGIN_V",
    "Pane",
    "pane_rects",
    "narration_lines",
    "caption_style_clear_of_seam",
    "run",
]

#: ``ProjectResult.kind`` for every video this module produces.
KIND = "split"

#: Silence between two narrated lines.  Handed to both
#: :func:`aiclipper.tts.synthesize_lines` and
#: :func:`~aiclipper.pipelines.common.concat_audio` so their clocks agree.
LINE_GAP = 0.18

#: Air after the last narrated word, in seconds.
TAIL_SECONDS = 0.6

#: A music bed sits well under the panes even before ducking.
MUSIC_GAIN_DB = -22.0

#: A timeline shorter than this is not worth an encode.
MIN_RENDERABLE = 0.2

#: Lines a caption group is assumed to wrap to when reserving room for it --
#: the same working worst case :mod:`aiclipper.pipelines.texts` reserves.
CAPTION_LINES = 3

#: Clear air kept between the pane seam and the top of the caption block, in
#: caption reference pixels.
CAPTION_CLEARANCE = 24

#: Captions are never pushed closer to the bottom edge than this.
CAPTION_MIN_MARGIN_V = 90

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


# --------------------------------------------------------------------------- #
# geometry
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Pane:
    """One half of the canvas, in canvas pixels."""

    x: int
    y: int
    w: int
    h: int

    @property
    def aspect(self) -> float:
        """Width over height -- what a crop window for this pane must match."""
        return (self.w / self.h) if self.h else 0.0

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


def pane_rects(width: int, height: int) -> tuple[Pane, Pane]:
    """Split a canvas into the top and bottom pane.

    Both panes get an **even** width and height (``yuv420p`` refuses odd
    dimensions, and so does the renderer's ``crop``), they start at ``x = 0``,
    the bottom one starts exactly where the top one ends, and their heights sum
    to the canvas height -- so the two rectangles tile it with no gap and no
    overlap.

    An odd ``width``/``height`` is rounded *down* to even first, which is the
    same normalisation :func:`aiclipper.pipelines.common.canvas_size` applies to
    the canvas itself: the returned rects always tile the canvas the renderer
    will actually use.  When that even height does not halve evenly (``962``)
    the extra two pixels go to the bottom pane.
    """
    w = int(width) - (int(width) % 2)
    h = int(height) - (int(height) % 2)
    if w < 2 or h < 4:
        raise ValueError(f"a split canvas needs at least 2x4 pixels, got {width}x{height}")

    top_h = h // 2
    top_h -= top_h % 2
    top_h = max(2, top_h)
    bottom_h = h - top_h
    return Pane(0, 0, w, top_h), Pane(0, top_h, w, bottom_h)


# --------------------------------------------------------------------------- #
# narration text
# --------------------------------------------------------------------------- #

def caption_style_clear_of_seam(
    style: CaptionStyle | None, *, width: int, height: int
) -> CaptionStyle | None:
    """Keep a centred preset off the seam between the two panes.

    The panes are exact halves, so the seam runs along the middle of the canvas
    -- which is precisely where a ``position="center"`` preset puts its text.
    Seven of the sixteen presets are centred, ``clean`` (this pipeline's own
    default) among them, so the default render used to slice every caption in
    half along the pane edge.

    Such a preset is re-seated into the bottom pane, which is the filler one:
    the text lands on quiet pixels instead of across the join, and the top pane
    -- the clip the user actually came for -- is left alone.  A preset that is
    already anchored top or bottom is returned untouched.
    """
    if style is None or style.position != "center":
        return style

    _, play_h = common.caption_resolution(width, height)
    seam = play_h / 2.0
    block = style.font_size * 1.32 * CAPTION_LINES + CAPTION_CLEARANCE
    # margin_v is measured up from the bottom edge, so the block clears the seam
    # while its bottom stays at most this far up.
    allowed = int(play_h - seam - block)
    if allowed < CAPTION_MIN_MARGIN_V:
        log.warning("split: the caption block is taller than a pane; captions may cross the seam")
    # Centre the block in the lower pane when there is room for it.
    margin_v = max(CAPTION_MIN_MARGIN_V, min(int(seam / 2.0), allowed))
    log.debug("split: centred captions re-seated into the bottom pane (margin_v=%d)", margin_v)
    return replace(style, position="bottom", margin_v=margin_v)


def narration_lines(text: str | None) -> list[str]:
    """Break narration text into the lines the speech backend speaks one by one.

    Explicit line breaks are honoured first (a caller who laid their script out
    by hand gets exactly those beats), then each line is split on sentence
    punctuation so a single paragraph still lands as separate files -- which is
    what gives :func:`~aiclipper.pipelines.common.concat_audio` per-line offsets
    to hang captions on.  Blank input yields an empty list, i.e. no narration.
    """
    out: list[str] = []
    for raw in (text or "").splitlines():
        chunk = raw.strip()
        if not chunk:
            continue
        for piece in _SENTENCE_SPLIT.split(chunk):
            spoken = piece.strip()
            if spoken:
                out.append(spoken)
    return out


# --------------------------------------------------------------------------- #
# sources
# --------------------------------------------------------------------------- #

def _library_background(name: str | None, settings: Settings, seed: int | None = None) -> Asset:
    """A background loop for the lower pane.

    :func:`aiclipper.assets.pick_background` generates the procedural starter
    set by itself when the library holds no backgrounds, and it generates only
    that *kind*, so a user who ships backgrounds but no music beds does not wait
    for beds nobody asked for.
    """
    from .. import assets as assets_module

    return assets_module.pick_background(name or None, settings=settings, seed=seed)


def _resolve_bottom(
    bottom: str | Path | None,
    *,
    settings: Settings,
    workspace: Path,
    seed: int | None,
) -> tuple[MediaInfo | None, Asset | None]:
    """Resolve the lower pane to either a second clip or a library background.

    ``None`` (and an empty string) picks a background from the asset library,
    generating the procedural starter set when the library is empty -- a split
    with nothing to put underneath is still a usable video.  A path or URL is
    ingested as a real second source.  Anything else is looked up as a library
    asset *name*, so ``bottom="ember_mist"`` reaches for the catalogue instead
    of the filesystem.
    """
    raw = "" if bottom is None else str(bottom).strip()
    if not raw:
        return None, _library_background(None, settings, seed)

    path = Path(raw).expanduser()
    if isinstance(bottom, Path) or path.exists() or ingest.is_url(raw):
        media = ingest.resolve(bottom, settings=settings, workspace=workspace)
        if not media.has_video or media.width <= 0 or media.height <= 0:
            raise IngestError(f"the bottom pane needs a video source; {media.path} has none")
        return media, None

    log.debug("bottom %r is not a file or URL; looking it up in the asset library", raw)
    return None, _library_background(raw, settings, seed)


# --------------------------------------------------------------------------- #
# duration
# --------------------------------------------------------------------------- #

def _duration(
    seconds: float | None,
    *,
    spoken: float,
    words: list[Word],
    top: MediaInfo,
    bottom: MediaInfo | None,
) -> tuple[float, str]:
    """The timeline length and the reason it is that length.

    Precedence, highest first: an explicit ``seconds``; the narration (its real
    length plus :data:`TAIL_SECONDS`); otherwise the shorter of the two sources
    -- or the top source alone when the lower pane is a library loop, which can
    be repeated for as long as the timeline needs.
    """
    if seconds is not None and float(seconds) > 0:
        return round(max(MIN_RENDERABLE, float(seconds)), 3), "seconds"

    narrated = max(float(spoken), common.words_span(words))
    if narrated > 0:
        return round(narrated + TAIL_SECONDS, 3), "narration"

    spans = [float(top.duration)]
    if bottom is not None:
        spans.append(float(bottom.duration))
    usable = [span for span in spans if span > 0]
    return round(min(usable) if usable else MIN_RENDERABLE, 3), "sources"


# --------------------------------------------------------------------------- #
# panes
# --------------------------------------------------------------------------- #

def _crop_for(
    media: MediaInfo,
    pane: Pane,
    *,
    reframe: bool,
    loops: bool,
    settings: Settings,
    end: float,
) -> tuple[CropPath | None, str]:
    """The crop window that fills ``pane`` from ``media``, plus how it was made.

    The target is ``pane.aspect`` -- half a vertical canvas is a *landscape*
    rectangle, so cropping to the canvas ratio here would be wrong twice over.
    Tracking is skipped for a pane that has to loop: an animated crop is driven
    by a ``sendcmd`` script on the filter timeline, which does not rewind with
    the source, so a static window is the honest choice there.
    """
    from .. import crop as crop_module

    if not reframe or loops:
        mode = "loop-center" if (reframe and loops) else "center"
        return _non_empty(crop_module.center_crop(media, pane.aspect), mode)

    path = crop_module.simplify(
        crop_module.track(
            media.path, target_aspect=pane.aspect, settings=settings, start=0.0, end=end
        )
    )
    return _non_empty(path, "track-static" if path.is_static else "track")


def _non_empty(path: CropPath, mode: str) -> tuple[CropPath | None, str]:
    """A keyframe-less path means "no crop" to the renderer; say so explicitly."""
    if not path.keyframes:
        return None, "none"
    return path, mode


def _source_layer(
    media: MediaInfo,
    pane: Pane,
    *,
    duration: float,
    crop_path: CropPath | None,
    take_audio: bool,
    z: int,
    label: str,
) -> VisualLayer:
    """One pane filled by a real clip, looped when the clip runs out early."""
    return VisualLayer(
        kind="video",
        src=str(media.path),
        start=0.0,
        end=duration,
        src_start=0.0,
        x=pane.x,
        y=pane.y,
        w=pane.w,
        h=pane.h,
        fit="cover",
        loop=_loops(media, duration),
        crop=crop_path,
        take_audio=bool(take_audio),
        volume=1.0,
        z=z,
        label=label,
    )


def _library_layer(asset: Asset, pane: Pane, *, duration: float, z: int, settings: Settings) -> VisualLayer:
    """One pane filled by a library loop, opened at a seeded offset."""
    layer = common.background_layer(
        asset, width=pane.w, height=pane.h, duration=duration, z=z, settings=settings
    )
    return replace(layer, x=pane.x, y=pane.y)


def _loops(media: MediaInfo | None, duration: float) -> bool:
    """Does this source have to repeat to cover a ``duration``-second timeline?"""
    if media is None:
        return False
    span = float(media.duration)
    return span <= 0.0 or span < duration


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def run(
    top: str | Path,
    *,
    bottom: str | Path | None = None,
    narration: str | None = None,
    voice: str = "",
    style: str = "clean",
    music: str | None = None,
    seconds: float | None = None,
    captions: bool = True,
    reframe: bool = True,
    out_path: Path | None = None,
    settings: Settings | None = None,
    provider: Any = None,
) -> ProjectResult:
    """Stack ``top`` over ``bottom`` into one vertical short.

    ``top`` is anything :func:`aiclipper.ingest.resolve` accepts.  ``bottom`` is
    a second clip, the *name* of a library background, or ``None`` to let the
    asset library supply the lower pane.  Each pane is reframed to fill its own
    half of the canvas (``reframe=False`` uses a static centred crop instead of
    following the action).

    ``narration`` is spoken verbatim -- it is the caller's own words, never a
    generated script -- and when present it replaces the top pane's audio and
    supplies the burned captions (``captions=False`` burns none, ``style`` names
    the preset).  ``music`` names a bed from the library; without one the panes
    and the narration carry the whole soundtrack.

    ``seconds`` forces the length.  Without it the narration decides, and
    without narration the shorter of the two sources does -- a library loop for
    the lower pane never shortens anything, it simply repeats.

    ``provider`` is accepted so every pipeline's ``run`` has the same shape, and
    is only recorded in the metadata: a split writes no script, it speaks the
    words it was handed.

    Raises :class:`~aiclipper.errors.IngestError` when a source carries no
    usable video, :class:`~aiclipper.errors.AssetError` when a named background
    does not exist, and :class:`~aiclipper.errors.RenderError` when the timeline
    will not render.
    """
    s = settings or get_settings()
    width, height = common.canvas_size(s)
    top_pane, bottom_pane = pane_rects(width, height)

    caption_style = common.caption_style(style, captions)
    lines = narration_lines(narration)
    work = common.workspace(f"{KIND}-{Path(str(top)).stem}", s)

    top_media = ingest.resolve(top, settings=s, workspace=work / "top")
    if not top_media.has_video or top_media.width <= 0 or top_media.height <= 0:
        raise IngestError(f"the top pane needs a video source; {top_media.path} has none")
    if top_media.duration <= MIN_RENDERABLE and seconds is None and not lines:
        raise IngestError(f"top source is too short to split: {top_media.path} is {top_media.duration:.2f}s")

    bottom_media, bottom_asset = _resolve_bottom(
        bottom, settings=s, workspace=work / "bottom", seed=s.seed
    )

    narration_path: Path | None = None
    words: list[Word] = []
    spoken = 0.0
    voice_spec = None
    speaker: Any = None
    if lines:
        voice_spec = common.resolve_voice(voice, settings=s)
        speaker = tts.get_provider(settings=s)
        results = tts.synthesize_lines(
            lines, work / "vo", voice=voice_spec, provider=speaker, gap=LINE_GAP, settings=s
        )
        narration_path, offsets = common.concat_audio(
            [r.audio_path for r in results], work / "narration.wav", gap=LINE_GAP, settings=s
        )
        words = common.narration_words(
            results, offsets, audio=narration_path, text=" ".join(lines), settings=s
        )
        spoken = common.probe_duration(narration_path, settings=s) or tts.total_duration(results, LINE_GAP)
        log.info("split: %d narrated line(s), %.2fs of speech", len(lines), spoken)

    duration, duration_source = _duration(
        seconds, spoken=spoken, words=words, top=top_media, bottom=bottom_media
    )

    top_crop, top_mode = _crop_for(
        top_media, top_pane, reframe=reframe, loops=_loops(top_media, duration),
        settings=s, end=min(duration, float(top_media.duration) or duration),
    )
    top_audio = narration_path is None and bool(top_media.has_audio)
    timeline = Timeline(
        width=width,
        height=height,
        fps=int(s.fps),
        duration=duration,
        background="#000000",
        title=top_media.title or Path(top_media.path).stem,
    )
    timeline.add_visual(
        _source_layer(
            top_media, top_pane, duration=duration, crop_path=top_crop,
            take_audio=top_audio, z=0, label="top",
        )
    )

    bottom_mode = "library"
    if bottom_media is not None:
        bottom_crop, bottom_mode = _crop_for(
            bottom_media, bottom_pane, reframe=reframe, loops=_loops(bottom_media, duration),
            settings=s, end=min(duration, float(bottom_media.duration) or duration),
        )
        timeline.add_visual(
            _source_layer(
                bottom_media, bottom_pane, duration=duration, crop_path=bottom_crop,
                take_audio=False, z=1, label="bottom",
            )
        )
    else:
        assert bottom_asset is not None  # _resolve_bottom always returns one of the two
        timeline.add_visual(
            _library_layer(bottom_asset, bottom_pane, duration=duration, z=1, settings=s)
        )

    if narration_path is not None:
        timeline.add_audio(common.voice_track(narration_path))

    music_asset: Asset | None = None
    if music is not None and str(music).strip():
        from .. import assets as assets_module

        music_asset = assets_module.pick_music(str(music).strip(), settings=s)
        timeline.add_audio(
            common.music_track(
                music_asset, duration=duration, gain_db=MUSIC_GAIN_DB,
                duck=narration_path is not None,
            )
        )

    timeline.subtitles = common.caption_track(
        words, work / "captions",
        style=caption_style_clear_of_seam(caption_style, width=width, height=height),
        width=width, height=height,
        enabled=captions and bool(words), settings=s,
    )

    title = timeline.title or KIND
    out = common.resolve_output(out_path, f"{title}-{KIND}", s)
    rendered = render_module.render(timeline, out, settings=s)
    log.info("split: %s over %s -> %s", top_media.path.name,
             bottom_media.path.name if bottom_media else (bottom_asset.name if bottom_asset else "-"),
             rendered.path)

    bottom_path = bottom_media.path if bottom_media is not None else (
        bottom_asset.path if bottom_asset is not None else None
    )
    return ProjectResult(
        output=rendered.path,
        kind=KIND,
        title=title,
        duration=rendered.duration or duration,
        transcript=Transcript.from_words(words) if words else None,
        metadata={
            "top": str(top_media.path),
            "top_title": top_media.title,
            "top_url": top_media.source_url,
            "top_size": [int(top_media.width), int(top_media.height)],
            "top_duration": round(float(top_media.duration), 3),
            "top_audio": top_audio,
            "bottom": str(bottom_path) if bottom_path is not None else None,
            "bottom_kind": "source" if bottom_media is not None else "library",
            "bottom_name": (
                bottom_media.path.stem if bottom_media is not None
                else (bottom_asset.name if bottom_asset is not None else "")
            ),
            "bottom_duration": round(
                float(bottom_media.duration) if bottom_media is not None
                else float(getattr(bottom_asset, "duration", 0.0) or 0.0), 3
            ),
            "panes": {"top": list(top_pane.as_tuple()), "bottom": list(bottom_pane.as_tuple())},
            "pane_aspect": round(top_pane.aspect, 6),
            "canvas": [width, height],
            "fps": int(s.fps),
            "narration": narration_path is not None,
            "narration_text": (narration or "").strip(),
            "narration_lines": len(lines),
            "narration_seconds": round(spoken, 3),
            "voice": (voice_spec.voice_id or voice_spec.provider) if voice_spec else "",
            "voice_request": voice,
            "tts_provider": common.provider_name(speaker),
            "captions": timeline.subtitles is not None,
            "style": common.style_name(caption_style) if timeline.subtitles is not None else "",
            "words": len(words),
            "music": music_asset.name if music_asset is not None else None,
            "reframed": top_mode.startswith("track") or bottom_mode.startswith("track"),
            "crop": {"top": top_mode, "bottom": bottom_mode},
            "duration_source": duration_source,
            "seconds_requested": float(seconds) if seconds is not None else None,
            "timeline_seconds": duration,
            "provider": common.provider_name(provider),
        },
    )
