"""The ``texts`` pipeline: a text-message conversation plays out over footage.

A caller hands in a topic -- or their own conversation -- and gets back a
vertical video in which an animated chat arrives bubble by bubble on top of a
looping background, every message spoken aloud in a different voice per side,
with a music bed ducked underneath.

The order of work:

1. :func:`aiclipper.scriptgen.write_chat` (or
   :func:`~aiclipper.scriptgen.parse_chat` when the caller brought their own
   text) produces a :class:`~aiclipper.models.ChatScript`.
2. Every spoken message is synthesised into its *own* file, with the voice that
   belongs to its side of the conversation.  This is why the module does not use
   :func:`aiclipper.tts.synthesize_lines`: that helper speaks one voice and lays
   the lines on a fixed ``gap``, while a chat needs two voices and a gap that
   changes from message to message.
3. :func:`plan_beats` walks the conversation and puts it on a clock (below).
4. :func:`aiclipper.overlays.render_chat` renders one transparent PNG per
   conversation *state*, and :func:`state_spans` maps those states onto the
   clock -- using each :class:`~aiclipper.overlays.ChatState`'s own ``visible``
   and ``typing`` fields rather than re-deriving the order, so a typing frame
   can never be mistaken for a bubble frame.
5. One :class:`~aiclipper.models.Timeline`: the background, one image layer per
   state covering exactly that state's span, one voice track per message at its
   own offset, and a ducked music bed.

**The clock.**  Walking the messages, for each one in turn:

* wait ``message.delay`` seconds (nothing new appears; the previous state stays
  on screen);
* when ``message.typing`` is set, show the typing indicator for that long;
* the bubble appears **and** its narration starts at the same instant;
* the next message waits for that narration to finish.

So the state for message *i* is visible from the moment it appears until the
state for message *i + 1* appears -- through the next message's delay and typing
beat -- and the last state stays up until the end of the video, which is the
last narration's end plus :data:`TAIL_SECONDS` of air.

Nothing here imports an optional third-party dependency at module scope and
nothing here touches the network: with the heuristic LLM provider, the offline
speech backend and the Pillow overlay backend the whole pipeline runs in a
container with no egress.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .. import assets as assets_module
from .. import overlays as overlays_module
from .. import render as render_module
from .. import scriptgen, tts
from ..config import Settings, get_settings
from ..errors import AiclipperError
from ..models import (
    CaptionStyle,
    ChatMessage,
    ChatScript,
    ProjectResult,
    Timeline,
    Transcript,
    TTSResult,
    VisualLayer,
    VoiceSpec,
)
from . import common

log = logging.getLogger(__name__)

__all__ = [
    "KIND",
    "TAIL_SECONDS",
    "MIN_BUBBLE_SECONDS",
    "MIN_STATE_SECONDS",
    "MUSIC_GAIN_DB",
    "CAPTION_CLEARANCE",
    "CAPTION_MIN_MARGIN_V",
    "CAPTION_LINES",
    "overlay_bottom",
    "caption_style_below_chat",
    "DEFAULT_VOICE",
    "DEFAULT_REPLY_VOICE",
    "Beat",
    "build_chat",
    "message_hold",
    "plan_beats",
    "state_spans",
    "run",
]

#: ``ProjectResult.kind`` for every video this module produces.
KIND = "texts"

#: Air after the last message's narration, so the ending is not abrupt.
TAIL_SECONDS = 1.2

#: A bubble is never on screen for less than this, however short its line is.
MIN_BUBBLE_SECONDS = 0.45

#: Floor for the final state's span, used only if the tail were ever zero.
MIN_STATE_SECONDS = 0.1

#: The bed sits well under the voices even before ducking kicks in.
MUSIC_GAIN_DB = -24.0

#: Clear air kept between the bottom of the bubble column and the top of the
#: caption block, in caption reference pixels.
CAPTION_CLEARANCE = 24

#: Captions are never pushed closer to the bottom edge than this, even when the
#: bubbles leave no room -- text under the platform's own UI helps nobody.
CAPTION_MIN_MARGIN_V = 90

#: Lines a caption group is assumed to wrap to when reserving room for it.
#: Presets cap a group at a handful of words, but a big uppercase face puts each
#: of them on its own line, so three is the working worst case.
CAPTION_LINES = 3

#: Catalogue voices used when the caller names none: two clearly different
#: reads, so the two sides of the conversation never sound like one person.
DEFAULT_VOICE = "storyteller"
DEFAULT_REPLY_VOICE = "storyteller_male"


# --------------------------------------------------------------------------- #
# the clock
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Beat:
    """One message placed on the video's clock.

    ``typing_start`` is when the typing indicator appears (``None`` when the
    message has no typing beat), ``start`` is when the bubble appears *and* its
    narration starts, and ``end`` is when that narration (or, for a silent
    message, its reading hold) finishes.  ``end`` is **not** when the bubble
    disappears: it stays up until the next state appears, which is what
    :func:`state_spans` works out.
    """

    index: int
    delay: float
    typing_start: float | None
    start: float
    end: float
    spoken: bool = False
    audio: Path | None = None

    @property
    def hold(self) -> float:
        """How long the narration (or the silent reading hold) lasts."""
        return max(0.0, self.end - self.start)


def message_hold(message: ChatMessage, duration: float = 0.0) -> float:
    """How long message ``message`` occupies the clock once its bubble appears.

    ``duration`` is the real length of its narration, probed from the synthesised
    file; a message that is never spoken (``read_aloud=False``, or no text) gets
    a reading hold estimated at the same pace the offline speech backend uses, so
    a silent bubble does not flash past.
    """
    span = max(0.0, float(duration or 0.0))
    if span <= 0.0:
        span = float(tts.estimate_duration(message.text or ""))
    return max(MIN_BUBBLE_SECONDS, span)


def plan_beats(
    messages: Sequence[ChatMessage],
    holds: Sequence[float],
    *,
    audio: Sequence[Path | None] | None = None,
) -> list[Beat]:
    """Lay ``messages`` on the clock, honouring every ``delay`` and ``typing``.

    ``holds[i]`` is how long message ``i`` holds the clock after its bubble
    appears -- normally the probed length of its narration, from
    :func:`message_hold`.  ``audio[i]`` is that narration's file, or ``None`` for
    a message nobody speaks.

    The returned beats are in message order and strictly increasing: every
    ``typing_start`` (when present) is before its ``start``, and every ``start``
    is after the previous beat's ``end``.
    """
    beats: list[Beat] = []
    cursor = 0.0
    for index, message in enumerate(messages):
        delay = max(0.0, float(getattr(message, "delay", 0.0) or 0.0))
        typing = max(0.0, float(getattr(message, "typing", 0.0) or 0.0))
        cursor += delay

        typing_start: float | None = None
        if typing > 0.0:
            typing_start = round(cursor, 6)
            cursor += typing

        start = round(cursor, 6)
        hold = max(MIN_BUBBLE_SECONDS, float(holds[index]) if index < len(holds) else 0.0)
        end = round(start + hold, 6)
        path = audio[index] if audio is not None and index < len(audio) else None
        beats.append(
            Beat(
                index=index,
                delay=delay,
                typing_start=typing_start,
                start=start,
                end=end,
                spoken=path is not None,
                audio=path,
            )
        )
        cursor = end
    return beats


def state_spans(
    script: ChatScript, beats: Sequence[Beat], total: float
) -> list[tuple[float, float]]:
    """Map every overlay state onto ``[start, end)`` on the clock.

    The states come from :func:`aiclipper.overlays.chat_states`, the same list
    :func:`~aiclipper.overlays.render_chat` renders, and each one is placed from
    its *own* fields: a typing state (``visible=i``) appears at beat ``i``'s
    ``typing_start``, and a bubble state (``visible=i+1``) appears at beat
    ``i``'s ``start``.  Each span runs until the next state appears, and the last
    one runs to ``total`` -- so the spans are contiguous, never overlap and cover
    everything from the first state to the end of the video.
    """
    states = overlays_module.chat_states(script)
    starts: list[float] = []
    for state in states:
        if state.typing:
            beat = beats[state.visible]
            starts.append(beat.start if beat.typing_start is None else beat.typing_start)
        else:
            starts.append(beats[state.visible - 1].start)

    spans: list[tuple[float, float]] = []
    for position, begin in enumerate(starts):
        if position + 1 < len(starts):
            end = starts[position + 1]
        else:
            end = max(float(total), begin + MIN_STATE_SECONDS)
        spans.append((begin, max(end, begin + MIN_STATE_SECONDS)))
    return spans


# --------------------------------------------------------------------------- #
# script sourcing
# --------------------------------------------------------------------------- #

def _cfg(settings: Settings | None) -> Settings:
    return settings or get_settings()


def build_chat(
    topic: str | None,
    script: str | ChatScript | None,
    *,
    turns: int = 10,
    settings: Settings | None = None,
    provider: Any = None,
) -> tuple[ChatScript, str, str]:
    """Return ``(script, source, provider_name)`` for this request.

    ``source`` is ``"supplied"`` when the caller brought the conversation (a
    :class:`~aiclipper.models.ChatScript` is used as-is, a string goes through
    :func:`aiclipper.scriptgen.parse_chat`) and ``"generated"`` when the
    language-model provider wrote it.  A supplied conversation never reaches a
    provider.

    Raises :class:`~aiclipper.errors.AiclipperError` when neither a topic nor a
    usable conversation was given, or when the conversation has no messages.
    """
    s = _cfg(settings)
    subject = (topic or "").strip()

    if script is not None and not isinstance(script, (str, ChatScript)):
        raise AiclipperError(f"script must be text or a ChatScript, got {type(script).__name__}")

    if isinstance(script, ChatScript):
        result, source, provider_name = script, "supplied", ""
    elif isinstance(script, str) and script.strip():
        result, source, provider_name = scriptgen.parse_chat(script), "supplied", ""
    elif subject:
        prov = common.llm_provider(provider, s)
        result = scriptgen.write_chat(subject, turns=max(2, int(turns)), settings=s, provider=prov)
        source, provider_name = "generated", common.provider_name(prov)
    else:
        raise AiclipperError("the texts pipeline needs a topic or a conversation")

    if not [m for m in result.messages if (m.text or "").strip()]:
        raise AiclipperError("the conversation has no messages to show")
    return result, source, provider_name


# --------------------------------------------------------------------------- #
# voices, theme and captions
# --------------------------------------------------------------------------- #

def _resolve_theme(theme: str, script: ChatScript) -> str:
    """The canonical chat theme for this run (unknown names fall back)."""
    for candidate in (theme or "", script.theme or ""):
        key = candidate.strip().lower()
        if key in overlays_module.CHAT_THEMES:
            return key
        if key:
            log.warning("unknown chat theme %r; using %r", candidate, overlays_module.DEFAULT_CHAT_THEME)
    return overlays_module.DEFAULT_CHAT_THEME


def _side_voices(voice: str, reply_voice: str, settings: Settings) -> tuple[VoiceSpec, VoiceSpec]:
    """``(outgoing, incoming)`` voice specs -- one per side of the conversation.

    ``voice`` speaks the outgoing ("me") side, ``reply_voice`` the incoming side.
    When the caller names neither, two different catalogue voices are used so the
    sides stay distinguishable; a name that resolves to nothing degrades to a
    usable spec rather than raising (see
    :func:`aiclipper.pipelines.common.resolve_voice`).
    """
    mine = common.resolve_voice(voice or settings.tts_voice or DEFAULT_VOICE, settings=settings)
    fallback = DEFAULT_REPLY_VOICE if mine.voice_id != DEFAULT_REPLY_VOICE else DEFAULT_VOICE
    theirs = common.resolve_voice(reply_voice or fallback, settings=settings)
    return mine, theirs


def _message_voice(
    message: ChatMessage, mine: VoiceSpec, theirs: VoiceSpec, settings: Settings
) -> VoiceSpec:
    """The voice for one message: its own override, else its side's voice."""
    override = (message.voice or "").strip()
    if override:
        return common.resolve_voice(override, settings=settings)
    return mine if message.outgoing else theirs


# --------------------------------------------------------------------------- #
# overlay rendering
# --------------------------------------------------------------------------- #

class _FallbackWatch(logging.Handler):
    """Notices :mod:`aiclipper.overlays` giving up on the browser backend.

    ``render_chat`` does not report which backend it used, but it logs a warning
    on its way from ``chromium`` down to ``pillow``.  Listening for that is the
    only honest way to put the backend that *actually* drew the frames in the
    result metadata.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.fell_back = False

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = record.getMessage()
        except Exception:  # pragma: no cover - a malformed log record is not our problem
            return
        if "falling back to pillow" in message:
            self.fell_back = True


def _render_states(
    script: ChatScript,
    out_dir: Path,
    *,
    width: int,
    height: int,
    backend: str | None,
    settings: Settings,
) -> tuple[list[overlays_module.OverlayImage], str]:
    """Render the conversation states and report the backend that drew them."""
    available = overlays_module.available_backends(settings)
    chosen = (backend or "").strip().lower() or (available[0] if available else "pillow")

    watch = _FallbackWatch()
    overlay_log = logging.getLogger(overlays_module.__name__)
    overlay_log.addHandler(watch)
    try:
        images = overlays_module.render_chat(
            script, out_dir, width=width, height=height, settings=settings, backend=backend
        )
    finally:
        overlay_log.removeHandler(watch)

    if watch.fell_back:
        chosen = "pillow"
    return images, chosen


# --------------------------------------------------------------------------- #
# keeping captions off the bubbles
# --------------------------------------------------------------------------- #

def overlay_bottom(images: Sequence[overlays_module.OverlayImage]) -> int:
    """The lowest row any chat state paints on, in canvas pixels.

    The bubble column is bottom-anchored, so this is where the conversation
    stops and the free band underneath it begins.  Unreadable or missing images
    contribute nothing rather than raising: a caption that cannot be placed
    precisely is still better than no video.
    """
    from PIL import Image

    bottom = 0
    for image in images:
        try:
            with Image.open(image.path) as handle:
                box = handle.convert("RGBA").getchannel("A").getbbox()
        except (OSError, ValueError) as exc:  # pragma: no cover - unreadable state PNG
            log.warning("could not measure chat state %s: %s", image.path, exc)
            continue
        if box:
            bottom = max(bottom, int(box[3]))
    return bottom


def caption_style_below_chat(
    style: CaptionStyle | None,
    images: Sequence[overlays_module.OverlayImage],
    *,
    width: int,
    height: int,
) -> CaptionStyle | None:
    """Move ``style`` into the clear band under the newest bubble.

    Burned captions are optional here because the bubbles already *are* the
    text, but when a caller asks for them they must not be stamped across the
    conversation.  Most presets sit either dead centre or a long way up from the
    bottom edge -- both land inside the bubble column -- so the block is pinned
    to the bottom and its margin tightened until it clears the lowest painted
    row.  A preset that already sits low enough is returned untouched.
    """
    if style is None or not images:
        return style

    bottom = overlay_bottom(images)
    if bottom <= 0 or bottom >= height:
        return style

    _, play_h = common.caption_resolution(width, height)
    bottom_ref = bottom * (play_h / float(height or 1))
    block = style.font_size * 1.32 * CAPTION_LINES + CAPTION_CLEARANCE
    allowed = int(play_h - bottom_ref - block)

    if allowed < CAPTION_MIN_MARGIN_V:
        log.warning("texts: the chat fills the canvas; captions may sit over the last bubble")
    margin_v = max(CAPTION_MIN_MARGIN_V, min(int(style.margin_v), allowed))
    if style.position == "bottom" and margin_v == int(style.margin_v):
        return style
    log.debug("texts: captions pinned to the bottom with margin_v=%d (bubbles end at %d)",
              margin_v, bottom)
    return replace(style, position="bottom", margin_v=margin_v)


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def _synthesise(
    script: ChatScript,
    work: Path,
    *,
    mine: VoiceSpec,
    theirs: VoiceSpec,
    settings: Settings,
    speaker: Any,
) -> tuple[list[TTSResult | None], list[Path | None], list[float]]:
    """Speak every spoken message into its own file.

    Returns three lists the length of ``script.messages``: the raw
    :class:`~aiclipper.models.TTSResult` (``None`` for a silent message), its
    audio path, and the hold each message takes on the clock.
    """
    vo_dir = work / "vo"
    vo_dir.mkdir(parents=True, exist_ok=True)

    results: list[TTSResult | None] = []
    paths: list[Path | None] = []
    holds: list[float] = []
    for index, message in enumerate(script.messages):
        text = (message.text or "").strip()
        if not message.read_aloud or not text:
            results.append(None)
            paths.append(None)
            holds.append(message_hold(message))
            continue
        spec = _message_voice(message, mine, theirs, settings)
        result = speaker.synthesize(text, vo_dir / f"msg_{index:03d}.wav", voice=spec)
        spoken = common.probe_duration(result.audio_path, settings=settings) or result.duration
        results.append(result)
        paths.append(Path(result.audio_path))
        holds.append(message_hold(message, spoken))
    return results, paths, holds


def _state_layer(
    image: overlays_module.OverlayImage, start: float, end: float, *, width: int, height: int
) -> VisualLayer:
    """One conversation state, held over the canvas for exactly its span.

    ``fit="contain"`` keeps the layer RGBA through the scale (the frames are
    already canvas sized, so nothing is actually padded) and the renderer gates
    it with ``enable='between(t,start,end)'``.
    """
    return VisualLayer(
        kind="image",
        src=str(image.path),
        start=float(start),
        end=float(end),
        x=0,
        y=0,
        w=int(width),
        h=int(height),
        fit="contain",
        opacity=1.0,
        take_audio=False,
        z=image.index + 1,
        label=f"chat:{image.index:03d}",
    )


def _message_report(script: ChatScript, beats: Sequence[Beat], voices: Sequence[str]) -> list[dict]:
    """Per-message timing, for :attr:`ProjectResult.metadata`."""
    report: list[dict] = []
    for beat, message in zip(beats, script.messages, strict=True):
        report.append({
            "index": beat.index,
            "sender": message.sender,
            "outgoing": bool(message.outgoing),
            "delay": round(beat.delay, 3),
            "typing": round(0.0 if beat.typing_start is None else beat.start - beat.typing_start, 3),
            "start": round(beat.start, 3),
            "end": round(beat.end, 3),
            "duration": round(beat.hold, 3),
            "spoken": beat.spoken,
            "voice": voices[beat.index] if beat.index < len(voices) else "",
        })
    return report


def run(
    topic: str | None = None,
    *,
    script: str | ChatScript | None = None,
    theme: str = "classic",
    voice: str = "",
    reply_voice: str = "",
    background: str | None = None,
    music: str | None = None,
    backend: str | None = None,
    turns: int = 10,
    captions: bool = False,
    style: str = "clean",
    out_path: Path | None = None,
    settings: Settings | None = None,
    provider: Any = None,
) -> ProjectResult:
    """Turn ``topic`` (or ``script``) into one text-conversation short.

    ``script`` may be a :class:`~aiclipper.models.ChatScript`, a block of text in
    the :func:`aiclipper.scriptgen.parse_chat` form, or ``None`` to have a
    conversation written about ``topic``.  Reading a file (or ``"-"`` for stdin)
    is the CLI's job, not this function's.

    ``voice`` speaks the outgoing side and ``reply_voice`` the incoming one; a
    message's own :attr:`~aiclipper.models.ChatMessage.voice` overrides both.
    ``backend`` picks the overlay renderer (``"chromium"``/``"pillow"``); the
    default tries the browser and falls back.  Captions are **off** by default:
    the bubbles already carry the text.

    Raises :class:`~aiclipper.errors.AiclipperError` when there is no
    conversation to show, :class:`~aiclipper.errors.AssetError` when no
    background or music can be resolved, :class:`~aiclipper.errors.OverlayError`
    when the states cannot be drawn, and
    :class:`~aiclipper.errors.RenderError` when the timeline will not render.
    """
    s = _cfg(settings)
    chat, source, provider_name = build_chat(
        topic, script, turns=turns, settings=s, provider=provider
    )
    caption_style = common.caption_style(style, captions)
    chat = replace(chat, theme=_resolve_theme(theme, chat))
    title = chat.title or (topic or "").strip() or chat.contact or "Untitled Thread"
    log.info("texts: %d messages from a %s conversation (%s theme)",
             len(chat.messages), source, chat.theme)

    work = common.workspace(f"{KIND}-{title}", s)
    width, height = common.canvas_size(s)

    mine, theirs = _side_voices(voice, reply_voice, s)
    speaker = tts.get_provider(settings=s)
    results, paths, holds = _synthesise(
        chat, work, mine=mine, theirs=theirs, settings=s, speaker=speaker
    )

    beats = plan_beats(chat.messages, holds, audio=paths)
    duration = round(beats[-1].end + TAIL_SECONDS, 3)

    images, used_backend = _render_states(
        chat, work / "states", width=width, height=height, backend=backend, settings=s
    )
    spans = state_spans(chat, beats, duration)
    if len(images) != len(spans):  # pragma: no cover - both come from chat_states
        raise AiclipperError(
            f"the overlay produced {len(images)} states but the clock has {len(spans)} spans"
        )

    spoken = [(r, b) for r, b in zip(results, beats, strict=True) if r is not None]
    words = common.narration_words(
        [r for r, _ in spoken], [b.start for _, b in spoken], settings=s
    )

    if not assets_module.library(s):
        assets_module.ensure_placeholders(s)
    background_asset = assets_module.pick_background(background, settings=s)
    music_asset = assets_module.pick_music(music, settings=s)

    timeline = Timeline(width=width, height=height, fps=s.fps, duration=duration, title=title)
    timeline.add_visual(
        common.background_layer(background_asset, width=width, height=height,
                                duration=duration, settings=s)
    )
    for image, (begin, end) in zip(images, spans, strict=True):
        timeline.add_visual(_state_layer(image, begin, end, width=width, height=height))
    for beat in beats:
        if beat.audio is not None:
            timeline.add_audio(
                common.voice_track(beat.audio, start=beat.start, label=f"voice:{beat.index:03d}")
            )
    timeline.add_audio(
        common.music_track(music_asset, duration=duration, gain_db=MUSIC_GAIN_DB,
                           duck=timeline.has_voice)
    )
    timeline.subtitles = common.caption_track(
        words, work / "captions",
        style=caption_style_below_chat(caption_style, images, width=width, height=height),
        width=width, height=height, enabled=captions, settings=s,
    )

    out = common.resolve_output(out_path, title, s)
    rendered = render_module.render(timeline, out, settings=s)

    voice_names = [
        (_message_voice(m, mine, theirs, s).voice_id or "default") for m in chat.messages
    ]
    return ProjectResult(
        output=rendered.path,
        kind=KIND,
        title=title,
        duration=rendered.duration or duration,
        transcript=Transcript.from_words(words) if words else None,
        metadata={
            "topic": (topic or "").strip(),
            "script_source": source,
            "provider": provider_name,
            "tts_provider": common.provider_name(speaker),
            "theme": chat.theme,
            "backend": used_backend,
            "backend_requested": (backend or "").strip().lower() or "auto",
            "contact": chat.contact,
            "messages": len(chat.messages),
            "spoken_messages": sum(1 for b in beats if b.spoken),
            "states": len(spans),
            "voices": {
                "outgoing": mine.voice_id or mine.provider,
                "incoming": theirs.voice_id or theirs.provider,
            },
            "voice_request": voice,
            "reply_voice_request": reply_voice,
            "message_timings": _message_report(chat, beats, voice_names),
            "background": background_asset.name,
            "music": music_asset.name,
            "captions": bool(captions and timeline.subtitles is not None),
            "style": common.style_name(caption_style),
            "words": len(words),
            "tail_seconds": TAIL_SECONDS,
            "timeline_seconds": duration,
            "script": asdict(chat),
        },
    )
