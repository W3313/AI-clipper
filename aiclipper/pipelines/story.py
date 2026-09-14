"""The ``story`` pipeline: a topic (or a supplied script) becomes a narrated short.

This is the AI video generator.  A caller hands in a subject -- or their own
words -- and gets back a finished vertical video: a written script, spoken over
a looping background loop, with a music bed ducked under the voice and the
narration burned in as captions.

The order of work, and why:

1. :func:`aiclipper.scriptgen.write_script` (or :func:`~aiclipper.scriptgen.parse_script`
   when the user supplied the text) turns the request into a
   :class:`~aiclipper.models.VideoScript` budgeted to ``seconds``.
2. :func:`aiclipper.tts.synthesize_lines` speaks one file per line, then
   :func:`aiclipper.pipelines.common.concat_audio` glues them into a single
   narration track and reports where each line landed inside it.
3. :func:`aiclipper.pipelines.common.narration_words` puts every word on that
   same clock, so captions cannot drift from the voice.
4. The library supplies a background loop and a music bed, generating
   procedural placeholders first when there is nothing on disk.
5. Everything becomes one :class:`~aiclipper.models.Timeline` and one
   :func:`aiclipper.render.render` call.

**The clock is the narration, not the request.**  ``seconds`` only budgets how
many words get written; the finished video is as long as the voice actually
turned out to be, plus :data:`TAIL_SECONDS` of air after the last word.  A
backend that speaks faster or slower than its own estimate therefore cannot
leave a gap of silence at the end or cut the final word off.

Nothing here imports an optional third-party dependency and nothing here
touches the network: with the heuristic LLM provider and the offline speech
backend the whole pipeline runs in a container with no egress.
"""

from __future__ import annotations

import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .. import assets as assets_module
from .. import render as render_module
from .. import scriptgen, tts
from ..config import Settings, get_settings
from ..errors import AiclipperError
from ..models import ProjectResult, Timeline, Transcript, VideoScript
from . import common

log = logging.getLogger(__name__)

__all__ = ["KIND", "LINE_GAP", "MUSIC_GAIN_DB", "TAIL_SECONDS", "build_script", "run"]

#: ``ProjectResult.kind`` for every video this module produces.
KIND = "story"

#: Silence between two narrated lines.  The same value is handed to
#: :func:`aiclipper.tts.synthesize_lines` and to
#: :func:`~aiclipper.pipelines.common.concat_audio` so the word timings that come
#: back from the first agree with the offsets that come back from the second.
LINE_GAP = 0.18

#: Air after the last narrated word, in seconds.  Deliberately shorter than
#: :data:`aiclipper.pipelines.common.TAIL_SECONDS`: a talking short wants to end
#: on the beat, not to sit on a silent frame.
TAIL_SECONDS = 0.6

#: The music bed sits well under the narration even before ducking kicks in.
MUSIC_GAIN_DB = -22.0


# --------------------------------------------------------------------------- #
# script sourcing
# --------------------------------------------------------------------------- #

def _cfg(settings: Settings | None) -> Settings:
    return settings or get_settings()


def build_script(
    topic: str | None,
    script: str | VideoScript | None,
    *,
    seconds: int = 35,
    tone: str = "punchy",
    settings: Settings | None = None,
    provider: Any = None,
) -> tuple[VideoScript, str, str]:
    """Return ``(script, source, provider_name)`` for this request.

    ``source`` is ``"supplied"`` when the caller brought their own words (a
    :class:`~aiclipper.models.VideoScript` is used as-is, a string is parsed) and
    ``"generated"`` when the language-model provider wrote it.  A supplied script
    never reaches a provider, so an offline caller with their own text never even
    constructs one.

    Raises :class:`~aiclipper.errors.AiclipperError` when neither a topic nor a
    usable script was given, or when the result has nothing to narrate.
    """
    s = _cfg(settings)
    subject = (topic or "").strip()

    if script is not None and not isinstance(script, (str, VideoScript)):
        raise AiclipperError(f"script must be text or a VideoScript, got {type(script).__name__}")

    if isinstance(script, VideoScript):
        result, source, provider_name = script, "supplied", ""
    elif isinstance(script, str) and script.strip():
        result, source, provider_name = scriptgen.parse_script(script), "supplied", ""
    elif subject:
        prov = common.llm_provider(provider, s)
        result = scriptgen.write_script(subject, seconds=max(3, int(seconds)), tone=tone,
                                        settings=s, provider=prov)
        source, provider_name = "generated", common.provider_name(prov)
    else:
        raise AiclipperError("the story pipeline needs a topic or a script")

    if not result.lines:
        raise AiclipperError("the story script has no narration to speak")
    return result, source, provider_name


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def run(
    topic: str | None = None,
    *,
    script: str | VideoScript | None = None,
    seconds: int = 35,
    voice: str = "",
    background: str | None = None,
    music: str | None = None,
    style: str = "bold_yellow",
    captions: bool = True,
    out_path: Path | None = None,
    settings: Settings | None = None,
    provider: Any = None,
) -> ProjectResult:
    """Turn ``topic`` (or ``script``) into one narrated short and return it.

    ``script`` may be a :class:`~aiclipper.models.VideoScript`, a block of text
    to parse, or ``None`` to have one written about ``topic``.  Reading a file
    (or ``"-"`` for stdin) is the CLI's job, not this function's.

    ``seconds`` budgets the *script*; the rendered video lasts as long as the
    narration really did plus :data:`TAIL_SECONDS`.  ``captions=False`` skips
    caption generation entirely -- no ASS file is written and the timeline
    carries no subtitle track.

    Raises :class:`~aiclipper.errors.AiclipperError` when there is nothing to
    narrate, :class:`~aiclipper.errors.AssetError` when no background or music
    can be resolved, and :class:`~aiclipper.errors.RenderError` when the timeline
    will not render.
    """
    s = _cfg(settings)
    video_script, source, provider_name = build_script(
        topic, script, seconds=seconds, settings=s, provider=provider
    )
    caption_style = common.caption_style(style, captions)
    lines = video_script.lines
    title = video_script.title or (topic or "").strip() or "Untitled Short"
    log.info("story: %d narration lines from a %s script", len(lines), source)

    work = common.workspace(f"{KIND}-{title}", s)
    voice_spec = common.resolve_voice(voice, settings=s)
    speaker = tts.get_provider(settings=s)

    results = tts.synthesize_lines(lines, work / "vo", voice=voice_spec, provider=speaker,
                                   gap=LINE_GAP, settings=s)
    narration, offsets = common.concat_audio(
        [r.audio_path for r in results], work / "narration.wav", gap=LINE_GAP, settings=s
    )
    words = common.narration_words(
        results, offsets, audio=narration, text=video_script.narration, settings=s
    )

    spoken = common.probe_duration(narration, settings=s) or tts.total_duration(results, LINE_GAP)
    duration = round(max(spoken, common.words_span(words)) + TAIL_SECONDS, 3)

    if not assets_module.library(s):
        assets_module.ensure_placeholders(s)
    background_asset = assets_module.pick_background(background, settings=s)
    music_asset = assets_module.pick_music(music, settings=s)

    width, height = common.canvas_size(s)
    timeline = Timeline(width=width, height=height, fps=s.fps, duration=duration, title=title)
    timeline.add_visual(
        common.background_layer(background_asset, width=width, height=height,
                                duration=duration, settings=s)
    )
    timeline.add_audio(common.voice_track(narration))
    timeline.add_audio(
        common.music_track(music_asset, duration=duration, gain_db=MUSIC_GAIN_DB, duck=True)
    )
    timeline.subtitles = common.caption_track(
        words, work / "captions", style=caption_style, width=width, height=height,
        enabled=captions, settings=s,
    )

    out = common.resolve_output(out_path, title, s)
    rendered = render_module.render(timeline, out, settings=s)

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
            "voice": voice_spec.voice_id or voice_spec.provider,
            "voice_request": voice,
            "background": background_asset.name,
            "music": music_asset.name,
            "style": common.style_name(caption_style),
            "captions": bool(captions and timeline.subtitles is not None),
            "words": len(words),
            "seconds_requested": int(seconds),
            "narration_seconds": round(spoken, 3),
            "timeline_seconds": duration,
            "narration": video_script.narration,
            "script": asdict(video_script),
        },
    )
