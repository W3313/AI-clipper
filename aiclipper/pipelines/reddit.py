"""The ``reddit`` pipeline: a forum-style story, read aloud over a background loop.

The shape of the finished video is the point of the workflow:

* the **card** -- our own generic forum card, never a clone of a real site --
  sits over the background while the *title* is read, then cuts away;
* the **body** is narrated over the bare background, with the burned captions
  starting exactly where the body starts.

That split is the whole trick, and it is why this pipeline narrates in two
stages rather than one.  The title is spoken into its own file, the body into
one file per paragraph, and :func:`aiclipper.pipelines.common.concat_audio`
reports where each of those parts landed inside the joined narration.  The card
therefore ends, and the captions therefore begin, at the *measured* second the
body's voice starts -- not at an estimate, and not at zero.  Captioning from
zero would print the title underneath the card that is already showing it, which
is the one mistake this pipeline exists to avoid.  The same applies when a
caller pins the card past the start of the body with ``card_seconds``: the
captions wait for the card to leave rather than printing across it.

The order of work:

1. :func:`aiclipper.scriptgen.write_reddit` writes a
   :class:`~aiclipper.models.RedditPost` about ``topic`` -- unless the caller
   supplied one, in which case no provider is ever constructed.
2. :func:`aiclipper.tts.synthesize_lines` speaks ``[title, *paragraphs]``, and
   ``concat_audio`` glues them into one narration file plus a list of offsets.
3. :func:`aiclipper.pipelines.common.narration_words` puts the **body** parts'
   words on that clock; they are what the captions are built from.
4. :func:`aiclipper.overlays.render_forum_card` draws the card as a transparent
   canvas-sized PNG, placed as an image layer spanning ``[0, body offset]`` (or
   ``card_seconds`` when the caller pinned it).
5. Background, card, voice, ducked music and subtitles become one
   :class:`~aiclipper.models.Timeline` for :func:`aiclipper.render.render`.

The video lasts as long as the narration really turned out to be plus
:data:`TAIL_SECONDS`, so a faster or slower speech backend changes the length of
the video rather than desynchronising anything inside it.

Nothing here imports an optional third-party dependency and nothing here touches
the network: with the heuristic LLM provider, the offline speech backend and the
Pillow overlay backend the whole pipeline runs with no egress at all.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .. import assets as assets_module
from .. import overlays as overlays_module
from .. import render as render_module
from .. import scriptgen, tts
from ..config import Settings, get_settings
from ..errors import AiclipperError
from ..models import ProjectResult, RedditPost, Timeline, Transcript, VisualLayer, Word
from . import common

log = logging.getLogger(__name__)

__all__ = [
    "KIND",
    "LINE_GAP",
    "MUSIC_GAIN_DB",
    "TAIL_SECONDS",
    "MIN_CARD_SECONDS",
    "DEFAULT_WORDS",
    "CARD_Z",
    "body_parts",
    "build_post",
    "run",
]

#: ``ProjectResult.kind`` for every video this module produces.
KIND = "reddit"

#: Silence between two narration parts.  Handed to both
#: :func:`aiclipper.tts.synthesize_lines` and
#: :func:`~aiclipper.pipelines.common.concat_audio` so the word timings that come
#: back from the first agree with the offsets that come back from the second.
LINE_GAP = 0.22

#: Air after the last narrated word, in seconds.
TAIL_SECONDS = 0.7

#: The music bed sits well under the narration even before ducking kicks in.
MUSIC_GAIN_DB = -22.0

#: The card is never shown for less than this, however short the title read was.
MIN_CARD_SECONDS = 0.4

#: Default word budget handed to :func:`aiclipper.scriptgen.write_reddit`.
DEFAULT_WORDS = 180

#: The card composites above the background.
CARD_Z = 10

#: A card that outlasts the title read by less than this is not worth
#: rescheduling captions around -- it is rounding, not an overlap.
_CARD_OVERLAP_TOLERANCE = 0.05

#: Paragraph break: one or more blank lines.
_PARAGRAPH = re.compile(r"\n\s*\n+")

#: How many opening words of the body stand in for a missing title.
_TITLE_FALLBACK_WORDS = 9


# --------------------------------------------------------------------------- #
# post sourcing
# --------------------------------------------------------------------------- #

def _cfg(settings: Settings | None) -> Settings:
    return settings or get_settings()


def body_parts(body: str) -> list[str]:
    """Split a post body into the paragraphs that get narrated separately.

    Paragraphs are the natural unit: they give the speech backend a breath
    between them, they keep each synthesised file small, and they give
    :func:`~aiclipper.pipelines.common.narration_words` more anchor points to
    hang word timings on.  Blank paragraphs are dropped -- a part with no text
    would only stretch the clock with silence.
    """
    chunks = [chunk.strip() for chunk in _PARAGRAPH.split(body or "")]
    return [" ".join(chunk.split()) for chunk in chunks if chunk.strip()]


def _fallback_title(post: RedditPost, topic: str) -> str:
    """A title for a post that arrived without one."""
    subject = (topic or "").strip()
    if subject:
        return subject[:1].upper() + subject[1:]
    words = (post.body or "").split()[:_TITLE_FALLBACK_WORDS]
    return " ".join(words).rstrip(",.;:") or "Untitled Post"


def build_post(
    topic: str | None,
    post: RedditPost | None,
    *,
    words: int = DEFAULT_WORDS,
    theme: str = "dark",
    settings: Settings | None = None,
    provider: Any = None,
) -> tuple[RedditPost, str, str]:
    """Return ``(post, source, provider_name)`` for this request.

    ``source`` is ``"supplied"`` when the caller brought their own
    :class:`~aiclipper.models.RedditPost` and ``"generated"`` when the
    language-model provider wrote it.  A supplied post never reaches a provider,
    so an offline caller with their own story never even constructs one.

    The returned post is always a *copy*: the caller's object is never mutated,
    even though the theme (and possibly a missing title) is filled in here.  A
    blank ``theme`` keeps whatever theme the post already carried.

    Raises :class:`~aiclipper.errors.AiclipperError` when neither a topic nor a
    usable post was given, or when the post has nothing to narrate.
    """
    s = _cfg(settings)
    subject = (topic or "").strip()

    if post is not None and not isinstance(post, RedditPost):
        raise AiclipperError(f"post must be a RedditPost, got {type(post).__name__}")

    if isinstance(post, RedditPost):
        result, source, provider_name = post, "supplied", ""
    elif subject:
        prov = common.llm_provider(provider, s)
        result = scriptgen.write_reddit(subject, words=max(20, int(words)), settings=s, provider=prov)
        source, provider_name = "generated", common.provider_name(prov)
    else:
        raise AiclipperError("the reddit pipeline needs a topic or a post")

    if not result.narration.strip():
        raise AiclipperError("the forum post has nothing to narrate")

    title = (result.title or "").strip() or _fallback_title(result, subject)
    resolved_theme = (theme or "").strip() or (result.theme or "").strip() or "dark"
    return replace(result, title=title, theme=resolved_theme), source, provider_name


def _card_span(card_seconds: float | None, body_offset: float, duration: float) -> float:
    """When the card leaves the screen, in seconds on the narration clock.

    With no explicit request the card holds until the body's voice starts -- it
    is showing the title, so it belongs on screen for exactly as long as the
    title is being read (plus the breath after it).  An explicit ``card_seconds``
    wins, clamped so the layer is neither degenerate nor left hanging past the
    end of the video.
    """
    wanted = body_offset if card_seconds is None else float(card_seconds)
    return round(max(MIN_CARD_SECONDS, min(wanted, duration)), 3)


def _uncovered_words(body_words: Sequence[Word], card_end: float, body_offset: float) -> list[Word]:
    """The body words that are *not* hidden underneath the card.

    Most caption presets sit in the middle of the canvas -- which is exactly
    where the card sits -- so a card pinned past the start of the body read
    would have caption text stamped across its title.  When that happens the
    captions wait for the card to leave; in the normal case (the card goes the
    moment the body starts) every word is kept, untouched.
    """
    if card_end <= body_offset + _CARD_OVERLAP_TOLERANCE:
        return list(body_words)
    return [w for w in body_words if w.start >= card_end]


# --------------------------------------------------------------------------- #
# the pipeline
# --------------------------------------------------------------------------- #

def run(
    topic: str | None = None,
    *,
    post: RedditPost | None = None,
    theme: str = "dark",
    voice: str = "",
    background: str | None = None,
    music: str | None = None,
    style: str = "clean",
    card_seconds: float | None = None,
    out_path: Path | None = None,
    settings: Settings | None = None,
    provider: Any = None,
    captions: bool = True,
    words: int = DEFAULT_WORDS,
    backend: str | None = None,
) -> ProjectResult:
    """Turn ``topic`` (or a supplied ``post``) into one narrated forum-story short.

    ``theme`` picks the card's look from :data:`aiclipper.overlays.FORUM_THEMES`;
    pass ``""`` to keep the theme a supplied post already carried.  ``style`` is
    a caption preset, ``captions=False`` skips captions entirely, and ``backend``
    forces the overlay renderer (``"pillow"`` or ``"chromium"``; ``None`` prefers
    chromium and falls back silently).

    ``card_seconds`` pins how long the card stays up.  Left at ``None`` it is
    measured: the card leaves the screen the moment the body's narration begins.
    The captions cover the **body only** -- the card already shows the title --
    and start at that same second.

    Raises :class:`~aiclipper.errors.AiclipperError` when there is nothing to
    narrate, :class:`~aiclipper.errors.OverlayError` when the card cannot be
    drawn, :class:`~aiclipper.errors.AssetError` when no background or music can
    be resolved, and :class:`~aiclipper.errors.RenderError` when the timeline
    will not render.
    """
    s = _cfg(settings)
    story_post, source, provider_name = build_post(
        topic, post, words=words, theme=theme, settings=s, provider=provider
    )
    caption_style = common.caption_style(style, captions)

    title = story_post.title
    paragraphs = body_parts(story_post.body)
    body_text = "\n\n".join(paragraphs)
    log.info("reddit: %s post %r with %d body paragraph(s)", source, title, len(paragraphs))

    work = common.workspace(f"{KIND}-{title}", s)
    voice_spec = common.resolve_voice(voice, settings=s)
    speaker = tts.get_provider(settings=s)

    results = tts.synthesize_lines([title, *paragraphs], work / "vo", voice=voice_spec,
                                   provider=speaker, gap=LINE_GAP, settings=s)
    narration, offsets = common.concat_audio(
        [r.audio_path for r in results], work / "narration.wav", gap=LINE_GAP, settings=s
    )

    spoken = common.probe_duration(narration, settings=s) or tts.total_duration(results, LINE_GAP)
    title_seconds = common.probe_duration(results[0].audio_path, settings=s)
    body_offset = round(offsets[1] if len(offsets) > 1 else spoken, 3)

    title_words = common.narration_words(results[:1], offsets[:1], settings=s)
    body_words = common.narration_words(results[1:], offsets[1:], settings=s)
    if not body_words and body_text.strip():
        # Every backend failed to time the body: spread it across its own span so
        # the captions still animate -- but starting at the body, never at zero.
        body_words = common.proportional_words(body_text, body_offset, max(0.0, spoken - body_offset))

    duration = round(max(spoken, common.words_span(body_words)) + TAIL_SECONDS, 3)
    card_end = _card_span(card_seconds, body_offset, duration)

    width, height = common.canvas_size(s)
    card = overlays_module.render_forum_card(
        story_post, work / "card.png", width=width, height=height, settings=s, backend=backend
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
    timeline.add_visual(
        VisualLayer(
            kind="image",
            src=str(card.path),
            start=0.0,
            end=card_end,
            x=0,
            y=0,
            w=width,
            h=height,
            fit="contain",
            take_audio=False,
            z=CARD_Z,
            label="card:forum",
        )
    )
    timeline.add_audio(common.voice_track(narration))
    timeline.add_audio(
        common.music_track(music_asset, duration=duration, gain_db=MUSIC_GAIN_DB, duck=True)
    )
    caption_words = _uncovered_words(body_words, card_end, body_offset)
    if len(caption_words) != len(body_words):
        log.info("reddit: holding %d caption word(s) back until the card leaves at %.2fs",
                 len(body_words) - len(caption_words), card_end)
    timeline.subtitles = common.caption_track(
        caption_words, work / "captions", style=caption_style, width=width, height=height,
        enabled=captions, settings=s,
    )

    out = common.resolve_output(out_path, title, s)
    rendered = render_module.render(timeline, out, settings=s)

    spoken_words = title_words + body_words
    return ProjectResult(
        output=rendered.path,
        kind=KIND,
        title=title,
        duration=rendered.duration or duration,
        transcript=Transcript.from_words(spoken_words) if spoken_words else None,
        metadata={
            "topic": (topic or "").strip(),
            "post_source": source,
            "provider": provider_name,
            "tts_provider": common.provider_name(speaker),
            "community": story_post.community,
            "author": story_post.author,
            "title": title,
            "theme": story_post.theme,
            "upvotes": story_post.upvotes,
            "comments": story_post.comments,
            "voice": voice_spec.voice_id or voice_spec.provider,
            "voice_request": voice,
            "background": background_asset.name,
            "music": music_asset.name,
            "style": common.style_name(caption_style),
            "captions": bool(captions and timeline.subtitles is not None),
            "card_backend": backend or "auto",
            "card_image": str(card.path),
            "card_seconds": card_end,
            "card_seconds_requested": None if card_seconds is None else float(card_seconds),
            "title_seconds": round(title_seconds, 3),
            "body_offset": body_offset,
            "body_paragraphs": len(paragraphs),
            "narration_seconds": round(spoken, 3),
            "timeline_seconds": duration,
            "words": len(spoken_words),
            "caption_words": len(caption_words),
            "narration": story_post.narration,
            "post": asdict(story_post),
        },
    )
