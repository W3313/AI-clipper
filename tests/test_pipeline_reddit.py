"""End-to-end tests for :mod:`aiclipper.pipelines.reddit`.

Every test drives the *whole* pipeline offline -- heuristic LLM provider, offline
speech backend, Pillow overlay backend, procedurally generated asset library --
at a 180x320 canvas, and asserts against files that really exist on disk.
Nothing is stubbed except where a test needs to inspect the timeline the pipeline
built (a spy that still calls the real renderer) or needs to prove that a code
path was *not* taken.

The assertions that matter most are about the two clocks meeting: the card must
leave the screen exactly when the body's voice arrives, and the captions -- which
cover the body only, because the card is already showing the title -- must start
at that very second and never print a word of the title.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aiclipper import config
from aiclipper import ffmpeg as ff
from aiclipper.errors import AiclipperError
from aiclipper.models import ProjectResult, RedditPost, Timeline
from aiclipper.pipelines import reddit

pytestmark = pytest.mark.needs_ffmpeg

#: A hand-written post.  "Kettle" appears in the title and nowhere in the body,
#: so a caption containing it proves the title leaked into the caption track.
POST = RedditPost(
    community="r/quiettales",
    author="u/deskplant",
    title="The Kettle That Ran My Whole Street",
    body=(
        "It started when the power went out on a Tuesday evening.\n\n"
        "By morning nine neighbours were queueing on my doorstep with mugs."
    ),
    upvotes=41200,
    comments=938,
    theme="dark",
)

_DIALOGUE = re.compile(
    r"^Dialogue:\s*\d+,(?P<start>[\d:.]+),(?P<end>[\d:.]+),[^,]*,[^,]*,"
    r"[^,]*,[^,]*,[^,]*,[^,]*,(?P<text>.*)$"
)


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #

@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A tiny canvas, a throwaway asset tree and a two-item starter library."""
    monkeypatch.setenv("AICLIP_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("AICLIP_WIDTH", "180")
    monkeypatch.setenv("AICLIP_HEIGHT", "320")
    monkeypatch.setenv("AICLIP_FPS", "12")
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    monkeypatch.setenv("AICLIP_SEED", "1234")
    config.reset_settings()
    settings = config.get_settings().ensure_dirs()

    ff.run_ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc2=s=120x214:r=12:d=3", "-t", "3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(settings.backgrounds_dir / "soft_drift.mp4"),
    ], settings=settings)
    ff.make_tone(4.0, settings.music_dir / "warm_bed.wav", frequency=180.0)

    yield settings
    config.reset_settings()


@dataclass
class RenderSpy:
    """Records the timeline handed to the renderer, then renders it for real."""

    timelines: list[Timeline] = field(default_factory=list)
    commands: list[list[str]] = field(default_factory=list)

    @property
    def timeline(self) -> Timeline:
        assert len(self.timelines) == 1, f"expected exactly one render, saw {len(self.timelines)}"
        return self.timelines[0]

    @property
    def command(self) -> str:
        assert len(self.commands) == 1
        return " ".join(self.commands[0])


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> RenderSpy:
    recorder = RenderSpy()
    real = reddit.render_module.render

    def _spy(timeline, out_path, **kwargs):
        recorder.timelines.append(timeline)
        result = real(timeline, out_path, **kwargs)
        recorder.commands.append(list(result.command))
        return result

    monkeypatch.setattr(reddit.render_module, "render", _spy)
    return recorder


def _run(**kwargs) -> ProjectResult:
    """The pipeline with the library assets and the fast overlay backend pinned."""
    kwargs.setdefault("post", POST)
    kwargs.setdefault("background", "soft_drift")
    kwargs.setdefault("music", "warm_bed")
    kwargs.setdefault("backend", "pillow")
    return reddit.run(**kwargs)


def _probe(result: ProjectResult):
    assert result.output.exists(), f"{result.output} was never written"
    assert result.output.suffix == ".mp4"
    assert result.output.stat().st_size > 2000
    return ff.probe(result.output)


def _work(settings) -> Path:
    found = sorted(p.parent for p in settings.work_dir.glob("reddit-*/narration.wav"))
    assert found, "the pipeline wrote no narration file"
    return found[-1]


def _part_durations(settings) -> list[float]:
    parts = sorted((_work(settings) / "vo").glob("line_*.wav"))
    assert parts, "the pipeline synthesised no narration parts"
    return [ff.probe(p).duration for p in parts]


def _card_layer(timeline: Timeline):
    cards = [layer for layer in timeline.visuals if layer.label == "card:forum"]
    assert len(cards) == 1, f"expected exactly one card layer, saw {len(cards)}"
    return cards[0]


def _cues(ass_path: Path) -> list[tuple[float, float, str]]:
    """Every ``Dialogue:`` line of an ASS file as ``(start, end, text)``."""
    out: list[tuple[float, float, str]] = []
    for line in Path(ass_path).read_text(encoding="utf-8").splitlines():
        match = _DIALOGUE.match(line.strip())
        if match:
            out.append((_seconds(match["start"]), _seconds(match["end"]), match["text"]))
    assert out, f"{ass_path} carries no dialogue lines"
    return out


def _seconds(stamp: str) -> float:
    hours, minutes, seconds = stamp.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #

def test_reddit_end_to_end_produces_a_real_video(env, spy: RenderSpy):
    result = _run(voice="narrator_deep", style="bold_yellow")

    info = _probe(result)
    assert info.has_video and info.has_audio, "a narrated story needs both streams"
    assert (info.width, info.height) == (180, 320)
    assert result.kind == "reddit"
    assert result.title == POST.title
    assert info.duration == pytest.approx(result.duration, abs=0.15)
    assert spy.timeline.validate() == []


def test_duration_is_the_title_plus_the_body_plus_the_tail(env, spy: RenderSpy):
    result = _run()

    parts = _part_durations(env)
    assert len(parts) == 3, "one part for the title and one per body paragraph"
    title, *body = parts
    expected = title + sum(body) + reddit.LINE_GAP * (len(parts) - 1) + reddit.TAIL_SECONDS

    assert result.duration == pytest.approx(expected, abs=0.1)
    assert spy.timeline.duration == pytest.approx(expected, abs=0.1)
    assert result.metadata["title_seconds"] == pytest.approx(title, abs=0.05)
    assert result.metadata["narration_seconds"] == pytest.approx(expected - reddit.TAIL_SECONDS, abs=0.1)
    assert _probe(result).duration == pytest.approx(expected, abs=0.2)


# --------------------------------------------------------------------------- #
# the card
# --------------------------------------------------------------------------- #

def test_the_card_spans_the_title_read_and_then_cuts_out(env, spy: RenderSpy):
    result = _run()

    title_seconds = _part_durations(env)[0]
    card = _card_layer(spy.timeline)

    assert card.kind == "image"
    assert card.start == 0.0, "the card is up from the first frame"
    assert card.end is not None
    # it holds for the title read and the breath after it, then the body starts
    assert card.end == pytest.approx(title_seconds + reddit.LINE_GAP, abs=0.02)
    assert card.end == pytest.approx(result.metadata["body_offset"], abs=1e-6)
    assert card.end == pytest.approx(result.metadata["card_seconds"], abs=1e-6)
    assert card.end < spy.timeline.duration, "the card must leave before the video ends"
    # ...and the background stays for the whole video underneath it
    background = [layer for layer in spy.timeline.visuals if layer.label.startswith("bg:")]
    assert len(background) == 1
    assert background[0].start == 0.0
    assert background[0].end == pytest.approx(spy.timeline.duration, abs=1e-6)
    assert background[0].z < card.z, "the card composites above the background"


def test_the_card_image_is_a_real_canvas_sized_transparent_png(env, spy: RenderSpy):
    from PIL import Image

    result = _run()
    card = _card_layer(spy.timeline)
    path = Path(card.src or "")

    assert path.exists() and path.suffix == ".png"
    assert str(path) == result.metadata["card_image"]
    with Image.open(path) as image:
        assert image.size == (180, 320)
        assert image.mode == "RGBA"
        assert image.getextrema()[3][0] == 0, "the card must be transparent around its edges"
    assert card.fit == "contain", "a transparent overlay must be padded, never cropped"


def test_card_seconds_overrides_the_measured_title_span(env, spy: RenderSpy):
    result = _run(card_seconds=1.25)

    card = _card_layer(spy.timeline)
    assert card.end == pytest.approx(1.25, abs=1e-6)
    assert result.metadata["card_seconds"] == pytest.approx(1.25, abs=1e-6)
    assert result.metadata["card_seconds_requested"] == 1.25
    # the narration clock is untouched -- only the card moved
    assert result.metadata["body_offset"] > 1.25
    assert _probe(result).has_video


def test_an_over_long_card_request_is_clamped_to_the_timeline(env, spy: RenderSpy):
    result = _run(card_seconds=900.0)

    card = _card_layer(spy.timeline)
    assert card.end == pytest.approx(spy.timeline.duration, abs=1e-6)
    assert spy.timeline.validate() == [], "a card outliving the timeline must not be emitted"
    assert result.metadata["card_seconds"] == pytest.approx(result.metadata["timeline_seconds"], abs=1e-3)


def test_a_zero_card_request_still_leaves_a_renderable_layer(env, spy: RenderSpy):
    _run(card_seconds=0.0)

    card = _card_layer(spy.timeline)
    assert card.end == pytest.approx(reddit.MIN_CARD_SECONDS, abs=1e-6)
    assert card.end > card.start, "a zero-length layer would fail validation"
    assert spy.timeline.validate() == []


def test_the_theme_reaches_the_card_renderer(env, monkeypatch: pytest.MonkeyPatch):
    seen: list[str] = []
    real = reddit.overlays_module.render_forum_card

    def _spy(post, out_path, **kwargs):
        seen.append(post.theme)
        return real(post, out_path, **kwargs)

    monkeypatch.setattr(reddit.overlays_module, "render_forum_card", _spy)

    result = _run(theme="paper")
    assert seen == ["paper"]
    assert result.metadata["theme"] == "paper"
    assert POST.theme == "dark", "the caller's post must never be mutated"


# --------------------------------------------------------------------------- #
# captions cover the body, never the title
# --------------------------------------------------------------------------- #

def test_captions_start_at_the_body_and_never_show_the_title(env, spy: RenderSpy):
    result = _run(style="clean")

    track = spy.timeline.subtitles
    assert track is not None
    cues = _cues(Path(track.ass_path))
    body_offset = result.metadata["body_offset"]
    assert body_offset > 0.5, "the title read has to be long enough for this test to mean anything"

    # ASS stamps are centisecond-resolution, hence the small tolerance
    assert min(start for start, _, _ in cues) >= body_offset - 0.02
    for start, end, text in cues:
        assert start >= body_offset - 0.02, f"a cue at {start}s starts before the body at {body_offset}s"
        assert end <= spy.timeline.duration + 0.05
        assert "kettle" not in text.lower(), f"the title leaked into a caption: {text!r}"

    spoken_body = " ".join(cue[2].lower() for cue in cues)
    assert "neighbours" in spoken_body, "the body itself must actually be captioned"
    assert "subtitles=" in spy.command


def test_the_caption_words_are_exactly_the_body_words(env, spy: RenderSpy):
    result = _run(style="clean")

    assert result.transcript is not None
    every_word = [w.text.lower().strip(".,!?") for w in result.transcript.words]
    body_offset = result.metadata["body_offset"]
    late = [w for w in result.transcript.words if w.start >= body_offset - 0.02]

    assert "kettle" in every_word, "the transcript carries the whole narration"
    assert result.metadata["words"] == len(result.transcript.words)
    assert result.metadata["caption_words"] == len(late)
    assert result.metadata["caption_words"] < result.metadata["words"]
    assert all("kettle" not in w.text.lower() for w in late)

    # what was actually burned in: the body's tokens, and nothing else
    track = spy.timeline.subtitles
    assert track is not None
    burned = {
        token.lower().strip(".,!?")
        for _, _, text in _cues(Path(track.ass_path))
        for token in re.sub(r"\{[^}]*\}", " ", text).replace("\\N", " ").split()
    }
    burned.discard("")
    body_tokens = {token.lower().strip(".,!?") for token in POST.body.split()}
    title_only = {token.lower().strip(".,!?") for token in POST.title.split()} - body_tokens
    assert burned <= body_tokens, f"a caption carried words the body never had: {burned - body_tokens}"
    assert not burned & title_only, f"title-only words were burned in: {burned & title_only}"
    assert len(burned) > 5


def test_a_pinned_card_holds_the_captions_back_instead_of_printing_over_it(env, spy: RenderSpy):
    """A card pinned past the body read must never have captions stamped on it.

    Most presets -- ``clean`` among them -- centre their text, which is exactly
    where the card sits, so an overlap is unreadable rather than merely untidy.
    """
    pinned_at = 5.0
    result = _run(style="clean", card_seconds=pinned_at)

    assert result.metadata["body_offset"] < pinned_at, "the pin has to outlast the title read"
    assert result.metadata["narration_seconds"] > pinned_at, "and still land inside the body"

    card = _card_layer(spy.timeline)
    assert card.end == pytest.approx(pinned_at, abs=1e-6), "the pinned card still wins"

    track = spy.timeline.subtitles
    assert track is not None, "the body after the card still gets captions"
    cues = _cues(Path(track.ass_path))
    assert min(start for start, _, _ in cues) >= pinned_at - 0.02
    assert 0 < result.metadata["caption_words"] < result.metadata["words"]


def test_an_unpinned_card_leaves_every_body_word_captioned(env, spy: RenderSpy):
    """The default path is untouched: the card goes as the body arrives."""
    result = _run(style="clean")

    card = _card_layer(spy.timeline)
    assert card.end == pytest.approx(result.metadata["body_offset"], abs=1e-6)
    assert result.metadata["caption_words"] == len(
        [w for w in result.transcript.words if w.start >= result.metadata["body_offset"] - 0.02]
    )


@pytest.mark.parametrize(
    ("card_end", "offset", "kept"),
    [(2.0, 2.0, 3), (2.0, 1.999, 3), (4.0, 2.0, 1), (0.0, 2.0, 3)],
)
def test_uncovered_words_only_trims_a_card_that_outlasts_the_body(card_end, offset, kept):
    from aiclipper.models import Word

    words = [Word("a", 2.0, 2.5), Word("b", 3.0, 3.5), Word("c", 4.0, 4.5)]
    assert len(reddit._uncovered_words(words, card_end, offset)) == kept


def test_captions_false_skips_caption_generation_entirely(env, spy: RenderSpy):
    result = _run(captions=False)

    assert spy.timeline.subtitles is None
    assert "subtitles=" not in spy.command
    assert not list(env.work_dir.rglob("*.ass")), "no ASS file should have been written"
    assert result.metadata["captions"] is False
    assert result.metadata["style"] == ""
    assert _probe(result).has_video


def test_an_unknown_caption_style_fails_before_any_speech_is_synthesised(env, monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("synthesis started despite a bad style")

    monkeypatch.setattr(reddit.tts, "synthesize_lines", _boom)
    with pytest.raises(ValueError):
        _run(style="not-a-real-preset")


# --------------------------------------------------------------------------- #
# the mix
# --------------------------------------------------------------------------- #

def test_music_is_ducked_under_the_one_narration_track(env, spy: RenderSpy):
    _run()
    timeline = spy.timeline

    voice = [t for t in timeline.audio if t.role == "voice"]
    music = [t for t in timeline.audio if t.role == "music"]
    assert len(voice) == 1 and len(music) == 1
    assert voice[0].start == 0.0 and voice[0].duck is False
    assert voice[0].src.endswith("narration.wav")
    assert music[0].duck is True, "the bed must sidechain against the narration"
    assert music[0].gain_db == reddit.MUSIC_GAIN_DB
    assert music[0].label == "music:warm_bed"
    assert music[0].end == pytest.approx(timeline.duration, abs=0.01)
    assert "sidechaincompress" in spy.command


# --------------------------------------------------------------------------- #
# supplied posts vs generated ones
# --------------------------------------------------------------------------- #

def test_a_supplied_post_bypasses_generation(env, monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("write_reddit was called for a supplied post")

    monkeypatch.setattr(reddit.scriptgen, "write_reddit", _boom)

    result = _run(topic="ignored topic")

    assert result.metadata["post_source"] == "supplied"
    assert result.metadata["provider"] == ""
    assert result.metadata["topic"] == "ignored topic"
    assert result.metadata["community"] == "r/quiettales"
    assert result.metadata["author"] == "u/deskplant"
    assert result.title == POST.title
    assert _probe(result).has_audio


def test_a_generated_post_names_its_provider_and_is_narrated(env):
    result = reddit.run(
        "the night the lift broke", words=30, background="soft_drift", music="warm_bed",
        backend="pillow",
    )

    meta = result.metadata
    assert meta["post_source"] == "generated"
    assert meta["provider"] == "heuristic"
    assert meta["community"].startswith("r/")
    assert meta["author"].startswith("u/")
    assert meta["title"]
    assert meta["body_paragraphs"] >= 1
    assert meta["caption_words"] > 0
    assert _probe(result).has_audio


def test_metadata_is_complete_and_json_serialisable(env):
    result = _run(voice="narrator_deep", style="neon")
    meta = result.metadata

    assert meta["community"] == "r/quiettales"
    assert meta["author"] == "u/deskplant"
    assert meta["title"] == POST.title
    assert meta["theme"] == "dark"
    assert meta["voice"] == "narrator_deep"
    assert meta["voice_request"] == "narrator_deep"
    assert meta["style"] == "neon"
    assert meta["captions"] is True
    assert meta["tts_provider"] == "offline"
    assert meta["background"] == "soft_drift"
    assert meta["music"] == "warm_bed"
    assert meta["card_backend"] == "pillow"
    assert meta["card_seconds"] > 0
    assert meta["upvotes"] == 41200 and meta["comments"] == 938
    assert meta["post"]["body"] == POST.body
    assert POST.title in meta["narration"] and "neighbours" in meta["narration"]
    assert json.dumps(meta)


# --------------------------------------------------------------------------- #
# build_post and body_parts on their own
# --------------------------------------------------------------------------- #

def test_build_post_requires_a_topic_or_a_post(env):
    with pytest.raises(AiclipperError, match="topic or a post"):
        reddit.build_post(None, None, settings=env)
    with pytest.raises(AiclipperError, match="topic or a post"):
        reddit.build_post("   ", None, settings=env)


def test_build_post_rejects_a_thing_that_is_not_a_post(env):
    with pytest.raises(AiclipperError, match="RedditPost"):
        reddit.build_post(None, "a string story", settings=env)  # type: ignore[arg-type]


def test_build_post_rejects_a_silent_post(env):
    with pytest.raises(AiclipperError, match="nothing to narrate"):
        reddit.build_post(None, RedditPost(community="r/x", author="u/y"), settings=env)


def test_build_post_fills_a_missing_title_and_keeps_the_caller_intact(env):
    bare = RedditPost(body="the lift broke again and nobody called anyone about it")
    post, source, provider = reddit.build_post("a broken lift", bare, settings=env)

    assert (source, provider) == ("supplied", "")
    assert post.title == "A broken lift"
    assert bare.title == "", "the caller's post must never be mutated"


def test_build_post_keeps_a_posts_own_theme_when_the_theme_is_blank(env):
    post, _, _ = reddit.build_post(None, RedditPost(title="T", body="b b b", theme="paper"),
                                   theme="", settings=env)
    assert post.theme == "paper"


def test_build_post_generates_from_a_topic(env):
    post, source, provider = reddit.build_post("a broken lift", None, words=40, settings=env)

    assert isinstance(post, RedditPost)
    assert (source, provider) == ("generated", "heuristic")
    assert post.title and post.body
    assert 20 <= len(post.body.split()) <= 120


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("one\n\ntwo\n\n\nthree", ["one", "two", "three"]),
        ("  \n\n  ", []),
        ("a single\nwrapped paragraph", ["a single wrapped paragraph"]),
        ("", []),
    ],
)
def test_body_parts_splits_on_blank_lines(body: str, expected: list[str]):
    assert reddit.body_parts(body) == expected


def test_one_paragraph_bodies_still_split_title_from_body(env, spy: RenderSpy):
    post = RedditPost(community="r/x", author="u/y", title="A Very Short Title",
                      body="One paragraph is all there is to this story.")
    result = _run(post=post)

    assert result.metadata["body_paragraphs"] == 1
    assert _card_layer(spy.timeline).end == pytest.approx(result.metadata["body_offset"], abs=1e-6)
    assert result.metadata["body_offset"] > 0
    assert spy.timeline.subtitles is not None


# --------------------------------------------------------------------------- #
# output naming
# --------------------------------------------------------------------------- #

def test_an_explicit_out_path_is_honoured(env, tmp_path: Path):
    target = tmp_path / "renders" / "my story.mp4"
    result = _run(out_path=target)

    assert result.output == target
    assert target.exists()


def test_repeat_runs_do_not_clobber_the_previous_file(env):
    first = _run()
    second = _run()

    assert first.output != second.output
    assert first.output.exists() and second.output.exists()
    assert second.output.name.endswith("-2.mp4")
