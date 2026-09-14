"""End-to-end tests for :mod:`aiclipper.pipelines.story`.

Every test here drives the *whole* pipeline offline -- heuristic LLM provider,
offline speech backend, procedurally generated library -- at a 180x320 canvas,
and asserts against files that really exist on disk.  Nothing is stubbed except
where a test needs to see the timeline the pipeline built (a spy that still
calls the real renderer) or needs to prove that a code path was *not* taken.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aiclipper import assets as assets_module
from aiclipper import config, scriptgen
from aiclipper import ffmpeg as ff
from aiclipper.errors import AiclipperError
from aiclipper.models import ProjectResult, Timeline, VideoScript
from aiclipper.pipelines import story

pytestmark = pytest.mark.needs_ffmpeg

#: A script short enough that the narration is unmistakably not ``seconds`` long.
SHORT_SCRIPT = """# Salt Water Clock
Nobody checks the water first.
That one habit saved the whole trip.
CTA: Follow for more.
"""


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
    real = story.render_module.render

    def _spy(timeline, out_path, **kwargs):
        recorder.timelines.append(timeline)
        result = real(timeline, out_path, **kwargs)
        recorder.commands.append(list(result.command))
        return result

    monkeypatch.setattr(story.render_module, "render", _spy)
    return recorder


def _probe(result: ProjectResult):
    assert result.output.exists(), f"{result.output} was never written"
    assert result.output.suffix == ".mp4"
    assert result.output.stat().st_size > 2000
    return ff.probe(result.output)


def _narration_path(settings) -> Path:
    found = sorted(settings.work_dir.glob("story-*/narration.wav"))
    assert found, "the pipeline wrote no narration file"
    return found[-1]


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #

def test_story_end_to_end_produces_a_real_video(env, spy: RenderSpy):
    result = story.run(
        "why tiny habits win", seconds=8, voice="narrator_deep",
        background="soft_drift", music="warm_bed", style="neon",
    )

    info = _probe(result)
    assert info.has_video and info.has_audio, "a narrated short needs both streams"
    assert (info.width, info.height) == (180, 320)
    assert result.kind == "story"
    assert result.title
    assert info.duration == pytest.approx(result.duration, abs=0.15)

    # the clock: the real narration file plus the tail, not the requested seconds
    spoken = ff.probe(_narration_path(env)).duration
    assert spoken > 0
    assert result.duration == pytest.approx(spoken + story.TAIL_SECONDS, abs=0.25)
    assert spy.timeline.duration == pytest.approx(spoken + story.TAIL_SECONDS, abs=0.05)

    # the transcript carries the narration the caller can search
    assert result.transcript is not None and result.transcript.words
    assert len(result.transcript.words) == result.metadata["words"]


def test_metadata_reports_everything_a_caller_needs(env):
    result = story.run(
        "why tiny habits win", seconds=8, voice="narrator_deep",
        background="soft_drift", music="warm_bed", style="neon",
    )
    meta = result.metadata

    assert meta["topic"] == "why tiny habits win"
    assert meta["provider"] == "heuristic"
    assert meta["tts_provider"] == "offline"
    assert meta["voice"] == "narrator_deep"
    assert meta["background"] == "soft_drift"
    assert meta["music"] == "warm_bed"
    assert meta["style"] == "neon"
    assert meta["captions"] is True
    assert meta["script_source"] == "generated"
    assert meta["words"] > 0
    assert meta["seconds_requested"] == 8

    script = meta["script"]
    assert isinstance(script, dict)
    assert set(script) >= {"title", "hook", "beats", "cta", "hashtags"}
    assert script["hook"]
    assert json.dumps(meta)  # the whole thing is JSON-serialisable for a CLI/report

    # the assets named in the metadata are the ones the timeline actually used
    narration_words = " ".join(w.text for w in (result.transcript.words if result.transcript else []))
    assert narration_words.split()[:2] == script["hook"].split()[:2]


def test_chosen_assets_land_on_the_timeline(env, spy: RenderSpy):
    story.run("a quiet morning", seconds=6, background="soft_drift", music="warm_bed")

    layers = spy.timeline.visuals
    assert len(layers) == 1
    assert layers[0].src.endswith("soft_drift.mp4")
    assert layers[0].label == "bg:soft_drift"
    assert (layers[0].w, layers[0].h) == (180, 320)
    assert layers[0].fit == "cover"
    # the 3s asset is shorter than the timeline, so it has to loop
    assert layers[0].loop is True

    music = [t for t in spy.timeline.audio if t.role == "music"]
    assert [t.label for t in music] == ["music:warm_bed"]


# --------------------------------------------------------------------------- #
# duration comes from the narration, not the request
# --------------------------------------------------------------------------- #

def test_duration_follows_the_narration_not_the_requested_seconds(env, spy: RenderSpy):
    # A four-line script asked to fill 40 seconds: the narration is nowhere near
    # that long, and the video must follow the voice rather than the budget.
    result = story.run(script=SHORT_SCRIPT, seconds=40)

    spoken = ff.probe(_narration_path(env)).duration
    assert spoken < 20, "the supplied script should be much shorter than the budget"
    assert result.duration == pytest.approx(spoken + story.TAIL_SECONDS, abs=0.25)
    assert abs(result.duration - 40) > 10, "the request must not drive the duration"
    assert result.metadata["narration_seconds"] == pytest.approx(spoken, abs=0.05)
    assert _probe(result).duration == pytest.approx(result.duration, abs=0.15)


# --------------------------------------------------------------------------- #
# captions
# --------------------------------------------------------------------------- #

def test_captions_are_burned_and_the_style_is_honoured(env, spy: RenderSpy):
    result = story.run("why tiny habits win", seconds=6, style="bold_yellow")

    track = spy.timeline.subtitles
    assert track is not None
    ass = Path(track.ass_path)
    assert ass.exists() and ass.stat().st_size > 200
    body = ass.read_text(encoding="utf-8")
    assert "[Events]" in body and "Dialogue:" in body
    assert "subtitles=" in spy.command
    assert result.metadata["style"] == "bold_yellow"
    assert result.metadata["captions"] is True


def test_captions_false_skips_caption_generation_entirely(env, spy: RenderSpy):
    result = story.run("why tiny habits win", seconds=6, captions=False)

    assert spy.timeline.subtitles is None, "captions=False must leave the timeline unsubtitled"
    assert "subtitles=" not in spy.command
    assert not list(env.work_dir.rglob("*.ass")), "no ASS file should have been written"
    assert result.metadata["captions"] is False
    assert result.metadata["style"] == ""
    assert _probe(result).has_video


def test_captions_true_and_false_differ_only_in_the_subtitle_track(env, spy: RenderSpy):
    story.run("why tiny habits win", seconds=6, out_path=env.output_dir / "with.mp4")
    story.run("why tiny habits win", seconds=6, captions=False,
              out_path=env.output_dir / "without.mp4")

    with_captions, without = spy.timelines
    assert with_captions.subtitles is not None and without.subtitles is None
    assert with_captions.duration == pytest.approx(without.duration, abs=0.01)
    assert [layer.src for layer in with_captions.visuals] == [layer.src for layer in without.visuals]
    assert [t.role for t in with_captions.audio] == [t.role for t in without.audio]


def test_an_unknown_caption_style_fails_before_any_speech_is_synthesised(env, monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("synthesis started despite a bad style")

    monkeypatch.setattr(story.tts, "synthesize_lines", _boom)
    with pytest.raises(ValueError):
        story.run("why tiny habits win", seconds=6, style="not-a-real-preset")


# --------------------------------------------------------------------------- #
# the mix: ducked music under the voice
# --------------------------------------------------------------------------- #

def test_music_is_ducked_under_the_voice(env, spy: RenderSpy):
    story.run("why tiny habits win", seconds=6, background="soft_drift", music="warm_bed")
    timeline = spy.timeline

    assert timeline.validate() == []
    assert timeline.has_voice

    voice = [t for t in timeline.audio if t.role == "voice"]
    music = [t for t in timeline.audio if t.role == "music"]
    assert len(voice) == 1 and len(music) == 1
    assert voice[0].duck is False and voice[0].start == 0.0
    assert voice[0].src.endswith("narration.wav")
    assert music[0].duck is True, "the bed must sidechain against the narration"
    assert music[0].gain_db == story.MUSIC_GAIN_DB
    assert music[0].gain_db < voice[0].gain_db - 10, "music must sit well below the voice"
    assert music[0].end == pytest.approx(timeline.duration, abs=0.01)
    assert "sidechaincompress" in spy.command


# --------------------------------------------------------------------------- #
# supplied scripts
# --------------------------------------------------------------------------- #

def test_a_supplied_script_bypasses_generation(env, monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("write_script was called for a supplied script")

    monkeypatch.setattr(story.scriptgen, "write_script", _boom)

    result = story.run(script=SHORT_SCRIPT, seconds=12)

    assert result.metadata["script_source"] == "supplied"
    assert result.metadata["provider"] == ""
    assert result.title == "Salt Water Clock"
    assert "Nobody checks the water first." in result.metadata["narration"]
    assert result.metadata["script"]["cta"] == "Follow for more."
    assert _probe(result).has_audio


def test_a_videoscript_object_is_used_verbatim(env, spy: RenderSpy):
    script = VideoScript(title="Hand Written", hook="One line only.", cta="")
    result = story.run(script=script, seconds=30)

    assert result.metadata["script"]["title"] == "Hand Written"
    assert result.metadata["narration"] == "One line only."
    assert result.metadata["words"] == 3
    assert spy.timeline.title == "Hand Written"
    assert "hand-written" in result.output.name


def test_a_topic_alongside_a_script_keeps_the_script_and_records_the_topic(env):
    result = story.run("ignored topic", script=SHORT_SCRIPT, seconds=10)

    assert result.metadata["script_source"] == "supplied"
    assert result.metadata["topic"] == "ignored topic"
    assert result.title == "Salt Water Clock"


# --------------------------------------------------------------------------- #
# build_script on its own
# --------------------------------------------------------------------------- #

def test_build_script_requires_a_topic_or_a_script(env):
    with pytest.raises(AiclipperError, match="topic or a script"):
        story.build_script(None, None, settings=env)
    with pytest.raises(AiclipperError, match="topic or a script"):
        story.build_script("   ", "  ", settings=env)


def test_build_script_rejects_a_thing_that_is_not_a_script(env):
    with pytest.raises(AiclipperError, match="VideoScript"):
        story.build_script(None, Path("/tmp/script.txt"), settings=env)  # type: ignore[arg-type]


def test_build_script_rejects_a_silent_script(env):
    with pytest.raises(AiclipperError, match="no narration"):
        story.build_script(None, VideoScript(title="Silent"), settings=env)


def test_build_script_generates_from_a_topic_and_names_the_provider(env):
    script, source, provider = story.build_script("why tiny habits win", None, seconds=8, settings=env)

    assert isinstance(script, VideoScript)
    assert script.lines and script.narration.strip()
    assert (source, provider) == ("generated", "heuristic")
    assert scriptgen.estimate_seconds(script) == pytest.approx(8, abs=6)


# --------------------------------------------------------------------------- #
# output naming and the asset library
# --------------------------------------------------------------------------- #

def test_an_explicit_out_path_is_honoured(env, tmp_path: Path):
    target = tmp_path / "renders" / "my story.mp4"
    result = story.run("why tiny habits win", seconds=6, out_path=target)

    assert result.output == target
    assert target.exists()


def test_repeat_runs_do_not_clobber_the_previous_file(env):
    first = story.run("why tiny habits win", seconds=6)
    second = story.run("why tiny habits win", seconds=6)

    assert first.output != second.output
    assert first.output.exists() and second.output.exists()
    assert second.output.name.endswith("-2.mp4")


def test_an_empty_library_is_generated_before_picking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env, spy: RenderSpy,
):
    """With nothing on disk the pipeline generates placeholders rather than failing."""
    for existing in list(env.backgrounds_dir.iterdir()) + list(env.music_dir.iterdir()):
        existing.unlink()
    assert assets_module.library(env) == []

    real = assets_module._generate_placeholders
    calls: list[str] = []

    def _tiny(settings=None, **kwargs):
        calls.append("generate")
        kwargs.update(width=180, height=320, fps=12, background_seconds=0.5, music_seconds=0.5)
        return real(settings, **kwargs)

    monkeypatch.setattr(assets_module, "_generate_placeholders", _tiny)

    result = story.run("why tiny habits win", seconds=6)

    assert calls, "ensure_placeholders should have run for an empty library"
    names = {(a.kind, a.name) for a in assets_module.library(env)}
    assert result.metadata["background"] in {n for k, n in names if k == "background"}
    assert result.metadata["music"] in {n for k, n in names if k == "music"}
    assert _probe(result).has_video
    assert spy.timeline.validate() == []


def test_a_stocked_library_is_never_regenerated(env, monkeypatch: pytest.MonkeyPatch):
    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("placeholders were generated over a stocked library")

    monkeypatch.setattr(assets_module, "_generate_placeholders", _boom)
    result = story.run("why tiny habits win", seconds=6)

    assert result.metadata["background"] == "soft_drift"
    assert result.metadata["music"] == "warm_bed"


# --------------------------------------------------------------------------- #
# voices
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("requested", ["", "definitely-not-a-voice"])
def test_a_missing_or_unknown_voice_still_renders(env, requested: str):
    result = story.run("why tiny habits win", seconds=6, voice=requested)

    assert result.metadata["voice_request"] == requested
    assert result.metadata["voice"]  # degraded to a usable spec rather than raising
    assert _probe(result).has_audio


def test_word_timings_stay_inside_the_narration(env, spy: RenderSpy):
    result = story.run("why tiny habits win", seconds=8)

    assert result.transcript is not None
    words = result.transcript.words
    spoken = ff.probe(_narration_path(env)).duration
    assert words[0].start >= 0.0
    assert words[-1].end <= spoken + 0.05, "a caption outlived its audio"
    assert words[-1].end <= spy.timeline.duration
