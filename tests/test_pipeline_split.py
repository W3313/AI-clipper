"""End-to-end tests for :mod:`aiclipper.pipelines.split`.

Sources are built on the fly with ffmpeg at deliberately awkward aspect ratios
(4:3 on top, 9:16 underneath) so a crop that targeted the canvas ratio instead
of the *pane* ratio could not possibly pass.  Everything runs offline at a
180x320 canvas with a two-item asset library seeded into a tmp dir.

Two fixtures drive the pipeline: ``plan`` swaps the renderer for a real
``dry_run`` build (the graph is still validated, every source file still has to
exist, no encode happens) so timeline assertions stay cheap, and ``spy`` records
the timeline while really encoding, for the tests that must prove a file lands
on disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from aiclipper import config
from aiclipper import ffmpeg as ff
from aiclipper.errors import AssetError, IngestError
from aiclipper.models import ProjectResult, Timeline
from aiclipper.pipelines import split

pytestmark = pytest.mark.needs_ffmpeg

CANVAS_W = 180
CANVAS_H = 320

#: 180x160 panes -- landscape, and nothing like the 180x320 canvas.
PANE_ASPECT = 180 / 160
CANVAS_ASPECT = CANVAS_W / CANVAS_H

NARRATION = "Two panes. One clock. Nobody notices the seam."


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #

@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A tiny canvas plus a seeded asset library, so nothing is ever generated."""
    monkeypatch.setenv("AICLIP_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("AICLIP_WIDTH", str(CANVAS_W))
    monkeypatch.setenv("AICLIP_HEIGHT", str(CANVAS_H))
    monkeypatch.setenv("AICLIP_FPS", "12")
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    monkeypatch.setenv("AICLIP_SEED", "4321")
    config.reset_settings()
    settings = config.get_settings().ensure_dirs()

    ff.run_ffmpeg([
        "-y", "-f", "lavfi", "-i", "gradients=s=120x214:r=12:d=3", "-t", "3",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(settings.backgrounds_dir / "slow_drift.mp4"),
    ], settings=settings)
    ff.make_tone(4.0, settings.music_dir / "warm_bed.wav", frequency=180.0)

    yield settings
    config.reset_settings()


@pytest.fixture
def wide(make_video) -> Path:
    """A 4:3 clip with audio -- the usual "podcast on top" source."""
    return make_video("wide.mp4", seconds=2.0, width=320, height=240, fps=12)


@pytest.fixture
def tall(make_video) -> Path:
    """An already-vertical clip with audio, for the lower pane."""
    return make_video("tall.mp4", seconds=2.0, width=180, height=320, fps=12,
                      source="testsrc2=size=180x320:rate=12")


@pytest.fixture
def silent_wide(make_video) -> Path:
    return make_video("mute.mp4", seconds=2.0, width=320, height=240, fps=12, audio=False)


@dataclass
class Recorder:
    """Captures every timeline handed to the renderer."""

    timelines: list[Timeline] = field(default_factory=list)

    @property
    def timeline(self) -> Timeline:
        assert len(self.timelines) == 1, f"expected one render, saw {len(self.timelines)}"
        return self.timelines[0]

    @property
    def layers(self) -> list:
        return self.timeline.ordered_visuals

    @property
    def top(self):
        return self.layers[0]

    @property
    def bottom(self):
        return self.layers[1]


@pytest.fixture
def plan(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Record the timeline and build the real ffmpeg command without encoding."""
    recorder = Recorder()
    real = split.render_module.render

    def _dry(timeline, out_path, **kwargs):
        recorder.timelines.append(timeline)
        kwargs["dry_run"] = True
        return real(timeline, out_path, **kwargs)

    monkeypatch.setattr(split.render_module, "render", _dry)
    return recorder


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch) -> Recorder:
    """Record the timeline and render it for real."""
    recorder = Recorder()
    real = split.render_module.render

    def _spy(timeline, out_path, **kwargs):
        recorder.timelines.append(timeline)
        return real(timeline, out_path, **kwargs)

    monkeypatch.setattr(split.render_module, "render", _spy)
    return recorder


def _aspect(layer) -> float:
    assert layer.crop is not None and layer.crop.keyframes, f"{layer.label} has no crop window"
    first = layer.crop.keyframes[0]
    return first.w / first.h


# --------------------------------------------------------------------------- #
# geometry: the panes tile the canvas
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("width", [180, 181, 1080])
@pytest.mark.parametrize("height", [4, 6, 320, 321, 322, 962, 1920])
def test_pane_rects_tile_the_canvas_with_no_gap_or_overlap(width: int, height: int):
    top, bottom = split.pane_rects(width, height)
    canvas_w = width - width % 2
    canvas_h = height - height % 2

    # every dimension even -- yuv420p and the renderer's crop both insist
    for pane in (top, bottom):
        assert pane.x % 2 == 0 and pane.y % 2 == 0
        assert pane.w % 2 == 0 and pane.h % 2 == 0
        assert pane.w == canvas_w
        assert pane.h >= 2

    assert top.y == 0
    assert bottom.y == top.h, "the bottom pane must start exactly where the top one ends"
    assert top.h + bottom.h == canvas_h, "the panes must sum back to the canvas height"
    assert abs(top.h - bottom.h) <= 2, "the panes are halves, not an arbitrary split"


def test_pane_rects_reject_a_canvas_with_nothing_to_split():
    with pytest.raises(ValueError):
        split.pane_rects(180, 3)
    with pytest.raises(ValueError):
        split.pane_rects(1, 320)


def test_pane_aspect_is_not_the_canvas_aspect():
    top, _ = split.pane_rects(1080, 1920)
    assert top.aspect == pytest.approx(1080 / 960)
    assert top.aspect != pytest.approx(1080 / 1920)


def test_an_odd_canvas_height_still_tiles_the_rendered_canvas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, env, wide: Path, plan: Recorder
):
    monkeypatch.setenv("AICLIP_HEIGHT", "321")
    config.reset_settings()

    split.run(wide, bottom="slow_drift", seconds=1.0)
    timeline = plan.timeline

    assert timeline.height == 320, "the renderer's canvas is always evened first"
    top, bottom = (layer.rect(timeline.width, timeline.height) for layer in plan.layers)
    assert top[1] == 0 and bottom[1] == top[3]
    assert top[3] + bottom[3] == timeline.height
    assert top[3] % 2 == 0 and bottom[3] % 2 == 0
    assert not timeline.validate()


# --------------------------------------------------------------------------- #
# the layers
# --------------------------------------------------------------------------- #

def test_layers_occupy_the_two_halves(env, wide: Path, tall: Path, plan: Recorder):
    split.run(wide, bottom=tall)
    timeline = plan.timeline

    assert len(timeline.visuals) == 2
    assert plan.top.rect(CANVAS_W, CANVAS_H) == (0, 0, 180, 160)
    assert plan.bottom.rect(CANVAS_W, CANVAS_H) == (0, 160, 180, 160)
    assert [layer.fit for layer in plan.layers] == ["cover", "cover"], "panes are filled, not fitted"
    assert plan.top.src.endswith("wide.mp4")
    assert plan.bottom.src.endswith("tall.mp4")


def test_each_pane_crops_to_the_pane_aspect_not_the_canvas_aspect(
    env, wide: Path, tall: Path, plan: Recorder
):
    split.run(wide, bottom=tall)

    # 320x240 is wider than a 9:8 pane -> keep the height, lose the sides
    assert _aspect(plan.top) == pytest.approx(PANE_ASPECT, abs=0.02)
    assert plan.top.crop.keyframes[0].h == 240

    # 180x320 is *narrower* than the pane -> keep the width, lose top and bottom
    assert _aspect(plan.bottom) == pytest.approx(PANE_ASPECT, abs=0.02)
    assert plan.bottom.crop.keyframes[0].w == 180
    assert plan.bottom.crop.keyframes[0].h == 160

    for layer in plan.layers:
        assert _aspect(layer) != pytest.approx(CANVAS_ASPECT, abs=0.05), "cropped to the canvas ratio"
        first = layer.crop.keyframes[0]
        assert first.w % 2 == 0 and first.h % 2 == 0


def test_reframe_off_keeps_a_static_centred_crop(env, wide: Path, plan: Recorder):
    split.run(wide, bottom="slow_drift", reframe=False)

    assert plan.top.crop.is_static
    assert _aspect(plan.top) == pytest.approx(PANE_ASPECT, abs=0.02)


def test_a_short_source_loops_to_fill_a_longer_timeline(env, wide: Path, tall: Path, plan: Recorder):
    split.run(wide, bottom=tall, seconds=5.0)

    assert plan.timeline.duration == pytest.approx(5.0)
    assert plan.top.loop is True and plan.bottom.loop is True
    # a looping pane cannot be driven by a sendcmd script that never rewinds
    assert plan.top.crop.is_static


# --------------------------------------------------------------------------- #
# the bottom pane
# --------------------------------------------------------------------------- #

def test_a_missing_bottom_source_falls_back_to_a_library_background(env, wide: Path, plan: Recorder):
    result = split.run(wide)

    assert plan.bottom.label == "bg:slow_drift"
    assert Path(plan.bottom.src).parent == env.backgrounds_dir
    assert plan.bottom.rect(CANVAS_W, CANVAS_H) == (0, 160, 180, 160)
    assert plan.bottom.take_audio is False
    assert result.metadata["bottom_kind"] == "library"
    assert result.metadata["bottom_name"] == "slow_drift"


def test_a_bare_bottom_name_resolves_against_the_asset_library(env, wide: Path, plan: Recorder):
    result = split.run(wide, bottom="slow_drift")

    assert plan.bottom.src.endswith("slow_drift.mp4")
    assert result.metadata["bottom_kind"] == "library"


def test_an_unknown_bottom_name_is_an_error_not_a_silent_swap(env, wide: Path, plan: Recorder):
    with pytest.raises(AssetError):
        split.run(wide, bottom="no_such_asset")


def test_a_bottom_source_never_contributes_audio(env, wide: Path, tall: Path, plan: Recorder):
    split.run(wide, bottom=tall)

    assert plan.bottom.take_audio is False
    assert plan.top.take_audio is True, "the top pane is the one that keeps its sound"


# --------------------------------------------------------------------------- #
# narration
# --------------------------------------------------------------------------- #

def test_narration_lines_split_on_newlines_and_sentences():
    assert split.narration_lines("One two. Three four!\n\nFive six") == [
        "One two.", "Three four!", "Five six",
    ]
    assert split.narration_lines("   ") == []
    assert split.narration_lines(None) == []


def test_narration_mutes_the_top_pane_and_becomes_the_voice_track(env, wide: Path, plan: Recorder):
    result = split.run(wide, bottom="slow_drift", narration=NARRATION, voice="narrator_deep")
    timeline = plan.timeline

    assert plan.top.take_audio is False, "narration replaces the top pane's audio"
    voices = [t for t in timeline.audio if t.role == "voice"]
    assert len(voices) == 1
    assert Path(voices[0].src).name == "narration.wav"
    assert Path(voices[0].src).exists()
    assert voices[0].start == 0.0

    # captions come from the same word timings the voice was built on
    assert timeline.subtitles is not None
    assert timeline.subtitles.ass_path.exists()
    assert result.transcript is not None
    words = result.transcript.words
    assert len(words) == result.metadata["words"] > 0
    assert max(w.end for w in words) <= timeline.duration + 1e-6
    assert result.metadata["narration"] is True
    assert result.metadata["voice"] == "narrator_deep"
    assert result.metadata["narration_lines"] == 3


def test_without_narration_there_is_no_voice_track_and_no_captions(env, wide: Path, plan: Recorder):
    result = split.run(wide, bottom="slow_drift")

    assert [t.role for t in plan.timeline.audio] == []
    assert plan.timeline.subtitles is None
    assert plan.top.take_audio is True
    assert result.transcript is None
    assert result.metadata["narration"] is False
    assert result.metadata["captions"] is False


def test_captions_can_be_switched_off_while_narrating(env, wide: Path, plan: Recorder):
    result = split.run(wide, bottom="slow_drift", narration=NARRATION, captions=False)

    assert plan.timeline.subtitles is None
    assert [t.role for t in plan.timeline.audio] == ["voice"]
    assert result.metadata["captions"] is False
    assert not list((env.work_dir).glob("split-*/captions/*.ass"))


def test_a_silent_top_source_does_not_claim_audio(env, silent_wide: Path, spy: Recorder):
    result = split.run(silent_wide, bottom="slow_drift", seconds=1.0)

    assert spy.top.take_audio is False, "take_audio on a silent source would fail the render"
    assert result.metadata["top_audio"] is False
    assert result.output.exists()


# --------------------------------------------------------------------------- #
# duration precedence: seconds > narration > the shorter source
# --------------------------------------------------------------------------- #

def test_seconds_wins_over_narration_and_the_sources(env, wide: Path, plan: Recorder):
    result = split.run(wide, bottom="slow_drift", narration=NARRATION, seconds=1.5)

    assert plan.timeline.duration == pytest.approx(1.5)
    assert result.metadata["duration_source"] == "seconds"
    assert result.metadata["narration_seconds"] > 0, "the narration was still synthesised"


def test_narration_wins_over_the_sources(env, wide: Path, plan: Recorder):
    result = split.run(wide, bottom="slow_drift", narration=NARRATION)

    narration = env.work_dir / "split-wide" / "narration.wav"
    spoken = ff.probe(narration).duration
    assert spoken > 2.0, "this narration must outlast the 2s sources for the test to mean anything"
    assert plan.timeline.duration == pytest.approx(spoken + split.TAIL_SECONDS, abs=0.05)
    assert result.metadata["duration_source"] == "narration"
    assert result.metadata["narration_seconds"] == pytest.approx(spoken, abs=0.05)


def test_without_narration_the_shorter_source_sets_the_length(env, make_video, plan: Recorder):
    short = make_video("short.mp4", seconds=1.2, width=320, height=240, fps=12)
    long = make_video("long.mp4", seconds=3.0, width=240, height=320, fps=12,
                      source="testsrc2=size=240x320:rate=12")

    result = split.run(short, bottom=long)
    assert plan.timeline.duration == pytest.approx(1.2, abs=0.2)
    assert result.metadata["duration_source"] == "sources"

    plan.timelines.clear()
    result = split.run(long, bottom=short)
    assert plan.timeline.duration == pytest.approx(1.2, abs=0.2)


def test_a_library_bottom_never_shortens_the_timeline(env, wide: Path, plan: Recorder):
    # the seeded background is 3s, the top source 2s: the loop must not win
    result = split.run(wide, bottom="slow_drift")

    assert plan.timeline.duration == pytest.approx(2.0, abs=0.2)
    assert result.metadata["duration_source"] == "sources"


# --------------------------------------------------------------------------- #
# music
# --------------------------------------------------------------------------- #

def test_music_ducks_only_when_there_is_a_voice_to_duck_under(env, wide: Path, plan: Recorder):
    split.run(wide, bottom="slow_drift", music="warm_bed", seconds=1.0)
    silent = [t for t in plan.timeline.audio if t.role == "music"]
    assert len(silent) == 1
    assert silent[0].duck is False, "ducking with no voice track fails Timeline.validate()"
    assert not plan.timeline.validate()

    plan.timelines.clear()
    split.run(wide, bottom="slow_drift", music="warm_bed", narration=NARRATION, seconds=2.0)
    narrated = [t for t in plan.timeline.audio if t.role == "music"]
    assert narrated[0].duck is True
    assert narrated[0].end == pytest.approx(plan.timeline.duration), "the bed covers the timeline"
    assert narrated[0].loop is False, "the 4s bed already outlasts this 2s timeline"
    assert not plan.timeline.validate()


def test_no_music_is_added_unless_one_is_named(env, wide: Path, plan: Recorder):
    split.run(wide, bottom="slow_drift")
    assert [t for t in plan.timeline.audio if t.role == "music"] == []


# --------------------------------------------------------------------------- #
# bad input
# --------------------------------------------------------------------------- #

def test_a_missing_top_source_raises(env, tmp_path: Path):
    with pytest.raises(IngestError):
        split.run(tmp_path / "nope.mp4")


def test_an_audio_only_top_source_raises(env, make_audio, tmp_path: Path):
    with pytest.raises(IngestError):
        split.run(make_audio("tone.wav", seconds=1.0))


def test_an_audio_only_bottom_source_raises(env, wide: Path, make_audio):
    with pytest.raises(IngestError):
        split.run(wide, bottom=make_audio("bed.wav", seconds=1.0))


# --------------------------------------------------------------------------- #
# the real thing
# --------------------------------------------------------------------------- #

def test_split_end_to_end_produces_a_real_two_pane_video(env, wide: Path, tall: Path, spy: Recorder):
    result = split.run(wide, bottom=tall, narration=NARRATION, voice="narrator_deep",
                       style="bold_yellow", music="warm_bed", seconds=2.0)

    assert isinstance(result, ProjectResult)
    assert result.output.exists() and result.output.suffix == ".mp4"
    assert result.output.stat().st_size > 2000

    info = ff.probe(result.output)
    assert (info.width, info.height) == (CANVAS_W, CANVAS_H)
    assert info.has_video and info.has_audio
    assert info.duration == pytest.approx(2.0, abs=0.2)
    assert result.duration == pytest.approx(info.duration, abs=0.2)
    assert result.kind == "split"
    assert not spy.timeline.validate()

    # both panes really are in the graph, at their own rects
    assert spy.top.rect(CANVAS_W, CANVAS_H) == (0, 0, 180, 160)
    assert spy.bottom.rect(CANVAS_W, CANVAS_H) == (0, 160, 180, 160)


def test_the_rendered_pixels_really_are_two_panes(env, make_video, tmp_path: Path):
    """Decode the output and look: a red top half over a blue bottom half."""
    from PIL import Image

    top = make_video("red.mp4", seconds=1.5, fps=12, audio=False,
                     source="color=c=red:size=320x240:rate=12")
    bottom = make_video("blue.mp4", seconds=1.5, fps=12, audio=False,
                        source="color=c=blue:size=180x320:rate=12")

    result = split.run(top, bottom=bottom, seconds=1.0)
    frame = tmp_path / "frame.png"
    ff.run_ffmpeg(["-y", "-ss", "0.5", "-i", str(result.output), "-frames:v", "1", str(frame)],
                  settings=env)

    image = Image.open(frame).convert("RGB")
    assert image.size == (CANVAS_W, CANVAS_H)
    seam = split.pane_rects(CANVAS_W, CANVAS_H)[0].h
    for y, name in ((4, "top"), (seam - 3, "top"), (seam + 2, "bottom"), (CANVAS_H - 4, "bottom")):
        r, g, b = image.getpixel((CANVAS_W // 2, y))
        if name == "top":
            assert r > 140 and b < 90, f"y={y} should be the red pane, got {(r, g, b)}"
        else:
            assert b > 140 and r < 90, f"y={y} should be the blue pane, got {(r, g, b)}"


def test_metadata_reports_what_a_caller_needs(env, wide: Path, tall: Path, plan: Recorder):
    result = split.run(wide, bottom=tall, narration=NARRATION, voice="narrator_deep",
                       style="neon", music="warm_bed", seconds=2.0)
    meta = result.metadata

    assert meta["top"] == str(wide)
    assert meta["bottom"] == str(tall)
    assert meta["bottom_kind"] == "source"
    assert meta["panes"] == {"top": [0, 0, 180, 160], "bottom": [0, 160, 180, 160]}
    assert meta["pane_aspect"] == pytest.approx(PANE_ASPECT)
    assert meta["canvas"] == [CANVAS_W, CANVAS_H]
    assert meta["fps"] == 12
    assert meta["narration"] is True
    assert meta["narration_text"] == NARRATION
    assert meta["voice"] == "narrator_deep"
    assert meta["tts_provider"] == "offline"
    assert meta["captions"] is True
    assert meta["style"] == "neon"
    assert meta["music"] == "warm_bed"
    assert meta["top_audio"] is False
    assert meta["duration_source"] == "seconds"
    assert meta["timeline_seconds"] == pytest.approx(2.0)
    assert meta["crop"]["top"].startswith("track") or meta["crop"]["top"] == "center"
    assert json.dumps(meta), "metadata must survive a CLI dumping it to JSON"


def test_output_path_is_honoured_exactly(env, wide: Path, tmp_path: Path, spy: Recorder):
    target = tmp_path / "renders" / "stacked.mp4"
    result = split.run(wide, bottom="slow_drift", out_path=target, seconds=1.0)

    assert result.output == target
    assert target.exists()


def test_a_generated_name_lands_in_the_output_dir(env, wide: Path, plan: Recorder):
    result = split.run(wide, bottom="slow_drift", seconds=1.0)

    assert result.output.parent == env.output_dir
    assert result.output.name == "wide-split.mp4"


# --------------------------------------------------------------------------- #
# captions and the pane seam
# --------------------------------------------------------------------------- #

def test_a_centred_preset_is_re_seated_into_the_bottom_pane():
    """The seam runs along the canvas centre, which is where a centred preset sits."""
    from aiclipper.captions import PRESETS
    from aiclipper.pipelines.split import caption_style_clear_of_seam

    clean = PRESETS["clean"]
    assert clean.position == "center", "this test is about the centred default"

    moved = caption_style_clear_of_seam(clean, width=1080, height=1920)
    assert moved.position == "bottom"
    # The whole block has to live below the seam, i.e. in the lower half.
    play_h = 1920
    block = moved.font_size * 1.32 * split.CAPTION_LINES + split.CAPTION_CLEARANCE
    assert play_h - moved.margin_v - block >= play_h / 2, "the block still crosses the seam"
    # Nothing else about the look changes.
    assert moved.font_size == clean.font_size
    assert moved.primary_color == clean.primary_color
    assert moved.animation == clean.animation


def test_a_preset_that_already_clears_the_seam_is_untouched():
    from aiclipper.captions import PRESETS
    from aiclipper.pipelines.split import caption_style_clear_of_seam

    lower = PRESETS["subtle_lower"]
    assert lower.position == "bottom"
    assert caption_style_clear_of_seam(lower, width=1080, height=1920) is lower
    assert caption_style_clear_of_seam(None, width=1080, height=1920) is None


def test_the_rendered_split_burns_its_captions_below_the_seam(env, wide: Path, plan: Recorder):
    """End to end: the .ass the pipeline wrote must not be centred on the canvas."""
    split.run(wide, bottom="slow_drift", narration=NARRATION)

    subtitles = plan.timeline.subtitles
    assert subtitles is not None
    lines = subtitles.ass_path.read_text(encoding="utf-8").splitlines()
    names = [f.strip() for f in next(ln for ln in lines if ln.startswith("Format:")).split(":", 1)[1].split(",")]
    style_line = next(ln for ln in lines if ln.startswith("Style:"))
    style = dict(zip(names, [f.strip() for f in style_line.split(":", 1)[1].split(",")], strict=True))

    # 2 = bottom-centre; 5 would be dead centre of the canvas, i.e. on the seam.
    assert int(style["Alignment"]) == 2, f"captions are still centred on the seam: {style_line}"
    assert int(style["MarginV"]) > 0
