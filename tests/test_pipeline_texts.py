"""End-to-end tests for :mod:`aiclipper.pipelines.texts`.

Every test drives the real pipeline offline -- heuristic LLM provider, offline
speech backend, Pillow overlay backend, procedurally generated library -- at a
180x320 canvas, and asserts against files that really exist on disk.  Nothing is
stubbed except where a test needs to see the timeline the pipeline built (a spy
that still calls the real renderer) or to prove a code path was *not* taken.

The clock is what these tests are really about: the states the overlay renders,
the spans the layers cover, and the instant each message's voice starts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from PIL import Image

from aiclipper import captions as captions_module
from aiclipper import config, overlays
from aiclipper import ffmpeg as ff
from aiclipper.errors import AiclipperError
from aiclipper.models import ChatMessage, ChatScript, ProjectResult, Timeline, VisualLayer
from aiclipper.pipelines import texts

pytestmark = pytest.mark.needs_ffmpeg

#: A four-message conversation with delays the parser understands and no typing.
PARSED_SCRIPT = """# Locked Out
Alex: [1s] are you home?
me: [0.5s] no why
Alex: [0.75s] the door was open when I got back
me: [0.5s] call me
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
    monkeypatch.setenv("AICLIP_SEED", "4321")
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
    real = texts.render_module.render

    def _spy(timeline, out_path, **kwargs):
        recorder.timelines.append(timeline)
        result = real(timeline, out_path, **kwargs)
        recorder.commands.append(list(result.command))
        return result

    monkeypatch.setattr(texts.render_module, "render", _spy)
    return recorder


def run(**kwargs) -> ProjectResult:
    """The pipeline with the test defaults: the library assets and Pillow."""
    kwargs.setdefault("background", "soft_drift")
    kwargs.setdefault("music", "warm_bed")
    kwargs.setdefault("backend", "pillow")
    return texts.run(**kwargs)


def typed_script() -> ChatScript:
    """A conversation with known delays and one deliberate typing beat."""
    return ChatScript(
        title="Night Shift",
        contact="Robin",
        theme="mint",
        messages=[
            ChatMessage(sender="Robin", text="you still up", delay=0.5),
            ChatMessage(sender="Me", text="yeah what happened", outgoing=True, delay=0.4),
            ChatMessage(sender="Robin", text="the lights in the yard came on by themselves",
                        delay=0.6, typing=0.9),
            ChatMessage(sender="Me", text="i am calling you", outgoing=True, delay=0.3),
        ],
    )


def chat_layers(timeline: Timeline) -> list[VisualLayer]:
    return [layer for layer in timeline.visuals if layer.label.startswith("chat:")]


def voice_tracks(timeline: Timeline):
    return sorted([t for t in timeline.audio if t.role == "voice"], key=lambda t: t.start)


def _alpha_bottom(path: Path) -> int:
    """The lowest row an overlay PNG paints on, measured independently."""
    with Image.open(path) as handle:
        box = handle.convert("RGBA").getchannel("A").getbbox()
    return int(box[3]) if box else 0


def _probe(result: ProjectResult):
    assert result.output.exists(), f"{result.output} was never written"
    assert result.output.suffix == ".mp4"
    assert result.output.stat().st_size > 2000
    return ff.probe(result.output)


# --------------------------------------------------------------------------- #
# the clock, on its own (no ffmpeg, no speech)
# --------------------------------------------------------------------------- #

def test_plan_beats_waits_for_the_delay_then_the_typing_beat_then_the_narration():
    messages = [
        ChatMessage(sender="A", text="one", delay=1.0),
        ChatMessage(sender="B", text="two", outgoing=True, delay=0.5, typing=0.8),
        ChatMessage(sender="A", text="three", delay=0.0),
    ]
    beats = texts.plan_beats(messages, [2.0, 1.0, 1.5])

    assert [b.index for b in beats] == [0, 1, 2]
    # message 0: no typing, one second of lead-in, two seconds of narration
    assert beats[0].typing_start is None
    assert (beats[0].start, beats[0].end) == (1.0, 3.0)
    # message 1: the delay, then the typing indicator, then the bubble + voice
    assert beats[1].typing_start == pytest.approx(3.5)
    assert beats[1].start == pytest.approx(4.3), "the bubble waits out the typing beat"
    assert beats[1].start - beats[1].typing_start == pytest.approx(0.8)
    assert beats[1].end == pytest.approx(5.3)
    # message 2: no delay at all, so it lands the instant the previous voice ends
    assert beats[2].start == pytest.approx(beats[1].end)
    assert beats[2].end == pytest.approx(6.8)
    assert [b.hold for b in beats] == pytest.approx([2.0, 1.0, 1.5])


def test_plan_beats_never_lets_a_bubble_flash_past():
    beats = texts.plan_beats([ChatMessage(sender="A", text="hi", delay=0.0)], [0.0])
    assert beats[0].hold == pytest.approx(texts.MIN_BUBBLE_SECONDS)


def test_state_spans_map_every_overlay_state_by_index():
    script = typed_script()
    holds = [1.0, 1.5, 2.0, 1.25]
    beats = texts.plan_beats(script.messages, holds)
    total = beats[-1].end + texts.TAIL_SECONDS
    states = overlays.chat_states(script)
    spans = texts.state_spans(script, beats, total)

    # one state per message plus the single typing frame, in overlay order
    assert [(s.visible, s.typing) for s in states] == [
        (1, False), (2, False), (2, True), (3, False), (4, False),
    ]
    assert len(spans) == len(states)

    # the typing frame occupies exactly the typing beat of message 2
    typing_span = spans[2]
    assert typing_span == (pytest.approx(beats[2].typing_start), pytest.approx(beats[2].start))
    assert typing_span[1] - typing_span[0] == pytest.approx(0.9)

    # every bubble frame starts exactly when its own message starts
    for position, state in enumerate(states):
        if not state.typing:
            assert spans[position][0] == pytest.approx(beats[state.visible - 1].start)

    # contiguous, no gaps, no overlaps, and the last one holds to the end
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == pytest.approx(next_start)
    assert all(end > start for start, end in spans)
    assert spans[-1][1] == pytest.approx(total)


# --------------------------------------------------------------------------- #
# the happy path
# --------------------------------------------------------------------------- #

def test_texts_end_to_end_produces_a_real_video(env, spy: RenderSpy):
    result = run(script=PARSED_SCRIPT, theme="dark")

    info = _probe(result)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (180, 320)
    assert result.kind == "texts"
    assert result.title == "Locked Out"
    assert info.duration == pytest.approx(result.metadata["timeline_seconds"], abs=0.2)
    assert spy.timeline.validate() == []
    assert result.metadata["messages"] == 4
    assert result.metadata["spoken_messages"] == 4


def test_state_count_equals_the_layer_count_and_the_spans_tile_the_timeline(env, spy: RenderSpy):
    script = typed_script()
    result = run(script=script)
    timeline = spy.timeline

    states = overlays.chat_states(script)
    layers = chat_layers(timeline)
    assert len(states) == 5, "four messages plus one typing frame"
    assert len(layers) == len(states) == result.metadata["states"]

    frames = sorted(Path(layer.src) for layer in layers)
    assert len(set(frames)) == len(frames), "every state has its own PNG"
    for path in frames:
        assert path.exists() and path.stat().st_size > 0

    spans = [(layer.start, layer.end) for layer in layers]
    assert spans == sorted(spans), "layers are added in state order"
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == pytest.approx(next_start), "a gap or an overlap between states"
    assert spans[-1][1] == pytest.approx(timeline.duration)
    assert all(end > start for start, end in spans)
    assert spans[0][0] == pytest.approx(0.5), "the first bubble waits out its own delay"


def test_the_typing_frame_covers_exactly_the_typing_beat(env, spy: RenderSpy):
    script = typed_script()
    result = run(script=script)

    states = overlays.chat_states(script)
    layers = chat_layers(spy.timeline)
    typing_positions = [i for i, state in enumerate(states) if state.typing]
    assert typing_positions == [2]

    typing_layer = layers[typing_positions[0]]
    bubble_layer = layers[typing_positions[0] + 1]
    assert typing_layer.end - typing_layer.start == pytest.approx(0.9, abs=0.01)
    assert typing_layer.end == pytest.approx(bubble_layer.start)

    timings = result.metadata["message_timings"]
    assert timings[2]["typing"] == pytest.approx(0.9)
    assert bubble_layer.start == pytest.approx(timings[2]["start"], abs=0.001)


def test_every_message_audio_starts_when_its_bubble_appears(env, spy: RenderSpy):
    script = typed_script()
    result = run(script=script)
    timeline = spy.timeline

    states = overlays.chat_states(script)
    layers = chat_layers(timeline)
    voices = voice_tracks(timeline)
    assert len(voices) == len(script.messages) == 4

    # independently recompute which layer shows message i's bubble
    for index, track in enumerate(voices):
        position = next(
            i for i, state in enumerate(states) if not state.typing and state.visible == index + 1
        )
        assert track.start == pytest.approx(layers[position].start), f"message {index} is out of sync"
        assert track.label == f"voice:{index:03d}"
        assert Path(track.src).exists()
        # the voice must finish before the *next* message's bubble arrives
        spoken = ff.probe(track.src).duration
        assert track.start + spoken <= timeline.duration + 0.01
        if index + 1 < len(voices):
            assert track.start + spoken <= voices[index + 1].start + 0.01

    starts = [t["start"] for t in result.metadata["message_timings"]]
    assert [t.start for t in voices] == pytest.approx(starts, abs=0.001)


def test_total_duration_is_delays_plus_typing_plus_narration_plus_tail(env, spy: RenderSpy):
    result = run(script=typed_script())
    timings = result.metadata["message_timings"]

    expected = sum(t["delay"] + t["typing"] + t["duration"] for t in timings) + texts.TAIL_SECONDS
    assert result.metadata["timeline_seconds"] == pytest.approx(expected, abs=0.005)
    assert spy.timeline.duration == pytest.approx(expected, abs=0.005)

    # and every "duration" really is the length of that message's narration file
    parts = sorted(env.work_dir.glob("texts-*/vo/msg_*.wav"))
    assert len(parts) == len(timings)
    for timing, part in zip(timings, parts, strict=True):
        assert timing["duration"] == pytest.approx(ff.probe(part).duration, abs=0.05)

    # the tail is real air after the last voice stops
    last = timings[-1]
    assert result.metadata["timeline_seconds"] - last["end"] == pytest.approx(texts.TAIL_SECONDS)
    assert _probe(result).duration == pytest.approx(result.metadata["timeline_seconds"], abs=0.2)


# --------------------------------------------------------------------------- #
# generated conversations
# --------------------------------------------------------------------------- #

def test_a_generated_conversation_renders_and_names_its_provider(env, spy: RenderSpy):
    result = run(topic="the neighbour who never sleeps", turns=4)

    assert result.metadata["script_source"] == "generated"
    assert result.metadata["provider"] == "heuristic"
    assert result.metadata["messages"] == 4
    assert result.metadata["tts_provider"] == "offline"

    script_data = result.metadata["script"]
    assert len(script_data["messages"]) == 4
    assert any(m["typing"] > 0 for m in script_data["messages"]), "the generator adds typing beats"

    layers = chat_layers(spy.timeline)
    assert len(layers) == result.metadata["states"] > 4, "typing beats add extra states"
    assert layers[-1].end == pytest.approx(spy.timeline.duration)
    assert _probe(result).has_video


def test_a_supplied_conversation_never_reaches_the_generator(env, monkeypatch):
    def _boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("write_chat was called for a supplied conversation")

    monkeypatch.setattr(texts.scriptgen, "write_chat", _boom)
    result = run(script=PARSED_SCRIPT)

    assert result.metadata["script_source"] == "supplied"
    assert result.metadata["provider"] == ""
    assert result.metadata["contact"] == "Alex"


# --------------------------------------------------------------------------- #
# empty and malformed conversations
# --------------------------------------------------------------------------- #

def test_an_empty_conversation_is_refused(env):
    with pytest.raises(AiclipperError, match="no messages"):
        texts.build_chat(None, ChatScript(title="Silence"), settings=env)
    with pytest.raises(AiclipperError, match="no messages"):
        texts.build_chat(None, ChatScript(messages=[ChatMessage(sender="A", text="   ")]), settings=env)


def test_an_empty_script_string_without_a_topic_is_refused(env):
    with pytest.raises(AiclipperError, match="topic or a conversation"):
        texts.build_chat(None, None, settings=env)
    with pytest.raises(AiclipperError, match="topic or a conversation"):
        texts.build_chat("   ", "   ", settings=env)


def test_a_thing_that_is_not_a_conversation_is_refused(env):
    with pytest.raises(AiclipperError, match="ChatScript"):
        texts.build_chat(None, Path("/tmp/chat.txt"), settings=env)  # type: ignore[arg-type]


def test_running_an_empty_conversation_renders_nothing(env):
    with pytest.raises(AiclipperError):
        run(script=ChatScript(title="Silence"))


# --------------------------------------------------------------------------- #
# voices
# --------------------------------------------------------------------------- #

def test_each_side_of_the_conversation_gets_its_own_voice(env):
    result = run(script=typed_script(), voice="narrator_deep", reply_voice="bright_female")
    voices = result.metadata["voices"]

    assert voices["outgoing"] == "narrator_deep"
    assert voices["incoming"] == "bright_female"
    by_side = {(t["outgoing"], t["voice"]) for t in result.metadata["message_timings"]}
    assert by_side == {(True, "narrator_deep"), (False, "bright_female")}


def test_the_default_voices_still_differ_per_side(env):
    result = run(script=typed_script())
    voices = result.metadata["voices"]

    assert voices["outgoing"] and voices["incoming"]
    assert voices["outgoing"] != voices["incoming"]


def test_a_message_can_override_its_side_voice(env):
    script = typed_script()
    script.messages[1].voice = "newsroom"
    result = run(script=script, voice="narrator_deep", reply_voice="bright_female")

    timings = result.metadata["message_timings"]
    assert timings[1]["voice"] == "newsroom"
    assert timings[3]["voice"] == "narrator_deep"


def test_an_unknown_voice_degrades_instead_of_failing(env):
    result = run(script=PARSED_SCRIPT, voice="definitely-not-a-voice")

    assert result.metadata["voice_request"] == "definitely-not-a-voice"
    assert _probe(result).has_audio


# --------------------------------------------------------------------------- #
# the mix
# --------------------------------------------------------------------------- #

def test_music_is_ducked_under_the_voices(env, spy: RenderSpy):
    run(script=typed_script())
    timeline = spy.timeline

    music = [t for t in timeline.audio if t.role == "music"]
    assert [t.label for t in music] == ["music:warm_bed"]
    assert music[0].duck is True
    assert music[0].gain_db == texts.MUSIC_GAIN_DB
    assert music[0].end == pytest.approx(timeline.duration, abs=0.01)
    assert "sidechaincompress" in spy.command
    assert all(t.duck is False for t in voice_tracks(timeline))


def test_a_conversation_nobody_reads_aloud_still_renders(env, spy: RenderSpy):
    script = typed_script()
    for message in script.messages:
        message.read_aloud = False
    result = run(script=script)

    timeline = spy.timeline
    assert voice_tracks(timeline) == []
    music = [t for t in timeline.audio if t.role == "music"]
    assert music[0].duck is False, "ducking with no voice would fail validation"
    assert timeline.validate() == []
    assert result.metadata["spoken_messages"] == 0
    assert result.transcript is None
    assert len(chat_layers(timeline)) == result.metadata["states"]
    assert _probe(result).has_video


def test_the_background_covers_the_whole_timeline(env, spy: RenderSpy):
    run(script=PARSED_SCRIPT)
    timeline = spy.timeline

    background = [layer for layer in timeline.visuals if layer.label.startswith("bg:")]
    assert len(background) == 1
    assert background[0].src.endswith("soft_drift.mp4")
    assert background[0].z == 0
    assert background[0].end == pytest.approx(timeline.duration)
    assert all(layer.z > 0 for layer in chat_layers(timeline)), "states sit above the background"


# --------------------------------------------------------------------------- #
# captions, theme, backend, output
# --------------------------------------------------------------------------- #

def test_no_captions_are_burned_by_default(env, spy: RenderSpy):
    result = run(script=PARSED_SCRIPT)

    assert spy.timeline.subtitles is None
    assert "subtitles=" not in spy.command
    assert not list(env.work_dir.rglob("*.ass"))
    assert result.metadata["captions"] is False
    assert result.metadata["style"] == ""
    assert result.metadata["words"] > 0, "the words are still on the clock for the transcript"
    assert result.transcript is not None


def test_captions_can_be_switched_on(env, spy: RenderSpy):
    result = run(script=PARSED_SCRIPT, captions=True, style="bold_yellow")

    track = spy.timeline.subtitles
    assert track is not None
    body = Path(track.ass_path).read_text(encoding="utf-8")
    assert "[Events]" in body and "Dialogue:" in body
    assert "subtitles=" in spy.command
    assert result.metadata["captions"] is True
    assert result.metadata["style"] == "bold_yellow"


def test_burned_captions_clear_the_bubble_column(env, spy: RenderSpy):
    """Captions must land in the empty band under the chat, never across it.

    The bubbles are bottom-anchored, so the last one is exactly where a preset
    that sits centred -- or a long way up from the bottom edge -- would print.
    """
    run(script=PARSED_SCRIPT, captions=True, style="bold_yellow")

    states = sorted((env.work_dir).rglob("states/*.png"))
    assert states, "the chat states are what the captions have to clear"
    bottom = max(_alpha_bottom(path) for path in states)
    assert 0 < bottom < env.height

    ass = Path(spy.timeline.subtitles.ass_path).read_text(encoding="utf-8")
    play_y = int(re.search(r"PlayResY:\s*(\d+)", ass).group(1))
    margin_v = int(re.search(r"^Style:.*,(\d+),\d+\s*$", ass, re.M).group(1))
    bottom_ref = bottom * play_y / env.height

    # the caption block grows upward from ``play_y - margin_v``; reserving the
    # same worst case the pipeline reserves must still clear the last bubble
    reserved = texts.CAPTION_LINES * 1.32 * captions_module.get_style("bold_yellow").font_size
    assert play_y - margin_v - reserved >= bottom_ref, (
        f"a caption block ending at {play_y - margin_v} overlaps bubbles that reach {bottom_ref}"
    )
    assert margin_v >= texts.CAPTION_MIN_MARGIN_V


def test_caption_style_below_chat_pins_a_centred_preset_to_the_bottom(tmp_path: Path):
    from aiclipper.overlays import OverlayImage

    state = tmp_path / "state.png"
    canvas = Image.new("RGBA", (1080, 1920), (0, 0, 0, 0))
    canvas.paste((255, 0, 0, 255), (100, 100, 900, 1400))
    canvas.save(state)
    images = [OverlayImage(path=state, width=1080, height=1920, index=0)]

    centred = captions_module.get_style("clean")
    assert centred.position == "center"

    moved = texts.caption_style_below_chat(centred, images, width=1080, height=1920)
    assert moved is not None
    assert moved.position == "bottom"
    assert moved.margin_v < centred.margin_v
    assert texts.overlay_bottom(images) == 1400


def test_caption_style_below_chat_is_a_no_op_without_captions_or_states():
    assert texts.caption_style_below_chat(None, [], width=1080, height=1920) is None
    style = captions_module.get_style("subtle_lower")
    assert texts.caption_style_below_chat(style, [], width=1080, height=1920) is style


def test_a_bad_caption_style_fails_before_any_speech(env, monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(texts.tts, "get_provider", lambda **kw: calls.append("tts"))
    with pytest.raises(ValueError):
        run(script=PARSED_SCRIPT, captions=True, style="not-a-real-preset")
    assert calls == []


def test_the_theme_is_honoured_and_the_caller_script_is_left_alone(env):
    script = typed_script()
    result = run(script=script, theme="sunset")

    assert result.metadata["theme"] == "sunset"
    assert script.theme == "mint", "the pipeline must not mutate the caller's script"


def test_an_unknown_theme_falls_back_to_the_default(env):
    result = run(script=PARSED_SCRIPT, theme="not-a-theme")
    assert result.metadata["theme"] == overlays.DEFAULT_CHAT_THEME


def test_the_backend_that_drew_the_frames_is_reported(env):
    result = run(script=PARSED_SCRIPT)
    assert result.metadata["backend"] == "pillow"
    assert result.metadata["backend_requested"] == "pillow"

    auto = texts.run(script=PARSED_SCRIPT, background="soft_drift", music="warm_bed")
    assert auto.metadata["backend"] in overlays.BACKENDS
    assert auto.metadata["backend_requested"] == "auto"
    assert auto.metadata["backend"] in overlays.available_backends(env)


def test_metadata_is_json_serialisable_and_complete(env):
    result = run(script=typed_script(), voice="narrator_deep", reply_voice="bright_female")
    meta = result.metadata

    assert json.dumps(meta)
    assert set(meta) >= {
        "theme", "backend", "messages", "voices", "message_timings", "background",
        "music", "states", "timeline_seconds", "tail_seconds", "script",
    }
    assert meta["background"] == "soft_drift" and meta["music"] == "warm_bed"
    assert meta["script"]["contact"] == "Robin"
    assert len(meta["message_timings"]) == meta["messages"] == 4


def test_output_naming(env, tmp_path: Path):
    target = tmp_path / "renders" / "my thread.mp4"
    explicit = run(script=PARSED_SCRIPT, out_path=target)
    assert explicit.output == target and target.exists()

    first = run(script=PARSED_SCRIPT)
    second = run(script=PARSED_SCRIPT)
    assert first.output != second.output
    assert first.output.exists() and second.output.exists()
    assert second.output.name.endswith("-2.mp4")
