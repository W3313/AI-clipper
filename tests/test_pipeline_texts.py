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
    states = overlays.chat_states(script, header_state=True)
    spans = texts.state_spans(script, beats, total)

    # the header, then one state per message plus the single typing frame
    assert [(s.visible, s.typing) for s in states] == [
        (0, False), (1, False), (2, False), (2, True), (3, False), (4, False),
    ]
    assert len(spans) == len(states)

    # the typing frame occupies exactly the typing beat of message 2
    typing_span = spans[3]
    assert typing_span == (pytest.approx(beats[2].typing_start), pytest.approx(beats[2].start))
    assert typing_span[1] - typing_span[0] == pytest.approx(0.9)

    # every bubble frame starts exactly when its own message starts
    for position, state in enumerate(states):
        if not state.typing and state.visible:
            assert spans[position][0] == pytest.approx(beats[state.visible - 1].start)

    # contiguous, no gaps, no overlaps, and the last one holds to the end
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == pytest.approx(next_start)
    assert all(end > start for start, end in spans)
    assert spans[-1][1] == pytest.approx(total)


def test_state_spans_hold_the_header_from_the_very_first_frame():
    """The video must not open on a bare background during message 0's delay."""
    script = typed_script()
    beats = texts.plan_beats(script.messages, [1.0, 1.5, 2.0, 1.25])
    states = overlays.chat_states(script, header_state=True)
    spans = texts.state_spans(script, beats, beats[-1].end + texts.TAIL_SECONDS)

    assert states[0].visible == 0 and not states[0].typing, "state 0 is the chrome-only frame"
    assert spans[0][0] == 0.0, "the header has to be on screen at t=0"
    assert spans[0][1] == pytest.approx(beats[0].start) == pytest.approx(0.5)
    # and it is the *only* thing on screen until the first bubble lands
    assert spans[1][0] == pytest.approx(beats[0].start)


def test_a_first_message_at_t_zero_leaves_the_header_no_room_without_breaking_the_tiling():
    """A conversation that starts instantly still keeps image/span parity."""
    script = ChatScript(
        title="Instant",
        contact="Robin",
        messages=[
            ChatMessage(sender="Robin", text="now", delay=0.0),
            ChatMessage(sender="Me", text="ok", outgoing=True, delay=0.4),
        ],
    )
    beats = texts.plan_beats(script.messages, [1.0, 1.0])
    total = beats[-1].end + texts.TAIL_SECONDS
    states = overlays.chat_states(script, header_state=True)
    spans = texts.state_spans(script, beats, total)

    assert len(spans) == len(states) == 3
    assert spans[0] == (0.0, 0.0), "an empty header span, not an overlapping one"
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == pytest.approx(next_start), "a gap or an overlap between states"
    assert spans[-1][1] == pytest.approx(total)


def test_a_typing_hint_shorter_than_the_floor_does_not_overlap_the_bubble():
    """A sub-MIN_STATE_SECONDS beat must still tile, not overrun its successor.

    The floor can only ever extend the *last* span: every other state gives way
    the instant the next one appears, and :func:`texts.state_frames` places the
    cuts from the starts alone, so padding a short span would put an overlap in
    the metadata and buy the state nothing on screen.
    """
    tiny = texts.MIN_STATE_SECONDS / 10
    script = ChatScript(
        title="Blink",
        contact="Robin",
        messages=[
            ChatMessage(sender="Robin", text="a", delay=0.0, typing=tiny),
            ChatMessage(sender="Me", text="b", outgoing=True, delay=0.0, typing=tiny),
        ],
    )
    beats = texts.plan_beats(script.messages, [1.0, 1.0])
    total = beats[-1].end + texts.TAIL_SECONDS
    states = overlays.chat_states(script, header_state=True)
    spans = texts.state_spans(script, beats, total)

    assert len(spans) == len(states) == 5
    for (start, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end >= start, "a span that ends before it begins"
        assert end == pytest.approx(next_start), "a gap or an overlap between states"
    assert spans[-1][1] == pytest.approx(total)

    # and the composited layer still tiles the whole programme, frame for frame
    edges = texts.state_frames(spans, fps=30, duration=total)
    assert edges == sorted(edges)
    assert edges[0] == 0
    assert sum(edges[i + 1] - edges[i] for i in range(len(spans))) == edges[-1]


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


def test_the_conversation_is_one_layer_and_its_states_tile_the_timeline(env, spy: RenderSpy):
    script = typed_script()
    result = run(script=script)
    timeline = spy.timeline

    states = overlays.chat_states(script, header_state=True)
    layers = chat_layers(timeline)
    assert len(states) == 6, "the header, four messages and one typing frame"
    assert result.metadata["states"] == len(states)

    # one pre-composited layer, whatever the message count
    assert len(layers) == 1
    layer = layers[0]
    assert layer.kind == "video"
    assert Path(layer.src).exists() and Path(layer.src).stat().st_size > 0
    assert (layer.start, layer.end) == (0.0, pytest.approx(timeline.duration))

    frames = sorted((env.work_dir).rglob("states/chat_*.png"))
    assert len(frames) == len(states), "every state still has its own PNG"

    spans = [(t["start"], t["end"]) for t in result.metadata["state_timings"]]
    assert spans == sorted(spans), "states are reported in overlay order"
    for (_, end), (next_start, _) in zip(spans, spans[1:], strict=False):
        assert end == pytest.approx(next_start), "a gap or an overlap between states"
    assert spans[0][0] == 0.0, "the header is up from the first frame"
    assert spans[-1][1] == pytest.approx(timeline.duration)
    assert spans[1][0] == pytest.approx(0.5), "the first bubble waits out its own delay"


def test_the_graph_does_not_grow_with_the_message_count(env, spy: RenderSpy):
    """The performance fix, stated as an invariant: one overlay, always.

    Before the states were pre-composited the filter graph carried one
    full-canvas RGBA ``overlay`` per conversation state, so a long chat paid for
    every one of them on every frame of the finished video.
    """
    def conversation(count: int) -> ChatScript:
        return ChatScript(
            title=f"Thread {count}",
            contact="Robin",
            messages=[
                ChatMessage(sender="Me" if i % 2 else "Robin", text=f"message number {i}",
                            outgoing=bool(i % 2), delay=0.3, typing=0.4 if i % 3 == 2 else 0.0)
                for i in range(count)
            ],
        )

    short = run(script=conversation(3))
    long = run(script=conversation(9))
    assert long.metadata["states"] > short.metadata["states"] + 4

    # the background and the one conversation layer -- and nothing else, at any length
    overlays_in = [" ".join(command).count("overlay=") for command in spy.commands]
    assert overlays_in == [2, 2], (
        f"the graph grew from {overlays_in[0]} to {overlays_in[1]} overlays with the message count"
    )
    # one scale/pad chain per conversation state was the other half of the cost
    pads = [" ".join(command).count("pad=") for command in spy.commands]
    assert pads == [1, 1], f"the graph grew from {pads[0]} to {pads[1]} padded layers"
    assert len(chat_layers(spy.timelines[0])) == len(chat_layers(spy.timelines[1])) == 1


def test_the_typing_frame_covers_exactly_the_typing_beat(env, spy: RenderSpy):
    script = typed_script()
    result = run(script=script)

    states = result.metadata["state_timings"]
    typing_positions = [i for i, state in enumerate(states) if state["typing"]]
    assert typing_positions == [3]

    typing_state = states[typing_positions[0]]
    bubble_state = states[typing_positions[0] + 1]
    assert typing_state["end"] - typing_state["start"] == pytest.approx(0.9, abs=0.01)
    assert typing_state["end"] == pytest.approx(bubble_state["start"])

    timings = result.metadata["message_timings"]
    assert timings[2]["typing"] == pytest.approx(0.9)
    assert bubble_state["start"] == pytest.approx(timings[2]["start"], abs=0.001)
    assert bubble_state["visible"] == 3, "the bubble frame that follows the typing one"


def test_every_message_audio_starts_when_its_bubble_appears(env, spy: RenderSpy):
    script = typed_script()
    result = run(script=script)
    timeline = spy.timeline

    states = result.metadata["state_timings"]
    voices = voice_tracks(timeline)
    assert len(voices) == len(script.messages) == 4

    # independently recompute which state shows message i's bubble
    for index, track in enumerate(voices):
        state = next(
            s for s in states if not s["typing"] and s["visible"] == index + 1
        )
        assert track.start == pytest.approx(state["start"]), f"message {index} is out of sync"
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
# the pre-composited conversation layer
# --------------------------------------------------------------------------- #

def _concat_durations(work_dir: Path) -> list[float]:
    """The ``duration`` directives the pipeline wrote for the concat demuxer."""
    lists = sorted(work_dir.rglob("chat_states.concat"))
    assert len(lists) == 1, f"expected one concat list, found {lists}"
    return [
        float(line.split(" ", 1)[1])
        for line in lists[0].read_text(encoding="utf-8").splitlines()
        if line.startswith("duration ")
    ]


def _state_video(work_dir: Path) -> Path:
    videos = sorted(work_dir.rglob("chat_states.*"))
    videos = [v for v in videos if v.suffix != ".concat"]
    assert len(videos) == 1, f"expected one composited layer, found {videos}"
    return videos[0]


def _frames(video: Path, out_dir: Path) -> list[bytes]:
    """Every frame of ``video``, as raw RGBA bytes, in order."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ff.run_ffmpeg(["-y", "-i", str(video), "-fps_mode", "passthrough",
                   str(out_dir / "f_%05d.png")])
    frames = []
    for path in sorted(out_dir.glob("f_*.png")):
        with Image.open(path) as handle:
            frames.append(handle.convert("RGBA").tobytes())
    return frames


def _rgba(path: Path) -> bytes:
    with Image.open(path) as handle:
        return handle.convert("RGBA").tobytes()


def test_state_frames_land_where_the_old_per_state_overlay_gate_did():
    """``enable='between(t,start,end)'`` showed a state from ``ceil(start*fps)``."""
    spans = [(0.0, 0.5), (0.5, 2.053991), (2.053991, 3.807982), (3.807982, 12.227)]
    assert texts.state_frames(spans, fps=30, duration=12.227) == [0, 15, 62, 115, 368]
    # a boundary that is a whole number of frames is not pushed onto the next one
    assert texts.state_frames([(0.0, 1.0), (1.0, 2.0)], fps=12, duration=2.0) == [0, 12, 25]
    # two states that fall inside one frame collapse rather than going backwards
    # (the second used to win the overlay anyway, being the higher layer)
    edges = texts.state_frames([(0.0, 0.42), (0.42, 0.5), (0.5, 3.0)], fps=12, duration=3.0)
    assert edges == [0, 6, 6, 37]


def test_the_concat_list_places_every_cut_on_a_whole_frame(env, spy: RenderSpy):
    """The timing guarantee, in the file ffmpeg is actually handed."""
    result = run(script=typed_script())
    timeline = spy.timeline

    spans = [(s["start"], s["end"]) for s in result.metadata["state_timings"]]
    edges = texts.state_frames(spans, fps=env.fps, duration=timeline.duration)
    expected = [
        (edges[i + 1] - edges[i]) / texts.CONCAT_TICK_RATE
        for i in range(len(spans)) if edges[i + 1] > edges[i]
    ]
    assert _concat_durations(env.work_dir) == pytest.approx(expected, abs=1e-9)

    video = _state_video(env.work_dir)
    assert video.exists() and video.stat().st_size > 0
    assert Path(chat_layers(timeline)[0].src) == video
    info = ff.probe(video)
    assert info.has_video and not info.has_audio
    assert (info.width, info.height) == (env.width, env.height)
    # the layer covers the whole timeline, never a frame short of it
    assert info.duration >= timeline.duration
    assert info.duration < timeline.duration + 2.0 / env.fps


def test_the_composited_layer_cuts_on_exactly_the_frames_the_clock_asked_for(env, tmp_path: Path):
    """Not "close": every state's artwork lands on its own frame, to the frame.

    Composition must not move a bubble even one frame away from the voice that
    starts with it, so the frame each state appears on is compared against the
    frame the per-state overlay gate produced, not against a tolerance.
    """
    result = run(script=typed_script())
    frames = _frames(_state_video(env.work_dir), tmp_path / "composite")
    stills = sorted(env.work_dir.rglob("states/chat_*.png"))

    spans = [(s["start"], s["end"]) for s in result.metadata["state_timings"]]
    edges = texts.state_frames(
        spans, fps=env.fps, duration=result.metadata["timeline_seconds"]
    )
    assert len(stills) == len(spans)
    assert len(frames) >= edges[-1] - 1
    for position, still in enumerate(stills):
        if edges[position + 1] <= edges[position]:
            continue
        wanted = _rgba(still)
        first = next((i for i, frame in enumerate(frames) if frame == wanted), None)
        assert first == edges[position], (
            f"state {position} appears on frame {first}, not on frame {edges[position]} "
            f"(t={spans[position][0]:.6f}s at {env.fps}fps)"
        )


def test_the_video_opens_on_the_chat_chrome_and_not_a_bare_background(env, tmp_path: Path):
    """The blank-open defect, checked on the rendered pixels.

    State 0 is the header-only frame; it has to be composited over the very
    first frame of the finished mp4, not left until the first bubble lands.
    """
    result = run(script=typed_script())
    header = sorted(env.work_dir.rglob("states/chat_*.png"))[0]

    spans = result.metadata["state_timings"]
    assert spans[0]["visible"] == 0 and spans[0]["start"] == 0.0
    assert spans[0]["end"] == pytest.approx(0.5), "held until the first bubble"

    # a pixel the header paints solidly, with a solid 5x5 neighbourhood so
    # chroma subsampling in the mp4 cannot move it
    with Image.open(header) as handle:
        art = handle.convert("RGBA")
    pixels = art.load()
    spot = None
    for y in range(2, art.height - 2):
        for x in range(2, art.width - 2):
            block = {pixels[x + dx, y + dy] for dx in (-2, 0, 2) for dy in (-2, 0, 2)}
            if len(block) == 1 and next(iter(block))[3] == 255:
                spot = (x, y, next(iter(block))[:3])
                break
        if spot:
            break
    assert spot is not None, "the header state paints nothing at all"

    x, y, colour = spot
    first_frame = tmp_path / "open.png"
    ff.run_ffmpeg(["-y", "-i", str(result.output), "-frames:v", "1", str(first_frame)])
    with Image.open(first_frame) as handle:
        got = handle.convert("RGB").load()[x, y]
    assert max(abs(a - b) for a, b in zip(got, colour, strict=True)) <= 24, (
        f"the first frame shows {got} at {(x, y)} where the chat header paints {colour}"
    )


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
    assert len(layers) == 1, "the conversation is pre-composited into one layer"
    assert result.metadata["states"] > 5, "the header and the typing beats add extra states"
    assert layers[0].end == pytest.approx(spy.timeline.duration)
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
    assert len(chat_layers(timeline)) == 1
    assert len(result.metadata["state_timings"]) == result.metadata["states"]
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


def test_the_default_filename_comes_from_the_topic(env):
    """``--topic "a lost cat"`` must not land on a file named after an invention."""
    result = run(topic="a lost cat", turns=4)

    assert result.output.stem == "a-lost-cat", result.output.name
    # The generated headline is still what a human sees, in both places.
    assert result.title
    assert result.title.lower() != "a lost cat"
    assert result.metadata["script"]["title"] == result.title
    assert result.metadata["topic"] == "a lost cat"


def test_a_supplied_conversation_still_names_its_file_from_the_title(env):
    """With no topic there is nothing but the title to name the file after."""
    result = run(script=PARSED_SCRIPT)

    assert result.output.stem == "locked-out", result.output.name
    assert result.title == "Locked Out"


# --------------------------------------------------------------------------- #
# speech backend fallback
# --------------------------------------------------------------------------- #

def test_a_backend_that_dies_mid_conversation_falls_back_to_offline(env, monkeypatch, caplog):
    """edge-tts with no network must not take the render down with it.

    The conversation is re-spoken from the first message by the next backend, so
    the finished video never mixes two voices on one side.
    """
    from aiclipper.errors import TTSError

    class HalfDeadTTS:
        name = "halfdead"
        calls = 0

        def available(self) -> bool:
            return True

        def synthesize(self, text, out_path, *, voice):  # noqa: ANN001
            HalfDeadTTS.calls += 1
            if HalfDeadTTS.calls > 1:
                raise TTSError("synthesis failed: the network went away")
            return texts.tts.OfflineTTS().synthesize(text, out_path, voice=voice)

    monkeypatch.setattr(texts.tts, "get_provider", lambda **kw: HalfDeadTTS())

    with caplog.at_level("WARNING"):
        result = run(script=PARSED_SCRIPT)

    assert result.output.exists()
    assert result.metadata["tts_provider"] == "offline", "the backend that really spoke is reported"
    assert HalfDeadTTS.calls > 1, "the dead backend was actually tried"
    assert any("halfdead" in record.getMessage() for record in caplog.records), caplog.text
    assert _probe(result).has_audio


def test_no_backend_left_means_the_original_failure_propagates(env, monkeypatch):
    """With nothing behind it, a failing backend still raises its own error."""
    from aiclipper.errors import TTSError

    class DeadTTS:
        name = "dead"

        def available(self) -> bool:
            return True

        def synthesize(self, text, out_path, *, voice):  # noqa: ANN001
            raise TTSError("synthesis failed: the network went away")

    monkeypatch.setattr(texts.tts, "get_provider", lambda **kw: DeadTTS())
    monkeypatch.setattr(texts.tts, "fallback_chain", lambda provider, **kw: [provider])

    with pytest.raises(TTSError, match="the network went away"):
        run(script=PARSED_SCRIPT)
