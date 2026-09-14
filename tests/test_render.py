"""Tests for :mod:`aiclipper.render` -- the ffmpeg filtergraph core.

Graph shape is asserted with ``dry_run=True`` (no ffmpeg process); the last
section renders four real (tiny, ultrafast) videos to prove the graphs the
builder emits actually execute and put the right pixels in the right place.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from aiclipper import ffmpeg as ff
from aiclipper.errors import RenderError
from aiclipper.models import (
    AudioTrack,
    CropKeyframe,
    CropPath,
    RenderOptions,
    SubtitleTrack,
    Timeline,
    VisualLayer,
)
from aiclipper.render import build_command, render

FAST = RenderOptions(preset="ultrafast", crf=32, audio_bitrate="64k")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def graph_of(cmd: list[str]) -> str:
    return cmd[cmd.index("-filter_complex") + 1]


def chains(cmd: list[str]) -> list[str]:
    return graph_of(cmd).split(";")


def chain_with(cmd: list[str], needle: str) -> str:
    found = [c for c in chains(cmd) if needle in c]
    assert found, f"no chain containing {needle!r} in:\n" + "\n".join(chains(cmd))
    assert len(found) == 1, f"{needle!r} appears in {len(found)} chains"
    return found[0]


def input_blocks(cmd: list[str]) -> list[list[str]]:
    """Split the argv into one list per ``-i`` block, in command-line order."""
    stop = cmd.index("-filter_complex")
    blocks: list[list[str]] = []
    current: list[str] = []
    for token in cmd[1:stop]:
        if token == "-y":
            continue
        current.append(token)
        if len(current) >= 2 and current[-2] == "-i":
            blocks.append(current)
            current = []
    assert not current, f"dangling input options: {current}"
    return blocks


def input_paths(cmd: list[str]) -> list[str]:
    return [block[-1] for block in input_blocks(cmd)]


def fake(path: Path) -> Path:
    """A file that merely exists -- enough for graph-only assertions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\0")
    return path


ASS_TEXT = """[Script Info]
ScriptType: v4.00+
PlayResX: 320
PlayResY: 568
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,DejaVu Sans,36,&H00FFFFFF,&H000000FF,&H00000000,&H64000000,-1,0,0,0,100,100,0,0,1,3,1,2,20,20,60,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,burned in
Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,second cue
"""


@pytest.fixture
def ass_path(tmp_path: Path) -> Path:
    path = tmp_path / "subs.ass"
    path.write_text(ASS_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def vid(tmp_path: Path) -> Path:
    return fake(tmp_path / "clip.mp4")


@pytest.fixture
def img(tmp_path: Path) -> Path:
    return fake(tmp_path / "card.png")


@pytest.fixture
def snd(tmp_path: Path) -> Path:
    return fake(tmp_path / "voice.wav")


def basic(vid: Path, **kw) -> Timeline:
    tl = Timeline(width=320, height=568, fps=30, duration=4.0, background="#101820", **kw)
    tl.add_visual(VisualLayer(kind="video", src=str(vid)))
    return tl


def frame_at(path: Path, t: float, out: Path):
    """Decode a single frame and return it as a Pillow image."""
    from PIL import Image

    ff.run_ffmpeg(["-y", "-ss", f"{t:.3f}", "-i", str(path), "-frames:v", "1", str(out)])
    with Image.open(out) as im:
        return im.convert("RGB").copy()


def mean_abs_diff(a, b) -> float:
    """Average per-channel difference between two same-sized frames."""
    pa, pb = list(a.getdata()), list(b.getdata())
    assert len(pa) == len(pb)
    total = sum(abs(x - y) for px, py in zip(pa, pb, strict=True) for x, y in zip(px, py, strict=True))
    return total / (len(pa) * 3)


def dominant(pixel: tuple[int, int, int]) -> str:
    r, g, b = pixel
    if r > 120 and r > b + 60:
        return "red"
    if b > 120 and b > r + 60:
        return "blue"
    return f"other{pixel}"


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #

def test_inputs_follow_z_order_then_audio(tmp_path: Path, vid: Path, img: Path, snd: Path):
    music = fake(tmp_path / "bed.wav")
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="image", src=str(img), z=5))
    tl.add_visual(VisualLayer(kind="color", color="#223344", z=1))
    tl.add_visual(VisualLayer(kind="video", src=str(vid), loop=True, z=0))
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(music), role="music"))

    cmd = build_command(tl, tmp_path / "out.mp4")

    # colour layer consumes no input; visuals come first, in z order, then audio
    assert input_paths(cmd) == [str(vid), str(img), str(snd), str(music)]
    assert input_blocks(cmd)[0] == ["-stream_loop", "-1", "-i", str(vid)]
    assert input_blocks(cmd)[1] == ["-loop", "1", "-i", str(img)]
    assert input_blocks(cmd)[2] == ["-i", str(snd)]
    # and the graph references them by the same indices
    assert "[0:v]" in graph_of(cmd) and "[1:v]" in graph_of(cmd)
    assert "[2:a]" in graph_of(cmd) and "[3:a]" in graph_of(cmd)


def test_non_looping_video_uses_input_seek_and_looping_uses_trim(tmp_path: Path, vid: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), src_start=12.5, z=0))
    tl.add_visual(VisualLayer(kind="video", src=str(vid), src_start=3.0, loop=True, z=1))
    cmd = build_command(tl, tmp_path / "out.mp4")

    assert input_blocks(cmd)[0] == ["-ss", "12.5", "-i", str(vid)]
    assert input_blocks(cmd)[1] == ["-stream_loop", "-1", "-i", str(vid)]

    seeked = chain_with(cmd, "[0:v]")
    assert "trim=start=" not in seeked          # the seek already happened at input level
    assert "setpts=PTS-STARTPTS" in seeked

    looped = chain_with(cmd, "[1:v]")
    assert "trim=start=3,setpts=PTS-STARTPTS" in looped


def test_every_video_layer_is_rate_locked(tmp_path: Path, vid: Path):
    tl = basic(vid)
    tl.fps = 24
    cmd = build_command(tl, tmp_path / "out.mp4")
    assert "fps=24" in chain_with(cmd, "[0:v]")


# --------------------------------------------------------------------------- #
# canvas, geometry, compositing
# --------------------------------------------------------------------------- #

def test_base_canvas_matches_timeline(tmp_path: Path, vid: Path):
    cmd = build_command(basic(vid), tmp_path / "out.mp4")
    assert chains(cmd)[0] == "color=c=0x101820:s=320x568:r=30:d=4[base]"


@pytest.mark.parametrize(
    "fit, expected",
    [
        ("stretch", ["scale=200:100"]),
        ("cover", ["scale=200:100:force_original_aspect_ratio=increase", "crop=200:100"]),
        (
            "contain",
            [
                "scale=200:100:force_original_aspect_ratio=decrease",
                "format=rgba",
                "pad=200:100:(ow-iw)/2:(oh-ih)/2:color=black@0",
            ],
        ),
    ],
)
def test_fit_modes(tmp_path: Path, vid: Path, fit: str, expected: list[str]):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), x=5, y=7, w=200, h=100, fit=fit))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "[0:v]")
    assert ",".join(expected) in chain
    assert chain.endswith("setsar=1[v0]")


def test_layer_rect_drives_overlay_position(tmp_path: Path, vid: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), x=12, y=34, w=100, h=80))
    overlay = chain_with(build_command(tl, tmp_path / "out.mp4"), "overlay=")
    assert "overlay=x=12:y=34" in overlay
    assert "shortest=0" in overlay


def test_overlay_accumulator_threads_layers_in_z_order(tmp_path: Path, vid: Path, img: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="image", src=str(img), z=9, label="top"))
    tl.add_visual(VisualLayer(kind="video", src=str(vid), z=-1, label="bottom"))
    cmd = build_command(tl, tmp_path / "out.mp4")
    graph = graph_of(cmd)

    assert "[base][v0]overlay=" in graph
    assert "[c0][v1]overlay=" in graph
    # the bottom (z=-1) layer is input 0 and is composited first
    assert graph.index("[0:v]") < graph.index("[1:v]")
    # every intermediate pad is produced once and consumed once
    labels = re.findall(r"\[([a-z][a-z0-9]*)\]", graph)
    for produced in ("base", "c0", "v0", "v1"):
        assert labels.count(produced) == 2, (produced, labels)
    # the final pad is produced inside the graph and consumed by -map, not by a filter
    assert labels.count("vout") == 1
    assert cmd[cmd.index("[vout]") - 1] == "-map"


def test_enable_window_only_when_layer_is_time_bounded(tmp_path: Path, vid: Path, img: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), z=0))                       # full span
    tl.add_visual(VisualLayer(kind="image", src=str(img), start=1.0, end=2.5, z=1))   # windowed
    tl.add_visual(VisualLayer(kind="image", src=str(img), start=3.0, z=2))            # open ended
    cmd = build_command(tl, tmp_path / "out.mp4")
    first, second, third = [c for c in chains(cmd) if "overlay=" in c]

    assert "enable=" not in first
    assert "enable='between(t,1,2.5)'" in second
    assert "enable='between(t,3,4)'" in third     # clamped to the timeline duration


def test_delayed_video_layer_is_time_padded(tmp_path: Path, vid: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), start=1.25))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "[0:v]")
    assert "tpad=start_duration=1.25:start_mode=add:color=black@0" in chain


def test_opacity_uses_rgba_then_alpha_mixer(tmp_path: Path, vid: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), opacity=0.35))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "[0:v]")
    assert chain.endswith("setsar=1,format=rgba,colorchannelmixer=aa=0.35[v0]")
    assert chain.index("format=rgba") < chain.index("colorchannelmixer")


def test_opaque_layer_has_no_alpha_stage(tmp_path: Path, vid: Path):
    chain = chain_with(build_command(basic(vid), tmp_path / "out.mp4"), "[0:v]")
    assert "colorchannelmixer" not in chain


def test_colour_layer_is_a_lavfi_source_not_an_input(tmp_path: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="color", color="#ff0044", x=0, y=500, w=320, h=60, opacity=0.5))
    cmd = build_command(tl, tmp_path / "out.mp4")
    assert input_paths(cmd) == []
    assert "color=c=0xff0044:s=320x60:r=30:d=4,format=rgba,colorchannelmixer=aa=0.5[v0]" in graph_of(cmd)


# --------------------------------------------------------------------------- #
# crop paths
# --------------------------------------------------------------------------- #

def test_static_crop_path_is_a_plain_crop(tmp_path: Path, vid: Path):
    path = CropPath.static(120, 30, 360, 640, 1280, 720)
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path))
    workdir = tmp_path / "wd"
    chain = chain_with(build_command(tl, tmp_path / "out.mp4", workdir=workdir), "[0:v]")

    assert "crop=w=360:h=640:x=120:y=30" in chain
    assert "sendcmd" not in chain
    assert list(workdir.glob("*.cmd")) == []  # nothing to steer, nothing written


def test_repeated_identical_keyframes_stay_static(tmp_path: Path, vid: Path):
    path = CropPath(
        [CropKeyframe(0.0, 10, 20, 360, 640), CropKeyframe(2.0, 10, 20, 360, 640)], 1280, 720
    )
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "[0:v]")
    assert "sendcmd" not in chain
    assert "crop=w=360:h=640:x=10:y=20" in chain


def test_animated_crop_writes_a_sendcmd_script(tmp_path: Path, vid: Path):
    path = CropPath(
        [
            CropKeyframe(0.0, 0, 0, 360, 640),
            CropKeyframe(1.5, 200, 40, 360, 640),
            CropKeyframe(3.0, 420, 80, 360, 640),
        ],
        1280,
        720,
    )
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path, label="reframe"))
    workdir = tmp_path / "wd"
    cmd = build_command(tl, tmp_path / "out.mp4", workdir=workdir)
    chain = chain_with(cmd, "[0:v]")

    scripts = list(workdir.glob("*.cmd"))
    assert len(scripts) == 1, scripts
    script = scripts[0]

    # sendcmd feeds a uniquely named crop instance, and comes before it
    m = re.search(r"sendcmd=f=(\S+?),(crop@\w+)=w=360:h=640:x=0:y=0", chain)
    assert m, chain
    assert m.group(1) == ff.escape_filter_path(script)
    target = m.group(2)

    lines = script.read_text(encoding="utf-8").strip().splitlines()
    assert lines == [
        f"0 {target} x 0, {target} y 0;",
        f"1.5 {target} x 200, {target} y 40;",
        f"3 {target} x 420, {target} y 80;",
    ]
    for line in lines:
        assert re.fullmatch(r"[\d.]+ crop@\w+ x -?\d+, crop@\w+ y -?\d+;", line)


def test_sendcmd_times_are_relative_to_src_start(tmp_path: Path, vid: Path):
    path = CropPath(
        [CropKeyframe(10.0, 0, 0, 360, 640), CropKeyframe(12.0, 100, 0, 360, 640)], 1280, 720
    )
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), src_start=10.0, crop=path))
    workdir = tmp_path / "wd"
    build_command(tl, tmp_path / "out.mp4", workdir=workdir)
    lines = next(iter(workdir.glob("*.cmd"))).read_text(encoding="utf-8").split()
    assert lines[0] == "0"      # the keyframe at source t=10 happens at chain t=0


def test_two_animated_crops_get_distinct_targets(tmp_path: Path, vid: Path):
    def path() -> CropPath:
        return CropPath(
            [CropKeyframe(0.0, 0, 0, 100, 100), CropKeyframe(1.0, 50, 0, 100, 100)], 200, 200
        )

    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path(), z=0))
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path(), z=1))
    workdir = tmp_path / "wd"
    graph = graph_of(build_command(tl, tmp_path / "out.mp4", workdir=workdir))
    targets = set(re.findall(r"(crop@\w+)=", graph))
    assert len(targets) == 2
    assert len(list(workdir.glob("*.cmd"))) == 2


def test_variable_size_crop_path_is_rejected(tmp_path: Path, vid: Path):
    path = CropPath(
        [CropKeyframe(0.0, 0, 0, 360, 640), CropKeyframe(1.0, 0, 0, 300, 533)], 1280, 720
    )
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path, label="zoomy"))
    with pytest.raises(RenderError) as excinfo:
        build_command(tl, tmp_path / "out.mp4")
    assert any("300x533" in p and "360x640" in p for p in excinfo.value.problems)


# --------------------------------------------------------------------------- #
# audio
# --------------------------------------------------------------------------- #

def test_audio_start_becomes_adelay_in_milliseconds(tmp_path: Path, vid: Path, snd: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), start=0.75, role="voice"))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "[1:a]")
    assert "adelay=750|750" in chain             # milliseconds, one value per channel
    assert chain.index("asetpts") < chain.index("adelay")


def test_audio_trim_gain_fades_and_loop(tmp_path: Path, vid: Path, snd: Path):
    tl = basic(vid)
    tl.add_audio(
        AudioTrack(
            src=str(snd), src_start=2.0, start=0.0, end=3.0, gain_db=-7.5,
            loop=True, fade_in=0.4, fade_out=0.6, role="music",
        )
    )
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "[1:a]")
    assert "atrim=start=2,asetpts=PTS-STARTPTS" in chain
    assert "aloop=loop=-1:size=" in chain
    assert "atrim=duration=3" in chain
    assert "volume=-7.5dB" in chain
    assert "afade=t=in:st=0:d=0.4" in chain
    assert "afade=t=out:st=2.4:d=0.6" in chain   # end of the 3s span
    assert chain.index("aloop") < chain.index("volume") < chain.index("afade=t=in")


def test_tracks_are_mixed_limited_and_resampled(tmp_path: Path, vid: Path, snd: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(snd), role="music"))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "amix=")
    assert "[a0][a1]amix=inputs=2:normalize=0" in chain
    assert "alimiter" in chain
    assert "aresample=async=1" in chain
    assert chain.endswith("[aout]")
    assert chain.index("amix=") < chain.index("alimiter") < chain.index("aresample")


def test_ducking_splits_the_voice_bus_into_a_sidechain(tmp_path: Path, vid: Path, snd: Path):
    music = fake(tmp_path / "bed.wav")
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(music), role="music", duck=True))
    graph = graph_of(build_command(tl, tmp_path / "out.mp4"))

    assert "[a0]asplit=2[a0m][a0k]" in graph                    # voice reused, never consumed twice
    assert "[a1][a0k]sidechaincompress=" in graph               # music keyed off the voice
    assert "[a0m][a1d]amix=inputs=2" in graph                   # ducked music goes into the mix


def test_two_voices_are_summed_into_one_key_bus(tmp_path: Path, vid: Path, snd: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(snd), role="music", duck=True))
    graph = graph_of(build_command(tl, tmp_path / "out.mp4"))
    assert "[a0k][a1k]amix=inputs=2:normalize=0[voicebus]" in graph
    assert "[a2][voicebus]sidechaincompress=" in graph


def test_no_sidechain_without_a_ducked_track(tmp_path: Path, vid: Path, snd: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(snd), role="music"))
    graph = graph_of(build_command(tl, tmp_path / "out.mp4"))
    assert "sidechaincompress" not in graph
    assert "asplit" not in graph


def test_silent_timeline_still_produces_an_audio_stream(tmp_path: Path, vid: Path):
    cmd = build_command(basic(vid), tmp_path / "out.mp4")
    assert "anullsrc=r=48000:cl=stereo:d=4" in chain_with(cmd, "anullsrc")
    assert "-map" in cmd and "[aout]" in cmd


def test_layer_audio_is_taken_from_the_visual_input(tmp_path: Path, vid: Path, snd: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), take_audio=True, volume=0.5))
    tl.add_audio(AudioTrack(src=str(snd), role="music"))
    graph = graph_of(build_command(tl, tmp_path / "out.mp4"))
    assert "[0:a]" in graph and "volume=0.5" in graph
    assert "[1:a]" in graph
    assert "amix=inputs=2" in graph


# --------------------------------------------------------------------------- #
# loudness normalisation (graph)
# --------------------------------------------------------------------------- #

#: The mix tail exactly as it looked before loudness normalisation existed.
LEGACY_TAIL = (
    "apad,atrim=duration=4,asetpts=PTS-STARTPTS,"
    "alimiter=limit=0.95:level=0,aresample=async=1,"
    "aformat=sample_fmts=fltp:sample_rates=48000:channel_layouts=stereo[aout]"
)


def test_mix_is_loudness_normalised_between_the_mix_and_the_limiter(
    tmp_path: Path, vid: Path, snd: Path
):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(snd), role="music"))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4"), "amix=")

    assert "loudnorm=I=-14:TP=-1.5:LRA=11" in chain
    # after the mix so it measures the whole programme, before the limiter so
    # the limiter stays the last thing to touch the samples
    assert chain.index("amix=") < chain.index("loudnorm=") < chain.index("alimiter")


def test_loudness_target_none_reproduces_the_un_normalised_command(
    tmp_path: Path, vid: Path, snd: Path
):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    tl.add_audio(AudioTrack(src=str(snd), role="music"))
    cmd = build_command(tl, tmp_path / "out.mp4", loudness_target=None)
    chain = chain_with(cmd, "amix=")

    assert "loudnorm" not in graph_of(cmd)
    assert chain == "[a0][a1]amix=inputs=2:normalize=0:dropout_transition=0," + LEGACY_TAIL


def test_loudness_target_comes_from_render_options(tmp_path: Path, vid: Path, snd: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))

    opts = RenderOptions()
    opts.loudness_target = -16.5
    assert "loudnorm=I=-16.5:TP=-1.5:LRA=11" in graph_of(build_command(tl, tmp_path / "a.mp4",
                                                                      options=opts))

    off = RenderOptions()
    off.loudness_target = None
    assert "loudnorm" not in graph_of(build_command(tl, tmp_path / "b.mp4", options=off))

    # the explicit argument wins over the option
    assert "loudnorm=I=-20" in graph_of(
        build_command(tl, tmp_path / "c.mp4", options=off, loudness_target=-20.0)
    )


def test_loudness_target_outside_the_loudnorm_range_is_rejected(tmp_path: Path, vid: Path,
                                                                snd: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    with pytest.raises(RenderError) as excinfo:
        build_command(tl, tmp_path / "out.mp4", loudness_target=0.0)
    assert any("loudness_target" in problem for problem in excinfo.value.problems)


def test_a_silent_timeline_is_not_loudness_normalised(tmp_path: Path, vid: Path):
    """Nothing to normalise on the ``anullsrc`` path -- and nothing to amplify."""
    graph = graph_of(build_command(basic(vid), tmp_path / "out.mp4"))
    assert "anullsrc" in graph and "loudnorm" not in graph


def short_tail(tmp_path: Path, vid: Path, snd: Path, duration: float) -> str:
    tl = Timeline(width=320, height=568, fps=30, duration=duration)
    tl.add_visual(VisualLayer(kind="video", src=str(vid)))
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    return chain_with(build_command(tl, tmp_path / "out.mp4"), "amix=")


def test_every_duration_is_loudness_normalised_there_is_no_three_second_cliff(
    tmp_path: Path, vid: Path, snd: Path
):
    """The defect: ``loudnorm`` used to be skipped below three seconds, so the
    same source exported ~8 dB quieter at 2.9s than at 3.0s."""
    from aiclipper.render import LOUDNESS_MIN_DURATION

    assert LOUDNESS_MIN_DURATION == 3.0
    for duration in (0.5, 1.0, 2.0, 2.5, 2.999, 3.0, 4.5, 6.0):
        chain = short_tail(tmp_path, vid, snd, duration)
        assert "loudnorm=I=-14:TP=-1.5:LRA=11" in chain, duration
        assert "alimiter" in chain          # the limiter is never skipped


def test_a_programme_shorter_than_the_floor_is_looped_up_to_it_then_cut_back(
    tmp_path: Path, vid: Path, snd: Path
):
    """loudnorm needs three seconds of programme to gate on -- so it is *given*
    three seconds of the programme, and only the last copy reaches the output."""
    chain = short_tail(tmp_path, vid, snd, 2.0)
    samples = 2 * 48000
    copies = 1 + 2                          # ceil(3.0 / 2.0) extra copies
    assert f"atrim=end_sample={samples}" in chain           # exact-length programme
    assert f"aloop=loop={copies - 1}:size={samples}" in chain
    assert chain.index("aloop=") < chain.index("loudnorm=")
    # the kept copy is the final one, and the cut is still ahead of the limiter.
    # It is expressed in seconds because loudnorm hands on 192 kHz, not 48 kHz.
    assert f"atrim=start={(copies - 1) * 2}" in chain
    assert "start_sample" not in chain.split("loudnorm=")[1]
    assert chain.index("loudnorm=") < chain.index("atrim=start=") < chain.index("alimiter")

    # ...and a programme at or above the floor keeps the plain, unlooped tail
    long_chain = short_tail(tmp_path, vid, snd, 3.0)
    assert "aloop=" not in long_chain and "_sample=" not in long_chain
    assert "apad,atrim=duration=3,asetpts=PTS-STARTPTS,loudnorm=" in long_chain


def test_a_short_programme_with_normalisation_off_keeps_the_plain_tail(
    tmp_path: Path, vid: Path, snd: Path
):
    """The loop only exists to feed ``loudnorm``: with no target there is none."""
    tl = Timeline(width=320, height=568, fps=30, duration=2.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid)))
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    chain = chain_with(build_command(tl, tmp_path / "out.mp4", loudness_target=None), "amix=")
    assert "aloop=" not in chain and "_sample=" not in chain
    assert chain.endswith(
        "apad,atrim=duration=2,asetpts=PTS-STARTPTS,alimiter=limit=0.95:level=0,"
        "aresample=async=1,aformat=sample_fmts=fltp:sample_rates=48000:"
        "channel_layouts=stereo[aout]"
    )


# --------------------------------------------------------------------------- #
# subtitles and output options
# --------------------------------------------------------------------------- #

def test_subtitles_are_burned_on_the_final_chain(tmp_path: Path, vid: Path, img: Path, ass_path: Path):
    fonts = tmp_path / "fonts"
    fonts.mkdir()
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), z=0))
    tl.add_visual(VisualLayer(kind="image", src=str(img), z=1))
    tl.subtitles = SubtitleTrack(ass_path=ass_path, fonts_dir=fonts)
    graph = graph_of(build_command(tl, tmp_path / "out.mp4"))

    assert graph.index("subtitles=") > graph.rindex("overlay=")
    tail = chain_with(build_command(tl, tmp_path / "out.mp4"), "subtitles=")
    assert tail.startswith("[c1]subtitles=")
    assert f"filename={ff.escape_filter_path(ass_path)}" in tail
    assert f"fontsdir={ff.escape_filter_path(fonts)}" in tail
    assert tail.endswith("format=yuv420p[vout]")


def test_fontsdir_is_omitted_when_unset(tmp_path: Path, vid: Path, ass_path: Path):
    tl = basic(vid)
    tl.subtitles = SubtitleTrack(ass_path=ass_path)
    tail = chain_with(build_command(tl, tmp_path / "out.mp4"), "subtitles=")
    assert "fontsdir=" not in tail


def test_paths_with_awkward_characters_are_escaped(tmp_path: Path, vid: Path):
    """A filter option value passes through two parsers.

    The graph parser eats one level of backslashes before the per-filter option
    parser sees the string, so ``,`` (special only to the graph parser) needs
    one backslash while ``:`` and ``'`` (special to the option parser too) need
    theirs to survive the first pass.  Getting this wrong does not raise -- it
    silently drops the apostrophe or swallows every later option.
    """
    odd = tmp_path / "a b, c: d"
    odd.mkdir()
    ass = odd / "sub's.ass"
    ass.write_text(ASS_TEXT, encoding="utf-8")
    tl = basic(vid)
    tl.subtitles = SubtitleTrack(ass_path=ass, fonts_dir=odd)
    tail = chain_with(build_command(tl, tmp_path / "out.mp4"), "subtitles=")

    assert r"a b\, c\\: d" in tail        # one backslash for the comma, two for the colon
    assert r"sub\\\'s.ass" in tail        # three for the apostrophe
    assert "sub's.ass" not in tail
    # graph-level escaping alone is not enough, which is the easy mistake here
    assert ff.escape_filter_path(ass) not in tail
    # and both option values got the same treatment
    assert tail.count(r"a b\, c") == 2


def test_output_options_and_explicit_duration(tmp_path: Path, vid: Path):
    out = tmp_path / "out.mp4"
    options = RenderOptions(
        crf=27, preset="ultrafast", pix_fmt="yuv420p", audio_bitrate="96k",
        threads=2, extra_args=["-metadata", "comment=aiclipper"],
    )
    cmd = build_command(basic(vid), out, options=options)

    assert cmd[0].endswith("ffmpeg")
    assert cmd[1] == "-y"
    assert cmd[-1] == str(out)
    assert cmd[-3:-1] == ["-metadata", "comment=aiclipper"]
    assert "-shortest" not in cmd
    assert cmd[cmd.index("-t") + 1] == "4"
    assert cmd[cmd.index("-r") + 1] == "30"
    assert cmd[cmd.index("-crf") + 1] == "27"
    assert cmd[cmd.index("-preset") + 1] == "ultrafast"
    assert cmd[cmd.index("-pix_fmt") + 1] == "yuv420p"
    assert cmd[cmd.index("-c:v") + 1] == "libx264"
    assert cmd[cmd.index("-c:a") + 1] == "aac"
    assert cmd[cmd.index("-b:a") + 1] == "96k"
    assert cmd[cmd.index("-threads") + 1] == "2"
    assert cmd[cmd.index("-movflags") + 1] == "+faststart"
    assert cmd.count("-map") == 2


def test_overwrite_and_faststart_can_be_switched_off(tmp_path: Path, vid: Path):
    options = RenderOptions(overwrite=False, faststart=False)
    cmd = build_command(basic(vid), tmp_path / "out.mp4", options=options)
    assert "-y" not in cmd
    assert "-movflags" not in cmd


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #

def test_invalid_timeline_raises_with_every_problem(tmp_path: Path):
    tl = Timeline(width=321, height=568, fps=0, duration=0.0)
    tl.add_visual(VisualLayer(kind="video", src=None))
    with pytest.raises(RenderError) as excinfo:
        build_command(tl, tmp_path / "out.mp4")
    problems = excinfo.value.problems
    assert len(problems) >= 4
    assert any("even" in p for p in problems)
    assert any("fps" in p for p in problems)
    assert any("duration" in p for p in problems)
    assert any("no src" in p for p in problems)


def test_missing_visual_source_raises(tmp_path: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(tmp_path / "ghost.mp4")))
    with pytest.raises(RenderError) as excinfo:
        build_command(tl, tmp_path / "out.mp4")
    assert any("ghost.mp4" in p and "not found" in p for p in excinfo.value.problems)


def test_missing_audio_or_subtitle_file_raises(tmp_path: Path, vid: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(tmp_path / "gone.wav"), role="voice"))
    tl.subtitles = SubtitleTrack(ass_path=tmp_path / "gone.ass")
    with pytest.raises(RenderError) as excinfo:
        build_command(tl, tmp_path / "out.mp4")
    joined = " ".join(excinfo.value.problems)
    assert "gone.wav" in joined and "gone.ass" in joined


def test_ducking_without_a_voice_track_is_a_timeline_problem(tmp_path: Path, vid: Path, snd: Path):
    tl = basic(vid)
    tl.add_audio(AudioTrack(src=str(snd), role="music", duck=True))
    with pytest.raises(RenderError) as excinfo:
        build_command(tl, tmp_path / "out.mp4")
    assert any("ducking" in p for p in excinfo.value.problems)


def test_dry_run_returns_the_command_without_touching_the_disk(tmp_path: Path, vid: Path):
    out = tmp_path / "nested" / "out.mp4"
    result = render(basic(vid), out, options=FAST, dry_run=True)
    assert result.command[0].endswith("ffmpeg")
    assert result.command[-1] == str(out)
    assert (result.width, result.height, result.duration) == (320, 568, 4.0)
    assert not out.exists()
    assert not out.parent.exists()


# --------------------------------------------------------------------------- #
# real renders -- the graphs have to actually run
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_render_background_loop_audio_and_subtitles(tmp_path, make_video, make_audio, ass_path):
    src = make_video("loop.mp4", seconds=0.6, width=160, height=120, fps=12, audio=False)
    bed = make_audio("bed.wav", seconds=0.8)
    out = tmp_path / "story.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=2.0, background="#101820")
    tl.add_visual(VisualLayer(kind="video", src=str(src), loop=True, fit="cover", z=0))
    tl.add_audio(AudioTrack(src=str(bed), role="voice", loop=True, fade_out=0.3))
    tl.subtitles = SubtitleTrack(ass_path=ass_path)

    result = render(tl, out, options=FAST)

    assert out.exists() and result.size_bytes > 0
    info = ff.probe(out)
    assert info.has_video and info.has_audio
    assert (info.width, info.height) == (320, 568)
    assert abs(info.duration - 2.0) < 0.25
    assert 11.0 <= info.fps <= 13.0
    assert result.command[0].endswith("ffmpeg")

    # both sampled moments lie past the end of the 0.6s source: the layer is
    # still painting (not the bare background) and still advancing, so
    # -stream_loop really is looping rather than freezing the last frame.
    late = frame_at(out, 1.2, tmp_path / "late0.png")
    later = frame_at(out, 1.9, tmp_path / "late1.png")
    low, high = late.convert("L").getextrema()
    assert high - low > 40, "the looping layer does not seem to be drawn"
    assert mean_abs_diff(late, later) > 5.0, "the looping layer froze"


@pytest.mark.needs_ffmpeg
def test_render_two_pane_stack(tmp_path, make_video):
    top = make_video("top.mp4", seconds=1.2, source="color=c=red:s=160x120:rate=12", audio=False)
    bottom = make_video("bottom.mp4", seconds=1.2, source="color=c=blue:s=160x120:rate=12", audio=False)
    out = tmp_path / "split.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=1.0, background="#000000")
    tl.add_visual(VisualLayer(kind="video", src=str(top), x=0, y=0, w=320, h=284, fit="cover", z=0))
    tl.add_visual(VisualLayer(kind="video", src=str(bottom), x=0, y=284, w=320, h=284, fit="cover", z=1))

    result = render(tl, out, options=FAST)
    info = ff.probe(out)
    assert (info.width, info.height) == (320, 568)
    assert abs(info.duration - 1.0) < 0.25
    assert result.width == 320

    frame = frame_at(out, 0.5, tmp_path / "split.png")
    assert dominant(frame.getpixel((160, 140))) == "red"
    assert dominant(frame.getpixel((160, 420))) == "blue"


@pytest.mark.needs_ffmpeg
def test_render_animated_crop_path_pans(tmp_path, make_video):
    # a 320x120 source: red on the left half, blue on the right half
    wide = tmp_path / "wide.mp4"
    ff.run_ffmpeg([
        "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:rate=12:duration=1.5",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:rate=12:duration=1.5",
        "-filter_complex", "[0:v][1:v]hstack=inputs=2,format=yuv420p[v]",
        "-map", "[v]", "-t", "1.5", "-c:v", "libx264", "-preset", "ultrafast", str(wide),
    ])
    out = tmp_path / "pan.mp4"

    path = CropPath(
        [CropKeyframe(0.0, 0, 0, 120, 120), CropKeyframe(1.2, 200, 0, 120, 120)],
        source_width=320,
        source_height=120,
    )
    tl = Timeline(width=320, height=568, fps=12, duration=1.4, background="#000000")
    # a lower layer whose fit=cover puts another crop filter *earlier* in the
    # graph: the sendcmd script has to address its own crop instance, or the
    # pan silently steers this one instead.
    under = make_video("under.mp4", seconds=1.5, source="color=c=green:s=160x120:rate=12", audio=False)
    tl.add_visual(VisualLayer(kind="video", src=str(under), fit="cover", z=-1))
    tl.add_visual(VisualLayer(kind="video", src=str(wide), crop=path, fit="cover", z=0))

    workdir = tmp_path / "graph"
    result = render(tl, out, options=FAST)
    assert "sendcmd" in graph_of(result.command)

    info = ff.probe(out)
    assert (info.width, info.height) == (320, 568)
    assert abs(info.duration - 1.4) < 0.25

    start = frame_at(out, 0.1, tmp_path / "pan0.png")
    end = frame_at(out, 1.25, tmp_path / "pan1.png")
    assert dominant(start.getpixel((160, 284))) == "red"
    assert dominant(end.getpixel((160, 284))) == "blue"

    # the same timeline built through build_command writes its script where asked
    build_command(tl, out, options=FAST, workdir=workdir)
    assert len(list(workdir.glob("*.cmd"))) == 1


@pytest.mark.needs_ffmpeg
def test_render_ducked_music_under_voice(tmp_path, make_video, make_audio):
    src = make_video("bg.mp4", seconds=1.0, width=160, height=120, fps=12, audio=False)
    voice = make_audio("voice.wav", seconds=1.0, frequency=440)
    music = make_audio("music.wav", seconds=0.5, frequency=110)
    out = tmp_path / "ducked.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=1.0)
    tl.add_visual(VisualLayer(kind="video", src=str(src), fit="contain", opacity=0.8, z=0))
    tl.add_visual(VisualLayer(kind="color", color="#204060", x=0, y=500, w=320, h=68, start=0.2, z=1))
    tl.add_audio(AudioTrack(src=str(voice), role="voice", start=0.1))
    tl.add_audio(AudioTrack(src=str(music), role="music", loop=True, gain_db=-6, duck=True))

    result = render(tl, out, options=FAST)
    assert "sidechaincompress" in graph_of(result.command)
    info = ff.probe(out)
    assert info.has_video and info.has_audio
    assert abs(info.duration - 1.0) < 0.25


# --------------------------------------------------------------------------- #
# contract, determinism, degenerate timelines
# --------------------------------------------------------------------------- #

def test_public_api_matches_the_contract():
    import inspect

    from aiclipper import render as mod

    assert set(mod.__all__) == {"render", "build_command"}

    sig = inspect.signature(mod.render)
    assert list(sig.parameters) == [
        "timeline", "out_path", "options", "settings", "log_path", "dry_run", "loudness_target",
    ]
    assert sig.parameters["timeline"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("options", "settings", "log_path", "dry_run", "loudness_target"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    assert sig.parameters["options"].default is None
    assert sig.parameters["settings"].default is None
    assert sig.parameters["log_path"].default is None
    assert sig.parameters["dry_run"].default is False

    sig = inspect.signature(mod.build_command)
    assert list(sig.parameters) == ["timeline", "out_path", "options", "settings", "workdir",
                                    "loudness_target"]
    for name in ("options", "settings", "workdir", "loudness_target"):
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY, name
    for name in ("options", "settings", "workdir"):
        assert sig.parameters[name].default is None
    # the added argument defaults to "ask ``options``", so existing callers are unaffected
    assert sig.parameters["loudness_target"].default is mod.UNSET


def test_render_returns_a_render_result(tmp_path: Path, vid: Path):
    from aiclipper.models import RenderResult

    result = render(basic(vid), tmp_path / "out.mp4", dry_run=True)
    assert isinstance(result, RenderResult)
    assert isinstance(result.path, Path) and isinstance(result.command, list)
    assert all(isinstance(token, str) for token in result.command)


def test_build_command_is_deterministic(tmp_path: Path, vid: Path, img: Path, snd: Path):
    def make() -> Timeline:
        tl = Timeline(width=320, height=568, fps=30, duration=4.0)
        for z in (3, 1, 2):
            tl.add_visual(VisualLayer(kind="video", src=str(vid), z=z, label=f"L{z}"))
        tl.add_visual(VisualLayer(kind="image", src=str(img), z=0))
        tl.add_audio(AudioTrack(src=str(snd), role="voice"))
        return tl

    first = build_command(make(), tmp_path / "out.mp4")
    second = build_command(make(), tmp_path / "out.mp4")
    assert first == second
    # z order, not insertion order, decides the input order
    assert [b[-1] for b in input_blocks(first)] == [str(img), str(vid), str(vid), str(vid), str(snd)]


def test_timeline_without_visuals_still_renders_a_canvas(tmp_path: Path, snd: Path):
    tl = Timeline(width=320, height=568, fps=30, duration=4.0, background="#123456")
    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    cmd = build_command(tl, tmp_path / "out.mp4")

    assert chains(cmd)[0] == "color=c=0x123456:s=320x568:r=30:d=4[base]"
    assert "overlay=" not in graph_of(cmd)
    assert chain_with(cmd, "[vout]").startswith("[base]")
    assert input_paths(cmd) == [str(snd)]


def test_take_audio_is_ignored_for_layers_that_have_no_audio_stream(tmp_path: Path, img: Path, snd: Path):
    """An image decodes to a single video stream and a colour layer is lavfi --
    referencing ``[n:a]`` for either would make ffmpeg abort."""
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="image", src=str(img), take_audio=True, z=0))
    tl.add_visual(VisualLayer(kind="color", color="#ffffff", take_audio=True, z=1))
    graph = graph_of(build_command(tl, tmp_path / "out.mp4"))
    assert "[0:a]" not in graph
    assert "anullsrc" in graph          # nothing contributed, so the mix is synthesised

    tl.add_audio(AudioTrack(src=str(snd), role="voice"))
    graph = graph_of(build_command(tl, tmp_path / "out.mp4"))
    assert "[0:a]" not in graph and "[1:a]" in graph


def test_dry_run_still_writes_the_sendcmd_script(tmp_path: Path, vid: Path):
    """The command handed back has to be runnable, so its side files must exist."""
    path = CropPath(
        [CropKeyframe(0.0, 0, 0, 100, 80), CropKeyframe(1.0, 60, 0, 100, 80)], 200, 160
    )
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path))
    out = tmp_path / "nested" / "out.mp4"

    result = render(tl, out, dry_run=True)
    script = re.search(r"sendcmd=f=(\S+?),crop@", graph_of(result.command)).group(1)
    written = script.replace("\\", "")
    assert Path(written).is_file()
    assert not out.exists()


def animated(vid: Path) -> Timeline:
    path = CropPath(
        [CropKeyframe(0.0, 0, 0, 100, 80), CropKeyframe(1.0, 60, 0, 100, 80)], 200, 160
    )
    tl = Timeline(width=320, height=568, fps=30, duration=4.0)
    tl.add_visual(VisualLayer(kind="video", src=str(vid), crop=path))
    return tl


def test_animated_crop_defaults_to_the_settings_work_dir(tmp_path: Path, vid: Path, settings):
    build_command(animated(vid), tmp_path / "out.mp4")
    scripts = list((settings.work_dir / "render").rglob("*.cmd"))
    assert len(scripts) == 1
    assert scripts[0].parent.name.startswith("out-")      # readable stem + key


def test_render_keeps_its_scratch_files_out_of_the_output_directory(
    tmp_path: Path, vid: Path, settings
):
    """The defect: ``render`` wrote its sendcmd scripts next to the finished
    video, littering the user's output directory with ``.<stem>-render/``."""
    out = tmp_path / "clips" / "source-wide-01-part-1.mp4"
    out.parent.mkdir(parents=True)

    render(animated(vid), out, dry_run=True)

    assert list(out.parent.iterdir()) == [], f"build litter in the output dir: {out.parent}"
    assert list((settings.work_dir / "render").rglob("*.cmd"))


def test_the_default_workdir_cannot_collide_between_concurrent_renders(
    tmp_path: Path, vid: Path, settings
):
    """Two renders whose outputs merely share a stem used to share one scratch
    directory, so each would overwrite the other's sendcmd scripts."""
    def script_dir(out: Path) -> Path:
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_command(animated(vid), out)
        return Path(re.search(r"sendcmd=f=(\S+?),crop@", graph_of(cmd)).group(1)
                    .replace("\\", "")).parent

    one = script_dir(tmp_path / "a" / "out.mp4")
    two = script_dir(tmp_path / "b" / "out.mp4")
    assert one != two, "same scratch dir for two different outputs"
    assert one.parent == two.parent == settings.work_dir / "render"
    # ...but the same output is still the same (reproducible) directory
    assert script_dir(tmp_path / "a" / "out.mp4") == one


# --------------------------------------------------------------------------- #
# more real renders
# --------------------------------------------------------------------------- #

def mean_volume(path: Path, start: float, length: float) -> float:
    """Mean dBFS over ``[start, start+length)`` of a rendered file."""
    proc = ff.run_ffmpeg(
        ["-ss", f"{start:.3f}", "-t", f"{length:.3f}", "-i", str(path),
         "-af", "volumedetect", "-f", "null", "-"],
        quiet=False,
    )
    values = [
        float(line.split("mean_volume:")[1].split("dB")[0])
        for line in proc.stderr.splitlines() if "mean_volume:" in line
    ]
    assert values, proc.stderr
    return values[-1]


@pytest.fixture
def rgb_clip(tmp_path: Path) -> Path:
    """3s at 20fps: red for 0-1s, green for 1-2s, blue for 2-3s."""
    out = tmp_path / "rgb.mp4"
    ff.run_ffmpeg([
        "-y",
        "-f", "lavfi", "-i", "color=c=red:s=160x120:rate=20:duration=1",
        "-f", "lavfi", "-i", "color=c=lime:s=160x120:rate=20:duration=1",
        "-f", "lavfi", "-i", "color=c=blue:s=160x120:rate=20:duration=1",
        "-filter_complex", "[0:v][1:v][2:v]concat=n=3:v=1:a=0,format=yuv420p[v]",
        "-map", "[v]", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18", str(out),
    ])
    return out


def hue(pixel: tuple[int, int, int]) -> str:
    r, g, b = pixel
    if r > 110 and g < 90 and b < 90:
        return "red"
    if g > 110 and r < 90 and b < 90:
        return "green"
    if b > 110 and r < 90 and g < 90:
        return "blue"
    if max(pixel) < 45:
        return "black"
    return f"other{pixel}"


@pytest.mark.needs_ffmpeg
def test_render_seeks_src_start_to_the_right_frame(tmp_path: Path, rgb_clip: Path):
    out = tmp_path / "seek.mp4"
    tl = Timeline(width=320, height=568, fps=20, duration=2.0, background="#000000")
    tl.add_visual(VisualLayer(kind="video", src=str(rgb_clip), src_start=1.0, fit="cover"))
    render(tl, out, options=FAST)

    # the green second of the source is now the first second of the output
    assert hue(frame_at(out, 0.2, tmp_path / "s0.png").getpixel((160, 284))) == "green"
    assert hue(frame_at(out, 1.3, tmp_path / "s1.png").getpixel((160, 284))) == "blue"


@pytest.mark.needs_ffmpeg
def test_render_delays_a_layer_without_eating_its_first_frames(tmp_path: Path, rgb_clip: Path):
    """``start`` must postpone the layer, not scrub into it: at ``start`` the
    viewer sees the source's *first* frame, and the background before that."""
    out = tmp_path / "late.mp4"
    tl = Timeline(width=320, height=568, fps=20, duration=3.0, background="#000000")
    tl.add_visual(VisualLayer(kind="video", src=str(rgb_clip), start=1.0, end=2.5, fit="cover"))
    render(tl, out, options=FAST)

    assert hue(frame_at(out, 0.4, tmp_path / "l0.png").getpixel((160, 284))) == "black"
    assert hue(frame_at(out, 1.2, tmp_path / "l1.png").getpixel((160, 284))) == "red"
    assert hue(frame_at(out, 2.1, tmp_path / "l2.png").getpixel((160, 284))) == "green"
    assert hue(frame_at(out, 2.8, tmp_path / "l3.png").getpixel((160, 284))) == "black"


@pytest.mark.needs_ffmpeg
def test_render_places_a_delayed_track_at_the_right_second(tmp_path: Path, make_video):
    """Proves ``adelay`` really is milliseconds: a 1s tone asked for at t=2 has
    to be silent for the first two seconds and audible in the third."""
    src = make_video("bg.mp4", seconds=1.0, width=160, height=120, fps=12, audio=False)
    tone = tmp_path / "tone.wav"
    ff.run_ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
                   "-af", "volume=6", "-c:a", "pcm_s16le", str(tone)])
    out = tmp_path / "delay.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=3.0)
    tl.add_visual(VisualLayer(kind="video", src=str(src), loop=True, z=0))
    tl.add_audio(AudioTrack(src=str(tone), role="voice", start=2.0))
    render(tl, out, options=RenderOptions(preset="ultrafast", crf=32, audio_bitrate="128k"))

    quiet = mean_volume(out, 0.1, 1.5)
    loud = mean_volume(out, 2.1, 0.8)
    assert loud - quiet > 30, (quiet, loud)


@pytest.mark.needs_ffmpeg
def test_render_rejects_take_audio_from_a_source_without_audio(tmp_path: Path, make_video):
    src = make_video("mute.mp4", seconds=1.0, width=160, height=120, fps=12, audio=False)
    tl = Timeline(width=320, height=568, fps=12, duration=1.0)
    tl.add_visual(VisualLayer(kind="video", src=str(src), take_audio=True, label="pane"))

    # the command still builds -- build_command never decodes anything
    assert "[0:a]" in graph_of(build_command(tl, tmp_path / "out.mp4"))
    with pytest.raises(RenderError) as excinfo:
        render(tl, tmp_path / "out.mp4", options=FAST)
    assert any("take_audio" in p and "no audio stream" in p for p in excinfo.value.problems)
    assert not (tmp_path / "out.mp4").exists()


@pytest.mark.needs_ffmpeg
def test_render_writes_the_ffmpeg_log_and_probes_the_output(tmp_path: Path, make_video):
    src = make_video("bg.mp4", seconds=1.0, width=160, height=120, fps=12, audio=True)
    out = tmp_path / "logged.mp4"
    log = tmp_path / "logs" / "render.log"

    tl = Timeline(width=320, height=568, fps=12, duration=1.0)
    tl.add_visual(VisualLayer(kind="video", src=str(src), take_audio=True, fit="cover"))
    result = render(tl, out, options=FAST, log_path=log)

    text = log.read_text(encoding="utf-8")
    assert "-filter_complex" in text and str(out) in text
    # the result reports the file, not just the timeline's wishes
    assert result.size_bytes == out.stat().st_size > 0
    assert (result.width, result.height) == (320, 568)
    assert 11.0 <= result.fps <= 13.0


@pytest.mark.needs_ffmpeg
def test_render_burns_subtitles_from_an_awkward_path_with_a_fonts_dir(tmp_path: Path):
    """The ASS file and the fonts dir both sit in a directory whose name carries
    a comma and an apostrophe: unescaped, the filter string would not even
    parse, so a successful render with visible glyphs proves the escaping."""
    font = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if not font.is_file():
        pytest.skip("no DejaVu font on this machine")

    odd = tmp_path / "sub's, fonts"
    odd.mkdir()
    (odd / font.name).write_bytes(font.read_bytes())
    ass = odd / "cue.ass"
    ass.write_text(
        ASS_TEXT.replace(
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,burned in",
            "Dialogue: 0,0:00:00.00,0:00:01.00,Default,,0,0,0,,WWWWWWWW",
        ).replace("Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,second cue\n", ""),
        encoding="utf-8",
    )
    out = tmp_path / "subs.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=2.0, background="#000000")
    tl.subtitles = SubtitleTrack(ass_path=ass, fonts_dir=odd)
    result = render(tl, out, options=FAST)

    assert "fontsdir=" in graph_of(result.command)
    during = frame_at(out, 0.5, tmp_path / "sub0.png").crop((0, 440, 320, 540))
    after = frame_at(out, 1.5, tmp_path / "sub1.png").crop((0, 440, 320, 540))
    assert during.convert("L").getextrema()[1] > 180, "no glyphs were burned in"
    assert after.convert("L").getextrema()[1] < 40, "the cue outlived its window"


# --------------------------------------------------------------------------- #
# loudness normalisation (real renders, measured with ebur128)
# --------------------------------------------------------------------------- #

def loudness(path: Path) -> tuple[float, float]:
    """``(integrated LUFS, true peak dBFS)`` of a rendered file, via ``ebur128``.

    ebur128 gates at -70 LUFS absolute, so a silent file reports exactly
    ``-70.0`` with a ``-inf`` peak: that pair *is* the signature of digital
    silence.
    """
    proc = ff.run_ffmpeg(
        ["-i", str(path), "-filter_complex", "ebur128=peak=true", "-f", "null", "-"],
        quiet=False,
    )
    summary = proc.stderr[proc.stderr.rfind("Integrated loudness"):]

    def grab(pattern: str) -> float:
        match = re.search(pattern, summary)
        assert match, summary
        return float("-inf") if "inf" in match.group(1) else float(match.group(1))

    return grab(r"I:\s+(-inf|-?[\d.]+) LUFS"), grab(r"Peak:\s+(-inf|-?[\d.]+) dBFS")


@pytest.fixture
def quiet_mix(tmp_path: Path, make_video, make_audio):
    """Factory for the kind of mix every workflow builds: narration over a bed."""
    bg = make_video("bg.mp4", seconds=3.0, width=160, height=120, fps=12, audio=False)
    voice = make_audio("voice.wav", seconds=3.0, frequency=420.0)
    bed = make_audio("bed.wav", seconds=3.0, frequency=180.0)

    def _make() -> Timeline:
        tl = Timeline(width=320, height=568, fps=12, duration=3.0)
        tl.add_visual(VisualLayer(kind="video", src=str(bg), loop=True, fit="cover"))
        tl.add_audio(AudioTrack(src=str(voice), role="voice"))
        tl.add_audio(AudioTrack(src=str(bed), role="music", gain_db=-8.0))
        return tl

    return _make


@pytest.mark.needs_ffmpeg
def test_render_lands_on_the_loudness_target(tmp_path: Path, quiet_mix):
    """The defect: the raw mix exports around -34 LUFS, ~20 LU below platform level."""
    before_path, after_path = tmp_path / "raw.mp4", tmp_path / "normalised.mp4"
    render(quiet_mix(), before_path, options=FAST, loudness_target=None)
    render(quiet_mix(), after_path, options=FAST)

    before, _ = loudness(before_path)
    after, peak = loudness(after_path)

    assert before < -25.0, f"the un-normalised mix was expected to be quiet, got {before}"
    assert abs(after + 14.0) <= 1.5, f"integrated loudness {after} LUFS is not near -14"
    assert after - before > 10.0
    assert peak < 0.0, f"true peak {peak} dBFS clips"


@pytest.mark.needs_ffmpeg
def test_normalised_output_does_not_clip(tmp_path: Path, make_video):
    """A deliberately hot source: the limiter downstream of loudnorm holds the ceiling."""
    bg = make_video("bg.mp4", seconds=3.5, width=160, height=120, fps=12, audio=False)
    hot = tmp_path / "hot.wav"
    ff.run_ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=3.5",
                   "-af", "volume=4", "-c:a", "pcm_s16le", str(hot)])
    out = tmp_path / "hot.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=3.5)
    tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
    tl.add_audio(AudioTrack(src=str(hot), role="voice"))
    render(tl, out, options=FAST)

    integrated, peak = loudness(out)
    assert peak <= -0.5, f"true peak {peak} dBFS leaves no headroom"
    assert mean_volume(out, 0.2, 3.0) < 0.0
    assert integrated <= -12.0, f"normalisation overshot upward: {integrated} LUFS"


@pytest.mark.needs_ffmpeg
def test_loudness_normalisation_leaves_digital_silence_silent(tmp_path: Path, make_video):
    """loudnorm's -70 LUFS absolute gate is what keeps a silent track silent."""
    bg = make_video("bg.mp4", seconds=3.5, width=160, height=120, fps=12, audio=False)
    silent = ff.make_silence(3.5, tmp_path / "narration.wav")
    out = tmp_path / "silent.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=3.5)
    tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
    tl.add_audio(AudioTrack(src=str(silent), role="voice"))
    render(tl, out, options=FAST)

    integrated, peak = loudness(out)
    assert integrated <= -70.0, f"silence was amplified to {integrated} LUFS"
    assert peak == float("-inf") or peak < -80.0
    assert mean_volume(out, 0.2, 3.0) < -80.0


@pytest.mark.needs_ffmpeg
def test_loudness_normalisation_does_not_lift_an_inaudible_track_into_hiss(
    tmp_path: Path, make_video
):
    """A -60 dBFS track is below the gate too: it stays inaudible instead of
    being hauled up 50 dB into a wall of noise."""
    bg = make_video("bg.mp4", seconds=3.5, width=160, height=120, fps=12, audio=False)
    whisper = tmp_path / "whisper.wav"
    ff.run_ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=3.5",
                   "-af", "volume=-60dB", "-c:a", "pcm_s16le", str(whisper)])
    out = tmp_path / "whisper.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=3.5)
    tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
    tl.add_audio(AudioTrack(src=str(whisper), role="voice"))
    render(tl, out, options=FAST)

    integrated, _ = loudness(out)
    assert integrated <= -70.0, f"an inaudible track was normalised up to {integrated} LUFS"
    assert mean_volume(out, 0.2, 3.0) < -70.0


@pytest.mark.needs_ffmpeg
def test_split_narration_routing_is_fine_the_offline_tts_is_just_silent(
    tmp_path: Path, make_video
):
    """Settles the audit's "-70 LUFS split render" finding: not a routing bug.

    The same two-pane timeline is rendered twice, with only the narration file
    swapped.  The filter graph is byte-identical, so the routing cannot differ;
    ``ffmpeg.make_silence`` -- exactly what ``tts/offline.py`` writes in the
    sandbox -- reaches the mix as digital silence, while an audible tone in the
    very same slot reaches the output on target.
    """
    top = make_video("top.mp4", seconds=3.5, width=160, height=120, fps=12, audio=False)
    bottom = make_video("bottom.mp4", seconds=3.5, width=160, height=120, fps=12, audio=False)
    offline = ff.make_silence(3.5, tmp_path / "offline-tts.wav")     # what tts/offline.py writes
    audible = ff.make_tone(3.5, tmp_path / "real-voice.wav", frequency=320.0, volume=0.25)

    def split(narration: Path) -> Timeline:
        tl = Timeline(width=320, height=568, fps=12, duration=3.5)
        tl.add_visual(VisualLayer(kind="video", src=str(top), x=0, y=0, w=320, h=284,
                                  fit="cover", z=0))
        tl.add_visual(VisualLayer(kind="video", src=str(bottom), x=0, y=284, w=320, h=284,
                                  fit="cover", z=1))
        tl.add_audio(AudioTrack(src=str(narration), role="voice"))
        return tl

    silent_out, audible_out = tmp_path / "offline.mp4", tmp_path / "audible.mp4"
    silent_cmd = render(split(offline), silent_out, options=FAST).command
    audible_cmd = render(split(audible), audible_out, options=FAST).command

    # identical routing: the graphs differ only in which narration file is read
    assert graph_of(silent_cmd) == graph_of(audible_cmd)
    assert "[2:a]" in graph_of(silent_cmd) and "amix=inputs=1" in graph_of(silent_cmd)

    quiet_i, quiet_peak = loudness(silent_out)
    loud_i, _ = loudness(audible_out)
    assert quiet_i <= -70.0 and quiet_peak == float("-inf")   # the offline provider, not routing
    assert abs(loud_i + 14.0) <= 1.5, f"the narration bus is broken: {loud_i} LUFS"


@pytest.mark.needs_ffmpeg
def test_a_short_silent_render_is_not_blasted_to_full_scale(tmp_path: Path, make_video):
    """The reason :data:`LOUDNESS_MIN_DURATION` exists.

    ffmpeg 6.1's ``loudnorm,alimiter`` pair hands back below-gate audio at
    about 0 dBFS when the filter is handed less than three seconds -- a silent
    2s render would come out as a full-scale blast.  The renderer loops the
    programme up to the floor before the filter sees it, so the gate engages
    and silence stays silence *and* the short render is still normalised.
    """
    bg = make_video("bg.mp4", seconds=2.0, width=160, height=120, fps=12, audio=False)
    silent = ff.make_silence(2.0, tmp_path / "narration.wav")
    out = tmp_path / "short.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=2.0)
    tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
    tl.add_audio(AudioTrack(src=str(silent), role="voice"))
    result = render(tl, out, options=FAST)

    assert mean_volume(out, 0.1, 1.8) < -80.0, "a silent short render came out loud"
    assert loudness(out)[1] == float("-inf")
    assert "loudnorm" in graph_of(result.command)


@pytest.mark.needs_ffmpeg
def test_a_short_inaudible_render_stays_below_the_gate(tmp_path: Path, make_video):
    """The other half of the short-programme blow-up: handed 2s directly,
    ffmpeg 6.1's loudnorm hauls a -60 dBFS track up to about -2 LUFS."""
    bg = make_video("bg.mp4", seconds=2.0, width=160, height=120, fps=12, audio=False)
    whisper = tmp_path / "whisper.wav"
    ff.run_ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
                   "-af", "volume=-60dB", "-c:a", "pcm_s16le", str(whisper)])
    out = tmp_path / "whisper-short.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=2.0)
    tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
    tl.add_audio(AudioTrack(src=str(whisper), role="voice"))
    render(tl, out, options=FAST)

    integrated, _ = loudness(out)
    assert integrated <= -70.0, f"a 2s inaudible track was normalised up to {integrated} LUFS"
    assert mean_volume(out, 0.1, 1.8) < -70.0


@pytest.mark.needs_ffmpeg
def test_there_is_no_loudness_step_at_the_three_second_boundary(tmp_path: Path, make_video):
    """The defect: the same source measured -21.8 LUFS at 2.9s and -14.0 LUFS
    at 3.0s -- a 7.8 dB cliff a caller cannot see, straddled by two clips out
    of one batch."""
    bg = make_video("bg.mp4", seconds=4.0, width=160, height=120, fps=12, audio=False)
    voice = ff.make_tone(4.0, tmp_path / "voice.wav", frequency=220.0, volume=0.06)

    def measure(duration: float) -> float:
        out = tmp_path / f"d{duration}.mp4"
        tl = Timeline(width=320, height=568, fps=12, duration=duration)
        tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
        tl.add_audio(AudioTrack(src=str(voice), role="voice"))
        render(tl, out, options=FAST)
        return loudness(out)[0]

    below, above = measure(2.9), measure(3.0)
    assert abs(below - above) <= 1.5, f"{below} LUFS at 2.9s vs {above} LUFS at 3.0s"
    assert abs(below + 14.0) <= 1.5, f"2.9s exported at {below} LUFS, nowhere near -14"


@pytest.mark.needs_ffmpeg
def test_short_renders_land_on_the_target_across_the_whole_range(tmp_path: Path, make_video):
    """Sampled across the sub-floor range: every duration reaches the target,
    the curve has no step, and the limiter still holds the ceiling."""
    bg = make_video("bg.mp4", seconds=4.0, width=160, height=120, fps=12, audio=False)
    voice = ff.make_tone(4.0, tmp_path / "voice.wav", frequency=220.0, volume=0.06)

    curve: dict[float, float] = {}
    for duration in (0.5, 1.5, 2.5, 3.5):
        out = tmp_path / f"c{duration}.mp4"
        tl = Timeline(width=320, height=568, fps=12, duration=duration)
        tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
        tl.add_audio(AudioTrack(src=str(voice), role="voice"))
        render(tl, out, options=FAST)
        integrated, peak = loudness(out)
        curve[duration] = integrated
        assert peak < -0.5, f"{duration}s render peaks at {peak} dBFS"
        assert abs(ff.probe(out).duration - duration) < 0.2, "the loop leaked into the output"

    assert all(abs(value + 14.0) <= 1.5 for value in curve.values()), curve
    assert max(curve.values()) - min(curve.values()) <= 1.5, curve


@pytest.mark.needs_ffmpeg
def test_a_looped_short_render_still_outputs_the_programme_not_a_repeat(
    tmp_path: Path, make_video
):
    """Only the *filter* sees the looped mix: the audio that lands in the file
    is the programme itself, so a clip that goes quiet stays quiet."""
    bg = make_video("bg.mp4", seconds=2.0, width=160, height=120, fps=12, audio=False)
    burst = tmp_path / "burst.wav"
    ff.run_ffmpeg(["-y", "-f", "lavfi", "-i", "sine=frequency=300:duration=0.5",
                   "-af", "volume=0.2,apad", "-t", "2.0", "-c:a", "pcm_s16le", str(burst)])
    out = tmp_path / "burst.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=2.0)
    tl.add_visual(VisualLayer(kind="video", src=str(bg), fit="cover"))
    tl.add_audio(AudioTrack(src=str(burst), role="voice"))
    render(tl, out, options=FAST)

    assert mean_volume(out, 0.05, 0.4) > -30.0, "the burst did not survive"
    assert mean_volume(out, 1.0, 0.9) < -60.0, "a looped copy leaked into the output"


@pytest.mark.needs_ffmpeg
def test_a_real_render_leaves_nothing_but_the_output_file_behind(tmp_path: Path, make_video):
    """The output directory receives the finished video and nothing else."""
    bg = make_video("bg.mp4", seconds=2.0, width=160, height=120, fps=12, audio=False)
    crop = CropPath(
        [CropKeyframe(0.0, 0, 0, 80, 120), CropKeyframe(1.0, 80, 0, 80, 120)], 160, 120
    )
    out_dir = tmp_path / "delivery"
    out = out_dir / "short.mp4"

    tl = Timeline(width=320, height=568, fps=12, duration=2.0)
    tl.add_visual(VisualLayer(kind="video", src=str(bg), crop=crop, fit="cover"))
    render(tl, out, options=FAST)

    assert "sendcmd" in graph_of(render(tl, out, options=FAST, dry_run=True).command)
    assert [p.name for p in out_dir.iterdir()] == ["short.mp4"]
