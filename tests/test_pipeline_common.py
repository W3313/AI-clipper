"""Tests for :mod:`aiclipper.pipelines.common`.

Every audio fixture here is generated with ffmpeg, so the assertions are made
against files that really exist and really have the durations we claim.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import pytest

from aiclipper import ffmpeg as ff
from aiclipper import transcribe as transcribe_module
from aiclipper.assets import Asset
from aiclipper.models import CaptionStyle, RenderOptions, Timeline, TTSResult, VoiceSpec, Word
from aiclipper.pipelines import common

pytestmark = pytest.mark.needs_ffmpeg


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture
def make_part(tmp_path: Path) -> Callable[..., Path]:
    """Factory for a narration-sized audio part with a controllable real length."""

    def _make(
        name: str,
        seconds: float,
        *,
        sample_rate: int = 44100,
        channels: int = 1,
        frequency: float = 300.0,
        trim: float | None = None,
    ) -> Path:
        out = tmp_path / name
        args = [
            "-y", "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={seconds}",
            "-ac", str(channels),
        ]
        if trim is not None:
            args += ["-t", f"{trim}"]
        args += ["-c:a", "pcm_s16le", str(out)]
        ff.run_ffmpeg(args)
        return out

    return _make


def _stream_format(path: Path) -> tuple[int, int]:
    raw = ff.run_ffprobe(
        ["-select_streams", "a:0", "-show_entries", "stream=sample_rate,channels",
         "-of", "csv=p=0", str(path)]
    ).strip()
    rate, channels = raw.split(",")[:2]
    return (int(rate), int(channels))


def _assert_monotonic(words: list[Word]) -> None:
    assert words, "expected a non-empty word list"
    previous = 0.0
    for word in words:
        assert word.end > word.start, f"{word.text!r} has no duration"
        assert word.start >= previous - 1e-9, f"{word.text!r} overlaps the previous word"
        previous = word.end


# --------------------------------------------------------------------------- #
# concat_audio
# --------------------------------------------------------------------------- #

def test_concat_offsets_follow_probed_not_requested_durations(tmp_path: Path, make_part):
    # The middle part is *asked* for 2.0s but truncated to 0.63s on the way out:
    # offsets built from the request would be a full second wrong.
    first = make_part("a.wav", 0.5)
    short = make_part("b.wav", 2.0, trim=0.63)
    third = make_part("c.wav", 0.4)
    probed = common.probe_durations([first, short, third])
    assert probed[1] == pytest.approx(0.63, abs=0.02)

    out, offsets = common.concat_audio([first, short, third], tmp_path / "joined.wav")

    assert out.exists() and out.stat().st_size > 1000
    assert offsets[0] == 0.0
    assert offsets[1] == pytest.approx(probed[0], abs=1e-3)
    assert offsets[2] == pytest.approx(probed[0] + probed[1], abs=1e-3)
    assert common.probe_duration(out) == pytest.approx(sum(probed), abs=0.05)


def test_concat_gap_arithmetic(tmp_path: Path, make_part):
    parts = [make_part("p0.wav", 0.4), make_part("p1.wav", 0.6), make_part("p2.wav", 0.5)]
    probed = common.probe_durations(parts)

    out, offsets = common.concat_audio(parts, tmp_path / "gapped.wav", gap=0.25)

    assert offsets[1] - offsets[0] == pytest.approx(probed[0] + 0.25, abs=1e-3)
    assert offsets[2] - offsets[1] == pytest.approx(probed[1] + 0.25, abs=1e-3)
    # Two gaps between three parts -- never a trailing one.
    assert common.probe_duration(out) == pytest.approx(sum(probed) + 0.5, abs=0.05)


def test_concat_single_part_is_still_normalised(tmp_path: Path, make_part):
    odd = make_part("odd.wav", 0.55, sample_rate=8000, channels=2)
    assert _stream_format(odd) == (8000, 2)

    out, offsets = common.concat_audio([odd], tmp_path / "single.wav")

    assert offsets == [0.0]
    assert _stream_format(out) == (common.NARRATION_SAMPLE_RATE, 1)
    assert common.probe_duration(out) == pytest.approx(0.55, abs=0.03)


def test_concat_mixed_sample_rates_and_channels(tmp_path: Path, make_part):
    parts = [
        make_part("m0.wav", 0.4, sample_rate=8000, channels=2),
        make_part("m1.wav", 0.5, sample_rate=22050, channels=1),
        make_part("m2.wav", 0.3, sample_rate=48000, channels=2),
    ]
    probed = common.probe_durations(parts)

    out, offsets = common.concat_audio(parts, tmp_path / "mixed.wav", gap=0.1)

    assert _stream_format(out) == (common.NARRATION_SAMPLE_RATE, 1)
    assert common.probe_duration(out) == pytest.approx(sum(probed) + 0.2, abs=0.05)
    assert offsets[2] == pytest.approx(probed[0] + probed[1] + 0.2, abs=1e-3)


def test_concat_rejects_no_parts(tmp_path: Path):
    with pytest.raises(ValueError):
        common.concat_audio([], tmp_path / "nothing.wav")


def test_concat_rejects_missing_part(tmp_path: Path, make_part):
    good = make_part("good.wav", 0.3)
    with pytest.raises(FileNotFoundError):
        common.concat_audio([good, tmp_path / "ghost.wav"], tmp_path / "out.wav")


def test_concat_rejects_silent_video(tmp_path: Path, make_part, make_video):
    mute = make_video("mute.mp4", seconds=0.5, width=64, height=64, fps=12, audio=False)
    with pytest.raises(ValueError, match="no audio stream"):
        common.concat_audio([make_part("x.wav", 0.3), mute], tmp_path / "out.wav")


# --------------------------------------------------------------------------- #
# narration_words
# --------------------------------------------------------------------------- #

@pytest.fixture
def narration(tmp_path: Path) -> tuple[list[Path], Path, list[float]]:
    parts = [ff.make_silence(0.8, tmp_path / "n0.wav"), ff.make_silence(1.2, tmp_path / "n1.wav")]
    joined, offsets = common.concat_audio(parts, tmp_path / "narration.wav", gap=0.2)
    return parts, joined, offsets


def test_narration_words_uses_provider_timings(narration):
    parts, joined, offsets = narration
    results = [
        TTSResult(parts[0], 0.8, [Word("hello", 0.0, 0.4), Word("there", 0.4, 0.8)], VoiceSpec(), "hello there"),
        TTSResult(parts[1], 1.2, [Word("good", 0.0, 0.5), Word("bye", 0.5, 1.2)], VoiceSpec(), "good bye"),
    ]

    words = common.narration_words(results, offsets, audio=joined)

    _assert_monotonic(words)
    assert [w.text for w in words] == ["hello", "there", "good", "bye"]
    assert words[2].start == pytest.approx(offsets[1], abs=1e-6)
    assert words[-1].end == pytest.approx(offsets[1] + 1.2, abs=1e-6)


def test_narration_words_is_idempotent_for_preshifted_results(narration):
    """``tts.synthesize_lines`` hands back words already on a shared clock."""
    parts, joined, offsets = narration
    results = [
        TTSResult(parts[0], 0.8, [Word("one", 0.0, 0.8)], VoiceSpec(), "one"),
        TTSResult(parts[1], 1.2, [Word("two", offsets[1], offsets[1] + 1.2)], VoiceSpec(), "two"),
    ]

    words = common.narration_words(results, offsets)

    _assert_monotonic(words)
    assert words[1].start == pytest.approx(offsets[1], abs=1e-6)
    assert words[1].end == pytest.approx(offsets[1] + 1.2, abs=1e-6)


def test_narration_words_falls_back_to_forced_alignment(narration, monkeypatch: pytest.MonkeyPatch):
    parts, joined, offsets = narration
    calls: list[tuple[Path, str]] = []

    def fake_align(audio, text, *, settings=None, language="en", cache=False):
        calls.append((Path(audio), text))
        tokens = text.split()
        step = 1.2 / len(tokens)
        return [Word(tok, i * step, (i + 1) * step) for i, tok in enumerate(tokens)]

    monkeypatch.setattr(transcribe_module, "available", lambda: True)
    monkeypatch.setattr(transcribe_module, "align", fake_align)

    results = [
        TTSResult(parts[0], 0.8, [Word("intro", 0.0, 0.8)], VoiceSpec(), "intro"),
        TTSResult(parts[1], 1.2, None, VoiceSpec(), "aligned words land here"),
    ]
    words = common.narration_words(results, offsets, audio=joined)

    assert calls and calls[0][0] == parts[1]
    _assert_monotonic(words)
    assert [w.text for w in words] == ["intro", "aligned", "words", "land", "here"]
    assert words[1].start == pytest.approx(offsets[1], abs=1e-6)


def test_narration_words_falls_back_to_proportional_timings(narration, monkeypatch: pytest.MonkeyPatch):
    parts, joined, offsets = narration
    monkeypatch.setattr(transcribe_module, "available", lambda: False)

    def boom(*args, **kwargs):  # pragma: no cover - must never be reached
        raise AssertionError("align must not be called when the backend is unavailable")

    monkeypatch.setattr(transcribe_module, "align", boom)

    text = "proportional timings keep the captions animating"
    results = [
        TTSResult(parts[0], 0.8, None, VoiceSpec(), "opening line"),
        TTSResult(parts[1], 1.2, None, VoiceSpec(), text),
    ]
    words = common.narration_words(results, offsets)

    _assert_monotonic(words)
    assert [w.text for w in words] == ["opening", "line", *text.split()]
    # The second part is laid across its *real* probed duration, not its request.
    real = common.probe_duration(parts[1])
    assert words[2].start == pytest.approx(offsets[1], abs=1e-6)
    assert words[-1].end == pytest.approx(offsets[1] + real, abs=1e-3)


def test_narration_words_never_empty_for_non_empty_text(narration, monkeypatch: pytest.MonkeyPatch):
    _parts, joined, _offsets = narration
    monkeypatch.setattr(transcribe_module, "available", lambda: False)

    words = common.narration_words([], [], audio=joined, text="nothing came back from the provider")

    _assert_monotonic(words)
    assert [w.text for w in words] == "nothing came back from the provider".split()
    assert words[-1].end == pytest.approx(common.probe_duration(joined), abs=1e-3)


def test_narration_words_empty_for_empty_text():
    assert common.narration_words([], []) == []


def test_proportional_words_covers_span_exactly():
    words = common.proportional_words("a much longer token here", 2.0, 4.0)
    _assert_monotonic(words)
    assert words[0].start == pytest.approx(2.0)
    assert words[-1].end == pytest.approx(6.0)
    assert common.words_span(words) == pytest.approx(6.0)


# --------------------------------------------------------------------------- #
# timeline pieces
# --------------------------------------------------------------------------- #

@pytest.fixture
def background_asset(tmp_path: Path, make_video) -> Asset:
    path = make_video("bg.mp4", seconds=6.0, width=120, height=200, fps=12, audio=False)
    return Asset(path=path, name="bg", kind="background", tags=["test"], duration=6.0)


def test_background_layer_offset_is_seeded_and_bounded(background_asset: Asset):
    a = common.background_layer(background_asset, width=180, height=320, duration=2.0, seed=7)
    b = common.background_layer(background_asset, width=180, height=320, duration=2.0, seed=7)
    c = common.background_layer(background_asset, width=180, height=320, duration=2.0, seed=99)

    assert a.src_start == b.src_start
    assert a.src_start != c.src_start
    for layer in (a, b, c):
        assert 0.0 <= layer.src_start <= background_asset.duration - 2.0
        assert layer.loop is False
        assert (layer.w, layer.h) == (180, 320)
        assert layer.fit == "cover"
        assert layer.take_audio is False


def test_background_layer_loops_when_asset_is_too_short(background_asset: Asset):
    layer = common.background_layer(background_asset, width=180, height=320, duration=30.0, seed=3)

    assert layer.src_start == 0.0
    assert layer.loop is True
    assert layer.end == pytest.approx(30.0)


def test_background_layer_handles_a_still(tmp_path: Path):
    still = tmp_path / "card.png"
    ff.run_ffmpeg(["-y", "-f", "lavfi", "-i", "color=c=blue:s=64x64", "-frames:v", "1", str(still)])
    asset = Asset(path=still, name="card", kind="background", duration=0.0)

    layer = common.background_layer(asset, width=180, height=320, duration=4.0)

    assert layer.kind == "image"
    assert layer.src_start == 0.0
    assert layer.loop is False


def test_music_and_voice_tracks(tmp_path: Path):
    bed = ff.make_tone(1.0, tmp_path / "bed.wav")
    asset = Asset(path=bed, name="bed", kind="music", duration=1.0)

    music = common.music_track(asset, duration=5.0)
    voice = common.voice_track(bed, start=0.5, gain_db=-1.0)

    assert music.role == "music" and music.duck is True and music.loop is True
    assert music.end == pytest.approx(5.0)
    assert music.gain_db == pytest.approx(-18.0)
    assert voice.role == "voice" and voice.start == pytest.approx(0.5) and voice.loop is False

    long_asset = Asset(path=bed, name="bed", kind="music", duration=40.0)
    assert common.music_track(long_asset, duration=5.0).loop is False
    # Fades never eat more than half of a very short timeline.
    tiny = common.music_track(asset, duration=1.0)
    assert tiny.fade_in <= 0.5 and tiny.fade_out <= 0.5


def test_caption_track_writes_ass_and_respects_the_off_switch(tmp_path: Path, sample_words):
    track = common.caption_track(sample_words, tmp_path / "subs", style="bold_yellow", width=180, height=320)

    assert track is not None
    body = track.ass_path.read_text(encoding="utf-8")
    assert "[Script Info]" in body and "Dialogue:" in body
    # The presets are authored in reference-canvas pixels, so the ASS declares the
    # reference resolution and lets libass scale it down to the real canvas.  A
    # literal "PlayResX: 180" would put a 96px font and a 380px margin on a
    # 320px-tall canvas, i.e. captions nobody can see.
    assert "PlayResX: 1080" in body and "PlayResY: 1920" in body

    assert common.caption_track(sample_words, tmp_path / "off", enabled=False) is None
    assert common.caption_track([], tmp_path / "empty") is None
    assert common.caption_track(sample_words, tmp_path / "nostyle", style=None) is None
    assert not (tmp_path / "off").exists()


def test_caption_resolution_normalises_the_canvas_to_reference_units():
    assert common.caption_resolution(1080, 1920) == (1080, 1920)
    # Small and large vertical canvases both normalise to the reference height.
    assert common.caption_resolution(180, 320) == (1080, 1920)
    assert common.caption_resolution(2160, 3840) == (1080, 1920)
    # The canvas aspect ratio is preserved exactly: a mismatch stretches glyphs.
    width, height = common.caption_resolution(1920, 1080)
    assert height == 1920
    assert width / height == pytest.approx(1920 / 1080, rel=1e-3)


@pytest.mark.parametrize("style", ["bold_yellow", "big_impact"])
def test_captions_are_actually_visible_on_a_small_canvas(tmp_path: Path, sample_words, style: str):
    """Burn captions at 180x320 and prove the text lands on the canvas.

    ``bold_yellow`` is bottom-anchored (its 380px margin is larger than the whole
    canvas) and ``big_impact`` is centred, so between them they cover both ways
    unscaled preset metrics used to fail.
    """
    from aiclipper import render

    timeline = Timeline(width=180, height=320, fps=12, duration=2.0, background="#101010")
    timeline.subtitles = common.caption_track(
        sample_words, tmp_path / "subs", style=style, width=180, height=320
    )
    assert timeline.subtitles is not None
    assert timeline.validate() == []

    out = render.render(timeline, tmp_path / f"burned-{style}.mp4",
                        options=RenderOptions(preset="ultrafast", crf=30)).path
    frame = tmp_path / f"frame-{style}.png"
    ff.run_ffmpeg(["-y", "-ss", "0.6", "-i", str(out), "-frames:v", "1", str(frame)])

    from PIL import Image

    pixels = list(Image.open(frame).convert("RGB").getdata())
    bright = sum(1 for r, g, b in pixels if r > 170 and g > 130)
    assert bright > 60, f"{style}: only {bright} lit pixels -- the captions are off canvas"


def test_helpers_compose_into_a_renderable_timeline(tmp_path: Path, background_asset: Asset, sample_words):
    from aiclipper import render

    bed = ff.make_tone(2.0, tmp_path / "bed.wav")
    voice = ff.make_silence(1.5, tmp_path / "voice.wav")
    music_asset = Asset(path=bed, name="bed", kind="music", duration=2.0)

    timeline = Timeline(width=180, height=320, fps=12, duration=1.5)
    timeline.add_visual(common.background_layer(background_asset, width=180, height=320, duration=1.5, seed=5))
    timeline.add_audio(common.voice_track(voice))
    timeline.add_audio(common.music_track(music_asset, duration=1.5))
    timeline.subtitles = common.caption_track(sample_words, tmp_path, width=180, height=320)

    assert timeline.validate() == []
    out = common.resolve_output(None, "Composed Timeline!")
    result = render.render(timeline, out, options=RenderOptions(preset="ultrafast", crf=32))

    assert result.path.exists() and result.path.stat().st_size > 0
    assert result.duration == pytest.approx(1.5, abs=0.3)


# --------------------------------------------------------------------------- #
# naming and output paths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Hello, World!", "hello-world"),
        ("  spaced   out  ", "spaced-out"),
        ("Café Münster", "cafe-munster"),
        ("MiXeD CaSe", "mixed-case"),
        ("under_score-and.dot", "under-score-and-dot"),
    ],
)
def test_safe_stem_simple_cases(text: str, expected: str):
    assert common.safe_stem(text) == expected


def test_safe_stem_edge_cases():
    assert common.safe_stem("") == "video"
    assert common.safe_stem("   ") == "video"
    assert common.safe_stem("", fallback="story") == "story"

    # Punctuation-only and non-Latin input still produce a usable, distinct name.
    punct = common.safe_stem("!!! ??? ...")
    other = common.safe_stem("***")
    assert punct.startswith("video-") and other.startswith("video-")
    assert punct != other

    japanese = common.safe_stem("日本語のタイトル")
    assert japanese and japanese != common.safe_stem("한국어 제목")

    long_stem = common.safe_stem("word " * 60)
    assert 0 < len(long_stem) <= 60
    assert long_stem == common.safe_stem("word " * 60)
    # Two long titles sharing a prefix must not collapse onto one filename.
    a = common.safe_stem("the same opening words repeated for a very long time indeed alpha")
    b = common.safe_stem("the same opening words repeated for a very long time indeed beta")
    assert len(a) <= 60 and len(b) <= 60 and a != b

    single_long = common.safe_stem("x" * 200)
    assert 0 < len(single_long) <= 60 and single_long.startswith("x")

    assert "/" not in common.safe_stem("a/b/c") and "\\" not in common.safe_stem("a\\b")


def test_resolve_output_defaults_to_the_output_dir(settings):
    path = common.resolve_output(None, "My Great Clip")

    assert path.parent == settings.output_dir
    assert path.name == "my-great-clip.mp4"
    assert path.parent.is_dir()


def test_resolve_output_disambiguates_instead_of_overwriting(settings):
    first = common.resolve_output(None, "collide")
    first.write_bytes(b"one")
    second = common.resolve_output(None, "collide")
    second.write_bytes(b"two")
    third = common.resolve_output(None, "collide")

    assert [p.name for p in (first, second, third)] == ["collide.mp4", "collide-2.mp4", "collide-3.mp4"]
    assert first.read_bytes() == b"one" and second.read_bytes() == b"two"


def test_resolve_output_honours_an_explicit_file(tmp_path: Path):
    target = tmp_path / "nested" / "exact.mp4"

    assert common.resolve_output(target, "ignored") == target
    assert target.parent.is_dir()


def test_resolve_output_treats_a_directory_as_a_destination(tmp_path: Path):
    directory = tmp_path / "shorts"
    directory.mkdir()
    (directory / "clip-one.mp4").write_bytes(b"x")

    path = common.resolve_output(directory, "Clip One")

    assert path.parent == directory
    assert path.name == "clip-one-2.mp4"


# --------------------------------------------------------------------------- #
# misc helpers and the package itself
# --------------------------------------------------------------------------- #

def test_canvas_size_forces_even_dimensions(settings):
    assert common.canvas_size(settings, 181, 321) == (180, 320)
    assert common.canvas_size(settings) == (settings.width, settings.height)


def test_seeded_random_is_reproducible(settings):
    a = common.seeded_random(settings, "asset", 3).random()
    b = common.seeded_random(settings, "asset", 3).random()
    c = common.seeded_random(settings, "other", 3).random()

    assert a == b and a != c


def test_resolve_voice_never_raises(settings):
    assert common.resolve_voice("narrator_deep").voice_id
    assert isinstance(common.resolve_voice("definitely not a catalogued voice"), VoiceSpec)
    assert isinstance(common.resolve_voice(""), VoiceSpec)
    passthrough = common.resolve_voice("edge:en-GB-RyanNeural")
    assert passthrough.voice_id == "en-GB-RyanNeural"


def test_workspace_is_named_safely(settings):
    path = common.workspace("A Messy: Title/With Slashes", settings)

    assert path.is_dir()
    assert path.parent == settings.work_dir
    assert path.name == "a-messy-title-with-slashes"


def test_pipelines_package_imports_submodules_lazily():
    code = (
        "import sys, aiclipper.pipelines as p;"
        "eager = [m for m in sys.modules if m.startswith('aiclipper.pipelines.')];"
        "assert not eager, eager;"
        "mod = p.common;"
        "assert 'aiclipper.pipelines.common' in sys.modules;"
        "assert mod is p.common;"
        "print('ok')"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)

    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "ok"


def test_pipelines_package_rejects_unknown_attributes():
    import aiclipper.pipelines as pipelines

    with pytest.raises(AttributeError):
        _ = pipelines.not_a_workflow
    assert set(common.__all__) <= set(dir(common))
    assert "story" in dir(pipelines)


def test_narration_words_tolerates_short_offsets(narration, monkeypatch: pytest.MonkeyPatch):
    """A caller with fewer offsets than results still gets a sane clock."""
    parts, _joined, _offsets = narration
    monkeypatch.setattr(transcribe_module, "available", lambda: False)
    results = [
        TTSResult(parts[0], 0.8, [Word("first", 0.0, 0.8)], VoiceSpec(), "first"),
        TTSResult(parts[1], 1.2, None, VoiceSpec(), "then this"),
    ]

    words = common.narration_words(results, [0.0])

    _assert_monotonic(words)
    assert [w.text for w in words] == ["first", "then", "this"]
    assert words[1].start == pytest.approx(0.8, abs=1e-6)


# --------------------------------------------------------------------------- #
# the helpers every pipeline shares
# --------------------------------------------------------------------------- #

def test_caption_style_resolves_a_preset_once_for_every_pipeline():
    resolved = common.caption_style("bold_yellow")

    assert isinstance(resolved, CaptionStyle)
    assert resolved.name == "bold_yellow"
    assert common.caption_style("Bold-Yellow").name == "bold_yellow"


def test_caption_style_is_none_when_captions_are_off():
    assert common.caption_style("bold_yellow", enabled=False) is None
    assert common.caption_style(None) is None


def test_caption_style_rejects_an_unknown_preset():
    with pytest.raises(ValueError, match="unknown caption style"):
        common.caption_style("not-a-real-preset")


def test_style_name_reports_the_preset_behind_either_form():
    assert common.style_name(None) == ""
    assert common.style_name("neon") == "neon"
    assert common.style_name(common.caption_style("neon")) == "neon"


def test_provider_name_survives_anything_a_pipeline_can_hold():
    class Named:
        name = "heuristic"

    class Anonymous:
        name = ""

    assert common.provider_name(None) == ""
    assert common.provider_name(Named()) == "heuristic"
    assert common.provider_name(Anonymous()) == "Anonymous"


def test_llm_provider_resolves_names_and_passes_objects_through(settings):
    resolved = common.llm_provider(None, settings)
    assert resolved.name == "heuristic"  # offline settings pin the heuristic backend
    assert common.llm_provider("heuristic", settings).name == "heuristic"

    sentinel = object()
    assert common.llm_provider(sentinel, settings) is sentinel


def test_every_pipeline_uses_the_shared_helpers_rather_than_its_own():
    """No pipeline may carry a private copy of a helper that lives in common."""
    import aiclipper.pipelines as pipelines_pkg

    root = Path(pipelines_pkg.__file__).parent
    duplicated = ("_caption_style", "_style_name", "_llm_provider")
    for module in ("clip", "story", "texts", "reddit", "split"):
        source = (root / f"{module}.py").read_text(encoding="utf-8")
        for name in duplicated:
            assert f"def {name}(" not in source, f"{module}.py redefines {name}"
