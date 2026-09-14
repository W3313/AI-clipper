"""Tests for :mod:`aiclipper.pipelines.clip`.

Speech recognition cannot run here (no model to download), so
``aiclipper.transcribe.transcribe`` is monkeypatched to hand back a synthetic
word-level :class:`~aiclipper.models.Transcript`.  Everything else is real: the
sources are built with ffmpeg, the crop paths come out of the real tracker, the
ASS files are really written, and one test encodes two shorts end to end and
probes what landed on disk.

Structural tests stub :func:`aiclipper.render.render` so they can assert on the
timeline the pipeline built (and still validate it, and still build the real
ffmpeg command) without paying for an encode.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import pytest

from aiclipper import crop as crop_module
from aiclipper import ffmpeg as ff
from aiclipper import render as render_module
from aiclipper import transcribe as transcribe_module
from aiclipper.config import get_settings, reset_settings
from aiclipper.errors import IngestError, MissingDependency
from aiclipper.models import RenderResult, Transcript, Word
from aiclipper.pipelines import clip

pytestmark = pytest.mark.needs_ffmpeg

CANVAS_W = 180
CANVAS_H = 320


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _tiny_canvas(monkeypatch: pytest.MonkeyPatch):
    """Every render in this file targets a 180x320 canvas at 12 fps."""
    monkeypatch.setenv("AICLIP_WIDTH", str(CANVAS_W))
    monkeypatch.setenv("AICLIP_HEIGHT", str(CANVAS_H))
    monkeypatch.setenv("AICLIP_FPS", "12")
    reset_settings()
    yield
    reset_settings()


def _speech(duration: float, *, per_word: float = 0.3) -> Transcript:
    """A synthetic transcript of hook-laden sentences filling ``duration``."""
    sentences = [
        "here is the part nobody tells you about shipping something people want.",
        "what happened next completely changed how we thought about the problem.",
        "revenue moved from 12 thousand to 240 thousand in 90 days.",
        "why does this keep working when every reasonable model says it should not?",
    ]
    words: list[Word] = []
    cursor = 0.0
    index = 0
    while cursor < duration - per_word:
        for token in sentences[index % len(sentences)].split():
            if cursor >= duration - per_word:
                break
            words.append(Word(token, round(cursor, 3), round(cursor + per_word, 3)))
            cursor += per_word
        cursor += 0.2
        index += 1
    transcript = Transcript.from_words(words, language="en")
    transcript.duration = duration
    return transcript


@pytest.fixture
def fake_asr(monkeypatch: pytest.MonkeyPatch) -> Callable[[Transcript | Exception], list]:
    """Install a synthetic transcript (or failure) for ``transcribe.transcribe``.

    Returns the list of media paths it was asked for, so a test can prove the
    source is transcribed once and not once per candidate.
    """

    def _install(result: Transcript | Exception) -> list:
        seen: list = []

        def _fake(media, **kwargs):  # noqa: ANN001, ANN003 - mirrors the real signature loosely
            seen.append(Path(media))
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(transcribe_module, "transcribe", _fake)
        return seen

    return _install


@pytest.fixture
def stub_render(monkeypatch: pytest.MonkeyPatch) -> list:
    """Record every timeline instead of encoding it -- but still prove it builds."""
    calls: list = []

    def _fake(timeline, out_path, *, options=None, settings=None, log_path=None, dry_run=False):
        problems = timeline.validate()
        assert problems == [], problems
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        # A real ffmpeg argv, so a broken graph still fails the test.
        command = render_module.build_command(
            timeline, out, settings=settings, workdir=out.parent / ".graph"
        )
        assert command[0]
        out.write_bytes(b"stub")
        calls.append((timeline, out, command))
        return RenderResult(
            path=out,
            duration=float(timeline.duration),
            width=timeline.width,
            height=timeline.height,
            fps=float(timeline.fps),
            command=command,
            size_bytes=out.stat().st_size,
        )

    monkeypatch.setattr(render_module, "render", _fake)
    return calls


@pytest.fixture
def no_tracking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ``crop.track`` an error: proves a code path never reframes."""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003
        raise AssertionError("crop.track must not run on this path")

    monkeypatch.setattr(crop_module, "track", _boom)


def _mean_volume_db(path: Path) -> float:
    """Mean audio level of ``path``; ``-inf`` for a genuinely silent track."""
    proc = ff.run_ffmpeg(
        ["-i", str(path), "-map", "0:a:0", "-af", "volumedetect", "-f", "null", "-"], quiet=False
    )
    match = re.search(r"mean_volume:\s*(-?\d+(?:\.\d+)?|-inf)\s*dB", proc.stderr)
    assert match, proc.stderr
    return float("-inf") if match.group(1) == "-inf" else float(match.group(1))


# --------------------------------------------------------------------------- #
# even_windows (pure, no ffmpeg needed)
# --------------------------------------------------------------------------- #

def test_even_windows_are_spaced_and_never_overlap():
    windows = clip.even_windows(60.0, count=3, min_duration=10.0, max_duration=15.0)

    assert len(windows) == 3
    assert [round(w.duration, 3) for w in windows] == [15.0, 15.0, 15.0]
    assert [w.start for w in windows] == [0.0, 20.0, 40.0]
    for first, second in zip(windows, windows[1:], strict=False):
        assert first.end <= second.start
    assert all(w.title for w in windows)


def test_even_windows_shrink_to_what_the_source_holds():
    # 25s cannot host three 10s windows; it hosts two.
    assert len(clip.even_windows(25.0, count=3, min_duration=10.0, max_duration=15.0)) == 2
    # Shorter than min_duration: one window covering everything rather than none.
    short = clip.even_windows(3.0, count=3, min_duration=10.0, max_duration=15.0)
    assert len(short) == 1
    assert (short[0].start, short[0].end) == (0.0, 3.0)
    assert clip.even_windows(0.0, count=3) == []
    assert clip.even_windows(60.0, count=0) == []


# --------------------------------------------------------------------------- #
# the real thing
# --------------------------------------------------------------------------- #

def test_real_render_end_to_end(make_video, fake_asr, tmp_path: Path):
    source = make_video("talk.mp4", seconds=12.0, width=640, height=360, fps=12)
    fake_asr(_speech(12.0))
    out_dir = tmp_path / "shorts"

    results = clip.run(
        source,
        count=2,
        min_duration=2.0,
        max_duration=4.0,
        style="bold_yellow",
        out_dir=out_dir,
    )

    assert len(results) == 2
    for rank, result in enumerate(results, start=1):
        assert result.output.exists() and result.output.stat().st_size > 2000
        assert result.output.parent == out_dir
        assert result.output.name.startswith(f"talk-{rank:02d}-")
        assert result.output.suffix == ".mp4"

        info = ff.probe(result.output)
        assert (info.width, info.height) == (CANVAS_W, CANVAS_H)
        assert info.has_video and info.has_audio
        window = result.metadata["window"]
        assert info.duration == pytest.approx(window[1] - window[0], abs=0.35)
        assert 2.0 - 0.05 <= window[1] - window[0] <= 4.0 + 0.05

        # The source's own audio really made it through, not an anullsrc bed.
        assert _mean_volume_db(result.output) > -50.0

        assert result.kind == "clip"
        assert result.metadata["rank"] == rank
        assert result.metadata["reframed"] is True
        assert result.metadata["captions"] is True
        assert result.metadata["style"] == "bold_yellow"
        assert result.metadata["words"] > 0
        assert result.metadata["source"] == str(source)
        assert result.transcript is not None and result.transcript.words
        # transcript.slice rebased the window onto the clip's own clock.
        assert result.transcript.words[0].start < 1.0

    # Ranked best-first, non-overlapping, and each knows about the other.
    scores = [r.metadata["score"] for r in results]
    assert scores == sorted(scores, reverse=True)
    first, second = (r.metadata["window"] for r in results)
    assert first[1] <= second[0] or second[1] <= first[0]
    assert results[0].siblings == [results[1].output]
    assert results[1].siblings == [results[0].output]


# --------------------------------------------------------------------------- #
# timeline shape
# --------------------------------------------------------------------------- #

def test_timeline_carries_source_audio_and_the_candidate_window(
    make_video, fake_asr, stub_render
):
    source = make_video("talk.mp4", seconds=10.0, width=640, height=360, fps=12)
    fake_asr(_speech(10.0))

    results = clip.run(source, count=1, min_duration=2.0, max_duration=3.0)

    assert len(results) == 1
    timeline, out, command = stub_render[0]
    candidate_start = results[0].metadata["source_start"]

    assert len(timeline.visuals) == 1
    layer = timeline.visuals[0]
    assert layer.kind == "video"
    assert layer.src == str(source)
    assert layer.take_audio is True
    assert layer.fit == "cover"
    assert layer.src_start == pytest.approx(candidate_start)
    assert (layer.w, layer.h) == (CANVAS_W, CANVAS_H)
    assert (timeline.width, timeline.height) == (CANVAS_W, CANVAS_H)
    assert timeline.duration == pytest.approx(results[0].metadata["clip_duration"])
    assert timeline.subtitles is not None
    assert Path(timeline.subtitles.ass_path).exists()
    assert "[Events]" in Path(timeline.subtitles.ass_path).read_text(encoding="utf-8")
    # take_audio means the graph really maps the source's audio stream.
    assert any(arg.startswith("crop=") or "crop@" in arg for arg in command)
    assert out.exists()


def test_captions_off_leaves_no_subtitle_track(make_video, fake_asr, stub_render):
    source = make_video("talk.mp4", seconds=8.0, width=640, height=360, fps=12)
    fake_asr(_speech(8.0))

    results = clip.run(source, count=1, min_duration=2.0, max_duration=3.0, captions=False)

    timeline, _out, _cmd = stub_render[0]
    assert timeline.subtitles is None
    assert results[0].metadata["captions"] is False
    assert results[0].metadata["style"] is None


# --------------------------------------------------------------------------- #
# reframing decisions
# --------------------------------------------------------------------------- #

def test_vertical_source_skips_reframing(make_video, fake_asr, stub_render, no_tracking):
    source = make_video("portrait.mp4", seconds=8.0, width=180, height=320, fps=12)
    fake_asr(_speech(8.0))

    results = clip.run(source, count=1, min_duration=2.0, max_duration=3.0)

    assert results, "a vertical source should still produce a clip"
    assert results[0].metadata["crop"] == "vertical"
    assert results[0].metadata["reframed"] is False
    timeline, _out, _cmd = stub_render[0]
    assert timeline.visuals[0].crop is not None
    assert timeline.visuals[0].crop.is_static


def test_reframe_false_takes_the_centre_crop(make_video, fake_asr, stub_render, no_tracking):
    source = make_video("wide.mp4", seconds=8.0, width=640, height=360, fps=12)
    fake_asr(_speech(8.0))

    results = clip.run(source, count=1, min_duration=2.0, max_duration=3.0, reframe=False)

    assert results[0].metadata["crop"] == "center"
    assert results[0].metadata["reframed"] is False
    timeline, _out, _cmd = stub_render[0]
    path = timeline.visuals[0].crop
    expected = crop_module.center_crop(ff.probe(source), CANVAS_W / CANVAS_H)
    assert path is not None and path.is_static
    assert (path.keyframes[0].x, path.keyframes[0].y) == (expected.keyframes[0].x, expected.keyframes[0].y)
    assert path.size == expected.size


def test_reframe_true_on_a_wide_source_tracks(make_video, fake_asr, stub_render):
    source = make_video("wide.mp4", seconds=8.0, width=640, height=360, fps=12)
    fake_asr(_speech(8.0))

    results = clip.run(source, count=1, min_duration=2.0, max_duration=3.0)

    assert results[0].metadata["crop"].startswith("track")
    assert results[0].metadata["reframed"] is True
    assert results[0].metadata["crop_keyframes"] >= 1
    path = stub_render[0][0].visuals[0].crop
    assert path is not None
    # Keyframes stay in *source* time; the renderer rebases them by src_start.
    assert path.keyframes[0].t >= results[0].metadata["source_start"] - 1e-6


# --------------------------------------------------------------------------- #
# degrading gracefully
# --------------------------------------------------------------------------- #

def test_the_source_is_transcribed_once_not_once_per_candidate(make_video, fake_asr, stub_render):
    source = make_video("talk.mp4", seconds=12.0, width=640, height=360, fps=12)
    seen = fake_asr(_speech(12.0))

    results = clip.run(source, count=3, min_duration=2.0, max_duration=3.0)

    assert len(results) == 3
    assert seen == [Path(source)]


def test_source_with_no_speech_still_produces_clips(make_video, fake_asr, stub_render):
    source = make_video("silent-ish.mp4", seconds=12.0, width=640, height=360, fps=12)
    fake_asr(Transcript(segments=[], language="en", duration=12.0))

    results = clip.run(source, count=3, min_duration=2.0, max_duration=3.0)

    assert len(results) == 3
    assert all(r.metadata["selection"] == "even" for r in results)
    assert all(r.metadata["captions"] is False for r in results)  # nothing to caption
    windows = [r.metadata["window"] for r in results]
    assert windows[0][0] == 0.0
    for first, second in zip(windows, windows[1:], strict=False):
        assert first[1] <= second[0] + 1e-6
    assert all(2.0 - 0.05 <= w[1] - w[0] <= 3.0 + 0.05 for w in windows)


def test_missing_asr_backend_degrades_to_even_windows(make_video, fake_asr, stub_render):
    source = make_video("talk.mp4", seconds=10.0, width=640, height=360, fps=12)
    fake_asr(MissingDependency("faster-whisper", purpose="speech recognition"))

    results = clip.run(source, count=2, min_duration=2.0, max_duration=3.0)

    assert len(results) == 2
    assert all(r.metadata["transcription"] == "unavailable" for r in results)
    assert all(r.metadata["selection"] == "even" for r in results)


def test_more_clips_requested_than_the_source_supports(make_video, fake_asr, stub_render):
    source = make_video("short.mp4", seconds=6.0, width=640, height=360, fps=12)
    fake_asr(_speech(6.0))

    results = clip.run(source, count=5, min_duration=4.0, max_duration=5.0)

    assert len(results) == 1
    assert results[0].metadata["requested"] == 5
    assert results[0].metadata["produced"] == 1
    assert results[0].siblings == []


def test_source_shorter_than_min_duration_still_yields_one_clip(make_video, fake_asr, stub_render):
    source = make_video("tiny.mp4", seconds=3.0, width=640, height=360, fps=12)
    fake_asr(Transcript(segments=[], language="en", duration=3.0))

    results = clip.run(source, count=1, min_duration=10.0, max_duration=20.0)

    assert len(results) == 1
    window = results[0].metadata["window"]
    assert window[0] == 0.0
    assert window[1] == pytest.approx(3.0, abs=0.15)
    assert results[0].metadata["selection"] == "even"


def test_count_zero_returns_nothing(make_video, fake_asr, stub_render):
    source = make_video("talk.mp4", seconds=6.0, width=640, height=360, fps=12)
    fake_asr(_speech(6.0))

    assert clip.run(source, count=0) == []
    assert stub_render == []


def test_audio_only_source_is_rejected(make_audio, fake_asr, stub_render):
    source = make_audio("voice.wav", seconds=3.0)
    fake_asr(_speech(3.0))

    with pytest.raises(IngestError):
        clip.run(source, count=1, min_duration=1.0, max_duration=2.0)


# --------------------------------------------------------------------------- #
# naming and output placement
# --------------------------------------------------------------------------- #

def test_output_names_are_ranked_slugs_of_the_title(make_video, fake_asr, stub_render, tmp_path: Path):
    source = make_video("My Big Talk.mp4", seconds=10.0, width=640, height=360, fps=12)
    fake_asr(_speech(10.0))
    out_dir = tmp_path / "renders"

    results = clip.run(source, count=2, min_duration=2.0, max_duration=3.0, out_dir=out_dir)

    assert len(results) == 2
    for rank, result in enumerate(results, start=1):
        name = result.output.name
        assert result.output.parent == out_dir
        assert re.fullmatch(rf"my-big-talk-{rank:02d}-[a-z0-9-]+\.mp4", name), name


def test_no_out_dir_lands_in_the_settings_output_dir(make_video, fake_asr, stub_render):
    source = make_video("talk.mp4", seconds=8.0, width=640, height=360, fps=12)
    fake_asr(_speech(8.0))

    results = clip.run(source, count=1, min_duration=2.0, max_duration=3.0)

    assert results[0].output.parent == Path(get_settings().output_dir)


def test_an_out_dir_that_looks_like_a_file_still_gives_every_clip_its_own_name(
    make_video, fake_asr, stub_render, tmp_path: Path
):
    """``--out shorts.mp4`` must not make every short overwrite the one before it."""
    source = make_video("talk.mp4", seconds=10.0, width=640, height=360, fps=12)
    fake_asr(_speech(10.0))
    looks_like_a_file = tmp_path / "shorts.mp4"

    results = clip.run(source, count=2, min_duration=2.0, max_duration=3.0, out_dir=looks_like_a_file)

    assert len(results) == 2
    assert len({r.output for r in results}) == 2
    for result in results:
        assert result.output.parent == looks_like_a_file
        assert result.output.name.endswith(".mp4")
    assert results[0].siblings == [results[1].output]


def test_an_unknown_caption_style_fails_before_the_source_is_touched(
    make_video, fake_asr, stub_render, monkeypatch: pytest.MonkeyPatch
):
    """The preset is resolved first, so a typo costs nothing but the error."""
    source = make_video("talk.mp4", seconds=8.0, width=640, height=360, fps=12)
    fake_asr(_speech(8.0))

    def _no_ingest(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("ingest.resolve was reached despite an unknown style")

    from aiclipper import ingest as ingest_module

    monkeypatch.setattr(ingest_module, "resolve", _no_ingest)
    with pytest.raises(ValueError, match="unknown caption style"):
        clip.run(source, count=1, style="not-a-real-preset")
    assert stub_render == []


def test_captions_off_never_validates_the_style(make_video, fake_asr, stub_render):
    """With no captions to burn there is no preset to reject."""
    source = make_video("talk.mp4", seconds=8.0, width=640, height=360, fps=12)
    fake_asr(_speech(8.0))

    results = clip.run(
        source, count=1, min_duration=2.0, max_duration=3.0, style="not-a-real-preset", captions=False
    )

    assert results and results[0].metadata["style"] is None


# --------------------------------------------------------------------------- #
# the clip's clock: captions must not outlive the video, words must be whole
# --------------------------------------------------------------------------- #

def _offset_speech(duration: float, *, per_word: float = 0.4, stride: float = 0.45) -> Transcript:
    """Words on a grid that deliberately straddles round-number boundaries.

    ``stride`` is wider than ``per_word``, so word *n* covers
    ``[0.45n, 0.45n + 0.4)`` -- every whole second lands inside a word rather
    than in the gap between two.  That is what an evenly spaced fallback window
    (or a window clamped to the probed media duration) cuts through.
    """
    tokens = "here is the part nobody tells you about shipping something people want".split()
    words: list[Word] = []
    index = 0
    while index * stride + per_word <= duration:
        words.append(
            Word(tokens[index % len(tokens)], round(index * stride, 3), round(index * stride + per_word, 3))
        )
        index += 1
    transcript = Transcript.from_words(words, language="en")
    transcript.duration = duration
    return transcript


def _dialogue_spans(ass_path: Path) -> list[tuple[float, float]]:
    """Every ``Dialogue:`` line of an ASS file as ``(start, end)`` seconds."""

    def _seconds(stamp: str) -> float:
        hours, minutes, rest = stamp.strip().split(":")
        return int(hours) * 3600 + int(minutes) * 60 + float(rest)

    spans: list[tuple[float, float]] = []
    for line in Path(ass_path).read_text(encoding="utf-8").splitlines():
        if line.startswith("Dialogue:"):
            fields = line.split(",")
            spans.append((_seconds(fields[1]), _seconds(fields[2])))
    assert spans, f"{ass_path} carries no dialogue lines"
    return spans


@pytest.mark.parametrize(
    ("seconds", "count", "low", "high"),
    [(10.0, 1, 2.0, 3.0), (12.0, 2, 2.5, 4.0), (12.0, 3, 3.0, 5.0)],
)
def test_no_caption_cue_outlives_the_clip(
    make_video, fake_asr, stub_render, seconds: float, count: int, low: float, high: float
):
    """A cue that ends after the last frame is a caption nobody ever sees."""
    source = make_video("talk.mp4", seconds=seconds, width=640, height=360, fps=12)
    fake_asr(_offset_speech(seconds))

    results = clip.run(source, count=count, min_duration=low, max_duration=high)

    assert results
    for (timeline, _out, _cmd), result in zip(stub_render, results, strict=True):
        assert timeline.subtitles is not None
        spans = _dialogue_spans(timeline.subtitles.ass_path)
        duration = float(timeline.duration)
        assert duration == pytest.approx(result.metadata["clip_duration"])
        assert max(end for _start, end in spans) <= duration + 1e-6
        assert all(start < duration for start, _end in spans)
        assert min(start for start, _end in spans) >= -1e-6


def test_the_clip_ends_on_a_whole_word(make_video, fake_asr, stub_render):
    """No word may straddle either edge of the window the clip is cut from."""
    source = make_video("talk.mp4", seconds=12.0, width=640, height=360, fps=12)
    transcript = _offset_speech(12.0)
    fake_asr(transcript)

    results = clip.run(source, count=3, min_duration=2.0, max_duration=3.0)

    assert results
    for result in results:
        start, end = result.metadata["window"]
        straddling = [
            w
            for w in transcript.words
            if (w.start < start - 1e-6 < w.end) or (w.start < end - 1e-6 < w.end - 1e-6)
        ]
        assert straddling == [], f"window {start}-{end} cuts {straddling}"
        # The rebased words really are whole: same length as in the source.
        assert result.transcript is not None and result.transcript.words
        for rebased, source_word in zip(
            result.transcript.words,
            [w for w in transcript.words if w.start >= start - 1e-6 and w.end <= end + 1e-6],
            strict=True,
        ):
            assert rebased.text == source_word.text
            assert rebased.end - rebased.start == pytest.approx(source_word.end - source_word.start)
        assert result.transcript.words[-1].end <= result.metadata["clip_duration"] + 1e-6


def test_even_window_fallback_snaps_to_word_edges(make_video, fake_asr, stub_render, monkeypatch):
    """Evenly spaced windows know nothing about speech -- the pipeline must."""
    from aiclipper import highlight as highlight_module

    monkeypatch.setattr(highlight_module, "select", lambda *a, **k: [])
    source = make_video("talk.mp4", seconds=12.0, width=640, height=360, fps=12)
    transcript = _offset_speech(12.0)
    fake_asr(transcript)

    results = clip.run(source, count=2, min_duration=4.0, max_duration=5.0)

    assert results and all(r.metadata["selection"] == "even" for r in results)
    for result in results:
        start, end = result.metadata["window"]
        assert any(w.start == pytest.approx(start) for w in transcript.words) or start == 0.0
        assert any(w.end == pytest.approx(end) for w in transcript.words)
        assert result.transcript is not None
        last = result.transcript.words[-1]
        assert last.end <= result.metadata["clip_duration"] + 1e-6
        assert last.end - last.start == pytest.approx(0.4)


def test_a_window_clamped_to_the_media_drops_the_half_word(make_video, fake_asr, stub_render):
    """ASR that runs past the probed duration must not leave a cut word behind."""
    # The media is 6.0s; the transcript's last word runs to 6.25s.
    source = make_video("talk.mp4", seconds=6.0, width=640, height=360, fps=12)
    transcript = _offset_speech(6.4)
    assert transcript.words[-1].end > 6.0
    fake_asr(transcript)

    results = clip.run(source, count=1, min_duration=5.0, max_duration=8.0)

    assert results
    start, end = results[0].metadata["window"]
    assert end <= 6.0 + 1e-6
    assert any(w.end == pytest.approx(end) for w in transcript.words)
    assert results[0].transcript is not None
    assert results[0].transcript.words[-1].end - results[0].transcript.words[-1].start == pytest.approx(0.4)


def test_audio_and_caption_clocks_agree_at_the_boundary(make_video, fake_asr, tmp_path: Path):
    """Render for real: the encoded clip is as long as its caption track claims."""
    source = make_video("talk.mp4", seconds=12.0, width=640, height=360, fps=12)
    fake_asr(_offset_speech(12.0))

    results = clip.run(
        source, count=1, min_duration=3.0, max_duration=4.0, out_dir=tmp_path / "shorts"
    )

    assert len(results) == 1
    result = results[0]
    info = ff.probe(result.output)
    window = result.metadata["window"]
    clip_duration = window[1] - window[0]
    assert info.duration == pytest.approx(clip_duration, abs=0.2)

    burned = [
        path
        for path in Path(get_settings().work_dir).glob("**/captions/*.ass")
        if path.stem.startswith("talk-01-")
    ]
    assert len(burned) == 1, burned
    spans = _dialogue_spans(burned[0])
    assert max(end for _s, end in spans) <= clip_duration + 1e-6
    # The final cue reaches the end of the clip: no dead tail, no overrun.
    assert max(end for _s, end in spans) >= clip_duration - 0.35


# --------------------------------------------------------------------------- #
# whole-word snapping must never under-deliver min_duration
# --------------------------------------------------------------------------- #

def _grid(spans: list[tuple[float, float]], duration: float) -> Transcript:
    """A transcript whose words sit exactly on ``spans``."""
    tokens = "here is the part nobody tells you about shipping something people want".split()
    words = [
        Word(tokens[index % len(tokens)], round(start, 3), round(end, 3))
        for index, (start, end) in enumerate(spans)
    ]
    transcript = Transcript.from_words(words, language="en")
    transcript.duration = duration
    return transcript


def _run_of(spans: list[tuple[float, float]], *, step: float, first: float, last: float):
    """``first`` .. ``last`` filled with ``step``-long words, appended to ``spans``."""
    cursor = first
    while cursor + step <= last + 1e-9:
        spans.append((cursor, cursor + step))
        cursor += step
    return spans


def _long_tail() -> Transcript:
    """Short words, then one long trailing word straddling the requested end."""
    spans = _run_of([], step=0.4, first=0.0, last=14.0)
    spans.append((14.0, 16.5))
    return _grid(spans, 20.0)


def _long_head() -> Transcript:
    """One long opening word straddling the requested start."""
    spans: list[tuple[float, float]] = [(2.0, 6.5)]
    _run_of(spans, step=0.4, first=6.5, last=24.0)
    return _grid(spans, 25.0)


def _long_both_edges() -> Transcript:
    """A long word at each edge of the requested window."""
    spans: list[tuple[float, float]] = [(1.0, 5.4)]
    _run_of(spans, step=0.4, first=5.4, last=18.0)
    spans.append((18.0, 20.4))
    return _grid(spans, 24.0)


@pytest.mark.parametrize(
    ("build", "seconds", "window", "low", "high"),
    [
        (_long_tail, 20.0, (0.0, 15.0), 15.0, 20.0),
        (_long_head, 25.0, (5.0, 20.0), 15.0, 20.0),
        (_long_both_edges, 24.0, (5.0, 20.0), 15.0, 22.0),
    ],
)
def test_whole_word_snapping_never_returns_less_than_min_duration(
    make_video, fake_asr, stub_render, monkeypatch, build, seconds, window, low, high
):
    """Honouring whole words must not hand back a third of the requested length."""
    from aiclipper import highlight as highlight_module
    from aiclipper.models import ClipCandidate

    transcript = build()
    source = make_video("talk.mp4", seconds=seconds, width=320, height=240, fps=12)
    fake_asr(transcript)
    monkeypatch.setattr(
        highlight_module,
        "select",
        lambda *a, **k: [
            ClipCandidate(start=window[0], end=window[1], title="Moment", hook="", reason="stub",
                          score=1.0, tags=[])
        ],
    )

    results = clip.run(source, count=1, min_duration=low, max_duration=high, reframe=False)

    assert results, "a window that cannot honour the minimum must not silently shrink"
    for result in results:
        start, end = result.metadata["window"]
        assert end - start >= low - 1e-6, f"{end - start:.3f}s delivered for a {low:.1f}s request"
        assert end - start <= high + 1e-6
        assert end <= seconds + 1e-6
        straddling = [w for w in transcript.words if w.start < start - 1e-6 < w.end]
        straddling += [w for w in transcript.words if w.start < end - 1e-6 < w.end - 1e-6]
        assert straddling == [], f"window {start}-{end} cuts {straddling}"


def test_a_window_that_cannot_meet_the_minimum_is_dropped(
    make_video, fake_asr, stub_render, monkeypatch
):
    """Better no clip at all than a 2s clip billed as a 15s one."""
    from aiclipper import highlight as highlight_module
    from aiclipper.models import ClipCandidate

    # Two words: a short one, then one that runs to the end of the media.  No
    # whole-word window of >= 15s fits inside a 16s ceiling.
    transcript = _grid([(0.0, 2.0), (2.0, 20.0)], 20.0)
    source = make_video("talk.mp4", seconds=20.0, width=320, height=240, fps=12)
    fake_asr(transcript)
    monkeypatch.setattr(
        highlight_module,
        "select",
        lambda *a, **k: [
            ClipCandidate(start=0.0, end=15.0, title="Moment", hook="", reason="stub",
                          score=1.0, tags=[])
        ],
    )

    results = clip.run(source, count=1, min_duration=15.0, max_duration=16.0, reframe=False)

    assert results == []
    assert stub_render == []


def test_a_source_shorter_than_the_minimum_still_yields_a_clip(make_video, fake_asr, stub_render):
    """The floor defends the caller's minimum; it does not veto a short source."""
    source = make_video("talk.mp4", seconds=6.0, width=320, height=240, fps=12)
    transcript = _offset_speech(6.4)
    fake_asr(transcript)

    results = clip.run(source, count=1, min_duration=15.0, max_duration=60.0, reframe=False)

    assert len(results) == 1
    start, end = results[0].metadata["window"]
    assert end <= 6.0 + 1e-6
    assert any(w.end == pytest.approx(end) for w in transcript.words)
