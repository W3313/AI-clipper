"""Shared pytest fixtures.

Everything here is generated on the fly with ffmpeg's lavfi sources, so the test
suite carries no binary media and runs fully offline.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from aiclipper import ffmpeg as ff
from aiclipper.config import get_settings, reset_settings
from aiclipper.models import Segment, Transcript, Word


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Point every settings-derived directory at a per-test tmp dir."""
    monkeypatch.setenv("AICLIP_WORK_DIR", str(tmp_path / "work"))
    monkeypatch.setenv("AICLIP_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    reset_settings()
    yield
    reset_settings()


@pytest.fixture
def settings(tmp_path: Path):
    return get_settings().ensure_dirs()


@pytest.fixture(scope="session")
def has_ffmpeg() -> bool:
    return ff.have_ffmpeg()


@pytest.fixture(autouse=True)
def _skip_without_ffmpeg(request, has_ffmpeg: bool):
    if request.node.get_closest_marker("needs_ffmpeg") and not has_ffmpeg:
        pytest.skip("ffmpeg is not installed")


@pytest.fixture
def make_video(tmp_path: Path) -> Callable[..., Path]:
    """Factory: build a small test clip. ``make_video(name, seconds=2, ...)``."""

    def _make(
        name: str = "clip.mp4",
        *,
        seconds: float = 2.0,
        width: int = 320,
        height: int = 240,
        fps: int = 24,
        source: str | None = None,
        audio: bool = True,
    ) -> Path:
        out = tmp_path / name
        src = source or f"testsrc2=size={width}x{height}:rate={fps}"
        args = ["-y", "-f", "lavfi", "-i", f"{src}:duration={seconds}"]
        if audio:
            args += ["-f", "lavfi", "-i", f"sine=frequency=330:duration={seconds}"]
        args += ["-t", f"{seconds}", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
        if audio:
            args += ["-c:a", "aac", "-shortest"]
        args += [str(out)]
        ff.run_ffmpeg(args)
        return out

    return _make


@pytest.fixture
def make_audio(tmp_path: Path) -> Callable[..., Path]:
    """Factory: build a short tone. ``make_audio("bed.wav", seconds=3)``."""

    def _make(name: str = "tone.wav", *, seconds: float = 2.0, frequency: float = 220.0) -> Path:
        return ff.make_tone(seconds, tmp_path / name, frequency=frequency)

    return _make


@pytest.fixture
def sample_words() -> list[Word]:
    """A tidy word sequence with a deliberate pause before 'Here'."""
    spoken = [
        ("This", 0.00, 0.28), ("is", 0.28, 0.44), ("the", 0.44, 0.58), ("part", 0.58, 0.94),
        ("nobody", 0.94, 1.40), ("tells", 1.40, 1.72), ("you", 1.72, 1.94),
        ("Here", 3.10, 3.44), ("is", 3.44, 3.60), ("why", 3.60, 3.98),
        ("it", 3.98, 4.14), ("actually", 4.14, 4.70), ("works", 4.70, 5.10),
    ]
    return [Word(text, start, end) for text, start, end in spoken]


@pytest.fixture
def sample_transcript(sample_words: list[Word]) -> Transcript:
    return Transcript.from_words(sample_words, language="en")


@pytest.fixture
def long_transcript() -> Transcript:
    """A 90s transcript: a flat stretch, a hook-laden stretch, then numbers."""
    segments: list[Segment] = []
    t = 0.0

    def push(sentence: str, per_word: float = 0.32) -> None:
        nonlocal t
        words: list[Word] = []
        for token in sentence.split():
            words.append(Word(token, t, t + per_word))
            t += per_word
        segments.append(Segment(sentence, words[0].start, words[-1].end, words))
        t += 0.25

    for _ in range(6):
        push("we kept the same process running quietly in the background all week.")
    push("here is the part nobody tells you about shipping something people actually want.")
    push("what happened next completely changed how we thought about the whole problem.")
    push("why does this keep working when every reasonable model says it should not?")
    for _ in range(3):
        push("revenue moved from 12 thousand to 240 thousand in 90 days across 3 markets.")
    return Transcript(segments=segments, language="en", duration=t)
