"""The ``edge-tts`` backend: free neural speech with word boundaries.

``edge_tts`` is an optional extra and is imported lazily inside the methods that
need it, so importing this module on a bare interpreter is safe and
:class:`~aiclipper.errors.MissingDependency` is raised at call time instead.

Three details are worth knowing because they are easy to get wrong:

* **Prosody is a string.**  The service wants ``rate="+15%"`` and
  ``pitch="+24Hz"``, not numbers.  :func:`format_rate` and :func:`format_pitch`
  map our provider-agnostic :class:`~aiclipper.models.VoiceSpec` onto that.
* **Boundary offsets are 100-nanosecond ticks**, not seconds and not
  milliseconds.  :func:`ticks_to_seconds` is the only place that conversion
  happens.
* **The stream is async.**  :func:`run_async` drives it with :func:`asyncio.run`
  and falls back to a worker thread when the caller already has a running event
  loop, so this stays callable from a notebook or an async web handler.

The service returns mp3; we transcode to the caller's ``out_path`` with ffmpeg
so every backend hands downstream code the same kind of file, and take
``TTSResult.duration`` from ``ffprobe`` rather than trusting the boundary
events.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import logging
import tempfile
from collections.abc import Awaitable, Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, TypeVar

from .. import ffmpeg
from ..config import Settings
from ..errors import MissingDependency, TTSError
from ..models import TTSResult, VoiceSpec, Word
from .base import _settings as _resolve_settings
from .voices import resolve_voice_id

log = logging.getLogger(__name__)

__all__ = [
    "EdgeTTS", "DEFAULT_EDGE_VOICE", "TICKS_PER_SECOND",
    "format_rate", "format_pitch", "ticks_to_seconds", "words_from_boundaries",
    "run_async", "edge_available",
]

T = TypeVar("T")

#: Used when the caller gave no voice at all.
DEFAULT_EDGE_VOICE = "en-US-GuyNeural"

#: The service reports offsets and durations in 100-nanosecond ticks.
TICKS_PER_SECOND = 10_000_000

#: Semitones are converted to Hz against this reference pitch -- roughly the
#: centre of a spoken voice -- because the service only accepts an absolute Hz
#: offset.  One semitone comes out near 12 Hz, which is the right order of
#: magnitude for speech without being so coarse it sounds robotic.
PITCH_REFERENCE_HZ = 200.0

#: The service rejects extreme prosody; clamp rather than 400 out.
_RATE_LIMITS = (0.5, 2.0)
_PITCH_LIMITS = (-12.0, 12.0)


def edge_available() -> bool:
    """True when the ``edge_tts`` package is importable (no network check)."""
    try:
        return importlib.util.find_spec("edge_tts") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install only
        return False


def format_rate(rate: float) -> str:
    """``1.15`` -> ``"+15%"``, ``0.8`` -> ``"-20%"``, ``1.0`` -> ``"+0%"``.

    The multiplier is clamped to 0.5x..2.0x, which is the usable range.
    """
    try:
        value = float(rate)
    except (TypeError, ValueError):
        value = 1.0
    if value <= 0:
        value = 1.0
    value = max(_RATE_LIMITS[0], min(_RATE_LIMITS[1], value))
    percent = int(round((value - 1.0) * 100))
    return f"{percent:+d}%"


def format_pitch(semitones: float) -> str:
    """``2`` -> ``"+24Hz"``, ``-1`` -> ``"-11Hz"``, ``0`` -> ``"+0Hz"``.

    Semitones are musical (a ratio); the service wants an absolute Hz offset, so
    the ratio is taken against :data:`PITCH_REFERENCE_HZ`.  Clamped to +/- one
    octave.
    """
    try:
        value = float(semitones)
    except (TypeError, ValueError):
        value = 0.0
    value = max(_PITCH_LIMITS[0], min(_PITCH_LIMITS[1], value))
    hz = int(round(PITCH_REFERENCE_HZ * (2.0 ** (value / 12.0) - 1.0)))
    return f"{hz:+d}Hz"


def ticks_to_seconds(ticks: float) -> float:
    """100-nanosecond ticks -> seconds.  ``10_000_000`` ticks is one second."""
    try:
        return float(ticks) / TICKS_PER_SECOND
    except (TypeError, ValueError):
        return 0.0


def words_from_boundaries(events: Iterable[dict[str, Any]], *, limit: float | None = None) -> list[Word]:
    """Turn ``WordBoundary`` chunks into :class:`~aiclipper.models.Word` objects.

    Each event carries ``offset`` and ``duration`` in ticks plus the spoken
    ``text``.  Events are sorted by offset, zero-length words are given a
    minimum width, and everything is clipped to ``limit`` (the measured audio
    duration) when one is given.  A word with no room left inside ``limit`` is
    dropped rather than emitted zero-length: every returned word satisfies
    ``end > start``, which is what the caption grouper assumes.
    """
    raw: list[Word] = []
    for event in events or ():
        if not isinstance(event, dict):
            continue
        text = str(event.get("text") or "").strip()
        if not text:
            continue
        start = ticks_to_seconds(event.get("offset", 0))
        length = ticks_to_seconds(event.get("duration", 0))
        raw.append(Word(text, start, start + max(length, 0.01), 1.0))

    raw.sort(key=lambda w: (w.start, w.end))

    out: list[Word] = []
    for word in raw:
        start = word.start
        if out and start < out[-1].end:
            start = out[-1].end
        end = max(word.end, start + 0.01)
        if limit is not None and limit > 0:
            if start >= limit:
                # The service placed this word past the end of the audio we
                # actually received; a zero-length caption helps nobody.
                continue
            end = min(end, limit)
        if end <= start:
            continue
        out.append(Word(word.text, start, end, word.prob))
    return out


def run_async(factory: Callable[[], Awaitable[T]]) -> T:
    """Run ``factory()`` to completion, even from inside a running event loop.

    ``factory`` is a zero-argument callable rather than a coroutine so that the
    coroutine is created on whichever thread ends up awaiting it -- creating one
    and never awaiting it (the loop-is-running path) would warn.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(factory())
    # A loop is already spinning on this thread; asyncio.run() would explode, so
    # hand the work to a thread that has no loop of its own.
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="aiclip-edge") as pool:
        return pool.submit(lambda: asyncio.run(factory())).result()


class EdgeTTS:
    """Speech via Microsoft's Edge read-aloud service, through ``edge-tts``."""

    name = "edge"

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = _resolve_settings(settings)

    def available(self) -> bool:
        """``edge_tts`` importable, ffmpeg present, and not running offline."""
        if self.settings.offline:
            return False
        return edge_available() and ffmpeg.have_ffmpeg(self.settings)

    # -- internals --------------------------------------------------------- #
    def _require(self) -> Any:
        if self.settings.offline:
            raise TTSError(
                "edge-tts needs network access but settings.offline is set; "
                "use the 'offline' provider or unset AICLIP_OFFLINE"
            )
        try:
            import edge_tts  # noqa: PLC0415 - optional extra, imported at call time
        except ImportError as exc:
            raise MissingDependency("edge-tts", extra="tts", purpose="neural speech synthesis") from exc
        return edge_tts

    def voice_id(self, voice: VoiceSpec | None) -> str:
        """Native Edge voice name for ``voice`` (catalogue name or native id)."""
        return resolve_voice_id(voice, "edge", default=DEFAULT_EDGE_VOICE) or DEFAULT_EDGE_VOICE

    def _communicate(self, edge_tts: Any, text: str, voice: VoiceSpec) -> Any:
        kwargs: dict[str, Any] = {
            "rate": format_rate(voice.rate),
            "pitch": format_pitch(voice.pitch_semitones),
        }
        # edge-tts >= 7 defaults to SentenceBoundary; we need per-word events.
        try:
            params = inspect.signature(edge_tts.Communicate.__init__).parameters
        except (TypeError, ValueError):  # pragma: no cover - exotic stubs only
            params = {}
        if "boundary" in params:
            kwargs["boundary"] = "WordBoundary"
        return edge_tts.Communicate(text, self.voice_id(voice), **kwargs)

    def _stream(self, text: str, voice: VoiceSpec) -> tuple[bytes, list[dict[str, Any]]]:
        edge_tts = self._require()

        async def _collect() -> tuple[bytes, list[dict[str, Any]]]:
            audio = bytearray()
            boundaries: list[dict[str, Any]] = []
            communicate = self._communicate(edge_tts, text, voice)
            async for chunk in communicate.stream():
                kind = chunk.get("type")
                if kind == "audio":
                    data = chunk.get("data")
                    if data:
                        audio.extend(data)
                elif kind == "WordBoundary":
                    boundaries.append(dict(chunk))
            return bytes(audio), boundaries

        try:
            return run_async(_collect)
        except (MissingDependency, TTSError):
            raise
        except Exception as exc:  # noqa: BLE001 - the library raises many types
            raise TTSError(f"edge-tts synthesis failed: {exc}") from exc

    # -- interface --------------------------------------------------------- #
    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        """Speak ``text`` and write it to ``out_path`` (transcoded from mp3)."""
        if not (text or "").strip():
            raise TTSError("edge-tts cannot synthesise empty text")
        spec = voice or VoiceSpec()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        audio, boundaries = self._stream(text, spec)
        if not audio:
            raise TTSError(f"edge-tts returned no audio for voice {self.voice_id(spec)!r}")

        with tempfile.TemporaryDirectory(prefix="aiclip-edge-") as tmp:
            mp3 = Path(tmp) / "speech.mp3"
            mp3.write_bytes(audio)
            _transcode(mp3, out, settings=self.settings)

        duration = ffmpeg.probe(out, settings=self.settings).duration
        words = words_from_boundaries(boundaries, limit=duration) or None
        if words is None:
            log.debug("edge-tts reported no word boundaries for %r", text[:40])

        return TTSResult(
            audio_path=out,
            duration=duration,
            words=words,
            voice=spec,
            text=text,
        )


def _transcode(src: Path, dst: Path, *, settings: Settings | None = None, sample_rate: int = 44100) -> Path:
    """mp3 -> the caller's container, mono, so downstream is uniform."""
    args: Sequence[str] = [
        "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", str(sample_rate), str(dst),
    ]
    ffmpeg.run_ffmpeg(args, settings=settings)
    return dst
