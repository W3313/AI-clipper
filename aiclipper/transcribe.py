"""Speech recognition and forced alignment.

Two jobs live here:

* :func:`transcribe` turns any media file into a word-level :class:`Transcript`
  using `faster-whisper <https://github.com/SYSTRAN/faster-whisper>`_.  Audio is
  decoded to 16 kHz mono WAV through :mod:`aiclipper.ffmpeg` first, and the
  result is cached next to the media as ``<stem>.transcript.json``.
* :func:`align` force-aligns *known* narration text (a TTS script) against the
  audio that was synthesised from it.  The ASR pass supplies the timings, the
  reference text supplies the spelling and punctuation, so burned-in captions
  match the script character for character even when the recogniser mishears.

faster-whisper is an optional extra: this module imports cleanly without it and
raises :class:`~aiclipper.errors.MissingDependency` only when a call actually
needs the recogniser.  Everything else here -- tokenisation, the alignment
mapping, the transcript cache -- is pure standard library and works offline.
"""

from __future__ import annotations

import difflib
import importlib.util
import logging
import re
import tempfile
import unicodedata
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .config import Settings, get_settings
from .errors import MissingDependency, TranscriptionError
from .ffmpeg import FFmpegError, extract_audio, probe_duration
from .models import Segment, Transcript, Word

log = logging.getLogger(__name__)

__all__ = ["transcribe", "align", "available", "cache_path"]

#: Shortest timing we will ever emit for a word, in seconds.
MIN_WORD_DURATION = 0.02

#: Speaking rate used when timings have to be invented from nothing.
WORDS_PER_SECOND = 2.6

_SAMPLE_RATE = 16000
_PUNCT_RE = re.compile(r"[^\w']+", re.UNICODE)
_APOSTROPHES = str.maketrans({"’": "'", "ʼ": "'", "‘": "'", "´": "'"})


# --------------------------------------------------------------------------- #
# availability
# --------------------------------------------------------------------------- #

def available() -> bool:
    """True when :mod:`faster_whisper` is importable (no import performed)."""
    try:
        return importlib.util.find_spec("faster_whisper") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken installs only
        return False


def _require_backend() -> None:
    if not available():
        raise MissingDependency("faster-whisper", extra="transcribe", purpose="speech recognition")


def _resolve_device(device: str | None) -> str:
    """``auto`` resolves to ``cpu`` -- this engine assumes no GPU."""
    value = (device or "").strip().lower()
    return "cpu" if value in {"", "auto"} else value


# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #

def _tokenize(text: str) -> list[str]:
    """Split reference text into tokens, keeping spelling and punctuation."""
    return [tok for tok in (text or "").split() if tok.strip()]


def _normalize(token: str) -> str:
    """Matching key for one token: casefolded, unaccented, punctuation-free."""
    raw = (token or "").translate(_APOSTROPHES).strip()
    folded = unicodedata.normalize("NFKD", raw.casefold())
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    key = _PUNCT_RE.sub("", folded).strip("'")
    return key or folded.strip()


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #

def _spread(tokens: Sequence[str], start: float, end: float, prob: float = 0.0) -> list[Word]:
    """Lay ``tokens`` out evenly across ``[start, end]``."""
    count = len(tokens)
    if count == 0:
        return []
    lo = max(0.0, float(start))
    span = max(float(end) - lo, count * MIN_WORD_DURATION)
    step = span / count
    return [Word(tok, lo + i * step, lo + (i + 1) * step, prob) for i, tok in enumerate(tokens)]


def _repair(tokens: Sequence[str], starts: list[float], ends: list[float], probs: list[float]) -> list[Word]:
    """Force the timings to be ordered, non-overlapping and non-degenerate."""
    out: list[Word] = []
    prev_end = 0.0
    for i, tok in enumerate(tokens):
        start = max(0.0, starts[i], prev_end)
        end = max(ends[i], start + MIN_WORD_DURATION)
        out.append(Word(tok, start, end, probs[i]))
        prev_end = end
    return out


def _align_tokens(
    ref_tokens: Sequence[str],
    asr_words: Sequence[Word],
    *,
    duration: float | None = None,
) -> list[Word]:
    """Map ASR timings onto ``ref_tokens``, keeping the reference spelling.

    Every reference token comes back exactly once, in order, with a positive
    duration and no overlap with its neighbours.  Tokens the recogniser matched
    inherit its timing and probability; tokens inside an unmatched run get
    timings interpolated linearly across the span that run occupies.
    """
    ref = [tok for tok in ref_tokens if tok and tok.strip()]
    if not ref:
        return []

    asr = [w for w in asr_words if (w.text or "").strip()]
    if not asr:
        total = float(duration) if duration and duration > 0 else len(ref) / WORDS_PER_SECOND
        return _spread(ref, 0.0, total, prob=0.0)

    n = len(ref)
    starts = [0.0] * n
    ends = [0.0] * n
    probs = [0.0] * n
    matched = [False] * n

    ref_keys = [_normalize(tok) for tok in ref]
    asr_keys = [_normalize(w.text) for w in asr]
    matcher = difflib.SequenceMatcher(a=ref_keys, b=asr_keys, autojunk=False)

    gaps: list[tuple[int, int, int, int]] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                word = asr[j1 + k]
                starts[i1 + k] = float(word.start)
                ends[i1 + k] = max(float(word.end), float(word.start))
                probs[i1 + k] = float(word.prob)
                matched[i1 + k] = True
        elif i2 > i1:  # replace / delete -- reference tokens with no direct match
            gaps.append((i1, i2, j1, j2))
        # "insert" means the recogniser heard extra words; they simply vanish.

    for i1, i2, j1, j2 in gaps:
        count = i2 - i1
        covering = asr[j1:j2]
        width = count / WORDS_PER_SECOND
        if covering:
            # The recogniser heard *something* here: reuse exactly that span.
            lo = float(covering[0].start)
            hi = max(float(covering[-1].end), lo)
            prob = sum(float(w.prob) for w in covering) / len(covering)
        else:
            # A pure deletion: nothing was heard where these tokens belong, so
            # slot them into the hole between the surrounding anchors.  Gaps are
            # visited left to right, so ``ends[i1 - 1]`` is already final.
            before = ends[i1 - 1] if i1 > 0 else None
            after = starts[i2] if i2 < n else None
            if before is None and after is None:  # pragma: no cover - asr is non-empty here
                lo, hi = 0.0, width
            elif before is None:  # the run sits *before* the first recognised word
                hi = float(after)
                lo = max(0.0, hi - width)
            elif after is None:  # the run sits *after* the last recognised word
                lo = float(before)
                hi = lo + width
            else:
                lo, hi = float(before), float(after)
            neighbours = [probs[i] for i in (i1 - 1, i2) if 0 <= i < n and matched[i]]
            prob = sum(neighbours) / len(neighbours) if neighbours else 0.0
        if hi <= lo:
            hi = lo + width
        step = (hi - lo) / count
        for k in range(count):
            starts[i1 + k] = lo + k * step
            ends[i1 + k] = lo + (k + 1) * step
            probs[i1 + k] = prob

    return _repair(ref, starts, ends, probs)


def align(
    audio: str | Path,
    text: str,
    *,
    settings: Settings | None = None,
    language: str = "en",
    cache: bool = False,
) -> list[Word]:
    """Force-align known narration ``text`` to ``audio``.

    Transcribes the audio, then maps the ASR words onto the reference tokens so
    the returned words carry the reference spelling and punctuation with ASR
    timings.  Unmatched reference tokens get interpolated timings; none are ever
    dropped.
    """
    ref = _tokenize(text)
    if not ref:
        return []
    transcript = transcribe(audio, settings=settings, language=language or None, cache=cache)
    return _align_tokens(ref, transcript.words, duration=transcript.duration)


# --------------------------------------------------------------------------- #
# transcript cache
# --------------------------------------------------------------------------- #

def cache_path(media: str | Path) -> Path:
    """Where the cached transcript for ``media`` lives (next to the media)."""
    path = Path(media)
    return path.with_name(f"{path.stem}.transcript.json")


def _read_cache(media: Path, *, language: str | None = None) -> Transcript | None:
    """Return the cached transcript when it is newer than the media file."""
    path = cache_path(media)
    try:
        if path.stat().st_mtime < media.stat().st_mtime:
            return None
    except OSError:
        return None
    try:
        transcript = Transcript.load(path)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        log.warning("ignoring unreadable transcript cache %s: %s", path, exc)
        return None
    if language and transcript.language and transcript.language != language:
        return None
    return transcript


def _write_cache(media: Path, transcript: Transcript) -> Path | None:
    path = cache_path(media)
    try:
        return transcript.save(path)
    except OSError as exc:  # pragma: no cover - read-only media directories
        log.warning("could not write transcript cache %s: %s", path, exc)
        return None


# --------------------------------------------------------------------------- #
# faster-whisper plumbing
# --------------------------------------------------------------------------- #

def _load_model(
    name: str,
    *,
    device: str = "cpu",
    compute_type: str = "int8",
    local_files_only: bool = False,
) -> Any:
    """Instantiate a ``WhisperModel``; downloads weights on first use.

    ``local_files_only`` forbids that download -- it is how ``settings.offline``
    is honoured, so an offline run either uses already-cached weights or fails
    loudly instead of reaching for the network.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise MissingDependency("faster-whisper", extra="transcribe", purpose="speech recognition") from exc
    try:
        return WhisperModel(
            name, device=device, compute_type=compute_type, local_files_only=local_files_only
        )
    except Exception as exc:  # noqa: BLE001 - ctranslate2/hub raise a zoo of types
        hint = (
            "settings.offline is set, so the weights must already be in the local cache."
            if local_files_only
            else "The first run downloads the weights, which needs network access."
        )
        raise TranscriptionError(
            f"could not load whisper model {name!r} on device {device!r}: {exc}. {hint}"
        ) from exc


def _word_from_asr(raw: Any) -> Word | None:
    text = str(getattr(raw, "word", "") or "").strip()
    if not text:
        return None
    start = float(getattr(raw, "start", 0.0) or 0.0)
    end = float(getattr(raw, "end", start) or start)
    prob = float(getattr(raw, "probability", 1.0) or 0.0)
    return Word(text, start, max(start, end), prob)


def _to_transcript(segments: Any, info: Any, *, fallback_duration: float = 0.0) -> Transcript:
    """Convert faster-whisper segments/info into our :class:`Transcript`."""
    out: list[Segment] = []
    for raw in segments:
        words = [w for w in (_word_from_asr(r) for r in (getattr(raw, "words", None) or [])) if w]
        text = str(getattr(raw, "text", "") or "").strip()
        if not text and words:
            text = " ".join(w.text for w in words)
        if not text and not words:
            continue
        start = float(getattr(raw, "start", 0.0) or 0.0)
        end = float(getattr(raw, "end", start) or start)
        if words:
            start = min(start, words[0].start)
            end = max(end, words[-1].end)
        out.append(Segment(text=text, start=start, end=max(start, end), words=words))

    language = str(getattr(info, "language", "") or "") or "en"
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    if duration <= 0.0:
        duration = max(fallback_duration, out[-1].end if out else 0.0)
    return Transcript(segments=out, language=language, duration=max(0.0, duration))


def transcribe(
    media: str | Path,
    *,
    settings: Settings | None = None,
    language: str | None = None,
    model: str | None = None,
    vad: bool = True,
    cache: bool = True,
) -> Transcript:
    """Transcribe ``media`` to a word-level :class:`Transcript`.

    The audio is decoded to 16 kHz mono WAV in a temporary directory, then run
    through faster-whisper with ``word_timestamps=True``.  With ``cache`` (the
    default) a transcript sitting next to the media and newer than it is reused
    verbatim, and a fresh result is written back there.
    """
    s = settings or get_settings()
    path = Path(media).expanduser()
    if not path.exists():
        raise TranscriptionError(f"media file not found: {path}")

    if cache:
        cached = _read_cache(path, language=language)
        if cached is not None:
            log.debug("transcript cache hit for %s", path)
            return cached

    _require_backend()
    name = model or s.whisper_model
    try:
        fallback_duration = probe_duration(path, settings=s)
    except (FFmpegError, OSError, ValueError):
        fallback_duration = 0.0

    with tempfile.TemporaryDirectory(prefix="aiclip-asr-") as tmp:
        wav = Path(tmp) / f"{path.stem}.16k.wav"
        try:
            extract_audio(path, wav, sample_rate=_SAMPLE_RATE, mono=True, settings=s)
        except (FFmpegError, OSError) as exc:
            raise TranscriptionError(f"could not extract audio from {path}: {exc}") from exc

        whisper = _load_model(
            name,
            device=_resolve_device(s.whisper_device),
            compute_type=s.whisper_compute_type,
            local_files_only=bool(s.offline),
        )
        try:
            segments, info = whisper.transcribe(
                str(wav),
                language=language,
                word_timestamps=True,
                vad_filter=bool(vad),
            )
            materialised = list(segments)  # the generator does the actual work
        except Exception as exc:  # noqa: BLE001 - backend failures are opaque
            raise TranscriptionError(f"transcription of {path.name} failed: {exc}") from exc
        transcript = _to_transcript(materialised, info, fallback_duration=fallback_duration)

    if cache:
        _write_cache(path, transcript)
    return transcript
