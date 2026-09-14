"""The offline speech backend: timed silence with plausible word timings.

There is no speech here.  The point is that *everything downstream* --
captions, chat overlays, the ducking mix, the timeline arithmetic -- needs a
narration track of the right length with word-level timings inside it, and this
produces exactly that with nothing but ffmpeg.  It is what makes the whole
engine runnable in CI, in a container with no egress, and on a laptop with no
API key.

The model: English narration runs at about :data:`WORDS_PER_SECOND` words per
second, scaled by ``VoiceSpec.rate``.  Inside that budget each word gets a slice
proportional to its length, separated by a short gap that widens after
punctuation, plus a small deterministic jitter seeded from ``settings.seed`` so
repeated runs of the same script produce byte-identical timings.

Exactness matters more than realism here.  The rendered file is measured with
``ffprobe`` after the fact and every timing is rescaled onto that measurement,
so ``result.words[-1].end == result.duration`` -- a caption can never outlive
its audio and the concatenation arithmetic in
:func:`aiclipper.tts.base.synthesize_lines` stays exact.
"""

from __future__ import annotations

import random
import re
import zlib
from pathlib import Path

from .. import ffmpeg
from ..config import Settings
from ..models import TTSResult, VoiceSpec, Word
from .base import _settings as _resolve_settings

__all__ = ["OfflineTTS", "WORDS_PER_SECOND", "plan_words", "estimate_duration"]

#: Comfortable narration pace for English, before ``VoiceSpec.rate``.
WORDS_PER_SECOND = 2.6

#: Silence between two ordinary words, and after a clause/sentence break.
_WORD_GAP = 0.055
_CLAUSE_GAP = 0.14
_SENTENCE_GAP = 0.26

#: How far a word's slice may be jittered, either side of its weight.
_JITTER = 0.12

#: Length weights are clamped so one very long word cannot starve the rest.
_MIN_WEIGHT = 1.5
_MAX_WEIGHT = 12.0

#: A blank line still occupies a beat rather than a zero-length file.
_EMPTY_DURATION = 0.28

_ALNUM_RE = re.compile(r"[^\W_]+", re.UNICODE)
_CLAUSE_END = ",;:-–—"
_SENTENCE_END = ".!?…"


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _tokenize(text: str) -> list[str]:
    return [tok for tok in (text or "").split() if tok.strip()]


def _weight(token: str) -> float:
    letters = "".join(_ALNUM_RE.findall(token))
    return _clamp(float(len(letters) or 1), _MIN_WEIGHT, _MAX_WEIGHT)


def _gap_after(token: str) -> float:
    tail = token.rstrip("\"')]}”’")
    if tail.endswith(tuple(_SENTENCE_END)):
        return _SENTENCE_GAP
    if tail.endswith(tuple(_CLAUSE_END)):
        return _CLAUSE_GAP
    return _WORD_GAP


def estimate_duration(text: str, *, rate: float = 1.0) -> float:
    """Seconds of narration ``text`` would occupy at ``rate``.

    Deterministic, no ffmpeg, no I/O -- the pipelines use it to budget a script
    before anything is rendered.
    """
    tokens = _tokenize(text)
    if not tokens:
        return _EMPTY_DURATION
    speed = _clamp(float(rate or 1.0), 0.25, 4.0)
    return len(tokens) / (WORDS_PER_SECOND * speed)


def plan_words(text: str, *, rate: float = 1.0, seed: int = 0, total: float | None = None) -> list[Word]:
    """Synthesise word timings for ``text``, spanning exactly ``total`` seconds.

    ``total`` defaults to :func:`estimate_duration`.  The returned words are
    strictly ordered, never overlap, and the last one ends exactly at ``total``.
    Pure: no ffmpeg, no filesystem, and identical output for identical inputs.
    """
    tokens = _tokenize(text)
    span = float(total) if total is not None else estimate_duration(text, rate=rate)
    if not tokens or span <= 0:
        return []

    rng = random.Random((int(seed) & 0xFFFFFFFF) ^ zlib.crc32(text.encode("utf-8", "replace")))
    speed = _clamp(float(rate or 1.0), 0.25, 4.0)

    weights = [_weight(tok) * rng.uniform(1.0 - _JITTER, 1.0 + _JITTER) for tok in tokens]
    gaps = [_gap_after(tok) / speed for tok in tokens[:-1]]

    # Gaps never get to eat more than half the budget, however punctuated.
    gap_total = sum(gaps)
    budget = span * 0.5
    if gap_total > budget:
        shrink = budget / gap_total
        gaps = [g * shrink for g in gaps]
        gap_total = budget

    speech = max(span * 0.5, span - gap_total)
    weight_total = sum(weights) or 1.0

    words: list[Word] = []
    cursor = 0.0
    for i, token in enumerate(tokens):
        length = speech * weights[i] / weight_total
        words.append(Word(token, cursor, cursor + length, 1.0))
        cursor += length
        if i < len(gaps):
            cursor += gaps[i]

    return _rescale(words, span)


def _rescale(words: list[Word], target: float) -> list[Word]:
    """Stretch ``words`` so the last one ends exactly at ``target``."""
    if not words or target <= 0:
        return words
    raw = words[-1].end
    if raw <= 0:
        return words
    factor = target / raw
    if factor == 1.0:
        return words
    return [Word(w.text, w.start * factor, w.end * factor, w.prob) for w in words]


class OfflineTTS:
    """Renders timed silence and synthetic word timings.  Never fails, never
    touches the network, needs only ffmpeg."""

    name = "offline"

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = _resolve_settings(settings)

    def available(self) -> bool:
        """True whenever ffmpeg is installed -- there is nothing else to check."""
        return ffmpeg.have_ffmpeg(self.settings)

    def usable(self, *, timeout: float | None = None, refresh: bool = False) -> bool:
        """The capability probe, which for this backend is the availability check.

        There is no service to reach and no key to validate, so "could it
        speak?" collapses to "is ffmpeg here?" -- which is why this backend is
        the floor of :func:`aiclipper.tts.base.fallback_chain`.  ``timeout`` is
        accepted for interface symmetry and never needed.  See
        :func:`aiclipper.tts.base.provider_usable` for the distinction this
        method draws elsewhere.
        """
        del timeout, refresh  # nothing to bound and nothing worth caching
        return self.available()

    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        """Write a silent wav of the right length and describe its word timings.

        ``VoiceSpec.rate`` scales the pace (``1.15`` -> 15% faster, so 15%
        shorter); ``voice_id``, ``pitch_semitones`` and ``style`` are recorded on
        the result but cannot change silence.
        """
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        spec = voice or VoiceSpec()
        target = estimate_duration(text, rate=spec.rate)
        ffmpeg.make_silence(target, out, settings=self.settings)

        # Measure what ffmpeg actually produced -- sample-rate quantisation and
        # the 3-decimal duration argument both move it a little -- and put the
        # word timings on *that* clock rather than on our estimate.
        duration = ffmpeg.probe(out, settings=self.settings).duration or target
        words = plan_words(text, rate=spec.rate, seed=self.settings.seed, total=duration)

        return TTSResult(
            audio_path=out,
            duration=duration,
            words=words,
            voice=spec,
            text=text or "",
        )
