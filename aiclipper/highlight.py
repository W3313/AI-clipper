"""Clip selection: turn a :class:`~aiclipper.models.Transcript` into ranked clips.

Two paths feed the same funnel:

* **LLM path** -- when a provider is available the transcript is offered to
  :mod:`aiclipper.llm` and the model is asked for self-contained moments by
  timestamp.  Everything that comes back is treated as hostile: timestamps may
  be floats, numeric strings, ``"MM:SS"`` strings, ``None`` or nonsense, and any
  window that is inverted or outside the media is dropped.  If the package is
  missing, the provider is unavailable, or the call raises for any reason we
  silently fall back to the heuristic path.
* **Heuristic path** -- slide windows of several lengths across the transcript
  and score each one (see below).  This always runs, so a clip selection is
  produced with zero network access.

Candidates from both paths are snapped to word edges, (re)scored, de-duplicated
by overlap keeping the higher score, trimmed to ``count`` and returned ranked.

Scoring weights
---------------

``score_window`` is a weighted sum of eight features, each normalised to 0..1,
minus two penalties, clamped to 0..1.  The positive weights sum to exactly 1.0:

===============  ======  ====================================================
feature          weight  what it measures
===============  ======  ====================================================
``hook``          0.22   known hook phrases ("the crazy part", "here's why")
``pace``          0.15   words-per-second relative to the transcript median
``boundary``      0.15   starts on a sentence start / ends on a sentence end
``numbers``       0.12   digits, number words and superlatives
``question``      0.10   question marks
``density``       0.10   share of the window that is actually speech
``entities``      0.08   crude capitalised-token named-entity count
``reaction``      0.08   laughter and reaction markers
===============  ======  ====================================================

Penalties: ``mid_sentence`` (0.12) when the window opens mid-sentence, and
``silence`` (up to 0.18) scaled by the longest internal pause beyond 1.0s.

Boundaries
----------

Every returned window sits on word edges, holds at least one whole word and
respects ``[min_duration, max_duration]`` whenever any nearby window of whole
words can -- a long pause can leave no legal cut, and a transcript shorter than
``min_duration`` obviously cannot host one; both degrade rather than raise.
Re-snapping an already snapped window is a no-op.

Transcripts whose segments carry no word timings at all (``Segment.words``
empty -- "coarse ASR", which :mod:`aiclipper.transcribe` does emit when the
model returns no word timestamps) are handled by spreading each segment's text
evenly over its span, so selection still works with interpolated edges.

Determinism
-----------

Selection is deterministic.  ``settings.seed`` only picks the sub-step phase of
the sliding-window grid, so the same settings always produce the same windows.
"""

from __future__ import annotations

import bisect
import itertools
import logging
import math
import random
import re
import statistics
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from .config import Settings, get_settings
from .models import ClipCandidate, Transcript, Word

log = logging.getLogger(__name__)

__all__ = [
    "select",
    "score_window",
    "snap_to_speech",
    "WEIGHTS",
    "PENALTIES",
    "HOOK_PHRASES",
]


# --------------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------------- #

WEIGHTS: dict[str, float] = {
    "hook": 0.22,
    "pace": 0.15,
    "boundary": 0.15,
    "numbers": 0.12,
    "question": 0.10,
    "density": 0.10,
    "entities": 0.08,
    "reaction": 0.08,
}

PENALTIES: dict[str, float] = {
    "mid_sentence": 0.12,
    "silence": 0.18,
}

#: Phrases that reliably introduce a payoff in spoken content.
HOOK_PHRASES: tuple[str, ...] = (
    "the crazy part",
    "here's why",
    "here is why",
    "nobody tells you",
    "what nobody tells you",
    "what happened next",
    "here's the thing",
    "here is the thing",
    "the best part",
    "the worst part",
    "this is the part",
    "let me explain",
    "believe it or not",
    "you won't believe",
    "the truth is",
    "no one talks about",
    "nobody talks about",
    "turns out",
    "it turns out",
    "the secret",
    "the reason why",
    "and then it hit me",
    "plot twist",
    "i'll never forget",
    "listen to this",
    "watch this",
    "the mistake everyone makes",
    "most people think",
    "but here's the catch",
)

_NUMBER_WORDS = frozenset(
    """zero one two three four five six seven eight nine ten eleven twelve twenty thirty forty
    fifty sixty seventy eighty ninety hundred thousand million billion trillion percent
    half double triple dozen first second third"""
    .split()
)

_SUPERLATIVES = frozenset(
    """best worst biggest smallest fastest slowest hardest easiest craziest wildest weirdest
    largest cheapest richest strongest greatest highest lowest most least only never always
    huge insane massive ridiculous unbelievable incredible impossible perfect"""
    .split()
)

_REACTIONS = frozenset(
    """laughs laughter laughing haha hahaha lol lmao wow whoa woah omg oh wait what damn
    unbelievable seriously exactly yikes applause gasp"""
    .split()
)

#: Capitalised tokens that are almost never entities.
_ENTITY_STOPWORDS = frozenset(
    """i i'm i've a an the and but or so if then that this these those it its he she they we you
    your our their there here what when where why how who which is are was were be been am do does
    did no not yes ok okay well now just like because"""
    .split()
)

_SENTENCE_END_CHARS = (".", "!", "?", "…")

#: A pause longer than this splits a sentence even without punctuation.
_GAP_BOUNDARY = 0.70
#: How far ``snap_to_speech`` will look for a nicer sentence boundary.
_SENTENCE_PULL = 2.5
#: Bonus added to the score of a candidate proposed by the language model.
_LLM_BONUS = 0.08
#: Default pace assumed when a transcript has no usable segment timings.
_DEFAULT_WPS = 2.6
#: Providers whose "moments" are invented by walking the schema rather than read
#: out of the transcript.  Their timestamps carry no information about this
#: media, so the local window scan is strictly better and the LLM path is skipped.
_SYNTHETIC_PROVIDERS = ("heuristic", "offline", "null", "dummy", "stub")

_CLEAN_RE = re.compile(r"[^a-z0-9]+")
_APOSTROPHES = str.maketrans("", "", "'’ʼ`")
_TIMECODE_RE = re.compile(r"^\s*(\d{1,3}):([0-5]?\d)(?::([0-5]?\d))?(\.\d+)?\s*$")


def _clean(text: str) -> str:
    """Lowercase, drop apostrophes, collapse everything else to single spaces."""
    return _CLEAN_RE.sub(" ", text.lower().translate(_APOSTROPHES)).strip()


_HOOKS: tuple[str, ...] = tuple(sorted({_clean(p) for p in HOOK_PHRASES} - {""}, key=len, reverse=True))


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _clamp(value: float, low: float, high: float) -> float:
    if high < low:
        return low
    return max(low, min(high, value))


def _ramp(value: float, low: float, high: float) -> float:
    """Map ``value`` from the range ``low..high`` onto 0..1."""
    if high <= low:
        return 0.0
    return _clamp((value - low) / (high - low), 0.0, 1.0)


# --------------------------------------------------------------------------- #
# transcript index
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Index:
    """Per-word feature flags plus sorted edges, so windows score in O(window)."""

    words: tuple[Word, ...]
    starts: tuple[float, ...]
    ends: tuple[float, ...]
    #: Running maximum of ``ends``.  ``ends`` itself can dip when one word is
    #: nested inside another (rare, but real ASR does it), and every lookup here
    #: is a bisect that needs a sorted sequence.
    ends_mono: tuple[float, ...]
    clean: tuple[str, ...]
    question: tuple[bool, ...]
    number: tuple[bool, ...]
    superlative: tuple[bool, ...]
    entity: tuple[bool, ...]
    reaction: tuple[bool, ...]
    sent_start: tuple[bool, ...]
    sent_end: tuple[bool, ...]
    median_wps: float
    limit: float

    @property
    def empty(self) -> bool:
        return not self.words

    @property
    def first(self) -> float:
        return self.starts[0] if self.starts else 0.0

    @property
    def last(self) -> float:
        return self.ends[-1] if self.ends else 0.0


def _interpolated_words(seg: Any) -> list[Word]:
    """Fake word timings for a coarse segment by spreading its text over its span.

    ``Segment.words`` is allowed to be empty ("coarse ASR", see
    :mod:`aiclipper.models`) and :func:`aiclipper.transcribe.transcribe` really
    does emit such segments when the model returns no word timestamps.  Rather
    than selecting nothing at all, treat each token as an evenly spaced pseudo
    word: boundaries are then approximate, but every other feature still works.
    """
    text = str(getattr(seg, "text", "") or "").strip()
    start = getattr(seg, "start", None)
    end = getattr(seg, "end", None)
    if not text or not _finite(start) or not _finite(end):
        return []
    start, end = float(start), float(end)
    tokens = text.split()
    if not tokens or end <= start:
        return []
    step = (end - start) / len(tokens)
    return [Word(tok, start + i * step, start + (i + 1) * step) for i, tok in enumerate(tokens)]


def _collect_words(transcript: Transcript) -> tuple[list[Word], set[int]]:
    """Flatten segments into a usable word list plus the indices that open one."""
    segments = list(getattr(transcript, "segments", []) or [])
    flat: list[Word] = []
    seg_first: set[int] = set()
    for seg in segments:
        usable = [
            w
            for w in (getattr(seg, "words", None) or [])
            if _finite(getattr(w, "start", None)) and _finite(getattr(w, "end", None)) and w.text.strip()
        ]
        if not usable:
            continue
        seg_first.add(len(flat))
        flat.extend(usable)
    if not flat:
        for seg in segments:
            usable = _interpolated_words(seg)
            if not usable:
                continue
            seg_first.add(len(flat))
            flat.extend(usable)
    # Defensive: if the segments were not in chronological order, sort and forget
    # the segment-derived boundaries (punctuation and pauses still supply them).
    if any(b.start < a.start for a, b in zip(flat, flat[1:], strict=False)):
        flat.sort(key=lambda w: (w.start, w.end))
        seg_first = {0} if flat else set()
    return flat, seg_first


def _build_index(transcript: Transcript) -> _Index:
    words, seg_first = _collect_words(transcript)
    n = len(words)

    clean = [_clean(w.text) for w in words]
    question = [("?" in w.text) for w in words]
    number = [
        bool(any(ch.isdigit() for ch in w.text) or clean[i] in _NUMBER_WORDS) for i, w in enumerate(words)
    ]
    superlative = [
        bool(clean[i] in _SUPERLATIVES or (len(clean[i]) > 5 and clean[i].endswith("est")))
        for i in range(n)
    ]
    reaction = [
        bool(clean[i] in _REACTIONS or (w.text.strip()[:1] in "[(" and clean[i] in _REACTIONS))
        for i, w in enumerate(words)
    ]

    sent_end = [False] * n
    for i, w in enumerate(words):
        text = w.text.strip().rstrip("\"')]")
        if text.endswith(_SENTENCE_END_CHARS):
            sent_end[i] = True
    for i in range(1, n):
        if words[i].start - words[i - 1].end > _GAP_BOUNDARY or i in seg_first:
            sent_end[i - 1] = True
    if n:
        sent_end[n - 1] = True

    sent_start = [False] * n
    for i in range(n):
        sent_start[i] = i == 0 or sent_end[i - 1]

    entity = [False] * n
    for i, w in enumerate(words):
        token = w.text.strip().strip("\"'([{")
        if not token or not token[:1].isupper() or not token[:1].isalpha():
            continue
        if sent_start[i] or clean[i] in _ENTITY_STOPWORDS or not clean[i]:
            continue
        entity[i] = True

    rates: list[float] = []
    for seg in getattr(transcript, "segments", []) or []:
        span = float(getattr(seg, "end", 0.0)) - float(getattr(seg, "start", 0.0))
        count = len(getattr(seg, "words", None) or [])
        if span > 0.05 and count:
            rates.append(count / span)
    median_wps = float(statistics.median(rates)) if rates else _DEFAULT_WPS
    if not math.isfinite(median_wps) or median_wps <= 0.0:
        median_wps = _DEFAULT_WPS

    declared = float(getattr(transcript, "duration", 0.0) or 0.0)
    limit = max(declared if math.isfinite(declared) else 0.0, words[-1].end if words else 0.0)

    ends = [w.end for w in words]
    return _Index(
        words=tuple(words),
        starts=tuple(w.start for w in words),
        ends=tuple(ends),
        ends_mono=tuple(itertools.accumulate(ends, max)),
        clean=tuple(clean),
        question=tuple(question),
        number=tuple(number),
        superlative=tuple(superlative),
        entity=tuple(entity),
        reaction=tuple(reaction),
        sent_start=tuple(sent_start),
        sent_end=tuple(sent_end),
        median_wps=median_wps,
        limit=limit,
    )


def _span(idx: _Index, start: float, end: float) -> tuple[int, int]:
    """Indices of the first and last word overlapping ``[start, end)``."""
    if idx.empty:
        return (0, -1)
    lo = bisect.bisect_right(idx.ends_mono, start)
    hi = bisect.bisect_left(idx.starts, end) - 1
    return (lo, hi)


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #

def _features(idx: _Index, start: float, end: float) -> dict[str, float]:
    lo, hi = _span(idx, start, end)
    if lo > hi or end <= start:
        return {}

    count = hi - lo + 1
    duration = end - start
    joined = " ".join(idx.clean[lo : hi + 1])

    hooks = sum(1 for phrase in _HOOKS if phrase and phrase in joined)
    questions = sum(1 for i in range(lo, hi + 1) if idx.question[i])
    numbers = sum(1 for i in range(lo, hi + 1) if idx.number[i] or idx.superlative[i])
    entities = sum(1 for i in range(lo, hi + 1) if idx.entity[i])
    reactions = sum(1 for i in range(lo, hi + 1) if idx.reaction[i])

    wps = count / duration
    spoken = sum(max(0.0, idx.ends[i] - idx.starts[i]) for i in range(lo, hi + 1))

    longest_gap = 0.0
    for i in range(lo, hi):
        longest_gap = max(longest_gap, idx.starts[i + 1] - idx.ends[i])
    longest_gap = max(longest_gap, idx.starts[lo] - start, end - idx.ends[hi])

    boundary = 0.0
    if idx.sent_start[lo]:
        boundary += 0.5
    if idx.sent_end[hi]:
        boundary += 0.5

    return {
        "hook": min(1.0, hooks / 2.0),
        "question": min(1.0, questions / 2.0),
        "numbers": min(1.0, numbers / 4.0),
        "entities": min(1.0, entities / 3.0),
        "reaction": min(1.0, reactions / 2.0),
        "pace": _ramp(wps / idx.median_wps, 0.80, 1.40),
        "boundary": boundary,
        "density": _ramp(spoken / duration, 0.40, 0.95),
        "_mid_sentence": 0.0 if idx.sent_start[lo] else 1.0,
        "_silence": _ramp(longest_gap, 1.0, 4.0),
        "_words": float(count),
    }


def _score_from(features: dict[str, float]) -> float:
    if not features:
        return 0.0
    total = sum(WEIGHTS[key] * features.get(key, 0.0) for key in WEIGHTS)
    total -= PENALTIES["mid_sentence"] * features.get("_mid_sentence", 0.0)
    total -= PENALTIES["silence"] * features.get("_silence", 0.0)
    return _clamp(total, 0.0, 1.0)


def score_window(transcript: Transcript, start: float, end: float) -> float:
    """Score ``[start, end)`` of ``transcript`` for short-form appeal, 0..1.

    An empty, inverted or speech-free window scores ``0.0``.  See the module
    docstring for the feature weights.
    """
    if not _finite(start) or not _finite(end) or end <= start:
        return 0.0
    return _score_from(_features(_build_index(transcript), float(start), float(end)))


# --------------------------------------------------------------------------- #
# boundary snapping
# --------------------------------------------------------------------------- #

def _nearest(values: Sequence[float], target: float) -> int:
    """Index of the value in the sorted sequence closest to ``target``."""
    pos = bisect.bisect_left(values, target)
    if pos <= 0:
        return 0
    if pos >= len(values):
        return len(values) - 1
    before, after = values[pos - 1], values[pos]
    return pos - 1 if (target - before) <= (after - target) else pos


_TOL = 1e-9


def _fits(idx: _Index, i: int, j: int, min_duration: float, max_duration: float) -> bool:
    span = idx.ends[j] - idx.starts[i]
    return min_duration - _TOL <= span <= max_duration + _TOL


def _fit_from(idx: _Index, i: int, min_duration: float, max_duration: float) -> int | None:
    """Best end index for a window opening at word ``i``, or ``None`` if none fits.

    The longest window that still respects ``max_duration`` is taken, then pulled
    back to the last sentence end that keeps it at or above ``min_duration``.
    """
    j = bisect.bisect_right(idx.ends_mono, idx.starts[i] + max_duration + _TOL) - 1
    if j < i or idx.ends[j] - idx.starts[i] < min_duration - _TOL:
        return None
    for k in range(j, i, -1):
        if idx.sent_end[k] and idx.ends[k] - idx.starts[i] >= min_duration - _TOL:
            return k
    return j


def _refit(idx: _Index, i: int, j: int, min_duration: float, max_duration: float) -> tuple[int, int]:
    """Repair a window that word granularity pushed outside ``[min, max]``.

    Growing by one word can leap a long pause, so the plain grow/shrink walk can
    settle below ``min_duration`` even though a window nearby would fit.  Try the
    word starts around the anchor, nearest first, and take the first one that
    honours both bounds; if nothing does, keep what we had.
    """
    if _fits(idx, i, j, min_duration, max_duration):
        return i, j
    anchor = idx.starts[i]
    reach = max_duration + _SENTENCE_PULL
    lo = bisect.bisect_left(idx.starts, anchor - reach)
    hi = bisect.bisect_right(idx.starts, anchor + reach)
    for k in sorted(range(lo, hi), key=lambda k: (abs(idx.starts[k] - anchor), k)):
        fitted = _fit_from(idx, k, min_duration, max_duration)
        if fitted is not None:
            return k, fitted
    return i, j


def _snap_indexed(
    idx: _Index, start: float, end: float, min_duration: float, max_duration: float
) -> tuple[int, int]:
    starts, ends = idx.starts, idx.ends
    n = len(starts)
    lo_t, hi_t = starts[0], ends[-1]

    start = _clamp(start, lo_t, hi_t)
    end = _clamp(end, lo_t, hi_t)
    if end <= start:
        end = hi_t

    i = _nearest(starts, start)
    j = _nearest(idx.ends_mono, end)
    if j < i:
        j = i

    # 1. pull the opening onto a nearby sentence start -- but only when it is not
    #    already on one, or repeated snapping would creep backwards for ever.
    best = i
    if not idx.sent_start[i]:
        k = i
        while k > 0 and starts[i] - starts[k - 1] <= _SENTENCE_PULL:
            k -= 1
            if idx.sent_start[k]:
                best = k
                break
        forward = i
        while forward < j and starts[forward] - starts[i] <= _SENTENCE_PULL:
            if idx.sent_start[forward]:
                break
            forward += 1
        if forward <= j and idx.sent_start[forward] and (
            forward - i < i - best or not idx.sent_start[best]
        ):
            best = forward
    if ends[j] - starts[best] <= max_duration or starts[best] > starts[i]:
        i = best
        if j < i:
            j = i

    # 2. grow to reach the minimum duration: end first, then the start.
    while j < n - 1 and ends[j] - starts[i] < min_duration:
        j += 1
    while i > 0 and ends[j] - starts[i] < min_duration:
        i -= 1

    # 3. prefer finishing on a sentence end, as long as the maximum allows it.
    if not idx.sent_end[j]:
        k = j
        while k < n - 1 and ends[k] - starts[i] <= max_duration:
            k += 1
            if idx.sent_end[k] and ends[k] - starts[i] <= max_duration:
                j = k
                break

    # 4. enforce the maximum duration, always keeping at least one word.
    while j > i and ends[j] - starts[i] > max_duration:
        j -= 1
    while j > i and ends[j] - starts[i] > max_duration:
        i += 1

    # 5. growing or trimming can leave the opening mid-sentence; step back onto the
    #    sentence start when one is close and the maximum still allows it.  This
    #    runs last so that re-snapping an already snapped window is a no-op.
    if not idx.sent_start[i]:
        k = i
        while k > 0 and starts[i] - starts[k - 1] <= _SENTENCE_PULL:
            k -= 1
            if idx.sent_start[k]:
                if ends[j] - starts[k] <= max_duration:
                    i = k
                break

    i = max(0, min(i, n - 1))
    j = max(i, min(j, n - 1))

    # 6. a long pause can make steps 2-4 overshoot in both directions at once;
    #    look for a nearby window that honours the bounds properly.
    return _refit(idx, i, j, min_duration, max_duration)


def snap_to_speech(
    transcript: Transcript,
    start: float,
    end: float,
    min_duration: float,
    max_duration: float,
) -> tuple[float, float]:
    """Move ``[start, end)`` onto word edges without ever splitting a word.

    The boundaries are pulled toward sentence starts/ends when one is nearby,
    then the window is grown or shrunk to sit inside
    ``[min_duration, max_duration]``.  The result always contains at least one
    whole word and never reaches outside the transcript.  Three cases cannot be
    fully honoured and are documented rather than raised on: a transcript whose
    whole speech span is shorter than ``min_duration`` (the full span is
    returned), a single word longer than ``max_duration`` (that word is
    returned, because returning no words is worse), and a stretch of speech so
    broken up by pauses that no window of whole words near the request lands
    inside the bounds (the closest attempt is returned).

    A transcript whose segments carry no word timings at all (coarse ASR) is
    still usable: the segment text is spread evenly over the segment span, so
    the returned edges are interpolated rather than measured.
    """
    if not _finite(start):
        start = 0.0
    if not _finite(end):
        end = float(start) + float(max(min_duration, 0.0))
    start, end = float(start), float(end)
    min_duration = max(0.0, float(min_duration)) if _finite(min_duration) else 0.0
    max_duration = float(max_duration) if _finite(max_duration) and max_duration > 0 else max(
        min_duration, 1.0
    )
    if max_duration < min_duration:
        max_duration = min_duration

    idx = _build_index(transcript)
    if idx.empty:
        # No words at all: clamp into the media and keep a positive duration.
        limit = idx.limit if idx.limit > 0 else max(end, start + min_duration, 1.0)
        start = _clamp(start, 0.0, limit)
        end = _clamp(max(end, start + min_duration), start, limit)
        if end <= start:
            end = limit
        return (start, end)

    i, j = _snap_indexed(idx, start, end, min_duration, max_duration)
    return (idx.starts[i], idx.ends[j])


# --------------------------------------------------------------------------- #
# LLM path
# --------------------------------------------------------------------------- #

_FALLBACK_SYSTEM = (
    "You find self-contained moments in a transcript that work as standalone vertical "
    "short-form videos. You answer with JSON only."
)

_FALLBACK_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["moments"],
    "properties": {
        "moments": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "title", "hook", "reason"],
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "title": {"type": "string"},
                    "hook": {"type": "string"},
                    "reason": {"type": "string"},
                },
            },
        }
    },
}

_MOMENT_KEYS = ("moments", "clips", "highlights", "candidates", "segments", "results", "items")
_START_KEYS = ("start", "start_time", "start_seconds", "from", "begin", "t0")
_END_KEYS = ("end", "end_time", "end_seconds", "to", "finish", "t1")


class _SafeDict(dict):
    """``format_map`` helper: unknown placeholders render as an empty string."""

    def __missing__(self, key: str) -> str:  # pragma: no cover - trivial
        return ""


@dataclass(frozen=True)
class _Prompting:
    """Whatever :mod:`aiclipper.llm.prompts` currently offers for the job."""

    system: str
    schema: dict[str, Any]
    template: str | None = None
    builder: Any = None


def _prompt_parts() -> _Prompting:
    """Fetch the highlight prompt pieces, tolerating a missing llm package."""
    system, schema, template, builder = _FALLBACK_SYSTEM, _FALLBACK_SCHEMA, None, None
    try:
        from .llm import prompts as _prompts  # noqa: PLC0415 - optional, lazy on purpose
    except Exception:  # pragma: no cover - depends on a sibling module
        return _Prompting(system, schema, template, builder)
    candidate = getattr(_prompts, "HIGHLIGHT_SYSTEM", None)
    if isinstance(candidate, str) and candidate.strip():
        system = candidate
    candidate = getattr(_prompts, "HIGHLIGHT_SCHEMA", None)
    if isinstance(candidate, dict) and candidate:
        schema = candidate
    for name in ("HIGHLIGHT_PROMPT", "HIGHLIGHT_USER", "HIGHLIGHT_TEMPLATE"):
        candidate = getattr(_prompts, name, None)
        if isinstance(candidate, str) and candidate.strip():
            template = candidate
            break
    candidate = getattr(_prompts, "highlight_prompt", None)
    if callable(candidate):
        builder = candidate
    return _Prompting(system, schema, template, builder)


def _transcript_digest(transcript: Transcript, max_chars: int = 12000) -> str:
    """One ``[start-end] text`` line per segment, sampled to fit ``max_chars``.

    A three-hour recording does not fit in a prompt.  Dropping the tail would
    mean every proposed moment came from the first twenty minutes, so segments
    are thinned evenly instead and the whole recording stays represented.
    """
    lines: list[str] = []
    for seg in getattr(transcript, "segments", []) or []:
        text = (getattr(seg, "text", "") or "").strip()
        if not text or not _finite(getattr(seg, "start", None)) or not _finite(getattr(seg, "end", None)):
            continue
        lines.append(f"[{float(seg.start):.1f}-{float(seg.end):.1f}] {text}")
    total = sum(len(line) + 1 for line in lines)
    if total <= max_chars or not lines:
        return "\n".join(lines)

    keep = max(1, int(len(lines) * max_chars / total))
    stride = len(lines) / keep
    out: list[str] = []
    previous = -1
    for n in range(keep):
        i = min(len(lines) - 1, int(n * stride))
        if i <= previous:
            continue
        if i > previous + 1:
            out.append("... (segments omitted)")
        out.append(lines[i])
        previous = i
    if previous < len(lines) - 1:
        out.append("... (segments omitted)")
    return "\n".join(out)


def _build_prompt(
    transcript: Transcript, count: int, min_duration: float, max_duration: float, parts: _Prompting
) -> str:
    """The user turn: the sibling module's builder when it has one, else ours."""
    digest = _transcript_digest(transcript)
    total = float(getattr(transcript, "duration", 0.0) or 0.0)
    if parts.builder is not None:
        try:
            rendered = parts.builder(
                digest,
                count=count,
                min_duration=min_duration,
                max_duration=max_duration,
                total_duration=total or None,
            )
            if isinstance(rendered, str) and rendered.strip():
                return rendered
        except Exception:
            log.debug("highlight: prompts.highlight_prompt did not accept our arguments")
    fields = _SafeDict(
        transcript=digest,
        text=digest,
        count=count,
        n=count,
        min_duration=f"{min_duration:.1f}",
        max_duration=f"{max_duration:.1f}",
        duration=f"{total:.1f}",
        language=getattr(transcript, "language", "en"),
    )
    if parts.template:
        try:
            rendered = parts.template.format_map(fields)
            if rendered.strip():
                return rendered
        except Exception:
            log.debug("highlight: HIGHLIGHT prompt template did not format, using the built-in one")
    return (
        f"Transcript of a {fields['duration']}s video, one line per segment with "
        f"[start-end] timestamps in seconds:\n\n{fields['transcript']}\n\n"
        f"Pick the {count} strongest self-contained moments for vertical short-form video. "
        f"Each must last between {fields['min_duration']} and {fields['max_duration']} seconds, "
        f"start on a complete thought and end on a payoff. Report start and end in seconds as "
        f"numbers, plus a short title, the spoken hook line, and why it works."
    )


def _parse_time(value: Any) -> float | None:
    """Accept float seconds, numeric strings and ``MM:SS`` / ``HH:MM:SS``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if math.isfinite(float(value)) else None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    match = _TIMECODE_RE.match(text)
    if match:
        a, b, c, frac = match.groups()
        fraction = float(frac) if frac else 0.0
        if c is None:
            return int(a) * 60 + int(b) + fraction
        return int(a) * 3600 + int(b) * 60 + int(c) + fraction
    try:
        parsed = float(text.rstrip("sS ").strip())
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _as_moments(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, dict):
        for key in _MOMENT_KEYS:
            value = data.get(key)
            if isinstance(value, list):
                return [m for m in value if isinstance(m, dict)]
        for value in data.values():
            if isinstance(value, list) and any(isinstance(m, dict) for m in value):
                return [m for m in value if isinstance(m, dict)]
        if any(k in data for k in _START_KEYS):
            return [data]
        return []
    if isinstance(data, list):
        return [m for m in data if isinstance(m, dict)]
    return []


def _first_str(moment: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = moment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _llm_moments(
    transcript: Transcript,
    idx: _Index,
    *,
    count: int,
    min_duration: float,
    max_duration: float,
    settings: Settings,
    provider: Any,
) -> list[ClipCandidate]:
    """Ask the language model for moments.  Never raises; returns [] on trouble."""
    try:
        if provider is None:
            from .llm import get_provider  # noqa: PLC0415 - optional, lazy on purpose

            provider = get_provider(getattr(settings, "llm_provider", None) or None, settings=settings)
        if provider is None:
            return []
        name = str(getattr(provider, "name", "") or "").lower()
        if any(marker in name for marker in _SYNTHETIC_PROVIDERS):
            return []
        checker = getattr(provider, "available", None)
        if callable(checker) and not checker():
            return []
        complete_json = getattr(provider, "complete_json", None)
        if not callable(complete_json):
            return []
        parts = _prompt_parts()
        prompt = _build_prompt(transcript, count, min_duration, max_duration, parts)
        data = complete_json(prompt, parts.schema, system=parts.system)
    except Exception as exc:
        log.debug("highlight: LLM path unavailable (%s), using heuristics", exc)
        return []

    limit = max(float(getattr(transcript, "duration", 0.0) or 0.0), idx.limit)
    out: list[ClipCandidate] = []
    try:
        moments = _as_moments(data)
    except Exception:  # pragma: no cover - defensive
        return []
    for moment in moments:
        start = None
        end = None
        for key in _START_KEYS:
            if key in moment:
                start = _parse_time(moment.get(key))
                if start is not None:
                    break
        for key in _END_KEYS:
            if key in moment:
                end = _parse_time(moment.get(key))
                if end is not None:
                    break
        if start is None or end is None:
            continue
        if end <= start or start < 0.0:
            continue
        if limit > 0.0 and (start >= limit or end > limit + 0.5):
            continue
        end = min(end, limit) if limit > 0.0 else end
        if end <= start:
            continue
        tags = moment.get("tags")
        clean_tags = (
            [str(t) for t in tags if isinstance(t, (str, int, float))] if isinstance(tags, list) else []
        )
        out.append(
            ClipCandidate(
                start=start,
                end=end,
                title=_first_str(moment, "title", "name", "headline"),
                hook=_first_str(moment, "hook", "quote", "opening"),
                reason=_first_str(moment, "reason", "why", "rationale", "explanation"),
                tags=clean_tags,
            )
        )
    return out


# --------------------------------------------------------------------------- #
# heuristic path
# --------------------------------------------------------------------------- #

def _window_lengths(min_duration: float, max_duration: float) -> list[float]:
    span = max_duration - min_duration
    if span <= 0.5:
        return [max_duration]
    return sorted({min_duration, min_duration + span * 0.35, min_duration + span * 0.7, max_duration})


def _heuristic_windows(
    idx: _Index, *, count: int, min_duration: float, max_duration: float, seed: int
) -> list[tuple[float, float, float]]:
    """Sliding-window scan -> a shortlist of ``(score, start, end)`` triples."""
    if idx.empty:
        return []
    first, last = idx.first, idx.last
    total = last - first
    if total <= 0.0:
        return []

    step = _clamp(min_duration / 5.0, 0.5, 4.0)
    phase = random.Random(seed).random() * step
    grid = {first}
    t = first + phase
    while t < last:
        grid.add(t)
        t += step
    # Every sentence start is worth trying as an opening too.
    grid.update(idx.starts[i] for i in range(len(idx.starts)) if idx.sent_start[i])

    scored: list[tuple[float, float, float]] = []
    for start in sorted(grid):
        for length in _window_lengths(min_duration, max_duration):
            end = min(start + length, last)
            if end - start < min(min_duration, total) - 1e-6:
                continue
            features = _features(idx, start, end)
            if not features:
                continue
            scored.append((_score_from(features), start, end))
    scored.sort(key=lambda item: (-item[0], item[1]))

    # Keep the best windows outright, but also a greedy non-overlapping spread so
    # that `count` distinct moments survive de-duplication even when one region
    # of the transcript dominates the top of the ranking.
    shortlist: list[tuple[float, float, float]] = list(scored[: max(count * 6, 24)])
    spread: list[tuple[float, float, float]] = []
    for item in scored:
        if all(item[2] <= other[1] or item[1] >= other[2] for other in spread):
            spread.append(item)
            if len(spread) >= count * 4:
                break
    for item in spread:
        if item not in shortlist:
            shortlist.append(item)
    return shortlist


# --------------------------------------------------------------------------- #
# public entry point
# --------------------------------------------------------------------------- #

def _describe(features: dict[str, float]) -> tuple[str, list[str]]:
    """Human-readable reason plus tags for the features that actually fired."""
    labels = {
        "hook": "hook phrase",
        "question": "poses a question",
        "numbers": "concrete numbers",
        "entities": "named people or places",
        "reaction": "audible reaction",
        "pace": "fast delivery",
        "boundary": "clean sentence boundaries",
        "density": "dense speech",
    }
    fired = [(WEIGHTS[k] * features.get(k, 0.0), k) for k in WEIGHTS if features.get(k, 0.0) > 0.25]
    fired.sort(reverse=True)
    tags = [key for _, key in fired[:4]]
    reason = ", ".join(labels[key] for key in tags) or "best available stretch of speech"
    return reason.capitalize(), tags


def _title_for(idx: _Index, start: float, end: float) -> tuple[str, str]:
    lo, hi = _span(idx, start, end)
    if lo > hi:
        return ("", "")
    words = [idx.words[i].text.strip() for i in range(lo, min(hi, lo + 11) + 1)]
    hook = " ".join(words).strip()
    title = " ".join(words[:8]).strip().rstrip(",;:")
    return (title, hook)


def _adopt(kept: ClipCandidate, other: ClipCandidate) -> None:
    """Copy the model's description onto an overlapping heuristic window."""
    kept.title = other.title or kept.title
    kept.hook = other.hook or kept.hook
    kept.reason = other.reason or kept.reason
    if other.tags:
        kept.tags = list(other.tags)


def _finalise(
    idx: _Index,
    raw: Sequence[tuple[ClipCandidate, float]],
    *,
    count: int,
    min_duration: float,
    max_duration: float,
    allow_short: bool = False,
) -> list[ClipCandidate]:
    """Snap, rescore, drop overlaps keeping the higher score, rank and trim.

    Windows that end up shorter than ``min_duration`` are dropped unless
    ``allow_short`` is set, which is the last-resort pass for a transcript that
    simply cannot host a clip of the requested length.
    """
    built: list[tuple[ClipCandidate, bool]] = []
    for candidate, bonus in raw:
        start, end = candidate.start, candidate.end
        if idx.empty:
            continue
        i, j = _snap_indexed(idx, start, end, min_duration, max_duration)
        start, end = idx.starts[i], idx.ends[j]
        if end <= start:
            continue
        features = _features(idx, start, end)
        if not features or features.get("_words", 0.0) < 1.0:
            continue
        span = end - start
        # A single word longer than max_duration is the only allowed overshoot.
        if span > max_duration + 1e-6 and features["_words"] > 1.0:
            continue
        if span < min_duration - 1e-6 and not allow_short:
            continue
        score = _clamp(_score_from(features) + bonus, 0.0, 1.0)
        reason, tags = _describe(features)
        title, hook = _title_for(idx, start, end)
        built.append(
            (
                ClipCandidate(
                    start=start,
                    end=end,
                    title=candidate.title or title,
                    hook=candidate.hook or hook,
                    reason=candidate.reason or reason,
                    score=round(score, 6),
                    tags=list(candidate.tags) or tags,
                ),
                bonus > 0.0,
            )
        )

    built.sort(key=lambda item: (-item[0].score, item[0].start))
    kept: list[list[Any]] = []
    for candidate, from_llm in built:
        clash = next((row for row in kept if candidate.overlaps(row[0], tolerance=0.05)), None)
        if clash is not None:
            # The better-scoring window wins, but it inherits the model's
            # description when it displaced an overlapping LLM proposal.
            if from_llm and not clash[1]:
                _adopt(clash[0], candidate)
                clash[1] = True
            continue
        if len(kept) >= count:
            continue
        kept.append([candidate, from_llm])
    rows = [row[0] for row in kept]
    rows.sort(key=lambda c: (-c.score, c.start))
    return rows


def select(
    transcript: Transcript,
    *,
    count: int = 3,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    settings: Settings | None = None,
    provider: Any = None,
) -> list[ClipCandidate]:
    """Pick up to ``count`` non-overlapping clips from ``transcript``, best first.

    The language-model path runs first when a provider is available; anything it
    returns is validated, snapped and rescored alongside heuristic windows, so a
    broken or unavailable provider simply degrades to the offline heuristic.
    """
    s = settings or get_settings()
    count = max(0, int(count))
    if count == 0:
        return []
    min_duration = max(0.1, float(min_duration))
    max_duration = max(min_duration, float(max_duration))

    idx = _build_index(transcript)
    if idx.empty or idx.last <= idx.first:
        return []

    raw: list[tuple[ClipCandidate, float]] = []
    for candidate in _llm_moments(
        transcript,
        idx,
        count=count,
        min_duration=min_duration,
        max_duration=max_duration,
        settings=s,
        provider=provider,
    ):
        raw.append((candidate, _LLM_BONUS))

    seed = int(getattr(s, "seed", 0) or 0)
    for score, start, end in _heuristic_windows(
        idx, count=count, min_duration=min_duration, max_duration=max_duration, seed=seed
    ):
        raw.append((ClipCandidate(start=start, end=end, score=score), 0.0))

    clips = _finalise(idx, raw, count=count, min_duration=min_duration, max_duration=max_duration)
    if clips:
        return clips
    # Nothing reached ``min_duration`` -- a transcript with barely any speech, or
    # speech so broken up that no whole-word window fits.  Returning the best
    # short moment beats returning nothing at all.
    return _finalise(
        idx, raw, count=count, min_duration=min_duration, max_duration=max_duration, allow_short=True
    )
