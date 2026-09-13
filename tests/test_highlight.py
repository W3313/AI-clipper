"""Tests for :mod:`aiclipper.highlight`.

Everything here is synthetic: transcripts are built word by word so the exact
word edges, sentence boundaries and window scores are known up front.  No
network, no ffmpeg, no optional dependency is touched.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from aiclipper.highlight import score_window, select, snap_to_speech
from aiclipper.models import ClipCandidate, Segment, Transcript, Word

EPS = 1e-6


# --------------------------------------------------------------------------- #
# builders
# --------------------------------------------------------------------------- #

def build(sentences: list[str], *, per_word: float = 0.32, gap: float = 0.25) -> tuple[Transcript, list]:
    """Build a transcript from sentences; also return each sentence's span."""
    segments: list[Segment] = []
    spans: list[tuple[float, float]] = []
    t = 0.0
    for sentence in sentences:
        words: list[Word] = []
        for token in sentence.split():
            words.append(Word(token, round(t, 6), round(t + per_word, 6)))
            t = round(t + per_word, 6)
        segments.append(Segment(sentence, words[0].start, words[-1].end, words))
        spans.append((words[0].start, words[-1].end))
        t = round(t + gap, 6)
    return Transcript(segments=segments, language="en", duration=round(t, 6)), spans


BORING = "we kept the same process running quietly in the background all week."
HOOKY = "here is the part nobody tells you about shipping something people actually want."
HOOKY2 = "what happened next completely changed how we thought about the whole problem."
QUESTION = "why does this keep working when every reasonable model says it should not?"
NUMBERS = "revenue moved from 12 thousand to 240 thousand in 90 days across 3 markets."


@pytest.fixture
def mixed() -> tuple[Transcript, list]:
    """Six boring sentences, three hook-laden ones, three numbers-heavy ones."""
    sentences = [BORING] * 6 + [HOOKY, HOOKY2, QUESTION] + [NUMBERS] * 3
    return build(sentences)


def boring_span(spans: list) -> tuple[float, float]:
    return (spans[0][0], spans[5][1])


def hook_span(spans: list) -> tuple[float, float]:
    return (spans[6][0], spans[8][1])


def numbers_span(spans: list) -> tuple[float, float]:
    return (spans[9][0], spans[11][1])


def splits_a_word(transcript: Transcript, start: float, end: float) -> bool:
    for w in transcript.words:
        if w.start + EPS < start < w.end - EPS or w.start + EPS < end < w.end - EPS:
            return True
    return False


# --------------------------------------------------------------------------- #
# fake LLM providers
# --------------------------------------------------------------------------- #

class FakeProvider:
    """Stands in for an :mod:`aiclipper.llm` provider."""

    name = "fake"

    def __init__(self, payload: Any = None, *, is_available: bool = True, raises: Exception | None = None):
        self.payload = payload
        self._available = is_available
        self.raises = raises
        self.calls: list[tuple[str, Any, str]] = []

    def available(self) -> bool:
        if isinstance(self._available, Exception):
            raise self._available
        return self._available

    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None) -> str:
        return ""

    def complete_json(self, prompt: str, schema: Any, *, system: str = "", max_tokens: int | None = None):
        self.calls.append((prompt, schema, system))
        if self.raises is not None:
            raise self.raises
        return self.payload


# --------------------------------------------------------------------------- #
# score_window
# --------------------------------------------------------------------------- #

def test_hook_window_outscores_boring_window(mixed):
    transcript, spans = mixed
    boring = score_window(transcript, *boring_span(spans))
    hooky = score_window(transcript, *hook_span(spans))
    assert hooky > boring
    assert hooky - boring > 0.1


def test_numbers_window_outscores_boring_window(mixed):
    transcript, spans = mixed
    assert score_window(transcript, *numbers_span(spans)) > score_window(transcript, *boring_span(spans))


def test_scores_stay_normalised(mixed):
    transcript, spans = mixed
    for start, end in spans:
        value = score_window(transcript, start, end)
        assert 0.0 <= value <= 1.0
        assert math.isfinite(value)


def test_degenerate_windows_score_zero(mixed):
    transcript, _ = mixed
    assert score_window(transcript, 5.0, 5.0) == 0.0
    assert score_window(transcript, 9.0, 4.0) == 0.0
    assert score_window(transcript, 10_000.0, 10_060.0) == 0.0
    assert score_window(transcript, float("nan"), 10.0) == 0.0
    assert score_window(Transcript(), 0.0, 10.0) == 0.0


def test_question_and_hook_features_are_additive():
    plain, plain_spans = build(["the model keeps working the way it did before."])
    hooked, hook_spans = build(["here is why the model keeps working the way it did before?"])
    assert score_window(hooked, *hook_spans[0]) > score_window(plain, *plain_spans[0])


def test_long_internal_pause_lowers_the_score():
    tight, _ = build([NUMBERS, NUMBERS])
    words = tight.words
    split = len(words) // 2
    gapped_words = [Word(w.text, w.start, w.end) for w in words[:split]]
    gapped_words += [Word(w.text, w.start + 3.0, w.end + 3.0) for w in words[split:]]
    gapped = Transcript.from_words(gapped_words, max_gap=5.0)
    tight_score = score_window(tight, 0.0, tight.words[-1].end)
    gapped_score = score_window(gapped, 0.0, gapped_words[-1].end)
    assert gapped_score < tight_score


# --------------------------------------------------------------------------- #
# snap_to_speech
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("request_start", [0.0, 1.7, 4.3, 9.9, 17.25, 30.5, 44.0])
def test_snapping_never_splits_a_word(mixed, request_start):
    transcript, _ = mixed
    start, end = snap_to_speech(transcript, request_start, request_start + 11.0, 8.0, 20.0)
    assert not splits_a_word(transcript, start, end)
    assert any(w.start >= start - EPS and w.end <= end + EPS for w in transcript.words)


@pytest.mark.parametrize("request_start", [0.0, 3.3, 12.0, 26.4, 40.0])
@pytest.mark.parametrize("bounds", [(8.0, 20.0), (12.0, 15.0), (5.0, 45.0)])
def test_snapping_respects_min_and_max(mixed, request_start, bounds):
    transcript, _ = mixed
    min_d, max_d = bounds
    start, end = snap_to_speech(transcript, request_start, request_start + 6.0, min_d, max_d)
    assert end - start >= min_d - EPS
    assert end - start <= max_d + EPS
    assert start >= transcript.words[0].start - EPS
    assert end <= transcript.words[-1].end + EPS
    assert not splits_a_word(transcript, start, end)


def test_snapping_lands_on_word_edges(mixed):
    transcript, _ = mixed
    starts = {w.start for w in transcript.words}
    ends = {w.end for w in transcript.words}
    start, end = snap_to_speech(transcript, 13.7, 21.1, 6.0, 25.0)
    assert start in starts
    assert end in ends


def test_snapping_grows_a_too_short_request(mixed):
    transcript, spans = mixed
    start, end = snap_to_speech(transcript, spans[2][0], spans[2][0] + 1.0, 14.0, 30.0)
    assert end - start >= 14.0 - EPS


def test_snapping_shrinks_a_too_long_request(mixed):
    transcript, _ = mixed
    start, end = snap_to_speech(transcript, 0.0, transcript.duration + 40.0, 5.0, 12.0)
    assert end - start <= 12.0 + EPS
    assert end - start >= 5.0 - EPS


def test_snapping_prefers_sentence_starts(mixed):
    transcript, spans = mixed
    # Ask just *after* a sentence opens; the boundary should be pulled onto it.
    wanted = spans[6][0]
    start, _ = snap_to_speech(transcript, wanted + 0.45, wanted + 12.0, 8.0, 20.0)
    assert start == pytest.approx(wanted, abs=0.01)


def test_snapping_clamps_inside_the_transcript(mixed):
    transcript, _ = mixed
    start, end = snap_to_speech(transcript, -500.0, 9_000.0, 5.0, 20.0)
    assert start >= transcript.words[0].start - EPS
    assert end <= transcript.words[-1].end + EPS
    assert end > start


def test_snapping_a_single_word_transcript_keeps_the_word():
    transcript = Transcript.from_words([Word("hello", 1.0, 1.6)])
    start, end = snap_to_speech(transcript, 0.0, 40.0, 15.0, 60.0)
    assert (start, end) == (1.0, 1.6)


def test_snapping_an_empty_transcript_does_not_crash():
    start, end = snap_to_speech(Transcript(duration=30.0), 5.0, 9.0, 15.0, 60.0)
    assert end > start
    assert 0.0 <= start <= 30.0
    assert end <= 30.0


def test_snapping_tolerates_garbage_inputs(mixed):
    transcript, _ = mixed
    start, end = snap_to_speech(transcript, float("nan"), float("inf"), 10.0, 25.0)
    assert end > start
    assert not splits_a_word(transcript, start, end)


# --------------------------------------------------------------------------- #
# select -- heuristic path
# --------------------------------------------------------------------------- #

def test_select_honours_count(mixed):
    transcript, _ = mixed
    for count in (1, 2, 3):
        clips = select(transcript, count=count, min_duration=8.0, max_duration=18.0)
        assert len(clips) == count


def test_select_returns_nothing_for_count_zero(mixed):
    transcript, _ = mixed
    assert select(transcript, count=0) == []


def test_select_ranks_by_score_and_never_overlaps(mixed):
    transcript, _ = mixed
    clips = select(transcript, count=3, min_duration=8.0, max_duration=18.0)
    assert clips
    assert [c.score for c in clips] == sorted((c.score for c in clips), reverse=True)
    for a, b in zip(clips, clips[1:], strict=False):
        assert not a.overlaps(b, tolerance=0.05)
    for a in clips:
        for b in clips:
            if a is not b:
                assert not (a.start < b.end - 0.05 and b.start < a.end - 0.05)


def test_select_windows_are_snapped_and_within_bounds(mixed):
    transcript, _ = mixed
    clips = select(transcript, count=3, min_duration=9.0, max_duration=20.0)
    for clip in clips:
        assert not splits_a_word(transcript, clip.start, clip.end)
        assert clip.duration <= 20.0 + EPS
        assert clip.start >= transcript.words[0].start - EPS
        assert clip.end <= transcript.words[-1].end + EPS
        assert transcript.slice(clip.start, clip.end).words


def test_select_prefers_the_hook_laden_stretch(mixed):
    transcript, spans = mixed
    best = select(transcript, count=1, min_duration=8.0, max_duration=20.0)[0]
    h_start, h_end = hook_span(spans)
    overlap = min(best.end, h_end) - max(best.start, h_start)
    assert overlap > 0.5 * (h_end - h_start)
    assert best.score > score_window(transcript, *boring_span(spans))


def test_select_fills_in_metadata(mixed):
    transcript, _ = mixed
    clip = select(transcript, count=1, min_duration=8.0, max_duration=20.0)[0]
    assert clip.title.strip()
    assert clip.hook.strip()
    assert clip.reason.strip()
    assert 0.0 <= clip.score <= 1.0


def test_select_is_deterministic(mixed):
    transcript, _ = mixed
    first = select(transcript, count=3, min_duration=8.0, max_duration=18.0)
    second = select(transcript, count=3, min_duration=8.0, max_duration=18.0)
    assert [(c.start, c.end, c.score) for c in first] == [(c.start, c.end, c.score) for c in second]


def test_select_on_empty_and_tiny_transcripts():
    assert select(Transcript()) == []
    assert select(Transcript(segments=[Segment("", 0.0, 0.0, [])], duration=5.0)) == []
    single = Transcript.from_words([Word("hello", 0.5, 1.1)])
    clips = select(single, count=3, min_duration=15.0, max_duration=60.0)
    assert len(clips) <= 1
    for clip in clips:
        assert (clip.start, clip.end) == (0.5, 1.1)


def test_select_handles_a_transcript_shorter_than_min_duration():
    transcript, _ = build([BORING])
    clips = select(transcript, count=2, min_duration=30.0, max_duration=60.0)
    assert len(clips) <= 1
    for clip in clips:
        assert clip.duration <= transcript.words[-1].end + EPS


# --------------------------------------------------------------------------- #
# select -- LLM path
# --------------------------------------------------------------------------- #

def test_llm_moments_are_used_and_timestamps_parsed(mixed):
    transcript, spans = mixed
    h_start, _ = hook_span(spans)
    mm_ss = f"{int(h_start) // 60}:{int(h_start) % 60:02d}"
    provider = FakeProvider(
        {
            "moments": [
                {"start": mm_ss, "end": h_start + 12.0, "title": "Nobody tells you", "hook": "here is",
                 "reason": "payoff", "tags": ["hook"]},
                {"start": str(spans[9][0]), "end": spans[11][1], "title": "The numbers"},
            ]
        }
    )
    clips = select(transcript, count=3, min_duration=8.0, max_duration=20.0, provider=provider)
    assert provider.calls, "the provider should have been asked"
    titles = {c.title for c in clips}
    assert "Nobody tells you" in titles
    picked = next(c for c in clips if c.title == "Nobody tells you")
    assert picked.start == pytest.approx(h_start, abs=1.0)
    assert picked.tags == ["hook"]
    assert not splits_a_word(transcript, picked.start, picked.end)


MALFORMED_PAYLOAD = {
    "moments": [
        {"start": "not a timestamp", "end": 12.0, "title": "BOGUS-1"},
        {"start": 30.0, "end": 10.0, "title": "BOGUS-2"},          # inverted
        {"start": None, "end": None, "title": "BOGUS-3"},
        {"start": -12.0, "end": 3.0, "title": "BOGUS-4"},          # negative
        {"start": 9_000.0, "end": 9_060.0, "title": "BOGUS-5"},    # past the media
        {"start": 5.0, "end": 99_999.0, "title": "BOGUS-6"},       # end past the media
        {"start": float("nan"), "end": 20.0, "title": "BOGUS-7"},
        {"start": float("inf"), "end": float("inf"), "title": "BOGUS-8"},
        {"start": True, "end": False, "title": "BOGUS-9"},
        {"start": 4.0, "end": 4.0, "title": "BOGUS-10"},           # zero length
        {"title": "BOGUS-11"},                                     # no timestamps at all
        "a bare string",
        None,
        42,
        [],
    ]
}


def test_malformed_llm_output_is_discarded_without_raising(mixed):
    transcript, _ = mixed
    provider = FakeProvider(MALFORMED_PAYLOAD)
    clips = select(transcript, count=3, min_duration=8.0, max_duration=18.0, provider=provider)
    assert clips, "the heuristic path must still produce clips"
    assert not any(c.title.startswith("BOGUS") for c in clips)
    for clip in clips:
        assert clip.start >= 0.0
        assert clip.end <= transcript.words[-1].end + EPS
        assert clip.end > clip.start


@pytest.mark.parametrize(
    "payload",
    [None, [], {}, {"moments": None}, {"moments": "nope"}, "not json at all", 17, {"moments": [{}]}],
)
def test_odd_llm_payload_shapes_fall_back_to_heuristics(mixed, payload):
    transcript, _ = mixed
    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0, provider=FakeProvider(payload))
    heuristic = select(transcript, count=2, min_duration=8.0, max_duration=18.0, provider=None)
    assert [(c.start, c.end) for c in clips] == [(c.start, c.end) for c in heuristic]


def test_provider_that_raises_falls_back(mixed):
    transcript, _ = mixed
    provider = FakeProvider(raises=RuntimeError("api on fire"))
    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0, provider=provider)
    assert len(clips) == 2


def test_unavailable_provider_is_never_called(mixed):
    transcript, _ = mixed
    provider = FakeProvider({"moments": []}, is_available=False)
    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0, provider=provider)
    assert provider.calls == []
    assert len(clips) == 2


def test_provider_whose_availability_check_explodes_falls_back(mixed):
    transcript, _ = mixed
    provider = FakeProvider({"moments": []}, is_available=ValueError("broken"))
    assert len(select(transcript, count=2, min_duration=8.0, max_duration=18.0, provider=provider)) == 2


def test_provider_without_the_expected_interface_falls_back(mixed):
    transcript, _ = mixed

    class Useless:
        name = "useless"

    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0, provider=Useless())
    assert len(clips) == 2


def test_llm_candidates_are_deduplicated_keeping_the_higher_score(mixed):
    transcript, spans = mixed
    h_start, h_end = hook_span(spans)
    b_start, b_end = boring_span(spans)
    provider = FakeProvider(
        {
            "moments": [
                # Three overlapping views of the same hook-laden moment plus one
                # dull window that does not overlap anything.
                {"start": h_start, "end": h_start + 12.0, "title": "A"},
                {"start": h_start + 1.0, "end": h_start + 13.0, "title": "B"},
                {"start": h_start + 2.0, "end": h_start + 14.0, "title": "C"},
                {"start": b_start, "end": b_start + 12.0, "title": "D"},
            ]
        }
    )
    clips = select(transcript, count=4, min_duration=8.0, max_duration=18.0, provider=provider)
    from_llm = [c for c in clips if c.title in {"A", "B", "C"}]
    assert len(from_llm) == 1, "overlapping proposals must collapse to one"
    scores = {
        title: score_window(transcript, *snap_to_speech(transcript, s, e, 8.0, 18.0))
        for title, (s, e) in {
            "A": (h_start, h_start + 12.0),
            "B": (h_start + 1.0, h_start + 13.0),
            "C": (h_start + 2.0, h_start + 14.0),
        }.items()
    }
    # Whatever survived must be at least as good as the best of the three.
    assert from_llm[0].score >= max(scores.values()) - 1e-6
    assert b_end > b_start and h_end > h_start  # fixture sanity, not a claim about select
    # The dull, non-overlapping proposal is a separate moment and survives too.
    assert any(c.title == "D" for c in clips)


def test_the_prompt_carries_the_transcript_and_the_bounds(mixed):
    transcript, _ = mixed
    provider = FakeProvider({"moments": []})
    select(transcript, count=2, min_duration=11.0, max_duration=19.0, provider=provider)
    prompt, schema, system = provider.calls[0]
    assert "nobody tells you" in prompt          # the transcript itself is in the prompt
    assert "11" in prompt and "19" in prompt     # ... and so are the duration bounds
    assert "2" in prompt                         # ... and the requested count
    assert isinstance(schema, dict) and schema
    assert isinstance(system, str) and system


def test_the_prompt_uses_the_shipped_llm_prompt_module(mixed):
    """Interop: the sibling module owns the system prompt, schema and user turn."""
    prompts = pytest.importorskip("aiclipper.llm.prompts")
    transcript, _ = mixed
    provider = FakeProvider({"clips": []})
    select(transcript, count=2, min_duration=11.0, max_duration=19.0, provider=provider)
    prompt, schema, system = provider.calls[0]
    assert system == prompts.HIGHLIGHT_SYSTEM
    assert schema == prompts.HIGHLIGHT_SCHEMA
    assert prompt == prompts.highlight_prompt(
        prompt.split("\n\n", 1)[1].strip(), count=2, min_duration=11.0, max_duration=19.0,
        total_duration=transcript.duration,
    )


def test_a_broken_prompt_module_falls_back_to_the_built_in_prompt(mixed, monkeypatch):
    """A renamed or reshaped prompts module must not break selection."""
    from aiclipper.llm import prompts as real

    def explode(*args: Any, **kwargs: Any) -> str:
        raise TypeError("signature changed")

    monkeypatch.setattr(real, "highlight_prompt", explode)
    monkeypatch.delattr(real, "HIGHLIGHT_SYSTEM")
    transcript, _ = mixed
    provider = FakeProvider({"moments": []})
    clips = select(transcript, count=2, min_duration=11.0, max_duration=19.0, provider=provider)
    prompt, schema, system = provider.calls[0]
    assert "nobody tells you" in prompt
    assert system.strip()
    assert len(clips) == 2


def test_candidates_are_clip_candidates(mixed):
    transcript, _ = mixed
    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0)
    assert all(isinstance(c, ClipCandidate) for c in clips)


def test_offline_heuristic_provider_is_not_trusted_for_timestamps(mixed):
    """The offline provider invents timestamps from the schema; ignore them."""
    transcript, _ = mixed
    fabricated = FakeProvider(
        {"clips": [{"start": 1.0, "end": 20.0, "title": "INVENTED", "hook": "x", "reason": "y"}]}
    )
    fabricated.name = "heuristic"
    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0, provider=fabricated)
    assert fabricated.calls == []
    assert not any(c.title == "INVENTED" for c in clips)
    assert len(clips) == 2


def test_clips_key_is_understood(mixed):
    """The shipped highlight schema names the array "clips", not "moments"."""
    transcript, spans = mixed
    start, _ = numbers_span(spans)
    provider = FakeProvider({"clips": [{"start": start, "end": start + 11.0, "title": "Numbers"}]})
    clips = select(transcript, count=3, min_duration=8.0, max_duration=18.0, provider=provider)
    assert any(c.title == "Numbers" for c in clips)


# --------------------------------------------------------------------------- #
# regressions: duration bounds around pauses, and snapping stability
# --------------------------------------------------------------------------- #

def paused() -> Transcript:
    """20 words, a 12s silence (applause, music, a slide change), then 40 more."""
    words: list[Word] = []
    t = 0.0
    for i in range(20):
        words.append(Word(f"word{i}.", round(t, 3), round(t + 0.4, 3)))
        t = round(t + 0.45, 3)
    t += 12.0
    for i in range(40):
        words.append(Word(f"later{i}.", round(t, 3), round(t + 0.4, 3)))
        t = round(t + 0.45, 3)
    return Transcript.from_words(words, max_gap=5.0)


@pytest.mark.parametrize("request_start", [0.0, 1.0, 3.0, 6.0, 8.5])
def test_snapping_reaches_min_duration_across_a_long_pause(request_start):
    """Growing by one word can leap a 12s pause; the window must still fit."""
    transcript = paused()
    start, end = snap_to_speech(transcript, request_start, request_start + 16.0, 15.0, 20.0)
    assert end - start >= 15.0 - EPS
    assert end - start <= 20.0 + EPS
    assert not splits_a_word(transcript, start, end)


def test_select_never_returns_a_clip_shorter_than_min_duration():
    transcript = paused()
    clips = select(transcript, count=3, min_duration=15.0, max_duration=20.0)
    assert clips
    for clip in clips:
        assert clip.duration >= 15.0 - EPS
        assert clip.duration <= 20.0 + EPS


def test_select_still_returns_something_when_min_duration_is_unreachable():
    """A transcript far shorter than ``min_duration`` still yields one clip."""
    transcript, _ = build([BORING])
    clips = select(transcript, count=2, min_duration=120.0, max_duration=180.0)
    assert len(clips) == 1
    assert clips[0].start == transcript.words[0].start
    assert clips[0].end == transcript.words[-1].end


@pytest.mark.parametrize("request_start", [0.0, 2.2, 9.9, 21.0, 33.0])
@pytest.mark.parametrize("bounds", [(8.0, 20.0), (12.0, 15.0)])
def test_snapping_an_already_snapped_window_is_stable(mixed, request_start, bounds):
    """Re-snapping must not creep: pipelines snap, pad, and snap again."""
    transcript, _ = mixed
    min_d, max_d = bounds
    first = snap_to_speech(transcript, request_start, request_start + 10.0, min_d, max_d)
    assert snap_to_speech(transcript, *first, min_d, max_d) == first


def test_snapping_does_not_drag_the_start_back_past_a_sentence_start(mixed):
    """A request that already sits on a sentence start stays exactly there."""
    transcript, spans = mixed
    for index in (1, 6, 9):
        wanted = spans[index][0]
        start, _ = snap_to_speech(transcript, wanted, wanted + 10.0, 8.0, 20.0)
        assert start == pytest.approx(wanted, abs=EPS)


def test_snapping_tolerates_inverted_duration_bounds(mixed):
    transcript, _ = mixed
    start, end = snap_to_speech(transcript, 5.0, 12.0, 20.0, 5.0)
    assert end > start
    assert not splits_a_word(transcript, start, end)


# --------------------------------------------------------------------------- #
# transcripts that are not tidy
# --------------------------------------------------------------------------- #

def coarse() -> Transcript:
    """Segment-level ASR: text and timings, but no word timestamps at all."""
    return Transcript(
        segments=[
            Segment(HOOKY, 0.0, 4.0),
            Segment(NUMBERS, 4.5, 9.0),
            Segment(BORING, 9.5, 14.0),
            Segment(QUESTION, 14.5, 19.0),
        ],
        duration=20.0,
    )


def test_coarse_segment_only_transcript_still_selects():
    """``Segment.words`` may be empty (models.py); transcribe.py really emits that."""
    transcript = coarse()
    assert transcript.words == []
    clips = select(transcript, count=2, min_duration=5.0, max_duration=10.0)
    assert len(clips) == 2
    for clip in clips:
        assert clip.end > clip.start
        assert 5.0 - EPS <= clip.duration <= 10.0 + EPS
        assert 0.0 <= clip.start
        assert clip.end <= 19.0 + EPS
        assert clip.title.strip()
    assert not clips[0].overlaps(clips[1], tolerance=0.05)


def test_coarse_transcript_scores_its_hook_segment_highest():
    transcript = coarse()
    assert score_window(transcript, 0.0, 4.0) > score_window(transcript, 9.5, 14.0)


def test_segments_without_text_or_timings_are_ignored():
    transcript = Transcript(
        segments=[Segment("", 0.0, 5.0), Segment("hello there", 9.0, 9.0), Segment("  ", 1.0, 2.0)],
        duration=10.0,
    )
    assert select(transcript, count=2) == []
    assert score_window(transcript, 0.0, 10.0) == 0.0


def test_out_of_order_and_nested_word_timings_do_not_break_selection():
    """Real ASR occasionally hands back unsorted or overlapping word spans."""
    early = [Word(f"first{i}.", 0.5 * i, 0.5 * i + 0.45) for i in range(20)]
    late = [Word(f"second{i}.", 20.0 + 0.5 * i, 20.0 + 0.5 * i + 0.45) for i in range(20)]
    late[3] = Word("swallowed", late[3].start, late[9].end)  # nested inside its neighbours
    transcript = Transcript(
        segments=[
            Segment(" ".join(w.text for w in late), late[0].start, late[-1].end, late),
            Segment(" ".join(w.text for w in early), early[0].start, early[-1].end, early),
        ],
        duration=32.0,
    )
    clips = select(transcript, count=2, min_duration=5.0, max_duration=12.0)
    assert clips
    starts = {w.start for w in transcript.words}
    ends = {w.end for w in transcript.words}
    for clip in clips:
        assert clip.start in starts
        assert clip.end in ends
        assert clip.end > clip.start
        assert clip.duration <= 12.0 + EPS


def test_unpunctuated_transcript_still_selects():
    """Whisper without punctuation: boundaries then come only from pauses."""
    words = [Word(f"token{i}", 0.4 * i, 0.4 * i + 0.35) for i in range(200)]
    transcript = Transcript.from_words(words, max_gap=0.6)
    clips = select(transcript, count=3, min_duration=10.0, max_duration=20.0)
    assert len(clips) == 3
    for clip in clips:
        assert 10.0 - EPS <= clip.duration <= 20.0 + EPS
        assert not splits_a_word(transcript, clip.start, clip.end)


# --------------------------------------------------------------------------- #
# degradation
# --------------------------------------------------------------------------- #

def test_select_works_with_the_llm_package_unimportable(mixed, monkeypatch):
    """`aiclipper.llm` is optional to this module; selection must survive without it."""
    import sys

    for name in [n for n in list(sys.modules) if n.startswith("aiclipper.llm")]:
        monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "aiclipper.llm", None)
    transcript, _ = mixed
    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0)
    assert len(clips) == 2


def test_select_with_the_default_provider_needs_no_network(mixed):
    """conftest forces ``settings.offline``; the default path must still deliver."""
    transcript, _ = mixed
    clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0)
    assert len(clips) == 2
    assert all(c.score > 0.0 for c in clips)


def test_an_llm_window_longer_than_max_duration_is_trimmed(mixed):
    transcript, _ = mixed
    provider = FakeProvider(
        {"clips": [{"start": 0.0, "end": transcript.words[-1].end, "title": "Whole thing"}]}
    )
    clips = select(transcript, count=1, min_duration=8.0, max_duration=18.0, provider=provider)
    assert len(clips) == 1
    assert clips[0].duration <= 18.0 + EPS
    assert clips[0].duration >= 8.0 - EPS


def test_clip_scores_match_score_window(mixed):
    """A caller can re-derive a heuristic clip's score from its boundaries."""
    transcript, _ = mixed
    for clip in select(transcript, count=3, min_duration=8.0, max_duration=18.0):
        assert clip.score == pytest.approx(score_window(transcript, clip.start, clip.end), abs=1e-6)


def test_a_different_seed_still_produces_valid_clips(mixed):
    """``settings.seed`` only shifts the scan grid; every run must stay legal."""
    from aiclipper.config import Settings

    transcript, _ = mixed
    seen = []
    for seed in (0, 1, 1234, 99_999):
        clips = select(transcript, count=2, min_duration=8.0, max_duration=18.0,
                       settings=Settings(seed=seed))
        assert len(clips) == 2
        for clip in clips:
            assert 8.0 - EPS <= clip.duration <= 18.0 + EPS
            assert not splits_a_word(transcript, clip.start, clip.end)
        assert not clips[0].overlaps(clips[1], tolerance=0.05)
        seen.append([(c.start, c.end) for c in clips])
    # ... and the same seed twice running is identical.
    again = select(transcript, count=2, min_duration=8.0, max_duration=18.0, settings=Settings(seed=0))
    assert [(c.start, c.end) for c in again] == seen[0]


def test_a_long_transcript_is_sampled_not_truncated_for_the_prompt():
    """A three-hour recording must not be represented by its first minutes only."""
    sentences = [f"segment number {i} says something {'x' * 90} about the topic." for i in range(900)]
    transcript, _ = build(sentences)
    provider = FakeProvider({"clips": []})
    select(transcript, count=3, min_duration=15.0, max_duration=60.0, provider=provider)
    prompt = provider.calls[0][0]
    assert len(prompt) < 40_000
    assert "segment number 0 " in prompt
    assert any(f"segment number {i} " in prompt for i in range(860, 900)), "the tail is missing"
    last_line = [ln for ln in prompt.splitlines() if ln.startswith("[")][-1]
    assert float(last_line[1:].split("-", 1)[0]) > 0.8 * transcript.duration
