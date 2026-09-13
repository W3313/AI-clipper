"""Tests for :mod:`aiclipper.transcribe`.

The alignment mapping, the tokeniser and the transcript cache are pure logic and
are exercised for real.  The recogniser itself is stubbed out: a genuine
faster-whisper run needs to download model weights, so it is guarded by an
``importorskip`` plus a check that the weights are already cached locally.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from aiclipper import transcribe as tr
from aiclipper.errors import MissingDependency, TranscriptionError
from aiclipper.models import Segment, Transcript, Word

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def words(*spec: tuple[str, float, float]) -> list[Word]:
    return [Word(text, start, end, 0.9) for text, start, end in spec]


def assert_well_formed(out: list[Word], ref: list[str]) -> None:
    """Every reference token, in order, with sane non-overlapping timings."""
    assert [w.text for w in out] == ref
    prev_end = -1.0
    for w in out:
        assert w.start >= 0.0
        assert w.end > w.start, f"zero-length timing on {w.text!r}"
        assert w.start >= prev_end - 1e-9, f"{w.text!r} starts before the previous word ends"
        prev_end = w.end


class StubWord:
    def __init__(self, word: str, start: float, end: float, probability: float = 0.87):
        self.word = word
        self.start = start
        self.end = end
        self.probability = probability


class StubSegment:
    def __init__(self, text: str, start: float, end: float, words_: list[StubWord] | None = None):
        self.text = text
        self.start = start
        self.end = end
        self.words = words_ or []


class StubInfo:
    def __init__(self, language: str = "en", duration: float = 0.0):
        self.language = language
        self.duration = duration


class StubModel:
    """Stands in for ``faster_whisper.WhisperModel``."""

    def __init__(self, segments: list[StubSegment], info: StubInfo):
        self._segments = segments
        self._info = info
        self.calls: list[dict] = []

    def transcribe(self, path, **kwargs):
        self.calls.append({"path": str(path), **kwargs})
        return iter(self._segments), self._info


def install_stub(monkeypatch: pytest.MonkeyPatch, segments, info) -> dict:
    """Patch ``_load_model`` with a stub; returns a record of how it was built."""
    record: dict = {}
    model = StubModel(segments, info)

    def fake_load(name, *, device="cpu", compute_type="int8", local_files_only=False):
        record.update(
            name=name, device=device, compute_type=compute_type,
            local_files_only=local_files_only, model=model,
        )
        return model

    monkeypatch.setattr(tr, "_load_model", fake_load)
    record["model"] = model
    return record


# --------------------------------------------------------------------------- #
# availability / device
# --------------------------------------------------------------------------- #

def test_module_imports_without_touching_faster_whisper():
    import importlib
    import sys

    sys.modules.pop("faster_whisper", None)
    importlib.reload(tr)
    assert "faster_whisper" not in sys.modules
    assert isinstance(tr.available(), bool)


def test_available_matches_find_spec():
    import importlib.util

    assert tr.available() is (importlib.util.find_spec("faster_whisper") is not None)


@pytest.mark.parametrize(
    ("given", "expected"),
    [("auto", "cpu"), ("", "cpu"), ("AUTO", "cpu"), (None, "cpu"), ("cuda", "cuda"), ("cpu", "cpu")],
)
def test_resolve_device(given, expected):
    assert tr._resolve_device(given) == expected


# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #

def test_tokenize_preserves_spelling_and_punctuation():
    assert tr._tokenize("Hello,  world!\nIt's  fine.") == ["Hello,", "world!", "It's", "fine."]


def test_tokenize_empty_inputs():
    assert tr._tokenize("") == []
    assert tr._tokenize("   \n\t ") == []


def test_normalize_folds_case_punctuation_and_accents():
    assert tr._normalize("Hello,") == tr._normalize("hello") == "hello"
    assert tr._normalize("it’s") == tr._normalize("It's") == "it's"
    assert tr._normalize("Café!") == "cafe"
    assert tr._normalize("—") == "—"  # pure punctuation keeps a key of its own


# --------------------------------------------------------------------------- #
# alignment
# --------------------------------------------------------------------------- #

def test_align_exact_match_copies_asr_timings():
    asr = words(("this", 0.0, 0.3), ("really", 0.3, 0.8), ("works", 0.8, 1.2))
    out = tr._align_tokens(["this", "really", "works"], asr)
    assert_well_formed(out, ["this", "really", "works"])
    assert [(w.start, w.end) for w in out] == [(0.0, 0.3), (0.3, 0.8), (0.8, 1.2)]
    assert all(w.prob == pytest.approx(0.9) for w in out)


def test_align_keeps_reference_casing_and_punctuation():
    ref = ["Hello,", "World!", "It's", "fine."]
    asr = words(("hello", 0.0, 0.4), ("world", 0.4, 0.9), ("its", 0.9, 1.1), ("fine", 1.1, 1.5))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    assert [(w.start, w.end) for w in out] == [(0.0, 0.4), (0.4, 0.9), (0.9, 1.1), (1.1, 1.5)]


def test_align_when_asr_drops_a_word():
    ref = ["the", "quick", "brown", "fox", "jumps"]
    asr = words(("the", 0.0, 0.2), ("quick", 0.2, 0.6), ("fox", 1.0, 1.3), ("jumps", 1.3, 1.8))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    brown = out[2]
    assert brown.start >= 0.6 - 1e-9 and brown.end <= 1.0 + 1e-9
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end == pytest.approx(1.8)
    assert brown.prob == pytest.approx(0.9)


def test_align_when_asr_drops_several_adjacent_words():
    ref = ["one", "two", "three", "four", "five"]
    asr = words(("one", 0.0, 0.4), ("five", 2.0, 2.4))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    middle = out[1:4]
    assert middle[0].start >= 0.4 - 1e-9
    assert middle[-1].end <= 2.0 + 1e-9
    spans = [round(w.end - w.start, 6) for w in middle]
    assert spans[0] == spans[1] == spans[2]  # distributed linearly


def test_align_when_asr_inserts_a_word():
    ref = ["keep", "it", "short"]
    asr = words(("keep", 0.0, 0.3), ("uh", 0.3, 0.5), ("it", 0.5, 0.7), ("short", 0.7, 1.1))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    assert [(w.start, w.end) for w in out] == [(0.0, 0.3), (0.5, 0.7), (0.7, 1.1)]


def test_align_when_asr_substitutes_a_word():
    ref = ["I", "bought", "eighty", "widgets"]
    asr = words(("i", 0.0, 0.2), ("bought", 0.2, 0.6), ("80", 0.6, 1.0), ("widgets", 1.0, 1.6))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    assert (out[2].start, out[2].end) == pytest.approx((0.6, 1.0))


def test_align_completely_disjoint_text_spreads_over_the_asr_span():
    ref = ["alpha", "beta", "gamma", "delta"]
    asr = words(("nothing", 1.0, 1.5), ("like", 1.5, 2.0), ("it", 2.0, 3.0))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    assert out[0].start == pytest.approx(1.0)
    assert out[-1].end == pytest.approx(3.0)
    spans = {round(w.end - w.start, 6) for w in out}
    assert len(spans) == 1


def test_align_never_drops_reference_tokens_when_asr_is_much_shorter():
    ref = "one two three four five six seven eight nine ten".split()
    asr = words(("three", 1.0, 1.4))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)


def test_align_empty_reference_returns_empty():
    assert tr._align_tokens([], words(("hi", 0.0, 0.5))) == []
    assert tr._align_tokens(["", "   "], words(("hi", 0.0, 0.5))) == []


def test_align_without_any_asr_words_spreads_over_the_duration():
    ref = ["one", "two", "three", "four"]
    out = tr._align_tokens(ref, [], duration=4.0)
    assert_well_formed(out, ref)
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end == pytest.approx(4.0)
    assert all(w.prob == 0.0 for w in out)


def test_align_without_asr_words_or_duration_uses_a_speaking_rate():
    ref = ["one", "two", "three", "four", "five"]
    out = tr._align_tokens(ref, [])
    assert_well_formed(out, ref)
    assert out[-1].end == pytest.approx(len(ref) / tr.WORDS_PER_SECOND)


def test_align_repairs_degenerate_asr_timings():
    """Zero-length and out-of-order ASR words must not leak into the output."""
    asr = [Word("a", 0.5, 0.5, 0.4), Word("b", 0.4, 0.4, 0.4), Word("c", 0.4, 0.9, 0.4)]
    out = tr._align_tokens(["a", "b", "c"], asr)
    assert_well_formed(out, ["a", "b", "c"])


def test_align_handles_dense_edits_and_stays_monotonic():
    ref = "we shipped the whole thing in under four weeks flat".split()
    asr = words(
        ("we", 0.0, 0.2), ("ship", 0.2, 0.5), ("whole", 0.9, 1.3), ("thing", 1.3, 1.6),
        ("um", 1.6, 1.8), ("in", 1.8, 1.95), ("under", 1.95, 2.3), ("for", 2.3, 2.6),
        ("weeks", 2.6, 3.0),
    )
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    assert out[-1].end >= 3.0


def test_align_uses_transcript_and_reference_spelling(monkeypatch, make_audio):
    audio = make_audio("narration.wav", seconds=1.0)
    segments = [
        StubSegment(
            "hello their world",
            0.0, 1.2,
            [StubWord("hello", 0.0, 0.4), StubWord("their", 0.4, 0.8), StubWord("world", 0.8, 1.2)],
        )
    ]
    install_stub(monkeypatch, segments, StubInfo("en", 1.2))
    out = tr.align(audio, "Hello, there world!")
    assert [w.text for w in out] == ["Hello,", "there", "world!"]
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end == pytest.approx(1.2)


def test_align_with_blank_text_does_not_transcribe(monkeypatch, tmp_path):
    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("align must not transcribe for empty text")

    monkeypatch.setattr(tr, "transcribe", boom)
    assert tr.align(tmp_path / "missing.wav", "   ") == []


# --------------------------------------------------------------------------- #
# transcribe: errors and caching
# --------------------------------------------------------------------------- #

def test_transcribe_missing_file():
    with pytest.raises(TranscriptionError, match="not found"):
        tr.transcribe("/nonexistent/definitely-not-here.wav")


def test_transcribe_without_backend_raises_missing_dependency(monkeypatch, tmp_path):
    media = tmp_path / "a.wav"
    media.write_bytes(b"not really audio")
    monkeypatch.setattr(tr, "available", lambda: False)
    with pytest.raises(MissingDependency) as excinfo:
        tr.transcribe(media)
    message = str(excinfo.value)
    assert "faster-whisper" in message
    assert "pip install 'aiclipper[transcribe]'" in message


def test_load_model_failure_becomes_transcription_error(monkeypatch):
    """A failed weight download must surface as TranscriptionError, not urllib noise."""
    import sys
    import types

    fake = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, name, device="cpu", compute_type="int8", local_files_only=False):
            raise OSError("could not reach the model hub")

    fake.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    with pytest.raises(TranscriptionError) as excinfo:
        tr._load_model("base", device="cpu", compute_type="int8")
    message = str(excinfo.value)
    assert "could not reach the model hub" in message
    assert "network" in message


def test_load_model_passes_device_and_compute_type(monkeypatch):
    import sys
    import types

    seen: dict = {}
    fake = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, name, device="cpu", compute_type="int8", local_files_only=False):
            seen.update(name=name, device=device, compute_type=compute_type,
                        local_files_only=local_files_only)

    fake.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    tr._load_model("small", device="cpu", compute_type="int8")
    assert seen == {"name": "small", "device": "cpu", "compute_type": "int8", "local_files_only": False}
    tr._load_model("small", device="cpu", compute_type="int8", local_files_only=True)
    assert seen["local_files_only"] is True


@pytest.mark.needs_ffmpeg
def test_transcribe_surfaces_model_load_errors(monkeypatch, make_audio):
    audio = make_audio("tone.wav", seconds=0.5)

    def fake_load(name, *, device="cpu", compute_type="int8", local_files_only=False):
        raise TranscriptionError("could not load whisper model 'base'")

    monkeypatch.setattr(tr, "_load_model", fake_load)
    with pytest.raises(TranscriptionError, match="could not load whisper model"):
        tr.transcribe(audio)
    assert not tr.cache_path(audio).exists()  # nothing cached on failure


@pytest.mark.needs_ffmpeg
def test_transcribe_backend_failure_becomes_transcription_error(monkeypatch, make_audio):
    audio = make_audio("tone.wav", seconds=0.5)

    class Exploding:
        def transcribe(self, *a, **k):
            raise RuntimeError("inference exploded")

    monkeypatch.setattr(tr, "_load_model", lambda *a, **k: Exploding())
    with pytest.raises(TranscriptionError, match="inference exploded"):
        tr.transcribe(audio)


def test_load_model_raises_missing_dependency_when_import_fails(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "faster_whisper":
            raise ImportError("no module named faster_whisper")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(MissingDependency):
        tr._load_model("base")


@pytest.mark.needs_ffmpeg
def test_transcribe_converts_segments_and_writes_cache(monkeypatch, make_audio):
    audio = make_audio("speech.wav", seconds=1.0)
    segments = [
        StubSegment(
            " Hello there. ", 0.0, 0.9,
            [StubWord(" Hello", 0.0, 0.45, 0.91), StubWord(" there.", 0.45, 0.9, 0.62)],
        ),
        StubSegment(
            " Second bit.", 1.1, 2.0,
            [StubWord(" Second", 1.1, 1.6, 0.5), StubWord(" bit.", 1.6, 2.0, 0.4)],
        ),
    ]
    record = install_stub(monkeypatch, segments, StubInfo("fr", 2.5))

    transcript = tr.transcribe(audio, vad=False, model="tiny")

    assert record["name"] == "tiny"
    assert record["device"] == "cpu"  # settings default is "auto" -> cpu
    call = record["model"].calls[0]
    assert call["word_timestamps"] is True
    assert call["vad_filter"] is False
    assert call["path"].endswith(".16k.wav")

    assert transcript.language == "fr"
    assert transcript.duration == pytest.approx(2.5)
    assert [s.text for s in transcript.segments] == ["Hello there.", "Second bit."]
    assert [w.text for w in transcript.words] == ["Hello", "there.", "Second", "bit."]
    assert transcript.words[0].prob == pytest.approx(0.91)
    assert transcript.words[1].prob == pytest.approx(0.62)

    cached = tr.cache_path(audio)
    assert cached.exists()
    assert cached.name == "speech.transcript.json"
    on_disk = json.loads(cached.read_text())
    assert on_disk["language"] == "fr"
    assert Transcript.from_dict(on_disk).words[0].text == "Hello"


@pytest.mark.needs_ffmpeg
def test_transcribe_reuses_a_fresh_cache(monkeypatch, make_audio):
    audio = make_audio("cached.wav", seconds=0.5)
    hand_built = Transcript(
        segments=[Segment("from the cache", 0.0, 1.0, words(("from", 0.0, 0.3), ("the", 0.3, 0.5),
                                                            ("cache", 0.5, 1.0)))],
        language="en",
        duration=1.0,
    )
    hand_built.save(tr.cache_path(audio))

    def boom(*a, **k):  # pragma: no cover - must never run
        raise AssertionError("cache hit must not load a model")

    monkeypatch.setattr(tr, "_load_model", boom)
    out = tr.transcribe(audio)
    assert out.text == "from the cache"
    assert [w.text for w in out.words] == ["from", "the", "cache"]
    assert out.duration == pytest.approx(1.0)


@pytest.mark.needs_ffmpeg
def test_stale_cache_is_ignored(monkeypatch, make_audio):
    audio = make_audio("stale.wav", seconds=0.5)
    cached = tr.cache_path(audio)
    Transcript(segments=[Segment("old news", 0.0, 1.0, words(("old", 0.0, 0.5), ("news", 0.5, 1.0)))],
               language="en", duration=1.0).save(cached)
    # Media edited after the transcript was written.
    os.utime(cached, (1_000_000, 1_000_000))
    os.utime(audio, (2_000_000, 2_000_000))

    segments = [StubSegment("fresh news", 0.0, 0.8,
                            [StubWord("fresh", 0.0, 0.4), StubWord("news", 0.4, 0.8)])]
    install_stub(monkeypatch, segments, StubInfo("en", 0.8))
    out = tr.transcribe(audio)
    assert out.text == "fresh news"
    assert Transcript.load(cached).text == "fresh news"  # rewritten


@pytest.mark.needs_ffmpeg
def test_cache_false_bypasses_read_and_write(monkeypatch, make_audio):
    audio = make_audio("nocache.wav", seconds=0.5)
    cached = tr.cache_path(audio)
    Transcript(segments=[Segment("stale", 0.0, 0.5, words(("stale", 0.0, 0.5)))],
               language="en", duration=0.5).save(cached)

    segments = [StubSegment("live run", 0.0, 0.5,
                            [StubWord("live", 0.0, 0.25), StubWord("run", 0.25, 0.5)])]
    install_stub(monkeypatch, segments, StubInfo("en", 0.5))
    out = tr.transcribe(audio, cache=False)
    assert out.text == "live run"
    assert Transcript.load(cached).text == "stale"  # untouched


@pytest.mark.needs_ffmpeg
def test_cache_is_ignored_for_a_different_language(monkeypatch, make_audio):
    audio = make_audio("lang.wav", seconds=0.5)
    Transcript(segments=[Segment("bonjour", 0.0, 0.5, words(("bonjour", 0.0, 0.5)))],
               language="fr", duration=0.5).save(tr.cache_path(audio))

    segments = [StubSegment("hello", 0.0, 0.5, [StubWord("hello", 0.0, 0.5)])]
    install_stub(monkeypatch, segments, StubInfo("en", 0.5))
    assert tr.transcribe(audio, language="en").text == "hello"


@pytest.mark.needs_ffmpeg
def test_corrupt_cache_is_ignored(monkeypatch, make_audio):
    audio = make_audio("corrupt.wav", seconds=0.5)
    tr.cache_path(audio).write_text("{not json", encoding="utf-8")
    segments = [StubSegment("recovered", 0.0, 0.5, [StubWord("recovered", 0.0, 0.5)])]
    install_stub(monkeypatch, segments, StubInfo("en", 0.5))
    assert tr.transcribe(audio).text == "recovered"


def test_cache_path_sits_next_to_the_media():
    assert tr.cache_path("/tmp/deep/clip.mp4") == Path("/tmp/deep/clip.transcript.json")


@pytest.mark.needs_ffmpeg
def test_transcript_duration_falls_back_to_the_media_length(monkeypatch, make_audio):
    audio = make_audio("fallback.wav", seconds=1.5)
    segments = [StubSegment("hi", 0.0, 0.4, [StubWord("hi", 0.0, 0.4)])]
    install_stub(monkeypatch, segments, StubInfo("en", 0.0))
    out = tr.transcribe(audio, cache=False)
    assert out.duration == pytest.approx(1.5, abs=0.15)


@pytest.mark.needs_ffmpeg
def test_segments_without_words_are_still_kept(monkeypatch, make_audio):
    audio = make_audio("coarse.wav", seconds=0.5)
    segments = [StubSegment("coarse only", 0.0, 0.5, []), StubSegment("   ", 0.5, 0.6, [])]
    install_stub(monkeypatch, segments, StubInfo("en", 0.6))
    out = tr.transcribe(audio, cache=False)
    assert [s.text for s in out.segments] == ["coarse only"]
    assert out.words == []


# --------------------------------------------------------------------------- #
# the real thing (skipped unless the weights are already on disk)
# --------------------------------------------------------------------------- #

def _weights_cached(model: str = "tiny") -> bool:
    home = Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    hub = home / "hub" if home.name != "hub" else home
    return any(hub.glob(f"models--*faster-whisper-{model}*")) if hub.exists() else False


@pytest.mark.needs_ffmpeg
@pytest.mark.slow
def test_real_faster_whisper_roundtrip(make_audio):
    pytest.importorskip("faster_whisper")
    if not _weights_cached("tiny"):
        pytest.skip("whisper weights are not cached locally and downloading them needs network")
    audio = make_audio("real.wav", seconds=1.0)
    try:
        transcript = tr.transcribe(audio, model="tiny", vad=False, cache=False)
    except TranscriptionError as exc:  # pragma: no cover - depends on the host
        pytest.skip(f"whisper unavailable here: {exc}")
    assert isinstance(transcript, Transcript)
    assert transcript.duration > 0
    for word in transcript.words:
        assert word.end >= word.start


# --------------------------------------------------------------------------- #
# regressions added during review
# --------------------------------------------------------------------------- #

def test_align_leading_run_the_asr_missed_keeps_matched_timings():
    """A missed intro must be slotted *before* the first ASR word.

    Regression: the unmatched run used to be laid out starting at the first ASR
    word, which shoved every recognised word later and collapsed it to the
    minimum duration.
    """
    ref = ["Hello", "there", "one", "two"]
    asr = words(("one", 2.0, 2.3), ("two", 2.3, 2.6))
    out = tr._align_tokens(ref, asr, duration=3.0)
    assert_well_formed(out, ref)
    assert (out[2].start, out[2].end) == pytest.approx((2.0, 2.3))
    assert (out[3].start, out[3].end) == pytest.approx((2.3, 2.6))
    assert out[1].end == pytest.approx(2.0)
    assert out[0].start > 0.0  # room existed, so it was used


def test_align_leading_run_clamps_at_zero_when_there_is_no_room():
    ref = ["missed", "words", "here", "one"]
    asr = words(("one", 0.05, 0.4))
    out = tr._align_tokens(ref, asr)
    assert_well_formed(out, ref)
    assert out[0].start == pytest.approx(0.0)
    assert out[-1].end >= 0.4


def test_align_trailing_run_extends_past_the_last_asr_word():
    ref = ["one", "two", "goodbye", "friends"]
    asr = words(("one", 0.0, 0.3), ("two", 0.3, 0.6))
    out = tr._align_tokens(ref, asr, duration=5.0)
    assert_well_formed(out, ref)
    assert (out[0].start, out[0].end) == pytest.approx((0.0, 0.3))
    assert out[2].start == pytest.approx(0.6)
    assert out[2].end == pytest.approx(out[3].start)


@pytest.mark.needs_ffmpeg
def test_transcribe_offline_never_downloads_weights(monkeypatch, make_audio):
    """Hard rule 3: with ``settings.offline`` the model load must stay local."""
    audio = make_audio("offline.wav", seconds=0.5)
    record = install_stub(monkeypatch, [StubSegment("hi", 0.0, 0.4, [StubWord("hi", 0.0, 0.4)])],
                          StubInfo("en", 0.4))
    tr.transcribe(audio, cache=False)  # conftest sets AICLIP_OFFLINE=1
    assert record["local_files_only"] is True


@pytest.mark.needs_ffmpeg
def test_transcribe_online_allows_the_weight_download(monkeypatch, make_audio):
    from aiclipper.config import reset_settings

    audio = make_audio("online.wav", seconds=0.5)
    monkeypatch.setenv("AICLIP_OFFLINE", "0")
    reset_settings()
    record = install_stub(monkeypatch, [StubSegment("hi", 0.0, 0.4, [StubWord("hi", 0.0, 0.4)])],
                          StubInfo("en", 0.4))
    tr.transcribe(audio, cache=False)
    assert record["local_files_only"] is False


def test_offline_model_load_error_mentions_offline_mode(monkeypatch):
    import sys
    import types

    fake = types.ModuleType("faster_whisper")

    class WhisperModel:
        def __init__(self, name, device="cpu", compute_type="int8", local_files_only=False):
            raise OSError("model not found in the local cache")

    fake.WhisperModel = WhisperModel
    monkeypatch.setitem(sys.modules, "faster_whisper", fake)
    with pytest.raises(TranscriptionError, match="offline"):
        tr._load_model("base", local_files_only=True)


@pytest.mark.needs_ffmpeg
def test_align_forwards_language_and_leaves_no_cache_beside_the_audio(monkeypatch, make_audio):
    audio = make_audio("narr.wav", seconds=0.6)
    record = install_stub(
        monkeypatch,
        [StubSegment("bonjour le monde", 0.0, 0.6,
                     [StubWord("bonjour", 0.0, 0.2), StubWord("le", 0.2, 0.4),
                      StubWord("monde", 0.4, 0.6)])],
        StubInfo("fr", 0.6),
    )
    out = tr.align(audio, "Bonjour, le monde!", language="fr")
    assert [w.text for w in out] == ["Bonjour,", "le", "monde!"]
    assert record["model"].calls[0]["language"] == "fr"
    assert not tr.cache_path(audio).exists()  # TTS audio must not litter transcripts


@pytest.mark.needs_ffmpeg
def test_transcribe_rejects_undecodable_media(tmp_path, monkeypatch):
    junk = tmp_path / "not-audio.wav"
    junk.write_bytes(b"this is not a wav file at all")
    install_stub(monkeypatch, [], StubInfo("en", 0.0))
    with pytest.raises(TranscriptionError, match="could not extract audio"):
        tr.transcribe(junk, cache=False)


def test_align_without_the_backend_raises_missing_dependency(monkeypatch, tmp_path):
    media = tmp_path / "voice.wav"
    media.write_bytes(b"placeholder")
    monkeypatch.setattr(tr, "available", lambda: False)
    with pytest.raises(MissingDependency):
        tr.align(media, "some narration text")


def test_package_degrades_with_faster_whisper_uninstalled(tmp_path):
    """Import, ``available()`` and the error paths with the extra truly absent."""
    import subprocess
    import sys

    media = tmp_path / "clip.wav"
    media.write_bytes(b"placeholder")
    script = tmp_path / "blocked.py"
    script.write_text(
        "import sys\n"
        "class Block:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == 'faster_whisper' or name.startswith('faster_whisper.'):\n"
        "            raise ModuleNotFoundError('blocked for the test')\n"
        "        return None\n"
        "sys.meta_path.insert(0, Block())\n"
        "sys.modules.pop('faster_whisper', None)\n"
        "import aiclipper.transcribe as tr\n"
        "from aiclipper.errors import MissingDependency\n"
        "from aiclipper.models import Word\n"
        "assert 'faster_whisper' not in sys.modules\n"
        "assert tr.available() is False\n"
        f"media = {str(media)!r}\n"
        "for call in (lambda: tr.transcribe(media), lambda: tr.align(media, 'hello there')):\n"
        "    try:\n"
        "        call()\n"
        "    except MissingDependency as exc:\n"
        "        assert \"pip install 'aiclipper[transcribe]'\" in str(exc), exc\n"
        "    else:\n"
        "        raise AssertionError('expected MissingDependency')\n"
        "out = tr._align_tokens(['a', 'b'], [Word('a', 0.0, 0.5, 1.0)])\n"
        "assert [w.text for w in out] == ['a', 'b']\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().endswith("ok")


def test_converter_handles_real_faster_whisper_result_objects():
    """The converter is fed genuine faster-whisper dataclasses (no weights needed)."""
    ft = pytest.importorskip("faster_whisper.transcribe")

    fw_words = [
        ft.Word(start=0.0, end=0.42, word=" Hello", probability=0.93),
        ft.Word(start=0.42, end=0.90, word=" there.", probability=0.51),
    ]
    segment = ft.Segment(
        id=1, seek=0, start=0.0, end=0.9, text=" Hello there.", tokens=[50364, 2425],
        avg_logprob=-0.21, compression_ratio=1.04, no_speech_prob=0.02,
        words=fw_words, temperature=0.0,
    )
    info = ft.TranscriptionInfo(
        language="en", language_probability=0.99, duration=1.25, duration_after_vad=1.25,
        all_language_probs=None, transcription_options=None, vad_options=None,
    )

    out = tr._to_transcript(iter([segment]), info, fallback_duration=9.0)
    assert isinstance(out, Transcript)
    assert out.language == "en"
    assert out.duration == pytest.approx(1.25)
    assert all(isinstance(s, Segment) for s in out.segments)
    assert all(isinstance(w, Word) for w in out.words)
    assert [w.text for w in out.words] == ["Hello", "there."]
    assert [round(w.prob, 2) for w in out.words] == [0.93, 0.51]
    assert out.segments[0].text == "Hello there."
