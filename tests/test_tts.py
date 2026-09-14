"""Tests for :mod:`aiclipper.tts`.

Everything here runs offline.  The only backend actually exercised end to end is
:class:`~aiclipper.tts.offline.OfflineTTS` -- that is the CI path, so its
arithmetic is checked hard.  The two network backends are tested through their
pure helpers (prosody formatting, tick conversion, request building) and their
refusal paths; nothing in this file opens a socket.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

from aiclipper import ffmpeg as ff
from aiclipper.config import Settings
from aiclipper.errors import MissingDependency, TTSError
from aiclipper.models import TTSResult, VoiceSpec, Word
from aiclipper.tts import (
    VOICES,
    EdgeTTS,
    ElevenLabsTTS,
    NarrationResults,
    OfflineTTS,
    Voice,
    estimate_duration,
    fallback_chain,
    find_voice,
    find_voice_entry,
    get_provider,
    list_voices,
    plan_words,
    provider_usable,
    reset_usable_cache,
    resolve_voice_id,
    synthesize_lines,
    total_duration,
)
from aiclipper.tts import edge as edge_mod
from aiclipper.tts import eleven as eleven_mod
from aiclipper.tts.base import PROVIDER_ALIASES, TTSProvider, run_bounded

EPS = 1e-3


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _online_settings(**overrides) -> Settings:
    """A Settings object that is *not* offline, for routing tests."""
    s = Settings()
    s.offline = False
    for key, value in overrides.items():
        setattr(s, key, value)
    return s


def _speech_bytes(tmp_path: Path, seconds: float = 1.5) -> bytes:
    """Encoded audio standing in for a service response.

    Real mp3 when this ffmpeg can encode one, wav otherwise -- ffmpeg sniffs the
    content rather than the extension, so the transcode path under test is the
    same either way.
    """
    src = tmp_path / "_tone.wav"
    ff.make_tone(seconds, src)
    if ff.has_encoder("libmp3lame"):
        mp3 = tmp_path / "_tone.mp3"
        ff.run_ffmpeg(["-y", "-i", str(src), "-c:a", "libmp3lame", str(mp3)])
        return mp3.read_bytes()
    return src.read_bytes()


def _install_fake_edge(monkeypatch: pytest.MonkeyPatch, chunks: list[dict]) -> list[dict]:
    """Put a fake ``edge_tts`` in ``sys.modules`` that streams ``chunks``."""
    seen: list[dict] = []

    class FakeCommunicate:
        # The 7.x signature: keyword-only prosody plus the boundary opt-in.
        def __init__(self, text, voice, *, rate="+0%", pitch="+0Hz", boundary="SentenceBoundary"):
            seen.append({"text": text, "voice": voice, "rate": rate,
                         "pitch": pitch, "boundary": boundary})

        async def stream(self):
            for chunk in chunks:
                await asyncio.sleep(0)
                yield chunk

    module = types.ModuleType("edge_tts")
    module.Communicate = FakeCommunicate
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    return seen


def _assert_timings_sane(words: list[Word], duration: float) -> None:
    assert words, "expected word timings"
    assert words[0].start >= -EPS
    for word in words:
        assert word.end > word.start, f"{word.text!r} has non-positive duration"
    for previous, current in zip(words, words[1:], strict=False):
        assert current.start >= previous.end - EPS, "word timings overlap"
        assert current.start >= previous.start, "word timings are not monotonic"
    assert words[-1].end <= duration + EPS, "captions outlive the audio"
    assert abs(words[-1].end - duration) < EPS, "the last word should land on the end of the audio"


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #

def test_importing_the_package_pulls_in_no_optional_dependency():
    """Rule 3: ``import aiclipper.tts`` must not drag in edge-tts/aiohttp."""
    code = (
        "import sys; import aiclipper.tts as t;"
        "assert 'edge_tts' not in sys.modules, 'edge_tts imported at module scope';"
        "assert 'aiohttp' not in sys.modules, 'aiohttp imported at module scope';"
        "print(len(t.VOICES))"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert int(proc.stdout.strip()) >= 40


def test_the_package_works_with_every_optional_extra_uninstalled():
    """Rule 2, for real: block the optional imports, then use the package.

    A meta-path finder that refuses ``edge_tts`` and friends is the closest we
    can get to a bare interpreter without disturbing the shared venv.  Importing
    must still work, ``auto`` must degrade to the offline backend, and naming
    ``edge`` explicitly must raise ``MissingDependency`` with its pip line.
    """
    code = """
import sys
BLOCKED = {"edge_tts", "aiohttp", "anthropic", "cv2", "faster_whisper", "yt_dlp", "playwright"}

class Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ImportError("blocked: " + name)
        return None

sys.meta_path.insert(0, Blocker())
from pathlib import Path
import aiclipper.tts as t
from aiclipper.errors import MissingDependency
from aiclipper.models import VoiceSpec

print(t.get_provider("auto").name)
try:
    t.EdgeTTS().synthesize("hi", Path("unused.wav"), voice=VoiceSpec())
except MissingDependency as exc:
    print(exc)
else:
    raise SystemExit("edge should have raised MissingDependency")
"""
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                          env={**os.environ, "AICLIP_OFFLINE": "0"})
    assert proc.returncode == 0, proc.stderr
    first, second = proc.stdout.splitlines()[:2]
    assert first == "offline", "auto must degrade to the offline backend"
    assert "edge-tts" in second and "pip install 'aiclipper[tts]'" in second


# --------------------------------------------------------------------------- #
# the offline provider, end to end
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_offline_synthesis_produces_a_probeable_wav(tmp_path: Path):
    provider = OfflineTTS()
    assert provider.name == "offline"
    assert provider.available()
    assert isinstance(provider, TTSProvider)

    text = "Here is the part nobody tells you about shipping something people want."
    out = tmp_path / "vo" / "line.wav"
    result = provider.synthesize(text, out, voice=VoiceSpec())

    assert isinstance(result, TTSResult)
    assert result.audio_path == out
    assert out.exists() and out.stat().st_size > 0
    info = ff.probe(out)
    assert info.has_audio and not info.has_video
    assert info.duration > 0
    assert abs(result.duration - info.duration) < EPS
    assert result.text == text
    assert [w.text for w in result.words] == text.split()
    _assert_timings_sane(result.words, result.duration)


@pytest.mark.needs_ffmpeg
@pytest.mark.parametrize("text", [
    "one",
    "two words here",
    "a slightly longer sentence with a comma, and then a full stop.",
    "Numbers 3 and 42 plus punctuation -- does it still line up? Yes!",
])
def test_offline_timings_are_sane_for_varied_text(tmp_path: Path, text: str):
    result = OfflineTTS().synthesize(text, tmp_path / "x.wav", voice=VoiceSpec())
    assert len(result.words) == len(text.split())
    _assert_timings_sane(result.words, result.duration)


@pytest.mark.needs_ffmpeg
def test_offline_duration_scales_with_word_count(tmp_path: Path):
    provider = OfflineTTS()
    durations = []
    for count in (4, 8, 16):
        text = " ".join(["word"] * count)
        result = provider.synthesize(text, tmp_path / f"n{count}.wav", voice=VoiceSpec())
        durations.append(result.duration)
        # ~2.6 words/second, within the 3-decimal rounding of make_silence.
        assert result.duration == pytest.approx(count / 2.6, abs=0.01)
    assert durations[0] < durations[1] < durations[2]
    assert durations[2] == pytest.approx(durations[0] * 4, rel=0.02)


@pytest.mark.needs_ffmpeg
def test_offline_duration_scales_inversely_with_rate(tmp_path: Path):
    provider = OfflineTTS()
    text = " ".join(["word"] * 12)
    slow = provider.synthesize(text, tmp_path / "slow.wav", voice=VoiceSpec(rate=0.5))
    normal = provider.synthesize(text, tmp_path / "normal.wav", voice=VoiceSpec(rate=1.0))
    fast = provider.synthesize(text, tmp_path / "fast.wav", voice=VoiceSpec(rate=2.0))

    assert slow.duration > normal.duration > fast.duration
    assert normal.duration == pytest.approx(slow.duration / 2, rel=0.02)
    assert fast.duration == pytest.approx(normal.duration / 2, rel=0.02)
    for result in (slow, normal, fast):
        _assert_timings_sane(result.words, result.duration)


@pytest.mark.needs_ffmpeg
def test_offline_is_deterministic_for_the_same_seed(tmp_path: Path):
    text = "determinism is the whole point of seeding from settings"
    first = OfflineTTS(settings=Settings()).synthesize(text, tmp_path / "a.wav", voice=VoiceSpec())
    second = OfflineTTS(settings=Settings()).synthesize(text, tmp_path / "b.wav", voice=VoiceSpec())
    assert [(w.text, w.start, w.end) for w in first.words] == [(w.text, w.start, w.end) for w in second.words]

    other = Settings()
    other.seed = 999999
    third = OfflineTTS(settings=other).synthesize(text, tmp_path / "c.wav", voice=VoiceSpec())
    assert [w.start for w in third.words] != [w.start for w in first.words]


@pytest.mark.needs_ffmpeg
def test_offline_handles_blank_text(tmp_path: Path):
    result = OfflineTTS().synthesize("   ", tmp_path / "blank.wav", voice=VoiceSpec())
    assert result.words == []
    assert result.duration > 0
    assert ff.probe(result.audio_path).duration > 0


@pytest.mark.needs_ffmpeg
def test_offline_records_the_voice_it_was_asked_for(tmp_path: Path):
    spec = find_voice("british_male", provider="edge")
    result = OfflineTTS().synthesize("hello world", tmp_path / "v.wav", voice=spec)
    assert result.voice is spec
    assert result.voice.voice_id == "en-GB-RyanNeural"


# --------------------------------------------------------------------------- #
# the pure timing planner (no ffmpeg)
# --------------------------------------------------------------------------- #

def test_plan_words_spans_exactly_the_requested_total():
    words = plan_words("alpha beta gamma delta epsilon", total=7.5, seed=1234)
    assert len(words) == 5
    assert words[0].start == 0.0
    assert words[-1].end == pytest.approx(7.5, abs=1e-9)
    _assert_timings_sane(words, 7.5)


def test_plan_words_gives_longer_words_longer_slices():
    words = plan_words("a extraordinarily a", total=6.0, seed=7)
    assert words[1].duration > words[0].duration
    assert words[1].duration > words[2].duration


def test_plan_words_is_empty_for_empty_text():
    assert plan_words("", total=3.0) == []
    assert plan_words("   ", total=3.0) == []


def test_plan_words_survives_heavy_punctuation():
    """Gaps must never eat the whole budget, however punctuated the line is."""
    text = "a. b. c. d. e. f. g. h."
    words = plan_words(text, total=0.6, seed=5)
    assert len(words) == 8
    _assert_timings_sane(words, 0.6)


def test_estimate_duration_matches_the_documented_rate():
    assert estimate_duration(" ".join(["w"] * 26)) == pytest.approx(10.0)
    assert estimate_duration(" ".join(["w"] * 26), rate=2.0) == pytest.approx(5.0)
    assert estimate_duration("") > 0


# --------------------------------------------------------------------------- #
# synthesize_lines: the rebasing contract
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_synthesize_lines_rebases_onto_one_timeline(tmp_path: Path):
    lines = [
        "First line of the narration.",
        "A second line that is quite a bit longer than the first one was.",
        "Third.",
        "And a fourth to finish it off.",
    ]
    gap = 0.25
    results = synthesize_lines(lines, tmp_path / "vo", voice=VoiceSpec(), provider="offline", gap=gap)

    assert len(results) == len(lines)
    assert [r.audio_path.name for r in results] == [f"line_{i:03d}.wav" for i in range(len(lines))]
    for result, line in zip(results, lines, strict=True):
        assert result.audio_path.exists()
        assert result.text == line
        assert ff.probe(result.audio_path).duration == pytest.approx(result.duration, abs=EPS)

    # Rebasing arithmetic, recomputed independently of the implementation.
    offset = 0.0
    previous_end = -1.0
    for index, result in enumerate(results):
        expected = sum(r.duration for r in results[:index]) + gap * index
        assert offset == pytest.approx(expected, abs=1e-9)
        assert result.words[0].start >= offset - EPS
        assert result.words[-1].end == pytest.approx(offset + result.duration, abs=EPS)
        assert result.words[0].start > previous_end, "lines must not overlap on the shared timeline"
        _assert_timings_sane([Word(w.text, w.start - offset, w.end - offset) for w in result.words],
                             result.duration)
        previous_end = result.words[-1].end
        offset = expected + result.duration + gap

    every_word = [w for r in results for w in r.words]
    assert [w.text for w in every_word] == " ".join(lines).split()
    for previous, current in zip(every_word, every_word[1:], strict=False):
        assert current.start >= previous.end - EPS

    assert total_duration(results, gap) == pytest.approx(
        sum(r.duration for r in results) + gap * (len(results) - 1)
    )
    assert every_word[-1].end == pytest.approx(total_duration(results, gap), abs=EPS)


@pytest.mark.needs_ffmpeg
def test_rebased_words_match_the_actually_concatenated_audio(tmp_path: Path):
    """The contract, checked against real audio rather than against arithmetic.

    Concatenate the line files with ``gap`` seconds of silence between them --
    exactly what the docstring tells a caller to do -- and the last rebased word
    must land on the end of that file.
    """
    lines = ["First line here.", "Second line, a little longer than the first.", "Third."]
    gap = 0.3
    results = synthesize_lines(lines, tmp_path / "vo", voice=VoiceSpec(), provider="offline", gap=gap)

    silence = tmp_path / "gap.wav"
    ff.make_silence(gap, silence)
    parts: list[Path] = []
    for index, result in enumerate(results):
        if index:
            parts.append(silence)
        parts.append(result.audio_path)
    listing = tmp_path / "concat.txt"
    listing.write_text("\n".join(f"file '{p}'" for p in parts), encoding="utf-8")

    joined = tmp_path / "joined.wav"
    ff.run_ffmpeg(["-y", "-f", "concat", "-safe", "0", "-i", str(listing), "-c", "copy", str(joined)])

    joined_duration = ff.probe(joined).duration
    assert joined_duration == pytest.approx(total_duration(results, gap), abs=0.01)

    words = [w for r in results for w in r.words]
    assert words[-1].end == pytest.approx(joined_duration, abs=0.01), "captions outlive the narration"
    assert words[0].start == pytest.approx(0.0, abs=EPS)


@pytest.mark.needs_ffmpeg
def test_synthesize_lines_gap_shifts_every_later_line(tmp_path: Path):
    lines = ["one two three", "four five six"]
    tight = synthesize_lines(lines, tmp_path / "tight", voice=VoiceSpec(), provider="offline", gap=0.0)
    loose = synthesize_lines(lines, tmp_path / "loose", voice=VoiceSpec(), provider="offline", gap=1.0)

    assert tight[1].words[0].start == pytest.approx(tight[0].duration, abs=EPS)
    assert loose[1].words[0].start == pytest.approx(loose[0].duration + 1.0, abs=EPS)
    assert loose[1].words[0].start - tight[1].words[0].start == pytest.approx(1.0, abs=EPS)
    # The audio itself is unaffected by the gap: only the timeline moves.
    assert tight[1].duration == pytest.approx(loose[1].duration, abs=EPS)


@pytest.mark.needs_ffmpeg
def test_synthesize_lines_keeps_blank_lines_in_place(tmp_path: Path):
    results = synthesize_lines(
        ["spoken line", "", "another spoken line"],
        tmp_path / "vo", voice=VoiceSpec(), provider="offline", gap=0.1,
    )
    assert len(results) == 3
    assert results[1].words == []
    assert results[1].audio_path.exists()
    expected = results[0].duration + 0.1 + results[1].duration + 0.1
    assert results[2].words[0].start >= expected - EPS


@pytest.mark.needs_ffmpeg
def test_synthesize_lines_accepts_a_provider_instance(tmp_path: Path):
    provider = OfflineTTS()
    results = synthesize_lines(["hello there"], tmp_path / "vo", voice=VoiceSpec(), provider=provider)
    assert len(results) == 1 and results[0].words


@pytest.mark.needs_ffmpeg
def test_synthesize_lines_writes_exactly_one_file_per_line(tmp_path: Path):
    out_dir = tmp_path / "deep" / "vo"          # created on demand
    results = synthesize_lines(["a b", "", "c d"], out_dir, voice=VoiceSpec(), provider="offline")
    assert sorted(p.name for p in out_dir.iterdir()) == ["line_000.wav", "line_001.wav", "line_002.wav"]
    assert len({r.audio_path for r in results}) == 3, "lines must not overwrite each other"


def test_synthesize_lines_with_no_lines_is_empty(tmp_path: Path):
    assert synthesize_lines([], tmp_path / "vo", voice=VoiceSpec(), provider="offline") == []


def test_total_duration_of_nothing_is_zero():
    assert total_duration([]) == 0.0
    one = TTSResult(audio_path=Path("x.wav"), duration=2.0)
    assert total_duration([one], gap=5.0) == pytest.approx(2.0)


@pytest.mark.needs_ffmpeg
def test_synthesize_lines_leaves_wordless_results_alone(tmp_path: Path):
    """A backend with no boundaries still advances the offset for later lines."""

    class Wordless:
        name = "wordless"

        def available(self) -> bool:
            return True

        def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
            ff.make_silence(1.0, out_path)
            return TTSResult(audio_path=Path(out_path), duration=1.0, words=None, text=text)

    results = synthesize_lines(["a", "b"], tmp_path / "vo", voice=VoiceSpec(),
                               provider=Wordless(), gap=0.5)
    assert [r.words for r in results] == [None, None]
    assert total_duration(results, 0.5) == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# the voice catalogue
# --------------------------------------------------------------------------- #

def test_catalogue_is_large_and_well_formed():
    assert len(VOICES) >= 40
    names = [v.name for v in VOICES]
    assert len(set(names)) == len(names), "voice names must be unique"
    for voice in VOICES:
        assert isinstance(voice, Voice)
        assert voice.name and voice.name == voice.name.lower()
        assert voice.description.strip(), f"{voice.name} has no description"
        assert voice.providers, f"{voice.name} has no provider id at all"
        assert len(voice.tags) >= 3, f"{voice.name} needs gender/accent/energy/use-case tags"


def test_catalogue_covers_the_documented_tag_axes():
    genders = {"male", "female", "neutral"}
    accents = {"american", "british", "australian", "irish", "indian", "canadian"}
    for voice in VOICES:
        tags = set(voice.tags)
        assert tags & genders, f"{voice.name} has no gender tag"
    assert accents <= {tag for v in VOICES for tag in v.tags}
    speakable = [v for v in VOICES if v.edge or v.elevenlabs]
    assert len(speakable) >= 40, "at least 40 voices must map to a real backend"


def test_catalogue_introspection_helpers():
    from aiclipper.tts.voices import KNOWN_PROVIDERS, canonical_provider, tag_values, voice_names

    names = voice_names()
    assert names == [v.name for v in VOICES]
    assert "narrator_deep" in names

    tags = tag_values()
    assert tags == sorted(set(tags))
    assert {"male", "female", "british", "calm", "narration"} <= set(tags)

    assert find_voice_entry("narrator_deep").has_tag("DEEP")
    assert not find_voice_entry("narrator_deep").has_tag("neon")

    assert canonical_provider("Edge-TTS") == "edge"
    assert canonical_provider("11labs") == "elevenlabs"
    assert canonical_provider(None) == "auto"
    assert canonical_provider("weird") == "weird"
    assert set(KNOWN_PROVIDERS) == {"edge", "elevenlabs", "offline"}
    assert find_voice_entry("narrator_deep").supports("edge")
    assert not find_voice_entry("newsroom").supports("elevenlabs")


def test_find_voice_exact_name():
    spec = find_voice("narrator_deep")
    assert isinstance(spec, VoiceSpec)
    assert spec.voice_id == "narrator_deep"
    assert find_voice_entry("narrator_deep").edge == "en-US-GuyNeural"


def test_find_voice_is_case_insensitive():
    assert find_voice_entry("NARRATOR_DEEP").name == "narrator_deep"
    assert find_voice_entry("British_Female").name == "british_female"
    assert find_voice_entry("  bright_female  ").name == "bright_female"


def test_find_voice_falls_back_to_tags():
    entry = find_voice_entry("british female")
    assert "british" in entry.tags and "female" in entry.tags

    deep = find_voice_entry("deep documentary male")
    assert {"deep", "documentary", "male"} <= set(deep.tags)

    # Hyphens and underscores are interchangeable with spaces in a tag query.
    assert find_voice_entry("australian-male").name == find_voice_entry("australian male").name


def test_tag_query_never_confuses_male_with_female():
    """Regression: substring matching made ``"male"`` match ``"female"``."""
    for query in ("male", "male energetic", "male bright promo", "american male"):
        entry = find_voice_entry(query)
        assert "male" in entry.tags and "female" not in entry.tags, f"{query!r} -> {entry.name}"
    for query in ("female", "female energetic", "female bright promo"):
        entry = find_voice_entry(query)
        assert "female" in entry.tags, f"{query!r} -> {entry.name}"
    # A prefix is still enough, which is what makes loose queries usable.
    assert find_voice_entry("brit female").name == find_voice_entry("british female").name


def test_find_voice_tag_query_picks_the_first_catalogue_match():
    """A tag query must resolve to one predictable entry, not "some match"."""
    query = ("warm", "narration", "male")
    expected = next(v for v in VOICES if set(query) <= set(v.tags))
    assert find_voice_entry(" ".join(query)).name == expected.name == "narrator_warm"
    # ...and the same holds when the pool is narrowed to one backend.
    eleven_expected = next(v for v in list_voices("elevenlabs") if set(query) <= set(v.tags))
    assert find_voice_entry(" ".join(query), provider="elevenlabs").name == eleven_expected.name


def test_offline_timings_are_reproducible_in_a_fresh_interpreter():
    """Rule 7: seeded output must not depend on this process's hash seed."""
    code = (
        "from aiclipper.tts import plan_words;"
        "print([round(w.start, 9) for w in plan_words('the quick brown fox jumps', "
        "total=4.0, seed=1234)])"
    )
    runs = [
        subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120,
                       env={**os.environ, "PYTHONHASHSEED": seed})
        for seed in ("0", "12345")
    ]
    for proc in runs:
        assert proc.returncode == 0, proc.stderr
    assert runs[0].stdout == runs[1].stdout
    in_process = [round(w.start, 9) for w in plan_words("the quick brown fox jumps", total=4.0, seed=1234)]
    assert runs[0].stdout.strip() == str(in_process)


def test_find_voice_resolves_a_provider_prefixed_spec():
    spec = find_voice("edge:narrator_deep")
    assert spec.provider == "edge"
    assert spec.voice_id == "en-US-GuyNeural"

    native = find_voice("edge:en-GB-ThisVoiceIsNotInOurCatalogue")
    assert native.provider == "edge"
    assert native.voice_id == "en-GB-ThisVoiceIsNotInOurCatalogue"


def test_find_voice_applies_rate_and_pitch():
    spec = find_voice("narrator_deep", provider="edge", rate=1.15, pitch_semitones=-2.0)
    assert (spec.rate, spec.pitch_semitones) == (1.15, -2.0)
    assert spec.voice_id == "en-US-GuyNeural"


def test_find_voice_rejects_nonsense_and_suggests_neighbours():
    with pytest.raises(TTSError) as excinfo:
        find_voice("narrator_deeep")
    message = str(excinfo.value)
    assert "narrator_deeep" in message
    assert "narrator_deep" in message, "the closest catalogue name should be suggested"

    with pytest.raises(TTSError):
        find_voice("")
    with pytest.raises(TTSError):
        find_voice("klingon warlord battlecry")


def test_find_voice_honours_the_provider_filter():
    assert find_voice_entry("narrator_deep", provider="elevenlabs").elevenlabs
    with pytest.raises(TTSError) as excinfo:
        # 'newsroom' is edge-only; asking ElevenLabs for it must fail loudly.
        find_voice_entry("newsroom", provider="elevenlabs")
    assert "elevenlabs" in str(excinfo.value)


def test_list_voices_filters_by_provider():
    everything = list_voices()
    assert everything == list(VOICES)
    assert list_voices("auto") == everything
    assert list_voices(None) == everything

    edge_voices = list_voices("edge")
    assert edge_voices and len(edge_voices) < len(everything)
    assert all(v.edge for v in edge_voices)
    assert set(edge_voices) <= set(everything)

    eleven_voices = list_voices("elevenlabs")
    assert eleven_voices and all(v.elevenlabs for v in eleven_voices)
    assert len(eleven_voices) < len(edge_voices)

    # The silent fallback can speak every name in the catalogue.
    assert list_voices("offline") == everything
    # Provider spellings are normalised.
    assert list_voices("edge-tts") == edge_voices
    assert list_voices("11labs") == eleven_voices
    assert list_voices("nonesuch-backend") == []


def test_resolve_voice_id_maps_catalogue_names_and_passes_natives_through():
    assert resolve_voice_id(VoiceSpec(voice_id="narrator_deep"), "edge") == "en-US-GuyNeural"
    assert resolve_voice_id(VoiceSpec(voice_id="NARRATOR_DEEP"), "edge") == "en-US-GuyNeural"
    assert resolve_voice_id(VoiceSpec(voice_id="narrator_deep"), "elevenlabs") == "pNInz6obpgDQGcFmaJgB"
    # Not one of ours -> verbatim.
    assert resolve_voice_id(VoiceSpec(voice_id="en-GB-RyanNeural"), "edge") == "en-GB-RyanNeural"
    # Known voice, but this backend has no id for it -> the default.
    assert resolve_voice_id(VoiceSpec(voice_id="newsroom"), "elevenlabs", default="d") == "d"
    # Nothing asked for -> the default.
    assert resolve_voice_id(VoiceSpec(), "edge", default="fallback") == "fallback"
    assert resolve_voice_id(None, "edge", default="fallback") == "fallback"


def test_voice_to_spec_round_trips_through_voicespec_parse():
    spec = find_voice_entry("british_male").to_spec("edge")
    assert VoiceSpec.parse(f"{spec.provider}:{spec.voice_id}") == VoiceSpec(
        provider="edge", voice_id="en-GB-RyanNeural"
    )


# --------------------------------------------------------------------------- #
# VoiceSpec.parse (the contract the CLI hands us)
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("raw, provider, voice_id", [
    ("", "auto", ""),
    ("   ", "auto", ""),
    ("narrator_male", "auto", "narrator_male"),
    ("edge:en-US-GuyNeural", "edge", "en-US-GuyNeural"),
    ("elevenlabs:21m00Tcm4TlvDq8ikWAM", "elevenlabs", "21m00Tcm4TlvDq8ikWAM"),
    ("  edge : en-GB-RyanNeural  ", "edge", "en-GB-RyanNeural"),
    (":bare", "auto", "bare"),
])
def test_voicespec_parse_forms(raw: str, provider: str, voice_id: str):
    spec = VoiceSpec.parse(raw)
    assert (spec.provider, spec.voice_id) == (provider, voice_id)
    assert (spec.rate, spec.pitch_semitones, spec.language) == (1.0, 0.0, "en")


# --------------------------------------------------------------------------- #
# get_provider routing
# --------------------------------------------------------------------------- #

def test_get_provider_routes_explicit_names():
    settings = _online_settings()
    assert get_provider("offline", settings=settings).name == "offline"
    assert get_provider("edge", settings=settings).name == "edge"
    assert get_provider("elevenlabs", settings=settings).name == "elevenlabs"
    # Aliases.
    assert get_provider("silence", settings=settings).name == "offline"
    assert get_provider("edge-tts", settings=settings).name == "edge"
    assert get_provider("11labs", settings=settings).name == "elevenlabs"
    assert get_provider("EDGE", settings=settings).name == "edge"


def test_get_provider_rejects_unknown_names():
    with pytest.raises(TTSError) as excinfo:
        get_provider("festival", settings=_online_settings())
    assert "festival" in str(excinfo.value)
    assert "offline" in str(excinfo.value)
    assert set(PROVIDER_ALIASES.values()) == {"auto", "edge", "elevenlabs", "offline"}


def test_get_provider_accepts_every_catalogue_provider_spelling():
    """The two alias tables are one table; a name one accepts, the other accepts."""
    from aiclipper.tts.voices import PROVIDER_ALIASES as VOICE_ALIASES

    settings = _online_settings()
    assert set(VOICE_ALIASES) <= set(PROVIDER_ALIASES)
    for spelling, canonical in VOICE_ALIASES.items():
        if canonical == "auto":
            continue
        assert get_provider(spelling, settings=settings).name == canonical, spelling


def test_every_backend_satisfies_the_provider_protocol():
    settings = _online_settings()
    for name in ("offline", "edge", "elevenlabs"):
        provider = get_provider(name, settings=settings)
        assert isinstance(provider, TTSProvider)
        assert provider.name == name
        assert isinstance(provider.available(), bool)


def test_get_provider_defaults_to_the_configured_provider():
    settings = _online_settings(tts_provider="offline")
    assert get_provider(settings=settings).name == "offline"
    assert get_provider(None, settings=settings).name == "offline"


def test_auto_prefers_edge_when_importable(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(edge_mod, "edge_available", lambda: True)
    assert get_provider("auto", settings=_online_settings()).name == "edge"


def test_auto_falls_back_to_elevenlabs_when_edge_is_missing(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(edge_mod, "edge_available", lambda: False)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    assert get_provider("auto", settings=_online_settings()).name == "elevenlabs"


def test_auto_falls_back_to_offline_with_no_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(edge_mod, "edge_available", lambda: False)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert get_provider("auto", settings=_online_settings()).name == "offline"


def test_auto_is_offline_when_settings_say_offline(monkeypatch: pytest.MonkeyPatch):
    """AICLIP_OFFLINE beats both network backends, key or no key."""
    monkeypatch.setattr(edge_mod, "edge_available", lambda: True)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    settings = Settings()
    settings.offline = True
    assert get_provider("auto", settings=settings).name == "offline"
    assert get_provider("auto").name == "offline"  # conftest sets AICLIP_OFFLINE=1


# --------------------------------------------------------------------------- #
# edge backend: pure helpers and refusal paths
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rate, expected", [
    (1.0, "+0%"), (1.15, "+15%"), (1.5, "+50%"), (0.8, "-20%"), (0.5, "-50%"),
    (2.0, "+100%"), (1.07, "+7%"), (0.999, "+0%"),
    (5.0, "+100%"), (0.01, "-50%"), (0.0, "+0%"), (-3.0, "+0%"), (None, "+0%"),
])
def test_format_rate(rate, expected: str):
    assert edge_mod.format_rate(rate) == expected


@pytest.mark.parametrize("semitones, expected", [
    (0.0, "+0Hz"), (2.0, "+24Hz"), (-1.0, "-11Hz"), (1.0, "+12Hz"),
    (12.0, "+200Hz"), (-12.0, "-100Hz"),
    (40.0, "+200Hz"), (-40.0, "-100Hz"), (None, "+0Hz"),
])
def test_format_pitch(semitones, expected: str):
    formatted = edge_mod.format_pitch(semitones)
    assert formatted == expected
    assert formatted[0] in "+-" and formatted.endswith("Hz")


def test_ticks_to_seconds():
    assert edge_mod.TICKS_PER_SECOND == 10_000_000
    assert edge_mod.ticks_to_seconds(10_000_000) == 1.0
    assert edge_mod.ticks_to_seconds(0) == 0.0
    assert edge_mod.ticks_to_seconds(12_500_000) == 1.25
    assert edge_mod.ticks_to_seconds(1_000_000) == pytest.approx(0.1)
    assert edge_mod.ticks_to_seconds("nope") == 0.0


def test_words_from_boundaries_converts_ticks_and_orders_words():
    events = [
        {"type": "WordBoundary", "offset": 5_000_000, "duration": 2_500_000, "text": "second"},
        {"type": "WordBoundary", "offset": 0, "duration": 4_000_000, "text": "first"},
        {"type": "WordBoundary", "offset": 80_000_000, "duration": 3_000_000, "text": "third"},
    ]
    words = edge_mod.words_from_boundaries(events)
    assert [w.text for w in words] == ["first", "second", "third"]
    assert words[0].start == 0.0 and words[0].end == pytest.approx(0.4)
    # The second word overlaps the first once converted; it gets pushed clear.
    assert words[1].start == pytest.approx(0.5)
    assert words[1].end == pytest.approx(0.75)
    assert words[2].start == pytest.approx(8.0)
    for previous, current in zip(words, words[1:], strict=False):
        assert current.start >= previous.end


def test_words_from_boundaries_clips_to_the_measured_duration():
    """A word straddling the end of the audio is truncated to it."""
    events = [
        {"type": "WordBoundary", "offset": 0, "duration": 10_000_000, "text": "long"},
        {"type": "WordBoundary", "offset": 20_000_000, "duration": 10_000_000, "text": "straddles"},
    ]
    words = edge_mod.words_from_boundaries(events, limit=2.5)
    assert [w.text for w in words] == ["long", "straddles"]
    assert all(w.end <= 2.5 + 1e-9 for w in words)
    assert words[-1].start == pytest.approx(2.0)
    assert words[-1].end == pytest.approx(2.5)


def test_words_from_boundaries_drops_words_past_the_audio_rather_than_zero_length_ones():
    """A zero-width caption is worse than no caption: the grouper assumes end > start."""
    events = [
        {"type": "WordBoundary", "offset": 0, "duration": 10_000_000, "text": "long"},
        {"type": "WordBoundary", "offset": 30_000_000, "duration": 10_000_000, "text": "over"},
    ]
    words = edge_mod.words_from_boundaries(events, limit=2.5)
    assert [w.text for w in words] == ["long"]
    assert all(w.end > w.start for w in words)

    # A word starting exactly at the end of the audio has no room either.
    at_limit = [{"type": "WordBoundary", "offset": 25_000_000, "duration": 10_000_000, "text": "edge"}]
    assert edge_mod.words_from_boundaries(at_limit, limit=2.5) == []


def test_words_from_boundaries_ignores_junk():
    assert edge_mod.words_from_boundaries([]) == []
    assert edge_mod.words_from_boundaries([{"type": "audio"}, {"text": "   "}, "nonsense"]) == []


def test_run_async_without_a_loop():
    async def answer() -> int:
        await asyncio.sleep(0)
        return 42

    assert edge_mod.run_async(answer) == 42


def test_run_async_inside_a_running_loop():
    """A notebook or async web handler must not blow up on asyncio.run()."""

    async def answer() -> int:
        await asyncio.sleep(0)
        return 7

    async def outer() -> int:
        return edge_mod.run_async(answer)

    assert asyncio.run(outer()) == 7


def test_edge_voice_id_resolution():
    provider = EdgeTTS(settings=_online_settings())
    assert provider.name == "edge"
    assert provider.voice_id(VoiceSpec(voice_id="narrator_deep")) == "en-US-GuyNeural"
    assert provider.voice_id(VoiceSpec(voice_id="en-AU-NatashaNeural")) == "en-AU-NatashaNeural"
    assert provider.voice_id(VoiceSpec()) == edge_mod.DEFAULT_EDGE_VOICE
    assert provider.voice_id(None) == edge_mod.DEFAULT_EDGE_VOICE


def test_edge_is_unavailable_offline():
    settings = Settings()
    settings.offline = True
    provider = EdgeTTS(settings=settings)
    assert provider.available() is False
    with pytest.raises(TTSError) as excinfo:
        provider.synthesize("hello", Path("unused.wav"), voice=VoiceSpec())
    assert "offline" in str(excinfo.value)


def test_edge_refuses_empty_text():
    with pytest.raises(TTSError):
        EdgeTTS(settings=_online_settings()).synthesize("  ", Path("unused.wav"), voice=VoiceSpec())


def test_edge_raises_missing_dependency_when_the_library_is_absent(monkeypatch: pytest.MonkeyPatch):
    """Rule 2: a missing optional extra must name its pip install line."""
    monkeypatch.setitem(sys.modules, "edge_tts", None)
    with pytest.raises(MissingDependency) as excinfo:
        EdgeTTS(settings=_online_settings()).synthesize("hello", Path("unused.wav"), voice=VoiceSpec())
    assert "edge-tts" in str(excinfo.value)
    assert "pip install" in str(excinfo.value)


def test_edge_builds_the_communicate_call_from_the_voice_spec():
    """The prosody strings and the WordBoundary opt-in, without a socket."""
    captured: dict[str, object] = {}

    class FakeCommunicate:
        def __init__(self, text, voice, *, rate="+0%", pitch="+0Hz", boundary="SentenceBoundary", **kw):
            captured.update(text=text, voice=voice, rate=rate, pitch=pitch, boundary=boundary)

    fake = type("FakeEdgeTTS", (), {"Communicate": FakeCommunicate})
    provider = EdgeTTS(settings=_online_settings())
    provider._communicate(fake, "hello there", VoiceSpec(voice_id="british_male", rate=1.15,
                                                        pitch_semitones=2.0))
    assert captured == {
        "text": "hello there",
        "voice": "en-GB-RyanNeural",
        "rate": "+15%",
        "pitch": "+24Hz",
        "boundary": "WordBoundary",
    }


def test_edge_omits_the_boundary_kwarg_on_older_libraries():
    """edge-tts 6.x has no ``boundary`` parameter; passing one would TypeError."""
    captured: dict[str, object] = {}

    class OldCommunicate:
        def __init__(self, text, voice, *, rate="+0%", pitch="+0Hz"):
            captured.update(text=text, voice=voice, rate=rate, pitch=pitch)

    fake = type("FakeEdgeTTS", (), {"Communicate": OldCommunicate})
    EdgeTTS(settings=_online_settings())._communicate(fake, "hi", VoiceSpec())
    assert "boundary" not in captured
    assert captured["rate"] == "+0%"


@pytest.mark.needs_ffmpeg
def test_edge_synthesis_transcodes_the_stream_and_keeps_the_word_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The whole edge path except the socket: stream -> mp3 -> wav -> words."""
    audio = _speech_bytes(tmp_path, seconds=1.5)
    half = len(audio) // 2
    seen = _install_fake_edge(monkeypatch, [
        {"type": "WordBoundary", "offset": 0, "duration": 5_000_000, "text": "hello"},
        {"type": "audio", "data": audio[:half]},
        {"type": "WordBoundary", "offset": 6_000_000, "duration": 5_000_000, "text": "world"},
        {"type": "audio", "data": audio[half:]},
        {"type": "SentenceBoundary", "offset": 0, "duration": 0, "text": "hello world"},
    ])

    out = tmp_path / "vo" / "edge.wav"
    result = EdgeTTS(settings=_online_settings()).synthesize(
        "hello world", out, voice=VoiceSpec(voice_id="narrator_deep", rate=1.15)
    )

    assert seen == [{"text": "hello world", "voice": "en-US-GuyNeural",
                     "rate": "+15%", "pitch": "+0Hz", "boundary": "WordBoundary"}]
    assert out.exists() and out.stat().st_size > 0
    info = ff.probe(out)
    assert info.has_audio and not info.has_video
    assert info.duration == pytest.approx(1.5, abs=0.05), "the received audio must reach the file"
    assert result.duration == pytest.approx(info.duration, abs=EPS), "duration comes from ffprobe"
    assert [w.text for w in result.words] == ["hello", "world"]
    assert result.words[0].start == 0.0 and result.words[0].end == pytest.approx(0.5)
    assert result.words[1].start == pytest.approx(0.6)
    assert result.words[-1].end <= result.duration + EPS
    assert result.text == "hello world"
    assert result.voice.voice_id == "narrator_deep", "the spec is recorded, not the native id"


@pytest.mark.needs_ffmpeg
def test_edge_synthesis_reports_no_words_when_the_service_gives_no_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """``words is None`` is the documented signal to force-align instead."""
    audio = _speech_bytes(tmp_path, seconds=0.6)
    _install_fake_edge(monkeypatch, [{"type": "audio", "data": audio}])

    result = EdgeTTS(settings=_online_settings()).synthesize(
        "hello world", tmp_path / "silent.wav", voice=VoiceSpec()
    )
    assert result.words is None
    assert result.duration > 0


def test_edge_synthesis_raises_when_the_stream_carries_no_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _install_fake_edge(monkeypatch, [
        {"type": "WordBoundary", "offset": 0, "duration": 100, "text": "hello"},
    ])
    with pytest.raises(TTSError) as excinfo:
        EdgeTTS(settings=_online_settings()).synthesize("hello", tmp_path / "x.wav", voice=VoiceSpec())
    assert "no audio" in str(excinfo.value)


def test_edge_wraps_library_failures_in_ttserror(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A transport blow-up must reach the caller as our own error type."""

    class Exploding:
        def __init__(self, *args, **kwargs):
            pass

        async def stream(self):
            raise RuntimeError("websocket closed: 1006")
            yield {}  # pragma: no cover - unreachable, makes this an async generator

    module = types.ModuleType("edge_tts")
    module.Communicate = Exploding
    monkeypatch.setitem(sys.modules, "edge_tts", module)

    with pytest.raises(TTSError) as excinfo:
        EdgeTTS(settings=_online_settings()).synthesize("hello", tmp_path / "x.wav", voice=VoiceSpec())
    assert "1006" in str(excinfo.value), "the underlying reason must survive the wrapping"


@pytest.mark.skip(reason="edge-tts needs network access; no egress in CI")
def test_edge_live_synthesis():  # pragma: no cover - documentation of the live path
    raise AssertionError("never runs")


# --------------------------------------------------------------------------- #
# elevenlabs backend: request shape and refusal paths
# --------------------------------------------------------------------------- #

def test_eleven_build_request_shape():
    request = eleven_mod.build_request("hello world", "voice-123", "sk-secret")
    assert request.method == "POST"
    assert request.full_url == f"{eleven_mod.API_ROOT}/voice-123"
    headers = {k.lower(): v for k, v in request.headers.items()}
    assert headers["Xi-api-key".lower()] == "sk-secret"
    assert headers["content-type"] == "application/json"
    assert headers["accept"] == "audio/mpeg"

    payload = json.loads(request.data.decode("utf-8"))
    assert payload["text"] == "hello world"
    assert payload["model_id"] == eleven_mod.DEFAULT_MODEL
    assert set(payload["voice_settings"]) == {"stability", "similarity_boost", "style"}


def test_eleven_build_request_validates_its_inputs():
    with pytest.raises(TTSError):
        eleven_mod.build_request("", "voice", "key")
    with pytest.raises(TTSError):
        eleven_mod.build_request("hi", "", "key")
    with pytest.raises(TTSError):
        eleven_mod.build_request("hi", "voice", "")


def test_eleven_availability_depends_on_the_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    assert ElevenLabsTTS(settings=_online_settings()).available() is False
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    assert ElevenLabsTTS(settings=_online_settings()).available() is True

    offline = Settings()
    offline.offline = True
    assert ElevenLabsTTS(settings=offline).available() is False


def test_eleven_refuses_to_run_offline(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    settings = Settings()
    settings.offline = True
    with pytest.raises(TTSError) as excinfo:
        ElevenLabsTTS(settings=settings).synthesize("hi", Path("unused.wav"), voice=VoiceSpec())
    assert "offline" in str(excinfo.value)


def test_eleven_reports_a_missing_key(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    with pytest.raises(TTSError) as excinfo:
        ElevenLabsTTS(settings=_online_settings()).synthesize("hi", Path("u.wav"), voice=VoiceSpec())
    assert "ELEVENLABS_API_KEY" in str(excinfo.value)


def test_eleven_surfaces_the_response_body_on_an_http_error(monkeypatch: pytest.MonkeyPatch):
    import urllib.error

    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")

    def boom(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url, 401, "Unauthorized", {},
            _FakeBody(b'{"detail":{"status":"invalid_api_key"}}'),
        )

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", boom)
    with pytest.raises(TTSError) as excinfo:
        ElevenLabsTTS(settings=_online_settings()).synthesize("hi", Path("u.wav"), voice=VoiceSpec())
    message = str(excinfo.value)
    assert "401" in message
    assert "invalid_api_key" in message, "the response body must reach the caller"


def test_eleven_voice_id_resolution():
    provider = ElevenLabsTTS(settings=_online_settings())
    assert provider.voice_id(VoiceSpec(voice_id="narrator_deep")) == "pNInz6obpgDQGcFmaJgB"
    assert provider.voice_id(VoiceSpec(voice_id="custom-clone-id")) == "custom-clone-id"
    # Catalogue voice with no ElevenLabs mapping -> the documented default.
    assert provider.voice_id(VoiceSpec(voice_id="newsroom")) == eleven_mod.DEFAULT_VOICE_ID
    assert provider.voice_id(VoiceSpec()) == eleven_mod.DEFAULT_VOICE_ID


@pytest.mark.parametrize("rate, expected", [
    (1.0, []),
    (1.0004, []),
    (0.0, []),
    (1.5, ["atempo=1.5000"]),
    (0.75, ["atempo=0.7500"]),
    (3.0, ["atempo=2", "atempo=1.5000"]),
    (0.25, ["atempo=0.5", "atempo=0.5000"]),
])
def test_eleven_tempo_filters(rate, expected):
    assert eleven_mod.tempo_filters(rate) == expected


def test_eleven_tempo_filters_multiply_back_to_the_rate():
    for rate in (0.4, 0.8, 1.3, 2.6, 3.9):
        chain = eleven_mod.tempo_filters(rate)
        product = 1.0
        for item in chain:
            product *= float(item.split("=")[1])
        assert product == pytest.approx(rate, rel=1e-3)


@pytest.mark.needs_ffmpeg
def test_eleven_synthesis_writes_a_wav_and_leaves_words_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The whole ElevenLabs path except the socket."""
    audio = _speech_bytes(tmp_path, seconds=1.5)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    sent: dict[str, object] = {}

    def fake_urlopen(request, timeout=None):
        sent.update(url=request.full_url, timeout=timeout,
                    key=request.headers["Xi-api-key"],
                    payload=json.loads(request.data.decode("utf-8")))
        return _FakeResponse(audio)

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", fake_urlopen)
    out = tmp_path / "vo" / "eleven.wav"
    result = ElevenLabsTTS(settings=_online_settings()).synthesize(
        "hello world", out, voice=VoiceSpec(voice_id="narrator_deep")
    )

    assert sent["url"] == f"{eleven_mod.API_ROOT}/pNInz6obpgDQGcFmaJgB"
    assert sent["key"] == "sk-test-not-a-real-key"
    assert sent["payload"]["text"] == "hello world"
    assert sent["timeout"] == eleven_mod.DEFAULT_TIMEOUT
    assert out.exists() and ff.probe(out).has_audio
    assert result.words is None, "the API reports no boundaries; callers force-align"
    assert result.duration == pytest.approx(1.5, abs=0.05)
    assert result.text == "hello world"


@pytest.mark.needs_ffmpeg
def test_eleven_applies_the_rate_as_a_tempo_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """``VoiceSpec.rate`` has no API field, so it must survive the transcode."""
    audio = _speech_bytes(tmp_path, seconds=1.5)
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen",
                        lambda request, timeout=None: _FakeResponse(audio))

    provider = ElevenLabsTTS(settings=_online_settings())
    plain = provider.synthesize("hi", tmp_path / "plain.wav", voice=VoiceSpec())
    fast = provider.synthesize("hi", tmp_path / "fast.wav", voice=VoiceSpec(rate=2.0))
    assert fast.duration == pytest.approx(plain.duration / 2, rel=0.05)


def test_eleven_rejects_a_non_200_response(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen",
                        lambda request, timeout=None: _FakeResponse(b"rate limited", status=429))
    with pytest.raises(TTSError) as excinfo:
        ElevenLabsTTS(settings=_online_settings()).synthesize("hi", tmp_path / "x.wav", voice=VoiceSpec())
    assert "429" in str(excinfo.value) and "rate limited" in str(excinfo.value)


def test_eleven_rejects_an_empty_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """An empty 200 must not become a zero-byte "audio" file downstream."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen",
                        lambda request, timeout=None: _FakeResponse(b""))
    with pytest.raises(TTSError) as excinfo:
        ElevenLabsTTS(settings=_online_settings()).synthesize("hi", tmp_path / "x.wav", voice=VoiceSpec())
    assert "empty" in str(excinfo.value)
    assert not (tmp_path / "x.wav").exists()


def test_eleven_wraps_transport_failures(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import urllib.error

    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")

    def unreachable(request, timeout=None):
        raise urllib.error.URLError("name or service not known")

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", unreachable)
    with pytest.raises(TTSError) as excinfo:
        ElevenLabsTTS(settings=_online_settings()).synthesize("hi", tmp_path / "x.wav", voice=VoiceSpec())
    assert "name or service not known" in str(excinfo.value)

    def timed_out(request, timeout=None):
        raise TimeoutError("timed out")

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", timed_out)
    with pytest.raises(TTSError):
        ElevenLabsTTS(settings=_online_settings()).synthesize("hi", tmp_path / "x.wav", voice=VoiceSpec())


@pytest.mark.skip(reason="ElevenLabs needs a paid key and network access; no egress in CI")
def test_eleven_live_synthesis():  # pragma: no cover - documentation of the live path
    raise AssertionError("never runs")


class _FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns."""

    def __init__(self, payload: bytes, *, status: int = 200) -> None:
        self._payload = payload
        self.status = status
        self.closed = False

    def read(self, *_args) -> bytes:
        return self._payload

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_exc) -> bool:
        self.close()
        return False


class _FakeBody:
    """Minimal file-like stand-in for ``HTTPError``'s response body."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self, *_args) -> bytes:
        return self._payload

    def close(self) -> None:
        return None


# --------------------------------------------------------------------------- #
# runtime fallback: a backend that dies part way through a narration
# --------------------------------------------------------------------------- #

class _Flaky:
    """A backend that speaks a distinctive tone until it drops the connection.

    Its audio is deliberately nothing like the offline backend's -- a long tone
    instead of a short silence -- so a test can tell, from the files on disk,
    *which* backend spoke each line.
    """

    name = "flaky"

    def __init__(self, *, fail_on: int, seconds: float = 3.0) -> None:
        self.fail_on = fail_on          # 1-based line number that blows up
        self.seconds = seconds
        self.calls: list[str] = []

    def available(self) -> bool:
        return True

    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        self.calls.append(text)
        if len(self.calls) >= self.fail_on:
            raise TTSError("the service dropped the connection")
        ff.make_tone(self.seconds, Path(out_path))
        return TTSResult(audio_path=Path(out_path), duration=self.seconds,
                         words=[Word(text, 0.0, self.seconds, 1.0)], voice=voice, text=text)


@pytest.mark.needs_ffmpeg
def test_a_backend_that_dies_mid_narration_does_not_kill_the_render(tmp_path: Path):
    """The bug: one blip on line 3 used to abort a render with a live offline path."""
    flaky = _Flaky(fail_on=3)
    lines = ["First line here.", "Second line here.", "Third line here.", "Fourth line here."]
    out_dir = tmp_path / "vo"

    results = synthesize_lines(lines, out_dir, voice=VoiceSpec(), provider=flaky, gap=0.1)

    assert isinstance(results, NarrationResults)
    assert len(results) == len(lines), "every line must still be spoken"
    assert results.provider == "offline", "the reported backend must be the one that ran"
    assert results.attempted == ("flaky", "offline")
    assert results.fell_back is True
    assert flaky.calls == lines[:3], "the failed backend must not be asked for the rest"


@pytest.mark.needs_ffmpeg
def test_the_fallback_respeaks_every_line_so_the_voice_stays_consistent(tmp_path: Path):
    """Lines 1-2 were already on disk in the flaky voice; they must be redone."""
    flaky = _Flaky(fail_on=3, seconds=3.0)
    lines = ["First line here.", "Second line here.", "Third line here."]
    out_dir = tmp_path / "vo"

    results = synthesize_lines(lines, out_dir, voice=VoiceSpec(), provider=flaky, gap=0.1)

    for index, line in enumerate(lines):
        path = out_dir / f"line_{index:03d}.wav"
        assert path.exists()
        measured = ff.probe(path).duration
        assert abs(measured - flaky.seconds) > 0.5, (
            f"line {index} is still the flaky backend's audio -- half the narration "
            "would be in a different voice"
        )
        assert abs(results[index].duration - estimate_duration(line)) < 0.25
        assert abs(measured - results[index].duration) < 0.1
    assert all(r.words for r in results), "the fallback's word timings must reach the caller"

    # The rebasing contract still holds across the re-synthesis.
    offset = 0.0
    for result in results:
        assert result.words[0].start >= offset - EPS
        offset += result.duration + 0.1


@pytest.mark.needs_ffmpeg
def test_a_backend_that_fails_on_the_very_first_line_falls_back_too(tmp_path: Path):
    results = synthesize_lines(["only line"], tmp_path / "vo", voice=VoiceSpec(),
                               provider=_Flaky(fail_on=1))
    assert results.provider == "offline"
    assert results[0].words


@pytest.mark.needs_ffmpeg
def test_fallback_false_still_raises(tmp_path: Path):
    """Strictness is a caller's right: no silent substitution when asked not to."""
    flaky = _Flaky(fail_on=2)
    with pytest.raises(TTSError, match="dropped the connection"):
        synthesize_lines(["a line", "another line"], tmp_path / "vo", voice=VoiceSpec(),
                         provider=flaky, fallback=False)
    assert flaky.calls == ["a line", "another line"]
    assert not (tmp_path / "vo" / "line_001.wav").exists()


@pytest.mark.needs_ffmpeg
def test_a_chain_that_fails_all_the_way_down_raises_the_last_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    def explode(self, text, out_path, *, voice):
        raise TTSError("ffmpeg is gone too")

    monkeypatch.setattr(OfflineTTS, "synthesize", explode)
    with pytest.raises(TTSError, match="ffmpeg is gone too"):
        synthesize_lines(["a line"], tmp_path / "vo", voice=VoiceSpec(), provider=_Flaky(fail_on=1))


@pytest.mark.needs_ffmpeg
def test_a_narration_that_works_reports_its_own_backend(tmp_path: Path):
    results = synthesize_lines(["hello there"], tmp_path / "vo", voice=VoiceSpec(), provider="offline")
    assert results.provider == "offline"
    assert results.attempted == ("offline",)
    assert results.fell_back is False
    assert isinstance(results, list) and len(results) == 1


def test_an_empty_narration_reports_nothing(tmp_path: Path):
    results = synthesize_lines([], tmp_path / "vo", voice=VoiceSpec(), provider="offline")
    assert results == []
    assert results.provider == "" and results.attempted == ()


@pytest.mark.needs_ffmpeg
def test_fallback_chain_always_ends_at_the_offline_backend():
    """Whatever the starting point, the floor is the backend that cannot fail."""
    assert [p.name for p in fallback_chain("offline")] == ["offline"]

    chain = [p.name for p in fallback_chain(_Flaky(fail_on=99))]
    assert chain[0] == "flaky" and chain[-1] == "offline"
    assert len(set(chain)) == len(chain), "a backend must not be tried twice"


@pytest.mark.needs_ffmpeg
def test_fallback_chain_offers_the_other_available_backends(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(edge_mod, "edge_available", lambda: True)
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    names = [p.name for p in fallback_chain("elevenlabs", settings=_online_settings())]
    assert names == ["elevenlabs", "edge", "offline"], (
        "an explicitly named backend leads; only backends that could run follow it"
    )


@pytest.mark.needs_ffmpeg
def test_fallback_chain_skips_the_network_backends_when_offline():
    settings = Settings()
    settings.offline = True
    assert [p.name for p in fallback_chain("auto", settings=settings)] == ["offline"]


# --------------------------------------------------------------------------- #
# available() vs usable(): what a doctor must ask
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _forget_probe_verdicts():
    """Probe verdicts are cached for the process; tests must not inherit them."""
    reset_usable_cache()
    yield
    reset_usable_cache()


def _install_probeable_edge(monkeypatch: pytest.MonkeyPatch, list_voices) -> None:
    """A fake ``edge_tts`` whose only interesting member is ``list_voices``."""
    module = types.ModuleType("edge_tts")
    module.list_voices = list_voices
    monkeypatch.setitem(sys.modules, "edge_tts", module)
    monkeypatch.setattr(edge_mod, "edge_available", lambda: True)


@pytest.mark.needs_ffmpeg
def test_edge_available_says_yes_where_edge_can_never_speak(monkeypatch: pytest.MonkeyPatch):
    """The bug: ``doctor`` printed OK for edge on a box with no egress."""
    asked: list[int] = []

    async def dead_service():
        asked.append(1)
        raise OSError("Cannot connect to host speech.platform.bing.com:443")

    _install_probeable_edge(monkeypatch, dead_service)
    provider = EdgeTTS(settings=_online_settings())

    assert provider.available() is True, "the import check cannot see the network"
    assert provider.usable(timeout=1.0) is False, "the capability probe must"
    assert asked == [1], "usable() has to actually ask the service"


@pytest.mark.needs_ffmpeg
def test_edge_usable_is_true_when_the_service_answers_and_is_cached_per_process(
    monkeypatch: pytest.MonkeyPatch
):
    asked: list[int] = []

    async def live_service():
        asked.append(1)
        return [{"Name": "en-US-GuyNeural"}]

    _install_probeable_edge(monkeypatch, live_service)
    settings = _online_settings()

    assert EdgeTTS(settings=settings).usable(timeout=1.0) is True
    assert EdgeTTS(settings=settings).usable(timeout=1.0) is True, "a fresh instance reuses the verdict"
    assert len(asked) == 1, "the service must be probed once per process, not once per call"

    assert EdgeTTS(settings=settings).usable(timeout=1.0, refresh=True) is True
    assert len(asked) == 2, "refresh=True must re-probe"

    reset_usable_cache()
    assert EdgeTTS(settings=settings).usable(timeout=1.0) is True
    assert len(asked) == 3


@pytest.mark.needs_ffmpeg
def test_edge_usable_times_out_rather_than_hanging(monkeypatch: pytest.MonkeyPatch):
    async def never_answers():
        await asyncio.sleep(30)
        return ["never"]

    _install_probeable_edge(monkeypatch, never_answers)
    start = time.perf_counter()
    assert EdgeTTS(settings=_online_settings()).usable(timeout=0.2) is False
    assert time.perf_counter() - start < 10.0, "a diagnostic that hangs is a diagnostic nobody runs"


@pytest.mark.needs_ffmpeg
def test_edge_usable_survives_a_probe_that_blocks_the_event_loop(monkeypatch: pytest.MonkeyPatch):
    """A synchronous wedge (DNS, TLS) never yields, so the bound must be outside the loop."""

    def blocks_hard():
        time.sleep(30)
        return ["never"]

    _install_probeable_edge(monkeypatch, blocks_hard)
    start = time.perf_counter()
    assert EdgeTTS(settings=_online_settings()).usable(timeout=0.2) is False
    assert time.perf_counter() - start < 10.0


def test_edge_usable_is_false_offline_without_probing(monkeypatch: pytest.MonkeyPatch):
    def must_not_run():
        raise AssertionError("offline mode must never touch the network")

    _install_probeable_edge(monkeypatch, must_not_run)
    settings = Settings()
    settings.offline = True
    assert EdgeTTS(settings=settings).usable(timeout=1.0) is False


@pytest.mark.needs_ffmpeg
def test_eleven_usable_checks_the_key_against_the_api(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-test-not-a-real-key")
    seen: list[tuple[str, str | None, float | None]] = []

    def fake_urlopen(request, timeout=None):
        seen.append((request.full_url, request.get_header("Xi-api-key"), timeout))
        return _FakeResponse(b'{"subscription": {"tier": "free"}}')

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", fake_urlopen)
    provider = ElevenLabsTTS(settings=_online_settings())

    assert provider.available() is True
    assert provider.usable(timeout=2.0) is True
    assert seen[0][0] == eleven_mod.PROBE_URL, "the probe must not spend a synthesis"
    assert seen[0][1] == "sk-test-not-a-real-key"
    assert seen[0][2] == 2.0, "the probe has to carry its timeout down to the socket"

    assert provider.usable(timeout=2.0) is True
    assert len(seen) == 1, "the verdict is cached for the process"


@pytest.mark.needs_ffmpeg
def test_eleven_usable_is_false_for_a_key_the_api_rejects(monkeypatch: pytest.MonkeyPatch):
    import urllib.error

    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-revoked-yesterday")

    def rejected(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, _FakeBody(b"nope"))

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", rejected)
    provider = ElevenLabsTTS(settings=_online_settings())
    assert provider.available() is True, "a revoked key is still a non-empty string"
    assert provider.usable(timeout=1.0) is False


@pytest.mark.needs_ffmpeg
def test_eleven_usable_is_cached_per_key_not_globally(monkeypatch: pytest.MonkeyPatch):
    import urllib.error

    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-good")

    def by_key(request, timeout=None):
        if request.get_header("Xi-api-key") == "sk-good":
            return _FakeResponse(b"{}")
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, _FakeBody(b"nope"))

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", by_key)
    assert ElevenLabsTTS(settings=_online_settings()).usable(timeout=1.0) is True

    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk-bad")
    assert ElevenLabsTTS(settings=_online_settings()).usable(timeout=1.0) is False, (
        "a new key is a new question, not a cache hit"
    )


@pytest.mark.needs_ffmpeg
def test_eleven_usable_is_false_without_a_key_and_never_calls_out(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    def must_not_run(request, timeout=None):
        raise AssertionError("no key means nothing to ask")

    monkeypatch.setattr(eleven_mod.urllib.request, "urlopen", must_not_run)
    assert ElevenLabsTTS(settings=_online_settings()).usable(timeout=1.0) is False


@pytest.mark.needs_ffmpeg
def test_the_offline_backend_is_always_usable():
    assert OfflineTTS().usable() is True
    assert provider_usable("offline") is True


@pytest.mark.needs_ffmpeg
def test_provider_usable_degrades_to_available_for_a_backend_without_a_probe():
    """A third-party or test-double provider is still a valid provider."""
    assert provider_usable(_Flaky(fail_on=99)) is True

    class Unavailable(_Flaky):
        def available(self) -> bool:
            return False

    assert provider_usable(Unavailable(fail_on=99)) is False


def test_provider_usable_never_raises():
    class Angry:
        name = "angry"

        def available(self) -> bool:
            raise RuntimeError("everything is on fire")

        def usable(self, *, timeout=None, refresh=False) -> bool:
            raise RuntimeError("this too")

        def synthesize(self, text, out_path, *, voice):
            raise RuntimeError("and this")

    assert provider_usable(Angry()) is False
    assert provider_usable("no-such-backend") is False


# --------------------------------------------------------------------------- #
# run_bounded, the thing that keeps a probe from wedging the caller
# --------------------------------------------------------------------------- #

def test_run_bounded_returns_the_default_when_the_call_wedges():
    start = time.perf_counter()
    assert run_bounded(lambda: time.sleep(30) or True, 0.15, default=False) is False
    assert time.perf_counter() - start < 5.0


def test_run_bounded_returns_the_value_when_the_call_finishes():
    assert run_bounded(lambda: "done", 5.0, default="timeout") == "done"


def test_run_bounded_propagates_failures():
    def boom():
        raise ValueError("probe blew up")

    with pytest.raises(ValueError, match="probe blew up"):
        run_bounded(boom, 5.0, default=None)
