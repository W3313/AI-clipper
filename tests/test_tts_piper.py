"""Tests for :mod:`aiclipper.tts.piper`.

Piper is *not* installed here and these tests never require it.  The binary is
faked with a small Python script on a temporary PATH that records the argv and
stdin it was handed and writes a real wav through :mod:`aiclipper.ffmpeg`, so
every assertion about the result -- the file exists, ``ffprobe`` agrees with
``TTSResult.duration`` -- is made against genuine audio rather than a mock.  The
in-process path is exercised with a fake ``piper`` module injected into
``sys.modules``.

Nothing here opens a socket, and the whole file runs with ``AICLIP_OFFLINE=1``
(the conftest default) on purpose: a local backend that stopped working offline
would be missing the point.
"""

from __future__ import annotations

import importlib.machinery
import json
import os
import re
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from pathlib import Path

import pytest

from aiclipper import ffmpeg as ff
from aiclipper.config import Settings, reset_settings
from aiclipper.errors import MissingDependency, TTSError
from aiclipper.models import TTSResult, VoiceSpec
from aiclipper.tts import VOICES, find_voice, find_voice_entry, get_provider, list_voices
from aiclipper.tts import piper as piper_mod
from aiclipper.tts.base import (
    FALLBACK_ORDER,
    PROVIDER_ALIASES,
    TTSProvider,
    fallback_chain,
    provider_usable,
    reset_usable_cache,
    synthesize_lines,
)
from aiclipper.tts.piper import (
    MODEL_SUFFIX,
    PROBE_TEXT,
    VOICE_DOWNLOADS,
    PiperTTS,
    installed_voices,
    length_scale,
    resolve_model,
    speak_args,
)
from aiclipper.tts.voices import KNOWN_PROVIDERS

EPS = 1e-3

#: A fake ``piper``: it logs how it was called, then writes a real wav whose
#: length follows the text and the length scale, so a test can prove that what
#: came back was measured rather than guessed.  The heavy import happens after
#: the optional sleep so the timeout test kills a child that really is wedged.
_STUB = '''#!{python}
import json
import os
import sys
import time
from pathlib import Path

argv = sys.argv[1:]
stdin = sys.stdin.read()

log = Path(os.environ["PIPER_STUB_LOG"])
calls = json.loads(log.read_text()) if log.exists() else []
calls.append({{"argv": argv, "stdin": stdin}})
log.write_text(json.dumps(calls))

time.sleep(float(os.environ.get("PIPER_STUB_SLEEP") or 0))

failure = os.environ.get("PIPER_STUB_FAIL")
if failure:
    print(failure, file=sys.stderr)
    raise SystemExit(3)

if os.environ.get("PIPER_STUB_NOWRITE"):
    raise SystemExit(0)

from aiclipper import ffmpeg as ff

out = argv[argv.index("--output_file") + 1]
scale = float(argv[argv.index("--length_scale") + 1]) if "--length_scale" in argv else 1.0
seconds = max(1, len(stdin.split())) / 2.6 * scale
ff.make_silence(seconds, out)
'''


@dataclass
class FakePiper:
    """The fake install: where the binary is, where the voices live, what ran."""

    bin_dir: Path
    voice_dir: Path
    log: Path
    monkeypatch: pytest.MonkeyPatch

    @property
    def binary(self) -> Path:
        return self.bin_dir / "piper"

    def calls(self) -> list[dict[str, object]]:
        if not self.log.exists():
            return []
        return json.loads(self.log.read_text())

    def add_voice(self, name: str = "en_US-lessac-medium") -> Path:
        """Install a voice model (an empty file: the fake never reads it)."""
        self.voice_dir.mkdir(parents=True, exist_ok=True)
        model = self.voice_dir / f"{name}{MODEL_SUFFIX}"
        model.write_bytes(b"")
        (self.voice_dir / f"{name}{MODEL_SUFFIX}.json").write_text("{}", encoding="utf-8")
        return model

    def unplug(self) -> None:
        """Take the binary off PATH again."""
        self.monkeypatch.setenv("PATH", self.original_path)

    def settings(self, **overrides: object) -> Settings:
        s = Settings()
        for key, value in overrides.items():
            setattr(s, key, value)
        return s

    original_path: str = ""


@pytest.fixture(autouse=True)
def _forget_probe_verdicts():
    """Probe verdicts are cached for the process; tests must not inherit them."""
    reset_usable_cache()
    yield
    reset_usable_cache()


@pytest.fixture
def fake_piper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakePiper:
    """A fake ``piper`` executable on PATH plus an empty voice directory."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "piper"
    script.write_text(_STUB.format(python=sys.executable), encoding="utf-8")
    script.chmod(0o755)

    voice_dir = tmp_path / "voices"
    voice_dir.mkdir()
    log = tmp_path / "piper-calls.json"

    original = os.environ.get("PATH", "")
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{original}")
    monkeypatch.setenv("PIPER_STUB_LOG", str(log))
    monkeypatch.setenv("AICLIP_PIPER_VOICES", str(voice_dir))
    monkeypatch.delenv("AICLIP_PIPER_VOICE", raising=False)
    reset_settings()
    return FakePiper(
        bin_dir=bin_dir, voice_dir=voice_dir, log=log, monkeypatch=monkeypatch, original_path=original
    )


def _install_fake_module(
    monkeypatch: pytest.MonkeyPatch,
    recorder: dict[str, object],
    *,
    seconds: float = 1.0,
) -> types.ModuleType:
    """A fake ``piper`` package with the older ``synthesize(text, wav, length_scale=)`` API."""
    module = types.ModuleType("piper")
    module.__spec__ = importlib.machinery.ModuleSpec("piper", None)

    class _Voice:
        @staticmethod
        def load(path: str) -> _Voice:
            recorder["model"] = path
            return _Voice()

        def synthesize(self, text: str, wav, length_scale: float = 1.0) -> None:
            recorder["text"] = text
            recorder["length_scale"] = length_scale
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(22050)
            wav.writeframes(b"\x00" * int(22050 * 2 * seconds))

    module.PiperVoice = _Voice
    monkeypatch.setitem(sys.modules, "piper", module)
    return module


class _Broken:
    """A backend that is willing but always fails -- the thing piper must catch."""

    name = "broken"

    def available(self) -> bool:
        return True

    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        raise TTSError("the service went away mid-narration")


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #

def test_importing_the_module_runs_nothing(fake_piper: FakePiper):
    """Rule 3: no subprocess and no optional import at module scope."""
    code = (
        "import sys; import aiclipper.tts.piper as p;"
        "assert 'piper' not in sys.modules or sys.modules['piper'] is None;"
        "print(p.PiperTTS.name)"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "piper"
    assert fake_piper.calls() == [], "importing the module must not execute piper"


def test_the_backend_satisfies_the_provider_protocol():
    provider = PiperTTS()
    assert isinstance(provider, TTSProvider)
    assert provider.name == "piper"
    assert isinstance(provider.available(), bool)


# --------------------------------------------------------------------------- #
# rate -> --length_scale
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("rate, expected", [
    (1.0, 1.0),
    (1.25, 0.8),
    (0.8, 1.25),
    (2.0, 0.5),
    (0.5, 2.0),
    (1.15, 0.8696),
])
def test_rate_is_the_inverse_of_the_length_scale(rate: float, expected: float):
    assert length_scale(rate) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("rate, expected", [
    (4.0, 0.5),     # clamped to the 2.0x ceiling
    (100.0, 0.5),
    (0.1, 2.0),     # clamped to the 0.5x floor
    (0.0, 1.0),     # nonsense -> neutral rather than a division by zero
    (-2.0, 1.0),
    (None, 1.0),
    ("fast", 1.0),
])
def test_length_scale_clamps_nonsense_into_a_speakable_band(rate, expected: float):
    assert length_scale(rate) == pytest.approx(expected, abs=1e-4)


def test_length_scale_round_trips():
    """The conversion is its own inverse inside the band."""
    for rate in (0.5, 0.75, 1.0, 1.33, 2.0):
        assert length_scale(length_scale(rate)) == pytest.approx(rate, abs=1e-3)


def test_speak_args_omits_a_neutral_length_scale():
    """Each voice ships its own tuned default; 1.0 must not overwrite it."""
    assert speak_args("v.onnx", "o.wav") == ["--model", "v.onnx", "--output_file", "o.wav"]
    assert speak_args("v.onnx", "o.wav", rate=1.0) == ["--model", "v.onnx", "--output_file", "o.wav"]
    assert speak_args("v.onnx", "o.wav", rate=1.25) == [
        "--model", "v.onnx", "--output_file", "o.wav", "--length_scale", "0.8",
    ]


# --------------------------------------------------------------------------- #
# the command it actually runs
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_synthesis_runs_the_documented_command_with_the_text_on_stdin(
    fake_piper: FakePiper, tmp_path: Path
):
    model = fake_piper.add_voice()
    provider = PiperTTS(settings=fake_piper.settings())
    out = tmp_path / "vo" / "line.wav"

    provider.synthesize("  Hello\n  there  ", out, voice=VoiceSpec(voice_id="en_US-lessac-medium"))

    calls = fake_piper.calls()
    assert len(calls) == 1
    assert calls[0]["argv"] == ["--model", str(model), "--output_file", str(out)]
    # One utterance: a newline in the text would make piper speak two and keep
    # only the last, so it is flattened before it is sent.
    assert calls[0]["stdin"] == "Hello there\n"


@pytest.mark.needs_ffmpeg
def test_the_rate_reaches_piper_as_a_length_scale(fake_piper: FakePiper, tmp_path: Path):
    model = fake_piper.add_voice()
    provider = PiperTTS(settings=fake_piper.settings())
    out = tmp_path / "fast.wav"

    provider.synthesize("one two three", out, voice=VoiceSpec(voice_id=str(model), rate=1.25))

    argv = fake_piper.calls()[0]["argv"]
    assert argv[-2:] == ["--length_scale", "0.8"]


@pytest.mark.needs_ffmpeg
def test_empty_text_is_refused_without_running_anything(fake_piper: FakePiper, tmp_path: Path):
    fake_piper.add_voice()
    provider = PiperTTS(settings=fake_piper.settings())
    with pytest.raises(TTSError, match="empty text"):
        provider.synthesize("   ", tmp_path / "nope.wav", voice=VoiceSpec())
    assert fake_piper.calls() == []


@pytest.mark.needs_ffmpeg
def test_a_failing_binary_becomes_a_ttserror_carrying_its_stderr(
    fake_piper: FakePiper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_piper.add_voice()
    monkeypatch.setenv("PIPER_STUB_FAIL", "Error: unable to load model")
    provider = PiperTTS(settings=fake_piper.settings(piper_voice="en_US-lessac-medium"))
    with pytest.raises(TTSError) as excinfo:
        provider.synthesize("hello", tmp_path / "x.wav", voice=VoiceSpec())
    assert "exited 3" in str(excinfo.value)
    assert "unable to load model" in str(excinfo.value)


@pytest.mark.needs_ffmpeg
def test_a_binary_that_writes_nothing_is_reported_rather_than_returned(
    fake_piper: FakePiper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_piper.add_voice()
    monkeypatch.setenv("PIPER_STUB_NOWRITE", "1")
    provider = PiperTTS(settings=fake_piper.settings(piper_voice="en_US-lessac-medium"))
    with pytest.raises(TTSError, match="wrote no audio"):
        provider.synthesize("hello", tmp_path / "x.wav", voice=VoiceSpec())


def test_a_missing_install_names_the_pip_line(fake_piper: FakePiper, tmp_path: Path):
    """Rule 2: an absent optional dependency explains how to get it."""
    model = fake_piper.add_voice()
    fake_piper.unplug()
    provider = PiperTTS(settings=fake_piper.settings())
    with pytest.raises(MissingDependency) as excinfo:
        provider.synthesize("hello", tmp_path / "x.wav", voice=VoiceSpec(voice_id=str(model)))
    assert "pip install piper-tts" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# the result
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_the_result_describes_the_file_that_was_really_written(
    fake_piper: FakePiper, tmp_path: Path
):
    fake_piper.add_voice()
    provider = PiperTTS(settings=fake_piper.settings())
    out = tmp_path / "line_000.wav"
    spec = VoiceSpec(voice_id="en_US-lessac-medium", rate=1.0, pitch_semitones=2.0)

    result = provider.synthesize("four quick words here", out, voice=spec)

    assert isinstance(result, TTSResult)
    assert result.audio_path == out and out.is_file()
    assert result.duration == pytest.approx(ff.probe(out).duration, abs=EPS)
    assert result.duration > 0
    # Piper's simple interface reports no word boundaries; the engine's
    # three-tier timing fallback covers it.  See the module docstring.
    assert result.words is None
    assert result.voice == spec
    assert result.text == "four quick words here"


@pytest.mark.needs_ffmpeg
def test_synthesize_lines_drives_the_backend_and_leaves_the_timings_to_the_engine(
    fake_piper: FakePiper, tmp_path: Path
):
    fake_piper.add_voice()
    settings = fake_piper.settings()
    results = synthesize_lines(
        ["first line", "second line"],
        tmp_path / "vo",
        voice=VoiceSpec(voice_id="en_US-lessac-medium"),
        provider="piper",
        settings=settings,
    )
    assert results.provider == "piper"
    assert [r.words for r in results] == [None, None]
    assert all(r.audio_path.is_file() and r.duration > 0 for r in results)


# --------------------------------------------------------------------------- #
# voice model resolution
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_model_resolution_accepts_an_absolute_path(fake_piper: FakePiper, tmp_path: Path):
    model = tmp_path / "elsewhere" / "custom-voice.onnx"
    model.parent.mkdir()
    model.write_bytes(b"")
    settings = fake_piper.settings()
    assert resolve_model(VoiceSpec(voice_id=str(model)), settings=settings) == model
    assert resolve_model(str(model), settings=settings) == model


def test_model_resolution_accepts_a_relative_path(
    fake_piper: FakePiper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = tmp_path / "rel" / "voice.onnx"
    model.parent.mkdir()
    model.write_bytes(b"")
    monkeypatch.chdir(tmp_path)
    found = resolve_model(VoiceSpec(voice_id="rel/voice.onnx"), settings=fake_piper.settings())
    assert found.resolve() == model.resolve()


def test_model_resolution_finds_a_native_name_in_the_voice_directory(fake_piper: FakePiper):
    model = fake_piper.add_voice("en_GB-alan-medium")
    settings = fake_piper.settings()
    assert resolve_model(VoiceSpec(voice_id="en_GB-alan-medium"), settings=settings) == model
    # The ``.onnx`` may be spelled out; it is the same file either way.
    assert resolve_model(VoiceSpec(voice_id="en_GB-alan-medium.onnx"), settings=settings) == model


def test_model_resolution_maps_a_catalogue_name_through_the_catalogue(fake_piper: FakePiper):
    """``narrator_deep`` is ours; ``en_US-ryan-high`` is what Piper calls it."""
    model = fake_piper.add_voice("en_US-ryan-high")
    found = resolve_model(VoiceSpec(voice_id="narrator_deep"), settings=fake_piper.settings())
    assert found == model


def test_model_resolution_falls_back_to_the_configured_voice(fake_piper: FakePiper):
    model = fake_piper.add_voice("en_US-amy-medium")
    settings = fake_piper.settings(piper_voice="en_US-amy-medium")
    # Nothing asked for at all.
    assert resolve_model(None, settings=settings) == model
    assert resolve_model(VoiceSpec(), settings=settings) == model
    # Asked for a catalogue entry Piper has no id for.
    assert resolve_model(VoiceSpec(voice_id="kid_bright"), settings=settings) == model


def test_model_resolution_accepts_a_path_in_the_configured_voice(fake_piper: FakePiper, tmp_path: Path):
    model = tmp_path / "configured.onnx"
    model.write_bytes(b"")
    settings = fake_piper.settings(piper_voice=str(model))
    assert resolve_model(VoiceSpec(), settings=settings) == model


def test_an_explicit_voice_beats_the_configured_default(fake_piper: FakePiper):
    asked = fake_piper.add_voice("en_GB-alan-medium")
    fake_piper.add_voice("en_US-amy-medium")
    settings = fake_piper.settings(piper_voice="en_US-amy-medium")
    assert resolve_model(VoiceSpec(voice_id="en_GB-alan-medium"), settings=settings) == asked


def test_nothing_resolvable_raises_and_says_where_it_looked(fake_piper: FakePiper):
    settings = fake_piper.settings()
    with pytest.raises(TTSError) as excinfo:
        resolve_model(VoiceSpec(voice_id="narrator_deep"), settings=settings)
    message = str(excinfo.value)
    assert str(fake_piper.voice_dir) in message, "the error must name the directory it searched"
    assert VOICE_DOWNLOADS in message, "the error must point at the voice downloads"
    assert "narrator_deep" in message and "en_US-ryan-high" in message, "both spellings tried"
    assert "no .onnx voice" in message


def test_the_missing_model_error_lists_what_is_installed(fake_piper: FakePiper):
    fake_piper.add_voice("en_GB-alan-medium")
    fake_piper.add_voice("en_US-amy-medium")
    with pytest.raises(TTSError) as excinfo:
        resolve_model(VoiceSpec(voice_id="en_US-ryan-high"), settings=fake_piper.settings())
    message = str(excinfo.value)
    assert "en_GB-alan-medium" in message and "en_US-amy-medium" in message


def test_a_voice_with_no_piper_id_and_no_default_is_a_clean_failure(fake_piper: FakePiper):
    """No guessing: ``kid_bright`` has no Piper voice, so nothing is substituted."""
    fake_piper.add_voice("en_US-amy-medium")
    with pytest.raises(TTSError):
        resolve_model(VoiceSpec(voice_id="kid_bright"), settings=fake_piper.settings())


def test_installed_voices_lists_the_directory(fake_piper: FakePiper):
    settings = fake_piper.settings()
    assert installed_voices(settings) == []
    fake_piper.add_voice("en_US-ryan-high")
    fake_piper.add_voice("en_GB-alan-medium")
    assert [p.stem for p in installed_voices(settings)] == ["en_GB-alan-medium", "en_US-ryan-high"]


# --------------------------------------------------------------------------- #
# available(): a dependency check, and nothing more
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_available_follows_the_binary(fake_piper: FakePiper):
    provider = PiperTTS(settings=fake_piper.settings())
    assert provider.available() is True
    fake_piper.unplug()
    assert PiperTTS(settings=fake_piper.settings()).available() is False


@pytest.mark.needs_ffmpeg
def test_available_accepts_an_absolute_binary_path(fake_piper: FakePiper):
    fake_piper.unplug()
    settings = fake_piper.settings(piper_binary=str(fake_piper.binary))
    assert PiperTTS(settings=settings).available() is True


@pytest.mark.needs_ffmpeg
def test_available_executes_nothing(fake_piper: FakePiper, monkeypatch: pytest.MonkeyPatch):
    """It runs on the hot path of every render: a PATH lookup, never a process."""
    def _forbidden(*args, **kwargs):
        raise AssertionError("available() must not execute a subprocess")

    monkeypatch.setattr(subprocess, "run", _forbidden)
    monkeypatch.setattr(subprocess, "Popen", _forbidden)
    assert PiperTTS(settings=fake_piper.settings()).available() is True
    assert fake_piper.calls() == []


@pytest.mark.needs_ffmpeg
def test_available_is_true_with_only_the_python_package(
    fake_piper: FakePiper, monkeypatch: pytest.MonkeyPatch
):
    fake_piper.unplug()
    assert PiperTTS(settings=fake_piper.settings()).available() is False
    _install_fake_module(monkeypatch, {})
    assert PiperTTS(settings=fake_piper.settings()).available() is True


@pytest.mark.needs_ffmpeg
def test_available_ignores_offline_because_piper_is_local(fake_piper: FakePiper):
    """The whole point: no network, so a boxed-in machine changes nothing."""
    settings = fake_piper.settings()
    settings.offline = True
    assert PiperTTS(settings=settings).available() is True


@pytest.mark.needs_ffmpeg
def test_available_says_nothing_about_voices(fake_piper: FakePiper):
    """A Piper with no voice models is still *installed* -- that is usable()'s job."""
    provider = PiperTTS(settings=fake_piper.settings())
    assert installed_voices(provider.settings) == []
    assert provider.available() is True


# --------------------------------------------------------------------------- #
# usable(): the diagnostic
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_usable_is_false_when_no_voice_is_installed(fake_piper: FakePiper):
    """The bug this prevents: ``doctor`` printing OK for a Piper that cannot speak."""
    provider = PiperTTS(settings=fake_piper.settings())
    assert provider.available() is True
    assert provider.usable() is False
    assert fake_piper.calls() == [], "no voice means no reason to run anything"


@pytest.mark.needs_ffmpeg
def test_usable_speaks_one_word_and_caches_the_verdict(fake_piper: FakePiper):
    fake_piper.add_voice()
    provider = PiperTTS(settings=fake_piper.settings())

    assert provider.usable() is True
    calls = fake_piper.calls()
    assert len(calls) == 1
    assert calls[0]["stdin"].strip() == PROBE_TEXT

    assert provider.usable() is True
    assert len(fake_piper.calls()) == 1, "the probe is cached for the process"
    assert provider.usable(refresh=True) is True
    assert len(fake_piper.calls()) == 2


@pytest.mark.needs_ffmpeg
def test_usable_is_false_when_the_binary_fails(fake_piper: FakePiper, monkeypatch: pytest.MonkeyPatch):
    fake_piper.add_voice()
    monkeypatch.setenv("PIPER_STUB_FAIL", "Error: onnxruntime is missing")
    assert PiperTTS(settings=fake_piper.settings()).usable() is False


@pytest.mark.needs_ffmpeg
def test_usable_times_out_rather_than_hanging(fake_piper: FakePiper, monkeypatch: pytest.MonkeyPatch):
    fake_piper.add_voice()
    monkeypatch.setenv("PIPER_STUB_SLEEP", "30")
    started = time.monotonic()
    assert PiperTTS(settings=fake_piper.settings()).usable(timeout=0.4) is False
    assert time.monotonic() - started < 5.0, "a diagnostic that hangs is a diagnostic nobody runs"


@pytest.mark.needs_ffmpeg
def test_usable_is_false_without_an_install(fake_piper: FakePiper):
    fake_piper.add_voice()
    fake_piper.unplug()
    assert PiperTTS(settings=fake_piper.settings()).usable() is False


@pytest.mark.needs_ffmpeg
def test_provider_usable_routes_to_the_backends_own_probe(fake_piper: FakePiper):
    fake_piper.add_voice()
    settings = fake_piper.settings()
    assert provider_usable("piper", settings=settings) is True
    assert len(fake_piper.calls()) == 1


# --------------------------------------------------------------------------- #
# the in-process path
# --------------------------------------------------------------------------- #

@pytest.mark.needs_ffmpeg
def test_the_python_package_is_preferred_over_the_binary(
    fake_piper: FakePiper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    model = fake_piper.add_voice()
    recorder: dict[str, object] = {}
    _install_fake_module(monkeypatch, recorder)

    provider = PiperTTS(settings=fake_piper.settings())
    out = tmp_path / "module.wav"
    result = provider.synthesize("hello there", out, voice=VoiceSpec(voice_id="en_US-lessac-medium",
                                                                    rate=1.25))

    assert fake_piper.calls() == [], "the importable package means no subprocess at all"
    assert recorder["model"] == str(model)
    assert recorder["text"] == "hello there"
    assert recorder["length_scale"] == pytest.approx(0.8)
    assert result.duration == pytest.approx(ff.probe(out).duration, abs=EPS)
    assert result.duration > 0
    assert result.words is None


@pytest.mark.needs_ffmpeg
def test_a_package_that_cannot_speak_falls_back_to_the_binary(
    fake_piper: FakePiper, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_piper.add_voice()
    module = types.ModuleType("piper")
    module.__spec__ = importlib.machinery.ModuleSpec("piper", None)  # importable, but useless
    monkeypatch.setitem(sys.modules, "piper", module)

    provider = PiperTTS(settings=fake_piper.settings(piper_voice="en_US-lessac-medium"))
    result = provider.synthesize("hello", tmp_path / "out.wav", voice=VoiceSpec())
    assert len(fake_piper.calls()) == 1, "an unusable package must not fail the render"
    assert result.duration > 0


# --------------------------------------------------------------------------- #
# the catalogue
# --------------------------------------------------------------------------- #

#: ``en_US-lessac-medium``, ``en_GB-northern_english_male-medium``, ...
_PIPER_ID = re.compile(r"^[a-z]{2}_[A-Z]{2}-[a-z0-9_]+-(x_low|low|medium|high)$")


def test_every_piper_id_looks_like_a_real_piper_voice():
    mapped = [v for v in VOICES if v.piper]
    assert len(mapped) >= 15, "a local backend with almost no voices is not much of a backend"
    for voice in mapped:
        assert _PIPER_ID.match(voice.piper), f"{voice.name}: {voice.piper!r} is not a piper voice name"


def test_the_catalogue_does_not_claim_accents_piper_does_not_have():
    """Piper's English voices are en_US and en_GB; the rest must stay ``None``."""
    elsewhere = {"australian", "irish", "indian", "canadian", "nz", "african", "asian"}
    for voice in VOICES:
        if voice.piper and set(voice.tags) & elsewhere:
            raise AssertionError(f"{voice.name} claims a Piper voice for an accent Piper lacks")
        if voice.piper:
            region = voice.piper.split("-")[0]
            assert region in {"en_US", "en_GB"}, f"{voice.name}: unexpected language {region}"


def test_the_catalogue_keeps_gender_and_accent_honest():
    entry = find_voice_entry("british_male")
    assert entry.piper == "en_GB-alan-medium"
    assert find_voice_entry("bright_female").piper == "en_US-amy-medium"
    assert find_voice_entry("silent").piper is None


def test_list_voices_tells_the_truth_about_what_piper_can_speak():
    speakable = list_voices("piper")
    assert speakable, "at least some catalogue voices must map onto piper"
    assert all(v.piper for v in speakable)
    assert len(speakable) < len(VOICES), "piper must not claim the whole catalogue"
    assert speakable == [v for v in VOICES if v.piper]
    # Spellings normalise onto the same list.
    assert list_voices("local") == speakable
    assert list_voices("onnx") == speakable


def test_find_voice_still_works_for_piper():
    spec = find_voice("narrator_deep", provider="piper")
    assert (spec.provider, spec.voice_id) == ("piper", "en_US-ryan-high")
    assert find_voice("piper:en_US-lessac-medium").voice_id == "en_US-lessac-medium"
    assert find_voice_entry("british female calm", provider="piper").piper
    # ``asmr_soft`` is real, but nothing Piper can speak -- including as a tag query.
    with pytest.raises(TTSError, match="piper"):
        find_voice_entry("asmr_soft", provider="piper")


def test_the_voice_entry_reports_piper_among_its_providers():
    entry = find_voice_entry("narrator_deep")
    assert entry.providers["piper"] == "en_US-ryan-high"
    assert entry.provider_id("local") == "en_US-ryan-high"
    assert entry.supports("piper")
    assert not find_voice_entry("kid_bright").supports("piper")
    assert "piper" in KNOWN_PROVIDERS


# --------------------------------------------------------------------------- #
# routing and fallback
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("spelling", ["piper", "PIPER", "piper-tts", "local", "onnx"])
def test_get_provider_routes_every_piper_spelling(spelling: str, fake_piper: FakePiper):
    provider = get_provider(spelling, settings=fake_piper.settings())
    assert provider.name == "piper"
    assert isinstance(provider, PiperTTS)


def test_the_alias_table_knows_piper():
    assert PROVIDER_ALIASES["piper"] == "piper"
    assert PROVIDER_ALIASES["local"] == "piper"
    assert PROVIDER_ALIASES["onnx"] == "piper"


def test_piper_sits_ahead_of_the_silent_floor():
    assert FALLBACK_ORDER.index("piper") < FALLBACK_ORDER.index("offline")
    assert FALLBACK_ORDER[-1] == "offline", "offline stays the floor"


@pytest.mark.needs_ffmpeg
def test_the_fallback_chain_reaches_piper_before_silence(fake_piper: FakePiper):
    settings = fake_piper.settings()
    names = [p.name for p in fallback_chain(_Broken(), settings=settings)]
    assert names[0] == "broken"
    assert names.index("piper") < names.index("offline")

    fake_piper.unplug()
    without = [p.name for p in fallback_chain(_Broken(), settings=fake_piper.settings())]
    assert "piper" not in without, "an uninstalled backend must not be offered"


@pytest.mark.needs_ffmpeg
def test_a_failing_backend_degrades_to_a_local_voice_not_to_silence(
    fake_piper: FakePiper, tmp_path: Path
):
    """The reason piper is in the chain: a real voice beats timed silence."""
    fake_piper.add_voice()
    settings = fake_piper.settings(piper_voice="en_US-lessac-medium")

    results = synthesize_lines(
        ["first line here", "second line here"],
        tmp_path / "vo",
        voice=VoiceSpec(voice_id="narrator_deep"),
        provider=_Broken(),
        settings=settings,
    )

    assert results.provider == "piper", "it degraded past the local voice to silence"
    assert results.attempted == ("broken", "piper")
    assert results.fell_back
    assert len(fake_piper.calls()) == 2, "every line is re-spoken by the backend that took over"
    assert all(r.audio_path.is_file() and r.duration > 0 for r in results)


@pytest.mark.needs_ffmpeg
def test_auto_prefers_a_local_voice_over_silence(fake_piper: FakePiper):
    """With no network backend available, an installed Piper beats the silent one."""
    settings = fake_piper.settings()
    settings.offline = True  # kills edge and elevenlabs, not piper
    assert get_provider("auto", settings=settings).name == "piper"

    fake_piper.unplug()
    offline_settings = fake_piper.settings()
    offline_settings.offline = True
    assert get_provider("auto", settings=offline_settings).name == "offline"


def test_naming_piper_explicitly_returns_it_even_when_it_is_missing(fake_piper: FakePiper):
    """House rule: an explicit name gets that backend's own diagnostic, not silence."""
    fake_piper.unplug()
    provider = get_provider("piper", settings=fake_piper.settings())
    assert provider.name == "piper"
    assert provider.available() is False


def test_the_piper_module_is_reachable_through_the_registry():
    assert piper_mod.PiperTTS is PiperTTS
