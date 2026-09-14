"""The Piper backend: neural speech that never leaves the machine.

Piper is a small, CPU-fast VITS synthesiser: one binary (or one Python package)
plus one ``.onnx`` model per voice, a few tens of megabytes each.  It is the only
backend here that is both a *real voice* and completely local, which is why
:data:`aiclipper.tts.base.FALLBACK_ORDER` puts it directly ahead of the silent
floor -- it cannot be taken away by a dead network, an expired key or a rate
limit, so it is a far better thing to degrade to than silence.

**Two ways to drive it, one behaviour.**  When the ``piper`` Python package is
importable it is called in-process -- no subprocess, no PATH hunt.  Otherwise
the binary named by ``settings.piper_binary`` is run with the text on stdin::

    piper --model en_US-lessac-medium.onnx --output_file line_000.wav

Both paths are reached lazily, inside the call that speaks, so importing this
module imports nothing optional, runs no subprocess and touches no network.

**No word timings, by design.**  Piper's simple interface reports no word
boundaries, so :attr:`aiclipper.models.TTSResult.words` is always ``None``.
That is not a gap.  :func:`aiclipper.pipelines.common.narration_words` has three
tiers -- words reported by the backend, else forced alignment of the known
narration text against the rendered audio (:func:`aiclipper.transcribe.align`),
else timings spread proportionally across the part's *probed* duration -- so
captions, chat overlays and the ducking mix stay in sync behind a backend that
reports nothing.  ``TTSResult.duration`` comes from :func:`aiclipper.ffmpeg.probe`
rather than from an estimate, which is what keeps those tiers exact.

**Voices are files.**  A voice is ``<name>.onnx`` beside its ``<name>.onnx.json``
config, downloaded once from :data:`VOICE_DOWNLOADS` into
``settings.piper_voice_dir``.  :func:`resolve_model` is the whole lookup, in
order: an explicit ``.onnx`` path, that name inside the voice directory, the
catalogue's Piper id for a name like ``narrator_deep``, then
``settings.piper_voice``.  Nothing is guessed: when none of those exist the error
names the directory that was searched and where to get a voice, because speaking
in a voice the caller did not ask for is worse than failing clearly.

**``rate`` is inverted.**  Piper measures pace as ``--length_scale`` -- the
length of the output, not its speed -- so our ``rate`` of 1.25 (25% faster) is a
length scale of 0.8.  :func:`length_scale` is that conversion and the only place
it happens.  Piper exposes no pitch control, so ``VoiceSpec.pitch_semitones`` is
recorded on the result and otherwise ignored.

Being local, this backend ignores ``settings.offline`` entirely: a machine with
no egress is exactly where it earns its place.
"""

from __future__ import annotations

import importlib.util
import inspect
import logging
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .. import ffmpeg
from ..config import Settings
from ..errors import MissingDependency, TTSError
from ..models import TTSResult, VoiceSpec
from . import base
from .base import _settings as _resolve_settings
from .voices import resolve_voice_id

log = logging.getLogger(__name__)

__all__ = [
    "PiperTTS", "MODEL_SUFFIX", "VOICE_DOWNLOADS", "PROBE_TEXT", "PIP_PACKAGE",
    "RATE_LIMITS", "LENGTH_SCALE_LIMITS", "SYNTHESIS_TIMEOUT",
    "length_scale", "speak_args", "resolve_binary", "piper_module_available",
    "installed_voices", "resolve_model",
]

#: Every Piper voice model is an ONNX graph with this extension.
MODEL_SUFFIX = ".onnx"

#: Where voices come from.  Named in the error a missing model raises, so the
#: person reading it does not have to search for the download.
VOICE_DOWNLOADS = "https://huggingface.co/rhasspy/piper-voices"

#: What ``pip install`` line to print when neither the package nor the binary
#: resolves.  The binary ships inside this wheel, so one line fixes both.
PIP_PACKAGE = "piper-tts"

#: One short word, spoken by the capability probe.  Short enough that the probe
#: costs a model load and nothing else.
PROBE_TEXT = "hi"

#: ``VoiceSpec.rate`` is clamped to this band before conversion: beyond it the
#: voice stops sounding like speech.
RATE_LIMITS = (0.5, 2.0)

#: ``--length_scale`` band, the reciprocal of :data:`RATE_LIMITS`.
LENGTH_SCALE_LIMITS = (0.5, 2.0)

#: Wall-clock bound on one synthesis, in seconds.  Generous: the first call
#: after boot pays for loading the model, and a long line on a slow CPU is
#: legitimately slow.  The capability probe uses its own, much tighter bound.
SYNTHESIS_TIMEOUT = 120.0

#: How much of a failed run's stderr goes into the exception message.
_STDERR_LINES = 12


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def length_scale(rate: float) -> float:
    """``VoiceSpec.rate`` -> Piper's ``--length_scale``: its reciprocal.

    Piper scales the *length* of the audio, so a faster read is a smaller
    number: ``1.25`` (25% faster) becomes ``0.8``, ``0.8`` becomes ``1.25`` and
    ``1.0`` stays ``1.0``.  The rate is clamped to :data:`RATE_LIMITS` first, so
    a nonsensical request (zero, negative, ``None``, ``10.0``) lands inside a
    band that still sounds like a voice instead of being rejected.
    """
    try:
        value = float(rate)
    except (TypeError, ValueError):
        value = 1.0
    if value <= 0:
        value = 1.0
    value = _clamp(value, *RATE_LIMITS)
    return round(_clamp(1.0 / value, *LENGTH_SCALE_LIMITS), 4)


def speak_args(model: str | Path, out_path: str | Path, *, rate: float = 1.0) -> list[str]:
    """Arguments for one synthesis, without the binary name.

    Pure, so the command shape is testable with no Piper installed.
    ``--length_scale`` is passed **only** when the caller actually asked for a
    different pace: each voice ships a tuned default in its ``.onnx.json``, and
    sending ``1.0`` would override that with our own idea of neutral.
    """
    args = ["--model", str(model), "--output_file", str(out_path)]
    scale = length_scale(rate)
    if abs(scale - 1.0) >= 1e-3:
        args += ["--length_scale", f"{scale:g}"]
    return args


def resolve_binary(settings: Settings) -> str | None:
    """Full path to the Piper executable, or ``None``.

    A PATH lookup and nothing else -- :meth:`PiperTTS.available` is on the hot
    path of every render, so this must never execute anything.  An absolute (or
    ``./``-relative) ``settings.piper_binary`` is accepted as-is by
    :func:`shutil.which`, which still checks that it is executable.
    """
    name = (getattr(settings, "piper_binary", "") or "").strip() or "piper"
    try:
        return shutil.which(name)
    except (OSError, ValueError):  # pragma: no cover - broken PATH only
        return None


def piper_module_available() -> bool:
    """True when the ``piper`` Python package is importable (it is not imported)."""
    try:
        return importlib.util.find_spec("piper") is not None
    except (ImportError, ValueError, AttributeError):  # pragma: no cover - broken install only
        return False


def installed_voices(settings: Settings) -> list[Path]:
    """Every ``.onnx`` sitting in ``settings.piper_voice_dir``, sorted by name.

    What a diagnostic needs: Piper with no voices installed cannot speak, and
    the list is short enough to print back at whoever asked.
    """
    voice_dir = Path(settings.piper_voice_dir).expanduser()
    try:
        return sorted(p for p in voice_dir.glob(f"*{MODEL_SUFFIX}") if p.is_file())
    except OSError:  # pragma: no cover - unreadable directory only
        return []


def _voice_token(voice: VoiceSpec | str | None) -> str:
    if voice is None:
        return ""
    if isinstance(voice, str):
        return voice.strip()
    return (voice.voice_id or "").strip()


def _spellings(token: str) -> list[str]:
    """One request -> the names to look for: as given, then our catalogue's id.

    ``resolve_voice_id`` passes a native name (``en_US-lessac-medium``) straight
    through and maps a catalogue name (``narrator_deep``) onto the Piper id we
    know for it, so this covers both without a second lookup table.
    """
    token = (token or "").strip()
    if not token:
        return []
    out = [token]
    mapped = resolve_voice_id(VoiceSpec(voice_id=token), "piper", default="")
    if mapped and mapped not in out:
        out.append(mapped)
    return out


def _existing_model(token: str, voice_dir: Path) -> Path | None:
    """``token`` as a model file: a path to an ``.onnx``, or a name in ``voice_dir``."""
    text = (token or "").strip()
    if not text:
        return None
    given = Path(text).expanduser()
    if given.suffix == MODEL_SUFFIX and given.is_file():
        return given
    if given.is_absolute():
        return None
    named = voice_dir / (text if text.endswith(MODEL_SUFFIX) else text + MODEL_SUFFIX)
    return named if named.is_file() else None


def resolve_model(voice: VoiceSpec | str | None, *, settings: Settings) -> Path:
    """Find the ``.onnx`` that speaks ``voice``.

    In order: an absolute or relative path to an ``.onnx`` that exists;
    ``<settings.piper_voice_dir>/<voice_id>.onnx``; the catalogue's Piper id for
    that voice name, looked up in the same directory; then the same two lookups
    for ``settings.piper_voice``.

    Raises :class:`~aiclipper.errors.TTSError` when none of them exist, naming
    every spelling tried, the directory searched, what is installed there and
    where to download a voice.  Nothing is guessed and no other voice is
    substituted: a narration in a voice nobody chose is worse than a clear stop.
    """
    voice_dir = Path(settings.piper_voice_dir).expanduser()
    tried: list[str] = []
    for token in (*_spellings(_voice_token(voice)), *_spellings(getattr(settings, "piper_voice", ""))):
        if token in tried:
            continue
        tried.append(token)
        found = _existing_model(token, voice_dir)
        if found is not None:
            return found
    raise TTSError(_no_model_message(tried, voice_dir, settings))


def _no_model_message(tried: Sequence[str], voice_dir: Path, settings: Settings) -> str:
    wanted = ", ".join(repr(t) for t in tried) if tried else "<no voice given>"
    have = installed_voices(settings)
    inventory = (
        "installed there: " + ", ".join(p.stem for p in have)
        if have
        else f"that directory holds no {MODEL_SUFFIX} voice"
    )
    return (
        f"piper has no voice model for {wanted}: searched {voice_dir} ({inventory}). "
        f"Download one (e.g. en_US-lessac-medium) from {VOICE_DOWNLOADS} into that directory, "
        f"or point AICLIP_PIPER_VOICE at a {MODEL_SUFFIX} file."
    )


def _one_line(text: str) -> str:
    """Flatten ``text`` to a single line.

    Piper treats every stdin line as its own utterance and writes them all to the
    same ``--output_file``, so a line break would silently cost us audio.
    """
    return " ".join((text or "").split())


def _tail(stderr: str) -> str:
    lines = (stderr or "").strip().splitlines()
    return "\n".join(lines[-_STDERR_LINES:]) if lines else "<no stderr>"


class PiperTTS:
    """Local neural speech via Piper.  No key, no network, no telemetry."""

    name = "piper"

    def __init__(self, *, settings: Settings | None = None) -> None:
        self.settings = _resolve_settings(settings)

    # -- interface --------------------------------------------------------- #
    def available(self) -> bool:
        """Is Piper installed here at all?

        A dependency check and nothing more: a PATH lookup for the binary, a
        module-spec lookup for the package, and the ffmpeg check every backend
        needs to measure what it produced.  It executes no subprocess and opens
        no socket, because :func:`aiclipper.tts.base.fallback_chain` calls it on
        every render.  It deliberately does *not* look for a voice model, and
        deliberately ignores ``settings.offline`` -- Piper is local, so being
        offline is no reason to skip it.  Whether it can really speak (a voice
        installed, a model that loads) is :meth:`usable`.
        """
        if not ffmpeg.have_ffmpeg(self.settings):
            return False
        return resolve_binary(self.settings) is not None or piper_module_available()

    def usable(self, *, timeout: float | None = None, refresh: bool = False) -> bool:
        """Could this Piper actually speak a syllable right now?

        :meth:`available` proves only that the binary or the package resolved --
        and a Piper with no ``.onnx`` installed passes that while failing every
        render, which is exactly the kind of ``OK`` a diagnostic must not print.
        So this also insists on a voice model and then *uses* it: one real
        synthesis of :data:`PROBE_TEXT` into a temporary file, bounded by
        ``timeout`` (default :data:`aiclipper.tts.base.USABLE_TIMEOUT`) both on
        the subprocess and on the thread running it, so a wedged model load
        returns ``False`` instead of hanging the caller.

        Cached for the process, keyed by the binary, the model and the bound, so
        a diagnostic pays for the model load once.  ``refresh=True`` re-probes.
        A cold first load of a large voice can exceed the default bound on a slow
        CPU; pass a longer ``timeout`` there rather than reading ``False`` as
        "broken".
        """
        if not self.available():
            return False
        model = self.probe_model()
        if model is None:
            log.debug(
                "piper is installed but no %s voice was found in %s",
                MODEL_SUFFIX, self.settings.piper_voice_dir,
            )
            return False
        limit = float(timeout) if timeout and timeout > 0 else float(base.USABLE_TIMEOUT)
        key = f"piper:{resolve_binary(self.settings) or 'module'}:{model}:{limit:g}"
        return base.cached_usable(key, lambda: self._probe(model, limit), refresh=refresh)

    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        """Speak ``text`` into ``out_path``.

        ``words`` is always ``None`` -- see the module docstring: the engine's
        three-tier timing fallback covers it.  ``duration`` is measured from the
        file Piper wrote, never estimated.  The wav lands where the caller asked
        with no transcode: every narration part is resampled to one format by
        :func:`aiclipper.pipelines.common.concat_audio` anyway, so an extra
        ffmpeg pass here would only cost quality and time.
        """
        if not (text or "").strip():
            raise TTSError("piper cannot synthesise empty text")
        spec = voice or VoiceSpec()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        model = self.model_path(spec)
        self._speak(_one_line(text), out, model=model, rate=spec.rate)
        if not out.is_file() or out.stat().st_size == 0:
            raise TTSError(f"piper reported success but wrote no audio to {out}")

        return TTSResult(
            audio_path=out,
            duration=ffmpeg.probe(out, settings=self.settings).duration,
            words=None,
            voice=spec,
            text=text,
        )

    # -- voices ------------------------------------------------------------ #
    def model_path(self, voice: VoiceSpec | str | None) -> Path:
        """The ``.onnx`` this backend would speak ``voice`` with."""
        return resolve_model(voice, settings=self.settings)

    def probe_model(self) -> Path | None:
        """A model to prove the install with: the configured one, else any installed one."""
        try:
            return resolve_model(None, settings=self.settings)
        except TTSError:
            pass
        found = installed_voices(self.settings)
        return found[0] if found else None

    # -- internals --------------------------------------------------------- #
    def _probe(self, model: Path, timeout: float) -> bool:
        """One real, tiny synthesis; ``False`` for anything that is not audio."""

        def _try() -> bool:
            with tempfile.TemporaryDirectory(prefix="aiclip-piper-") as tmp:
                out = Path(tmp) / "probe.wav"
                try:
                    self._speak(PROBE_TEXT, out, model=model, rate=1.0, timeout=timeout)
                except (TTSError, MissingDependency) as exc:
                    log.debug("piper capability probe failed: %s", exc)
                    return False
                return out.is_file() and out.stat().st_size > 0

        # The subprocess timeout bounds the child; this bounds everything else,
        # including a model load that wedges inside the Python package.
        return base.run_bounded(_try, timeout + 0.5, default=False)

    def _speak(
        self,
        text: str,
        out: Path,
        *,
        model: Path,
        rate: float,
        timeout: float | None = None,
    ) -> None:
        """Render ``text`` to ``out``: in-process when we can, by subprocess otherwise."""
        troubles: list[str] = []
        if piper_module_available():
            try:
                if self._speak_with_module(text, out, model=model, rate=rate):
                    return
                troubles.append("the installed 'piper' package exposes no usable synthesis call")
            except TTSError as exc:
                troubles.append(str(exc))
            log.debug("piper python package could not speak (%s); trying the binary", troubles[-1])

        binary = resolve_binary(self.settings)
        if binary is None:
            if troubles:
                raise TTSError("; ".join(troubles))
            raise MissingDependency(
                PIP_PACKAGE,
                purpose=f"local neural speech synthesis ({self.settings.piper_binary!r} is not on PATH)",
            )
        self._run(binary, speak_args(model, out, rate=rate), text=text, timeout=timeout)

    def _speak_with_module(self, text: str, out: Path, *, model: Path, rate: float) -> bool:
        """The in-process path.  ``False`` means "this package cannot, try the binary"."""
        try:
            import piper  # noqa: PLC0415 - optional extra, imported at call time
        except ImportError:
            return False
        load = getattr(getattr(piper, "PiperVoice", None), "load", None)
        if not callable(load):
            return False

        scale = length_scale(rate)
        try:
            loaded = load(str(model))
            with wave.open(str(out), "wb") as wav:
                if not _module_synthesize(piper, loaded, text, wav, scale):
                    return False
        except Exception as exc:  # noqa: BLE001 - the package raises many types
            raise TTSError(f"piper (python package) failed on {model.name}: {exc}") from exc
        return out.is_file() and out.stat().st_size > 0

    def _run(self, binary: str, args: Sequence[str], *, text: str, timeout: float | None) -> None:
        """The subprocess path: text on stdin, a wav path out."""
        limit = float(timeout) if timeout and timeout > 0 else SYNTHESIS_TIMEOUT
        cmd = [binary, *args]
        log.debug("piper %s", " ".join(cmd[1:]))
        try:
            proc = subprocess.run(
                cmd, input=f"{text}\n", capture_output=True, text=True, timeout=limit,
            )
        except subprocess.TimeoutExpired as exc:
            raise TTSError(f"piper did not finish within {limit:g}s") from exc
        except OSError as exc:
            raise TTSError(f"piper could not be executed ({binary}): {exc}") from exc
        if proc.returncode != 0:
            raise TTSError(f"piper exited {proc.returncode}\n--- piper stderr ---\n{_tail(proc.stderr)}")


def _module_synthesize(module: Any, voice: Any, text: str, wav: Any, scale: float) -> bool:
    """Call whichever synthesis method this build of ``piper`` offers.

    The package has moved from ``synthesize(text, wav, length_scale=...)`` to
    ``synthesize_wav(text, wav, syn_config=SynthesisConfig(...))``; both are
    driven from the signature rather than from a version number, and an
    unrecognised build returns ``False`` so the caller can fall back to the
    binary instead of failing the render.
    """
    for method in ("synthesize_wav", "synthesize"):
        call = getattr(voice, method, None)
        if callable(call):
            call(text, wav, **_synthesis_kwargs(module, call, scale))
            return True
    return False


def _synthesis_kwargs(module: Any, call: Any, scale: float) -> dict[str, Any]:
    """How to ask *this* build for a length scale -- or not to ask at all."""
    if abs(scale - 1.0) < 1e-3:
        return {}
    try:
        params = inspect.signature(call).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables only
        return {}
    if "length_scale" in params:
        return {"length_scale": scale}
    config = getattr(module, "SynthesisConfig", None)
    if "syn_config" in params and config is not None:
        try:
            return {"syn_config": config(length_scale=scale)}
        except TypeError:  # pragma: no cover - a config that wants other arguments
            return {}
    return {}
