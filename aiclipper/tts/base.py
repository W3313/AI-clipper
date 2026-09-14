"""The provider interface every speech backend implements.

Four backends exist -- :class:`~aiclipper.tts.edge.EdgeTTS` (free, network,
gives word boundaries), :class:`~aiclipper.tts.eleven.ElevenLabsTTS` (keyed,
network, no word boundaries), :class:`~aiclipper.tts.piper.PiperTTS` (local
neural speech from a downloaded ``.onnx`` voice, no network, no word boundaries)
and :class:`~aiclipper.tts.offline.OfflineTTS` (timed silence, always
available).  Callers never construct one directly; they ask :func:`get_provider`
and program against :class:`TTSProvider`.

Every backend is imported lazily inside :func:`get_provider` and none imports
its third-party dependency (or runs its binary) at module scope, so ``import
aiclipper.tts`` stays cheap and safe on a bare interpreter.

**Two different questions, two different calls.**  ``provider.available()`` is a
*routing* check: dependency, credential and offline flags only, no network, fast
enough to call on every render.  :func:`provider_usable` (backed by the optional
``provider.usable()`` method) is a *diagnostic* check: a real, cheap,
timeout-bounded capability probe that answers "could this backend actually speak
a syllable on this machine right now?".  ``available()`` says edge-tts is
installed; ``usable()`` says the service answers.  A doctor-style command must
ask the second question -- an ``OK`` that turns into a failed render is worse
than a ``MISSING`` -- and everything else should ask the first.  Probe results
are cached for the life of the process, so a diagnostic pays for them once.

**Runtime failure is not fatal.**  :func:`synthesize_lines` walks a
:func:`fallback_chain`: when the selected backend raises part way through a
narration the whole narration is re-synthesised with the next usable backend,
ending at :class:`~aiclipper.tts.offline.OfflineTTS`, which needs nothing but
ffmpeg -- via :class:`~aiclipper.tts.piper.PiperTTS` when a local voice is
installed, because a real voice that no network blip can take away is a much
better degradation target than silence.  Re-doing every line rather than
continuing from the failure is the point: half a narration in one voice and half
in another is a broken video, and a render that finishes in the offline voice
beats a render that dies.  The
backend that actually spoke is reported on the returned
:class:`NarrationResults`, so pipelines can record it in their metadata.
"""

from __future__ import annotations

import inspect
import logging
import threading
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any, Protocol, TypeVar, runtime_checkable

from ..config import Settings, get_settings
from ..errors import TTSError
from ..models import TTSResult, VoiceSpec
from .voices import PROVIDER_ALIASES as _VOICE_ALIASES

log = logging.getLogger(__name__)

__all__ = [
    "TTSProvider", "PROVIDER_ALIASES", "FALLBACK_ORDER", "USABLE_TIMEOUT", "NarrationResults",
    "get_provider", "fallback_chain", "synthesize_lines", "total_duration",
    "provider_usable", "cached_usable", "reset_usable_cache", "run_bounded",
]

T = TypeVar("T")

#: Accepted ``name`` values, mapped onto the canonical backend name.  Derived
#: from the catalogue's spellings so the two can never drift apart: a name that
#: :func:`aiclipper.tts.voices.canonical_provider` understands is a name
#: :func:`get_provider` accepts, and vice versa.
PROVIDER_ALIASES: dict[str, str] = {**_VOICE_ALIASES, "silent": "offline"}

#: Order :func:`fallback_chain` degrades through when a backend fails at
#: runtime.  ``piper`` sits directly ahead of ``offline``: it is local, so
#: unlike the two network backends it cannot fail from a blip, an expired key or
#: a rate limit, which makes it the last chance at a real voice.  ``offline`` is
#: last and unconditional: it needs only ffmpeg, so it is the floor below which
#: a render cannot fall.
FALLBACK_ORDER: tuple[str, ...] = ("edge", "elevenlabs", "piper", "offline")

#: Wall-clock bound, in seconds, on one :func:`provider_usable` probe.  A
#: diagnostic that hangs is a diagnostic nobody runs.
USABLE_TIMEOUT = 5.0


@runtime_checkable
class TTSProvider(Protocol):
    """What every speech backend exposes."""

    name: str

    def available(self) -> bool:
        """True when this backend can be used right now.

        A dependency/credential/offline check only -- it must never perform
        network access, because it is on the hot path of every render.  For the
        stronger "would it actually work?" question, see :func:`provider_usable`.
        """
        ...

    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        """Speak ``text`` into ``out_path`` and describe the result.

        ``TTSResult.words`` carries word timings when the backend reports them
        and is ``None`` when it does not -- callers then force-align with
        :func:`aiclipper.transcribe.align`.
        """
        ...

    # Backends may also implement::
    #
    #     def usable(self, *, timeout: float | None = None, refresh: bool = False) -> bool
    #
    # a real, timeout-bounded capability probe for diagnostics.  It is
    # deliberately *not* part of the protocol -- a third-party or test double
    # that only implements the two methods above is still a valid provider --
    # so call it through :func:`provider_usable`, which degrades to
    # ``available()`` when a backend does not offer one.


def _settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def get_provider(name: str | None = None, *, settings: Settings | None = None) -> TTSProvider:
    """Resolve a backend by name.

    ``None`` falls back to ``settings.tts_provider``.  ``"auto"`` prefers
    :class:`~aiclipper.tts.edge.EdgeTTS` when ``edge_tts`` is importable and
    ``settings.offline`` is false, then
    :class:`~aiclipper.tts.eleven.ElevenLabsTTS` when ``ELEVENLABS_API_KEY`` is
    set, then :class:`~aiclipper.tts.piper.PiperTTS` when Piper is installed
    locally -- offline or not, since it needs no network -- and finally
    :class:`~aiclipper.tts.offline.OfflineTTS`, which is always available.  The
    same reasoning as :data:`FALLBACK_ORDER`: picking silence over an installed
    local voice would be a strange thing to do automatically.  An unrecognised
    name raises :class:`~aiclipper.errors.TTSError`.

    Naming an unavailable backend explicitly still returns it -- so the caller
    gets that backend's own diagnostic (a ``MissingDependency`` naming the pip
    line, say) instead of silent, surprising silence.
    """
    s = _settings(settings)
    requested = (name if name is not None else s.tts_provider) or "auto"
    key = requested.strip().lower()
    canonical = PROVIDER_ALIASES.get(key)
    if canonical is None:
        known = ", ".join(sorted(k for k in PROVIDER_ALIASES if k))
        raise TTSError(f"unknown tts provider {requested!r}; expected one of: {known}")

    from .offline import OfflineTTS  # local import: keeps module import cheap

    if canonical == "offline":
        return OfflineTTS(settings=s)

    from .edge import EdgeTTS

    if canonical == "edge":
        return EdgeTTS(settings=s)

    from .eleven import ElevenLabsTTS

    if canonical == "elevenlabs":
        return ElevenLabsTTS(settings=s)

    from .piper import PiperTTS

    if canonical == "piper":
        return PiperTTS(settings=s)

    edge = EdgeTTS(settings=s)
    if edge.available():
        return edge
    eleven = ElevenLabsTTS(settings=s)
    if eleven.available():
        return eleven
    piper = PiperTTS(settings=s)
    if piper.available():
        return piper
    return OfflineTTS(settings=s)


def _coerce_provider(provider: TTSProvider | str | None, settings: Settings | None) -> TTSProvider:
    if provider is None or isinstance(provider, str):
        return get_provider(provider, settings=settings)
    return provider


def fallback_chain(
    provider: TTSProvider | str | None = None,
    *,
    settings: Settings | None = None,
) -> list[TTSProvider]:
    """The backends to try, in order, for one narration.

    The first entry is whatever ``provider`` resolves to (a name, an
    already-constructed backend, or ``None`` for ``settings.tts_provider``).
    After it come the other backends from :data:`FALLBACK_ORDER` that
    ``available()`` says could run here, and finally
    :class:`~aiclipper.tts.offline.OfflineTTS` -- unconditionally, because it is
    the one backend that cannot be taken away by a dead network or an expired
    key, and a render that ends in a synthetic voice beats a render that ends in
    a traceback.

    Only ``available()`` is consulted, never the network: this sits on the hot
    path of every render, and a backend that is *listed* here still has to fail
    for real before :func:`synthesize_lines` moves past it.
    """
    s = _settings(settings)
    first = _coerce_provider(provider, s)
    chain: list[TTSProvider] = [first]
    seen = {str(getattr(first, "name", "") or "")}
    for name in FALLBACK_ORDER:
        if name in seen:
            continue
        try:
            candidate = get_provider(name, settings=s)
        except TTSError:  # pragma: no cover - FALLBACK_ORDER holds known names
            continue
        if name != "offline":
            try:
                if not candidate.available():
                    continue
            except Exception:  # noqa: BLE001 - a broken probe is an unusable backend
                log.debug("tts backend %r failed its availability check", name, exc_info=True)
                continue
        seen.add(name)
        chain.append(candidate)
    return chain


class NarrationResults(list[TTSResult]):
    """``list[TTSResult]`` that also says which backend actually spoke it.

    A plain list everywhere it matters -- indexing, slicing, ``==`` against a
    list, iteration -- with two extra attributes so a pipeline can put the truth
    in its metadata instead of the backend it *asked* for:

    ``provider``
        Name of the backend whose audio is on disk (``"edge"``, ``"offline"``...).
    ``attempted``
        Every backend tried, in order, ending with ``provider``.  Longer than
        one entry means a fallback happened.
    """

    provider: str
    attempted: tuple[str, ...]

    def __init__(self, results: Iterable[TTSResult] = (), *, provider: str = "",
                 attempted: Iterable[str] = ()) -> None:
        super().__init__(results)
        self.provider = provider
        self.attempted = tuple(attempted) or ((provider,) if provider else ())

    @property
    def fell_back(self) -> bool:
        """True when the backend that spoke is not the one first asked for."""
        return len(self.attempted) > 1


def synthesize_lines(
    lines: Sequence[str],
    out_dir: Path,
    *,
    voice: VoiceSpec,
    provider: TTSProvider | str | None = None,
    gap: float = 0.18,
    settings: Settings | None = None,
    fallback: bool = True,
) -> NarrationResults:
    """Speak each line into its own file, on one shared timeline.

    **The rebasing contract.**  Line ``i`` is written to
    ``out_dir/line_<i>.wav`` and its :class:`~aiclipper.models.TTSResult`
    reports ``duration`` for *that file alone*.  The ``words`` on that result,
    however, are shifted onto a single continuous timeline in which line ``i``
    begins at::

        offset(i) = sum(results[k].duration for k in range(i)) + gap * i

    -- that is, the concatenation of every previous line plus one ``gap`` of
    silence between each adjacent pair.  So a caller that concatenates the audio
    files with ``gap`` seconds of silence between them can feed the collected
    ``words`` straight to :mod:`aiclipper.captions` and the captions stay in
    sync, with no further arithmetic.  ``offset(0)`` is ``0.0``; the whole
    narration runs ``total_duration(results, gap)`` seconds.

    A backend that reports no word boundaries leaves ``words`` as ``None``; such
    results are passed through unchanged (nothing to rebase) but still advance
    the offset for the lines after them.

    Blank lines are honoured rather than dropped, so the returned list is always
    the same length and order as ``lines``: they are rendered as a short silence
    by :class:`~aiclipper.tts.offline.OfflineTTS` regardless of ``provider``,
    because the network backends reject empty text.

    **Fallback (``fallback=True``, the default).**  A backend that raises -- a
    network blip on line 3, an expired key, a service that starts returning 502
    -- does not kill the render.  The failure is logged as a warning and the
    *whole narration is spoken again from line 0* by the next backend in
    :func:`fallback_chain`, ending at the always-there offline backend.
    Re-speaking every line is the entire point: the alternative is a video whose
    first two lines are one voice and whose rest is another, which is worse than
    either voice alone.  The decision is therefore made once per render, never
    per line.  Pass ``fallback=False`` to demand the chosen backend or nothing --
    the original exception then propagates untouched.

    The returned :class:`NarrationResults` is a list of results that also
    carries ``provider`` (the backend that really spoke) and ``attempted``, so
    the caller can record what happened rather than what it asked for.
    """
    if not lines:
        return NarrationResults()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = _settings(settings)

    chain = fallback_chain(provider, settings=s) if fallback else [_coerce_provider(provider, s)]
    attempted: list[str] = []
    failure: Exception | None = None

    for index, candidate in enumerate(chain):
        name = str(getattr(candidate, "name", "") or "tts")
        attempted.append(name)
        try:
            results = _speak_all(lines, out, voice=voice, provider=candidate, gap=gap, settings=s)
        except Exception as exc:  # noqa: BLE001 - any backend failure is recoverable
            failure = exc
            following = chain[index + 1:]
            if not following:
                break
            log.warning(
                "tts backend %r failed (%s); re-synthesising all %d line(s) with %r so the "
                "narration keeps one voice",
                name, exc, len(lines), str(getattr(following[0], "name", "") or "tts"),
            )
            continue
        return NarrationResults(results, provider=name, attempted=attempted)

    if failure is None:  # pragma: no cover - the loop only leaves here on a failure
        raise TTSError("no tts backend was available to speak this narration")
    raise failure


def _speak_all(
    lines: Sequence[str],
    out: Path,
    *,
    voice: VoiceSpec,
    provider: TTSProvider,
    gap: float,
    settings: Settings,
) -> list[TTSResult]:
    """One full pass over ``lines`` with a single backend.

    Either every line is spoken by ``provider`` (blank lines excepted, which are
    always silence) or this raises and nothing is returned -- which is what lets
    :func:`synthesize_lines` treat a fallback as all-or-nothing.
    """
    blank_provider: TTSProvider | None = None
    results: list[TTSResult] = []
    offset = 0.0
    for index, line in enumerate(lines):
        path = out / f"line_{index:03d}.wav"
        if line and line.strip():
            result = provider.synthesize(line, path, voice=voice)
        else:
            if blank_provider is None:
                from .offline import OfflineTTS

                blank_provider = OfflineTTS(settings=settings)
            result = blank_provider.synthesize("", path, voice=voice)
        if result.words:
            result.words = [w.shifted(offset) for w in result.words]
        results.append(result)
        offset += max(0.0, result.duration) + max(0.0, gap)
    return results


def total_duration(results: Sequence[TTSResult], gap: float = 0.18) -> float:
    """Length of the concatenated narration produced by :func:`synthesize_lines`."""
    if not results:
        return 0.0
    return sum(max(0.0, r.duration) for r in results) + max(0.0, gap) * (len(results) - 1)


# --------------------------------------------------------------------------- #
# capability probes: "would this backend actually work here?"
# --------------------------------------------------------------------------- #

#: Probe results, keyed by the probing backend plus whatever configuration the
#: answer depends on.  Process-lifetime by design: a diagnostic that probes the
#: same backend three times is three chances to hang.
_USABLE_CACHE: dict[str, bool] = {}
_USABLE_LOCK = threading.Lock()


def run_bounded(call: Callable[[], T], timeout: float, *, default: T) -> T:
    """Run ``call`` on a daemon thread and give up after ``timeout`` seconds.

    ``concurrent.futures`` cannot help here: its shutdown *waits* for the worker,
    so a probe that wedges inside a blocking socket read would hang the caller
    at the end of the ``with`` block instead of at the call.  A daemon thread we
    simply stop waiting for cannot keep the process alive, so a wedged probe
    costs one abandoned thread and nothing else.  Exceptions raised by ``call``
    propagate; a timeout returns ``default``.
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=_target, name="aiclip-tts-probe", daemon=True)
    thread.start()
    thread.join(max(0.0, float(timeout)))
    if thread.is_alive():
        return default
    if "error" in box:
        raise box["error"]
    return box.get("value", default)  # type: ignore[return-value]


def cached_usable(key: str, probe: Callable[[], bool], *, refresh: bool = False) -> bool:
    """Memoise one capability probe under ``key`` for the life of the process.

    A probe that raises is a backend that does not work, so the answer is
    ``False`` and it is cached like any other -- the caller wanted a verdict, not
    an exception.  ``refresh=True`` re-probes and overwrites.
    """
    if not refresh:
        with _USABLE_LOCK:
            hit = _USABLE_CACHE.get(key)
        if hit is not None:
            return hit
    try:
        verdict = bool(probe())
    except Exception as exc:  # noqa: BLE001 - a failed probe is simply "not usable"
        log.debug("tts capability probe %s failed: %s", key, exc, exc_info=True)
        verdict = False
    with _USABLE_LOCK:
        _USABLE_CACHE[key] = verdict
    return verdict


def reset_usable_cache(key: str | None = None) -> None:
    """Forget cached probe verdicts -- all of them, or just ``key``.

    Mostly for tests and for a long-lived process that has just been handed new
    credentials or regained its network.
    """
    with _USABLE_LOCK:
        if key is None:
            _USABLE_CACHE.clear()
        else:
            _USABLE_CACHE.pop(key, None)


def provider_usable(
    provider: TTSProvider | str | None = None,
    *,
    settings: Settings | None = None,
    timeout: float | None = None,
    refresh: bool = False,
) -> bool:
    """Can this backend really speak, here, now?

    This is what a ``doctor``-style diagnostic should call.  ``available()``
    only proves the import worked or the key is a non-empty string; this runs
    the backend's own probe -- a voice-list fetch for edge, a cheap authenticated
    GET for ElevenLabs, one tiny local synthesis for piper, an ffmpeg check for
    offline -- bounded by ``timeout`` (default :data:`USABLE_TIMEOUT`) and cached
    for the process, so "OK" means the next render will actually produce audio.

    A backend with no ``usable()`` of its own (a third-party provider, a test
    double) degrades to ``available()``.  Never raises: an unusable backend and
    a backend that explodes while being asked are both ``False``.
    """
    try:
        prov = _coerce_provider(provider, _settings(settings))
    except Exception as exc:  # noqa: BLE001 - diagnostics never crash the caller
        log.debug("tts provider %r could not be resolved: %s", provider, exc)
        return False

    probe = getattr(prov, "usable", None)
    if not callable(probe):
        try:
            return bool(prov.available())
        except Exception:  # noqa: BLE001
            log.debug("tts backend %r failed its availability check", prov, exc_info=True)
            return False

    try:
        params = inspect.signature(probe).parameters
    except (TypeError, ValueError):  # pragma: no cover - exotic callables only
        params = {}
    kwargs: dict[str, Any] = {}
    if "timeout" in params:
        kwargs["timeout"] = timeout
    if "refresh" in params:
        kwargs["refresh"] = refresh
    try:
        return bool(probe(**kwargs))
    except Exception:  # noqa: BLE001 - a probe that throws is a backend that does not work
        log.debug("tts capability probe for %r raised", getattr(prov, "name", prov), exc_info=True)
        return False
