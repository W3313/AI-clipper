"""The provider interface every speech backend implements.

Three backends exist -- :class:`~aiclipper.tts.edge.EdgeTTS` (free, network,
gives word boundaries), :class:`~aiclipper.tts.eleven.ElevenLabsTTS` (keyed,
network, no word boundaries) and :class:`~aiclipper.tts.offline.OfflineTTS`
(timed silence, always available).  Callers never construct one directly; they
ask :func:`get_provider` and program against :class:`TTSProvider`.

Both network backends are imported lazily inside :func:`get_provider` and never
import their third-party dependency at module scope, so ``import aiclipper.tts``
stays cheap and safe on a bare interpreter.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..errors import TTSError
from ..models import TTSResult, VoiceSpec
from .voices import PROVIDER_ALIASES as _VOICE_ALIASES

__all__ = ["TTSProvider", "PROVIDER_ALIASES", "get_provider", "synthesize_lines", "total_duration"]

#: Accepted ``name`` values, mapped onto the canonical backend name.  Derived
#: from the catalogue's spellings so the two can never drift apart: a name that
#: :func:`aiclipper.tts.voices.canonical_provider` understands is a name
#: :func:`get_provider` accepts, and vice versa.
PROVIDER_ALIASES: dict[str, str] = {**_VOICE_ALIASES, "silent": "offline"}


@runtime_checkable
class TTSProvider(Protocol):
    """What every speech backend exposes."""

    name: str

    def available(self) -> bool:
        """True when this backend can be used right now.

        A dependency/credential/offline check only -- it must never perform
        network access.
        """
        ...

    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        """Speak ``text`` into ``out_path`` and describe the result.

        ``TTSResult.words`` carries word timings when the backend reports them
        and is ``None`` when it does not -- callers then force-align with
        :func:`aiclipper.transcribe.align`.
        """
        ...


def _settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def get_provider(name: str | None = None, *, settings: Settings | None = None) -> TTSProvider:
    """Resolve a backend by name.

    ``None`` falls back to ``settings.tts_provider``.  ``"auto"`` prefers
    :class:`~aiclipper.tts.edge.EdgeTTS` when ``edge_tts`` is importable and
    ``settings.offline`` is false, then
    :class:`~aiclipper.tts.eleven.ElevenLabsTTS` when ``ELEVENLABS_API_KEY`` is
    set, and finally :class:`~aiclipper.tts.offline.OfflineTTS`, which is always
    available.  An unrecognised name raises
    :class:`~aiclipper.errors.TTSError`.

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

    edge = EdgeTTS(settings=s)
    if edge.available():
        return edge
    eleven = ElevenLabsTTS(settings=s)
    if eleven.available():
        return eleven
    return OfflineTTS(settings=s)


def _coerce_provider(provider: TTSProvider | str | None, settings: Settings | None) -> TTSProvider:
    if provider is None or isinstance(provider, str):
        return get_provider(provider, settings=settings)
    return provider


def synthesize_lines(
    lines: Sequence[str],
    out_dir: Path,
    *,
    voice: VoiceSpec,
    provider: TTSProvider | str | None = None,
    gap: float = 0.18,
    settings: Settings | None = None,
) -> list[TTSResult]:
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
    """
    if not lines:
        return []
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = _settings(settings)
    prov = _coerce_provider(provider, s)

    blank_provider: TTSProvider | None = None
    results: list[TTSResult] = []
    offset = 0.0
    for index, line in enumerate(lines):
        path = out / f"line_{index:03d}.wav"
        if line and line.strip():
            result = prov.synthesize(line, path, voice=voice)
        else:
            if blank_provider is None:
                from .offline import OfflineTTS

                blank_provider = OfflineTTS(settings=s)
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
