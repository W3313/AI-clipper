"""The speech-synthesis layer.

Ask :func:`get_provider` for a backend and program against
:class:`~aiclipper.tts.base.TTSProvider`::

    from aiclipper.tts import find_voice, get_provider, synthesize_lines

    voice = find_voice("narrator_deep")            # or "edge:en-GB-RyanNeural"
    results = synthesize_lines(script.lines, workspace / "vo", voice=voice)

``"auto"`` picks :class:`~aiclipper.tts.edge.EdgeTTS` when ``edge_tts`` is
installed and ``settings.offline`` is false, then
:class:`~aiclipper.tts.eleven.ElevenLabsTTS` when ``ELEVENLABS_API_KEY`` is set,
then :class:`~aiclipper.tts.piper.PiperTTS` when a local Piper install and voice
model are present, and finally :class:`~aiclipper.tts.offline.OfflineTTS` --
which renders timed silence with synthetic word timings and is what keeps every
workflow runnable with no key and no egress.

Piper is the one real voice that needs neither a key nor a network, so it sits
ahead of silence in both ``"auto"`` and the runtime fallback chain: a machine
whose only speech backend is local should narrate, not go quiet.  It returns no
word boundaries, which costs nothing here -- ``narration_words`` force-aligns
the audio instead.

:func:`~aiclipper.tts.base.synthesize_lines` is the one function most callers
need: it writes one file per line and rebases every word timing onto a single
continuous timeline that includes the inter-line ``gap``, so concatenating the
audio keeps the captions in sync.  See its docstring for the exact contract.

A backend that fails *while speaking* -- a network blip on line 3, a key that
expired this morning -- does not end the render: :func:`synthesize_lines` logs a
warning and re-synthesises the **whole** narration with the next backend in
:func:`~aiclipper.tts.base.fallback_chain`, so the voice stays the same from
first line to last, and reports the backend that really spoke on the returned
:class:`~aiclipper.tts.base.NarrationResults`.  Pass ``fallback=False`` to
demand the chosen backend or an exception.

Two questions, two calls: ``provider.available()`` is the cheap routing check
(dependency, key, offline flag -- never the network), while
:func:`~aiclipper.tts.base.provider_usable` is the diagnostic one, a real
timeout-bounded probe of whether the backend could speak here at all.  A
``doctor``-style command wants the second: ``edge-tts`` being importable on a
machine with no egress is an ``OK`` that becomes a failed render.

Importing this package pulls in no third-party dependency: ``edge_tts`` is
imported lazily inside the call that needs it, and the ElevenLabs backend speaks
plain :mod:`urllib.request`.
"""

from __future__ import annotations

from .base import (
    FALLBACK_ORDER,
    PROVIDER_ALIASES,
    USABLE_TIMEOUT,
    NarrationResults,
    TTSProvider,
    fallback_chain,
    get_provider,
    provider_usable,
    reset_usable_cache,
    synthesize_lines,
    total_duration,
)
from .edge import EdgeTTS
from .eleven import ElevenLabsTTS
from .offline import WORDS_PER_SECOND, OfflineTTS, estimate_duration, plan_words
from .piper import PiperTTS
from .voices import VOICES, Voice, find_voice, find_voice_entry, list_voices, resolve_voice_id

__all__ = [
    "TTSProvider", "get_provider", "synthesize_lines", "total_duration", "PROVIDER_ALIASES",
    "NarrationResults", "fallback_chain", "FALLBACK_ORDER",
    "provider_usable", "reset_usable_cache", "USABLE_TIMEOUT",
    "EdgeTTS", "ElevenLabsTTS", "PiperTTS", "OfflineTTS",
    "WORDS_PER_SECOND", "estimate_duration", "plan_words",
    "Voice", "VOICES", "find_voice", "find_voice_entry", "list_voices", "resolve_voice_id",
]
