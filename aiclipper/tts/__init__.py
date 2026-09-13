"""The speech-synthesis layer.

Ask :func:`get_provider` for a backend and program against
:class:`~aiclipper.tts.base.TTSProvider`::

    from aiclipper.tts import find_voice, get_provider, synthesize_lines

    voice = find_voice("narrator_deep")            # or "edge:en-GB-RyanNeural"
    results = synthesize_lines(script.lines, workspace / "vo", voice=voice)

``"auto"`` picks :class:`~aiclipper.tts.edge.EdgeTTS` when ``edge_tts`` is
installed and ``settings.offline`` is false, then
:class:`~aiclipper.tts.eleven.ElevenLabsTTS` when ``ELEVENLABS_API_KEY`` is set,
and finally :class:`~aiclipper.tts.offline.OfflineTTS` -- which renders timed
silence with synthetic word timings and is what keeps every workflow runnable
with no key and no egress.

:func:`~aiclipper.tts.base.synthesize_lines` is the one function most callers
need: it writes one file per line and rebases every word timing onto a single
continuous timeline that includes the inter-line ``gap``, so concatenating the
audio keeps the captions in sync.  See its docstring for the exact contract.

Importing this package pulls in no third-party dependency: ``edge_tts`` is
imported lazily inside the call that needs it, and the ElevenLabs backend speaks
plain :mod:`urllib.request`.
"""

from __future__ import annotations

from .base import PROVIDER_ALIASES, TTSProvider, get_provider, synthesize_lines, total_duration
from .edge import EdgeTTS
from .eleven import ElevenLabsTTS
from .offline import WORDS_PER_SECOND, OfflineTTS, estimate_duration, plan_words
from .voices import VOICES, Voice, find_voice, find_voice_entry, list_voices, resolve_voice_id

__all__ = [
    "TTSProvider", "get_provider", "synthesize_lines", "total_duration", "PROVIDER_ALIASES",
    "EdgeTTS", "ElevenLabsTTS", "OfflineTTS",
    "WORDS_PER_SECOND", "estimate_duration", "plan_words",
    "Voice", "VOICES", "find_voice", "find_voice_entry", "list_voices", "resolve_voice_id",
]
