"""The language-model layer.

Ask :func:`get_provider` for a backend and program against
:class:`~aiclipper.llm.base.LLMProvider`::

    from aiclipper.llm import get_provider, prompts

    provider = get_provider()                      # settings.llm_provider, default "auto"
    data = provider.complete_json(
        prompts.script_prompt("why bread rises", seconds=30),
        prompts.SCRIPT_SCHEMA,
        system=prompts.SCRIPT_SYSTEM,
    )

``"auto"`` picks :class:`~aiclipper.llm.claude.ClaudeProvider` when a credential
resolves and ``settings.offline`` is false; failing that
:class:`~aiclipper.llm.local.LocalProvider` when ``AICLIP_LLM_BASE_URL`` is set
in the environment, which is how a user says "I have a model running here"; and
otherwise :class:`~aiclipper.llm.heuristic.HeuristicProvider` -- always
available, never raises, and what keeps the whole engine runnable with no API
key at all.

:class:`~aiclipper.llm.local.LocalProvider` talks to any OpenAI-compatible
``/chat/completions`` endpoint, so one client covers Ollama, llama.cpp's server,
LM Studio and vLLM.  Smaller models are far less reliable at strict JSON than a
frontier model, so it asks for schema-guided output, steps down through weaker
response formats when a server rejects that, extracts the object from replies
wrapped in fences or prose, validates it, makes one repair round-trip, and only
then falls through to the heuristic answer.  A rambling reply costs quality, not
a render.

Importing this package pulls in no third-party dependency: the Anthropic SDK is
imported lazily inside the call that needs it, and the local backend speaks
plain :mod:`urllib.request`.
"""

from __future__ import annotations

from . import prompts
from .base import PROVIDER_ALIASES, LLMProvider, get_provider
from .claude import ClaudeProvider
from .heuristic import HeuristicProvider, build_instance, validate_instance
from .local import LocalProvider

__all__ = [
    "LLMProvider", "get_provider", "PROVIDER_ALIASES",
    "ClaudeProvider", "LocalProvider", "HeuristicProvider",
    "build_instance", "validate_instance",
    "prompts",
]
