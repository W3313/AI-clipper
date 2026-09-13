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
resolves and ``settings.offline`` is false, and
:class:`~aiclipper.llm.heuristic.HeuristicProvider` otherwise -- the heuristic
one is always available, never raises, and is what keeps the whole engine
runnable with no API key.

Importing this package pulls in no third-party dependency: the Anthropic SDK is
imported lazily, inside the call that needs it.
"""

from __future__ import annotations

from . import prompts
from .base import PROVIDER_ALIASES, LLMProvider, get_provider
from .claude import ClaudeProvider
from .heuristic import HeuristicProvider, build_instance, validate_instance

__all__ = [
    "LLMProvider", "get_provider", "PROVIDER_ALIASES",
    "ClaudeProvider", "HeuristicProvider",
    "build_instance", "validate_instance",
    "prompts",
]
