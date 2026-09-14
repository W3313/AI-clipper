"""The provider interface every language-model backend implements.

Three backends exist:

``claude``
    :class:`~aiclipper.llm.claude.ClaudeProvider` -- the Anthropic API.
``local``
    :class:`~aiclipper.llm.local.LocalProvider` -- a self-hosted model behind an
    OpenAI-compatible ``/chat/completions`` endpoint, which is what Ollama,
    llama.cpp's server, LM Studio and vLLM all speak.
``heuristic``
    :class:`~aiclipper.llm.heuristic.HeuristicProvider` -- rule-based, offline
    and, by contract, incapable of failing.

Callers never construct any of them directly: they ask :func:`get_provider` for
one and program against :class:`LLMProvider`.

``"auto"`` prefers Claude when a credential resolves, then -- only when the user
has explicitly exported :data:`LOCAL_BASE_URL_ENV`, which is an unambiguous
statement of intent and costs no network call -- the local backend, and
otherwise the heuristic one.  Every concrete provider is imported lazily inside
:func:`get_provider`, so ``import aiclipper.llm`` stays free of optional
third-party imports and nothing here touches the network.
"""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..errors import LLMError

__all__ = ["LLMProvider", "get_provider", "PROVIDER_ALIASES", "LOCAL_BASE_URL_ENV"]

#: Setting this in the environment is how a user says "I have a model running
#: locally".  ``"auto"`` reads it directly rather than comparing
#: ``settings.llm_base_url`` against its default, because the default is a
#: plausible URL (``http://localhost:11434/v1``) and not a statement of intent.
LOCAL_BASE_URL_ENV = "AICLIP_LLM_BASE_URL"

#: Accepted ``name`` values, mapped onto the canonical backend name.  The local
#: backend answers to every server that speaks its protocol, because "which
#: program is serving the model" is not a distinction this package needs to make.
PROVIDER_ALIASES: dict[str, str] = {
    "auto": "auto",
    "default": "auto",
    "": "auto",
    "claude": "claude",
    "anthropic": "claude",
    "heuristic": "heuristic",
    "offline": "heuristic",
    "rule": "heuristic",
    "none": "heuristic",
    "local": "local",
    "ollama": "local",
    "openai": "local",
    "openai-compatible": "local",
    "llamacpp": "local",
    "llama.cpp": "local",
    "lmstudio": "local",
    "lm-studio": "local",
    "vllm": "local",
}


@runtime_checkable
class LLMProvider(Protocol):
    """What every backend exposes.

    ``complete`` returns free text; ``complete_json`` returns an object that
    conforms to ``schema`` (a JSON Schema dict as accepted by the Anthropic
    structured-output API -- see :mod:`aiclipper.llm.prompts`).
    """

    name: str

    def available(self) -> bool:
        """True when this backend can actually be used right now.

        Must never perform network access: it is a credential/dependency check.
        """
        ...

    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None) -> str:
        """Answer ``prompt`` with plain text."""
        ...

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Answer ``prompt`` with an object validating against ``schema``."""
        ...


def _settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def _local_requested() -> bool:
    """Has the user explicitly pointed this process at a local endpoint?

    Presence of the environment variable, not the resolved setting: the setting
    has a usable default, so it says nothing about what the user wants.  No
    network, no import, cheap enough for the routing path.
    """
    return bool(os.environ.get(LOCAL_BASE_URL_ENV, "").strip())


def get_provider(name: str | None = None, *, settings: Settings | None = None) -> LLMProvider:
    """Resolve a provider by name.

    ``None`` falls back to ``settings.llm_provider``.  ``"auto"`` picks Claude
    when its credentials resolve and ``settings.offline`` is false; failing that
    it picks :class:`~aiclipper.llm.local.LocalProvider` when
    :data:`LOCAL_BASE_URL_ENV` is set in the environment and that backend is
    available; failing that, the heuristic provider, which always is.  An
    unrecognised name raises :class:`~aiclipper.errors.LLMError`.

    Naming a backend explicitly returns it even when it is unavailable, so the
    caller gets that backend's own diagnostic rather than a silent substitution.
    """
    s = _settings(settings)
    requested = (name if name is not None else s.llm_provider) or "auto"
    key = requested.strip().lower()
    canonical = PROVIDER_ALIASES.get(key)
    if canonical is None:
        known = ", ".join(sorted(k for k in PROVIDER_ALIASES if k))
        raise LLMError(f"unknown llm provider {requested!r}; expected one of: {known}")

    from .heuristic import HeuristicProvider  # local import: keeps module import cheap

    if canonical == "heuristic":
        return HeuristicProvider(settings=s)

    if canonical == "local":
        from .local import LocalProvider

        return LocalProvider(settings=s)

    from .claude import ClaudeProvider

    claude = ClaudeProvider(settings=s)
    if canonical == "claude":
        return claude
    if claude.available():
        return claude
    if _local_requested():
        from .local import LocalProvider

        local = LocalProvider(settings=s)
        if local.available():
            return local
    return HeuristicProvider(settings=s)
