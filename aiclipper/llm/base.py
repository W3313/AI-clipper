"""The provider interface every language-model backend implements.

Two backends exist: :class:`~aiclipper.llm.claude.ClaudeProvider`, which talks to
the Anthropic API, and :class:`~aiclipper.llm.heuristic.HeuristicProvider`, which
is rule-based, offline and cannot fail.  Callers never construct either directly
-- they ask :func:`get_provider` for one and program against
:class:`LLMProvider`.

Both concrete providers are imported lazily inside :func:`get_provider` so that
``import aiclipper.llm`` stays free of optional third-party imports.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..config import Settings, get_settings
from ..errors import LLMError

__all__ = ["LLMProvider", "get_provider", "PROVIDER_ALIASES"]

#: Accepted ``name`` values, mapped onto the canonical backend name.
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


def get_provider(name: str | None = None, *, settings: Settings | None = None) -> LLMProvider:
    """Resolve a provider by name.

    ``None`` falls back to ``settings.llm_provider``.  ``"auto"`` picks Claude
    when its credentials resolve and ``settings.offline`` is false, otherwise the
    heuristic provider.  An unrecognised name raises
    :class:`~aiclipper.errors.LLMError`.
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

    from .claude import ClaudeProvider

    claude = ClaudeProvider(settings=s)
    if canonical == "claude":
        return claude
    return claude if claude.available() else HeuristicProvider(settings=s)
