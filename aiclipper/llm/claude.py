"""The Anthropic-API provider.

``anthropic`` is an optional extra, so it is imported lazily inside the methods
that need it; importing this module on a bare interpreter is safe and
:class:`~aiclipper.errors.MissingDependency` is raised at call time instead.

Request shape (Opus 5 era -- do not "modernise" this from memory):

* structured output goes through ``output_config.format`` as a
  ``json_schema``; plain completions keep ``output_config.effort`` and drop
  ``format``;
* never send ``thinking`` -- adaptive thinking is the default on Opus 5 -- and
  never send ``budget_tokens``;
* never prefill an assistant turn and never force ``tool_choice``;
* check ``stop_reason == "refusal"`` before touching ``content``.

Transport failures are retried exactly once (429, 5xx, connection/timeout) and
then give up with :class:`~aiclipper.errors.LLMError`.  Callers that must not
fail -- the render pipelines -- catch that and fall back to
:class:`~aiclipper.llm.heuristic.HeuristicProvider`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ..config import Settings
from ..errors import LLMError, MissingDependency
from .base import _settings as _resolve_settings

log = logging.getLogger(__name__)

__all__ = ["ClaudeProvider", "credentials_available"]

#: One retry, then give up.
_MAX_RETRIES = 1

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


class _Unreachable(Exception):
    """Never raised.  Stands in for an SDK exception class when it is absent."""


#: Used in ``except`` clauses when ``anthropic`` is not importable but a client
#: object was injected (tests, or an embedder supplying its own transport).
_NO_SDK = SimpleNamespace(
    APITimeoutError=_Unreachable,
    APIConnectionError=_Unreachable,
    RateLimitError=_Unreachable,
    InternalServerError=_Unreachable,
    APIStatusError=_Unreachable,
    AnthropicError=_Unreachable,
)


# --------------------------------------------------------------------------- #
# credential discovery (filesystem + environment only, never the network)
# --------------------------------------------------------------------------- #

def _config_dir() -> Path:
    """Where the SDK keeps profiles: ``ANTHROPIC_CONFIG_DIR`` or the OS default."""
    override = os.environ.get("ANTHROPIC_CONFIG_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    if sys.platform == "win32":  # pragma: no cover - posix CI
        appdata = os.environ.get("APPDATA")
        return (Path(appdata) if appdata else Path.home() / "AppData" / "Roaming") / "Anthropic"
    return Path.home() / ".config" / "anthropic"


def _active_profile(config_dir: Path) -> str:
    name = os.environ.get("ANTHROPIC_PROFILE", "").strip()
    if name:
        return name
    try:
        pointer = (config_dir / "active_config").read_text(encoding="utf-8").strip()
    except OSError:
        pointer = ""
    return pointer or "default"


def credentials_available() -> bool:
    """True when the SDK would find *some* credential without asking the network.

    Mirrors the SDK's resolution order: explicit key/token env vars, then an
    ``ant`` profile on disk, then workload-identity federation env vars.
    """
    env = os.environ
    if env.get("ANTHROPIC_API_KEY", "").strip() or env.get("ANTHROPIC_AUTH_TOKEN", "").strip():
        return True
    try:
        config_dir = _config_dir()
        profile = _active_profile(config_dir)
        if "/" in profile or "\\" in profile or profile.startswith("."):
            return False
        for kind in ("credentials", "configs"):
            if (config_dir / kind / f"{profile}.json").is_file():
                return True
    except OSError:  # pragma: no cover - unreadable home directory
        pass
    federated = (
        env.get("ANTHROPIC_FEDERATION_RULE_ID", "").strip()
        and env.get("ANTHROPIC_ORGANIZATION_ID", "").strip()
    )
    if federated:
        if env.get("ANTHROPIC_IDENTITY_TOKEN", "").strip():
            return True
        if env.get("ANTHROPIC_IDENTITY_TOKEN_FILE", "").strip():
            return True
    return False


def _sdk() -> Any:
    try:
        import anthropic
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise MissingDependency("anthropic", extra="llm", purpose="Claude provider") from exc
    return anthropic


def _sdk_errors() -> Any:
    try:
        import anthropic
    except ImportError:  # pragma: no cover - exercised only without the extra
        return _NO_SDK
    return anthropic


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #

class ClaudeProvider:
    """Talks to the Anthropic Messages API.

    ``client`` exists for tests and for embedders that already own a configured
    ``anthropic.Anthropic`` instance; left as ``None`` one is constructed on
    first use, which is where the SDK reads ``ANTHROPIC_API_KEY`` or the active
    ``ant`` profile.
    """

    name = "claude"

    def __init__(self, *, settings: Settings | None = None, client: Any | None = None) -> None:
        self._settings = _resolve_settings(settings)
        self._client = client
        self._injected = client is not None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ClaudeProvider(model={self._settings.llm_model!r})"

    @property
    def settings(self) -> Settings:
        return self._settings

    # -- capability -------------------------------------------------------- #
    def available(self) -> bool:
        """True when a credential resolves and we are not in offline mode.

        Purely a local check: no request is made to find out.
        """
        if self._settings.offline:
            return False
        if self._injected:
            return True
        try:
            import importlib.util

            if importlib.util.find_spec("anthropic") is None:
                return False
        except (ImportError, ValueError):  # pragma: no cover - broken import system
            return False
        return credentials_available()

    # -- completion -------------------------------------------------------- #
    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None) -> str:
        """Plain-text completion.  Keeps ``effort``, sends no output format."""
        response = self._request(prompt, system=system, max_tokens=max_tokens, schema=None)
        return _response_text(response)

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Structured completion via ``output_config.format`` (``json_schema``)."""
        response = self._request(prompt, system=system, max_tokens=max_tokens, schema=schema)
        return _parse_json(_response_text(response))

    # -- internals --------------------------------------------------------- #
    def client(self) -> Any:
        """The underlying SDK client, constructed on first use."""
        if self._client is None:
            if self._settings.offline:
                raise LLMError("settings.offline is set; refusing to build an Anthropic client")
            anthropic = _sdk()
            self._client = anthropic.Anthropic()
        return self._client

    def build_request(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The exact kwargs handed to ``client.messages.create``."""
        s = self._settings
        output_config: dict[str, Any] = {"effort": s.llm_effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request: dict[str, Any] = {
            "model": s.llm_model,
            "max_tokens": max(1, int(max_tokens or s.llm_max_tokens)),
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            request["system"] = system
        return request

    def _request(
        self,
        prompt: str,
        *,
        system: str,
        max_tokens: int | None,
        schema: dict[str, Any] | None,
    ) -> Any:
        if self._settings.offline:
            raise LLMError("settings.offline is set; the Claude provider cannot be used")
        client = self.client()
        err = _sdk_errors()
        request = self.build_request(prompt, system=system, max_tokens=max_tokens, schema=schema)

        attempts = _MAX_RETRIES + 1
        last: BaseException | None = None
        for attempt in range(attempts):
            try:
                return client.messages.create(**request)
            except err.APITimeoutError as exc:          # most specific first
                last = exc
            except err.APIConnectionError as exc:
                last = exc
            except err.RateLimitError as exc:           # 429
                last = exc
            except err.InternalServerError as exc:      # 5xx
                last = exc
            except err.APIStatusError as exc:
                status = int(getattr(exc, "status_code", 0) or 0)
                if not (status == 429 or status >= 500):
                    raise LLMError(f"Claude request failed with status {status}: {exc}") from exc
                last = exc
            except err.AnthropicError as exc:           # auth, bad request, validation
                raise LLMError(f"Claude request failed: {exc}") from exc
            except Exception as exc:
                # Anything the SDK did not classify -- most plausibly a client
                # too old to accept ``output_config``.  Surfacing it as an
                # LLMError is what lets a pipeline fall back instead of dying
                # with a bare TypeError three frames down.
                raise LLMError(f"Claude request failed: {type(exc).__name__}: {exc}") from exc
            if attempt + 1 < attempts:
                log.warning("Claude request failed (%s); retrying once", type(last).__name__)
        raise LLMError(f"Claude request failed after {attempts} attempts: {last}") from last


# --------------------------------------------------------------------------- #
# response handling
# --------------------------------------------------------------------------- #

def _response_text(response: Any) -> str:
    """First text block of a message, guarded against refusals."""
    stop_reason = getattr(response, "stop_reason", None)
    if stop_reason == "refusal":
        raise LLMError("Claude declined to complete this request (stop_reason='refusal')")
    blocks = getattr(response, "content", None) or []
    try:
        text = next(b.text for b in blocks if getattr(b, "type", None) == "text")
    except StopIteration:
        raise LLMError(f"Claude returned no text block (stop_reason={stop_reason!r})") from None
    if not isinstance(text, str) or not text.strip():
        raise LLMError("Claude returned an empty text block")
    if stop_reason == "max_tokens":
        log.warning("Claude response was truncated by max_tokens")
    return text


def _parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object out of a model reply, fences and preamble tolerated."""
    candidates = [text.strip()]
    fenced = _JSON_FENCE_RE.search(text)
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    start, end = text.find("{"), text.rfind("}")
    if 0 <= start < end:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        if not candidate:
            continue
        try:
            data = json.loads(candidate)
        except ValueError:
            continue
        if isinstance(data, dict):
            return data
        return {"result": data}
    raise LLMError(f"Claude returned text that is not JSON: {text[:200]!r}")
