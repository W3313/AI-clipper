"""The local, self-hosted provider: an OpenAI-compatible chat endpoint over urllib.

Ollama, llama.cpp's ``server``, LM Studio and vLLM all expose the same
``POST {base}/chat/completions`` contract, so one client reaches every one of
them and no third-party SDK is needed -- :mod:`urllib.request` is enough, which
keeps this module importable on a bare interpreter like every other backend.

**Structured output is the hard part.**  A 7B model quantised onto a laptop is
far less reliable at emitting strict JSON than a frontier model, so
:meth:`LocalProvider.complete_json` never assumes anything and degrades in a
fixed order:

1. ask for schema-guided decoding (``response_format`` ``json_schema``), which
   vLLM, llama.cpp and LM Studio implement as a real grammar constraint;
2. on a rejection (400/422, or a body naming ``response_format``) retry with
   ``json_object`` plus the schema restated in the prompt -- Ollama's older
   OpenAI shim only knows this one;
3. on a second rejection, plain text with the schema in the prompt.

The mode that worked is remembered on the instance, so a hundred-clip run probes
once, not a hundred times.  The reply is then parsed defensively (fences,
"Here is the JSON:", trailing prose -- see :func:`extract_json`), validated with
:func:`aiclipper.llm.heuristic.validate_instance`, and, if it does not validate,
given exactly *one* repair round-trip that quotes the violations back.  If that
still fails the answer comes from
:class:`~aiclipper.llm.heuristic.HeuristicProvider` and a warning is logged: a
render must never die because a local model rambled.

**Two different questions, two different calls** -- the same split the speech
layer draws in :mod:`aiclipper.tts.base`.  :meth:`LocalProvider.available` is
the *routing* check: base URL configured, not offline, no packets, fast enough
for the hot path of every render.  :meth:`LocalProvider.usable` is the
*diagnostic* check: a real, timeout-bounded ``GET {base}/models`` that answers
"is a model actually being served right now?", cached for the life of the
process.  ``available()`` says where the server should be; ``usable()`` says
something is listening there with a model loaded.
"""

from __future__ import annotations

import contextlib
import json
import logging
import re
import threading
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from typing import Any

from ..config import Settings
from ..errors import LLMError
from .base import _settings as _resolve_settings
from .heuristic import HeuristicProvider, validate_instance

log = logging.getLogger(__name__)

__all__ = [
    "LocalProvider", "JSON_MODES", "CHAT_PATH", "MODELS_PATH", "PROBE_TIMEOUT",
    "DEFAULT_TEMPERATURE", "BASE_URL_ENV", "MODEL_ENV", "TIMEOUT_ENV",
    "build_request", "join_url", "strip_fences", "extract_json", "schema_hint",
    "repair_prompt", "run_bounded", "cached_usable", "reset_usable_cache",
]

#: Structured-output strategies, strongest first.  :meth:`LocalProvider.complete_json`
#: walks this list and stops at the first one the endpoint accepts.
JSON_MODES: tuple[str, ...] = ("json_schema", "json_object", "prompt")

#: Paths appended to ``settings.llm_base_url`` (which carries the API prefix,
#: e.g. ``http://localhost:11434/v1``).
CHAT_PATH = "/chat/completions"
MODELS_PATH = "/models"

#: Wall-clock bound, in seconds, on one :meth:`LocalProvider.usable` probe.  A
#: diagnostic that hangs is a diagnostic nobody runs, and this one exists to be
#: run when the endpoint is *suspected* of being down.
PROBE_TIMEOUT = 4.0

#: Low but not zero: structured output is more reliable when the sampler is not
#: adventurous, and hard rule 7 asks for determinism where it is free.  ``seed``
#: travels with it, from ``settings.seed``.
DEFAULT_TEMPERATURE = 0.2

#: Environment variables named in error messages, so a failure tells the reader
#: which knob to turn.
BASE_URL_ENV = "AICLIP_LLM_BASE_URL"
MODEL_ENV = "AICLIP_LLM_LOCAL_MODEL"
TIMEOUT_ENV = "AICLIP_LLM_TIMEOUT"

#: Substrings in an error body that mean "I do not support that
#: ``response_format``" rather than "your request was bad".  Servers disagree
#: about the status code but all of them name the field.
_REJECTION_MARKERS: tuple[str, ...] = (
    "response_format", "response format", "json_schema", "json schema",
    "json_object", "json object", "structured output", "guided_json", "grammar",
)

#: Status codes that mean the same thing when a ``response_format`` was sent.
_REJECTION_CODES: frozenset[int] = frozenset({400, 422, 501})

#: ```` ```json ... ``` ```` including the unterminated fence a truncated reply leaves.
_FENCE_RE = re.compile(r"```[ \t]*[A-Za-z0-9_+.-]*[ \t]*\r?\n?(?P<body>.*?)(?:```|\Z)", re.DOTALL)

#: Cap on how many opening braces are tried as extraction starts, so a
#: pathological reply cannot make parsing quadratic.
_MAX_JSON_STARTS = 64

#: How much of a bad reply is quoted back in a repair prompt.
_REPAIR_QUOTE_CHARS = 2000

#: How many violations are quoted back.  Beyond a handful the model stops reading.
_REPAIR_MAX_PROBLEMS = 12


class _ModeRejected(Exception):
    """The endpoint refused this ``response_format``; try a weaker one.

    Deliberately *not* an :class:`~aiclipper.errors.LLMError`: it is a signal
    inside the mode ladder, and only the bottom rung's failure is a real error.
    """


# --------------------------------------------------------------------------- #
# capability probing: bounded, cached, and never on the render hot path
# --------------------------------------------------------------------------- #

_USABLE_CACHE: dict[str, bool] = {}
_USABLE_LOCK = threading.Lock()


def run_bounded(call: Callable[[], bool], timeout: float, *, default: bool = False) -> bool:
    """Run ``call`` on a daemon thread and give up after ``timeout`` seconds.

    ``urlopen``'s own timeout bounds each socket operation rather than the call
    as a whole, so a server that dribbles bytes can outlast it; and
    ``concurrent.futures`` cannot help, because its shutdown *waits* for the
    worker.  A daemon thread we simply stop waiting for cannot keep the process
    alive, so a wedged probe costs one abandoned thread and nothing else.
    """
    box: dict[str, Any] = {}

    def _target() -> None:
        try:
            box["value"] = call()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the calling thread
            box["error"] = exc

    thread = threading.Thread(target=_target, name="aiclip-llm-probe", daemon=True)
    thread.start()
    thread.join(max(0.0, float(timeout)))
    if thread.is_alive():
        return default
    if "error" in box:
        raise box["error"]
    return bool(box.get("value", default))


def cached_usable(key: str, probe: Callable[[], bool], *, refresh: bool = False) -> bool:
    """Memoise one capability probe under ``key`` for the life of the process.

    A probe that raises is an endpoint that does not work, so the answer is
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
        log.debug("local llm probe %s failed: %s", key, exc, exc_info=True)
        verdict = False
    with _USABLE_LOCK:
        _USABLE_CACHE[key] = verdict
    return verdict


def reset_usable_cache(key: str | None = None) -> None:
    """Forget cached probe verdicts -- all of them, or just ``key``.

    For tests, and for a long-lived process that has just been pointed at a
    different endpoint or watched its model server come back up.
    """
    with _USABLE_LOCK:
        if key is None:
            _USABLE_CACHE.clear()
        else:
            _USABLE_CACHE.pop(key, None)


# --------------------------------------------------------------------------- #
# pure helpers: url, request shape, prompt text
# --------------------------------------------------------------------------- #

def join_url(base: str, path: str) -> str:
    """``base`` + ``path`` with exactly one slash between them."""
    return f"{(base or '').rstrip('/')}/{(path or '').lstrip('/')}"


def build_request(url: str, payload: dict[str, Any], *, api_key: str = "") -> urllib.request.Request:
    """Build the POST for one chat completion, without sending it.

    Pure and network-free, which is what makes the request shape testable in an
    offline CI.  ``Authorization`` is sent only when a key is configured: local
    servers usually want no header at all, and sending an empty bearer token is
    a good way to earn a 401 from the ones that do check.
    """
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )


def strip_fences(text: str) -> str:
    """The inside of the first ``` fence, or the whole text stripped.

    Handles the fence a local model actually emits: with or without an info
    string, with or without the closing fence (a reply cut off by ``max_tokens``
    never closes it).
    """
    match = _FENCE_RE.search(text or "")
    if match is None:
        return (text or "").strip()
    return match.group("body").strip()


def _balanced_span(text: str, start: int, opener: str, closer: str) -> str | None:
    """The substring from ``start`` to its matching ``closer``, or ``None``.

    Brackets inside JSON strings do not count, and ``\\"`` does not end a string
    -- which is the whole reason this is not a ``find``/``rfind`` pair.
    """
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            depth += 1
        elif char == closer:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
    return None


def _candidates(text: str, opener: str, closer: str) -> Iterator[str]:
    """Balanced spans starting at each ``opener``, outermost first."""
    seen = 0
    for index, char in enumerate(text):
        if char != opener:
            continue
        span = _balanced_span(text, index, opener, closer)
        if span is not None:
            yield span
        seen += 1
        if seen >= _MAX_JSON_STARTS:
            return


def extract_json(text: str) -> dict[str, Any] | None:
    """Pull the JSON object out of a model reply, or ``None`` if there is none.

    Written for what local models really do: wrap the object in a ``` fence,
    open with "Here is the JSON:", close with a paragraph of explanation, or all
    three at once.  The fenced body is preferred, the raw text is the fallback,
    and within each the *outermost* balanced object wins -- degrading to an
    inner one only when the outer span is not parseable JSON, which is what
    rescues a reply whose preamble happens to contain a brace.

    An object is preferred wherever it sits, including inside an array the
    model wrapped it in -- every schema this package asks for has an object at
    the root.  A reply that really is a list of scalars is accepted too and
    wrapped as ``{"result": [...]}``, so the return type stays honest whatever
    the model felt like emitting.
    """
    raw = (text or "").strip()
    fenced = strip_fences(raw)
    sources = [fenced] if fenced else []
    if raw and raw != fenced:
        sources.append(raw)
    for source in sources:
        for opener, closer in (("{", "}"), ("[", "]")):
            for candidate in _candidates(source, opener, closer):
                try:
                    value = json.loads(candidate)
                except ValueError:
                    continue
                return value if isinstance(value, dict) else {"result": value}
    return None


def schema_hint(schema: dict[str, Any]) -> str:
    """The "reply with exactly this shape" block appended to a prompt.

    Used by the two weaker modes, where the schema cannot ride along in
    ``response_format`` and the prompt is the only place left to put it.
    """
    body = json.dumps(schema, indent=2, sort_keys=True, default=str)
    return (
        "Reply with a single JSON object and nothing else: no prose, no explanation, "
        "no markdown code fence.\n"
        "The object must validate against this JSON Schema:\n"
        f"{body}"
    )


def repair_prompt(prompt: str, reply: str, problems: list[str], schema: dict[str, Any]) -> str:
    """Quote the violations back and ask for one corrected object."""
    quoted = (reply or "").strip()[:_REPAIR_QUOTE_CHARS] or "<empty reply>"
    listed = "\n".join(f"  - {p}" for p in problems[:_REPAIR_MAX_PROBLEMS]) or "  - it was not JSON"
    return (
        "Your previous reply did not satisfy the required JSON Schema.\n\n"
        f"Your previous reply was:\n{quoted}\n\n"
        f"These problems must be fixed:\n{listed}\n\n"
        "Return the corrected object only. Keep everything that was already correct, "
        "change only what the problems above name.\n\n"
        f"The original request was:\n{prompt}\n\n"
        f"{schema_hint(schema)}"
    )


def _schema_name(schema: dict[str, Any]) -> str:
    """A ``json_schema.name`` some servers insist on validating."""
    raw = str(schema.get("title") or "response") if isinstance(schema, dict) else "response"
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", raw).strip("_")
    return (cleaned or "response")[:64]


# --------------------------------------------------------------------------- #
# transport
# --------------------------------------------------------------------------- #

def _transport(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
    """Send ``request`` and return ``(status, body)``.

    The single seam the whole module funnels through, which is what lets the
    tests replace the network with a recorder and still exercise every branch
    above and below it.
    """
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(getattr(response, "status", 200) or 200), response.read()


def _decode(body: Any) -> str:
    if isinstance(body, (bytes, bytearray)):
        text = bytes(body).decode("utf-8", "replace")
    else:
        text = str(body or "")
    text = text.strip()
    return text[:800] if text else "<empty body>"


def _read_error(exc: urllib.error.HTTPError) -> str:
    """The body of an ``HTTPError``, without leaking its socket."""
    raw: bytes = b""
    try:
        raw = exc.read()
    except Exception:  # noqa: BLE001 - a body we cannot read is simply absent
        raw = b""
    finally:
        with contextlib.suppress(Exception):
            exc.close()
    return _decode(raw)


def _describe_error(error: Any) -> str:
    if isinstance(error, dict):
        message = error.get("message") or error.get("detail") or error.get("type")
        if message:
            return str(message)[:800]
    return _decode(error)


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #

class LocalProvider:
    """A self-hosted model behind an OpenAI-compatible chat endpoint.

    ``json_mode`` pins the structured-output strategy instead of discovering it;
    left ``None`` the first :meth:`complete_json` probes down the ladder and the
    winner is remembered for the life of the instance.
    """

    name = "local"

    def __init__(self, *, settings: Settings | None = None, json_mode: str | None = None) -> None:
        self._settings = _resolve_settings(settings)
        if json_mode is not None and json_mode not in JSON_MODES:
            raise LLMError(f"unknown json mode {json_mode!r}; expected one of: {', '.join(JSON_MODES)}")
        self._json_mode = json_mode

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LocalProvider(model={self._settings.llm_local_model!r}, url={self.chat_url!r})"

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def model(self) -> str:
        return self._settings.llm_local_model

    @property
    def chat_url(self) -> str:
        return join_url(self._settings.llm_base_url, CHAT_PATH)

    @property
    def models_url(self) -> str:
        return join_url(self._settings.llm_base_url, MODELS_PATH)

    @property
    def timeout(self) -> float:
        """Seconds one completion may take.

        A quantised model on a laptop CPU is slow, so the default is generous.
        Zero or nonsense falls back to that default rather than to "wait
        forever", which is what an unbounded ``urlopen`` would mean here.
        """
        try:
            configured = float(self._settings.llm_timeout or 0)
        except (TypeError, ValueError):  # pragma: no cover - Settings coerces already
            configured = 0.0
        return max(1.0, configured or 120.0)

    @property
    def json_mode(self) -> str | None:
        """The structured-output mode this endpoint accepted, once one has."""
        return self._json_mode

    # -- capability -------------------------------------------------------- #
    def available(self) -> bool:
        """True when an endpoint is configured and we are not running offline.

        Deliberately a configuration check and nothing more: it is consulted on
        the hot path of every render, so it must never open a socket.  Whether
        anything is actually listening is :meth:`usable`.
        """
        if self._settings.offline:
            return False
        return bool((self._settings.llm_base_url or "").strip())

    def usable(self, *, timeout: float | None = None, refresh: bool = False) -> bool:
        """Is a model really being served at this endpoint, right now?

        This is what a ``doctor``-style diagnostic should call.  :meth:`available`
        proves only that a URL is configured -- a stopped Ollama, a typo in the
        port, a server with no model loaded all pass it, and reporting ``OK`` for
        one of those misleads the person running it.  This spends one cheap
        ``GET`` on ``{base}/models`` bounded to ``timeout`` seconds (default
        :data:`PROBE_TIMEOUT`) and counts an endpoint as usable only when it
        answers 200 *and* names at least one model.

        Cached for the process, keyed by URL and timeout, so a diagnostic pays
        once; ``refresh=True`` probes again regardless.  Never raises.
        """
        if not self.available():
            return False
        limit = float(timeout) if timeout and timeout > 0 else PROBE_TIMEOUT
        key = f"local:{self.models_url}:{limit:g}"
        return cached_usable(key, lambda: self._probe(limit), refresh=refresh)

    def _probe(self, timeout: float) -> bool:
        """One bounded ``GET {base}/models``; ``False`` for anything unexpected."""
        headers = {"Accept": "application/json"}
        if self._settings.llm_api_key:
            headers["Authorization"] = f"Bearer {self._settings.llm_api_key}"
        request = urllib.request.Request(self.models_url, headers=headers, method="GET")

        def _ask() -> bool:
            try:
                status, body = _transport(request, timeout)
            except urllib.error.HTTPError as exc:
                log.debug("local llm probe rejected with HTTP %s: %s", exc.code, _read_error(exc))
                return False
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                log.debug("local llm probe could not reach %s: %s", self.models_url, exc)
                return False
            if status != 200:
                return False
            served = _model_ids(body)
            if served and self.model and not _serves(served, self.model):
                log.warning(
                    "local llm at %s is serving %s but %s is set to %r; the request may 404",
                    self.models_url, ", ".join(sorted(served)[:6]), MODEL_ENV, self.model,
                )
            return bool(served)

        # urlopen's timeout bounds each socket operation rather than the call as
        # a whole, so the probe gets an outer bound as well.
        return run_bounded(_ask, timeout + 0.5, default=False)

    # -- completion -------------------------------------------------------- #
    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None) -> str:
        """Plain-text completion.  No schema, so no ``response_format``."""
        payload = self.build_payload(prompt, system=system, max_tokens=max_tokens)
        return self._send(payload)

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Structured completion, defended at every step.

        Walks :data:`JSON_MODES` until the endpoint accepts one, parses the reply
        with :func:`extract_json`, validates it, allows itself exactly one repair
        round-trip, and falls back to
        :class:`~aiclipper.llm.heuristic.HeuristicProvider` rather than raising if
        the model still cannot produce a conforming object.  Transport failures
        (nothing listening, a timeout, a 404 on the model) *do* raise
        :class:`~aiclipper.errors.LLMError`: those are worth telling the user
        about, and every call site already falls back on them.
        """
        text, mode = self._structured(prompt, schema, system=system, max_tokens=max_tokens)
        data, problems = self._interpret(text, schema)
        if data is not None and not problems:
            return data

        log.info(
            "local model %r returned a non-conforming object (%s); asking it to repair",
            self.model, "; ".join(problems[:3]) or "unparseable",
        )
        repaired = self._repair(
            prompt, schema, system=system, max_tokens=max_tokens, mode=mode,
            reply=text, problems=problems,
        )
        if repaired is not None:
            return repaired

        log.warning(
            "local model %r could not produce an object matching the schema after one repair; "
            "falling back to the heuristic provider so the render survives",
            self.model,
        )
        return HeuristicProvider(settings=self._settings).complete_json(
            prompt, schema, system=system, max_tokens=max_tokens,
        )

    # -- request building -------------------------------------------------- #
    def build_payload(
        self,
        prompt: str,
        *,
        system: str = "",
        max_tokens: int | None = None,
        schema: dict[str, Any] | None = None,
        mode: str = "prompt",
    ) -> dict[str, Any]:
        """The exact JSON body POSTed to ``{base}/chat/completions``.

        Separate from the call that sends it, so the request shape -- the part
        that silently rots -- is testable with no network at all.  ``schema`` is
        ``None`` for plain completions; otherwise ``mode`` decides whether it
        travels in ``response_format``, in the prompt, or both.
        """
        if mode not in JSON_MODES:
            raise LLMError(f"unknown json mode {mode!r}; expected one of: {', '.join(JSON_MODES)}")
        s = self._settings
        content = prompt
        if schema is not None and mode != "json_schema":
            content = f"{prompt}\n\n{schema_hint(schema)}"

        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})

        payload: dict[str, Any] = {
            "model": s.llm_local_model,
            "messages": messages,
            "max_tokens": max(1, int(max_tokens or s.llm_max_tokens)),
            "temperature": DEFAULT_TEMPERATURE,
            "seed": int(s.seed),
            "stream": False,
        }
        if schema is not None and mode == "json_schema":
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": _schema_name(schema), "schema": schema, "strict": True},
            }
        elif schema is not None and mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        return payload

    # -- internals --------------------------------------------------------- #
    def _modes(self) -> tuple[str, ...]:
        """The ladder still worth trying: everything from the remembered rung down."""
        if self._json_mode in JSON_MODES:
            return JSON_MODES[JSON_MODES.index(str(self._json_mode)):]
        return JSON_MODES

    def _structured(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str,
        max_tokens: int | None,
    ) -> tuple[str, str]:
        """``(reply text, mode that worked)``, downgrading through the ladder."""
        rejection: LLMError | None = None
        for mode in self._modes():
            payload = self.build_payload(
                prompt, system=system, max_tokens=max_tokens, schema=schema, mode=mode,
            )
            try:
                text = self._send(payload, mode=mode)
            except _ModeRejected as exc:
                log.warning(
                    "local llm at %s rejected response_format mode %r (%s); retrying with the next mode",
                    self.chat_url, mode, exc,
                )
                rejection = LLMError(str(exc))
                continue
            if self._json_mode != mode:
                log.debug("local llm at %s accepted response_format mode %r", self.chat_url, mode)
            self._json_mode = mode
            return text, mode
        raise rejection or LLMError(f"local llm at {self.chat_url} accepted no structured-output mode")

    def _interpret(self, text: str, schema: dict[str, Any]) -> tuple[dict[str, Any] | None, list[str]]:
        """Parse and validate one reply; ``problems`` empty means it is good."""
        data = extract_json(text)
        if data is None:
            return None, ["$: the reply contained no JSON object"]
        return data, validate_instance(data, schema)

    def _repair(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str,
        max_tokens: int | None,
        mode: str,
        reply: str,
        problems: list[str],
    ) -> dict[str, Any] | None:
        """Exactly one corrective round-trip.  ``None`` means "still no good"."""
        payload = self.build_payload(
            repair_prompt(prompt, reply, problems, schema),
            system=system, max_tokens=max_tokens, schema=schema, mode=mode,
        )
        try:
            text = self._send(payload, mode=mode)
        except (_ModeRejected, LLMError) as exc:
            # The first request already succeeded, so a failure here is a blip,
            # not a misconfiguration: take the heuristic answer instead of
            # turning a recoverable situation into a dead render.
            log.warning("local llm repair round-trip failed: %s", exc)
            return None
        data, remaining = self._interpret(text, schema)
        if data is not None and not remaining:
            return data
        log.debug("local llm repair still violated the schema: %s", "; ".join(remaining[:5]))
        return None

    def _send(self, payload: dict[str, Any], *, mode: str = "prompt") -> str:
        """POST one completion and return the assistant's text.

        ``mode`` names the ``response_format`` strategy that produced ``payload``
        so that a rejection of *the format* can be told apart from a rejection of
        the request; plain completions pass the default and never downgrade.
        """
        if self._settings.offline:
            raise LLMError("settings.offline is set; the local llm provider cannot be used")
        if not (self._settings.llm_base_url or "").strip():
            raise LLMError(
                f"no local llm endpoint is configured; set {BASE_URL_ENV} to the server's API root "
                "(e.g. http://localhost:11434/v1 for Ollama)"
            )

        url = self.chat_url
        request = build_request(url, payload, api_key=self._settings.llm_api_key)
        try:
            status, body = _transport(request, self.timeout)
        except urllib.error.HTTPError as exc:
            detail = _read_error(exc)
            status = int(getattr(exc, "code", 0) or 0)
            if _is_mode_rejection(status, detail, mode):
                raise _ModeRejected(f"HTTP {status}: {detail}") from exc
            raise self._http_error(status, detail) from exc
        except urllib.error.URLError as exc:
            raise self._reach_error(exc.reason) from exc
        except (TimeoutError, OSError) as exc:
            # A read timeout surfaces as a bare socket error, not a URLError.
            raise self._reach_error(exc) from exc

        if status != 200:
            detail = _decode(body)
            if _is_mode_rejection(status, detail, mode):
                raise _ModeRejected(f"HTTP {status}: {detail}")
            raise self._http_error(status, detail)
        return self._message_text(body)

    def _message_text(self, body: bytes) -> str:
        """``choices[0].message.content``, defended against every shape seen in the wild."""
        try:
            data = json.loads(bytes(body).decode("utf-8", "replace") or "{}")
        except ValueError as exc:
            raise LLMError(
                f"local llm at {self.chat_url} returned a body that is not JSON: {_decode(body)}"
            ) from exc
        if not isinstance(data, dict):
            raise LLMError(f"local llm at {self.chat_url} returned {type(data).__name__}, not an object")

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            error = data.get("error")
            if error:
                raise LLMError(f"local llm at {self.chat_url} reported: {_describe_error(error)}")
            raise LLMError(f"local llm at {self.chat_url} returned no choices: {_decode(body)}")

        first = choices[0] if isinstance(choices[0], dict) else {}
        message = first.get("message")
        content: Any = message.get("content") if isinstance(message, dict) else None
        if isinstance(content, list):  # some servers emit OpenAI "content parts"
            content = "".join(
                str(part.get("text", "")) for part in content if isinstance(part, dict)
            )
        if not isinstance(content, str) or not content.strip():
            legacy = first.get("text")  # /completions-style servers
            content = legacy if isinstance(legacy, str) else ""
        if not content.strip():
            raise LLMError(
                f"local model {self.model!r} returned an empty message "
                f"(finish_reason={first.get('finish_reason')!r})"
            )
        if first.get("finish_reason") == "length":
            log.warning(
                "local model %r hit its token limit; raise AICLIP_LLM_MAX_TOKENS if the reply is cut off",
                self.model,
            )
        return content

    # -- error mapping ----------------------------------------------------- #
    def _http_error(self, status: int, detail: str) -> LLMError:
        """An HTTP status turned into something the reader can act on."""
        url = self.chat_url
        if status == 404:
            return LLMError(
                f"local model {self.model!r} was not found at {url} (HTTP 404: {detail}). "
                f"Pull it first -- `ollama pull {self.model}` -- or point {MODEL_ENV} at a model "
                f"this server already serves; if the endpoint is not Ollama, check that "
                f"{BASE_URL_ENV} includes the API prefix (e.g. http://localhost:8000/v1)."
            )
        if status in (401, 403):
            return LLMError(
                f"local llm at {url} refused the credentials (HTTP {status}: {detail}). "
                "Set AICLIP_LLM_API_KEY to the token this server expects."
            )
        if status == 413:
            return LLMError(
                f"local llm at {url} rejected the request as too large (HTTP 413: {detail}). "
                "Lower AICLIP_LLM_MAX_TOKENS or serve the model with a larger context window."
            )
        return LLMError(f"local llm at {url} returned HTTP {status}: {detail}")

    def _reach_error(self, reason: Any) -> LLMError:
        """A transport failure turned into something the reader can act on."""
        url = self.chat_url
        text = str(reason)
        if isinstance(reason, TimeoutError) or "timed out" in text.lower():
            return LLMError(
                f"local llm at {url} did not answer within {self.timeout:g}s. Raise {TIMEOUT_ENV}, "
                f"serve a smaller model, or lower AICLIP_LLM_MAX_TOKENS."
            )
        if isinstance(reason, ConnectionRefusedError) or "refused" in text.lower():
            return LLMError(
                f"local llm endpoint {url} is not reachable (connection refused). Start the model "
                f"server (e.g. `ollama serve`) or point {BASE_URL_ENV} at one that is running."
            )
        return LLMError(
            f"local llm endpoint {url} is not reachable: {text}. Check that the server is running "
            f"and that {BASE_URL_ENV} is correct."
        )


# --------------------------------------------------------------------------- #
# response helpers
# --------------------------------------------------------------------------- #

def _is_mode_rejection(status: int, detail: str, mode: str) -> bool:
    """Did the endpoint refuse the *format* rather than the request?

    ``prompt`` mode sends no ``response_format`` at all, so nothing it gets back
    is ever a format rejection -- a 400 there is a real error and must surface
    as one instead of silently exhausting the ladder.
    """
    if mode == "prompt" or mode not in JSON_MODES:
        return False
    lowered = (detail or "").lower()
    if any(marker in lowered for marker in _REJECTION_MARKERS):
        return True
    return status in _REJECTION_CODES


def _model_ids(body: bytes) -> set[str]:
    """Model ids from a ``GET /models`` body, tolerating both list shapes."""
    try:
        data = json.loads(bytes(body).decode("utf-8", "replace") or "{}")
    except ValueError:
        return set()
    rows: Any = data.get("data") if isinstance(data, dict) else data
    if isinstance(data, dict) and not isinstance(rows, list):
        rows = data.get("models")
    if not isinstance(rows, list):
        return set()
    ids: set[str] = set()
    for row in rows:
        if isinstance(row, dict):
            value = row.get("id") or row.get("name") or row.get("model")
        else:
            value = row
        if isinstance(value, str) and value.strip():
            ids.add(value.strip())
    return ids


def _serves(served: set[str], wanted: str) -> bool:
    """Is ``wanted`` one of ``served``, allowing for Ollama's ``:latest`` tags?"""
    target = wanted.strip().lower()
    if not target:
        return True
    for name in served:
        lowered = name.strip().lower()
        if lowered == target or lowered.split(":", 1)[0] == target.split(":", 1)[0]:
            return True
    return False
