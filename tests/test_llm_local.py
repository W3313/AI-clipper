"""Tests for :mod:`aiclipper.llm.local`.

Everything here runs offline.  There is no model server on this machine and the
network is blocked, so the single transport seam -- ``local._transport`` -- is
replaced by a recorder that answers from a script and remembers every request it
was handed.  That is what makes the interesting parts testable: the exact body
sent in each ``response_format`` mode, the downgrade ladder, the repair
round-trip, and the error messages a user would actually see.  An autouse
fixture turns any real socket connection into a failure, so a test that reaches
for the network fails loudly instead of hanging.
"""

from __future__ import annotations

import io
import json
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

import pytest

from aiclipper.config import get_settings, reset_settings
from aiclipper.errors import LLMError
from aiclipper.llm import local
from aiclipper.llm.base import LOCAL_BASE_URL_ENV, PROVIDER_ALIASES, LLMProvider, get_provider
from aiclipper.llm.claude import ClaudeProvider
from aiclipper.llm.heuristic import HeuristicProvider, validate_instance
from aiclipper.llm.local import (
    JSON_MODES,
    LocalProvider,
    build_request,
    extract_json,
    join_url,
    repair_prompt,
    reset_usable_cache,
    schema_hint,
    strip_fences,
)

BASE_URL = "http://localhost:11434/v1"
CHAT_URL = f"{BASE_URL}/chat/completions"
MODELS_URL = f"{BASE_URL}/models"
MODEL = "llama3.1"
SEED = 7
MAX_TOKENS = 2048

PROMPT = "Pick the best 30 seconds of this talk and title it."
SYSTEM = "You are a ruthless short-form editor."

SCHEMA: dict[str, Any] = {
    "type": "object",
    "title": "Clip Plan",
    "additionalProperties": False,
    "required": ["title", "score"],
    "properties": {
        "title": {"type": "string", "minLength": 3, "maxLength": 40},
        "score": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE",
    "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)


# --------------------------------------------------------------------------- #
# fixtures and doubles
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn any outbound connection attempt into a test failure."""

    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the local llm provider must not touch the network in tests")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    monkeypatch.setattr(socket.socket, "connect_ex", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)


@pytest.fixture(autouse=True)
def _fresh_probe_cache() -> Any:
    reset_usable_cache()
    yield
    reset_usable_cache()


@pytest.fixture
def local_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """A configured, online, keyless local endpoint."""
    for name in CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "anthropic"))
    for name in ("AICLIP_LLM_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AICLIP_OFFLINE", "0")
    monkeypatch.setenv(LOCAL_BASE_URL_ENV, BASE_URL)
    monkeypatch.setenv("AICLIP_LLM_LOCAL_MODEL", MODEL)
    monkeypatch.setenv("AICLIP_LLM_TIMEOUT", "30")
    monkeypatch.setenv("AICLIP_LLM_MAX_TOKENS", str(MAX_TOKENS))
    monkeypatch.setenv("AICLIP_SEED", str(SEED))
    reset_settings()


@pytest.fixture
def provider(local_env: None) -> LocalProvider:
    return LocalProvider()


@dataclass
class Call:
    """One request the provider handed to the transport."""

    url: str
    method: str
    headers: dict[str, str] = field(default_factory=dict)
    body: dict[str, Any] | None = None
    timeout: float = 0.0

    @property
    def user_content(self) -> str:
        messages = (self.body or {}).get("messages") or []
        return "".join(m.get("content", "") for m in messages if m.get("role") == "user")


class FakeTransport:
    """Records every request and answers from a script, strictly in order.

    Running out of scripted replies is an error rather than a repeat: "the
    provider made one more call than it should have" is exactly the bug these
    tests are looking for.
    """

    def __init__(self, *replies: Any) -> None:
        self.replies = list(replies)
        self.calls: list[Call] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
        self.calls.append(
            Call(
                url=request.full_url,
                method=request.get_method(),
                headers={k.lower(): v for k, v in request.header_items()},
                body=json.loads(request.data.decode("utf-8")) if request.data else None,
                timeout=float(timeout),
            )
        )
        if not self.replies:
            raise AssertionError(f"unexpected request #{len(self.calls)} to {request.full_url}")
        reply = self.replies.pop(0)
        if callable(reply):
            return reply()
        if isinstance(reply, BaseException):
            raise reply
        return reply

    @property
    def bodies(self) -> list[dict[str, Any] | None]:
        return [call.body for call in self.calls]


def install(monkeypatch: pytest.MonkeyPatch, *replies: Any) -> FakeTransport:
    fake = FakeTransport(*replies)
    monkeypatch.setattr(local, "_transport", fake)
    return fake


def chat(content: str, *, finish_reason: str = "stop", status: int = 200) -> tuple[int, bytes]:
    """A well-formed OpenAI chat-completion response carrying ``content``."""
    body = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": MODEL,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": finish_reason,
            }
        ],
    }
    return status, json.dumps(body).encode("utf-8")


def raw(payload: Any, *, status: int = 200) -> tuple[int, bytes]:
    return status, json.dumps(payload).encode("utf-8")


def http_error(code: int, body: str = "{}", *, url: str = CHAT_URL) -> urllib.error.HTTPError:
    fp = io.BytesIO(body.encode("utf-8"))
    return urllib.error.HTTPError(url, code, "error", {}, fp)  # type: ignore[arg-type]


def rejecting(code: int = 400, body: str = '{"error":{"message":"response_format is not supported"}}'):
    """A scripted reply that raises a *fresh* ``HTTPError`` when reached."""

    def _raise() -> tuple[int, bytes]:
        raise http_error(code, body)

    return _raise


def raising(exc: BaseException):
    def _raise() -> tuple[int, bytes]:
        raise exc

    return _raise


GOOD = json.dumps({"title": "The pricing was never the problem", "score": 0.82})
BAD_SCORE = json.dumps({"title": "The pricing was never the problem", "score": 5})


# --------------------------------------------------------------------------- #
# request shape: the part that silently rots
# --------------------------------------------------------------------------- #

def test_json_schema_mode_body_is_exact(provider: LocalProvider) -> None:
    payload = provider.build_payload(PROMPT, system=SYSTEM, schema=SCHEMA, mode="json_schema")
    assert payload == {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": PROMPT},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": local.DEFAULT_TEMPERATURE,
        "seed": SEED,
        "stream": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "Clip_Plan", "schema": SCHEMA, "strict": True},
        },
    }


def test_json_object_mode_restates_the_schema_in_the_prompt(provider: LocalProvider) -> None:
    payload = provider.build_payload(PROMPT, schema=SCHEMA, mode="json_object")
    assert payload["response_format"] == {"type": "json_object"}
    content = payload["messages"][0]["content"]
    assert payload["messages"][0]["role"] == "user"
    assert content.startswith(PROMPT)
    assert schema_hint(SCHEMA) in content
    assert '"maxLength": 40' in content


def test_prompt_mode_sends_no_response_format_but_keeps_the_schema(provider: LocalProvider) -> None:
    payload = provider.build_payload(PROMPT, schema=SCHEMA, mode="prompt")
    assert "response_format" not in payload
    assert schema_hint(SCHEMA) in payload["messages"][0]["content"]


def test_a_plain_completion_carries_no_schema_at_all(provider: LocalProvider) -> None:
    payload = provider.build_payload(PROMPT)
    assert "response_format" not in payload
    assert payload["messages"] == [{"role": "user", "content": PROMPT}]


def test_an_unknown_mode_is_refused(provider: LocalProvider) -> None:
    with pytest.raises(LLMError, match="unknown json mode"):
        provider.build_payload(PROMPT, schema=SCHEMA, mode="grammar")
    with pytest.raises(LLMError, match="unknown json mode"):
        LocalProvider(json_mode="grammar")


def test_max_tokens_override(provider: LocalProvider) -> None:
    assert provider.build_payload(PROMPT, max_tokens=64)["max_tokens"] == 64
    assert provider.build_payload(PROMPT, max_tokens=0)["max_tokens"] == MAX_TOKENS
    assert provider.build_payload(PROMPT, max_tokens=-5)["max_tokens"] == 1


@pytest.mark.parametrize(
    ("base", "expected"),
    [
        ("http://localhost:11434/v1", "http://localhost:11434/v1/chat/completions"),
        ("http://localhost:11434/v1/", "http://localhost:11434/v1/chat/completions"),
        ("http://box:8000/v1//", "http://box:8000/v1/chat/completions"),
    ],
)
def test_the_url_is_the_base_plus_one_slash(
    local_env: None, monkeypatch: pytest.MonkeyPatch, base: str, expected: str
) -> None:
    monkeypatch.setenv(LOCAL_BASE_URL_ENV, base)
    reset_settings()
    assert LocalProvider().chat_url == expected
    assert join_url(base, "models") == expected.replace("chat/completions", "models")


def test_no_authorization_header_when_no_key_is_set(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, chat("hello"))
    provider.complete(PROMPT)
    headers = fake.calls[0].headers
    assert "authorization" not in headers
    assert headers["content-type"] == "application/json"
    assert fake.calls[0].method == "POST"
    assert fake.calls[0].url == CHAT_URL


def test_the_authorization_header_is_sent_when_a_key_is_set(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_LLM_API_KEY", "sk-local-abc")
    reset_settings()
    fake = install(monkeypatch, chat("hello"))
    LocalProvider().complete(PROMPT)
    assert fake.calls[0].headers["authorization"] == "Bearer sk-local-abc"


def test_build_request_is_pure() -> None:
    request = build_request(CHAT_URL, {"model": MODEL}, api_key="k")
    assert request.full_url == CHAT_URL
    assert json.loads(request.data.decode("utf-8")) == {"model": MODEL}
    assert request.get_header("Authorization") == "Bearer k"
    assert build_request(CHAT_URL, {}).get_header("Authorization") is None


def test_the_timeout_comes_from_settings(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_LLM_TIMEOUT", "12.5")
    reset_settings()
    fake = install(monkeypatch, chat("hi"))
    LocalProvider().complete(PROMPT)
    assert fake.calls[0].timeout == pytest.approx(12.5)


def test_plain_completion_returns_the_text_untouched(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, chat("  two sentences. and a second one.  "))
    assert provider.complete(PROMPT, system=SYSTEM) == "  two sentences. and a second one.  "


# --------------------------------------------------------------------------- #
# the structured-output ladder
# --------------------------------------------------------------------------- #

def test_the_first_call_asks_for_a_schema_and_remembers_that_it_worked(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, chat(GOOD), chat(GOOD))
    assert provider.json_mode is None
    assert provider.complete_json(PROMPT, SCHEMA)["score"] == 0.82
    assert provider.json_mode == "json_schema"

    provider.complete_json(PROMPT, SCHEMA)
    assert len(fake.calls) == 2
    assert all(b["response_format"]["type"] == "json_schema" for b in fake.bodies)


def test_a_400_downgrades_to_json_object_and_the_mode_is_remembered(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, rejecting(400), chat(GOOD), chat(GOOD))
    assert provider.complete_json(PROMPT, SCHEMA)["title"].startswith("The pricing")
    assert provider.json_mode == "json_object"
    assert [b["response_format"]["type"] for b in fake.bodies] == ["json_schema", "json_object"]

    # A second call must not re-probe the mode that already failed.
    provider.complete_json(PROMPT, SCHEMA)
    assert len(fake.calls) == 3
    assert fake.bodies[2]["response_format"] == {"type": "json_object"}


def test_a_422_downgrades_too(provider: LocalProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, rejecting(422, "unprocessable"), chat(GOOD))
    provider.complete_json(PROMPT, SCHEMA)
    assert provider.json_mode == "json_object"


def test_a_body_naming_response_format_downgrades_whatever_the_status(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(
        monkeypatch,
        rejecting(500, '{"error":"json_schema decoding is not compiled in"}'),
        chat(GOOD),
    )
    provider.complete_json(PROMPT, SCHEMA)
    assert provider.json_mode == "json_object"
    assert len(fake.calls) == 2


def test_two_rejections_land_on_plain_prompt_mode(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, rejecting(400), rejecting(400), chat(GOOD))
    provider.complete_json(PROMPT, SCHEMA)
    assert provider.json_mode == "prompt"
    assert "response_format" not in fake.bodies[2]
    assert schema_hint(SCHEMA) in fake.calls[2].user_content


def test_a_400_in_prompt_mode_is_a_real_error_not_another_downgrade(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(
        monkeypatch,
        rejecting(400),
        rejecting(400),
        rejecting(400, '{"error":{"message":"context length exceeded"}}'),
    )
    with pytest.raises(LLMError, match="HTTP 400"):
        provider.complete_json(PROMPT, SCHEMA)
    assert len(fake.calls) == 3


def test_a_pinned_mode_is_used_verbatim(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, chat(GOOD))
    pinned = LocalProvider(json_mode="prompt")
    pinned.complete_json(PROMPT, SCHEMA)
    assert "response_format" not in fake.bodies[0]


# --------------------------------------------------------------------------- #
# JSON extraction: what local models really emit
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("reply", "expected"),
    [
        ('{"a": 1}', {"a": 1}),
        ('  {"a": 1}  ', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('```JSON\r\n{"a": 1}\r\n```', {"a": 1}),
        ('```{"a": 1}```', {"a": 1}),
        ('Here is the JSON:\n{"a": 1}', {"a": 1}),
        ('Sure! Here you go: {"a": 1} Let me know if you want changes.', {"a": 1}),
        ('{"a": 1}\n\nI hope that helps!', {"a": 1}),
        ('Here is the JSON:\n```json\n{"a": 1}\n```\nThat is my answer.', {"a": 1}),
        ('{"a": {"b": {"c": [1, 2]}}}', {"a": {"b": {"c": [1, 2]}}}),
        ('{"a": "}{"}', {"a": "}{"}),
        ('{"a": "a } b { c"}\ntrailing', {"a": "a } b { c"}),
        (r'{"a": "he said \"}\" loudly"}', {"a": 'he said "}" loudly'}),
        (r'{"a": "back\\slash"}', {"a": "back\\slash"}),
        ('```json\n{"a": 1}\n', {"a": 1}),
        ('[1, 2, 3]', {"result": [1, 2, 3]}),
        ('Here: [1, 2, 3]', {"result": [1, 2, 3]}),
        # An object anywhere beats an array wrapping it: every schema this
        # package asks for has an object at the root.
        ('The answer is [{"a": 1}]', {"a": 1}),
        ('{"a": 1} {"b": 2}', {"a": 1}),
        ("", None),
        ("   ", None),
        ("I cannot help with that.", None),
        ("{not json at all", None),
        ("```\nnothing here\n```", None),
    ],
)
def test_extract_json(reply: str, expected: dict[str, Any] | None) -> None:
    assert extract_json(reply) == expected


def test_extraction_prefers_the_outermost_object() -> None:
    assert extract_json('{"outer": {"inner": 1}}') == {"outer": {"inner": 1}}


def test_extraction_falls_back_to_an_inner_object_when_the_outer_is_broken() -> None:
    # A preamble that opens a brace it never closes properly still must not hide
    # the real object further down the reply.
    assert extract_json('{ note: this is not json {"a": 1} }') == {"a": 1}


def test_extraction_survives_a_very_long_preamble() -> None:
    reply = ("thinking out loud. " * 500) + '{"a": 1}'
    assert extract_json(reply) == {"a": 1}


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("plain text", "plain text"),
        ("```json\n{}\n```", "{}"),
        ("```\nbody\n```", "body"),
        ("prefix\n```json\n{}\n```\nsuffix", "{}"),
        ("```json\n{} unterminated", "{} unterminated"),
    ],
)
def test_strip_fences(text: str, expected: str) -> None:
    assert strip_fences(text) == expected


# --------------------------------------------------------------------------- #
# validation, one repair, then the heuristic floor
# --------------------------------------------------------------------------- #

def test_a_schema_violation_triggers_exactly_one_repair(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, chat(BAD_SCORE), chat(GOOD))
    result = provider.complete_json(PROMPT, SCHEMA, system=SYSTEM)

    assert result == json.loads(GOOD)
    assert len(fake.calls) == 2, "exactly one repair round-trip"
    repair = fake.calls[1].user_content
    assert "above maximum 1" in repair, "the violations are quoted back"
    assert BAD_SCORE in repair, "so is the reply that caused them"
    assert PROMPT in repair, "and the original request"
    assert fake.bodies[1]["messages"][0] == {"role": "system", "content": SYSTEM}


def test_an_unparseable_reply_is_repaired_too(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, chat("I am not able to produce JSON."), chat(GOOD))
    assert provider.complete_json(PROMPT, SCHEMA) == json.loads(GOOD)
    assert len(fake.calls) == 2
    assert "no JSON object" in fake.calls[1].user_content


def test_the_repair_keeps_the_mode_that_the_endpoint_accepted(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, rejecting(400), chat(BAD_SCORE), chat(GOOD))
    provider.complete_json(PROMPT, SCHEMA)
    assert fake.bodies[2]["response_format"] == {"type": "json_object"}


def test_a_failed_repair_falls_back_to_the_heuristic_provider(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    fake = install(monkeypatch, chat(BAD_SCORE), chat(BAD_SCORE))
    with caplog.at_level("WARNING", logger="aiclipper.llm.local"):
        result = provider.complete_json(PROMPT, SCHEMA)

    assert len(fake.calls) == 2, "one attempt plus one repair, and no more"
    assert validate_instance(result, SCHEMA) == [], "a valid object still comes back"
    assert result != json.loads(BAD_SCORE)
    assert result == HeuristicProvider().complete_json(PROMPT, SCHEMA)
    assert "heuristic" in caplog.text


def test_a_reply_that_is_never_json_still_yields_a_valid_object(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, chat("no."), chat("still no."))
    assert validate_instance(provider.complete_json(PROMPT, SCHEMA), SCHEMA) == []


def test_a_transport_blip_during_the_repair_does_not_kill_the_render(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    refused = urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))
    install(monkeypatch, chat(BAD_SCORE), raising(refused))
    assert validate_instance(provider.complete_json(PROMPT, SCHEMA), SCHEMA) == []


def test_a_valid_reply_is_returned_without_any_repair(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, chat(f"Here you go:\n```json\n{GOOD}\n```"))
    assert provider.complete_json(PROMPT, SCHEMA) == json.loads(GOOD)
    assert len(fake.calls) == 1


def test_repair_prompt_is_pure_and_bounded() -> None:
    text = repair_prompt(PROMPT, "x" * 9000, ["$.score: 5 is above maximum 1"], SCHEMA)
    assert "above maximum 1" in text
    assert PROMPT in text
    assert len(text) < 9000


# --------------------------------------------------------------------------- #
# available(): routing, and not a single packet
# --------------------------------------------------------------------------- #

def test_available_is_true_when_a_base_url_is_configured(provider: LocalProvider) -> None:
    assert provider.available() is True


def test_available_is_false_when_offline(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    reset_settings()
    assert LocalProvider().available() is False


@pytest.mark.parametrize("value", ["", "   "])
def test_available_is_false_without_a_base_url(
    local_env: None, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    monkeypatch.setenv(LOCAL_BASE_URL_ENV, value)
    reset_settings()
    assert get_settings().llm_base_url == ""
    assert LocalProvider().available() is False


def test_available_opens_no_socket(provider: LocalProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    def _boom(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("available() must not perform network access")

    monkeypatch.setattr(local, "_transport", _boom)
    monkeypatch.setattr(urllib.request, "urlopen", _boom)
    assert [provider.available() for _ in range(50)] == [True] * 50


# --------------------------------------------------------------------------- #
# usable(): a real probe, bounded and cached
# --------------------------------------------------------------------------- #

def test_usable_probes_the_models_endpoint(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, raw({"object": "list", "data": [{"id": "llama3.1:latest"}]}))
    assert provider.usable() is True
    assert fake.calls[0].url == MODELS_URL
    assert fake.calls[0].method == "GET"
    assert fake.calls[0].timeout == pytest.approx(local.PROBE_TIMEOUT)


def test_usable_accepts_the_ollama_style_model_list(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, raw({"models": [{"name": "llama3.1:latest"}]}))
    assert provider.usable() is True


def test_usable_is_false_when_nothing_is_being_served(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, raw({"object": "list", "data": []}))
    assert provider.usable() is False


def test_usable_is_false_when_the_endpoint_refuses_the_connection(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, raising(urllib.error.URLError(ConnectionRefusedError(111, "refused"))))
    assert provider.usable() is False


def test_usable_is_false_on_an_http_error(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, rejecting(401, "unauthorized"))
    assert provider.usable() is False


def test_usable_warns_when_the_configured_model_is_not_served(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    install(monkeypatch, raw({"data": [{"id": "mistral"}]}))
    with caplog.at_level("WARNING", logger="aiclipper.llm.local"):
        assert provider.usable() is True
    assert "AICLIP_LLM_LOCAL_MODEL" in caplog.text


def test_usable_is_false_when_offline_and_never_probes(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    reset_settings()
    fake = install(monkeypatch)  # any request at all would blow up
    assert LocalProvider().usable() is False
    assert fake.calls == []


def test_usable_times_out_rather_than_hanging(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = threading.Event()

    def _wedged(request: urllib.request.Request, timeout: float) -> tuple[int, bytes]:
        release.wait(30.0)  # a server that accepts the connection and then dribbles
        return raw({"data": [{"id": MODEL}]})

    monkeypatch.setattr(local, "_transport", _wedged)
    started = time.monotonic()
    try:
        assert provider.usable(timeout=0.05) is False
    finally:
        release.set()
    assert time.monotonic() - started < 5.0


def test_usable_is_cached_for_the_process(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, raw({"data": [{"id": MODEL}]}))
    assert provider.usable() is True
    assert provider.usable() is True
    assert LocalProvider().usable() is True, "the cache is keyed by endpoint, not by instance"
    assert len(fake.calls) == 1


def test_a_negative_verdict_is_cached_too_and_refresh_reprobes(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(
        monkeypatch,
        raising(urllib.error.URLError(ConnectionRefusedError(111, "refused"))),
        raw({"data": [{"id": MODEL}]}),
    )
    assert provider.usable() is False
    assert provider.usable() is False
    assert len(fake.calls) == 1
    assert provider.usable(refresh=True) is True
    assert len(fake.calls) == 2


def test_run_bounded_returns_the_default_instead_of_waiting() -> None:
    release = threading.Event()
    try:
        assert local.run_bounded(lambda: release.wait(30.0) or True, 0.05, default=False) is False
    finally:
        release.set()
    assert local.run_bounded(lambda: True, 5.0, default=False) is True


def test_a_probe_that_explodes_is_simply_not_usable() -> None:
    def _boom() -> bool:
        raise RuntimeError("no")

    assert local.cached_usable("test:boom", _boom) is False


# --------------------------------------------------------------------------- #
# error mapping: messages a user can act on
# --------------------------------------------------------------------------- #

def test_connection_refused_names_the_endpoint_and_the_env_var(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, raising(urllib.error.URLError(ConnectionRefusedError(111, "Connection refused"))))
    with pytest.raises(LLMError) as excinfo:
        provider.complete(PROMPT)
    message = str(excinfo.value)
    assert CHAT_URL in message
    assert "not reachable" in message
    assert LOCAL_BASE_URL_ENV in message


def test_a_404_names_the_model_and_suggests_ollama_pull(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, rejecting(404, '{"error":{"message":"model \'llama3.1\' not found"}}'))
    with pytest.raises(LLMError) as excinfo:
        provider.complete(PROMPT)
    message = str(excinfo.value)
    assert MODEL in message
    assert f"ollama pull {MODEL}" in message
    assert "AICLIP_LLM_LOCAL_MODEL" in message


def test_a_404_during_structured_output_is_not_mistaken_for_a_mode_rejection(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = install(monkeypatch, rejecting(404, "model not found"))
    with pytest.raises(LLMError, match="ollama pull"):
        provider.complete_json(PROMPT, SCHEMA)
    assert len(fake.calls) == 1, "a missing model must not walk the whole ladder"


@pytest.mark.parametrize(
    "failure",
    [
        TimeoutError("timed out"),
        urllib.error.URLError(TimeoutError("timed out")),
        OSError("The read operation timed out"),
    ],
)
def test_a_timeout_names_the_timeout_setting(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    install(monkeypatch, raising(failure))
    with pytest.raises(LLMError) as excinfo:
        provider.complete(PROMPT)
    message = str(excinfo.value)
    assert "AICLIP_LLM_TIMEOUT" in message
    assert "30" in message


def test_a_401_points_at_the_api_key_variable(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, rejecting(401, "unauthorized"))
    with pytest.raises(LLMError, match="AICLIP_LLM_API_KEY"):
        provider.complete(PROMPT)


def test_an_unclassified_transport_failure_still_names_the_endpoint(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, raising(urllib.error.URLError("[Errno -2] Name or service not known")))
    with pytest.raises(LLMError, match="not reachable"):
        provider.complete(PROMPT)


def test_offline_settings_block_the_provider(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    reset_settings()
    fake = install(monkeypatch)
    with pytest.raises(LLMError, match="offline"):
        LocalProvider().complete(PROMPT)
    assert fake.calls == []


def test_an_unconfigured_endpoint_says_which_variable_to_set(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(LOCAL_BASE_URL_ENV, "")
    reset_settings()
    fake = install(monkeypatch)
    with pytest.raises(LLMError, match=LOCAL_BASE_URL_ENV):
        LocalProvider().complete(PROMPT)
    assert fake.calls == []


# --------------------------------------------------------------------------- #
# response handling
# --------------------------------------------------------------------------- #

def test_an_empty_message_raises(provider: LocalProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, chat("   "))
    with pytest.raises(LLMError, match="empty message"):
        provider.complete(PROMPT)


def test_an_error_body_with_no_choices_raises(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, raw({"error": {"message": "model is loading"}}))
    with pytest.raises(LLMError, match="model is loading"):
        provider.complete(PROMPT)


def test_a_non_json_body_raises(provider: LocalProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, (200, b"<html>bad gateway</html>"))
    with pytest.raises(LLMError, match="not JSON"):
        provider.complete(PROMPT)


def test_content_parts_are_joined(provider: LocalProvider, monkeypatch: pytest.MonkeyPatch) -> None:
    body = {"choices": [{"message": {"content": [{"text": "one "}, {"text": "two"}]}}]}
    install(monkeypatch, raw(body))
    assert provider.complete(PROMPT) == "one two"


def test_a_truncated_reply_is_flagged_but_still_returned(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    install(monkeypatch, chat("half a sen", finish_reason="length"))
    with caplog.at_level("WARNING", logger="aiclipper.llm.local"):
        assert provider.complete(PROMPT) == "half a sen"
    assert "token limit" in caplog.text


def test_a_non_200_status_without_an_exception_is_still_an_error(
    provider: LocalProvider, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, (503, b'{"error":"overloaded"}'))
    with pytest.raises(LLMError, match="HTTP 503"):
        provider.complete(PROMPT)


# --------------------------------------------------------------------------- #
# provider routing
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "name",
    ["local", "ollama", "openai", "openai-compatible", "llamacpp", "llama.cpp",
     "lmstudio", "lm-studio", "vllm", "  Ollama  ", "VLLM"],
)
def test_every_alias_resolves_to_the_local_backend(local_env: None, name: str) -> None:
    assert isinstance(get_provider(name), LocalProvider)


def test_an_explicit_local_provider_is_returned_even_when_unavailable(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    reset_settings()
    provider = get_provider("ollama")
    assert isinstance(provider, LocalProvider)
    assert provider.available() is False


def test_auto_prefers_local_when_the_base_url_is_set_explicitly(local_env: None) -> None:
    assert isinstance(get_provider("auto"), LocalProvider)
    assert isinstance(get_provider(None), LocalProvider)


def test_auto_falls_back_to_the_heuristic_without_the_env_var(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(LOCAL_BASE_URL_ENV, raising=False)
    reset_settings()
    assert get_settings().llm_base_url == "http://localhost:11434/v1", "the default is still a URL"
    assert isinstance(get_provider("auto"), HeuristicProvider), "but not a statement of intent"


def test_auto_still_prefers_claude_when_a_credential_resolves(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    reset_settings()
    assert isinstance(get_provider("auto"), ClaudeProvider)


def test_auto_ignores_the_base_url_when_offline(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    reset_settings()
    assert isinstance(get_provider("auto"), HeuristicProvider)


def test_settings_llm_provider_drives_the_default(
    local_env: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AICLIP_LLM", "lmstudio")
    reset_settings()
    assert isinstance(get_provider(), LocalProvider)


def test_the_alias_map_covers_exactly_the_known_backends() -> None:
    assert set(PROVIDER_ALIASES.values()) == {"auto", "claude", "heuristic", "local"}


def test_an_unknown_name_still_raises(local_env: None) -> None:
    with pytest.raises(LLMError, match="unknown llm provider"):
        get_provider("llama")


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #

def test_the_local_provider_satisfies_the_protocol(local_env: None) -> None:
    provider = LocalProvider()
    assert isinstance(provider, LLMProvider)
    assert provider.name == "local"
    assert provider.model == MODEL


def test_importing_the_module_touches_no_network_and_no_third_party() -> None:
    code = (
        "import sys, aiclipper.llm.local as m; "
        "print(all(n not in sys.modules for n in ('anthropic', 'requests', 'httpx')))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"


def test_json_modes_are_ordered_strongest_first() -> None:
    assert JSON_MODES == ("json_schema", "json_object", "prompt")


@pytest.mark.parametrize(
    ("configured", "expected"),
    [("30", 30.0), ("0", 120.0), ("-5", 1.0), ("", 120.0), ("nonsense", 120.0)],
)
def test_a_nonsense_timeout_never_means_wait_forever(
    local_env: None, monkeypatch: pytest.MonkeyPatch, configured: str, expected: float
) -> None:
    monkeypatch.setenv("AICLIP_LLM_TIMEOUT", configured)
    reset_settings()
    fake = install(monkeypatch, chat("hi"))
    LocalProvider().complete(PROMPT)
    assert fake.calls[0].timeout == pytest.approx(expected)
