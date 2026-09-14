"""Tests for :mod:`aiclipper.llm`.

Everything here runs offline.  The Anthropic provider is exercised against a
fake client object -- the point of those tests is the *shape of the request*,
which is the part that silently rots -- and an autouse fixture makes any real
socket connection an error, so a test that accidentally reaches for the network
fails loudly instead of hanging or being skipped.
"""

from __future__ import annotations

import json
import os
import re
import socket
import subprocess
import sys
from dataclasses import replace
from typing import Any

import pytest

from aiclipper.config import Settings, get_settings, reset_settings
from aiclipper.errors import LLMError, MissingDependency
from aiclipper.llm import prompts
from aiclipper.llm.base import get_provider
from aiclipper.llm.claude import ClaudeProvider, _parse_json, credentials_available
from aiclipper.llm.heuristic import (
    HeuristicProvider,
    build_instance,
    validate_instance,
    with_subject,
)
from aiclipper.models import ChatMessage, ChatScript, ClipCandidate, RedditPost, ScriptBeat, VideoScript

CREDENTIAL_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_PROFILE", "ANTHROPIC_CONFIG_DIR",
    "ANTHROPIC_FEDERATION_RULE_ID", "ANTHROPIC_ORGANIZATION_ID", "ANTHROPIC_IDENTITY_TOKEN",
    "ANTHROPIC_IDENTITY_TOKEN_FILE",
)

TOPIC = (
    "why sourdough bread rises so much better after a cold overnight proof in the fridge"
)
TRANSCRIPT = (
    "[0.0-4.2] we tried three different launch plans and only one of them actually worked.\n"
    "[4.2-9.0] the thing nobody tells you is that pricing was never really the problem.\n"
    "[9.0-15.5] revenue moved from 12 thousand to 240 thousand in 90 days across 3 markets."
)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn any outbound connection attempt into a test failure."""

    def _boom(*args: Any, **kwargs: Any):
        raise AssertionError("the llm layer must not touch the network in tests")

    monkeypatch.setattr(socket.socket, "connect", _boom)
    monkeypatch.setattr(socket.socket, "connect_ex", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    """No API key, no discoverable ``ant`` profile, online mode."""
    for name in CREDENTIAL_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ANTHROPIC_CONFIG_DIR", str(tmp_path / "anthropic"))
    monkeypatch.setenv("AICLIP_OFFLINE", "0")
    reset_settings()


def online_settings(**overrides: Any) -> Settings:
    return replace(get_settings(), offline=False, **overrides)


# --------------------------------------------------------------------------- #
# fake Anthropic client
# --------------------------------------------------------------------------- #

class _Block:
    def __init__(self, text: str, kind: str = "text") -> None:
        self.text = text
        self.type = kind


class _Response:
    def __init__(self, text: str = "{}", *, stop_reason: str = "end_turn", blocks=None) -> None:
        self.stop_reason = stop_reason
        self.content = blocks if blocks is not None else [_Block(text)]


class FakeMessages:
    """Records every ``create`` call; replays a scripted list of outcomes."""

    def __init__(self, outcomes: list[Any]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0) if self.outcomes else _Response()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


class FakeClient:
    def __init__(self, *outcomes: Any) -> None:
        self.messages = FakeMessages(list(outcomes))


def status_error(status: int, base_name: str = "APIStatusError") -> BaseException:
    """An SDK status error carrying ``status`` without needing a real response."""
    anthropic = pytest.importorskip("anthropic")
    base = getattr(anthropic, base_name)

    class _Err(base):  # type: ignore[misc, valid-type]
        def __init__(self) -> None:
            Exception.__init__(self, f"HTTP {status}")
            self.status_code = status
            self.response = None
            self.body = None

    return _Err()


# --------------------------------------------------------------------------- #
# prompts
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("job", prompts.JOBS)
def test_schemas_are_valid_structured_output_schemas(job: str) -> None:
    schema = prompts.get_schema(job)
    assert prompts.check_schema(schema) == []
    assert schema["type"] == "object"
    assert prompts.get_system(job).strip()


def test_check_schema_catches_loose_objects() -> None:
    loose = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
    }
    problems = prompts.check_schema(loose)
    assert any("additionalProperties" in p for p in problems)
    assert any("required is missing b" in p for p in problems)


def test_check_schema_recurses_into_items_and_defs() -> None:
    nested = {
        "type": "object",
        "additionalProperties": False,
        "required": ["rows"],
        "properties": {
            "rows": {
                "type": "array",
                "items": {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
            }
        },
    }
    assert any("$.rows[]" in p for p in prompts.check_schema(nested))


def test_prompt_builders_carry_their_arguments() -> None:
    assert "sourdough" in prompts.script_prompt(TOPIC, seconds=30)
    assert "78 words" in prompts.script_prompt(TOPIC, seconds=30)  # 30 * 2.6
    assert "9 messages" in prompts.chat_prompt("a premise", turns=9)
    assert "200 words" in prompts.forum_prompt("a premise", words=200)
    text = prompts.highlight_prompt(TRANSCRIPT, count=2, min_duration=20, max_duration=45)
    assert "2 best" in text and "20 and 45 seconds" in text and TRANSCRIPT.splitlines()[0] in text


def test_unknown_job_raises() -> None:
    with pytest.raises(KeyError):
        prompts.get_schema("nonsense")
    with pytest.raises(KeyError):
        prompts.get_system("nonsense")


# --------------------------------------------------------------------------- #
# the validator itself -- a vacuous checker would make every test below vacuous
# --------------------------------------------------------------------------- #

def test_validate_instance_detects_real_violations() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["name", "count", "kind", "tags"],
        "properties": {
            "name": {"type": "string", "minLength": 3, "maxLength": 8, "pattern": "^[a-z]+$"},
            "count": {"type": "integer", "minimum": 1, "maximum": 10, "multipleOf": 2},
            "kind": {"enum": ["a", "b"]},
            "tags": {"type": "array", "minItems": 2, "items": {"type": "string"}, "uniqueItems": True},
        },
    }
    good = {"name": "widget", "count": 4, "kind": "b", "tags": ["one", "two"]}
    assert validate_instance(good, schema) == []

    assert any("missing required" in p for p in validate_instance({"name": "widget"}, schema))
    assert any("additional property" in p for p in validate_instance({**good, "nope": 1}, schema))
    assert any("expected type" in p for p in validate_instance({**good, "count": "4"}, schema))
    assert any("multiple of" in p for p in validate_instance({**good, "count": 3}, schema))
    assert any("above maximum" in p for p in validate_instance({**good, "count": 12}, schema))
    assert any("not one of" in p for p in validate_instance({**good, "kind": "z"}, schema))
    short = validate_instance({**good, "tags": ["x"]}, schema)
    assert any("minItems" in p or "minimum 2" in p for p in short)
    assert any("duplicate" in p for p in validate_instance({**good, "tags": ["x", "x"]}, schema))
    assert any("pattern" in p for p in validate_instance({**good, "name": "Widget"}, schema))
    assert any("maxLength" in p for p in validate_instance({**good, "name": "abcdefghij"}, schema))


def test_validate_instance_handles_refs_and_combinators() -> None:
    schema = {
        "$defs": {"tag": {"type": "string", "minLength": 2}},
        "type": "object",
        "additionalProperties": False,
        "required": ["tag", "either"],
        "properties": {
            "tag": {"$ref": "#/$defs/tag"},
            "either": {"oneOf": [{"type": "string"}, {"type": "integer"}]},
        },
    }
    assert validate_instance({"tag": "ok", "either": 3}, schema) == []
    assert any("minLength" in p for p in validate_instance({"tag": "x", "either": 3}, schema))
    assert any("oneOf" in p for p in validate_instance({"tag": "ok", "either": [1]}, schema))


# --------------------------------------------------------------------------- #
# heuristic provider: the four real schemas
# --------------------------------------------------------------------------- #

@pytest.fixture
def heuristic() -> HeuristicProvider:
    return HeuristicProvider()


REAL_PROMPTS = {
    "highlight": prompts.highlight_prompt(TRANSCRIPT, count=3),
    "script": prompts.script_prompt(TOPIC, seconds=30),
    "chat": prompts.chat_prompt("my roommate Dana has been feeding the neighbour's cat for a month"),
    "forum": prompts.forum_prompt("my coworker took credit for my project so I let him present it alone"),
}


@pytest.mark.parametrize("job", prompts.JOBS)
def test_heuristic_matches_every_real_schema(heuristic: HeuristicProvider, job: str) -> None:
    schema = prompts.get_schema(job)
    data = heuristic.complete_json(REAL_PROMPTS[job], schema, system=prompts.get_system(job))
    assert validate_instance(data, schema) == [], data
    assert json.loads(json.dumps(data)) == data  # plain JSON, no stray objects


def test_heuristic_highlight_output_is_usable(heuristic: HeuristicProvider) -> None:
    data = heuristic.complete_json(REAL_PROMPTS["highlight"], prompts.HIGHLIGHT_SCHEMA)
    clips = [ClipCandidate(**clip) for clip in data["clips"]]
    assert clips
    for clip in clips:
        assert clip.duration > 0
        assert clip.title and clip.hook and clip.reason
        assert 0.0 <= clip.score <= 1.0
    assert [c.score for c in clips] == sorted((c.score for c in clips), reverse=True)


def test_heuristic_script_output_is_usable(heuristic: HeuristicProvider) -> None:
    data = heuristic.complete_json(REAL_PROMPTS["script"], prompts.SCRIPT_SCHEMA)
    script = VideoScript(
        title=data["title"],
        hook=data["hook"],
        beats=[ScriptBeat(**beat) for beat in data["beats"]],
        cta=data["cta"],
        hashtags=data["hashtags"],
    )
    assert len(script.beats) >= 2
    assert script.narration.count(" ") > 10
    assert all(tag.startswith("#") and tag[1:].isalnum() for tag in script.hashtags)
    assert "sourdough" in script.narration.lower() or "sourdough" in script.title.lower()


def test_heuristic_chat_output_is_usable(heuristic: HeuristicProvider) -> None:
    data = heuristic.complete_json(REAL_PROMPTS["chat"], prompts.CHAT_SCHEMA)
    chat = ChatScript(
        title=data["title"],
        contact=data["contact"],
        messages=[ChatMessage(**m) for m in data["messages"]],
    )
    assert len(chat.messages) >= 4
    assert {m.outgoing for m in chat.messages} == {True, False}
    assert len({m.text for m in chat.messages}) == len(chat.messages)  # no copy-paste bubbles
    assert chat.contact == "Dana"
    assert all(m.delay >= 0 for m in chat.messages)


def test_heuristic_forum_output_is_usable(heuristic: HeuristicProvider) -> None:
    data = heuristic.complete_json(REAL_PROMPTS["forum"], prompts.FORUM_SCHEMA)
    post = RedditPost(**data)
    # Plain names only -- the offline provider must not emit another service's
    # handle grammar, because the story card is our own design (hard rule 6).
    assert len(post.community) > 3 and not re.match(r"^/?[ru]/", post.community)
    assert len(post.author) > 3 and not re.match(r"^/?[ru]/", post.author)
    assert post.upvotes > 0 and post.comments > 0
    assert len(post.narration.split()) > 10


def test_heuristic_content_comes_from_the_prompt(heuristic: HeuristicProvider) -> None:
    prompt = prompts.forum_prompt("my landlord installed a jacuzzi on the roof without telling anyone")
    data = heuristic.complete_json(prompt, prompts.FORUM_SCHEMA)
    blob = json.dumps(data).lower()
    assert "jacuzzi" in blob or "landlord" in blob


def test_heuristic_is_deterministic_and_seed_sensitive() -> None:
    prompt = prompts.chat_prompt("one short premise line about a missing parcel")
    first = HeuristicProvider(settings=replace(get_settings(), seed=1234))
    same = HeuristicProvider(settings=replace(get_settings(), seed=1234))
    other = HeuristicProvider(settings=replace(get_settings(), seed=99))
    a = first.complete_json(prompt, prompts.CHAT_SCHEMA)
    b = same.complete_json(prompt, prompts.CHAT_SCHEMA)
    c = other.complete_json(prompt, prompts.CHAT_SCHEMA)
    assert a == b
    assert a != c
    assert validate_instance(c, prompts.CHAT_SCHEMA) == []


# --------------------------------------------------------------------------- #
# heuristic provider: adversarial schemas
# --------------------------------------------------------------------------- #

def _obj(props: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(props),
        "properties": props,
        **extra,
    }


DEEP = _obj({
    "level1": _obj({
        "level2": _obj({
            "level3": {
                "type": "array",
                "minItems": 2,
                "items": _obj({
                    "level4": _obj({
                        "leaf": {"type": "string", "minLength": 20, "maxLength": 60},
                        "flag": {"type": "boolean"},
                    }),
                }),
            }
        }),
    }),
})

ENUMS = _obj({
    "mood": {"enum": ["calm", "tense", "funny"]},
    "count": {"type": "integer", "enum": [3, 5, 8]},
    "fixed": {"const": "always-this"},
    "mixed": {"type": ["string", "null"]},
})

ARRAYS = _obj({
    "rows": {
        "type": "array",
        "minItems": 4,
        "maxItems": 4,
        "items": _obj({
            "id": {"type": "integer", "minimum": 10, "maximum": 99},
            "label": {"type": "string", "minLength": 1},
            "weights": {"type": "array", "minItems": 2,
                        "items": {"type": "number", "minimum": 0, "maximum": 1}},
        }),
    },
    "tuple": {"type": "array", "items": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}]},
    "unique": {"type": "array", "minItems": 3, "uniqueItems": True, "items": {"type": "string"}},
})

NUMBERS = _obj({
    "stepped": {"type": "integer", "minimum": 6, "maximum": 30, "multipleOf": 6},
    "exclusive": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1},
    "negative": {"type": "integer", "minimum": -50, "maximum": -10},
    "patterned": {"type": "string", "pattern": "^[a-z0-9-]+$", "minLength": 3},
})

COMBINATORS = {
    "$defs": {
        "named": _obj({"name": {"type": "string", "minLength": 2}}),
        "node": _obj({
            "name": {"type": "string", "minLength": 1},
            "children": {"type": "array", "items": {"$ref": "#/$defs/node"}, "maxItems": 2},
        }),
    },
    "type": "object",
    "additionalProperties": False,
    "required": ["ref", "either", "merged", "tree"],
    "properties": {
        "ref": {"$ref": "#/$defs/named"},
        "either": {"oneOf": [{"type": "string", "minLength": 4}, {"type": "integer"}]},
        # allOf branches are deliberately open: two closed objects combined with
        # allOf is unsatisfiable, and no generator can rescue that.
        "merged": {
            "allOf": [
                {"type": "object", "required": ["a"], "properties": {"a": {"type": "string"}}},
                {"type": "object", "required": ["b"], "properties": {"b": {"type": "integer"}}},
            ]
        },
        "tree": {"$ref": "#/$defs/node"},
    },
}

ADVERSARIAL = {
    "deep": DEEP,
    "enums": ENUMS,
    "arrays": ARRAYS,
    "numbers": NUMBERS,
    "combinators": COMBINATORS,
}


@pytest.mark.parametrize("name", sorted(ADVERSARIAL))
def test_heuristic_satisfies_adversarial_schemas(heuristic: HeuristicProvider, name: str) -> None:
    schema = ADVERSARIAL[name]
    data = heuristic.complete_json("A short premise about a delivery van and a locked gate.", schema)
    assert validate_instance(data, schema) == [], json.dumps(data, indent=1)


def test_adversarial_schema_details(heuristic: HeuristicProvider) -> None:
    data = heuristic.complete_json("delivery van, locked gate, 14 parcels", ARRAYS)
    assert len(data["rows"]) == 4
    assert all(len(row["weights"]) >= 2 for row in data["rows"])
    assert [type(v) for v in data["tuple"]] == [str, int, bool]
    assert len(set(data["unique"])) == 3

    enums = heuristic.complete_json("a tense little story", ENUMS)
    assert enums["mood"] in ("calm", "tense", "funny")
    assert enums["count"] in (3, 5, 8)
    assert enums["fixed"] == "always-this"

    deep = heuristic.complete_json("a tense little story", DEEP)
    leaves = deep["level1"]["level2"]["level3"]
    assert len(leaves) >= 2
    assert all(20 <= len(item["level4"]["leaf"]) <= 60 for item in leaves)


def test_heuristic_enum_prefers_a_word_from_the_prompt(heuristic: HeuristicProvider) -> None:
    schema = _obj({"mood": {"enum": ["calm", "tense", "funny"]}})
    data = heuristic.complete_json("Premise:\nthis one should be funny, funny and light", schema)
    assert data["mood"] == "funny"


# --------------------------------------------------------------------------- #
# heuristic provider: robustness
# --------------------------------------------------------------------------- #

WEIRD_PROMPTS = [
    "",
    "   \n\t  ",
    "?!?!...",
    "\x00\x01\x02 control bytes \x7f",
    "emoji only \U0001f680\U0001f9ea\U0001f4bb",
    "a" * 200_000,
    "word " * 50_000,
    "Topic:",
    "{'not': 'a prompt'}",
    "日本語のプロンプトです。これは短いテストです。",
]


@pytest.mark.parametrize("prompt", WEIRD_PROMPTS, ids=range(len(WEIRD_PROMPTS)))
def test_heuristic_never_raises_on_weird_prompts(heuristic: HeuristicProvider, prompt: str) -> None:
    for job in prompts.JOBS:
        schema = prompts.get_schema(job)
        data = heuristic.complete_json(prompt, schema)
        assert validate_instance(data, schema) == [], (job, data)
    assert isinstance(heuristic.complete(prompt), str)


WEIRD_SCHEMAS = [
    {},
    True,
    False,
    None,
    "not a schema",
    {"type": "object"},
    {"type": "array"},
    {"type": "unheard-of"},
    {"type": []},
    {"properties": {"a": {"type": "string"}}},
    {"$ref": "#/$defs/missing"},
    {"$ref": "http://example.invalid/schema.json"},
    {"type": "object", "additionalProperties": False, "required": ["ghost"], "properties": {}},
    {"type": "string", "pattern": "([unclosed"},
    {"type": "integer", "minimum": 5, "maximum": 1},
]


@pytest.mark.parametrize("schema", WEIRD_SCHEMAS, ids=range(len(WEIRD_SCHEMAS)))
def test_heuristic_never_raises_on_weird_schemas(heuristic: HeuristicProvider, schema: Any) -> None:
    data = heuristic.complete_json("a premise about a lost dog", schema)
    assert isinstance(data, dict)
    json.dumps(data)


def test_build_instance_supports_non_object_roots() -> None:
    assert isinstance(build_instance({"type": "string", "minLength": 4}, "a lost dog"), str)
    value = build_instance({"type": "array", "minItems": 2, "items": {"type": "integer"}}, "7 dogs")
    assert isinstance(value, list) and len(value) >= 2
    # complete_json must still hand back a dict for such a schema.
    wrapped = HeuristicProvider().complete_json("x", {"type": "string"})
    assert isinstance(wrapped["result"], str)


def test_heuristic_complete_is_extractive_and_budgeted(heuristic: HeuristicProvider) -> None:
    prompt = (
        "Topic:\n"
        "The bakery kept the starter alive for nineteen years. "
        "A cold proof slows fermentation and builds flavour. "
        "Nobody on the team could explain why the loaves rose higher in winter."
    )
    out = heuristic.complete(prompt)
    assert out
    assert any(fragment in out for fragment in ("bakery", "cold proof", "loaves"))
    assert len(heuristic.complete(prompt, max_tokens=20)) <= 80


def test_heuristic_is_always_available() -> None:
    assert HeuristicProvider().available() is True
    assert HeuristicProvider(settings=replace(get_settings(), offline=True)).available() is True


# --------------------------------------------------------------------------- #
# provider routing
# --------------------------------------------------------------------------- #

def test_auto_falls_back_to_heuristic_without_a_key(clean_env: None) -> None:
    assert credentials_available() is False
    assert isinstance(get_provider("auto"), HeuristicProvider)
    assert isinstance(get_provider(None), HeuristicProvider)


def test_auto_picks_claude_when_a_key_resolves(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    reset_settings()
    provider = get_provider("auto")
    assert isinstance(provider, ClaudeProvider)
    assert provider.available() is True


def test_auto_picks_claude_from_an_ant_profile(clean_env: None, tmp_path) -> None:
    config_dir = tmp_path / "anthropic"
    (config_dir / "configs").mkdir(parents=True)
    (config_dir / "configs" / "default.json").write_text("{}", encoding="utf-8")
    assert credentials_available() is True
    assert isinstance(get_provider("auto"), ClaudeProvider)


def test_offline_forces_the_heuristic_provider(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    monkeypatch.setenv("AICLIP_OFFLINE", "1")
    reset_settings()
    assert get_settings().offline is True
    assert isinstance(get_provider("auto"), HeuristicProvider)
    assert ClaudeProvider().available() is False


def test_settings_llm_provider_drives_the_default(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AICLIP_LLM", "claude")
    reset_settings()
    assert isinstance(get_provider(), ClaudeProvider)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("claude", ClaudeProvider),
        ("anthropic", ClaudeProvider),
        ("Claude", ClaudeProvider),
        ("heuristic", HeuristicProvider),
        ("offline", HeuristicProvider),
        ("  OFFLINE  ", HeuristicProvider),
    ],
)
def test_named_providers(clean_env: None, name: str, expected: type) -> None:
    assert isinstance(get_provider(name), expected)


def test_explicit_claude_is_returned_even_without_credentials(clean_env: None) -> None:
    provider = get_provider("claude")
    assert isinstance(provider, ClaudeProvider)
    assert provider.available() is False


@pytest.mark.parametrize("name", ["gpt", "llama", "claude-3", "heuristics", "!"])
def test_unknown_provider_raises(name: str) -> None:
    with pytest.raises(LLMError, match="unknown llm provider"):
        get_provider(name)


def test_get_provider_honours_an_explicit_settings_object(clean_env: None) -> None:
    offline = replace(get_settings(), offline=True)
    assert isinstance(get_provider("auto", settings=offline), HeuristicProvider)


# --------------------------------------------------------------------------- #
# Claude provider: availability, without a single packet
# --------------------------------------------------------------------------- #

def test_available_is_false_without_credentials(clean_env: None) -> None:
    provider = ClaudeProvider(settings=online_settings())
    assert provider.available() is False  # the _no_network fixture proves no call was made


def test_available_is_false_when_the_sdk_is_missing(clean_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-not-a-real-key")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # makes ``import anthropic`` fail
    reset_settings()
    assert ClaudeProvider(settings=online_settings()).available() is False
    with pytest.raises(MissingDependency, match="pip install"):
        ClaudeProvider(settings=online_settings()).client()


def test_available_is_true_with_an_injected_client(clean_env: None) -> None:
    provider = ClaudeProvider(settings=online_settings(), client=FakeClient())
    assert provider.available() is True


def test_credentials_available_reads_the_federation_trio(
    clean_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    assert credentials_available() is False
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "rule_1")
    assert credentials_available() is False
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_1")
    assert credentials_available() is False
    monkeypatch.setenv("ANTHROPIC_IDENTITY_TOKEN", "tok")
    assert credentials_available() is True


# --------------------------------------------------------------------------- #
# Claude provider: request shape
# --------------------------------------------------------------------------- #

def test_json_request_kwargs_are_exactly_right(clean_env: None) -> None:
    client = FakeClient(_Response('{"community": "r/x"}'))
    settings = online_settings(llm_model="claude-opus-5", llm_effort="medium", llm_max_tokens=16000)
    provider = ClaudeProvider(settings=settings, client=client)

    data = provider.complete_json("premise", prompts.FORUM_SCHEMA, system=prompts.FORUM_SYSTEM)
    assert data == {"community": "r/x"}

    (kwargs,) = client.messages.calls
    assert kwargs == {
        "model": "claude-opus-5",
        "max_tokens": 16000,
        "system": prompts.FORUM_SYSTEM,
        "messages": [{"role": "user", "content": "premise"}],
        "output_config": {
            "effort": "medium",
            "format": {"type": "json_schema", "schema": prompts.FORUM_SCHEMA},
        },
    }


def test_forbidden_parameters_are_never_sent(clean_env: None) -> None:
    client = FakeClient(_Response("{}"), _Response("plain"))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    provider.complete_json("p", prompts.CHAT_SCHEMA, system="s")
    provider.complete("p", system="s")
    for kwargs in client.messages.calls:
        assert "thinking" not in kwargs
        assert "budget_tokens" not in kwargs
        assert "budget_tokens" not in json.dumps(kwargs["output_config"])
        assert "tool_choice" not in kwargs
        assert "tools" not in kwargs
        assert [m["role"] for m in kwargs["messages"]] == ["user"]  # no assistant prefill


def test_plain_completion_keeps_effort_and_drops_format(clean_env: None) -> None:
    client = FakeClient(_Response("  a plain answer  "))
    provider = ClaudeProvider(settings=online_settings(llm_effort="high"), client=client)
    assert provider.complete("hello") == "  a plain answer  "
    (kwargs,) = client.messages.calls
    assert kwargs["output_config"] == {"effort": "high"}
    assert "system" not in kwargs  # an empty system prompt is omitted, not sent blank


def test_max_tokens_override(clean_env: None) -> None:
    client = FakeClient(_Response("x"))
    provider = ClaudeProvider(settings=online_settings(llm_max_tokens=16000), client=client)
    provider.complete("hello", max_tokens=512)
    assert client.messages.calls[0]["max_tokens"] == 512


def test_model_id_comes_from_settings(clean_env: None) -> None:
    client = FakeClient(_Response("x"))
    ClaudeProvider(settings=online_settings(llm_model="claude-haiku-4-5"), client=client).complete("hi")
    assert client.messages.calls[0]["model"] == "claude-haiku-4-5"


def test_build_request_is_pure(clean_env: None) -> None:
    provider = ClaudeProvider(settings=online_settings(), client=FakeClient())
    request = provider.build_request("p", system="s", schema=prompts.SCRIPT_SCHEMA)
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert provider._client.messages.calls == []


# --------------------------------------------------------------------------- #
# Claude provider: responses and failure handling
# --------------------------------------------------------------------------- #

def test_refusal_raises_before_content_is_read(clean_env: None) -> None:
    class _Exploding:
        type = "text"

        @property
        def text(self) -> str:  # pragma: no cover - must never be reached
            raise AssertionError("content was read despite a refusal")

    client = FakeClient(_Response(stop_reason="refusal", blocks=[_Exploding()]))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    with pytest.raises(LLMError, match="refusal"):
        provider.complete("hello")


def test_missing_text_block_raises(clean_env: None) -> None:
    client = FakeClient(_Response(blocks=[_Block("{}", kind="thinking")]))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    with pytest.raises(LLMError, match="no text block"):
        provider.complete("hello")


def test_five_hundred_triggers_exactly_one_retry(clean_env: None) -> None:
    client = FakeClient(status_error(500), status_error(500))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    with pytest.raises(LLMError, match="after 2 attempts"):
        provider.complete("hello")
    assert len(client.messages.calls) == 2


def test_retry_succeeds_on_the_second_attempt(clean_env: None) -> None:
    client = FakeClient(status_error(429), _Response("second time lucky"))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    assert provider.complete("hello") == "second time lucky"
    assert len(client.messages.calls) == 2


def test_typed_server_error_is_retried(clean_env: None) -> None:
    client = FakeClient(status_error(503, "InternalServerError"), _Response("recovered"))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    assert provider.complete("hello") == "recovered"
    assert len(client.messages.calls) == 2


def test_connection_error_is_retried_once(clean_env: None) -> None:
    anthropic = pytest.importorskip("anthropic")
    failure = anthropic.APIConnectionError(request=None)
    client = FakeClient(failure, failure)
    provider = ClaudeProvider(settings=online_settings(), client=client)
    with pytest.raises(LLMError):
        provider.complete("hello")
    assert len(client.messages.calls) == 2


def test_client_errors_are_not_retried(clean_env: None) -> None:
    client = FakeClient(status_error(400), _Response("never reached"))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    with pytest.raises(LLMError, match="status 400"):
        provider.complete("hello")
    assert len(client.messages.calls) == 1


def test_offline_settings_block_the_provider(clean_env: None) -> None:
    provider = ClaudeProvider(settings=replace(get_settings(), offline=True), client=FakeClient())
    with pytest.raises(LLMError, match="offline"):
        provider.complete("hello")


def test_json_parsing_tolerates_fences_and_preamble(clean_env: None) -> None:
    client = FakeClient(_Response('Sure, here it is:\n```json\n{"title": "ok"}\n```\n'))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    assert provider.complete_json("p", prompts.SCRIPT_SCHEMA) == {"title": "ok"}


def test_non_json_reply_raises(clean_env: None) -> None:
    client = FakeClient(_Response("I would rather write you a poem."))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    with pytest.raises(LLMError, match="not JSON"):
        provider.complete_json("p", prompts.SCRIPT_SCHEMA)


def test_parse_json_wraps_bare_arrays() -> None:
    assert _parse_json('[{"a": 1}]') == {"result": [{"a": 1}]}
    assert _parse_json('{"a": 1}') == {"a": 1}


# --------------------------------------------------------------------------- #
# import hygiene
# --------------------------------------------------------------------------- #

def test_importing_the_package_does_not_import_anthropic() -> None:
    code = "import sys, aiclipper.llm; print('anthropic' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


def test_both_providers_satisfy_the_protocol() -> None:
    from aiclipper.llm.base import LLMProvider

    assert isinstance(HeuristicProvider(), LLMProvider)
    assert isinstance(ClaudeProvider(), LLMProvider)
    assert not isinstance(object(), LLMProvider)
    assert HeuristicProvider().name == "heuristic"
    assert ClaudeProvider().name == "claude"


# --------------------------------------------------------------------------- #
# schema walking: the constraints a naive walker gets wrong
# --------------------------------------------------------------------------- #

def test_unique_items_keeps_the_declared_type(heuristic: HeuristicProvider) -> None:
    """A boolean array holds two booleans, not ``True, False, 'False 3'``.

    ``minItems: 4`` with boolean items is unsatisfiable -- there are only two
    booleans -- and the right answer is a short list of the correct type, never
    a padded one of the wrong type.
    """
    schema = _obj({"flags": {"type": "array", "minItems": 4, "uniqueItems": True,
                             "items": {"type": "boolean"}}})
    data = heuristic.complete_json("a premise about a locked gate", schema)
    assert all(isinstance(v, bool) for v in data["flags"])
    assert len(set(data["flags"])) == len(data["flags"])


def test_unique_items_of_objects_are_actually_distinct(heuristic: HeuristicProvider) -> None:
    schema = _obj({"rows": {
        "type": "array", "minItems": 4, "uniqueItems": True,
        "items": _obj({"label": {"type": "string", "minLength": 1},
                       "weight": {"type": "number", "minimum": 0, "maximum": 100}}),
    }})
    data = heuristic.complete_json("Topic:\nfourteen parcels, one locked gate, one van.", schema)
    assert validate_instance(data, schema) == [], data
    assert len(data["rows"]) == 4
    assert len({json.dumps(row, sort_keys=True) for row in data["rows"]}) == 4


def test_unique_items_survive_a_constrained_numeric_range(heuristic: HeuristicProvider) -> None:
    """Every re-roll clamps to the same number; the generator must still spread them."""
    schema = _obj({"ids": {"type": "array", "minItems": 5, "uniqueItems": True,
                           "items": {"type": "integer", "minimum": -20, "maximum": 1}}})
    data = heuristic.complete_json("Topic:\n14 parcels", schema)
    assert validate_instance(data, schema) == [], data


def test_nested_unique_lists_are_not_broken_from_the_outside(heuristic: HeuristicProvider) -> None:
    inner = {"type": "array", "minItems": 3, "uniqueItems": True,
             "items": _obj({"n": {"type": "number", "minimum": -6, "maximum": 56}})}
    schema = _obj({"grid": {"type": "array", "minItems": 2, "uniqueItems": True, "items": inner}})
    data = heuristic.complete_json("Topic:\n14 parcels and a gate", schema)
    assert validate_instance(data, schema) == [], data


def test_tuple_items_are_padded_up_to_min_items(heuristic: HeuristicProvider) -> None:
    schema = _obj({"row": {"type": "array", "minItems": 5,
                           "items": [{"type": "string"}, {"type": "integer"}]}})
    data = heuristic.complete_json("Topic:\na van, a gate, 14 parcels", schema)
    assert validate_instance(data, schema) == [], data
    assert len(data["row"]) == 5
    assert isinstance(data["row"][0], str) and isinstance(data["row"][1], int)


def test_multiple_of_lands_inside_exclusive_bounds(heuristic: HeuristicProvider) -> None:
    schema = _obj({
        "half": {"type": "number", "exclusiveMinimum": 0, "exclusiveMaximum": 1, "multipleOf": 0.5},
        "sixes": {"type": "integer", "minimum": 6, "maximum": 30, "multipleOf": 6},
    })
    data = heuristic.complete_json("Topic:\nnumbers", schema)
    assert validate_instance(data, schema) == [], data
    assert data["half"] == 0.5


def test_unsatisfiable_multiple_of_stays_inside_the_range(heuristic: HeuristicProvider) -> None:
    """No multiple of 5 lives in [-4, -2]: keep the range, not a value outside it."""
    schema = _obj({"n": {"type": "number", "minimum": -4, "maximum": -2, "multipleOf": 5}})
    data = heuristic.complete_json("Topic:\nnumbers", schema)
    assert -4 <= data["n"] <= -2


PATTERNS = [
    r"^#[a-z0-9]+$",
    r"^ch-[a-z]{3,20}$",
    r"^\d{4}-\d{2}-\d{2}$",
    r"^[A-Z]{2}-\d{3,5}$",
    r"^(alpha|beta|gamma)$",
    r"^v\d+\.\d+(\.\d+)?$",
    r"^\w+@\w+\.(com|org)$",
    r"^#?[0-9a-fA-F]{6}$",
    r"^[^0-9]{5,}$",
    r"^\[\d+\]$",
]


@pytest.mark.parametrize("pattern", PATTERNS)
def test_patterned_strings_are_synthesised_when_the_prompt_cannot_supply_them(
    heuristic: HeuristicProvider, pattern: str
) -> None:
    """No amount of reshaping turns prompt text into ``#abc``; the regex is sampled."""
    schema = _obj({"v": {"type": "string", "pattern": pattern}})
    data = heuristic.complete_json("Topic:\nA van, a gate and 14 parcels.", schema)
    assert validate_instance(data, schema) == [], data


def test_patterned_strings_respect_max_length(heuristic: HeuristicProvider) -> None:
    schema = _obj({"v": {"type": "string", "pattern": "^[a-z]+$", "minLength": 3, "maxLength": 5}})
    data = heuristic.complete_json("Topic:\nA van, a gate and fourteen parcels.", schema)
    assert validate_instance(data, schema) == [], data


def test_unique_patterned_strings_stay_matching(heuristic: HeuristicProvider) -> None:
    schema = _obj({"tags": {"type": "array", "minItems": 5, "uniqueItems": True,
                            "items": {"type": "string", "pattern": "^#[a-z0-9]+$"}}})
    data = heuristic.complete_json("Topic:\nA van and a gate.", schema)
    assert validate_instance(data, schema) == [], data
    assert len(set(data["tags"])) == 5


def test_exhausting_the_node_budget_still_yields_a_valid_instance(heuristic: HeuristicProvider) -> None:
    """Past the budget the walker emits placeholders -- they still carry required keys."""
    schema = _obj({"rows": {
        "type": "array", "minItems": 2000,
        "items": _obj({"title": {"type": "string", "minLength": 5},
                       "count": {"type": "integer", "minimum": 3}}),
    }})
    data = heuristic.complete_json("Topic:\nA van and a gate.", schema)
    assert validate_instance(data, schema) == [], data[0] if data else data
    assert len(data["rows"]) == 2000


def test_bounded_time_windows_do_not_collapse_onto_the_ceiling(heuristic: HeuristicProvider) -> None:
    """With a ``maximum`` the windows wrap instead of pinning to ``(max, max)``."""
    span = {"type": "number", "minimum": 0, "maximum": 60}
    schema = _obj({"clips": {"type": "array", "minItems": 6,
                             "items": _obj({"start": span, "end": span})}})
    data = heuristic.complete_json("Topic:\na recording", schema)
    assert validate_instance(data, schema) == [], data
    assert all(clip["end"] > clip["start"] for clip in data["clips"]), data["clips"]


def test_unbounded_highlight_windows_still_march_forward(heuristic: HeuristicProvider) -> None:
    data = heuristic.complete_json(REAL_PROMPTS["highlight"], prompts.HIGHLIGHT_SCHEMA)
    clips = data["clips"]
    assert [c["start"] for c in clips] == sorted(c["start"] for c in clips)
    assert all(c["end"] > c["start"] for c in clips)
    assert all(a["end"] <= b["start"] for a, b in zip(clips, clips[1:], strict=False))  # non-overlapping


def test_degenerate_nodes_keep_their_required_keys() -> None:
    from aiclipper.llm import heuristic as module

    node = _obj({"title": {"type": "string", "minLength": 4}, "count": {"type": "integer", "minimum": 3}})
    ctx = module._Ctx(root=node, mat=module._material("", 0))
    value = module._degenerate(node, ctx)
    assert set(value) == {"count", "title"}
    assert validate_instance(value, node) == []


def test_complete_json_returns_a_dict_even_if_the_walker_explodes(
    monkeypatch: pytest.MonkeyPatch, heuristic: HeuristicProvider
) -> None:
    """The 'never raises' contract is load-bearing: every pipeline leans on it."""
    from aiclipper.llm import heuristic as module

    def _boom(*args: Any, **kwargs: Any):
        raise RuntimeError("walker failed")

    monkeypatch.setattr(module, "_build", _boom)
    data = heuristic.complete_json("a premise", prompts.SCRIPT_SCHEMA)
    assert isinstance(data, dict)


# --------------------------------------------------------------------------- #
# prompt material
# --------------------------------------------------------------------------- #

def test_payload_split_keeps_material_above_an_inner_colon(heuristic: HeuristicProvider) -> None:
    """A transcript line ending in ':' looks like a label -- it must not eat the payload."""
    transcript = (
        "[0.0-4.2] we tried three launch plans and only one of them actually worked.\n"
        "[4.2-9.0] so here is my point:\n"
        "[9.0-15.5] revenue moved from 12 thousand to 240 thousand in 90 days.\n"
    )
    data = heuristic.complete_json(prompts.highlight_prompt(transcript, count=3),
                                   prompts.HIGHLIGHT_SCHEMA)
    blob = json.dumps(data).lower()
    assert "launch plans" in blob, blob


def test_content_is_drawn_from_the_payload_not_the_instructions(heuristic: HeuristicProvider) -> None:
    data = heuristic.complete_json(prompts.script_prompt("the great emu war of 1932"),
                                   prompts.SCRIPT_SCHEMA)
    blob = json.dumps(data).lower()
    assert "emu" in blob
    assert "spoken narration must be about" not in blob  # the instruction line itself


#: Wording that only ever comes from our own prompt scaffolding.
INSTRUCTION_VOCABULARY = (
    "vertical short", "spoken narration", "call to action", "punchy tone",
    "text-message story", "forum story post", "words in the body",
    "premise:", "topic:", "write a", "-second",
)

#: Topics carrying too little text to out-rank the boilerplate around them.
THIN_SUBJECTS = ["cats", "tiny things", "\U0001f431", "ok", "a" * 4_000]


@pytest.mark.parametrize("subject", THIN_SUBJECTS, ids=range(len(THIN_SUBJECTS)))
def test_a_tagged_subject_keeps_the_instructions_out_of_the_content(
    heuristic: HeuristicProvider, subject: str
) -> None:
    """A thin topic used to lose the sentence ranking to the prompt's own boilerplate."""
    for job, prompt in (
        ("script", prompts.script_prompt(subject, seconds=8)),
        ("chat", prompts.chat_prompt(subject, turns=6)),
        ("forum", prompts.forum_prompt(subject, words=60)),
    ):
        schema = prompts.get_schema(job)
        data = heuristic.complete_json(with_subject(prompt, subject), schema)
        assert validate_instance(data, schema) == [], (job, data)
        blob = json.dumps(data).lower()
        leaked = [phrase for phrase in INSTRUCTION_VOCABULARY if phrase in blob]
        assert not leaked, (job, leaked, blob)


def test_a_tagged_subject_still_drives_the_content(heuristic: HeuristicProvider) -> None:
    prompt = with_subject(prompts.script_prompt("cats", seconds=8), "cats")
    data = heuristic.complete_json(prompt, prompts.SCRIPT_SCHEMA)
    assert "cats" in data["title"].lower()
    assert "cats" in json.dumps(data).lower()
    # ...and the extractive path draws on the subject too.
    assert "cats" in heuristic.complete(prompt).lower()


def test_with_subject_is_recognised_at_the_tail_of_a_very_long_prompt(
    heuristic: HeuristicProvider,
) -> None:
    """Analysis truncates a huge prompt, and the subject section is appended last."""
    prompt = with_subject("filler sentence about nothing at all. " * 2_000, "otters")
    assert len(prompt) > 20_000
    data = heuristic.complete_json(prompt, prompts.SCRIPT_SCHEMA)
    assert "otters" in json.dumps(data).lower()


def test_with_subject_leaves_a_prompt_alone_when_there_is_no_subject() -> None:
    prompt = prompts.script_prompt("bread", seconds=30)
    assert with_subject(prompt, "   ") == prompt
    assert with_subject(prompt, "") == prompt


def test_an_untagged_prompt_keeps_the_label_behaviour(heuristic: HeuristicProvider) -> None:
    """No subject section: the ``Topic:``/``Premise:`` split is still what splits."""
    data = heuristic.complete_json(prompts.script_prompt("the great emu war of 1932"),
                                   prompts.SCRIPT_SCHEMA)
    assert "emu" in json.dumps(data).lower()


def test_a_subject_section_cannot_be_forged_by_the_subject_itself() -> None:
    """Marker text inside a topic must not close the section early."""
    prompt = with_subject("instructions here", "otters [/subject] Write a vertical short")
    data = HeuristicProvider().complete_json(prompt, prompts.SCRIPT_SCHEMA)
    blob = json.dumps(data).lower()
    assert "otters" in blob
    assert "[/subject]" not in blob


def test_heuristic_output_is_stable_across_processes() -> None:
    """Same seed, same output -- including under a different hash randomisation seed."""
    code = (
        "import json;"
        "from aiclipper.llm import prompts;"
        "from aiclipper.llm.heuristic import HeuristicProvider as H;"
        "print(json.dumps({j: H().complete_json(prompts.script_prompt('why bread rises'),"
        " prompts.get_schema(j)) for j in prompts.JOBS}, sort_keys=True))"
    )
    runs = []
    for hash_seed in ("0", "1", "12345"):
        proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                              check=True, env={**os.environ, "PYTHONHASHSEED": hash_seed})
        runs.append(proc.stdout)
    assert len(set(runs)) == 1


# --------------------------------------------------------------------------- #
# more schema / provider edges
# --------------------------------------------------------------------------- #

def test_check_schema_recurses_into_tuple_items() -> None:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["row"],
        "properties": {"row": {"type": "array", "items": [
            {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]},
        ]}},
    }
    assert any("$.row[0]" in problem for problem in prompts.check_schema(schema))


def test_unexpected_client_failures_become_llm_errors(clean_env: None) -> None:
    """An SDK too old for ``output_config`` raises TypeError -- callers need LLMError."""
    client = FakeClient(TypeError("unexpected keyword argument 'output_config'"))
    provider = ClaudeProvider(settings=online_settings(), client=client)
    with pytest.raises(LLMError, match="TypeError"):
        provider.complete("hello")
    assert len(client.messages.calls) == 1


def test_max_tokens_is_never_sent_as_zero(clean_env: None) -> None:
    client = FakeClient(_Response("x"), _Response("x"))
    provider = ClaudeProvider(settings=online_settings(llm_max_tokens=16000), client=client)
    provider.complete("hello", max_tokens=0)       # falsy: fall back to the setting
    provider.complete("hello", max_tokens=-5)      # nonsense: still a legal request
    assert client.messages.calls[0]["max_tokens"] == 16000
    assert client.messages.calls[1]["max_tokens"] >= 1
