"""The offline, rule-based language-model provider.

This backend never touches the network, never imports a third-party package and
-- by contract -- never raises.  It is what makes every workflow in the engine
runnable on a laptop with no API key and in CI with no egress, so the rest of
the codebase can call :meth:`HeuristicProvider.complete_json` and simply assume
it gets a usable object back.

Two jobs:

``complete``
    Extractive.  The prompt is split into sentences, the sentences are ranked by
    how much of the prompt's own vocabulary they carry, and the best few are
    returned in their original order.

``complete_json``
    A JSON Schema walker.  It handles ``$ref``, ``allOf``/``anyOf``/``oneOf``,
    ``enum``/``const``, objects (``properties``, ``required``,
    ``additionalProperties``), arrays (``items``, ``minItems``/``maxItems``),
    strings (``minLength``/``maxLength``/``pattern``/``format``) and numbers
    (``minimum``/``maximum``/``multipleOf``), and emits an instance that
    validates -- for *any* schema, not just the four in
    :mod:`aiclipper.llm.prompts`.  The values are derived from the prompt: its
    sentences, its numbers and its keywords, routed by property name, so the
    result reads like an answer to the question rather than filler.

A prompt is mostly *instructions*, and instruction wording must never surface as
content -- "Write a 30-second vertical short" is not a title.  Callers therefore
tag the thing the prompt is actually about with :func:`with_subject`, which
appends a delimited ``[subject]`` section; when one is present every generated
value is drawn from it alone.  Without it the older label heuristic
(``Topic:``/``Premise:``) still applies, so existing callers are unaffected.

:func:`validate_instance` is the small checker used to prove that; it is shared
with the test-suite rather than duplicated there.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from typing import Any

from ..config import Settings
from .base import _settings as _resolve_settings

__all__ = [
    "HeuristicProvider", "build_instance", "validate_instance",
    "with_subject", "SUBJECT_OPEN", "SUBJECT_CLOSE",
]

#: Analysis is capped so a pathological prompt cannot make generation slow.
_MAX_PROMPT_CHARS = 20_000
_MAX_SENTENCES = 120
_MAX_KEYWORDS = 80

#: Guard rails for schema walking (cyclic ``$ref``, deeply nested arrays).
_MAX_DEPTH = 12
_NODE_BUDGET = 4_000

#: Guard rails for the ``pattern`` sampler.
_MAX_PATTERN_CHARS = 400
_MAX_PATTERN_DEPTH = 10

_STOPWORDS = frozenset(
    """
    a about after all also am an and any are as at be because been before being but by can cannot
    could did do does doing done down each else even ever every for from get got had has have he
    her here hers him his how i if in into is it its just like make may me might more most must my
    no nor not now of off on once one only or other our out over own said same she should so some
    such than that the their them then there these they this those through to too under until up
    us use used very was we were what when where which while who why will with would you your
    write written writes about topic premise output json schema field fields must should return
    """.split()
)

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_SLUG_RE = re.compile(r"[^a-z0-9]+")

_FALLBACK_SENTENCES = (
    "The moment everything changed was smaller than anyone expected.",
    "Nobody wrote it down, so it kept happening for another three years.",
    "It worked, and the reason it worked is the interesting part.",
    "By the end of the week the numbers told a completely different story.",
)
_FALLBACK_KEYWORDS = ("story", "moment", "change", "detail", "reason", "result")

#: Sentence shells used to top up a prompt that is too short to fill a schema.
_FILLER_TEMPLATES = (
    "It started with {a} and ended somewhere nobody had planned for.",
    "The detail that mattered was {b}, not {c}.",
    "Nobody expected {a} to change the outcome by that much.",
    "Here is what {c} actually looked like from the inside.",
    "Every version of this came back to {b} in the end.",
    "The part about {a} is the part people get wrong.",
)


# --------------------------------------------------------------------------- #
# prompt material
# --------------------------------------------------------------------------- #

@dataclass
class _Material:
    """Everything the generator knows about one prompt."""

    sentences: list[str]
    keywords: list[str]
    numbers: list[float]
    rng: random.Random
    person: str = "Sam"
    cursor: int = 0
    sentence_cursor: int = -1
    keyword_cursor: int = -1

    def next_index(self) -> int:
        self.cursor += 1
        return self.cursor

    def sentence(self, index: int) -> str:
        return self.sentences[index % len(self.sentences)]

    def keyword(self, index: int) -> str:
        return self.keywords[index % len(self.keywords)]

    def number(self, index: int) -> float | None:
        return self.numbers[index % len(self.numbers)] if self.numbers else None

    # Sentences and keywords are handed out from their own cursors rather than
    # from a shared one: a shared counter advances by however many fields an
    # object has, which lands on the same sentence every time that count and the
    # sentence count share a factor -- every chat message getting the same body.
    def take_sentence(self) -> str:
        self.sentence_cursor += 1
        return self.sentence(self.sentence_cursor)

    def take_sentences(self, count: int) -> list[str]:
        """``count`` sentences, skipping ones already handed out in this call."""
        out: list[str] = []
        for _ in range(min(count, len(self.sentences))):
            for _ in range(len(self.sentences)):
                candidate = self.take_sentence()
                if candidate not in out:
                    out.append(candidate)
                    break
        return out or [self.sentence(0)]

    def take_keyword(self) -> str:
        self.keyword_cursor += 1
        return self.keyword(self.keyword_cursor)

    def take_phrase(self, count: int = 3) -> str:
        return " ".join(self.take_keyword() for _ in range(count))


def _clean_sentence(raw: str) -> str:
    text = " ".join(raw.split()).strip(" -*#>•\t")
    if len(text) > 220:
        text = text[:217].rsplit(" ", 1)[0] + "..."
    return text


#: Delimiters wrapping the part of a prompt that names its actual subject.
#: They are plain text so a real model reads them as a labelled section, and
#: distinctive enough that ordinary prose never trips over them.
SUBJECT_OPEN = "[subject]"
SUBJECT_CLOSE = "[/subject]"

_SUBJECT_RE = re.compile(
    r"^[ \t]*\[subject\][ \t]*$\n?(?P<subject>.*?)(?:^[ \t]*\[/subject\][ \t]*$|\Z)",
    re.IGNORECASE | re.DOTALL | re.MULTILINE,
)
_MARKER_RE = re.compile(r"\[/?subject\]", re.IGNORECASE)


def with_subject(prompt: str, subject: str) -> str:
    """Return ``prompt`` with ``subject`` appended as a delimited section.

    The heuristic provider draws every generated value from that section alone,
    so none of the surrounding instruction wording can leak into a title, a hook
    or a beat.  Any other provider simply sees the subject restated under a
    clear label.  An empty subject changes nothing.
    """
    body = _MARKER_RE.sub(" ", str(subject or "")).strip()
    if not body:
        return prompt or ""
    head = (prompt or "").rstrip()
    section = f"{SUBJECT_OPEN}\n{body}\n{SUBJECT_CLOSE}\n"
    return f"{head}\n\n{section}" if head else section


def _split_subject(text: str) -> tuple[str, str]:
    """``(subject, everything else)`` for a prompt carrying a subject section."""
    match = _SUBJECT_RE.search(text)
    if match is None:
        return "", text
    subject = match.group("subject").strip()
    if not subject:
        return "", text
    rest = f"{text[:match.start()]}\n{text[match.end():]}"
    return subject, rest


def _find_subject(raw: str) -> tuple[str, str]:
    """Locate the subject section in a prompt of any length.

    The section is appended last, so on a prompt long enough to be truncated for
    analysis it lives in the tail rather than the head; both ends are searched
    before giving up and falling back to the label heuristic.
    """
    head = raw[:_MAX_PROMPT_CHARS]
    subject, rest = _split_subject(head)
    if not subject and len(raw) > _MAX_PROMPT_CHARS:
        subject, _ = _split_subject(raw[-_MAX_PROMPT_CHARS:])
        rest = head
    return subject[:_MAX_PROMPT_CHARS], rest[:_MAX_PROMPT_CHARS]


def _split_payload(text: str) -> tuple[str, str]:
    """Separate instructions from payload.

    Our own prompt builders end with a label line -- ``Topic:``, ``Premise:``,
    ``Transcript (timestamps in seconds):`` -- followed by the material the
    caller actually cares about.  Everything after the *first* such label is the
    payload and gets first claim on the generated content.

    First, not last: payloads contain colons of their own (a transcript line
    that ends "so here is my point:" looks exactly like a label), and splitting
    on the last one throws away everything above it -- which, for a transcript,
    is nearly all of the material.
    """
    lines = text.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.endswith(":") or not 0 < len(stripped.split()) <= 8:
            continue
        if any(rest.strip() for rest in lines[i + 1:]):
            return "\n".join(lines[i + 1:]), "\n".join(lines[:i])
    return "", text


def _sentences_of(text: str) -> list[str]:
    parts = [_clean_sentence(part) for part in _SENTENCE_SPLIT_RE.split(text)]
    return [s for s in parts if len(s) >= 12 and any(c.isalpha() for c in s)]


def _rank_keywords(text: str, weight: int, counts: dict[str, int], order: dict[str, int]) -> None:
    for i, match in enumerate(_WORD_RE.finditer(text)):
        word = match.group(0).lower()
        if len(word) < 4 or word in _STOPWORDS:
            continue
        counts[word] = counts.get(word, 0) + weight
        order.setdefault(word, i)


#: Fallback display names for a prompt that names nobody.  A person field --
#: a chat contact, a message sender -- has to read as a person: a topic keyword
#: put there produces a thread headed "Text" or "Lawn", which looks like a bug
#: to whoever watches the video.  Picked deterministically from the prompt.
_PERSON_NAMES: tuple[str, ...] = (
    "Alex", "Robin", "Sam", "Jordan", "Casey", "Riley", "Morgan", "Taylor",
    "Devon", "Quinn", "Harper", "Rowan", "Micah", "Noor", "Ira", "Dana",
)


def _fallback_person(keywords: list[str], seed: int | None) -> str:
    """A plausible display name, stable for the same prompt and seed."""
    key = f"{seed}:person:{'|'.join(keywords[:3])}"
    return random.Random(key).choice(_PERSON_NAMES)


def _proper_noun(text: str) -> str:
    """A name-ish token from the prompt: a mid-sentence capitalised word."""
    for match in list(_WORD_RE.finditer(text))[1:]:
        word = match.group(0)
        if word[0].isupper() and not word.isupper() and word.lower() not in _STOPWORDS:
            before = text[max(0, match.start() - 2):match.start()].strip()
            if before and before[-1] not in ".!?:\n":
                return word
    return ""


def _topped_up(sentences: list[str], keywords: list[str], rng: random.Random) -> list[str]:
    """Keep at least four sentences around so content does not visibly repeat.

    The order the filler templates are drawn in is the one genuinely arbitrary
    choice in this module, so it is the one thing seeded from ``settings.seed``:
    same seed, same script.
    """
    out = list(dict.fromkeys(sentences))
    templates = list(_FILLER_TEMPLATES)
    rng.shuffle(templates)
    i = 0
    while len(out) < 4 and i < len(templates):
        out.append(
            templates[i].format(
                a=keywords[i % len(keywords)],
                b=keywords[(i + 1) % len(keywords)],
                c=keywords[(i + 2) % len(keywords)],
            )
        )
        i += 1
    return out


def _material(prompt: str, seed: int) -> _Material:
    """Split a prompt into the raw material every generated value is built from."""
    raw = prompt or ""
    subject, rest = _find_subject(raw)
    if subject:
        # An explicit subject is exclusive: a two-word topic must not be topped
        # up out of the instructions that surround it, because "write a
        # 30-second vertical short" then becomes the title of the video.
        payload, instructions = subject, ""
        text = rest
    else:
        text = raw[:_MAX_PROMPT_CHARS]
        payload, instructions = _split_payload(text)

    # Instruction text is only material of last resort: when the caller labelled
    # a payload, the generated content comes from the payload alone.
    sentences = (_sentences_of(payload) or _sentences_of(instructions))[:_MAX_SENTENCES]

    counts: dict[str, int] = {}
    order: dict[str, int] = {}
    _rank_keywords(payload, 3, counts, order)
    _rank_keywords(instructions, 1, counts, order)
    keywords = sorted(counts, key=lambda w: (-counts[w], order[w]))[:_MAX_KEYWORDS]
    from_prompt = bool(keywords)
    keywords = keywords or list(_FALLBACK_KEYWORDS)

    numbers: list[float] = []
    for match in _NUMBER_RE.finditer(payload if subject else (payload or text)):
        try:
            numbers.append(float(match.group(0)))
        except ValueError:  # pragma: no cover - regex guarantees parseability
            continue
        if len(numbers) >= 64:
            break

    person = (
        _proper_noun(payload)
        or ("" if subject else _proper_noun(text))
        or _fallback_person(keywords, seed)
    )
    rng = random.Random(seed)
    # A subject too short to yield a sentence -- "cats" -- still has vocabulary,
    # and the filler shells built from that vocabulary are at least *about* it.
    # The generic fallbacks are for a prompt that carries no words at all.
    seeds = sentences or ([] if from_prompt else list(_FALLBACK_SENTENCES))
    return _Material(
        sentences=_topped_up(seeds, keywords, rng),
        keywords=keywords,
        numbers=numbers,
        rng=rng,
        person=person[:24],
    )


def _titlecase(text: str) -> str:
    words = text.split()
    return " ".join(w if w.isupper() else w.capitalize() for w in words)


def _slug(text: str, *, sep: str = "") -> str:
    out = _SLUG_RE.sub(sep, text.lower()).strip(sep or None)
    return out or "story"


def _sentence_score(sentence: str, ranking: dict[str, int]) -> float:
    words = [w.lower() for w in _WORD_RE.findall(sentence)]
    if not words:
        return 0.0
    hits = sum(ranking.get(w, 0) for w in words)
    length_penalty = 1.0 + abs(len(words) - 18) / 40.0
    return hits / length_penalty


# --------------------------------------------------------------------------- #
# schema walking
# --------------------------------------------------------------------------- #

@dataclass
class _Ctx:
    """Mutable state carried through one :func:`build_instance` call."""

    root: dict[str, Any]
    mat: _Material
    budget: int = _NODE_BUDGET

    def spend(self) -> bool:
        self.budget -= 1
        return self.budget > 0


def _deref(schema: dict[str, Any], ctx: _Ctx) -> dict[str, Any]:
    """Resolve a local ``$ref`` chain (``#/$defs/x``) against the root schema."""
    seen = 0
    while isinstance(schema, dict) and isinstance(schema.get("$ref"), str) and seen < 8:
        ref = schema["$ref"]
        seen += 1
        if not ref.startswith("#"):
            return {}
        node: Any = ctx.root
        for part in ref.lstrip("#/").split("/"):
            if not part:
                continue
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(node, list):
                try:
                    node = node[int(part)]
                except (ValueError, IndexError):
                    return {}
            elif isinstance(node, dict):
                node = node.get(part)
            else:
                return {}
            if node is None:
                return {}
        schema = node if isinstance(node, dict) else {}
    return schema


def _merged(schema: dict[str, Any], ctx: _Ctx, depth: int) -> dict[str, Any]:
    """Flatten ``$ref``/``allOf`` into a single node (shallow merge)."""
    schema = _deref(schema, ctx)
    all_of = schema.get("allOf")
    if not isinstance(all_of, list) or not all_of:
        return schema
    merged: dict[str, Any] = {k: v for k, v in schema.items() if k != "allOf"}
    for branch in all_of:
        if not isinstance(branch, dict):
            continue
        branch = _merged(branch, ctx, depth + 1)
        for key, value in branch.items():
            if key == "properties" and isinstance(value, dict):
                props = dict(merged.get("properties") or {})
                props.update(value)
                merged["properties"] = props
            elif key == "required" and isinstance(value, list):
                merged["required"] = list(dict.fromkeys(list(merged.get("required") or []) + value))
            elif key not in merged:
                merged[key] = value
    return merged


def _infer_type(schema: dict[str, Any]) -> str:
    raw = schema.get("type")
    if isinstance(raw, list):
        for candidate in raw:
            if candidate != "null":
                return str(candidate)
        return "null"
    if isinstance(raw, str):
        return raw
    if "properties" in schema or "additionalProperties" in schema or "minProperties" in schema:
        return "object"
    if "items" in schema or "minItems" in schema:
        return "array"
    if any(k in schema for k in ("minimum", "maximum", "multipleOf", "exclusiveMinimum")):
        return "number"
    return "string"


def _empty_for(schema: dict[str, Any]) -> Any:
    """The cheapest value for a node, still honouring its scalar bounds."""
    kind = _infer_type(schema)
    if kind == "string":
        lo, hi = schema.get("minLength"), schema.get("maxLength")
        width = int(lo) if isinstance(lo, int) and lo > 0 else 0
        if isinstance(hi, int):
            width = min(width, max(hi, 0))
        return "x" * width
    if kind in ("number", "integer"):
        return _clamp_number(0.0, schema, integral=kind == "integer")
    return {"object": {}, "array": [], "boolean": False, "null": None}.get(kind, "")


def _degenerate(node: dict[str, Any], ctx: _Ctx) -> Any:
    """Value emitted when the depth cap or the node budget runs out.

    An object still carries its required keys -- dropping them is what turns a
    merely shallow instance into an invalid one -- but their values are the
    non-recursive :func:`_empty_for` placeholders, so this always terminates.
    """
    if _infer_type(node) != "object":
        return _empty_for(node)
    props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    out: dict[str, Any] = {}
    for key in node.get("required") or ():
        if not isinstance(key, str):
            continue
        sub = props.get(key)
        if not isinstance(sub, dict):
            sub = {}
        else:
            try:
                sub = _deref(sub, ctx)
            except Exception:  # pragma: no cover - defensive
                sub = {}
        out[key] = _empty_for(sub)
    return out


def _build(schema: Any, ctx: _Ctx, *, name: str = "", depth: int = 0, index: int = 0) -> Any:
    """Emit one value for ``schema``.  Defensive: never raises, always returns."""
    if not isinstance(schema, dict):
        # ``True`` means "anything"; ``False``/garbage means "nothing sensible".
        return ctx.mat.take_sentence() if schema is not False else None
    try:
        node = _merged(schema, ctx, depth)
    except Exception:  # pragma: no cover - defensive
        return ""
    if depth >= _MAX_DEPTH or not ctx.spend():
        return _degenerate(node, ctx)

    if "const" in node:
        return node["const"]
    enum = node.get("enum")
    if isinstance(enum, list) and enum:
        return _pick_enum(enum, ctx, name, index)

    for key in ("oneOf", "anyOf"):
        branches = node.get(key)
        if isinstance(branches, list) and branches:
            branch = branches[index % len(branches)] if key == "anyOf" else branches[0]
            merged = {k: v for k, v in node.items() if k not in ("oneOf", "anyOf")}
            if isinstance(branch, dict):
                merged.update(branch)
            return _build(merged, ctx, name=name, depth=depth + 1, index=index)

    kind = _infer_type(node)
    if kind == "object":
        return _build_object(node, ctx, name=name, depth=depth, index=index)
    if kind == "array":
        return _build_array(node, ctx, name=name, depth=depth, index=index)
    if kind == "boolean":
        return _build_boolean(name, index)
    if kind in ("number", "integer"):
        return _build_number(node, ctx, name=name, index=index, integral=kind == "integer")
    if kind == "null":
        return None
    return _build_string(node, ctx, name=name, index=index)


def _pick_enum(enum: list[Any], ctx: _Ctx, name: str, index: int) -> Any:
    """Prefer an option the prompt actually mentions, else rotate deterministically."""
    haystack = {k.lower() for k in ctx.mat.keywords}
    for option in enum:
        if isinstance(option, str) and option.lower() in haystack:
            return option
    return enum[index % len(enum)]


def _build_object(node: dict[str, Any], ctx: _Ctx, *, name: str, depth: int, index: int) -> dict[str, Any]:
    props = node.get("properties") if isinstance(node.get("properties"), dict) else {}
    required = [k for k in (node.get("required") or []) if isinstance(k, str)]
    out: dict[str, Any] = {}
    for key, sub in props.items():
        out[key] = _build(sub, ctx, name=str(key), depth=depth + 1, index=index)

    extra_schema = node.get("additionalProperties")
    for key in required:
        if key in out:
            continue
        sub = extra_schema if isinstance(extra_schema, dict) else {}
        out[key] = _build(sub, ctx, name=key, depth=depth + 1, index=index)

    minimum = int(node.get("minProperties") or 0)
    if len(out) < minimum and extra_schema is not False:
        sub = extra_schema if isinstance(extra_schema, dict) else {}
        while len(out) < minimum:
            key = f"{_slug(name or 'key')}_{len(out) + 1}"
            out[key] = _build(sub, ctx, name=key, depth=depth + 1, index=index)

    maximum = node.get("maxProperties")
    if isinstance(maximum, int) and len(out) > maximum:
        keep = list(dict.fromkeys(required + list(out)))[:maximum]
        out = {k: out[k] for k in keep if k in out}

    _harmonise_span(out, props)
    return out


def _harmonise_span(obj: dict[str, Any], props: dict[str, Any]) -> None:
    """Make ``end`` follow ``start`` when an object carries both."""
    start, end = obj.get("start"), obj.get("end")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        return
    if isinstance(start, bool) or isinstance(end, bool) or end > start:
        return
    schema = props.get("end") if isinstance(props.get("end"), dict) else {}
    ceiling = schema.get("maximum")
    candidate = float(start) + 30.0
    if isinstance(ceiling, (int, float)) and candidate > ceiling:
        return
    obj["end"] = int(candidate) if isinstance(end, int) else candidate


def _build_array(node: dict[str, Any], ctx: _Ctx, *, name: str, depth: int, index: int) -> list[Any]:
    items = node.get("items")
    if isinstance(items, list):  # tuple-style validation
        out = [_build(sub, ctx, name=name, depth=depth + 1, index=i) for i, sub in enumerate(items)]
        # ``minItems`` can exceed the tuple: the surplus is governed by
        # ``additionalItems`` (or by nothing at all, i.e. "any value").
        extra = node.get("additionalItems")
        tail = extra if isinstance(extra, dict) else {}
        wanted = node.get("minItems")
        wanted = int(wanted) if isinstance(wanted, int) and wanted > 0 else 0
        if extra is not False:
            for i in range(len(out), wanted):
                out.append(_build(tail, ctx, name=name, depth=depth + 1, index=i))
        return out

    lo = node.get("minItems")
    hi = node.get("maxItems")
    lo = int(lo) if isinstance(lo, int) and lo > 0 else 0
    # Near the depth cap (or out of budget) emit the bare minimum: that is what
    # lets a self-referential schema -- a tree of ``$ref``s -- terminate with an
    # instance that still validates instead of being truncated mid-object.
    if depth >= _MAX_DEPTH - 3 or ctx.budget < 64:
        default = 0
    else:
        default = 3 if depth <= 3 else 1
    count = max(lo, default)
    if isinstance(hi, int):
        count = min(count, max(hi, 0))
    if isinstance(items, dict) and _infer_type(_merged(items, ctx, depth)) == "array":
        count = min(count, max(lo, 2))

    sub = items if isinstance(items, dict) else {}
    if node.get("uniqueItems"):
        return _unique_items(sub, ctx, name=name, depth=depth, index=index, count=count)
    return [_build(sub, ctx, name=name, depth=depth + 1, index=index + i) for i in range(count)]


def _item_key(value: Any) -> str:
    """A stable identity for an emitted value, for ``uniqueItems`` de-duplication."""
    try:
        return json.dumps(value, sort_keys=True, default=repr)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return repr(value)


def _unique_items(
    sub: dict[str, Any], ctx: _Ctx, *, name: str, depth: int, index: int, count: int
) -> list[Any]:
    """``count`` distinct values, keeping every one the type the schema declares.

    Re-rolling with a fresh ``index`` is what actually varies the content (the
    material cursors advance too), so most of the work is just asking again.
    Only when that runs dry is a scalar nudged apart -- never by rewriting it as
    a string, which is how a ``boolean`` array used to come back holding
    ``"False 3"``.
    """
    out: list[Any] = []
    seen: set[str] = set()
    attempts = 0
    offset = index
    while len(out) < count and attempts < count * 3 + 12:
        value = _build(sub, ctx, name=name, depth=depth + 1, index=offset)
        key = _item_key(value)
        if key not in seen:
            seen.add(key)
            out.append(value)
        offset += 1
        attempts += 1
    base = out[0] if out else _build(sub, ctx, name=name, depth=depth + 1, index=offset)
    nudge = 0
    while len(out) < count and nudge < count * 3 + 12:
        nudge += 1
        for delta in (nudge, -nudge):  # a value pinned to its maximum has to step down
            value = _nudged(base, sub, delta)
            key = _item_key(value)
            if key in seen:
                continue
            seen.add(key)
            out.append(value)
            break
    return out


def _nudged(value: Any, sub: dict[str, Any], step: int) -> Any:
    """Move ``value`` off itself without leaving the type its schema declares.

    Booleans, nulls, enums and consts have a finite domain: a schema asking for
    more unique items than they hold is simply unsatisfiable, and we return the
    value unchanged rather than lying about its type.
    """
    if isinstance(value, bool) or value is None:
        return value
    if "enum" in sub or "const" in sub:  # a closed domain: moving off it means leaving the schema
        return value
    if isinstance(value, dict):
        props = sub.get("properties") if isinstance(sub.get("properties"), dict) else {}
        for key, item in value.items():
            inner = props.get(key)
            moved = _nudged(item, inner if isinstance(inner, dict) else {}, step)
            if _item_key(moved) != _item_key(item):
                clone = dict(value)
                clone[key] = moved
                return clone
        return value
    if isinstance(value, list):
        items = sub.get("items")
        inner = items if isinstance(items, dict) else {}
        for position, item in enumerate(value):
            moved = _nudged(item, inner, step)
            if _item_key(moved) == _item_key(item):
                continue
            clone = list(value)
            clone[position] = moved
            # Moving one element must not collide with its neighbours: a nested
            # ``uniqueItems`` list is easy to break from the outside.
            if sub.get("uniqueItems") and len({_item_key(v) for v in clone}) != len(clone):
                continue
            return clone
        return value
    if isinstance(value, str):
        hi = sub.get("maxLength")
        pattern = sub.get("pattern")
        for suffix in (f" {step}", str(abs(step)), "x" * max(1, abs(step))):
            head = value
            if isinstance(hi, int) and len(head) + len(suffix) > hi:
                head = head[:max(0, hi - len(suffix))]
            candidate = head + suffix
            if not isinstance(pattern, str) or not pattern:
                return candidate
            try:
                if re.search(pattern, candidate):
                    return candidate
            except re.error:  # pragma: no cover - re-raised nowhere, caller keeps the value
                return value
        sampled = _sample_pattern(str(pattern), max(1, abs(step) + 1))
        if sampled is not None and (not isinstance(hi, int) or len(sampled) <= hi):
            return sampled
        return value
    if isinstance(value, (int, float)):
        multiple = sub.get("multipleOf")
        grain = float(multiple) if isinstance(multiple, (int, float)) and multiple > 0 else 1.0
        return _clamp_number(float(value) + grain * step, sub, integral=isinstance(value, int))
    return value


def _build_boolean(name: str, index: int) -> bool:
    lowered = name.lower()
    if "outgoing" in lowered:
        return index % 2 == 1
    if "emphasis" in lowered or "highlight" in lowered:
        return index % 3 == 0
    if lowered.startswith(("is_", "has_", "should_")) or "enabled" in lowered:
        return True
    return index % 2 == 0


# --------------------------------------------------------------------------- #
# scalar content, routed by property name
# --------------------------------------------------------------------------- #

def _build_number(node: dict[str, Any], ctx: _Ctx, *, name: str, index: int, integral: bool) -> float | int:
    mat = ctx.mat
    lowered = name.lower()
    step = mat.next_index()

    if "start" in lowered or lowered in ("from", "begin", "offset"):
        value = _window(node, index)[0]
    elif "end" in lowered or lowered in ("to", "stop", "until"):
        value = _window(node, index)[1]
    elif "score" in lowered or "confidence" in lowered or "probability" in lowered:
        value = max(0.05, 0.92 - 0.07 * index)
    elif "delay" in lowered or "pause" in lowered or "gap" in lowered:
        value = 0.4 + 0.35 * (index % 4)
    elif "duration" in lowered or "seconds" in lowered or "length" in lowered:
        value = 28.0 + 4.0 * index
    elif "upvote" in lowered or "like" in lowered or "view" in lowered or "count" in lowered:
        value = 1000.0 + 917.0 * (index + 1)
    elif "comment" in lowered or "repl" in lowered:
        value = 140.0 + 63.0 * (index + 1)
    else:
        from_prompt = mat.number(step)
        value = from_prompt if from_prompt is not None else float(index + 1)

    return _clamp_number(value, node, integral=integral)


_WINDOW_LENGTH = 32.0
_WINDOW_STRIDE = 45.0


def _window(node: dict[str, Any], index: int) -> tuple[float, float]:
    """The ``(start, end)`` pair for the ``index``-th time window.

    With no ``maximum`` the windows march forward on a fixed stride -- the usual
    case, since a transcript has no declared ceiling.  When the schema *does*
    cap the value, the stride wraps inside the allowed range instead of pinning
    every window past the first onto the ceiling, which used to make every clip
    after the second a zero-length ``(max, max)``.
    """
    lo = node.get("minimum")
    floor = float(lo) if isinstance(lo, (int, float)) else 0.0
    start = floor + 12.0 + _WINDOW_STRIDE * index
    hi = node.get("maximum")
    if not isinstance(hi, (int, float)):
        return start, start + _WINDOW_LENGTH
    ceiling = float(hi)
    span = ceiling - floor
    if span <= 0.0:
        return floor, floor
    length = min(_WINDOW_LENGTH, span / 2.0)
    usable = max(span - length, span / 4.0)
    start = floor + (start - floor) % usable
    return start, min(start + length, ceiling)


def _clamp_number(value: float, node: dict[str, Any], *, integral: bool) -> float | int:
    lo = node.get("minimum")
    hi = node.get("maximum")
    ex_lo = node.get("exclusiveMinimum")
    ex_hi = node.get("exclusiveMaximum")
    if not math.isfinite(value):
        value = 0.0
    if isinstance(lo, (int, float)) and value < lo:
        value = float(lo)
    if isinstance(hi, (int, float)) and value > hi:
        value = float(hi)
    multiple = node.get("multipleOf")
    epsilon = 1.0 if integral else (float(multiple) if isinstance(multiple, (int, float)) and multiple > 0
                                    else 0.001)
    if isinstance(ex_lo, (int, float)) and value <= ex_lo:
        value = float(ex_lo) + epsilon
    if isinstance(ex_hi, (int, float)) and value >= ex_hi:
        value = float(ex_hi) - epsilon
    if isinstance(multiple, (int, float)) and multiple > 0:
        in_range = value
        value = round(value / multiple) * multiple
        # Snapping to the grid can walk back over a bound, so walk off it again
        # -- one whole step, which is the only move that stays on the grid.
        for _ in range(4):
            if isinstance(lo, (int, float)) and value < lo:
                value += multiple
            elif isinstance(ex_lo, (int, float)) and value <= ex_lo:
                value += multiple
            elif isinstance(hi, (int, float)) and value > hi:
                value -= multiple
            elif isinstance(ex_hi, (int, float)) and value >= ex_hi:
                value -= multiple
            else:
                break
        else:
            # No point of the grid lies inside the range: the schema cannot be
            # satisfied, so keep the value the bounds asked for rather than one
            # that is out of range *and* off the grid.
            value = in_range
    if integral:
        result = int(round(value))
        if isinstance(lo, (int, float)) and result < lo:
            result = int(math.ceil(lo))
        if isinstance(hi, (int, float)) and result > hi:
            result = int(math.floor(hi))
        return result
    return round(float(value), 3)


def _build_string(node: dict[str, Any], ctx: _Ctx, *, name: str, index: int) -> str:
    """Pick the content for one string field, routed by the property's name."""
    mat = ctx.mat
    lowered = name.lower()
    step = mat.next_index()
    fmt = str(node.get("format") or "").lower()

    if fmt in ("date-time", "datetime"):
        text = "2024-05-01T09:30:00Z"
    elif fmt == "date":
        text = "2024-05-01"
    elif fmt == "time":
        text = "09:30:00"
    elif fmt == "email" or "email" in lowered:
        text = f"{_slug(mat.take_keyword(), sep='.')}@example.invalid"
    elif fmt in ("uri", "url") or lowered in ("url", "uri", "link", "href"):
        text = f"https://example.invalid/{_slug(mat.take_keyword(), sep='-')}"
    elif fmt == "uuid":
        text = "00000000-0000-4000-8000-000000000000"
    elif "community" in lowered or "subreddit" in lowered or lowered == "forum":
        # Plain readable names, never a site-specific handle grammar: the story
        # card is our own design and must not borrow another service's prefixes.
        text = _titlecase(_slug(mat.take_keyword() + " " + mat.take_keyword(), sep=" "))
    elif "author" in lowered or "username" in lowered or "handle" in lowered or lowered == "user":
        text = _slug(mat.take_keyword() + " " + mat.take_keyword(), sep=" ")
    elif "hashtag" in lowered or lowered in ("tag", "tags"):
        # Indexed by array position, not the shared cursor, so a hashtag list
        # comes out as the prompt's strongest keywords rather than its dregs.
        text = "#" + _slug(mat.keyword(index))
    elif "title" in lowered or "headline" in lowered or "subject" in lowered:
        text = _titlecase(_trim_words(mat.take_sentence().rstrip(".!?"), 9))
    elif "hook" in lowered or "tagline" in lowered or "teaser" in lowered:
        text = _trim_words(mat.take_sentence().rstrip(".!?"), 11)
    elif "cta" in lowered or "call_to_action" in lowered:
        text = f"Follow for more on {mat.keyword(0)}."
    elif "reason" in lowered or "rationale" in lowered or lowered.startswith("why"):
        text = f"It lands because {_lower_first(_trim_words(mat.take_sentence(), 16))}"
    elif "image" in lowered or "visual" in lowered or "art" in lowered:
        text = (
            f"{_titlecase(mat.take_phrase(2))}, vertical 9:16 framing, "
            f"shallow depth of field, soft directional light"
        )
    elif "broll" in lowered or "b_roll" in lowered or "keyword" in lowered or "query" in lowered:
        text = mat.take_phrase(3)
    elif "sender" in lowered:
        # Mirror the parity used for ``outgoing`` so the two stay consistent.
        text = "Me" if index % 2 == 1 else mat.person
    elif "contact" in lowered or "speaker" in lowered or "recipient" in lowered:
        text = mat.person
    elif lowered.endswith("name") or lowered == "who":
        # A generic name field, unlike a chat contact, should vary per instance.
        text = _titlecase(mat.take_keyword())
    elif "body" in lowered or "story" in lowered or "description" in lowered or "summary" in lowered:
        text = " ".join(mat.take_sentences(3))
    else:
        text = mat.take_sentence()

    return _fit_string(text, node, mat, step)


def _lower_first(text: str) -> str:
    return text[:1].lower() + text[1:] if text else text


def _trim_words(text: str, count: int) -> str:
    words = text.split()
    return " ".join(words[:count]) if words else text


def _fit_string(text: str, node: dict[str, Any], mat: _Material, step: int) -> str:
    text = " ".join(text.split()) or mat.keyword(step)
    lo = node.get("minLength")
    hi = node.get("maxLength")
    if isinstance(hi, int) and hi >= 0 and len(text) > hi:
        clipped = text[:hi]
        if " " in clipped[max(0, hi - 24):]:
            clipped = clipped.rsplit(" ", 1)[0]
        text = clipped[:hi].rstrip(" ,;:-") or text[:hi]
    if isinstance(lo, int) and len(text) < lo:
        filler = 0
        while len(text) < lo and filler < 40:
            text = f"{text} {mat.keyword(step + filler)}".strip()
            filler += 1
        if len(text) < lo:
            text = (text + " " + "detail " * lo)[:lo]
        if isinstance(hi, int) and len(text) > hi:
            text = text[:hi]

    pattern = node.get("pattern")
    if isinstance(pattern, str) and pattern:
        text = _match_pattern(text, pattern, node, mat, step)
    return text


def _match_pattern(text: str, pattern: str, node: dict[str, Any], mat: _Material, step: int) -> str:
    """Bend ``text`` into something matching ``pattern``.

    Prompt-derived text wins whenever it already matches.  Otherwise a handful
    of reshapings are tried, and then the pattern itself is *sampled* -- a
    regex like ``^#[a-z0-9]+$`` describes exactly one family of strings and
    nothing extracted from the prompt will ever be in it.
    """
    try:
        compiled = re.compile(pattern)
    except re.error:
        return text
    lo = int(node.get("minLength") or 0)
    hi = node.get("maxLength")
    candidates = [
        text,
        _slug(text, sep="-"),
        _slug(text),
        "".join(ch for ch in text if ch.isalnum()),
        mat.keyword(step),
        _slug(mat.keyword(step)),
        "".join(ch for ch in text if ch.isdigit()) or "42",
    ]
    for reps in (max(1, lo), max(2, lo), 1, 3, 8):
        sample = _sample_pattern(pattern, reps)
        if sample is not None and sample not in candidates:
            candidates.append(sample)
    candidates += ["a" * max(1, lo), "0" * max(1, lo), ""]
    fallback = ""
    for candidate in candidates:
        if not compiled.search(candidate) or len(candidate) < lo:
            continue
        if isinstance(hi, int) and len(candidate) > hi:
            fallback = fallback or candidate
            continue
        return candidate
    return fallback or text


# --------------------------------------------------------------------------- #
# a small regular-expression sampler
# --------------------------------------------------------------------------- #
#
# Enough of the syntax to satisfy the patterns that turn up in JSON Schema:
# literals, escapes, classes, groups, alternation and quantifiers.  Anything it
# cannot parse yields ``None`` and the caller falls back to reshaped prompt
# text; anything it gets *wrong* is caught by the caller's ``re.search`` check.

_CLASS_POOL = "abcdefghijklmnopqrstuvwxyz0123456789-_.ABCDEFGHIJKLMNOPQRSTUVWXYZ "
_CLASS_SHORTHAND = {
    "d": "0123456789",
    "w": "abcdefghijklmnopqrstuvwxyz0123456789_",
    "s": " ",
    "D": "abcdefghijklmnopqrstuvwxyz",
    "W": "-. ",
    "S": "abcdefghijklmnopqrstuvwxyz0123456789",
}
_ESCAPE_LITERALS = {"n": "\n", "t": "\t", "r": "\r", "f": "\f", "v": "\v", "0": "\0"}
_ZERO_WIDTH = frozenset("bBAZ")


def _sample_pattern(pattern: str, reps: int) -> str | None:
    """One string the regex accepts, with unbounded repeats taken ``reps`` times."""
    if not pattern or len(pattern) > _MAX_PATTERN_CHARS:
        return None
    try:
        text, pos = _sample_alt(pattern, 0, max(0, reps), 0)
    except (ValueError, IndexError, RecursionError):
        return None
    return text if pos >= len(pattern) else None


def _sample_alt(src: str, pos: int, reps: int, depth: int) -> tuple[str, int]:
    """Sample the first branch of an alternation, parsing past the rest."""
    if depth > _MAX_PATTERN_DEPTH:
        raise ValueError("pattern nested too deeply")
    text, pos = _sample_seq(src, pos, reps, depth)
    while pos < len(src) and src[pos] == "|":
        _, pos = _sample_seq(src, pos + 1, reps, depth)
    return text, pos


def _sample_seq(src: str, pos: int, reps: int, depth: int) -> tuple[str, int]:
    out: list[str] = []
    while pos < len(src) and src[pos] not in "|)":
        atom, pos = _sample_atom(src, pos, reps, depth)
        low, high, pos = _read_quantifier(src, pos)
        count = max(low, min(reps, high)) if high is not None else max(low, reps)
        out.append(atom * count)
    return "".join(out), pos


def _sample_atom(src: str, pos: int, reps: int, depth: int) -> tuple[str, int]:
    char = src[pos]
    if char == "(":
        return _sample_group(src, pos, reps, depth)
    if char == "[":
        return _sample_class(src, pos)
    if char == "\\":
        return _sample_escape(src, pos)
    if char in "^$":
        return "", pos + 1
    if char == ".":
        return "a", pos + 1
    if char in "*+?{":  # a dangling quantifier: not something we can sample
        raise ValueError("quantifier with nothing to repeat")
    return char, pos + 1


def _sample_group(src: str, pos: int, reps: int, depth: int) -> tuple[str, int]:
    pos += 1  # past '('
    keep = True
    if src.startswith("?", pos):
        if src.startswith("?P<", pos) or src.startswith("?'", pos):
            closer = ">" if src.startswith("?P<", pos) else "'"
            end = src.find(closer, pos)
            if end < 0:
                raise ValueError("unterminated group name")
            pos = end + 1
        elif src.startswith("?:", pos):
            pos += 2
        elif src.startswith(("?=", "?!"), pos):
            keep, pos = False, pos + 2
        elif src.startswith(("?<=", "?<!"), pos):
            keep, pos = False, pos + 3
        else:  # inline flags, ``(?P=name)`` and friends: skip the whole group
            end = src.find(")", pos)
            if end < 0:
                raise ValueError("unterminated group")
            return "", end + 1
    text, pos = _sample_alt(src, pos, reps, depth + 1)
    if pos >= len(src) or src[pos] != ")":
        raise ValueError("unterminated group")
    return (text if keep else ""), pos + 1


def _sample_escape(src: str, pos: int) -> tuple[str, int]:
    if pos + 1 >= len(src):
        raise ValueError("trailing backslash")
    esc = src[pos + 1]
    if esc in _ZERO_WIDTH:
        return "", pos + 2
    if esc in _CLASS_SHORTHAND:
        return _CLASS_SHORTHAND[esc][0], pos + 2
    if esc in _ESCAPE_LITERALS:
        return _ESCAPE_LITERALS[esc], pos + 2
    if esc.isdigit():  # a back-reference: not something we can sample
        raise ValueError("back-reference")
    return esc, pos + 2


def _sample_class(src: str, pos: int) -> tuple[str, int]:
    pos += 1  # past '['
    negated = src.startswith("^", pos)
    pos += 1 if negated else 0
    allowed: set[str] = set()
    first = True
    while pos < len(src) and (src[pos] != "]" or first):
        first = False
        if src[pos] == "\\" and pos + 1 < len(src):
            esc = src[pos + 1]
            pos += 2
            allowed.update(_CLASS_SHORTHAND.get(esc, _ESCAPE_LITERALS.get(esc, esc)))
            continue
        char = src[pos]
        pos += 1
        if pos + 1 < len(src) and src[pos] == "-" and src[pos + 1] != "]":
            stop = src[pos + 1]
            pos += 2
            if ord(stop) >= ord(char):
                allowed.update(chr(c) for c in range(ord(char), min(ord(stop), ord(char) + 512) + 1))
            continue
        allowed.add(char)
    if pos >= len(src):
        raise ValueError("unterminated character class")
    pos += 1  # past ']'
    if negated:
        pick = next((c for c in _CLASS_POOL if c not in allowed), None)
    else:
        pick = next((c for c in _CLASS_POOL if c in allowed), None)
        if pick is None:
            pick = min(allowed) if allowed else None
    if pick is None:
        raise ValueError("character class matches nothing we can emit")
    return pick, pos


def _read_quantifier(src: str, pos: int) -> tuple[int, int | None, int]:
    """``(low, high, pos)`` for the quantifier at ``pos``; ``(1, 1, pos)`` if none."""
    if pos >= len(src):
        return 1, 1, pos
    char = src[pos]
    if char == "*":
        low, high, pos = 0, None, pos + 1
    elif char == "+":
        low, high, pos = 1, None, pos + 1
    elif char == "?":
        low, high, pos = 0, 1, pos + 1
    elif char == "{":
        match = re.match(r"\{(\d*)(,?)(\d*)\}", src[pos:])
        if not match or not (match.group(1) or match.group(3)):
            return 1, 1, pos
        start, comma, stop = match.groups()
        low = int(start or 0)
        high = int(stop) if stop else (None if comma else low)
        pos += match.end()
    else:
        return 1, 1, pos
    if pos < len(src) and src[pos] in "?+":  # lazy / possessive
        pos += 1
    return low, high, pos


# --------------------------------------------------------------------------- #
# public helpers
# --------------------------------------------------------------------------- #

def build_instance(schema: Any, prompt: str = "", *, seed: int = 1234) -> Any:
    """Emit a value that validates against ``schema``, derived from ``prompt``.

    Unlike :meth:`HeuristicProvider.complete_json` this does not insist on an
    object at the root, so it can be pointed at any schema fragment.
    """
    try:
        root = schema if isinstance(schema, dict) else {}
        ctx = _Ctx(root=root, mat=_material(prompt, seed))
        return _build(root, ctx)
    except Exception:  # pragma: no cover - the contract is "never raises"
        return _empty_for(schema if isinstance(schema, dict) else {})


_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, dict),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "boolean": lambda v: isinstance(v, bool),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "null": lambda v: v is None,
}


def validate_instance(data: Any, schema: Any, *, path: str = "$", _root: Any = None) -> list[str]:
    """Return a list of human-readable schema violations (empty means valid).

    A deliberately small JSON Schema subset -- the keywords this package
    actually uses -- kept here so the generator and the tests agree on what
    "valid" means without pulling in a validation library.
    """
    root = _root if _root is not None else schema
    if schema is True or schema == {}:
        return []
    if schema is False:
        return [f"{path}: schema forbids any value"]
    if not isinstance(schema, dict):
        return [f"{path}: schema node is not an object"]

    problems: list[str] = []
    if isinstance(schema.get("$ref"), str):
        ctx = _Ctx(root=root if isinstance(root, dict) else {}, mat=_material("", 0))
        return validate_instance(data, _deref(schema, ctx), path=path, _root=root)

    for branch in schema.get("allOf") or ():
        problems += validate_instance(data, branch, path=path, _root=root)
    any_of = schema.get("anyOf")
    if isinstance(any_of, list) and any_of:
        if all(validate_instance(data, b, path=path, _root=root) for b in any_of):
            problems.append(f"{path}: matches no anyOf branch")
    one_of = schema.get("oneOf")
    if isinstance(one_of, list) and one_of:
        matches = sum(1 for b in one_of if not validate_instance(data, b, path=path, _root=root))
        if matches != 1:
            problems.append(f"{path}: matches {matches} oneOf branches, expected exactly 1")

    if "const" in schema and data != schema["const"]:
        problems.append(f"{path}: expected const {schema['const']!r}, got {data!r}")
    enum = schema.get("enum")
    if isinstance(enum, list) and enum and data not in enum:
        problems.append(f"{path}: {data!r} is not one of {enum!r}")

    declared = schema.get("type")
    types = declared if isinstance(declared, list) else ([declared] if isinstance(declared, str) else [])
    if types and not any(_TYPE_CHECKS.get(t, lambda _v: True)(data) for t in types):
        problems.append(f"{path}: expected type {'|'.join(types)}, got {type(data).__name__}")
        return problems

    if isinstance(data, dict):
        problems += _validate_object(data, schema, path, root)
    elif isinstance(data, list):
        problems += _validate_array(data, schema, path, root)
    elif isinstance(data, str):
        problems += _validate_string(data, schema, path)
    elif isinstance(data, (int, float)) and not isinstance(data, bool):
        problems += _validate_number(data, schema, path)
    return problems


def _validate_object(data: dict[str, Any], schema: dict[str, Any], path: str, root: Any) -> list[str]:
    problems: list[str] = []
    props = schema.get("properties") if isinstance(schema.get("properties"), dict) else {}
    for key in schema.get("required") or ():
        if key not in data:
            problems.append(f"{path}: missing required property {key!r}")
    extra = schema.get("additionalProperties")
    for key, value in data.items():
        if key in props:
            problems += validate_instance(value, props[key], path=f"{path}.{key}", _root=root)
        elif extra is False:
            problems.append(f"{path}: additional property {key!r} is not allowed")
        elif isinstance(extra, dict):
            problems += validate_instance(value, extra, path=f"{path}.{key}", _root=root)
    lo, hi = schema.get("minProperties"), schema.get("maxProperties")
    if isinstance(lo, int) and len(data) < lo:
        problems.append(f"{path}: has {len(data)} properties, minimum {lo}")
    if isinstance(hi, int) and len(data) > hi:
        problems.append(f"{path}: has {len(data)} properties, maximum {hi}")
    return problems


def _validate_array(data: list[Any], schema: dict[str, Any], path: str, root: Any) -> list[str]:
    problems: list[str] = []
    items = schema.get("items")
    if isinstance(items, list):
        for i, sub in enumerate(items):
            if i < len(data):
                problems += validate_instance(data[i], sub, path=f"{path}[{i}]", _root=root)
    elif items is not None:
        for i, value in enumerate(data):
            problems += validate_instance(value, items, path=f"{path}[{i}]", _root=root)
    lo, hi = schema.get("minItems"), schema.get("maxItems")
    if isinstance(lo, int) and len(data) < lo:
        problems.append(f"{path}: has {len(data)} items, minimum {lo}")
    if isinstance(hi, int) and len(data) > hi:
        problems.append(f"{path}: has {len(data)} items, maximum {hi}")
    if schema.get("uniqueItems"):
        seen: list[Any] = []
        for value in data:
            if value in seen:
                problems.append(f"{path}: duplicate item {value!r}")
                break
            seen.append(value)
    return problems


def _validate_string(data: str, schema: dict[str, Any], path: str) -> list[str]:
    problems: list[str] = []
    lo, hi = schema.get("minLength"), schema.get("maxLength")
    if isinstance(lo, int) and len(data) < lo:
        problems.append(f"{path}: length {len(data)} is below minLength {lo}")
    if isinstance(hi, int) and len(data) > hi:
        problems.append(f"{path}: length {len(data)} exceeds maxLength {hi}")
    pattern = schema.get("pattern")
    if isinstance(pattern, str) and pattern:
        try:
            if not re.search(pattern, data):
                problems.append(f"{path}: {data!r} does not match pattern {pattern!r}")
        except re.error:
            pass
    return problems


def _validate_number(data: float, schema: dict[str, Any], path: str) -> list[str]:
    problems: list[str] = []
    lo, hi = schema.get("minimum"), schema.get("maximum")
    ex_lo, ex_hi = schema.get("exclusiveMinimum"), schema.get("exclusiveMaximum")
    if isinstance(lo, (int, float)) and data < lo:
        problems.append(f"{path}: {data} is below minimum {lo}")
    if isinstance(hi, (int, float)) and data > hi:
        problems.append(f"{path}: {data} is above maximum {hi}")
    if isinstance(ex_lo, (int, float)) and data <= ex_lo:
        problems.append(f"{path}: {data} is not above exclusiveMinimum {ex_lo}")
    if isinstance(ex_hi, (int, float)) and data >= ex_hi:
        problems.append(f"{path}: {data} is not below exclusiveMaximum {ex_hi}")
    multiple = schema.get("multipleOf")
    if isinstance(multiple, (int, float)) and multiple > 0:
        if abs(data / multiple - round(data / multiple)) > 1e-6:
            problems.append(f"{path}: {data} is not a multiple of {multiple}")
    return problems


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #

class HeuristicProvider:
    """Rule-based provider: no network, no optional imports, no exceptions."""

    name = "heuristic"

    def __init__(self, *, settings: Settings | None = None) -> None:
        self._settings = _resolve_settings(settings)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "HeuristicProvider()"

    @property
    def settings(self) -> Settings:
        return self._settings

    def available(self) -> bool:
        """Always true -- that is the point of this provider."""
        return True

    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None) -> str:
        """Extractive answer: the prompt's own best sentences, in order."""
        try:
            return self._complete(prompt, max_tokens=max_tokens)
        except Exception:  # pragma: no cover - the contract is "never raises"
            return ""

    def complete_json(
        self,
        prompt: str,
        schema: dict[str, Any],
        *,
        system: str = "",
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Emit an object that validates against ``schema``.

        A schema whose root is not an object still produces a dict: the value is
        wrapped under a ``"result"`` key so the return type stays honest.
        """
        value = build_instance(schema, prompt, seed=self._settings.seed)
        if isinstance(value, dict):
            return value
        return {"result": value}

    # -- internals --------------------------------------------------------- #
    def _complete(self, prompt: str, *, max_tokens: int | None) -> str:
        mat = _material(prompt, self._settings.seed)
        limit = max(80, min(4000, (max_tokens or 300) * 4))
        ranking = {word: len(mat.keywords) - i for i, word in enumerate(mat.keywords)}
        scored = sorted(
            ((_sentence_score(s, ranking), -i, s) for i, s in enumerate(mat.sentences)),
            reverse=True,
        )
        chosen = {s for _, _, s in scored[:4]}
        ordered = [s for s in mat.sentences if s in chosen]
        if not ordered:
            ordered = [mat.sentence(0)]

        out: list[str] = []
        total = 0
        for sentence in ordered:
            if total + len(sentence) + 1 > limit and out:
                break
            out.append(sentence)
            total += len(sentence) + 1
        text = " ".join(out)
        return text[:limit].rstrip()
