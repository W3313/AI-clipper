"""Script generation: a topic becomes a narrated script, a chat story or a forum post.

Three generators and two parsers live here.  The generators
(:func:`write_script`, :func:`write_chat`, :func:`write_reddit`) ask the language
model layer for structured JSON and then *repair* the answer: word counts are
budgeted to the requested duration at :data:`WORDS_PER_SECOND`, chat turns are
forced to alternate and to carry sensible delays, and engagement counts are
derived deterministically from ``settings.seed``.  If the LLM layer is missing,
broken, or hands back nonsense, every generator quietly falls back to a built-in
template so a render never dies for want of a model.

The parsers (:func:`parse_chat`, :func:`parse_script`) let a user bring their own
script in plain text.  They are deliberately forgiving: they never raise, and an
empty string yields an empty -- but well formed -- object.

Nothing here imports a third-party package at module import time.
"""

from __future__ import annotations

import random
import re
import zlib
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .config import Settings, get_settings
from .models import ChatMessage, ChatScript, RedditPost, ScriptBeat, VideoScript

__all__ = [
    "WORDS_PER_SECOND",
    "budget_words",
    "estimate_seconds",
    "fit_script_to_budget",
    "write_script",
    "write_chat",
    "write_reddit",
    "parse_chat",
    "parse_script",
    "format_chat",
    "strip_site_prefix",
]

#: Spoken words per second used to budget a script against a target duration.
#: 2.6 w/s is a brisk but readable short-form narration pace.
WORDS_PER_SECOND = 2.6

#: Used when the caller passes an empty topic.
DEFAULT_TOPIC = "a small thing that turned out to matter more than it looked"

_MIN_BEAT_WORDS = 3
_MAX_BEATS = 14
#: A spoken word: letters (any alphabet, so accented and non-Latin narration is
#: counted rather than silently measured as zero words) or a number.
_WORD_RE = re.compile(r"[^\W\d_]+(?:['’\-][^\W\d_]+)*|\d+(?:[.,]\d+)*")
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]*")
_HASHTAG_RE = re.compile(r"(?<!\w)#([A-Za-z0-9][A-Za-z0-9_\-]*)")
_DELAY_RE = re.compile(r"\[\s*(\d+(?:\.\d+)?)\s*s?\s*\]", re.IGNORECASE)
_SENDER_RE = re.compile(r"^(?P<dir>[<>])?\s*(?P<name>[^:]{1,32}?)\s*:\s*(?P<text>.*)$")
_TITLE_RE = re.compile(r"^#\s+(?P<title>\S.*)$")
_CTA_RE = re.compile(r"^(?:cta|call to action|outro)\s*:\s*(?P<text>.*)$", re.IGNORECASE)
_HOOK_RE = re.compile(r"^(?:hook|open|opener)\s*:\s*(?P<text>.*)$", re.IGNORECASE)
_LABEL_RE = re.compile(r"^(?:title)\s*:\s*(?P<text>.*)$", re.IGNORECASE)
_OUTGOING_NAMES = frozenset({"me", "i", "myself", "you", "self", "op"})

#: Phrases that mean a model echoed our own instructions back at us.
_INSTRUCTION_MARKERS = (
    "vertical short", "spoken narration", "premise:", "topic:",
    "forum story post", "text-message story", "words in the body",
)

_STOPWORDS = frozenset(
    """
    a an and are as at be been but by can could did do does for from had has have he her him his how
    i if in into is it its just like me my no not of on or our out she so than that the their them then
    there these they this to too up was we were what when where which who why will with would you your
    about after all also am any because before being between both during each few more most other over
    same some such only own very s t don now
    """.split()
)

_TONE_HOOKS: dict[str, tuple[str, ...]] = {
    "punchy": (
        "Here is the part about {kw} nobody warns you about.",
        "Everything you think about {kw} is one step off.",
    ),
    "calm": (
        "There is a quieter way to think about {kw}.",
        "Most of what goes wrong with {kw} starts small.",
    ),
    "curious": (
        "Why does {kw} keep working when it clearly should not?",
        "Nobody can quite explain what {kw} does here.",
    ),
    "dramatic": (
        "This is the moment {kw} stopped being simple.",
        "Everyone missed the one detail that made {kw} matter.",
    ),
}

_BODY_TEMPLATES: tuple[str, ...] = (
    "It starts with {kw}, which looks ordinary until you measure it.",
    "The first time it happened everyone blamed something else entirely.",
    "Then the numbers moved, and the easy explanation stopped fitting.",
    "What actually changed was the order things happened in, not the effort.",
    "Once you see that, {kw} stops looking like luck.",
    "The people who get this right do one boring thing early.",
    "Everyone else tries to fix it at the end, when it costs ten times more.",
    "That single difference is the whole story.",
)

_PAD_TEMPLATES: tuple[str, ...] = (
    "Once you notice it in {kw} you cannot stop seeing it anywhere.",
    "That is the detail everybody skips right before it gets good.",
    "Here is what actually happens the second time around.",
    "The {kw} part is the part that decides it.",
    "Try it once and watch what changes.",
    "It is smaller than you think.",
    "Nobody mentions that part.",
    "That is the whole trick.",
    "Every single time.",
    "It works.",
)

_CHAT_TEMPLATE: tuple[str, ...] = (
    "are you awake? something happened with the {kw}",
    "it is the middle of the night. what did you do",
    "i did not do anything. the {kw} did",
    "that is not an answer",
    "okay. do not be angry",
    "why would i be angry",
    "because i already told them it was fine",
    "you told WHO it was fine",
    "everyone. it is in writing now",
    "please tell me you are joking",
    "i have a screenshot if you want it",
    "do not send it. i am coming over",
    "bring the spare key. mine is somewhere in the {kw}",
    "of course it is",
    "we can fix this before morning",
    "we have forty minutes, not a morning",
)

_CHAT_ESCALATION: tuple[str, ...] = (
    "wait. say that again",
    "i am reading it twice and it still says the same thing",
    "this is so much worse than you think",
    "how long have you known",
    "long enough to feel sick about it",
    "you should have called",
    "i am calling now",
    "do not do anything until i get there",
)

_POST_TEMPLATES: tuple[str, ...] = (
    "I did not think {kw} was going to be a problem until the second week.",
    "It started the way these things always start, with something small I let slide.",
    "By the time I said anything out loud it had already stopped being fixable quietly.",
    "I tried the reasonable version first. That went about as well as you would guess.",
    "Then somebody else noticed, and suddenly it was not only my problem any more.",
    "The part I still think about is how ordinary it all looked from the outside.",
    "I would do most of it the same way again. Not all of it.",
    "So was I wrong to handle it like that, or would you have done the same?",
)


# --------------------------------------------------------------------------- #
# small text helpers
# --------------------------------------------------------------------------- #

def _settings(settings: Settings | None) -> Settings:
    return settings if settings is not None else get_settings()


def _looks_like_instruction(text: str) -> bool:
    """True when a line is our own prompt scaffolding echoed back as content."""
    low = (text or "").lower()
    return any(marker in low for marker in _INSTRUCTION_MARKERS)


def _model_title(raw: Any) -> str:
    """A model-supplied title, or ``""`` when it is our own prompt echoed back.

    Titles get the same guard as hooks, beats and messages because a title is
    not merely displayed -- it names the output file.
    """
    title = " ".join(str(raw or "").split())
    return "" if _looks_like_instruction(title) else title


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _count_words(text: str) -> int:
    return len(_words(text))


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text or "") if s.strip()]


def _shorten(text: str, max_words: int) -> str:
    """Cut ``text`` down to at most ``max_words`` counted words, re-punctuated."""
    if max_words <= 0:
        return ""
    tokens = (text or "").split()
    while tokens and _count_words(" ".join(tokens)) > max_words:
        tokens.pop()
    out = " ".join(tokens).rstrip(" ,;:-–—")
    if out and out[-1] not in ".!?…":
        out += "."
    return out


def _titlecase(text: str, max_words: int = 8) -> str:
    tokens = _words(text)[:max_words]
    if not tokens:
        return ""
    out = [tokens[0].capitalize()]
    out += [t if t.lower() in _STOPWORDS and len(t) <= 3 else t.capitalize() for t in tokens[1:]]
    return " ".join(out)


def _keywords(topic: str, limit: int = 6) -> list[str]:
    seen: list[str] = []
    for token in _words(topic):
        low = token.lower()
        if low in _STOPWORDS or len(low) < 3 or low in seen:
            continue
        seen.append(low)
        if len(seen) >= limit:
            break
    return seen


def _keyword(topic: str) -> str:
    keys = _keywords(topic, limit=4)
    if not keys:
        return "this"
    return max(keys, key=len)


def _clean_topic(topic: str | None) -> str:
    text = (topic or "").strip()
    return text if text else DEFAULT_TOPIC


def _as_list(value: Any) -> list[Any]:
    """Only a real sequence is a list of items; a bare string is not.

    Iterating a string would hand us one beat (or hashtag, or message) per
    *character*, which is how a sloppy model payload used to turn into a script
    of single letters.
    """
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def _rng(*parts: Any, seed: int = 0) -> random.Random:
    """A deterministic RNG keyed on ``seed`` plus a stable hash of ``parts``."""
    blob = "␟".join(str(p) for p in parts).encode("utf-8", "replace")
    return random.Random((zlib.crc32(blob) ^ (int(seed) * 2654435761)) & 0xFFFFFFFF)


def _normalize_hashtag(raw: str) -> str:
    token = re.sub(r"[^0-9A-Za-z_]", "", str(raw or "")).lower()
    return f"#{token}" if token else ""


def _initials(name: str) -> str:
    tokens = _words(name)[:2]
    return "".join(t[0].upper() for t in tokens)


# --------------------------------------------------------------------------- #
# word budgeting
# --------------------------------------------------------------------------- #

def budget_words(seconds: float) -> int:
    """How many spoken words fit in ``seconds`` at :data:`WORDS_PER_SECOND`."""
    return max(4, int(round(max(0.0, float(seconds)) * WORDS_PER_SECOND)))


def estimate_seconds(source: Any) -> float:
    """Estimated spoken duration of a string, a script, a post or a line list."""
    return _count_words(_narration_of(source)) / WORDS_PER_SECOND


def _narration_of(source: Any) -> str:
    if source is None:
        return ""
    if isinstance(source, str):
        return source
    if isinstance(source, ChatScript):
        return " ".join(m.text for m in source.spoken_lines)
    narration = getattr(source, "narration", None)
    if isinstance(narration, str):
        return narration
    if isinstance(source, (list, tuple)):
        return " ".join(_narration_of(item) for item in source)
    return str(source)


def fit_script_to_budget(script: VideoScript, seconds: float, *, seed: int = 1234) -> VideoScript:
    """Return a copy of ``script`` whose narration lands on the ``seconds`` budget.

    Over-long scripts lose trailing sentences, then trailing beats, then words.
    Short scripts gain filler beats sized to the exact shortfall, so the result
    is within a word or two of ``seconds * WORDS_PER_SECOND``.
    """
    target = budget_words(seconds)
    out = VideoScript(
        title=script.title,
        hook=script.hook,
        beats=[replace(b) for b in script.beats],
        cta=script.cta,
        hashtags=list(script.hashtags),
    )
    keyword = _keyword(f"{out.title} {out.hook}")
    _shrink_script(out, target)
    _pad_script(out, target, keyword, seed=seed)
    return out


def _shrink_script(script: VideoScript, target: int) -> None:
    def total() -> int:
        return _count_words(script.narration)

    # 1. drop trailing sentences, as long as that does not undershoot the target
    progress = True
    while progress and total() > target:
        progress = False
        for beat in reversed(script.beats):
            sents = _sentences(beat.text)
            if len(sents) < 2:
                continue
            if total() - _count_words(sents[-1]) >= target:
                beat.text = " ".join(sents[:-1])
                progress = True
                break

    # 2. drop whole trailing beats, again only while it does not undershoot
    while len(script.beats) > 1 and total() > target:
        if total() - _count_words(script.beats[-1].text) < target:
            break
        script.beats.pop()

    # 3. trim word by word from the tail: beats first, then the cta, then the hook
    guard = 0
    while total() > target and guard < 400:
        guard += 1
        before = total()
        excess = before - target
        if script.beats:
            beat = script.beats[-1]
            keep = _count_words(beat.text) - excess
            if keep < _MIN_BEAT_WORDS:
                script.beats.pop()
                continue
            beat.text = _shorten(beat.text, keep)
            if not beat.text:
                script.beats.pop()
        elif script.cta:
            keep = _count_words(script.cta) - excess
            script.cta = _shorten(script.cta, keep) if keep >= 2 else ""
        elif script.hook:
            script.hook = _shorten(script.hook, max(_count_words(script.hook) - excess, 2))
        else:
            break
        if total() >= before:
            break


def _pad_script(script: VideoScript, target: int, keyword: str, *, seed: int) -> None:
    pad = list(_PAD_TEMPLATES)
    offset = _rng(script.hook, script.title, seed=seed).randrange(len(pad))
    pool = list(_BODY_TEMPLATES) + pad[offset:] + pad[:offset]
    deficit = target - _count_words(script.narration)
    # A long script needs more beats, not one monstrous closing beat: one spoken
    # line per filler phrase, with a ceiling that only an absurd target reaches.
    max_beats = max(_MAX_BEATS, target // 6)

    for phrase in _filler_phrases(deficit, pool, keyword):
        if len(script.beats) < max_beats and _count_words(phrase) >= 4:
            script.beats.append(ScriptBeat(text=phrase))
        elif script.beats:
            # Ceiling reached: fold the phrase into a short beat, but never one
            # that already ends with it (that reads as a stutter).
            room = [b for b in script.beats if not b.text.rstrip().endswith(phrase)] or script.beats
            target_beat = min(room, key=lambda b: _count_words(b.text))
            target_beat.text = f"{target_beat.text} {phrase}".strip()
        else:
            script.hook = f"{script.hook} {phrase}".strip()


def _filler_phrases(deficit: int, pool: Sequence[str], keyword: str) -> list[str]:
    """Plan filler phrases whose word counts sum to (almost exactly) ``deficit``.

    ``pool`` is walked in narrative order so padding reads in sequence.  A phrase
    is only reused once every other phrase that still fits has been spent, and
    never twice in a row -- the old picker answered "the longest phrase that
    fits" over and over, which padded a two-minute script with the same sentence
    six times.
    """
    phrases = [t.format(kw=keyword) for t in pool]
    sized = [(_count_words(t), t) for t in phrases if _count_words(t) >= 2]
    out: list[str] = []
    used: set[str] = set()
    previous = ""
    remaining = int(deficit)

    for _ in range(max(64, remaining)):  # every pick spends >= 2 words, so this always ends
        if remaining < 2:
            break
        choice = _next_filler(sized, remaining, used, previous)
        if choice is None and used:
            used = set()  # every fitting phrase is spent: start a fresh pass
            choice = _next_filler(sized, remaining, used, previous)
        if choice is None:
            break
        out.append(choice)
        used.add(choice)
        previous = choice
        remaining -= _count_words(choice)
    return out


def _next_filler(items: Sequence[tuple[int, str]], budget: int, used: set[str], previous: str) -> str | None:
    """First phrase in narrative order that fits, is unused, and is not a repeat."""
    for count, text in items:
        if count <= budget and text not in used and text != previous:
            return text
    return None


def _fit_text_to_words(
    text: str,
    target: int,
    keyword: str,
    *,
    pool: Sequence[str] = _PAD_TEMPLATES,
) -> str:
    """Trim or pad a free-text body (paragraphs preserved) to ``target`` words."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paragraphs:
        paragraphs = [""]

    def total() -> int:
        return _count_words("\n\n".join(paragraphs))

    # trim trailing sentences while that keeps us at or above the target
    progress = True
    while progress and total() > target:
        progress = False
        for i in range(len(paragraphs) - 1, -1, -1):
            sents = _sentences(paragraphs[i])
            if not sents or (len(sents) == 1 and len(paragraphs) == 1):
                continue
            if total() - _count_words(sents[-1]) >= target:
                paragraphs[i] = " ".join(sents[:-1])
                if not paragraphs[i] and len(paragraphs) > 1:
                    paragraphs.pop(i)
                progress = True
                break

    guard = 0
    while total() > target and guard < 100:
        guard += 1
        before = total()
        excess = before - target
        last = paragraphs[-1]
        keep = _count_words(last) - excess
        if keep < _MIN_BEAT_WORDS and len(paragraphs) > 1:
            paragraphs.pop()
            continue
        paragraphs[-1] = _shorten(last, max(keep, 1))
        if total() >= before:
            break

    for phrase in _filler_phrases(target - total(), pool, keyword):
        if _count_words(paragraphs[-1]) > 70:
            paragraphs.append(phrase)          # keep forum paragraphs readable
        else:
            paragraphs[-1] = f"{paragraphs[-1]} {phrase}".strip()

    return "\n\n".join(p for p in paragraphs if p)


# --------------------------------------------------------------------------- #
# language-model plumbing (every import is lazy, every failure is survivable)
# --------------------------------------------------------------------------- #

_FALLBACK_SCHEMAS: dict[str, dict[str, Any]] = {
    "script": {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "hook", "beats", "cta", "hashtags"],
        "properties": {
            "title": {"type": "string"},
            "hook": {"type": "string"},
            "beats": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["text", "image_prompt", "broll", "emphasis"],
                    "properties": {
                        "text": {"type": "string"},
                        "image_prompt": {"type": "string"},
                        "broll": {"type": "string"},
                        "emphasis": {"type": "boolean"},
                    },
                },
            },
            "cta": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}},
        },
    },
    "chat": {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "contact", "messages"],
        "properties": {
            "title": {"type": "string"},
            "contact": {"type": "string"},
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sender", "text", "outgoing", "delay"],
                    "properties": {
                        "sender": {"type": "string"},
                        "text": {"type": "string"},
                        "outgoing": {"type": "boolean"},
                        "delay": {"type": "number"},
                    },
                },
            },
        },
    },
    "forum": {
        "type": "object",
        "additionalProperties": False,
        "required": ["community", "author", "title", "body", "upvotes", "comments"],
        "properties": {
            "community": {"type": "string"},
            "author": {"type": "string"},
            "title": {"type": "string"},
            "body": {"type": "string"},
            "upvotes": {"type": "integer"},
            "comments": {"type": "integer"},
        },
    },
}


def _fallback_prompt(job: str, topic: str, **kwargs: Any) -> str:
    if job == "script":
        seconds = int(kwargs.get("seconds", 35))
        tone = str(kwargs.get("tone", "punchy"))
        return (
            f"Write a {seconds}-second vertical short in a {tone} tone, about "
            f"{budget_words(seconds)} spoken words in total.\nTopic:\n{topic}\n"
        )
    if job == "chat":
        return f"Write a text-message story of about {int(kwargs.get('turns', 14))} messages.\nPremise:\n{topic}\n"
    return f"Write a forum story post of about {int(kwargs.get('words', 180))} words.\nPremise:\n{topic}\n"


def _prompt_bundle(job: str, topic: str, **kwargs: Any) -> tuple[str, dict[str, Any], str]:
    """``(prompt, schema, system)`` from :mod:`aiclipper.llm.prompts`, or a local stand-in."""
    try:
        from .llm import prompts as P

        builder = {"script": P.script_prompt, "chat": P.chat_prompt, "forum": P.forum_prompt}[job]
        prompt, schema, system = builder(topic, **kwargs), P.get_schema(job), P.get_system(job)
    except Exception:
        prompt, schema, system = _fallback_prompt(job, topic, **kwargs), _FALLBACK_SCHEMAS[job], ""
    return _tag_subject(prompt, topic), schema, system


def _tag_subject(prompt: str, topic: str) -> str:
    """Mark which part of ``prompt`` is the subject rather than our instructions.

    The offline provider builds its content out of the tagged section alone, so
    a two-word topic can no longer be out-voted by the boilerplate around it.
    Any provider that does not know the marker just reads a labelled restatement
    of the topic, and if the llm layer is unavailable the prompt is unchanged.
    """
    try:
        from .llm.heuristic import with_subject

        return with_subject(prompt, topic)
    except Exception:
        return prompt


def _resolve_provider(provider: Any, settings: Settings) -> Any:
    if provider is not None:
        return provider
    try:
        from .llm import get_provider

        return get_provider(settings=settings)
    except Exception:
        return None


def _ask(job: str, topic: str, provider: Any, settings: Settings, **kwargs: Any) -> dict[str, Any] | None:
    """Ask the model for structured JSON; ``None`` means "use the template path"."""
    prov = _resolve_provider(provider, settings)
    if prov is None:
        return None
    prompt, schema, system = _prompt_bundle(job, topic, **kwargs)
    try:
        data = prov.complete_json(prompt, schema, system=system, max_tokens=settings.llm_max_tokens)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


# --------------------------------------------------------------------------- #
# narrated script
# --------------------------------------------------------------------------- #

def write_script(
    topic: str,
    *,
    seconds: int = 35,
    tone: str = "punchy",
    settings: Settings | None = None,
    provider: Any = None,
) -> VideoScript:
    """Generate a narrated short about ``topic``, budgeted to ``seconds``."""
    cfg = _settings(settings)
    subject = _clean_topic(topic)
    seconds = max(3, int(seconds))

    data = _ask("script", subject, provider, cfg, seconds=seconds, tone=tone)
    script = _script_from_data(data) if data else None
    if script is None:
        script = _template_script(subject, tone, cfg)

    # Title/cta first: the cta is spoken, so it has to count against the budget
    # rather than be bolted on afterwards and push the narration over it.
    _finish_copy(script, subject)
    script = fit_script_to_budget(script, seconds, seed=cfg.seed)
    _finish_beats(script, subject)
    return script


def _script_from_data(data: dict[str, Any]) -> VideoScript | None:
    hook = str(data.get("hook") or "").strip()
    if _looks_like_instruction(hook):
        hook = ""
    beats: list[ScriptBeat] = []
    for raw in _as_list(data.get("beats")):
        if isinstance(raw, str):
            text, image_prompt, broll, emphasis = raw, None, None, False
        elif isinstance(raw, dict):
            text = str(raw.get("text") or "")
            image_prompt = raw.get("image_prompt") or None
            broll = raw.get("broll") or None
            emphasis = bool(raw.get("emphasis"))
        else:
            continue
        text = text.strip()
        if not text or _looks_like_instruction(text):
            continue
        beats.append(
            ScriptBeat(
                text=text,
                image_prompt=str(image_prompt) if image_prompt else None,
                broll=str(broll) if broll else None,
                emphasis=emphasis,
            )
        )
    if not hook and beats:
        hook = beats.pop(0).text
    if not hook or not beats:
        return None

    hashtags = []
    for tag in _as_list(data.get("hashtags")):
        normalized = _normalize_hashtag(tag)
        if normalized and normalized not in hashtags:
            hashtags.append(normalized)

    script = VideoScript(
        title=_model_title(data.get("title")),
        hook=hook,
        beats=beats,
        cta=str(data.get("cta") or "").strip(),
        hashtags=hashtags,
    )
    return script if _count_words(script.narration) >= 6 else None


def _template_script(topic: str, tone: str, settings: Settings) -> VideoScript:
    keyword = _keyword(topic)
    rng = _rng(topic, tone, seed=settings.seed)
    hooks = _TONE_HOOKS.get((tone or "").strip().lower(), _TONE_HOOKS["punchy"])
    hook = rng.choice(hooks).format(kw=keyword)
    body = list(_BODY_TEMPLATES)
    beats = [
        ScriptBeat(
            text=line.format(kw=keyword),
            image_prompt=f"{_titlecase(topic, 6) or keyword}, vertical 9:16, natural light, shallow depth",
            broll=" ".join(_keywords(topic, 3)) or keyword,
            emphasis=(i % 3 == 1),
        )
        for i, line in enumerate(body)
    ]
    hashtags = [f"#{k}" for k in _keywords(topic, 3)] or ["#story"]
    return VideoScript(
        title=_titlecase(topic) or "Untitled Short",
        hook=hook,
        beats=beats,
        cta=f"Follow for more on {keyword}.",
        hashtags=hashtags,
    )


def _finish_copy(script: VideoScript, topic: str) -> None:
    """Fill in title, call to action and hashtags -- the spoken parts, pre-budget."""
    keyword = _keyword(topic)
    if not script.title:
        # The topic first: a rejected or missing title must still name the video
        # after what it is about, and the hook may itself be generated filler.
        script.title = _titlecase(topic) or _titlecase(script.hook) or "Untitled Short"
    if not script.cta:
        script.cta = f"Follow for more on {keyword}."
    if not script.hashtags:
        script.hashtags = [f"#{k}" for k in _keywords(topic, 3)] or ["#story"]


def _finish_beats(script: VideoScript, topic: str) -> None:
    """Give every beat -- including padding added while budgeting -- visual cues."""
    keyword = _keyword(topic)
    for i, beat in enumerate(script.beats):
        if not beat.broll:
            beat.broll = " ".join(_keywords(f"{topic} {beat.text}", 3)) or keyword
        if not beat.image_prompt:
            beat.image_prompt = f"{_titlecase(beat.text, 6)}, vertical 9:16, natural light, shallow depth"
        beat.emphasis = bool(beat.emphasis) or (i == len(script.beats) - 1 and len(script.beats) > 2)


# --------------------------------------------------------------------------- #
# chat story
# --------------------------------------------------------------------------- #

def write_chat(
    topic: str,
    *,
    turns: int = 14,
    settings: Settings | None = None,
    provider: Any = None,
) -> ChatScript:
    """Generate an alternating text-message story of exactly ``turns`` messages."""
    cfg = _settings(settings)
    subject = _clean_topic(topic)
    turns = max(2, int(turns))

    data = _ask("chat", subject, provider, cfg, turns=turns)
    script = _chat_from_data(data) if data else None
    if script is None or len(script.messages) < 2:
        script = _template_chat(subject, turns, cfg)

    _resize_chat(script, subject, turns, cfg)
    _style_chat(script, subject)
    return script


def _chat_from_data(data: dict[str, Any]) -> ChatScript | None:
    messages: list[ChatMessage] = []
    for raw in _as_list(data.get("messages")):
        if isinstance(raw, str):
            sender, text, outgoing = "", raw, None
        elif isinstance(raw, dict):
            sender = str(raw.get("sender") or "").strip()
            text = str(raw.get("text") or "")
            outgoing = raw.get("outgoing")
        else:
            continue
        text = " ".join(text.split())
        if not text or _looks_like_instruction(text):
            continue
        messages.append(
            ChatMessage(
                sender=sender or "Unknown",
                text=text,
                outgoing=bool(outgoing),
                delay=_as_float(raw.get("delay") if isinstance(raw, dict) else None, 0.35),
            )
        )
    if len(messages) < 2:
        return None
    contact = str(data.get("contact") or "").strip()
    if not contact:
        contact = next((m.sender for m in messages if not m.outgoing and m.sender != "Unknown"), "")
    return ChatScript(
        title=_model_title(data.get("title")),
        contact=contact or "Unknown",
        messages=messages,
    )


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if out == out and abs(out) != float("inf") else default


def _template_chat(topic: str, turns: int, settings: Settings) -> ChatScript:
    keyword = _keyword(topic)
    lines = [line.format(kw=keyword) for line in _CHAT_TEMPLATE][: max(turns, 2)]
    contact = _contact_name(topic, settings)
    messages = [ChatMessage(sender=contact, text=line) for line in lines]
    return ChatScript(title=_titlecase(topic) or "Untitled Thread", contact=contact, messages=messages)


def _contact_name(topic: str, settings: Settings) -> str:
    names = (
        "Alex", "Jordan", "Priya", "Sam", "Noor", "Rhys", "Mika", "Dani",
        "Theo", "Ines", "Kai", "Lena", "Omar", "Rosa", "Tariq", "Wren",
    )
    return _rng(topic, "contact", seed=settings.seed).choice(names)


def _resize_chat(script: ChatScript, topic: str, turns: int, settings: Settings) -> None:
    """Force the message count to ``turns``: trim the middle, extend the tail."""
    messages = script.messages
    if len(messages) > turns:
        head = messages[: min(2, turns)]
        tail = messages[len(messages) - (turns - len(head)):] if turns > len(head) else []
        script.messages = head + tail
        return

    keyword = _keyword(topic)
    pool = [line.format(kw=keyword) for line in _CHAT_ESCALATION]
    rng = _rng(topic, "escalate", seed=settings.seed)
    start = rng.randrange(len(pool)) if pool else 0
    i = 0
    while len(script.messages) < turns:
        script.messages.append(ChatMessage(sender="", text=pool[(start + i) % len(pool)]))
        i += 1


def _style_chat(script: ChatScript, topic: str) -> None:
    """Alternate senders, derive delays from message length, add typing beats."""
    messages = script.messages
    if not messages:
        script.contact = script.contact or "Unknown"
        return

    contact = script.contact.strip()
    if not contact or contact.lower() in _OUTGOING_NAMES or contact == "Unknown":
        contact = next(
            (m.sender for m in messages if m.sender and not m.outgoing and m.sender.lower() not in _OUTGOING_NAMES),
            "",
        )
    script.contact = contact or "Unknown"
    script.avatar_initials = script.avatar_initials or _initials(script.contact) or "?"
    if not script.title:
        script.title = _titlecase(topic) or "Untitled Thread"

    first_outgoing = bool(messages[0].outgoing)
    last = len(messages) - 1
    previous_words = 0
    previous_outgoing = None
    for i, message in enumerate(messages):
        outgoing = first_outgoing if i % 2 == 0 else not first_outgoing
        message.outgoing = outgoing
        message.sender = "Me" if outgoing else script.contact
        message.read_aloud = True

        if i == 0:
            message.delay = 0.4
        else:
            delay = 0.55 + 0.11 * previous_words
            if previous_outgoing is not None and previous_outgoing != outgoing:
                delay += 0.3
            message.delay = round(min(3.2, max(0.4, delay)), 2)

        words = _count_words(message.text)
        if _is_dramatic(message.text, i == last, words, late=i >= 0.65 * last):
            message.typing = round(min(1.4, max(0.4, 0.45 + 0.05 * words)), 2)
        else:
            message.typing = 0.0

        previous_words = words
        previous_outgoing = outgoing


def _is_dramatic(text: str, is_last: bool, words: int, *, late: bool = False) -> bool:
    """True for beats worth a short typing indicator before the bubble lands."""
    stripped = text.strip()
    if not stripped:
        return False
    if is_last or words >= 12:
        return True
    if any(mark in stripped for mark in ("?", "!", "...", "…")):
        return True
    if any(len(token) > 2 and token.isupper() for token in stripped.split()):
        return True
    return late and words >= 6


# --------------------------------------------------------------------------- #
# forum post
# --------------------------------------------------------------------------- #

def write_reddit(
    topic: str,
    *,
    words: int = 180,
    settings: Settings | None = None,
    provider: Any = None,
) -> RedditPost:
    """Generate a first-person forum story of about ``words`` words."""
    cfg = _settings(settings)
    subject = _clean_topic(topic)
    words = max(20, int(words))

    data = _ask("forum", subject, provider, cfg, words=words)
    post = _post_from_data(data) if data else None
    if post is None:
        post = _template_post(subject, cfg)

    keyword = _keyword(subject)
    post.body = _fit_text_to_words(
        post.body, words, keyword, pool=_POST_TEMPLATES + _PAD_TEMPLATES
    )
    if not post.title:
        post.title = _titlecase(subject, 12) or "Something happened and I still think about it"
    post.community = _forum_name(post.community, subject, cfg, kind="community")
    post.author = _forum_name(post.author, subject, cfg, kind="author")
    if post.author.casefold() == post.community.casefold():
        # a model that reuses one slug for both would give us "X / by X"
        post.author = _forum_name("", subject, cfg, kind="author")
    post.upvotes, post.comments = engagement(subject, cfg.seed)
    post.theme = post.theme or "dark"
    return post


def _post_from_data(data: dict[str, Any]) -> RedditPost | None:
    title = _model_title(data.get("title"))
    body = str(data.get("body") or "").strip()
    if not body or _count_words(body) < 10:
        return None
    return RedditPost(
        community=str(data.get("community") or "").strip(),
        author=str(data.get("author") or "").strip(),
        title=title,
        body=body,
        theme="dark",
    )


def _template_post(topic: str, settings: Settings) -> RedditPost:
    keyword = _keyword(topic)
    body = "\n\n".join(
        [
            " ".join(line.format(kw=keyword) for line in _POST_TEMPLATES[:3]),
            " ".join(line.format(kw=keyword) for line in _POST_TEMPLATES[3:6]),
            " ".join(line.format(kw=keyword) for line in _POST_TEMPLATES[6:]),
        ]
    )
    return RedditPost(
        community=_forum_name("", topic, settings, kind="community"),
        author=_forum_name("", topic, settings, kind="author"),
        title=_titlecase(topic, 12) or "I did not expect this to get out of hand",
        body=body,
        theme="dark",
    )


#: A borrowed ``r/``/``u/`` handle prefix from one specific real forum.  The card
#: is our own design and labels its own fields, so we never *write* one -- and a
#: model or a user that supplies one has it stripped rather than rejected.
_SITE_PREFIX_RE = re.compile(r"^/?[ru]/", re.IGNORECASE)

#: Invented community names and author handles -- ours, in plain words.
_COMMUNITY_NAMES = (
    "Stories From Work", "Quiet Drama", "Told You So",
    "The Small Print", "Late Night Tales", "This Took A Turn",
)
_AUTHOR_NAMES = (
    "quiet desk plant", "box of cables", "third floor window",
    "no longer on the rota", "spare room tenant",
)


def strip_site_prefix(name: str) -> str:
    """``"r/quietdrama"`` -> ``"quietdrama"``; anything else is passed through."""
    return _SITE_PREFIX_RE.sub("", (name or "").strip(), count=1).strip()


def _forum_name(current: str, topic: str, settings: Settings, *, kind: str) -> str:
    """A plain, unprefixed community name or author name for the story card."""
    name = strip_site_prefix(current)
    if name:
        cleaned = re.sub(r"\s+", " ", re.sub(r"[^0-9A-Za-z _'&-]+", " ", name)).strip(" -_")[:32].strip()
        if cleaned and kind == "community":
            # a community reads as a label, so give it initial capitals
            cleaned = " ".join(word[:1].upper() + word[1:] for word in cleaned.split())
        if cleaned:
            return cleaned
    rng = _rng(topic, kind, seed=settings.seed)
    return rng.choice(_COMMUNITY_NAMES if kind == "community" else _AUTHOR_NAMES)


def engagement(topic: str, seed: int) -> tuple[int, int]:
    """Plausible ``(upvotes, comments)`` derived deterministically from ``seed``."""
    rng = _rng(topic, "engagement", seed=seed)
    upvotes = rng.randrange(1200, 74000)
    if upvotes > 10000:
        upvotes -= upvotes % 100
    elif upvotes > 2000:
        upvotes -= upvotes % 10
    comments = max(12, int(upvotes / rng.uniform(14.0, 48.0)))
    return upvotes, comments


# --------------------------------------------------------------------------- #
# parsing: user-supplied chat
# --------------------------------------------------------------------------- #

def parse_chat(text: str) -> ChatScript:
    """Parse a plain-text conversation into a :class:`ChatScript`.

    Recognised forms::

        # My Title              -> the video title
        Alex: hey               -> an incoming message from Alex
        me: hey                 -> an outgoing message ("me", "I", "self", ...)
        > Sam: hey              -> outgoing regardless of the name
        < Sam: hey              -> incoming regardless of the name
        Alex: [3s] hey          -> 3 second delay before this bubble
        [2s]                    -> delay applied to the next message
        (a line with no colon)  -> appended to the previous message
        (a blank line)          -> ends the current message

    Never raises; an empty string gives an empty script.
    """
    script = ChatScript()
    raw = (text or "").strip()
    if not raw:
        return script

    lines = raw.splitlines()
    if lines:
        match = _TITLE_RE.match(lines[0].strip())
        if match:
            script.title = match.group("title").strip()
            lines = lines[1:]

    current: ChatMessage | None = None
    pending_delay: float | None = None

    for line in lines:
        stripped = line.strip()
        if not stripped:
            current = None
            continue

        stripped, inline_delay = _extract_delay(stripped)
        if not stripped:
            pending_delay = inline_delay if inline_delay is not None else pending_delay
            current = None
            continue

        sender, body, direction = _split_sender(stripped)
        if sender is None:
            if current is not None:
                current.text = f"{current.text}\n{stripped}".strip()
            else:
                current = ChatMessage(sender="Unknown", text=stripped)
                script.messages.append(current)
                current.delay = _take(pending_delay, inline_delay, 0.35)
                pending_delay = None
            continue

        outgoing = direction if direction is not None else sender.lower() in _OUTGOING_NAMES
        current = ChatMessage(sender=sender, text=body.strip(), outgoing=outgoing)
        current.delay = _take(pending_delay, inline_delay, 0.35)
        pending_delay = None
        script.messages.append(current)

    script.messages = [m for m in script.messages if m.text.strip()]
    contact = next(
        (m.sender for m in script.messages if not m.outgoing and m.sender and m.sender != "Unknown"), ""
    )
    script.contact = contact or "Unknown"
    script.avatar_initials = _initials(script.contact) or "?"
    if not script.title and script.messages:
        script.title = _titlecase(script.messages[0].text) or "Untitled Thread"
    return script


def _take(pending: float | None, inline: float | None, default: float) -> float:
    for value in (inline, pending):
        if value is not None:
            return round(float(value), 2)
    return default


def _extract_delay(line: str) -> tuple[str, float | None]:
    found: list[float] = []

    def _grab(match: re.Match[str]) -> str:
        found.append(float(match.group(1)))
        return " "

    cleaned = _DELAY_RE.sub(_grab, line).strip()
    return " ".join(cleaned.split()), (found[0] if found else None)


def _split_sender(line: str) -> tuple[str | None, str, bool | None]:
    """``("Alex", "hey", None)`` for a sender line, ``(None, line, None)`` otherwise."""
    match = _SENDER_RE.match(line)
    if not match:
        return None, line, None
    name = match.group("name").strip()
    if not name or not any(ch.isalnum() for ch in name):
        return None, line, None
    if "/" in name or len(name) > 32 or len(name.split()) > 4:
        return None, line, None
    if name.lower() in {"http", "https", "ftp", "note", "www"}:
        return None, line, None
    marker = match.group("dir")
    direction = True if marker == ">" else (False if marker == "<" else None)
    return name, match.group("text"), direction


def format_chat(script: ChatScript) -> str:
    """Render a :class:`ChatScript` back into the :func:`parse_chat` text form."""
    out: list[str] = []
    if script.title:
        out.append(f"# {script.title}")
        out.append("")
    for message in script.messages:
        prefix = "> " if message.outgoing else ""
        delay = f"[{round(message.delay, 2):g}s] " if message.delay else ""
        out.append(f"{prefix}{message.sender}: {delay}{message.text}")
        out.append("")
    return "\n".join(out).strip() + "\n"


# --------------------------------------------------------------------------- #
# parsing: user-supplied narration
# --------------------------------------------------------------------------- #

def parse_script(text: str) -> VideoScript:
    """Parse free text (or one line per beat) into a :class:`VideoScript`.

    The first unit becomes the hook, a trailing ``CTA:`` line becomes the call to
    action, ``#hashtags`` are collected and removed, a leading ``# Title`` line
    sets the title, and ``*emphasis*`` markers set :attr:`ScriptBeat.emphasis`.
    A single block of prose is split into sentences.  Never raises.
    """
    script = VideoScript()
    raw = (text or "").strip()
    if not raw:
        return script

    lines = raw.splitlines()
    if lines:
        match = _TITLE_RE.match(lines[0].strip())
        if match:
            script.title = match.group("title").strip()
            lines = lines[1:]

    body = "\n".join(lines)
    for tag in _HASHTAG_RE.findall(body):
        normalized = _normalize_hashtag(tag)
        if normalized and normalized not in script.hashtags:
            script.hashtags.append(normalized)
    body = _HASHTAG_RE.sub(" ", body)

    units: list[str] = []
    for block in re.split(r"\n\s*\n", body):
        block_lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not block_lines:
            continue
        if len(block_lines) > 1:
            units.extend(block_lines)
        else:
            units.append(block_lines[0])

    cleaned: list[str] = []
    for unit in units:
        label = _LABEL_RE.match(unit)
        if label:
            script.title = script.title or label.group("text").strip()
            continue
        cta = _CTA_RE.match(unit)
        if cta:
            script.cta = cta.group("text").strip()
            continue
        hook = _HOOK_RE.match(unit)
        if hook:
            unit = hook.group("text").strip()
        unit = " ".join(unit.split())
        if unit:
            cleaned.append(unit)

    if len(cleaned) == 1:
        cleaned = _sentences(cleaned[0]) or cleaned

    if not cleaned:
        if not script.title and script.cta:
            script.title = _titlecase(script.cta)
        return script

    script.hook = _strip_emphasis(cleaned[0])[0]
    for unit in cleaned[1:]:
        text_value, emphasis = _strip_emphasis(unit)
        if text_value:
            script.beats.append(ScriptBeat(text=text_value, emphasis=emphasis))
    if not script.title:
        script.title = _titlecase(script.hook) or "Untitled Short"
    return script


def _strip_emphasis(unit: str) -> tuple[str, bool]:
    text = unit.strip()
    emphasis = False
    match = re.match(r"^\*{1,2}(?P<inner>.+?)\*{1,2}$", text)
    if match:
        text = match.group("inner").strip()
        emphasis = True
    if text.endswith("!"):
        emphasis = True
    return text, emphasis
