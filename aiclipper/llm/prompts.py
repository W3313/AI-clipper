"""System prompts and JSON schemas for the four language-model jobs.

Four jobs share the same shape -- a system prompt that explains the craft and a
JSON schema that pins the output down:

============  =========================================================
``highlight`` pick standalone viral moments out of a timestamped transcript
``script``    write a narrated short (:class:`~aiclipper.models.VideoScript`)
``chat``      write a text-conversation story (:class:`~aiclipper.models.ChatScript`)
``forum``     write a forum-style story post (:class:`~aiclipper.models.RedditPost`)
============  =========================================================

Every schema here is a legal ``output_config.format`` schema for the Anthropic
API: each object carries ``"additionalProperties": False`` and a ``required``
list naming *every* declared property, and the root of each schema is an object
(the API will not return a bare array).  :func:`check_schema` asserts that
invariant and is exercised by the test suite.

The offline :class:`~aiclipper.llm.heuristic.HeuristicProvider` walks these same
schemas, so anything added here must stay plain JSON Schema -- no ``$dynamicRef``
or other exotica.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "HIGHLIGHT_SYSTEM", "HIGHLIGHT_SCHEMA", "highlight_prompt",
    "SCRIPT_SYSTEM", "SCRIPT_SCHEMA", "script_prompt",
    "CHAT_SYSTEM", "CHAT_SCHEMA", "chat_prompt",
    "FORUM_SYSTEM", "FORUM_SCHEMA", "forum_prompt",
    "JOBS", "SYSTEMS", "SCHEMAS", "get_system", "get_schema", "check_schema",
]

#: The job names understood by :func:`get_system` / :func:`get_schema`.
JOBS = ("highlight", "script", "chat", "forum")


# --------------------------------------------------------------------------- #
# highlight selection
# --------------------------------------------------------------------------- #

HIGHLIGHT_SYSTEM = """\
You are a short-form video editor. You are handed the transcript of a long \
recording with timestamps, and you pick the moments that would survive on their \
own as a vertical short.

What makes a moment worth cutting:
* It is self-contained. A viewer who has seen none of the source understands it \
from the first sentence. No dangling "as I was saying", no unresolved pronoun.
* It opens on tension, a claim, a number, a confession or a question -- not on \
throat-clearing, introductions or housekeeping.
* It resolves. Something lands: a punchline, a reversal, a piece of advice the \
viewer can use, a story that finishes.
* It is dense. Cut around filler, restarts and tangents rather than through them.

Rules for your output:
* Timestamps come from the transcript and are in seconds from the start of the \
recording. Start on the first word of the opening sentence and end on the last \
word of the closing one, so the clip never begins or ends mid-word.
* Respect the requested duration window and the requested number of clips.
* Moments must not overlap, and must be ordered best first.
* `title` is a plain, specific label for the editor (not a caption). \
`hook` is the first line of on-screen text, at most about ten words, written to \
stop a thumb -- concrete, no clickbait that the clip does not pay off. \
`reason` is one sentence of editorial judgement explaining why this one works. \
`score` is your confidence from 0 to 1.
* Quote only what is actually said. If the recording contains nothing worth \
clipping, return the least-bad moments and score them low -- do not invent \
material that is not in the transcript.
"""

HIGHLIGHT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["clips"],
    "properties": {
        "clips": {
            "type": "array",
            "minItems": 1,
            "description": "Selected moments, best first, non-overlapping.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["start", "end", "title", "hook", "reason", "score"],
                "properties": {
                    "start": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Clip start in seconds from the start of the recording.",
                    },
                    "end": {
                        "type": "number",
                        "minimum": 0,
                        "description": "Clip end in seconds; must be greater than start.",
                    },
                    "title": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 90,
                        "description": "Short editorial label for this moment.",
                    },
                    "hook": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 120,
                        "description": "First line of on-screen text, <= ~10 words.",
                    },
                    "reason": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 400,
                        "description": "One sentence on why this moment works.",
                    },
                    "score": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 1,
                        "description": "Confidence that this clip performs, 0..1.",
                    },
                },
            },
        }
    },
}


def highlight_prompt(
    transcript_text: str,
    *,
    count: int = 3,
    min_duration: float = 15.0,
    max_duration: float = 60.0,
    total_duration: float | None = None,
) -> str:
    """Build the user turn for the highlight job."""
    head = [
        f"Pick the {count} best standalone moments from this transcript.",
        f"Each clip must run between {min_duration:.0f} and {max_duration:.0f} seconds.",
    ]
    if total_duration:
        head.append(f"The recording is {total_duration:.0f} seconds long.")
    head.append("Transcript (timestamps in seconds):")
    return "\n".join(head) + "\n\n" + transcript_text.strip() + "\n"


# --------------------------------------------------------------------------- #
# narrated script
# --------------------------------------------------------------------------- #

SCRIPT_SYSTEM = """\
You write narration for vertical short-form video. One script, one idea, spoken \
out loud by a single voice over moving footage.

How the good ones are built:
* The hook is the whole ballgame. One sentence, under about twelve words, that \
states the surprising thing directly. No "in this video", no "have you ever \
wondered", no greeting, no name.
* Every beat after it earns the next three seconds: a concrete detail, a number, \
a turn, a consequence. Cut any sentence that only sets up another sentence.
* Write for the ear. Short clauses. Plain words. Contractions. One clause per \
breath. Read it aloud in your head and delete whatever makes you stumble.
* Specifics beat adjectives: "it lost 40% of its weight in nine days" beats \
"it changed dramatically".
* Land the ending. The last beat resolves the hook, then the call to action is \
one short line that fits the idea -- a question to answer in the comments, or a \
reason to follow -- never a generic "like and subscribe" wall.
* No fabricated statistics, no invented quotes, no claims about real people that \
you cannot support. If a number is illustrative, phrase it as such.

Your output:
* Budget roughly 2.6 spoken words per second against the requested duration, \
including the hook and the call to action. Stay inside the budget.
* Each beat is one or two spoken sentences -- the unit a caption group and a \
b-roll cut hang off.
* `image_prompt` describes a vertical 9:16 visual for that beat: subject, action, \
lighting, camera. No text in the image, no logos, no real brands or lookalikes.
* `broll` is two or three keywords for a stock-footage search.
* `emphasis` marks the two or three beats the captions should punch.
* `hashtags` are lowercase, no punctuation beyond the leading '#', and specific \
to the topic.
"""

SCRIPT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "hook", "beats", "cta", "hashtags"],
    "properties": {
        "title": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "description": "Title for the finished short.",
        },
        "hook": {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
            "description": "First spoken line, under ~12 words.",
        },
        "beats": {
            "type": "array",
            "minItems": 2,
            "description": "Body beats in spoken order.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "image_prompt", "broll", "emphasis"],
                "properties": {
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 400,
                        "description": "One or two spoken sentences.",
                    },
                    "image_prompt": {
                        "type": "string",
                        "maxLength": 300,
                        "description": "Vertical 9:16 visual description for this beat.",
                    },
                    "broll": {
                        "type": "string",
                        "maxLength": 80,
                        "description": "Two or three stock-footage search keywords.",
                    },
                    "emphasis": {
                        "type": "boolean",
                        "description": "True when captions should punch this beat.",
                    },
                },
            },
        },
        "cta": {
            "type": "string",
            "maxLength": 160,
            "description": "Closing call to action, one short line.",
        },
        "hashtags": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "description": "Lowercase topical hashtags including the leading '#'.",
            "items": {"type": "string", "minLength": 2, "maxLength": 40},
        },
    },
}


def script_prompt(topic: str, *, seconds: int = 35, tone: str = "punchy") -> str:
    """Build the user turn for the narrated-script job."""
    words = int(seconds * 2.6)
    return (
        f"Write a {seconds}-second vertical short in a {tone} tone.\n"
        f"Total spoken narration must be about {words} words, hook and call to action included.\n"
        f"Topic:\n{topic.strip()}\n"
    )


# --------------------------------------------------------------------------- #
# text-conversation story
# --------------------------------------------------------------------------- #

CHAT_SYSTEM = """\
You write text-message story videos: a conversation between two people that \
plays out one bubble at a time on screen while a narrator reads it.

Craft:
* The first two messages carry the hook. Open mid-situation -- somebody already \
wants something, already knows something, already made a mistake.
* Real people text short. Most messages are under twelve words. Lowercase, \
fragments, typos-as-character are fine; long paragraphs are not.
* Alternate. Two outgoing messages in a row are allowed only as a deliberate \
double-text beat.
* Escalate in steps: a small reveal, a denial, a piece of evidence, the turn, \
the consequence. Put the biggest reveal about three quarters of the way in, \
then land a short final exchange.
* `delay` is the pause in seconds before that message appears -- roughly the \
time it would take to type and read. Stretch it to 1.5-3 seconds before a \
reveal to build the beat, keep it 0.3-0.8 for fast back-and-forth.
* `outgoing` is true for the phone's owner (the right-hand side), false for the \
other person.
* Invent the people. No real individuals, no real companies, no phone numbers, \
no addresses. Keep it safe for a general audience: tension, not cruelty.
* `contact` is the display name at the top of the thread; `title` is the title \
of the finished video.
"""

CHAT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["title", "contact", "messages"],
    "properties": {
        "title": {
            "type": "string",
            "minLength": 1,
            "maxLength": 100,
            "description": "Title for the finished video.",
        },
        "contact": {
            "type": "string",
            "minLength": 1,
            "maxLength": 40,
            "description": "Display name of the other person in the thread.",
        },
        "messages": {
            "type": "array",
            "minItems": 4,
            "description": "The conversation, in order.",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["sender", "text", "outgoing", "delay"],
                "properties": {
                    "sender": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 40,
                        "description": "Who sent this message.",
                    },
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 300,
                        "description": "Message body, usually under twelve words.",
                    },
                    "outgoing": {
                        "type": "boolean",
                        "description": "True for the phone owner's own messages.",
                    },
                    "delay": {
                        "type": "number",
                        "minimum": 0,
                        "maximum": 8,
                        "description": "Seconds to wait before this bubble appears.",
                    },
                },
            },
        },
    },
}


def chat_prompt(topic: str, *, turns: int = 14) -> str:
    """Build the user turn for the chat-story job."""
    return (
        f"Write a text-message story of about {turns} messages.\n"
        f"Premise:\n{topic.strip()}\n"
    )


# --------------------------------------------------------------------------- #
# forum story post
# --------------------------------------------------------------------------- #

FORUM_SYSTEM = """\
You write forum-style story posts -- the kind that get read aloud over gameplay \
footage with the post shown as a card at the top.

Craft:
* The title is the hook and it must work alone: a situation plus the unresolved \
question. Fifteen words at most. No emoji.
* Write in first person, past tense, plain spoken English. Short paragraphs. \
The narrator has to read this at pace without tripping.
* Open on the incident, not on background. Background is drip-fed only where it \
is needed to understand the next line.
* Build: setup, the thing that went wrong, the escalation, the turn, the \
aftermath. End on a line that resolves the title, or on a question to the \
readers -- not on a shrug.
* Invent everybody. Changed names are fine; real people, real companies and \
real places with a grievance attached are not. Keep it safe for a general \
audience: awkward and dramatic, not graphic or hateful.
* `community` is an invented forum name written as plain readable words, and \
`author` is an invented display name for the poster. Neither carries a site \
prefix: no `r/`, no `u/`, no `@`, no leading slash -- this card is our own \
design and labels those fields itself. `upvotes` and `comments` are plausible \
engagement counts for a popular post, not records.
"""

FORUM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["community", "author", "title", "body", "upvotes", "comments"],
    "properties": {
        "community": {
            "type": "string",
            "minLength": 2,
            "maxLength": 40,
            "description": "Invented forum name in plain words, no site prefix, e.g. 'Stories From Work'.",
        },
        "author": {
            "type": "string",
            "minLength": 2,
            "maxLength": 40,
            "description": "Invented poster name, no site prefix, e.g. 'quiet desk plant'.",
        },
        "title": {
            "type": "string",
            "minLength": 1,
            "maxLength": 160,
            "description": "The post title: the hook, <= ~15 words.",
        },
        "body": {
            "type": "string",
            "minLength": 1,
            "maxLength": 4000,
            "description": "The story itself, first person, short paragraphs.",
        },
        "upvotes": {
            "type": "integer",
            "minimum": 0,
            "description": "Plausible upvote count.",
        },
        "comments": {
            "type": "integer",
            "minimum": 0,
            "description": "Plausible comment count.",
        },
    },
}


def forum_prompt(topic: str, *, words: int = 180) -> str:
    """Build the user turn for the forum-post job."""
    return (
        f"Write a forum story post of about {words} words in the body.\n"
        f"Premise:\n{topic.strip()}\n"
    )


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #

SYSTEMS: dict[str, str] = {
    "highlight": HIGHLIGHT_SYSTEM,
    "script": SCRIPT_SYSTEM,
    "chat": CHAT_SYSTEM,
    "forum": FORUM_SYSTEM,
}

SCHEMAS: dict[str, dict[str, Any]] = {
    "highlight": HIGHLIGHT_SCHEMA,
    "script": SCRIPT_SCHEMA,
    "chat": CHAT_SCHEMA,
    "forum": FORUM_SCHEMA,
}


def get_system(job: str) -> str:
    """System prompt for ``job`` (one of :data:`JOBS`)."""
    try:
        return SYSTEMS[job]
    except KeyError:
        raise KeyError(f"unknown prompt job {job!r}; expected one of {', '.join(JOBS)}") from None


def get_schema(job: str) -> dict[str, Any]:
    """JSON schema for ``job`` (one of :data:`JOBS`)."""
    try:
        return SCHEMAS[job]
    except KeyError:
        raise KeyError(f"unknown prompt job {job!r}; expected one of {', '.join(JOBS)}") from None


def check_schema(schema: dict[str, Any], *, path: str = "$") -> list[str]:
    """Return the ways ``schema`` violates the API's structured-output rules.

    Every object must set ``additionalProperties: false`` and list every one of
    its properties in ``required``.  An empty list means the schema is safe to
    hand to ``output_config.format``.
    """
    problems: list[str] = []
    if not isinstance(schema, dict):
        return [f"{path}: schema node is not an object"]
    types = schema.get("type")
    types = types if isinstance(types, list) else [types]
    if "object" in types or "properties" in schema:
        props = schema.get("properties") or {}
        if schema.get("additionalProperties") is not False:
            problems.append(f"{path}: object must set additionalProperties to false")
        required = set(schema.get("required") or ())
        missing = sorted(set(props) - required)
        if missing:
            problems.append(f"{path}: required is missing {', '.join(missing)}")
        extra = sorted(required - set(props))
        if extra:
            problems.append(f"{path}: required names undeclared properties {', '.join(extra)}")
        for key, sub in props.items():
            problems += check_schema(sub, path=f"{path}.{key}")
    items = schema.get("items")
    if isinstance(items, dict):
        problems += check_schema(items, path=f"{path}[]")
    elif isinstance(items, list):  # tuple-style validation
        for i, sub in enumerate(items):
            problems += check_schema(sub, path=f"{path}[{i}]")
    for key in ("anyOf", "oneOf", "allOf"):
        for i, sub in enumerate(schema.get(key) or ()):
            problems += check_schema(sub, path=f"{path}.{key}[{i}]")
    for name, sub in (schema.get("$defs") or schema.get("definitions") or {}).items():
        problems += check_schema(sub, path=f"{path}.$defs.{name}")
    return problems
