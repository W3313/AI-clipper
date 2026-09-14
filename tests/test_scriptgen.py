"""Tests for :mod:`aiclipper.scriptgen`.

Everything here runs offline: the generators are driven either by the built-in
heuristic provider (via ``get_provider`` with ``AICLIP_OFFLINE=1``, set by the
autouse fixture in ``conftest.py``) or by the fake providers defined below.
"""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any

import pytest

from aiclipper import scriptgen as sg
from aiclipper.config import Settings, get_settings
from aiclipper.models import ChatMessage, ChatScript, RedditPost, ScriptBeat, VideoScript

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def seeded(seed: int = 1234) -> Settings:
    return replace(get_settings(), seed=seed)


#: One specific real forum's handle grammar -- hard rule 6 says the card is our
#: own design, so nothing we generate may carry it.
SITE_PREFIX = re.compile(r"^/?[ru]/", re.IGNORECASE)


def _has_site_prefix(name: str) -> bool:
    return bool(SITE_PREFIX.match(name or ""))


class FakeProvider:
    """Stands in for an LLM: returns a canned payload or raises."""

    name = "fake"

    def __init__(self, payload: Any = None, *, error: Exception | None = None) -> None:
        self.payload = payload
        self.error = error
        self.calls: list[tuple[str, dict, str]] = []

    def available(self) -> bool:
        return True

    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None) -> str:
        return ""

    def complete_json(self, prompt, schema, *, system: str = "", max_tokens: int | None = None):
        self.calls.append((prompt, schema, system))
        if self.error is not None:
            raise self.error
        return self.payload


GOOD_SCRIPT_PAYLOAD = {
    "title": "Why Bread Rises",
    "hook": "Your loaf is not rising because of the yeast.",
    "beats": [
        {"text": "Yeast makes gas, but gluten decides whether the gas stays.", "image_prompt": "dough",
         "broll": "dough closeup", "emphasis": True},
        {"text": "Knead until the sheet of dough goes translucent.", "image_prompt": "hands",
         "broll": "kneading hands", "emphasis": False},
        {"text": "Then let the oven do the last third of the work.", "image_prompt": "oven",
         "broll": "oven glow", "emphasis": False},
    ],
    "cta": "Follow for more kitchen physics.",
    "hashtags": ["bread", "#Baking", "!!"],
}

GOOD_CHAT_PAYLOAD = {
    "title": "The Spare Key",
    "contact": "Alex",
    "messages": [
        {"sender": "Alex", "text": "are you home", "outgoing": False, "delay": 0.4},
        {"sender": "Me", "text": "no why", "outgoing": True, "delay": 0.4},
        {
            "sender": "Alex",
            "text": "because there is a van in your driveway and two people carrying your sofa out of it",
            "outgoing": False,
            "delay": 0.4,
        },
        {"sender": "Me", "text": "call the police", "outgoing": True, "delay": 0.4},
    ],
}

GOOD_FORUM_PAYLOAD = {
    "community": "r/quietdrama",
    "author": "u/spareroomtenant",
    "title": "My landlord changed the locks while I was at work",
    "body": (
        "I got home on a Tuesday and my key did not turn. I stood there for a full minute "
        "thinking I had the wrong door. Then I saw the new brass plate around the lock and "
        "realised somebody had been in. I called him and he told me it was routine maintenance."
    ),
    "upvotes": 12,
    "comments": 3,
}

RICH_CHAT_TEXT = """\
# The Night Shift

Alex: hey are you up?
i cannot sleep

> me: [3s] what happened

[2s]
me: seriously, what happened

< Alex: i think i left the door open
Alex: [1.5s] and the dog is gone
this is not a joke
"""

SCRIPT_TEXT = """\
# Bread Rising

Here is why your bread never rises.

Yeast eats sugar and burps gas.
*The gluten net is what traps it.*
Steam does the rest in the oven!

CTA: follow for more kitchen physics
#bread #Baking #science
"""


# --------------------------------------------------------------------------- #
# constants and budgeting maths
# --------------------------------------------------------------------------- #


def test_words_per_second_constant_is_exposed():
    assert sg.WORDS_PER_SECOND == 2.6
    assert sg.budget_words(35) == 91
    assert sg.budget_words(10) == 26
    assert sg.budget_words(0) == 4  # never zero: a script always says something


def test_estimate_seconds_round_trips_the_budget():
    for seconds in (8, 15, 35, 60):
        words = sg.budget_words(seconds)
        text = " ".join(["word"] * words)
        assert sg.estimate_seconds(text) == pytest.approx(seconds, abs=0.5)


def test_estimate_seconds_accepts_models():
    script = VideoScript(hook="one two three", beats=[ScriptBeat("four five six")], cta="seven eight")
    assert sg.estimate_seconds(script) == pytest.approx(8 / 2.6, abs=1e-6)

    chat = ChatScript(messages=[ChatMessage("A", "one two"), ChatMessage("B", "three four five")])
    assert sg.estimate_seconds(chat) == pytest.approx(5 / 2.6, abs=1e-6)

    post = RedditPost(title="one two", body="three four five six")
    assert sg.estimate_seconds(post) == pytest.approx(6 / 2.6, abs=1e-6)


# --------------------------------------------------------------------------- #
# parse_chat
# --------------------------------------------------------------------------- #


def test_parse_chat_rich_sample_covers_every_syntax_feature():
    script = sg.parse_chat(RICH_CHAT_TEXT)

    assert script.title == "The Night Shift"
    assert script.contact == "Alex"
    assert script.avatar_initials == "A"
    assert len(script.messages) == 5

    first, second, third, fourth, fifth = script.messages

    # plain "Name: text" plus a continuation line with no colon
    assert first.sender == "Alex"
    assert first.outgoing is False
    assert first.text == "hey are you up?\ni cannot sleep"
    assert first.delay == pytest.approx(0.35)

    # "> me: [3s] ..." -> outgoing with an inline delay
    assert second.sender == "me"
    assert second.outgoing is True
    assert second.text == "what happened"
    assert second.delay == pytest.approx(3.0)

    # a standalone "[2s]" line sets the delay of the next message
    assert third.outgoing is True  # bare "me:" is outgoing too
    assert third.text == "seriously, what happened"
    assert third.delay == pytest.approx(2.0)

    # "< Name:" forces incoming even for a name that is not the contact
    assert fourth.sender == "Alex"
    assert fourth.outgoing is False

    assert fifth.delay == pytest.approx(1.5)
    assert fifth.text == "and the dog is gone\nthis is not a joke"


def test_parse_chat_round_trips_through_format_chat():
    original = sg.parse_chat(RICH_CHAT_TEXT)
    again = sg.parse_chat(sg.format_chat(original))

    assert again.title == original.title
    assert again.contact == original.contact
    shape = lambda s: [(m.sender, m.text, m.outgoing, round(m.delay, 2)) for m in s.messages]  # noqa: E731
    assert shape(again) == shape(original)


def test_parse_chat_is_forgiving_about_odd_lines():
    script = sg.parse_chat(
        "no sender here\n"
        "https://example.com/thing\n"
        "Sam: fine\n"
    )
    assert len(script.messages) == 2
    # the URL has a colon but is not a sender, so it joins the first message
    assert script.messages[0].sender == "Unknown"
    assert "https://example.com/thing" in script.messages[0].text
    assert script.messages[1].sender == "Sam"


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t\n"])
def test_parse_chat_handles_empty_input(text):
    script = sg.parse_chat(text)
    assert isinstance(script, ChatScript)
    assert script.messages == []
    assert script.title == ""


# --------------------------------------------------------------------------- #
# parse_script
# --------------------------------------------------------------------------- #


def test_parse_script_extracts_title_hook_beats_cta_and_hashtags():
    script = sg.parse_script(SCRIPT_TEXT)

    assert script.title == "Bread Rising"
    assert script.hook == "Here is why your bread never rises."
    assert [b.text for b in script.beats] == [
        "Yeast eats sugar and burps gas.",
        "The gluten net is what traps it.",
        "Steam does the rest in the oven!",
    ]
    assert script.beats[0].emphasis is False
    assert script.beats[1].emphasis is True  # *starred*
    assert script.beats[2].emphasis is True  # ends with "!"
    assert script.cta == "follow for more kitchen physics"
    assert script.hashtags == ["#bread", "#baking", "#science"]
    # hashtags are stripped out of the spoken narration
    assert "#" not in script.narration


def test_parse_script_splits_a_single_block_into_sentences():
    script = sg.parse_script("This is the hook. Second sentence lands here. Third one closes it.")
    assert script.hook == "This is the hook."
    assert [b.text for b in script.beats] == ["Second sentence lands here.", "Third one closes it."]
    assert script.title  # derived from the hook


def test_parse_script_one_line_per_beat():
    script = sg.parse_script("Hook: the first line\nsecond line\nthird line\nCTA: subscribe")
    assert script.hook == "the first line"
    assert [b.text for b in script.beats] == ["second line", "third line"]
    assert script.cta == "subscribe"


@pytest.mark.parametrize("text", ["", "   ", "\n \n"])
def test_parse_script_handles_empty_input(text):
    script = sg.parse_script(text)
    assert isinstance(script, VideoScript)
    assert script.hook == ""
    assert script.beats == []
    assert script.lines == []


# --------------------------------------------------------------------------- #
# word budgeting on generated scripts
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("seconds", [10, 18, 25, 35, 48, 60])
def test_write_script_hits_the_duration_budget(seconds):
    script = sg.write_script(
        "why sourdough bread rises so much in a very hot oven",
        seconds=seconds,
        settings=seeded(11),
    )
    assert isinstance(script, VideoScript)
    assert script.hook and script.beats and script.cta and script.hashtags
    tolerance = max(1.0, 0.1 * seconds)
    assert abs(sg.estimate_seconds(script) - seconds) <= tolerance
    for beat in script.beats:
        assert beat.text.strip()
        assert beat.broll and beat.image_prompt


def test_fit_script_to_budget_trims_an_overlong_script():
    long_beats = [ScriptBeat(f"Beat number {i} says something quite long and unnecessary here.") for i in range(12)]
    script = VideoScript(hook="A hook that is fairly wordy on its own.", beats=long_beats, cta="Follow now.")
    assert sg.estimate_seconds(script) > 40

    fitted = sg.fit_script_to_budget(script, 15)
    target = sg.budget_words(15)
    assert abs(len(fitted.narration.split()) - target) <= 3
    assert fitted.hook == script.hook  # the hook survives trimming
    assert len(fitted.beats) < len(script.beats)
    assert len(script.beats) == 12  # the input is not mutated


def test_fit_script_to_budget_pads_a_short_script():
    script = VideoScript(title="Bread", hook="Bread.", beats=[ScriptBeat("Yeast.")], cta="Follow.")
    fitted = sg.fit_script_to_budget(script, 30)
    target = sg.budget_words(30)
    assert abs(len(fitted.narration.split()) - target) <= 3
    assert len(fitted.beats) > 1
    assert all(b.text.strip() for b in fitted.beats)


def test_fit_script_to_budget_is_deterministic_for_a_seed():
    script = VideoScript(hook="Short hook.", beats=[ScriptBeat("Tiny beat.")], cta="Bye.")
    a = sg.fit_script_to_budget(script, 25, seed=5)
    b = sg.fit_script_to_budget(script, 25, seed=5)
    c = sg.fit_script_to_budget(script, 25, seed=6)
    assert a.narration == b.narration
    assert len(c.narration.split()) == len(a.narration.split())


# --------------------------------------------------------------------------- #
# write_chat
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("turns", [2, 4, 7, 14, 21])
def test_write_chat_produces_alternating_turns(turns):
    script = sg.write_chat("a roommate keeps eating my food and denies it", turns=turns, settings=seeded(3))

    assert isinstance(script, ChatScript)
    assert len(script.messages) == turns
    assert script.title and script.contact and script.avatar_initials

    senders = {m.sender for m in script.messages}
    assert len(senders) == min(2, turns)
    for previous, current in zip(script.messages, script.messages[1:], strict=False):
        assert previous.outgoing != current.outgoing
        assert previous.sender != current.sender
    for message in script.messages:
        assert message.text.strip()
        assert message.sender == ("Me" if message.outgoing else script.contact)
        assert 0.4 <= message.delay <= 3.2
        assert 0.0 <= message.typing <= 1.4


def test_write_chat_hook_and_escalation_markers():
    script = sg.write_chat("my sister sold my car while I was abroad", turns=12, settings=seeded(3))
    # the hook lives in the first two messages
    assert all(m.text.strip() for m in script.messages[:2])
    # dramatic beats get a typing indicator, and the last message always does
    assert script.messages[-1].typing > 0.0
    assert sum(1 for m in script.messages if m.typing > 0.0) >= 1


def test_write_chat_delay_grows_after_a_long_message():
    payload = {
        "title": "t",
        "contact": "Alex",
        "messages": [
            {"sender": "Alex", "text": "hi", "outgoing": False, "delay": 0.4},
            {"sender": "Me", "text": "hey", "outgoing": True, "delay": 0.4},
            {
                "sender": "Alex",
                "text": "so the thing is that I have been sitting on this for about three weeks now and I cannot keep it in",
                "outgoing": False,
                "delay": 0.4,
            },
            {"sender": "Me", "text": "go on", "outgoing": True, "delay": 0.4},
            {"sender": "Alex", "text": "ok", "outgoing": False, "delay": 0.4},
        ],
    }
    script = sg.write_chat("secret", turns=5, provider=FakeProvider(payload), settings=seeded(3))
    after_short = script.messages[2].delay   # follows "hey"
    after_long = script.messages[3].delay    # follows the long confession
    after_tiny = script.messages[4].delay    # follows "go on"
    assert after_long > after_short
    assert after_long > after_tiny


def test_write_chat_trims_an_overlong_conversation_keeping_hook_and_ending():
    script = sg.write_chat("x", turns=4, provider=FakeProvider(GOOD_CHAT_PAYLOAD), settings=seeded(3))
    assert len(script.messages) == 4
    assert script.messages[0].text == "are you home"
    assert script.messages[-1].text == "call the police"


# --------------------------------------------------------------------------- #
# write_reddit
# --------------------------------------------------------------------------- #


def test_write_reddit_is_deterministic_under_a_fixed_seed():
    topic = "my neighbour towed my car from my own driveway"
    first = sg.write_reddit(topic, words=120, settings=seeded(42))
    second = sg.write_reddit(topic, words=120, settings=seeded(42))
    assert first == second

    other = sg.write_reddit(topic, words=120, settings=seeded(43))
    assert (other.upvotes, other.comments) != (first.upvotes, first.comments)


def test_write_reddit_shape_and_body_budget():
    post = sg.write_reddit("my neighbour towed my car", words=140, settings=seeded(42))
    assert isinstance(post, RedditPost)
    # our own card labelling: a plain community name and a plain author, never
    # another forum's "r/" / "u/" handle grammar
    assert not _has_site_prefix(post.community) and len(post.community) > 2
    assert not _has_site_prefix(post.author) and len(post.author) > 2
    assert post.title.strip()
    assert post.upvotes > 0 and post.comments > 0
    assert post.comments < post.upvotes
    assert abs(len(post.body.split()) - 140) <= 14


def test_write_reddit_keeps_model_prose_and_normalises_handles():
    payload = dict(GOOD_FORUM_PAYLOAD, community="QuietDrama", author="spare room tenant")
    post = sg.write_reddit("locks", words=60, provider=FakeProvider(payload), settings=seeded(9))
    assert "my key did not turn" in post.body
    assert post.title == payload["title"]
    assert post.community == "QuietDrama"
    assert post.author == "spare room tenant"
    # engagement counts are derived from the seed, not taken from the model
    assert post.upvotes != payload["upvotes"]


def test_write_reddit_strips_a_site_prefix_a_model_supplied():
    """The payload arrives as "r/quietdrama" / "u/spareroomtenant"; we keep the
    names and drop one specific real forum's handle grammar."""
    post = sg.write_reddit("locks", words=60, provider=FakeProvider(GOOD_FORUM_PAYLOAD),
                           settings=seeded(9))
    assert not _has_site_prefix(post.community) and not _has_site_prefix(post.author)
    assert post.community.casefold().endswith("quietdrama")
    assert post.author == "spareroomtenant"


@pytest.mark.parametrize(
    "supplied, expected",
    [
        ("r/quietdrama", "quietdrama"),
        ("/r/QuietDrama", "QuietDrama"),
        ("u/spareroomtenant", "spareroomtenant"),
        ("/U/Someone", "Someone"),
        ("Quiet Drama", "Quiet Drama"),
        ("", ""),
        ("  ", ""),
    ],
)
def test_strip_site_prefix(supplied, expected):
    assert sg.strip_site_prefix(supplied) == expected


def test_write_reddit_never_reuses_one_name_for_community_and_author():
    """The offline provider slugs both fields from the same keyword; the card
    must not read "Towed / by towed"."""
    payload = dict(GOOD_FORUM_PAYLOAD, community="r/samestem", author="u/samestem")
    post = sg.write_reddit("a tow truck", words=40, provider=FakeProvider(payload), settings=seeded(3))
    assert post.community.casefold() != post.author.casefold()
    assert not _has_site_prefix(post.author)


def test_the_forum_prompt_asks_for_unprefixed_names():
    from aiclipper.llm import prompts

    system = prompts.FORUM_SYSTEM
    assert "no `r/`" in system and "no `u/`" in system
    for field in ("community", "author"):
        described = prompts.FORUM_SCHEMA["properties"][field]["description"]
        assert "no site prefix" in described
        assert "r/" not in described and "u/" not in described


# --------------------------------------------------------------------------- #
# model output is used when it is good, ignored when it is not
# --------------------------------------------------------------------------- #


def test_write_script_uses_a_good_payload():
    provider = FakeProvider(GOOD_SCRIPT_PAYLOAD)
    script = sg.write_script("bread", seconds=30, provider=provider, settings=seeded(1))
    assert provider.calls, "the provider should have been asked"
    assert script.hook == GOOD_SCRIPT_PAYLOAD["hook"]
    assert script.title == GOOD_SCRIPT_PAYLOAD["title"]
    assert script.hashtags == ["#bread", "#baking"]  # normalised, junk dropped
    assert script.beats[0].emphasis is True


BAD_PAYLOADS = [
    None,
    {},
    [],
    "not a dict",
    {"hook": "", "beats": []},
    {"hook": "   ", "beats": [123, {"text": "  "}, None], "cta": None, "hashtags": None},
    {"messages": []},
    {"messages": [{"sender": "A", "text": ""}]},
    {"title": "t", "body": "too short"},
    {"beats": "not a list"},
]


@pytest.mark.parametrize("payload", BAD_PAYLOADS)
def test_malformed_payloads_fall_back_to_the_template(payload):
    provider = FakeProvider(payload)
    script = sg.write_script("bread ovens", seconds=20, provider=provider, settings=seeded(2))
    assert script.hook and script.beats and script.cta and script.hashtags
    assert abs(sg.estimate_seconds(script) - 20) <= 2.0

    chat = sg.write_chat("bread ovens", turns=6, provider=FakeProvider(payload), settings=seeded(2))
    assert len(chat.messages) == 6
    assert all(m.text.strip() for m in chat.messages)

    post = sg.write_reddit("bread ovens", words=80, provider=FakeProvider(payload), settings=seeded(2))
    assert post.body.strip() and post.title.strip()
    assert abs(len(post.body.split()) - 80) <= 10


@pytest.mark.parametrize("error", [RuntimeError("boom"), ValueError("nope"), KeyError("k")])
def test_provider_errors_fall_back_to_the_template(error):
    provider = FakeProvider(error=error)
    script = sg.write_script("bread ovens", seconds=25, provider=provider, settings=seeded(4))
    chat = sg.write_chat("bread ovens", turns=8, provider=FakeProvider(error=error), settings=seeded(4))
    post = sg.write_reddit("bread ovens", words=90, provider=FakeProvider(error=error), settings=seeded(4))

    assert script.hook and len(script.beats) >= 1
    assert len(chat.messages) == 8
    assert post.body.strip()


def test_broken_prompt_module_still_reaches_the_provider(monkeypatch):
    import aiclipper.llm.prompts as prompts_module

    def boom(*args, **kwargs):
        raise RuntimeError("prompts exploded")

    monkeypatch.setattr(prompts_module, "script_prompt", boom)
    provider = FakeProvider(GOOD_SCRIPT_PAYLOAD)
    script = sg.write_script("bread ovens", seconds=30, provider=provider, settings=seeded(5))

    assert provider.calls, "a local prompt should stand in for the broken one"
    prompt, schema, _system = provider.calls[0]
    assert "bread ovens" in prompt
    assert schema["type"] == "object"
    assert script.hook == GOOD_SCRIPT_PAYLOAD["hook"]


def test_unavailable_llm_package_falls_back_to_the_template(monkeypatch):
    def boom(*args, **kwargs):
        raise ImportError("no llm layer here")

    monkeypatch.setattr("aiclipper.llm.get_provider", boom)
    script = sg.write_script("bread ovens", seconds=20, settings=seeded(6))
    chat = sg.write_chat("bread ovens", turns=6, settings=seeded(6))
    post = sg.write_reddit("bread ovens", words=70, settings=seeded(6))

    assert script.hook and script.beats
    assert abs(sg.estimate_seconds(script) - 20) <= 2.0
    assert len(chat.messages) == 6
    assert abs(len(post.body.split()) - 70) <= 10


# --------------------------------------------------------------------------- #
# degenerate inputs
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("topic", ["", "   ", "\n\t "])
def test_generators_handle_empty_topics(topic):
    script = sg.write_script(topic, seconds=15, settings=seeded(8))
    chat = sg.write_chat(topic, turns=6, settings=seeded(8))
    post = sg.write_reddit(topic, words=60, settings=seeded(8))

    assert script.hook.strip() and script.beats and script.hashtags
    assert abs(sg.estimate_seconds(script) - 15) <= 1.5
    assert len(chat.messages) == 6 and all(m.text.strip() for m in chat.messages)
    assert post.title.strip() and post.body.strip()
    assert post.community.strip() and not _has_site_prefix(post.community)


def test_generators_are_offline_with_the_heuristic_provider():
    from aiclipper.llm import get_provider

    provider = get_provider("heuristic", settings=seeded(13))
    script = sg.write_script("a deadline nobody told me about", seconds=20, provider=provider, settings=seeded(13))
    chat = sg.write_chat("a deadline nobody told me about", turns=6, provider=provider, settings=seeded(13))
    post = sg.write_reddit("a deadline nobody told me about", words=70, provider=provider, settings=seeded(13))

    assert abs(sg.estimate_seconds(script) - 20) <= 2.0
    assert len(chat.messages) == 6
    assert abs(len(post.body.split()) - 70) <= 10
    # repeat runs with the same seed agree
    assert sg.write_reddit("a deadline nobody told me about", words=70, provider=provider, settings=seeded(13)) == post


# --------------------------------------------------------------------------- #
# regression tests for bugs found while reviewing the module
# --------------------------------------------------------------------------- #


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


@pytest.mark.parametrize("seconds", [90, 120, 200])
def test_write_script_hits_long_duration_budgets(seconds):
    """Padding used to run out of steam: a 120s ask returned ~100s of narration."""
    script = sg.write_script("why bread rises in a hot oven", seconds=seconds, settings=seeded(7))
    target = sg.budget_words(seconds)
    assert abs(len(script.narration.split()) - target) <= 3
    # and it stays a script: many spoken lines, not one enormous closing beat
    assert len(script.beats) >= seconds // 8
    assert max(len(b.text.split()) for b in script.beats) <= 40


@pytest.mark.parametrize("seconds", [3, 5, 8])
def test_write_script_does_not_overshoot_a_tiny_budget(seconds):
    """The trimmer used to stop at the last beat, leaving hook + cta over budget."""
    script = sg.write_script("why bread rises in a hot oven", seconds=seconds, settings=seeded(7))
    assert len(script.narration.split()) <= sg.budget_words(seconds) + 1
    assert script.hook.strip()
    assert script.lines  # something is still said


@pytest.mark.parametrize("words", [300, 400, 700])
def test_write_reddit_hits_large_word_targets(words):
    post = sg.write_reddit("my neighbour towed my car", words=words, settings=seeded(7))
    assert abs(len(post.body.split()) - words) <= 3


def test_padding_does_not_stutter():
    """Filler is cycled, not "the longest phrase that fits" over and over."""
    script = sg.write_script("bread", seconds=200, settings=seeded(7))
    lines = sentences(script.narration)
    assert not any(a == b for a, b in zip(lines, lines[1:], strict=False)), "repeated sentence back to back"
    assert len(set(lines)) >= 15, "padding should draw on the whole phrase pool"

    body = sg.write_reddit("bread", words=500, settings=seeded(7)).body
    body_lines = sentences(body)
    assert not any(a == b for a, b in zip(body_lines, body_lines[1:], strict=False))


def test_string_payloads_are_not_iterated_character_by_character():
    """A string where a list belongs used to become one beat/tag/message per letter."""
    provider = FakeProvider({"hook": "", "beats": "not a list", "hashtags": "bread", "cta": "x"})
    script = sg.write_script("bread ovens", seconds=20, provider=provider, settings=seeded(2))
    assert all(len(b.text.split()) > 1 for b in script.beats)
    assert all(len(tag) > 2 for tag in script.hashtags)

    chat = sg.write_chat("bread ovens", turns=6, provider=FakeProvider({"messages": "hello"}), settings=seeded(2))
    assert all(len(m.text.split()) > 1 for m in chat.messages)


def test_prompt_echo_is_rejected_in_favour_of_the_template():
    echoed = {
        "title": "t",
        "hook": "Write a 20-second vertical short in a punchy tone.",
        "beats": [{"text": "Premise: bread ovens"}, {"text": "spoken narration goes here"}],
        "cta": "",
    }
    script = sg.write_script("bread ovens", seconds=20, provider=FakeProvider(echoed), settings=seeded(2))
    assert "vertical short" not in script.narration.lower()
    assert "premise:" not in script.narration.lower()
    assert script.hook.strip() and script.beats


def test_word_counting_understands_non_ascii_narration():
    """An ASCII-only word regex measured a Cyrillic script as zero words."""
    russian = "Привет это хук. Вторая строка здесь. Третья строка тут."
    parsed = sg.parse_script(russian)
    assert len(parsed.beats) == 2
    assert sg.estimate_seconds(parsed) == pytest.approx(9 / sg.WORDS_PER_SECOND, abs=1e-6)
    # accented Latin counts one word per word, not one per accent-free fragment
    assert sg.estimate_seconds("café naïve") == pytest.approx(2 / sg.WORDS_PER_SECOND, abs=1e-6)


def test_fit_script_to_budget_leaves_the_input_untouched():
    original = VideoScript(
        title="T", hook="A hook with several words in it.",
        beats=[ScriptBeat("One beat here.", broll="b"), ScriptBeat("Another beat here.")],
        cta="Follow.", hashtags=["#a"],
    )
    before = (original.title, original.hook, [b.text for b in original.beats], original.cta, original.hashtags)
    sg.fit_script_to_budget(original, 5)
    sg.fit_script_to_budget(original, 60)
    after = (original.title, original.hook, [b.text for b in original.beats], original.cta, original.hashtags)
    assert before == after


def test_generators_repeat_themselves_for_one_seed():
    topic = "the parcel that kept coming back"
    assert sg.write_script(topic, seconds=30, settings=seeded(21)) == sg.write_script(
        topic, seconds=30, settings=seeded(21)
    )
    first = sg.write_chat(topic, turns=8, settings=seeded(21))
    assert first == sg.write_chat(topic, turns=8, settings=seeded(21))
    # a different seed is still a valid chat of the requested length
    other = sg.write_chat(topic, turns=8, settings=seeded(22))
    assert len(other.messages) == 8


def test_parse_chat_handles_crlf_and_a_leading_delay_marker():
    script = sg.parse_chat("# T\r\n[5s]\r\nAlex: one\r\n> me: two\r\n")
    assert [(m.sender, m.outgoing, m.delay) for m in script.messages] == [
        ("Alex", False, 5.0),
        ("me", True, 0.35),
    ]
    assert all("\r" not in m.text for m in script.messages)


def test_format_chat_of_an_empty_script_round_trips():
    assert sg.parse_chat(sg.format_chat(ChatScript())).messages == []


@pytest.mark.parametrize("turns", [-3, 0, 1])
def test_write_chat_clamps_degenerate_turn_counts(turns):
    """A conversation needs two sides; anything below two is raised to two."""
    chat = sg.write_chat("a locked door", turns=turns, settings=seeded(3))
    assert len(chat.messages) == 2
    assert chat.messages[0].outgoing != chat.messages[1].outgoing


# --------------------------------------------------------------------------- #
# the prompt's own wording never becomes content
# --------------------------------------------------------------------------- #

#: Wording that can only have come from our own prompt scaffolding.
INSTRUCTION_VOCABULARY = (
    "vertical short", "spoken narration", "call to action", "punchy tone",
    "text-message story", "forum story post", "words in the body",
    "premise:", "topic:", "write a", "-second",
)

#: Topics too thin to out-weigh the instructions wrapped around them.
THIN_TOPICS = ["cats", "tiny things", "\U0001f431", "ok", "the parcel " * 400]


def _instruction_leaks(text: str) -> list[str]:
    low = (text or "").lower()
    return [phrase for phrase in INSTRUCTION_VOCABULARY if phrase in low]


@pytest.mark.parametrize("topic", THIN_TOPICS, ids=range(len(THIN_TOPICS)))
def test_offline_output_never_echoes_the_prompt_instructions(topic):
    """The title becomes the output filename, so a leak here is visible on disk."""
    script = sg.write_script(topic, seconds=10, settings=seeded(4))
    chat = sg.write_chat(topic, turns=6, settings=seeded(4))
    post = sg.write_reddit(topic, words=60, settings=seeded(4))

    assert _instruction_leaks(script.title) == [], script.title
    assert _instruction_leaks(chat.title) == [], chat.title
    assert _instruction_leaks(post.title) == [], post.title
    # ...and the rest of the copy is clean too, not only the titles.
    assert _instruction_leaks(f"{script.hook} {script.narration}") == []
    assert _instruction_leaks(" ".join(m.text for m in chat.messages)) == []
    assert _instruction_leaks(post.body) == []


@pytest.mark.parametrize(
    ("topic", "expected"),
    [("cats", ("cats",)), ("tiny things", ("tiny", "things")), ("otters", ("otters",))],
)
def test_offline_titles_are_about_the_topic(topic, expected):
    """Low-information topics still have to name the thing they are about."""
    titles = [
        sg.write_script(topic, seconds=10, settings=seeded(4)).title,
        sg.write_chat(topic, turns=6, settings=seeded(4)).title,
        sg.write_reddit(topic, words=60, settings=seeded(4)).title,
    ]
    for title in titles:
        assert any(word in title.lower() for word in expected), title


def test_a_model_title_that_echoes_our_instructions_is_replaced():
    """The guard that already covers hooks, beats and messages covers titles too."""
    echo = "Write a 30-second vertical short in a punchy tone"
    script = sg.write_script(
        "sourdough", seconds=30, settings=seeded(7),
        provider=FakeProvider(dict(GOOD_SCRIPT_PAYLOAD, title=echo)),
    )
    chat = sg.write_chat(
        "sourdough", turns=4, settings=seeded(7),
        provider=FakeProvider(dict(GOOD_CHAT_PAYLOAD, title="Write a text-message story")),
    )
    post = sg.write_reddit(
        "sourdough", words=60, settings=seeded(7),
        provider=FakeProvider(dict(GOOD_FORUM_PAYLOAD, title="Write a forum story post")),
    )
    for title in (script.title, chat.title, post.title):
        assert _instruction_leaks(title) == [], title
        assert "sourdough" in title.lower(), title
    # the rest of a good payload is still used
    assert script.hook == GOOD_SCRIPT_PAYLOAD["hook"]
    assert "my key did not turn" in post.body


def test_the_subject_reaches_the_provider_tagged():
    provider = FakeProvider(GOOD_SCRIPT_PAYLOAD)
    sg.write_script("bread ovens", seconds=30, provider=provider, settings=seeded(5))
    prompt, _schema, _system = provider.calls[0]
    assert "bread ovens" in prompt
    assert "[subject]" in prompt and "[/subject]" in prompt
