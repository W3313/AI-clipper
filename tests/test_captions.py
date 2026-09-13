"""Tests for :mod:`aiclipper.captions`.

Covers colour/time primitives, cue grouping invariants, the ASS document
structure for every preset, and one real ffmpeg burn-in so we know libass
actually accepts what we write.
"""

from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import pytest

from aiclipper import captions
from aiclipper import ffmpeg as ff
from aiclipper.models import CaptionCue, CaptionStyle, Word

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

DIALOGUE_RE = re.compile(
    r"^Dialogue: (\d+),(\d+:\d{2}:\d{2}\.\d{2}),(\d+:\d{2}:\d{2}\.\d{2}),([^,]*),([^,]*),"
    r"(\d+),(\d+),(\d+),([^,]*),(.*)$"
)
PER_WORD_ANIMATIONS = {"karaoke", "pop", "typewriter"}


def parse_time(stamp: str) -> float:
    hours, minutes, rest = stamp.split(":")
    secs, cs = rest.split(".")
    return int(hours) * 3600 + int(minutes) * 60 + int(secs) + int(cs) / 100.0


def dialogues(text: str) -> list[tuple[float, float, str]]:
    out: list[tuple[float, float, str]] = []
    for line in text.splitlines():
        if not line.startswith("Dialogue:"):
            continue
        match = DIALOGUE_RE.match(line)
        assert match, f"malformed dialogue line: {line!r}"
        out.append((parse_time(match.group(2)), parse_time(match.group(3)), match.group(10)))
    return out


def sections(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.startswith("[")]


def words_from(spec: str, *, per_word: float = 0.4, gap: float = 0.0, start: float = 0.0) -> list[Word]:
    out: list[Word] = []
    t = start
    for token in spec.split():
        out.append(Word(token, t, t + per_word))
        t += per_word + gap
    return out


LONG_TEXT = (
    "this is the part nobody tells you about shipping a product people want. "
    "what happened next changed how the whole team thought about the problem! "
    "why does it keep working when every model says that it should not? "
    "we went from 12 thousand to 240 thousand in 90 days across 3 markets"
)


# --------------------------------------------------------------------------- #
# colours
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("#FFFFFF", "&H00FFFFFF"),
        ("#000000", "&H00000000"),
        ("#FFE400", "&H0000E4FF"),      # R=FF G=E4 B=00 -> BGR 00E4FF
        ("#102030", "&H00302010"),
        ("#35F08A", "&H008AF035"),
        ("FFE400", "&H0000E4FF"),       # leading '#' optional
        ("#fff", "&H00FFFFFF"),         # 3-digit shorthand
        ("#f00", "&H000000FF"),
        ("#FF102030", "&H00302010"),    # opaque alpha -> ASS 00
        ("#00102030", "&HFF302010"),    # transparent alpha -> ASS FF
        ("#CC101018", "&H33181010"),    # CC (204) -> 255-204 = 0x33
        ("#80000000", "&H7F000000"),
        ("&H00FF8000", "&H00FF8000"),   # already an ASS literal
    ],
)
def test_ass_color_table(value: str, expected: str) -> None:
    assert captions.ass_color(value) == expected


def test_ass_color_alpha_is_inverted() -> None:
    assert captions.ass_color("#FF123456")[:4] == "&H00"
    assert captions.ass_color("#00123456")[:4] == "&HFF"
    # alpha byte is the only thing that changes between the two
    assert captions.ass_color("#FF123456")[4:] == captions.ass_color("#00123456")[4:]


def test_ass_color_without_alpha_is_inline_form() -> None:
    assert captions.ass_color("#FFE400", with_alpha=False) == "&H00E4FF&"
    assert captions.ass_color("#CC101018", with_alpha=False) == "&H181010&"


@pytest.mark.parametrize("bad", ["", "#", "#12345", "#GGHHII", "#1234567", "not-a-colour", "#12345678910"])
def test_ass_color_rejects_invalid(bad: str) -> None:
    with pytest.raises(ValueError):
        captions.ass_color(bad)


def test_ass_color_rejects_non_string() -> None:
    with pytest.raises(ValueError):
        captions.ass_color(0xFFEE00)  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# timestamps
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "0:00:00.00"),
        (0.01, "0:00:00.01"),
        (1.5, "0:00:01.50"),
        (3.456, "0:00:03.46"),       # centisecond rounding
        (3.454, "0:00:03.45"),
        (59.999, "0:01:00.00"),      # rounds up across the minute
        (61.25, "0:01:01.25"),
        (599.9, "0:09:59.90"),
        (3600.0, "1:00:00.00"),
        (3599.999, "1:00:00.00"),
        (7322.33, "2:02:02.33"),
        (36000.0, "10:00:00.00"),    # >9h keeps extra digits
        (-0.5, "0:00:00.00"),        # negatives clamp
        (-1000.0, "0:00:00.00"),
    ],
)
def test_ass_timestamp(seconds: float, expected: str) -> None:
    assert captions.ass_timestamp(seconds) == expected


def test_ass_timestamp_handles_non_finite() -> None:
    assert captions.ass_timestamp(float("nan")) == "0:00:00.00"
    assert captions.ass_timestamp(float("-inf")) == "0:00:00.00"


# --------------------------------------------------------------------------- #
# escaping
# --------------------------------------------------------------------------- #

def test_escape_text_handles_braces_backslashes_newlines() -> None:
    assert captions.escape_text("{pos}") == "\\{pos\\}"
    assert captions.escape_text("a\\b") == "a\\\\b"
    assert captions.escape_text("one\ntwo") == "one\\Ntwo"
    assert captions.escape_text("one\r\ntwo") == "one\\Ntwo"
    # a backslash is escaped before braces, so we never manufacture an override
    assert captions.escape_text("\\{") == "\\\\\\{"


def test_escaped_specials_survive_into_the_file(tmp_path: Path) -> None:
    style = replace(captions.get_style("clean"), animation="none", max_words=4, max_chars=99)
    cues = captions.group_words(words_from("{fx} back\\slash plain"), style)
    out = captions.write_ass(cues, tmp_path / "esc.ass", style=style, width=1080, height=1920)
    body = out.read_text(encoding="utf-8")
    text = dialogues(body)[0][2]
    assert "\\{fx\\}" in text
    assert "back\\\\slash" in text
    # no unescaped brace can open an override block by accident
    assert re.search(r"(?<!\\)\{", text) is None


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #

def test_group_respects_max_words() -> None:
    style = replace(captions.get_style("clean"), max_words=2, max_chars=999)
    cues = captions.group_words(words_from("a b c d e"), style)
    assert [len(c.words) for c in cues] == [2, 2, 1]


def test_group_respects_max_chars() -> None:
    style = replace(captions.get_style("clean"), max_words=99, max_chars=10)
    cues = captions.group_words(words_from("alpha beta gamma delta"), style)
    assert [c.text for c in cues] == ["alpha beta", "gamma", "delta"]
    assert all(len(c.text) <= 10 for c in cues)


def test_group_keeps_oversized_word_alone() -> None:
    style = replace(captions.get_style("clean"), max_words=5, max_chars=4)
    cues = captions.group_words(words_from("hi extraordinarily ok"), style)
    assert [c.text for c in cues] == ["hi", "extraordinarily", "ok"]


def test_group_splits_on_long_gap(sample_words: list[Word]) -> None:
    style = replace(captions.get_style("clean"), max_words=99, max_chars=999, animation="none")
    cues = captions.group_words(sample_words, style)
    # the fixture pauses 1.16s before "Here"
    assert len(cues) == 2
    assert cues[0].words[-1].text == "you"
    assert cues[1].words[0].text == "Here"


def test_gap_split_brackets_the_gap_split_threshold() -> None:
    """The split happens within a centisecond of :data:`GAP_SPLIT`, either side."""
    style = replace(captions.get_style("clean"), max_words=99, max_chars=999)
    under = [Word("one", 0.0, 0.4), Word("two", 0.4 + captions.GAP_SPLIT - 0.01, 1.5)]
    over = [Word("one", 0.0, 0.4), Word("two", 0.4 + captions.GAP_SPLIT + 0.01, 1.5)]
    assert len(captions.group_words(under, style)) == 1
    assert len(captions.group_words(over, style)) == 2


def test_group_does_not_split_on_short_gap() -> None:
    style = replace(captions.get_style("clean"), max_words=99, max_chars=999)
    words = [Word("one", 0.0, 0.4), Word("two", 1.0, 1.4)]  # 0.6s gap, under the 0.7 threshold
    assert len(captions.group_words(words, style)) == 1


def test_group_splits_after_sentence_punctuation() -> None:
    style = replace(captions.get_style("clean"), max_words=99, max_chars=999)
    words = words_from("stop. go on! really? yes")
    cues = captions.group_words(words, style)
    assert [c.text for c in cues] == ["stop.", "go on!", "really?", "yes"]


def test_group_does_not_split_on_initials_or_decimals() -> None:
    style = replace(captions.get_style("clean"), max_words=99, max_chars=999)
    cues = captions.group_words(words_from("J. R. Smith paid 3.5 today"), style)
    assert len(cues) == 1


def test_group_skips_blank_words() -> None:
    style = captions.get_style("clean")
    words = [Word("real", 0.0, 0.3), Word("   ", 0.3, 0.4), Word("words", 0.4, 0.8)]
    cues = captions.group_words(words, style)
    assert " ".join(c.text for c in cues) == "real words"


def test_group_returns_nothing_for_empty_input() -> None:
    assert captions.group_words([], captions.get_style("clean")) == []


@pytest.mark.parametrize("name", sorted(captions.PRESETS))
def test_group_never_drops_or_reorders_words(name: str) -> None:
    style = captions.get_style(name)
    words = words_from(LONG_TEXT, per_word=0.3, gap=0.05)
    words[20] = replace(words[20], start=words[20].start + 2.0, end=words[20].end + 2.0)
    for i in range(21, len(words)):
        words[i] = replace(words[i], start=words[i].start + 2.0, end=words[i].end + 2.0)
    cues = captions.group_words(words, style)
    flat = [w.text for cue in cues for w in cue.words]
    assert flat == [w.text for w in words]
    assert " ".join(cue.text for cue in cues) == " ".join(w.text for w in words)
    assert all(len(cue.words) <= style.max_words for cue in cues)


@pytest.mark.parametrize("name", sorted(captions.PRESETS))
def test_cue_times_are_monotonic_and_non_overlapping(name: str) -> None:
    style = captions.get_style(name)
    cues = captions.group_words(words_from(LONG_TEXT, per_word=0.3, gap=0.05), style)
    assert cues
    for cue in cues:
        assert cue.start >= 0.0
        assert cue.end > cue.start
        assert cue.start <= cue.words[0].start + 1e-9
        assert cue.end >= cue.words[-1].end - 1e-9
    for first, second in zip(cues, cues[1:], strict=False):
        assert first.end <= second.start + 1e-9
        assert first.start < second.start


def test_cue_end_extends_into_the_gap_but_not_past_the_next_cue() -> None:
    style = replace(captions.get_style("clean"), max_words=1, max_chars=99)
    words = [Word("first", 0.0, 0.5), Word("second", 4.0, 4.5)]
    a, b = captions.group_words(words, style)
    assert 0.5 < a.end <= 0.5 + captions.HOLD_SECONDS
    assert a.end < b.start
    assert b.end == pytest.approx(4.5 + captions.HOLD_SECONDS)


def test_degenerate_style_limits_are_clamped() -> None:
    """``max_words``/``max_chars`` of 0 must not wedge the packer or drop words."""
    style = replace(captions.get_style("clean"), max_words=0, max_chars=0)
    cues = captions.group_words(words_from("one two three"), style)
    assert [c.text for c in cues] == ["one", "two", "three"]


def test_overlapping_input_never_yields_a_zero_length_cue() -> None:
    """Out-of-order ASR output still has to produce cues that are on screen."""
    style = replace(captions.get_style("clean"), max_words=1, max_chars=99)
    words = [Word("a", 0.0, 1.0), Word("b", 0.5, 0.6), Word("c", 0.2, 0.3)]
    cues = captions.group_words(words, style)
    assert [c.text for c in cues] == ["a", "b", "c"]  # nothing dropped or reordered
    for cue in cues:
        assert cue.end > cue.start


def test_tight_words_do_not_produce_overlapping_cues() -> None:
    style = replace(captions.get_style("clean"), max_words=1, max_chars=99)
    words = [Word("a", 0.0, 0.2), Word("b", 0.2, 0.4), Word("c", 0.4, 0.6)]
    cues = captions.group_words(words, style)
    for first, second in zip(cues, cues[1:], strict=False):
        assert first.end <= second.start + 1e-9


# --------------------------------------------------------------------------- #
# styles
# --------------------------------------------------------------------------- #

def test_presets_cover_the_documented_names() -> None:
    required = {
        "clean", "bold_yellow", "karaoke_green", "outline_pop", "boxed", "minimal_serif",
        "neon", "comic", "subtle_lower", "big_impact", "gradient_pop", "mono_terminal",
        "handwritten", "shadow_deep", "tiktok_white", "podcast_bar",
    }
    assert required <= set(captions.PRESETS)
    assert len(captions.PRESETS) >= 16


def test_presets_are_self_consistent_and_described() -> None:
    for key, style in captions.PRESETS.items():
        assert style.name == key
        assert style.description and len(style.description.split()) >= 3
        assert style.font_size > 0
        assert style.max_words >= 1 and style.max_chars >= 1
        assert style.position in {"top", "center", "bottom"}
        assert style.animation in {"none", "karaoke", "pop", "bounce", "fade", "typewriter"}
        for colour in (style.primary_color, style.highlight_color, style.outline_color, style.shadow_color):
            captions.ass_color(colour)  # must parse
        if style.back_color:
            captions.ass_color(style.back_color)


def test_presets_are_visually_distinct() -> None:
    fingerprints = {
        (s.font, s.font_size, s.primary_color, s.highlight_color, s.outline, s.position, s.animation)
        for s in captions.PRESETS.values()
    }
    assert len(fingerprints) == len(captions.PRESETS)
    # every animation mode is represented by at least one preset
    assert {s.animation for s in captions.PRESETS.values()} == {
        "none", "karaoke", "pop", "bounce", "fade", "typewriter"
    }
    assert len({s.font for s in captions.PRESETS.values()}) >= 3
    assert len({s.position for s in captions.PRESETS.values()}) >= 2


def test_get_style_lookup_is_forgiving() -> None:
    assert captions.get_style("clean").name == "clean"
    assert captions.get_style("  BOLD-YELLOW ").name == "bold_yellow"
    assert captions.get_style("mono terminal").name == "mono_terminal"
    assert captions.get_style("").name == captions.DEFAULT_STYLE


def test_get_style_returns_a_copy_not_the_preset() -> None:
    style = captions.get_style("clean")
    assert style is not captions.PRESETS["clean"]
    style.font_size = 12
    assert captions.PRESETS["clean"].font_size != 12


def test_get_style_passes_instances_through() -> None:
    custom = CaptionStyle(name="mine", font_size=40)
    assert captions.get_style(custom) is custom


def test_get_style_rejects_unknown_names() -> None:
    with pytest.raises(ValueError) as excinfo:
        captions.get_style("definitely_not_a_preset")
    assert "clean" in str(excinfo.value)


def test_list_styles_is_sorted_and_complete() -> None:
    names = [s.name for s in captions.list_styles()]
    assert names == sorted(captions.PRESETS)


# --------------------------------------------------------------------------- #
# ASS document structure
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(captions.PRESETS))
def test_every_preset_writes_a_parseable_ass_file(tmp_path: Path, name: str) -> None:
    style = captions.get_style(name)
    words = words_from(LONG_TEXT, per_word=0.3, gap=0.05)
    cues = captions.group_words(words, style)
    out = captions.write_ass(cues, tmp_path / f"{name}.ass", style=style, width=1080, height=1920)
    body = out.read_text(encoding="utf-8")

    assert sections(body) == ["[Script Info]", "[V4+ Styles]", "[Events]"]
    assert "ScriptType: v4.00+" in body
    # WrapStyle 2 disables wrapping outright and lets long cues run off canvas
    assert "WrapStyle: 0" in body
    assert "ScaledBorderAndShadow: yes" in body
    assert "PlayResX: 1080" in body
    assert "PlayResY: 1920" in body

    style_lines = [ln for ln in body.splitlines() if ln.startswith("Style:")]
    assert len(style_lines) == 1
    assert len([ln for ln in body.splitlines() if ln.startswith("Format:")]) == 2

    fields = style_lines[0][len("Style: "):].split(",")
    assert len(fields) == 23
    assert fields[0] == "Default"
    assert fields[1] == style.font
    assert fields[2] == str(style.font_size)
    assert fields[3] == captions.ass_color(style.primary_color)
    # libass paints the opaque box with OutlineColour, so back_color lives there
    assert fields[5] == captions.ass_color(style.back_color or style.outline_color)
    assert fields[6] == captions.ass_color(style.shadow_color)
    assert fields[7] == ("-1" if style.bold else "0")
    assert fields[8] == ("-1" if style.italic else "0")
    assert fields[15] == ("3" if style.back_color else "1")
    assert fields[18] == str({"top": 8, "center": 5, "bottom": 2}[style.position])
    assert fields[19] == fields[20] == str(style.margin_h)
    assert fields[21] == str(style.margin_v)

    events = dialogues(body)
    expected = (
        sum(len(c.words) for c in cues) if style.animation in PER_WORD_ANIMATIONS else len(cues)
    )
    assert len(events) == expected
    for start, end, text in events:
        assert end >= start
        assert "\n" not in text and text.strip()
    starts = [e[0] for e in events]
    assert starts == sorted(starts)


def test_empty_words_produce_a_header_only_file(tmp_path: Path) -> None:
    out = captions.build([], tmp_path / "empty.ass", style="clean")
    body = out.read_text(encoding="utf-8")
    assert sections(body) == ["[Script Info]", "[V4+ Styles]", "[Events]"]
    assert body.rstrip().endswith(f"Format: {captions._EVENT_FORMAT}")
    assert dialogues(body) == []
    assert len([ln for ln in body.splitlines() if ln.startswith("Style:")]) == 1


def test_canvas_size_drives_playres(tmp_path: Path) -> None:
    out = captions.build(words_from("small canvas here"), tmp_path / "c.ass", width=720, height=1280)
    body = out.read_text(encoding="utf-8")
    assert "PlayResX: 720" in body
    assert "PlayResY: 1280" in body


def test_back_color_switches_border_style_and_box_colour(tmp_path: Path) -> None:
    """BorderStyle 3 must carry ``back_color`` in the *outline* slot.

    libass/VSFilter fill the opaque box with OutlineColour and keep BackColour
    for the drop shadow; writing the slab colour into BackColour makes the box
    render in the outline colour instead (verified against a real burn-in).
    """
    plain = replace(captions.get_style("clean"), back_color=None,
                    outline_color="#ABCDEF", shadow_color="#123456")
    boxed = replace(plain, back_color="#CC101018")
    for style, expected_border, expected_outline in (
        (plain, "1", captions.ass_color("#ABCDEF")),
        (boxed, "3", captions.ass_color("#CC101018")),
    ):
        out = captions.write_ass([], tmp_path / f"{expected_border}.ass", style=style, width=1080, height=1920)
        fields = [ln for ln in out.read_text().splitlines() if ln.startswith("Style:")][0].split(",")
        assert fields[15] == expected_border  # "Style: Default" collapses into one field
        assert fields[5] == expected_outline
        assert fields[6] == captions.ass_color("#123456")  # BackColour stays the shadow


def test_karaoke_highlights_one_word_at_a_time_and_restores_colour(tmp_path: Path) -> None:
    style = replace(captions.get_style("clean"), animation="karaoke", max_words=3, max_chars=99)
    words = words_from("alpha beta gamma")
    out = captions.build(words, tmp_path / "k.ass", style=style)
    events = dialogues(out.read_text(encoding="utf-8"))
    assert len(events) == 3
    high = captions.ass_color(style.highlight_color, with_alpha=False)
    base = captions.ass_color(style.primary_color, with_alpha=False)
    for index, (_, _, text) in enumerate(events):
        token = words[index].text
        assert f"{{\\c{high}}}{token}{{\\c{base}}}" in text
        assert text.count(f"\\c{high}") == 1
        # the whole cue is always on screen
        for other in words:
            assert other.text in text
    # each dialogue starts when its word does
    assert [round(e[0], 2) for e in events] == [0.0, 0.4, 0.8]


def test_pop_scales_the_active_word(tmp_path: Path) -> None:
    style = replace(captions.get_style("big_impact"), max_words=2, max_chars=99, scale_pop=1.25)
    out = captions.build(words_from("hey there"), tmp_path / "p.ass", style=style)
    events = dialogues(out.read_text(encoding="utf-8"))
    assert len(events) == 2
    assert "\\t(0,120,\\fscx125\\fscy125)" in events[0][2]
    assert events[0][2].index("\\t(") < events[0][2].index("HEY")  # override precedes its word
    assert "\\fscx100\\fscy100}" in events[0][2]


def test_typewriter_reveals_word_by_word(tmp_path: Path) -> None:
    style = replace(captions.get_style("mono_terminal"), max_words=3, max_chars=99)
    out = captions.build(words_from("one two three"), tmp_path / "t.ass", style=style)
    events = dialogues(out.read_text(encoding="utf-8"))
    assert [text for _, _, text in events] == ["one", "one two", "one two three"]


def test_fade_and_bounce_emit_one_dialogue_per_cue(tmp_path: Path) -> None:
    faded = replace(captions.get_style("boxed"), animation="fade", max_words=4, max_chars=99)
    out = captions.build(words_from("a b c d"), tmp_path / "f.ass", style=faded)
    events = dialogues(out.read_text(encoding="utf-8"))
    assert len(events) == 1
    assert events[0][2].startswith("{\\fad(120,120)}")

    bouncy = replace(captions.get_style("comic"), animation="bounce", max_words=4, max_chars=99)
    out = captions.build(words_from("a b c d"), tmp_path / "b.ass", style=bouncy, width=1080, height=1920)
    text = dialogues(out.read_text(encoding="utf-8"))[0][2]
    move = re.match(r"^\{\\move\((\d+),(\d+),(\d+),(\d+),0,(\d+)\)\}", text)
    assert move, text
    x1, y1, x2, y2, ms = (int(g) for g in move.groups())
    assert x1 == x2 == 540
    assert y2 == 1920 - bouncy.margin_v
    assert y1 > y2  # starts lower and rises into place
    assert 0 < ms <= 400


def test_none_animation_is_plain_text(tmp_path: Path) -> None:
    style = replace(captions.get_style("subtle_lower"), animation="none", max_words=4, max_chars=99)
    out = captions.build(words_from("just plain words"), tmp_path / "n.ass", style=style)
    events = dialogues(out.read_text(encoding="utf-8"))
    assert len(events) == 1
    assert events[0][2] == "just plain words"


def test_uppercase_applies_to_rendered_text_only(tmp_path: Path) -> None:
    style = replace(captions.get_style("clean"), uppercase=True, animation="none", max_words=3, max_chars=99)
    words = words_from("shout it out")
    cues = captions.group_words(words, style)
    assert cues[0].text == "shout it out"  # cue keeps the source spelling
    out = captions.write_ass(cues, tmp_path / "u.ass", style=style, width=1080, height=1920)
    assert dialogues(out.read_text(encoding="utf-8"))[0][2] == "SHOUT IT OUT"


def test_dialogue_times_stay_inside_their_cue(tmp_path: Path) -> None:
    for name in ("clean", "mono_terminal", "big_impact"):
        style = captions.get_style(name)
        cues = captions.group_words(words_from(LONG_TEXT, per_word=0.3, gap=0.05), style)
        out = captions.write_ass(cues, tmp_path / f"{name}-span.ass", style=style, width=1080, height=1920)
        events = dialogues(out.read_text(encoding="utf-8"))
        # timestamps are quantised to centiseconds, so allow one cs of slack
        for start, end, _ in events:
            assert any(c.start - 0.01 <= start <= end <= c.end + 0.01 for c in cues)


def test_build_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deeper" / "caps.ass"
    out = captions.build(words_from("nested output path"), target, style="neon")
    assert out == target and target.exists()


def test_build_accepts_a_style_object(tmp_path: Path) -> None:
    style = CaptionStyle(name="custom", font="DejaVu Sans", font_size=48, animation="none",
                         max_words=2, max_chars=40)
    out = captions.build(words_from("two at a time"), tmp_path / "custom.ass", style=style)
    body = out.read_text(encoding="utf-8")
    assert ",DejaVu Sans,48," in body
    assert len(dialogues(body)) == 2


def test_build_rejects_unknown_style_name(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        captions.build(words_from("nope"), tmp_path / "x.ass", style="nope_not_here")


def test_output_is_deterministic(tmp_path: Path) -> None:
    words = words_from(LONG_TEXT, per_word=0.3, gap=0.05)
    first = captions.build(words, tmp_path / "a.ass", style="karaoke_green").read_bytes()
    second = captions.build(words, tmp_path / "b.ass", style="karaoke_green").read_bytes()
    assert first == second


def test_write_ass_overwrites_rather_than_appends(tmp_path: Path) -> None:
    target = tmp_path / "twice.ass"
    style = replace(captions.get_style("clean"), animation="none", max_words=9, max_chars=99)
    captions.write_ass(captions.group_words(words_from("a b c d e"), style), target,
                       style=style, width=1080, height=1920)
    captions.write_ass(captions.group_words(words_from("x"), style), target,
                       style=style, width=1080, height=1920)
    body = target.read_text(encoding="utf-8")
    assert len(dialogues(body)) == 1
    assert dialogues(body)[0][2] == "x"
    assert body.count("[Events]") == 1


def test_non_ascii_text_survives_as_utf8(tmp_path: Path) -> None:
    style = replace(captions.get_style("clean"), animation="none", uppercase=True,
                    max_words=4, max_chars=99)
    words = [Word("café", 0.0, 0.4), Word("naïve", 0.4, 0.8), Word("日本語", 0.8, 1.2)]
    out = captions.build(words, tmp_path / "utf8.ass", style=style)
    text = dialogues(out.read_text(encoding="utf-8"))[0][2]
    assert text == "CAFÉ NAÏVE 日本語"
    assert "CAFÉ".encode() in out.read_bytes()  # written as UTF-8, not latin-1/escapes


def test_font_name_cannot_break_the_style_line(tmp_path: Path) -> None:
    style = CaptionStyle(name="odd", font="Bad, Font", animation="none", max_words=2, max_chars=40)
    out = captions.build(words_from("still valid"), tmp_path / "font.ass", style=style)
    fields = [ln for ln in out.read_text().splitlines() if ln.startswith("Style:")][0]
    assert len(fields[len("Style: "):].split(",")) == 23
    assert "Bad Font" in fields


def test_degenerate_canvas_size_is_clamped(tmp_path: Path) -> None:
    out = captions.build(words_from("tiny"), tmp_path / "tiny.ass", width=0, height=-5)
    body = out.read_text(encoding="utf-8")
    assert "PlayResX: 2" in body and "PlayResY: 2" in body


def test_cues_can_be_written_directly(tmp_path: Path) -> None:
    cue = CaptionCue(start=0.0, end=1.0, words=[Word("manual", 0.0, 1.0)])
    out = captions.write_ass([cue], tmp_path / "manual.ass", style=captions.get_style("clean"),
                             width=1080, height=1920)
    assert len(dialogues(out.read_text(encoding="utf-8"))) == 1


# --------------------------------------------------------------------------- #
# the real thing: let libass parse it
# --------------------------------------------------------------------------- #

def _burn(ass_path: Path | None, out_path: Path, *, width: int = 320, height: int = 568,
          background: str = "black", pix_fmt: str = "yuv420p") -> None:
    """Render a 1s flat clip, optionally with ``ass_path`` burned in by libass."""
    args = ["-y", "-f", "lavfi", "-i", f"color=c={background}:s={width}x{height}:r=12:d=1"]
    if ass_path is not None:
        args += ["-vf", f"subtitles={ff.escape_filter_path(ass_path)}"]
    args += ["-t", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", pix_fmt, str(out_path)]
    ff.run_ffmpeg(args, timeout=120)


def _rgb_frame(video: Path, frame: Path, *, at: float = 0.5):
    """Grab one frame as an ``(h, w, 3)`` int array."""
    import numpy as np
    from PIL import Image

    ff.run_ffmpeg(["-y", "-ss", f"{at}", "-i", str(video), "-frames:v", "1", str(frame)], timeout=120)
    return np.asarray(Image.open(frame).convert("RGB")).astype(int)


def _bright_pixels(video: Path, frame: Path, *, at: float = 0.5) -> int:
    import numpy as np

    rgb = _rgb_frame(video, frame, at=at)
    return int((np.asarray(rgb).mean(axis=2) > 200).sum())


@pytest.mark.needs_ffmpeg
def test_ffmpeg_burns_in_generated_subtitles(tmp_path: Path) -> None:
    """libass must accept our file, and the burn must actually draw something."""
    if not ff.has_filter("subtitles"):
        pytest.skip("this ffmpeg build has no subtitles filter")
    style = replace(captions.get_style("clean"), font_size=30, outline=2.0, shadow=1.0,
                    margin_v=120, margin_h=20, max_words=3, max_chars=18)
    words = [Word("burned", 0.0, 0.3), Word("in", 0.3, 0.55), Word("captions", 0.55, 1.0)]
    ass_path = captions.build(words, tmp_path / "burn.ass", style=style, width=320, height=568)

    out = tmp_path / "burned.mp4"
    _burn(ass_path, out)
    assert out.exists() and out.stat().st_size > 0

    info = ff.probe(out)
    assert info.has_video and info.width == 320 and info.height == 568
    assert info.duration == pytest.approx(1.0, abs=0.25)

    # the same canvas without subtitles is pure black, so bright pixels == drawn glyphs
    blank = tmp_path / "blank.mp4"
    _burn(None, blank)
    assert _bright_pixels(blank, tmp_path / "blank.png") == 0
    assert _bright_pixels(out, tmp_path / "frame.png") > 50, "libass drew no caption pixels"


@pytest.mark.needs_ffmpeg
@pytest.mark.parametrize("name", ["karaoke_green", "mono_terminal", "comic", "podcast_bar"])
def test_ffmpeg_accepts_every_animation_kind(tmp_path: Path, name: str) -> None:
    if not ff.has_filter("subtitles"):
        pytest.skip("this ffmpeg build has no subtitles filter")
    style = replace(captions.get_style(name), font_size=26, margin_v=90, margin_h=16)
    words = words_from("libass has to parse all of this", per_word=0.12)
    ass_path = captions.build(words, tmp_path / f"{name}.ass", style=style, width=320, height=568)
    _burn(ass_path, tmp_path / f"{name}.mp4")
    assert (tmp_path / f"{name}.mp4").stat().st_size > 0


@pytest.mark.needs_ffmpeg
def test_ffmpeg_renders_colours_in_the_right_byte_order(tmp_path: Path) -> None:
    """``#RRGGBB`` -> ``&HBBGGRR``: a red style must burn in *red*, not blue."""
    if not ff.has_filter("subtitles"):
        pytest.skip("this ffmpeg build has no subtitles filter")
    style = replace(captions.get_style("clean"), animation="none", font_size=48,
                    primary_color="#FF0000", outline_color="#000000", outline=2.0, shadow=0.0,
                    margin_v=200, margin_h=10, max_words=2, max_chars=20)
    ass_path = captions.build([Word("red", 0.0, 0.5), Word("text", 0.5, 1.0)],
                              tmp_path / "red.ass", style=style, width=320, height=568)
    out = tmp_path / "red.mp4"
    _burn(ass_path, out)
    rgb = _rgb_frame(out, tmp_path / "red.png", at=0.2)
    red = int(((rgb[:, :, 0] > 150) & (rgb[:, :, 1] < 80) & (rgb[:, :, 2] < 80)).sum())
    blue = int(((rgb[:, :, 2] > 150) & (rgb[:, :, 0] < 80) & (rgb[:, :, 1] < 80)).sum())
    assert red > 100, "no red glyph pixels: the RGB->BGR conversion is wrong"
    assert blue == 0, "glyphs rendered blue: red and blue bytes are swapped"


@pytest.mark.needs_ffmpeg
def test_ffmpeg_opaque_box_is_drawn_in_back_color(tmp_path: Path) -> None:
    """A ``back_color`` slab must actually appear in that colour, alpha included."""
    if not ff.has_filter("subtitles"):
        pytest.skip("this ffmpeg build has no subtitles filter")
    style = replace(captions.get_style("boxed"), font_size=40, outline=6.0, shadow=0.0,
                    outline_color="#FF0000", back_color="#CC0000FF", primary_color="#FFFFFF",
                    margin_v=200, margin_h=20, max_words=2, max_chars=20, animation="none")
    ass_path = captions.build([Word("box", 0.0, 0.5), Word("test", 0.5, 1.0)],
                              tmp_path / "box.ass", style=style, width=320, height=568)
    out = tmp_path / "box.mp4"
    # green ground so a translucent blue box is still unambiguous
    _burn(ass_path, out, background="green", pix_fmt="yuv444p")
    rgb = _rgb_frame(out, tmp_path / "box.png", at=0.2)
    blue_box = int(((rgb[:, :, 2] > 90) & (rgb[:, :, 0] < 90)).sum())
    red_box = int(((rgb[:, :, 0] > 120) & (rgb[:, :, 1] < 90) & (rgb[:, :, 2] < 90)).sum())
    assert blue_box > 500, "the opaque box was not painted in back_color"
    assert red_box == 0, "the box was painted with outline_color instead of back_color"


@pytest.mark.needs_ffmpeg
def test_ffmpeg_long_lines_wrap_inside_the_canvas(tmp_path: Path) -> None:
    """A dense preset must wrap, not run off the edges (WrapStyle 2 would not)."""
    if not ff.has_filter("subtitles"):
        pytest.skip("this ffmpeg build has no subtitles filter")
    import numpy as np

    margin_h = 20
    style = replace(captions.get_style("podcast_bar"), back_color=None, font_size=40,
                    margin_v=100, margin_h=margin_h, max_words=9, max_chars=48, animation="none")
    words = words_from("an extremely long caption line that will certainly overflow", per_word=0.1)
    ass_path = captions.build(words, tmp_path / "wrap.ass", style=style, width=320, height=568)
    out = tmp_path / "wrap.mp4"
    _burn(ass_path, out)
    grey = _rgb_frame(out, tmp_path / "wrap.png", at=0.2).mean(axis=2)
    cols = np.where((grey > 150).any(axis=0))[0]
    rows = np.where((grey > 150).any(axis=1))[0]
    assert len(cols), "nothing was drawn"
    assert cols.min() >= margin_h - 8 and cols.max() <= 320 - margin_h + 8, "text ran past the margins"
    assert rows.max() - rows.min() > style.font_size * 0.9, "the long line did not wrap onto 2+ lines"
