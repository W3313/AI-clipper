"""Tests for :mod:`aiclipper.overlays`.

The Pillow backend is exercised end to end (it is always available); the
Chromium backend is only checked for parity when Playwright *and* a browser
binary are actually present -- these tests never download a browser.
"""

from __future__ import annotations

import dataclasses
import inspect
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from aiclipper import overlays
from aiclipper.config import Settings, get_settings
from aiclipper.errors import MissingDependency, OverlayError
from aiclipper.models import ChatMessage, ChatScript, RedditPost

CANVAS = (1080, 1920)
SMALL = (540, 960)

_CHROMIUM_OK: bool | None = None


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def chromium_or_skip(settings: Settings) -> None:
    """Skip unless Playwright is installed *and* a browser actually launches."""
    global _CHROMIUM_OK
    pytest.importorskip("playwright")
    if _CHROMIUM_OK is None:
        from playwright.sync_api import sync_playwright

        try:
            with sync_playwright() as pw:
                overlays._launch_chromium(pw, settings).close()
            _CHROMIUM_OK = True
        except Exception as exc:  # noqa: BLE001 - any launch failure means "not usable here"
            print(f"chromium unavailable: {exc}")
            _CHROMIUM_OK = False
    if not _CHROMIUM_OK:
        pytest.skip("chromium could not be launched in this environment")


def alpha_of(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        assert img.mode == "RGBA"
        return np.array(img)[..., 3]


def opaque_count(path: Path) -> int:
    return int((alpha_of(path) > 0).sum())


def conversation() -> ChatScript:
    return ChatScript(
        title="The keys",
        contact="Mara",
        avatar_initials="M",
        theme="classic",
        messages=[
            ChatMessage("Mara", "so did you actually go last night?", False),
            ChatMessage("me", "i did. and you will not believe who was there", True, typing=0.9),
            ChatMessage("Mara", "no way", False),
            ChatMessage("me", "yes way, she walked up and asked for the keys back", True, typing=1.1),
            ChatMessage("Mara", "what did you say??", False),
            ChatMessage("me", "nothing. i just smiled", True),
        ],
    )


def long_conversation(count: int = 16) -> ChatScript:
    return ChatScript(
        contact="",  # no header, so every opaque pixel belongs to the scrolling feed
        theme="dark",
        messages=[
            ChatMessage(
                "Sam" if i % 2 == 0 else "me",
                f"message {i} carrying enough words that the stack definitely overflows the safe area",
                outgoing=bool(i % 2),
            )
            for i in range(count)
        ],
    )


def a_post(**kw) -> RedditPost:
    data = dict(
        community="Nightshift",
        author="quietcorridor",
        title="My neighbour keeps leaving notes in my mailbox and one had my handwriting",
        body="I moved in six months ago. The first note said welcome. The second listed the exact "
             "hours I leave for work, which was strange enough to make me change my route.",
        upvotes=24800,
        comments=1320,
        theme="dark",
    )
    data.update(kw)
    return RedditPost(**data)


# --------------------------------------------------------------------------- #
# state planning
# --------------------------------------------------------------------------- #

def test_chat_states_insert_a_frame_before_every_typing_message():
    states = overlays.chat_states(conversation())
    assert [s.visible for s in states] == [1, 1, 2, 3, 3, 4, 5, 6]
    assert [s.typing for s in states] == [False, True, False, False, True, False, False, False]
    # the typing frames sit on the side of the message that is about to arrive
    assert [s.outgoing for s in states if s.typing] == [True, True]
    assert [s.index for s in states] == list(range(len(states)))


def test_chat_states_of_an_empty_script_is_empty():
    assert overlays.chat_states(ChatScript()) == []


def test_helpers_format_counts_and_initials():
    assert overlays._short_count(999) == "999"
    assert overlays._short_count(1200) == "1.2k"
    assert overlays._short_count(24800) == "24.8k"
    assert overlays._short_count(2_000_000) == "2M"
    assert overlays._short_count(999_999) == "1M"      # not "1000k"
    assert overlays._short_count(-1_500) == "-1.5k"
    assert overlays._short_count(0) == "0"
    assert overlays._initials("Mara Quinn") == "MQ"
    assert overlays._badge_letter("Nightshift") == "N"
    assert overlays._badge_letter("r/nightshift") == "N"


@pytest.mark.parametrize(
    "supplied, plain, line",
    [
        ("r/nightshift", "nightshift", "by nightshift"),
        ("/r/Nightshift", "Nightshift", "by Nightshift"),
        ("u/quietcorridor", "quietcorridor", "by quietcorridor"),
        ("/U/Quiet", "Quiet", "by Quiet"),
        ("Nightshift", "Nightshift", "by Nightshift"),
        ("  spaced  ", "spaced", "by spaced"),
        ("", "", ""),
    ],
)
def test_the_card_labels_its_own_fields(supplied, plain, line):
    """Hard rule 6: no "r/"/"u/" handle grammar borrowed from a real forum."""
    assert overlays.plain_name(supplied) == plain
    assert overlays.byline(supplied) == line


# --------------------------------------------------------------------------- #
# pillow backend, end to end
# --------------------------------------------------------------------------- #

def test_pillow_chat_renders_one_state_per_message_plus_typing(tmp_path: Path):
    script = conversation()
    images = overlays.render_chat(script, tmp_path, width=CANVAS[0], height=CANVAS[1], backend="pillow")

    assert len(images) == 8  # 6 messages + 2 typing frames
    assert [im.index for im in images] == list(range(8))
    assert [im.path.name for im in images] == [f"chat_{i:03d}.png" for i in range(8)]

    previous = -1
    for image in images:
        assert image.path.is_file()
        with Image.open(image.path) as img:
            assert img.size == CANVAS
            assert img.mode == "RGBA"
        assert (image.width, image.height) == CANVAS
        count = opaque_count(image.path)
        assert count > 0, "every state must draw something"
        assert count > previous, "each state adds a bubble, so it must add opaque pixels"
        previous = count


def test_pillow_chat_is_deterministic(tmp_path: Path):
    script = conversation()
    first = overlays.render_chat(script, tmp_path / "a", width=SMALL[0], height=SMALL[1], backend="pillow")
    second = overlays.render_chat(script, tmp_path / "b", width=SMALL[0], height=SMALL[1], backend="pillow")
    assert [p.path.read_bytes() for p in first] == [p.path.read_bytes() for p in second]


def test_long_message_wraps_and_stays_inside_the_canvas(tmp_path: Path):
    text = ("this is a single enormous message " * 12) + "Supercalifragilisticexpialidociousandthensome"
    script = ChatScript(contact="Kit", messages=[ChatMessage("Kit", text, False)])
    (image,) = overlays.render_chat(script, tmp_path, width=SMALL[0], height=SMALL[1], backend="pillow")

    with Image.open(image.path) as img:
        box = img.getbbox()
    assert box is not None
    left, top, right, bottom = box
    assert left >= 1 and top >= 1
    assert right <= SMALL[0] - 1 and bottom <= SMALL[1] - 1
    # the bubble respects the side margins rather than bleeding to the canvas edge
    assert left >= int(SMALL[0] * 0.03)
    assert right <= int(SMALL[0] * 0.97)


def test_overflowing_stack_scrolls_instead_of_growing(tmp_path: Path):
    script = long_conversation(16)
    images = overlays.render_chat(script, tmp_path, width=SMALL[0], height=SMALL[1], backend="pillow")
    assert len(images) == 16

    width, height = SMALL
    alpha = alpha_of(images[-1].path)
    rows = np.where(alpha.max(axis=1) > 0)[0]
    assert rows.size

    # the newest bubble is still parked low in the frame
    assert int((alpha[height // 2:] > 0).sum()) > 0
    assert rows.max() >= height * 0.6

    # nothing spills above the feed area: older bubbles are clipped, not shrunk
    assert rows.min() >= height * 0.10
    # ... and the top of the feed is faded out rather than cut hard
    top_band = alpha[rows.min(): rows.min() + max(2, height // 100)]
    assert 0 < top_band.max() < 255
    assert alpha.max() == 255

    # a mid conversation state and the final state occupy a similar amount of ink:
    # the stack scrolled rather than piling up forever
    assert opaque_count(images[-1].path) < 2 * opaque_count(images[9].path)


def test_empty_conversation_renders_nothing(tmp_path: Path):
    out = tmp_path / "states"
    assert overlays.render_chat(ChatScript(contact="Nobody"), out, width=320, height=568) == []
    assert list(out.glob("*.png")) == []


def test_forum_card_renders_with_opaque_content(tmp_path: Path):
    out = tmp_path / "card.png"
    image = overlays.render_forum_card(a_post(), out, width=SMALL[0], height=SMALL[1], backend="pillow")

    assert image.path == out and image.index == 0
    assert (image.width, image.height) == SMALL
    with Image.open(out) as img:
        assert img.size == SMALL
        assert img.mode == "RGBA"
    alpha = alpha_of(out)
    assert alpha.max() == 255
    # the card is centred: solid pixels around the middle, empty at the very top
    rows = np.where(alpha.max(axis=1) > 0)[0]
    assert rows.min() > 0 and rows.max() < SMALL[1] - 1
    middle = (rows.min() + rows.max()) // 2
    assert abs(middle - SMALL[1] // 2) < SMALL[1] * 0.12
    # side margins stay clear of the card itself (only the soft drop shadow reaches them)
    assert alpha[:, : SMALL[0] // 20].max() < 160


def _mask_of(path: Path, color: str, tol: int = 12) -> np.ndarray:
    want = np.array(overlays._hex_rgb(color), dtype=np.int16)
    with Image.open(path) as img:
        arr = np.array(img).astype(np.int16)
    return (np.abs(arr[..., :3] - want).max(axis=2) <= tol) & (arr[..., 3] > 200)


def _runs(flags: np.ndarray) -> list[tuple[int, int]]:
    """``[True, True, False, True]`` -> ``[(0, 2), (3, 4)]``."""
    out: list[tuple[int, int]] = []
    start: int | None = None
    for i, on in enumerate(list(flags) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            out.append((start, i))
            start = None
    return out


def test_a_site_prefix_is_never_drawn_on_the_card(tmp_path: Path):
    """Defect: the card printed "r/x" and "u/y" -- one real forum's handle
    grammar.  A supplied prefix is now stripped, so both render identically."""
    prefixed = overlays.render_forum_card(
        a_post(community="r/nightshift", author="u/quietcorridor"),
        tmp_path / "prefixed.png", width=SMALL[0], height=SMALL[1], backend="pillow")
    plain = overlays.render_forum_card(
        a_post(community="nightshift", author="quietcorridor"),
        tmp_path / "plain.png", width=SMALL[0], height=SMALL[1], backend="pillow")

    with Image.open(prefixed.path) as a, Image.open(plain.path) as b:
        assert np.array_equal(np.array(a), np.array(b))
    # and the byline really is drawn -- an authorless card has none of its ink
    bare = overlays.render_forum_card(a_post(author=""), tmp_path / "bare.png",
                                      width=SMALL[0], height=SMALL[1], backend="pillow")
    ink = overlays.FORUM_THEMES["dark"].meta_fg
    assert _mask_of(bare.path, ink).sum() < _mask_of(plain.path, ink).sum()


def test_the_engagement_marks_are_our_own_geometry(tmp_path: Path):
    """The upvote triangle is gone: the score mark is three rising bars and the
    replies mark two stacked bars, both of our own design."""
    theme = overlays.FORUM_THEMES["dark"]
    card = overlays.render_forum_card(a_post(), tmp_path / "marks.png",
                                      width=CANVAS[0], height=CANVAS[1], backend="pillow")
    fm = overlays._forum_metrics(theme, *CANVAS)

    # the chip row sits one card padding above the bottom edge of the card
    card_rows = np.nonzero(_mask_of(card.path, theme.card_bg).any(axis=1))[0]
    strip = slice(card_rows.max() - fm.pad - fm.chip_h, card_rows.max() - fm.pad)

    accent = _mask_of(card.path, theme.accent)[strip]
    bars = _runs(accent.any(axis=0))
    assert len(bars) == 3, f"score mark is not three bars: {bars}"
    heights = [int(accent[:, a:b].any(axis=1).sum()) for a, b in bars]
    assert heights[0] < heights[1] < heights[2], f"bars do not rise: {heights}"
    bottoms = [int(np.nonzero(accent[:, a:b].any(axis=1))[0].max()) for a, b in bars]
    assert max(bottoms) - min(bottoms) <= 2, f"bars are not bottom aligned: {bottoms}"

    # the replies mark leads the second chip, in the quiet chip colour
    chips = _runs(_mask_of(card.path, theme.chip_bg)[strip].any(axis=0))
    assert len(chips) == 2, f"expected two count chips: {chips}"
    left = chips[1][0] + fm.chip_pad
    chip = _mask_of(card.path, theme.chip_fg)[strip][:, left:left + fm.mark_w]
    lanes = _runs(chip.any(axis=1))
    assert len(lanes) == 2, f"replies mark is not two stacked bars: {lanes}"
    widths = [int(chip[a:b].any(axis=0).sum()) for a, b in lanes]
    assert widths[1] < widths[0], f"the lower reply bar should be shorter: {widths}"


def test_the_forum_template_carries_no_borrowed_iconography():
    css, html = _css("forum.css"), (overlays.TEMPLATE_DIR / "forum.html").read_text(encoding="utf-8")
    for gone in (".arrow", ".speech", "#author"):
        assert gone not in css and gone not in html, f"{gone} is still in the forum template"
    assert "arrow" not in html and "speech" not in html
    # the mark geometry is data in overlays.py; the stylesheet only echoes it
    for left, height in overlays._SCORE_BARS:
        assert left == 0.0 or f"{left:.2f}" in css
        assert height == 1.0 or f"{height:.2f}" in css
    for top, width in overlays._REPLY_BARS:
        assert f"{top:.2f}" in css
        assert width == 1.0 or f"{width:.2f}" in css
    assert f"{overlays._SCORE_BAR_W:.2f}" in css and f"{overlays._REPLY_BAR_H:.2f}" in css


def test_forum_card_handles_a_missing_body_and_huge_counts(tmp_path: Path):
    out = tmp_path / "bare.png"
    post = a_post(body="", title="Short one", upvotes=3_400_000, comments=0, theme="light")
    image = overlays.render_forum_card(post, out, width=SMALL[0], height=SMALL[1], backend="pillow")
    assert opaque_count(image.path) > 0


def test_every_theme_renders(tmp_path: Path):
    script = ChatScript(
        contact="Ray",
        messages=[ChatMessage("Ray", "are you up?", False), ChatMessage("me", "sadly yes", True)],
    )
    for name in overlays.CHAT_THEMES:
        script.theme = name
        images = overlays.render_chat(
            script, tmp_path / f"chat-{name}", width=360, height=640, backend="pillow"
        )
        assert len(images) == 2
        assert opaque_count(images[-1].path) > 0, f"chat theme {name} drew nothing"

    for name in overlays.FORUM_THEMES:
        out = tmp_path / f"forum-{name}.png"
        overlays.render_forum_card(a_post(theme=name), out, width=360, height=640, backend="pillow")
        assert opaque_count(out) > 0, f"forum theme {name} drew nothing"


def test_unknown_theme_falls_back_instead_of_raising(tmp_path: Path):
    script = ChatScript(contact="Ray", theme="no-such-theme",
                        messages=[ChatMessage("Ray", "hello", False)])
    images = overlays.render_chat(script, tmp_path, width=360, height=640, backend="pillow")
    assert opaque_count(images[0].path) > 0
    out = tmp_path / "f.png"
    overlays.render_forum_card(a_post(theme="nope"), out, width=360, height=640, backend="pillow")
    assert opaque_count(out) > 0


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #

def test_available_backends_always_offers_pillow():
    backends = overlays.available_backends()
    assert backends[-1] == "pillow"
    assert set(backends) <= set(overlays.BACKENDS)
    if len(backends) == 2:
        assert backends[0] == "chromium"


def test_unknown_backend_is_rejected(tmp_path: Path):
    with pytest.raises(OverlayError):
        overlays.render_chat(conversation(), tmp_path, width=320, height=568, backend="imagemagick")
    with pytest.raises(OverlayError):
        overlays.render_forum_card(a_post(), tmp_path / "x.png", width=320, height=568, backend="nope")


def test_explicit_chromium_without_playwright_raises_missing_dependency(tmp_path, monkeypatch):
    monkeypatch.setattr(overlays, "_have_playwright", lambda: False)
    with pytest.raises(MissingDependency):
        overlays.render_chat(conversation(), tmp_path, width=320, height=568, backend="chromium")


def test_automatic_backend_falls_back_to_pillow(tmp_path: Path, monkeypatch, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("no browser today")

    monkeypatch.setattr(overlays, "available_backends", lambda settings=None: ["chromium", "pillow"])
    monkeypatch.setattr(overlays, "_render_chat_chromium", boom)
    monkeypatch.setattr(overlays, "_render_forum_chromium", boom)

    with caplog.at_level("WARNING"):
        images = overlays.render_chat(conversation(), tmp_path, width=320, height=568)
        card = overlays.render_forum_card(a_post(), tmp_path / "c.png", width=320, height=568)

    assert len(images) == 8
    assert opaque_count(card.path) > 0
    assert sum("falling back to pillow" in r.message for r in caplog.records) == 2


def test_explicit_chromium_failure_is_not_silently_swallowed(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(overlays, "_have_playwright", lambda: True)
    monkeypatch.setattr(overlays, "_render_chat_chromium",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("crashed")))
    with pytest.raises(OverlayError, match="crashed"):
        overlays.render_chat(conversation(), tmp_path, width=320, height=568, backend="chromium")


def test_configured_chromium_path_is_honoured(monkeypatch, tmp_path: Path):
    fake = tmp_path / "my-chrome"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("AICLIP_CHROMIUM", str(fake))
    from aiclipper.config import reset_settings

    reset_settings()
    try:
        assert overlays._chromium_executable(get_settings()) == str(fake)
    finally:
        reset_settings()


def test_bad_canvas_size_is_rejected(tmp_path: Path):
    with pytest.raises(OverlayError):
        overlays.render_chat(conversation(), tmp_path, width=0, height=100)
    with pytest.raises(OverlayError):
        overlays.render_forum_card(a_post(), tmp_path / "x.png", width=100, height=-1)


# --------------------------------------------------------------------------- #
# chromium backend (skipped when no browser is installed)
# --------------------------------------------------------------------------- #

def test_chromium_chat_matches_pillow_state_for_state(tmp_path: Path):
    settings = get_settings()
    chromium_or_skip(settings)

    script = conversation()
    chrome = overlays.render_chat(script, tmp_path / "cr", width=SMALL[0], height=SMALL[1],
                                  backend="chromium")
    pillow = overlays.render_chat(script, tmp_path / "pil", width=SMALL[0], height=SMALL[1],
                                  backend="pillow")

    assert len(chrome) == len(pillow) == len(overlays.chat_states(script))
    assert [im.index for im in chrome] == [im.index for im in pillow]
    assert [im.path.name for im in chrome] == [im.path.name for im in pillow]

    previous = -1
    for image in chrome:
        with Image.open(image.path) as img:
            assert img.size == SMALL
            assert img.mode == "RGBA"
        count = opaque_count(image.path)
        assert count > 0
        assert count > previous
        previous = count

    # the newest bubble lands in the same part of the frame in both backends
    for shot in (chrome[-1], pillow[-1]):
        rows = np.where(alpha_of(shot.path).max(axis=1) > 0)[0]
        assert SMALL[1] * 0.55 < rows.max() < SMALL[1] * 0.85


def test_chromium_forum_card_matches_pillow(tmp_path: Path):
    settings = get_settings()
    chromium_or_skip(settings)

    post = a_post()
    chrome = overlays.render_forum_card(post, tmp_path / "cr.png", width=SMALL[0], height=SMALL[1],
                                        backend="chromium")
    pillow = overlays.render_forum_card(post, tmp_path / "pil.png", width=SMALL[0], height=SMALL[1],
                                        backend="pillow")
    for image in (chrome, pillow):
        with Image.open(image.path) as img:
            assert img.size == SMALL
            assert img.mode == "RGBA"
        alpha = alpha_of(image.path)
        assert alpha.max() == 255
        rows = np.where(alpha.max(axis=1) > 0)[0]
        assert abs((rows.min() + rows.max()) // 2 - SMALL[1] // 2) < SMALL[1] * 0.12


def test_templates_are_installed_and_self_contained():
    for name in ("chat.html", "chat.css", "forum.html", "forum.css"):
        assert (overlays.TEMPLATE_DIR / name).is_file()
    for html, css in (("chat.html", "chat.css"), ("forum.html", "forum.css")):
        page = overlays._page_html(html, css)
        assert "<style>" in page and "<!--STYLE-->" not in page
        assert "http://" not in page and "https://" not in page  # no network at render time


# --------------------------------------------------------------------------- #
# contract + import hygiene
# --------------------------------------------------------------------------- #

def test_public_contract_matches_the_architecture_spec():
    """Names, argument names, keyword-onlyness and defaults are binding."""
    assert [f.name for f in dataclasses.fields(overlays.OverlayImage)] == \
        ["path", "width", "height", "index"]
    image = overlays.OverlayImage(Path("x.png"), 10, 20, 3)
    assert (image.path, image.width, image.height, image.index) == (Path("x.png"), 10, 20, 3)

    chat = inspect.signature(overlays.render_chat)
    assert list(chat.parameters) == ["script", "out_dir", "width", "height", "settings", "backend",
                                     "header_state"]
    assert chat.parameters["out_dir"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    for name in ("width", "height", "settings", "backend", "header_state"):
        assert chat.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert chat.parameters["width"].default is inspect.Parameter.empty
    assert chat.parameters["height"].default is inspect.Parameter.empty
    assert chat.parameters["settings"].default is None
    assert chat.parameters["backend"].default is None
    # the extra header state is opt-in: today's callers (the texts pipeline maps
    # states to beats by index) keep exactly the states they had before
    assert chat.parameters["header_state"].default is False
    assert inspect.signature(overlays.chat_states).parameters["header_state"].default is False

    card = inspect.signature(overlays.render_forum_card)
    assert list(card.parameters) == ["post", "out_path", "width", "height", "settings", "backend"]
    for name in ("width", "height", "settings", "backend"):
        assert card.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
    assert card.parameters["settings"].default is None and card.parameters["backend"].default is None

    backends = inspect.signature(overlays.available_backends)
    assert list(backends.parameters) == ["settings"]
    assert backends.parameters["settings"].default is None


def test_importing_the_module_pulls_in_no_optional_dependency():
    """Hard rule 2: playwright is an extra, so it must not be imported eagerly."""
    code = (
        "import sys; import aiclipper.overlays as m; "
        "print(sorted(n for n in ('playwright', 'PIL', 'numpy', 'cv2') if n in sys.modules)); "
        "print(m.available_backends()[-1])"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    leaked, last = proc.stdout.strip().splitlines()
    assert leaked == "[]", f"overlays imported {leaked} at module scope"
    assert last == "pillow"


def test_render_results_are_reproducible_in_a_fresh_process(tmp_path: Path):
    """Hard rule 7: same seed, same bytes -- even across interpreter runs."""
    code = (
        "import hashlib, sys;"
        "from aiclipper import overlays;"
        "from aiclipper.config import Settings;"
        "from aiclipper.models import ChatMessage, ChatScript;"
        "s = ChatScript(contact='Mara Quinn', messages=[ChatMessage('Mara', 'hi there', False)]);"
        "im = overlays.render_chat(s, sys.argv[1], width=180, height=320, backend='pillow',"
        " settings=Settings(seed=7))[0];"
        "print(hashlib.sha256(im.path.read_bytes()).hexdigest())"
    )
    digests = []
    for run in ("a", "b"):
        proc = subprocess.run([sys.executable, "-c", code, str(tmp_path / run)],
                              capture_output=True, text=True, check=True)
        digests.append(proc.stdout.strip())
    assert digests[0] == digests[1]


def test_settings_seed_drives_the_avatar_colour(tmp_path: Path):
    script = ChatScript(contact="Mara Quinn", messages=[ChatMessage("Mara", "hi there", False)])
    seen = set()
    for seed in range(12):
        image = overlays.render_chat(script, tmp_path / str(seed), width=180, height=320,
                                     backend="pillow", settings=Settings(seed=seed))[0]
        seen.add(image.path.read_bytes())
    assert len(seen) > 1, "settings.seed is ignored: every seed produced the same pixels"


# --------------------------------------------------------------------------- #
# regressions
# --------------------------------------------------------------------------- #

def test_wrap_never_exceeds_the_line_budget():
    font = overlays._load_font(get_settings(), 24, bold=False)
    text = "short words then Supercalifragilisticexpialidociousandthensomemoreletters here"
    lines = overlays._wrap(text, font, 120.0)
    assert len(lines) > 1
    for line in lines:
        assert overlays._text_width(font, line) <= 120.0
    # the over-long word is split rather than dropped
    assert "".join(lines).replace(" ", "").startswith("shortwordsthenSupercalifragilistic")


def test_forum_card_with_a_runaway_title_stays_inside_the_canvas(tmp_path: Path):
    """An LLM-written title must never push the card (or its chips) off-canvas."""
    post = a_post(title="A very long title word " * 30, body="filler sentence. " * 60)
    image = overlays.render_forum_card(post, tmp_path / "long.png", width=SMALL[0], height=SMALL[1],
                                       backend="pillow")
    alpha = alpha_of(image.path)
    rows = np.where(alpha.max(axis=1) > 0)[0]
    assert rows.min() > 0 and rows.max() < SMALL[1] - 1
    # the card body itself (not just the soft shadow) is fully inside the frame
    solid = np.where(alpha.max(axis=1) == 255)[0]
    assert solid.min() > 0 and solid.max() < SMALL[1] - 1


def test_stale_frames_from_an_earlier_render_are_swept(tmp_path: Path):
    """Workspaces are reused, so a shorter conversation must not leave old frames."""
    out = tmp_path / "states"
    long_script = ChatScript(contact="A", messages=[ChatMessage("A", f"m{i}", bool(i % 2))
                                                    for i in range(5)])
    overlays.render_chat(long_script, out, width=200, height=356, backend="pillow")
    keep = out / "notes.txt"
    keep.write_text("mine")
    assert len(list(out.glob("chat_*.png"))) == 5

    short = ChatScript(contact="A", messages=[ChatMessage("A", "only one", False)])
    images = overlays.render_chat(short, out, width=200, height=356, backend="pillow")
    assert [p.name for p in sorted(out.glob("chat_*.png"))] == ["chat_000.png"]
    assert images[0].path.is_file() and keep.is_file()

    assert overlays.render_chat(ChatScript(), out, width=200, height=356, backend="pillow") == []
    assert list(out.glob("chat_*.png")) == []


def test_overlays_are_transparent_outside_the_artwork(tmp_path: Path):
    """The renderer composites these over video, so the background must be alpha 0."""
    script = ChatScript(contact="Ray", messages=[ChatMessage("Ray", "are you up?", False)])
    image = overlays.render_chat(script, tmp_path / "c", width=360, height=640, backend="pillow")[0]
    alpha = alpha_of(image.path)
    assert alpha[0, 0] == alpha[0, -1] == alpha[-1, 0] == alpha[-1, -1] == 0
    assert alpha.max() == 255
    assert int((alpha == 0).sum()) > alpha.size * 0.5

    card = overlays.render_forum_card(a_post(), tmp_path / "f.png", width=360, height=640,
                                      backend="pillow")
    calpha = alpha_of(card.path)
    assert calpha[0, 0] == calpha[-1, -1] == 0


def test_typing_indicator_before_the_very_first_message(tmp_path: Path):
    script = ChatScript(contact="Ada", messages=[ChatMessage("Ada", "guess what", False, typing=1.2)])
    states = overlays.chat_states(script)
    assert [(s.visible, s.typing, s.outgoing) for s in states] == [(0, True, False), (1, False, False)]

    images = overlays.render_chat(script, tmp_path, width=360, height=640, backend="pillow")
    assert len(images) == 2

    # the typing frame draws an indicator bubble in the feed, not just the header
    alpha = alpha_of(images[0].path)
    feed = alpha[int(640 * 0.35):]
    assert feed.max() == 255, "the typing state drew nothing below the contact header"
    rows = np.where(feed.max(axis=1) > 0)[0]
    assert rows.max() > 640 * 0.2  # anchored low, like a real bubble
    cols = np.where(feed.max(axis=0) > 0)[0]
    assert cols.min() < 360 * 0.25, "an incoming indicator belongs on the left"

    # a non-positive typing hint adds no state at all
    quiet = ChatScript(messages=[ChatMessage("A", "x", False, typing=-1.0)])
    assert [(s.visible, s.typing) for s in overlays.chat_states(quiet)] == [(1, False)]


def test_chat_state_indices_stay_contiguous_with_many_typing_hints():
    script = ChatScript(messages=[ChatMessage("A", f"m{i}", bool(i % 2), typing=0.4)
                                  for i in range(9)])
    states = overlays.chat_states(script)
    assert len(states) == 18
    assert [s.index for s in states] == list(range(18))
    assert [s.visible for s in states] == [0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9]


def test_chromium_forum_card_with_a_runaway_title_stays_inside_the_canvas(tmp_path: Path):
    chromium_or_skip(get_settings())
    post = a_post(title="A very long title word " * 30, body="filler sentence. " * 60)
    image = overlays.render_forum_card(post, tmp_path / "long.png", width=SMALL[0], height=SMALL[1],
                                       backend="chromium")
    rows = np.where(alpha_of(image.path).max(axis=1) > 0)[0]
    assert rows.min() > 0 and rows.max() < SMALL[1] - 1


# --------------------------------------------------------------------------- #
# one theme, one definition (chromium/pillow parity)
# --------------------------------------------------------------------------- #

def _css(name: str) -> str:
    return (overlays.TEMPLATE_DIR / name).read_text(encoding="utf-8")


def test_the_stylesheets_carry_no_palette_and_no_measurements():
    """Regression: the CSS used to hold a second copy of every theme.

    Two copies meant the documented "silently falls back to pillow" silently
    changed the artwork, so the values now live only in :data:`overlays.THEMES`.
    """
    for name in ("chat.css", "forum.css"):
        text = re.sub(r"/\*.*?\*/", " ", _css(name), flags=re.S)
        values = " ".join(re.findall(r":([^;{}]*)[;}]", text))
        assert not re.search(r"#[0-9A-Fa-f]{3,8}\b", values), f"{name} still hardcodes a colour"
        # the only literal colours left are the black/transparent stops of the feed mask
        leftovers = [c for c in re.findall(r"rgba?\([^)]*\)", values)
                     if not re.fullmatch(r"rgba\(0, 0, 0, [01]\)", c)]
        assert leftovers == [], f"{name} still hardcodes {leftovers}"
        assert not re.search(r"\d\s*(vw|vh)\b", values), f"{name} still hardcodes a canvas measurement"
        assert "data-theme" not in text, f"{name} still switches palettes itself"


def test_every_css_variable_the_templates_use_is_supplied_by_python():
    """Nothing may fall back to a browser default the Pillow backend cannot see."""
    chat_vars = set(overlays._chat_css_vars(
        overlays.CHAT_THEMES["classic"],
        overlays._chat_metrics(overlays.CHAT_THEMES["classic"], *CANVAS),
        "#123456", "",
    ))
    forum_vars = set(overlays._forum_css_vars(
        overlays.FORUM_THEMES["dark"],
        overlays._forum_metrics(overlays.FORUM_THEMES["dark"], *CANVAS),
        "",
    ))
    for name, supplied in (("chat.css", chat_vars), ("forum.css", forum_vars)):
        used = set(re.findall(r"var\((--[\w-]+)", _css(name)))
        assert used, f"{name} uses no custom properties at all"
        assert used <= supplied, f"{name} uses undefined {sorted(used - supplied)}"


def test_the_injected_css_variables_are_the_pillow_metrics():
    """The browser is handed the numbers Pillow lays out with, not its own copy."""
    theme = overlays.CHAT_THEMES["dark"]
    m = overlays._chat_metrics(theme, *CANVAS)
    css = overlays._chat_css_vars(theme, m, "#0A0B0C", "")
    assert css["--fs"] == f"{m.font_size}px"
    assert css["--radius"] == f"{m.radius}px"
    assert css["--margin"] == f"{m.margin}px"
    assert css["--avatar-d"] == f"{m.avatar_d}px"
    assert css["--bubble-max"] == f"{m.max_bubble_w}px"
    assert css["--in-bg"] == theme.in_bg and css["--out-bg"] == theme.out_bg
    assert css["--edge"] == overlays._css_rgba(theme.outline, theme.outline_alpha)
    assert css["--avatar-bg"] == "#0A0B0C"   # the caller's resolved colour, not theme.avatar_bg

    forum = overlays.FORUM_THEMES["paper"]
    fm = overlays._forum_metrics(forum, *CANVAS)
    fcss = overlays._forum_css_vars(forum, fm, "")
    assert fcss["--card-bg"] == forum.card_bg and fcss["--accent"] == forum.accent
    assert fcss["--card-radius"] == f"{fm.radius}px"
    assert fcss["--title-clamp"] == str(fm.title_lines)


def test_the_chromium_payload_carries_the_avatar_colour_pillow_would_draw():
    """Defect: pillow picked a seeded accent, the browser always used theme.avatar_bg."""
    settings = get_settings()
    script = ChatScript(contact="Kit", theme="classic",
                        messages=[ChatMessage("Kit", "hey", False)])
    payload = overlays._chat_payload(script, *CANVAS, settings)
    assert payload["vars"]["--avatar-bg"] == overlays._avatar_color(script, settings)


def _centroid(path: Path, color: str, tol: int = 10) -> tuple[float, float] | None:
    """Centre of mass of the pixels matching ``color``, or ``None`` if unpainted."""
    want = np.array(overlays._hex_rgb(color), dtype=np.int16)
    with Image.open(path) as img:
        arr = np.array(img).astype(np.int16)
    hit = (np.abs(arr[..., :3] - want).max(axis=2) <= tol) & (arr[..., 3] > 200)
    if not hit.any():
        return None
    ys, xs = np.nonzero(hit)
    return (float(xs.mean()), float(ys.mean()))


def test_chromium_and_pillow_paint_the_same_theme(tmp_path: Path):
    """Both backends must agree on colour *and* placement, within a few pixels.

    "Kit" is the reproducer from the audit: the seeded avatar accent resolves to
    a teal, which the browser used to ignore in favour of the stylesheet's blue.
    """
    settings = get_settings()
    chromium_or_skip(settings)

    script = ChatScript(
        contact="Kit", avatar_initials="K", theme="classic",
        messages=[
            ChatMessage("Kit", "so did you actually go last night?", False),
            ChatMessage("me", "i did. you will not believe who was there", True),
            ChatMessage("Kit", "no way", False),
        ],
    )
    theme = overlays.CHAT_THEMES["classic"]
    shots = {
        backend: overlays.render_chat(script, tmp_path / backend, width=SMALL[0], height=SMALL[1],
                                      backend=backend)[-1].path
        for backend in ("chromium", "pillow")
    }
    slack_x, slack_y = SMALL[0] * 0.02, SMALL[1] * 0.02

    landmarks = {
        "avatar": overlays._avatar_color(script, settings),
        "incoming bubble": theme.in_bg,
        "outgoing bubble": theme.out_bg,
    }
    for what, color in landmarks.items():
        spots = {b: _centroid(p, color) for b, p in shots.items()}
        for backend, spot in spots.items():
            assert spot is not None, f"{backend} never painted the {what} in {color}"
        (cx, cy), (px, py) = spots["chromium"], spots["pillow"]
        assert abs(cx - px) < slack_x, f"{what} sits at a different x: {cx} vs {px}"
        assert abs(cy - py) < slack_y, f"{what} sits at a different y: {cy} vs {py}"

    # ... and the whole composition occupies the same band of the canvas
    boxes = {}
    for backend, path in shots.items():
        alpha = alpha_of(path)
        rows = np.where(alpha.max(axis=1) > 0)[0]
        cols = np.where(alpha.max(axis=0) > 0)[0]
        boxes[backend] = (cols.min(), rows.min(), cols.max(), rows.max())
    for chrome, pil, slack in zip(boxes["chromium"], boxes["pillow"],
                                  (slack_x, slack_y, slack_x, slack_y), strict=True):
        assert abs(int(chrome) - int(pil)) < slack


def test_chromium_and_pillow_paint_the_same_forum_card(tmp_path: Path):
    settings = get_settings()
    chromium_or_skip(settings)

    post = a_post(theme="light")
    theme = overlays.FORUM_THEMES["light"]
    shots = {
        backend: overlays.render_forum_card(post, tmp_path / f"{backend}.png", width=SMALL[0],
                                            height=SMALL[1], backend=backend).path
        for backend in ("chromium", "pillow")
    }
    slack_x, slack_y = SMALL[0] * 0.03, SMALL[1] * 0.03
    for what, color in (("card", theme.card_bg), ("accent", theme.accent), ("chip", theme.chip_bg)):
        spots = {b: _centroid(p, color) for b, p in shots.items()}
        for backend, spot in spots.items():
            assert spot is not None, f"{backend} never painted the {what} in {color}"
        (cx, cy), (px, py) = spots["chromium"], spots["pillow"]
        assert abs(cx - px) < slack_x and abs(cy - py) < slack_y, f"{what}: {spots}"


# --------------------------------------------------------------------------- #
# the leading header-only state
# --------------------------------------------------------------------------- #

def test_header_state_is_opt_in_and_adds_one_leading_frame():
    script = conversation()
    plain = overlays.chat_states(script)
    withhead = overlays.chat_states(script, header_state=True)

    assert [s.visible for s in plain] == [1, 1, 2, 3, 3, 4, 5, 6]
    assert len(withhead) == len(plain) + 1
    assert withhead[0] == overlays.ChatState(index=0, visible=0, typing=False, outgoing=False)
    # every message state keeps its identity, one place further along
    assert [(s.visible, s.typing) for s in withhead[1:]] == [(s.visible, s.typing) for s in plain]
    assert [s.index for s in withhead] == list(range(len(withhead)))
    # an empty conversation still has nothing to show
    assert overlays.chat_states(ChatScript(contact="Nobody"), header_state=True) == []


@pytest.mark.parametrize("backend", ["pillow", "chromium"])
def test_the_first_frame_is_the_chrome_with_no_bubbles(tmp_path: Path, backend: str):
    """Defect: the video opened on a bare background during the first delay."""
    if backend == "chromium":
        chromium_or_skip(get_settings())
    script = conversation()
    images = overlays.render_chat(script, tmp_path / backend, width=SMALL[0], height=SMALL[1],
                                  backend=backend, header_state=True)

    assert len(images) == len(overlays.chat_states(script)) + 1
    assert [im.index for im in images] == list(range(len(images)))
    assert [im.path.name for im in images] == [f"chat_{i:03d}.png" for i in range(len(images))]

    header = alpha_of(images[0].path)
    assert (header > 0).sum() > 0, "the first frame must not be empty"
    # ...and all of its ink is the header: nothing is painted down in the feed
    feed_top = overlays._chat_metrics(overlays.CHAT_THEMES[script.theme], *SMALL).content_top
    assert int((header[feed_top:] > 0).sum()) == 0
    # the bubbles start one frame later and still grow one state at a time
    counts = [opaque_count(im.path) for im in images]
    assert counts == sorted(counts)
    assert counts[1] > counts[0]


def test_header_state_keeps_the_message_frames_identical(tmp_path: Path):
    """Turning the flag on must only *prepend*, never re-render the rest."""
    script = conversation()
    plain = overlays.render_chat(script, tmp_path / "off", width=SMALL[0], height=SMALL[1],
                                 backend="pillow")
    shifted = overlays.render_chat(script, tmp_path / "on", width=SMALL[0], height=SMALL[1],
                                   backend="pillow", header_state=True)
    assert [p.path.read_bytes() for p in plain] == [p.path.read_bytes() for p in shifted[1:]]


# --------------------------------------------------------------------------- #
# font resolution
# --------------------------------------------------------------------------- #

def test_font_resolution_prefers_a_colour_emoji_font(tmp_path: Path, monkeypatch):
    """A colour face wins over a monochrome one -- see the module docstring."""
    fonts = tmp_path / "assets" / "fonts"
    fonts.mkdir(parents=True)
    (fonts / "NotoEmoji-Regular.ttf").write_bytes(b"\0")
    settings = Settings(assets_dir=tmp_path / "assets")
    monkeypatch.setattr(overlays, "_emoji_font_cache", {})
    monkeypatch.setattr(overlays, "_FONT_DIRS", ())          # only our sandbox counts
    assert overlays._emoji_font(settings) == (str(fonts / "NotoEmoji-Regular.ttf"), False)

    (fonts / "NotoColorEmoji.ttf").write_bytes(b"\0")
    monkeypatch.setattr(overlays, "_emoji_font_cache", {})
    assert overlays._emoji_font(settings) == (str(fonts / "NotoColorEmoji.ttf"), True)
    assert overlays._emoji_family(settings) == "Noto Color Emoji"

    # the emoji face is a *fallback*: putting it first would hand the browser its
    # very wide space glyph for ordinary text
    stack = overlays._font_stack("Noto Color Emoji")
    assert stack.endswith('"Noto Color Emoji"')
    assert stack.index("DejaVu Sans") < stack.index("Noto Color Emoji")
    assert overlays._font_stack("") == overlays._font_stack()


def test_no_emoji_font_at_all_is_not_an_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(overlays, "_emoji_font_cache", {})
    monkeypatch.setattr(overlays, "_FONT_DIRS", ())
    settings = Settings(assets_dir=tmp_path / "nope")
    assert overlays._emoji_font(settings) is None
    assert overlays._emoji_family(settings) == ""


def test_chromium_and_pillow_forum_shadows_reach_the_same_distance(tmp_path: Path):
    """The card's drop shadow must not be twice as wide in one backend.

    CSS ``box-shadow``'s blur radius is *twice* the Gaussian sigma, so handing
    the raw blur to ``ImageFilter.GaussianBlur`` used to spread the Pillow card's
    shadow about 2x further than Chromium's -- a soft halo instead of a drop.
    """
    settings = get_settings()
    chromium_or_skip(settings)

    post = a_post()
    made = {
        name: overlays.render_forum_card(post, tmp_path / f"{name}.png",
                                         width=CANVAS[0], height=CANVAS[1], backend=name)
        for name in ("chromium", "pillow")
    }
    boxes = {}
    for name, image in made.items():
        rows = np.where(alpha_of(image.path).max(axis=1) > 0)[0]
        cols = np.where(alpha_of(image.path).max(axis=0) > 0)[0]
        boxes[name] = (int(cols.min()), int(rows.min()), int(cols.max()), int(rows.max()))
    chrome, pillow = boxes["chromium"], boxes["pillow"]
    for axis, (a, b) in enumerate(zip(chrome, pillow, strict=True)):
        assert abs(a - b) <= 12, f"painted extent {axis} differs: {chrome} vs {pillow}"
