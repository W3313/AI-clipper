"""Overlay artwork: chat-conversation states and forum story cards.

Two renderers produce the same thing so the rest of the engine never has to
care which one ran:

* ``chromium`` -- Playwright drives a real browser over the HTML/CSS templates
  in :mod:`aiclipper.templates`.  The browser is launched **once** per call; a
  JavaScript entry point is called for every conversation state and the page is
  screenshotted with ``omit_background=True`` so the PNG keeps its alpha.
* ``pillow`` -- a pure-Python layout engine.  Always available, no browser, no
  system dependencies beyond Pillow itself.  It draws at 2x and box-filters the
  result down so rounded bubbles and glyph edges stay clean.

Both backends emit exactly the same number of images, in the same order, at
exactly the requested canvas size, so a pipeline can switch backends without
re-timing anything.

**One theme, one definition.**  Every colour, radius, spacing and font size
lives exactly once, in :data:`THEMES` (i.e. in :data:`CHAT_THEMES` and
:data:`FORUM_THEMES`), as unitless fractions of the canvas or of the base font
size.  :func:`_chat_metrics` / :func:`_forum_metrics` turn a theme plus a
canvas size into concrete pixels, and *both* backends consume that one result:
Pillow draws with it directly, and the Chromium backend injects it into the
page as CSS custom properties before screenshotting.  The stylesheets therefore
contain no palette and no magic numbers -- if they did, "silently falls back to
pillow" would silently change how the video looks.

``render_chat`` turns a :class:`~aiclipper.models.ChatScript` into one
transparent PNG per *conversation state*: image ``i`` shows messages ``0..i``.
A message with ``typing > 0`` additionally gets a state just before it that
shows a typing indicator on that message's side of the screen.  With
``header_state=True`` an extra leading state is emitted that shows the chrome
(avatar + contact name) with no bubbles at all, so a caller can hold something
on screen from ``t=0`` instead of opening on a bare background.  The newest
bubble is anchored low in the safe area and older bubbles scroll up (the
topmost fading out) once the stack overflows -- text size never shrinks.

Fonts are resolved from ``settings.fonts_dir`` and the usual system directories;
a **colour** emoji font (``NotoColorEmoji``, ``Apple Color Emoji``,
``seguiemj``) is preferred over a monochrome one when the machine has one.
Known limitation: Chromium falls back per glyph, so it picks the colour font up
automatically, but Pillow/FreeType has no per-glyph font fallback -- it draws
every character from the single face it was handed.  On a machine whose text
font only carries outline emoji (DejaVu Sans does), the Pillow backend will
therefore still draw emoji as monochrome outlines.  Installing a colour emoji
font does not change that for Pillow; only the Chromium backend benefits.

All artwork here is original: generic bubbles, a generic card, our own
palettes.  Nothing imitates the trade dress of any real product.
"""

from __future__ import annotations

import logging
import os
import random
import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import Settings, get_settings
from .errors import MissingDependency, OverlayError
from .models import ChatMessage, ChatScript, RedditPost

if TYPE_CHECKING:  # pragma: no cover - typing only
    from PIL.Image import Image as PilImage
    from PIL.ImageDraw import ImageDraw as PilDraw

log = logging.getLogger(__name__)

__all__ = [
    "OverlayImage",
    "ChatState",
    "ChatTheme",
    "ForumTheme",
    "CHAT_THEMES",
    "FORUM_THEMES",
    "THEMES",
    "TEMPLATE_DIR",
    "BACKENDS",
    "chat_states",
    "plain_name",
    "byline",
    "render_chat",
    "render_forum_card",
    "available_backends",
]

TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"
BACKENDS = ("chromium", "pillow")


# --------------------------------------------------------------------------- #
# public dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class OverlayImage:
    """One rendered overlay PNG (RGBA, exactly canvas sized)."""

    path: Path
    width: int
    height: int
    index: int


@dataclass(frozen=True)
class ChatState:
    """One frame of the conversation.

    ``visible`` messages are fully drawn; when ``typing`` is true an extra
    indicator bubble is drawn on the ``outgoing`` side for the message that is
    about to arrive.
    """

    index: int
    visible: int
    typing: bool = False
    outgoing: bool = False


@dataclass(frozen=True)
class ChatTheme:
    """One chat look: colours **and** geometry, defined here and nowhere else.

    Colours are hex strings.  Geometry is unitless so it scales to any canvas:
    ``*_w`` fractions are of the canvas width, ``*_h`` fractions of the canvas
    height and ``*_em`` multiples of the base font size.  :func:`_chat_metrics`
    resolves a theme against a canvas into pixels, and both backends render
    from that one result.
    """

    name: str
    in_bg: str
    in_fg: str
    out_bg: str
    out_fg: str
    header_fg: str
    meta_fg: str
    avatar_bg: str
    avatar_fg: str
    dot: str
    outline: str = "#000000"
    outline_alpha: float = 0.16

    # -- geometry ---------------------------------------------------------- #
    font_w: float = 0.0390          # base font size
    margin_w: float = 0.0550        # side margin of the bubble column
    avatar_w: float = 0.1050        # avatar disc diameter
    header_top_h: float = 0.0450    # top of the avatar disc
    header_gap_h: float = 0.0135    # avatar -> contact name
    feed_gap_h: float = 0.0190      # contact name -> top of the feed
    feed_bottom_h: float = 0.7180   # bottom edge of the newest bubble
    fade_h: float = 0.0700          # length of the fade at the top of the feed
    row_gap_em: float = 0.46        # vertical gap between bubbles
    pad_x_em: float = 0.66
    pad_y_em: float = 0.46
    radius_em: float = 0.88
    line_em: float = 1.26           # bubble line height
    label_em: float = 0.62          # sender caption font size
    label_gap_em: float = 0.22      # sender caption -> bubble
    contact_em: float = 0.92        # contact name font size
    initials_em: float = 1.15       # avatar initials font size
    bubble_max: float = 0.74        # of the feed width
    bubble_min_em: float = 2.60
    dot_em: float = 0.34            # typing dot diameter
    dot_gap_em: float = 0.34        # gap between typing dots
    edge_em: float = 0.024          # bubble hairline


@dataclass(frozen=True)
class ForumTheme:
    """One forum-card look: colours **and** geometry, defined here and nowhere else.

    Same convention as :class:`ChatTheme`; :func:`_forum_metrics` resolves it.
    """

    name: str
    card_bg: str
    title_fg: str
    body_fg: str
    meta_fg: str
    accent: str
    chip_bg: str
    chip_fg: str
    badge_fg: str
    shadow: str = "#000000"
    shadow_alpha: float = 0.45
    outline: str = "#000000"
    outline_alpha: float = 0.10
    rule_alpha: float = 0.16          # hairline above the engagement row

    # -- geometry ---------------------------------------------------------- #
    font_w: float = 0.0300          # base font size
    margin_w: float = 0.0700        # gap between card and canvas edge
    stage_pad_h: float = 0.0450     # minimum gap above/below the card
    pad_w: float = 0.0520           # card padding
    radius_w: float = 0.0380        # card corner radius
    badge_em: float = 2.00          # community badge diameter
    meta_line: float = 1.30
    title_em: float = 1.62
    title_line: float = 1.20
    body_em: float = 1.06
    body_line: float = 1.34
    gap_title_em: float = 0.95      # meta row -> title
    gap_body_em: float = 0.90       # title -> excerpt
    gap_row_em: float = 1.15        # excerpt -> hairline
    rule_gap_em: float = 0.95       # hairline -> engagement row
    chip_h_em: float = 2.30
    chip_pad_em: float = 0.70
    chip_gap_em: float = 0.70
    icon_em: float = 0.66           # engagement mark unit (mark box is 1.5x this)
    mark_gap_em: float = 0.50       # mark -> its count
    badge_text_em: float = 1.05     # community badge letter
    meta_gap_em: float = 0.60       # badge -> community name
    author_em: float = 0.94         # byline font size (quieter than the community)
    author_gap_em: float = 0.20     # community name -> byline
    shadow_drop_em: float = 0.55
    shadow_blur_em: float = 0.90
    edge_em: float = 0.06
    title_lines: int = 8            # wrapped lines kept before eliding
    body_lines: int = 7


CHAT_THEMES: dict[str, ChatTheme] = {
    "classic": ChatTheme(
        name="classic",
        in_bg="#ECEBF2", in_fg="#1B1B26",
        out_bg="#4C5FD7", out_fg="#FFFFFF",
        header_fg="#FFFFFF", meta_fg="#D4D6E4",
        avatar_bg="#4C5FD7", avatar_fg="#FFFFFF", dot="#6C6E80",
    ),
    "dark": ChatTheme(
        name="dark",
        in_bg="#23262F", in_fg="#E8EAF2",
        out_bg="#6F5AE6", out_fg="#FFFFFF",
        header_fg="#F2F3F8", meta_fg="#9DA2B4",
        avatar_bg="#6F5AE6", avatar_fg="#FFFFFF", dot="#9DA2B4",
        outline="#000000", outline_alpha=0.35,
    ),
    "mint": ChatTheme(
        name="mint",
        in_bg="#E7F5EE", in_fg="#0F3A2C",
        out_bg="#17A97C", out_fg="#FFFFFF",
        header_fg="#EAFBF3", meta_fg="#BFE6D6",
        avatar_bg="#17A97C", avatar_fg="#FFFFFF", dot="#4F8C79",
    ),
    "sunset": ChatTheme(
        name="sunset",
        in_bg="#FFF0E4", in_fg="#3A1F14",
        out_bg="#E2653C", out_fg="#FFFFFF",
        header_fg="#FFF3EA", meta_fg="#F0C4AC",
        avatar_bg="#E2653C", avatar_fg="#FFFFFF", dot="#A9765E",
    ),
    "mono": ChatTheme(
        name="mono",
        in_bg="#F2F2F2", in_fg="#141414",
        out_bg="#1E1E1E", out_fg="#F7F7F7",
        header_fg="#FFFFFF", meta_fg="#C9C9C9",
        avatar_bg="#1E1E1E", avatar_fg="#FFFFFF", dot="#7A7A7A",
    ),
}

FORUM_THEMES: dict[str, ForumTheme] = {
    "dark": ForumTheme(
        name="dark",
        card_bg="#1A1C24", title_fg="#F4F5FA", body_fg="#C3C7D6", meta_fg="#8A90A6",
        accent="#6F8CFF", chip_bg="#262A36", chip_fg="#D8DCEA", badge_fg="#101219",
        shadow="#000000", shadow_alpha=0.55, outline="#FFFFFF", outline_alpha=0.08,
    ),
    "light": ForumTheme(
        name="light",
        card_bg="#FFFFFF", title_fg="#16181F", body_fg="#43474F", meta_fg="#7A8090",
        accent="#3B54C4", chip_bg="#EFF1F6", chip_fg="#3A3F4C", badge_fg="#FFFFFF",
        shadow="#101322", shadow_alpha=0.40, outline="#101322", outline_alpha=0.10,
    ),
    "paper": ForumTheme(
        name="paper",
        card_bg="#FBF6EC", title_fg="#2A2218", body_fg="#544838", meta_fg="#8C7B63",
        accent="#B2622A", chip_bg="#F0E6D4", chip_fg="#54462F", badge_fg="#FFFFFF",
        shadow="#3A2C18", shadow_alpha=0.38, outline="#3A2C18", outline_alpha=0.12,
    ),
}

#: The single source of truth for every overlay look, keyed by surface.  Both
#: the Pillow backend and the stylesheets behind the Chromium backend read from
#: here -- the CSS files carry no palette and no measurements of their own.
THEMES: dict[str, dict[str, ChatTheme] | dict[str, ForumTheme]] = {
    "chat": CHAT_THEMES,
    "forum": FORUM_THEMES,
}

#: Opacity of the three typing dots, oldest first -- shared by both backends.
_TYPING_DOT_ALPHAS = (0.55, 0.775, 1.0)

DEFAULT_CHAT_THEME = "classic"
DEFAULT_FORUM_THEME = "dark"


def _chat_theme(name: str | None) -> ChatTheme:
    key = (name or "").strip().lower()
    if key in CHAT_THEMES:
        return CHAT_THEMES[key]
    if key:
        log.warning("unknown chat theme %r, using %r", name, DEFAULT_CHAT_THEME)
    return CHAT_THEMES[DEFAULT_CHAT_THEME]


def _forum_theme(name: str | None) -> ForumTheme:
    key = (name or "").strip().lower()
    if key in FORUM_THEMES:
        return FORUM_THEMES[key]
    if key:
        log.warning("unknown forum theme %r, using %r", name, DEFAULT_FORUM_THEME)
    return FORUM_THEMES[DEFAULT_FORUM_THEME]


# --------------------------------------------------------------------------- #
# state planning (shared by both backends)
# --------------------------------------------------------------------------- #

def chat_states(script: ChatScript, *, header_state: bool = False) -> list[ChatState]:
    """Plan the conversation states for ``script``.

    One state per message, plus one extra *before* every message that carries a
    ``typing`` hint.  Both backends consume this list, which is what keeps them
    in lockstep.

    With ``header_state=True`` the list gains a leading state at index ``0``
    with ``visible=0``: the header (avatar + contact name) and an empty feed.
    It exists so a caller can put *something* on screen from ``t=0`` instead of
    opening on a bare background while the first message waits out its delay.
    It shifts every following index by one, so pass the same value here and to
    :func:`render_chat` and read the mapping from ``state.visible`` rather than
    from the position in the list::

        state.visible == 0                -> header only, no messages
        state.visible == i + 1, typing=0  -> messages 0..i are on screen
        state.visible == i,     typing=1  -> messages 0..i-1 plus a typing dot
                                             bubble for the message about to land
    """
    states: list[ChatState] = []
    if not script.messages:  # an empty conversation has nothing to hold, header or not
        return states
    if header_state:
        states.append(ChatState(index=0, visible=0, typing=False, outgoing=False))
    for i, msg in enumerate(script.messages):
        if getattr(msg, "typing", 0.0) and msg.typing > 0:
            states.append(ChatState(index=len(states), visible=i, typing=True, outgoing=bool(msg.outgoing)))
        states.append(ChatState(index=len(states), visible=i + 1, typing=False, outgoing=bool(msg.outgoing)))
    return states


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _hex_rgb(color: str) -> tuple[int, int, int]:
    raw = (color or "#000000").strip().lstrip("#")
    if len(raw) == 3:
        raw = "".join(c * 2 for c in raw)
    if len(raw) != 6:
        raw = "000000"
    try:
        return (int(raw[0:2], 16), int(raw[2:4], 16), int(raw[4:6], 16))
    except ValueError:  # pragma: no cover - defensive
        return (0, 0, 0)


def _rgba(color: str, alpha: float = 1.0) -> tuple[int, int, int, int]:
    r, g, b = _hex_rgb(color)
    return (r, g, b, max(0, min(255, int(round(alpha * 255)))))


def _font_stack(emoji_family: str = "") -> str:
    """CSS font stack for the Chromium page.

    The colour emoji face goes **last**: it is a fallback for glyphs the text
    fonts do not carry, and putting it first would hand the browser that font's
    (very wide) space and digit glyphs for ordinary text.
    """
    stack = '"DejaVu Sans", "Liberation Sans", "Noto Sans", Arial, sans-serif'
    return f'{stack}, "{emoji_family}"' if emoji_family else stack


def _css_rgba(color: str, alpha: float = 1.0) -> str:
    """``"#000000", 0.16`` -> ``"rgba(0, 0, 0, 0.16)"`` -- the same colour Pillow gets."""
    r, g, b = _hex_rgb(color)
    return f"rgba({r}, {g}, {b}, {max(0.0, min(1.0, float(alpha))):.4f})"


def _initials(text: str, limit: int = 2) -> str:
    parts = [p for p in (text or "").replace("_", " ").split() if p]
    letters = [p[0] for p in parts if p[0].isalnum()]
    if not letters:
        letters = [c for c in (text or "") if c.isalnum()][:limit]
    return "".join(letters[:limit]).upper() or "?"


#: A borrowed ``r/``/``u/`` (or ``/r/``, ``/u/``) handle prefix from one specific
#: real forum.  Our card labels its own fields, so it is stripped for display --
#: callers may still store whatever string they like on the post.
_SITE_PREFIX_RE = re.compile(r"^/?[ru]/", re.IGNORECASE)


def plain_name(value: str) -> str:
    """``"r/nightshift"`` -> ``"nightshift"``; anything else is passed through."""
    return _SITE_PREFIX_RE.sub("", (value or "").strip(), count=1).strip()


def byline(author: str) -> str:
    """Our own author labelling: ``"u/quietcorridor"`` -> ``"by quietcorridor"``."""
    name = plain_name(author)
    return f"by {name}" if name else ""


def _badge_letter(community: str) -> str:
    """First meaningful letter of a community name (``"r/nightshift"`` -> ``"N"``)."""
    name = plain_name(community)
    if "/" in name:
        name = name.rsplit("/", 1)[-1]
    return _initials(name or community or "?", 1)


def _ellipsize(line: str) -> str:
    """Trim trailing punctuation/ellipsis off ``line`` and append a single ellipsis."""
    return line.rstrip("…").rstrip(" .,;:") + "…"


def _short_count(value: int) -> str:
    n = int(value)
    sign = "-" if n < 0 else ""
    n = abs(n)
    if n < 1_000:
        return f"{sign}{n}"
    unit, step = ("M", 1_000_000) if n >= 1_000_000 else ("k", 1_000)
    scaled = round(n / step, 1)
    if unit == "k" and scaled >= 1000:  # 999_999 reads as "1M", never "1000k"
        unit, scaled = "M", round(n / 1_000_000, 1)
    return f"{sign}{scaled:.1f}{unit}".replace(f".0{unit}", unit)


# --------------------------------------------------------------------------- #
# font resolution
# --------------------------------------------------------------------------- #

_FONT_DIRS = (
    "/usr/share/fonts/truetype/dejavu",
    "/usr/share/fonts/truetype/liberation",
    "/usr/share/fonts/truetype/freefont",
    "/usr/share/fonts/TTF",
    "/usr/share/fonts",
    "/usr/local/share/fonts",
    "/Library/Fonts",
    "/System/Library/Fonts",
    "C:/Windows/Fonts",
)

_REGULAR_NAMES = (
    "DejaVuSans.ttf", "LiberationSans-Regular.ttf", "FreeSans.ttf",
    "NotoSans-Regular.ttf", "Arial.ttf", "Helvetica.ttc", "arial.ttf",
)
_BOLD_NAMES = (
    "DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "FreeSansBold.ttf",
    "NotoSans-Bold.ttf", "Arial Bold.ttf", "arialbd.ttf",
)
#: Colour emoji faces, best first.  Preferred over the monochrome outline emoji
#: that ship inside the text fonts above -- see the module docstring for why
#: only the Chromium backend can actually use them.
_EMOJI_NAMES = (
    "NotoColorEmoji.ttf", "NotoColorEmoji.ttc", "Noto Color Emoji.ttf",
    "AppleColorEmoji.ttf", "Apple Color Emoji.ttc", "seguiemj.ttf",
    "TwemojiMozilla.ttf", "EmojiOneColor.otf",
)
#: Monochrome emoji faces, used only when no colour font exists at all.
_EMOJI_FALLBACK_NAMES = ("NotoEmoji-Regular.ttf", "Symbola.ttf", "OpenSansEmoji.ttf")

_font_file_cache: dict[tuple[str, bool], str | None] = {}
_emoji_font_cache: dict[str, tuple[str, bool] | None] = {}


def _search_fonts(settings: Settings, names: tuple[str, ...]) -> str | None:
    """First of ``names`` that exists under ``settings.fonts_dir`` or a system dir."""
    roots: list[Path] = [settings.fonts_dir, *(Path(d) for d in _FONT_DIRS)]
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:  # pragma: no cover - unreadable mount
            continue
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return str(candidate)
        for pattern in ("*/{}", "*/*/{}"):  # e.g. /usr/share/fonts/truetype/noto/<name>
            for name in names:
                try:
                    nested = sorted(root.glob(pattern.format(name)))
                except OSError:  # pragma: no cover - unreadable mount
                    nested = []
                if nested:
                    return str(nested[0])
    return None


def _emoji_font(settings: Settings) -> tuple[str, bool] | None:
    """Best emoji face on this machine as ``(path, is_colour)``, colour first.

    Returns ``None`` when the machine has no dedicated emoji font at all, in
    which case emoji fall back to whatever outline glyphs the text font carries.
    """
    key = str(settings.fonts_dir)
    if key in _emoji_font_cache:
        return _emoji_font_cache[key]
    found = _search_fonts(settings, _EMOJI_NAMES)
    result: tuple[str, bool] | None = (found, True) if found else None
    if result is None:
        mono = _search_fonts(settings, _EMOJI_FALLBACK_NAMES)
        result = (mono, False) if mono else None
    _emoji_font_cache[key] = result
    return result


def _emoji_family(settings: Settings) -> str:
    """CSS family name for the preferred emoji face (``""`` when there is none)."""
    found = _emoji_font(settings)
    if not found:
        return ""
    stem = Path(found[0]).stem
    # "NotoColorEmoji" -> "Noto Color Emoji", "seguiemj" -> "Segoe UI Emoji"
    special = {"seguiemj": "Segoe UI Emoji", "TwemojiMozilla": "Twemoji Mozilla"}
    if stem in special:
        return special[stem]
    out: list[str] = []
    for char in stem.replace("-", " ").replace("_", " "):
        if char.isupper() and out and out[-1] not in " ":
            out.append(" ")
        out.append(char)
    return "".join(out).strip()


def _font_file(settings: Settings, bold: bool) -> str | None:
    """Locate a real TTF, preferring ``settings.fonts_dir``."""
    key = (str(settings.fonts_dir), bold)
    if key in _font_file_cache:
        return _font_file_cache[key]
    names = _BOLD_NAMES if bold else _REGULAR_NAMES
    found: str | None = _search_fonts(settings, names)
    if found is None:
        # last resort: any ttf sitting in the project font directory
        try:
            pool = sorted(settings.fonts_dir.glob("*.ttf")) if settings.fonts_dir.is_dir() else []
        except OSError:  # pragma: no cover - unreadable mount
            pool = []
        picked = [p for p in pool if ("Bold" in p.name) == bold] or pool
        found = str(picked[0]) if picked else None
    _font_file_cache[key] = found
    return found


def _load_font(settings: Settings, size: int, *, bold: bool = False) -> Any:
    from PIL import ImageFont

    size = max(8, int(size))
    path = _font_file(settings, bold)
    if path:
        try:
            return ImageFont.truetype(path, size)
        except OSError:  # pragma: no cover - corrupt font file
            log.warning("could not load font %s, falling back to the bitmap default", path)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # pragma: no cover - Pillow < 10.1
        return ImageFont.load_default()


def _line_height(font: Any) -> int:
    try:
        ascent, descent = font.getmetrics()
        return int(ascent + descent)
    except (AttributeError, OSError):  # pragma: no cover - bitmap fallback
        box = font.getbbox("AÁgjq")
        return int(box[3] - box[1]) + 4


def _text_width(font: Any, text: str) -> float:
    try:
        return float(font.getlength(text))
    except (AttributeError, OSError):  # pragma: no cover - bitmap fallback
        box = font.getbbox(text)
        return float(box[2] - box[0])


def _fit_word(word: str, font: Any, max_w: float) -> list[str]:
    if _text_width(font, word) <= max_w:
        return [word]
    pieces: list[str] = []
    cur = ""
    for ch in word:
        if cur and _text_width(font, cur + ch) > max_w:
            pieces.append(cur)
            cur = ch
        else:
            cur += ch
    if cur:
        pieces.append(cur)
    return pieces


def _wrap(text: str, font: Any, max_w: float) -> list[str]:
    """Greedy word wrap; over-long words are broken by character."""
    out: list[str] = []
    paragraphs = (text or "").splitlines() or [""]
    for para in paragraphs:
        words = para.split()
        if not words:
            out.append("")
            continue
        cur = ""
        for word in words:
            for piece in _fit_word(word, font, max_w):
                candidate = f"{cur} {piece}" if cur else piece
                if cur and _text_width(font, candidate) > max_w:
                    out.append(cur)
                    cur = piece
                else:
                    cur = candidate
        if cur:
            out.append(cur)
    return out or [""]


def _downsample(img: PilImage, factor: int) -> PilImage:
    """Box-filter ``img`` down by ``factor``, premultiplying so edges stay clean.

    Averaging straight (non-premultiplied) RGBA would drag the black of the
    transparent background into every bubble edge, so alpha is folded in before
    the reduction and divided back out afterwards.
    """
    if factor <= 1:
        return img
    import numpy as np
    from PIL import Image, ImageChops

    red, green, blue, alpha = img.split()
    premul = Image.merge("RGBA", (
        ImageChops.multiply(red, alpha),
        ImageChops.multiply(green, alpha),
        ImageChops.multiply(blue, alpha),
        alpha,
    ))
    small = np.asarray(premul.reduce(factor), dtype=np.uint16)
    out_alpha = small[..., 3]
    denom = np.maximum(out_alpha, 1)[..., None]
    rgb = np.minimum((small[..., :3] * 255 + denom // 2) // denom, 255)
    out = np.dstack([rgb.astype(np.uint8), out_alpha.astype(np.uint8)])
    return Image.fromarray(out, "RGBA")


# --------------------------------------------------------------------------- #
# chat layout (pillow backend)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _ChatMetrics:
    """A :class:`ChatTheme` resolved against one canvas, in device pixels.

    Built once by :func:`_chat_metrics` and consumed by *both* backends: Pillow
    lays out with these numbers, and :func:`_chat_css_vars` hands the very same
    numbers to the browser as CSS custom properties.
    """

    w: int
    h: int
    font_size: int
    margin: int
    header_top: int
    avatar_d: int
    initials_size: int
    header_gap: int
    contact_size: int
    contact_h: int
    content_top: int
    anchor_y: int
    gap: int
    pad_x: int
    pad_y: int
    radius: int
    line_h: int
    label_size: int
    label_line: int
    label_h: int
    name_h: int
    outline: int
    max_bubble_w: int
    min_bubble_w: int
    typing_w: int
    typing_h: int
    dot_d: int
    dot_gap: int
    fade_span: int


def _chat_metrics(theme: ChatTheme, width: int, height: int, scale: int = 1) -> _ChatMetrics:
    """Resolve ``theme``'s geometry against a ``width``x``height`` canvas."""
    w, h = int(width) * scale, int(height) * scale
    fs = max(10, round(w * theme.font_w))
    pad_x = round(fs * theme.pad_x_em)
    pad_y = round(fs * theme.pad_y_em)
    line_h = round(fs * theme.line_em)
    label_size = max(8, round(fs * theme.label_em))
    label_h = round(label_size * theme.line_em) + round(fs * theme.label_gap_em)
    contact_size = max(9, round(fs * theme.contact_em))
    contact_h = round(contact_size * theme.line_em)
    avatar_d = round(w * theme.avatar_w)
    header_top = round(h * theme.header_top_h)
    header_gap = round(h * theme.header_gap_h)
    margin = round(w * theme.margin_w)
    dot_d = max(2, round(fs * theme.dot_em))
    dot_gap = max(1, round(fs * theme.dot_gap_em))
    return _ChatMetrics(
        w=w, h=h, font_size=fs, margin=margin,
        header_top=header_top, avatar_d=avatar_d,
        initials_size=max(10, round(fs * theme.initials_em)),
        header_gap=header_gap, contact_size=contact_size, contact_h=contact_h,
        content_top=header_top + avatar_d + header_gap + contact_h + round(h * theme.feed_gap_h),
        anchor_y=round(h * theme.feed_bottom_h),
        gap=round(fs * theme.row_gap_em),
        pad_x=pad_x, pad_y=pad_y,
        radius=round(fs * theme.radius_em),
        line_h=line_h,
        label_size=label_size, label_line=round(label_size * theme.line_em),
        label_h=label_h, name_h=label_h,
        outline=max(1, round(fs * theme.edge_em)),
        max_bubble_w=round((w - 2 * margin) * theme.bubble_max),
        min_bubble_w=round(fs * theme.bubble_min_em),
        typing_w=2 * pad_x + 3 * dot_d + 2 * dot_gap,
        typing_h=2 * pad_y + line_h,
        dot_d=dot_d, dot_gap=dot_gap,
        fade_span=round(h * theme.fade_h),
    )


def _chat_css_vars(theme: ChatTheme, m: _ChatMetrics, avatar_bg: str, emoji_family: str) -> dict[str, str]:
    """The same theme + metrics, as CSS custom properties for the Chromium page."""
    return {
        "--font-stack": _font_stack(emoji_family),
        "--in-bg": theme.in_bg,
        "--in-fg": theme.in_fg,
        "--out-bg": theme.out_bg,
        "--out-fg": theme.out_fg,
        "--header-fg": theme.header_fg,
        "--meta-fg": theme.meta_fg,
        "--avatar-bg": avatar_bg,
        "--avatar-fg": theme.avatar_fg,
        "--dot": theme.dot,
        "--edge": _css_rgba(theme.outline, theme.outline_alpha),
        "--fs": f"{m.font_size}px",
        "--line-h": f"{m.line_h}px",
        "--pad-x": f"{m.pad_x}px",
        "--pad-y": f"{m.pad_y}px",
        "--radius": f"{m.radius}px",
        "--margin": f"{m.margin}px",
        "--row-gap": f"{m.gap}px",
        "--label-fs": f"{m.label_size}px",
        "--label-h": f"{m.label_h}px",
        "--label-line": f"{m.label_line}px",
        "--label-gap": f"{m.label_h - m.label_line}px",
        "--contact-fs": f"{m.contact_size}px",
        "--contact-h": f"{m.contact_h}px",
        "--initials-fs": f"{m.initials_size}px",
        "--avatar-d": f"{m.avatar_d}px",
        "--head-top": f"{m.header_top}px",
        "--head-gap": f"{m.header_gap}px",
        "--feed-top": f"{m.content_top}px",
        "--feed-h": f"{max(1, m.anchor_y - m.content_top)}px",
        "--fade": f"{m.fade_span}px",
        "--bubble-max": f"{m.max_bubble_w}px",
        "--bubble-min": f"{m.min_bubble_w}px",
        "--typing-w": f"{m.typing_w}px",
        "--typing-h": f"{m.typing_h}px",
        "--dot-d": f"{m.dot_d}px",
        "--dot-gap": f"{m.dot_gap}px",
        "--edge-w": f"{m.outline}px",
        "--dot-a1": f"{_TYPING_DOT_ALPHAS[0]}",
        "--dot-a2": f"{_TYPING_DOT_ALPHAS[1]}",
        "--dot-a3": f"{_TYPING_DOT_ALPHAS[2]}",
    }


@dataclass
class _Bubble:
    lines: list[str] = field(default_factory=list)
    w: int = 0
    body_h: int = 0
    h: int = 0
    outgoing: bool = False
    label: str = ""
    typing: bool = False


class _ChatCtx:
    """Fonts + metrics + palette for one Pillow chat render."""

    def __init__(self, script: ChatScript, width: int, height: int, settings: Settings, scale: int):
        self.theme = _chat_theme(script.theme)
        self.m = _chat_metrics(self.theme, width, height, scale)
        m = self.m
        self.font = _load_font(settings, m.font_size, bold=False)
        self.name_font = _load_font(settings, m.label_size, bold=True)
        self.header_font = _load_font(settings, m.contact_size, bold=True)
        self.avatar_font = _load_font(settings, m.initials_size, bold=True)
        # CSS centres each glyph inside its line box (half-leading); Pillow draws
        # from the ascender, so shift by the same half-leading to line up.
        self.text_dy = max(0, (m.line_h - _line_height(self.font)) // 2)
        self.label_dy = max(0, (m.label_line - _line_height(self.name_font)) // 2)
        self.contact_dy = max(0, (m.contact_h - _line_height(self.header_font)) // 2)

    # -- measuring --------------------------------------------------------- #
    def measure(self, msg: ChatMessage, label: str) -> _Bubble:
        m = self.m
        inner = m.max_bubble_w - 2 * m.pad_x
        lines = _wrap(msg.text or "", self.font, inner)
        if len(lines) > 1:
            # A wrapped bubble fills the column.  CSS shrink-to-fit resolves to
            # max-width as soon as the text no longer fits on one line, so this
            # is what the Chromium backend does -- match it or the two drift.
            w = m.max_bubble_w
        else:
            text_w = max((_text_width(self.font, ln) for ln in lines), default=0.0)
            w = int(max(m.min_bubble_w, min(m.max_bubble_w, round(text_w) + 2 * m.pad_x)))
        body_h = 2 * m.pad_y + len(lines) * m.line_h
        head = m.name_h if label else 0
        return _Bubble(lines=lines, w=w, body_h=body_h, h=body_h + head,
                       outgoing=bool(msg.outgoing), label=label)

    def typing_bubble(self, outgoing: bool) -> _Bubble:
        m = self.m
        return _Bubble(lines=[], w=m.typing_w, body_h=m.typing_h, h=m.typing_h,
                       outgoing=outgoing, label="", typing=True)


def _sender_labels(script: ChatScript) -> list[str]:
    """Name captions: shown on an incoming bubble that starts a new run."""
    labels: list[str] = []
    previous: tuple[str, bool] | None = None
    for msg in script.messages:
        key = (msg.sender or "", bool(msg.outgoing))
        show = (not msg.outgoing) and key != previous and bool((msg.sender or "").strip())
        labels.append(msg.sender.strip() if show else "")
        previous = key
    return labels


def _avatar_color(script: ChatScript, settings: Settings) -> str:
    theme = _chat_theme(script.theme)
    palette = [theme.avatar_bg, theme.out_bg, "#8A6BE0", "#2F9BB5", "#D4735B", "#3F8F6A"]
    rng = random.Random(f"{settings.seed}:{script.contact}:{theme.name}")
    return rng.choice(palette)


def _draw_chat_header(d: PilDraw, ctx: _ChatCtx, script: ChatScript, settings: Settings) -> None:
    m, theme = ctx.m, ctx.theme
    if not (script.contact or "").strip():
        return
    cx = m.w // 2
    top = m.header_top
    r = m.avatar_d // 2
    d.ellipse((cx - r, top, cx + r, top + m.avatar_d), fill=_rgba(_avatar_color(script, settings)))
    initials = (script.avatar_initials or "").strip() or _initials(script.contact)
    d.text((cx, top + r), initials, font=ctx.avatar_font, fill=_rgba(theme.avatar_fg), anchor="mm")
    d.text((cx, top + m.avatar_d + m.header_gap + ctx.contact_dy), script.contact.strip(),
           font=ctx.header_font, fill=_rgba(theme.header_fg), anchor="ma")


def _draw_bubble(d: PilDraw, ctx: _ChatCtx, bubble: _Bubble, top: int) -> None:
    m, theme = ctx.m, ctx.theme
    x = (m.w - m.margin - bubble.w) if bubble.outgoing else m.margin
    y = top
    if bubble.label:
        label_x = x + bubble.w - m.pad_x if bubble.outgoing else x + m.pad_x
        d.text((label_x, y + ctx.label_dy), bubble.label, font=ctx.name_font,
               fill=_rgba(theme.meta_fg), anchor="ra" if bubble.outgoing else "la")
        y += m.name_h
    bg = theme.out_bg if bubble.outgoing else theme.in_bg
    fg = theme.out_fg if bubble.outgoing else theme.in_fg
    d.rounded_rectangle(
        (x, y, x + bubble.w, y + bubble.body_h),
        radius=m.radius,
        fill=_rgba(bg),
        outline=_rgba(theme.outline, theme.outline_alpha),
        width=m.outline,
    )
    if bubble.typing:
        cy = y + bubble.body_h // 2
        step = m.dot_d + m.dot_gap
        left = x + bubble.w // 2 - (3 * m.dot_d + 2 * m.dot_gap) // 2
        radius = m.dot_d / 2.0
        dot = theme.out_fg if bubble.outgoing else theme.dot
        for i in range(3):
            px = left + i * step + radius
            d.ellipse((px - radius, cy - radius, px + radius, cy + radius),
                      fill=_rgba(dot, _TYPING_DOT_ALPHAS[i]))
        return
    ty = y + m.pad_y + ctx.text_dy
    for line in bubble.lines:
        d.text((x + m.pad_x, ty), line, font=ctx.font, fill=_rgba(fg))
        ty += m.line_h


def _feed_mask(width: int, height: int, top: int, fade: int) -> Any:
    """Alpha ramp for the scrolling feed: hidden above ``top``, fading in over ``fade`` px."""
    import numpy as np
    from PIL import Image

    ramp = np.clip((np.arange(height, dtype=np.float32) - top) / max(1, fade), 0.0, 1.0)
    column = (ramp * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(np.repeat(column[:, None], width, axis=1), "L")


def _render_chat_pillow(
    script: ChatScript,
    states: list[ChatState],
    out_dir: Path,
    width: int,
    height: int,
    settings: Settings,
) -> list[OverlayImage]:
    from PIL import Image, ImageChops, ImageDraw

    scale = 2 if width * height <= 1080 * 1920 else 1
    ctx = _ChatCtx(script, width, height, settings, scale)
    m = ctx.m
    labels = _sender_labels(script)
    bubbles = [ctx.measure(msg, labels[i]) for i, msg in enumerate(script.messages)]
    typing = {True: ctx.typing_bubble(True), False: ctx.typing_bubble(False)}
    mask = _feed_mask(m.w, m.h, m.content_top, m.fade_span)

    images: list[OverlayImage] = []
    for state in states:
        img = Image.new("RGBA", (m.w, m.h), (0, 0, 0, 0))
        _draw_chat_header(ImageDraw.Draw(img, "RGBA"), ctx, script, settings)

        stack = list(bubbles[: state.visible])
        if state.typing:
            stack.append(typing[state.outgoing])

        placed: list[tuple[_Bubble, int]] = []
        y = m.anchor_y
        for bubble in reversed(stack):
            top = y - bubble.h
            placed.append((bubble, top))
            y = top - m.gap

        feed = Image.new("RGBA", (m.w, m.h), (0, 0, 0, 0))
        fd = ImageDraw.Draw(feed, "RGBA")
        for bubble, top in reversed(placed):
            if top + bubble.h <= m.content_top:
                continue  # scrolled fully out of the safe area
            _draw_bubble(fd, ctx, bubble, top)
        # older bubbles are clipped at the top of the feed and fade as they leave
        feed.putalpha(ImageChops.multiply(feed.getchannel("A"), mask))
        img.alpha_composite(feed)

        out = _downsample(img, scale)
        path = out_dir / f"chat_{state.index:03d}.png"
        out.save(path)
        images.append(OverlayImage(path=path, width=width, height=height, index=state.index))
    return images


# --------------------------------------------------------------------------- #
# forum card metrics (shared by both backends)
# --------------------------------------------------------------------------- #

# Our own engagement marks, as fractions of the mark box -- one definition for
# both backends.  ``score`` is three rising bars, ``replies`` two stacked bars;
# deliberately abstract geometry, borrowed from no existing service's icon set.
_MARK_W_UNITS = 1.5             # mark box width, in ``icon`` units
_MARK_H_UNITS = 1.4             # mark box height, in ``icon`` units
#: score mark: ``(left, height)`` per bar, fractions of the mark box.
_SCORE_BARS = ((0.00, 0.40), (0.38, 0.68), (0.76, 1.00))
_SCORE_BAR_W = 0.24             # bar width, fraction of the mark box width
#: replies mark: ``(top, width)`` per bar, fractions of the mark box.
_REPLY_BARS = ((0.16, 1.00), (0.60, 0.55))
_REPLY_BAR_H = 0.24             # bar height, fraction of the mark box height


@dataclass(frozen=True)
class _ForumMetrics:
    """A :class:`ForumTheme` resolved against one canvas, in device pixels."""

    w: int
    h: int
    base: int
    margin: int
    card_w: int
    pad: int
    inner: int
    radius: int
    stage_pad: int
    badge_d: int
    badge_size: int
    meta_size: int
    meta_line: int
    author_size: int
    author_line: int
    author_gap: int
    meta_gap: int
    meta_h: int
    title_size: int
    title_lh: int
    body_size: int
    body_lh: int
    gap_title: int
    gap_body: int
    gap_row: int
    rule_gap: int
    chip_h: int
    chip_pad: int
    chip_gap: int
    icon: int
    mark_w: int
    mark_h: int
    mark_gap: int
    edge: int
    shadow_drop: int
    shadow_blur: int
    title_lines: int
    body_lines: int


def _forum_metrics(theme: ForumTheme, width: int, height: int, scale: int = 1) -> _ForumMetrics:
    """Resolve ``theme``'s geometry against a ``width``x``height`` canvas."""
    w, h = int(width) * scale, int(height) * scale
    base = max(10, round(w * theme.font_w))
    margin = round(w * theme.margin_w)
    pad = round(w * theme.pad_w)
    card_w = w - 2 * margin
    badge_d = round(base * theme.badge_em)
    meta_line = round(base * theme.meta_line)
    author_size = max(8, round(base * theme.author_em))
    author_line = round(author_size * theme.meta_line)
    author_gap = round(base * theme.author_gap_em)
    meta_h = max(badge_d, meta_line + author_gap + author_line)
    icon = round(base * theme.icon_em)
    return _ForumMetrics(
        w=w, h=h, base=base, margin=margin, card_w=card_w, pad=pad,
        inner=card_w - 2 * pad,
        radius=round(w * theme.radius_w),
        stage_pad=round(h * theme.stage_pad_h),
        badge_d=badge_d,
        badge_size=max(8, round(base * theme.badge_text_em)),
        meta_size=base, meta_line=meta_line,
        author_size=author_size, author_line=author_line, author_gap=author_gap,
        meta_gap=round(base * theme.meta_gap_em), meta_h=meta_h,
        title_size=max(10, round(base * theme.title_em)),
        title_lh=round(base * theme.title_em * theme.title_line),
        body_size=max(9, round(base * theme.body_em)),
        body_lh=round(base * theme.body_em * theme.body_line),
        gap_title=round(base * theme.gap_title_em),
        gap_body=round(base * theme.gap_body_em),
        gap_row=round(base * theme.gap_row_em),
        rule_gap=round(base * theme.rule_gap_em),
        chip_h=round(base * theme.chip_h_em),
        chip_pad=round(base * theme.chip_pad_em),
        chip_gap=round(base * theme.chip_gap_em),
        icon=icon,
        mark_w=round(icon * _MARK_W_UNITS),
        mark_h=round(icon * _MARK_H_UNITS),
        mark_gap=round(base * theme.mark_gap_em),
        edge=max(1, round(base * theme.edge_em)),
        shadow_drop=round(base * theme.shadow_drop_em),
        shadow_blur=round(base * theme.shadow_blur_em),
        title_lines=int(theme.title_lines),
        body_lines=int(theme.body_lines),
    )


def _forum_css_vars(theme: ForumTheme, m: _ForumMetrics, emoji_family: str) -> dict[str, str]:
    """The same theme + metrics, as CSS custom properties for the Chromium page."""
    return {
        "--font-stack": _font_stack(emoji_family),
        "--card-bg": theme.card_bg,
        "--title-fg": theme.title_fg,
        "--body-fg": theme.body_fg,
        "--meta-fg": theme.meta_fg,
        "--accent": theme.accent,
        "--chip-bg": theme.chip_bg,
        "--chip-fg": theme.chip_fg,
        "--badge-fg": theme.badge_fg,
        "--shadow": _css_rgba(theme.shadow, theme.shadow_alpha),
        "--edge": _css_rgba(theme.outline, theme.outline_alpha),
        "--rule": _css_rgba(theme.outline, theme.rule_alpha),
        "--fs": f"{m.meta_size}px",
        "--meta-line": f"{m.meta_line}px",
        "--author-fs": f"{m.author_size}px",
        "--author-line": f"{m.author_line}px",
        "--author-gap": f"{m.author_gap}px",
        "--meta-gap": f"{m.meta_gap}px",
        "--meta-h": f"{m.meta_h}px",
        "--badge-d": f"{m.badge_d}px",
        "--badge-fs": f"{m.badge_size}px",
        "--title-fs": f"{m.title_size}px",
        "--title-lh": f"{m.title_lh}px",
        "--body-fs": f"{m.body_size}px",
        "--body-lh": f"{m.body_lh}px",
        "--gap-title": f"{m.gap_title}px",
        "--gap-body": f"{m.gap_body}px",
        "--gap-row": f"{m.gap_row}px",
        "--rule-gap": f"{m.rule_gap}px",
        "--chip-h": f"{m.chip_h}px",
        "--chip-pad": f"{m.chip_pad}px",
        "--chip-gap": f"{m.chip_gap}px",
        "--mark-w": f"{m.mark_w}px",
        "--mark-h": f"{m.mark_h}px",
        "--mark-gap": f"{m.mark_gap}px",
        "--edge-w": f"{m.edge}px",
        "--card-margin": f"{m.margin}px",
        "--card-pad": f"{m.pad}px",
        "--card-radius": f"{m.radius}px",
        "--stage-pad": f"{m.stage_pad}px",
        "--shadow-y": f"{m.shadow_drop}px",
        "--shadow-blur": f"{m.shadow_blur}px",
        "--title-clamp": f"{m.title_lines}",
        "--body-clamp": f"{m.body_lines}",
    }


# --------------------------------------------------------------------------- #
# forum card (pillow backend)
# --------------------------------------------------------------------------- #

def _draw_score_mark(d: PilDraw, x: float, y: float, w: float, h: float, ink: tuple[int, ...]) -> None:
    """Our score mark: three rising bars, bottom aligned in the ``w`` x ``h`` box."""
    bar_w = w * _SCORE_BAR_W
    for left, tall in _SCORE_BARS:
        bx = x + w * left
        d.rounded_rectangle((bx, y + h * (1.0 - tall), bx + bar_w, y + h),
                            radius=bar_w / 2.0, fill=ink)


def _draw_replies_mark(d: PilDraw, x: float, y: float, w: float, h: float, ink: tuple[int, ...]) -> None:
    """Our replies mark: two stacked bars, centred in the ``w`` x ``h`` box."""
    bar_h = h * _REPLY_BAR_H
    for top, wide in _REPLY_BARS:
        by = y + h * top
        d.rounded_rectangle((x, by, x + w * wide, by + bar_h), radius=bar_h / 2.0, fill=ink)


def _render_forum_pillow(
    post: RedditPost,
    out_path: Path,
    width: int,
    height: int,
    settings: Settings,
) -> OverlayImage:
    from PIL import Image, ImageDraw, ImageFilter

    scale = 2 if width * height <= 1080 * 1920 else 1
    theme = _forum_theme(post.theme)
    fm = _forum_metrics(theme, width, height, scale)
    w, h = fm.w, fm.h

    f_meta = _load_font(settings, fm.author_size, bold=False)
    f_meta_b = _load_font(settings, fm.meta_size, bold=True)
    f_count = _load_font(settings, fm.meta_size, bold=True)
    f_title = _load_font(settings, fm.title_size, bold=True)
    f_body = _load_font(settings, fm.body_size, bold=False)
    f_badge = _load_font(settings, fm.badge_size, bold=True)

    margin, card_w, pad, inner, radius = fm.margin, fm.card_w, fm.pad, fm.inner, fm.radius
    badge_d, meta_h = fm.badge_d, fm.meta_h
    title_lh, body_lh = fm.title_lh, fm.body_lh
    chip_h, gap_title, gap_body, gap_row = fm.chip_h, fm.gap_title, fm.gap_body, fm.gap_row
    # CSS centres a glyph in its line box; Pillow draws from the ascender.
    title_dy = max(0, (title_lh - _line_height(f_title)) // 2)
    body_dy = max(0, (body_lh - _line_height(f_body)) // 2)

    title_lines = _wrap(post.title or "", f_title, inner)
    if len(title_lines) > fm.title_lines:
        title_lines = title_lines[:fm.title_lines]
        title_lines[-1] = _ellipsize(title_lines[-1])

    body_lines: list[str] = []
    if (post.body or "").strip():
        body_lines = _wrap(post.body.strip(), f_body, inner)
        if len(body_lines) > fm.body_lines:
            body_lines = body_lines[:fm.body_lines]
            body_lines[-1] = _ellipsize(body_lines[-1])

    def card_height(n_title: int, n_body: int) -> int:
        total = pad * 2 + meta_h + gap_title + n_title * title_lh
        if n_body:
            total += gap_body + n_body * body_lh
        return total + gap_row + fm.edge + fm.rule_gap + chip_h

    # An over-long title or body must never push the card (and its vote chips)
    # off the canvas: trim the excerpt first, then the title, adding an ellipsis.
    max_card_h = h - 2 * fm.stage_pad
    while body_lines and card_height(len(title_lines), len(body_lines)) > max_card_h:
        body_lines.pop()
        if body_lines:
            body_lines[-1] = _ellipsize(body_lines[-1])
    while len(title_lines) > 1 and card_height(len(title_lines), len(body_lines)) > max_card_h:
        title_lines.pop()
        title_lines[-1] = _ellipsize(title_lines[-1])
    card_h = card_height(len(title_lines), len(body_lines))

    x0 = margin
    y0 = max(fm.stage_pad, (h - card_h) // 2)
    x1, y1 = x0 + card_w, y0 + card_h

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    sd.rounded_rectangle((x0, y0 + fm.shadow_drop, x1, y1 + fm.shadow_drop), radius=radius,
                         fill=_rgba(theme.shadow, theme.shadow_alpha))
    # CSS `box-shadow ... <blur>` spreads over a Gaussian of *sigma = blur / 2*
    # (CSS Backgrounds 3, "shadow blur radius"), while Pillow's GaussianBlur
    # takes sigma directly.  Handing it the blur radius made the Pillow card's
    # shadow reach roughly twice as far as the Chromium one's -- a soft halo
    # instead of a drop -- so halve it and the two backends match.
    shadow = shadow.filter(ImageFilter.GaussianBlur(fm.shadow_blur / 2.0))
    img.alpha_composite(shadow)

    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=_rgba(theme.card_bg),
                        outline=_rgba(theme.outline, theme.outline_alpha), width=fm.edge)

    cx = x0 + pad
    cy = y0 + pad

    # community badge + community / author line
    br = badge_d // 2
    bcy = cy + meta_h // 2
    d.ellipse((cx, bcy - br, cx + badge_d, bcy + br), fill=_rgba(theme.accent))
    d.text((cx + br, bcy), _badge_letter(post.community), font=f_badge,
           fill=_rgba(theme.badge_fg), anchor="mm")
    tx = cx + badge_d + fm.meta_gap
    community = plain_name(post.community)
    line = byline(post.author)
    stack_h = fm.meta_line + (fm.author_gap + fm.author_line if line else 0)
    top = bcy - stack_h // 2
    d.text((tx, top + max(0, (fm.meta_line - _line_height(f_meta_b)) // 2)), community,
           font=f_meta_b, fill=_rgba(theme.accent), anchor="la")
    if line:
        ly = top + fm.meta_line + fm.author_gap + max(0, (fm.author_line - _line_height(f_meta)) // 2)
        d.text((tx, ly), line, font=f_meta, fill=_rgba(theme.meta_fg), anchor="la")

    cy += meta_h + gap_title
    for line in title_lines:
        d.text((cx, cy + title_dy), line, font=f_title, fill=_rgba(theme.title_fg))
        cy += title_lh

    if body_lines:
        cy += gap_body
        for line in body_lines:
            d.text((cx, cy + body_dy), line, font=f_body, fill=_rgba(theme.body_fg))
            cy += body_lh

    # a hairline separates the story from its counts, so the counts read as the
    # quiet secondary information they are
    cy = y1 - pad - chip_h - fm.rule_gap - fm.edge
    d.rectangle((cx, cy, cx + inner, cy + fm.edge - 1), fill=_rgba(theme.outline, theme.rule_alpha))

    cy = y1 - pad - chip_h
    chip_pad, mark_gap = fm.chip_pad, fm.mark_gap
    mark_w, mark_h = fm.mark_w, fm.mark_h
    my = cy + (chip_h - mark_h) / 2.0

    for value, draw_mark, ink in (
        (post.upvotes, _draw_score_mark, theme.accent),
        (post.comments, _draw_replies_mark, theme.chip_fg),
    ):
        text = _short_count(int(value))
        chip_w = chip_pad * 2 + mark_w + mark_gap + round(_text_width(f_count, text))
        d.rounded_rectangle((cx, cy, cx + chip_w, cy + chip_h), radius=chip_h // 2,
                            fill=_rgba(theme.chip_bg))
        draw_mark(d, cx + chip_pad, my, mark_w, mark_h, _rgba(ink))
        d.text((cx + chip_pad + mark_w + mark_gap, cy + chip_h // 2), text, font=f_count,
               fill=_rgba(theme.chip_fg), anchor="lm")
        cx += chip_w + fm.chip_gap

    out = _downsample(img, scale)
    out.save(out_path)
    return OverlayImage(path=out_path, width=width, height=height, index=0)


# --------------------------------------------------------------------------- #
# chromium backend
# --------------------------------------------------------------------------- #

_CHROMIUM_ARGS = [
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
    "--hide-scrollbars",
    "--force-color-profile=srgb",
    "--font-render-hinting=none",
]

_CHROMIUM_RELATIVE = (
    "chrome-linux/chrome",
    "chrome-linux/headless_shell",
    "chrome-linux/chrome-headless-shell",
    "chrome-headless-shell-linux64/chrome-headless-shell",
    "Chromium.app/Contents/MacOS/Chromium",
    "chrome-mac/Chromium.app/Contents/MacOS/Chromium",
    "chrome-win/chrome.exe",
)


def _browser_roots() -> list[Path]:
    roots: list[Path] = []
    env = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env and env != "0":
        roots.append(Path(env))
    roots.append(Path.home() / ".cache" / "ms-playwright")
    roots.append(Path("/ms-playwright"))
    roots.append(Path("/opt/pw-browsers"))
    return roots


def _chromium_executable(settings: Settings) -> str | None:
    """Resolve a chromium binary without ever running ``playwright install``."""
    configured = (settings.chromium or "").strip()
    if configured:
        path = Path(configured).expanduser()
        if path.is_file():
            return str(path)
        which = shutil.which(configured)
        if which:
            return which
        log.warning("settings.chromium=%r does not exist; searching the browser cache", configured)
    for root in _browser_roots():
        try:
            if not root.is_dir():
                continue
            builds = sorted(root.glob("chromium-*"), reverse=True)
            builds += sorted(root.glob("chromium_headless_shell-*"), reverse=True)
        except OSError:  # pragma: no cover - unreadable mount
            continue
        for build in builds:
            for rel in _CHROMIUM_RELATIVE:
                candidate = build / rel
                if candidate.is_file():
                    return str(candidate)
    for name in ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable"):
        which = shutil.which(name)
        if which:
            return which
    return None


def _have_playwright() -> bool:
    from importlib.util import find_spec

    try:
        return find_spec("playwright.sync_api") is not None
    except (ImportError, ValueError):  # pragma: no cover - broken install
        return False


def _require_playwright() -> Any:
    if not _have_playwright():
        raise MissingDependency("playwright", extra="overlays", purpose="chromium overlay backend")
    from playwright.sync_api import sync_playwright

    return sync_playwright


def _launch_chromium(pw: Any, settings: Settings) -> Any:
    """Launch chromium once, honouring ``settings.chromium`` when it is set."""
    configured = _chromium_executable(settings) if (settings.chromium or "").strip() else None
    try:
        return pw.chromium.launch(executable_path=configured, args=list(_CHROMIUM_ARGS))
    except Exception as exc:  # noqa: BLE001 - any launch failure gets one retry
        fallback = _chromium_executable(settings)
        if fallback and fallback != configured:
            log.debug("chromium launch failed (%s); retrying with %s", exc, fallback)
            return pw.chromium.launch(executable_path=fallback, args=list(_CHROMIUM_ARGS))
        raise


def _template(name: str) -> str:
    path = TEMPLATE_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise OverlayError(f"overlay template {name!r} is missing at {path}") from exc


def _page_html(html_name: str, css_name: str) -> str:
    html = _template(html_name)
    css = _template(css_name)
    if "<!--STYLE-->" not in html:  # pragma: no cover - template guard
        raise OverlayError(f"template {html_name!r} has no <!--STYLE--> marker")
    return html.replace("<!--STYLE-->", f"<style>\n{css}\n</style>")


def _chat_payload(script: ChatScript, width: int, height: int, settings: Settings) -> dict[str, Any]:
    """Everything the page needs, including the resolved theme as CSS variables."""
    labels = _sender_labels(script)
    theme = _chat_theme(script.theme)
    metrics = _chat_metrics(theme, width, height)
    return {
        "title": script.title,
        "contact": script.contact,
        "initials": (script.avatar_initials or "").strip() or _initials(script.contact),
        "theme": theme.name,
        "vars": _chat_css_vars(theme, metrics, _avatar_color(script, settings), _emoji_family(settings)),
        "messages": [
            {"sender": m.sender, "text": m.text, "outgoing": bool(m.outgoing), "label": labels[i]}
            for i, m in enumerate(script.messages)
        ],
    }


def _render_chat_chromium(
    script: ChatScript,
    states: list[ChatState],
    out_dir: Path,
    width: int,
    height: int,
    settings: Settings,
) -> list[OverlayImage]:
    sync_playwright = _require_playwright()
    html = _page_html("chat.html", "chat.css")
    payload = _chat_payload(script, width, height, settings)
    images: list[OverlayImage] = []
    with sync_playwright() as pw:
        browser = _launch_chromium(pw, settings)
        try:
            page = browser.new_page(viewport={"width": int(width), "height": int(height)})
            page.set_viewport_size({"width": int(width), "height": int(height)})
            page.set_content(html, wait_until="load")
            page.evaluate("data => window.__chatInit(data)", payload)
            for state in states:
                page.evaluate(
                    "s => window.__chatState(s)",
                    {"visible": state.visible, "typing": state.typing, "outgoing": state.outgoing},
                )
                path = out_dir / f"chat_{state.index:03d}.png"
                page.screenshot(path=str(path), omit_background=True)
                images.append(OverlayImage(path=path, width=width, height=height, index=state.index))
        finally:
            browser.close()
    return images


def _render_forum_chromium(
    post: RedditPost,
    out_path: Path,
    width: int,
    height: int,
    settings: Settings,
) -> OverlayImage:
    sync_playwright = _require_playwright()
    html = _page_html("forum.html", "forum.css")
    theme = _forum_theme(post.theme)
    payload = {
        "community": plain_name(post.community),
        "byline": byline(post.author),
        "title": post.title,
        "body": post.body,
        "upvotes": _short_count(post.upvotes),
        "comments": _short_count(post.comments),
        "badge": _badge_letter(post.community),
        "theme": theme.name,
        "vars": _forum_css_vars(theme, _forum_metrics(theme, width, height), _emoji_family(settings)),
    }
    with sync_playwright() as pw:
        browser = _launch_chromium(pw, settings)
        try:
            page = browser.new_page(viewport={"width": int(width), "height": int(height)})
            page.set_viewport_size({"width": int(width), "height": int(height)})
            page.set_content(html, wait_until="load")
            page.evaluate("data => window.__forumRender(data)", payload)
            page.screenshot(path=str(out_path), omit_background=True)
        finally:
            browser.close()
    return OverlayImage(path=out_path, width=width, height=height, index=0)


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def available_backends(settings: Settings | None = None) -> list[str]:
    """Backends that will actually work here, best first.

    ``pillow`` is always present; ``chromium`` only when Playwright is importable
    *and* a browser binary can be located (we never download one).
    """
    s = settings or get_settings()
    out: list[str] = []
    if _have_playwright() and _chromium_executable(s):
        out.append("chromium")
    out.append("pillow")
    return out


def _pick_backend(backend: str | None, settings: Settings) -> tuple[str, bool]:
    """Return ``(backend, may_fall_back)``."""
    if backend is None:
        return ("chromium" if "chromium" in available_backends(settings) else "pillow", True)
    name = backend.strip().lower()
    if name not in BACKENDS:
        raise OverlayError(f"unknown overlay backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    if name == "chromium" and not _have_playwright():
        raise MissingDependency("playwright", extra="overlays", purpose="chromium overlay backend")
    return (name, False)


def _sweep_stale_states(out_dir: Path, keep: set[str]) -> None:
    """Delete ``chat_*.png`` left over from an earlier, longer render.

    Workspaces are reused between runs (``settings.keep_work`` defaults to
    true), so a shorter conversation would otherwise leave frames from the
    previous one lying next to the new ones.
    """
    try:
        stale = [p for p in out_dir.glob("chat_*.png") if p.name not in keep]
    except OSError:  # pragma: no cover - unreadable directory
        return
    for path in stale:
        try:
            path.unlink()
        except OSError:  # pragma: no cover - locked file
            log.debug("could not remove stale overlay frame %s", path)


def render_chat(
    script: ChatScript,
    out_dir: Path,
    *,
    width: int,
    height: int,
    settings: Settings | None = None,
    backend: str | None = None,
    header_state: bool = False,
) -> list[OverlayImage]:
    """Render one transparent PNG per conversation state into ``out_dir``.

    The returned list is ordered and its ``index`` values are ``0..n-1``, which
    are also the ``chat_NNN.png`` file numbers.  **How the indices map to
    messages** (an off-by-one here is a visible bug, so read it from
    :func:`chat_states` rather than counting positions by hand):

    ``header_state=False`` (the default, and what the ``texts`` pipeline
    currently assumes) -- state ``0`` is the first message::

        [msg 0] [typing?] [msg 1] ... [msg n-1]

    ``header_state=True`` -- one extra leading state is inserted, so every
    message state moves one place to the right::

        [header only] [msg 0] [typing?] [msg 1] ... [msg n-1]

    The leading state paints the chrome (avatar + contact name) over an empty
    feed.  A caller should hold it from ``t=0`` until the first message lands;
    without it the video opens on a bare background, because the header lives
    inside the state images.  The robust way to place a state is by its own
    fields -- ``visible == 0`` is the header, ``visible == i + 1`` with
    ``typing`` false shows messages ``0..i`` -- which is exactly what
    :func:`chat_states` returns, called with the *same* ``header_state`` value.

    A conversation with no messages renders nothing under either setting.
    """
    s = settings or get_settings()
    if width <= 0 or height <= 0:
        raise OverlayError(f"canvas size must be positive, got {width}x{height}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = chat_states(script, header_state=header_state)
    name, may_fall_back = _pick_backend(backend, s)
    if not states:
        log.info("chat script %r has no messages; nothing to render", script.title or script.contact)
        _sweep_stale_states(out_dir, set())
        return []

    images: list[OverlayImage] | None = None
    if name == "chromium":
        try:
            images = _render_chat_chromium(script, states, out_dir, width, height, s)
        except Exception as exc:  # noqa: BLE001 - fall back on *any* browser trouble
            if not may_fall_back:
                raise OverlayError(f"chromium chat overlay failed: {exc}") from exc
            log.warning("chromium overlay backend failed (%s); falling back to pillow", exc)
    if images is None:
        images = _render_chat_pillow(script, states, out_dir, width, height, s)
    _sweep_stale_states(out_dir, {im.path.name for im in images})
    return images


def render_forum_card(
    post: RedditPost,
    out_path: Path,
    *,
    width: int,
    height: int,
    settings: Settings | None = None,
    backend: str | None = None,
) -> OverlayImage:
    """Render the forum story card as a single transparent PNG."""
    s = settings or get_settings()
    if width <= 0 or height <= 0:
        raise OverlayError(f"canvas size must be positive, got {width}x{height}")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    name, may_fall_back = _pick_backend(backend, s)
    if name == "chromium":
        try:
            return _render_forum_chromium(post, out_path, width, height, s)
        except Exception as exc:  # noqa: BLE001 - fall back on *any* browser trouble
            if not may_fall_back:
                raise OverlayError(f"chromium forum overlay failed: {exc}") from exc
            log.warning("chromium overlay backend failed (%s); falling back to pillow", exc)
    return _render_forum_pillow(post, out_path, width, height, s)
