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

``render_chat`` turns a :class:`~aiclipper.models.ChatScript` into one
transparent PNG per *conversation state*: image ``i`` shows messages ``0..i``.
A message with ``typing > 0`` additionally gets a state just before it that
shows a typing indicator on that message's side of the screen.  The newest
bubble is anchored low in the safe area and older bubbles scroll up (the
topmost fading out) once the stack overflows -- text size never shrinks.

All artwork here is original: generic bubbles, a generic card, our own
palettes.  Nothing imitates the trade dress of any real product.
"""

from __future__ import annotations

import logging
import os
import random
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
    "TEMPLATE_DIR",
    "BACKENDS",
    "chat_states",
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
    """Colour palette for the chat overlay (original design)."""

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


@dataclass(frozen=True)
class ForumTheme:
    """Colour palette for the forum story card (original design)."""

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

DEFAULT_CHAT_THEME = "classic"
DEFAULT_FORUM_THEME = "dark"

#: How many wrapped lines of the forum card's title/body survive before eliding.
#: Mirrored by the ``-webkit-line-clamp`` rules in ``templates/forum.css``.
_FORUM_TITLE_LINES = 8
_FORUM_BODY_LINES = 7


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

def chat_states(script: ChatScript) -> list[ChatState]:
    """Plan the conversation states for ``script``.

    One state per message, plus one extra *before* every message that carries a
    ``typing`` hint.  Both backends consume this list, which is what keeps them
    in lockstep.
    """
    states: list[ChatState] = []
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


def _initials(text: str, limit: int = 2) -> str:
    parts = [p for p in (text or "").replace("_", " ").split() if p]
    letters = [p[0] for p in parts if p[0].isalnum()]
    if not letters:
        letters = [c for c in (text or "") if c.isalnum()][:limit]
    return "".join(letters[:limit]).upper() or "?"


def _badge_letter(community: str) -> str:
    """First meaningful letter of a community name (``"r/nightshift"`` -> ``"N"``)."""
    name = (community or "").strip()
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

_font_file_cache: dict[tuple[str, bool], str | None] = {}


def _font_file(settings: Settings, bold: bool) -> str | None:
    """Locate a real TTF, preferring ``settings.fonts_dir``."""
    key = (str(settings.fonts_dir), bold)
    if key in _font_file_cache:
        return _font_file_cache[key]
    names = _BOLD_NAMES if bold else _REGULAR_NAMES
    roots: list[Path] = [settings.fonts_dir, *(Path(d) for d in _FONT_DIRS)]
    found: str | None = None
    for root in roots:
        try:
            if not root.is_dir():
                continue
        except OSError:  # pragma: no cover - unreadable mount
            continue
        for name in names:
            candidate = root / name
            if candidate.is_file():
                found = str(candidate)
                break
        if found:
            break
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

@dataclass
class _ChatMetrics:
    w: int
    h: int
    margin: int
    header_top: int
    avatar_d: int
    content_top: int
    anchor_y: int
    gap: int
    pad_x: int
    pad_y: int
    radius: int
    line_h: int
    line_gap: int
    name_h: int
    outline: int
    max_bubble_w: int
    min_bubble_w: int
    typing_w: int
    typing_h: int
    dot_r: int
    dot_gap: int
    fade_span: int


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
        w, h = width * scale, height * scale
        font_size = max(10, round(w * 0.0390))
        self.font = _load_font(settings, font_size, bold=False)
        self.name_font = _load_font(settings, max(8, round(font_size * 0.62)), bold=True)
        self.header_font = _load_font(settings, max(9, round(font_size * 0.92)), bold=True)
        self.avatar_font = _load_font(settings, max(10, round(font_size * 1.15)), bold=True)

        line_h = _line_height(self.font)
        pad_x = round(font_size * 0.66)
        pad_y = round(font_size * 0.46)
        dot_r = max(1, round(font_size * 0.17))
        dot_gap = max(2, round(font_size * 0.52))
        typing_w = 2 * pad_x + 2 * dot_gap + 2 * dot_r
        typing_h = 2 * pad_y + line_h
        header_top = round(h * 0.045)
        avatar_d = round(w * 0.105)
        name_h = round(_line_height(self.name_font) * 1.25)
        content_top = header_top + avatar_d + round(_line_height(self.header_font) * 1.5) + round(h * 0.018)
        bottom_safe = round(h * 0.140)
        anchor_y = content_top + round((h - bottom_safe - content_top) * 0.80)
        margin = round(w * 0.055)
        self.m = _ChatMetrics(
            w=w, h=h, margin=margin,
            header_top=header_top, avatar_d=avatar_d,
            content_top=content_top, anchor_y=anchor_y,
            gap=round(font_size * 0.46),
            pad_x=pad_x, pad_y=pad_y,
            radius=round(font_size * 0.88),
            line_h=line_h, line_gap=round(line_h * 0.22),
            name_h=name_h,
            outline=max(1, round(font_size * 0.045)),
            max_bubble_w=round((w - 2 * margin) * 0.74),
            min_bubble_w=typing_w + round(font_size * 0.9),
            typing_w=typing_w, typing_h=typing_h,
            dot_r=dot_r, dot_gap=dot_gap,
            fade_span=round(h * 0.07),
        )

    # -- measuring --------------------------------------------------------- #
    def measure(self, msg: ChatMessage, label: str) -> _Bubble:
        m = self.m
        inner = m.max_bubble_w - 2 * m.pad_x
        lines = _wrap(msg.text or "", self.font, inner)
        text_w = max((_text_width(self.font, ln) for ln in lines), default=0.0)
        w = int(max(m.min_bubble_w, min(m.max_bubble_w, round(text_w) + 2 * m.pad_x)))
        body_h = 2 * m.pad_y + len(lines) * m.line_h + (len(lines) - 1) * m.line_gap
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
    d.text((cx, top + m.avatar_d + round(m.name_h * 0.45)), script.contact.strip(),
           font=ctx.header_font, fill=_rgba(theme.header_fg), anchor="ma")


def _draw_bubble(d: PilDraw, ctx: _ChatCtx, bubble: _Bubble, top: int) -> None:
    m, theme = ctx.m, ctx.theme
    x = (m.w - m.margin - bubble.w) if bubble.outgoing else m.margin
    y = top
    if bubble.label:
        label_x = x + bubble.w - m.pad_x if bubble.outgoing else x + m.pad_x
        d.text((label_x, y), bubble.label, font=ctx.name_font,
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
        cx = x + bubble.w // 2 - m.dot_gap
        dot = theme.out_fg if bubble.outgoing else theme.dot
        for i in range(3):
            px = cx + i * m.dot_gap
            d.ellipse((px - m.dot_r, cy - m.dot_r, px + m.dot_r, cy + m.dot_r),
                      fill=_rgba(dot, 0.55 + 0.225 * i))
        return
    ty = y + m.pad_y
    for line in bubble.lines:
        d.text((x + m.pad_x, ty), line, font=ctx.font, fill=_rgba(fg))
        ty += m.line_h + m.line_gap


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
# forum card (pillow backend)
# --------------------------------------------------------------------------- #

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
    w, h = width * scale, height * scale

    base = max(10, round(w * 0.030))
    f_meta = _load_font(settings, base, bold=False)
    f_meta_b = _load_font(settings, base, bold=True)
    f_title = _load_font(settings, round(base * 1.62), bold=True)
    f_body = _load_font(settings, round(base * 1.06), bold=False)
    f_badge = _load_font(settings, round(base * 1.05), bold=True)

    margin = round(w * 0.070)
    card_w = w - 2 * margin
    pad = round(w * 0.052)
    inner = card_w - 2 * pad
    radius = round(w * 0.038)

    badge_d = round(base * 2.0)
    meta_h = max(badge_d, round(_line_height(f_meta) * 2.2))
    title_lines = _wrap(post.title or "", f_title, inner)
    if len(title_lines) > _FORUM_TITLE_LINES:
        title_lines = title_lines[:_FORUM_TITLE_LINES]
        title_lines[-1] = _ellipsize(title_lines[-1])
    title_lh = round(_line_height(f_title) * 1.20)

    body_lines: list[str] = []
    body_lh = round(_line_height(f_body) * 1.34)
    if (post.body or "").strip():
        body_lines = _wrap(post.body.strip(), f_body, inner)
        if len(body_lines) > _FORUM_BODY_LINES:
            body_lines = body_lines[:_FORUM_BODY_LINES]
            body_lines[-1] = _ellipsize(body_lines[-1])

    chip_h = round(base * 2.3)
    gap_title = round(base * 0.95)
    gap_body = round(base * 1.05)
    gap_row = round(base * 1.25)

    def card_height(n_title: int, n_body: int) -> int:
        total = pad * 2 + meta_h + gap_title + n_title * title_lh
        if n_body:
            total += gap_body + n_body * body_lh
        return total + gap_row + chip_h

    # An over-long title or body must never push the card (and its vote chips)
    # off the canvas: trim the excerpt first, then the title, adding an ellipsis.
    max_card_h = h - 2 * round(h * 0.045)
    while body_lines and card_height(len(title_lines), len(body_lines)) > max_card_h:
        body_lines.pop()
        if body_lines:
            body_lines[-1] = _ellipsize(body_lines[-1])
    while len(title_lines) > 1 and card_height(len(title_lines), len(body_lines)) > max_card_h:
        title_lines.pop()
        title_lines[-1] = _ellipsize(title_lines[-1])
    card_h = card_height(len(title_lines), len(body_lines))

    x0 = margin
    y0 = max(round(h * 0.05), (h - card_h) // 2)
    x1, y1 = x0 + card_w, y0 + card_h

    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))

    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow, "RGBA")
    drop = round(base * 0.55)
    sd.rounded_rectangle((x0, y0 + drop, x1, y1 + drop), radius=radius,
                         fill=_rgba(theme.shadow, theme.shadow_alpha))
    shadow = shadow.filter(ImageFilter.GaussianBlur(round(base * 0.9)))
    img.alpha_composite(shadow)

    d = ImageDraw.Draw(img, "RGBA")
    d.rounded_rectangle((x0, y0, x1, y1), radius=radius, fill=_rgba(theme.card_bg),
                        outline=_rgba(theme.outline, theme.outline_alpha),
                        width=max(1, round(base * 0.06)))

    cx = x0 + pad
    cy = y0 + pad

    # community badge + community / author line
    br = badge_d // 2
    bcy = cy + meta_h // 2
    d.ellipse((cx, bcy - br, cx + badge_d, bcy + br), fill=_rgba(theme.accent))
    d.text((cx + br, bcy), _badge_letter(post.community), font=f_badge,
           fill=_rgba(theme.badge_fg), anchor="mm")
    tx = cx + badge_d + round(base * 0.6)
    community = (post.community or "").strip()
    author = (post.author or "").strip()
    d.text((tx, bcy - round(_line_height(f_meta_b) * 0.52)), community, font=f_meta_b,
           fill=_rgba(theme.accent), anchor="ls")
    if author:
        d.text((tx, bcy + round(_line_height(f_meta) * 0.92)), author, font=f_meta,
               fill=_rgba(theme.meta_fg), anchor="ls")

    cy += meta_h + gap_title
    for line in title_lines:
        d.text((cx, cy), line, font=f_title, fill=_rgba(theme.title_fg))
        cy += title_lh

    if body_lines:
        cy += gap_body
        for line in body_lines:
            d.text((cx, cy), line, font=f_body, fill=_rgba(theme.body_fg))
            cy += body_lh

    cy = y1 - pad - chip_h
    chip_pad = round(base * 0.7)
    icon = round(base * 0.52)

    votes = _short_count(post.upvotes)
    chip1_w = chip_pad * 2 + icon * 2 + round(base * 0.5) + round(_text_width(f_meta_b, votes))
    d.rounded_rectangle((cx, cy, cx + chip1_w, cy + chip_h), radius=chip_h // 2, fill=_rgba(theme.chip_bg))
    acx = cx + chip_pad + icon
    acy = cy + chip_h // 2
    d.polygon([(acx, acy - icon), (acx - icon, acy + icon * 0.5), (acx + icon, acy + icon * 0.5)],
              fill=_rgba(theme.accent))
    d.text((acx + icon + round(base * 0.5), acy), votes, font=f_meta_b,
           fill=_rgba(theme.chip_fg), anchor="lm")

    comments = _short_count(post.comments)
    cx2 = cx + chip1_w + round(base * 0.7)
    chip2_w = chip_pad * 2 + icon * 2 + round(base * 0.5) + round(_text_width(f_meta_b, comments))
    d.rounded_rectangle((cx2, cy, cx2 + chip2_w, cy + chip_h), radius=chip_h // 2, fill=_rgba(theme.chip_bg))
    bx = cx2 + chip_pad
    d.rounded_rectangle((bx, acy - icon, bx + icon * 2, acy + icon * 0.7),
                        radius=max(2, round(icon * 0.45)), fill=_rgba(theme.chip_fg))
    d.polygon([(bx + icon * 0.45, acy + icon * 0.6), (bx + icon * 1.05, acy + icon * 0.6),
               (bx + icon * 0.5, acy + icon * 1.35)], fill=_rgba(theme.chip_fg))
    d.text((cx2 + chip_pad + icon * 2 + round(base * 0.5), acy), comments, font=f_meta_b,
           fill=_rgba(theme.chip_fg), anchor="lm")

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


def _chat_payload(script: ChatScript) -> dict[str, Any]:
    labels = _sender_labels(script)
    return {
        "title": script.title,
        "contact": script.contact,
        "initials": (script.avatar_initials or "").strip() or _initials(script.contact),
        "theme": _chat_theme(script.theme).name,
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
    payload = _chat_payload(script)
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
    payload = {
        "community": post.community,
        "author": post.author,
        "title": post.title,
        "body": post.body,
        "upvotes": _short_count(post.upvotes),
        "comments": _short_count(post.comments),
        "badge": _badge_letter(post.community),
        "theme": _forum_theme(post.theme).name,
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
) -> list[OverlayImage]:
    """Render one transparent PNG per conversation state into ``out_dir``.

    Image ``i`` shows messages ``0..i``; a message with ``typing > 0`` gets an
    extra preceding state that shows the typing indicator on its side.  The
    returned list is ordered and its ``index`` values are ``0..n-1``.
    """
    s = settings or get_settings()
    if width <= 0 or height <= 0:
        raise OverlayError(f"canvas size must be positive, got {width}x{height}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    states = chat_states(script)
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
