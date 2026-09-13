"""Word-level captions: grouping, named style presets and hand-written ASS output.

The pipeline is deliberately small and dependency free::

    words -> :func:`group_words` -> [CaptionCue] -> :func:`write_ass` -> .ass file

The renderer burns the resulting file in with ffmpeg's ``subtitles`` filter, so
everything emitted here has to be valid *Advanced SubStation Alpha v4.00+* that
libass accepts.  We write the format by hand -- no subtitle library -- because
the animation overrides (``\\k``-style karaoke recolouring, ``\\t`` scale pops,
``\\move`` bounces, ``\\fad`` fades and word-by-word typewriter reveals) are
easier to express directly than through a generic object model.

Colours are authored as familiar ``#RRGGBB`` / ``#AARRGGBB`` strings and
converted to ASS' ``&HAABBGGRR`` (byte-reversed, *inverted* alpha where ``00``
means fully opaque) by :func:`ass_color`.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from .models import CaptionCue, CaptionStyle, Word

__all__ = [
    "PRESETS",
    "DEFAULT_STYLE",
    "GAP_SPLIT",
    "HOLD_SECONDS",
    "group_words",
    "write_ass",
    "build",
    "get_style",
    "list_styles",
    "ass_color",
    "ass_timestamp",
    "escape_text",
]

# --------------------------------------------------------------------------- #
# tunables
# --------------------------------------------------------------------------- #

DEFAULT_STYLE = "clean"

#: A silence longer than this (seconds) always starts a new cue.
GAP_SPLIT = 0.7

#: How long a cue may linger past its last word when there is room for it.
HOLD_SECONDS = 0.30

#: Smallest cue/dialogue duration we are willing to emit.
MIN_CUE = 0.05

#: Fade and pop transition lengths, in milliseconds.
FADE_MS = 120
POP_MS = 120
MOVE_MS = 140

_ALIGNMENT = {"top": 8, "center": 5, "bottom": 2}
_SENTENCE_END = (".", "!", "?", "\u2026")
_TRAILING_PUNCT = "\"')]}\u00bb\u201d\u2019\u203a"
_INITIAL_RE = re.compile(r"^[A-Za-z]\.$")
_STYLE_FORMAT = (
    "Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
    "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
    "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
)
_EVENT_FORMAT = "Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"
_STYLE_NAME = "Default"


# --------------------------------------------------------------------------- #
# colour + time primitives
# --------------------------------------------------------------------------- #

def ass_color(value: str, *, with_alpha: bool = True) -> str:
    """Convert ``#RRGGBB`` / ``#AARRGGBB`` to an ASS colour literal.

    ASS stores colours byte-reversed as ``&HAABBGGRR`` and treats alpha as
    *transparency*: ``00`` is fully opaque, ``FF`` fully transparent.  The input
    uses the usual convention (``FF`` = opaque), so the alpha byte is inverted.

    With ``with_alpha=False`` the 6-digit ``&HBBGGRR&`` form used by inline
    ``\\c`` overrides is returned instead.
    """
    if not isinstance(value, str):
        raise ValueError(f"colour must be a string, got {type(value).__name__}")
    text = value.strip()
    if text.lower().startswith("&h"):
        text = text[2:].rstrip("&")
        if len(text) not in (6, 8) or not _is_hex(text):
            raise ValueError(f"invalid ASS colour literal: {value!r}")
        body = text.rjust(8, "0").upper()
        return f"&H{body}" if with_alpha else f"&H{body[2:]}&"
    text = text.lstrip("#").strip()
    if not _is_hex(text):
        raise ValueError(f"invalid colour {value!r}: expected hex digits")
    if len(text) == 3:  # #RGB
        text = "".join(ch * 2 for ch in text)
    elif len(text) == 4:  # #ARGB
        text = "".join(ch * 2 for ch in text)
    if len(text) == 6:
        alpha_in = 255
        rr, gg, bb = text[0:2], text[2:4], text[4:6]
    elif len(text) == 8:
        alpha_in = int(text[0:2], 16)
        rr, gg, bb = text[2:4], text[4:6], text[6:8]
    else:
        raise ValueError(f"invalid colour {value!r}: expected 3, 4, 6 or 8 hex digits")
    bgr = f"{bb}{gg}{rr}".upper()
    if not with_alpha:
        return f"&H{bgr}&"
    return f"&H{255 - alpha_in:02X}{bgr}"


def _is_hex(text: str) -> bool:
    return bool(text) and all(ch in "0123456789abcdefABCDEF" for ch in text)


def ass_timestamp(seconds: float) -> str:
    """Format ``seconds`` as ASS' ``H:MM:SS.cc``.  Negatives clamp to zero."""
    value = float(seconds)
    if not math.isfinite(value) or value <= 0.0:
        return "0:00:00.00"
    total_cs = int(round(value * 100))
    hours, rest = divmod(total_cs, 360000)
    minutes, rest = divmod(rest, 6000)
    secs, cs = divmod(rest, 100)
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{cs:02d}"


def escape_text(text: str) -> str:
    """Escape ASS specials: backslashes, braces and newlines."""
    out = text.replace("\\", "\\\\")
    out = out.replace("{", "\\{").replace("}", "\\}")
    out = out.replace("\r\n", "\\N").replace("\r", "\\N").replace("\n", "\\N")
    return out


def _field(text: str) -> str:
    """Sanitise a value going into a comma-separated ``Style:`` field."""
    return " ".join(str(text).replace(",", " ").split())


def _num(value: float) -> str:
    number = float(value)
    if not math.isfinite(number):
        return "0"
    return str(int(number)) if number.is_integer() else f"{number:g}"


# --------------------------------------------------------------------------- #
# style presets
# --------------------------------------------------------------------------- #

def _style(name: str, description: str, **kwargs: object) -> CaptionStyle:
    return CaptionStyle(name=name, description=description, **kwargs)  # type: ignore[arg-type]


PRESETS: dict[str, CaptionStyle] = {
    "clean": _style(
        "clean", "Crisp white sans with a yellow karaoke sweep, centred.",
        font="DejaVu Sans", font_size=84, bold=True, primary_color="#FFFFFF",
        highlight_color="#FFE400", outline_color="#000000", shadow_color="#000000",
        outline=6.0, shadow=2.0, position="center", margin_v=320, margin_h=90,
        max_words=3, max_chars=24, animation="karaoke",
    ),
    "bold_yellow": _style(
        "bold_yellow", "Chunky uppercase yellow that pops word by word near the bottom.",
        font="DejaVu Sans", font_size=96, bold=True, uppercase=True, primary_color="#FFE400",
        highlight_color="#FFFFFF", outline_color="#1A1400", shadow_color="#000000",
        outline=8.0, shadow=3.0, position="bottom", margin_v=380, margin_h=80,
        max_words=4, max_chars=26, animation="pop", scale_pop=1.16,
    ),
    "karaoke_green": _style(
        "karaoke_green", "Longer lines with a mint-green karaoke highlight.",
        font="DejaVu Sans", font_size=88, bold=True, primary_color="#FFFFFF",
        highlight_color="#35F08A", outline_color="#06200F", shadow_color="#000000",
        outline=7.0, shadow=2.0, position="center", margin_v=300, margin_h=80,
        max_words=5, max_chars=30, animation="karaoke",
    ),
    "outline_pop": _style(
        "outline_pop", "Two-word bursts inside a very heavy black outline.",
        font="DejaVu Sans", font_size=104, bold=True, uppercase=True, primary_color="#FFFFFF",
        highlight_color="#FF4F9A", outline_color="#000000", shadow_color="#000000",
        outline=12.0, shadow=0.0, position="center", margin_v=340, margin_h=110,
        max_words=2, max_chars=16, animation="pop", scale_pop=1.22,
    ),
    "boxed": _style(
        "boxed", "Soft translucent slab behind quiet fading lines.",
        font="DejaVu Sans", font_size=72, bold=True, primary_color="#FFFFFF",
        highlight_color="#9BD7FF", outline_color="#000000", shadow_color="#000000",
        back_color="#CC101018", outline=4.0, shadow=0.0, position="bottom",
        margin_v=260, margin_h=120, max_words=6, max_chars=34, animation="fade",
    ),
    "minimal_serif": _style(
        "minimal_serif", "Understated serif documentary subtitles, no animation.",
        font="DejaVu Serif", font_size=64, bold=False, primary_color="#F4F1EA",
        highlight_color="#D9B26A", outline_color="#14110C", shadow_color="#000000",
        outline=3.0, shadow=2.0, position="bottom", margin_v=180, margin_h=140,
        max_words=7, max_chars=40, animation="none",
    ),
    "neon": _style(
        "neon", "Electric cyan with a magenta sweep and a glowing drop shadow.",
        font="DejaVu Sans", font_size=92, bold=True, uppercase=True, primary_color="#7CFBFF",
        highlight_color="#FF3FD8", outline_color="#0B0F2E", shadow_color="#00E5FF",
        outline=5.0, shadow=7.0, spacing=2.0, position="center", margin_v=320, margin_h=90,
        max_words=3, max_chars=22, animation="bounce",
    ),
    "comic": _style(
        "comic", "Playful italic captions that bounce in from below.",
        font="DejaVu Sans", font_size=90, bold=True, italic=True, primary_color="#FFFFFF",
        highlight_color="#FF6A3D", outline_color="#1A1A1A", shadow_color="#3B1E0B",
        outline=9.0, shadow=4.0, position="bottom", margin_v=420, margin_h=90,
        max_words=3, max_chars=24, animation="bounce",
    ),
    "subtle_lower": _style(
        "subtle_lower", "Small, dense lower-third text that stays out of the way.",
        font="DejaVu Sans", font_size=50, bold=False, primary_color="#FFFFFF",
        highlight_color="#BFD9FF", outline_color="#000000", shadow_color="#000000",
        outline=2.0, shadow=1.0, position="bottom", margin_v=140, margin_h=150,
        max_words=8, max_chars=44, animation="none",
    ),
    "big_impact": _style(
        "big_impact", "Enormous uppercase hero text with a hard red accent.",
        font="DejaVu Sans", font_size=128, bold=True, uppercase=True, primary_color="#FFFFFF",
        highlight_color="#FF2D2D", outline_color="#000000", shadow_color="#000000",
        outline=10.0, shadow=5.0, position="center", margin_v=360, margin_h=70,
        max_words=2, max_chars=14, animation="pop", scale_pop=1.28,
    ),
    "gradient_pop": _style(
        "gradient_pop", "Blush-to-violet palette with a karaoke sweep.",
        font="DejaVu Sans", font_size=98, bold=True, uppercase=True, primary_color="#FFE9F6",
        highlight_color="#7B5BFF", outline_color="#2A0F4B", shadow_color="#12042A",
        outline=8.0, shadow=3.0, spacing=1.0, position="center", margin_v=300, margin_h=90,
        max_words=3, max_chars=20, animation="karaoke", scale_pop=1.18,
    ),
    "mono_terminal": _style(
        "mono_terminal", "Green-on-black monospace that types itself out.",
        font="DejaVu Sans Mono", font_size=58, bold=False, primary_color="#9BFF8F",
        highlight_color="#FFFFFF", outline_color="#000000", shadow_color="#000000",
        back_color="#E6050A05", outline=5.0, shadow=0.0, spacing=1.0, position="bottom",
        margin_v=220, margin_h=110, max_words=6, max_chars=36, animation="typewriter",
    ),
    "handwritten": _style(
        "handwritten", "Warm cream italic serif that fades line by line.",
        font="DejaVu Serif", font_size=78, bold=True, italic=True, primary_color="#FFF6E0",
        highlight_color="#FFB55C", outline_color="#3A2A12", shadow_color="#1B1207",
        outline=5.0, shadow=3.0, position="center", margin_v=280, margin_h=120,
        max_words=4, max_chars=28, animation="fade",
    ),
    "shadow_deep": _style(
        "shadow_deep", "Thin outline carried by a long, heavy drop shadow.",
        font="DejaVu Sans", font_size=86, bold=True, primary_color="#FFFFFF",
        highlight_color="#FFD166", outline_color="#000000", shadow_color="#000000",
        outline=2.0, shadow=12.0, position="bottom", margin_v=300, margin_h=100,
        max_words=4, max_chars=28, animation="fade",
    ),
    "tiktok_white": _style(
        "tiktok_white", "Plain white vertical-video captions with a light pop.",
        font="DejaVu Sans", font_size=80, bold=True, primary_color="#FFFFFF",
        highlight_color="#00E0FF", outline_color="#000000", shadow_color="#000000",
        outline=5.0, shadow=0.0, position="bottom", margin_v=460, margin_h=90,
        max_words=4, max_chars=26, animation="pop", scale_pop=1.10,
    ),
    "podcast_bar": _style(
        "podcast_bar", "Wide dark caption bar for talking-head clips.",
        font="DejaVu Sans", font_size=56, bold=False, primary_color="#FFFFFF",
        highlight_color="#FFC857", outline_color="#000000", shadow_color="#000000",
        back_color="#D9000000", outline=6.0, shadow=0.0, position="bottom",
        margin_v=120, margin_h=60, max_words=9, max_chars=48, animation="none",
    ),
}


def list_styles() -> list[CaptionStyle]:
    """Every preset, ordered by name."""
    return [PRESETS[key] for key in sorted(PRESETS)]


def get_style(name: str | CaptionStyle) -> CaptionStyle:
    """Resolve a preset name (case/dash insensitive) or pass a style through."""
    if isinstance(name, CaptionStyle):
        return name
    if not isinstance(name, str):
        raise ValueError(f"style must be a name or CaptionStyle, got {type(name).__name__}")
    key = name.strip().lower().replace("-", "_").replace(" ", "_")
    if not key:
        key = DEFAULT_STYLE
    if key not in PRESETS:
        raise ValueError(f"unknown caption style {name!r}. Available: {', '.join(sorted(PRESETS))}")
    return replace(PRESETS[key])


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #

def _ends_sentence(text: str) -> bool:
    stripped = text.strip().rstrip(_TRAILING_PUNCT)
    if not stripped or not stripped.endswith(_SENTENCE_END):
        return False
    return not _INITIAL_RE.match(stripped)


def group_words(words: Sequence[Word], style: CaptionStyle) -> list[CaptionCue]:
    """Pack ``words`` into on-screen cues according to ``style``.

    Words are never dropped (except blank ones) nor reordered.  A new cue starts
    when the previous one is full (``max_words`` / ``max_chars``), when the pause
    before the next word exceeds :data:`GAP_SPLIT`, or right after a
    sentence-final word.  Each cue lingers up to :data:`HOLD_SECONDS` past its
    last word but never into the next cue.
    """
    max_words = max(1, int(style.max_words))
    max_chars = max(1, int(style.max_chars))

    groups: list[list[Word]] = []
    current: list[Word] = []
    current_chars = 0

    for word in words:
        text = word.text.strip()
        if not text:
            continue
        if current:
            gap = word.start - current[-1].end
            would_be = current_chars + 1 + len(text)
            if (
                gap > GAP_SPLIT
                or len(current) >= max_words
                or would_be > max_chars
                or _ends_sentence(current[-1].text)
            ):
                groups.append(current)
                current = []
                current_chars = 0
        current.append(word)
        current_chars = len(text) if len(current) == 1 else current_chars + 1 + len(text)
    if current:
        groups.append(current)

    cues: list[CaptionCue] = []
    for index, group in enumerate(groups):
        start = max(0.0, float(group[0].start))
        end = max(float(group[-1].end), start + MIN_CUE)
        next_start = float(groups[index + 1][0].start) if index + 1 < len(groups) else None
        if next_start is None:
            end += HOLD_SECONDS
        else:
            room = next_start - end
            if room > 0:
                end += min(HOLD_SECONDS, room * 0.5)
            else:
                end = min(end, max(start, next_start))
        # Out-of-order / overlapping input can collapse the window above; never
        # emit a cue that would be on screen for zero seconds.
        if end <= start:
            end = start + MIN_CUE
        cues.append(CaptionCue(start=start, end=end, words=list(group)))
    return cues


# --------------------------------------------------------------------------- #
# ASS writing
# --------------------------------------------------------------------------- #

def _tokens(cue: CaptionCue, style: CaptionStyle) -> list[str]:
    out: list[str] = []
    for word in cue.words:
        text = word.text.strip()
        if not text:
            continue
        out.append(escape_text(text.upper() if style.uppercase else text))
    return out


def _anchor(style: CaptionStyle, width: int, height: int) -> tuple[int, int]:
    x = width // 2
    if style.position == "top":
        y = int(style.margin_v)
    elif style.position == "bottom":
        y = height - int(style.margin_v)
    else:
        y = height // 2
    return x, max(0, min(height, y))


def _header(style: CaptionStyle, width: int, height: int, *, title: str = "aiclipper captions") -> str:
    """The ``[Script Info]`` + ``[V4+ Styles]`` + ``[Events]`` preamble.

    Note the colour mapping for ``back_color``: libass (like VSFilter) paints the
    ``BorderStyle: 3`` opaque box with the *OutlineColour*, not the BackColour --
    BackColour stays the drop shadow in both border styles.  So a style with a
    ``back_color`` puts that colour in the outline slot (there is no separate
    glyph outline once the box replaces it) and keeps the shadow in BackColour.
    """
    border_style = 3 if style.back_color else 1
    outline = style.back_color or style.outline_color
    back = style.shadow_color
    alignment = _ALIGNMENT.get(style.position, 5)
    style_line = ",".join(
        [
            _STYLE_NAME,
            _field(style.font) or "DejaVu Sans",
            _num(style.font_size),
            ass_color(style.primary_color),
            ass_color(style.highlight_color),
            ass_color(outline),
            ass_color(back),
            "-1" if style.bold else "0",
            "-1" if style.italic else "0",
            "0",
            "0",
            "100",
            "100",
            _num(style.spacing),
            "0",
            str(border_style),
            _num(style.outline),
            _num(style.shadow),
            str(alignment),
            str(int(style.margin_h)),
            str(int(style.margin_h)),
            str(int(style.margin_v)),
            "1",
        ]
    )
    return "\n".join(
        [
            "[Script Info]",
            f"; {title}",
            f"; style: {style.name}",
            "ScriptType: v4.00+",
            # 0 = balanced automatic wrapping inside the margins.  WrapStyle 2
            # disables wrapping entirely, which lets a long cue run off canvas.
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            "YCbCr Matrix: TV.709",
            f"PlayResX: {width}",
            f"PlayResY: {height}",
            "",
            "[V4+ Styles]",
            f"Format: {_STYLE_FORMAT}",
            f"Style: {style_line}",
            "",
            "[Events]",
            f"Format: {_EVENT_FORMAT}",
        ]
    )


def _dialogue(start: float, end: float, text: str, *, layer: int = 0) -> str:
    return (
        f"Dialogue: {layer},{ass_timestamp(start)},{ass_timestamp(end)},"
        f"{_STYLE_NAME},,0,0,0,,{text}"
    )


def _word_spans(cue: CaptionCue, count: int) -> list[tuple[float, float]]:
    """Per-word [start, end) slices covering the cue, clamped and monotonic."""
    spans: list[tuple[float, float]] = []
    words = [w for w in cue.words if w.text.strip()]
    cursor = cue.start
    for i in range(count):
        start = max(cursor, min(float(words[i].start), cue.end))
        if i == 0:
            start = cue.start
        if i + 1 < count:
            end = max(start, min(float(words[i + 1].start), cue.end))
        else:
            end = max(start, cue.end)
        spans.append((start, end))
        cursor = end
    return spans


def _cue_dialogues(cue: CaptionCue, style: CaptionStyle, width: int, height: int) -> list[str]:
    tokens = _tokens(cue, style)
    if not tokens:
        return []
    animation = style.animation
    base_color = ass_color(style.primary_color, with_alpha=False)
    high_color = ass_color(style.highlight_color, with_alpha=False)
    plain = " ".join(tokens)

    if animation == "karaoke":
        lines = []
        for i, (start, end) in enumerate(_word_spans(cue, len(tokens))):
            parts = list(tokens)
            parts[i] = f"{{\\c{high_color}}}{tokens[i]}{{\\c{base_color}}}"
            lines.append(_dialogue(start, end, " ".join(parts)))
        return lines

    if animation == "pop":
        scale = max(1, int(round(float(style.scale_pop) * 100)))
        lines = []
        for i, (start, end) in enumerate(_word_spans(cue, len(tokens))):
            parts = list(tokens)
            grow = f"{{\\fscx100\\fscy100\\t(0,{POP_MS},\\fscx{scale}\\fscy{scale})}}"
            parts[i] = f"{grow}{tokens[i]}{{\\fscx100\\fscy100}}"
            lines.append(_dialogue(start, end, " ".join(parts)))
        return lines

    if animation == "typewriter":
        lines = []
        for i, (start, end) in enumerate(_word_spans(cue, len(tokens))):
            lines.append(_dialogue(start, end, " ".join(tokens[: i + 1])))
        return lines

    if animation == "bounce":
        x, y = _anchor(style, width, height)
        drop = max(6, int(round(style.font_size * 0.35)))
        move = f"{{\\move({x},{y + drop},{x},{y},0,{MOVE_MS})}}"
        return [_dialogue(cue.start, cue.end, f"{move}{plain}")]

    if animation == "fade":
        return [_dialogue(cue.start, cue.end, f"{{\\fad({FADE_MS},{FADE_MS})}}{plain}")]

    return [_dialogue(cue.start, cue.end, plain)]


def write_ass(
    cues: Sequence[CaptionCue],
    out_path: str | Path,
    *,
    style: CaptionStyle,
    width: int,
    height: int,
) -> Path:
    """Write ``cues`` as an ASS v4.00+ file sized for a ``width`` x ``height`` canvas."""
    canvas_w = max(2, int(width))
    canvas_h = max(2, int(height))
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [_header(style, canvas_w, canvas_h)]
    for cue in cues:
        lines.extend(_cue_dialogues(cue, style, canvas_w, canvas_h))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def build(
    words: Sequence[Word],
    out_path: str | Path,
    *,
    style: str | CaptionStyle = DEFAULT_STYLE,
    width: int = 1080,
    height: int = 1920,
) -> Path:
    """Group ``words``, render them with ``style`` and write the ASS file."""
    resolved = get_style(style)
    cues = group_words(words, resolved)
    return write_ass(cues, out_path, style=resolved, width=width, height=height)
