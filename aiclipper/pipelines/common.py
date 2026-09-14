"""Shared building blocks for the five workflow pipelines.

Every pipeline (``clip``, ``story``, ``texts``, ``reddit``, ``split``) does the
same handful of chores: glue narration parts into one audio file, put word
timings on that file's clock, lay a looping background under the canvas, add a
ducked music bed, burn captions, and pick a non-destructive output path.  Those
chores live here so the pipeline modules stay thin and so every workflow agrees
on the arithmetic.

The keystone is the *narration clock*.  :func:`concat_audio` returns both the
joined file and the start offset of every part, computed from the parts' real
probed durations; :func:`narration_words` puts each part's words on that same
clock.  Overlays, captions and audio tracks then share one timebase with no
further arithmetic anywhere else.

Nothing in this module imports an optional third-party dependency at import
time, and nothing here touches the network.
"""

from __future__ import annotations

import hashlib
import logging
import random
import re
import unicodedata
import zlib
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .. import ffmpeg as ff
from ..config import Settings, get_settings
from ..models import (
    AudioTrack,
    CaptionStyle,
    SubtitleTrack,
    TTSResult,
    VisualLayer,
    VoiceSpec,
    Word,
)

if TYPE_CHECKING:  # pragma: no cover - import-time cost is not worth a type hint
    from ..assets import Asset

log = logging.getLogger(__name__)

__all__ = [
    "MIN_WORD_DURATION",
    "WORDS_PER_SECOND",
    "NARRATION_SAMPLE_RATE",
    "TAIL_SECONDS",
    "CAPTION_REFERENCE",
    "concat_audio",
    "narration_words",
    "proportional_words",
    "words_span",
    "background_layer",
    "music_track",
    "voice_track",
    "caption_resolution",
    "caption_track",
    "safe_stem",
    "resolve_output",
    "caption_style",
    "style_name",
    "provider_name",
    "llm_provider",
    "probe_duration",
    "probe_durations",
    "canvas_size",
    "resolve_voice",
    "seeded_random",
    "workspace",
]

#: Nothing shorter than this is ever emitted as a word, in seconds.
MIN_WORD_DURATION = 0.02

#: Speaking rate used when a duration has to be invented from nothing.
WORDS_PER_SECOND = 2.6

#: Every narration file this module writes is normalised to this format.
NARRATION_SAMPLE_RATE = 44100
NARRATION_CHANNELS = 1

#: A little breathing room after the last narrated word, in seconds.
TAIL_SECONDS = 0.9

#: The canvas the caption presets are authored for; see :func:`caption_resolution`.
CAPTION_REFERENCE = (1080, 1920)

_SLUG_MAX = 60
_SLUG_SPLIT = re.compile(r"[^a-z0-9]+")
_DISAMBIGUATION_LIMIT = 9999


# --------------------------------------------------------------------------- #
# small shared utilities
# --------------------------------------------------------------------------- #

def _cfg(settings: Settings | None) -> Settings:
    return settings or get_settings()


def _fmt(value: float) -> str:
    return f"{float(value):.6f}".rstrip("0").rstrip(".") or "0"


def seeded_random(settings: Settings | None = None, *parts: object, seed: int | None = None) -> random.Random:
    """A deterministic RNG keyed by ``seed`` (or ``settings.seed``) plus ``parts``.

    ``parts`` are stringified and folded in with CRC32 rather than :func:`hash`,
    because Python randomises string hashing per process and hard rule 7 wants
    the same inputs to give the same output on every run.
    """
    base = int(_cfg(settings).seed if seed is None else seed)
    for part in parts:
        base ^= zlib.crc32(str(part).encode("utf-8", "replace"))
    return random.Random(base & 0xFFFFFFFF)


def caption_style(style: str | CaptionStyle | None, enabled: bool = True) -> CaptionStyle | None:
    """Resolve a caption preset *before* a pipeline does any expensive work.

    Returns ``None`` when captions are switched off (or no preset was named),
    which is exactly what :func:`caption_track` wants for ``style``.  Otherwise
    the name goes through :func:`aiclipper.captions.get_style`, so an unknown
    preset raises straight away rather than after a render's worth of speech
    synthesis, frame decoding and crop tracking.
    """
    if not enabled or style is None:
        return None

    from .. import captions as captions_module

    return captions_module.get_style(style)


def style_name(style: str | CaptionStyle | None) -> str:
    """The preset name behind ``style`` (``""`` when there is none)."""
    if style is None:
        return ""
    return style.name if isinstance(style, CaptionStyle) else str(style)


def provider_name(provider: object) -> str:
    """A reportable name for an LLM/TTS provider object, for the metadata."""
    if provider is None:
        return ""
    name = getattr(provider, "name", "")
    return str(name) if name else type(provider).__name__


def llm_provider(provider: object, settings: Settings | None = None) -> object:
    """Resolve ``provider`` to something :mod:`aiclipper.scriptgen` can ask.

    ``None`` and a plain name both go through :func:`aiclipper.llm.get_provider`
    (which picks the heuristic provider offline); an already-constructed provider
    object is passed straight through.  Resolving it in the pipeline rather than
    letting ``scriptgen`` do it means the pipeline knows *which* provider wrote
    the script and can say so in the metadata.
    """
    if provider is None or isinstance(provider, str):
        from ..llm import get_provider

        return get_provider(provider, settings=_cfg(settings))
    return provider


def canvas_size(settings: Settings | None = None, width: int | None = None,
                height: int | None = None) -> tuple[int, int]:
    """Canvas dimensions for a render, forced even so ``yuv420p`` is happy."""
    s = _cfg(settings)
    w = int(width or s.width)
    h = int(height or s.height)
    return (max(2, w - (w % 2)), max(2, h - (h % 2)))


def workspace(name: str, settings: Settings | None = None) -> Path:
    """A scratch directory for one job, named from ``name`` via :func:`safe_stem`."""
    return _cfg(settings).workspace(safe_stem(name, fallback="job"))


def probe_duration(path: str | Path, *, settings: Settings | None = None, default: float = 0.0) -> float:
    """Probed duration of ``path`` in seconds, or ``default`` when unprobeable."""
    try:
        return max(0.0, ff.probe(path, settings=_cfg(settings)).duration)
    except (ff.FFmpegError, FileNotFoundError, OSError, ValueError):
        return default


def probe_durations(paths: Sequence[str | Path], *, settings: Settings | None = None) -> list[float]:
    """Probed duration of every path, in order."""
    s = _cfg(settings)
    return [probe_duration(p, settings=s) for p in paths]


def words_span(words: Sequence[Word]) -> float:
    """End time of the last word (0.0 for an empty list)."""
    return max((w.end for w in words), default=0.0)


def resolve_voice(name: str = "", *, provider: str | None = None,
                  settings: Settings | None = None) -> VoiceSpec:
    """Resolve a user voice string to a :class:`VoiceSpec`, never raising.

    An empty ``name`` falls back to ``settings.tts_voice`` and then to a bare
    :class:`VoiceSpec`, so an offline run with no catalogue match still speaks.
    """
    from ..errors import TTSError
    from ..tts import voices as voice_catalogue

    s = _cfg(settings)
    for candidate in (name or "", s.tts_voice or ""):
        raw = candidate.strip()
        if not raw:
            continue
        try:
            return voice_catalogue.find_voice(raw, provider=provider)
        except TTSError:
            log.debug("voice %r did not resolve; trying the next candidate", raw)
    return VoiceSpec(provider=provider or "auto")


# --------------------------------------------------------------------------- #
# narration audio
# --------------------------------------------------------------------------- #

def _aformat(sample_rate: int, channels: int) -> str:
    layout = "mono" if channels <= 1 else "stereo"
    return f"aformat=sample_fmts=s16:sample_rates={sample_rate}:channel_layouts={layout}"


def concat_audio(
    parts: Sequence[str | Path],
    out_path: str | Path,
    *,
    gap: float = 0.0,
    settings: Settings | None = None,
    sample_rate: int = NARRATION_SAMPLE_RATE,
    channels: int = NARRATION_CHANNELS,
) -> tuple[Path, list[float]]:
    """Join narration ``parts`` into one file with ``gap`` seconds between them.

    Returns ``(path, offsets)`` where ``offsets[i]`` is the second at which part
    ``i`` begins inside the joined file -- the clock that overlays, captions and
    audio tracks are all placed on.  Offsets come from each part's **probed**
    duration, never from whatever length the caller (or a TTS backend) asked
    for, so a backend that overshoots or truncates cannot desynchronise the
    captions.

    The concat *filter* is used rather than the concat demuxer because the parts
    may legitimately differ in codec, sample rate or channel count: every part is
    resampled to a single format first, and the result is a uniform PCM wav even
    when there is only one part.  Silence between parts is appended to each part
    with ``apad`` so no extra inputs are needed.

    Raises :class:`ValueError` when ``parts`` is empty or a part carries no audio,
    and :class:`FileNotFoundError` when a part does not exist.
    """
    s = _cfg(settings)
    sources = [Path(p) for p in parts]
    if not sources:
        raise ValueError("concat_audio needs at least one part")

    pad = max(0.0, float(gap))
    rate = max(8000, int(sample_rate))
    chans = 1 if int(channels) <= 1 else 2

    durations: list[float] = []
    for part in sources:
        if not part.exists():
            raise FileNotFoundError(f"narration part not found: {part}")
        info = ff.probe(part, settings=s)
        if not info.has_audio:
            raise ValueError(f"narration part has no audio stream: {part}")
        durations.append(max(0.0, info.duration))

    offsets: list[float] = []
    cursor = 0.0
    for duration in durations:
        offsets.append(round(cursor, 6))
        cursor += duration + pad

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    args: list[str] = ["-y"]
    for part in sources:
        args += ["-i", str(part)]

    chains: list[str] = []
    labels: list[str] = []
    last = len(sources) - 1
    for index in range(len(sources)):
        filters = [_aformat(rate, chans), "asetpts=N/SR/TB"]
        if pad > 0 and index != last:
            filters.append(f"apad=pad_dur={_fmt(pad)}")
        label = f"p{index}"
        chains.append(f"[{index}:a]" + ",".join(filters) + f"[{label}]")
        labels.append(label)

    if len(labels) == 1:
        graph = ";".join(chains)
        final = labels[0]
    else:
        joined = "".join(f"[{label}]" for label in labels)
        graph = ";".join(chains) + f";{joined}concat=n={len(labels)}:v=0:a=1[out]"
        final = "out"

    args += ["-filter_complex", graph, "-map", f"[{final}]", "-vn"]
    if out.suffix.lower() == ".wav":
        args += ["-c:a", "pcm_s16le"]
    args += [str(out)]

    ff.run_ffmpeg(args, settings=s)
    return out, offsets


# --------------------------------------------------------------------------- #
# narration word timings
# --------------------------------------------------------------------------- #

def _token_weight(token: str) -> float:
    letters = sum(1 for ch in token if ch.isalnum())
    return float(max(1, letters))


def proportional_words(text: str, start: float, duration: float) -> list[Word]:
    """Lay every token of ``text`` across ``[start, start + duration]``.

    The last resort when neither the TTS backend nor forced alignment produced
    timings: tokens are spread in proportion to their length so captions still
    animate roughly in step with the voice.  Every token of ``text`` appears
    exactly once, timings are strictly increasing, and the final word ends at
    ``start + duration``.
    """
    tokens = [tok for tok in (text or "").split() if tok.strip()]
    if not tokens:
        return []
    begin = max(0.0, float(start))
    span = max(float(duration), len(tokens) * MIN_WORD_DURATION)
    weights = [_token_weight(tok) for tok in tokens]
    total = sum(weights) or float(len(tokens))

    words: list[Word] = []
    cursor = begin
    for index, token in enumerate(tokens):
        if index == len(tokens) - 1:
            end = begin + span
        else:
            end = cursor + span * weights[index] / total
        end = max(end, cursor + MIN_WORD_DURATION)
        words.append(Word(token, cursor, end, 0.0))
        cursor = end
    return words


def _monotonic(words: Iterable[Word]) -> list[Word]:
    """Force a word list to be ordered, non-overlapping and non-degenerate."""
    ordered = sorted(words, key=lambda w: (w.start, w.end))
    out: list[Word] = []
    previous = 0.0
    for word in ordered:
        start = max(0.0, word.start, previous)
        end = max(word.end, start + MIN_WORD_DURATION)
        out.append(Word(word.text, start, end, word.prob))
        previous = end
    return out


def _rebased(words: Sequence[Word], offset: float) -> list[Word]:
    """Move ``words`` so their first word starts at ``offset``.

    ``tts.synthesize_lines`` already shifts each result's words onto a shared
    clock, while a bare provider call leaves them part-local.  Rebasing on the
    first word absorbs both conventions, which also makes
    :func:`narration_words` idempotent.
    """
    base = min(w.start for w in words)
    delta = offset - base
    return [w.shifted(delta) for w in words]


def _aligner_ready() -> bool:
    from .. import transcribe

    try:
        return bool(transcribe.available())
    except Exception:  # pragma: no cover - a broken install only
        return False


def _try_align(audio: str | Path, text: str, settings: Settings) -> list[Word]:
    from .. import transcribe

    try:
        return list(transcribe.align(audio, text, settings=settings))
    except Exception as exc:  # noqa: BLE001 - any backend failure degrades to timings we invent
        log.debug("forced alignment of %s failed (%s); using proportional timings", audio, exc)
        return []


def _part_words(result: TTSResult, offset: float, settings: Settings) -> list[Word]:
    if result.words:
        return _rebased(result.words, offset)

    text = (result.text or "").strip()
    if not text:
        return []

    audio = Path(result.audio_path) if result.audio_path else None
    span = probe_duration(audio, settings=settings) if audio else 0.0
    if span <= 0:
        span = max(0.0, float(result.duration or 0.0))
    if span <= 0:
        span = len(text.split()) / WORDS_PER_SECOND

    if audio is not None and audio.exists() and _aligner_ready():
        aligned = _try_align(audio, text, settings)
        if aligned:
            return _rebased(aligned, offset)

    return proportional_words(text, offset, span)


def narration_words(
    results: Sequence[TTSResult],
    offsets: Sequence[float],
    *,
    audio: str | Path | None = None,
    text: str = "",
    settings: Settings | None = None,
) -> list[Word]:
    """Collect every narrated word onto the concatenated narration's clock.

    ``offsets`` are the part offsets from :func:`concat_audio`.  For each result
    in turn:

    1. words reported by the backend are rebased onto that part's offset;
    2. a wordless result with text is force-aligned against its own audio with
       :func:`aiclipper.transcribe.align`;
    3. when alignment is unavailable or fails, timings are spread across the
       part's *real* duration by :func:`proportional_words`.

    When no result yields anything and ``text`` is non-empty, the same two
    fallbacks run once over the whole ``audio`` file, so a caller with narration
    text never gets an empty list back.  The result is always sorted,
    non-overlapping and free of zero-length words.
    """
    s = _cfg(settings)
    collected: list[Word] = []
    cursor = 0.0
    for index, result in enumerate(results):
        offset = float(offsets[index]) if index < len(offsets) else cursor
        collected.extend(_part_words(result, offset, s))
        cursor = offset + max(0.0, float(result.duration or 0.0))

    if not collected and (text or "").strip():
        span = probe_duration(audio, settings=s) if audio is not None else 0.0
        if audio is not None and span > 0 and _aligner_ready():
            collected = _try_align(audio, text, s)
        if not collected:
            if span <= 0:
                span = len(text.split()) / WORDS_PER_SECOND
            collected = proportional_words(text, 0.0, span)

    return _monotonic(collected)


# --------------------------------------------------------------------------- #
# timeline pieces
# --------------------------------------------------------------------------- #

def background_layer(
    asset: Asset,
    *,
    width: int,
    height: int,
    duration: float,
    z: int = 0,
    seed: int | None = None,
    settings: Settings | None = None,
    opacity: float = 1.0,
) -> VisualLayer:
    """A full-canvas looping background cut from ``asset``.

    The loop starts at a seeded offset inside the asset so two runs of the same
    pipeline do not open on the same frame.  The offset can never reach past
    ``asset.duration - duration``: when the asset is long enough the layer plays
    straight through from there, and when it is shorter than the timeline the
    layer loops from the top instead.  Stills are emitted as ``image`` layers,
    which the renderer holds for the whole timeline.
    """
    canvas_w, canvas_h = canvas_size(settings, width, height)
    span = max(0.0, float(duration))
    asset_duration = max(0.0, float(getattr(asset, "duration", 0.0) or 0.0))
    is_image = bool(getattr(asset, "is_image", False))

    src_start = 0.0
    loop = True
    if not is_image:
        # Keep a little margin so a probe that rounds up cannot seek into the tail.
        headroom = asset_duration - span - 0.05
        if headroom > 0:
            src_start = round(seeded_random(settings, asset.name, span, seed=seed).uniform(0.0, headroom), 3)
            loop = False
    else:
        loop = False

    return VisualLayer(
        kind="image" if is_image else "video",
        src=str(asset.path),
        start=0.0,
        end=span or None,
        src_start=src_start,
        x=0,
        y=0,
        w=canvas_w,
        h=canvas_h,
        fit="cover",
        loop=loop,
        opacity=max(0.0, min(1.0, float(opacity))),
        take_audio=False,
        z=z,
        label=f"bg:{asset.name}",
    )


def music_track(
    asset: Asset,
    *,
    duration: float,
    gain_db: float = -18.0,
    duck: bool = True,
    fade_in: float = 0.6,
    fade_out: float = 1.2,
) -> AudioTrack:
    """A music bed under the whole timeline, looped when the asset is short.

    ``duck`` asks the renderer to sidechain it against the voice bus; the
    timeline only validates that when a voice track exists, so pipelines with no
    narration should pass ``duck=False``.
    """
    span = max(0.0, float(duration))
    asset_duration = max(0.0, float(getattr(asset, "duration", 0.0) or 0.0))
    return AudioTrack(
        src=str(asset.path),
        start=0.0,
        src_start=0.0,
        end=span or None,
        gain_db=float(gain_db),
        loop=asset_duration <= 0.0 or asset_duration < span,
        fade_in=max(0.0, min(float(fade_in), span / 2 if span else float(fade_in))),
        fade_out=max(0.0, min(float(fade_out), span / 2 if span else float(fade_out))),
        duck=bool(duck),
        role="music",
        label=f"music:{asset.name}",
    )


def voice_track(
    path: str | Path,
    *,
    start: float = 0.0,
    gain_db: float = 0.0,
    end: float | None = None,
    label: str = "voice",
) -> AudioTrack:
    """One narration file placed at ``start`` on the timeline's clock."""
    return AudioTrack(
        src=str(path),
        start=max(0.0, float(start)),
        src_start=0.0,
        end=end,
        gain_db=float(gain_db),
        loop=False,
        duck=False,
        role="voice",
        label=label,
    )


def caption_resolution(width: int, height: int) -> tuple[int, int]:
    """The ``PlayRes`` an ASS file should declare for a ``width`` x ``height`` canvas.

    Every preset in :mod:`aiclipper.captions` is authored in the units of the
    reference canvas -- ``font_size=96``, ``margin_v=380``, ``outline=8`` and so
    on are pixels at :data:`CAPTION_REFERENCE`.  :func:`aiclipper.captions.build`
    copies those numbers into the style line verbatim and declares ``PlayResX`` /
    ``PlayResY`` as whatever canvas it was handed, so handing it the *real* size
    of a small render makes the captions absurd: at 180x320 a ``bold_yellow``
    line is 96px tall with a 380px bottom margin, which puts it off the canvas
    entirely, and a centred preset such as ``big_impact`` renders two clipped
    letters per line.

    libass scales ``PlayRes`` to the output frame, so the fix is to declare the
    canvas in reference units instead: keep the canvas' exact aspect ratio (an
    aspect mismatch would stretch the glyphs horizontally) and normalise the
    height to :data:`CAPTION_REFERENCE`.  A 1080x1920 render is unchanged; a
    180x320 render gets the same captions, scaled down by six.
    """
    w = max(2, int(width))
    h = max(2, int(height))
    scale = CAPTION_REFERENCE[1] / h
    return (max(2, round(w * scale)), max(2, round(h * scale)))


def caption_track(
    words: Sequence[Word],
    out_dir: str | Path,
    *,
    style: str | CaptionStyle | None = "clean",
    width: int = 1080,
    height: int = 1920,
    enabled: bool = True,
    name: str = "captions.ass",
    settings: Settings | None = None,
) -> SubtitleTrack | None:
    """Burn ``words`` into an ASS file and wrap it in a :class:`SubtitleTrack`.

    ``width`` and ``height`` are the render canvas; the ASS file itself is
    written at the caption reference resolution for that shape (see
    :func:`caption_resolution`) so a preset looks the same at every canvas size.

    Returns ``None`` -- and writes nothing -- when captions are switched off or
    there are no words to show, which is exactly what a pipeline wants to assign
    straight to ``Timeline.subtitles``.
    """
    if not enabled or style is None or not words:
        return None

    from .. import captions as captions_module

    s = _cfg(settings)
    target = Path(out_dir)
    target.mkdir(parents=True, exist_ok=True)
    play_w, play_h = caption_resolution(width, height)
    ass_path = captions_module.build(words, target / name, style=style, width=play_w, height=play_h)
    fonts = s.fonts_dir
    return SubtitleTrack(ass_path=ass_path, fonts_dir=fonts if fonts.is_dir() else None)


# --------------------------------------------------------------------------- #
# naming and output paths
# --------------------------------------------------------------------------- #

def _digest(text: str, size: int = 6) -> str:
    return hashlib.blake2s(text.encode("utf-8", "replace"), digest_size=8).hexdigest()[:size]


def safe_stem(text: str, fallback: str = "video") -> str:
    """A lowercase, filesystem-safe slug for ``text``.

    Accents are folded to ASCII, punctuation and whitespace collapse to single
    hyphens, and the result is capped at 60 characters.  A short digest of the
    full input is appended whenever the slug had to be truncated or the input
    slugified to nothing (non-Latin scripts, punctuation-only titles), so two
    different titles never quietly land on the same filename.  The return value
    is never empty.
    """
    raw = (text or "").strip()
    folded = unicodedata.normalize("NFKD", raw)
    ascii_only = "".join(ch for ch in folded if not unicodedata.combining(ch))
    ascii_only = ascii_only.encode("ascii", "ignore").decode("ascii").lower()
    parts = [p for p in _SLUG_SPLIT.split(ascii_only) if p]

    slug = ""
    for part in parts:
        candidate = f"{slug}-{part}" if slug else part
        if len(candidate) > _SLUG_MAX:
            break
        slug = candidate
    truncated = bool(parts) and slug != "-".join(parts)
    if parts and not slug:  # a single word longer than the cap
        slug = parts[0][:_SLUG_MAX]
        truncated = True

    if not slug:
        base = safe_stem(fallback, fallback="") if fallback else ""
        slug = base or "video"
        if raw:
            return f"{slug}-{_digest(raw)}"
        return slug
    if truncated:
        return f"{slug[: _SLUG_MAX - 7].rstrip('-')}-{_digest(raw)}"
    return slug


def resolve_output(
    out: str | Path | None,
    stem: str,
    settings: Settings | None = None,
    *,
    suffix: str = ".mp4",
) -> Path:
    """Where a pipeline should write its finished file.

    * An explicit file path is honoured exactly -- the caller asked for it.
    * An explicit *directory* (existing, trailing-separator, or extension-less)
      receives ``<stem><suffix>`` inside it.
    * With no ``out`` the file lands in ``settings.output_dir``.

    In every generated case an existing file is never silently clobbered: the
    name gains a ``-2``, ``-3``, ... suffix until it is free.
    """
    s = _cfg(settings)
    slug = safe_stem(stem)

    if out is not None:
        raw = str(out)
        path = Path(raw).expanduser()
        looks_like_dir = path.is_dir() or raw.endswith(("/", "\\")) or not path.suffix
        if not looks_like_dir:
            path.parent.mkdir(parents=True, exist_ok=True)
            return path
        directory = path
    else:
        directory = Path(s.output_dir).expanduser()

    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / f"{slug}{suffix}"
    index = 2
    while candidate.exists() and index <= _DISAMBIGUATION_LIMIT:
        candidate = directory / f"{slug}-{index}{suffix}"
        index += 1
    return candidate
