"""Core data contracts shared by every module in :mod:`aiclipper`.

Everything in the engine speaks these types.  Producers (ingest, transcribe,
highlight, script, tts, crop, captions, overlays) emit them; the renderer
consumes a :class:`Timeline` and nothing else.  Keep this module dependency
free -- standard library only -- so it can be imported anywhere.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any, Literal

__all__ = [
    "Word", "Segment", "Transcript",
    "MediaInfo",
    "ClipCandidate",
    "ScriptBeat", "VideoScript",
    "ChatMessage", "ChatScript", "RedditPost",
    "VoiceSpec", "TTSResult",
    "CropKeyframe", "CropPath",
    "CaptionStyle", "CaptionCue",
    "VisualLayer", "AudioTrack", "SubtitleTrack", "Timeline",
    "RenderOptions", "RenderResult", "ProjectResult",
    "to_jsonable",
]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def to_jsonable(obj: Any) -> Any:
    """Recursively convert dataclasses/Paths into JSON-serialisable values."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --------------------------------------------------------------------------- #
# transcripts
# --------------------------------------------------------------------------- #

@dataclass
class Word:
    """A single spoken word with its timing, in seconds from media start."""

    text: str
    start: float
    end: float
    prob: float = 1.0

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def shifted(self, offset: float) -> Word:
        return Word(self.text, self.start + offset, self.end + offset, self.prob)


@dataclass
class Segment:
    """A sentence-ish chunk of speech.  ``words`` may be empty for coarse ASR."""

    text: str
    start: float
    end: float
    words: list[Word] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def shifted(self, offset: float) -> Segment:
        return Segment(
            self.text,
            self.start + offset,
            self.end + offset,
            [w.shifted(offset) for w in self.words],
        )


@dataclass
class Transcript:
    """Word-level transcript of one media file."""

    segments: list[Segment] = field(default_factory=list)
    language: str = "en"
    duration: float = 0.0

    # -- views ------------------------------------------------------------- #
    @property
    def words(self) -> list[Word]:
        out: list[Word] = []
        for seg in self.segments:
            out.extend(seg.words)
        return out

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.segments if s.text.strip())

    def __iter__(self) -> Iterator[Segment]:
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)

    @property
    def is_empty(self) -> bool:
        return not any(s.text.strip() for s in self.segments)

    # -- operations -------------------------------------------------------- #
    def slice(self, start: float, end: float, rebase: bool = True) -> Transcript:
        """Return the portion overlapping ``[start, end)``.

        With ``rebase`` the result is shifted so ``start`` becomes ``0.0`` --
        which is what clip rendering wants.  Segments are trimmed, not dropped,
        when they straddle a boundary.
        """
        out: list[Segment] = []
        for seg in self.segments:
            if seg.end <= start or seg.start >= end:
                continue
            words = [w for w in seg.words if w.end > start and w.start < end]
            if seg.words and not words:
                continue
            text = " ".join(w.text.strip() for w in words).strip() if words else seg.text
            out.append(
                Segment(
                    text=text,
                    start=max(seg.start, start),
                    end=min(seg.end, end),
                    words=[Word(w.text, max(w.start, start), min(w.end, end), w.prob) for w in words],
                )
            )
        offset = -start if rebase else 0.0
        if offset:
            out = [s.shifted(offset) for s in out]
        return Transcript(segments=out, language=self.language, duration=max(0.0, end - start))

    def word_at(self, t: float) -> Word | None:
        for w in self.words:
            if w.start <= t < w.end:
                return w
        return None

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        return to_jsonable(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        segs = [
            Segment(
                text=s.get("text", ""),
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", 0.0)),
                words=[
                    Word(w["text"], float(w["start"]), float(w["end"]), float(w.get("prob", 1.0)))
                    for w in s.get("words", [])
                ],
            )
            for s in data.get("segments", [])
        ]
        return cls(segments=segs, language=data.get("language", "en"), duration=float(data.get("duration", 0.0)))

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        return p

    @classmethod
    def load(cls, path: str | Path) -> Transcript:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_words(cls, words: Sequence[Word], language: str = "en", max_gap: float = 0.6) -> Transcript:
        """Group a flat word list into segments, splitting on pauses/punctuation."""
        segs: list[Segment] = []
        cur: list[Word] = []
        for w in words:
            if cur and (w.start - cur[-1].end > max_gap or cur[-1].text.rstrip().endswith((".", "!", "?"))):
                segs.append(Segment(" ".join(x.text.strip() for x in cur), cur[0].start, cur[-1].end, list(cur)))
                cur = []
            cur.append(w)
        if cur:
            segs.append(Segment(" ".join(x.text.strip() for x in cur), cur[0].start, cur[-1].end, list(cur)))
        return cls(segments=segs, language=language, duration=segs[-1].end if segs else 0.0)


# --------------------------------------------------------------------------- #
# media
# --------------------------------------------------------------------------- #

@dataclass
class MediaInfo:
    """Probe result for one local media file."""

    path: Path
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    has_video: bool = False
    has_audio: bool = False
    title: str | None = None
    source_url: str | None = None
    size_bytes: int = 0

    @property
    def aspect(self) -> float:
        return (self.width / self.height) if self.height else 0.0

    @property
    def is_vertical(self) -> bool:
        return self.height > self.width


# --------------------------------------------------------------------------- #
# clip selection
# --------------------------------------------------------------------------- #

@dataclass
class ClipCandidate:
    """One proposed short, in source-media time."""

    start: float
    end: float
    title: str = ""
    hook: str = ""
    reason: str = ""
    score: float = 0.0
    tags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def padded(self, before: float, after: float, limit: float | None = None) -> ClipCandidate:
        start = max(0.0, self.start - before)
        end = self.end + after
        if limit is not None:
            end = min(end, limit)
        return ClipCandidate(start, end, self.title, self.hook, self.reason, self.score, list(self.tags))

    def overlaps(self, other: ClipCandidate, tolerance: float = 0.0) -> bool:
        return self.start < other.end - tolerance and other.start < self.end - tolerance


# --------------------------------------------------------------------------- #
# generated scripts
# --------------------------------------------------------------------------- #

@dataclass
class ScriptBeat:
    """One narrated line, optionally paired with a visual cue."""

    text: str
    image_prompt: str | None = None
    broll: str | None = None
    emphasis: bool = False


@dataclass
class VideoScript:
    """A narrated short: hook, body beats, call to action."""

    title: str = ""
    hook: str = ""
    beats: list[ScriptBeat] = field(default_factory=list)
    cta: str = ""
    hashtags: list[str] = field(default_factory=list)

    @property
    def lines(self) -> list[str]:
        out = [self.hook] + [b.text for b in self.beats] + [self.cta]
        return [ln.strip() for ln in out if ln and ln.strip()]

    @property
    def narration(self) -> str:
        return " ".join(self.lines)


# --------------------------------------------------------------------------- #
# chat + forum story sources
# --------------------------------------------------------------------------- #

@dataclass
class ChatMessage:
    """One bubble in a text-conversation story video."""

    sender: str
    text: str
    outgoing: bool = False
    delay: float = 0.35
    reaction: str | None = None
    read_aloud: bool = True
    voice: str | None = None
    typing: float = 0.0


@dataclass
class ChatScript:
    """A full text-conversation story."""

    title: str = ""
    contact: str = "Unknown"
    messages: list[ChatMessage] = field(default_factory=list)
    theme: str = "classic"
    avatar_initials: str = ""

    @property
    def spoken_lines(self) -> list[ChatMessage]:
        return [m for m in self.messages if m.read_aloud and m.text.strip()]


@dataclass
class RedditPost:
    """A forum-style story card (original styling, not a site clone)."""

    community: str = "r/stories"
    author: str = "u/anonymous"
    title: str = ""
    body: str = ""
    upvotes: int = 0
    comments: int = 0
    theme: str = "dark"

    @property
    def narration(self) -> str:
        return f"{self.title} {self.body}".strip()


# --------------------------------------------------------------------------- #
# speech synthesis
# --------------------------------------------------------------------------- #

@dataclass
class VoiceSpec:
    """Which voice to synthesise with, provider agnostic."""

    provider: str = "auto"
    voice_id: str = ""
    rate: float = 1.0
    pitch_semitones: float = 0.0
    style: str | None = None
    language: str = "en"

    @classmethod
    def parse(cls, spec: str) -> VoiceSpec:
        """``"edge:en-US-GuyNeural"`` / ``"narrator_male"`` / ``""``."""
        spec = (spec or "").strip()
        if not spec:
            return cls()
        if ":" in spec:
            provider, _, voice = spec.partition(":")
            return cls(provider=provider.strip() or "auto", voice_id=voice.strip())
        return cls(provider="auto", voice_id=spec)


@dataclass
class TTSResult:
    """Synthesised narration.  ``words`` is ``None`` when the provider gave no
    word boundaries -- callers then force-align with the ASR backend."""

    audio_path: Path
    duration: float = 0.0
    words: list[Word] | None = None
    voice: VoiceSpec | None = None
    text: str = ""


# --------------------------------------------------------------------------- #
# reframing
# --------------------------------------------------------------------------- #

@dataclass
class CropKeyframe:
    """Crop window in *source* pixel coordinates at time ``t`` (seconds)."""

    t: float
    x: int
    y: int
    w: int
    h: int


@dataclass
class CropPath:
    """Animated crop window used to reframe a wide source to vertical."""

    keyframes: list[CropKeyframe] = field(default_factory=list)
    source_width: int = 0
    source_height: int = 0

    @property
    def is_static(self) -> bool:
        if len(self.keyframes) <= 1:
            return True
        first = self.keyframes[0]
        return all(k.x == first.x and k.y == first.y and k.w == first.w and k.h == first.h for k in self.keyframes)

    @property
    def size(self) -> tuple[int, int]:
        if not self.keyframes:
            return (self.source_width, self.source_height)
        return (self.keyframes[0].w, self.keyframes[0].h)

    def at(self, t: float) -> CropKeyframe:
        """Linearly interpolated crop window at time ``t``."""
        if not self.keyframes:
            return CropKeyframe(t, 0, 0, self.source_width, self.source_height)
        kfs = self.keyframes
        if t <= kfs[0].t:
            k = kfs[0]
            return CropKeyframe(t, k.x, k.y, k.w, k.h)
        if t >= kfs[-1].t:
            k = kfs[-1]
            return CropKeyframe(t, k.x, k.y, k.w, k.h)
        for a, b in zip(kfs, kfs[1:], strict=False):  # pairwise: last keyframe has no successor
            if a.t <= t <= b.t:
                span = (b.t - a.t) or 1e-6
                f = _clamp((t - a.t) / span, 0.0, 1.0)
                return CropKeyframe(
                    t,
                    int(round(a.x + (b.x - a.x) * f)),
                    int(round(a.y + (b.y - a.y) * f)),
                    int(round(a.w + (b.w - a.w) * f)),
                    int(round(a.h + (b.h - a.h) * f)),
                )
        k = kfs[-1]
        return CropKeyframe(t, k.x, k.y, k.w, k.h)

    @classmethod
    def static(cls, x: int, y: int, w: int, h: int, source_width: int, source_height: int) -> CropPath:
        return cls([CropKeyframe(0.0, x, y, w, h)], source_width, source_height)


# --------------------------------------------------------------------------- #
# captions
# --------------------------------------------------------------------------- #

@dataclass
class CaptionStyle:
    """A named subtitle look.  Consumed by the ASS writer."""

    name: str = "clean"
    font: str = "DejaVu Sans"
    font_size: int = 84
    bold: bool = True
    italic: bool = False
    uppercase: bool = False
    primary_color: str = "#FFFFFF"
    highlight_color: str = "#FFE400"
    outline_color: str = "#000000"
    shadow_color: str = "#000000"
    back_color: str | None = None
    outline: float = 6.0
    shadow: float = 2.0
    spacing: float = 0.0
    position: Literal["top", "center", "bottom"] = "center"
    margin_v: int = 320
    margin_h: int = 90
    max_words: int = 3
    max_chars: int = 24
    animation: Literal["none", "karaoke", "pop", "bounce", "fade", "typewriter"] = "karaoke"
    scale_pop: float = 1.12
    description: str = ""


@dataclass
class CaptionCue:
    """One rendered caption group: the words shown together on screen."""

    start: float
    end: float
    words: list[Word] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(w.text.strip() for w in self.words).strip()


# --------------------------------------------------------------------------- #
# timeline (the renderer's only input)
# --------------------------------------------------------------------------- #

Fit = Literal["cover", "contain", "stretch"]


@dataclass
class VisualLayer:
    """One picture element composited onto the canvas.

    Geometry is in *canvas* pixels.  ``w``/``h`` default to the full canvas.
    ``crop`` reframes the source before scaling (used for vertical reframing).
    """

    kind: Literal["video", "image", "color"] = "video"
    src: str | None = None
    start: float = 0.0
    end: float | None = None
    src_start: float = 0.0
    x: int = 0
    y: int = 0
    w: int | None = None
    h: int | None = None
    fit: Fit = "cover"
    loop: bool = False
    opacity: float = 1.0
    color: str = "#000000"
    crop: CropPath | None = None
    take_audio: bool = False
    volume: float = 1.0
    z: int = 0
    label: str = ""

    def rect(self, canvas_w: int, canvas_h: int) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w or canvas_w, self.h or canvas_h)


@dataclass
class AudioTrack:
    """One sound source on the mix bus."""

    src: str
    start: float = 0.0
    src_start: float = 0.0
    end: float | None = None
    gain_db: float = 0.0
    loop: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0
    duck: bool = False
    role: Literal["voice", "music", "sfx"] = "sfx"
    label: str = ""


@dataclass
class SubtitleTrack:
    """Burned-in subtitles, as a path to an ASS file."""

    ass_path: Path
    fonts_dir: Path | None = None


@dataclass
class Timeline:
    """Everything the renderer needs to produce one video file."""

    width: int = 1080
    height: int = 1920
    fps: int = 30
    duration: float = 0.0
    background: str = "#000000"
    visuals: list[VisualLayer] = field(default_factory=list)
    audio: list[AudioTrack] = field(default_factory=list)
    subtitles: SubtitleTrack | None = None
    title: str = ""

    # -- building ---------------------------------------------------------- #
    def add_visual(self, layer: VisualLayer) -> VisualLayer:
        self.visuals.append(layer)
        return layer

    def add_audio(self, track: AudioTrack) -> AudioTrack:
        self.audio.append(track)
        return track

    @property
    def ordered_visuals(self) -> list[VisualLayer]:
        return sorted(self.visuals, key=lambda layer: (layer.z, self.visuals.index(layer)))

    @property
    def has_voice(self) -> bool:
        return any(t.role == "voice" for t in self.audio)

    def fit_duration(self) -> float:
        """Longest explicit end time across layers (fallback when duration unset)."""
        ends: list[float] = [self.duration]
        for layer in self.visuals:
            if layer.end is not None:
                ends.append(layer.end)
        for track in self.audio:
            if track.end is not None:
                ends.append(track.end)
        return max(ends) if ends else 0.0

    def validate(self) -> list[str]:
        """Return a list of human-readable problems; empty means renderable."""
        problems: list[str] = []
        if self.width <= 0 or self.height <= 0:
            problems.append("canvas size must be positive")
        if self.width % 2 or self.height % 2:
            problems.append("canvas dimensions must be even for yuv420p")
        if self.fps <= 0:
            problems.append("fps must be positive")
        if self.duration <= 0:
            problems.append("timeline duration must be positive")
        for i, layer in enumerate(self.visuals):
            if layer.kind in ("video", "image") and not layer.src:
                problems.append(f"visual[{i}] ({layer.label or layer.kind}) has no src")
            if layer.end is not None and layer.end <= layer.start:
                problems.append(f"visual[{i}] ends before it starts")
            if not math.isfinite(layer.opacity) or not 0.0 <= layer.opacity <= 1.0:
                problems.append(f"visual[{i}] opacity must be within 0..1")
        for i, track in enumerate(self.audio):
            if not track.src:
                problems.append(f"audio[{i}] has no src")
            if track.end is not None and track.end <= track.start:
                problems.append(f"audio[{i}] ends before it starts")
        if any(t.duck for t in self.audio) and not self.has_voice:
            problems.append("a track requests ducking but no voice track exists")
        return problems


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #

@dataclass
class RenderOptions:
    crf: int = 20
    preset: str = "veryfast"
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    audio_bitrate: str = "192k"
    pix_fmt: str = "yuv420p"
    threads: int = 0
    overwrite: bool = True
    faststart: bool = True
    extra_args: list[str] = field(default_factory=list)


@dataclass
class RenderResult:
    path: Path
    duration: float = 0.0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    command: list[str] = field(default_factory=list)
    size_bytes: int = 0


@dataclass
class ProjectResult:
    """What a pipeline hands back to the CLI."""

    output: Path
    kind: str = ""
    title: str = ""
    duration: float = 0.0
    transcript: Transcript | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    siblings: list[Path] = field(default_factory=list)

    def summary(self) -> str:
        return f"{self.kind}: {self.title or self.output.name} -> {self.output} ({self.duration:.1f}s)"
