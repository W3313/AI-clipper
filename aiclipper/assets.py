"""Background and music asset library.

The engine needs two kinds of stock media: vertical background loops and
instrumental music beds.  Users drop their own files into
``settings.backgrounds_dir`` / ``settings.music_dir``; everything found there is
catalogued in a ``library.json`` manifest that also carries hand-written tags.

When the library is empty nothing has to be downloaded: :func:`ensure_placeholders`
*synthesises* a small starter set with ffmpeg's ``lavfi`` sources -- four
visually distinct 1080x1920 loops (a spiralling gradient aurora, a cellular
bloom, a violet fractal tide and a soft ember mist) and two music beds built
from layered sine/aevalsrc voices with an envelope.  Generation is idempotent:
an asset that already exists on disk is never re-rendered.

Only :mod:`aiclipper.ffmpeg` is used to invoke ffmpeg, and nothing outside the
standard library is imported at module import time.
"""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import ffmpeg as ff
from .config import Settings, get_settings
from .errors import AssetError

log = logging.getLogger(__name__)

__all__ = [
    "Asset",
    "MANIFEST_NAME",
    "VIDEO_EXTENSIONS",
    "IMAGE_EXTENSIONS",
    "AUDIO_EXTENSIONS",
    "PLACEHOLDER_WIDTH",
    "PLACEHOLDER_HEIGHT",
    "PLACEHOLDER_FPS",
    "PLACEHOLDER_BACKGROUND_SECONDS",
    "PLACEHOLDER_MUSIC_SECONDS",
    "library",
    "pick_background",
    "pick_music",
    "ensure_placeholders",
]

MANIFEST_NAME = "library.json"
MANIFEST_VERSION = 2

VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg"})
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".webp", ".bmp"})
AUDIO_EXTENSIONS = frozenset({".mp3", ".m4a", ".wav", ".aac", ".flac", ".ogg", ".opus", ".wma"})

KINDS = ("background", "music")

PLACEHOLDER_WIDTH = 1080
PLACEHOLDER_HEIGHT = 1920
PLACEHOLDER_FPS = 30
PLACEHOLDER_BACKGROUND_SECONDS = 20.0
PLACEHOLDER_MUSIC_SECONDS = 30.0

_SAMPLE_RATE = 44100
_TAG_SPLIT = str.maketrans({"-": " ", "_": " ", ".": " ", "+": " "})


# --------------------------------------------------------------------------- #
# the record
# --------------------------------------------------------------------------- #

@dataclass
class Asset:
    """One catalogued media file: a background loop or a music bed."""

    path: Path
    name: str = ""
    kind: str = "background"
    tags: list[str] = field(default_factory=list)
    duration: float = 0.0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        if not self.name:
            self.name = self.path.stem
        self.tags = [t.strip().lower() for t in self.tags if str(t).strip()]

    @property
    def suffix(self) -> str:
        return self.path.suffix.lower()

    @property
    def is_image(self) -> bool:
        return self.suffix in IMAGE_EXTENSIONS

    @property
    def is_video(self) -> bool:
        return self.suffix in VIDEO_EXTENSIONS

    @property
    def is_audio(self) -> bool:
        return self.suffix in AUDIO_EXTENSIONS

    def score(self, tags: Iterable[str] | str) -> int:
        """How many of ``tags`` this asset carries (case-insensitive)."""
        return len(_norm_tags(tags) & set(self.tags))


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _norm_tags(tags: Iterable[str] | str | None) -> set[str]:
    """Normalise a tag argument.  A bare string counts as *one* tag, not a set of letters."""
    if tags is None:
        return set()
    if isinstance(tags, (str, Path)):
        tags = [str(tags)]
    return {str(t).strip().lower() for t in tags if str(t).strip()}


def _cfg(settings: Settings | None) -> Settings:
    return settings or get_settings()


def _dir_for(settings: Settings, kind: str) -> Path:
    if kind == "background":
        return settings.backgrounds_dir
    if kind == "music":
        return settings.music_dir
    raise AssetError(f"unknown asset kind {kind!r} (expected one of {KINDS})")


def _extensions_for(kind: str) -> frozenset[str]:
    return AUDIO_EXTENSIONS if kind == "music" else (VIDEO_EXTENSIONS | IMAGE_EXTENSIONS)


def _infer_tags(stem: str) -> list[str]:
    """Derive tags from a filename: ``sunset-loop_02`` -> ``["sunset", "loop", "02"]``."""
    out: list[str] = []
    for part in stem.translate(_TAG_SPLIT).split():
        tag = part.strip().lower()
        if tag and tag not in out:
            out.append(tag)
    return out


def _iter_media(root: Path, kind: str) -> list[Path]:
    if not root.is_dir():
        return []
    allowed = _extensions_for(kind)
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        # skip dotfiles and anything inside a hidden folder (.trash, .cache, ...)
        if any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        if path.name == MANIFEST_NAME or path.suffix.lower() not in allowed:
            continue
        found.append(path)
    return found


def _unique_name(path: Path, root: Path, taken: set[str]) -> str:
    name = path.stem
    if name not in taken:
        return name
    try:
        rel = path.relative_to(root).with_suffix("")
    except ValueError:  # pragma: no cover - defensive
        rel = Path(path.stem)
    candidate = "-".join(rel.parts)
    suffix = 2
    while candidate in taken:
        candidate = f"{path.stem}-{suffix}"
        suffix += 1
    return candidate


def _probe_duration(path: Path, settings: Settings) -> float:
    try:
        return round(max(0.0, ff.probe_duration(path, settings=settings)), 3)
    except Exception as exc:  # ffmpeg missing, unreadable file, image without duration
        log.debug("could not probe %s: %s", path, exc)
        return 0.0


def _stat_key(path: Path) -> tuple[int, int]:
    """Size + nanosecond mtime -- the cache key that says "this file is unchanged"."""
    try:
        st = path.stat()
    except OSError:  # pragma: no cover - race with deletion
        return (0, 0)
    return (int(st.st_size), int(st.st_mtime_ns))


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

def _manifest_path(settings: Settings) -> Path:
    return settings.assets_dir / MANIFEST_NAME


def _read_manifest(path: Path) -> dict[tuple[str, str], dict]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    entries = raw.get("assets") if isinstance(raw, dict) else raw
    if not isinstance(entries, list):
        return {}
    out: dict[tuple[str, str], dict] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        kind = str(entry.get("kind") or "background")
        name = str(entry.get("name") or Path(str(entry.get("file") or "")).stem)
        if not name:
            continue
        out[(kind, name)] = entry
    return out


def _entry_for(asset: Asset, settings: Settings) -> dict:
    try:
        file = asset.path.relative_to(settings.assets_dir).as_posix()
    except ValueError:
        file = asset.path.as_posix()
    size, mtime_ns = _stat_key(asset.path)
    return {
        "name": asset.name,
        "kind": asset.kind,
        "file": file,
        "tags": list(asset.tags),
        "duration": round(float(asset.duration), 3),
        "size": size,
        "mtime_ns": mtime_ns,
    }


def _write_manifest(path: Path, entries: Sequence[dict]) -> None:
    """Write the manifest, but only when its content actually changed."""
    payload = {"version": MANIFEST_VERSION, "assets": list(entries)}
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    try:
        if path.exists() and path.read_text(encoding="utf-8") == text:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.part")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)  # atomic: a concurrent reader never sees a half-written manifest
    except OSError as exc:  # a read-only asset dir must not break a render
        log.warning("could not write asset manifest %s: %s", path, exc)


def _upsert_manifest(settings: Settings, assets: Sequence[Asset]) -> None:
    """Merge ``assets`` into the manifest, keeping every other entry."""
    path = _manifest_path(settings)
    records = _read_manifest(path)
    for asset in assets:
        records[(asset.kind, asset.name)] = _entry_for(asset, settings)
    ordered = sorted(records.items(), key=lambda item: (item[0][0], item[0][1]))
    _write_manifest(path, [entry for _, entry in ordered])


# --------------------------------------------------------------------------- #
# the library
# --------------------------------------------------------------------------- #

def library(settings: Settings | None = None) -> list[Asset]:
    """Scan the asset directories and return every catalogued :class:`Asset`.

    Tags and durations come from ``library.json`` when it already describes the
    file (matched on size + mtime); otherwise tags are inferred from the
    filename and the duration is probed with ffprobe.  The manifest is then
    refreshed so the next call is cheap.
    """
    s = _cfg(settings)
    records = _read_manifest(_manifest_path(s))

    assets: list[Asset] = []
    for kind in KINDS:
        root = _dir_for(s, kind)
        taken: set[str] = set()
        for path in _iter_media(root, kind):
            name = _unique_name(path, root, taken)
            taken.add(name)
            entry = records.get((kind, name))
            tags = [str(t) for t in entry.get("tags", [])] if isinstance(entry, dict) else []
            if not tags:
                tags = _infer_tags(name)
            duration = 0.0
            fresh = False
            if isinstance(entry, dict):
                size, mtime_ns = _stat_key(path)
                if int(entry.get("size", -1)) == size and int(entry.get("mtime_ns", -1)) == mtime_ns:
                    fresh = True
                    try:
                        duration = float(entry.get("duration", 0.0))
                    except (TypeError, ValueError):
                        duration, fresh = 0.0, False
            # A still image legitimately has no duration; trusting the cached 0.0 keeps the
            # scan from re-probing every image on every call.
            if duration <= 0.0 and not (fresh and path.suffix.lower() in IMAGE_EXTENSIONS):
                duration = _probe_duration(path, s)
            assets.append(Asset(path=path, name=name, kind=kind, tags=tags, duration=duration))

    assets.sort(key=lambda a: (a.kind, a.name))
    if assets or _manifest_path(s).exists():
        _write_manifest(_manifest_path(s), [_entry_for(a, s) for a in assets])
    return assets


# --------------------------------------------------------------------------- #
# picking
# --------------------------------------------------------------------------- #

def _choose(pool: Sequence[Asset], seed: int | None, settings: Settings) -> Asset:
    ordered = sorted(pool, key=lambda a: (a.name, str(a.path)))
    rng = random.Random(settings.seed if seed is None else seed)
    return rng.choice(ordered)


def _asset_from_path(raw: str, kind: str, settings: Settings) -> Asset | None:
    """Allow ``pick_*("/some/file.mp4")`` -- an explicit path wins over the library."""
    looks_like_path = os.sep in raw or "/" in raw or bool(Path(raw).suffix)
    if not raw or not looks_like_path:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        return None
    return Asset(
        path=path,
        name=path.stem,
        kind=kind,
        tags=_infer_tags(path.stem),
        duration=_probe_duration(path, settings),
    )


def _pool(kind: str, settings: Settings) -> list[Asset]:
    pool = [a for a in library(settings) if a.kind == kind]
    if pool:
        return pool
    # Only synthesise the kind that is actually missing: a user who ships their own
    # backgrounds but no music should not get four rendered background loops dropped
    # into their curated folder (and should not wait ~40s for them).
    try:
        generated = _generate_placeholders(settings, kinds=(kind,))
    except AssetError:
        raise
    except Exception as exc:  # pragma: no cover - unexpected ffmpeg failure shape
        raise AssetError(f"no {kind} assets and placeholders could not be generated: {exc}") from exc
    pool = [a for a in generated if a.kind == kind]
    if not pool:
        raise AssetError(
            f"no {kind} assets available in {_dir_for(settings, kind)} and none could be generated"
        )
    return pool


def _pick(
    kind: str,
    name: str | Path | None,
    tags: Iterable[str] | str | None,
    settings: Settings | None,
    seed: int | None,
) -> Asset:
    s = _cfg(settings)
    wanted_name = "" if name is None else str(name).strip()
    if wanted_name:
        direct = _asset_from_path(wanted_name, kind, s)
        if direct is not None:
            return direct

    pool = _pool(kind, s)

    if wanted_name:
        key = wanted_name.lower()
        for asset in sorted(pool, key=lambda a: a.name):
            if asset.name.lower() == key:
                return asset
        tagged = [a for a in pool if key in a.tags]
        if tagged:
            return _choose(tagged, seed, s)
        available = ", ".join(sorted(a.name for a in pool)) or "<none>"
        raise AssetError(f"no {kind} asset named {wanted_name!r}. Available: {available}")

    wanted_tags = _norm_tags(tags)
    if wanted_tags:
        best = max(a.score(wanted_tags) for a in pool)
        if best > 0:
            return _choose([a for a in pool if a.score(wanted_tags) == best], seed, s)
        log.debug("no %s asset matched tags %s; falling back to the whole library", kind, sorted(wanted_tags))

    return _choose(pool, seed, s)


def pick_background(
    name: str | Path | None = None,
    *,
    tags: Iterable[str] | str | None = None,
    settings: Settings | None = None,
    seed: int | None = None,
) -> Asset:
    """Resolve a background: by exact name (or path), else by tag, else seeded choice."""
    return _pick("background", name, tags, settings, seed)


def pick_music(
    name: str | Path | None = None,
    *,
    tags: Iterable[str] | str | None = None,
    settings: Settings | None = None,
    seed: int | None = None,
) -> Asset:
    """Resolve a music bed: by exact name (or path), else by tag, else seeded choice."""
    return _pick("music", name, tags, settings, seed)


# --------------------------------------------------------------------------- #
# procedural placeholders
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _Geometry:
    """Output canvas plus the (cheap) low-resolution synthesis stage."""

    width: int
    height: int
    fps: int
    seconds: float
    stage_w: int
    stage_h: int

    @property
    def frames(self) -> int:
        return max(1, int(round(self.seconds * self.fps)))

    @property
    def blur(self) -> int:
        return max(1, int(round(self.stage_w / 60.0)))


def _even(value: float, minimum: int = 2) -> int:
    n = max(minimum, int(round(value)))
    return n if n % 2 == 0 else n + 1


def _geometry(width: int, height: int, fps: int, seconds: float, stage: int = 360) -> _Geometry:
    w, h = _even(width, 16), _even(height, 16)
    if w <= stage:
        sw, sh = w, h
    else:
        sw = _even(stage, 16)
        sh = _even(h * (sw / w), 16)
    return _Geometry(width=w, height=h, fps=max(1, int(fps)), seconds=max(0.2, float(seconds)),
                     stage_w=sw, stage_h=sh)


# -- background filter graphs ----------------------------------------------- #

def _graph_aurora_spiral(g: _Geometry, seed: int) -> str:
    """Slow spiral of saturated colour, soft-focused -- an aurora sweep."""
    return (
        f"gradients=s={g.stage_w}x{g.stage_h}:r={g.fps}:d={g.seconds:.3f}"
        f":c0=0x2B0B4E:c1=0x1B4FD8:c2=0x11C3B0:c3=0xF25C8A:nb_colors=4"
        f":speed=0.09:type=spiral:seed={seed},"
        "hue=h=20*sin(2*PI*t/11):s=1.20,"
        f"boxblur={g.blur}:1,"
        "eq=contrast=1.12:brightness=0.01,"
        f"scale={g.width}:{g.height}:flags=bicubic,"
        "vignette=PI/5"
    )


def _graph_cell_bloom(g: _Geometry, seed: int) -> str:
    """Drifting colonies of turquoise cells with a cool decay trail."""
    lw = _even(g.stage_w / 5.0, 16)
    lh = _even(g.stage_h / 5.0, 16)
    return (
        f"life=s={lw}x{lh}:r={g.fps}:mold=22:ratio=0.17"
        f":death_color=0x08161C:life_color=0x5FE8CB:mold_color=0x14415E:stitch=0:random_seed={seed},"
        f"scale={g.stage_w}:{g.stage_h}:flags=bicubic,"
        f"boxblur={g.blur}:1,"
        "colorchannelmixer=rr=0.85:gg=1.00:bb=1.10,"
        "hue=h=12*sin(2*PI*t/9):s=1.15,"
        f"scale={g.width}:{g.height}:flags=bicubic,"
        "vignette=PI/4.5"
    )


def _graph_violet_fractal(g: _Geometry, seed: int) -> str:
    """A slowly zooming fractal coastline, recoloured to violet and steel."""
    period = max(60, g.frames)
    return (
        f"mandelbrot=s={g.stage_w}x{g.stage_h}:r={g.fps}:maxiter=100"
        f":start_scale=1.8:end_scale=0.2:end_pts={max(30, int(g.frames * 0.75))}"
        ":inner=mincol:outer=normalized_iteration_count,"
        "hue=h=200+10*sin(2*PI*t/15):s=0.58,"
        "colorchannelmixer=rr=0.55:rg=0.10:rb=0.45:gr=0.10:gg=0.30:gb=0.50:br=0.30:bg=0.20:bb=0.95,"
        f"boxblur={g.blur}:1,"
        f"zoompan=z='1.05+0.04*sin(2*PI*on/{period})':d=1:s={g.stage_w}x{g.stage_h}:fps={g.fps},"
        "eq=brightness=0.02:contrast=1.08,"
        f"scale={g.width}:{g.height}:flags=bicubic,"
        "vignette=PI/4"
    )


def _graph_ember_mist(g: _Geometry, seed: int) -> str:
    """A warm, very soft cloud field that rolls instead of flickering."""
    nw = _even(g.stage_w / 40.0, 8)
    nh = _even(nw * (g.stage_h / g.stage_w), 8)
    wide = _even(g.stage_w * 1.3, 16)
    tall = _even(g.stage_h * 1.3, 16)
    return (
        f"color=c=0x808080:s={nw}x{nh}:r={g.fps},"
        f"noise=alls=100:allf=t:all_seed={seed},"
        "tmix=frames=24,"
        f"scale={wide}:{tall}:flags=bicubic,"
        "eq=contrast=2.2:saturation=0:brightness=0.03,"
        "colorchannelmixer=rr=1.05:rg=0.25:gg=0.40:gb=0.10:bb=0.30:br=0.35,"
        f"boxblur={max(2, g.blur * 2)}:2,"
        "rotate=0.09*sin(2*PI*t/17):c=none:ow=iw:oh=ih,"
        f"crop={g.stage_w}:{g.stage_h},"
        f"scale={g.width}:{g.height}:flags=bicubic,"
        "vignette=PI/4.5"
    )


# -- music filter graphs ----------------------------------------------------- #

def _fades(seconds: float) -> tuple[float, float, float]:
    fade_in = min(3.0, max(0.1, seconds * 0.15))
    fade_out = min(4.0, max(0.1, seconds * 0.2))
    return fade_in, fade_out, max(0.0, seconds - fade_out)


def _graph_ambient_glow(seconds: float) -> str:
    """A slow A-minor pad: four sine voices breathing under a gentle tremolo."""
    d = f"{seconds:.3f}"
    fade_in, fade_out, fade_at = _fades(seconds)
    return (
        f"sine=f=110:r={_SAMPLE_RATE}:d={d},volume=0.30[g0];"
        f"sine=f=164.81:r={_SAMPLE_RATE}:d={d},volume=0.20[g1];"
        f"sine=f=220:r={_SAMPLE_RATE}:d={d},volume=0.13[g2];"
        "aevalsrc=exprs='0.10*sin(2*PI*329.63*t)*(0.55+0.45*sin(2*PI*t/6.5))'"
        f":s={_SAMPLE_RATE}:d={d}:c=mono[g3];"
        "aevalsrc=exprs='0.05*sin(2*PI*493.88*t)*(0.5+0.5*sin(2*PI*t/9+1.2))'"
        f":s={_SAMPLE_RATE}:d={d}:c=mono[g4];"
        "[g0][g1][g2][g3][g4]amix=inputs=5:normalize=0,"
        "tremolo=f=0.25:d=0.30,"
        "highpass=f=45,lowpass=f=1600,volume=2.6,"
        f"afade=t=in:d={fade_in:.3f},afade=t=out:st={fade_at:.3f}:d={fade_out:.3f},"
        "alimiter=limit=0.9,aformat=channel_layouts=stereo"
    )


def _graph_soft_pulse(seconds: float) -> str:
    """An 80 BPM heartbeat bass under a D-minor shimmer and a brushed hiss."""
    d = f"{seconds:.3f}"
    fade_in, fade_out, fade_at = _fades(seconds)
    return (
        "aevalsrc=exprs='0.40*sin(2*PI*73.42*t)*exp(-4.5*mod(t\\,0.75))'"
        f":s={_SAMPLE_RATE}:d={d}:c=mono[p0];"
        "aevalsrc=exprs='0.16*sin(2*PI*293.66*t)*(0.45+0.55*sin(2*PI*t/5))'"
        f":s={_SAMPLE_RATE}:d={d}:c=mono[p1];"
        "aevalsrc=exprs='0.12*sin(2*PI*349.23*t)*(0.45+0.55*sin(2*PI*t/7+0.9))'"
        f":s={_SAMPLE_RATE}:d={d}:c=mono[p2];"
        "aevalsrc=exprs='0.09*sin(2*PI*440*t)*(0.4+0.6*sin(2*PI*t/11+2.1))'"
        f":s={_SAMPLE_RATE}:d={d}:c=mono[p3];"
        f"anoisesrc=color=pink:amplitude=0.05:r={_SAMPLE_RATE}:d={d},"
        "highpass=f=3500,tremolo=f=2.667:d=0.95[p4];"
        "[p0][p1][p2][p3][p4]amix=inputs=5:normalize=0,"
        "lowpass=f=6000,volume=1.2,"
        f"afade=t=in:d={fade_in:.3f},afade=t=out:st={fade_at:.3f}:d={fade_out:.3f},"
        "alimiter=limit=0.9,aformat=channel_layouts=stereo"
    )


# -- specs ------------------------------------------------------------------- #

@dataclass(frozen=True)
class _PlaceholderSpec:
    name: str
    kind: str
    suffix: str
    tags: tuple[str, ...]
    description: str


_BACKGROUND_GRAPHS: dict[str, Callable[[_Geometry, int], str]] = {
    "aurora_spiral": _graph_aurora_spiral,
    "cell_bloom": _graph_cell_bloom,
    "violet_fractal": _graph_violet_fractal,
    "ember_mist": _graph_ember_mist,
}

_MUSIC_GRAPHS: dict[str, Callable[[float], str]] = {
    "ambient_glow": _graph_ambient_glow,
    "soft_pulse": _graph_soft_pulse,
}

PLACEHOLDER_SPECS: tuple[_PlaceholderSpec, ...] = (
    _PlaceholderSpec(
        "aurora_spiral", "background", ".mp4",
        ("placeholder", "abstract", "loop", "gradient", "aurora", "cool", "vibrant"),
        "Spiralling aurora gradient",
    ),
    _PlaceholderSpec(
        "cell_bloom", "background", ".mp4",
        ("placeholder", "abstract", "loop", "cells", "organic", "teal", "dark"),
        "Blooming turquoise cell colonies",
    ),
    _PlaceholderSpec(
        "violet_fractal", "background", ".mp4",
        ("placeholder", "abstract", "loop", "fractal", "violet", "dreamy"),
        "Slow violet fractal tide",
    ),
    _PlaceholderSpec(
        "ember_mist", "background", ".mp4",
        ("placeholder", "abstract", "loop", "mist", "warm", "ember", "soft"),
        "Warm drifting ember mist",
    ),
    _PlaceholderSpec(
        "ambient_glow", "music", ".m4a",
        ("placeholder", "ambient", "calm", "pad", "warm", "loop"),
        "Breathing minor pad",
    ),
    _PlaceholderSpec(
        "soft_pulse", "music", ".m4a",
        ("placeholder", "ambient", "calm", "pulse", "rhythmic", "loop"),
        "Soft 80 BPM pulse bed",
    ),
)


def _part_path(out: Path) -> Path:
    """Per-process scratch name so two concurrent renders never share a temp file."""
    return out.with_name(f"{out.name}.{os.getpid()}.part")


def _spec_seed(settings: Settings, index: int) -> int:
    return abs(int(settings.seed) + index * 7919) % (2**31 - 1)


def _render_background(
    spec: _PlaceholderSpec, out: Path, geometry: _Geometry, seed: int, settings: Settings
) -> None:
    graph = _BACKGROUND_GRAPHS[spec.name](geometry, seed)
    tmp = _part_path(out)
    ff.run_ffmpeg(
        [
            "-y", "-f", "lavfi", "-i", graph,
            "-t", f"{geometry.seconds:.3f}", "-r", str(geometry.fps),
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-f", "mp4", str(tmp),
        ],
        settings=settings,
        timeout=600,
    )
    tmp.replace(out)


def _render_music(spec: _PlaceholderSpec, out: Path, seconds: float, settings: Settings) -> None:
    graph = _MUSIC_GRAPHS[spec.name](seconds)
    tmp = _part_path(out)
    ff.run_ffmpeg(
        [
            "-y", "-f", "lavfi", "-i", graph,
            "-t", f"{seconds:.3f}",
            "-c:a", "aac", "-b:a", "128k", "-ar", str(_SAMPLE_RATE), "-ac", "2",
            "-f", "mp4", str(tmp),
        ],
        settings=settings,
        timeout=600,
    )
    tmp.replace(out)


def _generate_placeholders(
    settings: Settings | None = None,
    *,
    width: int = PLACEHOLDER_WIDTH,
    height: int = PLACEHOLDER_HEIGHT,
    fps: int = PLACEHOLDER_FPS,
    background_seconds: float = PLACEHOLDER_BACKGROUND_SECONDS,
    music_seconds: float = PLACEHOLDER_MUSIC_SECONDS,
    kinds: Sequence[str] = KINDS,
    force: bool = False,
) -> list[Asset]:
    """Render the placeholder set.  Existing files are kept unless ``force``.

    ``kinds`` narrows generation to ``("background",)`` / ``("music",)``.  Tests use
    the smaller-parameter path; :func:`ensure_placeholders` calls this with the real
    1080x1920/30fps defaults.
    """
    s = _cfg(settings)
    geometry = _geometry(width, height, fps, background_seconds)
    wanted = tuple(kinds)
    for kind in wanted:
        try:
            _dir_for(s, kind).mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise AssetError(f"cannot create asset directory {_dir_for(s, kind)}: {exc}") from exc

    made: list[Asset] = []
    for index, spec in enumerate(PLACEHOLDER_SPECS):
        if spec.kind not in wanted:
            continue
        out = _dir_for(s, spec.kind) / f"{spec.name}{spec.suffix}"
        if force or not out.is_file() or out.stat().st_size == 0:
            log.info("generating placeholder %s asset %s", spec.kind, out.name)
            try:
                if spec.kind == "background":
                    _render_background(spec, out, geometry, _spec_seed(s, index), s)
                else:
                    _render_music(spec, out, max(0.2, float(music_seconds)), s)
            except Exception as exc:
                raise AssetError(f"could not generate placeholder asset {out.name}: {exc}") from exc
        made.append(
            Asset(
                path=out,
                name=spec.name,
                kind=spec.kind,
                tags=list(spec.tags),
                duration=_probe_duration(out, s),
            )
        )

    _upsert_manifest(s, made)
    names = {(a.kind, a.name) for a in made}
    catalogued = [a for a in library(s) if (a.kind, a.name) in names]
    return catalogued or made


def ensure_placeholders(settings: Settings | None = None) -> list[Asset]:
    """Make sure a usable starter library exists, generating it with ffmpeg lavfi.

    Four 20s 1080x1920 30fps h264 loops and two 30s music beds, synthesised from
    ``gradients``/``life``/``mandelbrot``/``noise`` and layered ``sine``/``aevalsrc``
    voices.  Idempotent -- assets already on disk are left untouched.  Raises
    :class:`AssetError` when generation is impossible (no ffmpeg, unwritable dir).
    """
    return _generate_placeholders(settings)
