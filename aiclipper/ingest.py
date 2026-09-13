"""Media ingest: turn a local path or an http(s) URL into a probed :class:`MediaInfo`.

This is the front door of the engine.  Everything downstream assumes it is
handed a local file that ffmpeg can decode, so this module has three jobs:

* :func:`resolve` -- accept whatever the user typed (path or URL) and give back
  a :class:`~aiclipper.models.MediaInfo`.
* :func:`download` -- fetch a URL with ``yt-dlp`` (Python module preferred, CLI
  as a fallback) into a content-addressed cache under ``settings.cache_dir`` so
  repeated runs never touch the network twice.
* :func:`normalize` -- re-encode an arbitrary input to a known-good mp4
  (h264 + aac, even dimensions, optional target size/fps).  This is what keeps
  exotic containers, odd pixel sizes and variable frame rates from reaching the
  renderer.

``yt-dlp`` is an optional extra: it is imported lazily inside the fetch helper
and :class:`~aiclipper.errors.MissingDependency` is raised at call time.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .config import Settings, get_settings
from .errors import IngestError, MissingDependency
from .ffmpeg import FFmpegError, probe, run_ffmpeg
from .models import MediaInfo

log = logging.getLogger(__name__)

__all__ = ["DEFAULT_QUALITY", "cache_key", "download", "is_url", "normalize", "resolve"]

#: Default yt-dlp format selector: best <=1080p video + best audio, muxed.
DEFAULT_QUALITY = "bv*[height<=1080]+ba/b[height<=1080]"

_ALLOWED_SCHEMES = frozenset({"http", "https"})

#: Sidecar file recorded next to a cached download.
_META_NAME = "source.json"

#: Files inside a download directory that are never the media itself: yt-dlp's
#: scratch files plus the sidecars/thumbnails/subtitles it can drop alongside.
_NON_MEDIA_SUFFIXES = frozenset({
    ".json", ".part", ".ytdl", ".tmp", ".temp", ".txt", ".log", ".description",
    ".vtt", ".srt", ".ass", ".lrc", ".jpg", ".jpeg", ".png", ".webp",
})


# --------------------------------------------------------------------------- #
# url handling
# --------------------------------------------------------------------------- #

def is_url(source: str) -> bool:
    """True when ``source`` is a fetchable ``http``/``https`` URL.

    Anything else -- a local path, ``file://``, ``ftp://``, a Windows drive
    letter, a URL with whitespace/control characters or with ``..`` traversal
    segments in its path -- is rejected, so callers can safely branch on this
    to decide between "probe a file" and "hit the network".
    """
    if not isinstance(source, str):
        return False
    text = source.strip()
    if not text:
        return False
    if any(ch.isspace() or ord(ch) < 32 or ord(ch) == 127 for ch in text):
        return False
    try:
        parts = urlsplit(text)
    except ValueError:
        return False
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        return False
    try:
        host = parts.hostname
    except ValueError:
        return False
    if not host:
        return False
    segments = parts.path.split("/")
    if ".." in segments or "%2e%2e" in (s.lower() for s in segments):
        return False
    return True


def cache_key(url: str, quality: str = DEFAULT_QUALITY) -> str:
    """Stable 16-hex-char cache key for ``(url, quality)``.

    ``sha256`` of the url and the format selector (NUL separated so a url that
    ends in the next field's text cannot collide with another pair).
    """
    digest = hashlib.sha256(f"{url}\x00{quality}".encode()).hexdigest()
    return digest[:16]


# --------------------------------------------------------------------------- #
# cache plumbing
# --------------------------------------------------------------------------- #

def _cache_slot(settings: Settings, url: str, quality: str) -> Path:
    return settings.cache_dir / cache_key(url, quality)


def _scan_media(directory: Path) -> Path | None:
    """The largest plausible media file in ``directory``.

    Only used to locate what a fetch backend just wrote; cache-hit detection
    goes through :func:`_cached_media`, which is far stricter.
    """
    if not directory.is_dir():
        return None
    candidates = [
        p for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() not in _NON_MEDIA_SUFFIXES and p.stat().st_size > 0
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_size)


def _cached_media(slot: Path) -> tuple[Path, dict[str, Any]] | None:
    """The media + sidecar of a *complete* cache slot, else ``None``.

    ``source.json`` is written only once the download has finished and the file
    is in place, so it is the sole evidence that a slot is trustworthy.  A slot
    without a readable sidecar naming an existing file is a crashed or
    half-written attempt; reporting it as a miss is what lets the next call
    re-fetch instead of serving a fragment (or a permanent probe failure)
    forever.
    """
    if not slot.is_dir():
        return None
    meta = _read_meta(slot)
    name = meta.get("file")
    if not isinstance(name, str) or not name:
        return None
    media = slot / Path(name).name
    try:
        if not media.is_file() or media.stat().st_size <= 0:
            return None
    except OSError:  # pragma: no cover - racy filesystem
        return None
    return media, meta


def _clear_slot(slot: Path) -> None:
    """Empty a half-finished cache slot so a retry starts from clean ground."""
    if not slot.is_dir():
        return
    for child in slot.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
        except OSError:  # pragma: no cover - racy filesystem
            log.debug("could not remove stale cache entry %s", child, exc_info=True)


def _read_meta(slot: Path) -> dict[str, Any]:
    path = slot / _META_NAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        log.debug("ignoring unreadable cache sidecar %s", path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _write_meta(slot: Path, url: str, quality: str, media: Path, info: dict[str, Any]) -> dict[str, Any]:
    """Record the sidecar that marks ``slot`` as a complete cache entry."""
    payload: dict[str, Any] = {
        "url": url,
        "quality": quality,
        "file": media.name,
        "title": info.get("title") or None,
        "duration": info.get("duration"),
        "extractor": info.get("extractor_key") or info.get("extractor"),
        "id": info.get("id"),
    }
    path = slot / _META_NAME
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # atomic: a slot is never half-marked as complete
    return payload


def _place(media: Path, dest_dir: Path) -> Path:
    """Make ``media`` available inside ``dest_dir`` (hardlink, else copy)."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    if media.parent.resolve() == dest_dir.resolve():
        return media
    target = dest_dir / media.name
    if target.exists():
        if target.is_dir():
            raise IngestError(f"cannot place {media.name} into {dest_dir}: a directory is in the way")
        try:
            if target.samefile(media):
                return target
        except OSError:  # pragma: no cover - racy filesystem
            pass
        # Same name, different inode: it is a stale copy, not this download.
        # Size-matching was the old test here and happily served the wrong file.
        target.unlink()
    try:
        os.link(media, target)
    except OSError:
        shutil.copy2(media, target)
    return target


# --------------------------------------------------------------------------- #
# yt-dlp backends
# --------------------------------------------------------------------------- #

def _load_yt_dlp() -> Any | None:
    """Import ``yt_dlp`` lazily; ``None`` when it is not installed."""
    try:
        import yt_dlp
    except Exception:  # pragma: no cover - depends on the environment
        return None
    return yt_dlp


def _yt_dlp_cli() -> str | None:
    return shutil.which("yt-dlp") or shutil.which("youtube-dl")


def _outtmpl(dest_dir: Path) -> str:
    return str(dest_dir / "%(title).80s-%(id)s.%(ext)s")


def _fetch_with_module(module: Any, url: str, dest_dir: Path, quality: str) -> tuple[Path, dict[str, Any]]:
    opts = {
        "outtmpl": _outtmpl(dest_dir),
        "format": quality,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 3,
        "ignoreerrors": False,
    }
    try:
        with module.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=True)
    except Exception as exc:  # yt-dlp raises a zoo of DownloadError subclasses
        raise IngestError(f"yt-dlp could not download {url}: {exc}") from exc
    if not isinstance(info, dict):
        raise IngestError(f"yt-dlp returned no metadata for {url}")
    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if isinstance(e, dict)]
        if not entries:
            raise IngestError(f"{url} resolved to an empty playlist")
        info = entries[0]

    path: Path | None = None
    for requested in info.get("requested_downloads") or []:
        candidate = requested.get("filepath") or requested.get("_filename")
        if candidate and Path(candidate).is_file():
            path = Path(candidate)
            break
    if path is None:
        candidate = info.get("filepath") or info.get("_filename")
        if candidate and Path(candidate).is_file():
            path = Path(candidate)
    if path is None:
        path = _scan_media(dest_dir)
    if path is None:
        raise IngestError(f"yt-dlp reported success but produced no file for {url}")
    return path, info


def _fetch_with_cli(exe: str, url: str, dest_dir: Path, quality: str) -> tuple[Path, dict[str, Any]]:
    cmd = [
        exe, "--no-playlist", "--no-progress", "--no-warnings",
        "-f", quality, "--merge-output-format", "mp4",
        "-o", _outtmpl(dest_dir),
        "--print", "after_move:%(title)s",
        "--print", "after_move:filepath",
        url,
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        raise IngestError(f"yt-dlp CLI failed for {url}: {exc}") from exc
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-10:])
        raise IngestError(f"yt-dlp CLI exited {proc.returncode} for {url}\n{tail}")

    lines = [ln.strip() for ln in (proc.stdout or "").splitlines() if ln.strip()]
    path: Path | None = None
    for line in reversed(lines):
        if Path(line).is_file():
            path = Path(line)
            break
    if path is None:
        path = _scan_media(dest_dir)
    if path is None:
        raise IngestError(f"yt-dlp CLI produced no file for {url}")
    title = lines[0] if lines and not Path(lines[0]).is_file() else path.stem
    return path, {"title": title}


def _fetch(url: str, dest_dir: Path, *, quality: str, settings: Settings) -> tuple[Path, dict[str, Any]]:
    """Download ``url`` into ``dest_dir``.  The only place that touches the network.

    Tests monkeypatch this symbol to prove a cache hit never reaches it.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    module = _load_yt_dlp()
    if module is not None:
        return _fetch_with_module(module, url, dest_dir, quality)
    exe = _yt_dlp_cli()
    if exe:
        return _fetch_with_cli(exe, url, dest_dir, quality)
    raise MissingDependency("yt-dlp", extra="ingest", purpose="downloading media from a URL")


# --------------------------------------------------------------------------- #
# public api
# --------------------------------------------------------------------------- #

def _probe(path: Path, *, settings: Settings) -> MediaInfo:
    try:
        return probe(path, settings=settings)
    except FileNotFoundError as exc:
        raise IngestError(f"no such media file: {path}") from exc
    except FFmpegError as exc:
        raise IngestError(f"could not probe {path}: {exc}") from exc


def download(
    url: str,
    dest_dir: Path,
    *,
    settings: Settings | None = None,
    quality: str = DEFAULT_QUALITY,
) -> MediaInfo:
    """Fetch ``url`` with yt-dlp and return its :class:`MediaInfo`.

    The bytes live in a cache slot under ``settings.cache_dir`` keyed by
    ``sha256(url + quality)[:16]``; ``dest_dir`` gets a hardlink (or a copy when
    hardlinking is impossible).  A complete cache slot short-circuits before any
    network access, and its ``source.json`` sidecar restores ``title`` /
    ``source_url`` on those hits.  The sidecar is written last and atomically,
    so a slot missing one is treated as a crashed attempt: it is cleared and
    re-fetched rather than serving a fragment forever.  With
    ``settings.offline`` set, a miss raises instead of reaching the network.
    """
    s = settings or get_settings()
    if not is_url(url):
        raise IngestError(f"not an http(s) URL: {url!r}")

    dest = Path(dest_dir)
    slot = _cache_slot(s, url, quality)

    hit = _cached_media(slot)
    if hit is not None:
        media, meta = hit
    else:
        if s.offline:
            stale = " -- an incomplete cache entry is present" if slot.is_dir() else ""
            raise IngestError(f"offline mode is on and {url} is not in the cache ({slot}){stale}")
        _clear_slot(slot)  # discard anything a previous, crashed attempt left behind
        slot.mkdir(parents=True, exist_ok=True)
        log.info("downloading %s", url)
        fetched, info = _fetch(url, slot, quality=quality, settings=s)
        media = _place(Path(fetched), slot)
        meta = _write_meta(slot, url, quality, media, info)

    local = _place(media, dest)
    out = _probe(local, settings=s)
    out.title = meta.get("title") or out.title or local.stem
    out.source_url = meta.get("url") or url
    return out


def resolve(
    source: str | Path,
    *,
    settings: Settings | None = None,
    workspace: Path | None = None,
    quality: str = DEFAULT_QUALITY,
) -> MediaInfo:
    """Resolve a user-supplied ``source`` to a probed local media file.

    A :class:`~pathlib.Path`, or a string that is not an http(s) URL, is treated
    as a local file and merely probed.  An http(s) URL is downloaded via
    :func:`download` -- into ``workspace`` when given, otherwise straight into
    the download cache.
    """
    s = settings or get_settings()
    text = str(source) if isinstance(source, Path) else str(source).strip()
    if not text:
        raise IngestError("empty media source: pass a file path or an http(s) URL")
    if isinstance(source, Path) or not is_url(text):
        path = Path(text).expanduser()
        if not path.exists():
            raise IngestError(f"no such media file: {path}")
        if path.is_dir():
            raise IngestError(f"expected a media file, got a directory: {path}")
        info = _probe(path, settings=s)
        info.title = info.title or path.stem
        return info

    dest = Path(workspace) if workspace is not None else _cache_slot(s, text, quality)
    return download(text, dest, settings=s, quality=quality)


def _even(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def _scale_filters(width: int | None, height: int | None) -> list[str]:
    """Scale filters that preserve aspect ratio and guarantee even dimensions."""
    if width and height:
        return [
            f"scale={_even(width)}:{_even(height)}:force_original_aspect_ratio=decrease",
            "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        ]
    if width:
        return [f"scale={_even(width)}:-2"]
    if height:
        return [f"scale=-2:{_even(height)}"]
    return ["scale=trunc(iw/2)*2:trunc(ih/2)*2"]


def normalize(
    info: MediaInfo,
    dest: Path,
    *,
    width: int | None = None,
    height: int | None = None,
    fps: float | None = None,
    settings: Settings | None = None,
) -> MediaInfo:
    """Re-encode ``info`` to a known-good mp4 and return the new :class:`MediaInfo`.

    h264 + aac, ``yuv420p``, square pixels, even dimensions, faststart.  Scaling
    only happens when ``width`` and/or ``height`` are given and always preserves
    the source aspect ratio (with both given the picture is fitted *inside* that
    box).  ``fps`` retimes the video; ``None`` keeps the source rate.  A silent
    stereo track is added when the source has no audio so downstream mixing
    always has something to work with.

    ``width`` / ``height`` / ``fps`` must be positive when given, and ``dest``
    may not resolve to ``info.path`` -- ffmpeg would truncate the file it is
    still reading.  Both raise :class:`~aiclipper.errors.IngestError`.
    """
    s = settings or get_settings()
    for name, value in (("width", width), ("height", height), ("fps", fps)):
        if value is not None and not value > 0:
            raise IngestError(f"normalize(): {name} must be positive, got {value!r}")

    src = Path(info.path)
    if not src.is_file():
        raise IngestError(f"no such media file: {src}")

    out = Path(dest)
    if not out.suffix:
        out = out.with_suffix(".mp4")
    if out.resolve() == src.resolve():
        raise IngestError(f"normalize() cannot write over its own source: {src}")
    out.parent.mkdir(parents=True, exist_ok=True)

    has_video = bool(info.has_video)
    has_audio = bool(info.has_audio)
    if not has_video and not has_audio:
        raise IngestError(f"{src} has neither a video nor an audio stream")
    add_silence = has_video and not has_audio

    args: list[str] = ["-y", "-i", str(src)]
    if add_silence:
        args += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000"]
        args += ["-map", "0:v:0", "-map", "1:a:0"]

    if has_video:
        filters = _scale_filters(width, height)
        if fps is not None:
            filters.append(f"fps={float(fps):g}")
        filters.append("setsar=1")
        args += ["-vf", ",".join(filters)]
        args += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        if fps is not None:
            args += ["-r", f"{float(fps):g}"]
    else:
        args += ["-vn"]

    if has_audio or add_silence:
        args += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2"]
    else:
        args += ["-an"]
    if add_silence:
        args += ["-shortest"]

    args += ["-movflags", "+faststart", str(out)]

    try:
        run_ffmpeg(args, settings=s)
    except FFmpegError as exc:
        raise IngestError(f"could not normalize {src}: {exc}") from exc

    result = _probe(out, settings=s)
    result.title = info.title
    result.source_url = info.source_url
    return result
