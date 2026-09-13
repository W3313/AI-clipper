"""Thin, dependency-free wrapper around the ffmpeg/ffprobe binaries.

Every module that shells out to ffmpeg goes through here so that binary
resolution, error reporting and logging behave identically everywhere.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Settings, get_settings
from .models import MediaInfo

log = logging.getLogger(__name__)

__all__ = [
    "FFmpegError", "FFmpegMissing",
    "ffmpeg_bin", "ffprobe_bin", "have_ffmpeg",
    "run_ffmpeg", "run_ffprobe", "probe", "probe_duration",
    "extract_audio", "audio_duration", "make_silence", "make_tone",
    "has_encoder", "has_filter", "escape_filter_path",
]


class FFmpegError(RuntimeError):
    """Raised when ffmpeg/ffprobe exits non-zero."""

    def __init__(self, message: str, *, command: Sequence[str] | None = None, stderr: str = ""):
        super().__init__(message)
        self.command = list(command or [])
        self.stderr = stderr

    def __str__(self) -> str:  # pragma: no cover - formatting only
        base = super().__str__()
        tail = "\n".join(self.stderr.strip().splitlines()[-20:])
        return f"{base}\n--- ffmpeg stderr ---\n{tail}" if tail else base


class FFmpegMissing(FFmpegError):
    """ffmpeg is not installed or not on PATH."""


def ffmpeg_bin(settings: Settings | None = None) -> str:
    return (settings or get_settings()).ffmpeg


def ffprobe_bin(settings: Settings | None = None) -> str:
    return (settings or get_settings()).ffprobe


def have_ffmpeg(settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    return bool(shutil.which(s.ffmpeg) and shutil.which(s.ffprobe))


def _require(binary: str) -> str:
    resolved = shutil.which(binary)
    if not resolved:
        raise FFmpegMissing(
            f"{binary!r} not found on PATH. Install ffmpeg (apt install ffmpeg / brew install ffmpeg) "
            f"or set AICLIP_FFMPEG / AICLIP_FFPROBE."
        )
    return resolved


def run_ffmpeg(
    args: Sequence[str],
    *,
    settings: Settings | None = None,
    log_path: Path | None = None,
    timeout: float | None = None,
    quiet: bool = True,
) -> subprocess.CompletedProcess:
    """Run ffmpeg with ``args`` (without the binary name). Raises on failure."""
    s = settings or get_settings()
    cmd = [_require(s.ffmpeg), "-hide_banner"]
    if quiet:
        cmd += ["-loglevel", "error", "-nostats"]
    cmd += list(args)
    log.debug("ffmpeg %s", " ".join(cmd[1:]))
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(" ".join(cmd) + "\n\n" + proc.stderr, encoding="utf-8")
    if proc.returncode != 0:
        raise FFmpegError(f"ffmpeg exited {proc.returncode}", command=cmd, stderr=proc.stderr)
    return proc


def run_ffprobe(args: Sequence[str], *, settings: Settings | None = None, timeout: float | None = 60) -> str:
    s = settings or get_settings()
    cmd = [_require(s.ffprobe), "-hide_banner", "-loglevel", "error", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise FFmpegError(f"ffprobe exited {proc.returncode}", command=cmd, stderr=proc.stderr)
    return proc.stdout


def _probe_json(path: str | Path, *, settings: Settings | None = None) -> dict[str, Any]:
    out = run_ffprobe(
        ["-print_format", "json", "-show_format", "-show_streams", str(path)],
        settings=settings,
    )
    return json.loads(out or "{}")


def _parse_fps(rate: str | None) -> float:
    if not rate or rate in {"0/0", "N/A"}:
        return 0.0
    if "/" in rate:
        num, _, den = rate.partition("/")
        try:
            d = float(den)
            return float(num) / d if d else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0


def probe(path: str | Path, *, settings: Settings | None = None) -> MediaInfo:
    """Return a :class:`MediaInfo` for a local media file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"media file not found: {p}")
    data = _probe_json(p, settings=settings)
    fmt = data.get("format", {}) or {}
    streams = data.get("streams", []) or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for candidate in (fmt.get("duration"), (video or {}).get("duration"), (audio or {}).get("duration")):
        try:
            duration = float(candidate)
            if duration > 0:
                break
        except (TypeError, ValueError):
            continue

    tags = fmt.get("tags", {}) or {}
    return MediaInfo(
        path=p,
        duration=max(0.0, duration),
        width=int((video or {}).get("width") or 0),
        height=int((video or {}).get("height") or 0),
        fps=_parse_fps((video or {}).get("avg_frame_rate") or (video or {}).get("r_frame_rate")),
        has_video=video is not None,
        has_audio=audio is not None,
        title=tags.get("title"),
        size_bytes=int(fmt.get("size") or (p.stat().st_size if p.exists() else 0)),
    )


def probe_duration(path: str | Path, *, settings: Settings | None = None) -> float:
    return probe(path, settings=settings).duration


audio_duration = probe_duration


def extract_audio(
    src: str | Path,
    dst: str | Path,
    *,
    sample_rate: int = 16000,
    mono: bool = True,
    settings: Settings | None = None,
) -> Path:
    """Decode any media file to WAV suitable for ASR."""
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-y", "-i", str(src), "-vn", "-ac", "1" if mono else "2",
         "-ar", str(sample_rate), "-c:a", "pcm_s16le", str(out)],
        settings=settings,
    )
    return out


def make_silence(
    duration: float,
    dst: str | Path,
    *,
    sample_rate: int = 44100,
    settings: Settings | None = None,
) -> Path:
    """Generate a silent audio file -- the offline TTS fallback leans on this."""
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-y", "-f", "lavfi", "-i", f"anullsrc=r={sample_rate}:cl=mono",
         "-t", f"{max(0.05, duration):.3f}", "-c:a", "pcm_s16le", str(out)],
        settings=settings,
    )
    return out


def make_tone(
    duration: float,
    dst: str | Path,
    *,
    frequency: float = 220.0,
    volume: float = 0.08,
    sample_rate: int = 44100,
    settings: Settings | None = None,
) -> Path:
    """Generate a quiet sine tone (placeholder music bed / offline narration)."""
    out = Path(dst)
    out.parent.mkdir(parents=True, exist_ok=True)
    run_ffmpeg(
        ["-y", "-f", "lavfi",
         "-i", f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={max(0.05, duration):.3f}",
         "-af", f"volume={volume}", "-c:a", "pcm_s16le", str(out)],
        settings=settings,
    )
    return out


def _capability(kind: str, name: str, settings: Settings | None = None) -> bool:
    s = settings or get_settings()
    try:
        proc = subprocess.run(
            [_require(s.ffmpeg), "-hide_banner", f"-{kind}"],
            capture_output=True, text=True, timeout=30,
        )
    except (FFmpegMissing, subprocess.SubprocessError):
        return False
    return any(name in line.split() for line in proc.stdout.splitlines())


def has_encoder(name: str, settings: Settings | None = None) -> bool:
    return _capability("encoders", name, settings)


def has_filter(name: str, settings: Settings | None = None) -> bool:
    return _capability("filters", name, settings)


def escape_filter_path(path: str | Path) -> str:
    """Escape a filesystem path for use inside an ffmpeg filter argument.

    ``subtitles=`` and friends parse ``:``, ``'`` and ``\\`` specially.
    """
    text = str(path)
    text = text.replace("\\", "/")
    text = text.replace(":", r"\:")
    text = text.replace("'", r"\'")
    text = text.replace("[", r"\[").replace("]", r"\]")
    text = text.replace(",", r"\,")
    return text
