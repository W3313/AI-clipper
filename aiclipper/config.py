"""Runtime configuration.

Everything is overridable by environment variable so the same code runs on a
laptop, in CI, and inside a container with no edits.  Nothing here performs
network access or imports optional dependencies.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def _env_path(name: str, default: Path) -> Path:
    raw = _env(name)
    return Path(raw).expanduser() if raw else default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _env(name).lower()
    if not raw:
        return default
    return raw in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


@dataclass
class Settings:
    """Resolved settings for one run."""

    # -- filesystem -------------------------------------------------------- #
    work_dir: Path = field(default_factory=lambda: _env_path("AICLIP_WORK_DIR", PROJECT_ROOT / "work"))
    output_dir: Path = field(default_factory=lambda: _env_path("AICLIP_OUTPUT_DIR", PROJECT_ROOT / "out"))
    assets_dir: Path = field(default_factory=lambda: _env_path("AICLIP_ASSETS_DIR", PROJECT_ROOT / "assets"))

    # -- binaries ---------------------------------------------------------- #
    ffmpeg: str = field(default_factory=lambda: _env("AICLIP_FFMPEG", "ffmpeg"))
    ffprobe: str = field(default_factory=lambda: _env("AICLIP_FFPROBE", "ffprobe"))
    chromium: str = field(default_factory=lambda: _env("AICLIP_CHROMIUM"))

    # -- canvas ------------------------------------------------------------ #
    width: int = field(default_factory=lambda: _env_int("AICLIP_WIDTH", 1080))
    height: int = field(default_factory=lambda: _env_int("AICLIP_HEIGHT", 1920))
    fps: int = field(default_factory=lambda: _env_int("AICLIP_FPS", 30))

    # -- speech recognition ------------------------------------------------ #
    whisper_model: str = field(default_factory=lambda: _env("AICLIP_WHISPER_MODEL", "base"))
    whisper_device: str = field(default_factory=lambda: _env("AICLIP_WHISPER_DEVICE", "auto"))
    whisper_compute_type: str = field(default_factory=lambda: _env("AICLIP_WHISPER_COMPUTE", "int8"))

    # -- language model ---------------------------------------------------- #
    llm_provider: str = field(default_factory=lambda: _env("AICLIP_LLM", "auto"))
    llm_model: str = field(default_factory=lambda: _env("AICLIP_LLM_MODEL", "claude-opus-5"))
    #: OpenAI-compatible endpoint for a locally hosted model.  Ollama, llama.cpp's
    #: server, LM Studio and vLLM all speak this protocol, so one client reaches
    #: every one of them.  Include the version prefix, e.g. ".../v1".
    llm_base_url: str = field(default_factory=lambda: _env("AICLIP_LLM_BASE_URL", "http://localhost:11434/v1"))
    llm_local_model: str = field(default_factory=lambda: _env("AICLIP_LLM_LOCAL_MODEL", "llama3.1"))
    llm_timeout: float = field(default_factory=lambda: _env_float("AICLIP_LLM_TIMEOUT", 120.0))
    llm_effort: str = field(default_factory=lambda: _env("AICLIP_LLM_EFFORT", "medium"))
    llm_max_tokens: int = field(default_factory=lambda: _env_int("AICLIP_LLM_MAX_TOKENS", 16000))

    # -- speech synthesis -------------------------------------------------- #
    tts_provider: str = field(default_factory=lambda: _env("AICLIP_TTS", "auto"))
    tts_voice: str = field(default_factory=lambda: _env("AICLIP_VOICE", ""))
    #: Local neural speech.  ``piper_binary`` is resolved on PATH unless given an
    #: absolute path; ``piper_voice_dir`` holds the downloaded .onnx voices.
    piper_binary: str = field(default_factory=lambda: _env("AICLIP_PIPER", "piper"))
    piper_voice: str = field(default_factory=lambda: _env("AICLIP_PIPER_VOICE", ""))

    # -- behaviour --------------------------------------------------------- #
    offline: bool = field(default_factory=lambda: _env_bool("AICLIP_OFFLINE", False))
    keep_work: bool = field(default_factory=lambda: _env_bool("AICLIP_KEEP_WORK", True))
    seed: int = field(default_factory=lambda: _env_int("AICLIP_SEED", 1234))

    # -- derived ----------------------------------------------------------- #
    @property
    def backgrounds_dir(self) -> Path:
        return self.assets_dir / "backgrounds"

    @property
    def music_dir(self) -> Path:
        return self.assets_dir / "music"

    @property
    def fonts_dir(self) -> Path:
        return self.assets_dir / "fonts"

    @property
    def cache_dir(self) -> Path:
        return self.work_dir / "cache"

    @property
    def piper_voice_dir(self) -> Path:
        return _env_path("AICLIP_PIPER_VOICES", self.assets_dir / "piper")

    @property
    def llm_api_key(self) -> str:
        """Token for the OpenAI-compatible endpoint.

        Local servers usually need none; a placeholder is sent so that clients
        which insist on an Authorization header still work.
        """
        for var in ("AICLIP_LLM_API_KEY", "OPENAI_API_KEY"):
            value = os.environ.get(var, "").strip()
            if value:
                return value
        return ""

    @property
    def anthropic_api_key(self) -> str:
        return os.environ.get("ANTHROPIC_API_KEY", "").strip()

    @property
    def elevenlabs_api_key(self) -> str:
        return os.environ.get("ELEVENLABS_API_KEY", "").strip()

    def ensure_dirs(self) -> Settings:
        for path in (self.work_dir, self.output_dir, self.cache_dir, self.backgrounds_dir, self.music_dir):
            path.mkdir(parents=True, exist_ok=True)
        return self

    def workspace(self, name: str) -> Path:
        """A clean-ish scratch directory for one job."""
        safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name)[:80] or "job"
        path = self.work_dir / safe
        path.mkdir(parents=True, exist_ok=True)
        return path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings singleton (call :func:`reset_settings` in tests)."""
    return Settings()


def reset_settings() -> None:
    get_settings.cache_clear()
