"""Exception hierarchy for :mod:`aiclipper`."""

from __future__ import annotations

__all__ = [
    "AiclipperError", "MissingDependency", "IngestError", "TranscriptionError",
    "LLMError", "TTSError", "RenderError", "AssetError", "OverlayError",
]


class AiclipperError(RuntimeError):
    """Base class for every error raised by this package."""


class MissingDependency(AiclipperError):
    """An optional extra is required for this code path but is not installed."""

    def __init__(self, package: str, *, extra: str | None = None, purpose: str = ""):
        self.package = package
        self.extra = extra
        install = f"pip install 'aiclipper[{extra}]'" if extra else f"pip install {package}"
        detail = f" ({purpose})" if purpose else ""
        super().__init__(f"{package} is required{detail}. Install it with: {install}")


class IngestError(AiclipperError):
    """Could not fetch or probe the requested media."""


class TranscriptionError(AiclipperError):
    """Speech recognition failed."""


class LLMError(AiclipperError):
    """The language-model provider failed or returned unusable output."""


class TTSError(AiclipperError):
    """Speech synthesis failed."""


class RenderError(AiclipperError):
    """The timeline could not be rendered."""

    def __init__(self, message: str, *, problems: list[str] | None = None):
        self.problems = problems or []
        if self.problems:
            message = message + "\n  - " + "\n  - ".join(self.problems)
        super().__init__(message)


class AssetError(AiclipperError):
    """A background, music bed or font could not be resolved."""


class OverlayError(AiclipperError):
    """Chat or forum-card image generation failed."""
