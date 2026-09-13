"""aiclipper -- a clean-room short-form video engine.

Auto-clipping, AI story videos, animated text-conversation and forum-story
templates, word-level captions and a declarative ffmpeg render core.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import Settings, get_settings
from .errors import AiclipperError, MissingDependency, RenderError
from .models import (
    AudioTrack, CaptionCue, CaptionStyle, ChatMessage, ChatScript, ClipCandidate,
    CropKeyframe, CropPath, MediaInfo, ProjectResult, RedditPost, RenderOptions,
    RenderResult, ScriptBeat, Segment, SubtitleTrack, Timeline, Transcript,
    TTSResult, VideoScript, VisualLayer, VoiceSpec, Word,
)

__all__ = [
    "__version__", "Settings", "get_settings",
    "AiclipperError", "MissingDependency", "RenderError",
    "AudioTrack", "CaptionCue", "CaptionStyle", "ChatMessage", "ChatScript",
    "ClipCandidate", "CropKeyframe", "CropPath", "MediaInfo", "ProjectResult",
    "RedditPost", "RenderOptions", "RenderResult", "ScriptBeat", "Segment",
    "SubtitleTrack", "Timeline", "Transcript", "TTSResult", "VideoScript",
    "VisualLayer", "VoiceSpec", "Word",
]
