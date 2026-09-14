"""The ElevenLabs backend: REST over :mod:`urllib.request`, no SDK.

One POST to ``/v1/text-to-speech/{voice_id}`` with ``xi-api-key`` returns mp3
bytes.  That is the entire integration, so there is no reason to take a
dependency for it -- the standard library is enough and the module stays
importable everywhere.

The API gives no word boundaries, so ``TTSResult.words`` is ``None`` and callers
that need caption timings force-align with :func:`aiclipper.transcribe.align`.
``VoiceSpec.rate`` has no API equivalent either; it is applied during the mp3
transcode as an ``atempo`` filter so the setting still means something.

:func:`build_request` is deliberately separate from the call that sends it: the
request shape is then testable with no network at all.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from .. import ffmpeg
from ..config import Settings
from ..errors import TTSError
from ..models import TTSResult, VoiceSpec
from . import base
from .base import _settings as _resolve_settings
from .voices import resolve_voice_id

log = logging.getLogger(__name__)

__all__ = [
    "ElevenLabsTTS", "API_ROOT", "PROBE_URL", "DEFAULT_MODEL", "DEFAULT_VOICE_ID",
    "build_request", "tempo_filters",
]

#: Documented endpoint root.
API_ROOT = "https://api.elevenlabs.io/v1/text-to-speech"

#: Multilingual model; overridable per call.
DEFAULT_MODEL = "eleven_multilingual_v2"

#: "Adam", one of the public sample voices, used when nothing else resolves.
DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"

#: Network timeout for one synthesis request, in seconds.
DEFAULT_TIMEOUT = 60.0

#: The cheapest authenticated endpoint there is: it returns the caller's own
#: user record, so it costs no credits and proves the key is live.  Used only by
#: the capability probe, never by a render.
PROBE_URL = "https://api.elevenlabs.io/v1/user"

#: ``atempo`` only accepts 0.5..2.0 per instance, so larger changes chain.
_ATEMPO_MIN = 0.5
_ATEMPO_MAX = 2.0


def build_request(
    text: str,
    voice_id: str,
    api_key: str,
    *,
    model: str = DEFAULT_MODEL,
    stability: float = 0.5,
    similarity_boost: float = 0.75,
    style: float = 0.0,
) -> urllib.request.Request:
    """Build the POST for one synthesis, without sending it.

    Pure and network-free, which is what makes the request shape testable in an
    offline CI.
    """
    if not (text or "").strip():
        raise TTSError("elevenlabs cannot synthesise empty text")
    if not voice_id:
        raise TTSError("elevenlabs needs a voice id")
    if not api_key:
        raise TTSError("ELEVENLABS_API_KEY is not set")

    payload = {
        "text": text,
        "model_id": model,
        "voice_settings": {
            "stability": float(stability),
            "similarity_boost": float(similarity_boost),
            "style": float(style),
        },
    }
    return urllib.request.Request(
        f"{API_ROOT}/{voice_id}",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "xi-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        method="POST",
    )


def tempo_filters(rate: float) -> list[str]:
    """``atempo`` chain implementing a playback-rate multiplier.

    ``atempo`` is limited to 0.5..2.0 per instance, so 3x becomes
    ``atempo=2.0,atempo=1.5``.  Returns ``[]`` for a rate of 1.
    """
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return []
    if value <= 0 or abs(value - 1.0) < 1e-3:
        return []
    value = max(0.25, min(4.0, value))
    chain: list[str] = []
    while value > _ATEMPO_MAX:
        chain.append(f"atempo={_ATEMPO_MAX:g}")
        value /= _ATEMPO_MAX
    while value < _ATEMPO_MIN:
        chain.append(f"atempo={_ATEMPO_MIN:g}")
        value /= _ATEMPO_MIN
    if abs(value - 1.0) >= 1e-3:
        chain.append(f"atempo={value:.4f}")
    return chain


class ElevenLabsTTS:
    """Speech via the ElevenLabs REST API."""

    name = "elevenlabs"

    def __init__(self, *, settings: Settings | None = None, model: str = DEFAULT_MODEL) -> None:
        self.settings = _resolve_settings(settings)
        self.model = model

    def available(self) -> bool:
        """A key is set, ffmpeg is present, and we are not running offline.

        A credential *presence* check, deliberately -- it is consulted on every
        render and must not touch the network.  Whether the key is accepted is
        :meth:`usable`.
        """
        if self.settings.offline:
            return False
        return bool(self.settings.elevenlabs_api_key) and ffmpeg.have_ffmpeg(self.settings)

    def usable(self, *, timeout: float | None = None, refresh: bool = False) -> bool:
        """Does the key this process holds actually open the API?

        :meth:`available` proves only that ``ELEVENLABS_API_KEY`` is a non-empty
        string -- a revoked, mistyped or out-of-quota key passes it, and a
        diagnostic that reports ``OK`` for one of those has misled the person
        running it.  This spends one authenticated GET on :data:`PROBE_URL`
        (the account record: no credits, no audio) bounded to ``timeout``
        seconds, and treats any non-200 or transport failure as unusable.

        Cached for the process, keyed by a fingerprint of the key and the
        timeout, so re-keying the environment re-probes but asking twice does
        not.  ``refresh=True`` probes again regardless.
        """
        if not self.available():
            return False
        limit = float(timeout) if timeout and timeout > 0 else float(base.USABLE_TIMEOUT)
        key = f"elevenlabs:{_fingerprint(self.settings.elevenlabs_api_key)}:{limit:g}"
        return base.cached_usable(key, lambda: self._probe(limit), refresh=refresh)

    def _probe(self, timeout: float) -> bool:
        """One cheap authenticated GET; ``False`` for anything but a 200."""
        api_key = self.settings.elevenlabs_api_key
        request = urllib.request.Request(
            PROBE_URL,
            headers={"xi-api-key": api_key, "Accept": "application/json"},
            method="GET",
        )

        def _ask() -> bool:
            try:
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    return int(getattr(response, "status", 200) or 200) == 200
            except urllib.error.HTTPError as exc:
                with contextlib.suppress(Exception):
                    exc.close()
                log.debug("elevenlabs probe rejected with HTTP %s", getattr(exc, "code", "?"))
                return False
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                log.debug("elevenlabs probe could not reach the API: %s", exc)
                return False

        # urlopen's timeout bounds each socket operation rather than the call as
        # a whole, so the probe gets an outer bound as well.
        return base.run_bounded(_ask, timeout + 0.5, default=False)

    def voice_id(self, voice: VoiceSpec | None) -> str:
        """Native ElevenLabs voice id for ``voice``."""
        return resolve_voice_id(voice, "elevenlabs", default=DEFAULT_VOICE_ID) or DEFAULT_VOICE_ID

    def _fetch(self, text: str, voice: VoiceSpec) -> bytes:
        if self.settings.offline:
            raise TTSError(
                "elevenlabs needs network access but settings.offline is set; "
                "use the 'offline' provider or unset AICLIP_OFFLINE"
            )
        key = self.settings.elevenlabs_api_key
        if not key:
            raise TTSError("ELEVENLABS_API_KEY is not set; export it or pick another tts provider")

        request = build_request(text, self.voice_id(voice), key, model=self.model)
        try:
            with urllib.request.urlopen(request, timeout=DEFAULT_TIMEOUT) as response:
                status = getattr(response, "status", 200) or 200
                body: bytes = response.read()
        except urllib.error.HTTPError as exc:
            try:
                detail = _decode(exc.read() if hasattr(exc, "read") else b"")
            finally:
                # HTTPError *is* the response object; leaving it open leaks a socket.
                with contextlib.suppress(Exception):
                    exc.close()
            raise TTSError(f"elevenlabs returned HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise TTSError(f"elevenlabs request failed: {exc.reason}") from exc
        except (TimeoutError, OSError) as exc:
            # A read timeout surfaces as a bare socket error, not a URLError.
            raise TTSError(f"elevenlabs request failed: {exc}") from exc

        if status != 200:
            raise TTSError(f"elevenlabs returned HTTP {status}: {_decode(body)}")
        if not body:
            raise TTSError("elevenlabs returned an empty response body")
        return body

    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult:
        """Speak ``text`` into ``out_path``.  ``words`` is always ``None``."""
        if not (text or "").strip():
            raise TTSError("elevenlabs cannot synthesise empty text")
        spec = voice or VoiceSpec()
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        audio = self._fetch(text, spec)
        with tempfile.TemporaryDirectory(prefix="aiclip-eleven-") as tmp:
            mp3 = Path(tmp) / "speech.mp3"
            mp3.write_bytes(audio)
            args: list[str] = ["-y", "-i", str(mp3), "-vn", "-ac", "1", "-ar", "44100"]
            chain = tempo_filters(spec.rate)
            if chain:
                args += ["-af", ",".join(chain)]
            args += [str(out)]
            ffmpeg.run_ffmpeg(args, settings=self.settings)

        return TTSResult(
            audio_path=out,
            duration=ffmpeg.probe(out, settings=self.settings).duration,
            words=None,
            voice=spec,
            text=text,
        )


def _fingerprint(api_key: str) -> str:
    """A short, non-reversible tag for a key, so the probe cache can be keyed on
    *which* key answered without ever holding the key itself."""
    return hashlib.sha256((api_key or "").encode("utf-8", "replace")).hexdigest()[:12]


def _decode(body: Any) -> str:
    if isinstance(body, bytes):
        text = body.decode("utf-8", "replace")
    else:
        text = str(body or "")
    text = text.strip()
    return text[:800] if text else "<empty body>"
