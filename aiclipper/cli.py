"""Command line interface: ``aiclip <command> [options]``.

Nine subcommands.  Five of them drive a pipeline (``clip``, ``story``, ``texts``,
``reddit``, ``split``) and print the produced path(s) on stdout, one per line, so
the tool composes in a shell.  Four report on the installation instead
(``voices``, ``styles``, ``assets``, ``doctor``).

Global flags (``--work-dir``, ``--output-dir``, ``--offline``, ``--seed``,
``--width``, ``--height``, ``--fps``, ``-v``, ``--dry-run``) live on a *parent*
parser, so they are accepted both before and after the subcommand name::

    aiclip --width 720 story --topic "..."
    aiclip story --width 720 --topic "..."

Configuration goes through the environment, which is the supported way to reach
:class:`~aiclipper.config.Settings`: each global flag sets its ``AICLIP_*``
variable, :func:`~aiclipper.config.reset_settings` clears the settings cache, and
the previous environment is restored when the command finishes -- so calling
:func:`main` from a test or another program leaves no residue.

:func:`validate` runs before any command does work: every enumerated option
(caption style, chat and forum theme, voice name, overlay backend) and every
numeric floor is checked up front, so a typo costs a line of stderr instead of a
finished render.  The numeric floors are enforced by the parser itself.

:func:`main` returns ``0`` on success, ``1`` on a handled failure (one clear line
on stderr; a traceback only with ``-v``) and ``2`` on bad usage.  Nothing in this
module imports an optional dependency at import time.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import traceback
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from . import __version__
from .config import Settings, get_settings, reset_settings

__all__ = [
    "PROG",
    "GLOBAL_ENV",
    "OPTIONAL_EXTRAS",
    "MIN_COUNT",
    "MIN_TURNS",
    "MIN_WORDS",
    "MIN_SECONDS",
    "DOCTOR_TTS_TIMEOUT",
    "UsageError",
    "build_parser",
    "validate",
    "environment_for",
    "read_script",
    "main",
]

PROG = "aiclip"

#: Global flag ``dest`` -> the environment variable :class:`Settings` reads.
GLOBAL_ENV: dict[str, str] = {
    "work_dir": "AICLIP_WORK_DIR",
    "output_dir": "AICLIP_OUTPUT_DIR",
    "offline": "AICLIP_OFFLINE",
    "seed": "AICLIP_SEED",
    "width": "AICLIP_WIDTH",
    "height": "AICLIP_HEIGHT",
    "fps": "AICLIP_FPS",
}

#: ``doctor`` rows for the optional extras: label -> (module, pip extra).
OPTIONAL_EXTRAS: tuple[tuple[str, str, str], ...] = (
    ("faster-whisper", "faster_whisper", "transcribe"),
    ("yt-dlp", "yt_dlp", "ingest"),
    ("opencv", "cv2", "vision"),
    ("anthropic", "anthropic", "llm"),
    ("pydantic", "pydantic", "llm"),
    ("edge-tts", "edge_tts", "tts"),
    ("playwright", "playwright", "overlays"),
)

_TTS_BACKENDS = ("edge", "elevenlabs", "piper", "offline")

#: Floors for the numeric options.  A degenerate value (no clips, a fifth of a
#: second of video) is a usage error, not something to render and then explain.
MIN_COUNT = 1
MIN_TURNS = 1
MIN_WORDS = 20
MIN_SECONDS = 1.0

#: Seconds ``doctor`` gives each TTS backend to prove it can synthesise.
DOCTOR_TTS_TIMEOUT = 4.0

log = logging.getLogger("aiclipper.cli")


class UsageError(Exception):
    """The arguments parsed, but the combination cannot be acted on (exit 2)."""


# --------------------------------------------------------------------------- #
# settings plumbing
# --------------------------------------------------------------------------- #

def environment_for(ns: argparse.Namespace) -> dict[str, str]:
    """The ``AICLIP_*`` overrides implied by the global flags actually given."""
    env: dict[str, str] = {}
    for dest, name in GLOBAL_ENV.items():
        value = getattr(ns, dest, None)
        if value is None or value is False:
            continue
        env[name] = "1" if value is True else str(value)
    return env


@contextmanager
def _applied(env: dict[str, str]) -> Iterator[Settings]:
    """Apply ``env``, refresh the settings cache, restore both on the way out."""
    previous = {name: os.environ.get(name) for name in env}
    os.environ.update(env)
    reset_settings()
    try:
        yield get_settings()
    finally:
        for name, old in previous.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        reset_settings()


def _configure_logging(verbosity: int) -> None:
    level = logging.WARNING if verbosity <= 0 else (logging.INFO if verbosity == 1 else logging.DEBUG)
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s", stream=sys.stderr)
    logging.getLogger().setLevel(level)


# --------------------------------------------------------------------------- #
# small input helpers
# --------------------------------------------------------------------------- #

def read_script(value: str | None) -> str | None:
    """Resolve a ``--script`` argument: ``None``, ``-`` (stdin) or a file path.

    A script that cannot be read is bad *input*, not a failed render, so every
    way of getting it wrong raises :class:`UsageError` -- exit 2, the same as an
    unknown caption preset or a missing ``--topic`` -- and the message names the
    path and says whether it was missing, unreadable or empty.
    """
    if value is None:
        return None
    if value == "-":
        text = sys.stdin.read()
        if not text.strip():
            raise UsageError("--script -: the script read from stdin is empty")
        return text

    path = Path(value).expanduser()
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise UsageError(f"--script {path}: the script file is missing") from exc
    except OSError as exc:
        raise UsageError(f"--script {path}: cannot read the script file ({exc.strerror or exc})") from exc
    if not text.strip():
        raise UsageError(f"--script {path}: the script file is empty")
    return text


def _out_path(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def _mute_stdout() -> None:
    """Point stdout at ``/dev/null`` so interpreter shutdown cannot re-raise EPIPE."""
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, sys.stdout.fileno())
    except (OSError, ValueError, AttributeError):  # pragma: no cover - captured stdout
        pass


def _one_line(exc: BaseException) -> str:
    """Collapse an exception message to a single, still-complete line."""
    message = " ".join(str(exc).split())
    return message or exc.__class__.__name__


def _emit(results: Any) -> int:
    """Print every produced path, one per line.  Returns the process exit code."""
    items = list(results) if isinstance(results, (list, tuple)) else [results]
    if not items:
        print(f"{PROG}: the pipeline produced nothing", file=sys.stderr)
        return 1
    for item in items:
        log.info("%s", item.summary())
        print(item.output)
    return 0


def _dry_run(command: str, primary: dict[str, Any], kwargs: dict[str, Any]) -> int:
    """Describe the call that ``--dry-run`` is holding back, and succeed."""
    print(f"dry-run: {command}")
    for key, value in list(primary.items()) + sorted(kwargs.items()):
        if value is None or value == "":
            continue
        print(f"  {key}: {value}")
    settings = get_settings()
    print(f"  canvas: {settings.width}x{settings.height}@{settings.fps}")
    print(f"  output-dir: {settings.output_dir}")
    return 0


# --------------------------------------------------------------------------- #
# up-front validation
# --------------------------------------------------------------------------- #
#
# Every enumerated option is checked before the command runs, so a typo costs a
# line of stderr instead of minutes of rendering.  The message names the bad
# value and lists what was allowed, and the exit code is 2 -- it is bad usage.

def _bounded_int(flag: str, minimum: int) -> Callable[[str], int]:
    """An argparse ``type`` accepting whole numbers ``>= minimum``."""

    def parse(text: str) -> int:
        try:
            value = int(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} wants a whole number, not {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(f"{flag} must be at least {minimum}, not {value}")
        return value

    return parse


def _bounded_float(flag: str, minimum: float) -> Callable[[str], float]:
    """An argparse ``type`` accepting numbers ``>= minimum`` seconds."""

    def parse(text: str) -> float:
        try:
            value = float(text)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{flag} wants a number of seconds, not {text!r}") from None
        if value < minimum:
            raise argparse.ArgumentTypeError(
                f"{flag} must be at least {minimum:g} second{'' if minimum == 1 else 's'}, not {value:g}"
            )
        return value

    return parse


def _style_key(value: str) -> str:
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def _theme_key(value: str) -> str:
    return value.strip().lower()


def _check_choice(what: str, value: str | None, known: Sequence[str],
                  normalise: Callable[[str], str]) -> None:
    """Raise :class:`UsageError` unless ``value`` names one of ``known``."""
    raw = (value or "").strip()
    if not raw:
        return
    if normalise(raw) in known:
        return
    raise UsageError(f"unknown {what} {raw!r}. Available: {', '.join(known)}")


def _check_style(value: str | None) -> None:
    from . import captions

    _check_choice("caption style", value, sorted(captions.PRESETS), _style_key)


def _check_theme(kind: str, value: str | None) -> None:
    from . import overlays

    themes = overlays.THEMES[kind]
    _check_choice(f"{kind} theme", value, sorted(themes), _theme_key)


def _check_voice(flag: str, value: str | None) -> None:
    """A voice name that will not resolve is a usage error, not a silent default."""
    raw = (value or "").strip()
    if not raw:
        return
    from .errors import TTSError
    from .tts import voices as catalogue

    try:
        catalogue.find_voice(raw)
    except TTSError as exc:
        raise UsageError(f"{flag}: {_one_line(exc)} (run: {PROG} voices)") from None


def _check_backend(value: str | None) -> None:
    from . import overlays

    _check_choice("overlay backend", value, sorted(overlays.BACKENDS), _theme_key)


def validate(ns: argparse.Namespace) -> None:
    """Reject every bad enumerated or degenerate value, before any work starts."""
    command = getattr(ns, "command", "")
    if command not in ("clip", "story", "texts", "reddit", "split"):
        return

    _check_style(getattr(ns, "style", ""))
    _check_backend(getattr(ns, "backend", None))
    _check_voice("--voice", getattr(ns, "voice", ""))
    _check_voice("--reply-voice", getattr(ns, "reply_voice", ""))
    if command == "texts":
        _check_theme("chat", ns.theme)
    if command == "reddit":
        _check_theme("forum", ns.theme)
    if command == "clip" and ns.min_duration > ns.max_duration:
        raise UsageError(
            f"--min {ns.min_duration:g} is longer than --max {ns.max_duration:g}; "
            "--min must not exceed --max"
        )


# --------------------------------------------------------------------------- #
# pipeline commands
# --------------------------------------------------------------------------- #

def _cmd_clip(ns: argparse.Namespace) -> int:
    from .pipelines import clip

    kwargs: dict[str, Any] = {
        "count": ns.count,
        "min_duration": ns.min_duration,
        "max_duration": ns.max_duration,
        "style": ns.style,
        "reframe": not ns.no_reframe,
        "captions": not ns.no_captions,
        "out_dir": _out_path(ns.out),
    }
    if ns.dry_run:
        return _dry_run("clip", {"source": ns.source}, kwargs)
    return _emit(clip.run(ns.source, **kwargs))


def _cmd_story(ns: argparse.Namespace) -> int:
    from .pipelines import story

    script = read_script(ns.script)
    if not ns.topic and script is None:
        raise UsageError("story needs --topic TEXT or --script FILE|-")
    kwargs: dict[str, Any] = {
        "topic": ns.topic,
        "script": script,
        "seconds": ns.seconds,
        "voice": ns.voice,
        "background": ns.background,
        "music": ns.music,
        "style": ns.style,
        "captions": not ns.no_captions,
        "out_path": _out_path(ns.out),
    }
    if ns.dry_run:
        return _dry_run("story", {}, {**kwargs, "script": _script_note(script)})
    return _emit(story.run(**kwargs))


def _cmd_texts(ns: argparse.Namespace) -> int:
    from .pipelines import texts

    script = read_script(ns.script)
    if not ns.topic and script is None:
        raise UsageError("texts needs --topic TEXT or --script FILE|-")
    kwargs: dict[str, Any] = {
        "topic": ns.topic,
        "script": script,
        "theme": ns.theme,
        "voice": ns.voice,
        "reply_voice": ns.reply_voice,
        "background": ns.background,
        "music": ns.music,
        "backend": ns.backend,
        "turns": ns.turns,
        "captions": ns.captions,
        "style": ns.style,
        "out_path": _out_path(ns.out),
    }
    if ns.dry_run:
        return _dry_run("texts", {}, {**kwargs, "script": _script_note(script)})
    return _emit(texts.run(**kwargs))


def _cmd_reddit(ns: argparse.Namespace) -> int:
    from .pipelines import reddit

    if not ns.topic:
        raise UsageError("reddit needs --topic TEXT")
    kwargs: dict[str, Any] = {
        "topic": ns.topic,
        "theme": ns.theme,
        "voice": ns.voice,
        "background": ns.background,
        "music": ns.music,
        "style": ns.style,
        "card_seconds": ns.card_seconds,
        "captions": not ns.no_captions,
        "words": ns.words,
        "backend": ns.backend,
        "out_path": _out_path(ns.out),
    }
    if ns.dry_run:
        return _dry_run("reddit", {}, kwargs)
    return _emit(reddit.run(**kwargs))


def _cmd_split(ns: argparse.Namespace) -> int:
    from .pipelines import split

    kwargs: dict[str, Any] = {
        "bottom": ns.bottom,
        "narration": ns.narration,
        "voice": ns.voice,
        "style": ns.style,
        "music": ns.music,
        "seconds": ns.seconds,
        "captions": not ns.no_captions,
        "reframe": not ns.no_reframe,
        "out_path": _out_path(ns.out),
    }
    if ns.dry_run:
        return _dry_run("split", {"top": ns.top}, kwargs)
    return _emit(split.run(ns.top, **kwargs))


def _script_note(script: str | None) -> str:
    if script is None:
        return ""
    words = len(script.split())
    return f"<{words} word{'' if words == 1 else 's'} supplied>"


# --------------------------------------------------------------------------- #
# catalogue commands
# --------------------------------------------------------------------------- #

def _cmd_voices(ns: argparse.Namespace) -> int:
    from .tts import voices as catalogue

    provider = (ns.provider or "").strip()
    if provider:
        canonical = catalogue.canonical_provider(provider)
        known = ("auto", *catalogue.KNOWN_PROVIDERS)
        if canonical not in known:
            raise UsageError(f"unknown tts provider {provider!r}; choose one of: {', '.join(known)}")
        provider = canonical

    entries = catalogue.list_voices(provider or None)
    tags = [t.strip().lower() for t in (ns.tag or []) if t.strip()]
    if tags:
        entries = [v for v in entries if all(v.has_tag(t) for t in tags)]
    if not entries:
        wanted = " ".join(filter(None, [provider, *tags])) or "that filter"
        print(f"{PROG}: no voice matches {wanted}", file=sys.stderr)
        print(f"known tags: {', '.join(catalogue.tag_values())}", file=sys.stderr)
        return 1

    width = max(len(v.name) for v in entries)
    for voice in entries:
        backends = ",".join(sorted(voice.providers))
        print(f"{voice.name:<{width}}  {backends:<22}  {', '.join(voice.tags)}")
        if ns.verbose and voice.description:
            print(f"{'':<{width}}  {voice.description}")
    return 0


def _cmd_styles(ns: argparse.Namespace) -> int:
    from . import captions

    presets = captions.list_styles()
    width = max(len(s.name) for s in presets)
    for style in presets:
        print(f"{style.name:<{width}}  {style.description}")
        if ns.verbose:
            print(
                f"{'':<{width}}  {style.animation} / {style.position} / "
                f"{style.max_words} words / {style.font} {style.font_size}px"
            )
    return 0


def _cmd_assets(ns: argparse.Namespace) -> int:
    from . import assets as library

    settings = get_settings()
    if ns.generate and ns.dry_run:
        print("dry-run: assets --generate")
        print(f"  assets-dir: {settings.assets_dir}")
        print(f"  existing: {len(library.library(settings))}")
        return 0
    if ns.generate:
        library.ensure_placeholders(settings)

    catalogue = library.library(settings)
    if not catalogue:
        print(f"{PROG}: the asset library is empty ({settings.assets_dir})", file=sys.stderr)
        print(f"generate a starter set with: {PROG} assets --generate", file=sys.stderr)
        return 0

    width = max(len(a.name) for a in catalogue)
    for asset in sorted(catalogue, key=lambda a: (a.kind, a.name)):
        tags = ", ".join(asset.tags)
        print(f"{asset.kind:<10}  {asset.name:<{width}}  {asset.duration:6.1f}s  {tags:<28}  {asset.path}")
    kinds = ", ".join(
        f"{sum(1 for a in catalogue if a.kind == kind)} {kind}" for kind in library.KINDS
    )
    print(f"{len(catalogue)} assets ({kinds}) in {settings.assets_dir}")
    return 0


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #

def _module_present(name: str) -> bool:
    """True when ``name`` is importable, without importing it."""
    from importlib.util import find_spec

    try:
        return find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):  # pragma: no cover - broken installs
        return False


def _first_line(text: str, fallback: str) -> str:
    lines = (text or "").strip().splitlines()
    return lines[0].strip() if lines else fallback


def _chromium_report(settings: Settings) -> tuple[bool, str]:
    """Actually launch Chromium (never downloads one) and report what happened."""
    from . import overlays

    if not _module_present("playwright"):
        return False, "playwright is not installed (pip install 'aiclipper[overlays]')"
    if "chromium" not in overlays.available_backends(settings):
        return False, "no chromium binary found; set AICLIP_CHROMIUM (never run playwright install)"
    from playwright.sync_api import sync_playwright

    launch = getattr(overlays, "_launch_chromium", None)
    with sync_playwright() as pw:
        browser = launch(pw, settings) if callable(launch) else pw.chromium.launch()
        try:
            return True, f"chromium {browser.version}"
        finally:
            browser.close()


def _tts_blocker(provider: Any, settings: Settings) -> str:
    """Why ``provider`` cannot be routed to, when ``available()`` says no.

    ``available()`` folds three different answers into one ``False`` -- the
    package is missing, the credential is missing, or ``settings.offline``
    forbids the network -- and reporting all three as "not installed"
    contradicts the extras row printed a few lines above it (``OK edge-tts
    importable``).  A diagnostic that disagrees with itself is the same kind of
    lie the capability probe was added to stop telling, so each backend names
    its own blocker.  Anything that is not one of our two network backends (a
    third-party provider, a test double) keeps the plain wording.
    """
    from .tts.edge import EdgeTTS, edge_available
    from .tts.eleven import ElevenLabsTTS

    offline = "installed, but offline mode is set (unset AICLIP_OFFLINE)"
    if isinstance(provider, EdgeTTS):
        if not edge_available():
            return "not installed (pip install 'aiclipper[tts]')"
        return offline if settings.offline else "installed, but ffmpeg is missing"
    if isinstance(provider, ElevenLabsTTS):
        if not settings.elevenlabs_api_key:
            return "no ELEVENLABS_API_KEY set"
        return offline if settings.offline else "key set, but ffmpeg is missing"
    return "not installed"


def _tts_report(name: str, settings: Settings) -> tuple[bool, str]:
    """Installed is not usable: ask the backend to prove it can really speak.

    ``available()`` only says the import worked or that a key is a non-empty
    string, which is how a machine with no route to the voice service still
    reported ``OK tts:edge``.  :func:`aiclipper.tts.provider_usable` runs the
    backend's own bounded, cached probe instead; it never raises.  When even
    ``available()`` says no, :func:`_tts_blocker` names which of its several
    conditions actually failed.
    """
    from . import tts

    provider = tts.get_provider(name, settings=settings)
    try:
        installed = bool(provider.available())
    except Exception:  # noqa: BLE001 - doctor reports, it does not crash
        log.debug("tts backend %r failed its availability check", name, exc_info=True)
        installed = False
    usable = tts.provider_usable(provider, settings=settings, timeout=DOCTOR_TTS_TIMEOUT)
    if usable:
        return True, f"provider {provider.name}: installed and usable"
    if installed:
        return False, f"provider {provider.name}: installed, but it cannot synthesise here"
    return False, f"provider {provider.name}: {_tts_blocker(provider, settings)}"


def _library_report(settings: Settings) -> tuple[bool, str]:
    from . import assets

    catalogue = assets.library(settings)
    kinds = ", ".join(
        f"{sum(1 for a in catalogue if a.kind == kind)} {kind}" for kind in assets.KINDS
    )
    if not catalogue:
        return False, f"empty ({settings.assets_dir}) -- run: {PROG} assets --generate"
    return True, f"{len(catalogue)} assets ({kinds}) in {settings.assets_dir}"


def _claude_report(settings: Settings) -> tuple[bool, str]:
    from .llm import claude

    if not claude.credentials_available():
        return False, "no ANTHROPIC_API_KEY / auth token / ant profile found"
    if settings.offline:
        return True, "credentials resolve (offline mode: the heuristic provider is used)"
    return True, "credentials resolve"


def _check(label: str, probe: Callable[[], tuple[bool, str]]) -> tuple[str, str, str]:
    """Run one probe.  A broken component is a MISSING row, never a traceback."""
    try:
        ok, detail = probe()
    except Exception as exc:  # noqa: BLE001 - doctor must survive anything
        log.debug("doctor probe %s failed", label, exc_info=True)
        return ("MISSING", label, _one_line(exc))
    return ("OK" if ok else "MISSING", label, detail)


def _cmd_doctor(ns: argparse.Namespace) -> int:
    settings = get_settings()
    print(f"{PROG} {__version__} doctor")
    print(f"  canvas     {settings.width}x{settings.height} @{settings.fps}fps, seed {settings.seed}")
    print(f"  offline    {'yes' if settings.offline else 'no'}")
    print(f"  work       {settings.work_dir}")
    print(f"  output     {settings.output_dir}")
    print(f"  assets     {settings.assets_dir}")
    print("")

    rows: list[tuple[str, str, str]] = []
    rows.append(_check("ffmpeg", lambda: _ffmpeg_probe(settings)))
    rows.append(_check("ffprobe", lambda: _ffprobe_probe(settings)))
    for label, module, extra in OPTIONAL_EXTRAS:
        rows.append(
            _check(
                label,
                lambda module=module, extra=extra: (
                    _module_present(module),
                    "importable" if _module_present(module) else f"pip install 'aiclipper[{extra}]'",
                ),
            )
        )
    rows.append(_check("claude key", lambda: _claude_report(settings)))
    rows.append(_check("llm provider", lambda: _llm_report(settings)))
    for name in _TTS_BACKENDS:
        rows.append(_check(f"tts:{name}", lambda name=name: _tts_report(name, settings)))
    rows.append(_check("transcribe", lambda: _transcribe_report(settings)))
    rows.append(_check("chromium", lambda: _chromium_report(settings)))
    rows.append(_check("assets", lambda: _library_report(settings)))

    width = max(len(label) for _, label, _ in rows)
    for status, label, detail in rows:
        print(f"{status:<7} {label:<{width}}  {detail}")

    core = dict((label, status) for status, label, _ in rows)
    healthy = core.get("ffmpeg") == "OK" and core.get("ffprobe") == "OK"
    print("")
    print("core ok: ffmpeg and ffprobe are ready" if healthy else "core missing: install ffmpeg")
    return 0 if healthy else 1


def _ffmpeg_probe(settings: Settings) -> tuple[bool, str]:
    """``ffmpeg -version``, always through :mod:`aiclipper.ffmpeg`."""
    from . import ffmpeg as ff

    if not shutil.which(settings.ffmpeg):
        return False, f"{settings.ffmpeg!r} not found on PATH (apt install ffmpeg / brew install ffmpeg)"
    proc = ff.run_ffmpeg(["-version"], settings=settings, quiet=False, timeout=20)
    return True, _first_line(proc.stdout, settings.ffmpeg)


def _ffprobe_probe(settings: Settings) -> tuple[bool, str]:
    from . import ffmpeg as ff

    if not shutil.which(settings.ffprobe):
        return False, f"{settings.ffprobe!r} not found on PATH (it ships with ffmpeg)"
    return True, _first_line(ff.run_ffprobe(["-version"], settings=settings, timeout=20), settings.ffprobe)


def _llm_report(settings: Settings) -> tuple[bool, str]:
    """Installed is not usable, the same rule the ``tts:`` rows follow.

    ``available()`` only says a credential is a non-empty string or that a base
    URL is configured -- which is how a machine with nothing listening on that
    URL would still report ``OK``.  A backend that offers the stronger
    ``usable()`` probe (the local one does) is asked that instead, under a bound,
    so a configured-but-dead endpoint is caught here rather than halfway through
    a render.
    """
    from . import llm
    from .tts.base import run_bounded

    provider = llm.get_provider(settings=settings)
    resolves = f"{settings.llm_provider!r} resolves to {provider.name}"
    try:
        installed = bool(provider.available())
    except Exception:  # noqa: BLE001 - doctor reports, it does not crash
        log.debug("llm backend %r failed its availability check", provider.name, exc_info=True)
        installed = False

    probe = getattr(provider, "usable", None)
    if not callable(probe):
        return installed, resolves
    if not installed:
        return False, f"{resolves}, which is not configured here"
    def _probe() -> bool:
        try:
            return bool(probe())
        except Exception:  # noqa: BLE001 - an unreachable endpoint is a row, not a crash
            log.debug("llm backend %r failed its usability probe", provider.name, exc_info=True)
            return False

    if run_bounded(_probe, DOCTOR_TTS_TIMEOUT, default=False):
        return True, f"{resolves}, serving {getattr(provider, 'model', '?')!r}"
    return False, f"{resolves}, but nothing is answering at {getattr(provider, 'chat_url', '?')}"


def _transcribe_report(settings: Settings) -> tuple[bool, str]:
    """Installed *and* able to run: the weights have to be on this disk too.

    ``faster-whisper`` importing proves nothing on its own -- the first real
    transcription downloads a model, which is exactly what an offline box cannot
    do.  So this row follows the same installed-vs-usable rule the ``tts:`` rows
    follow, and says which of the two is missing.
    """
    from . import transcribe

    if not transcribe.available():
        return False, "pip install 'aiclipper[transcribe]' (no ASR: clip falls back to even windows)"
    name = settings.whisper_model
    cached = _whisper_weights_cached(name)
    if cached is False:
        return False, (
            f"faster-whisper installed, but model {name!r} is not cached here; the first run "
            "downloads it (without ASR, clip falls back to even windows)"
        )
    if cached is None:
        return True, f"faster-whisper is importable (model {name!r} not verified)"
    return True, f"faster-whisper, model {name!r} cached locally"


def _whisper_weights_cached(model: str) -> bool | None:
    """Are ``model``'s weights already on disk?  ``None`` when we cannot tell.

    Never touches the network: a local directory is checked directly, and the
    Hugging Face cache is queried with ``local_files_only=True``.
    """
    if not model:
        return None
    if Path(model).expanduser().is_dir():
        return True
    try:
        from faster_whisper.utils import download_model
    except Exception:  # noqa: BLE001 - an internal helper we are allowed to lose
        return None
    try:
        download_model(model, local_files_only=True)
    except Exception:  # noqa: BLE001 - any failure here means "not cached"
        return False
    return True


# --------------------------------------------------------------------------- #
# parser
# --------------------------------------------------------------------------- #

def _global_parser() -> argparse.ArgumentParser:
    """The parent parser holding every global flag.

    Every option defaults to :data:`argparse.SUPPRESS` so a subparser never
    clobbers a value that was given *before* the subcommand name.
    """
    parent = argparse.ArgumentParser(add_help=False)
    group = parent.add_argument_group("global options")
    group.add_argument("--work-dir", metavar="DIR", default=argparse.SUPPRESS,
                       help="scratch directory (AICLIP_WORK_DIR)")
    group.add_argument("--output-dir", metavar="DIR", default=argparse.SUPPRESS,
                       help="where finished videos land (AICLIP_OUTPUT_DIR)")
    group.add_argument("--offline", action="store_true", default=argparse.SUPPRESS,
                       help="never touch the network (AICLIP_OFFLINE)")
    group.add_argument("--seed", type=int, metavar="N", default=argparse.SUPPRESS,
                       help="seed every random choice (AICLIP_SEED)")
    group.add_argument("--width", type=int, metavar="PX", default=argparse.SUPPRESS,
                       help="canvas width (AICLIP_WIDTH)")
    group.add_argument("--height", type=int, metavar="PX", default=argparse.SUPPRESS,
                       help="canvas height (AICLIP_HEIGHT)")
    group.add_argument("--fps", type=int, metavar="N", default=argparse.SUPPRESS,
                       help="canvas frame rate (AICLIP_FPS)")
    group.add_argument("-v", "--verbose", action="count", default=argparse.SUPPRESS,
                       help="-v for INFO logging, -vv for DEBUG")
    group.add_argument("--dry-run", action="store_true", default=argparse.SUPPRESS,
                       help="print what would run, produce nothing")
    return parent


def build_parser() -> argparse.ArgumentParser:
    """The full ``aiclip`` parser: global flags plus the nine subcommands."""
    parent = _global_parser()
    parser = argparse.ArgumentParser(
        prog=PROG,
        parents=[parent],
        description="Short-form video engine: auto-clipping, narrated stories, chat and forum shorts.",
    )
    parser.add_argument("--version", action="version", version=f"{PROG} {__version__}")
    subparsers = parser.add_subparsers(dest="command", metavar="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        return subparsers.add_parser(name, parents=[parent], help=help_text, description=help_text)

    # -- clip -------------------------------------------------------------- #
    clip = add("clip", "cut a long video or URL into vertical shorts")
    clip.add_argument("source", help="local file or video URL")
    clip.add_argument("--count", type=_bounded_int("--count", MIN_COUNT), default=3, metavar="N",
                      help=f"how many shorts to cut (at least {MIN_COUNT})")
    clip.add_argument("--min", dest="min_duration", type=_bounded_float("--min", MIN_SECONDS),
                      default=15.0, metavar="S",
                      help=f"shortest acceptable clip, in seconds (at least {MIN_SECONDS:g}, "
                           "and never above --max)")
    clip.add_argument("--max", dest="max_duration", type=_bounded_float("--max", MIN_SECONDS),
                      default=60.0, metavar="S",
                      help=f"longest acceptable clip, in seconds (at least {MIN_SECONDS:g})")
    clip.add_argument("--style", default="clean", metavar="NAME", help="caption preset")
    clip.add_argument("--no-reframe", action="store_true", help="keep the source framing")
    clip.add_argument("--no-captions", action="store_true", help="do not burn captions")
    clip.add_argument("--out", metavar="DIR", help="directory for the shorts")
    clip.set_defaults(func=_cmd_clip)

    # -- story ------------------------------------------------------------- #
    story = add("story", "turn a topic or a script into a narrated short")
    source = story.add_mutually_exclusive_group()
    source.add_argument("--topic", metavar="TEXT", help="what the short is about")
    source.add_argument("--script", metavar="FILE", help="script file, or - for stdin")
    story.add_argument("--seconds", type=_bounded_int("--seconds", int(MIN_SECONDS)), default=35,
                       metavar="N", help=f"target length in seconds (at least {int(MIN_SECONDS)})")
    story.add_argument("--voice", default="", metavar="NAME", help="catalogue voice name or tag query")
    story.add_argument("--background", metavar="NAME", help="library background name or path")
    story.add_argument("--music", metavar="NAME", help="library music name or path")
    story.add_argument("--style", default="bold_yellow", metavar="NAME", help="caption preset")
    story.add_argument("--no-captions", action="store_true", help="do not burn captions")
    story.add_argument("--out", metavar="FILE", help="output file")
    story.set_defaults(func=_cmd_story)

    # -- texts ------------------------------------------------------------- #
    texts = add("texts", "build an animated text-message conversation")
    source = texts.add_mutually_exclusive_group()
    source.add_argument("--topic", metavar="TEXT", help="what the conversation is about")
    source.add_argument("--script", metavar="FILE", help="conversation file, or - for stdin")
    texts.add_argument("--theme", default="classic", metavar="NAME", help="chat theme")
    texts.add_argument("--voice", default="", metavar="NAME", help="voice for the outgoing (\"me\") side")
    texts.add_argument("--reply-voice", default="", metavar="NAME", help="voice for the incoming side")
    texts.add_argument("--background", metavar="NAME", help="library background name or path")
    texts.add_argument("--music", metavar="NAME", help="library music name or path")
    texts.add_argument("--backend", choices=["chromium", "pillow"], help="overlay renderer")
    texts.add_argument("--turns", type=_bounded_int("--turns", MIN_TURNS), default=10, metavar="N",
                       help=f"messages to generate (at least {MIN_TURNS})")
    texts.add_argument("--captions", action="store_true", help="also burn captions (off by default)")
    texts.add_argument("--style", default="clean", metavar="NAME", help="caption preset for --captions")
    texts.add_argument("--out", metavar="FILE", help="output file")
    texts.set_defaults(func=_cmd_texts)

    # -- reddit ------------------------------------------------------------ #
    reddit = add("reddit", "narrate a forum-style story behind a title card")
    reddit.add_argument("--topic", metavar="TEXT", help="what the post is about")
    reddit.add_argument("--theme", default="dark", metavar="NAME", help="card theme")
    reddit.add_argument("--voice", default="", metavar="NAME", help="narration voice")
    reddit.add_argument("--background", metavar="NAME", help="library background name or path")
    reddit.add_argument("--music", metavar="NAME", help="library music name or path")
    reddit.add_argument("--style", default="clean", metavar="NAME", help="caption preset")
    reddit.add_argument("--card-seconds", type=_bounded_float("--card-seconds", MIN_SECONDS),
                        metavar="S",
                        help=f"how long the card stays up (at least {MIN_SECONDS:g} second)")
    reddit.add_argument("--words", type=_bounded_int("--words", MIN_WORDS), default=180, metavar="N",
                        help=f"length of the story in words (at least {MIN_WORDS})")
    reddit.add_argument("--backend", choices=["chromium", "pillow"], help="overlay renderer")
    reddit.add_argument("--no-captions", action="store_true", help="do not burn captions")
    reddit.add_argument("--out", metavar="FILE", help="output file")
    reddit.set_defaults(func=_cmd_reddit)

    # -- split ------------------------------------------------------------- #
    split = add("split", "stack two panes: your clip over a second one")
    split.add_argument("top", help="the clip that fills the top half")
    split.add_argument("--bottom", metavar="PATH", help="second clip, or a library background name")
    split.add_argument("--narration", metavar="TEXT", help="spoken line that replaces the top audio")
    split.add_argument("--voice", default="", metavar="NAME", help="narration voice")
    split.add_argument("--style", default="clean", metavar="NAME", help="caption preset")
    split.add_argument("--music", metavar="NAME", help="library music name or path")
    split.add_argument("--seconds", type=_bounded_float("--seconds", MIN_SECONDS), metavar="S",
                       help=f"trim the result to this length (at least {MIN_SECONDS:g} second)")
    split.add_argument("--no-captions", action="store_true", help="do not burn captions")
    split.add_argument("--no-reframe", action="store_true", help="keep the source framing")
    split.add_argument("--out", metavar="FILE", help="output file")
    split.set_defaults(func=_cmd_split)

    # -- catalogues -------------------------------------------------------- #
    voices = add("voices", "list the voice catalogue")
    voices.add_argument("--provider", metavar="NAME", help="only voices this backend can speak")
    voices.add_argument("--tag", action="append", metavar="TAG", help="repeatable; all tags must match")
    voices.set_defaults(func=_cmd_voices)

    styles = add("styles", "list the caption presets")
    styles.set_defaults(func=_cmd_styles)

    assets = add("assets", "list the background and music library")
    assets.add_argument("--generate", action="store_true", help="create the starter library if needed")
    assets.set_defaults(func=_cmd_assets)

    doctor = add("doctor", "report what is installed and reachable")
    doctor.set_defaults(func=_cmd_doctor)

    return parser


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #

def main(argv: Sequence[str] | None = None) -> int:
    """Run one ``aiclip`` invocation.  0 = success, 1 = failure, 2 = bad usage."""
    parser = build_parser()
    try:
        ns = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:  # argparse owns --help (0) and usage errors (2)
        return int(exc.code or 0)

    verbosity = int(getattr(ns, "verbose", 0) or 0)
    ns.verbose = verbosity
    ns.dry_run = bool(getattr(ns, "dry_run", False))
    _configure_logging(verbosity)

    try:
        with _applied(environment_for(ns)):
            validate(ns)
            log.debug("running %s with %s", ns.command, vars(ns))
            return int(ns.func(ns))
    except UsageError as exc:
        print(f"{PROG} {ns.command}: error: {exc}", file=sys.stderr)
        return 2
    except SystemExit as exc:  # a handler deferred to argparse
        return int(exc.code or 0)
    except BrokenPipeError:  # a downstream `head`/`less` closed the pipe: stay quiet
        _mute_stdout()
        return 0
    except KeyboardInterrupt:
        print(f"{PROG}: interrupted", file=sys.stderr)
        return 1
    except Exception as exc:  # noqa: BLE001 - the CLI reports, it does not crash
        if verbosity:
            traceback.print_exc()
        print(f"{PROG}: {_one_line(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover - module entry point
    raise SystemExit(main())
