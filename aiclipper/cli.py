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
    "UsageError",
    "build_parser",
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

_TTS_BACKENDS = ("edge", "elevenlabs", "offline")

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
    """Resolve a ``--script`` argument: ``None``, ``-`` (stdin) or a file path."""
    if value is None:
        return None
    if value == "-":
        text = sys.stdin.read()
        where = "stdin"
    else:
        path = Path(value).expanduser()
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"cannot read script {path}: {exc.strerror or exc}") from exc
        where = str(path)
    if not text.strip():
        raise RuntimeError(f"the script read from {where} is empty")
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


def _tts_report(name: str, settings: Settings) -> tuple[bool, str]:
    from . import tts

    provider = tts.get_provider(name, settings=settings)
    return bool(provider.available()), f"provider {provider.name}"


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
    rows.append(_check("transcribe", lambda: _transcribe_report()))
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
    from . import llm

    provider = llm.get_provider(settings=settings)
    return bool(provider.available()), f"{settings.llm_provider!r} resolves to {provider.name}"


def _transcribe_report() -> tuple[bool, str]:
    from . import transcribe

    if not transcribe.available():
        return False, "pip install 'aiclipper[transcribe]' (no ASR: clip falls back to even windows)"
    return True, "faster-whisper is importable"


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
    clip.add_argument("--count", type=int, default=3, metavar="N", help="how many shorts to cut")
    clip.add_argument("--min", dest="min_duration", type=float, default=15.0, metavar="S",
                      help="shortest acceptable clip")
    clip.add_argument("--max", dest="max_duration", type=float, default=60.0, metavar="S",
                      help="longest acceptable clip")
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
    story.add_argument("--seconds", type=int, default=35, metavar="N", help="target length")
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
    texts.add_argument("--turns", type=int, default=10, metavar="N", help="messages to generate")
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
    reddit.add_argument("--card-seconds", type=float, metavar="S", help="how long the card stays up")
    reddit.add_argument("--words", type=int, default=180, metavar="N", help="length of the story")
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
    split.add_argument("--seconds", type=float, metavar="S", help="trim the result to this length")
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
