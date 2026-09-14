"""Tests for :mod:`aiclipper.cli`.

The CLI's real job is translation: user words in, exact pipeline keyword
arguments out.  Most tests therefore monkeypatch the pipeline ``run`` functions
and assert on what the CLI handed them (including that the CLI's defaults still
match the pipelines' own defaults).  One test runs the story pipeline for real at
a 180x320 canvas and probes the file whose path the CLI printed.
"""

from __future__ import annotations

import inspect
import io
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from aiclipper import ffmpeg as ff
from aiclipper.cli import build_parser, environment_for, main
from aiclipper.config import get_settings
from aiclipper.errors import AssetError, RenderError
from aiclipper.models import ProjectResult

# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class Recorder:
    """Stand-in for a pipeline ``run``: records the call, returns a result."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        self.settings: list[Any] = []
        self.defaults: dict[str, Any] = {}

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        self.settings.append(get_settings())
        if isinstance(self.result, Exception):
            raise self.result
        return self.result

    @property
    def kwargs(self) -> dict[str, Any]:
        assert self.calls, "the pipeline was never called"
        return self.calls[-1][1]

    @property
    def args(self) -> tuple[Any, ...]:
        assert self.calls, "the pipeline was never called"
        return self.calls[-1][0]


def result_for(tmp_path: Path, name: str = "video.mp4", kind: str = "story") -> ProjectResult:
    out = tmp_path / name
    out.write_bytes(b"\x00" * 16)
    return ProjectResult(output=out, kind=kind, title=name, duration=8.0)


def pipeline_defaults(name: str) -> dict[str, Any]:
    """The real ``run`` signature's keyword defaults, straight from the source."""
    module = __import__(f"aiclipper.pipelines.{name}", fromlist=["run"])
    return {
        key: param.default
        for key, param in inspect.signature(module.run).parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def patch_pipeline(monkeypatch: pytest.MonkeyPatch, name: str, result: Any) -> Recorder:
    recorder = Recorder(result)
    recorder.defaults = pipeline_defaults(name)  # captured before the real run is replaced
    monkeypatch.setattr(f"aiclipper.pipelines.{name}.run", recorder)
    return recorder


def assert_defaults_match(recorder: Recorder, name: str, skip: set[str]) -> None:
    defaults = recorder.defaults
    for key, value in recorder.kwargs.items():
        if key in skip:
            continue
        assert key in defaults, f"{name}.run has no parameter {key!r}"
        assert value == defaults[key], f"cli default for {key!r} drifted from {name}.run"


@pytest.fixture
def tiny_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A two-item asset library small enough to render in about a second."""
    root = tmp_path / "assets"
    (root / "backgrounds").mkdir(parents=True)
    (root / "music").mkdir(parents=True)
    ff.run_ffmpeg([
        "-y", "-f", "lavfi", "-i", "testsrc2=size=180x320:rate=12:duration=8",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        str(root / "backgrounds" / "slow_drift.mp4"),
    ])
    ff.make_tone(8.0, root / "music" / "quiet_bed.wav", frequency=180.0)
    monkeypatch.setenv("AICLIP_ASSETS_DIR", str(root))
    return root


# --------------------------------------------------------------------------- #
# argument translation: one test per pipeline
# --------------------------------------------------------------------------- #


def test_clip_passes_every_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    first = result_for(tmp_path, "short-01.mp4", kind="clip")
    second = result_for(tmp_path, "short-02.mp4", kind="clip")
    run = patch_pipeline(monkeypatch, "clip", [first, second])

    code = main([
        "clip", "lecture.mp4", "--count", "2", "--min", "8", "--max", "24",
        "--style", "neon", "--no-reframe", "--no-captions", "--out", str(tmp_path / "shorts"),
    ])

    assert code == 0
    assert run.args == ("lecture.mp4",)
    assert run.kwargs == {
        "count": 2,
        "min_duration": 8.0,
        "max_duration": 24.0,
        "style": "neon",
        "reframe": False,
        "captions": False,
        "out_dir": tmp_path / "shorts",
    }
    out = capsys.readouterr().out.splitlines()
    assert out == [str(first.output), str(second.output)]


def test_clip_defaults_match_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "clip", [result_for(tmp_path, kind="clip")])
    assert main(["clip", "lecture.mp4"]) == 0
    assert run.args == ("lecture.mp4",)
    assert_defaults_match(run, "clip", skip=set())


def test_story_passes_every_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    result = result_for(tmp_path)
    run = patch_pipeline(monkeypatch, "story", result)

    code = main([
        "story", "--topic", "deep sea vents", "--seconds", "18", "--voice", "narrator_deep",
        "--background", "ember_mist", "--music", "quiet_bed", "--style", "neon",
        "--no-captions", "--out", str(tmp_path / "vents.mp4"),
    ])

    assert code == 0
    assert run.args == ()
    assert run.kwargs == {
        "topic": "deep sea vents",
        "script": None,
        "seconds": 18,
        "voice": "narrator_deep",
        "background": "ember_mist",
        "music": "quiet_bed",
        "style": "neon",
        "captions": False,
        "out_path": tmp_path / "vents.mp4",
    }
    assert capsys.readouterr().out.strip() == str(result.output)


def test_story_defaults_match_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "story", result_for(tmp_path))
    assert main(["story", "--topic", "sourdough"]) == 0
    assert_defaults_match(run, "story", skip={"topic", "script"})


def test_texts_passes_every_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "texts", result_for(tmp_path, kind="texts"))

    code = main([
        "texts", "--topic", "a missed train", "--theme", "sunset", "--voice", "narrator_female",
        "--reply-voice", "narrator_deep", "--background", "ember_mist", "--music", "quiet_bed",
        "--backend", "pillow", "--turns", "6", "--captions", "--style", "boxed",
        "--out", str(tmp_path / "train.mp4"),
    ])

    assert code == 0
    assert run.kwargs == {
        "topic": "a missed train",
        "script": None,
        "theme": "sunset",
        "voice": "narrator_female",
        "reply_voice": "narrator_deep",
        "background": "ember_mist",
        "music": "quiet_bed",
        "backend": "pillow",
        "turns": 6,
        "captions": True,
        "style": "boxed",
        "out_path": tmp_path / "train.mp4",
    }


def test_texts_defaults_match_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "texts", result_for(tmp_path, kind="texts"))
    assert main(["texts", "--topic", "a missed train"]) == 0
    assert_defaults_match(run, "texts", skip={"topic", "script"})


def test_reddit_passes_every_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "reddit", result_for(tmp_path, kind="reddit"))

    code = main([
        "reddit", "--topic", "the quiet neighbour", "--theme", "light", "--voice", "narrator_deep",
        "--background", "ember_mist", "--music", "quiet_bed", "--style", "boxed",
        "--card-seconds", "4.5", "--words", "90", "--backend", "pillow", "--no-captions",
        "--out", str(tmp_path / "post.mp4"),
    ])

    assert code == 0
    assert run.kwargs == {
        "topic": "the quiet neighbour",
        "theme": "light",
        "voice": "narrator_deep",
        "background": "ember_mist",
        "music": "quiet_bed",
        "style": "boxed",
        "card_seconds": 4.5,
        "captions": False,
        "words": 90,
        "backend": "pillow",
        "out_path": tmp_path / "post.mp4",
    }


def test_reddit_defaults_match_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "reddit", result_for(tmp_path, kind="reddit"))
    assert main(["reddit", "--topic", "the quiet neighbour"]) == 0
    assert_defaults_match(run, "reddit", skip={"topic"})


def test_split_passes_every_argument(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "split", result_for(tmp_path, kind="split"))

    code = main([
        "split", "gameplay.mp4", "--bottom", "ember_mist", "--narration", "watch this part",
        "--voice", "narrator_deep", "--style", "neon", "--music", "quiet_bed", "--seconds", "12",
        "--no-captions", "--no-reframe", "--out", str(tmp_path / "stack.mp4"),
    ])

    assert code == 0
    assert run.args == ("gameplay.mp4",)
    assert run.kwargs == {
        "bottom": "ember_mist",
        "narration": "watch this part",
        "voice": "narrator_deep",
        "style": "neon",
        "music": "quiet_bed",
        "seconds": 12.0,
        "captions": False,
        "reframe": False,
        "out_path": tmp_path / "stack.mp4",
    }


def test_split_defaults_match_the_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "split", result_for(tmp_path, kind="split"))
    assert main(["split", "gameplay.mp4"]) == 0
    assert_defaults_match(run, "split", skip={"top"})


# --------------------------------------------------------------------------- #
# script input
# --------------------------------------------------------------------------- #


def test_script_from_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    script = tmp_path / "story.txt"
    script.write_text("# Title\nThe first beat.\nThe second beat.\n", encoding="utf-8")
    run = patch_pipeline(monkeypatch, "story", result_for(tmp_path))

    assert main(["story", "--script", str(script)]) == 0
    assert run.kwargs["script"] == script.read_text(encoding="utf-8")
    assert run.kwargs["topic"] is None


def test_script_from_stdin(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Alex: hey\nme: hey back\n"))
    run = patch_pipeline(monkeypatch, "texts", result_for(tmp_path, kind="texts"))

    assert main(["texts", "--script", "-"]) == 0
    assert run.kwargs["script"] == "Alex: hey\nme: hey back\n"


def test_missing_script_file_is_a_usage_error(tmp_path: Path, capsys) -> None:
    """A bad --script is bad input, so it exits 2 like every other bad input."""
    missing = tmp_path / "nope.txt"
    code = main(["story", "--script", str(missing)])
    err = capsys.readouterr().err

    assert code == 2
    assert "missing" in err and str(missing) in err
    assert "Traceback" not in err


def test_empty_script_file_is_a_usage_error(tmp_path: Path, capsys) -> None:
    script = tmp_path / "empty.txt"
    script.write_text("   \n\n", encoding="utf-8")

    assert main(["story", "--script", str(script)]) == 2
    err = capsys.readouterr().err
    assert "empty" in err and str(script) in err


def test_empty_stdin_script_is_a_usage_error(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    """`--script -` with nothing on stdin is the same class of error."""
    monkeypatch.setattr("sys.stdin", io.StringIO("   \n"))

    assert main(["texts", "--script", "-"]) == 2
    err = capsys.readouterr().err
    assert "empty" in err and "stdin" in err
    assert "Traceback" not in err


def test_an_unreadable_script_path_is_a_usage_error(tmp_path: Path, capsys) -> None:
    """A directory handed to --script names the path rather than crashing."""
    assert main(["story", "--script", str(tmp_path)]) == 2
    assert str(tmp_path) in capsys.readouterr().err


def test_topic_and_script_are_mutually_exclusive(capsys) -> None:
    assert main(["story", "--topic", "x", "--script", "-"]) == 2
    assert "not allowed with" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# global flags
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "argv",
    [
        ["--width", "180", "--height", "320", "--fps", "12", "--seed", "77", "story", "--topic", "x"],
        ["story", "--width", "180", "--height", "320", "--fps", "12", "--seed", "77", "--topic", "x"],
        ["--width", "180", "story", "--height", "320", "--fps", "12", "--seed", "77", "--topic", "x"],
    ],
    ids=["before", "after", "mixed"],
)
def test_global_flags_reach_get_settings(
    argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = patch_pipeline(monkeypatch, "story", result_for(tmp_path))

    assert main(argv) == 0

    settings = run.settings[-1]
    assert (settings.width, settings.height, settings.fps, settings.seed) == (180, 320, 12, 77)


def test_directory_flags_reach_get_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = patch_pipeline(monkeypatch, "story", result_for(tmp_path))
    work = tmp_path / "scratch"
    out = tmp_path / "finished"

    assert main(["--work-dir", str(work), "--output-dir", str(out), "story", "--topic", "x"]) == 0

    settings = run.settings[-1]
    assert settings.work_dir == work
    assert settings.output_dir == out
    assert settings.cache_dir == work / "cache"


def test_offline_flag_sets_the_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AICLIP_OFFLINE", "0")
    run = patch_pipeline(monkeypatch, "story", result_for(tmp_path))

    assert main(["story", "--topic", "x"]) == 0
    assert run.settings[-1].offline is False

    assert main(["--offline", "story", "--topic", "x"]) == 0
    assert run.settings[-1].offline is True


def test_environment_is_restored_after_main(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AICLIP_WIDTH", raising=False)
    patch_pipeline(monkeypatch, "story", result_for(tmp_path))
    before = get_settings().width

    assert main(["--width", "180", "story", "--topic", "x"]) == 0

    import os

    assert "AICLIP_WIDTH" not in os.environ
    assert get_settings().width == before


def test_environment_for_only_lists_flags_that_were_given() -> None:
    parser = build_parser()
    bare = parser.parse_args(["styles"])
    assert environment_for(bare) == {}

    rich = parser.parse_args(["--width", "180", "--offline", "--seed", "9", "styles"])
    assert environment_for(rich) == {
        "AICLIP_WIDTH": "180",
        "AICLIP_OFFLINE": "1",
        "AICLIP_SEED": "9",
    }


@pytest.mark.parametrize(
    ("argv", "level"),
    [(["styles"], logging.WARNING), (["-v", "styles"], logging.INFO), (["-vv", "styles"], logging.DEBUG)],
)
def test_verbosity_sets_the_logging_level(argv: list[str], level: int) -> None:
    assert main(argv) == 0
    assert logging.getLogger().level == level


def test_verbose_after_the_subcommand_also_counts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    patch_pipeline(monkeypatch, "story", result_for(tmp_path))
    assert main(["story", "--topic", "x", "-vv"]) == 0
    assert logging.getLogger().level == logging.DEBUG


# --------------------------------------------------------------------------- #
# --dry-run
# --------------------------------------------------------------------------- #


def test_dry_run_describes_the_call_without_running_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    run = patch_pipeline(monkeypatch, "clip", [result_for(tmp_path, kind="clip")])

    code = main(["--dry-run", "--width", "180", "clip", "lecture.mp4", "--count", "5"])

    assert code == 0
    assert run.calls == []
    out = capsys.readouterr().out
    assert "dry-run: clip" in out
    assert "source: lecture.mp4" in out
    assert "count: 5" in out
    assert "canvas: 180x1920@30" in out


def test_dry_run_after_the_subcommand(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    run = patch_pipeline(monkeypatch, "story", result_for(tmp_path))
    assert main(["story", "--topic", "x", "--dry-run"]) == 0
    assert run.calls == []
    assert "dry-run: story" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# voices / styles / assets
# --------------------------------------------------------------------------- #


def test_voices_lists_name_tags_and_providers(capsys) -> None:
    from aiclipper.tts import VOICES

    assert main(["voices"]) == 0
    out = capsys.readouterr().out
    lines = out.splitlines()
    assert len(lines) == len(VOICES)

    sample = next(v for v in VOICES if v.edge)
    row = next(line for line in lines if line.startswith(sample.name))
    assert "edge" in row and "offline" in row
    for tag in sample.tags:
        assert tag in row


def test_voices_filtered_by_provider_and_tag(capsys) -> None:
    assert main(["voices", "--provider", "elevenlabs", "--tag", "female"]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out

    from aiclipper.tts import list_voices

    expected = [v.name for v in list_voices("elevenlabs") if v.has_tag("female")]
    assert [line.split()[0] for line in out] == expected


def test_voices_verbose_adds_descriptions(capsys) -> None:
    assert main(["-v", "voices", "--tag", "documentary"]) == 0
    out = capsys.readouterr().out
    from aiclipper.tts import VOICES

    entry = next(v for v in VOICES if v.has_tag("documentary"))
    assert entry.description in out


def test_voices_unknown_provider_is_bad_usage(capsys) -> None:
    assert main(["voices", "--provider", "banana"]) == 2
    assert "unknown tts provider" in capsys.readouterr().err


def test_voices_with_no_match_fails_clearly(capsys) -> None:
    assert main(["voices", "--tag", "klingon"]) == 1
    err = capsys.readouterr().err
    assert "no voice matches" in err
    assert "known tags" in err


def test_styles_lists_every_preset_with_a_description(capsys) -> None:
    from aiclipper.captions import list_styles

    assert main(["styles"]) == 0
    out = capsys.readouterr().out
    presets = list_styles()
    assert len(out.splitlines()) == len(presets)
    for style in presets:
        assert style.name in out
        assert style.description in out


def test_assets_lists_the_library(tiny_library: Path, capsys) -> None:
    assert main(["assets"]) == 0
    out = capsys.readouterr().out
    assert "slow_drift" in out
    assert "quiet_bed" in out
    assert str(tiny_library / "backgrounds" / "slow_drift.mp4") in out
    assert "2 assets (1 background, 1 music)" in out


def test_assets_reports_an_empty_library(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setenv("AICLIP_ASSETS_DIR", str(tmp_path / "nothing"))
    assert main(["assets"]) == 0
    err = capsys.readouterr().err
    assert "empty" in err
    assert "assets --generate" in err


def test_assets_generate_calls_ensure_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    calls: list[Any] = []

    def fake_ensure(settings: Any = None) -> list[Any]:
        calls.append(settings)
        root = Path(settings.assets_dir)
        (root / "backgrounds").mkdir(parents=True, exist_ok=True)
        ff.run_ffmpeg([
            "-y", "-f", "lavfi", "-i", "color=c=black:size=64x64:rate=6:duration=1",
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(root / "backgrounds" / "made_up.mp4"),
        ])
        return []

    monkeypatch.setenv("AICLIP_ASSETS_DIR", str(tmp_path / "fresh"))
    monkeypatch.setattr("aiclipper.assets.ensure_placeholders", fake_ensure)

    assert main(["assets", "--generate"]) == 0
    assert len(calls) == 1
    assert "made_up" in capsys.readouterr().out


def test_assets_generate_dry_run_generates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    def explode(settings: Any = None) -> list[Any]:
        raise AssertionError("--dry-run must not generate assets")

    monkeypatch.setenv("AICLIP_ASSETS_DIR", str(tmp_path / "fresh"))
    monkeypatch.setattr("aiclipper.assets.ensure_placeholders", explode)

    assert main(["assets", "--generate", "--dry-run"]) == 0
    assert "dry-run: assets --generate" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# doctor
# --------------------------------------------------------------------------- #


@pytest.mark.needs_ffmpeg
def test_doctor_reports_every_component(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("aiclipper.cli._chromium_report", lambda settings: (True, "chromium 1.2.3"))

    assert main(["doctor"]) == 0

    out = capsys.readouterr().out
    assert "ffmpeg version" in out
    assert "ffprobe version" in out
    for label in ("faster-whisper", "yt-dlp", "opencv", "anthropic", "edge-tts", "playwright",
                  "claude key", "llm provider", "tts:edge", "tts:offline", "chromium", "assets"):
        assert label in out
    assert "chromium 1.2.3" in out
    assert "core ok" in out
    for line in out.splitlines():
        if line.startswith(("OK", "MISSING")):
            assert len(line.split()) >= 2


@pytest.mark.needs_ffmpeg
def test_doctor_survives_a_probe_that_explodes(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def boom(settings: Any) -> tuple[bool, str]:
        raise RuntimeError("browser exploded")

    monkeypatch.setattr("aiclipper.cli._chromium_report", boom)

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "MISSING chromium" in out.replace("  ", " ")
    assert "browser exploded" in out


def test_doctor_honours_the_global_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("aiclipper.cli._chromium_report", lambda settings: (True, "chromium 1.2.3"))
    out_dir = tmp_path / "finished"

    assert main(["--output-dir", str(out_dir), "--width", "540", "--height", "960", "doctor"]) == 0

    out = capsys.readouterr().out
    assert str(out_dir) in out
    assert "540x960" in out


def test_doctor_fails_when_ffmpeg_is_missing(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("aiclipper.cli.shutil.which", lambda name: None)
    monkeypatch.setattr("aiclipper.cli._chromium_report", lambda settings: (False, "not checked"))

    assert main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "MISSING ffmpeg" in out.replace("  ", " ")
    assert "core missing" in out


def test_doctor_launches_chromium_for_real(capsys) -> None:
    """The real probe: either a version string or a clear MISSING reason."""
    pytest.importorskip("playwright")
    from aiclipper.cli import _chromium_report

    ok, detail = _chromium_report(get_settings())
    assert isinstance(detail, str) and detail
    if ok:
        assert "chromium" in detail


# --------------------------------------------------------------------------- #
# exit codes and error reporting
# --------------------------------------------------------------------------- #


def test_pipeline_failure_is_one_clean_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    patch_pipeline(monkeypatch, "story", AssetError("no background named 'nope'"))

    code = main(["story", "--topic", "x", "--background", "nope"])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.strip() == "aiclip: no background named 'nope'"


def test_multiline_errors_collapse_to_one_line(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    patch_pipeline(monkeypatch, "story", RenderError("timeline is invalid", problems=["no layers", "no audio"]))

    assert main(["story", "--topic", "x"]) == 1
    err = capsys.readouterr().err.strip()
    assert err.count("\n") == 0
    assert "no layers" in err and "no audio" in err


def test_verbose_adds_a_traceback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    patch_pipeline(monkeypatch, "story", AssetError("boom"))

    assert main(["-v", "story", "--topic", "x"]) == 1
    err = capsys.readouterr().err
    assert "Traceback" in err
    assert "aiclip: boom" in err


def test_empty_result_list_is_a_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    patch_pipeline(monkeypatch, "clip", [])
    assert main(["clip", "lecture.mp4"]) == 1
    assert "produced nothing" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["nosuchcommand"],
        ["clip"],
        ["clip", "a.mp4", "--count", "many"],
        ["split"],
        ["reddit"],
        ["texts", "--backend", "webgl"],
    ],
)
def test_bad_usage_exits_two(argv: list[str], capsys) -> None:
    assert main(argv) == 2
    assert capsys.readouterr().err


def test_help_exits_zero(capsys) -> None:
    assert main(["--help"]) == 0
    out = capsys.readouterr().out
    for command in ("clip", "story", "texts", "reddit", "split", "voices", "styles", "assets", "doctor"):
        assert command in out


def test_subcommand_help_exits_zero(capsys) -> None:
    assert main(["story", "--help"]) == 0
    assert "--seconds" in capsys.readouterr().out


def test_version_exits_zero(capsys) -> None:
    from aiclipper import __version__

    assert main(["--version"]) == 0
    assert __version__ in capsys.readouterr().out


def test_keyboard_interrupt_is_handled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    def interrupt(*args: Any, **kwargs: Any) -> Any:
        raise KeyboardInterrupt

    monkeypatch.setattr("aiclipper.pipelines.story.run", interrupt)
    assert main(["story", "--topic", "x"]) == 1
    assert "interrupted" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# end to end
# --------------------------------------------------------------------------- #


@pytest.mark.needs_ffmpeg
@pytest.mark.slow
def test_story_end_to_end_prints_a_real_file(tiny_library: Path, tmp_path: Path, capsys) -> None:
    code = main([
        "--width", "180", "--height", "320", "--fps", "12",
        "story", "--topic", "why bread rises", "--seconds", "6", "--style", "clean",
    ])

    captured = capsys.readouterr()
    assert code == 0, captured.err

    lines = captured.out.splitlines()
    assert len(lines) == 1
    produced = Path(lines[0])
    assert produced.is_file()
    assert produced.stat().st_size > 2_000

    info = ff.probe(produced)
    assert info.width == 180
    assert info.height == 320
    assert info.has_audio
    assert 3.0 < info.duration < 40.0


@pytest.mark.needs_ffmpeg
@pytest.mark.slow
def test_story_end_to_end_honours_out_and_script(tiny_library: Path, tmp_path: Path, capsys) -> None:
    script = tmp_path / "beats.txt"
    script.write_text("# Rising Dough\nYeast eats sugar.\nThe gas is trapped.\n", encoding="utf-8")
    target = tmp_path / "custom" / "dough.mp4"

    code = main([
        "--width", "180", "--height", "320", "--fps", "12",
        "story", "--script", str(script), "--no-captions", "--out", str(target),
    ])

    captured = capsys.readouterr()
    assert code == 0, captured.err
    assert captured.out.strip() == str(target)
    assert target.is_file()
    assert ff.probe(target).duration > 1.0


def test_callable_entry_point_is_exported() -> None:
    from aiclipper import cli

    assert isinstance(cli.main, Callable)
    assert "main" in cli.__all__


# --------------------------------------------------------------------------- #
# up-front validation of every enumerated option
# --------------------------------------------------------------------------- #


def explode_pipeline(monkeypatch: pytest.MonkeyPatch, name: str) -> None:
    """Replace a pipeline ``run`` with a landmine: reaching it fails the test."""

    def landmine(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name}.run was reached; validation should have rejected the arguments")

    monkeypatch.setattr(f"aiclipper.pipelines.{name}.run", landmine)


@pytest.mark.parametrize(
    ("argv", "pipeline", "needle"),
    [
        (["texts", "--topic", "x", "--theme", "nosuch"], "texts", "unknown chat theme 'nosuch'"),
        (["reddit", "--topic", "x", "--theme", "nosuch"], "reddit", "unknown forum theme 'nosuch'"),
        (["story", "--topic", "x", "--style", "nosuch"], "story", "unknown caption style 'nosuch'"),
        (["clip", "a.mp4", "--style", "nosuch"], "clip", "unknown caption style 'nosuch'"),
        (["split", "a.mp4", "--style", "nosuch"], "split", "unknown caption style 'nosuch'"),
        (["texts", "--topic", "x", "--style", "nosuch"], "texts", "unknown caption style 'nosuch'"),
        (["story", "--topic", "x", "--voice", "nosuchvoice"], "story", "unknown voice 'nosuchvoice'"),
        (["reddit", "--topic", "x", "--voice", "nosuchvoice"], "reddit", "unknown voice 'nosuchvoice'"),
        (["split", "a.mp4", "--voice", "nosuchvoice"], "split", "unknown voice 'nosuchvoice'"),
        (["texts", "--topic", "x", "--reply-voice", "nope"], "texts", "--reply-voice"),
    ],
)
def test_unknown_enumerated_value_exits_two_before_the_pipeline_runs(
    argv: list[str], pipeline: str, needle: str, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    explode_pipeline(monkeypatch, pipeline)

    assert main(argv) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert needle in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("argv", "known"),
    [
        (["texts", "--topic", "x", "--theme", "nosuch"], lambda: sorted(_chat_themes())),
        (["reddit", "--topic", "x", "--theme", "nosuch"], lambda: sorted(_forum_themes())),
        (["story", "--topic", "x", "--style", "nosuch"], lambda: sorted(_style_names())),
    ],
)
def test_the_rejection_lists_every_valid_value(
    argv: list[str], known: Callable[[], list[str]], monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    explode_pipeline(monkeypatch, argv[0])

    assert main(argv) == 2
    err = capsys.readouterr().err
    for name in known():
        assert name in err, f"{name!r} missing from the rejection message"


def _chat_themes() -> list[str]:
    from aiclipper.overlays import CHAT_THEMES

    return list(CHAT_THEMES)


def _forum_themes() -> list[str]:
    from aiclipper.overlays import FORUM_THEMES

    return list(FORUM_THEMES)


def _style_names() -> list[str]:
    from aiclipper.captions import PRESETS

    return list(PRESETS)


@pytest.mark.parametrize(
    "argv",
    [
        ["texts", "--topic", "x", "--theme", "CLASSIC"],
        ["texts", "--topic", "x", "--theme", " mint "],
        ["story", "--topic", "x", "--style", "bold-yellow"],
        ["story", "--topic", "x", "--voice", "narrator_deep"],
        ["story", "--topic", "x", "--voice", "british female"],
        ["story", "--topic", "x", "--voice", "edge:en-GB-RyanNeural"],
        ["story", "--topic", "x"],
    ],
)
def test_validation_lets_good_values_through(
    argv: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    name = argv[0]
    patch_pipeline(monkeypatch, name, result_for(tmp_path, kind=name))
    assert main(argv) == 0


def test_a_bad_theme_is_rejected_even_under_dry_run(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    explode_pipeline(monkeypatch, "texts")
    assert main(["texts", "--topic", "x", "--theme", "nosuch", "--dry-run"]) == 2
    assert "unknown chat theme" in capsys.readouterr().err


# --------------------------------------------------------------------------- #
# degenerate numbers are a usage error, not a pointless render
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("argv", "pipeline", "needle"),
    [
        (["clip", "a.mp4", "--count", "0"], "clip", "--count must be at least 1"),
        (["clip", "a.mp4", "--count", "-3"], "clip", "--count must be at least 1"),
        (["clip", "a.mp4", "--min", "0"], "clip", "--min must be at least 1"),
        (["clip", "a.mp4", "--max", "0.5"], "clip", "--max must be at least 1"),
        (["clip", "a.mp4", "--min", "60", "--max", "10"], "clip", "--min must not exceed --max"),
        (["split", "a.mp4", "--seconds", "0.05"], "split", "--seconds must be at least 1"),
        (["split", "a.mp4", "--seconds", "0"], "split", "--seconds must be at least 1"),
        (["story", "--topic", "x", "--seconds", "0"], "story", "--seconds must be at least 1"),
        (["texts", "--topic", "x", "--turns", "0"], "texts", "--turns must be at least 1"),
        (["reddit", "--topic", "x", "--words", "3"], "reddit", "--words must be at least 20"),
        (["reddit", "--topic", "x", "--card-seconds", "0.2"], "reddit", "--card-seconds must be at least 1"),
    ],
)
def test_degenerate_numbers_exit_two_before_the_pipeline_runs(
    argv: list[str], pipeline: str, needle: str, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    explode_pipeline(monkeypatch, pipeline)

    assert main(argv) == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert needle in captured.err
    assert "Traceback" not in captured.err


def test_the_floors_are_documented_in_the_help(capsys) -> None:
    from aiclipper.cli import MIN_COUNT, MIN_SECONDS, MIN_TURNS, MIN_WORDS

    assert main(["clip", "--help"]) == 0
    clip_help = capsys.readouterr().out
    assert f"at least {MIN_COUNT}" in clip_help
    assert f"at least {MIN_SECONDS:g}" in clip_help

    assert main(["texts", "--help"]) == 0
    assert f"at least {MIN_TURNS}" in capsys.readouterr().out

    assert main(["reddit", "--help"]) == 0
    assert f"at least {MIN_WORDS}" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("argv", "pipeline"),
    [
        (["clip", "a.mp4", "--count", "1", "--min", "1", "--max", "1"], "clip"),
        (["split", "a.mp4", "--seconds", "1"], "split"),
        (["reddit", "--topic", "x", "--words", "20"], "reddit"),
    ],
)
def test_the_floor_itself_is_accepted(
    argv: list[str], pipeline: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = result_for(tmp_path, kind=pipeline)
    patch_pipeline(monkeypatch, pipeline, [result] if pipeline == "clip" else result)
    assert main(argv) == 0


# --------------------------------------------------------------------------- #
# doctor: installed is not usable
# --------------------------------------------------------------------------- #


class FakeProvider:
    def __init__(self, name: str, available: bool) -> None:
        self.name = name
        self._available = available

    def available(self) -> bool:
        return self._available

    def synthesize(self, text: str, out_path: Path, *, voice: Any) -> Any:  # pragma: no cover
        raise AssertionError("doctor must never synthesise")


def patch_tts(monkeypatch: pytest.MonkeyPatch, usable: dict[str, bool], available: dict[str, bool]) -> None:
    import aiclipper.tts as tts

    monkeypatch.setattr(
        tts, "get_provider", lambda name=None, **kw: FakeProvider(str(name), available[str(name)])
    )
    monkeypatch.setattr(tts, "provider_usable", lambda provider, **kw: usable[provider.name])


def test_doctor_separates_installed_from_usable(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr("aiclipper.cli._chromium_report", lambda settings: (True, "chromium 1.2.3"))
    patch_tts(
        monkeypatch,
        usable={"edge": False, "elevenlabs": False, "offline": True},
        available={"edge": True, "elevenlabs": False, "offline": True},
    )

    assert main(["doctor"]) == 0

    rows = {
        line.split()[1]: line
        for line in capsys.readouterr().out.splitlines()
        if line.startswith(("OK", "MISSING"))
    }
    # importable, but no route to the service: a MISSING row that says why.
    assert rows["tts:edge"].startswith("MISSING")
    assert "installed" in rows["tts:edge"]
    assert "cannot synthesise" in rows["tts:edge"]
    # not installed at all: a different reason, still MISSING.
    assert rows["tts:elevenlabs"].startswith("MISSING")
    assert "not installed" in rows["tts:elevenlabs"]
    # the one that really works.
    assert rows["tts:offline"].startswith("OK")
    assert "usable" in rows["tts:offline"]


def test_doctor_uses_the_capability_probe_not_just_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """The row must follow ``provider_usable``, not ``available()``."""
    import aiclipper.tts as tts
    from aiclipper.cli import _tts_report

    monkeypatch.setattr(tts, "get_provider", lambda name=None, **kw: FakeProvider(str(name), True))
    asked: list[str] = []

    def probe(provider: Any, **kwargs: Any) -> bool:
        asked.append(provider.name)
        return False

    monkeypatch.setattr(tts, "provider_usable", probe)

    ok, detail = _tts_report("edge", get_settings())
    assert asked == ["edge"]
    assert ok is False
    assert "installed" in detail and "cannot synthesise" in detail


def test_doctor_survives_a_tts_backend_that_explodes(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    import aiclipper.tts as tts

    def boom(name: Any = None, **kwargs: Any) -> Any:
        raise RuntimeError("tts backend exploded")

    monkeypatch.setattr("aiclipper.cli._chromium_report", lambda settings: (True, "chromium 1.2.3"))
    monkeypatch.setattr(tts, "get_provider", boom)

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out.replace("  ", " ")
    assert "MISSING tts:edge" in out
    assert "tts backend exploded" in out


def test_doctor_names_the_real_blocker_instead_of_saying_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A backend held back by offline mode or a missing key is not "not installed".

    ``available()`` returns ``False`` for three different reasons; reporting all
    of them as "not installed" contradicts the extras rows above, which say
    ``edge-tts`` imported fine.
    """
    import aiclipper.tts as tts
    from aiclipper.cli import _tts_report
    from aiclipper.tts.edge import EdgeTTS

    monkeypatch.setattr(tts, "provider_usable", lambda provider, **kwargs: False)
    monkeypatch.setattr("aiclipper.tts.edge.edge_available", lambda: True)

    # AICLIP_OFFLINE is set by the autouse fixture: edge is installed, just fenced off.
    settings = get_settings()
    assert settings.offline
    assert isinstance(tts.get_provider("edge", settings=settings), EdgeTTS)
    ok, detail = _tts_report("edge", settings)
    assert ok is False
    assert "installed" in detail and "offline" in detail
    assert "not installed" not in detail

    # No key is its own reason, and never "not installed" either.
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    ok, detail = _tts_report("elevenlabs", settings)
    assert ok is False
    assert "ELEVENLABS_API_KEY" in detail
    assert "not installed" not in detail

    # The package genuinely absent still reads as "not installed".
    monkeypatch.setattr("aiclipper.tts.edge.edge_available", lambda: False)
    ok, detail = _tts_report("edge", settings)
    assert ok is False
    assert "not installed" in detail


@pytest.mark.needs_ffmpeg
def test_doctor_still_exits_zero_with_working_ffmpeg_and_a_dead_backend(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    monkeypatch.setattr("aiclipper.cli._chromium_report", lambda settings: (True, "chromium 1.2.3"))
    patch_tts(
        monkeypatch,
        usable={"edge": False, "elevenlabs": False, "offline": False},
        available={"edge": True, "elevenlabs": True, "offline": True},
    )

    assert main(["doctor"]) == 0
    out = capsys.readouterr().out
    assert "core ok" in out
    assert "MISSING tts:offline" in out.replace("  ", " ")


def test_validate_rejects_an_unknown_overlay_backend() -> None:
    """The parser's ``choices`` catch this first; ``validate`` is the backstop."""
    from aiclipper.cli import UsageError, validate

    ns = build_parser().parse_args(["texts", "--topic", "x", "--backend", "pillow"])
    validate(ns)  # a good backend passes

    ns.backend = "webgl"
    with pytest.raises(UsageError) as excinfo:
        validate(ns)
    assert "unknown overlay backend 'webgl'" in str(excinfo.value)
    assert "chromium" in str(excinfo.value) and "pillow" in str(excinfo.value)


def test_validate_ignores_the_catalogue_commands() -> None:
    """``voices``/``styles``/``assets``/``doctor`` carry none of these options."""
    from aiclipper.cli import validate

    for argv in (["voices"], ["styles"], ["assets"], ["doctor"]):
        validate(build_parser().parse_args(argv))


def test_doctor_separates_a_whisper_install_from_its_weights(monkeypatch: pytest.MonkeyPatch) -> None:
    """Importing faster-whisper is not the same as being able to transcribe.

    The weights are downloaded on first use, so an offline box with the package
    installed still cannot run ASR -- and ``doctor`` has to say so rather than
    printing a bare OK.
    """
    from aiclipper.cli import _transcribe_report

    monkeypatch.setattr("aiclipper.transcribe.available", lambda: True)

    monkeypatch.setattr("aiclipper.cli._whisper_weights_cached", lambda model: True)
    ok, detail = _transcribe_report(get_settings())
    assert ok is True and "cached locally" in detail

    monkeypatch.setattr("aiclipper.cli._whisper_weights_cached", lambda model: False)
    ok, detail = _transcribe_report(get_settings())
    assert ok is False, detail
    assert "installed" in detail and "not cached here" in detail

    monkeypatch.setattr("aiclipper.cli._whisper_weights_cached", lambda model: None)
    ok, detail = _transcribe_report(get_settings())
    assert ok is True and "not verified" in detail

    monkeypatch.setattr("aiclipper.transcribe.available", lambda: False)
    ok, detail = _transcribe_report(get_settings())
    assert ok is False and "aiclipper[transcribe]" in detail


def test_the_whisper_weight_probe_never_needs_a_network(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from aiclipper.cli import _whisper_weights_cached

    assert _whisper_weights_cached("") is None
    # a model given as a local directory needs no cache lookup at all
    assert _whisper_weights_cached(str(tmp_path)) is True
    # a name that is not on this disk is reported as missing, not as an error
    assert _whisper_weights_cached("aiclipper-no-such-model-9f3a") is False
