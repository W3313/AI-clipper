"""Tests for :mod:`aiclipper.ingest`.

No external network is touched.  The only media fixture is a two-second clip
generated with ffmpeg's lavfi sources; the download path is exercised twice --
once with ``ingest._fetch`` swapped for a local copy (so a cache hit can be
proved not to call it) and once against a loopback http server, which drives the
real yt-dlp module and CLI backends end to end.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import pytest

from aiclipper import ffmpeg
from aiclipper.config import Settings
from aiclipper.errors import IngestError, MissingDependency
from aiclipper.ingest import DEFAULT_QUALITY, _fetch_with_cli, cache_key, download, is_url, normalize, resolve
from aiclipper.models import MediaInfo

pytestmark = pytest.mark.skipif(not ffmpeg.have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")

URL = "https://example.invalid/watch?v=abc123"


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def sample_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2s 320x240 testsrc2 clip with a sine tone (video + audio)."""
    out = tmp_path_factory.mktemp("ingest-media") / "sample clip.mp4"
    ffmpeg.run_ffmpeg([
        "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-t", "2", str(out),
    ])
    return out


@pytest.fixture(scope="module")
def silent_video(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 2s 320x240 clip with no audio stream at all."""
    out = tmp_path_factory.mktemp("ingest-silent") / "mute.mp4"
    ffmpeg.run_ffmpeg([
        "-y",
        "-f", "lavfi", "-i", "testsrc2=size=320x240:rate=30:duration=2",
        "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
        "-t", "2", str(out),
    ])
    return out


@pytest.fixture(scope="module")
def audio_only(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 1s tone with no video stream at all."""
    out = tmp_path_factory.mktemp("ingest-audio") / "tone.m4a"
    ffmpeg.run_ffmpeg([
        "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
        "-c:a", "aac", "-t", "1", str(out),
    ])
    return out


@pytest.fixture
def cfg(tmp_path: Path) -> Settings:
    """Explicit settings for this module.

    The shared conftest forces ``AICLIP_OFFLINE=1`` for the whole suite, so the
    download-path tests build their own object with ``offline=False`` (the fetch
    helper is either monkeypatched or pointed at a loopback server; no test here
    reaches beyond ``127.0.0.1``).
    """
    return Settings(
        work_dir=tmp_path / "work",
        output_dir=tmp_path / "out",
        assets_dir=tmp_path / "assets",
        offline=False,
    )


@dataclass
class LocalServer:
    """A loopback http server serving the fixture clip, with a request counter."""

    url: str
    _counter: list[int]

    @property
    def hits(self) -> int:
        return self._counter[0]


@pytest.fixture(scope="module")
def local_server(sample_video: Path):
    counter = [0]

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, directory=str(sample_video.parent), **kwargs)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's API
            counter[0] += 1
            super().do_GET()

        def log_message(self, fmt: str, *args: Any) -> None:
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    url = f"http://{host}:{port}/{quote(sample_video.name)}"
    try:
        yield LocalServer(url, counter)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class FakeFetch:
    """Stand-in for ``ingest._fetch`` that copies a local file instead of downloading."""

    def __init__(self, source: Path, title: str = "Fixture Clip") -> None:
        self.source = source
        self.title = title
        self.calls: list[str] = []

    def __call__(
        self, url: str, dest_dir: Path, *, quality: str, settings: Settings
    ) -> tuple[Path, dict[str, Any]]:
        self.calls.append(url)
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / "fetched.mp4"
        target.write_bytes(self.source.read_bytes())
        return target, {"title": self.title, "id": "abc123", "extractor_key": "Fake"}


# --------------------------------------------------------------------------- #
# is_url
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("value", [
    "http://example.com/v.mp4",
    "https://example.com/watch?v=abc",
    "HTTPS://Example.COM/x",
    "https://example.com:8443/a/b.mp4#t=3",
])
def test_is_url_accepts_http_urls(value: str) -> None:
    assert is_url(value) is True


@pytest.mark.parametrize("value", [
    "",
    "   ",
    "video.mp4",
    "/abs/path/video.mp4",
    "./relative/video.mp4",
    "../../etc/passwd",
    "~/movies/clip.mkv",
    "file:///etc/passwd",
    "ftp://example.com/v.mp4",
    "rtmp://example.com/live",
    "C:\\videos\\clip.mp4",
    "http://",
    "https:///no/host.mp4",
    "https://example.com/a/../../etc/passwd",
    "https://example.com/a/%2e%2e/secret",
    "https://exa mple.com/v.mp4",
    "https://example.com/v\nmp4",
])
def test_is_url_rejects_everything_else(value: str) -> None:
    assert is_url(value) is False


def test_is_url_ignores_non_strings() -> None:
    assert is_url(Path("/tmp/x.mp4")) is False  # type: ignore[arg-type]


def test_cache_key_is_stable_and_quality_sensitive() -> None:
    a = cache_key(URL, DEFAULT_QUALITY)
    assert a == cache_key(URL, DEFAULT_QUALITY)
    assert len(a) == 16 and all(c in "0123456789abcdef" for c in a)
    assert a != cache_key(URL, "worst")
    assert a != cache_key(URL + "x", DEFAULT_QUALITY)


# --------------------------------------------------------------------------- #
# resolve on local files
# --------------------------------------------------------------------------- #

def test_resolve_local_file(sample_video: Path, cfg: Settings) -> None:
    info = resolve(sample_video, settings=cfg)
    assert isinstance(info, MediaInfo)
    assert info.path == sample_video
    assert info.width == 320 and info.height == 240
    assert info.has_video and info.has_audio
    assert info.duration == pytest.approx(2.0, abs=0.3)
    assert info.fps == pytest.approx(30.0, abs=0.5)
    assert info.size_bytes > 0
    assert info.source_url is None
    assert info.title == "sample clip"
    assert info.aspect == pytest.approx(4 / 3, abs=0.01)
    assert info.is_vertical is False


def test_resolve_accepts_a_string_path(sample_video: Path, cfg: Settings) -> None:
    info = resolve(str(sample_video), settings=cfg)
    assert info.width == 320 and info.path == sample_video


def test_resolve_missing_path_raises_ingest_error(tmp_path: Path, cfg: Settings) -> None:
    with pytest.raises(IngestError) as excinfo:
        resolve(tmp_path / "nope.mp4", settings=cfg)
    assert "nope.mp4" in str(excinfo.value)


def test_resolve_directory_raises_ingest_error(tmp_path: Path, cfg: Settings) -> None:
    with pytest.raises(IngestError):
        resolve(tmp_path, settings=cfg)


def test_resolve_unreadable_file_raises_ingest_error(tmp_path: Path, cfg: Settings) -> None:
    bogus = tmp_path / "not-media.mp4"
    bogus.write_text("this is not a video", encoding="utf-8")
    with pytest.raises(IngestError):
        resolve(bogus, settings=cfg)


# --------------------------------------------------------------------------- #
# normalize
# --------------------------------------------------------------------------- #

def test_normalize_applies_fps_and_width(sample_video: Path, cfg: Settings, tmp_path: Path) -> None:
    info = resolve(sample_video, settings=cfg)
    out = normalize(info, tmp_path / "norm.mp4", width=160, fps=15, settings=cfg)

    assert out.path.is_file() and out.path != info.path
    assert (out.width, out.height) == (160, 120)          # aspect ratio preserved
    assert out.fps == pytest.approx(15.0, abs=0.5)
    assert out.has_video and out.has_audio
    assert out.duration == pytest.approx(2.0, abs=0.3)

    # still probe-able through the normal path
    reprobed = resolve(out.path, settings=cfg)
    assert (reprobed.width, reprobed.height) == (160, 120)
    assert reprobed.has_audio


def test_normalize_fits_inside_a_box_without_distorting(
    sample_video: Path, cfg: Settings, tmp_path: Path
) -> None:
    info = resolve(sample_video, settings=cfg)
    out = normalize(info, tmp_path / "boxed.mp4", width=200, height=200, settings=cfg)
    assert (out.width, out.height) == (200, 150)
    assert out.aspect == pytest.approx(info.aspect, abs=0.01)


def test_normalize_scales_by_height_only(sample_video: Path, cfg: Settings, tmp_path: Path) -> None:
    info = resolve(sample_video, settings=cfg)
    out = normalize(info, tmp_path / "tall.mp4", height=120, settings=cfg)
    assert (out.width, out.height) == (160, 120)


def test_normalize_without_size_keeps_dimensions_and_carries_metadata(
    sample_video: Path, cfg: Settings, tmp_path: Path
) -> None:
    info = resolve(sample_video, settings=cfg)
    info.title = "Carried Title"
    info.source_url = URL
    out = normalize(info, tmp_path / "same", settings=cfg)
    assert out.path.suffix == ".mp4"                      # extension supplied for us
    assert (out.width, out.height) == (320, 240)
    assert out.fps == pytest.approx(info.fps, abs=0.5)
    assert out.title == "Carried Title"
    assert out.source_url == URL


def test_normalize_gives_a_silent_source_an_audio_track(
    silent_video: Path, cfg: Settings, tmp_path: Path
) -> None:
    info = resolve(silent_video, settings=cfg)
    assert info.has_audio is False
    out = normalize(info, tmp_path / "with-audio.mp4", fps=10, settings=cfg)
    assert out.has_audio is True
    assert out.has_video is True
    assert out.duration == pytest.approx(2.0, abs=0.4)


def test_normalize_missing_source_raises(cfg: Settings, tmp_path: Path) -> None:
    info = MediaInfo(path=tmp_path / "ghost.mp4", has_video=True)
    with pytest.raises(IngestError):
        normalize(info, tmp_path / "out.mp4", settings=cfg)


# --------------------------------------------------------------------------- #
# download + cache
# --------------------------------------------------------------------------- #

def test_download_caches_and_second_call_never_fetches(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)

    dest = tmp_path / "dl"
    first = download(URL, dest, settings=cfg)
    assert fake.calls == [URL]
    assert first.path.is_file() and first.path.parent == dest
    assert first.title == "Fixture Clip"
    assert first.source_url == URL
    assert (first.width, first.height) == (320, 240)

    # the cache slot holds the media plus the sidecar
    slot = cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)
    sidecar = slot / "source.json"
    assert sidecar.is_file()
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    assert meta["url"] == URL and meta["title"] == "Fixture Clip"

    # second call is served from the cache: the fetch function is never entered
    monkeypatch.setattr(
        "aiclipper.ingest._fetch",
        lambda *a, **k: pytest.fail("cache hit must not touch the network"),
    )
    second = download(URL, tmp_path / "dl2", settings=cfg)
    assert fake.calls == [URL]
    assert second.title == "Fixture Clip"
    assert second.source_url == URL
    assert (second.width, second.height) == (320, 240)
    assert second.path.is_file() and second.path.parent == tmp_path / "dl2"


def test_download_cache_is_keyed_by_quality(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)
    download(URL, tmp_path / "a", settings=cfg, quality="best")
    download(URL, tmp_path / "b", settings=cfg, quality="worst")
    assert fake.calls == [URL, URL]
    download(URL, tmp_path / "c", settings=cfg, quality="best")
    assert fake.calls == [URL, URL]


def test_resolve_url_downloads_into_workspace_then_reuses_cache(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(sample_video, title="From URL")
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)

    workspace = tmp_path / "job"
    info = resolve(URL, settings=cfg, workspace=workspace)
    assert info.path.parent == workspace
    assert info.source_url == URL and info.title == "From URL"

    again = resolve(URL, settings=cfg, workspace=workspace)
    assert len(fake.calls) == 1
    assert again.path == info.path


def test_resolve_url_without_workspace_lands_in_the_cache(
    sample_video: Path, cfg: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)
    info = resolve(URL, settings=cfg)
    assert info.path.parent == cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)


def test_download_rejects_non_http_sources(cfg: Settings, tmp_path: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aiclipper.ingest._fetch",
        lambda *a, **k: pytest.fail("must not fetch a rejected url"),
    )
    for bad in ("file:///etc/passwd", "ftp://example.com/v.mp4", "../../etc/passwd"):
        with pytest.raises(IngestError):
            download(bad, tmp_path / "dl", settings=cfg)


def test_download_offline_without_cache_raises(cfg: Settings, tmp_path: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "aiclipper.ingest._fetch",
        lambda *a, **k: pytest.fail("offline mode must not fetch"),
    )
    cfg.offline = True
    with pytest.raises(IngestError) as excinfo:
        download(URL, tmp_path / "dl", settings=cfg)
    assert "offline" in str(excinfo.value).lower()


def test_download_offline_still_serves_a_cache_hit(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)
    download(URL, tmp_path / "dl", settings=cfg)

    cfg.offline = True
    monkeypatch.setattr(
        "aiclipper.ingest._fetch",
        lambda *a, **k: pytest.fail("offline mode must not fetch"),
    )
    info = download(URL, tmp_path / "dl-offline", settings=cfg)
    assert info.title == "Fixture Clip"


def test_download_without_yt_dlp_raises_missing_dependency(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("aiclipper.ingest._load_yt_dlp", lambda: None)
    monkeypatch.setattr("aiclipper.ingest._yt_dlp_cli", lambda: None)
    with pytest.raises(MissingDependency) as excinfo:
        download(URL, tmp_path / "dl", settings=cfg)
    assert "yt-dlp" in str(excinfo.value)
    assert "pip install" in str(excinfo.value)


def test_download_failure_becomes_ingest_error(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Boom:
        def YoutubeDL(self, opts: dict[str, Any]) -> Any:  # noqa: N802 - mirrors yt_dlp's API
            raise RuntimeError("network is down")

    monkeypatch.setattr("aiclipper.ingest._load_yt_dlp", lambda: Boom())
    with pytest.raises(IngestError) as excinfo:
        download(URL, tmp_path / "dl", settings=cfg)
    assert "network is down" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# argument validation + edge cases the brief calls for
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad", ["", "   ", "\t\n"])
def test_resolve_rejects_an_empty_source(bad: str, cfg: Settings) -> None:
    with pytest.raises(IngestError) as excinfo:
        resolve(bad, settings=cfg)
    assert "empty" in str(excinfo.value).lower()


def test_resolve_strips_surrounding_whitespace_from_a_url(
    sample_video: Path, cfg: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pasted URL usually arrives with a trailing newline; it must still fetch."""
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)
    info = resolve(f"  {URL}\n", settings=cfg)
    assert fake.calls == [URL], "the url must be trimmed before it is cached/fetched"
    assert info.source_url == URL
    assert info.path.parent == cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)


@pytest.mark.parametrize("kwargs", [
    {"width": 0}, {"width": -320}, {"height": 0}, {"height": -2}, {"fps": 0}, {"fps": -30},
])
def test_normalize_rejects_non_positive_geometry(
    sample_video: Path, cfg: Settings, tmp_path: Path, kwargs: dict[str, Any]
) -> None:
    info = resolve(sample_video, settings=cfg)
    with pytest.raises(IngestError):
        normalize(info, tmp_path / "bad.mp4", settings=cfg, **kwargs)
    assert not (tmp_path / "bad.mp4").exists()


def test_normalize_rounds_an_odd_width_down_to_even(
    sample_video: Path, cfg: Settings, tmp_path: Path
) -> None:
    """yuv420p needs even dimensions; an odd request must not blow up the encode."""
    info = resolve(sample_video, settings=cfg)
    out = normalize(info, tmp_path / "odd.mp4", width=161, settings=cfg)
    assert out.width % 2 == 0 and out.height % 2 == 0
    assert out.width == 160


def test_normalize_refuses_to_overwrite_its_own_source(
    sample_video: Path, cfg: Settings, tmp_path: Path
) -> None:
    copy = tmp_path / "inplace.mp4"
    copy.write_bytes(sample_video.read_bytes())
    before = copy.read_bytes()
    info = resolve(copy, settings=cfg)
    with pytest.raises(IngestError) as excinfo:
        normalize(info, copy, settings=cfg)
    assert "source" in str(excinfo.value)
    assert copy.read_bytes() == before, "the source file was clobbered"


def test_normalize_creates_missing_parent_directories(
    sample_video: Path, cfg: Settings, tmp_path: Path
) -> None:
    info = resolve(sample_video, settings=cfg)
    out = normalize(info, tmp_path / "deep" / "nested" / "clip.mp4", settings=cfg)
    assert out.path.is_file() and out.path.parent == tmp_path / "deep" / "nested"


def test_normalize_keeps_an_audio_only_source_audio_only(
    audio_only: Path, cfg: Settings, tmp_path: Path
) -> None:
    info = resolve(audio_only, settings=cfg)
    assert info.has_audio and not info.has_video
    out = normalize(info, tmp_path / "audio.m4a", settings=cfg)
    assert out.has_audio is True
    assert out.has_video is False
    assert out.duration == pytest.approx(1.0, abs=0.3)


def test_normalize_rejects_a_stream_less_source(cfg: Settings, tmp_path: Path) -> None:
    empty = tmp_path / "nothing.mp4"
    empty.write_bytes(b"")
    info = MediaInfo(path=empty, has_video=False, has_audio=False)
    with pytest.raises(IngestError):
        normalize(info, tmp_path / "out.mp4", settings=cfg)


# --------------------------------------------------------------------------- #
# cache robustness
# --------------------------------------------------------------------------- #

def test_download_refetches_after_a_crashed_download_left_a_fragment(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slot with no sidecar is a crashed attempt, not a cache hit.

    Without this the leftover fragment is served forever and every later call
    dies in ffprobe with no way to recover short of deleting the cache by hand.
    """
    slot = cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)
    slot.mkdir(parents=True)
    fragment = slot / "clip-abc123.f137.mp4"
    fragment.write_bytes(b"\x00" * 4096)          # a truncated, unprobeable fragment

    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)

    info = download(URL, tmp_path / "dl", settings=cfg)
    assert fake.calls == [URL], "the poisoned slot was served instead of re-fetching"
    assert info.has_video and (info.width, info.height) == (320, 240)
    assert not fragment.exists(), "the stale fragment must be cleared, not left to win on size"
    assert (slot / "source.json").is_file()


def test_download_refetches_when_the_sidecar_is_corrupt(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)
    download(URL, tmp_path / "one", settings=cfg)

    slot = cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)
    (slot / "source.json").write_text("{not json", encoding="utf-8")

    download(URL, tmp_path / "two", settings=cfg)
    assert fake.calls == [URL, URL]
    assert json.loads((slot / "source.json").read_text(encoding="utf-8"))["url"] == URL


def test_download_refetches_when_the_cached_file_was_deleted(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)
    download(URL, tmp_path / "one", settings=cfg)

    slot = cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)
    media = slot / json.loads((slot / "source.json").read_text(encoding="utf-8"))["file"]
    media.unlink()

    info = download(URL, tmp_path / "two", settings=cfg)
    assert fake.calls == [URL, URL]
    assert info.path.is_file()


def test_download_offline_rejects_an_incomplete_cache_slot(
    cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    slot = cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)
    slot.mkdir(parents=True)
    (slot / "half.mp4").write_bytes(b"\x00" * 512)
    monkeypatch.setattr(
        "aiclipper.ingest._fetch",
        lambda *a, **k: pytest.fail("offline mode must not fetch"),
    )
    cfg.offline = True
    with pytest.raises(IngestError) as excinfo:
        download(URL, tmp_path / "dl", settings=cfg)
    assert "offline" in str(excinfo.value).lower()


def test_download_copies_when_hardlinking_is_not_available(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """dest_dir is often on a different filesystem than the cache."""
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)

    def no_link(*args: Any, **kwargs: Any) -> None:
        raise OSError(18, "Invalid cross-device link")

    monkeypatch.setattr("aiclipper.ingest.os.link", no_link)

    dest = tmp_path / "dl"
    info = download(URL, dest, settings=cfg)
    assert info.path.is_file() and info.path.parent == dest
    assert (info.width, info.height) == (320, 240)

    slot = cfg.cache_dir / cache_key(URL, DEFAULT_QUALITY)
    cached = slot / json.loads((slot / "source.json").read_text(encoding="utf-8"))["file"]
    assert cached.is_file()
    assert not info.path.samefile(cached), "expected an independent copy, not a link"
    assert info.path.read_bytes() == cached.read_bytes()


def test_download_replaces_a_stale_same_sized_file_in_dest(
    sample_video: Path, silent_video: Path, cfg: Settings, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-existing file with the right name must not be trusted on size alone."""
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)

    dest = tmp_path / "dl"
    dest.mkdir()
    decoy = dest / "fetched.mp4"
    decoy.write_bytes(b"\x00" * sample_video.stat().st_size)   # same size, junk content

    info = download(URL, dest, settings=cfg)
    assert info.path == decoy
    assert info.path.read_bytes() != b"\x00" * sample_video.stat().st_size
    assert (info.width, info.height) == (320, 240)


def test_cache_slot_layout_is_deterministic_across_settings_objects(
    sample_video: Path, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second Settings pointed at the same work_dir sees the same cache."""
    fake = FakeFetch(sample_video)
    monkeypatch.setattr("aiclipper.ingest._fetch", fake)
    download(URL, tmp_path / "a", settings=cfg)

    twin = Settings(work_dir=cfg.work_dir, output_dir=cfg.output_dir,
                    assets_dir=cfg.assets_dir, offline=True)
    info = download(URL, tmp_path / "b", settings=twin)   # offline: cache only
    assert fake.calls == [URL]
    assert info.title == "Fixture Clip"


# --------------------------------------------------------------------------- #
# the real yt-dlp backends, driven against a loopback http server
# --------------------------------------------------------------------------- #

def _no_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep yt-dlp off the sandbox's outbound proxy for loopback requests."""
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")


def test_download_uses_yt_dlp_and_then_the_cache(
    local_server: LocalServer, cfg: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end through the real yt-dlp module -- no external network involved."""
    pytest.importorskip("yt_dlp")
    _no_proxy(monkeypatch)

    before = local_server.hits
    info = download(local_server.url, tmp_path / "dl", settings=cfg, quality="best")
    assert local_server.hits > before
    assert info.path.is_file() and info.path.parent == tmp_path / "dl"
    assert (info.width, info.height) == (320, 240)
    assert info.has_video and info.has_audio
    assert info.source_url == local_server.url
    assert info.title

    sidecar = cfg.cache_dir / cache_key(local_server.url, "best") / "source.json"
    assert json.loads(sidecar.read_text(encoding="utf-8"))["url"] == local_server.url

    after = local_server.hits
    again = download(local_server.url, tmp_path / "dl2", settings=cfg, quality="best")
    assert local_server.hits == after, "cache hit issued an http request"
    assert (again.width, again.height) == (320, 240)


def test_cli_backend_downloads_and_reports_its_path(
    local_server: LocalServer, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")
    if not Path(exe).is_file():
        pytest.skip("yt-dlp CLI not installed")
    _no_proxy(monkeypatch)

    path, meta = _fetch_with_cli(exe, local_server.url, tmp_path / "cli", "best")
    assert path.is_file() and path.parent == tmp_path / "cli"
    assert meta["title"]
    assert ffmpeg.probe(path).width == 320


# --------------------------------------------------------------------------- #
# import hygiene + the (skipped) live path
# --------------------------------------------------------------------------- #

def test_module_imports_without_pulling_in_yt_dlp() -> None:
    code = "import sys, aiclipper.ingest; print('yt_dlp' in sys.modules)"
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False"


def test_live_download(cfg: Settings, tmp_path: Path) -> None:
    """Real network fetch -- opt-in only; skipped in CI and in this sandbox."""
    if os.environ.get("AICLIP_LIVE_TESTS") != "1":
        pytest.skip("live-network test disabled (set AICLIP_LIVE_TESTS=1 to run)")
    import socket

    try:
        socket.create_connection(("www.youtube.com", 443), timeout=3).close()
    except OSError as exc:  # pragma: no cover - network dependent
        pytest.skip(f"no network access: {exc}")
    url = "https://www.youtube.com/watch?v=aqz-KE-bpKQ"
    info = download(url, tmp_path / "live", settings=cfg, quality="worst")
    assert info.has_video and info.duration > 0
