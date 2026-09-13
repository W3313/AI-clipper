"""Tests for :mod:`aiclipper.assets` -- library scanning and procedural placeholders."""

from __future__ import annotations

import array
import itertools
import json
import os
import wave
from pathlib import Path

import pytest

from aiclipper import assets, config
from aiclipper import ffmpeg as ff
from aiclipper.errors import AssetError

needs_ffmpeg = pytest.mark.skipif(not ff.have_ffmpeg(), reason="ffmpeg/ffprobe not on PATH")

# Deliberately tiny: the placeholder graphs are exercised for real, just at a
# size where the whole file stays fast.
SMALL = dict(width=180, height=320, fps=12, background_seconds=0.5, music_seconds=0.5)


@pytest.fixture
def settings(tmp_path, monkeypatch):
    """Settings pointed at a throwaway asset tree."""
    monkeypatch.setenv("AICLIP_ASSETS_DIR", str(tmp_path / "assets"))
    monkeypatch.setenv("AICLIP_SEED", "1234")
    config.reset_settings()
    yield config.get_settings()
    config.reset_settings()


def _make_clip(path: Path, seconds: float = 0.5) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    ff.run_ffmpeg([
        "-y", "-f", "lavfi", "-i", f"testsrc2=s=64x112:r=10:d={seconds}",
        "-t", f"{seconds}", "-c:v", "libx264", "-preset", "ultrafast",
        "-pix_fmt", "yuv420p", str(path),
    ])
    return path


def _small_placeholders(s):
    return assets._generate_placeholders(s, **SMALL)


def _patch_small_generation(monkeypatch, **overrides):
    """Make the *internal* generation call made by ``pick_*`` render at a tiny size."""
    real = assets._generate_placeholders
    small = {**SMALL, **overrides}

    def _tiny(s=None, **kwargs):
        kwargs.update(small)
        return real(s, **kwargs)

    monkeypatch.setattr(assets, "_generate_placeholders", _tiny)


def _thumbnail_frames(path: Path, s) -> list[bytes]:
    """Decode a clip down to 4x4 rgb24 frames -- enough to compare colour and motion."""
    raw = path.with_suffix(".raw")
    ff.run_ffmpeg([
        "-y", "-i", str(path), "-vf", "scale=4:4:flags=area",
        "-pix_fmt", "rgb24", "-f", "rawvideo", str(raw),
    ], settings=s)
    data = raw.read_bytes()
    stride = 4 * 4 * 3
    return [data[i * stride:(i + 1) * stride] for i in range(len(data) // stride)]


def _mean_rgb(frame: bytes) -> tuple[float, float, float]:
    return tuple(sum(frame[i::3]) / len(frame[i::3]) for i in range(3))  # type: ignore[return-value]


def _pcm(path: Path, s) -> array.array:
    wav = path.with_suffix(".probe.wav")
    ff.run_ffmpeg([
        "-y", "-i", str(path), "-ac", "1", "-ar", "8000", "-c:a", "pcm_s16le", str(wav),
    ], settings=s)
    with wave.open(str(wav)) as handle:
        samples = array.array("h")
        samples.frombytes(handle.readframes(handle.getnframes()))
    return samples


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #

def test_empty_library_is_empty_and_writes_no_manifest(settings):
    assert assets.library(settings) == []
    assert not (settings.assets_dir / assets.MANIFEST_NAME).exists()


@needs_ffmpeg
def test_library_catalogues_user_files_and_infers_tags(settings):
    _make_clip(settings.backgrounds_dir / "night-drive_city.mp4")
    _make_clip(settings.backgrounds_dir / "nested" / "slow_pan.mp4")
    ff.make_tone(0.4, settings.music_dir / "lofi_rain.wav", settings=settings)

    found = assets.library(settings)
    by_name = {a.name: a for a in found}

    assert set(by_name) == {"night-drive_city", "slow_pan", "lofi_rain"}
    assert by_name["night-drive_city"].kind == "background"
    assert by_name["lofi_rain"].kind == "music"
    assert by_name["night-drive_city"].tags == ["night", "drive", "city"]
    assert by_name["lofi_rain"].tags == ["lofi", "rain"]
    assert by_name["night-drive_city"].duration == pytest.approx(0.5, abs=0.15)
    assert by_name["lofi_rain"].duration == pytest.approx(0.4, abs=0.15)
    # nested files are found too, and kind follows the directory not the name
    assert by_name["slow_pan"].path.parent.name == "nested"
    # stray non-media files are ignored
    (settings.backgrounds_dir / "notes.txt").write_text("hi", encoding="utf-8")
    assert {a.name for a in assets.library(settings)} == set(by_name)


@needs_ffmpeg
def test_manifest_round_trip_supplies_tags_and_cached_durations(settings, monkeypatch):
    _make_clip(settings.backgrounds_dir / "loop_one.mp4")
    assets.library(settings)

    manifest = settings.assets_dir / assets.MANIFEST_NAME
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["version"] == assets.MANIFEST_VERSION
    entry = next(e for e in data["assets"] if e["name"] == "loop_one")
    assert entry["kind"] == "background"
    assert entry["file"] == "backgrounds/loop_one.mp4"
    assert entry["duration"] > 0

    # hand-edited tags in the manifest win over filename inference ...
    entry["tags"] = ["moody", "handpicked"]
    manifest.write_text(json.dumps(data), encoding="utf-8")

    # ... and the cached duration is reused instead of re-probing.
    def _boom(*args, **kwargs):
        raise AssertionError("library() should not re-probe an unchanged file")

    monkeypatch.setattr(assets.ff, "probe_duration", _boom)
    again = assets.library(settings)
    assert [a.tags for a in again] == [["moody", "handpicked"]]
    assert again[0].duration == pytest.approx(entry["duration"])

    # rewriting the file invalidates the cache entry
    monkeypatch.undo()
    _make_clip(settings.backgrounds_dir / "loop_one.mp4", seconds=1.0)
    refreshed = assets.library(settings)
    assert refreshed[0].duration == pytest.approx(1.0, abs=0.2)
    assert refreshed[0].tags == ["moody", "handpicked"]


# --------------------------------------------------------------------------- #
# placeholder generation
# --------------------------------------------------------------------------- #

@needs_ffmpeg
def test_generated_placeholders_are_real_playable_media(settings):
    made = _small_placeholders(settings)

    backgrounds = [a for a in made if a.kind == "background"]
    music = [a for a in made if a.kind == "music"]
    assert len(backgrounds) >= 3
    assert len(music) >= 2

    for asset in backgrounds:
        info = ff.probe(asset.path, settings=settings)
        assert info.has_video and not info.has_audio
        assert (info.width, info.height) == (SMALL["width"], SMALL["height"])
        assert info.duration == pytest.approx(SMALL["background_seconds"], abs=0.2)
        assert asset.duration == pytest.approx(SMALL["background_seconds"], abs=0.2)
        assert {"placeholder", "abstract", "loop"} <= set(asset.tags)

    for asset in music:
        info = ff.probe(asset.path, settings=settings)
        assert info.has_audio and not info.has_video
        assert info.duration == pytest.approx(SMALL["music_seconds"], abs=0.2)
        assert {"placeholder", "ambient", "calm"} <= set(asset.tags)

    # every generated asset is visible through the normal library scan
    assert {(a.kind, a.name) for a in assets.library(settings)} == {(a.kind, a.name) for a in made}


@needs_ffmpeg
def test_placeholder_graphs_are_distinct(settings):
    """Each background comes from a different lavfi source, not one preset."""
    made = _small_placeholders(settings)
    sizes = {a.name: a.path.stat().st_size for a in made if a.kind == "background"}
    assert len(sizes) >= 3
    # different synthesis chains compress very differently; identical sizes
    # would mean we rendered the same thing more than once
    assert len(set(sizes.values())) == len(sizes)


@needs_ffmpeg
def test_ensure_placeholders_is_idempotent(settings, monkeypatch):
    first = _small_placeholders(settings)
    stamps = {a.path: (a.path.stat().st_mtime_ns, a.path.stat().st_size) for a in first}

    def _boom(*args, **kwargs):
        raise AssertionError("existing placeholders must not be re-rendered")

    monkeypatch.setattr(assets.ff, "run_ffmpeg", _boom)
    second = _small_placeholders(settings)

    assert {a.name for a in second} == {a.name for a in first}
    for asset in second:
        assert (asset.path.stat().st_mtime_ns, asset.path.stat().st_size) == stamps[asset.path]


@needs_ffmpeg
def test_placeholders_use_the_declared_canvas_at_full_width(settings):
    """The shipped defaults really are 1080x1920 h264 at 30fps (short duration here)."""
    made = assets._generate_placeholders(
        settings,
        width=assets.PLACEHOLDER_WIDTH,
        height=assets.PLACEHOLDER_HEIGHT,
        fps=assets.PLACEHOLDER_FPS,
        background_seconds=1.0,
        music_seconds=0.5,
    )
    background = next(a for a in made if a.kind == "background")
    info = ff.probe(background.path, settings=settings)
    assert (info.width, info.height) == (1080, 1920)
    assert info.fps == pytest.approx(30.0, abs=0.1)
    assert (assets.PLACEHOLDER_BACKGROUND_SECONDS, assets.PLACEHOLDER_MUSIC_SECONDS) == (20.0, 30.0)


@pytest.mark.skipif(
    not os.environ.get("AICLIP_SLOW_ASSET_TESTS"),
    reason="set AICLIP_SLOW_ASSET_TESTS=1 to render the full 20s/30s placeholder set",
)
@needs_ffmpeg
def test_full_placeholder_set(settings):  # pragma: no cover - opt-in
    made = assets.ensure_placeholders(settings)
    assert len([a for a in made if a.kind == "background"]) >= 3
    assert len([a for a in made if a.kind == "music"]) >= 2
    for asset in made:
        info = ff.probe(asset.path, settings=settings)
        expected = 20.0 if asset.kind == "background" else 30.0
        assert info.duration == pytest.approx(expected, abs=0.3)
        if asset.kind == "background":
            assert (info.width, info.height) == (1080, 1920)
            assert info.fps == pytest.approx(30.0, abs=0.1)
            assert info.has_video and not info.has_audio
        else:
            assert info.has_audio and not info.has_video


def test_generation_failure_becomes_asset_error(settings, monkeypatch):
    def _missing(*args, **kwargs):
        raise ff.FFmpegMissing("no ffmpeg here")

    monkeypatch.setattr(assets.ff, "run_ffmpeg", _missing)
    with pytest.raises(AssetError):
        assets.ensure_placeholders(settings)
    with pytest.raises(AssetError):
        assets.pick_background(settings=settings)


# --------------------------------------------------------------------------- #
# picking
# --------------------------------------------------------------------------- #

@needs_ffmpeg
def test_pick_by_exact_name(settings):
    _small_placeholders(settings)
    assert assets.pick_background("cell_bloom", settings=settings).name == "cell_bloom"
    assert assets.pick_music("soft_pulse", settings=settings).name == "soft_pulse"


@needs_ffmpeg
def test_pick_unknown_name_raises(settings):
    _small_placeholders(settings)
    with pytest.raises(AssetError) as excinfo:
        assets.pick_background("does_not_exist", settings=settings)
    assert "cell_bloom" in str(excinfo.value)  # the message lists what is available


@needs_ffmpeg
def test_pick_by_tag(settings):
    _small_placeholders(settings)
    assert assets.pick_background(tags=["fractal"], settings=settings).name == "violet_fractal"
    assert assets.pick_background(tags=["cells", "organic"], settings=settings).name == "cell_bloom"
    assert assets.pick_music(tags=["pulse"], settings=settings).name == "soft_pulse"
    # a name that is not an asset but is a tag still resolves
    assert assets.pick_background("aurora", settings=settings).name == "aurora_spiral"
    # unmatched tags degrade to a normal seeded choice instead of failing
    assert assets.pick_background(tags=["no-such-tag"], settings=settings).kind == "background"


@needs_ffmpeg
def test_pick_is_deterministic_per_seed(settings):
    _small_placeholders(settings)
    baseline = assets.pick_background(settings=settings).name
    assert all(assets.pick_background(settings=settings).name == baseline for _ in range(5))
    assert assets.pick_background(settings=settings, seed=settings.seed).name == baseline

    picks = {assets.pick_background(settings=settings, seed=n).name for n in range(12)}
    assert len(picks) > 1  # the seed actually steers the choice
    assert picks <= {a.name for a in assets.library(settings) if a.kind == "background"}


@needs_ffmpeg
def test_pick_accepts_an_explicit_path(settings, tmp_path):
    outside = _make_clip(tmp_path / "elsewhere" / "my-own-clip.mp4")
    asset = assets.pick_background(str(outside), settings=settings)
    assert asset.path == outside
    assert asset.name == "my-own-clip"
    assert asset.tags == ["my", "own", "clip"]
    assert asset.duration > 0
    # nothing was imported into the library
    assert assets.library(settings) == []


@needs_ffmpeg
def test_pick_generates_placeholders_when_library_is_empty(settings, monkeypatch):
    _patch_small_generation(monkeypatch)
    assert assets.library(settings) == []
    picked = assets.pick_background(settings=settings)
    assert picked.kind == "background"
    assert picked.path.is_file()
    assert "placeholder" in picked.tags
    assert assets.pick_music(settings=settings).kind == "music"


@needs_ffmpeg
def test_pick_only_generates_the_missing_kind(settings, monkeypatch):
    """A user with their own backgrounds must not get four loops dumped on them."""
    _make_clip(settings.backgrounds_dir / "my_own_bg.mp4")
    _patch_small_generation(monkeypatch)

    music = assets.pick_music(settings=settings)
    assert music.kind == "music" and music.path.is_file()
    # the background folder was left exactly as the user had it
    assert sorted(p.name for p in settings.backgrounds_dir.iterdir()) == ["my_own_bg.mp4"]
    assert assets.pick_background(settings=settings).name == "my_own_bg"


@needs_ffmpeg
def test_pick_music_accepts_an_explicit_path(settings, tmp_path):
    bed = ff.make_tone(0.3, tmp_path / "outside" / "my-bed.wav", settings=settings)
    asset = assets.pick_music(bed, settings=settings)  # a Path, not a str
    assert asset.path == bed and asset.kind == "music" and asset.is_audio
    assert asset.duration > 0


@needs_ffmpeg
def test_asset_score_and_flags(settings):
    made = _small_placeholders(settings)
    background = next(a for a in made if a.name == "cell_bloom")
    music = next(a for a in made if a.name == "ambient_glow")
    assert background.is_video and not background.is_audio and not background.is_image
    assert music.is_audio
    assert background.score(["cells", "TEAL", "nope"]) == 2
    assert background.score([]) == 0


# --------------------------------------------------------------------------- #
# scanning edge cases
# --------------------------------------------------------------------------- #

@needs_ffmpeg
def test_duplicate_stems_get_distinct_names(settings):
    """Two files called ``dup.mp4`` in different folders must both be reachable."""
    top = _make_clip(settings.backgrounds_dir / "dup.mp4")
    nested = _make_clip(settings.backgrounds_dir / "sub" / "dup.mp4")

    names = {a.name: a.path for a in assets.library(settings)}
    assert set(names) == {"dup", "sub-dup"}
    assert names["dup"] == top and names["sub-dup"] == nested
    assert assets.pick_background("sub-dup", settings=settings).path == nested


@needs_ffmpeg
def test_files_of_the_wrong_kind_are_ignored(settings):
    _make_clip(settings.backgrounds_dir / "real_bg.mp4")
    ff.make_tone(0.3, settings.music_dir / "real_bed.wav", settings=settings)
    # an audio file in backgrounds/ and a video file in music/ are not catalogued
    ff.make_tone(0.3, settings.backgrounds_dir / "stray_tone.wav", settings=settings)
    _make_clip(settings.music_dir / "stray_video.mp4")

    found = {(a.kind, a.name) for a in assets.library(settings)}
    assert found == {("background", "real_bg"), ("music", "real_bed")}


@needs_ffmpeg
def test_still_images_are_valid_backgrounds_and_are_not_reprobed(settings, monkeypatch):
    still = settings.backgrounds_dir / "poster_frame.png"
    still.parent.mkdir(parents=True, exist_ok=True)
    ff.run_ffmpeg(["-y", "-f", "lavfi", "-i", "color=c=0x224466:s=64x112:d=0.1",
                   "-frames:v", "1", str(still)], settings=settings)

    first = assets.library(settings)
    assert [(a.name, a.is_image, a.is_video) for a in first] == [("poster_frame", True, False)]
    assert first[0].duration == 0.0  # a still has no duration, and that is not an error

    def _boom(*args, **kwargs):
        raise AssertionError("an unchanged still image must not be re-probed on every scan")

    monkeypatch.setattr(assets.ff, "probe_duration", _boom)
    assert [a.name for a in assets.library(settings)] == ["poster_frame"]


@needs_ffmpeg
def test_corrupt_manifest_is_tolerated_and_rewritten(settings):
    _make_clip(settings.backgrounds_dir / "solo.mp4")
    manifest = settings.assets_dir / assets.MANIFEST_NAME
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text("{ this is not json", encoding="utf-8")

    found = assets.library(settings)
    assert [a.name for a in found] == ["solo"] and found[0].duration > 0
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert [e["name"] for e in data["assets"]] == ["solo"]


@needs_ffmpeg
def test_manifest_forgets_deleted_files(settings):
    clip = _make_clip(settings.backgrounds_dir / "temporary.mp4")
    _make_clip(settings.backgrounds_dir / "keeper.mp4")
    assets.library(settings)

    clip.unlink()
    assert [a.name for a in assets.library(settings)] == ["keeper"]
    data = json.loads((settings.assets_dir / assets.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert [e["name"] for e in data["assets"]] == ["keeper"]


# --------------------------------------------------------------------------- #
# the placeholders are real artwork, not test cards
# --------------------------------------------------------------------------- #

@needs_ffmpeg
def test_backgrounds_are_visually_distinct_and_actually_move(settings):
    made = [a for a in _small_placeholders(settings) if a.kind == "background"]
    assert len(made) >= 3

    signatures = {}
    for asset in made:
        frames = _thumbnail_frames(asset.path, settings)
        assert len(frames) >= 3
        motion = max(
            sum(abs(x - y) for x, y in zip(frames[0], f, strict=True)) / len(f) for f in frames[1:]
        )
        assert motion > 1.0, f"{asset.name} is a still image, not a loop (motion={motion:.2f})"
        signatures[asset.name] = _mean_rgb(frames[0])

    for (name_a, rgb_a), (name_b, rgb_b) in itertools.combinations(signatures.items(), 2):
        distance = sum(abs(x - y) for x, y in zip(rgb_a, rgb_b, strict=True))
        assert distance > 20.0, f"{name_a} and {name_b} look the same ({distance:.1f})"


@needs_ffmpeg
def test_music_beds_are_audible_and_have_an_envelope(settings):
    made = [a for a in _small_placeholders(settings) if a.kind == "music"]
    assert len(made) >= 2

    for asset in made:
        samples = _pcm(asset.path, settings)
        assert len(samples) > 100
        head = max(abs(v) for v in samples[: len(samples) // 10])
        middle = max(abs(v) for v in samples[len(samples) // 3: 2 * len(samples) // 3])
        assert middle > 2000, f"{asset.name} is near-silent (peak {middle})"
        assert head < middle, f"{asset.name} has no fade-in (head {head} >= middle {middle})"


@needs_ffmpeg
def test_only_the_missing_placeholder_is_regenerated(settings):
    made = _small_placeholders(settings)
    victim = next(a for a in made if a.name == "violet_fractal")
    survivors = {a.path: a.path.stat().st_mtime_ns for a in made if a.path != victim.path}
    victim.path.unlink()

    again = _small_placeholders(settings)
    assert {a.name for a in again} == {a.name for a in made}
    assert victim.path.is_file() and victim.path.stat().st_size > 0
    for path, stamp in survivors.items():
        assert path.stat().st_mtime_ns == stamp, f"{path.name} was pointlessly re-rendered"


@needs_ffmpeg
def test_zero_byte_placeholder_is_replaced(settings):
    """A render killed mid-flight leaves an empty file; it must not be trusted."""
    made = _small_placeholders(settings)
    victim = next(a for a in made if a.kind == "background")
    victim.path.write_bytes(b"")

    again = _small_placeholders(settings)
    restored = next(a for a in again if a.name == victim.name)
    assert restored.path.stat().st_size > 0 and restored.duration > 0


# --------------------------------------------------------------------------- #
# picker argument handling
# --------------------------------------------------------------------------- #

@needs_ffmpeg
def test_pick_accepts_a_path_object(settings, tmp_path):
    clip = _make_clip(tmp_path / "elsewhere" / "hand-picked.mp4")
    asset = assets.pick_background(clip, settings=settings)  # Path, not str
    assert asset.path == clip and asset.name == "hand-picked"


@needs_ffmpeg
def test_tags_may_be_a_bare_string(settings):
    _small_placeholders(settings)
    assert assets.pick_background(tags="fractal", settings=settings).name == "violet_fractal"
    assert assets.pick_music(tags="pulse", settings=settings).name == "soft_pulse"
    asset = next(a for a in assets.library(settings) if a.name == "cell_bloom")
    assert asset.score("cells") == 1
    assert asset.score(["cells", "teal"]) == 2


@needs_ffmpeg
def test_pick_is_seeded_from_settings_not_global_random(settings):
    """Another module reseeding the global RNG must not change what we pick."""
    import random as _random

    _small_placeholders(settings)
    baseline = assets.pick_background(settings=settings).name
    _random.seed(99)
    _random.random()
    assert assets.pick_background(settings=settings).name == baseline
