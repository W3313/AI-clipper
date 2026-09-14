"""The render core: a :class:`~aiclipper.models.Timeline` becomes one ffmpeg run.

Everything the engine produces ends up here.  :func:`build_command` turns a
timeline into a single ``ffmpeg`` invocation built around one
``-filter_complex`` graph; :func:`render` executes it and probes the result.

Design rules that the rest of the package relies on:

* **One process.**  No intermediate files, no concat demuxer -- the whole
  composite (canvas, layers, crops, audio mix, burned subtitles) is expressed
  as a single filter graph.
* **Every pad is labelled and consumed exactly once.**  The video accumulator
  is threaded through the overlays in z order; the audio bus is split with
  ``asplit`` whenever a stream needs to be reused (ducking).
* **Explicit duration.**  The base canvas, every audio chain and the output all
  carry the timeline duration; ``-shortest`` is never used.
* **Paths are escaped** with :func:`aiclipper.ffmpeg.escape_filter_path` before
  they enter the graph.
* **The mix is loudness-normalised.**  A single-pass ``loudnorm`` aimed at
  :data:`DEFAULT_LOUDNESS_TARGET` sits between ``amix`` and the limiter, so
  every workflow exports at roughly the level short-form platforms normalise
  to instead of the -33..-44 LUFS the raw mix lands at.  Programmes shorter
  than :data:`LOUDNESS_MIN_DURATION` skip it (the filter misbehaves there).
  Pass ``loudness_target=None`` -- to :func:`render`/:func:`build_command`, or
  as a ``loudness_target`` attribute on
  :class:`~aiclipper.models.RenderOptions` -- to switch it off and get the
  un-normalised graph back verbatim.

The command returned by :func:`build_command` includes the ffmpeg binary as
element ``0`` so it is directly runnable/loggable; :func:`render` hands
``cmd[1:]`` to :func:`aiclipper.ffmpeg.run_ffmpeg`.

Known limitation: a :class:`~aiclipper.models.VisualLayer` with ``take_audio``
joins the mix as a voice-role source, but :meth:`Timeline.validate` only counts
:class:`~aiclipper.models.AudioTrack` entries when it checks that a ducked
track has something to duck under.  A timeline whose only voice is a layer's
own audio must therefore not set ``duck`` on its music bed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from . import ffmpeg as ff
from .config import Settings, get_settings
from .errors import RenderError
from .models import (
    AudioTrack,
    RenderOptions,
    RenderResult,
    Timeline,
    VisualLayer,
)

log = logging.getLogger(__name__)

__all__ = ["render", "build_command"]


# --------------------------------------------------------------------------- #
# constants
# --------------------------------------------------------------------------- #

SAMPLE_RATE = 48000
CHANNEL_LAYOUT = "stereo"
CHANNELS = 2

_AFORMAT = f"aformat=sample_fmts=fltp:sample_rates={SAMPLE_RATE}:channel_layouts={CHANNEL_LAYOUT}"
_DUCK = "sidechaincompress=threshold=0.05:ratio=8:attack=20:release=300"
_LIMITER = "alimiter=limit=0.95:level=0"
_CROP_INSTANCE = "pan"

#: Integrated loudness the finished file aims for, in LUFS.  Short-form
#: platforms normalise playback to roughly -14 LUFS; anything quieter is turned
#: up (or the viewer reaches for the volume), anything louder is turned down.
DEFAULT_LOUDNESS_TARGET = -14.0

#: Ceiling handed to ``loudnorm``.  It leaves headroom for the limiter that
#: follows, so normalisation can never be the thing that clips.
LOUDNESS_TRUE_PEAK = -1.5

#: Target loudness *range*.  11 LU keeps a narration/music mix lively without
#: letting ``loudnorm`` squash it flat.
LOUDNESS_RANGE = 11.0

#: ``loudnorm`` only accepts a target inside this window.
_LOUDNESS_LIMITS = (-70.0, -5.0)

#: Shortest programme ``loudnorm`` is trusted with, in seconds.  See
#: :func:`_loudnorm`: below this it never establishes its gate and a silent mix
#: comes out of the chain at full scale.
LOUDNESS_MIN_DURATION = 3.0


class _Unset:
    """Sentinel: "take the loudness target from ``options``"."""

    __slots__ = ()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<unset>"


UNSET = _Unset()


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #

def _fmt(value: float) -> str:
    """Compact, stable float formatting for filter arguments."""
    text = f"{float(value):.6f}".rstrip("0").rstrip(".")
    if text in ("", "-0", "-"):
        return "0"
    return text


def _filter_path(path: str | Path) -> str:
    """Escape a path for use as a *filter option value*.

    :func:`aiclipper.ffmpeg.escape_filter_path` escapes for the filtergraph
    parser, which is the right thing for ``,``, ``;`` and ``[]`` -- but that
    parser strips one level of backslashes before the per-filter option parser
    ever sees the string, and ``:`` (the option separator) and ``'`` (the
    option quote) are special to *that* parser too.  Their backslash therefore
    has to survive the first pass, so it gets one of its own.

    Verified against ffmpeg 6.1: without this a ``subtitles=filename=`` for a
    path holding an apostrophe silently loses the apostrophe, and one holding
    a colon swallows every option after it.
    """
    text = ff.escape_filter_path(path)
    return text.replace("\\:", "\\\\:").replace("\\'", "\\\\\\'")


def _color(value: str) -> str:
    """Normalise ``#RRGGBB`` to the ``0xRRGGBB`` form ffmpeg prefers."""
    text = (value or "").strip() or "black"
    if text.startswith("#"):
        return "0x" + text[1:]
    return text


def _loudnorm(target: float) -> str:
    """Single-pass (dynamic) ``loudnorm`` aimed at ``target`` LUFS.

    Single pass means no measurement run and no second ffmpeg process: the
    filter gates and normalises as it goes.  Three properties of that mode are
    load-bearing here, all measured against ffmpeg 6.1:

    * It honours the EBU R128 **absolute gate at -70 LUFS**.  Material below it
      -- digital silence, and anything quiet enough to be inaudible, such as the
      timed silence the offline TTS provider produces -- is passed through
      *unchanged* rather than being lifted 56 dB into a wall of hiss.
    * Its output never exceeds ``TP``, which we keep below the downstream
      limiter's ceiling, so normalisation cannot introduce clipping.
    * **It needs three seconds to get there.**  Given less,
      ``loudnorm,alimiter`` hands back below-gate input at roughly *0 dBFS* --
      a 2s silent mix measures -91 dB through the limiter alone and -0.0 dB
      through the pair.  (Each filter on its own is fine; it is the
      combination, and only under three seconds.)  That is why callers go
      through :data:`LOUDNESS_MIN_DURATION` instead of calling this blindly.
    """
    return f"loudnorm=I={_fmt(target)}:TP={_fmt(LOUDNESS_TRUE_PEAK)}:LRA={_fmt(LOUDNESS_RANGE)}"


def _resolve_loudness(options: RenderOptions, override: float | None | _Unset) -> float | None:
    """Work out the loudness target: explicit argument, then ``options``, then the default.

    ``RenderOptions`` is the contract in :mod:`aiclipper.models`; this reads
    ``loudness_target`` off it when it is there and falls back to
    :data:`DEFAULT_LOUDNESS_TARGET` when it is not, so the renderer behaves the
    same whether or not the field has been declared.  ``None`` -- from either
    place -- disables normalisation and restores the pre-normalisation graph.
    """
    if isinstance(override, _Unset):
        value = getattr(options, "loudness_target", DEFAULT_LOUDNESS_TARGET)
    else:
        value = override
    if value is None:
        return None
    target = float(value)
    low, high = _LOUDNESS_LIMITS
    if not low <= target <= high:
        raise RenderError(
            "loudness target out of range",
            problems=[f"loudness_target={_fmt(target)} LUFS is outside {_fmt(low)}..{_fmt(high)}"],
        )
    return target


@dataclass
class _Input:
    """One ``-i`` block, in the order it will appear on the command line."""

    args: list[str]
    seek: float = 0.0


@dataclass
class _AudioSpec:
    """A normalised audio source: an :class:`AudioTrack` or a layer's own audio."""

    index: int
    seek: float = 0.0
    src_start: float = 0.0
    start: float = 0.0
    end: float | None = None
    gain_db: float = 0.0
    volume: float = 1.0
    loop: bool = False
    fade_in: float = 0.0
    fade_out: float = 0.0
    duck: bool = False
    role: str = "sfx"

    @classmethod
    def from_track(cls, index: int, seek: float, track: AudioTrack) -> _AudioSpec:
        return cls(
            index=index,
            seek=seek,
            src_start=max(0.0, track.src_start),
            start=max(0.0, track.start),
            end=track.end,
            gain_db=track.gain_db,
            loop=track.loop,
            fade_in=max(0.0, track.fade_in),
            fade_out=max(0.0, track.fade_out),
            duck=track.duck,
            role=track.role,
        )

    @classmethod
    def from_layer(cls, index: int, seek: float, layer: VisualLayer) -> _AudioSpec:
        return cls(
            index=index,
            seek=seek,
            src_start=max(0.0, layer.src_start),
            start=max(0.0, layer.start),
            end=layer.end,
            volume=layer.volume,
            loop=layer.loop,
            role="voice",
        )


# --------------------------------------------------------------------------- #
# graph builder
# --------------------------------------------------------------------------- #

class _GraphBuilder:
    """Builds the filter graph and the full argv for one timeline."""

    def __init__(
        self,
        timeline: Timeline,
        out_path: Path,
        options: RenderOptions,
        settings: Settings,
        workdir: Path | None,
        loudness_target: float | None = None,
    ) -> None:
        self.timeline = timeline
        self.out_path = Path(out_path)
        self.options = options
        self.settings = settings
        self.loudness_target = loudness_target
        self._workdir = workdir
        self.inputs: list[_Input] = []
        self.chains: list[str] = []
        self.duration = float(timeline.duration)
        self.fps = int(timeline.fps)
        self.audio_specs: list[_AudioSpec] = []

    # -- infrastructure ---------------------------------------------------- #
    @property
    def workdir(self) -> Path:
        if self._workdir is None:
            self._workdir = Path(self.settings.work_dir) / f"render-{self.out_path.stem or 'timeline'}"
        self._workdir.mkdir(parents=True, exist_ok=True)
        return self._workdir

    def add_input(self, args: list[str], seek: float = 0.0) -> int:
        self.inputs.append(_Input(list(args), seek))
        return len(self.inputs) - 1

    # -- validation -------------------------------------------------------- #
    def validate(self) -> None:
        tl = self.timeline
        problems = list(tl.validate())
        for i, layer in enumerate(tl.visuals):
            if layer.kind in ("video", "image") and layer.src:
                if not Path(layer.src).exists():
                    problems.append(f"visual[{i}] source not found: {layer.src}")
        for i, track in enumerate(tl.audio):
            if track.src and not Path(track.src).exists():
                problems.append(f"audio[{i}] source not found: {track.src}")
        if tl.subtitles is not None:
            ass = Path(tl.subtitles.ass_path)
            if not ass.exists():
                problems.append(f"subtitle file not found: {ass}")
            fonts = tl.subtitles.fonts_dir
            if fonts is not None and not Path(fonts).exists():
                problems.append(f"subtitle fonts dir not found: {fonts}")
        if problems:
            raise RenderError("timeline cannot be rendered", problems=problems)

    # -- video ------------------------------------------------------------- #
    def build_video(self) -> None:
        tl = self.timeline
        self.chains.append(
            f"color=c={_color(tl.background)}:s={tl.width}x{tl.height}:"
            f"r={self.fps}:d={_fmt(self.duration)}[base]"
        )
        current = "base"
        for n, layer in enumerate(tl.ordered_visuals):
            label = self._layer_chain(n, layer)
            x, y, _, _ = layer.rect(tl.width, tl.height)
            opts = [f"x={x}", f"y={y}"]
            enable = self._enable(layer)
            if enable:
                opts.append(enable)
            opts += ["shortest=0", "eof_action=pass"]
            nxt = f"c{n}"
            self.chains.append(f"[{current}][{label}]overlay={':'.join(opts)}[{nxt}]")
            current = nxt

        tail: list[str] = []
        if tl.subtitles is not None:
            args = [f"filename={_filter_path(tl.subtitles.ass_path)}"]
            if tl.subtitles.fonts_dir is not None:
                args.append(f"fontsdir={_filter_path(tl.subtitles.fonts_dir)}")
            tail.append("subtitles=" + ":".join(args))
        tail.append(f"format={self.options.pix_fmt}")
        self.chains.append(f"[{current}]{','.join(tail)}[vout]")

    def _enable(self, layer: VisualLayer) -> str:
        start = max(0.0, layer.start)
        if start <= 0.0 and layer.end is None:
            return ""
        end = self.duration if layer.end is None else float(layer.end)
        return f"enable='between(t,{_fmt(start)},{_fmt(end)})'"

    def _layer_chain(self, n: int, layer: VisualLayer) -> str:
        """Append one layer's chain and return the label it produces."""
        tl = self.timeline
        _, _, rect_w, rect_h = layer.rect(tl.width, tl.height)
        label = f"v{n}"
        filters: list[str] = []
        rgba = False

        if layer.kind == "color":
            head = (
                f"color=c={_color(layer.color)}:s={rect_w}x{rect_h}:"
                f"r={self.fps}:d={_fmt(self.duration)},"
            )
        else:
            src = str(layer.src)
            args: list[str] = []
            seek = 0.0
            if layer.kind == "image":
                args += ["-loop", "1"]
            elif layer.loop:
                args += ["-stream_loop", "-1"]
            elif layer.src_start > 0:
                # Fast, accurate input-level seek; only safe when we are not looping.
                seek = float(layer.src_start)
                args += ["-ss", _fmt(seek)]
            args += ["-i", src]
            index = self.add_input(args, seek)
            head = f"[{index}:v]"

            if layer.kind == "video":
                remaining = max(0.0, float(layer.src_start) - seek)
                if remaining > 0:
                    filters.append(f"trim=start={_fmt(remaining)}")
                filters.append("setpts=PTS-STARTPTS")
            filters.append(f"fps={self.fps}")

            filters += self._crop_filters(n, layer)
            filters += _fit_filters(layer.fit, rect_w, rect_h)
            if layer.fit == "contain":
                rgba = True
            filters.append("setsar=1")

            if layer.kind == "video" and layer.start > 0:
                if not rgba:
                    filters.append("format=rgba")
                    rgba = True
                filters.append(
                    f"tpad=start_duration={_fmt(layer.start)}:start_mode=add:color=black@0"
                )

        if layer.opacity < 1.0:
            if not rgba:
                filters.append("format=rgba")
                rgba = True
            filters.append(f"colorchannelmixer=aa={_fmt(layer.opacity)}")

        body = ",".join(filters)
        self.chains.append(f"{head}{body}[{label}]" if body else f"{head}null[{label}]")
        return label

    def _crop_filters(self, n: int, layer: VisualLayer) -> list[str]:
        path = layer.crop
        if path is None or not path.keyframes:
            return []
        kfs = sorted(path.keyframes, key=lambda k: k.t)
        first = kfs[0]
        for k in kfs:
            if k.w != first.w or k.h != first.h:
                raise RenderError(
                    "crop path size must stay constant",
                    problems=[
                        f"visual[{n}] ({layer.label or layer.kind}) crop keyframe at t={_fmt(k.t)} "
                        f"is {k.w}x{k.h}, expected {first.w}x{first.h}"
                    ],
                )
        if first.w <= 0 or first.h <= 0:
            raise RenderError(
                "crop path size must be positive",
                problems=[f"visual[{n}] crop size is {first.w}x{first.h}"],
            )
        geometry = f"w={first.w}:h={first.h}:x={first.x}:y={first.y}"
        if path.is_static or len(kfs) == 1:
            return [f"crop={geometry}"]
        # The crop instance needs a unique name: ``sendcmd`` addresses filters by
        # class name unless one is given, which would steer every other crop in
        # the graph (the ``fit=cover`` crops, for instance) as well.
        target = f"crop@{_CROP_INSTANCE}{n}"
        script = self._write_sendcmd(n, layer, kfs, target)
        return [f"sendcmd=f={_filter_path(script)}", f"{target}={geometry}"]

    def _write_sendcmd(self, n: int, layer: VisualLayer, kfs: list, target: str) -> Path:
        offset = max(0.0, float(layer.src_start))
        lines: list[str] = []
        for k in kfs:
            t = max(0.0, float(k.t) - offset)
            lines.append(f"{_fmt(t)} {target} x {int(k.x)}, {target} y {int(k.y)};")
        name = f"crop-{n}-{layer.label or layer.kind}.cmd"
        safe = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in name)
        path = self.workdir / safe
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    # -- audio ------------------------------------------------------------- #
    def collect_audio(self) -> None:
        """Layer audio first (it shares the visual inputs), then the audio tracks."""
        for n, layer in enumerate(self.timeline.ordered_visuals):
            # Only a video input can carry an audio stream: colour layers are
            # lavfi sources and stills decode to a single video stream, so
            # honouring ``take_audio`` on them would reference a pad ffmpeg
            # never creates.
            if not layer.take_audio or layer.kind != "video" or not layer.src:
                continue
            index = self._visual_input_index(n)
            if index is None:
                continue
            self.audio_specs.append(_AudioSpec.from_layer(index, self.inputs[index].seek, layer))
        for track in self.timeline.audio:
            index = self.add_input(["-i", str(track.src)])
            self.audio_specs.append(_AudioSpec.from_track(index, 0.0, track))

    def _visual_input_index(self, n: int) -> int | None:
        """Input index of the n-th ordered visual layer (``None`` for colour layers)."""
        seen = 0
        for i, layer in enumerate(self.timeline.ordered_visuals):
            if layer.kind == "color" or not layer.src:
                continue
            if i == n:
                return seen
            seen += 1
        return None

    def build_audio(self) -> None:
        if not self.audio_specs:
            self.chains.append(
                f"anullsrc=r={SAMPLE_RATE}:cl={CHANNEL_LAYOUT}:d={_fmt(self.duration)},"
                f"{_AFORMAT}[aout]"
            )
            return

        labels = [self._audio_chain(i, spec) for i, spec in enumerate(self.audio_specs)]
        mix = dict(enumerate(labels))

        voices = [i for i, spec in enumerate(self.audio_specs) if spec.role == "voice"]
        ducked = [i for i, spec in enumerate(self.audio_specs) if spec.duck and spec.role != "voice"]

        if ducked and voices:
            keys: list[str] = []
            for i in voices:
                a, b = f"{labels[i]}m", f"{labels[i]}k"
                self.chains.append(f"[{labels[i]}]asplit=2[{a}][{b}]")
                mix[i] = a
                keys.append(b)
            if len(keys) == 1:
                bus = keys[0]
            else:
                bus = "voicebus"
                self.chains.append(
                    "".join(f"[{k}]" for k in keys) + f"amix=inputs={len(keys)}:normalize=0[{bus}]"
                )
            if len(ducked) == 1:
                sidechains = [bus]
            else:
                sidechains = [f"duckkey{j}" for j in range(len(ducked))]
                self.chains.append(
                    f"[{bus}]asplit={len(ducked)}" + "".join(f"[{s}]" for s in sidechains)
                )
            for j, i in enumerate(ducked):
                out = f"{labels[i]}d"
                self.chains.append(f"[{mix[i]}][{sidechains[j]}]{_DUCK}[{out}]")
                mix[i] = out

        inputs = "".join(f"[{mix[i]}]" for i in range(len(labels)))
        tail = [
            "apad",
            f"atrim=duration={_fmt(self.duration)}",
            "asetpts=PTS-STARTPTS",
        ]
        # Normalise the finished mix, then limit: loudnorm sits *between* the
        # mix and the limiter so the limiter is still the last thing to touch
        # the samples and remains the clipping backstop.  Programmes too short
        # for loudnorm to establish its gate are left alone -- see
        # :func:`_loudnorm` -- which is also the floor that keeps a silent
        # timeline silent.
        if self.loudness_target is not None and self.duration >= LOUDNESS_MIN_DURATION:
            tail.append(_loudnorm(self.loudness_target))
        tail += [
            _LIMITER,
            "aresample=async=1",
            _AFORMAT,
        ]
        self.chains.append(
            f"{inputs}amix=inputs={len(labels)}:normalize=0:dropout_transition=0,{','.join(tail)}[aout]"
        )

    def _audio_chain(self, i: int, spec: _AudioSpec) -> str:
        label = f"a{i}"
        end = self.duration if spec.end is None else float(spec.end)
        span = max(0.05, end - spec.start)
        filters: list[str] = [_AFORMAT]

        trim_start = max(0.0, spec.src_start - spec.seek)
        if trim_start > 0:
            filters.append(f"atrim=start={_fmt(trim_start)}")
            filters.append("asetpts=PTS-STARTPTS")
        if spec.loop:
            filters.append(f"aloop=loop=-1:size={int(SAMPLE_RATE * (span + 1.0))}")
        filters.append("apad")
        filters.append(f"atrim=duration={_fmt(span)}")
        filters.append("asetpts=PTS-STARTPTS")

        if abs(spec.gain_db) > 1e-6:
            filters.append(f"volume={_fmt(spec.gain_db)}dB")
        if abs(spec.volume - 1.0) > 1e-6:
            filters.append(f"volume={_fmt(spec.volume)}")
        if spec.fade_in > 0:
            filters.append(f"afade=t=in:st=0:d={_fmt(min(spec.fade_in, span))}")
        if spec.fade_out > 0:
            fade = min(spec.fade_out, span)
            filters.append(f"afade=t=out:st={_fmt(span - fade)}:d={_fmt(fade)}")
        if spec.start > 0:
            ms = int(round(spec.start * 1000))
            filters.append("adelay=" + "|".join([str(ms)] * CHANNELS))

        self.chains.append(f"[{spec.index}:a]{','.join(filters)}[{label}]")
        return label

    # -- assembly ---------------------------------------------------------- #
    def build(self) -> list[str]:
        self.validate()
        self.build_video()
        self.collect_audio()
        self.build_audio()

        opts = self.options
        cmd: list[str] = [ff.ffmpeg_bin(self.settings)]
        if opts.overwrite:
            cmd.append("-y")
        for item in self.inputs:
            cmd += item.args
        cmd += ["-filter_complex", ";".join(self.chains)]
        cmd += ["-map", "[vout]", "-map", "[aout]"]
        cmd += ["-c:v", opts.video_codec, "-preset", opts.preset, "-crf", str(opts.crf)]
        cmd += ["-pix_fmt", opts.pix_fmt]
        cmd += ["-c:a", opts.audio_codec, "-b:a", opts.audio_bitrate]
        cmd += ["-ar", str(SAMPLE_RATE), "-ac", str(CHANNELS)]
        cmd += ["-r", str(self.fps), "-t", _fmt(self.duration)]
        if opts.faststart:
            cmd += ["-movflags", "+faststart"]
        if opts.threads:
            cmd += ["-threads", str(opts.threads)]
        cmd += [str(a) for a in opts.extra_args]
        cmd.append(str(self.out_path))
        return cmd


def _fit_filters(fit: str, w: int, h: int) -> list[str]:
    """Scale (and crop/pad) the source into a ``w`` x ``h`` rectangle."""
    if fit == "stretch":
        return [f"scale={w}:{h}"]
    if fit == "contain":
        return [
            f"scale={w}:{h}:force_original_aspect_ratio=decrease",
            "format=rgba",
            f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2:color=black@0",
        ]
    return [
        f"scale={w}:{h}:force_original_aspect_ratio=increase",
        f"crop={w}:{h}",
    ]


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #

def _check_layer_audio(timeline: Timeline, settings: Settings) -> None:
    """Fail loudly when a ``take_audio`` layer has no audio stream to take.

    ``build_command`` stays offline and trusts the flag -- it must work on a
    timeline whose sources are not decodable yet, and ``dry_run`` must never
    shell out.  Just before ffmpeg actually runs we can afford one probe per
    flagged layer, which turns an opaque ffmpeg "invalid stream specifier"
    exit into a problem list the caller can read.
    """
    problems: list[str] = []
    for i, layer in enumerate(timeline.visuals):
        if not layer.take_audio or layer.kind != "video" or not layer.src:
            continue
        try:
            info = ff.probe(layer.src, settings=settings)
        except Exception as exc:  # probing is a courtesy; never block a render on it
            log.warning("could not probe take_audio source %s: %s", layer.src, exc)
            continue
        if not info.has_audio:
            problems.append(
                f"visual[{i}] ({layer.label or layer.kind}) sets take_audio but "
                f"{layer.src} has no audio stream"
            )
    if problems:
        raise RenderError("timeline cannot be rendered", problems=problems)


def build_command(
    timeline: Timeline,
    out_path: Path,
    *,
    options: RenderOptions | None = None,
    settings: Settings | None = None,
    workdir: Path | None = None,
    loudness_target: float | None | _Unset = UNSET,
) -> list[str]:
    """Return the complete ffmpeg argv (``argv[0]`` is the ffmpeg binary).

    Raises :class:`~aiclipper.errors.RenderError` when the timeline does not
    validate, when a referenced file is missing, or when an animated crop path
    changes size between keyframes.  Any ``sendcmd`` scripts needed by animated
    crops are written into ``workdir`` (defaults to a directory under
    ``settings.work_dir``) as a side effect -- including under ``dry_run``,
    since the returned command is only runnable if those scripts exist.

    No media is decoded here: sources are checked for existence, never probed,
    so a command can be built for files ffmpeg has not seen yet.

    ``loudness_target`` overrides the integrated-loudness target in LUFS for
    this call; left alone it comes from ``options.loudness_target`` when that
    field exists and from :data:`DEFAULT_LOUDNESS_TARGET` otherwise, and
    ``None`` disables normalisation entirely.
    """
    settings = settings or get_settings()
    options = options or RenderOptions()
    builder = _GraphBuilder(
        timeline,
        Path(out_path),
        options,
        settings,
        Path(workdir) if workdir is not None else None,
        _resolve_loudness(options, loudness_target),
    )
    return builder.build()


def render(
    timeline: Timeline,
    out_path: Path,
    *,
    options: RenderOptions | None = None,
    settings: Settings | None = None,
    log_path: Path | None = None,
    dry_run: bool = False,
    loudness_target: float | None | _Unset = UNSET,
) -> RenderResult:
    """Render ``timeline`` to ``out_path`` in a single ffmpeg invocation.

    With ``dry_run`` the command is built (and any ``sendcmd`` script written)
    but ffmpeg is never started; the returned :class:`RenderResult` carries the
    timeline's own geometry.  Otherwise the output is probed and the result
    reports what actually landed on disk.

    ``loudness_target`` is forwarded to :func:`build_command`; ``None``
    disables loudness normalisation.
    """
    settings = settings or get_settings()
    options = options or RenderOptions()
    out = Path(out_path)
    workdir = out.parent / f".{out.stem or 'timeline'}-render"
    cmd = build_command(
        timeline,
        out,
        options=options,
        settings=settings,
        workdir=workdir,
        loudness_target=loudness_target,
    )

    if dry_run:
        return RenderResult(
            path=out,
            duration=float(timeline.duration),
            width=timeline.width,
            height=timeline.height,
            fps=float(timeline.fps),
            command=cmd,
        )

    _check_layer_audio(timeline, settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    ff.run_ffmpeg(cmd[1:], settings=settings, log_path=log_path)

    result = RenderResult(
        path=out,
        duration=float(timeline.duration),
        width=timeline.width,
        height=timeline.height,
        fps=float(timeline.fps),
        command=cmd,
        size_bytes=out.stat().st_size if out.exists() else 0,
    )
    try:
        info = ff.probe(out, settings=settings)
    except Exception as exc:  # pragma: no cover - a probe failure is informational
        log.warning("could not probe rendered output %s: %s", out, exc)
        return result
    result.duration = info.duration or result.duration
    result.width = info.width or result.width
    result.height = info.height or result.height
    result.fps = info.fps or result.fps
    result.size_bytes = info.size_bytes or result.size_bytes
    return result
