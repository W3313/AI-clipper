# Pipelines and CLI (phase two)

The nine core modules are the parts. Pipelines are the assembly: each one takes a
user request, drives the modules in order, builds a `Timeline`, and hands it to
`render.render`. Each returns a `ProjectResult`.

Read `docs/ARCHITECTURE.md` first -- its hard rules apply here too. Read the real
signatures in the modules before wiring them; the spec below is the shape, the
source is the truth.

## `pipelines/common.py`

Shared helpers every pipeline needs. Keep pipeline modules thin by putting the
reusable work here.

```python
def concat_audio(parts: Sequence[Path], out_path: Path, *, gap: float = 0.0,
                 settings=None) -> tuple[Path, list[float]]
```
Concatenate narration parts with `gap` seconds of silence between them, using the
ffmpeg concat filter. Returns the joined file and the start offset of each part,
so callers can place overlays and captions on the same clock.

```python
def narration_words(results: Sequence[TTSResult], offsets: Sequence[float],
                    *, audio: Path | None = None, text: str = "",
                    settings=None) -> list[Word]
```
Collect word timings from the TTS results, shifting each by its offset. When a
provider returned no words and `audio`+`text` are given, fall back to
`transcribe.align`; when that is unavailable, fall back to proportional timings
so captions still animate. Never return an empty list for non-empty text.

```python
def background_layer(asset: Asset, *, width: int, height: int, duration: float,
                     z: int = 0, seed: int | None = None) -> VisualLayer
def music_track(asset: Asset, *, duration: float, gain_db: float = -18.0,
                duck: bool = True) -> AudioTrack
def voice_track(path: Path, *, start: float = 0.0, gain_db: float = 0.0) -> AudioTrack
def caption_track(words, out_dir: Path, *, style, width, height) -> SubtitleTrack | None
def safe_stem(text: str, fallback: str = "video") -> str
def resolve_output(out: Path | None, stem: str, settings=None) -> Path
```
`background_layer` starts the loop at a seeded random offset inside the asset so
repeated runs do not all open on the same frame.

## `pipelines/clip.py` -- long video to shorts

```python
def run(source: str | Path, *, count: int = 3, min_duration: float = 15.0,
        max_duration: float = 60.0, style: str = "clean", reframe: bool = True,
        captions: bool = True, out_dir: Path | None = None,
        settings=None, provider=None) -> list[ProjectResult]
```
1. `ingest.resolve(source)`
2. `transcribe.transcribe(media)` (cached)
3. `highlight.select(transcript, count=..., min_duration=..., max_duration=...)`
4. per candidate: `crop.track(media, start=c.start, end=c.end)` when `reframe`
   and the source is not already vertical, else `crop.center_crop`
5. captions from `transcript.slice(c.start, c.end)` (rebased -- that is what the
   `rebase` argument is for)
6. `Timeline` with one video layer (`src_start=c.start`, `take_audio=True`,
   `crop=path`, `fit="cover"`) plus the subtitle track
7. `render.render` to `<stem>-01-<slug>.mp4`

Each `ProjectResult` carries the candidate's title, score and source window in
`metadata`. On a vertical source, skip reframing entirely.

## `pipelines/story.py` -- topic to narrated short

```python
def run(topic: str | None = None, *, script: str | VideoScript | None = None,
        seconds: int = 35, voice: str = "", background: str | None = None,
        music: str | None = None, style: str = "bold_yellow", captions: bool = True,
        out_path: Path | None = None, settings=None, provider=None) -> ProjectResult
```
`scriptgen.write_script` (or `parse_script` when the user supplied one) ->
`tts.synthesize_lines` -> `concat_audio` -> `narration_words` -> captions ->
`assets.pick_background` / `pick_music` (calling `ensure_placeholders` first) ->
timeline: looping background layer, voice track, ducked music track, subtitles.
Duration is the narration length plus a short tail.

## `pipelines/texts.py` -- text-conversation story

```python
def run(topic: str | None = None, *, script: str | ChatScript | None = None,
        theme: str = "classic", voice: str = "", reply_voice: str = "",
        background: str | None = None, music: str | None = None,
        backend: str | None = None, out_path: Path | None = None,
        settings=None, provider=None) -> ProjectResult
```
`scriptgen.write_chat` / `parse_chat` -> synthesise each spoken message, using a
different voice per side -> lay messages on a clock honouring each message's
`delay` and `typing` -> `overlays.render_chat` for the state images -> timeline:
background, one image layer per state with `start`/`end` covering that state's
span, one voice track per message at its offset, ducked music. No burned captions
by default (the bubbles are the text).

## `pipelines/reddit.py` -- forum story

```python
def run(topic: str | None = None, *, post: RedditPost | None = None,
        theme: str = "dark", voice: str = "", background: str | None = None,
        music: str | None = None, style: str = "clean", card_seconds: float | None = None,
        out_path: Path | None = None, settings=None, provider=None) -> ProjectResult
```
`scriptgen.write_reddit` -> narrate title, then body -> `overlays.render_forum_card`
shown for the title read (or `card_seconds`) -> timeline: background, card image
layer for its span, voice, ducked music, captions over the body only.

## `pipelines/split.py` -- stacked two-pane

```python
def run(top: str | Path, *, bottom: str | Path | None = None,
        narration: str | None = None, voice: str = "", style: str = "clean",
        music: str | None = None, seconds: float | None = None,
        out_path: Path | None = None, settings=None, provider=None) -> ProjectResult
```
Top pane is the supplied clip (reframed to fill the top half), bottom pane is a
second clip or a library background. Optional narration replaces the top pane's
audio. Panes are exact halves of the canvas with no gap, each `fit="cover"`.

## `cli.py`

`argparse` with subcommands. No third-party CLI library.

```
aiclip clip <source> [--count N] [--min S] [--max S] [--style NAME] [--no-reframe]
                     [--no-captions] [--out DIR]
aiclip story [--topic TEXT | --script FILE|-] [--seconds N] [--voice NAME]
             [--background NAME] [--music NAME] [--style NAME] [--out FILE]
aiclip texts [--topic TEXT | --script FILE|-] [--theme NAME] [--voice NAME]
             [--reply-voice NAME] [--background NAME] [--out FILE]
aiclip reddit [--topic TEXT] [--theme NAME] [--voice NAME] [--style NAME] [--out FILE]
aiclip split <top> [--bottom PATH] [--narration TEXT] [--voice NAME] [--out FILE]
aiclip voices [--provider NAME] [--tag TAG]      # list the voice catalogue
aiclip styles                                     # list caption presets
aiclip assets [--generate]                        # list or generate the library
aiclip doctor                                     # report what is installed and reachable
```
Global flags: `--work-dir`, `--output-dir`, `--offline`, `--seed`, `--width`,
`--height`, `--fps`, `-v/--verbose` (sets logging level), `--dry-run` where it
makes sense. Every subcommand prints the resulting path(s) on success and a clear
one-line error on failure (exit code 1; 2 for bad usage). `main(argv=None) -> int`.

`doctor` is the support tool: report ffmpeg version, which optional extras import,
whether a Claude key resolves, which TTS providers are usable, whether Chromium
launches, and the asset library size -- each as an OK/MISSING line.

## Tests

`tests/test_pipelines.py` and `tests/test_cli.py`. Everything must pass offline
with the heuristic LLM provider and the offline TTS provider. Keep renders tiny
(set `AICLIP_WIDTH=180`, `AICLIP_HEIGHT=320`, short durations). Cover at least:
each pipeline end-to-end producing a probe-able file of roughly the expected
duration; the clip pipeline with a monkeypatched transcriber so no model download
is needed; CLI argument parsing for every subcommand; `voices`/`styles`/`doctor`
output; exit codes on bad input.
