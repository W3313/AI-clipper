# aiclipper architecture

A clean-room short-form video engine. Five user-facing workflows share one
render core:

| Workflow | What it does |
|---|---|
| `clip` | long video / YouTube URL -> N vertical shorts, auto-reframed and captioned |
| `story` | topic or script -> narrated short over background footage |
| `texts` | a text-message conversation -> animated chat story video |
| `reddit` | a forum-style post -> narrated story video with a title card |
| `split` | narration + a second video pane stacked vertically |

## Hard rules

1. **`models.py` is the contract.** Never redefine a type that lives there; import it.
2. **Everything optional degrades.** `faster-whisper`, `yt-dlp`, `opencv`, `anthropic`,
   `edge-tts`, `playwright` are all optional extras. Import them lazily *inside*
   the function that needs them. If missing, raise `MissingDependency` (from
   `aiclipper.errors`) with the exact `pip install` line, or fall back to the
   offline path where one exists. `import aiclipper.<anything>` must never fail
   on a bare interpreter with only numpy + Pillow installed.
3. **No network at import time**, and none at all when `settings.offline` is true.
4. **ffmpeg only through `aiclipper.ffmpeg`.** Never call `subprocess` on ffmpeg directly.
5. **The renderer's only input is a `Timeline`.** Producers never emit ffmpeg args.
6. **Original artwork only.** The chat and forum templates are our own visual
   design -- generic bubbles, generic card. Do not imitate the trade dress,
   logos, colour schemes or iconography of any real product.
7. **Deterministic where possible.** Seed randomness from `settings.seed` so the
   same inputs give the same output.
8. Type-hint everything, `from __future__ import annotations` at the top, module
   docstring, `__all__`. Target Python 3.10+. Format: 110 columns, ruff-clean.

## Layout

```
aiclipper/
  models.py        DONE - data contract (Word/Transcript/Timeline/...)
  config.py        DONE - Settings, env overrides, get_settings()
  ffmpeg.py        DONE - run_ffmpeg/probe/extract_audio/make_silence/make_tone/has_filter
  errors.py        DONE - AiclipperError, MissingDependency, ...
  ingest.py        media in: local path or URL -> MediaInfo
  assets.py        background/music library + procedural placeholders
  transcribe.py    audio -> Transcript (word level), plus forced alignment
  llm/             provider interface, Claude provider, offline heuristic
  highlight.py     Transcript -> [ClipCandidate]
  scriptgen.py     topic -> VideoScript / ChatScript / RedditPost
  tts/             voice catalogue + edge/elevenlabs/offline providers
  crop.py          reframing: wide source -> CropPath (face/motion tracked)
  captions.py      Words -> CaptionCue groups -> ASS file, 16 style presets
  overlays.py      chat + forum card images (Chromium or Pillow backend)
  templates/       HTML/CSS for the overlay renderer
  render.py        Timeline -> ffmpeg filtergraph -> mp4
  pipelines/       one module per workflow, wiring the above together
  cli.py           `aiclip clip|story|texts|reddit|split|voices|styles`
```

## Module contracts

Signatures below are binding. Add helpers freely; do not change these.

### `ingest.py`
```python
def resolve(source: str | Path, *, settings=None, workspace: Path | None = None,
            quality: str = "bv*[height<=1080]+ba/b[height<=1080]") -> MediaInfo
def download(url: str, dest_dir: Path, *, settings=None, quality: str = ...) -> MediaInfo
def is_url(source: str) -> bool
def normalize(info: MediaInfo, dest: Path, *, width=None, height=None, fps=None, settings=None) -> MediaInfo
```
`resolve` accepts a local path (probe it) or URL (download via yt-dlp, prefer the
Python module, fall back to the CLI). Cache downloads in `settings.cache_dir`
keyed by a hash of the URL + quality; a cache hit must not hit the network.

### `assets.py`
```python
@dataclass
class Asset:  # path, name, kind ("background"|"music"), tags: list[str], duration: float
def library(settings=None) -> list[Asset]
def pick_background(name: str | None = None, *, tags=None, settings=None, seed=None) -> Asset
def pick_music(name: str | None = None, *, tags=None, settings=None, seed=None) -> Asset
def ensure_placeholders(settings=None) -> list[Asset]
```
`ensure_placeholders` **generates** assets with ffmpeg lavfi when the library is
empty, so the engine runs end-to-end with zero downloads: at least three
visually distinct 20s 1080x1920 loops (e.g. animated gradient, drifting cells,
soft noise field -- use `gradients`, `life`, `mandelbrot`, `testsrc2`, `noise`,
`hue`, `zoompan`) and two 30s music beds built from layered `sine`/`aevalsrc`
with an envelope. Cache them in `settings.assets_dir`; regenerate only if absent.
A `library.json` manifest alongside them records name/kind/tags/duration.

### `transcribe.py`
```python
def transcribe(media: str | Path, *, settings=None, language: str | None = None,
               model: str | None = None, vad: bool = True) -> Transcript
def align(audio: str | Path, text: str, *, settings=None, language: str = "en") -> list[Word]
def available() -> bool
```
faster-whisper with `word_timestamps=True`. `align` force-aligns known narration
text to synthesised audio: transcribe, then map the ASR words onto the reference
token sequence so the returned words carry the *reference* spelling with ASR
timings (a simple difflib/SequenceMatcher alignment is fine; interpolate timings
for unmatched reference tokens). Cache transcripts next to the media as
`<stem>.transcript.json` and reuse when newer than the media file.

### `llm/`
`llm/base.py`
```python
class LLMProvider(Protocol):
    name: str
    def available(self) -> bool: ...
    def complete(self, prompt: str, *, system: str = "", max_tokens: int | None = None) -> str: ...
    def complete_json(self, prompt: str, schema: dict, *, system: str = "",
                      max_tokens: int | None = None) -> dict: ...
def get_provider(name: str | None = None, *, settings=None) -> LLMProvider
```
`llm/claude.py` -- `ClaudeProvider`. Uses the official `anthropic` SDK:
```python
import anthropic
client = anthropic.Anthropic()                     # reads ANTHROPIC_API_KEY / ant profile
resp = client.messages.create(
    model=settings.llm_model,                      # default "claude-opus-5"
    max_tokens=settings.llm_max_tokens,            # 16000
    system=system,
    messages=[{"role": "user", "content": prompt}],
    output_config={"effort": settings.llm_effort,
                   "format": {"type": "json_schema", "schema": schema}},
)
text = next(b.text for b in resp.content if b.type == "text")
data = json.loads(text)
```
Schemas passed to `output_config.format` need `"additionalProperties": False` and
a `required` list covering every property. Rules: never send `thinking`
(adaptive is the default on Opus 5), never send `budget_tokens`, never prefill an
assistant turn, never force `tool_choice`. Guard `resp.stop_reason == "refusal"`
before reading content. Retry once on `anthropic.APIStatusError` with 5xx/429,
then fall back to the heuristic provider rather than crashing a render.

`llm/heuristic.py` -- `HeuristicProvider`, zero network. Implements the same
interface with rule-based text: sentence ranking, template filling, and schema
walking that emits a valid (if unremarkable) object for any schema it is given.
This is what makes every workflow runnable offline, so it must never raise.

`llm/prompts.py` -- system prompts + JSON schemas for highlight selection,
script writing, chat writing and forum-post writing, as module constants.

`get_provider("auto")` -> Claude when `ANTHROPIC_API_KEY` (or an `ant` profile)
resolves and `settings.offline` is false, else heuristic.

### `highlight.py`
```python
def select(transcript: Transcript, *, count: int = 3, min_duration: float = 15.0,
           max_duration: float = 60.0, settings=None, provider=None) -> list[ClipCandidate]
def score_window(transcript: Transcript, start: float, end: float) -> float
def snap_to_speech(transcript: Transcript, start: float, end: float,
                   min_duration: float, max_duration: float) -> tuple[float, float]
```
LLM path asks for moments by timestamp and title; heuristic path scores sliding
windows (hook words, question marks, numbers, named entities, laughter markers,
words-per-second, self-contained sentence boundaries). Always snap boundaries to
word edges, enforce min/max duration, drop overlaps (keep the higher score), and
return at most `count`, ranked. Never return a window with no words.

### `scriptgen.py`
```python
def write_script(topic: str, *, seconds: int = 35, tone: str = "punchy",
                 settings=None, provider=None) -> VideoScript
def write_chat(topic: str, *, turns: int = 14, settings=None, provider=None) -> ChatScript
def write_reddit(topic: str, *, words: int = 180, settings=None, provider=None) -> RedditPost
def parse_chat(text: str) -> ChatScript      # "Alex: hey" / "> me: hey" plain-text form
def parse_script(text: str) -> VideoScript   # plain text or one-line-per-beat
```
`parse_*` let a user supply their own script instead of generating one.
Budget the word count to the target duration (~2.6 words/second).

### `tts/`
`tts/base.py`
```python
class TTSProvider(Protocol):
    name: str
    def available(self) -> bool: ...
    def synthesize(self, text: str, out_path: Path, *, voice: VoiceSpec) -> TTSResult: ...
def get_provider(name: str | None = None, *, settings=None) -> TTSProvider
def synthesize_lines(lines: Sequence[str], out_dir: Path, *, voice: VoiceSpec,
                     provider=None, gap: float = 0.18, settings=None,
                     fallback: bool = True) -> NarrationResults
```
`tts/edge.py` (`edge-tts`, free, gives WordBoundary events -> fill `TTSResult.words`),
`tts/eleven.py` (`ELEVENLABS_API_KEY`, REST via urllib, words left `None`),
`tts/offline.py` (always available: renders timed silence with `ffmpeg.make_silence`
at ~2.6 words/second and synthesises plausible `Word` timings so captions and
overlays still animate -- the CI/offline path).
`tts/voices.py` -- a `VOICES` catalogue of at least 40 named voices
(`narrator_deep`, `bright_female`, `documentary`, ...) each mapping to per-provider
ids plus tags (gender, accent, energy); `find_voice(name) -> VoiceSpec`,
`list_voices(provider=None) -> list[...]`.

### `crop.py`
```python
def track(media: str | Path, *, target_aspect: float = 9/16, settings=None,
          sample_fps: float = 4.0, smooth_seconds: float = 1.2,
          start: float = 0.0, end: float | None = None) -> CropPath
def center_crop(info: MediaInfo, target_aspect: float = 9/16) -> CropPath
def simplify(path: CropPath, tolerance: float = 8.0) -> CropPath
```
Sample frames with OpenCV, detect faces (bundled Haar cascade via
`cv2.data.haarcascades`), fall back to frame-difference motion energy when no
face is found, produce a centre-of-interest per sample, smooth it (moving average
+ deadzone so it does not jitter), clamp inside frame bounds, and emit keyframes.
`simplify` drops keyframes within `tolerance` px of the interpolated path.
Without OpenCV installed, `track` must transparently return `center_crop`.

### `captions.py`
```python
PRESETS: dict[str, CaptionStyle]                       # >= 16 named looks
def group_words(words: Sequence[Word], style: CaptionStyle, *,
                hold: float = HOLD_SECONDS, gap_split: float = GAP_SPLIT) -> list[CaptionCue]
def write_ass(cues: Sequence[CaptionCue], out_path: Path, *, style: CaptionStyle,
              width: int, height: int) -> Path
def build(words: Sequence[Word], out_path: Path, *, style: str | CaptionStyle = "clean",
          width: int = 1080, height: int = 1920,
          hold: float = HOLD_SECONDS, gap_split: float = GAP_SPLIT) -> Path
def get_style(name: str | CaptionStyle) -> CaptionStyle
```
Write ASS v4.00+ by hand (no library). Colours convert to `&HAABBGGRR`.
Group by `max_words`/`max_chars` and split on gaps > 0.7s. Animations:
`karaoke` re-emits the group once per word with the active word in
`highlight_color`; `pop` adds `\t` scale on the active word; `bounce` adds a
short `\move`; `fade` uses `\fad`; `typewriter` reveals word by word.
Escape `{`, `}` and newlines. Presets must be visually distinct and include at
least: clean, bold_yellow, karaoke_green, outline_pop, boxed, minimal_serif,
neon, comic, subtle_lower, big_impact, gradient_pop, mono_terminal,
handwritten, shadow_deep, tiktok_white, podcast_bar.

### `overlays.py`
```python
@dataclass
class OverlayImage:  # path: Path, width: int, height: int, index: int
def render_chat(script: ChatScript, out_dir: Path, *, width: int, height: int,
                settings=None, backend: str | None = None,
                header_state: bool = False) -> list[OverlayImage]
def render_forum_card(post: RedditPost, out_path: Path, *, width: int, height: int,
                      settings=None, backend: str | None = None) -> OverlayImage
def available_backends(settings=None) -> list[str]
```
`render_chat` returns one image per *conversation state*: image `i` shows
messages `0..i` (plus an optional typing indicator frame), so the pipeline
overlays image `i` from the moment message `i` lands until message `i+1`. Keep
the newest bubble anchored in view; scroll older ones up when the stack
overflows the canvas. Images are transparent PNGs (RGBA) sized to the canvas.
Two backends: `chromium` (Playwright -> `templates/chat.html` + `chat.css`,
respecting `settings.chromium` / `PLAYWRIGHT_BROWSERS_PATH`) and `pillow`
(pure-Python, always available). `available_backends` reports what will work;
`backend=None` prefers chromium and silently falls back to pillow.
`header_state=True` prepends a chrome-only state so a caller can hold it from
`t=0` instead of opening on a bare background; it shifts every message state by
one, so map states by the documented indices, not by position.
Themes are single-sourced as data in `overlays.py` and consumed by BOTH backends,
so a silent fallback from chromium to pillow does not change how the video looks.
Original visual design only -- see hard rule 6. That extends to layout grammar,
not just colour: the story card must not reproduce the prefixes, iconography or
arrangement that identify a specific real service.

### `render.py`
```python
def render(timeline: Timeline, out_path: Path, *, options: RenderOptions | None = None,
           settings=None, log_path: Path | None = None, dry_run: bool = False,
           loudness_target: float | None = DEFAULT_LOUDNESS_TARGET) -> RenderResult
def build_command(timeline: Timeline, out_path: Path, *, options=None, settings=None,
                  workdir: Path | None = None,
                  loudness_target: float | None = DEFAULT_LOUDNESS_TARGET) -> list[str]
```
Builds one ffmpeg invocation with a `-filter_complex` graph:
* base canvas `color=c=<background>:s=WxH:r=FPS:d=DURATION`
* each visual layer: `-i src` (with `-stream_loop -1` when `loop`),
  `trim`/`setpts` for `src_start`, `crop` for `CropPath` (static -> plain crop;
  animated -> write a `sendcmd` script to `workdir` and chain
  `sendcmd=f=<file>,crop=...`), `scale`+`crop`/`pad` to honour `fit`,
  `format=rgba,colorchannelmixer=aa=<opacity>` when `opacity < 1`,
  then `overlay=x:y:enable='between(t,start,end)'`
* audio: per track `atrim`/`adelay`/`volume` (`gain_db` -> `volume=<db>dB`),
  `afade` in/out, `aloop` when looping; ducking uses `sidechaincompress` keyed
  off the summed voice bus; finally `amix` + `alimiter`, or `anullsrc` when the
  timeline is silent
* subtitles burned last: `subtitles=<escaped ass path>` (pass `fontsdir` when set)
* output: `-c:v <codec> -preset -crf -pix_fmt`, `-c:a aac -b:a`, `-movflags +faststart`,
  `-r fps`, `-t duration`, `-shortest` never (explicit `-t` instead)
* loudness: normalise the finished mix toward `loudness_target` LUFS (default
  `DEFAULT_LOUDNESS_TARGET`) with the limiter downstream of it, leaving genuine
  silence silent and peaks under `LOUDNESS_TRUE_PEAK`; `None` disables it
Scratch files the renderer generates for itself -- sendcmd scripts, intermediates,
command logs -- belong under `settings.work_dir`, keyed so concurrent renders
cannot collide. The output directory receives the finished file and nothing else.
Call `timeline.validate()` first and raise `RenderError` listing every problem.
`dry_run` returns the `RenderResult` with the command and no ffmpeg execution.
Use `ffmpeg.escape_filter_path` for every path inside the graph.

### `pipelines/` and `cli.py`
Written after the modules land; each pipeline returns a `ProjectResult`.

## Testing

`tests/` with pytest. Every module gets a test file. Rules:
* Tests must pass **offline** with only the base deps + ffmpeg installed.
* Mark tests needing an optional extra with
  `pytest.importorskip("faster_whisper")` etc.
* Prefer tiny generated fixtures (`ffmpeg.make_tone`, a 2s `testsrc2` clip)
  over checked-in media. Put helpers in `tests/conftest.py`.
* Render tests use `dry_run=True` for graph assertions plus **one** real short
  render (<= 2s, 320x568) to prove the graph actually executes.
