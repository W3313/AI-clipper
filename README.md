# aiclipper

A short-form video engine. It turns long videos into captioned vertical shorts,
and turns a topic into a narrated video with voiceover, background footage,
animated captions and music.

It is a clean-room rebuild of the product category that tools like Crayo and
Opus Clip occupy, built on open components. See [docs/FEATURE-MAP.md](docs/FEATURE-MAP.md)
for what is and is not implemented, and why.

## What it does

| Command | What you get |
|---|---|
| `aiclip clip` | a long video or URL becomes N ranked vertical shorts, auto-reframed and captioned |
| `aiclip story` | a topic becomes a narrated short over background footage |
| `aiclip texts` | a text conversation becomes an animated chat story video |
| `aiclip reddit` | a forum-style story is read aloud over a story card |
| `aiclip split` | two stacked panes: your clip on top, background below |

## Install

Requires Python 3.10+ and ffmpeg.

```bash
sudo apt install ffmpeg          # or: brew install ffmpeg
pip install -e '.[all]'
```

Every heavy dependency is optional. The base install runs; each extra unlocks a
capability: `transcribe` (speech recognition), `ingest` (URL downloads), `vision`
(face-tracked reframing), `llm` (Claude-written scripts), `tts` (neural voices),
`overlays` (browser-rendered chat graphics).

Two extras need one more step before they can do anything. `overlays` installs
the Playwright driver but not a browser -- run `playwright install chromium` for
that, or let the chat and forum graphics fall back to the built-in Pillow
renderer, which draws the same design. `transcribe` downloads its Whisper
weights the first time you clip something, so that first run needs network.
`aiclip doctor` tells you which of the two is missing.

Check what you have:

```bash
aiclip doctor
```

## Use it

```bash
# a topic to a finished video
aiclip story --topic "why sourdough starters die" --seconds 40 --voice narrator_deep

# your own script, your own look
aiclip story --script script.txt --style neon --background cells --music calm

# a long video into three shorts
aiclip clip talk.mp4 --count 3 --min 20 --max 45 --style bold_yellow

# a chat story
aiclip texts --topic "a text thread that goes wrong" --theme mint

# see the catalogues
aiclip voices --tag calm
aiclip styles
```

Output lands in `out/` unless you pass `--out`. Every command that renders prints
the paths it wrote to stdout, one per line, and nothing else -- progress and
warnings go to stderr -- so it composes in a shell pipeline.

## Use it as a library

```python
from aiclipper.pipelines import story

result = story.run(topic="why sourdough starters die", seconds=40)
print(result.output, result.duration)
```

Each pipeline returns a `ProjectResult` carrying the output path, the duration and
metadata about how it was made. The `clip` pipeline returns one per short.

## Run it fully locally

Nothing here has to touch a cloud service. The three model-backed stages each
have a local backend, and the rest of the engine -- reframing, captions, chat and
card graphics, all rendering -- is local already.

**Transcription** is local out of the box. It uses faster-whisper, which
downloads its weights once and then runs offline on CPU or GPU:

```bash
export AICLIP_WHISPER_MODEL=base        # tiny | base | small | medium | large-v3
export AICLIP_WHISPER_DEVICE=cuda       # or cpu (the default)
```

**Scripts and clip selection** can run on a self-hosted model through any
OpenAI-compatible endpoint, which covers Ollama, llama.cpp's server, LM Studio
and vLLM:

```bash
ollama pull llama3.1
export AICLIP_LLM=local
export AICLIP_LLM_BASE_URL=http://localhost:11434/v1
export AICLIP_LLM_LOCAL_MODEL=llama3.1
```

Smaller models are less reliable at strict JSON than a frontier model, so the
local backend asks for schema-guided output, validates what comes back, makes one
repair attempt, and falls through to the rule-based writer rather than letting a
rambling reply kill a render.

**Narration** can run on Piper, a small CPU-fast neural voice:

```bash
# install piper and drop a voice .onnx into assets/piper/
export AICLIP_TTS=piper
export AICLIP_PIPER_VOICE=en_US-lessac-medium
```

Then confirm the whole local stack in one command:

```bash
aiclip doctor
```

It reports each backend as installed, usable, or missing, and a local endpoint
that is configured but not answering shows up there rather than halfway through a
render.

## It runs with nothing configured

There is a complete offline path: a rule-based script writer, a speech synthesiser
that produces correctly timed silence, and procedurally generated backgrounds and
music. All five workflows run, and the whole test suite passes, with no API keys,
no network and no downloaded assets.

What degrades rather than fails: the scripts are templated instead of written,
the narration is silence of the right length instead of a voice, and `clip` -- if
the Whisper weights have never been downloaded -- cuts evenly spaced windows with
no captions instead of ranked, captioned moments. Nothing raises, and every
fallback says so on stderr.

Quality improves a lot with real providers, and nothing is blocked without them:

```bash
export ANTHROPIC_API_KEY=...     # Claude writes the scripts and picks the clips
export ELEVENLABS_API_KEY=...    # optional; edge-tts is the free default voice
```

Drop your own loops into `assets/backgrounds/` and beds into `assets/music/` to
replace the generated placeholders.

## How it fits together

Producers make data; one renderer consumes it. Everything upstream ends at a
`Timeline`, and the renderer compiles that into a single ffmpeg filtergraph.

```
ingest ──► transcribe ──► highlight ──┐
                                      ├──► Timeline ──► render ──► mp4
scriptgen ──► tts ──► captions ───────┤
                      overlays ───────┤
                      assets ─────────┘
                      crop  ──────────┘
```

That seam is deliberate. Adding a generative image provider, a new caption
animation or a visual editor means producing a `Timeline`, not touching the
renderer. [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) documents every module
interface; [docs/PIPELINES.md](docs/PIPELINES.md) covers the workflows and CLI.

## Configuration

Every setting is an environment variable, so the same code runs on a laptop, in
CI and in a container. The ones worth knowing: `AICLIP_WIDTH`, `AICLIP_HEIGHT`,
`AICLIP_FPS`, `AICLIP_WORK_DIR`, `AICLIP_OUTPUT_DIR`, `AICLIP_ASSETS_DIR`,
`AICLIP_OFFLINE`, `AICLIP_SEED`, `AICLIP_LLM_MODEL`, `AICLIP_VOICE`,
`AICLIP_WHISPER_MODEL`. The global CLI flags set them for you.
`aiclipper/config.py` is the full list -- provider and binary overrides
(`AICLIP_LLM`, `AICLIP_TTS`, `AICLIP_FFMPEG`, `AICLIP_CHROMIUM`) live there too.

Scratch files go to `AICLIP_WORK_DIR`; `AICLIP_OUTPUT_DIR` only ever receives
finished videos.

Runs are deterministic under a fixed `--seed`.

## Develop

```bash
pip install -e '.[all,dev]'
pytest -q          # offline, needs only ffmpeg
ruff check .
```

## License

MIT. See [LICENSE](LICENSE).
