# What this is, and how it maps to the commercial tools

`aiclipper` is a **clean-room** rebuild of the short-form video product category
that tools like Crayo, Opus Clip and their competitors occupy. Nothing here is
derived from any of their code, assets or APIs. The feature set below was
assembled from their public marketing pages and third-party reviews, then
reimplemented from scratch on open components.

Two boundaries were deliberately kept:

* **No trade dress.** The chat and forum templates are our own visual design.
  They are generic message bubbles and a generic story card, not reproductions
  of any real application's interface, palette, iconography or branding.
* **No scraping, no private APIs.** Media comes in through `yt-dlp` from URLs the
  user supplies, or from local files. The engine never touches a competitor's
  backend.

## Feature by feature

| Category feature | Status | Where it lives |
|---|---|---|
| Long video / URL to vertical shorts | Built | `pipelines/clip.py` |
| Highlight detection (pick the good moments) | Built, LLM + heuristic | `highlight.py` |
| Word-level transcription | Built | `transcribe.py` |
| Auto vertical reframing with face tracking | Built | `crop.py` |
| Script generation from a topic | Built, LLM + template | `scriptgen.py` |
| AI voiceover, multi-voice library | Built, 50 voices, 3 providers | `tts/` |
| Animated word-level captions, style presets | Built, 16 presets, 5 animations | `captions.py` |
| Background footage + music library | Built, plus procedural generation | `assets.py` |
| Music ducked under narration | Built | `render.py` sidechain bus |
| Narrated story videos over background | Built | `pipelines/story.py` |
| Text-conversation story videos | Built, original design | `pipelines/texts.py`, `overlays.py` |
| Forum/story-card videos | Built, original design | `pipelines/reddit.py`, `overlays.py` |
| Split-screen / two-pane format | Built | `pipelines/split.py` |
| Streamer-clip format | Covered by `clip` + `split` | — |
| Programmatic API | The Python package itself | `aiclipper.pipelines` |
| Command line interface | Built | `cli.py` |
| AI image generation per beat | **Not built** | `ScriptBeat.image_prompt` is carried but unused |
| Text-to-video model integration | **Not built** | would slot in as a `VisualLayer` source |
| Social publishing and scheduling | **Not built** | needs per-platform app review |
| Post analytics / account tracking | **Not built** | out of scope for a render engine |
| Hosted web editor, timeline UI | **Not built** | the `Timeline` type is the seam for one |
| Accounts, credits, billing | **Not built** | intentionally absent |

## Why the gaps are where they are

The three unbuilt items that matter are deliberate.

**Social publishing** is not a technical problem, it is an approval problem. Each
platform requires a reviewed application with its own content and disclosure
rules before it will accept uploads. The engine writes ordinary mp4 files, so any
existing publishing tool can take it from there.

**Generative images and video** are a provider integration, not an architecture
change. `ScriptBeat.image_prompt` is already generated and carried through the
script types; a provider module returning image paths would drop straight into a
`VisualLayer` with no change to the renderer.

**The web editor** is the largest remaining piece, and the `Timeline` dataclass is
the seam it would attach to. Everything upstream produces a `Timeline`, and the
renderer consumes nothing else, so a UI that edits a `Timeline` and re-renders is
a self-contained project rather than a rewrite.

## Running without a cloud provider

Every model-backed stage has a local backend, so the engine can run end to end on
one machine with no account anywhere:

| Stage | Local backend | Notes |
|---|---|---|
| Transcription | faster-whisper | local from the start; downloads weights once |
| Scripts, clip selection | any OpenAI-compatible endpoint | Ollama, llama.cpp, LM Studio, vLLM |
| Narration | Piper | small, CPU-fast, permissively licensed |
| Reframing, captions, overlays, rendering | OpenCV, ffmpeg, Pillow | no model involved |

The local language backend is deliberately defensive: it asks for schema-guided
output, validates the reply, makes one repair attempt, and falls through to the
rule-based writer instead of failing a render. Small models ramble, and a video
pipeline should survive that.

## What runs without anything

The engine has an offline path end to end: a rule-based language provider that
emits schema-valid output for any schema, a text-to-speech provider that
synthesises timed silence with plausible word timings, and procedurally generated
backgrounds and music beds. That means the full test suite runs with no API keys,
no network and no downloaded assets, and it means a user can try every workflow
before paying any provider. Quality improves considerably with a real language
model and a real voice, but nothing is blocked without them.
