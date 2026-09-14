"""The named voice catalogue.

The engine talks about voices by *our* names -- ``narrator_deep``,
``bright_female``, ``documentary`` -- not by a provider's internal identifier.
Each :class:`Voice` entry carries a human description, a bag of tags (gender,
accent, energy, use-case) and the per-provider ids we know for it.  That keeps
the CLI, the pipelines and the config file provider-agnostic: swap ``edge`` for
``elevenlabs`` and the same ``--voice narrator_deep`` keeps working.

Three entry points:

:func:`find_voice`
    Resolve user input to a :class:`~aiclipper.models.VoiceSpec`.  It accepts a
    catalogue name (exactly, or in any case), a ``provider:native_id`` string, or
    a loose tag query like ``"british female calm"``.  Failure raises
    :class:`~aiclipper.errors.TTSError` listing the closest catalogue names.

:func:`list_voices`
    The catalogue, optionally filtered to what one provider can actually speak.

:func:`resolve_voice_id`
    What the providers call: turn a :class:`~aiclipper.models.VoiceSpec` into the
    native id for *this* backend.  A ``voice_id`` that is not a catalogue name is
    passed straight through, so ``VoiceSpec.parse("edge:en-GB-RyanNeural")``
    still works for voices we never listed.

This module imports nothing outside the standard library.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..errors import TTSError
from ..models import VoiceSpec

__all__ = [
    "Voice", "VOICES", "PROVIDER_ALIASES", "KNOWN_PROVIDERS",
    "canonical_provider", "find_voice", "find_voice_entry", "list_voices",
    "resolve_voice_id", "voice_names", "tag_values",
]

#: Accepted provider spellings -> canonical backend name.
PROVIDER_ALIASES: dict[str, str] = {
    "edge": "edge",
    "edge-tts": "edge",
    "edge_tts": "edge",
    "microsoft": "edge",
    "eleven": "elevenlabs",
    "elevenlabs": "elevenlabs",
    "eleven-labs": "elevenlabs",
    "11labs": "elevenlabs",
    "xi": "elevenlabs",
    "piper": "piper",
    "piper-tts": "piper",
    "local": "piper",
    "onnx": "piper",
    "offline": "offline",
    "silence": "offline",
    "null": "offline",
    "none": "offline",
    "auto": "auto",
    "default": "auto",
    "": "auto",
}

#: Backends a catalogue entry may carry an id for.
KNOWN_PROVIDERS: tuple[str, ...] = ("edge", "elevenlabs", "piper", "offline")


def canonical_provider(name: str | None) -> str:
    """Normalise a provider spelling; unknown names come back lower-cased."""
    key = (name or "").strip().lower()
    return PROVIDER_ALIASES.get(key, key or "auto")


@dataclass(frozen=True)
class Voice:
    """One named voice and the provider ids that can speak it."""

    name: str
    description: str = ""
    tags: tuple[str, ...] = ()
    edge: str | None = None
    elevenlabs: str | None = None
    piper: str | None = None
    language: str = "en"
    style: str | None = None

    @property
    def providers(self) -> dict[str, str]:
        """Canonical provider name -> native id, for the backends we know.

        ``offline`` is always present: the silent fallback can "speak" anything.
        """
        out: dict[str, str] = {"offline": self.name}
        if self.edge:
            out["edge"] = self.edge
        if self.elevenlabs:
            out["elevenlabs"] = self.elevenlabs
        if self.piper:
            out["piper"] = self.piper
        return out

    def provider_id(self, provider: str | None) -> str | None:
        """Native id for ``provider``; ``None`` when this voice has none."""
        canonical = canonical_provider(provider)
        if canonical in ("auto", ""):
            return self.name
        return self.providers.get(canonical)

    def supports(self, provider: str | None) -> bool:
        return self.provider_id(provider) is not None

    def has_tag(self, tag: str) -> bool:
        return tag.strip().lower() in self.tags

    def to_spec(
        self,
        provider: str | None = None,
        *,
        rate: float = 1.0,
        pitch_semitones: float = 0.0,
    ) -> VoiceSpec:
        """Build a :class:`~aiclipper.models.VoiceSpec` for this entry.

        With no ``provider`` the spec stays provider-agnostic and carries the
        *catalogue* name -- every backend resolves that through
        :func:`resolve_voice_id` at synthesis time.
        """
        canonical = canonical_provider(provider)
        voice_id = self.provider_id(canonical) or self.name
        return VoiceSpec(
            provider=canonical or "auto",
            voice_id=voice_id,
            rate=rate,
            pitch_semitones=pitch_semitones,
            style=self.style,
            language=self.language,
        )


def _v(
    name: str,
    description: str,
    tags: str,
    *,
    edge: str | None = None,
    elevenlabs: str | None = None,
    piper: str | None = None,
    language: str = "en",
    style: str | None = None,
) -> Voice:
    return Voice(
        name=name,
        description=description,
        tags=tuple(t for t in tags.split() if t),
        edge=edge,
        elevenlabs=elevenlabs,
        piper=piper,
        language=language,
        style=style,
    )


#: The catalogue.  Names are ours; ``edge`` ids are Microsoft neural voice
#: names, ``elevenlabs`` ids are the public sample voice ids and ``piper`` ids
#: are downloadable model names (``<lang>-<speaker>-<quality>``), each given
#: where one is known (``None`` means "this backend has no mapping -- pick
#: another").  Piper's English catalogue is en_US and en_GB only and has no
#: style control, so every other accent -- and the child, ASMR and hype-promo
#: entries -- is left ``None`` rather than pointed at a voice that is not that
#: voice: ``list_voices("piper")`` must list what it can really speak.
VOICES: tuple[Voice, ...] = (
    # -- American -------------------------------------------------------- #
    _v("narrator_deep", "Low, unhurried male narration for documentary voice-over.",
       "male american deep calm narration documentary audiobook",
       edge="en-US-GuyNeural", elevenlabs="pNInz6obpgDQGcFmaJgB", piper="en_US-ryan-high"),
    _v("narrator_warm", "Warm, even male read that sits under music without fighting it.",
       "male american warm calm narration podcast audiobook",
       edge="en-US-ChristopherNeural", elevenlabs="ErXwobaYiN019PkySvjV", piper="en_US-joe-medium"),
    _v("narrator_female", "Measured female narration with a steady, trustworthy centre.",
       "female american warm calm narration audiobook documentary",
       edge="en-US-JennyNeural", elevenlabs="21m00Tcm4TlvDq8ikWAM", piper="en_US-lessac-medium"),
    _v("bright_female", "Bright, upbeat female delivery built for fast social hooks.",
       "female american bright energetic promo ads storytelling",
       edge="en-US-AriaNeural", elevenlabs="EXAVITQu4vr4xnSDxMaL", piper="en_US-amy-medium"),
    _v("bright_male", "Clean, energetic male read with plenty of forward lean.",
       "male american bright energetic promo ads tutorial",
       edge="en-US-AndrewNeural", elevenlabs="TxGEqnHWrfWFTfGW9XjX", piper="en_US-ryan-medium"),
    _v("documentary", "Serious, spacious delivery for archive-footage storytelling.",
       "male american deep serious documentary narration",
       edge="en-US-RogerNeural", elevenlabs="VR6AewLTigWG4xSOukaG", piper="en_US-norman-medium"),
    _v("newsroom", "Crisp anchor cadence: clear consonants, no ornament.",
       "male american crisp serious news commentary",
       edge="en-US-EricNeural", piper="en_US-john-medium"),
    _v("newsroom_female", "Studio-desk female read with tight, confident phrasing.",
       "female american crisp serious news commentary",
       edge="en-US-MichelleNeural", elevenlabs="AZnzlk1XvdvUeBnXmlld", piper="en_US-hfc_female-medium"),
    _v("storyteller", "Conversational female voice that leans into a plot twist.",
       "female american warm playful storytelling narration",
       edge="en-US-AvaNeural", elevenlabs="MF3mGyEYCl7XYWbV9V6O", piper="en_US-ljspeech-high"),
    _v("storyteller_male", "Easy, fireside male storytelling with a dry edge.",
       "male american warm playful storytelling narration",
       edge="en-US-BrianNeural", piper="en_US-bryce-medium"),
    _v("explainer", "Patient tutorial voice that lands each step cleanly.",
       "male american crisp calm tutorial narration",
       edge="en-US-SteffanNeural", piper="en_US-hfc_male-medium"),
    _v("explainer_female", "Friendly how-to female read, teacherly but never slow.",
       "female american warm crisp tutorial narration",
       edge="en-US-EmmaNeural", piper="en_US-lessac-high"),
    _v("hype_promo", "High-energy trailer read for a hard-sell opening line.",
       "male american energetic bright promo ads",
       edge="en-US-AndrewMultilingualNeural"),
    _v("kid_bright", "Young, playful voice for kids' and cartoon-style stories.",
       "female american bright playful kids storytelling",
       edge="en-US-AnaNeural"),
    _v("asmr_soft", "Very soft, close-mic female delivery for calm content.",
       "female american soft calm asmr meditation",
       edge="en-US-EmmaMultilingualNeural"),
    _v("meditation_male", "Slow, low male guide voice for breathing and sleep content.",
       "male american deep soft calm meditation asmr",
       edge="en-US-BrianMultilingualNeural"),
    _v("podcast_host", "Relaxed two-mic podcast energy with natural pauses.",
       "female american warm playful podcast commentary",
       edge="en-US-AvaMultilingualNeural", piper="en_US-kathleen-low"),
    _v("gaming_commentary", "Fast, reactive commentary voice for gameplay clips.",
       "male american energetic playful gaming commentary",
       edge="en-US-GuyNeural", elevenlabs="yoZ06aMxZJJ28mfd3POQ"),

    # -- British --------------------------------------------------------- #
    _v("british_male", "Neutral RP male read: composed, articulate, unhurried.",
       "male british calm crisp narration documentary",
       edge="en-GB-RyanNeural", piper="en_GB-alan-medium"),
    _v("british_female", "Neutral RP female read with a light, precise touch.",
       "female british calm crisp narration audiobook",
       edge="en-GB-SoniaNeural", piper="en_GB-cori-high"),
    _v("british_warm", "Softer British female voice, good for personal stories.",
       "female british warm soft storytelling audiobook",
       edge="en-GB-LibbyNeural", piper="en_GB-jenny_dioco-medium"),
    _v("british_gravitas", "Weighty British male delivery for history and mystery.",
       "male british deep serious documentary narration",
       edge="en-GB-ThomasNeural", piper="en_GB-northern_english_male-medium"),
    _v("british_young", "Youthful British voice for first-person confession stories.",
       "female british bright playful storytelling kids",
       edge="en-GB-MaisieNeural", piper="en_GB-southern_english_female-low"),

    # -- Irish / Scottish-adjacent --------------------------------------- #
    _v("irish_male", "Irish male read with a lilt that keeps long copy alive.",
       "male irish warm playful storytelling narration",
       edge="en-IE-ConnorNeural"),
    _v("irish_female", "Irish female voice, bright and conversational.",
       "female irish bright warm storytelling podcast",
       edge="en-IE-EmilyNeural"),

    # -- Australian / NZ -------------------------------------------------- #
    _v("aussie_male", "Australian male read: open vowels, relaxed pace.",
       "male australian warm playful commentary narration",
       edge="en-AU-WilliamNeural"),
    _v("aussie_female", "Australian female voice with easy, friendly energy.",
       "female australian bright warm storytelling ads",
       edge="en-AU-NatashaNeural"),
    _v("kiwi_male", "New Zealand male voice, understated and dry.",
       "male nz calm playful commentary narration",
       edge="en-NZ-MitchellNeural"),
    _v("kiwi_female", "New Zealand female voice with a light, quick cadence.",
       "female nz bright warm storytelling podcast",
       edge="en-NZ-MollyNeural"),

    # -- Canadian --------------------------------------------------------- #
    _v("canadian_male", "Canadian male read, neutral and even-tempered.",
       "male canadian calm crisp narration tutorial",
       edge="en-CA-LiamNeural"),
    _v("canadian_female", "Canadian female read, clear and approachable.",
       "female canadian warm crisp narration tutorial",
       edge="en-CA-ClaraNeural"),

    # -- Indian / South Asian --------------------------------------------- #
    _v("indian_male", "Indian English male voice, precise and confident.",
       "male indian crisp calm narration news",
       edge="en-IN-PrabhatNeural"),
    _v("indian_female", "Indian English female voice with a clear, warm tone.",
       "female indian warm crisp narration tutorial",
       edge="en-IN-NeerjaNeural"),

    # -- African ---------------------------------------------------------- #
    _v("nigerian_male", "Nigerian English male voice, resonant and deliberate.",
       "male african nigerian deep warm narration storytelling",
       edge="en-NG-AbeoNeural"),
    _v("nigerian_female", "Nigerian English female voice with a rich, even tone.",
       "female african nigerian warm calm narration storytelling",
       edge="en-NG-EzinneNeural"),
    _v("kenyan_male", "Kenyan English male voice, steady and clear.",
       "male african kenyan calm crisp narration news",
       edge="en-KE-ChilembaNeural"),
    _v("kenyan_female", "Kenyan English female voice, bright and articulate.",
       "female african kenyan bright crisp narration podcast",
       edge="en-KE-AsiliaNeural"),
    _v("south_african_male", "South African English male voice with a grounded tone.",
       "male african south-african calm deep narration documentary",
       edge="en-ZA-LukeNeural"),
    _v("south_african_female", "South African English female voice, warm and level.",
       "female african south-african warm calm narration audiobook",
       edge="en-ZA-LeahNeural"),
    _v("tanzanian_male", "Tanzanian English male voice, measured and friendly.",
       "male african tanzanian calm warm narration tutorial",
       edge="en-TZ-ElimuNeural"),
    _v("tanzanian_female", "Tanzanian English female voice with a soft edge.",
       "female african tanzanian soft warm narration storytelling",
       edge="en-TZ-ImaniNeural"),

    # -- Asia-Pacific English --------------------------------------------- #
    _v("hongkong_male", "Hong Kong English male voice, brisk and businesslike.",
       "male asian hongkong crisp serious news tutorial",
       edge="en-HK-SamNeural"),
    _v("hongkong_female", "Hong Kong English female voice, clear and quick.",
       "female asian hongkong crisp bright news tutorial",
       edge="en-HK-YanNeural"),
    _v("singapore_male", "Singapore English male voice, neutral and crisp.",
       "male asian singaporean crisp calm narration tutorial",
       edge="en-SG-WayneNeural"),
    _v("singapore_female", "Singapore English female voice, light and precise.",
       "female asian singaporean bright crisp narration podcast",
       edge="en-SG-LunaNeural"),
    _v("filipino_male", "Filipino English male voice, easy and conversational.",
       "male asian filipino warm playful storytelling podcast",
       edge="en-PH-JamesNeural"),
    _v("filipino_female", "Filipino English female voice, friendly and bright.",
       "female asian filipino bright warm storytelling ads",
       edge="en-PH-RosaNeural"),

    # -- Character / utility ---------------------------------------------- #
    _v("chat_incoming", "Neutral voice for the other side of a text conversation.",
       "neutral american calm crisp chat storytelling",
       edge="en-US-EricNeural", piper="en_US-ryan-medium"),
    _v("chat_outgoing", "Neutral voice for the viewer's own side of a chat.",
       "neutral american warm crisp chat storytelling",
       edge="en-US-JennyNeural", piper="en_US-amy-medium"),
    _v("silent", "No speech at all: timed silence with synthetic word timings.",
       "neutral utility offline silence narration"),
)


def voice_names() -> list[str]:
    """Every catalogue name, in catalogue order."""
    return [v.name for v in VOICES]


def tag_values() -> list[str]:
    """Every tag used anywhere in the catalogue, sorted."""
    seen: set[str] = set()
    for voice in VOICES:
        seen.update(voice.tags)
    return sorted(seen)


_BY_NAME: dict[str, Voice] = {v.name: v for v in VOICES}
_BY_LOWER: dict[str, Voice] = {v.name.lower(): v for v in VOICES}


def list_voices(provider: str | None = None) -> list[Voice]:
    """The catalogue, filtered to entries ``provider`` can actually speak.

    ``None`` (or ``"auto"``) returns everything.  ``"offline"`` also returns
    everything -- the silent provider speaks any name.
    """
    if provider is None:
        return list(VOICES)
    canonical = canonical_provider(provider)
    if canonical in ("auto", ""):
        return list(VOICES)
    return [v for v in VOICES if v.supports(canonical)]


def _normalise_query(name: str) -> list[str]:
    cleaned = name.strip().lower().replace("-", " ").replace("_", " ").replace(",", " ")
    return [tok for tok in cleaned.split() if tok]


def _token_matches_tag(token: str, tag: str) -> bool:
    """One query token against one tag: exact, or a prefix of it.

    Prefix -- not substring.  ``"male" in "female"`` is true and would have made
    ``find_voice("male energetic")`` hand back a female voice; ``"female"``
    does not *start* with ``"male"``, so a prefix test keeps genders apart while
    still letting ``"brit"`` find ``"british"``.
    """
    return tag == token or tag.startswith(token)


def _tag_matches(tokens: Sequence[str], candidates: Iterable[Voice]) -> list[Voice]:
    """Voices whose tags cover every token in ``tokens``."""
    out: list[Voice] = []
    for voice in candidates:
        if all(any(_token_matches_tag(tok, tag) for tag in voice.tags) for tok in tokens):
            out.append(voice)
    return out


def find_voice_entry(name: str, *, provider: str | None = None) -> Voice:
    """Resolve ``name`` to a catalogue :class:`Voice`.

    Resolution order: exact name, case-insensitive name, then a tag query where
    every whitespace-separated token must match one of the voice's tags
    (``"british female calm"``).  ``provider`` narrows the candidate pool to
    voices that backend can speak.

    Raises :class:`~aiclipper.errors.TTSError` -- with the closest catalogue
    names attached -- when nothing matches.
    """
    raw = (name or "").strip()
    pool = list_voices(provider)
    if not raw:
        raise TTSError("no voice name given; pass a catalogue name such as 'narrator_deep'")

    voice = _BY_NAME.get(raw)
    if voice is not None and voice in pool:
        return voice

    voice = _BY_LOWER.get(raw.lower())
    if voice is not None and voice in pool:
        return voice

    tokens = _normalise_query(raw)
    if tokens:
        joined = "_".join(tokens)
        voice = _BY_LOWER.get(joined)
        if voice is not None and voice in pool:
            return voice
        matches = _tag_matches(tokens, pool)
        if matches:
            return matches[0]

    close = difflib.get_close_matches(raw.lower(), [v.name for v in pool], n=5, cutoff=0.45)
    if not close:
        close = [v.name for v in pool[:5]]
    where = f" for provider {canonical_provider(provider)!r}" if provider else ""
    raise TTSError(f"unknown voice {name!r}{where}. Closest matches: {', '.join(close)}")


def find_voice(
    name: str,
    *,
    provider: str | None = None,
    rate: float = 1.0,
    pitch_semitones: float = 0.0,
) -> VoiceSpec:
    """Resolve user input to a :class:`~aiclipper.models.VoiceSpec`.

    Accepts a catalogue name (``"narrator_deep"``, any case), a tag query
    (``"british female"``), or a ``provider:native_id`` string
    (``"edge:en-GB-RyanNeural"``) which is passed through untouched when the id
    is not one of ours.
    """
    raw = (name or "").strip()
    if ":" in raw:
        spec = VoiceSpec.parse(raw)
        try:
            entry = find_voice_entry(spec.voice_id, provider=spec.provider)
        except TTSError:
            return VoiceSpec(
                provider=spec.provider,
                voice_id=spec.voice_id,
                rate=rate,
                pitch_semitones=pitch_semitones,
            )
        return entry.to_spec(spec.provider, rate=rate, pitch_semitones=pitch_semitones)

    entry = find_voice_entry(raw, provider=provider)
    return entry.to_spec(provider, rate=rate, pitch_semitones=pitch_semitones)


def resolve_voice_id(voice: VoiceSpec | None, provider: str, *, default: str = "") -> str:
    """Native id for ``provider``, given whatever the caller put in ``voice``.

    * empty ``voice_id`` -> ``default``
    * a catalogue name -> that entry's id for ``provider`` (``default`` when the
      entry has none)
    * anything else -> returned verbatim, so native ids we never catalogued
      (``"en-GB-RyanNeural"``) keep working.
    """
    vid = (voice.voice_id if voice else "").strip()
    if not vid:
        return default
    entry = _BY_NAME.get(vid) or _BY_LOWER.get(vid.lower())
    if entry is None:
        return vid
    return entry.provider_id(provider) or default
