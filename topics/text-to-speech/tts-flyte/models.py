"""Registry of open-source text-to-speech models we compare on Flyte.

Each entry is a `TTSModelSpec`: which HuggingFace repo(s) to pull, which *adapter*
(inference package) loads and drives it, and the per-model defaults that make a fair
"read the same script" comparison possible.

── Why an `adapter` field, and no shared pipeline class ─────────────────────────
The video demo could name a single diffusers pipeline class per model because every
model loaded the same way. TTS has no such common runtime: Qwen loads through the
`qwen-tts` package, Kokoro through `kokoro`, Chatterbox through `chatterbox-tts`, Dia
through transformers. So instead of a `pipeline` string, a spec names an `adapter`,
and tts_core.py has one loader/synth function per adapter. The adapter also decides
which image and which GPU TaskEnvironment the model runs in (see config.GPU_ENVS).

── What we're actually comparing ────────────────────────────────────────────────
This is the PURE-GENERATION task: every model reads the same text in its own default
/ built-in voice. Voice cloning is a separate task (it needs a reference clip and a
different fairness setup), so there is no clone plumbing here on purpose.

The axes that separate these models in the report:
  - naturalness / expressiveness  (the thing you judge by ear)
  - speed, as real-time factor    (synth seconds per second of audio; lower is faster)
  - size / hardware footprint     (82M CPU-friendly Kokoro vs a 1.7B transformer)
  - license                       (all of these are Apache-2.0 or MIT: commercial-safe)

── Sizes are on-disk download sizes ─────────────────────────────────────────────
`download_gb` is the measured size of the repo(s) we fetch, from the HF API. These
are all small enough that none of the video demo's memory-guard machinery is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace


@dataclass(frozen=True)
class Voice:
    """A named built-in voice a model can render, with a gender for M/F selection.

    `voice_id` is whatever the adapter's synth call wants: a Kokoro voicepack id
    (`af_heart`), a Qwen speaker name (`Vivian`), or, for Parler, a whole natural-
    language voice description. `label` is the short human name shown in the report
    (defaults to voice_id; set it for Parler so the card says "Laura", not a sentence).
    """
    suffix: str          # key suffix for the variant column, e.g. "f"
    gender: str          # "female" | "male"
    voice_id: str        # adapter-specific voice handle
    label: str = ""      # display name; falls back to voice_id

    def disp(self) -> str:
        return f"{self.label or self.voice_id} · {self.gender}"


@dataclass(frozen=True)
class TTSModelSpec:
    key: str                       # short handle used on the CLI and in reports
    repo: str                      # primary HuggingFace repo id
    adapter: str                   # which tts_core loader/synth drives it + which image
    family: str                    # backbone family, for the report
    license: str
    params: str                    # human string, e.g. "1.7B", "82M"
    download_gb: float             # measured size of what we fetch
    sample_rate: int               # native output rate (Hz)

    # Extra repos the adapter's loader will also request (e.g. Qwen's separate speech
    # tokenizer). Fetched alongside `repo` so the warm HF cache has everything.
    extra_repos: tuple[str, ...] = ()

    # Per-model synthesis defaults. Not every field applies to every adapter; each
    # adapter reads the ones it needs.
    voice: str = ""                # default/single voice (Kokoro pack / Qwen speaker / Parler desc)
    voice_label: str = ""          # display name for the voice in the report (set on expansion)
    voices: tuple[Voice, ...] = () # named M/F variants; empty = single default voice
    language: str = "English"      # Qwen language arg; Kokoro maps it to a lang_code
    dtype: str = "bfloat16"        # GB10 (Blackwell) is happiest in bf16
    speaker_tagged: bool = False   # Dia: text must be [S1]/[S2] tagged

    gated: bool = False
    native: str = ""               # what the model card actually recommends
    notes: str = ""

    # Selective download. Left empty = fetch the whole repo (fine for models this size).
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()

    @property
    def all_repos(self) -> tuple[str, ...]:
        return (self.repo, *self.extra_repos)


# Junk that is never load-bearing for a loader, in every repo.
_JUNK = ("*.md", "*.gif", "*.mp4", "*.png", "*.jpg", "assets/*", "examples/*", "samples/*")


SPECS: list[TTSModelSpec] = [
    # ── The flagship the stream is built around ──────────────────────────────────
    TTSModelSpec(
        key="qwen3-1.7b",
        repo="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        extra_repos=("Qwen/Qwen3-TTS-Tokenizer-12Hz",),
        adapter="qwen",
        family="Qwen3-TTS (12Hz)",
        license="Apache-2.0",
        params="1.7B",
        download_gb=5.2,
        sample_rate=24000,
        voice="Ryan",           # one of: Vivian Serena Uncle_Fu Dylan Eric Ryan Aiden Ono_Anna Sohee
        voices=(Voice("f", "female", "Vivian"), Voice("m", "male", "Ryan")),
        language="English",
        native="10 languages, streaming to ~97ms latency; expressive built-in speakers.",
        notes="Alibaba's Jan-2026 flagship. Expressive, multilingual, Apache-2.0. "
              "The one to beat here.",
    ),
    # ── Same model family, a third the size: the size/speed contrast ──────────────
    TTSModelSpec(
        key="qwen3-0.6b",
        repo="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
        extra_repos=("Qwen/Qwen3-TTS-Tokenizer-12Hz",),
        adapter="qwen",
        family="Qwen3-TTS (12Hz)",
        license="Apache-2.0",
        params="0.6B",
        download_gb=3.2,
        sample_rate=24000,
        voice="Eric",
        voices=(Voice("f", "female", "Serena"), Voice("m", "male", "Eric")),
        language="English",
        native="Lightweight Qwen3-TTS; same API as the 1.7B, ~4GB VRAM.",
        notes="The lightweight sibling. Listen for where the smaller model gives up "
              "expressiveness for speed.",
    ),
    # ── The cloning-capable one everyone benchmarks against ElevenLabs ────────────
    TTSModelSpec(
        key="chatterbox",
        repo="ResembleAI/chatterbox",
        adapter="chatterbox",
        family="Chatterbox (Llama + S3Gen)",
        license="MIT",
        params="0.5B",
        download_gb=13.9,       # whole repo; the base English weights are a subset
        sample_rate=24000,      # overridden at runtime by model.sr
        native="Emotion-exaggeration control; 10s zero-shot voice clone. Beat ElevenLabs "
               "in a blind test (65% preference).",
        notes="MIT, and the natural-speech bar for open TTS. We use its DEFAULT voice "
              "here; cloning is the separate task.",
    ),
    # ── The efficiency baseline: 82M, CPU-capable, no cloning ─────────────────────
    TTSModelSpec(
        key="kokoro-82m",
        repo="hexgrad/Kokoro-82M",
        adapter="kokoro",
        family="Kokoro (StyleTTS2-ish)",
        license="Apache-2.0",
        params="82M",
        download_gb=0.36,
        sample_rate=24000,
        voice="af_heart",       # American-English female; see the repo's voices/ dir
        voices=(Voice("f", "female", "af_heart"), Voice("m", "male", "am_michael")),
        language="English",
        native="36x real-time on a T4, 5x on CPU. 54 baked-in voices, no cloning.",
        notes="The speed/efficiency floor. Fixed voicepacks only. If quality is 'good "
              "enough' here, it's the cheapest thing to ship.",
    ),
    # ── The expressive-dialogue outlier: a different capability entirely ──────────
    TTSModelSpec(
        key="dia-1.6b",
        repo="nari-labs/Dia-1.6B-0626",
        adapter="dia",
        family="Dia (dialogue TTS)",
        license="Apache-2.0",
        params="1.6B",
        download_gb=6.4,
        sample_rate=44100,
        speaker_tagged=True,    # text is [S1]/[S2] tagged; the adapter adds [S1] if absent
        native="Multi-speaker dialogue with nonverbals (laughs, sighs). Begin with [S1], "
               "alternate [S1]/[S2].",
        notes="Not a like-for-like single-voice model: it's the best open PODCAST/dialogue "
              "voice. Shines on the two-speaker script.",
    ),
    # ── Voice controlled by a text DESCRIPTION: a different UX entirely ────────────
    TTSModelSpec(
        key="parler-mini",
        repo="parler-tts/parler-tts-mini-v1",
        adapter="parler",
        family="Parler-TTS (mini v1)",
        license="Apache-2.0",
        params="880M",
        download_gb=2.5,
        sample_rate=44100,
        # For Parler the "voice" IS a natural-language description. Named speakers
        # (Laura, Gary, Jon, Lea, Mike, Jenna...) give a consistent voice; gender and
        # style are just words in the prompt. That makes M/F trivial: swap the name.
        voice="Laura's voice is expressive and clear, with very high audio quality and no background noise.",
        voices=(
            Voice("f", "female",
                  "Laura's voice is expressive and clear, with very high audio quality and no background noise.",
                  label="Laura"),
            Voice("m", "male",
                  "Gary's voice is expressive and clear, with very high audio quality and no background noise.",
                  label="Gary"),
        ),
        native="Voice/style/pace/reverb all controlled by an English description prompt.",
        notes="The odd one out: you describe the voice in words. Great for showing "
              "prompt-controlled M/F without any cloning.",
    ),
    # ── Sesame's conversational model: very natural, LLM-based ─────────────────────
    TTSModelSpec(
        key="csm-1b",
        repo="sesame/csm-1b",
        adapter="csm",
        family="Sesame CSM (Llama backbone)",
        license="Apache-2.0",
        params="1B",
        download_gb=6.2,
        sample_rate=24000,
        # CSM's speaker is an id ([0]/[1]); without an audio context the voice is
        # model-chosen, so no controllable M/F here (that's the cloning task). Single voice.
        native="Conversational speech codec model; speaker id [0]/[1], best with context.",
        notes="Sesame's natural conversational voice. No named M/F voices without an "
              "audio prompt, so it runs single-voice here.",
    ),
]


MODELS: dict[str, TTSModelSpec] = {s.key: s for s in SPECS}

# The default comparison set: everything commercial-safe, one per capability niche.
DEFAULT_MODELS: list[str] = [
    "qwen3-1.7b", "qwen3-0.6b", "chatterbox", "kokoro-82m", "dia-1.6b",
    "parler-mini", "csm-1b",
]


def get_spec(key: str) -> TTSModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(
            f"Unknown model {key!r}. Known: {', '.join(MODELS)}"
        ) from None


def resolve_models(models: list[str] | None) -> list[TTSModelSpec]:
    """A list of keys (or None for the default set) -> the specs, order preserved."""
    keys = models or DEFAULT_MODELS
    return [get_spec(k) for k in keys]


def jobs_for(spec: TTSModelSpec, gender: str = "all") -> list[tuple[str, str, str]]:
    """Expand one model into its voice-variant columns.

    Returns (variant_key, voice_id, voice_label) per voice. A model with no named
    voices yields a single default job. `gender` filters to "female"/"male" (a model
    with no matching named voice falls back to its default, so it never drops out of
    the grid entirely).
    """
    if not spec.voices:
        return [(spec.key, spec.voice, spec.voice_label or (spec.voice or "default"))]
    variants = spec.voices
    if gender in ("female", "male"):
        variants = tuple(v for v in variants if v.gender == gender) or spec.voices
    return [(f"{spec.key}-{v.suffix}", v.voice_id, v.disp()) for v in variants]


def render_spec(base: TTSModelSpec, variant_key: str, voice_id: str, voice_label: str) -> TTSModelSpec:
    """A spec whose `.key` is the variant column id, for the report renderer."""
    return replace(base, key=variant_key, voice=voice_id, voice_label=voice_label)


if __name__ == "__main__":
    for s in SPECS:
        print(f"{s.key:14s} {s.params:>5s}  {s.download_gb:5.1f}GB  {s.license:11s}  "
              f"{s.adapter:11s}  {s.family}")
