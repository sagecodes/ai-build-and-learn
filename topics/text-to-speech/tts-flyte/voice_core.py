"""The voice loop: mic -> STT -> LLM -> TTS, streamed sentence by sentence.

Flyte-free like tts_core and metrics, so the identical loop runs in the Flyte app pod
and in a local script.

── Why this streams the way it does ─────────────────────────────────────────────
The naive version (what `topics/gemma4/voice/app.py` does today) waits for the LLM to
finish, then synthesizes the whole reply, then plays it. Time-to-first-audio is
therefore the WHOLE generation plus the WHOLE synthesis: for a 60-token reply at the
14 tok/s measured on this box, that's 4+ seconds of silence before anything happens.

Instead we chunk the token stream at sentence boundaries and synthesize each chunk
while the next one is still generating. First audio lands after the first sentence
(~1s), and because the LLM generates ~4x faster than Kokoro speaks, generation stays
comfortably ahead of playback for the rest of the reply. The user hears a continuous
answer that started a second after they stopped talking.

Two constraints shape `SentenceChunker`:

  1. Gradio's streaming audio wants chunks "larger than 1 second" for smooth playback,
     so a bare "Sure." (~0.4s) must NOT be emitted alone: short sentences get merged
     forward until the chunk is worth speaking. That's `min_chars`.
  2. Chunks must break on sentence boundaries anyway, because TTS models need a full
     clause to get the prosody right. Splitting mid-sentence makes every model here
     sound robotic, which defeats the point of the comparison next door.

── The three layers are independent on purpose ──────────────────────────────────
STT, LLM and TTS are separate swappable registries rather than one fused pipeline.
That's what lets the LLM slot hold anything (Gemma, Qwen, a 1B) instead of only models
with a native audio encoder. It costs one extra model in memory (Whisper, ~1.6GB) and
buys a free choice of LLM.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile
import time
from dataclasses import dataclass, field

import numpy as np
import soundfile as sf

log = logging.getLogger("voice")

# ── Which LLM backend, and therefore which deployment shape ──────────────────────
#
# The two are not interchangeable at runtime; they are different DEPLOYMENTS, because
# the box has one GPU (one node, `nvidia.com/gpu: 1`, no time-slicing):
#
#   LLM_BACKEND=vllm    vLLM holds the GPU in its OWN app pod (the flyteplugins-vllm
#                       pattern from gemma4-dgx-devbox/vllm_server.py). This voice app
#                       is then CPU-only and talks to it over the OpenAI API. Fastest
#                       LLM (a quantized checkpoint like nvidia/Gemma-4-26B-A4B-NVFP4 is
#                       16.5GB, smaller than ollama's Q4, on better kernels) and it
#                       leaves the pipelines runnable. Costs: switching models is a cold
#                       start, and Chatterbox voice cloning is out (2.2x real-time on a
#                       GPU is far under 1x on CPU).
#
#   LLM_BACKEND=ollama  ollama runs INSIDE this pod, which therefore holds the GPU.
#                       Model swaps take seconds and TTS/STT get the GPU, so the cloned
#                       voice works. Costs: weaker serving, and the compare/clone
#                       pipelines queue behind this app while it is up.
#
# Everything above the LLM layer (STT, chunking, TTS, the UI) is identical either way,
# which is the point of routing both through stream_llm().
LLM_BACKEND = os.environ.get("LLM_BACKEND", "vllm").strip().lower()

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
# Set by the app from the deployed vLLM app's URL. The OpenAI client wants the /v1 root.
VLLM_URL = os.environ.get("VLLM_URL", "").rstrip("/")
VLLM_MODEL_ID = os.environ.get("VLLM_MODEL_ID", "gemma-4-26b-a4b-it")


def _device() -> str:
    """cuda when this pod actually has a GPU, else cpu.

    Auto-detected rather than configured, because the same code runs in the GPU pod
    (ollama backend) and the CPU pod (vllm backend) and should not need to be told.
    """
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

# Speech runs at roughly 15 characters per second, so ~60 chars is ~4s of audio: well
# clear of Gradio's 1s floor while still being one or two natural sentences.
MIN_CHUNK_CHARS = 60


# ── Sentence chunking ────────────────────────────────────────────────────────────

# Split AFTER .!? plus following quotes/brackets, when followed by whitespace. The
# lookbehind keeps the punctuation attached to the sentence it ends, so the TTS model
# still sees "Really?" and can put the question contour on it.
_SENT_END = re.compile(r'(?<=[.!?])["\')\]]*\s+')

# Things that end in "." but do not end a sentence. Without this the chunker cuts after
# "Dr." or "3." and the TTS model reads a fragment with falling final intonation.
_NOT_END = re.compile(
    r'(?:\b(?:mr|mrs|ms|dr|prof|sr|jr|st|vs|etc|e\.g|i\.e|approx|fig|no)\.|'
    r'\b[A-Z]\.|\d+\.)\s*$',
    re.I,
)


class SentenceChunker:
    """Accumulates a token stream and emits speakable chunks.

    feed() returns the chunks that are ready (often none), flush() returns whatever is
    left when the stream ends. A chunk is emitted only at a sentence boundary AND once
    it is long enough to be worth a separate audio segment.
    """

    def __init__(self, min_chars: int = MIN_CHUNK_CHARS):
        self.min_chars = min_chars
        self.buf = ""

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out: list[str] = []
        while True:
            chunk = self._take()
            if chunk is None:
                return out
            out.append(chunk)

    def _take(self) -> str | None:
        """Pop one complete, long-enough chunk off the front of the buffer."""
        pos = 0
        while True:
            m = _SENT_END.search(self.buf, pos)
            if not m:
                return None                     # no boundary yet; wait for more tokens
            head = self.buf[: m.start()]
            # An abbreviation is not a sentence end: keep scanning past it, so
            # "Dr. Chen agrees." emits once, not twice.
            if _NOT_END.search(head):
                pos = m.end()
                continue
            if len(head.strip()) < self.min_chars:
                pos = m.end()                   # too short to speak alone: merge forward
                continue
            self.buf = self.buf[m.end():]
            return head.strip()

    def flush(self) -> str | None:
        tail, self.buf = self.buf.strip(), ""
        return tail or None


# ── Speech to text ───────────────────────────────────────────────────────────────

# Whisper via transformers, NOT faster-whisper: metrics.py already runs exactly this
# stack on this box for the clone scoring, and CTranslate2 (faster-whisper's backend)
# has no reliable arm64+CUDA wheel here. Sizes are swappable at runtime; turbo is the
# accuracy/speed sweet spot and the same checkpoint the clone scorer uses, so it is
# usually already in the HF cache.
STT_MODELS: dict[str, str] = {
    "whisper-turbo": "openai/whisper-large-v3-turbo",
    "whisper-small": "openai/whisper-small.en",
    "whisper-base": "openai/whisper-base.en",
}
DEFAULT_STT = "whisper-turbo"


@dataclass
class Transcriber:
    """Whisper, loaded once per size and cached across turns."""
    device: str = field(default_factory=_device)
    _cache: dict = field(default_factory=dict, repr=False)

    def _pipe(self, key: str):
        if key not in self._cache:
            import torch
            from transformers import pipeline
            repo = STT_MODELS.get(key, key)
            # fp16 is a GPU trick; on CPU it is emulated and SLOWER than fp32, which
            # matters because in the vllm deployment this runs on CPU.
            dtype = torch.float16 if self.device == "cuda" else torch.float32
            log.info(f"[stt] loading {repo} on {self.device} ({dtype})")
            self._cache[key] = pipeline(
                "automatic-speech-recognition", model=repo,
                torch_dtype=dtype, device=self.device,
            )
        return self._cache[key]

    def transcribe(self, audio_path: str, key: str = DEFAULT_STT) -> tuple[str, float]:
        """(text, seconds). Returns ("", secs) for silence rather than raising, because
        an empty mic recording is a normal thing for a user to do."""
        t0 = time.time()
        wav, sr = sf.read(audio_path, dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if wav.size < sr * 0.2:                 # <200ms: a stray click, not speech
            return "", time.time() - t0
        if sr != 16000:
            import torch
            import torchaudio
            wav = torchaudio.functional.resample(
                torch.from_numpy(wav), sr, 16000).numpy()
        out = self._pipe(key)(
            wav, generate_kwargs={"language": "english", "task": "transcribe"})
        return (out.get("text") or "").strip(), time.time() - t0


# ── The LLM (ollama) ─────────────────────────────────────────────────────────────
#
# ollama rather than vLLM for one decisive reason: a model DROPDOWN. vLLM serves one
# model per process, so switching would mean restarting the server; ollama swaps on
# demand, keeps recently-used models resident, and evicts on idle. It also serves GGUF
# quantized weights, which matters more here than it looks: the Spark is memory-
# bandwidth-bound, so tokens/sec scales with bytes-read-per-token. Measured on this box,
# gemma4:26b at Q4 (18GB) decodes 14.1 tok/s; the same model in bf16 (~52GB) would read
# ~3x more per token and fall to roughly speech rate, which is too slow to stay ahead
# of playback.

# The dropdown. Ordered small-to-large so the latency difference is the first thing you
# see. Anything ollama can pull works; this is just the curated set.
LLM_MODELS: list[str] = [
    "qwen3:0.6b",       # the "how fast can this possibly get" end
    "qwen3:1.7b",
    "gemma4:e4b",       # Gemma's edge variant, ~4B effective
    "gemma4:12b",       # dense, natively multimodal
    "gemma4:26b",       # MoE, 4B active: measured 14.1 tok/s on this box
]
DEFAULT_LLM = "gemma4:26b"

VOICE_SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are spoken aloud, so: keep them "
    "to one to three short sentences, use plain conversational English, and never use "
    "markdown, bullet points, code blocks, emoji or headings. Write numbers and symbols "
    "the way a person would say them out loud."
)


def list_models(fallback: list[str] | None = None) -> list[str]:
    """What the active backend can actually serve, so the dropdown never lies.

    vLLM serves exactly one model per process, so its "dropdown" is a single entry:
    switching models there means deploying a different app, not picking from a menu.
    That asymmetry is the main reason to keep the ollama path around.
    """
    if LLM_BACKEND == "vllm":
        try:
            from openai import OpenAI
            client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="not-used")
            return [m.id for m in client.models.list().data] or [VLLM_MODEL_ID]
        except Exception as e:
            log.warning(f"[llm] could not list vllm models: {e}")
            return [VLLM_MODEL_ID]
    try:
        import ollama
        have = sorted(m.model for m in ollama.Client(host=OLLAMA_HOST).list().models)
        return have or (fallback or LLM_MODELS)
    except Exception as e:
        log.warning(f"[llm] could not list ollama models: {e}")
        return fallback or LLM_MODELS


# Back-compat alias: the app used to call this by its ollama-specific name.
list_ollama_models = list_models


def stream_llm(messages: list[dict], model: str = DEFAULT_LLM,
               temperature: float = 0.6):
    """Yield content deltas from whichever backend is configured.

    THE seam. Everything else in this module (STT, chunking, TTS, the loop) is backend-
    agnostic, so swapping vLLM for ollama touches only this function. Reasoning tokens
    are off in both paths: a voice assistant that thinks for 400 tokens before speaking
    is a voice assistant with four seconds of dead air, even though the text chat app
    next door exposes thinking as a slider.
    """
    if LLM_BACKEND == "vllm":
        yield from _stream_vllm(messages, model, temperature)
    else:
        yield from _stream_ollama(messages, model, temperature)


def _stream_vllm(messages: list[dict], model: str, temperature: float):
    """The OpenAI-compatible path: the same client shape gemma4's chat_app.py uses."""
    from openai import OpenAI

    if not VLLM_URL:
        raise RuntimeError(
            "LLM_BACKEND=vllm but VLLM_URL is unset. Deploy the vLLM app first "
            "(python voice_vllm.py) and pass its URL through VLLM_URL.")
    client = OpenAI(base_url=f"{VLLM_URL}/v1", api_key="not-used")
    stream = client.chat.completions.create(
        model=model or VLLM_MODEL_ID, messages=messages, stream=True,
        temperature=float(temperature), max_tokens=512,
    )
    for part in stream:
        if not part.choices:
            continue
        if delta := (part.choices[0].delta.content or ""):
            yield delta


def _stream_ollama(messages: list[dict], model: str, temperature: float):
    import ollama

    stream = ollama.Client(host=OLLAMA_HOST).chat(
        model=model, messages=messages, stream=True, think=False,
        options={"temperature": float(temperature)},
    )
    try:
        for part in stream:
            if delta := ((part.get("message") or {}).get("content") or ""):
                yield delta
    finally:
        if close := getattr(stream, "close", None):
            close()


# ── Text to speech ───────────────────────────────────────────────────────────────
#
# The engines come from the demo next door (models.py / tts_core.py), so the voice app
# and the comparison report are speaking through the exact same adapters. Which ones are
# actually offered is decided at RUNTIME by what imports, because the TTS packages have
# mutually hostile pins and cannot all share one image (see config.py). An engine whose
# package is missing from this image simply does not appear in the dropdown, rather than
# appearing and then failing on click.

# Real-time factors measured on this box (see the compare demo's README): Kokoro 78x,
# Chatterbox 2.2x, Qwen3-TTS 1.4x, Dia 0.5x. Only Kokoro is comfortably faster than
# speech, so only Kokoro truly streams; the rest are offered so you can HEAR the
# difference, with the UI warning that they will lag.
TTS_ENGINES: dict[str, dict] = {
    "kokoro-82m":  {"realtime": True,  "note": "78x real-time. Streams smoothly."},
    "chatterbox":  {"realtime": False, "note": "2.2x real-time. Clones your voice; expect a lag."},
    "qwen3-1.7b":  {"realtime": False, "note": "1.4x real-time. Expressive but laggy."},
}
DEFAULT_TTS = "kokoro-82m"

# Kokoro's voicepacks, the ones worth putting in a dropdown.
KOKORO_VOICES = ["af_heart", "af_bella", "af_nicole", "am_michael", "am_fenrir",
                 "bf_emma", "bm_george"]


def available_tts() -> list[str]:
    """Engines whose package actually imports in THIS image."""
    import importlib.util
    probes = {"kokoro-82m": "kokoro", "chatterbox": "chatterbox", "qwen3-1.7b": "qwen_tts"}
    out = [k for k, mod in probes.items() if importlib.util.find_spec(mod) is not None]
    return out or [DEFAULT_TTS]


@dataclass
class Speaker:
    """Holds one loaded TTS engine, swapping only when the selection changes.

    Reloading a model per sentence would dwarf the synthesis itself, so the handle is
    cached and only torn down when the user picks a different engine.
    """
    ref_wav: str = ""              # a reference clip enables the cloned voice
    ref_text: str = ""
    _key: str = ""
    _handle: object = field(default=None, repr=False)
    _spec: object = field(default=None, repr=False)
    _ref: object = field(default=None, repr=False)

    def _load(self, key: str, voice: str):
        import tts_core
        from models import get_spec

        spec = get_spec(key)
        if voice and key == "kokoro-82m":
            from dataclasses import replace
            spec = replace(spec, voice=voice, voice_label=voice)

        if self._key == key and self._handle is not None:
            self._spec = spec          # a voicepack change needs no reload
            return
        if self._handle is not None:
            try:
                tts_core.close_model(self._spec, self._handle)
            except Exception:
                log.exception("[tts] close_model failed")
            self._handle = None
            tts_core.free_gpu_memory()

        log.info(f"[tts] loading {key}")
        self._handle = tts_core.load_model(spec)
        self._spec, self._key = spec, key

        if self.ref_wav and self._ref is None:
            self._ref = tts_core.RefVoice.from_file(self.ref_wav, self.ref_text)

    def say(self, text: str, key: str = DEFAULT_TTS, voice: str = "",
            clone: bool = False) -> tuple[np.ndarray, int, float]:
        import tts_core
        self._load(key, voice)
        if clone and self._ref is not None and getattr(self._spec, "clone_capable", False):
            return tts_core.synth_clone(self._handle, self._spec, text, self._ref)
        return tts_core.synth_one(self._handle, self._spec, text)

    def say_to_file(self, text: str, out_dir: str, idx: int, **kw) -> tuple[str, float, float]:
        """Synthesize one chunk to a wav on disk. Gradio's streaming audio output takes
        a .wav/.mp3 path (or bytes) per chunk, so this is the unit the UI yields."""
        import tts_core
        wav, sr, secs = self.say(text, **kw)
        path = os.path.join(out_dir, f"chunk_{idx:03d}.wav")
        tts_core.write_wav(wav, sr, path)
        return path, (wav.size / float(sr) if sr else 0.0), secs


# ── The loop ─────────────────────────────────────────────────────────────────────

@dataclass
class Turn:
    """What one exchange produced, for the UI and the latency readout."""
    user_text: str = ""
    reply: str = ""
    stt_seconds: float = 0.0
    first_audio_seconds: float = 0.0   # mic-stop -> first playable chunk. THE number.
    total_seconds: float = 0.0
    audio_seconds: float = 0.0
    chunks: int = 0
    error: str = ""

    @property
    def realtime_factor(self) -> float:
        """Audio produced per second of wall clock. Above 1.0 means synthesis outran
        playback, i.e. the listener never heard a gap."""
        return self.audio_seconds / self.total_seconds if self.total_seconds else 0.0


def converse(audio_path: str, history: list[dict], speaker: Speaker,
             transcriber: Transcriber, *, llm: str = DEFAULT_LLM,
             stt: str = DEFAULT_STT, tts: str = DEFAULT_TTS, voice: str = "",
             clone: bool = False, system: str = VOICE_SYSTEM_PROMPT,
             temperature: float = 0.6, min_chars: int = MIN_CHUNK_CHARS):
    """One full turn, yielding (Turn, audio_chunk_path_or_None) as it goes.

    The generator shape is what makes the UI stream: every yield is a chance for Gradio
    to push another audio chunk to the browser and update the transcript, so the reply
    is heard as it is written rather than after it.
    """
    t_start = time.time()
    turn = Turn()

    turn.user_text, turn.stt_seconds = transcriber.transcribe(audio_path, stt)
    if not turn.user_text:
        turn.error = "Didn't catch that: the recording was silent or too short."
        turn.total_seconds = time.time() - t_start
        yield turn, None
        return
    yield turn, None

    messages = [{"role": "system", "content": system}, *history,
                {"role": "user", "content": turn.user_text}]

    out_dir = tempfile.mkdtemp(prefix="voice_turn_")
    chunker = SentenceChunker(min_chars=min_chars)

    def _speak(text: str):
        """Synthesize one chunk and hand it to the UI, recording time-to-first-audio."""
        path, dur, _secs = speaker.say_to_file(
            text, out_dir, turn.chunks, key=tts, voice=voice, clone=clone)
        turn.chunks += 1
        turn.audio_seconds += dur
        if turn.first_audio_seconds == 0.0:
            turn.first_audio_seconds = time.time() - t_start
        return path

    try:
        for delta in stream_llm(messages, model=llm, temperature=temperature):
            turn.reply += delta
            spoke = False
            for chunk in chunker.feed(delta):
                yield turn, _speak(chunk)
                spoke = True
            if not spoke:
                # No chunk was ready, but yield anyway so the transcript stays live:
                # the text should appear as it generates, not only when audio does.
                yield turn, None

        if tail := chunker.flush():
            yield turn, _speak(tail)
    except Exception as e:
        log.exception("[turn] failed")
        turn.error = f"{type(e).__name__}: {e}"

    turn.total_seconds = time.time() - t_start
    yield turn, None
