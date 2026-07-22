"""The engine room: load a TTS model, synthesize, embed audio in a Flyte report.

Deliberately Flyte-free, so the exact same code runs inside the GPU task, the Gradio
studio, and run_local.py. If audio plays in the standalone HTML run_local writes, it
plays in the Flyte report, because it's the identical renderer.

Two things carry over from the video demo and matter here too:

  1. Flyte reports render under a CSP that drops external assets and <script> tags.
     Audio still plays, because HTML5 <audio> needs no JS: a base64 WAV in a data URI
     on a <audio controls> element is enough. That's `audio_data_uri`.
  2. The report needs a *visual* comparison surface, not just players, the way the
     video grid had a frame strip. For audio that's a waveform + spectrogram PNG:
     you can see clipping, silence, breath pauses and pitch contour at a glance, and
     it's the fallback if a player ever fails. That's `waveform_spectrogram_png`.

── One loader/synth per adapter ─────────────────────────────────────────────────
Unlike the video demo's single diffusers path, every TTS package loads and generates
differently, so `load_model` / `synth_one` dispatch on `spec.adapter`. Each adapter
imports its package LAZILY (inside the function), so an image that lacks a given
package, e.g. the Qwen image has no `kokoro`, never fails at import time.
"""

from __future__ import annotations

import base64
import gc
import html
import io
import re
import time
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")               # headless: no display, render straight to PNG bytes
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# Past this the <audio> data URI is dropped and only the spectrogram is shown. A ~20s
# 24kHz PCM16 clip is ~1MB, so this is generous headroom, not a real constraint.
MAX_EMBED_BYTES = 6_000_000


# ── GPU helpers (import-safe on a CPU-only host) ─────────────────────────────────

def free_gpu_memory() -> None:
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def reset_peak_memory() -> None:
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def peak_memory_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def prepare_gpu() -> None:
    """Clean the pool and reset the peak counter so a run's reported peak is its own.

    No preflight fit check: these models are tiny (the biggest is ~4GB in bf16) next
    to the GB10's unified pool, so the video demo's fit machinery would be theater.
    """
    free_gpu_memory()
    reset_peak_memory()


# ── Audio utilities ──────────────────────────────────────────────────────────────

def to_mono_float32(wav) -> np.ndarray:
    """Coerce whatever an adapter returns (numpy or torch, any channel layout) to a
    1-D float32 mono array. Adapters return maddeningly varied shapes; normalize once."""
    try:
        import torch
        if isinstance(wav, torch.Tensor):
            wav = wav.detach().to(torch.float32).cpu().numpy()
    except Exception:
        pass
    wav = np.asarray(wav, dtype=np.float32)
    wav = np.squeeze(wav)
    if wav.ndim == 2:
        # (channels, samples) or (samples, channels); average to mono either way.
        ch_axis = 0 if wav.shape[0] < wav.shape[1] else 1
        wav = wav.mean(axis=ch_axis)
    return np.ascontiguousarray(wav.reshape(-1))


def write_wav(wav: np.ndarray, sr: int, path) -> None:
    sf.write(str(path), to_mono_float32(wav), sr, subtype="PCM_16")


def audio_data_uri(wav: np.ndarray, sr: int, budget: int = MAX_EMBED_BYTES) -> tuple[str, str]:
    """(data_uri, note). An empty uri + a note means 'too big, spectrogram only'.

    Prefer OGG/Vorbis: it's ~10x smaller than PCM WAV and plays natively in <audio>,
    which keeps a many-column grid (7 models x M/F x N scripts) light enough to embed.
    Fall back to WAV if the image's libsndfile lacks the Vorbis encoder.
    """
    wav = to_mono_float32(wav)
    for fmt, sub, mime in (("OGG", "VORBIS", "audio/ogg"), ("WAV", "PCM_16", "audio/wav")):
        try:
            buf = io.BytesIO()
            sf.write(buf, wav, sr, format=fmt, subtype=sub)
            raw = buf.getvalue()
            if len(raw) > budget:
                return "", f"clip is {len(raw)/1e6:.1f} MB, over the embed budget; spectrogram shown"
            return f"data:{mime};base64,{base64.b64encode(raw).decode()}", ""
        except Exception:
            continue
    return "", "could not encode audio for embedding"


def waveform_spectrogram_png(wav: np.ndarray, sr: int) -> str:
    """A stacked waveform + spectrogram as a base64 PNG data URI.

    The audio analogue of the video demo's frame strip: the at-a-glance visual where
    clipping, dead air, breath pauses and pitch contour are all legible, and the
    fallback surface if a player fails. Uses matplotlib's own specgram, so no librosa.
    """
    wav = to_mono_float32(wav)
    if wav.size == 0:
        wav = np.zeros(int(sr * 0.1), dtype=np.float32)
    t = np.arange(wav.size) / float(sr)

    fig, (ax_w, ax_s) = plt.subplots(
        2, 1, figsize=(4.4, 2.4), dpi=110, gridspec_kw={"height_ratios": [1, 1.4]}
    )
    ax_w.plot(t, wav, linewidth=0.5, color="#6366f1")
    ax_w.set_xlim(0, max(t[-1], 0.1))
    ax_w.set_ylim(-1.05, 1.05)
    ax_w.set_yticks([])
    ax_w.set_xticks([])
    ax_w.margins(0)

    nfft = 512 if wav.size >= 512 else max(32, 1 << int(np.log2(max(wav.size, 32))))
    ax_s.specgram(wav, NFFT=nfft, Fs=sr, noverlap=nfft // 2, cmap="magma")
    ax_s.set_ylim(0, min(sr / 2, 8000))          # voice energy lives under ~8kHz
    ax_s.set_yticks([])
    ax_s.set_xlabel("seconds", fontsize=7, color="#6b7280")
    ax_s.tick_params(axis="x", labelsize=6, colors="#6b7280")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.98, bottom=0.14, hspace=0.08)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


# ── Text prep ────────────────────────────────────────────────────────────────────

_TAG = re.compile(r"\[S\d+\]")
_NONVERBAL = re.compile(r"\((?:laughs?|sighs?|coughs?|clears throat|gasps?)\)", re.I)


def clean_for_plain(text: str) -> str:
    """Strip [S1]/[S2] speaker tags and (laughs)-style nonverbals for single-voice
    models, so they read the dialogue script as one clean narrator instead of saying
    'bracket S one' out loud. Dia keeps the tags; everyone else gets this."""
    text = _TAG.sub(" ", text)
    text = _NONVERBAL.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Adapters: one loader + one synth per package ─────────────────────────────────
#
# load_model(spec)          -> an opaque handle (model, pipeline, or (processor, model))
# synth_one(handle, spec, t)-> (wav float32 mono, sample_rate, synth_seconds)

_KOKORO_LANG = {"English": "a", "British": "b", "Spanish": "e", "French": "f",
                "Hindi": "h", "Italian": "i", "Japanese": "j", "Portuguese": "p",
                "Chinese": "z"}

# ── Voxtral: the served model ─────────────────────────────────────────────────────
# Voxtral loads through a vLLM-omni server, not from_pretrained. The adapter starts the
# server as a subprocess, waits for its audio API, then POSTs text to /v1/audio/speech.
# The two-stage config (acoustic transformer + audio tokenizer) is embedded here so it
# rides along in the code bundle rather than as a stray .yaml. Stage-0 gpu util is 0.5
# so both stages fit the GB10's unified pool (from the voxtral/ Spark config).
VOXTRAL_STAGE_YAML = """async_chunk: true
stage_args:
  - stage_id: 0
    stage_type: llm
    runtime: {process: true, devices: "0"}
    engine_args:
      max_num_seqs: 32
      model_stage: audio_generation
      model_arch: VoxtralTTSForConditionalGeneration
      worker_type: ar
      worker_cls: vllm_omni.worker.gpu_ar_worker.GPUARWorker
      scheduler_cls: vllm_omni.core.sched.omni_ar_scheduler.OmniARScheduler
      gpu_memory_utilization: 0.5
      enforce_eager: false
      trust_remote_code: true
      async_scheduling: true
      engine_output_type: latent
      enable_prefix_caching: false
      tokenizer_mode: mistral
      config_format: mistral
      load_format: mistral
      skip_mm_profiling: true
      enable_chunked_prefill: false
      max_model_len: 4096
      custom_process_next_stage_input_func: vllm_omni.model_executor.stage_input_processors.voxtral_tts.generator2tokenizer_async_chunk
    output_connectors: {to_stage_1: connector_of_shared_memory}
    is_comprehension: true
    final_output: false
    final_output_type: text
    default_sampling_params: {temperature: 0.0, top_p: 1.0, top_k: -1, max_tokens: 2048, seed: 42, detokenize: True, repetition_penalty: 1.1}
  - stage_id: 1
    stage_type: llm
    runtime: {process: true, devices: "0"}
    engine_args:
      max_num_seqs: 32
      model_stage: audio_tokenizer
      model_arch: VoxtralTTSForConditionalGeneration
      worker_type: generation
      worker_cls: vllm_omni.worker.gpu_generation_worker.GPUGenerationWorker
      scheduler_cls: vllm_omni.core.sched.omni_generation_scheduler.OmniGenerationScheduler
      async_scheduling: false
      gpu_memory_utilization: 0.1
      enforce_eager: true
      trust_remote_code: true
      enable_prefix_caching: false
      skip_mm_profiling: true
      engine_output_type: audio
      tokenizer_mode: mistral
      config_format: mistral
      load_format: mistral
      max_num_batched_tokens: 65536
      max_model_len: 65536
    engine_input_source: [0]
    is_comprehension: false
    final_output: true
    final_output_type: audio
    input_connectors: {from_stage_0: connector_of_shared_memory}
    tts_args: {max_instructions_length: 500}
    default_sampling_params: {temperature: 0.9, top_p: 0.8, top_k: 40, max_tokens: 2048, seed: 42, detokenize: True, repetition_penalty: 1.05}
runtime:
  enabled: true
  defaults: {window_size: -1, max_inflight: 1}
  connectors:
    connector_of_shared_memory:
      name: SharedMemoryConnector
      extra: {shm_threshold_bytes: 65536, codec_streaming: true, connector_get_sleep_s: 0.01, connector_get_max_wait_first_chunk: 3000, connector_get_max_wait: 300, codec_chunk_frames: 25, codec_chunk_frames_at_begin: 5, codec_left_context_frames: 25}
  edges:
    - {from: 0, to: 1, window_size: -1}
"""


def _start_voxtral_server(spec, port: int = 8000, ready_timeout: int = 900):
    """Start vllm-omni serving Voxtral and block until its audio API answers.

    Returns {"proc", "base"}. Raises if the server exits early or never gets ready.
    """
    import subprocess
    import tempfile
    import time as _t

    import httpx

    yaml_path = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    yaml_path.write(VOXTRAL_STAGE_YAML)
    yaml_path.close()

    base = f"http://127.0.0.1:{port}"
    # Inherit stdout/stderr so vLLM's logs land in the pod log (essential for debugging
    # a server that won't start). This is the noisiest adapter by far.
    proc = subprocess.Popen(
        ["vllm-omni", "serve", spec.repo, "--omni",
         "--stage-configs-path", yaml_path.name, "--host", "127.0.0.1", "--port", str(port)]
    )
    deadline = _t.time() + ready_timeout
    while _t.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm-omni exited early with code {proc.returncode}")
        try:
            if httpx.get(f"{base}/v1/audio/voices", timeout=5).status_code == 200:
                return {"proc": proc, "base": base}
        except Exception:
            pass
        _t.sleep(4)
    proc.kill()
    raise RuntimeError(f"vllm-omni server not ready within {ready_timeout}s")


def load_model(spec):
    if spec.adapter == "qwen":
        import torch
        from qwen_tts import Qwen3TTSModel
        # attn_implementation="sdpa": flash-attn has no cu130 arm64 wheel, and sdpa is
        # plenty fast for a 1.7B model. The card's example uses flash_attention_2; don't.
        return Qwen3TTSModel.from_pretrained(
            spec.repo, device_map="cuda:0",
            dtype=getattr(torch, spec.dtype), attn_implementation="sdpa",
        )

    if spec.adapter == "kokoro":
        from kokoro import KPipeline
        return KPipeline(lang_code=_KOKORO_LANG.get(spec.language, "a"))

    if spec.adapter == "chatterbox":
        from chatterbox.tts import ChatterboxTTS
        return ChatterboxTTS.from_pretrained(device="cuda")

    if spec.adapter == "dia":
        import torch
        from transformers import AutoProcessor, DiaForConditionalGeneration
        processor = AutoProcessor.from_pretrained(spec.repo)
        model = DiaForConditionalGeneration.from_pretrained(
            spec.repo, torch_dtype=getattr(torch, spec.dtype)
        ).to("cuda")
        return (processor, model)

    if spec.adapter == "csm":
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration
        processor = AutoProcessor.from_pretrained(spec.repo)
        model = CsmForConditionalGeneration.from_pretrained(
            spec.repo, device_map="cuda", torch_dtype=getattr(torch, spec.dtype)
        )
        return (processor, model)

    if spec.adapter == "parler":
        import torch
        from parler_tts import ParlerTTSForConditionalGeneration
        from transformers import AutoTokenizer
        model = ParlerTTSForConditionalGeneration.from_pretrained(spec.repo).to("cuda")
        tokenizer = AutoTokenizer.from_pretrained(spec.repo)
        return (model, tokenizer)

    if spec.adapter == "voxtral":
        return _start_voxtral_server(spec)

    raise ValueError(f"No adapter for {spec.adapter!r}")


def close_model(spec, handle) -> None:
    """Release a loaded model. No-op for in-process models (GC + free_gpu handle those);
    for Voxtral it terminates the vLLM-omni server subprocess."""
    if spec.adapter == "voxtral" and isinstance(handle, dict) and handle.get("proc"):
        proc = handle["proc"]
        try:
            proc.terminate()
            proc.wait(timeout=20)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass


def synth_one(handle, spec, text: str) -> tuple[np.ndarray, int, float]:
    if spec.adapter == "qwen":
        t0 = time.time()
        wavs, sr = handle.generate_custom_voice(
            text=clean_for_plain(text), language=spec.language, speaker=spec.voice
        )
        return to_mono_float32(wavs[0]), int(sr), time.time() - t0

    if spec.adapter == "kokoro":
        t0 = time.time()
        chunks = [audio for (_g, _p, audio) in handle(clean_for_plain(text), voice=spec.voice, speed=1)]
        wav = np.concatenate([to_mono_float32(c) for c in chunks]) if chunks else np.zeros(1, np.float32)
        return wav, spec.sample_rate, time.time() - t0

    if spec.adapter == "chatterbox":
        t0 = time.time()
        wav = handle.generate(clean_for_plain(text))
        return to_mono_float32(wav), int(getattr(handle, "sr", spec.sample_rate)), time.time() - t0

    if spec.adapter == "dia":
        import torch
        processor, model = handle
        # Dia REQUIRES a leading speaker tag; add [S1] for the plain scripts.
        tagged = text if _TAG.search(text) else f"[S1] {text}"
        inputs = processor(text=[tagged], padding=True, return_tensors="pt").to(model.device)
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=3072, guidance_scale=3.0,
                                 temperature=1.8, top_p=0.90, top_k=45)
        decoded = processor.batch_decode(out)
        wav = decoded[0] if isinstance(decoded, (list, tuple)) else decoded
        return to_mono_float32(wav), spec.sample_rate, time.time() - t0

    if spec.adapter == "csm":
        import torch
        processor, model = handle
        # Speaker id 0. Strip [S1]/(laughs) tags; CSM uses its own [0]/[1] convention.
        tagged = f"[0]{clean_for_plain(text)}"
        inputs = processor(tagged, add_special_tokens=True).to(model.device)
        t0 = time.time()
        with torch.no_grad():
            audio = model.generate(**inputs, output_audio=True)
        wav = audio[0] if isinstance(audio, (list, tuple)) else audio
        return to_mono_float32(wav), spec.sample_rate, time.time() - t0

    if spec.adapter == "parler":
        import torch
        model, tokenizer = handle
        desc = spec.voice or "A clear, natural voice with very high audio quality and no background noise."
        input_ids = tokenizer(desc, return_tensors="pt").input_ids.to(model.device)
        prompt_ids = tokenizer(clean_for_plain(text), return_tensors="pt").input_ids.to(model.device)
        t0 = time.time()
        with torch.no_grad():
            gen = model.generate(input_ids=input_ids, prompt_input_ids=prompt_ids)
        sr = int(getattr(model.config, "sampling_rate", spec.sample_rate))
        return to_mono_float32(gen.cpu().numpy().squeeze()), sr, time.time() - t0

    if spec.adapter == "voxtral":
        import httpx
        payload = {"input": clean_for_plain(text), "model": spec.repo,
                   "response_format": "wav", "voice": spec.voice or "casual_male"}
        t0 = time.time()
        r = httpx.post(f"{handle['base']}/v1/audio/speech", json=payload, timeout=180.0)
        r.raise_for_status()
        wav, sr = sf.read(io.BytesIO(r.content), dtype="float32")
        return to_mono_float32(wav), int(sr), time.time() - t0

    raise ValueError(f"No adapter for {spec.adapter!r}")


# ── Report data + rendering ──────────────────────────────────────────────────────

@dataclass
class AudioResult:
    model_key: str
    text: str
    seconds: float = 0.0         # synth wall-clock
    audio_seconds: float = 0.0   # duration of the produced audio
    rtf: float = 0.0             # synth / audio; < 1 is faster than real time
    sample_rate: int = 0
    audio_uri: str = ""          # base64 wav data URI ("" if too big / failed)
    spec_uri: str = ""           # base64 waveform+spectrogram PNG
    peak_gb: float = 0.0
    embed_note: str = ""
    error: str = ""

    @property
    def speedup(self) -> float:
        return (self.audio_seconds / self.seconds) if self.seconds else 0.0


def build_audio_result(spec, text, wav, sr, seconds, peak_gb=0.0) -> AudioResult:
    wav = to_mono_float32(wav)
    audio_seconds = wav.size / float(sr) if sr else 0.0
    uri, note = audio_data_uri(wav, sr)
    return AudioResult(
        model_key=spec.key, text=text, seconds=seconds, audio_seconds=audio_seconds,
        rtf=(seconds / audio_seconds if audio_seconds else 0.0),
        sample_rate=sr, audio_uri=uri, spec_uri=waveform_spectrogram_png(wav, sr),
        peak_gb=peak_gb, embed_note=note,
    )


REPORT_CSS = """
<style>
  .tts-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
              color: #0b0b0b; }
  .tts-wrap h2 { margin: 0 0 4px; }
  .tts-meta { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
  .tts-grid { display: grid; gap: 16px;
              grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }
  .tts-cell { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
              background: #fff; display: flex; flex-direction: column; }
  .tts-spec { padding: 10px 11px 0; }
  .tts-spec img { width: 100%; height: auto; border-radius: 6px; display: block;
                  cursor: zoom-in; }
  .tts-audio { padding: 10px 11px 4px; }
  .tts-audio audio { width: 100%; display: block; }
  .tts-cap { padding: 6px 11px 11px; }
  .tts-model { font-weight: 600; font-size: 14px; }
  .tts-fam { display: inline-block; font-size: 11px; color: #374151; background: #f3f4f6;
             border-radius: 999px; padding: 1px 8px; margin: 4px 4px 0 0; }
  .tts-fast { display: inline-block; font-size: 11px; color: #065f46; background: #d1fae5;
              border-radius: 999px; padding: 1px 8px; margin: 4px 0 0; font-weight: 600; }
  .tts-voice { display: inline-block; font-size: 11px; color: #3730a3; background: #e0e7ff;
               border-radius: 999px; padding: 1px 8px; margin: 4px 4px 0 0; font-weight: 600; }
  .tts-sub { color: #6b7280; font-size: 12px; margin-top: 4px; line-height: 1.4; }
  .tts-err { padding: 16px; color: #b91c1c; font-size: 13px; white-space: pre-wrap; }
  .tts-note { padding: 8px 11px; color: #92400e; background: #fffbeb; font-size: 12px; }
  .tts-script { background: #f9fafb; border-left: 3px solid #6366f1; padding: 8px 12px;
                border-radius: 6px; margin: 6px 0 14px; font-size: 14px; }
  #tts-lb { position: fixed; inset: 0; z-index: 9999; display: none; cursor: zoom-out;
            flex-direction: column; align-items: center; justify-content: center;
            gap: 12px; padding: 24px; background: rgba(0,0,0,.88); }
  #tts-lb img { max-width: 96vw; max-height: 86vh; border-radius: 8px; }
  #tts-lb #tts-lb-cap { color: #e5e7eb; font-size: 14px; }
</style>
"""

_ZOOM = (
    "document.getElementById('tts-lb-img').src=this.src;"
    "document.getElementById('tts-lb-cap').textContent=this.dataset.cap;"
    "document.getElementById('tts-lb').style.display='flex'"
)
_LIGHTBOX = (
    "<div id=\"tts-lb\" onclick=\"this.style.display='none'\" style=\"display:none\">"
    '<img id="tts-lb-img" src="" alt="zoomed"/><div id="tts-lb-cap"></div></div>'
)


def _zoom_img(uri: str, cap: str) -> str:
    return (f'<img src="{uri}" alt="{html.escape(cap)}" '
            f'data-cap="{html.escape(cap, quote=True)}" onclick="{_ZOOM}"/>')


def _player(r: AudioResult) -> str:
    if not r.audio_uri:
        return ""
    # controls = native scrub/play/pause, no JS. Not autoplay: five clips auto-playing
    # at once is a wall of noise; the point is to play them one at a time and compare.
    return (f'<div class="tts-audio"><audio controls preload="metadata" '
            f'src="{r.audio_uri}"></audio></div>')


def _cell(spec, r: AudioResult) -> str:
    if r.error:
        return (f'<div class="tts-cell"><div class="tts-err">⚠️ {html.escape(r.error)}</div>'
                f'<div class="tts-cap"><div class="tts-model">{html.escape(spec.key)}</div>'
                f'</div></div>')

    spec_img = (f'<div class="tts-spec">{_zoom_img(r.spec_uri, f"{spec.key} · {r.text[:60]}")}</div>'
                if r.spec_uri else "")
    note = f'<div class="tts-note">{html.escape(r.embed_note)}</div>' if r.embed_note else ""
    fast = (f'<span class="tts-fast">{r.speedup:.1f}x real-time</span>'
            if r.speedup else "")
    voice = (f'<span class="tts-voice">🔊 {html.escape(spec.voice_label)}</span>'
             if getattr(spec, "voice_label", "") else "")
    peak = f' · peak {r.peak_gb:.1f}GB' if r.peak_gb else ""
    cap = (
        f'<div class="tts-cap"><div class="tts-model">{html.escape(spec.key)}</div>'
        f'<span class="tts-fam">{html.escape(spec.family)}</span>'
        f'<span class="tts-fam">{html.escape(spec.params)}</span>{voice}{fast}'
        f'<div class="tts-sub">{r.seconds:.1f}s to synth · {r.audio_seconds:.1f}s audio '
        f'@ {r.sample_rate/1000:.0f}kHz{peak}<br>{html.escape(spec.license)}</div></div>'
    )
    return f'<div class="tts-cell">{spec_img}{_player(r)}{note}{cap}</div>'


def render_grid(texts, specs, results, *, title="Text-to-speech comparison", meta="") -> str:
    """Full script x model grid: one block per script, models as the columns."""
    by_pair = {(r.text, r.model_key): r for r in results}
    blocks = []
    for text in texts:
        cells = "".join(
            _cell(s, by_pair.get((text, s.key), AudioResult(s.key, text, error="no result")))
            for s in specs
        )
        blocks.append(
            f'<div class="tts-script">🗣️ {html.escape(text)}</div>'
            f'<div class="tts-grid">{cells}</div>'
        )
    return (
        f'{REPORT_CSS}<div class="tts-wrap"><h2>{html.escape(title)}</h2>'
        f'<div class="tts-meta">{html.escape(meta)}</div>'
        + "".join(blocks) + "</div>" + _LIGHTBOX
    )


def render_status(title: str, body: str) -> str:
    return (f'{REPORT_CSS}<div class="tts-wrap"><h2>{html.escape(title)}</h2>'
            f'<div class="tts-meta">{html.escape(body)}</div></div>')
