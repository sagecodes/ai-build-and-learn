"""The Flyte comparison: read one script across several TTS models, one report.

Usage (runs on the DGX Spark devbox, not local Python):

    # the default 5-model, 5-script suite
    flyte run compare_pipeline.py compare

    # a quick 3-script pass on two models
    flyte run compare_pipeline.py compare \
        --suite quick --models '["qwen3-1.7b","kokoro-82m"]'

    # your own lines
    flyte run compare_pipeline.py compare \
        --texts '["The quick brown fox jumps over the lazy dog."]'

Shape (identical to the video demo next door):

    compare ─┬─ fetch_weights(model)   ·· CPU, cached: one HF download per model
             └─ GEN_TASKS[adapter](model, weights, texts)  ·· GPU: load once, say all

fetch is serial (parallel HF pulls just congest the uplink); the GPU tasks are
gathered, and since the box has one GPU the scheduler runs them one at a time. Each
GPU task lives in its OWN image/env (Qwen, Kokoro, Chatterbox and Dia cannot share a
Python environment; see config.py), so `compare` dispatches each model to the task
whose image has its package.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import flyte
import flyte.io
import flyte.report
import soundfile as sf

import tts_core
from config import GPU_ENVS, cpu_env, orch_env
from models import get_spec, jobs_for, render_spec, resolve_models
from prompts import get_suite

log = logging.getLogger("tts")
logging.basicConfig(level=logging.INFO)


# ── Data crossing the task boundary ──────────────────────────────────────────────
#
# NOTE these are the LIGHTWEIGHT carriers: metadata + a Dir of wavs, never the
# multi-hundred-KB base64 data URIs. The parent re-derives the report objects
# (AudioResult, with the embedded audio and spectrogram) from the wav files, exactly
# as the video demo re-reads its mp4s. A dataclass full of data URIs would bloat every
# task result needlessly.

@dataclass
class AudioItem:
    text: str
    filename: str = ""            # wav in the model's Dir; "" if this line failed
    seconds: float = 0.0          # synth wall-clock
    audio_seconds: float = 0.0
    sample_rate: int = 0
    peak_gb: float = 0.0
    error: str = ""


@dataclass
class ModelRun:
    model_key: str
    items: list[AudioItem] = field(default_factory=list)
    clips: flyte.io.Dir | None = None   # the wavs


# ── Fetch: cache the HF download, keyed per model ────────────────────────────────
#
# BUMP this when you change WHAT gets downloaded (the repos in a spec's all_repos, the
# allow/ignore patterns). Then a re-download is a deliberate act, not a side effect of
# editing a neighbouring function. Keyed on model_key, so each model keeps its entry.
_WEIGHTS_CACHE_VERSION = "v1"
_WEIGHTS_CACHE = flyte.Cache(behavior="override", version_override=_WEIGHTS_CACHE_VERSION)


@cpu_env.task(cache=_WEIGHTS_CACHE, retries=2)
async def fetch_weights(model_key: str) -> flyte.io.Dir:
    """Snapshot a model's HF repo(s) into an HF-cache-layout Dir. Cached forever.

    We download into `<dir>/` with snapshot_download(cache_dir=...), so the Dir is a
    ready-made HF hub cache. The GPU task points HF_HUB_CACHE at it, and every loader
    (qwen-tts, kokoro, chatterbox, transformers) reads from it, falling back to the
    network only on a miss. Runs on a CPU pod so no GPU sits idle during the pull.
    """
    from huggingface_hub import snapshot_download

    spec = get_spec(model_key)
    dest = Path(tempfile.mkdtemp(prefix=f"weights_{model_key}_"))
    token = os.environ.get("HF_TOKEN")
    for repo in spec.all_repos:
        log.info(f"[{model_key}] downloading {repo} -> {dest}")
        await asyncio.to_thread(
            snapshot_download, repo_id=repo, cache_dir=str(dest), token=token,
            allow_patterns=list(spec.allow_patterns) or None,
            ignore_patterns=list(spec.ignore_patterns) or None,
        )
    return await flyte.io.Dir.from_local(str(dest))


# ── The per-model work, shared by every adapter task ─────────────────────────────

async def _run_model(model_key: str, weights: flyte.io.Dir, texts: list[str],
                     voice: str = "", variant_key: str = "") -> ModelRun:
    """Load one model once, say every line, live-update the report, return a ModelRun.

    `voice` overrides the spec's default voice (a Kokoro pack / Qwen speaker / Parler
    description); `variant_key` is the column identity in the report (e.g.
    "kokoro-82m-f"), so two voices of the same model land in separate columns without
    re-downloading its weights. Runs INSIDE whichever adapter task called it.
    """
    base = get_spec(model_key)
    disp_key = variant_key or base.key
    the_voice = voice or base.voice
    # spec carries the base's repo/adapter but the variant's key + voice, so the report
    # and filenames are per-voice while load/synth still target the right model.
    spec = render_spec(base, disp_key, the_voice, base.voice_label or the_voice or "default")

    # Point HF at the pre-fetched Dir BEFORE any loader imports huggingface_hub (the
    # adapters import their packages lazily inside load_model, so nothing has read the
    # cache path yet). A miss just re-downloads; correctness holds either way.
    local = await weights.download()
    os.environ["HF_HUB_CACHE"] = str(local)
    os.environ["HF_HOME"] = str(local)

    out_dir = Path(tempfile.mkdtemp(prefix=f"tts_{disp_key}_"))
    items: list[AudioItem] = []
    results: list[tts_core.AudioResult] = []

    meta = f"{spec.repo} · voice={the_voice or 'default'} · {spec.license}"
    handle = None
    try:
        tts_core.prepare_gpu()
        log.info(f"[{disp_key}] loading via '{spec.adapter}' adapter (voice={the_voice or 'default'})")
        handle = tts_core.load_model(spec)

        for i, text in enumerate(texts):
            try:
                tts_core.reset_peak_memory()
                wav, sr, secs = tts_core.synth_one(handle, spec, text)
                peak = tts_core.peak_memory_gb()
                fn = f"{disp_key}__{i:02d}.wav"
                tts_core.write_wav(wav, sr, out_dir / fn)
                r = tts_core.build_audio_result(spec, text, wav, sr, secs, peak)
                results.append(r)
                items.append(AudioItem(text=text, filename=fn, seconds=secs,
                                       audio_seconds=r.audio_seconds, sample_rate=sr,
                                       peak_gb=peak))
                log.info(f"[{disp_key}] {i+1}/{len(texts)}: {secs:.1f}s -> "
                         f"{r.audio_seconds:.1f}s audio ({r.speedup:.1f}x RT)")
            except Exception as e:  # one bad line must not kill the model's other lines
                log.exception(f"[{disp_key}] line {i} failed")
                results.append(tts_core.AudioResult(disp_key, text, error=repr(e)))
                items.append(AudioItem(text=text, error=repr(e)))

            await flyte.report.replace.aio(
                tts_core.render_grid(texts[: i + 1], [spec], results,
                                     title=f"{disp_key} · synthesizing", meta=meta))
            await flyte.report.flush.aio()
    finally:
        try:
            if handle is not None:
                tts_core.close_model(spec, handle)   # tears down Voxtral's vLLM server
        except Exception:
            log.exception(f"[{disp_key}] close_model failed")
        handle = None
        tts_core.free_gpu_memory()

    clips = await flyte.io.Dir.from_local(str(out_dir))
    return ModelRun(model_key=disp_key, items=items, clips=clips)


# ── One task per adapter (one image/env each). Body is shared via _run_model ──────
#
# They look identical on purpose: the only thing that differs is which
# TaskEnvironment (hence image) they run in. model_key selects the spec, so the two
# Qwen models both flow through generate_qwen.

@GPU_ENVS["qwen"].task(report=True, retries=1)
async def generate_qwen(model_key: str, weights: flyte.io.Dir, texts: list[str],
                        voice: str = "", variant_key: str = "") -> ModelRun:
    return await _run_model(model_key, weights, texts, voice, variant_key)


@GPU_ENVS["kokoro"].task(report=True, retries=1)
async def generate_kokoro(model_key: str, weights: flyte.io.Dir, texts: list[str],
                          voice: str = "", variant_key: str = "") -> ModelRun:
    return await _run_model(model_key, weights, texts, voice, variant_key)


@GPU_ENVS["chatterbox"].task(report=True, retries=1)
async def generate_chatterbox(model_key: str, weights: flyte.io.Dir, texts: list[str],
                              voice: str = "", variant_key: str = "") -> ModelRun:
    return await _run_model(model_key, weights, texts, voice, variant_key)


@GPU_ENVS["dia"].task(report=True, retries=1)
async def generate_dia(model_key: str, weights: flyte.io.Dir, texts: list[str],
                       voice: str = "", variant_key: str = "") -> ModelRun:
    return await _run_model(model_key, weights, texts, voice, variant_key)


@GPU_ENVS["csm"].task(report=True, retries=1)
async def generate_csm(model_key: str, weights: flyte.io.Dir, texts: list[str],
                       voice: str = "", variant_key: str = "") -> ModelRun:
    return await _run_model(model_key, weights, texts, voice, variant_key)


@GPU_ENVS["parler"].task(report=True, retries=1)
async def generate_parler(model_key: str, weights: flyte.io.Dir, texts: list[str],
                          voice: str = "", variant_key: str = "") -> ModelRun:
    return await _run_model(model_key, weights, texts, voice, variant_key)


@GPU_ENVS["voxtral"].task(report=True, retries=1)
async def generate_voxtral(model_key: str, weights: flyte.io.Dir, texts: list[str],
                           voice: str = "", variant_key: str = "") -> ModelRun:
    return await _run_model(model_key, weights, texts, voice, variant_key)


GEN_TASKS = {
    "qwen": generate_qwen,
    "kokoro": generate_kokoro,
    "chatterbox": generate_chatterbox,
    "dia": generate_dia,
    "csm": generate_csm,
    "parler": generate_parler,
    "voxtral": generate_voxtral,
}


# ── Re-derive report objects from a run's wavs, in the parent ────────────────────

async def _to_results(spec, run: ModelRun) -> list[tts_core.AudioResult]:
    """Read a model's wavs back out and rebuild AudioResults (with the embedded audio
    and spectrogram) for the aggregate grid."""
    # A model that failed EVERY line still returns a non-None but EMPTY Dir, and
    # downloading an empty Dir raises DownloadQueueEmpty in the parent, which would
    # take the whole comparison down. So check there's something to download first.
    if not run.clips or not any(it.filename and not it.error for it in run.items):
        return [tts_core.AudioResult(spec.key, it.text, error=it.error or "no output")
                for it in run.items]

    local = Path(await run.clips.download())
    out: list[tts_core.AudioResult] = []
    for it in run.items:
        if it.error or not it.filename:
            out.append(tts_core.AudioResult(spec.key, it.text, error=it.error or "no output"))
            continue
        try:
            wav, sr = sf.read(str(local / it.filename), dtype="float32")
            out.append(tts_core.build_audio_result(spec, it.text, wav, sr, it.seconds, it.peak_gb))
        except Exception as e:
            out.append(tts_core.AudioResult(spec.key, it.text,
                                            error=f"could not read {it.filename}: {e}"))
    return out


# ── The orchestrator ─────────────────────────────────────────────────────────────

@orch_env.task(report=True)
async def compare(
    texts: list[str] | None = None,
    suite: str = "full",
    models: list[str] | None = None,
    voices: str = "all",
) -> list[ModelRun]:
    """Fan out one script across the models (and their voices), render one grid.

    `texts` wins over `suite` if given. `models` is a list of keys from models.MODELS;
    None uses the default set. `voices` expands each model's named voices into their own
    columns: "all" (every M/F variant), "female", "male", or "default" (one per model).
    """
    specs = resolve_models(models)
    if texts is None:
        texts = get_suite(suite)

    # Expand each model into its voice-variant columns. jobs = (render_spec, base_key,
    # voice_id, variant_key). "default" collapses to one column per model.
    jobs = []
    for s in specs:
        variants = [(s.key, s.voice, s.voice_label or (s.voice or "default"))] \
            if voices == "default" else jobs_for(s, voices)
        for variant_key, voice_id, voice_label in variants:
            jobs.append((render_spec(s, variant_key, voice_id, voice_label),
                         s.key, voice_id, variant_key))

    await flyte.report.replace.aio(tts_core.render_status(
        "Text-to-speech comparison",
        f"{len(specs)} models -> {len(jobs)} voice columns × {len(texts)} scripts. "
        f"Fetching weights, then synthesizing. Columns: {', '.join(j[3] for j in jobs)}."))
    await flyte.report.flush.aio()

    # Fetch each unique base model ONCE (voice variants share a repo), serial so parallel
    # HF pulls don't fight for the uplink. Tolerate a per-model fetch failure (a gated
    # repo, a network blip): it becomes an error COLUMN, it does not kill the run. One
    # bad model must never throw away every other model's audio.
    base_keys = list(dict.fromkeys(bk for _, bk, _, _ in jobs))
    weights: dict[str, flyte.io.Dir] = {}
    fetch_errors: dict[str, str] = {}
    for k in base_keys:
        try:
            weights[k] = await fetch_weights.override(short_name=f"fetch {k}")(k)
        except Exception as e:  # gated repo, download failure, ...
            log.exception(f"[{k}] fetch failed")
            fetch_errors[k] = f"weights fetch failed: {e}"

    # Parallel synth over the columns whose base fetched OK; one GPU means the scheduler
    # serializes. return_exceptions so one model's crash doesn't abort the gather.
    launch = [(rs, bk, vid, vk) for (rs, bk, vid, vk) in jobs if bk in weights]
    raw = await asyncio.gather(*[
        GEN_TASKS[rs.adapter].override(short_name=f"say {vk}")(bk, weights[bk], texts, vid, vk)
        for (rs, bk, vid, vk) in launch
    ], return_exceptions=True)

    render_specs = [rs for rs, _, _, _ in jobs]
    all_results: list[tts_core.AudioResult] = []
    runs: list[ModelRun] = []
    launched = {vk: res for (_, _, _, vk), res in zip(launch, raw)}
    for (rs, bk, _vid, vk) in jobs:
        res = launched.get(vk)
        if bk in fetch_errors:
            all_results.extend(tts_core.AudioResult(vk, t, error=fetch_errors[bk]) for t in texts)
        elif isinstance(res, Exception):
            all_results.extend(tts_core.AudioResult(vk, t, error=f"synth failed: {res}") for t in texts)
        else:
            runs.append(res)
            all_results.extend(await _to_results(rs, res))

    meta = (f"{len(specs)} models · {len(jobs)} voice columns · {len(texts)} scripts · "
            f"same text · play a row left-to-right to compare")
    await flyte.report.replace.aio(tts_core.render_grid(
        texts, render_specs, all_results,
        title="Open-source TTS: same script, side by side", meta=meta))
    await flyte.report.flush.aio()
    return runs


@orch_env.task(report=True)
async def generate_one(
    model_key: str = "qwen3-1.7b",
    text: str = "The morning light came slow and gold across the harbor.",
    voice: str = "",
) -> ModelRun:
    """Single-model smoke test: fetch, say one line, report. The cheapest 'does this
    model even load on the box' check without leaving Flyte. `voice` overrides the
    default (a Kokoro pack / Qwen speaker / Parler description)."""
    spec = get_spec(model_key)
    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)
    return await GEN_TASKS[spec.adapter].override(short_name=f"say {model_key}")(
        model_key, w, [text], voice)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(compare, suite="quick")
    print(f"Compare run: {run.name}")
    print(f"  {run.url}")
