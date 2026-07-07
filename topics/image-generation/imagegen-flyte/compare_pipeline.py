"""Flyte 2 image-model comparison pipeline.

Run one or more prompts across several open-source text-to-image models and get
a single side-by-side Flyte report (models are the columns, prompts the rows),
plus the full-resolution PNGs saved as a directory artifact.

    generate_for_model(flux1-schnell) ─┐
    generate_for_model(sdxl) ──────────┼─> compare ─> side-by-side report
    generate_for_model(qwen-image) ────┘

Each model is its own GPU task, so the model loads once and generates every
prompt before the next model starts. On a single-GPU devbox the tasks serialize
at the scheduler (only one can hold the GPU); on a multi-GPU box they fan out.

Usage (run on the devbox):
    # default 3-model set, one prompt
    flyte run compare_pipeline.py compare \
        --prompts '["a red panda barista latte-art, cozy cafe, 50mm bokeh"]'

    # pick the models and add prompts
    flyte run compare_pipeline.py compare \
        --prompts '["neon cyberpunk alley in the rain","a storefront sign that reads OPEN 24 HOURS"]' \
        --models '["sdxl","flux1-schnell","qwen-image"]'

    # single model, quick smoke (fetch cached weights, then generate)
    flyte run compare_pipeline.py generate_one \
        --model_key flux1-schnell --prompts '["a corgi astronaut, studio light"]'
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from config import cpu_task_env, gpu_task_env, orch_env
from imagegen_core import (
    GenResult,
    free_gpu_memory,
    load_pipeline,
    pil_to_data_uri,
    render_grid,
    timed_generate,
)
from models import get_spec, resolve_models

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = gpu_task_env

# Cap the size embedded in the report so a big grid stays light (JPEG thumbnails);
# full-res PNGs live in the returned directory. 768 keeps the whole grid base64'd
# into one HTML doc light enough for the console iframe, while giving the
# click-to-zoom lightbox a meaningfully bigger image than the ~260px grid cell.
REPORT_MAX_SIDE = 768


@dataclass
class GenItem:
    """One (prompt → image) result within a model's run."""
    prompt: str
    filename: str          # PNG name inside the run's images dir ("" on error)
    seconds: float
    error: str = ""


@dataclass
class ModelRun:
    """Everything one model produced over the prompt set."""
    model_key: str
    items: list[GenItem] = field(default_factory=list)
    images: flyte.io.Dir | None = None   # full-res PNGs


# ──────────────────────────────────────────────────────────────────────────────
# Weight download: a throughput watchdog that kills + resumes on a stall
# ──────────────────────────────────────────────────────────────────────────────
#
# On a lossy uplink a snapshot_download stalls mid-stream: bytes flatline (dead
# socket, or a trickle below any read timeout) and neither HF_HUB_DOWNLOAD_TIMEOUT
# nor a socket timeout ever fires. Verified a 44.9GB Qwen pull sat at +0 MB/30s
# for 20+ minutes. Flyte task retries don't help either: a retry is a fresh pod,
# so it restarts from 0. The fix: watch throughput, and when it flatlines KILL the
# download and re-spawn it against the same local dir. snapshot_download resumes
# from the .incomplete files, so a restart just re-establishes the connection and
# keeps every byte downloaded so far. A subprocess (not a thread) because a
# blocked/trickling socket can't be interrupted in-thread.

def _dir_size(path) -> int:
    try:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    except OSError:
        return 0


def _snapshot_worker(repo: str, local_dir: str, token, ignore_patterns) -> None:
    """Child process: pin DNS to IPv4 (HF's CDN black-holes IPv6 here), then pull."""
    import socket as _socket

    _orig = _socket.getaddrinfo
    _socket.getaddrinfo = lambda *a, **k: [r for r in _orig(*a, **k) if r[0] == _socket.AF_INET]
    _socket.setdefaulttimeout(120)
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo, local_dir=local_dir, token=token,
                      ignore_patterns=ignore_patterns)


def _download_with_watchdog(repo, dest, token, ignore_patterns, model_key,
                            poll=30, stall_windows=5, min_growth=5_000_000,
                            max_restarts=3) -> None:
    """Pull `repo` into `dest`, restarting the download whenever throughput stalls.

    A poll window that adds < `min_growth` bytes counts as stalled; `stall_windows`
    in a row (default 5 x 30s = 2.5 min) triggers a kill + resume. Each resume
    continues from the .incomplete files (it does NOT re-pull what's already down),
    so a restart is cheap. `max_restarts` (default 3) bounds it so a permanently
    dead link fails fast instead of looping forever; raise it for a huge model on
    a flaky link. Flyte's task retries is the from-scratch backstop beyond that.
    """
    import multiprocessing as _mp
    import time as _time

    ctx = _mp.get_context("spawn")   # clean child; safe to fork from a worker thread
    restarts = 0
    while True:
        p = ctx.Process(target=_snapshot_worker,
                        args=(repo, str(dest), token, ignore_patterns), daemon=True)
        p.start()
        last = _dir_size(dest)
        stalls = 0
        stalled = False
        while p.is_alive():
            _time.sleep(poll)
            cur = _dir_size(dest)
            grew = cur - last
            log.info(f"[{model_key}] {cur / 1e9:.1f} GB so far (+{grew / 1e6:.0f} MB/{poll}s)")
            stalls = stalls + 1 if grew < min_growth else 0
            last = cur
            if stalls >= stall_windows:
                stalled = True
                break
        if stalled:
            restarts += 1
            log.warning(f"[{model_key}] stalled at {last / 1e9:.1f} GB for "
                        f"{stall_windows * poll}s; killing + resuming "
                        f"(restart {restarts}/{max_restarts})")
            p.terminate()
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join()
            if restarts > max_restarts:
                raise RuntimeError(f"[{model_key}] download stalled past {max_restarts} restarts")
            continue
        p.join()
        if p.exitcode == 0:
            log.info(f"[{model_key}] download complete")
            return
        # Worker exited on a network error rather than a flatline; resume too.
        restarts += 1
        log.warning(f"[{model_key}] download worker exited {p.exitcode}; "
                    f"resuming (restart {restarts}/{max_restarts})")
        if restarts > max_restarts:
            raise RuntimeError(f"[{model_key}] download failed after {max_restarts} restarts")
        _time.sleep(5)


# ──────────────────────────────────────────────────────────────────────────────
# Task — download the weights once, cache the result
# ──────────────────────────────────────────────────────────────────────────────

# retries=2: the in-pod watchdog handles stalls by resuming (cheap), so the
# task-level retry is just a from-scratch backstop for a pod dying outright.
# Kept low on purpose: each task retry is a fresh pod that re-pulls from 0, and
# we don't want to burn bandwidth re-downloading tens of GB many times over.
@cpu_task_env.task(cache="auto", retries=2)
async def fetch_weights(model_key: str) -> flyte.io.Dir:
    """Snapshot a model's HuggingFace repo into a Dir and return it.

    `cache="auto"` keys on (model_key, task version), so the multi-GB download
    happens once: later runs get the Dir straight from the blob store, and the
    GPU task pulls it in-cluster instead of re-hitting HuggingFace. Runs on a
    CPU pod so no GPU sits idle during the download.
    """
    import asyncio as _asyncio
    import os as _os

    spec = get_spec(model_key)
    dest = Path(tempfile.mkdtemp(prefix=f"weights_{model_key}_")) / "repo"
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"[{model_key}] downloading {spec.repo} → {dest}")
    # Pull ONLY what diffusers' from_pretrained loads, not the whole repo. Big
    # model repos (SDXL especially) also ship: PyTorch .bin duplicates of every
    # safetensors, an fp16 *and* fp32 copy of each component, and single-file
    # combined checkpoints (sd_xl_base_1.0.safetensors, refiners, example LoRAs)
    # that the diffusers pipeline never touches. Downloading all of it is ~50GB
    # for SDXL vs ~13GB for the fp32 diffusers folders we actually use.
    ignore_patterns = [
        "*.pth", "*.onnx", "*.onnx_data", "*.ckpt", "*.gguf",  # other runtimes
        "*.bin", "*.msgpack", "*.h5",                          # non-safetensors dupes
        "*.fp16.safetensors",                                  # keep fp32 components
        # Root-level single-file checkpoints that DUPLICATE the diffusers
        # subfolders (transformer/, unet/, vae/, ...) which from_pretrained
        # actually loads. Each is a full second copy of the model, e.g. FLUX
        # ships flux1-schnell.safetensors (~22GB) alongside transformer/.
        "sd_xl_*.safetensors", "*_refiner*",                   # SDXL
        "flux1-*.safetensors", "flux2-*.safetensors", "ae.safetensors",  # FLUX
        "sd3*.safetensors", "sd3.5*.safetensors",              # SD3.5
    ]

    # Download in a subprocess under a throughput watchdog (see
    # _download_with_watchdog): it logs the same 30s GB heartbeat, and when bytes
    # flatline it kills + resumes the pull, which is the only thing that reliably
    # recovers a mid-stream stall on this link. Runs in a thread so the task's
    # event loop stays responsive.
    await _asyncio.to_thread(
        _download_with_watchdog, spec.repo, dest,
        _os.environ.get("HF_TOKEN"), ignore_patterns, model_key,
    )
    return await flyte.io.Dir.from_local(str(dest))


# ──────────────────────────────────────────────────────────────────────────────
# Task — one model over all prompts
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True, retries=3)
async def generate_for_model(
    model_key: str,
    weights: flyte.io.Dir,
    prompts: list[str],
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    width: int = -1,
    height: int = -1,
    negative_prompt: str = "",
) -> ModelRun:
    """Load `model_key` from the cached `weights` Dir, generate every prompt.

    -1 sentinels mean "use the model's default" (steps/guidance/width/height).
    """
    spec = get_spec(model_key)
    out_dir = Path(tempfile.mkdtemp(prefix=f"imgs_{model_key}_"))

    await flyte.report.replace.aio(
        f"<h3>Loading <code>{spec.repo}</code> ({spec.family})…</h3>"
    )
    await flyte.report.flush.aio()

    local_weights = await weights.download()
    pipe = None
    try:
        # Start from a clean allocator in case anything lingered, then load. A
        # transient CUDA OOM here (unified-memory pressure on the GB10) re-runs
        # via retries=3 in a fresh pod, which usually clears it.
        free_gpu_memory()
        pipe = load_pipeline(spec, model_path=local_weights)

        kw = dict(
            steps=None if steps < 0 else steps,
            guidance=None if guidance < 0 else guidance,
            seed=seed,
            width=None if width < 0 else width,
            height=None if height < 0 else height,
            negative_prompt=negative_prompt or None,
        )

        items: list[GenItem] = []
        results: list[GenResult] = []  # for the live per-model report (data URIs)
        for i, prompt in enumerate(prompts):
            log.info(f"[{model_key}] {i + 1}/{len(prompts)}: {prompt[:60]}")
            try:
                img, secs = timed_generate(pipe, spec, prompt, **kw)
                fname = f"{model_key}__{i:02d}.png"
                img.save(out_dir / fname)
                items.append(GenItem(prompt=prompt, filename=fname, seconds=secs))
                results.append(GenResult(
                    model_key=model_key, prompt=prompt, seconds=secs,
                    data_uri=pil_to_data_uri(img, max_side=REPORT_MAX_SIDE),
                ))
            except Exception as e:  # one bad prompt shouldn't sink the whole model
                # A CUDA OOM is different: it's a whole-pod condition, not a
                # bad prompt, and it's usually transient (unified-memory pressure
                # on the GB10). Free and re-raise so the task retries in a fresh
                # pod instead of silently marking every remaining cell failed.
                if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
                    log.warning(f"[{model_key}] CUDA OOM on prompt {i}; failing task to retry")
                    free_gpu_memory()
                    raise
                log.warning(f"[{model_key}] failed on prompt {i}: {e}")
                items.append(GenItem(prompt=prompt, filename="", seconds=0.0, error=str(e)))
                results.append(GenResult(model_key=model_key, prompt=prompt, seconds=0.0,
                                         error=str(e)))
            # Progressive report: redraw the prompts-so-far grid after each image.
            await flyte.report.replace.aio(render_grid(
                prompts[: i + 1], [spec], results,
                meta=f"{spec.repo} · {spec.license} · {i + 1}/{len(prompts)} prompts",
            ))
            await flyte.report.flush.aio()

        images = await flyte.io.Dir.from_local(str(out_dir))
        return ModelRun(model_key=model_key, items=items, images=images)
    finally:
        # Free the GPU before this task returns. The per-model tasks serialize on
        # the single-GPU devbox, so releasing here keeps the next model from
        # racing this pod's teardown (and cleans up even if generation threw).
        pipe = None
        free_gpu_memory()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator — many models, one side-by-side report
# ──────────────────────────────────────────────────────────────────────────────

async def _run_to_results(run: ModelRun) -> list[GenResult]:
    """Download a ModelRun's PNGs and turn them into report-ready GenResults."""
    from PIL import Image

    local = Path(await run.images.download()) if run.images else None
    out: list[GenResult] = []
    for it in run.items:
        if it.error or not it.filename or local is None:
            out.append(GenResult(run.model_key, it.prompt, 0.0, error=it.error or "missing"))
            continue
        img = Image.open(local / it.filename)
        out.append(GenResult(
            model_key=run.model_key, prompt=it.prompt, seconds=it.seconds,
            data_uri=pil_to_data_uri(img, max_side=REPORT_MAX_SIDE),
        ))
    return out


@orch_env.task(report=True)
async def generate_one(
    model_key: str,
    prompts: list[str],
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
) -> ModelRun:
    """CLI-friendly single model: fetch (cached) then generate. For quick smokes."""
    w = await fetch_weights(model_key)
    return await generate_for_model(
        model_key, w, prompts, steps=steps, guidance=guidance, seed=seed,
    )


@orch_env.task(report=True)
async def compare(
    prompts: list[str],
    models: list[str] | None = None,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    width: int = -1,
    height: int = -1,
    negative_prompt: str = "",
) -> list[ModelRun]:
    """Generate `prompts` on each model and render the side-by-side grid report."""
    specs = resolve_models(models)
    log.info(f"Comparing {[s.key for s in specs]} on {len(prompts)} prompt(s)")

    await flyte.report.replace.aio(
        f"<h2>Image model comparison</h2><p>Fetching weights + generating "
        f"{len(prompts)} prompt(s) × {len(specs)} models "
        f"({', '.join(s.key for s in specs)})…</p>"
    )
    await flyte.report.flush.aio()

    # Download every model's weights first, one model at a time. Serial on
    # purpose: parallel snapshot_downloads open dozens of concurrent sockets to
    # the HF CDN, and on a lossy uplink (e.g. the Spark over Wi-Fi) that
    # congestion is what black-holes a transfer mid-stream and hangs the fetch.
    # On a real cluster with a fat, reliable pipe, swap this for the parallel
    # form: `weights = await asyncio.gather(*[fetch_weights(s.key) for s in specs])`.
    # Each fetch is cache="auto", so already-downloaded models return instantly.
    weights = [await fetch_weights(s.key) for s in specs]

    # Then one GPU task per model. asyncio.gather submits them together; the
    # devbox scheduler runs as many as there are free GPUs (one at a time on a
    # single-GPU box), each loading from its cached weights Dir.
    runs: list[ModelRun] = await asyncio.gather(*[
        generate_for_model(
            s.key, w, prompts, steps=steps, guidance=guidance, seed=seed,
            width=width, height=height, negative_prompt=negative_prompt,
        )
        for s, w in zip(specs, weights)
    ])

    all_results: list[GenResult] = []
    for r in runs:
        all_results.extend(await _run_to_results(r))

    meta = (f"{len(prompts)} prompt(s) · seed={seed} · "
            f"models: {', '.join(s.key for s in specs)}")
    await flyte.report.replace.aio(render_grid(prompts, specs, all_results, meta=meta))
    await flyte.report.flush.aio()
    return runs


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(
        compare,
        prompts=["a red panda barista pouring latte art, cozy cafe, 50mm, bokeh"],
    )
    print(f"Compare run: {run.name}")
    print(f"  {run.url}")
