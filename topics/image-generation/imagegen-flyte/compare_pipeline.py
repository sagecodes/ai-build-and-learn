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
# full-res PNGs live in the returned directory. Keep this modest: the whole grid
# is base64'd into one HTML doc that the console iframe has to render.
REPORT_MAX_SIDE = 512


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
# Task — download the weights once, cache the result
# ──────────────────────────────────────────────────────────────────────────────

@cpu_task_env.task(cache="auto")
async def fetch_weights(model_key: str) -> flyte.io.Dir:
    """Snapshot a model's HuggingFace repo into a Dir and return it.

    `cache="auto"` keys on (model_key, task version), so the multi-GB download
    happens once: later runs get the Dir straight from the blob store, and the
    GPU task pulls it in-cluster instead of re-hitting HuggingFace. Runs on a
    CPU pod so no GPU sits idle during the download.
    """
    import os as _os

    from huggingface_hub import snapshot_download

    # IPv6 to the HF CDN is black-holed on this network (302 -> 0 bytes, hangs
    # forever with no timeout mid-shard). Pin name resolution to IPv4 so big
    # multi-shard pulls (Qwen, FLUX, future video weights) actually complete.
    import socket as _socket
    if not getattr(_socket, "_flyte_ipv4_only", False):
        _orig_gai = _socket.getaddrinfo
        _socket.getaddrinfo = lambda *a, **k: [r for r in _orig_gai(*a, **k) if r[0] == _socket.AF_INET]
        _socket._flyte_ipv4_only = True

    spec = get_spec(model_key)
    dest = Path(tempfile.mkdtemp(prefix=f"weights_{model_key}_")) / "repo"
    log.info(f"[{model_key}] downloading {spec.repo} → {dest}")
    # Pull ONLY what diffusers' from_pretrained loads, not the whole repo. Big
    # model repos (SDXL especially) also ship: PyTorch .bin duplicates of every
    # safetensors, an fp16 *and* fp32 copy of each component, and single-file
    # combined checkpoints (sd_xl_base_1.0.safetensors, refiners, example LoRAs)
    # that the diffusers pipeline never touches. Downloading all of it is ~50GB
    # for SDXL vs ~13GB for the fp32 diffusers folders we actually use.
    snapshot_download(
        repo_id=spec.repo,
        local_dir=str(dest),
        token=_os.environ.get("HF_TOKEN"),
        ignore_patterns=[
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
        ],
    )
    log.info(f"[{model_key}] download complete")
    return await flyte.io.Dir.from_local(str(dest))


# ──────────────────────────────────────────────────────────────────────────────
# Task — one model over all prompts
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
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

    # Download every model's weights first — CPU tasks, run in parallel, and
    # cached so this is free on re-runs.
    weights = await asyncio.gather(*[fetch_weights(s.key) for s in specs])

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
