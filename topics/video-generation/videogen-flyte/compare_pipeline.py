"""Flyte 2 video-model comparison pipeline.

Run one or more prompts across several open-source video models and get a single
side-by-side Flyte report with **playable clips**, plus the full-resolution .mp4s
saved as a directory artifact.

    fetch_weights(wan22-ti2v-5b) ─┐                         ┌─ generate(wan22)  ─┐
                                  ├─ (cached, once ever) ───┤                    ├─> report
    fetch_weights(ltx2-distilled)─┘                         └─ generate(ltx2)  ──┘

Each model is its own GPU task, so the model loads once and renders every prompt
before the next model starts. On the single-GPU Spark they serialize at the
scheduler; on a multi-GPU box they fan out.

Two entry points:

  compare  : text-to-video across N models, one grid report.
  animate  : image-to-video: generate a first frame with a small image model, then
             animate it. Shows the frame it started from next to the clips.

Usage (runs on the devbox; see README for the devbox setup):

    # the default pair (wan21-t2v-1.3b + wan22-ti2v-5b), one prompt
    flyte run compare_pipeline.py compare \
        --prompts '["a red panda barista pouring latte art, steam rising, 50mm"]'

    # the headline comparison: Apache-2.0 Wan vs LTX-2 (which also generates audio)
    flyte run compare_pipeline.py compare \
        --prompts '["waves crashing on black volcanic rock, slow motion"]' \
        --models '["wan22-ti2v-5b","ltx2-distilled"]'

    # image-to-video: make a first frame, then animate it
    flyte run compare_pipeline.py animate \
        --prompts '["a paper boat on a rain puddle, neon reflections"]'

    # single model smoke test (weights come from cache after the first run)
    flyte run compare_pipeline.py generate_one \
        --model_key wan21-t2v-1.3b --prompts '["a corgi astronaut, studio light"]'
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

from config import cpu_env, gpu_env, orch_env
from models import DEFAULT_IMAGE_MODEL, get_image_spec, get_spec, resolve_models
from videogen_core import (
    ClipResult,
    build_clip_result,
    free_gpu_memory,
    load_image_pipeline,
    load_pipeline,
    pil_to_data_uri,
    prepare_gpu,
    render_grid,
    render_status,
    timed_generate,
    write_mp4,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)


@dataclass
class ClipItem:
    """One (prompt -> clip) result within a model's run."""
    prompt: str
    filename: str          # .mp4 name inside the run's clips dir ("" on error)
    seconds: float
    n_frames: int = 0
    fps: int = 0
    has_audio: bool = False
    peak_gb: float = 0.0
    error: str = ""


@dataclass
class ModelRun:
    """Everything one model produced over the prompt set."""
    model_key: str
    items: list[ClipItem] = field(default_factory=list)
    clips: flyte.io.Dir | None = None   # full-resolution .mp4s


# ──────────────────────────────────────────────────────────────────────────────
# Weight download: a throughput watchdog that kills + resumes on a stall
# ──────────────────────────────────────────────────────────────────────────────
#
# Carried over from the image-gen demo, and it matters MORE here: these downloads
# are 30-126GB, so a stall that silently eats the run is expensive.
#
# On a lossy uplink a snapshot_download stalls mid-stream: bytes flatline (dead
# socket, or a trickle below any read timeout) and neither HF_HUB_DOWNLOAD_TIMEOUT
# nor a socket timeout ever fires. Flyte task retries don't help either: a retry is
# a fresh pod, so it restarts from 0. The fix: watch throughput, and when it
# flatlines KILL the download and re-spawn it against the same local dir.
# snapshot_download resumes from the .incomplete files, so a restart just
# re-establishes the connection and keeps every byte so far. A subprocess (not a
# thread) because a blocked socket can't be interrupted in-thread.

def _dir_size(path) -> int:
    try:
        return sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    except OSError:
        return 0


def _snapshot_worker(repo, local_dir, token, allow_patterns, ignore_patterns) -> None:
    """Child process: pin DNS to IPv4 (HF's CDN black-holes IPv6 here), then pull."""
    import socket as _socket

    _orig = _socket.getaddrinfo
    _socket.getaddrinfo = lambda *a, **k: [r for r in _orig(*a, **k) if r[0] == _socket.AF_INET]
    _socket.setdefaulttimeout(120)
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo, local_dir=local_dir, token=token,
        allow_patterns=list(allow_patterns) or None,
        ignore_patterns=list(ignore_patterns) or None,
    )


def _download_with_watchdog(repo, dest, token, allow_patterns, ignore_patterns, key,
                            poll=30, stall_windows=5, min_growth=5_000_000,
                            max_restarts=5) -> None:
    """Pull `repo` into `dest`, restarting whenever throughput flatlines.

    A poll window adding < `min_growth` bytes counts as stalled; `stall_windows` in
    a row (5 x 30s = 2.5 min) triggers a kill + resume. Each resume continues from
    the .incomplete files, so a restart is cheap. `max_restarts` bounds it so a
    dead link fails instead of looping forever. Raised to 5 here (from the image
    demo's 3) because these downloads are several times larger, so they're exposed
    to proportionally more chances to stall.
    """
    import multiprocessing as _mp
    import time as _time

    ctx = _mp.get_context("spawn")
    restarts = 0
    while True:
        p = ctx.Process(
            target=_snapshot_worker,
            args=(repo, str(dest), token, allow_patterns, ignore_patterns), daemon=True,
        )
        p.start()
        last, stalls, stalled = _dir_size(dest), 0, False
        while p.is_alive():
            _time.sleep(poll)
            cur = _dir_size(dest)
            grew = cur - last
            log.info(f"[{key}] {cur / 1e9:.1f} GB so far (+{grew / 1e6:.0f} MB/{poll}s)")
            stalls = stalls + 1 if grew < min_growth else 0
            last = cur
            if stalls >= stall_windows:
                stalled = True
                break
        if stalled:
            restarts += 1
            log.warning(f"[{key}] stalled at {last / 1e9:.1f} GB for {stall_windows * poll}s; "
                        f"killing + resuming (restart {restarts}/{max_restarts})")
            p.terminate()
            p.join(10)
            if p.is_alive():
                p.kill()
                p.join()
            if restarts > max_restarts:
                raise RuntimeError(f"[{key}] download stalled past {max_restarts} restarts")
            continue
        p.join()
        if p.exitcode == 0:
            log.info(f"[{key}] download complete ({_dir_size(dest) / 1e9:.1f} GB)")
            return
        restarts += 1
        log.warning(f"[{key}] worker exited {p.exitcode}; resuming "
                    f"(restart {restarts}/{max_restarts})")
        if restarts > max_restarts:
            raise RuntimeError(f"[{key}] download failed after {max_restarts} restarts")
        _time.sleep(5)


# ──────────────────────────────────────────────────────────────────────────────
# Tasks: fetch (cached), then generate
# ──────────────────────────────────────────────────────────────────────────────

@cpu_env.task(cache="auto", retries=2)
async def fetch_weights(model_key: str) -> flyte.io.Dir:
    """Snapshot a video model's HF repo into a Dir and return it. Cached forever.

    `cache="auto"` keys on (model_key, task version), so a 95GB download happens
    ONCE: later runs get the Dir straight from the blob store and the GPU task
    pulls it in-cluster instead of re-hitting HuggingFace. Runs on a CPU pod so no
    GPU sits idle for the hour a cold LTX-2 pull can take.

    The allow/ignore patterns come from the spec, and they are the difference
    between a 53GB download and a 372GB one (see models.py). We only ever fetch
    what `from_pretrained` actually loads.
    """
    import asyncio as _asyncio
    import os as _os

    spec = get_spec(model_key)
    dest = Path(tempfile.mkdtemp(prefix=f"weights_{model_key}_")) / "repo"
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"[{model_key}] downloading {spec.repo} (~{spec.download_gb:.0f} GB) -> {dest}")

    await _asyncio.to_thread(
        _download_with_watchdog, spec.repo, dest, _os.environ.get("HF_TOKEN"),
        spec.allow_patterns, spec.ignore_patterns, model_key,
    )
    return await flyte.io.Dir.from_local(str(dest))


@cpu_env.task(cache="auto", retries=2)
async def fetch_image_weights(model_key: str) -> flyte.io.Dir:
    """Same, for the small text-to-image model that makes an I2V first frame.

    Separate task (not a param on fetch_weights) so the two registries stay
    separate and each keeps its own cache entry.
    """
    import asyncio as _asyncio
    import os as _os

    spec = get_image_spec(model_key)
    dest = Path(tempfile.mkdtemp(prefix=f"imgw_{model_key}_")) / "repo"
    dest.mkdir(parents=True, exist_ok=True)
    log.info(f"[{model_key}] downloading {spec.repo} (~{spec.download_gb:.0f} GB)")

    await _asyncio.to_thread(
        _download_with_watchdog, spec.repo, dest, _os.environ.get("HF_TOKEN"),
        (), spec.ignore_patterns, model_key,
    )
    return await flyte.io.Dir.from_local(str(dest))


@gpu_env.task(report=True, retries=2)
async def make_first_frames(
    image_model: str,
    weights: flyte.io.Dir,
    prompts: list[str],
    width: int = 832,
    height: int = 480,
    seed: int = 1234,
) -> list[str]:
    """Generate one starting image per prompt, returned as JPEG data URIs.

    This is the front half of the image-to-video path: rather than make you supply
    a source image, we generate it. Data URIs (not a Dir) because they're small
    (a few hundred KB), they flow straight into the report, and the I2V task can
    decode them back to PIL without another blob round-trip.

    The image models here are deliberately tiny and few-step (sd-turbo is ~1s a
    frame); the video model is what we're actually looking at.
    """
    import base64
    import io as _io
    import torch
    from PIL import Image

    spec = get_image_spec(image_model)
    local = await weights.download()

    await flyte.report.replace.aio(
        render_status("Generating first frames",
                      f"{spec.repo} · {len(prompts)} prompt(s)")
    )
    await flyte.report.flush.aio()

    free_gpu_memory()
    pipe = None
    try:
        pipe = load_image_pipeline(spec, model_path=local)
        uris: list[str] = []
        for i, p in enumerate(prompts):
            g = torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None
            kw = dict(prompt=p, num_inference_steps=spec.steps, width=width,
                      height=height, generator=g)
            if spec.guidance is not None:
                kw["guidance_scale"] = spec.guidance
            img: Image.Image = pipe(**kw).images[0]
            buf = _io.BytesIO()
            img.convert("RGB").save(buf, format="JPEG", quality=92)
            uris.append("data:image/jpeg;base64,"
                        + base64.b64encode(buf.getvalue()).decode("ascii"))
            log.info(f"[{image_model}] first frame {i + 1}/{len(prompts)}")
        return uris
    finally:
        pipe = None
        free_gpu_memory()


def _decode_uri(uri: str):
    """data:image/...;base64,... -> PIL.Image"""
    import base64
    import io as _io

    from PIL import Image

    return Image.open(_io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")


@gpu_env.task(report=True, retries=2)
async def generate_for_model(
    model_key: str,
    weights: flyte.io.Dir,
    prompts: list[str],
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    width: int = -1,
    height: int = -1,
    num_frames: int = -1,
    negative_prompt: str = "",
    first_frames: list[str] | None = None,   # data URIs -> image-to-video
) -> ModelRun:
    """Load `model_key` from the cached weights Dir and render every prompt.

    -1 sentinels mean "use the model's default" (steps/guidance/size/frames).
    Passing `first_frames` switches this to the model's image-to-video pipeline.
    """
    spec = get_spec(model_key)
    i2v = bool(first_frames)
    if i2v and not spec.supports_i2v:
        raise ValueError(
            f"{model_key} has no image-to-video pipeline. I2V-capable models: "
            f"wan22-ti2v-5b, ltx2-distilled, cogvideox-5b."
        )

    out_dir = Path(tempfile.mkdtemp(prefix=f"clips_{model_key}_"))
    mode = "image-to-video" if i2v else "text-to-video"

    await flyte.report.replace.aio(
        render_status(f"Loading {spec.key}",
                      f"{spec.repo} · {spec.family} · {mode} · ~{spec.est_vram_gb:.0f}GB in bf16")
    )
    await flyte.report.flush.aio()

    local = await weights.download()
    pipe = None
    try:
        # Clean the pool, cap this process, and refuse a model that can't fit BEFORE
        # spending 20 minutes loading it. On unified memory an uncapped overshoot
        # locks the whole box instead of raising. See videogen_core.prepare_gpu.
        prepare_gpu(spec)
        pipe = load_pipeline(spec, model_path=local, i2v=i2v)

        kw = dict(
            steps=None if steps < 0 else steps,
            guidance=None if guidance < 0 else guidance,
            seed=seed,
            width=None if width < 0 else width,
            height=None if height < 0 else height,
            num_frames=None if num_frames < 0 else num_frames,
            negative_prompt=negative_prompt or None,
        )

        items: list[ClipItem] = []
        results: list[ClipResult] = []
        ff_map: dict[str, str] = {}

        for i, prompt in enumerate(prompts):
            log.info(f"[{model_key}] {i + 1}/{len(prompts)}: {prompt[:60]}")
            start_img = None
            if i2v:
                start_img = _decode_uri(first_frames[i])
                ff_map[prompt] = first_frames[i]

            # Live progress. A video step can take 10+ seconds on this box, so a
            # report that only updates once per CLIP would sit blank for minutes.
            # Reporting per denoise step is the difference between "is it hung?" and
            # watching it work. flyte.report isn't async-safe from the pipeline's
            # worker thread, so the callback just records and we render between
            # prompts; the step counter still lands in the pod logs live.
            def _on_step(k, total, _p=prompt, _i=i):
                if k == 1 or k % 5 == 0 or k == total:
                    log.info(f"[{model_key}] prompt {_i + 1}: step {k}/{total}")

            try:
                frames, audio, sr, secs = timed_generate(
                    pipe, spec, prompt, image=start_img, on_step=_on_step, **kw
                )
                fname = f"{model_key}__{i:02d}.mp4"
                write_mp4(frames, spec.fps, out_dir / fname, audio=audio, sample_rate=sr)

                r = build_clip_result(spec, prompt, frames, audio, sr, secs)
                results.append(r)
                items.append(ClipItem(
                    prompt=prompt, filename=fname, seconds=secs, n_frames=len(frames),
                    fps=spec.fps, has_audio=audio is not None, peak_gb=r.peak_gb,
                ))
                log.info(f"[{model_key}] prompt {i + 1} done in {secs:.0f}s "
                         f"(peak {r.peak_gb:.0f}GB)")
            except Exception as e:
                # An OOM is a whole-pod condition, not a bad prompt, and on this box
                # it's often transient (something else held the unified pool). Free
                # and re-raise so the task retries in a fresh pod rather than marking
                # every remaining cell failed.
                if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
                    log.warning(f"[{model_key}] CUDA OOM on prompt {i}; failing task to retry")
                    free_gpu_memory()
                    raise
                log.warning(f"[{model_key}] failed on prompt {i}: {e}")
                items.append(ClipItem(prompt=prompt, filename="", seconds=0.0, error=str(e)))
                results.append(ClipResult(model_key, prompt, 0.0, error=str(e)))

            await flyte.report.replace.aio(render_grid(
                prompts[: i + 1], [spec], results,
                title=f"{spec.key} ({mode})",
                meta=f"{spec.repo} · {spec.license} · {i + 1}/{len(prompts)} prompts",
                first_frames=ff_map or None,
            ))
            await flyte.report.flush.aio()

        clips = await flyte.io.Dir.from_local(str(out_dir))
        return ModelRun(model_key=model_key, items=items, clips=clips)
    finally:
        # Free before the task returns. The per-model tasks serialize on the
        # single-GPU box, so releasing here keeps the next model from racing this
        # pod's teardown (and cleans up even if generation threw).
        pipe = None
        free_gpu_memory()


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrators
# ──────────────────────────────────────────────────────────────────────────────

async def _to_results(run: ModelRun) -> list[ClipResult]:
    """Re-read a ModelRun's .mp4s and rebuild report-ready ClipResults.

    The child task already built these once for its own live report, but a
    dataclass crossing the task boundary can't carry multi-MB data URIs sanely, so
    the parent re-derives them from the clip artifacts. Reading frames back out of
    the mp4 (rather than shipping them) keeps the task interface small.
    """
    import av

    from videogen_core import frame_strip_data_uri, video_data_uri

    if not run.clips:
        return [ClipResult(run.model_key, it.prompt, 0.0, error=it.error or "missing")
                for it in run.items]

    local = Path(await run.clips.download())
    out: list[ClipResult] = []
    for it in run.items:
        if it.error or not it.filename:
            out.append(ClipResult(run.model_key, it.prompt, 0.0,
                                  error=it.error or "missing"))
            continue
        path = local / it.filename
        frames = []
        try:
            with av.open(str(path)) as c:
                for f in c.decode(video=0):
                    frames.append(f.to_image())
        except Exception as e:
            out.append(ClipResult(run.model_key, it.prompt, it.seconds,
                                  error=f"could not read {it.filename}: {e}"))
            continue

        # Embed the mp4 file itself rather than re-encoding: it's already the size
        # we generated at, and re-encoding would only lose quality. If it's over
        # budget, video_data_uri's re-encode path handles it.
        raw = path.read_bytes()
        from videogen_core import MAX_EMBED_BYTES

        if len(raw) <= MAX_EMBED_BYTES:
            import base64

            uri = "data:video/mp4;base64," + base64.b64encode(raw).decode("ascii")
            note = ""
        else:
            uri, note = video_data_uri(frames, it.fps or 16)

        w, h = frames[0].size if frames else (0, 0)
        out.append(ClipResult(
            model_key=run.model_key, prompt=it.prompt, seconds=it.seconds,
            video_uri=uri, strip_uri=frame_strip_data_uri(frames), embed_note=note,
            n_frames=it.n_frames or len(frames), fps=it.fps, width=w, height=h,
            has_audio=it.has_audio, peak_gb=it.peak_gb,
        ))
    return out


@orch_env.task(report=True)
async def generate_one(
    model_key: str,
    prompts: list[str],
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    num_frames: int = -1,
) -> ModelRun:
    """One model, fetch (cached) then generate. The quick smoke test."""
    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)
    return await generate_for_model.override(short_name=f"generate {model_key}")(
        model_key, w, prompts, steps=steps, guidance=guidance, seed=seed,
        num_frames=num_frames,
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
    num_frames: int = -1,
    negative_prompt: str = "",
) -> list[ModelRun]:
    """Text-to-video: render `prompts` on each model, emit one grid report."""
    specs = resolve_models(models)
    log.info(f"Comparing {[s.key for s in specs]} on {len(prompts)} prompt(s)")

    total_gb = sum(s.download_gb for s in specs)
    await flyte.report.replace.aio(render_status(
        "Video model comparison",
        f"Fetching weights (~{total_gb:.0f}GB, cached after the first run) and rendering "
        f"{len(prompts)} prompt(s) across {len(specs)} model(s): "
        f"{', '.join(s.key for s in specs)}. Video generation is slow on this box; "
        f"expect a few minutes per clip.",
    ))
    await flyte.report.flush.aio()

    # Download one model at a time. Serial on purpose: parallel snapshot_downloads
    # open dozens of concurrent sockets to the HF CDN, and on a lossy uplink that
    # congestion is exactly what black-holes a transfer mid-stream. It matters more
    # here than in the image demo because the payloads are 3-4x bigger. Each fetch
    # is cache="auto", so already-downloaded models return instantly.
    weights = [
        await fetch_weights.override(short_name=f"fetch {s.key}")(s.key) for s in specs
    ]

    # Then one GPU task per model. gather submits them together; the scheduler runs
    # as many as there are free GPUs (one at a time on the Spark).
    runs: list[ModelRun] = await asyncio.gather(*[
        generate_for_model.override(short_name=f"generate {s.key}")(
            s.key, w, prompts, steps=steps, guidance=guidance, seed=seed,
            width=width, height=height, num_frames=num_frames,
            negative_prompt=negative_prompt,
        )
        for s, w in zip(specs, weights)
    ])

    all_results: list[ClipResult] = []
    for r in runs:
        all_results.extend(await _to_results(r))

    await flyte.report.replace.aio(render_grid(
        prompts, specs, all_results,
        title="Video model comparison (text-to-video)",
        meta=f"{len(prompts)} prompt(s) · seed={seed} · "
             f"models: {', '.join(s.key for s in specs)}",
    ))
    await flyte.report.flush.aio()
    return runs


@orch_env.task(report=True)
async def animate(
    prompts: list[str],
    models: list[str] | None = None,
    image_model: str = DEFAULT_IMAGE_MODEL,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    width: int = 832,
    height: int = 480,
    num_frames: int = -1,
) -> list[ModelRun]:
    """Image-to-video: generate a first frame per prompt, then animate it.

    prompt --(sd-turbo)--> first frame --(wan22-ti2v-5b)--> clip

    This is the "join the two demos" path: the same trick the image-generation
    project does to make a picture, feeding the video model's I2V pipeline. The
    report shows the frame it started from next to each clip, so you can see what
    the video model kept and what it drifted away from.

    Only I2V-capable models are allowed (wan22-ti2v-5b, ltx2-distilled,
    cogvideox-5b); a text-only model like wan21-t2v-1.3b will fail fast with a
    message naming the ones that work.
    """
    specs = resolve_models(models or ["wan22-ti2v-5b"])
    bad = [s.key for s in specs if not s.supports_i2v]
    if bad:
        raise ValueError(
            f"These models are text-to-video only and can't animate a frame: {bad}. "
            f"I2V-capable: wan22-ti2v-5b, ltx2-distilled, cogvideox-5b."
        )

    ispec = get_image_spec(image_model)
    await flyte.report.replace.aio(render_status(
        "Image-to-video",
        f"Generating {len(prompts)} first frame(s) with {ispec.repo}, then animating "
        f"with {', '.join(s.key for s in specs)}.",
    ))
    await flyte.report.flush.aio()

    # First frames: fetch the (small) image model, generate one image per prompt.
    iw = await fetch_image_weights.override(short_name=f"fetch {image_model}")(image_model)
    first_frames = await make_first_frames.override(short_name=f"first frames {image_model}")(
        image_model, iw, prompts, width=width, height=height, seed=seed,
    )

    weights = [
        await fetch_weights.override(short_name=f"fetch {s.key}")(s.key) for s in specs
    ]
    runs: list[ModelRun] = await asyncio.gather(*[
        generate_for_model.override(short_name=f"animate {s.key}")(
            s.key, w, prompts, steps=steps, guidance=guidance, seed=seed,
            width=width, height=height, num_frames=num_frames, first_frames=first_frames,
        )
        for s, w in zip(specs, weights)
    ])

    all_results: list[ClipResult] = []
    for r in runs:
        all_results.extend(await _to_results(r))

    await flyte.report.replace.aio(render_grid(
        prompts, specs, all_results,
        title="Image-to-video (generated first frame, then animated)",
        meta=f"first frame: {image_model} · animated by: "
             f"{', '.join(s.key for s in specs)} · seed={seed}",
        first_frames=dict(zip(prompts, first_frames)),
    ))
    await flyte.report.flush.aio()
    return runs


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(
        compare,
        prompts=["a red panda barista pouring latte art, steam rising, cozy cafe, 50mm"],
    )
    print(f"Compare run: {run.name}")
    print(f"  {run.url}")
