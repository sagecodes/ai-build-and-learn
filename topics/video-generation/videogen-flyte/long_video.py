"""Long video by chaining clips: generate, take the last frame, generate again.

Every model in `models.py` has a fixed clip length baked into its latent shape
(`num_frames`), so a single call gets you ~2-5 seconds and no more. This module
buys length a different way: run the model's *image-to-video* pipeline, keep the
last frame, and feed it back in as the next chunk's starting frame.

    beat 1 ──> [chunk 1] ──last frame──> [chunk 2] ──last frame──> [chunk 3] ──> concat

The nice part is that each beat gets its own prompt, so the clip can tell a small
story ("the fox stops", "the fox looks up") rather than looping one motion. That is
something no single-shot call to these models can do at any length.

── Why this is a Flyte demo and not a for-loop ─────────────────────────────────
Because the alternative is one fragile 20-minute task. Chunking turns a long render
into N independently retryable units: chunk 4 OOMing doesn't throw away chunks 1-3,
and on a box where OOM is often transient (something else grabbed the unified pool)
that matters. It's also the honest way to show a dependency chain: chunk N+1
genuinely cannot start until chunk N is done, which is the shape a lot of real
pipelines have.

We keep the pipeline RESIDENT across chunks inside one task rather than making each
chunk its own task. Loading wan22-ti2v-5b costs ~60s and the model is 26GB; paying
that per chunk would dominate the runtime and, on the single-GPU box, each chunk
pod would serialize behind the last one's teardown anyway. So: one task, many
chunks, checkpointed into the report as it goes.

── What to actually look for (the result is the degradation) ───────────────────
This method drifts, and it should. Each hop re-encodes the previous chunk's *output*
frame, so VAE round-trip error compounds: expect contrast and saturation to creep
up, and fine texture to smear, the further you get from beat 1. The report prints
per-chunk brightness/contrast so the drift is measurable, not vibes.

That degradation is the point of the comparison with `skyreels-v2-df-1.3b`, which
does diffusion forcing: it keeps a real *latent history* across chunks instead of a
single RGB frame, so it should drift less. This module is the cheap trick that needs
no new model; SkyReels is the principled version. Showing both is the interesting
segment.

Usage (on the devbox):
    # 4 beats x ~2s = an ~8s clip from a 26GB model that can only do 2s at a time
    flyte run long_video.py long_video --beats '["a red fox trotting through tall grass, morning light", "the fox slows and stops, ears twitching", "the fox turns its head toward the camera", "the fox looks up at the sky"]'

    # more/longer chunks
    flyte run long_video.py long_video --beats '[...]' --num_frames 49
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import flyte
import flyte.io
import flyte.report

# Top-level imports: `flyte run` only bundles sibling modules it can see at import
# time, so these must not move inside a function.
from compare_pipeline import fetch_image_weights, fetch_weights, make_first_frames
from config import gpu_env, orch_env
from models import DEFAULT_IMAGE_MODEL, get_spec
from videogen_core import (
    ClipResult,
    build_clip_result,
    free_gpu_memory,
    load_pipeline,
    prepare_gpu,
    render_grid,
    render_status,
    timed_generate,
    write_mp4,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

# The model has to be able to start from an image, which is the whole mechanism.
DEFAULT_CHAIN_MODEL = "wan22-ti2v-5b"


@dataclass
class ChunkStat:
    """Per-chunk numbers, so drift across the chain is measurable."""
    beat: str
    seconds: float
    n_frames: int
    brightness: float = 0.0   # mean luma of the chunk's LAST frame (0-255)
    contrast: float = 0.0     # std of luma of that frame
    error: str = ""


@dataclass
class ChainRun:
    model_key: str
    chunks: list[ChunkStat] = field(default_factory=list)
    total_frames: int = 0
    fps: int = 0
    clips: flyte.io.Dir | None = None   # per-chunk mp4s + the concatenated one


def _luma_stats(img) -> tuple[float, float]:
    """(mean, std) of a PIL frame's luma. Cheap drift probe, no extra deps."""
    import numpy as np

    a = np.asarray(img.convert("L"), dtype="float32")
    return float(a.mean()), float(a.std())


def _renorm_to(img, ref_mean, ref_std, strength: float = 1.0):
    """Rescale `img`'s per-channel mean/std toward the anchor's, before hand-off.

    ── Why ─────────────────────────────────────────────────────────────────────
    What kills a chained video is **exposure bias**: the model was trained on clean
    history and gets its own imperfect output at inference, so per-hop error compounds
    instead of averaging out. We measured it: +32% contrast over 4 hops on the boat,
    and total subject collapse on the walking human.

    The literature's answer is to re-anchor each step rather than trust the drifted
    frame (StreamingT2V re-injects features from an anchor frame; FramePack generates
    endpoints first and interpolates; SkyReels adds noise to the conditioning). This is
    the cheapest member of that family: match the hand-off frame's first and second
    moments back to chunk 1's, per channel, so the VAE round-trip's contrast/colour
    creep is corrected *before* it can seed the next chunk.

    It only treats the statistics, not the semantics: it cannot stop a subject from
    turning around and walking at the camera (that's a geometry error, and it needs an
    actual anchor, i.e. VACE `reference_images`). Expect it to flatten the contrast
    curve and do nothing for the human run's collapse. Worth knowing which failure you
    are fixing.

    `strength` blends: 0 = off (raw hand-off), 1 = fully re-normalized.
    """
    import numpy as np
    from PIL import Image

    a = np.asarray(img, dtype="float32")
    out = np.empty_like(a)
    for c in range(a.shape[2]):
        ch = a[..., c]
        m, s = float(ch.mean()), float(ch.std())
        if s < 1e-6:
            out[..., c] = ch
            continue
        fixed = (ch - m) / s * ref_std[c] + ref_mean[c]
        out[..., c] = ch + (fixed - ch) * strength
    return Image.fromarray(np.clip(out, 0, 255).astype("uint8"))


def _rgb_stats(img) -> tuple[list[float], list[float]]:
    """Per-channel (means, stds) of a PIL RGB frame: the anchor for _renorm_to."""
    import numpy as np

    a = np.asarray(img.convert("RGB"), dtype="float32")
    return ([float(a[..., c].mean()) for c in range(3)],
            [float(a[..., c].std()) for c in range(3)])


@gpu_env.task(report=True, retries=2)
async def chain_from_frame(
    model_key: str,
    weights: flyte.io.Dir,
    beats: list[str],
    first_frame: str,                 # data URI: the image chunk 1 starts from
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    width: int = -1,
    height: int = -1,
    num_frames: int = -1,
    renorm: float = 0.0,
) -> ChainRun:
    """Render one chunk per beat, each starting from the previous chunk's last frame.

    `renorm` (0..1) re-anchors each hand-off frame's per-channel mean/std to chunk 1's
    before it seeds the next chunk: the cheap fix for the contrast creep this method
    accumulates. Defaults to **0 (off)** so the drift stays visible and the A/B against
    the measured baseline (+32% contrast over 4 hops) is honest. See `_renorm_to`.

    Returns the per-chunk mp4s plus `chained.mp4`, which is every chunk concatenated.
    """
    from compare_pipeline import _decode_uri

    spec = get_spec(model_key)
    if not spec.supports_i2v:
        raise ValueError(
            f"{model_key} can't start from a frame, so it can't be chained. "
            f"I2V-capable: wan22-ti2v-5b, ltx2-distilled, cogvideox-5b."
        )

    out_dir = Path(tempfile.mkdtemp(prefix=f"chain_{model_key}_"))
    await flyte.report.replace.aio(render_status(
        f"Chaining {len(beats)} chunks with {spec.key}",
        f"{spec.repo} · each chunk starts from the previous chunk's last frame",
    ))
    await flyte.report.flush.aio()

    local = await weights.download()
    pipe = None
    try:
        prepare_gpu(spec)
        pipe = load_pipeline(spec, model_path=local, i2v=True)

        kw = dict(
            steps=None if steps < 0 else steps,
            guidance=None if guidance < 0 else guidance,
            width=None if width < 0 else width,
            height=None if height < 0 else height,
            num_frames=None if num_frames < 0 else num_frames,
            negative_prompt=spec.negative_prompt or None,
        )

        start_img = _decode_uri(first_frame)
        all_frames: list = []
        stats: list[ChunkStat] = []
        results: list[ClipResult] = []

        for i, beat in enumerate(beats):
            log.info(f"[chain] chunk {i + 1}/{len(beats)}: {beat[:60]}")

            def _on_step(k, total, _i=i):
                if k == 1 or k % 5 == 0 or k == total:
                    log.info(f"[chain] chunk {_i + 1}: step {k}/{total}")

            try:
                # Vary the seed per chunk. With one fixed seed every chunk resolves
                # toward the same motion from a near-identical starting frame, and
                # the chain comes out looking like a stutter loop.
                frames, audio, sr, secs = timed_generate(
                    pipe, spec, beat, image=start_img, on_step=_on_step,
                    seed=(seed + i if seed >= 0 else seed), **kw,
                )
            except Exception as e:
                # OOM is a pod-level condition and often transient here: re-raise so
                # Flyte retries the whole task in a fresh pod. Any other failure ends
                # the chain, because chunk N+1 has no frame to start from without N.
                if "out of memory" in str(e).lower() or type(e).__name__ == "OutOfMemoryError":
                    log.warning(f"[chain] CUDA OOM on chunk {i}; failing task to retry")
                    free_gpu_memory()
                    raise
                log.warning(f"[chain] chunk {i} failed, stopping chain: {e}")
                stats.append(ChunkStat(beat=beat, seconds=0.0, n_frames=0, error=str(e)))
                break

            # Drop each later chunk's frame 0: it's a re-render of the frame we fed
            # in, so keeping it would duplicate the seam and read as a hitch.
            new = frames if i == 0 else frames[1:]
            all_frames.extend(new)

            b, c = _luma_stats(frames[-1])
            stats.append(ChunkStat(beat=beat, seconds=secs, n_frames=len(new),
                                   brightness=b, contrast=c))

            write_mp4(frames, spec.fps, out_dir / f"chunk_{i:02d}.mp4",
                      audio=audio, sample_rate=sr)
            results.append(build_clip_result(spec, f"chunk {i + 1}: {beat}",
                                             frames, audio, sr, secs))

            # The hand-off. Chunk 1's FIRST frame is the anchor: it is the only frame
            # in the whole chain that never went through a VAE round trip, so it is
            # the one honest reference for "what this scene is supposed to look like".
            start_img = frames[-1]
            if i == 0:
                anchor_mean, anchor_std = _rgb_stats(frames[0])
            if renorm > 0:
                before = _luma_stats(start_img)
                start_img = _renorm_to(start_img, anchor_mean, anchor_std, renorm)
                after = _luma_stats(start_img)
                log.info(f"[chain] renorm@{renorm}: hand-off luma "
                         f"{before[0]:.1f}±{before[1]:.1f} -> {after[0]:.1f}±{after[1]:.1f}")

            log.info(f"[chain] chunk {i + 1} done in {secs:.0f}s · "
                     f"luma {b:.1f}±{c:.1f} · {len(all_frames)} frames so far")

            await flyte.report.replace.aio(render_grid(
                [r.prompt for r in results], [spec], results,
                title=f"Chained long video · {spec.key}",
                meta=(f"{i + 1}/{len(beats)} chunks · {len(all_frames)} frames · "
                      f"{len(all_frames) / spec.fps:.1f}s so far"),
            ))
            await flyte.report.flush.aio()

        if not all_frames:
            raise RuntimeError("chain produced no frames; see the chunk error above")

        # The payoff: every chunk as one continuous clip.
        write_mp4(all_frames, spec.fps, out_dir / "chained.mp4")

        drift = ""
        ok = [s for s in stats if not s.error]
        if len(ok) >= 2:
            db = ok[-1].brightness - ok[0].brightness
            dc = ok[-1].contrast - ok[0].contrast
            drift = (f" · drift over {len(ok)} chunks: brightness {db:+.1f}, "
                     f"contrast {dc:+.1f} (0,0 = none)")
        log.info(f"[chain] {len(all_frames)} frames total{drift}")

        final = build_clip_result(spec, "the whole chain, concatenated",
                                  all_frames, None, None,
                                  sum(s.seconds for s in stats))
        await flyte.report.replace.aio(render_grid(
            ["the whole chain, concatenated"] + [r.prompt for r in results],
            [spec], [final] + results,
            title=f"Chained long video · {spec.key}",
            meta=(f"{len(ok)} chunks · {len(all_frames)} frames · "
                  f"{len(all_frames) / spec.fps:.1f}s @{spec.fps}fps{drift}"),
        ))
        await flyte.report.flush.aio()

        clips = await flyte.io.Dir.from_local(str(out_dir))
        return ChainRun(model_key=model_key, chunks=stats,
                        total_frames=len(all_frames), fps=spec.fps, clips=clips)
    finally:
        pipe = None
        free_gpu_memory()


@gpu_env.task(report=True, retries=2)
async def polish_windows(
    model_key: str,
    weights: flyte.io.Dir,
    source_clip: flyte.io.Dir,
    prompt: str,
    strength: float = 0.4,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
) -> ChainRun:
    """Re-render a chained clip with TRUE video-to-video, in independent windows.

    ── The right tool for the "chain then refine" idea ─────────────────────────
    The first attempt at this (`vace.py refine`) used VACE's control mode and made a
    good clip worse: it fed VACE **edge maps**, which discard all appearance, and
    asked one `reference_images` frame to rebuild the scene. The coat came back black
    and the window seams popped at 2.7-3.4x.

    `WanVideoToVideoPipeline` is the tool that job actually wanted:

      * it takes the **real frames**, not a control signal, so appearance is preserved
        by construction rather than reconstructed from nothing;
      * it has **`strength`** (img2img-style): 0.3 = a light touch-up, 0.8 = a heavy
        restyle. VACE's mask is binary (keep / generate) and has no such knob;
      * it loads the **wan22-ti2v-5b checkpoint we already have cached** — a different
        pipeline class over the same weights. No download, and 5B instead of VACE's
        1.3B.

    Windows are still independent (each one's input is its own slice of the ORIGINAL
    clip), so there is no hand-off for error to accumulate through — that part of the
    idea was always right. But because low `strength` barely moves each window, the
    boundary pops should be far milder than VACE's.

    Expect `strength` to be the whole story: too low and it changes nothing, too high
    and each window drifts off on its own and the seams come back.
    """
    import time

    import torch

    spec = get_spec(model_key)
    if not spec.supports_v2v:
        raise ValueError(f"{model_key} has no v2v pipeline. Try wan22-ti2v-5b.")

    out_dir = Path(tempfile.mkdtemp(prefix=f"polish_{model_key}_"))
    await flyte.report.replace.aio(render_status(
        f"Polish (true v2v, strength={strength})",
        f"{spec.repo} · {spec.v2v_pipeline} · re-rendering a chained clip in windows",
    ))
    await flyte.report.flush.aio()

    src_local = await source_clip.download()
    mp4s = sorted(Path(src_local).glob("*.mp4"))   # chained.mp4 sorts before chunk_*
    if not mp4s:
        raise ValueError(f"no .mp4 in {src_local}")

    import av
    from PIL import Image

    c = av.open(str(mp4s[0]))
    fps = int(round(float(c.streams.video[0].average_rate or 24)))
    frames = [Image.fromarray(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]
    n = spec.num_frames
    log.info(f"[polish] {mp4s[0].name}: {len(frames)} frames @{fps}fps -> "
             f"{-(-len(frames) // n)} windows of {n}")

    local = await weights.download()
    pipe = None
    try:
        prepare_gpu(spec)
        pipe = load_pipeline(spec, model_path=local, v2v=True)

        out_frames: list = []
        stats: list[ChunkStat] = []
        t0 = time.time()
        for wi in range(0, len(frames), n):
            window = frames[wi:wi + n]
            short = len(window)
            if short < n:
                window = window + [window[-1]] * (n - short)
            log.info(f"[polish] window {wi // n + 1}: frames {wi}-{wi + short - 1}")
            out = pipe(
                video=window,
                prompt=prompt,
                negative_prompt=spec.negative_prompt or None,
                height=spec.height, width=spec.width,
                num_inference_steps=spec.steps if steps < 0 else steps,
                guidance_scale=spec.guidance if guidance < 0 else guidance,
                strength=strength,
                generator=(torch.Generator(device="cuda").manual_seed(seed)
                           if seed >= 0 else None),
                output_type="pil",
            ).frames[0][:short]
            out_frames.extend(out)
            b, cst = _luma_stats(out[-1])
            stats.append(ChunkStat(beat=f"window {wi // n + 1}", seconds=0.0,
                                   n_frames=len(out), brightness=b, contrast=cst))
        secs = time.time() - t0

        write_mp4(out_frames, fps, out_dir / "polished.mp4")
        d = stats[-1].contrast - stats[0].contrast
        log.info(f"[polish] {len(out_frames)} frames in {secs:.0f}s · "
                 f"contrast drift {d:+.1f}")

        spec_v = get_spec(model_key)
        results = [
            build_clip_result(spec_v, "source (chained)", frames, None, None, 0.0, fps=fps),
            build_clip_result(spec_v, f"polished @strength={strength}", out_frames,
                              None, None, secs, fps=fps),
        ]
        await flyte.report.replace.aio(render_grid(
            [r.prompt for r in results], [spec_v], results,
            title=f"Polish: true v2v, strength={strength}",
            meta=(f"{len(out_frames)} frames · {len(stats)} independent windows · "
                  f"contrast drift {d:+.1f} · {secs:.0f}s"),
        ))
        await flyte.report.flush.aio()

        clips = await flyte.io.Dir.from_local(str(out_dir))
        return ChainRun(model_key=model_key, chunks=stats, total_frames=len(out_frames),
                        fps=fps, clips=clips)
    finally:
        pipe = None
        free_gpu_memory()


# WanVideoToVideoPipeline only loads a Wan 2.1 VAE (8x spatial, z_dim 16). The 2.2
# TI2V-5B checkpoint ships a 16x/z_dim-48 VAE and dies on a latent shape mismatch, so
# polish runs on the 1.3B. That is a real quality cost (1.3B re-rendering a 5B's clip)
# and the honest reason to reach for VACE-14B or Wan 2.1 14B instead if it shows.
POLISH_MODEL = "wan21-t2v-1.3b"


@orch_env.task(report=True)
async def polish(
    source_clip_uri: str,
    prompt: str,
    model: str = POLISH_MODEL,
    strength: float = 0.4,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
) -> ChainRun:
    """Chain first, then polish: true video-to-video over a chained clip.

        flyte run long_video.py polish --strength 0.4 \
            --source_clip_uri s3://flyte-data/.../chain_wan22-ti2v-5b_xxxx \
            --prompt "a person in a long red coat walking away down a wet neon street at night"

    See `polish_windows` for why this is the tool `vace.py refine` should have been.
    """
    w = await fetch_weights.override(short_name=f"fetch {model}")(model)
    return await polish_windows.override(short_name=f"polish @{strength}")(
        model, w, flyte.io.Dir.from_existing_remote(source_clip_uri), prompt,
        strength=strength, steps=steps, guidance=guidance, seed=seed,
    )


@orch_env.task(report=True)
async def long_video(
    beats: list[str],
    model: str = DEFAULT_CHAIN_MODEL,
    image_model: str = DEFAULT_IMAGE_MODEL,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    width: int = 832,
    height: int = 480,
    num_frames: int = -1,
    renorm: float = 0.0,
) -> ChainRun:
    """Beats -> one long clip, by chaining image-to-video on the last frame.

        beats[0] ──(sd-turbo)──> first frame ──(wan22-ti2v-5b)──> chunk 1
                                                     └─last frame─> chunk 2 ...

    Each beat is one chunk's prompt, so the scene can evolve across the clip
    instead of looping. Length is `len(beats) * num_frames`, which is the point:
    it breaks the model's fixed per-call frame budget.
    """
    if not beats:
        raise ValueError("pass at least one beat")

    spec = get_spec(model)
    if not spec.supports_i2v:
        raise ValueError(
            f"{model} can't start from a frame, so it can't be chained. "
            f"I2V-capable: wan22-ti2v-5b, ltx2-distilled, cogvideox-5b."
        )

    await flyte.report.replace.aio(render_status(
        "Long video by chaining",
        f"{len(beats)} beats · first frame from {image_model} · chained by {model}",
    ))
    await flyte.report.flush.aio()

    # Beat 1 also seeds the look of the whole chain, so it's the first frame's prompt.
    iw = await fetch_image_weights.override(short_name=f"fetch {image_model}")(image_model)
    frames = await make_first_frames.override(short_name=f"first frame {image_model}")(
        image_model, iw, [beats[0]], width=width, height=height, seed=seed,
    )

    w = await fetch_weights.override(short_name=f"fetch {model}")(model)
    run = await chain_from_frame.override(short_name=f"chain {len(beats)} chunks")(
        model, w, beats, frames[0], steps=steps, guidance=guidance, seed=seed,
        width=width, height=height, num_frames=num_frames, renorm=renorm,
    )
    return run


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
