"""Video-to-video with Wan VACE: put a video IN, get a restyled video OUT.

Every other model in this repo goes text -> video or image -> video. VACE is the
only one that takes **video as input**, which makes it the one genuinely new axis
here rather than another quality datapoint. It's also the smallest model we carry
(19GB), which is a nice inversion: the new capability is the cheap one.

    source clip ──(edge map per frame)──> control video ─┐
                                                          ├──(VACE)──> restyled clip
    "the same scene, but <new style>" ───────────────────┘

── How VACE actually works (read this before changing anything) ────────────────
`WanVACEPipeline.__call__` takes `video`, `mask` and `reference_images` instead of
just a prompt. The mask semantics are the opposite of the intuition, and they come
straight from the diffusers example:

    mask BLACK (0)   = KEEP this frame  (it is given; condition on it)
    mask WHITE (255) = GENERATE this frame

So the first-last-frame trick in the docs passes [first, gray, gray, ..., last] as
`video` with mask [black, white, ..., white, black]: keep the two real frames,
invent the middle. See `flf2v` below, which is that exact path.

For **control** tasks (what `restyle` does), the mask is all white (generate every
frame) and the `video` is not RGB footage at all: it's a *control signal* derived
from the source, which is what VACE was trained to consume (depth, pose, scribble).
Passing raw RGB frames with an all-white mask would just throw the source away and
degenerate into plain text-to-video. This is the single easiest thing to get wrong.

We use **edge maps** as the control signal (VACE's "scribble" mode). That's a
deliberate dependency choice: a depth model would mean another 1-2GB download and
another failure mode, while edges are ~15 lines of numpy and need nothing new. The
trade is that edges carry no depth ordering, so VACE has more freedom (and more room
to hallucinate) than a depth map would give it.

── Pick your source clip for its EDGES, not its looks ──────────────────────────
Verified locally against real clips from this repo (2026-07-16), and it decides
whether the whole thing works:

  boat on a dark puddle  -> a clean control map: hull outline, mast crease, the
                            reflection, the bokeh rings. VACE has real structure.
  fox in tall grass      -> useless. At any threshold the edges are spent on grass
                            blades and the fox has no outline at all. The texture IS
                            the gradient, so edge control cannot see the subject.

So high-contrast subjects on uncluttered backgrounds work; textured naturalistic
scenes do not. That's a property of edge control, not of VACE, and it's why the
`restyle` default source is the boat. A depth-map control signal would not have this
failure (it would see the fox as a shape) at the cost of another model download.

── Status ──────────────────────────────────────────────────────────────────────
NOT YET RUN on this box. The pipeline class, the repo, the mask semantics and the
flow_shift are all checked against diffusers 0.39 and the model card, but no clip has
come out of it here. Treat the first run as a debugging session.

Usage (on the devbox):
    # generate a source clip, then restyle it
    flyte run vace.py restyle \
        --source_prompt "a red fox trotting through tall grass, morning light" \
        --style_prompt  "the same scene as a pen and ink sketch, cross-hatched, monochrome"

    # restyle a clip we already made (skips regenerating it)
    flyte run vace.py restyle --style_prompt "..." \
        --source_clip_uri s3://flyte-data/.../clips_wan22-ti2v-5b_xxxx

    # first-last-frame: give it two images, it invents the motion between them
    flyte run vace.py flf2v --prompt "the boat drifts across the puddle"
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from compare_pipeline import (
    fetch_image_weights,
    fetch_weights,
    generate_for_model,
    make_first_frames,
)
from config import gpu_env, orch_env
from models import DEFAULT_IMAGE_MODEL, get_image_spec, get_spec
from videogen_core import (
    ClipResult,
    build_clip_result,
    free_gpu_memory,
    load_pipeline,
    prepare_gpu,
    render_grid,
    render_status,
    write_mp4,
)

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

VACE_MODEL = "wan21-vace-1.3b"
SOURCE_MODEL = "wan22-ti2v-5b"     # what makes the source clip when you don't supply one


@dataclass
class RestyleRun:
    model_key: str
    seconds: float = 0.0
    n_frames: int = 0
    fps: int = 0
    error: str = ""
    clips: flyte.io.Dir | None = None   # source.mp4, control.mp4, restyled.mp4


def _edge_frames(frames: list, keep_pct: float = 8.0, blur: float = 1.5) -> list:
    """RGB frames -> white-on-black edge maps (VACE's 'scribble' control signal).

    Deliberately numpy+PIL only: a depth estimator would be a better control signal
    but costs another model download and another failure mode, and edges are enough
    to prove the axis works.

    Two things here are not incidental, both learned by looking at the output:

    1. **Blur before Sobel.** A raw Sobel on a textured scene (grass, gravel, water)
       fires on every blade, and the *subject* vanishes into the texture. A ~1.5px
       Gaussian first suppresses fine detail and keeps real contours.
    2. **Threshold by percentile, not a constant.** The first version used a fixed
       cutoff on the max-normalized magnitude, which produced a **79% white** map on
       a fox-in-grass frame: the fox had no outline at all and the control video was
       noise. A percentile keeps edge density fixed (~`keep_pct`%) no matter how
       textured the scene is, which is the property we actually want.

    3. **Gradients per RGB channel, not on luma.** This one is not obvious and it is
       the difference between seeing your subject and not. A luma-only Sobel is
       COLOURBLIND: a red coat on a teal-dark street is a huge *colour* contrast and a
       weak *luma* one, so converting to "L" first made the most salient object in the
       frame vanish while every neon sign lit up. We take the per-channel gradient and
       keep the strongest, so colour boundaries survive.

    `keep_pct` is the knob: lower = only the strongest contours (VACE freer to
    invent), higher = denser structure (VACE more constrained).
    """
    import numpy as np
    from PIL import Image, ImageFilter

    def _sobel(g):
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[1:-1, 1:-1] = (
            g[:-2, 2:] + 2 * g[1:-1, 2:] + g[2:, 2:]
            - g[:-2, :-2] - 2 * g[1:-1, :-2] - g[2:, :-2]
        )
        gy[1:-1, 1:-1] = (
            g[2:, :-2] + 2 * g[2:, 1:-1] + g[2:, 2:]
            - g[:-2, :-2] - 2 * g[:-2, 1:-1] - g[:-2, 2:]
        )
        return np.hypot(gx, gy)

    out = []
    for f in frames:
        src = f.convert("RGB")
        if blur:
            src = src.filter(ImageFilter.GaussianBlur(radius=blur))
        a = np.asarray(src, dtype=np.float32)
        # Strongest response across R, G, B: a boundary that exists in ANY channel is
        # a boundary. (max, not mean: averaging dilutes a single-channel colour edge.)
        mag = np.maximum.reduce([_sobel(a[..., c]) for c in range(3)])
        thr = float(np.percentile(mag, 100.0 - keep_pct))
        edges = np.where(mag > thr, 255, 0).astype("uint8")
        out.append(Image.fromarray(edges).convert("RGB"))
    return out


def _read_mp4(path: str | Path) -> tuple[list, int]:
    """mp4 -> (list[PIL], fps). PyAV, same as everywhere else in this repo."""
    import av
    from PIL import Image

    c = av.open(str(path))
    stream = c.streams.video[0]
    fps = int(round(float(stream.average_rate or 24)))
    frames = [Image.fromarray(f.to_ndarray(format="rgb24")) for f in c.decode(video=0)]
    return frames, fps


@gpu_env.task(report=True, retries=2)
async def vace_restyle(
    weights: flyte.io.Dir,
    source_clip: flyte.io.Dir,
    style_prompt: str,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    conditioning_scale: float = 1.0,
) -> RestyleRun:
    """Take a source clip, derive an edge-map control video, and restyle it."""
    import torch

    spec = get_spec(VACE_MODEL)
    out_dir = Path(tempfile.mkdtemp(prefix="vace_"))

    await flyte.report.replace.aio(render_status(
        "VACE video-to-video",
        f"{spec.repo} · restyling to: {style_prompt}",
    ))
    await flyte.report.flush.aio()

    src_local = await source_clip.download()
    mp4s = sorted(Path(src_local).glob("*.mp4"))
    if not mp4s:
        raise ValueError(f"no .mp4 in the source clip dir {src_local}")
    frames, fps = _read_mp4(mp4s[0])
    log.info(f"[vace] source: {mp4s[0].name}, {len(frames)} frames @{fps}fps")

    # VACE's frame budget is fixed by the latent shape like every other model here,
    # so the control video has to match num_frames exactly.
    n = spec.num_frames
    if len(frames) < n:
        raise ValueError(
            f"source clip has {len(frames)} frames but VACE wants {n}. "
            f"Regenerate the source with --num_frames {n}."
        )
    frames = frames[:n]
    w, h = spec.width, spec.height
    frames = [f.resize((w, h)) for f in frames]

    control = _edge_frames(frames)
    write_mp4(frames, fps, out_dir / "source.mp4")
    write_mp4(control, fps, out_dir / "control.mp4")

    local = await weights.download()
    pipe = None
    try:
        prepare_gpu(spec)
        pipe = load_pipeline(spec, model_path=local)

        from PIL import Image

        # All-white mask = generate every frame; the control video is what steers it.
        # (Black would mean "keep this frame", which would just hand back the input.)
        mask = [Image.new("L", (w, h), 255)] * n

        import time

        t0 = time.time()
        result = pipe(
            prompt=style_prompt,
            negative_prompt=spec.negative_prompt or None,
            video=control,
            mask=mask,
            conditioning_scale=conditioning_scale,
            height=h,
            width=w,
            num_frames=n,
            num_inference_steps=spec.steps if steps < 0 else steps,
            guidance_scale=spec.guidance if guidance < 0 else guidance,
            generator=(torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None),
            output_type="pil",
        )
        secs = time.time() - t0
        out = result.frames[0]
        write_mp4(out, fps, out_dir / "restyled.mp4")
        log.info(f"[vace] restyled {len(out)} frames in {secs:.0f}s")

        results = [
            build_clip_result(spec, f"source ({SOURCE_MODEL})", frames, None, None, 0.0, fps=fps),
            build_clip_result(spec, "control (edge map)", control, None, None, 0.0, fps=fps),
            build_clip_result(spec, f"restyled: {style_prompt}", out, None, None, secs, fps=fps),
        ]
        await flyte.report.replace.aio(render_grid(
            [r.prompt for r in results], [spec], results,
            title="VACE video-to-video",
            meta=(f"{spec.repo} · edge-map control · conditioning_scale={conditioning_scale} "
                  f"· {secs:.0f}s · seed={seed}"),
        ))
        await flyte.report.flush.aio()

        clips = await flyte.io.Dir.from_local(str(out_dir))
        return RestyleRun(model_key=spec.key, seconds=secs, n_frames=len(out),
                          fps=fps, clips=clips)
    finally:
        pipe = None
        free_gpu_memory()


@orch_env.task(report=True)
async def restyle(
    style_prompt: str,
    # Default deliberately chosen for its CONTROL MAP, not its looks. See the
    # "pick your source" note in the module docstring: a boat on a dark puddle gives
    # crisp contours, a fox in grass gives 8% of the frame in grass blades and no
    # animal. The source clip's edge-friendliness decides whether this works.
    source_prompt: str = ("a yellow paper boat on a dark rain puddle at night, "
                          "cinematic, shallow depth of field"),
    source_clip_uri: str = "",
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    conditioning_scale: float = 1.0,
) -> RestyleRun:
    """Generate (or reuse) a source clip, then restyle it with VACE.

        source_prompt --(wan22-ti2v-5b)--> clip --(edges)--> control --(VACE)--> restyled

    Pass `source_clip_uri` (the Dir from any earlier run, e.g. a `compare` or
    `long_video` output) to skip regenerating the source, which is the fast path once
    you have a clip you like.
    """
    vspec = get_spec(VACE_MODEL)
    await flyte.report.replace.aio(render_status(
        "VACE video-to-video",
        f"source: {'reusing ' + source_clip_uri if source_clip_uri else source_prompt} · "
        f"style: {style_prompt}",
    ))
    await flyte.report.flush.aio()

    if source_clip_uri:
        source = flyte.io.Dir.from_existing_remote(source_clip_uri)
    else:
        # Match VACE's frame budget at generation time so the control video lines up.
        sw = await fetch_weights.override(short_name=f"fetch {SOURCE_MODEL}")(SOURCE_MODEL)
        run = await generate_for_model.override(short_name=f"source {SOURCE_MODEL}")(
            SOURCE_MODEL, sw, [source_prompt], seed=seed,
            width=vspec.width, height=vspec.height, num_frames=vspec.num_frames,
        )
        if run.clips is None:
            raise RuntimeError("source generation produced no clips")
        source = run.clips

    vw = await fetch_weights.override(short_name=f"fetch {VACE_MODEL}")(VACE_MODEL)
    return await vace_restyle.override(short_name="vace restyle")(
        vw, source, style_prompt, steps=steps, guidance=guidance, seed=seed,
        conditioning_scale=conditioning_scale,
    )


@gpu_env.task(report=True, retries=2)
async def flf2v(
    weights: flyte.io.Dir,
    first_frame: str,
    last_frame: str,
    prompt: str,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
) -> RestyleRun:
    """First-last-frame to video: give two images, VACE invents the motion between.

    This is the *documented* VACE path (it's the diffusers example verbatim), so it's
    the lower-risk way to prove the model loads and runs before trusting `restyle`.
    Frames are data URIs, same as `make_first_frames` produces.
    """
    import time

    import torch
    from PIL import Image

    from compare_pipeline import _decode_uri

    spec = get_spec(VACE_MODEL)
    out_dir = Path(tempfile.mkdtemp(prefix="vace_flf2v_"))
    w, h, n = spec.width, spec.height, spec.num_frames

    first = _decode_uri(first_frame).resize((w, h))
    last = _decode_uri(last_frame).resize((w, h))

    # The example's exact construction: real frames at the ends, gray in between,
    # and a mask that says "keep the ends, generate the middle".
    video = [first] + [Image.new("RGB", (w, h), (128, 128, 128))] * (n - 2) + [last]
    mask = ([Image.new("L", (w, h), 0)]
            + [Image.new("L", (w, h), 255)] * (n - 2)
            + [Image.new("L", (w, h), 0)])

    local = await weights.download()
    pipe = None
    try:
        prepare_gpu(spec)
        pipe = load_pipeline(spec, model_path=local)
        t0 = time.time()
        result = pipe(
            prompt=prompt,
            negative_prompt=spec.negative_prompt or None,
            video=video, mask=mask,
            height=h, width=w, num_frames=n,
            num_inference_steps=spec.steps if steps < 0 else steps,
            guidance_scale=spec.guidance if guidance < 0 else guidance,
            generator=(torch.Generator(device="cuda").manual_seed(seed) if seed >= 0 else None),
            output_type="pil",
        )
        secs = time.time() - t0
        out = result.frames[0]
        write_mp4(out, spec.fps, out_dir / "flf2v.mp4")

        r = build_clip_result(spec, f"first->last: {prompt}", out, None, None, secs)
        await flyte.report.replace.aio(render_grid(
            [r.prompt], [spec], [r],
            title="VACE first-last-frame to video",
            meta=f"{spec.repo} · {n} frames · {secs:.0f}s",
            first_frames={r.prompt: first_frame},
        ))
        await flyte.report.flush.aio()

        clips = await flyte.io.Dir.from_local(str(out_dir))
        return RestyleRun(model_key=spec.key, seconds=secs, n_frames=len(out),
                          fps=spec.fps, clips=clips)
    finally:
        pipe = None
        free_gpu_memory()


@orch_env.task(report=True)
async def bookend(
    first_prompt: str,
    last_prompt: str,
    prompt: str,
    image_model: str = DEFAULT_IMAGE_MODEL,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
) -> RestyleRun:
    """Generate BOTH endpoints, then have VACE invent the motion between them.

        first_prompt --(sdxl-turbo)--> frame A ─┐
                                                ├──(VACE flf2v)──> clip
        last_prompt  --(sdxl-turbo)--> frame B ─┘

    ── Why this is the answer to the chain's runaway ───────────────────────────
    `long_video.py` fails on a trajectory: the person turned around and each chunk
    faithfully walked them a bit closer until the camera was inside them (see the
    README). Nothing could correct it, because each chunk only ever sees ONE frame
    and has no idea where the shot is supposed to END.

    This inverts that. Fix both endpoints first and interpolate between them, so the
    trajectory is *bounded by construction*: the model cannot walk the subject into
    the lens because it has been told, up front, that the last frame is a small
    figure far down the street. That's FramePack's "anti-drifting sampling" — anchor
    the endpoints, generate the middle with bidirectional context — using VACE's
    first-last-frame path as the primitive.

    The trade: you get 49 frames, not an unbounded chain. Bounded is the point.

    Usage:
        flyte run vace.py bookend \
            --first_prompt "a person in a red coat close to the camera on a neon street at night, seen from behind" \
            --last_prompt  "the same neon street at night, a tiny distant figure in a red coat far down the street" \
            --prompt "the person walks away from the camera down the street, receding into the distance"
    """
    ispec = get_image_spec(image_model)
    await flyte.report.replace.aio(render_status(
        "VACE bookend (endpoint-pinned)",
        f"first: {first_prompt}<br>last: {last_prompt}<br>motion: {prompt}",
    ))
    await flyte.report.flush.aio()

    vspec = get_spec(VACE_MODEL)
    iw = await fetch_image_weights.override(short_name=f"fetch {image_model}")(image_model)
    frames = await make_first_frames.override(short_name="endpoints")(
        image_model, iw, [first_prompt, last_prompt],
        width=vspec.width, height=vspec.height, seed=seed,
    )
    vw = await fetch_weights.override(short_name=f"fetch {VACE_MODEL}")(VACE_MODEL)
    return await flf2v.override(short_name="vace flf2v")(
        vw, frames[0], frames[1], prompt, steps=steps, guidance=guidance, seed=seed,
    )


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
