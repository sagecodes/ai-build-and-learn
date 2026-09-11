"""Loading Cosmos 3 and rolling it forward, with the GB10-specific care it needs.

Split out of pipeline.py so the model code can be exercised from `smoke_test.py` on
the host without Flyte in the picture, which is the fast iteration loop.

── The three things that go wrong on this box ──────────────────────────────────
1. OVERSIZED LOADS HANG, THEY DO NOT OOM. The GB10 has one 119.7 GiB pool shared by
   the GPU, the OS and every other pod. Ask for more than is free and the box wedges
   rather than raising, so `guard_memory()` sets a hard cap BEFORE any weights are
   touched. It caps against FREE, not total: `set_per_process_memory_fraction` takes
   a share of TOTAL, and on a box where a leaky object store is already holding 40 GB
   a 0.9-of-total cap is not a cap at all. That mistake produces an OOM followed by a
   bare SIGSEGV (exit 139).

2. THE DOUBLE-COPY LOAD TRAP. `from_pretrained(...).to("cuda")` materializes the
   whole model on the host and then copies it, so it needs 2x the model on one pool.
   `device_map="cuda"` streams shards straight to the device instead, which is also
   what NVIDIA's own example runner does. Never add a `.to()` after this call.

3. NO torch.compile. Triton does not emit working SASS for sm_121a yet, so it fails
   or silently falls back to something slower than eager.

`faulthandler` is armed at import: a native crash inside a CUDA kernel or a codec
otherwise produces a bare exit 139 with no Python traceback at all, and the pod log
is the only place that information can come from.
"""

from __future__ import annotations

import faulthandler
import json
import logging
import os
import time

faulthandler.enable()

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# The canonical action-conditioning example ships INSIDE the checkpoint, which is
# why these are asset paths and not URLs: by the time we need them the snapshot is
# already on local disk, so there is no second network dependency to fail.
ACTION_FRAME = "assets/example_action_fd_agibotworld_first_frame.png"
ACTION_CHUNKS = "assets/example_action_fd_agibotworld_action_chunks.json"
I2V_FRAME = "assets/example_i2v_input.jpg"

# The reasoning example ships in the checkpoint too, and it is the tell that the
# understanding surface is a first-class mode rather than something bolted on: a
# photograph, a goal, and a token budget, with no video anywhere in it.
REASON_FRAME = "assets/example_reasoning_input.png"
REASON_PROMPT = "assets/example_reasoning_prompt.json"

# Inverse dynamics ships with ground truth, and that is the whole reason to run it.
# Two AV clips, 61 frames of 832x480 at 10 fps, each paired with the 60 actions that
# connect those frames as a [60, 9] float32 array. Forward dynamics can only be judged
# by eye; this one has a right answer, so the output is a number rather than a vibe.
INVERSE_CLIP = "assets/example_action_id_av_{i}_input.mp4"
INVERSE_TRUTH = "assets/example_action_id_av_{i}_output.json"
INVERSE_DOMAIN = "av"          # 9-D: the widths live in _EMBODIMENT_TO_RAW_ACTION_DIM
INVERSE_TIER = 480             # matches the clips' native 832x480, so no rescale
INVERSE_FPS = 10.0

# The action model was trained on structured JSON captions, and the pipeline builds
# that JSON itself from this sentence. The clips are forward driving footage, and
# nothing here tells the model what the actions were: it has to read them off the
# video, which is the point.
INVERSE_DESCRIPTION = "The ego vehicle drives forward along the road."


# Cosmos3-Nano is 16B in BF16: ~30 GiB of weights resident, plus the VAE decode of a
# multi-second latent, which is the real peak. The failed rollout died asking for a
# single 14.13 GiB block, so the budget has to cover weights AND that spike.
NANO_NEEDS_GIB = 46.0

# Never hand the allocator more than this share of the pool however free the box
# looks. The GB10 pool is shared with the OS, so an oversized load wedges the whole
# box rather than raising, and there is no recovering from that without a reboot.
MAX_FRACTION = 0.90


def _host_available_gib() -> float | None:
    """MemAvailable in GiB, or None if /proc/meminfo is unreadable.

    This, not `cuMemGetInfo`, is the honest number on the GB10. There is one 119.7 GiB
    pool, and `cuMemGetInfo` reports only what is *unused* -- it counts the kernel's
    page cache as taken, even though page cache is clean and gets evicted the instant
    anything asks for the memory. With 64 GiB of cache from a 35 GB model download
    sitting there, cuMemGetInfo says 32 GiB free while MemAvailable says 99 GiB, and
    the second one is right.

    MemAvailable is also the number that correctly *excludes* the thing that genuinely
    does steal from us: a leaky rustfs holding 12 GiB of anonymous heap is not
    reclaimable, and MemAvailable does not count it. So this stays a real cap.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024 / 2**30
    except (OSError, ValueError, IndexError):
        pass
    return None


def _cgroup_limit_gib() -> float | None:
    """This pod's own memory cgroup ceiling, in GiB, or None if unlimited/absent.

    /proc/meminfo is NOT namespaced, so inside a Flyte pod it reports the whole host.
    That is what we want for the GPU pool, but the pod also has its own `memory=96Gi`
    limit, and being killed by the cgroup OOM-killer is just as fatal. Take the lower
    of the two.
    """
    try:
        with open("/sys/fs/cgroup/memory.max") as fh:
            raw = fh.read().strip()
        return None if raw == "max" else int(raw) / 2**30
    except (OSError, ValueError):
        return None


def guard_memory(needs_gib: float = NANO_NEEDS_GIB, headroom_gib: float = 8.0) -> str:
    """Cap this process against what is genuinely reclaimable, or refuse to start.

    Returns a human-readable line for the report, because "how much of the box was
    free when this ran" is the single most useful number when a run that worked
    yesterday OOMs today.

    Raises RuntimeError when the budget will not cover the load. That is deliberate,
    and it is the lesson from the first rollout attempt: the old version clamped a
    negative budget up to a 0.05 floor and carried on, so a guard whose whole job was
    to prevent an OOM produced one -- "Tried to allocate 14.13 GiB ... 5.98 GiB
    allowed", where 5.98 GiB is exactly 0.05 of the pool. Failing in two seconds with
    an actionable message beats failing in eight minutes with a CUDA traceback.
    """
    import torch

    if not torch.cuda.is_available():
        return "no CUDA device"

    cuda_free, total = torch.cuda.mem_get_info()
    cuda_free_gib, total_gib = cuda_free / 2**30, total / 2**30

    # Take the most pessimistic real ceiling: host-reclaimable, the pod's own cgroup
    # limit, and the pool itself. Fall back to cuMemGetInfo only if /proc is missing.
    candidates = [c for c in (_host_available_gib(), _cgroup_limit_gib(), total_gib) if c]
    usable_gib = min(candidates) if candidates else cuda_free_gib
    budget_gib = usable_gib - headroom_gib

    if budget_gib < needs_gib:
        raise RuntimeError(
            f"only {usable_gib:.1f} GiB usable ({budget_gib:.1f} GiB after headroom), "
            f"and this load needs ~{needs_gib:.0f} GiB. Refusing to start rather than "
            f"OOM eight minutes in. Free the box first:\n"
            f"  kubectl rollout restart deploy/rustfs -n flyte   # the usual culprit\n"
            f"  sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'\n"
            f"or run ./preflight.sh, which does both and re-checks."
        )

    # set_per_process_memory_fraction is a share of TOTAL, so convert.
    fraction = min(MAX_FRACTION, budget_gib / total_gib)
    torch.cuda.set_per_process_memory_fraction(fraction)
    line = (
        f"{usable_gib:.1f} GiB usable of {total_gib:.1f} GiB "
        f"(cuMemGetInfo says {cuda_free_gib:.1f} free; the difference is reclaimable "
        f"page cache); capped this process at {fraction:.2f} "
        f"({fraction * total_gib:.1f} GiB)"
    )
    log.info("memory guard: %s", line)
    return line


def snapshot(repo: str) -> str:
    """Download the checkpoint (resumable) and return the local path."""
    from huggingface_hub import snapshot_download

    t0 = time.monotonic()
    # max_workers=4 rather than the default 8. Fetches on this box are more reliable
    # serialized than parallel; a stalled worker is what produces a "download" that
    # sits at the same byte count until the task times out.
    path = snapshot_download(repo_id=repo, max_workers=4)
    log.info("snapshot %s -> %s (%.1f min)", repo, path, (time.monotonic() - t0) / 60)
    return path


def load(repo: str, *, sound: bool = False):
    """Build a Cosmos3OmniPipeline on the GPU, ready to call.

    `enable_safety_checker=False` is deliberate and is explained at length in
    config.py: the default True constructs a CosmosSafetyChecker, which is a separate
    `cosmos_guardrail` install that pulls a GATED Llama Guard checkpoint, so leaving
    it on fails at construction on any box whose token has not accepted that licence.
    These runs therefore have no content guardrail.
    """
    import torch
    from diffusers import Cosmos3OmniPipeline

    path = snapshot(repo)
    guard_memory()

    t0 = time.monotonic()
    pipe = Cosmos3OmniPipeline.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,   # the ONLY precision NVIDIA tests for Cosmos 3
        device_map="cuda",            # streams shards to device; see the module docstring
        enable_safety_checker=False,
    )
    log.info("loaded %s in %.1f min", repo, (time.monotonic() - t0) / 60)

    if not sound and getattr(pipe, "sound_tokenizer", None) is not None:
        # The audio branch is ~2 GB of weights that never get used unless
        # enable_sound=True, and on this box the activation budget is the binding
        # constraint. `sound_tokenizer` is in the pipeline's `_optional_components`,
        # so unregistering it is supported rather than a hack; every task reloads the
        # pipeline from scratch anyway, so nothing downstream sees a half-built one.
        pipe.sound_tokenizer = None
        torch.cuda.empty_cache()
        log.info("dropped the sound tokenizer (enable_sound is off), ~2 GB reclaimed")
    return pipe


def generate(
    pipe,
    prompt: str,
    *,
    image=None,
    negative_prompt: str | None = None,
    num_frames: int = 61,
    height: int = 480,
    width: int = 832,
    fps: int = 24,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
    sound: bool = False,
):
    """Text-to-video, or image-to-video when `image` is given.

    The pipeline picks the mode from the inputs: `num_frames == 1` is text-to-image,
    an `image` anchors frame 0 and makes it image-to-video, otherwise it is
    text-to-video. There is no mode flag to set and no separate pipeline class.

    Defaults are deliberately smaller than NVIDIA's (189 frames at 720x1280). A clip
    is going to be base64'd into an HTML report, and 480p for ~2.5s is the largest
    thing that stays comfortably embeddable. Pass the bigger numbers when you want
    the quality shot rather than the report.
    """
    import torch

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image,
        num_frames=num_frames,
        height=height,
        width=width,
        fps=float(fps),
        num_inference_steps=steps,
        guidance_scale=guidance,
        enable_sound=sound,
        generator=generator,
        # The checker was never constructed (enable_safety_checker=False at load), so
        # this is belt and braces: it keeps the call working if someone turns the
        # constructor flag back on to get the guardrail without editing every task.
        enable_safety_check=False,
    )
    secs = time.monotonic() - t0
    log.info("generated %s frames in %.1fs (%.1fs/step)", num_frames, secs, secs / steps)
    return result, secs


def load_action_example(model_path: str) -> dict:
    """Read the action-conditioning example that ships inside the checkpoint.

    Returns the parsed JSON plus the conditioning frame. Everything the rollout needs
    is self-describing in that file: prompt, embodiment domain, viewpoint, fps,
    resolution tier, chunk size, and the chunks themselves as [num_chunks, 16, 29].
    """
    import torch
    from PIL import Image

    meta = json.loads(open(os.path.join(model_path, ACTION_CHUNKS)).read())
    meta["chunks"] = torch.tensor(meta["action_chunks"], dtype=torch.float32)
    meta["first_frame"] = Image.open(os.path.join(model_path, ACTION_FRAME)).convert("RGB")
    log.info(
        "action example: %s, domain=%s, chunks=%s",
        meta.get("prompt"), meta.get("domain_name"), tuple(meta["chunks"].shape),
    )
    return meta


def rollout(
    pipe,
    meta: dict,
    *,
    num_chunks: int = 2,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Action-conditioned forward dynamics: first frame + actions -> future video.

    This is the world-model claim in its most literal form. The prompt names the task
    ("Pickup items in the supermarket") but says NOTHING about what happens next; the
    future comes entirely from the action sequence and the model's learned dynamics.

    Chunks are rolled AUTOREGRESSIVELY: chunk 0 is conditioned on the real observed
    frame, and every chunk after it is conditioned on the last frame the model itself
    predicted. That is the honest version of a rollout, and it is also where error
    compounds, which is worth watching in the output. Each chunk of 16 actions yields
    17 frames (`chunk_size + 1`).

    Note what is NOT passed: height, width and num_frames must all be None for an
    action run. Resolution comes from `action.resolution_tier` and the frame count
    from `action.chunk_size`; passing them raises.
    """
    import torch
    from diffusers import CosmosActionCondition

    chunks = meta["chunks"]
    frame = meta["first_frame"]
    frames: list = []
    per_chunk: list[float] = []

    for i in range(min(num_chunks, chunks.shape[0])):
        generator = torch.Generator().manual_seed(seed + i) if seed is not None else None
        t0 = time.monotonic()
        result = pipe(
            prompt=meta["prompt"],
            action=CosmosActionCondition(
                mode="forward_dynamics",
                chunk_size=int(meta["action_chunk_size"]),
                domain_name=meta["domain_name"],
                resolution_tier=int(meta["image_size"]),
                raw_actions=chunks[i],
                image=frame,
                view_point=meta.get("view_point", "ego_view"),
            ),
            fps=float(meta.get("fps", 10)),
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generator,
            # NVIDIA's runner passes use_system_prompt=False for action modes: the
            # action model was trained on structured captions without the generic
            # "you are a helpful assistant who will generate videos" preamble.
            use_system_prompt=False,
            enable_safety_check=False,
        )
        secs = time.monotonic() - t0
        per_chunk.append(secs)
        chunk_frames = list(result.video)
        # Drop the conditioning frame on every chunk after the first, or the seam
        # shows up as a duplicated frame in the middle of the rollout.
        frames.extend(chunk_frames if i == 0 else chunk_frames[1:])
        # Autoregress: the next chunk starts from what this one predicted.
        frame = chunk_frames[-1]
        log.info("chunk %s: %s frames in %.1fs", i, len(chunk_frames), secs)

    return frames, per_chunk


def load_inverse_example(model_path: str, index: int = 0) -> dict:
    """Read one inverse-dynamics example, clip and answer key both.

    Decodes through `media.decode` rather than `diffusers.utils.load_video`, which
    the pipeline docstring recommends but which needs imageio-ffmpeg: the one codec
    this image leaves out on purpose. The frame count is asserted against the answer
    key below, so a decoder that silently drops a frame fails here and not eight
    minutes into a GPU task.
    """
    import torch

    import media

    clip = os.path.join(model_path, INVERSE_CLIP.format(i=index))
    frames = media.decode(clip)
    raw = json.loads(open(os.path.join(model_path, INVERSE_TRUTH.format(i=index))).read())
    truth = torch.tensor(raw["data"], dtype=torch.float32).reshape(*raw["shape"])

    # The conditioning video spans chunk_size + 1 frames, so 61 frames describe 60
    # transitions. Asserting it here turns a shape mismatch into one readable line
    # instead of a tensor error thrown eight minutes into a GPU task.
    assert truth.shape[0] == len(frames) - 1, (
        f"{len(frames)} frames should describe {len(frames) - 1} actions, "
        f"but the answer key has {truth.shape[0]}"
    )
    log.info(
        "inverse example %s: %s frames, truth %s",
        index, len(frames), tuple(truth.shape),
    )
    return {
        "index": index,
        "frames": frames,
        "truth": truth,
        "domain_name": INVERSE_DOMAIN,
        "resolution_tier": INVERSE_TIER,
        "fps": INVERSE_FPS,
        "description": INVERSE_DESCRIPTION,
    }


def invert(
    pipe,
    meta: dict,
    *,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Inverse dynamics: hand it the video, get back the actions that produced it.

    The mirror image of `rollout` above, and the thing a forward-only world model
    cannot do at all. DreamerV3's RSSM maps (state, action) to the next state and has
    no path in the other direction; Cosmos denoises the action channel the same way it
    denoises pixels, so running it backwards is a mode flag rather than a new model.

    Every vision latent frame is conditioning here, so the returned video is
    essentially the input handed back. `result.action` is the output that matters, and
    the pipeline has already sliced it to the embodiment's true width (9 for `av`)
    from the padded channel count the transformer works in.
    """
    import torch
    from diffusers import CosmosActionCondition

    frames = meta["frames"]
    chunk_size = len(frames) - 1

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    result = pipe(
        prompt=meta["description"],
        action=CosmosActionCondition(
            mode="inverse_dynamics",
            chunk_size=chunk_size,
            domain_name=meta["domain_name"],
            resolution_tier=int(meta["resolution_tier"]),
            video=frames,
            view_point="ego_view",
        ),
        fps=float(meta["fps"]),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        # Same reason as the forward rollout: the action model was trained on
        # structured captions with no assistant preamble.
        use_system_prompt=False,
        enable_safety_check=False,
    )
    secs = time.monotonic() - t0

    # The output dataclass types `action` as a list of tensors, and the action modes
    # populate it with a single [T, D] entry. Accept both so a future shape change
    # fails loudly at the assert below rather than silently indexing the wrong axis.
    pred = result.action
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = pred.detach().float().cpu()
    log.info("recovered actions %s in %.1fs", tuple(pred.shape), secs)
    return pred, secs


def action_error(truth, pred, moving_frac: float = 0.1) -> dict:
    """Score recovered actions against the answer key.

    Reported per channel as well as overall, because the mean alone hides the thing
    worth seeing: the `av` action is a mixed vector (translation next to what looks
    like a rotation basis sitting near 1.0), so channels are not commensurable and a
    single MAE silently weights them by their native scale.

    `moving_frac` is the reason there are two headline numbers rather than one. In a
    given clip most channels barely leave their start value: measured on the bundled
    examples, one channel spans ~0.41 and the other eight span ~0.02. A range-
    normalised error on those eight is noise divided by nothing, and it can exceed
    1.0 (example 1 scores 1.20 on channel 7) without the model having done anything
    wrong. So `mae_moving` covers only channels whose range is at least this fraction
    of the widest one, and that is the number worth quoting.
    """
    import torch

    n = min(truth.shape[0], pred.shape[0])
    truth, pred = truth[:n], pred[:n]
    assert truth.shape == pred.shape, (truth.shape, pred.shape)

    err = (pred - truth).abs()
    per_dim = err.mean(dim=0)
    # Range-normalised, so a channel that barely moves cannot look good by standing
    # still. Guarded because a constant channel has zero range.
    spread = (truth.max(dim=0).values - truth.min(dim=0).values).clamp(min=1e-6)
    moving = [d for d in range(truth.shape[-1])
              if float(spread[d]) >= moving_frac * float(spread.max())]
    return {
        "steps": int(n),
        "mae": float(err.mean()),
        "mae_moving": float(err[:, moving].mean()),
        "moving_dims": moving,
        "mae_per_dim": [float(x) for x in per_dim],
        "nmae_per_dim": [float(x) for x in (per_dim / spread)],
        "max_abs_err": float(err.max()),
        "truth_range": [[float(a), float(b)] for a, b in
                        zip(truth.min(dim=0).values, truth.max(dim=0).values)],
    }


# ── Everything below is a second surface of the same checkpoint ─────────────────
#
# The four functions above (generate / rollout / invert) cover text-to-video and the
# two action modes that ship with an answer key. What follows is the rest of what the
# checkpoint can actually do, and each one exists because it answers a question the
# first three cannot:
#
#   policy()          forward dynamics tells you what happens IF you do X. Policy mode
#                     asks the model to choose X itself, so it predicts the actions
#                     and the video together.
#   extend()          rollout is capped at the actions that ship in the asset. Video-
#                     to-video conditioning re-enters the model's own output as
#                     conditioning, which is how you go past the end of the data.
#   decode_latents()  the pipeline decodes once, at the end. Decoding mid-denoise is
#                     what turns "here is a clip" into "here is the clip resolving".
#   counterfactuals() the control experiment. A video model that ignores its actions
#                     and a world model that obeys them look identical until you
#                     change only the actions and diff the pixels.


def decode_latents(pipe, latents):
    """Decode vision latents to PIL frames, the same way the pipeline's last step does.

    Reproduced here rather than reached for because the pipeline inlines it: the
    de-normalisation (`latents / inv_std + mean`) happens once, at the bottom of
    `__call__`, and there is no public method that turns a latent into frames. Doing
    it by hand is what makes a mid-denoise snapshot possible.

    The `.view(1, -1, 1, 1, 1)` broadcast is load-bearing and looks wrong at a glance:
    `latents` is [C, T, H, W] with no batch axis, and broadcasting a 5-D stat against
    a 4-D tensor left-pads it to [1, C, T, H, W], which is exactly the shape the VAE
    wants. Adding a batch dimension by hand here produces [1, 1, C, T, H, W] and a
    shape error several frames deep inside the decoder.
    """
    import torch

    # Normalise away any leading batch axes before the broadcast below.
    #
    # This is load-bearing and the failure it prevents is not obvious. The pipeline's
    # own vision latents already carry a batch axis, and it hands the scheduler
    # `latents.unsqueeze(0)`, so anything captured from inside the solver (the x0
    # prediction in `generate_trajectory`) comes back with one axis MORE than the
    # latents handed to the callback. Both then broadcast against a 5-D statistic
    # without complaint, and the only symptom is the Wan VAE unpacking the result:
    #
    #     _, _, num_frame, height, width = z.shape
    #     ValueError: too many values to unpack (expected 5)
    #
    # Squeezing here rather than at each call site means every caller can hand over
    # whatever rank it happens to hold.
    while latents.ndim > 4 and latents.shape[0] == 1:
        latents = latents.squeeze(0)

    with torch.no_grad():
        dtype = pipe.vae.dtype
        mean = pipe._vae_latents_mean.to(device=latents.device, dtype=dtype)
        inv_std = pipe._vae_latents_inv_std.to(device=latents.device, dtype=dtype)
        z = latents.to(dtype) / inv_std.view(1, -1, 1, 1, 1) + mean.view(1, -1, 1, 1, 1)
        decoded = pipe.vae.decode(z).sample
        return pipe.video_processor.postprocess_video(decoded, output_type="pil")[0]


def generate_trajectory(
    pipe,
    prompt: str,
    *,
    snapshots: int = 8,
    show: str = "x0",
    negative_prompt: str | None = None,
    num_frames: int = 45,
    height: int = 480,
    width: int = 832,
    fps: int = 24,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Generate, and keep the partly-denoised clip at `snapshots` points along the way.

    Same call as `generate`, plus `callback_on_step_end`, which the pipeline invokes
    after each scheduler step. Decoding partway is the only way to see the thing
    everyone describes and nobody shows.

    ── show="x0" versus show="latent", which is the whole methodology ──────────────
    `callback_on_step_end` hands back `latents`, which is x_t: the still-NOISY sample
    at the current step. Decoding that is a real picture of the trajectory and it is
    NOT the picture people mean by "watch the image emerge". Measured on this
    checkpoint, decoding x_t gives eight snapshots that are visually noise with a flat
    sharpness of ~130 for thirty of the thirty five steps, jumping to 259 only at the
    last one, because under a flow-matching schedule x_t stays dominated by noise
    until the very end. Presenting that as "the world condensing" would be a story the
    chart underneath it contradicts.

    What people mean is x0: the model's current PREDICTION of the finished video,
    which exists at every step and is what the solver actually steers with. This
    scheduler is configured `predict_x0: true`, so `convert_model_output` returns
    exactly that, and it is a pure function of the velocity and the sample. Rather
    than recompute it (which would need `step_index` to be the value it had mid-step),
    this wraps that method and keeps what the solver itself computed, which cannot
    drift out of sync with the run by construction.

    The wrapper is installed on `pipe.scheduler` and removed in a `finally`. That is
    safe here because this is a text-to-video path: the pipeline deep-copies the
    scheduler for the sound and action streams, and a patched bound method is not
    something to hand to `copy.deepcopy`.

    The decode is NOT free. It is the same VAE pass that produces the final clip and
    it is the largest single allocation in the whole run (measured at 10 to 14s for 45
    frames at 480p), so this samples a handful of steps rather than all of them.

    Returns (stages, final_frames, seconds) where stages is [(step_number, frames)].
    """
    import torch

    total = max(int(steps), 1)
    # Evenly spaced over the schedule, always including the first and last step. The
    # last one matters: it is the same latent the pipeline itself decodes, so it is
    # the control that proves the snapshots are on the same trajectory as the output.
    picks = sorted({round(i * (total - 1) / max(snapshots - 1, 1)) for i in range(snapshots)})
    stages: list[tuple[int, list]] = []

    # Filled by the wrapper below with the x0 prediction the solver just computed.
    latest: dict = {}
    scheduler = pipe.scheduler
    original_convert = scheduler.convert_model_output

    def capture_x0(model_output, *args, sample=None, **kwargs):
        out = original_convert(model_output, *args, sample=sample, **kwargs)
        latest["x0"] = out
        return out

    def on_step_end(p, i, t, kwargs):
        if i in picks:
            snapshot = latest.get("x0") if show == "x0" else kwargs["latents"]
            if snapshot is not None:
                t0 = time.monotonic()
                stages.append((i + 1, decode_latents(p, snapshot)))
                log.info(
                    "snapshot at step %s/%s from %s (decode %.1fs)",
                    i + 1, total, show, time.monotonic() - t0,
                )
        # The pipeline does `callback_outputs.pop("latents", latents)`, so returning
        # the kwargs unchanged leaves the denoising trajectory untouched. Returning
        # None raises; returning a modified "latents" would edit the generation.
        return kwargs

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    if show == "x0":
        scheduler.convert_model_output = capture_x0
    try:
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            height=height,
            width=width,
            fps=float(fps),
            num_inference_steps=total,
            guidance_scale=guidance,
            generator=generator,
            callback_on_step_end=on_step_end,
            callback_on_step_end_tensor_inputs=["latents"],
            enable_safety_check=False,
        )
    finally:
        scheduler.convert_model_output = original_convert
    secs = time.monotonic() - t0
    log.info("trajectory: %s snapshots over %s steps in %.1fs", len(stages), total, secs)
    return stages, list(result.video), secs


def policy(
    pipe,
    meta: dict,
    *,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Policy mode: one frame and a task description in, video AND actions out.

    The third action mode, and the one that is neither of the other two. Forward
    dynamics is given actions and predicts pixels; inverse dynamics is given pixels
    and predicts actions. Policy mode is given NEITHER: it sees the first frame and
    the task ("Pickup items in the supermarket") and denoises the action channel and
    the vision channel jointly, so it is simultaneously deciding what the robot should
    do and rendering the consequence of having done it.

    That makes it the closest thing in this checkpoint to a robot policy, and the
    reason the same weights can be called a world model and a policy in one breath:
    the action tokens and the pixel tokens are the same sequence to the transformer,
    so which one is "input" is decided entirely by which one you leave noisy.

    Returns (frames, predicted_actions, seconds). No `raw_actions` is passed, which is
    the whole point; the ground-truth chunk in the asset is only used afterwards to
    score what came back.
    """
    import torch
    from diffusers import CosmosActionCondition

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    result = pipe(
        prompt=meta["prompt"],
        action=CosmosActionCondition(
            mode="policy",
            chunk_size=int(meta["action_chunk_size"]),
            domain_name=meta["domain_name"],
            resolution_tier=int(meta["image_size"]),
            image=meta["first_frame"],
            view_point=meta.get("view_point", "ego_view"),
        ),
        fps=float(meta.get("fps", 10)),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        use_system_prompt=False,
        enable_safety_check=False,
    )
    secs = time.monotonic() - t0

    pred = result.action
    if isinstance(pred, (list, tuple)):
        pred = pred[0]
    pred = pred.detach().float().cpu()
    log.info("policy: %s frames, actions %s in %.1fs", len(result.video), tuple(pred.shape), secs)
    return list(result.video), pred, secs


# Video-to-video conditioning keeps the first `max(condition_frame_indexes) *
# scale_factor_temporal + 1` pixel frames clean and denoises the rest. With the
# default indexes (0, 1) and the Wan VAE's 4x temporal compression that is 5 frames,
# so a chained clip reproduces the 5 frames it was handed and the stitch has to drop
# them or the seam shows up as a five-frame stutter.
V2V_CONDITION_INDEXES = (0, 1)
V2V_OVERLAP = max(V2V_CONDITION_INDEXES) * 4 + 1


def extend(
    pipe,
    frames: list,
    prompt: str,
    *,
    negative_prompt: str | None = None,
    num_frames: int = 45,
    height: int = 480,
    width: int = 832,
    fps: int = 24,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """Continue a clip: condition on its last frames and denoise the rest.

    This is the fourth generation mode, selected by passing `video=` with no `action=`.
    It is what lifts the model off a fixed clip length, because the video it continues
    can be the video it just produced, and that is the only way to get a horizon longer
    than one forward pass out of a model whose context is one clip.

    It is also the honest way to show a world model failing. Every chained segment
    conditions on the model's OWN output rather than on anything real, so whatever
    error is in segment N is treated as ground truth by segment N+1. That compounding
    is the central unsolved problem with generative world models, and chaining half a
    dozen segments is enough to watch it happen.

    `condition_video_keep="last"` is the load-bearing argument. The default is "first",
    which conditions on the START of whatever you hand it and would regenerate the
    clip you already have instead of continuing it.

    Returns (frames, seconds). The returned frames INCLUDE the V2V_OVERLAP
    reproduced conditioning frames; use `stitch` to join segments without them.
    """
    import torch

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        video=frames,
        condition_frame_indexes_vision=V2V_CONDITION_INDEXES,
        condition_video_keep="last",
        num_frames=num_frames,
        height=height,
        width=width,
        fps=float(fps),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        enable_safety_check=False,
    )
    secs = time.monotonic() - t0
    log.info("extended by %s frames in %.1fs", num_frames, secs)
    return list(result.video), secs


def stitch(segments: list[list], overlap: int = V2V_OVERLAP) -> list:
    """Join chained segments into one clip, dropping each segment's reproduced tail."""
    if not segments:
        return []
    out = list(segments[0])
    for seg in segments[1:]:
        out.extend(seg[overlap:])
    return out


# ── Counterfactual actions ──────────────────────────────────────────────────────
#
# Every variant is anchored at the RECORDED first action rather than invented from
# nothing, and that is a deliberate constraint rather than timidity. The agibotworld
# action is a 29-dimensional vector whose channels are joint targets on a specific
# humanoid; a vector of zeros is not "do nothing", it is "drive every joint to zero",
# which is both physically violent and far outside anything the model was trained on.
# A model that produces garbage from a garbage action has told you nothing. Anchoring
# at the true starting pose keeps all four sequences on the data manifold, so a
# difference in the output is a difference the model actually believes in.

def counterfactuals(chunk) -> list[tuple[str, object, str]]:
    """Build the action variants for the counterfactual test.

    Returns [(name, actions, description)] with the recorded chunk first, so the rest
    can be diffed against it.
    """
    first = chunk[:1]
    return [
        (
            "recorded",
            chunk.clone(),
            "The actions the real robot executed. Everything else is measured against this.",
        ),
        (
            "held",
            first.repeat(chunk.shape[0], 1),
            "The first action, repeated. The commanded pose never changes, so a model "
            "that is genuinely reading the action channel should predict a robot that "
            "stops moving.",
        ),
        (
            "reversed",
            chunk.flip(0).clone(),
            "The same motion executed backwards. The start and end poses swap, so the "
            "arm should travel the same path in the opposite direction.",
        ),
        (
            "amplified",
            (first + (chunk - first) * 2.0),
            "The recorded motion, doubled about its starting pose. Same direction, "
            "twice the displacement: the test for whether action MAGNITUDE registers "
            "or only action direction.",
        ),
    ]


def frame_divergence(a: list, b: list) -> float:
    """Mean absolute pixel difference between two clips, 0-255, on the shared prefix.

    The measurement the counterfactual test turns on. A video model that has learned
    to ignore its action channel and just continue the scene produces near-identical
    clips for every variant, and this number is how you tell that apart from a model
    that is genuinely conditioning on them. Comparing against the recorded rollout
    rather than against a fixed reference keeps it symmetric across variants.
    """
    import numpy as np

    n = min(len(a), len(b))
    if n == 0:
        return 0.0
    diffs = [
        float(np.abs(np.asarray(x, dtype="float32") - np.asarray(y, dtype="float32")).mean())
        for x, y in zip(a[:n], b[:n])
    ]
    return float(np.mean(diffs))


def rank_agreement(judged: dict, measured: dict) -> dict:
    """Do two rankings of the same clips agree, pair by pair?

    The right question to ask of a coarse judge. "Did it get `held` right" turns on one
    clip and one wording; this asks whether the ORDER it puts the clips in is consistent
    with the order a pixel measurement puts them in, which is the claim that actually
    matters and the only one a three-way bucket can support.

    Ties in the judged ranking are counted as neither concordant nor discordant rather
    than as failures: a bucket that lumps two clips together has declined to order them,
    and scoring that as a mistake would punish the judge for being appropriately coarse.
    Discordant pairs are the ones that count against it, and one is enough to matter.
    """
    names = [n for n in judged if judged[n] is not None and n in measured]
    concordant = discordant = tied = 0
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            dj, dm = judged[a] - judged[b], measured[a] - measured[b]
            if dj == 0:
                tied += 1
            elif dj * dm > 0:
                concordant += 1
            else:
                discordant += 1
    return {
        "pairs": concordant + discordant + tied,
        "concordant": concordant,
        "discordant": discordant,
        "tied": tied,
        "inverted": discordant > 0,
    }


def clip_stats(frames: list) -> dict:
    """Per-clip health numbers, for watching a long autoregressive rollout decay.

    Three numbers, each catching a different way a chained rollout dies:

      sharpness  variance of the Laplacian, the standard blur detector. Autoregressive
                 video collapse shows up here first: each generation smooths its own
                 input slightly, and re-feeding it compounds into mush long before a
                 human calls the clip broken.
      motion     mean absolute difference between consecutive frames. Falls to ~0 when
                 the rollout freezes, which is the other failure mode.
      luminance  mean brightness. Drifts monotonically when the model loses its
                 exposure anchor, a very common long-horizon artifact.
    """
    import numpy as np

    if not frames:
        return {"sharpness": 0.0, "motion": 0.0, "luminance": 0.0}

    grays = [np.asarray(f.convert("L"), dtype="float32") for f in frames]
    # A 4-neighbour Laplacian by slicing rather than scipy, which is not in the image.
    sharp = []
    for g in grays:
        lap = (
            g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:] - 4.0 * g[1:-1, 1:-1]
        )
        sharp.append(float(lap.var()))
    motion = (
        float(np.mean([np.abs(b - a).mean() for a, b in zip(grays, grays[1:])]))
        if len(grays) > 1
        else 0.0
    )
    return {
        "sharpness": float(np.mean(sharp)),
        "motion": motion,
        "luminance": float(np.mean([g.mean() for g in grays])),
    }


def rollout_chunk(
    pipe,
    meta: dict,
    actions,
    *,
    frame=None,
    steps: int = 35,
    guidance: float = 6.0,
    seed: int | None = 0,
):
    """One chunk of forward dynamics driven by a caller-supplied action tensor.

    `rollout` above walks the chunks that ship in the asset. This takes the actions as
    an argument instead, which is what makes a counterfactual possible: the
    conditioning frame, the prompt, the seed and the schedule are all held fixed and
    the action tensor is the only thing that changes, so any difference in the output
    is attributable to the actions and to nothing else.

    Returns (frames, seconds).
    """
    import torch
    from diffusers import CosmosActionCondition

    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    t0 = time.monotonic()
    result = pipe(
        prompt=meta["prompt"],
        action=CosmosActionCondition(
            mode="forward_dynamics",
            chunk_size=int(meta["action_chunk_size"]),
            domain_name=meta["domain_name"],
            resolution_tier=int(meta["image_size"]),
            raw_actions=actions,
            image=meta["first_frame"] if frame is None else frame,
            view_point=meta.get("view_point", "ego_view"),
        ),
        fps=float(meta.get("fps", 10)),
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=generator,
        use_system_prompt=False,
        enable_safety_check=False,
    )
    secs = time.monotonic() - t0
    return list(result.video), secs


# ── The other surface: the understanding expert of the SAME checkpoint ──────────
#
# Cosmos 3 is an omnimodal model, not a video generator with a chatbot bolted on. The
# marketing splits it into a Reasoner and a Generator; on disk they are one set of
# weights with two entry points, and the thing that makes that concrete is the file
# layout of the checkpoint itself:
#
#   model.safetensors.index.json        <- transformers reads this
#     -> transformer/diffusion_pytorch_model-0000{1..7}-of-00007.safetensors
#     -> vision_encoder/model.safetensors
#   model_index.json                    <- diffusers reads this
#     -> the same transformer/ and vision_encoder/ files
#
# Both indexes point at the same shards. So once the shared hostPath cache has the
# 33 GB staged for the generation tasks, the reasoning surface costs ZERO extra
# download, which is why this is worth having at all on a box with one disk.
#
# What differs is what gets materialised. `Cosmos3OmniPipeline` builds the diffusion
# expert plus the VAE and lands at ~30 GB resident; `Cosmos3OmniForConditionalGeneration`
# builds the language expert plus the Qwen3-VL vision tower and lands at 17.5 GB
# (measured, BF16, on the Spark). They do not fit at the same time alongside a VAE
# decode, so any task wanting both runs them in sequence with `release()` between.
#
# transformers >= 5.11 is already a hard requirement in config.py for the Qwen3-VL
# vision tower, and `Cosmos3OmniForConditionalGeneration` ships in the same release.
# No new dependency, no trust_remote_code.

# 17.5 GiB resident, measured. The rest is headroom for the vision tower's activations,
# which scale with (frames x pixels) and are the part that actually varies here.
REASONER_NEEDS_GIB = 26.0

# How many frames of a clip to show the reasoner. The vision tower charges per frame
# and the answers stop changing well before the frame budget does: 8 evenly spaced
# frames across a 45-frame segment is enough to describe the action and cheap enough
# (1.5 to 4.1s per question) to ask several questions per clip.
REASON_FRAMES = 8


def load_reasoner(repo: str):
    """Load the understanding expert. Returns (model, processor).

    Deliberately mirrors `load` above, including `device_map="cuda"` rather than a
    `.to("cuda")`: the double-copy trap in the module docstring applies here exactly
    as it does to the diffusion expert, and 17.5 GB copied through the host is still
    35 GB of one shared pool.

    The processor resolves to `Qwen3VLProcessor` off the checkpoint's own
    preprocessor_config.json, so the chat template, the image token id and the video
    sampler all come from the repo rather than from anything hardcoded here.
    """
    import torch
    from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

    path = snapshot(repo)
    guard_memory(REASONER_NEEDS_GIB)

    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(path)
    model = Cosmos3OmniForConditionalGeneration.from_pretrained(
        path,
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    log.info(
        "loaded the reasoner from %s in %.1f min (%.1f GB resident)",
        repo, (time.monotonic() - t0) / 60, torch.cuda.memory_allocated() / 2**30,
    )
    return model, processor


def release() -> float:
    """Give the pool back after dropping a model. Returns GiB still allocated.

    Call it with the caller's own reference already set to None -- `pipe = None;
    world.release()` -- because a helper that took the model as an argument would
    itself hold the last reference while trying to free it.

    NOT because the two experts cannot co-exist. They can, and this file claimed
    otherwise for a while on the strength of an estimate rather than a measurement.
    Measured on the Spark: generation expert alone 29.6 GiB, both resident 46.0 GiB,
    peak 50.5 GiB with a VAE decode running while the language expert sits there, and
    both still work. Against a 96 GiB pod limit that is comfortable.

    What releasing buys is headroom, not feasibility. A task that only needs one expert
    at a time runs at 30 GiB instead of 50, which matters on a box where a leaky object
    store can quietly take 40 GiB (see reference_gb10_cap_against_free) and where an
    oversized load hangs the machine rather than raising. So sequential stays the
    default for tasks that generate THEN judge; keeping both is for tasks that alternate
    between them, where paying 4 minutes of reloading per round is the real cost.

    Two steps, and neither is optional: `gc.collect()` because diffusers pipelines hold
    reference cycles and CPython will not free those on refcount alone, then
    `empty_cache()` because until it runs the caching allocator holds every freed block
    as reserved-but-unused and the next model sees a pool that is still full.
    """
    import gc

    import torch

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    still, reserved = torch.cuda.memory_allocated() / 2**30, torch.cuda.memory_reserved() / 2**30
    log.info("released; %.1f GiB still allocated, %.1f GiB reserved", still, reserved)
    return still


def sample_frames(frames: list, count: int = REASON_FRAMES) -> list:
    """Evenly spaced frames spanning the whole clip, endpoints included."""
    if not frames:
        return []
    if len(frames) <= count:
        return list(frames)
    idx = [round(i * (len(frames) - 1) / (count - 1)) for i in range(count)]
    return [frames[i] for i in idx]


def ask(
    model,
    processor,
    question: str,
    *,
    image=None,
    video: list | None = None,
    max_new_tokens: int = 512,
    frames: int = REASON_FRAMES,
) -> tuple[str, float]:
    """Put a question to the reasoner about an image, a clip, or nothing. (text, secs).

    Greedy decoding (`do_sample=False`) on purpose. These answers are being used as
    measurements -- a plausibility score per segment, a moving/stationary verdict per
    counterfactual -- and a sampled answer would put temperature noise into a number
    that gets plotted next to a deterministic pixel statistic.

    Note the video path takes an already-sampled list of PIL frames rather than a
    path. transformers will otherwise warn that it is guessing 24 fps because no
    video metadata came with the frames; sampling here means the frame budget is a
    decision this file makes rather than one the processor makes for us.
    """
    import torch

    content: list[dict] = []
    if image is not None:
        content.append({"type": "image", "image": image})
    if video:
        content.append({"type": "video", "video": sample_frames(video, frames)})
    content.append({"type": "text", "text": question})

    inputs = processor.apply_chat_template(
        [{"role": "user", "content": content}],
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    t0 = time.monotonic()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    # Slice off the prompt: generate() returns prompt + completion, and for a video
    # question the prompt is thousands of vision tokens.
    text = processor.batch_decode(
        out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
    )[0].strip()
    secs = time.monotonic() - t0
    log.info("asked (%s): %.1fs -> %s", question[:40], secs, text[:90])
    return text, secs


def parse_score(text: str, lo: float = 1.0, hi: float = 10.0) -> float | None:
    """Pull the leading 1-10 score out of an answer, or None if it did not give one.

    None rather than a default, and the caller plots the gaps. A task that quietly
    substituted 5.0 for "the model did not answer the question" would produce a chart
    that looks like a measurement and is partly fiction.
    """
    import re

    # Most specific first. "The video shows 3 robot arms, I rate it 8" is the failure
    # this ordering exists for: a bare first-number-in-range scan reads that as 3.
    patterns = (
        r"^\s*(\d+(?:\.\d+)?)",              # leading, which is what the prompt asks for
        r"(\d+(?:\.\d+)?)\s*(?:/|out of)\s*10",   # "8/10", "8 out of 10"
        r"(?:score|rating|rate[sd]?)\D{0,12}?(\d+(?:\.\d+)?)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            if lo <= value <= hi:
                return value
    for match in re.finditer(r"\d+(?:\.\d+)?", text):
        value = float(match.group())
        if lo <= value <= hi:
            return value
    return None


# Both directions of the moving/still question, so the verdict is read off whichever
# word the model reached for first rather than off a fixed position in the sentence.
_STILL_WORDS = ("still", "stationary", "motionless", "not moving", "does not move")
_MOVING_WORDS = ("moving", "moves", "in motion")


def parse_choice(text: str, options: tuple[str, ...]) -> int | None:
    """Index of whichever option the answer reaches for FIRST, or None if none appear.

    Earliest-occurring rather than first-in-the-list, so the reading does not depend on
    the order the options happen to be declared in, and a justification that later
    mentions a rejected option cannot overturn the answer that was given.
    """
    low = text.lower()
    hits = [(low.find(opt.lower()), i) for i, opt in enumerate(options) if opt.lower() in low]
    return min(hits)[1] if hits else None


def parse_still(text: str) -> bool | None:
    """True if the answer says the robot is holding still, False if moving, None if neither.

    Not `text.split()[0]`. The prompt asks for one word first and usually gets it, but
    "The robot is holding still" begins with "The", and a first-word check would score
    that as a moving verdict -- silently, and in the direction that would flatter the
    result this task exists to test.
    """
    low = text.lower()
    hits = [(low.find(w), True) for w in _STILL_WORDS if w in low]
    hits += [(low.find(w), False) for w in _MOVING_WORDS if w in low]
    return min(hits)[1] if hits else None


# Words that carry no scene information, so two descriptions of completely different
# scenes still share them. Kept short on purpose: the point of the overlap metric
# below is to be transparent, and a big curated stoplist is a thumb on the scale.
_STOPWORDS = frozenset(
    "a an the is are was were be been being of in on at to for with from by and or "
    "as it its this that these those there here into onto over under while during "
    "video shows showing appears seems captures depicts scene footage camera frame "
    "frames clip image images view".split()
)


def content_words(text: str) -> set:
    """Lowercased words of a description with the scaffolding removed."""
    import re

    return {w for w in re.findall(r"[a-z]+", text.lower()) if w not in _STOPWORDS and len(w) > 2}


def description_overlap(first: str, other: str) -> float:
    """Jaccard overlap of two descriptions' content words, 0 to 1.

    The semantic counterpart to `clip_stats`. Sharpness and inter-frame motion say
    whether a long rollout is still a well-formed video; they cannot say whether it is
    still a video OF THE SAME THING, and that is the drift that actually matters when
    the claim is "this could stand in for a simulator". Two descriptions from the same
    model of the same scene share most of their nouns; a supermarket that has quietly
    become a corridor does not.

    Deliberately a crude set overlap rather than an embedding distance. It needs no
    second model, and every point on the resulting chart can be checked by reading the
    two sentences printed next to it, which is not true of a cosine similarity.
    """
    a, b = content_words(first), content_words(other)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def load_reasoning_example(model_path: str) -> dict:
    """Read the planning example that ships inside the checkpoint: image + goal.

    Same shape as `load_action_example`: everything the task needs is self-describing
    in the repo, so there is no second network dependency and no prompt of ours
    standing in for one the model was actually trained to answer.
    """
    from PIL import Image

    meta = json.loads(open(os.path.join(model_path, REASON_PROMPT)).read())
    meta["image"] = Image.open(os.path.join(model_path, REASON_FRAME)).convert("RGB")
    log.info("reasoning example: %s (%s)", meta.get("prompt"), meta["image"].size)
    return meta


def split_subtasks(text: str, limit: int = 6) -> list[str]:
    """Break a generated plan into individual subtasks.

    The model answers this prompt as running prose -- "Move the arm to the flower.
    Grasp the flower. Move the arm to the red bottle. Place the flower in the red
    bottle." -- rather than as a JSON list, so this splits on sentences and on the
    numbered/bulleted forms it uses when the goal is longer. Whatever comes out is
    shown verbatim in the report next to the raw answer, so a bad split is visible
    rather than silently reshaping what the model said.
    """
    import re

    lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip()
             for ln in text.splitlines() if ln.strip()]
    # Numbered or bulleted plan: one subtask per line, and the prose splitter below
    # would wrongly cut "1. Move to the bottle. " into two.
    if len(lines) > 1:
        parts = [ln for ln in lines if len(ln) > 8]
    else:
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]
    return [p.rstrip(".") + "." for p in parts if len(p) > 8][:limit]


def forward_meta_from_inverse(example: dict) -> dict:
    """Turn an inverse-dynamics example into the meta a FORWARD rollout needs.

    The bundled inverse examples are the only assets in the checkpoint that pair a real
    video with the real actions that produced it, which makes them the only place a
    round trip can be scored against ground truth rather than against itself. But they
    are shaped for `invert` (frames in, actions out) and `rollout_chunk` wants the other
    shape, so this converts one into the other: same embodiment, same resolution tier,
    same fps, conditioning on the clip's own first frame.

    The action count becomes the chunk size, so a 60-action example generates 61 frames
    and lines up with the real clip frame for frame.
    """
    return {
        "prompt": example["description"],
        "action_chunk_size": int(example["truth"].shape[0]),
        "domain_name": example["domain_name"],
        "image_size": int(example["resolution_tier"]),
        "first_frame": example["frames"][0],
        "view_point": "ego_view",
        "fps": example["fps"],
    }


def sharpness(frames: list) -> float:
    """Variance of the Laplacian, averaged over frames. Same measure as `clip_stats`."""
    return clip_stats(frames)["sharpness"]


def blur_to_match(frames: list, target: float, max_radius: float = 8.0) -> tuple[list, float]:
    """Blur a clip until its sharpness matches `target`. Returns (frames, radius used).

    The control for the round-trip result. `cycle` shows that inverse dynamics reads a
    generated video about three times worse than the real one. The obvious reading is
    "the generated video is softer, so there is less to read", which is a hypothesis
    rather than a measurement, and the alternative has the opposite engineering
    consequence: the generated video might be perfectly legible and simply OUT OF
    DISTRIBUTION for a model trained on camera footage. One says wait for a better
    generator, the other says fine-tune the labeller, which is cheap and possible today.

    So this degrades the REAL clip to the dreamed clip's sharpness and lets the same
    inverse model read that. If the blurred-real error lands on the dreamed error, lost
    detail explains the tax. If it stays near the real baseline, detail does not explain
    it and something about generated video specifically is the problem.

    MEASURED: it stays at the baseline. 0.0160 becomes 0.0161 while the dreamed clip
    sits at 0.0534, so blur reproduces 0% and 2% of the gap on the two examples. The
    obvious reading is wrong, and the sharpness gap was only 9% and 29% to begin with.

    Binary search on Gaussian radius because the relationship between radius and
    Laplacian variance is monotone but not analytic. Searched on a subsample of frames
    and applied to all of them: the sharpness of a clip is an average over frames anyway,
    and blurring 61 frames eight times over to refine a radius is wasted work.

    Honest limit, and it belongs next to the result rather than in a footnote: this
    matches ONE axis. A generated clip differs from a filmed one in temporal coherence
    and in its artifacts as well as in spatial detail, so a blurred real clip is a
    control for detail alone. Landing on the dreamed error is therefore strong evidence
    that detail is sufficient to explain the gap; missing it is evidence for a
    distribution gap rather than proof of one.
    """
    from PIL import ImageFilter

    probe = sample_frames(frames, 8)
    current = sharpness(probe)
    if target >= current:
        log.info("blur: target %.0f already >= source %.0f, nothing to do", target, current)
        return list(frames), 0.0

    lo, hi = 0.0, max_radius
    best = hi
    for _ in range(8):
        mid = (lo + hi) / 2.0
        got = sharpness([f.filter(ImageFilter.GaussianBlur(mid)) for f in probe])
        if got > target:
            lo = mid          # still too sharp, blur harder
        else:
            hi = best = mid   # too soft, ease off
    blurred = [f.filter(ImageFilter.GaussianBlur(best)) for f in frames]
    log.info("blur: radius %.2f took sharpness %.0f -> %.0f (target %.0f)",
             best, current, sharpness(sample_frames(blurred, 8)), target)
    return blurred, best


def rank_string(values: dict) -> str:
    """Names ordered by value, low to high, as 'a < b < c'. For stating an ordering."""
    return " < ".join(n for n, _ in sorted(values.items(), key=lambda kv: kv[1]))


# ── Testing which embodiments the checkpoint actually knows ─────────────────────
#
# `_EMBODIMENT_TO_RAW_ACTION_DIM` in diffusers lists 15 domains. That table says what the
# ARCHITECTURE accepts, not what these weights were trained on, and the difference is not
# academic: `pusht` runs without error, generates fast, and produces a photorealistic
# robot arm on a wooden desk instead of PushT.
#
# The discriminating test is scene retention against a REAL in-domain frame. An earlier
# attempt scored action sensitivity instead (flat action versus large action, per
# embodiment, on a borrowed conditioning frame) and it failed completely: all 15 scored a
# divergence of 16 to 30 including the one already known to be broken, because arbitrary
# actions on a foreign frame make every domain wander. Only real in-domain data tells you
# anything.
#
# Aliases found while doing it, identical outputs meaning identical internal embodiment:
#   droid_lerobot == robomind-franka
#   agibotworld   == agibot_gear_gripper == agibot_gear_gripper_ext

# Decoded videos and parquet tables, keyed by (repo, video path). A LeRobot episode file
# is one mp4 for thousands of frames, so the cost is all in decoding it, and a task that
# wants many windows out of the same episode should pay that once.
_LEROBOT_CACHE: dict = {}


def load_lerobot_sample(
    repo: str,
    video_path: str,
    *,
    start: int = 30,
    count: int = 17,
    size: tuple[int, int] = (320, 192),
) -> dict:
    """Real frames and the real actions beside them, from a LeRobot dataset on the Hub.

    LeRobot datasets keep pixels in mp4s and everything else in parquet, indexed by the
    same frame order, so a slice of one lines up with a slice of the other. That pairing
    is the whole point: it is what makes the conditioning frame, the actions and the
    ground-truth continuation come from the same moment of the same episode.

    Returns the conditioning frame, the real continuation to compare against, and the
    raw actions in whatever width the dataset ships.
    """
    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    import media

    # Decode each video and read each parquet ONCE, however many windows are asked for.
    # Without this a task that wants 32 windows decodes the whole mp4 32 times, which
    # measured at ~26s per window and dominated the run: the training task spent longer
    # assembling its dataset than it did generating the dreams in it.
    key = (repo, video_path)
    if key not in _LEROBOT_CACHE:
        clip = hf_hub_download(repo, video_path, repo_type="dataset")
        table = pq.read_table(
            hf_hub_download(repo, "data/chunk-000/file-000.parquet", repo_type="dataset")
        )
        _LEROBOT_CACHE[key] = (media.decode(clip), table)
        log.info("cached %s: %s frames", repo, len(_LEROBOT_CACHE[key][0]))
    decoded, table = _LEROBOT_CACHE[key]
    frames = [f.resize(size) for f in decoded[start:start + count]]
    actions = np.stack(
        table.column("action").to_pylist()[start:start + count - 1]
    ).astype("float32")
    log.info("%s: %s frames %s, actions %s", repo, len(frames), size, actions.shape)
    return {"frames": frames, "first_frame": frames[0], "actions": actions}


def axis_angle_to_6d(actions):
    """LeRobot DROID's 7-D action to the 10-D one Cosmos wants.

    DROID ships 3 translation + 3 axis-angle rotation + 1 gripper. Cosmos's
    `droid_lerobot` is 10 wide, which is 3 translation + a 6D rotation + 1 gripper: the
    first two columns of the rotation matrix, flattened. That encoding is standard
    because it is continuous, which axis-angle and quaternions are not, and networks
    regress it far better.

    Rodrigues to build the matrix, then take columns 0 and 1. Verified only indirectly:
    the converted actions produce a coherent continuation of a real DROID clip, which is
    evidence for this encoding rather than proof of it.
    """
    import numpy as np

    out = np.zeros((len(actions), 10), dtype="float32")
    out[:, :3] = actions[:, :3]
    for i, r in enumerate(actions[:, 3:6]):
        theta = float(np.linalg.norm(r)) + 1e-9
        k = r / theta
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
        out[i, 3:9] = R[:, :2].T.reshape(-1)
    out[:, 9] = actions[:, 6]
    return out


def scene_retained(generated: list, real: list, tolerance: float = 0.15) -> dict:
    """Did the model continue the scene it was given, or leave for somewhere else?

    Mean luminance against the real continuation's, which sounds crude and is the number
    that actually separated the two cases by a mile. A real PushT clip sits at 249 and
    the model's replacement for it at 127 to 169; a real DROID clip sits at 97.3 and the
    model's continuation at 99.4. Nothing subtle was needed.

    The limitation belongs next to the result: this catches a model that swaps a bright
    synthetic scene for a dark photoreal one, and it would NOT catch a model that
    wandered to somewhere of similar brightness. The frame strips in the report are the
    real evidence; this is the number that makes it sortable.
    """
    import numpy as np

    def lum(fr):
        return float(np.mean([np.asarray(f.convert("L"), dtype="float32").mean() for f in fr]))

    gen_l, real_l = lum(generated), lum(real)
    ratio = gen_l / (real_l or 1.0)
    return {
        "luminance_generated": round(gen_l, 1),
        "luminance_real": round(real_l, 1),
        "ratio": round(ratio, 3),
        "divergence": round(frame_divergence(real, generated), 2),
        "motion_generated": round(clip_stats(generated)["motion"], 2),
        "motion_real": round(clip_stats(real)["motion"], 2),
        "kept": abs(ratio - 1.0) <= tolerance,
    }
