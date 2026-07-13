"""Flyte-free core: load a video pipeline, generate a clip, render the report.

Nothing here imports flyte, so it runs three ways unchanged: inside a Flyte GPU
task (compare_pipeline.py), inside the Gradio app, and directly on the Spark host
(run_local.py).

── Playing video INSIDE a Flyte report ─────────────────────────────────────────
This is the interesting problem, and the constraint is tight. The Flyte console
renders a report either as a document or by injecting the HTML via innerHTML, and
a CSP drops anything fetched from outside. So: **no <script>, no external assets,
no <source src="http://...">**. Everything must be inline and self-contained.

The good news is that a plain HTML5 <video> element needs no JavaScript at all:

    <video controls loop muted playsinline src="data:video/mp4;base64,...">

`controls` gives you scrub/play/pause for free, `loop` makes a 2-second clip
readable, and the base64 data URI means the bytes travel inside the HTML. So we
get *real playback in the report*, not a poster frame.

The catch is weight. Base64 inflates by 4/3, and every clip in a grid lands in
ONE html document. So a report clip is not the artifact clip:

  - the ARTIFACT mp4 (in the task's output Dir) is full resolution, full quality.
  - the REPORT mp4 is re-encoded small (REPORT_MAX_SIDE) purely to be embeddable.

And because embedded playback can still fail (an old console, a strict CSP, a
codec the browser dislikes), every cell ALSO carries a **frame strip**: N frames
sampled across the clip, composited into one JPEG. That is the fallback, and it's
independently useful: a strip shows you motion, drift, and temporal consistency at
a glance in a way a playing video actually hides. If a clip is over
MAX_EMBED_BYTES even after downscaling, we drop the <video> and ship the strip
alone rather than producing a report that won't load.
"""

from __future__ import annotations

import base64
import html
import io
import os
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from models import ImageModelSpec, VideoModelSpec

# Report-embedded clips are re-encoded to this long side. 512 keeps a 2-4s clip in
# the low hundreds of KB, so a 2x3 grid stays a few MB of HTML and the console
# iframe renders it instantly.
REPORT_MAX_SIDE = 512

# Hard ceiling for one embedded clip's base64 payload. Past this we ship the frame
# strip only: a report that fails to load is worse than one without playback.
MAX_EMBED_BYTES = 6_000_000

# Frames sampled into the fallback/evaluation strip.
STRIP_FRAMES = 6
STRIP_FRAME_WIDTH = 240


# ── Loading ─────────────────────────────────────────────────────────────────────

def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def load_pipeline(
    spec: VideoModelSpec,
    *,
    device: str = "cuda",
    model_path: str | None = None,
    i2v: bool = False,
):
    """Load `spec` into a ready-to-call diffusers pipeline.

    `model_path` is a local dir of already-downloaded weights (from the cached
    `fetch_weights` task); diffusers loads straight from it and never touches the
    hub. `i2v=True` selects the spec's image-to-video class instead of its T2V one.

    Two model-specific quirks are handled here rather than at the call site:

      1. **Wan's VAE must stay fp32.** Loading the whole pipeline in bf16 takes the
         VAE with it and the decode produces washed-out, banded output. The model
         card loads AutoencoderKLWan separately in fp32 and passes it in; so do we.
      2. **No CPU offload.** Every HF example calls enable_sequential_cpu_offload()
         because it assumes a 24GB discrete card. The GB10 has 128GB of *unified*
         memory: the GPU and CPU address the same pool, so "offloading" moves
         nothing and just adds synchronization. We keep the model resident. Set
         VIDEOGEN_OFFLOAD=1 to force it back on for a discrete-GPU box.
    """
    import diffusers

    dtype = _torch_dtype(spec.dtype)
    source = model_path or spec.repo
    cls_name = spec.i2v_pipeline if i2v else spec.pipeline

    load_kwargs: dict = {"torch_dtype": dtype}
    if model_path is None:
        load_kwargs["token"] = os.environ.get("HF_TOKEN")

    # Wan: hand the pipeline an fp32 VAE (see quirk 1).
    if spec.vae_dtype:
        vae_cls = getattr(diffusers, "AutoencoderKLWan", None)
        if vae_cls is not None:
            try:
                load_kwargs["vae"] = vae_cls.from_pretrained(
                    source, subfolder="vae", torch_dtype=_torch_dtype(spec.vae_dtype)
                )
            except Exception as e:
                print(f"[videogen] {spec.key}: fp32 VAE load failed ({e}); "
                      f"falling back to the pipeline's own VAE", flush=True)

    cls = getattr(diffusers, cls_name, None)
    if cls is None:
        # A brand-new class name that this diffusers doesn't have. DiffusionPipeline
        # reads the repo's own model_index.json and usually still resolves it.
        print(f"[videogen] diffusers has no {cls_name}; falling back to DiffusionPipeline",
              flush=True)
        cls = diffusers.DiffusionPipeline

    # ── The unified-memory double-copy trap (this is THE bug on a GB10) ──────────
    #
    # The obvious way to load a pipeline is `from_pretrained(...).to("cuda")`. On a
    # discrete GPU that's fine: the weights land in host RAM, get copied across PCIe
    # into separate VRAM, and the host copy is freed. Two different pools.
    #
    # On GB10 there is only ONE pool. `from_pretrained` materializes the model in
    # system RAM and `.to("cuda")` then allocates a SECOND full copy in the very same
    # 119.7GiB. So peak demand is 2x the model, and a 72GB model needs ~144GB it will
    # never have. Measured: LTX-2.3 died with "527MiB free" while PyTorch itself held
    # only 52GB, because the host-side copy was holding the rest.
    #
    # `device_map="balanced"` makes accelerate dispatch each module straight to the
    # device as it's read, so the full host-side copy never exists. With a single GPU
    # "balanced" simply means "all of it on cuda:0".
    #
    # The `max_memory` override is not optional. Accelerate sizes its budget from
    # torch.cuda.mem_get_info(), and on unified memory "free" reads as near-zero
    # (the OS page cache holds it, reclaimable but not counted). Left to itself,
    # accelerate concludes the GPU is full and offloads everything back to CPU,
    # which is the thing we're trying to avoid. So we tell it the real budget and
    # give the CPU a deliberately small allowance so nothing lands there.
    #
    # Only big models take this path: `.to()` is simpler, and for a 15-26GB model the
    # transient double copy fits comfortably.
    use_device_map = (
        device == "cuda"
        and spec.est_vram_gb >= DEVICE_MAP_MIN_GB
        and os.environ.get("VIDEOGEN_OFFLOAD") != "1"
    )
    if use_device_map:
        _, total = gpu_memory_gb()
        fraction = float(os.environ.get("VIDEOGEN_MEM_FRACTION", DEFAULT_MEM_FRACTION))
        # Mind the units. gpu_memory_gb() returns decimal GB (bytes/1e9) but accelerate's
        # max_memory string is GiB (bytes/2**30), and the two differ by ~7%. Feeding a
        # GB number in with a GiB label hands accelerate a budget ABOVE the cap that
        # set_per_process_memory_fraction enforces, so torch would OOM before accelerate
        # ever thought it was full. Convert properly and keep both numbers in agreement.
        total_gib = total * 1e9 / 2**30
        budget_gib = int(total_gib * fraction) if total else 96
        load_kwargs["device_map"] = "balanced"
        load_kwargs["max_memory"] = {0: f"{budget_gib}GiB", "cpu": "8GiB"}
        print(f"[videogen] {spec.key} is ~{spec.est_vram_gb:.0f}GB: loading straight to "
              f"the device (device_map=balanced, max {budget_gib}GiB of "
              f"{total_gib:.0f}GiB) to avoid the unified-memory double copy", flush=True)

    print(f"[videogen] loading {source} via {cls.__name__}", flush=True)
    try:
        pipe = cls.from_pretrained(source, **load_kwargs)
    except (ValueError, NotImplementedError) as e:
        # Not every pipeline accepts device_map. Fall back to the plain path rather
        # than failing outright; it may still fit if the model isn't huge.
        if not use_device_map:
            raise
        print(f"[videogen] {spec.key}: device_map rejected ({type(e).__name__}: {e}); "
              f"retrying with a plain load", flush=True)
        load_kwargs.pop("device_map", None)
        load_kwargs.pop("max_memory", None)
        use_device_map = False
        pipe = cls.from_pretrained(source, **load_kwargs)

    if os.environ.get("VIDEOGEN_OFFLOAD") == "1":
        pipe.enable_model_cpu_offload()
    elif not use_device_map:
        pipe = pipe.to(device)
    # else: accelerate already placed every module on the device. Calling .to() now
    # would either be a no-op or raise, so don't.

    # VAE tiling is the one memory optimization that genuinely matters for video:
    # decoding a 121-frame latent in one shot is the single biggest allocation in
    # the whole run and OOMs long before the transformer does. Tiling decodes it in
    # chunks. It does not change the output.
    for opt in ("enable_tiling", "enable_slicing"):
        vae = getattr(pipe, "vae", None)
        if vae is not None and hasattr(vae, opt):
            try:
                getattr(vae, opt)()
            except Exception:
                pass
    return pipe


def load_image_pipeline(spec: ImageModelSpec, *, device: str = "cuda",
                        model_path: str | None = None):
    """Load a small text-to-image pipeline, for generating an I2V first frame."""
    import diffusers

    source = model_path or spec.repo
    kwargs: dict = {"torch_dtype": _torch_dtype(spec.dtype)}
    if model_path is None:
        kwargs["token"] = os.environ.get("HF_TOKEN")
    cls = getattr(diffusers, spec.pipeline, diffusers.AutoPipelineForText2Image)
    print(f"[videogen] loading first-frame model {source} via {cls.__name__}", flush=True)
    return cls.from_pretrained(source, **kwargs).to(device)


def free_gpu_memory() -> None:
    """Hand a just-dropped pipeline's memory back to the allocator.

    On the GB10's unified memory there is no separate VRAM pool to evict to, so
    `.to("cpu")` frees nothing; the reclaim comes from dropping the Python objects
    and then returning PyTorch's cached blocks. Best-effort and import-safe.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


# ── Memory guardrails ───────────────────────────────────────────────────────────
#
# This is the part that keeps a bad config from taking the box down, and on the
# GB10 it is not optional.
#
# On a discrete GPU, asking for more memory than exists raises OutOfMemoryError and
# you move on. On Grace-Blackwell the GPU allocates from the SAME DRAM the OS lives
# in. There is nothing to OOM *into*: the driver spins on NV_ERR_NO_MEMORY while the
# kernel reclaims from everything else, and the whole machine locks up instead of
# raising. You lose the box, not just the run. (Same failure the GRPO workshop hit;
# same fix.)
#
# Video generation is exactly where you'd hit it: these are the biggest models we
# run here. LTX-2.3 in bf16 is ~72GB of weights against a 119.7GiB pool that the OS,
# the page cache, rustfs, and every other pod are also using. So:
#
#   1. `set_per_process_memory_fraction` caps the process. Past the cap PyTorch
#      raises a normal OutOfMemoryError, which Flyte's retry can act on, instead of
#      wedging the host. This is the guardrail that matters.
#   2. `preflight_fit` refuses a model that plainly cannot fit BEFORE spending 20
#      minutes loading 95GB of weights to find out.
#
# The fraction is a genuine tradeoff, not a free win. Too low and LTX-2.3 can't load
# at all; too high and there's no headroom for the VAE decode spike (the single
# largest allocation in a video run, which is why we always enable VAE tiling).
# 0.90 leaves ~12GB for the OS, which has held. Override with VIDEOGEN_MEM_FRACTION.

DEFAULT_MEM_FRACTION = 0.90

# Models estimated at or above this bf16 footprint load via device_map instead of
# `.to("cuda")`, to dodge the unified-memory double copy (see load_pipeline). Set
# below LTX-2.3 (~72GB) and above Wan 2.2 TI2V-5B (~26GB), which loads fine plainly.
DEVICE_MAP_MIN_GB = 40.0


def gpu_memory_gb() -> tuple[float, float]:
    """(free, total) GPU memory in GB. (0, 0) when there's no CUDA device."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0, 0.0
        free, total = torch.cuda.mem_get_info()
        return free / 1e9, total / 1e9
    except Exception:
        return 0.0, 0.0


def guard_gpu_memory(fraction: float | None = None) -> None:
    """Cap this process's share of the GPU pool so overshoot RAISES, not hangs.

    Call once before loading a pipeline. See the block comment above for why a cap
    is load-bearing on unified memory rather than a nicety.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return
        if fraction is None:
            fraction = float(os.environ.get("VIDEOGEN_MEM_FRACTION", DEFAULT_MEM_FRACTION))
        torch.cuda.set_per_process_memory_fraction(fraction)
        free, total = gpu_memory_gb()
        print(
            f"[videogen] {torch.cuda.get_device_name(0)}: {total:.1f}GB total, "
            f"{free:.1f}GB free; capping this process at {fraction:.0%} "
            f"({total * fraction:.1f}GB)",
            flush=True,
        )
    except Exception as e:
        print(f"[videogen] could not set a memory cap ({e}); continuing uncapped", flush=True)


def preflight_fit(spec: VideoModelSpec, fraction: float | None = None) -> None:
    """Refuse a model that cannot fit, before paying to load it.

    Raises MemoryError with an actionable message. Only fires when the estimate is
    unambiguous, because `est_vram_gb` is weights-only and approximate, so we compare it
    against the cap with headroom for activations and the VAE decode rather than
    treating it as exact. A model that's merely *tight* is allowed through: the cap
    from guard_gpu_memory turns a genuine overshoot into a clean OutOfMemoryError.
    """
    if not spec.est_vram_gb:
        return
    _, total = gpu_memory_gb()
    if not total:
        return
    if fraction is None:
        fraction = float(os.environ.get("VIDEOGEN_MEM_FRACTION", DEFAULT_MEM_FRACTION))
    budget = total * fraction

    # Activations + the VAE decode spike land on top of the weights. ~15% of the
    # weight footprint is a rough but serviceable allowance at our clip sizes.
    need = spec.est_vram_gb * 1.15
    if need > budget:
        raise MemoryError(
            f"{spec.key} needs about {need:.0f}GB (≈{spec.est_vram_gb:.0f}GB of bf16 "
            f"weights plus activations) but this process is capped at {budget:.0f}GB "
            f"of the {total:.0f}GB pool.\n"
            f"Options: pick a smaller model (wan22-ti2v-5b is ~26GB, wan21-t2v-1.3b "
            f"~15GB), raise VIDEOGEN_MEM_FRACTION if the box is otherwise idle, or "
            f"set VIDEOGEN_OFFLOAD=1 to stream layers (much slower, and it buys you "
            f"little on unified memory)."
        )
    # Deliberately NOT warning when est_vram_gb > free. On unified memory
    # `mem_get_info()`'s "free" excludes the OS page cache, which is reclaimable, so
    # it reads as near-zero almost always: we measured 5.4GB "free" on a box that
    # then happily ran a 16GB model. Warning on that number cries wolf every single
    # run. The cap above is what actually protects the host, and it keys off `total`.


def prepare_gpu(spec: VideoModelSpec) -> None:
    """Get the GPU into a known-clean, capped, verified state before a load.

    Always call this instead of the three pieces separately, because the ORDER matters:

      1. **Free first.** A previous pipeline in this same process (the Gradio app
         switching models, run_local looping over a grid) still holds its weights
         in PyTorch's cache until we drop it, and a stale pod can hold the pool
         too. Loading a 72GB model on top of a 26GB corpse is how you wedge the
         box. `free_gpu_memory()` is a no-op when there's nothing to reclaim, so
         this is free insurance.
      2. **Then cap**, so any overshoot from here raises instead of hanging.
      3. **Then check** what's actually free, which is only meaningful *after* the
         free in step 1, or we'd refuse a model over memory we were about to
         reclaim anyway.
      4. Reset the peak counter so the number we report belongs to this generation.
    """
    free_gpu_memory()
    guard_gpu_memory()
    preflight_fit(spec)
    reset_peak_memory()


def peak_memory_gb() -> float:
    """Peak allocation since the last reset. Reported in the run's report."""
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return 0.0


def reset_peak_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


# ── Generating ──────────────────────────────────────────────────────────────────

@dataclass
class ClipResult:
    """One (model, prompt) generation, ready to drop into a report."""
    model_key: str
    prompt: str
    seconds: float
    video_uri: str = ""       # base64 mp4 data URI, or "" if too big / failed
    strip_uri: str = ""       # base64 JPEG frame strip
    n_frames: int = 0
    fps: int = 0
    width: int = 0
    height: int = 0
    has_audio: bool = False
    peak_gb: float = 0.0      # peak GPU allocation; how close this ran to the ceiling
    embed_note: str = ""      # why the player is missing, when it is
    error: str = ""


def generate_video(
    pipe,
    spec: VideoModelSpec,
    prompt: str,
    *,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int = 1234,
    width: int | None = None,
    height: int | None = None,
    num_frames: int | None = None,
    negative_prompt: str | None = None,
    image=None,                       # PIL first frame -> image-to-video
    on_step=None,                     # callback(i, total) for a live report
):
    """Run one generation. Returns (frames: list[PIL], audio, sample_rate).

    `audio` is None for every model except LTX-2, which generates a synced track.
    """
    import torch

    steps = spec.steps if steps is None else steps
    guidance = spec.guidance if guidance is None else guidance
    width = spec.width if width is None else width
    height = spec.height if height is None else height
    num_frames = spec.num_frames if num_frames is None else num_frames
    if negative_prompt is None:
        negative_prompt = spec.negative_prompt

    gen_device = "cpu" if os.environ.get("VIDEOGEN_OFFLOAD") == "1" else "cuda"
    generator = (
        torch.Generator(device=gen_device).manual_seed(int(seed)) if seed >= 0 else None
    )

    kwargs: dict = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "num_frames": int(num_frames),
        "num_inference_steps": int(steps),
        "generator": generator,
        "output_type": "pil",
    }
    if guidance is not None:
        kwargs["guidance_scale"] = float(guidance)
    if negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if image is not None:
        kwargs["image"] = image

    # LTX-2 takes the frame rate as a sampling input (it conditions on it), unlike
    # Wan/CogVideoX where fps is purely an encoding choice made at export time.
    if spec.pipeline.startswith("LTX2"):
        kwargs["frame_rate"] = float(spec.fps)
        # The distilled model is trained to a FIXED 8-sigma schedule. Passing plain
        # num_inference_steps makes the scheduler invent its own, which produces
        # mush. Pass the real sigmas the checkpoint expects.
        if spec.distilled:
            try:
                from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES

                kwargs["sigmas"] = DISTILLED_SIGMA_VALUES
                kwargs["num_inference_steps"] = len(DISTILLED_SIGMA_VALUES)
            except ImportError:
                pass

    # Live progress: diffusers' callback is (pipe, i, t, kwargs) -> dict.
    if on_step is not None:
        total = int(kwargs["num_inference_steps"])

        def _cb(_pipe, i, _t, cb_kwargs):
            try:
                on_step(i + 1, total)
            except Exception:
                pass
            return cb_kwargs

        kwargs["callback_on_step_end"] = _cb

    out = pipe(**kwargs)

    # LTX-2 returns frames AND audio; everything else returns frames only.
    frames = out.frames[0]
    audio, sample_rate = None, None
    raw_audio = getattr(out, "audio", None)
    if raw_audio is not None and len(raw_audio):
        audio = raw_audio[0].float().cpu()
        # The vocoder knows its own output rate (24kHz for LTX-2). Getting this
        # wrong doesn't error, it just pitch-shifts the audio, so read it rather
        # than hardcode it.
        vocoder = getattr(pipe, "vocoder", None)
        sample_rate = int(getattr(getattr(vocoder, "config", None), "output_sampling_rate", 24000))
    return frames, audio, sample_rate


def timed_generate(pipe, spec, prompt, **kw):
    """generate_video() wrapped with wall-clock timing."""
    t0 = time.time()
    frames, audio, sr = generate_video(pipe, spec, prompt, **kw)
    return frames, audio, sr, time.time() - t0


# ── Encoding ────────────────────────────────────────────────────────────────────

def _stereo(audio):
    """Coerce a generated audio track to the (samples, 2) layout the muxer demands.

    diffusers' `_write_audio` accepts only 2-channel audio: it reshapes (N,) to
    (N, 1) and then *raises* ValueError, and a mono track would otherwise throw
    away an otherwise-good clip on the very last step. Models don't agree on
    layout (mono vs stereo, channels-first vs channels-last), so normalize rather
    than trust: pick the channel axis as the SHORT one (a clip has far more samples
    than channels), then duplicate mono up to stereo.
    """
    import torch

    a = audio if isinstance(audio, torch.Tensor) else torch.as_tensor(audio)
    a = a.squeeze()
    if a.ndim == 1:                       # mono -> duplicate into both channels
        return a[:, None].repeat(1, 2)
    if a.ndim != 2:
        raise ValueError(f"unexpected audio shape {tuple(a.shape)}")
    if a.shape[0] < a.shape[1]:           # channels-first -> channels-last
        a = a.T
    if a.shape[1] == 1:
        return a.repeat(1, 2)
    return a[:, :2]                       # >2 channels: keep the first two


def write_mp4(frames, fps: int, path: str | Path, audio=None, sample_rate: int | None = None):
    """Encode frames (+ optional audio) to an .mp4.

    diffusers' `encode_video` is the only export path that muxes an audio track,
    which LTX-2 needs; it's PyAV-backed. `export_to_video` is the frames-only
    fallback for a diffusers too old to have it.

    An audio failure never costs you the clip: if muxing throws, we fall back to
    writing the video silently rather than losing a generation that took minutes.
    """
    path = str(path)
    try:
        from diffusers.utils import encode_video
    except ImportError:
        from diffusers.utils import export_to_video

        export_to_video(frames, path, fps=int(fps))
        return path

    if audio is not None:
        try:
            encode_video(frames, fps=int(fps), output_path=path,
                         audio=_stereo(audio), audio_sample_rate=sample_rate)
            return path
        except Exception as e:
            print(f"[videogen] audio mux failed ({type(e).__name__}: {e}); "
                  f"writing the clip without sound", flush=True)

    encode_video(frames, fps=int(fps), output_path=path)
    return path


def _resize(frames, max_side: int):
    """Downscale frames so the long side is `max_side`. H.264 wants even dims."""
    w, h = frames[0].size
    if max(w, h) <= max_side:
        return frames
    scale = max_side / float(max(w, h))
    nw, nh = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    return [f.resize((nw, nh)) for f in frames]


def video_data_uri(
    frames, fps: int, audio=None, sample_rate: int | None = None,
    max_side: int = REPORT_MAX_SIDE, budget: int = MAX_EMBED_BYTES,
) -> tuple[str, str]:
    """Encode a REPORT-sized mp4 and base64 it. Returns (data_uri, note).

    An empty data_uri with a note means "too big to embed, use the strip". The
    caller always has the strip, so this degrades rather than breaks.
    """
    small = _resize(frames, max_side)
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "preview.mp4"
        try:
            write_mp4(small, fps, p, audio=audio, sample_rate=sample_rate)
        except Exception as e:
            return "", f"preview encode failed: {type(e).__name__}: {e}"
        raw = p.read_bytes()

    if len(raw) > budget:
        return "", (f"clip is {len(raw) / 1e6:.1f}MB, over the "
                    f"{budget / 1e6:.0f}MB embed budget; frames shown instead")
    b64 = base64.b64encode(raw).decode("ascii")
    return f"data:video/mp4;base64,{b64}", ""


def frame_strip_data_uri(frames, n: int = STRIP_FRAMES,
                         frame_width: int = STRIP_FRAME_WIDTH) -> str:
    """Sample `n` frames evenly across the clip into one horizontal JPEG strip.

    One image, not n, so the report stays light. This is both the no-playback
    fallback and, honestly, the better evaluation surface: temporal drift and
    identity collapse are obvious side by side and easy to miss while a 3-second
    clip is looping past you.
    """
    from PIL import Image

    if not frames:
        return ""
    n = min(n, len(frames))
    idx = [round(i * (len(frames) - 1) / max(n - 1, 1)) for i in range(n)]
    picks = [frames[i] for i in idx]

    w, h = picks[0].size
    fh = int(h * (frame_width / float(w)))
    gap = 3
    strip = Image.new("RGB", (frame_width * n + gap * (n - 1), fh), "#e1e0d9")
    for i, f in enumerate(picks):
        strip.paste(f.convert("RGB").resize((frame_width, fh)), (i * (frame_width + gap), 0))

    buf = io.BytesIO()
    strip.save(buf, format="JPEG", quality=82, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def pil_to_data_uri(img, max_side: int = 512, quality: int = 85) -> str:
    """A single still (the I2V first frame) as a JPEG data URI."""
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / float(max(w, h))
        img = img.resize((int(w * s), int(h * s)))
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def build_clip_result(
    spec: VideoModelSpec, prompt: str, frames, audio, sample_rate, seconds: float,
    fps: int | None = None,
) -> ClipResult:
    """Package a raw generation into the report-ready ClipResult (player + strip)."""
    fps = spec.fps if fps is None else fps
    w, h = frames[0].size
    uri, note = video_data_uri(frames, fps, audio=audio, sample_rate=sample_rate)
    return ClipResult(
        model_key=spec.key, prompt=prompt, seconds=seconds,
        video_uri=uri, strip_uri=frame_strip_data_uri(frames), embed_note=note,
        n_frames=len(frames), fps=fps, width=w, height=h,
        has_audio=audio is not None, peak_gb=peak_memory_gb(),
    )


# ── Report ──────────────────────────────────────────────────────────────────────
#
# Constraints (from the Flyte console, not from taste): no <script>, no external
# assets. <video controls> needs neither, so playback works. The only interactivity
# beyond the native player is an inline-onclick lightbox for the frame strips.

REPORT_CSS = """
<style>
  .vg-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
             color: #0b0b0b; }
  .vg-wrap h2 { margin: 0 0 4px; }
  .vg-meta { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
  .vg-grid { display: grid; gap: 16px;
             grid-template-columns: repeat(auto-fill, minmax(340px, 1fr)); }
  .vg-cell { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
             background: #fff; display: flex; flex-direction: column; }
  .vg-cell video { width: 100%; height: auto; display: block; background: #111; }
  .vg-cap { padding: 9px 11px; }
  .vg-model { font-weight: 600; font-size: 14px; }
  .vg-sub { color: #6b7280; font-size: 12px; margin-top: 3px; line-height: 1.4; }
  .vg-fam { display: inline-block; font-size: 11px; color: #374151; background: #f3f4f6;
            border-radius: 999px; padding: 1px 8px; margin-top: 4px; }
  .vg-audio { display: inline-block; font-size: 11px; color: #7c2d12; background: #ffedd5;
              border-radius: 999px; padding: 1px 8px; margin: 4px 0 0 4px; font-weight: 600; }
  .vg-err { padding: 16px; color: #b91c1c; font-size: 13px; white-space: pre-wrap; }
  .vg-note { padding: 8px 11px; color: #92400e; background: #fffbeb; font-size: 12px; }
  .vg-prompt { background: #f9fafb; border-left: 3px solid #6366f1; padding: 8px 12px;
               border-radius: 6px; margin: 6px 0 14px; font-size: 14px; }
  /* The frame strip. Always present, under the player: it is the fallback when
     playback fails AND the surface where temporal drift is actually legible. */
  .vg-strip { padding: 0 11px 10px; }
  .vg-strip img { width: 100%; height: auto; border-radius: 6px; display: block;
                  cursor: zoom-in; }
  .vg-strip .lbl { font-size: 11px; color: #898781; margin: 8px 0 4px;
                   text-transform: uppercase; letter-spacing: .04em; }
  .vg-first { padding: 0 11px 10px; }
  .vg-first img { width: 100%; height: auto; border-radius: 6px; display: block;
                  cursor: zoom-in; }
  /* Lightbox: pure inline handlers, no JS block, so it survives innerHTML injection. */
  #vg-lb { position: fixed; inset: 0; z-index: 9999; display: none; cursor: zoom-out;
           flex-direction: column; align-items: center; justify-content: center;
           gap: 12px; padding: 24px; background: rgba(0,0,0,.88); }
  #vg-lb img { max-width: 96vw; max-height: 86vh; border-radius: 8px; }
  #vg-lb #vg-lb-cap { color: #e5e7eb; font-size: 14px; }
</style>
"""

_ZOOM = (
    "document.getElementById('vg-lb-img').src=this.src;"
    "document.getElementById('vg-lb-cap').textContent=this.dataset.cap;"
    "document.getElementById('vg-lb').style.display='flex'"
)
_LIGHTBOX = (
    "<div id=\"vg-lb\" onclick=\"this.style.display='none'\" style=\"display:none\">"
    '<img id="vg-lb-img" src="" alt="zoomed"/><div id="vg-lb-cap"></div></div>'
)


def _player(r: ClipResult) -> str:
    """The <video> element. No JS: `controls` and `loop` are native attributes.

    muted + playsinline + autoplay: a muted autoplaying video is allowed to start
    without a user gesture in every browser, so the grid comes alive on load. The
    controls still let you unmute, which matters for LTX-2, whose audio track is
    the whole point.
    """
    if not r.video_uri:
        return ""
    return (
        f'<video controls loop muted autoplay playsinline preload="metadata" '
        f'src="{r.video_uri}"></video>'
    )


def _zoom_img(uri: str, cap: str) -> str:
    return (f'<img src="{uri}" alt="{html.escape(cap)}" '
            f'data-cap="{html.escape(cap, quote=True)}" onclick="{_ZOOM}"/>')


def _cell(spec: VideoModelSpec, r: ClipResult) -> str:
    if r.error:
        return (f'<div class="vg-cell"><div class="vg-err">⚠️ {html.escape(r.error)}</div>'
                f'<div class="vg-cap"><div class="vg-model">{html.escape(spec.key)}</div>'
                f'</div></div>')

    body = _player(r)
    note = f'<div class="vg-note">{html.escape(r.embed_note)}</div>' if r.embed_note else ""
    strip = ""
    if r.strip_uri:
        strip = (
            f'<div class="vg-strip"><div class="lbl">frames across the clip</div>'
            f'{_zoom_img(r.strip_uri, f"{spec.key} · {r.prompt}")}</div>'
        )
    dur = r.n_frames / r.fps if r.fps else 0
    badges = f'<span class="vg-fam">{html.escape(spec.family)}</span>'
    if r.has_audio:
        badges += '<span class="vg-audio">♪ audio</span>'
    peak = f' · peak {r.peak_gb:.0f}GB' if r.peak_gb else ""
    cap = (
        f'<div class="vg-cap"><div class="vg-model">{html.escape(spec.key)}</div>{badges}'
        f'<div class="vg-sub">{r.seconds:.0f}s to generate · {r.n_frames} frames · '
        f'{r.width}x{r.height} @{r.fps}fps ({dur:.1f}s clip){peak}<br>'
        f'{html.escape(spec.license)}</div></div>'
    )
    return f'<div class="vg-cell">{body}{note}{cap}{strip}</div>'


def render_grid(
    prompts: list[str],
    specs: list[VideoModelSpec],
    results: list[ClipResult],
    *,
    title: str = "Video model comparison",
    meta: str = "",
    first_frames: dict[str, str] | None = None,
) -> str:
    """Full prompt x model grid: one block per prompt, models as the columns.

    `first_frames` maps prompt -> data URI of the image the clip was animated from
    (image-to-video runs only), shown once above the row.
    """
    by_pair = {(r.prompt, r.model_key): r for r in results}
    blocks = []
    for p in prompts:
        cells = "".join(
            _cell(s, by_pair.get((p, s.key), ClipResult(s.key, p, 0.0, error="no result")))
            for s in specs
        )
        ff = ""
        if first_frames and first_frames.get(p):
            ff = (f'<div class="vg-first" style="max-width:340px">'
                  f'<div class="lbl" style="font-size:11px;color:#898781;margin:0 0 4px">'
                  f'first frame (generated, then animated)</div>'
                  f'{_zoom_img(first_frames[p], f"first frame · {p}")}</div>')
        blocks.append(
            f'<div class="vg-prompt">🎬 {html.escape(p)}</div>{ff}'
            f'<div class="vg-grid">{cells}</div>'
        )
    return (
        f'{REPORT_CSS}<div class="vg-wrap"><h2>{html.escape(title)}</h2>'
        f'<div class="vg-meta">{html.escape(meta)}</div>'
        + "".join(blocks) + "</div>" + _LIGHTBOX
    )


def render_status(title: str, body: str) -> str:
    """A plain in-progress report (shown while models load / denoise)."""
    return (f'{REPORT_CSS}<div class="vg-wrap"><h2>{html.escape(title)}</h2>'
            f'<div class="vg-meta">{body}</div></div>')
