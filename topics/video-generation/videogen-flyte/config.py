"""Shared Flyte config for the open-source video-generation demo.

Same shape as the image-generation demo next door (topics/image-generation), so
if you've read that one you already know this file:

  - compare_pipeline.py : GPU *tasks* that render prompts across video models and
    emit one side-by-side Flyte report with playable clips.
  - app.py              : a thin CPU Gradio *app* that launches runs and links the
    report. It holds no GPU and loads no model.

DGX-Spark-pinned (GB10 Blackwell, arm64, cu130 stack): aarch64 platform + the
devbox-local registry. Drop the pins for a generic Flyte 2 cluster.

── What's different from the image demo ────────────────────────────────────────
1. `av` (PyAV) is in the image. Video only becomes an .mp4 via an encoder, and
   diffusers' own `encode_video` hard-raises ImportError without PyAV. PyAV ships
   manylinux **aarch64** wheels, so it installs clean on the Spark; `imageio-ffmpeg`
   does not reliably, which is why we don't use it.
2. Bigger disks. A single LTX-2.3 pull is ~95GB and Wan 2.2 A14B is ~126GB.
3. The Spark env vars below. Video gen is the most bandwidth-hungry thing this box
   runs, and the difference between the defaults and these is not subtle.
"""

from __future__ import annotations

import os

import flyte
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# GB10 is Blackwell on the cu130 stack; PyTorch publishes matching aarch64 wheels.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

APP_NAME = "videogen-studio"
APP_PORT = 7863          # 7862 is the image-gen studio; don't collide

HF_HOME = "/tmp/hf"
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")


# ── Python deps ─────────────────────────────────────────────────────────────────
#
# diffusers>=0.39 is a hard floor, not a nicety: LTX2Pipeline / HunyuanVideo15Pipeline
# only exist from 0.39.0. (Several model cards still say "install diffusers from
# main"; that guidance is stale as of 0.39.)
VIDEO_SPEC = (
    "diffusers>=0.39.0",
    "transformers>=4.44",
    "accelerate>=0.34",
    "safetensors",
    "sentencepiece",     # T5/UMT5 tokenizers (Wan) and Gemma-3 (LTX-2)
    "protobuf",
    "pillow",
    "huggingface_hub",
    "hf_transfer",
    # PyAV: the mp4 encoder. diffusers.utils.encode_video (the only path that muxes
    # LTX-2's generated audio track into the clip) raises ImportError without it.
    "av",
    # imageio backs diffusers' plain export_to_video for the no-audio models.
    "imageio",
    # ftfy is an *undeclared hard dependency* of SkyReelsV2DiffusionForcingPipeline,
    # and diffusers won't tell you: it does `if is_ftfy_available(): import ftfy` at
    # module scope but then calls `ftfy.fix_text(text)` UNCONDITIONALLY when cleaning
    # the prompt, so a missing ftfy surfaces as `NameError: name 'ftfy' is not defined`
    # at generation time, not as an ImportError at load. Wan guards the same call site
    # (`if is_ftfy_available()`), which is why Wan runs fine without it and SkyReels
    # does not. Cheap pure-python wheel; just install it.
    "ftfy",
    # config.py imports kubernetes.client at module top for the app pod template,
    # and task pods import config too, so both images need it.
    "kubernetes",
)


def _torch_image(name: str, extra: tuple[str, ...] = ()) -> flyte.Image:
    """Debian base + cu130 torch + the diffusers video stack."""
    return (
        flyte.Image.from_debian_base(name=name, registry=REGISTRY, platform=PLATFORM)
        # ffmpeg for the system codecs; git because some HF repos vendor code via git.
        .with_apt_packages("git", "ffmpeg")
        .with_pip_packages("torch", "torchvision", index_url=TORCH_INDEX)
        .with_pip_packages(*VIDEO_SPEC, *extra)
    )


image = _torch_image("videogen-image")

# The studio app is a thin LAUNCHER: it submits runs and links the report, so it
# needs no torch/diffusers, just flyte + gradio and the model registry for the
# picker. Keeps the app pod tiny and means it never holds GPU memory.
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
studio_app_image = (
    flyte.Image.from_debian_base(
        name="videogen-studio-image", registry=REGISTRY, platform=PLATFORM
    )
    .with_pip_packages("flyte==2.2.1", "connectrpc==0.10.*", "gradio==5.42.0", "python-dotenv")
)


# ── DGX Spark tuning ────────────────────────────────────────────────────────────
#
# These are not cargo-cult. Sources: NVIDIA's own dev-forum threads on running
# ComfyUI video workflows on the Spark, where the same failures keep recurring.
#
#   CUDA_CACHE_MAXSIZE   The PTX->SASS JIT cache. Video models JIT a *lot* of
#                        kernels on first use; with the default (tiny) cache they
#                        get evicted and re-JIT-ed every denoise step. Sizing it
#                        to 4GB is reported to cut per-step time ~3x after warmup.
#   PYTORCH_ALLOC_CONF   expandable_segments lets the allocator grow/shrink
#                        segments instead of reserving fixed blocks. Matters more
#                        here than on a discrete GPU: "GPU memory" IS the same
#                        unified 128GB the OS and every other pod share, so a
#                        fragmented allocator OOMs under pressure a discrete card
#                        would never feel. (PYTORCH_CUDA_ALLOC_CONF is the older
#                        name; we set both so it works whichever torch reads.)
#   CUDA_MODULE_LOADING  EAGER: load modules up front rather than lazily mid-step.
#
# Two more knobs live OUTSIDE the container because they're host-level, and both
# are worth doing before a long session (see README):
#   sudo swapoff -a               swap thrash on unified memory = a silent freeze
#   sudo nvidia-smi -lgc 300,2100 clamp the clock: the hard crashes people hit on
#                                 this box are POWER spikes (OCP), not OOM
#
# And one anti-pattern: do NOT torch.compile the transformer. Triton doesn't emit
# working SASS for sm_121a yet, so it fails or silently falls back. The diffusers
# Wan docs recommend `torch.compile(mode="max-autotune")`; ignore that here.
_SPARK_ENV = {
    "CUDA_CACHE_MAXSIZE": "4294967296",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "EAGER",
}

_ENV_VARS = {"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"}

# The fetch task turns hf_transfer OFF on purpose. hf_transfer is a Rust downloader
# that does its own DNS, so it bypasses the IPv4 pin in fetch_weights; on a network
# with a black-holed IPv6 route to the HF CDN it hangs anyway. The plain Python
# downloader honors the pin, and HF_HUB_DOWNLOAD_TIMEOUT bounds a *stalled read*
# (per-read, not total), so a hung socket fails in ~60s and snapshot_download
# resumes from the .incomplete file rather than hanging forever.
_FETCH_ENV_VARS = {**_ENV_VARS, "HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_HUB_DOWNLOAD_TIMEOUT": "60"}

_GPU_ENV_VARS = {**_ENV_VARS, **_SPARK_ENV}


# ── Environments ────────────────────────────────────────────────────────────────
#
#   cpu_env   (videogen-fetch) : fetch_weights, a huge HF download, no GPU.
#   gpu_env   (videogen)       : generate_for_model / first frame, the GPU work.
#   orch_env  (videogen-orch)  : the orchestrators. CPU-only ON PURPOSE, because an
#                                orchestrator pod stays alive holding its resources
#                                while awaiting children, so if it asked for the GPU
#                                it would hold the box's only one and its own GPU
#                                child would deadlock on "Insufficient nvidia.com/gpu"
#                                forever.
#
# Cross-env calls need the caller to `depends_on` the callee's env or `flyte run`
# won't build the callee's image ("Environment '…' not found in image cache").

# disk=200Gi: an LTX-2.3 snapshot is ~95GB, and the task writes it to a temp dir
# and then uploads it, so headroom matters. Bump for wan22-t2v-a14b (126GB).
cpu_env = flyte.TaskEnvironment(
    name="videogen-fetch",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", disk="220Gi"),
    secrets=[HF_SECRET],
    env_vars=_FETCH_ENV_VARS,
)

# memory=96Gi: LTX-2.3 is a 22B transformer plus a Gemma-3 text encoder, and the
# VAE decode of a 121-frame latent is itself a large allocation. On the GB10 this
# is the single unified 119.7GiB pool shared with the OS, so this is close to the
# ceiling. If a pod goes Unschedulable, memory is the knob to turn down.
gpu_env = flyte.TaskEnvironment(
    name="videogen",
    image=image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="150Gi"),
    secrets=[HF_SECRET],
    env_vars=_GPU_ENV_VARS,
)

orch_env = flyte.TaskEnvironment(
    name="videogen-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    secrets=[HF_SECRET],
    env_vars=_ENV_VARS,
    depends_on=[cpu_env, gpu_env],
)

# AppEnvironment does NOT honor flyte.Resources(gpu=1) on this SDK: the serializer
# maps it to a bare `gpu` name that k8s drops, so the pod silently schedules CPU-only.
# The fix (verified in the magenta + imagegen demos) is a PodTemplate that sets
# nvidia.com/gpu directly, passing NO resources=. Unused while the studio is a pure
# launcher; kept for in-pod experiments.
app_gpu_pod = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                resources=V1ResourceRequirements(
                    requests={"cpu": "8", "memory": "96Gi", "ephemeral-storage": "150Gi"},
                    limits={"nvidia.com/gpu": "1"},
                ),
            )
        ]
    ),
)
