"""Shared Flyte config for the open-source image-generation demo.

Two runtimes share one image + one model registry (models.py):
  - compare_pipeline.py : GPU *tasks* that generate a prompt grid across models
    and emit a side-by-side Flyte report.
  - app.py              : a thin CPU Gradio *app* that launches compare runs and
    links the report (no GPU, loads no model).

DGX-Spark-pinned (GB10 Blackwell, arm64, cu130 stack): aarch64 platform +
devbox-local registry. Drop the pins for a generic Flyte 2 cluster.

── The one thing to verify first (same shape as the magenta demo) ──────────────
PyTorch + diffusers on **arm64 (sbsa) + CUDA 13** is the install risk. We build
on `from_debian_base` and pull torch from the cu130 wheel index. If those aarch64
wheels don't resolve, switch `_image` to a CUDA/PyTorch base image instead:

    _base = flyte.Image.from_base("nvcr.io/nvidia/pytorch:25.xx-py3")
    object.__setattr__(_base, "platform", PLATFORM)
    image = _base.clone(registry=REGISTRY, name="imagegen-image").with_pip_packages(
        "diffusers", "transformers", "accelerate", ...)   # torch already present

(That's exactly how gemma4/vllm_server.py builds on a vLLM base.) A CPU smoke
test — drop the torch index + `.to("cuda")` — proves download + pipeline load
before fighting CUDA.
"""

from __future__ import annotations

import os

import flyte
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# GB10 is Blackwell on the cu130 stack. PyTorch publishes matching aarch64 wheels
# under this index. Bump to a newer cuNNN if you move to a different CUDA.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

APP_NAME = "imagegen-studio"
APP_PORT = 7862

# HF cache lives on the (large) ephemeral disk so re-pulls across tasks are warm
# within a pod and weights never bloat the image.
HF_HOME = "/tmp/hf"

# Weights are pulled at runtime from HuggingFace; gated repos (SD3.5, FLUX.1/2
# dev) need an HF_TOKEN that accepted the license. Same secret the other demos
# use; harmless for the open-only default set.
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")


# ── Python deps ─────────────────────────────────────────────────────────────────
#
# diffusers/transformers/accelerate is the core; peft is for the LoRA trainer;
# sentencepiece + protobuf are needed by the T5 tokenizers in FLUX/SD3/Qwen;
# datasets is for the LoRA training data.
DIFFUSERS_SPEC = (
    "diffusers>=0.32",
    "transformers>=4.44",
    "accelerate>=0.34",
    "peft>=0.13",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "pillow",
    "huggingface_hub",
    "datasets>=3.0",
    # hf_transfer is HF's Rust downloader; enabling HF_HUB_ENABLE_HF_TRANSFER
    # parallelizes chunk downloads and saturates bandwidth far better than the
    # default python client (the multi-GB weight pulls are the slow part).
    "hf_transfer",
    # config.py imports kubernetes.client at module top (for the app's GPU
    # pod_template). Task pods import config too, so both images need it.
    "kubernetes",
)


def _torch_image(name: str, extra: tuple[str, ...] = ()) -> flyte.Image:
    """Debian base + cu130 torch + diffusers stack, plus any extra pip packages."""
    return (
        flyte.Image.from_debian_base(name=name, registry=REGISTRY, platform=PLATFORM)
        .with_apt_packages("git")  # some HF repos vendor code fetched via git
        .with_pip_packages(
            "torch",
            "torchvision",
            index_url=TORCH_INDEX,
        )
        .with_pip_packages(*DIFFUSERS_SPEC, *extra)
    )


# Task image (compare_pipeline.py, lora_finetune.py).
image = _torch_image("imagegen-image")

# The studio app is a thin LAUNCHER: it submits `compare` runs and links the
# report, so it needs no torch/diffusers, just flyte + gradio and the model
# registry (models.py, bundled in app.py) for the model picker. That keeps the
# app pod tiny and, crucially, means it never loads a model or holds GPU memory.
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
studio_app_image = (
    flyte.Image.from_debian_base(name="imagegen-studio-image", registry=REGISTRY, platform=PLATFORM)
    .with_pip_packages("flyte==2.2.1", "connectrpc==0.10.*", "gradio==5.42.0", "python-dotenv")
)

# Legacy heavyweight app image + GPU pod template from when the studio generated
# in-pod. The studio is now a launcher (studio_app_image above), so these are
# unused; kept only for lora_finetune-style in-pod experiments. Safe to delete.
app_image = _torch_image("imagegen-app-image", extra=("gradio==5.42.0",))


# ── GPU: tasks vs apps ──────────────────────────────────────────────────────────
#
# TaskEnvironment honors `flyte.Resources(gpu=1)` (see maze-rl). AppEnvironment
# does NOT — on this SDK the app serializer maps the GPU resource to a bare `gpu`
# name that k8s drops, so the pod schedules CPU-only. The magenta demo verified
# the fix: hand the app a PodTemplate whose primary "app" container sets
# `nvidia.com/gpu` directly, and pass NO `resources=` (which would re-introduce
# the broken mapping). Reused verbatim here.

# Three environments, split by what they need so nothing hogs the GPU:
#
#   cpu_task_env  (imagegen-fetch) : fetch_weights — a big HF download, no GPU.
#   gpu_task_env  (imagegen)       : generate_for_model / LoRA — the only GPU work.
#   orch_env      (imagegen-orch)  : the orchestrators (compare / generate_one /
#                                    lora_demo). CPU-only ON PURPOSE: an
#                                    orchestrator pod stays alive holding its
#                                    resources while it awaits child tasks, so if
#                                    it requested a GPU it would hold the box's
#                                    only GPU and its GPU child would deadlock
#                                    ("Insufficient nvidia.com/gpu" forever).
#
# Cross-env calls need the caller to `depends_on` the callee's env, or `flyte
# run` won't build the callee image ("Environment '…' not found in image cache").
# fetch/generate are leaves (no depends_on); orch calls both.

_ENV_VARS = {"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"}

# The fetch task turns hf_transfer OFF on purpose. hf_transfer is a Rust
# downloader that does its own DNS, so it bypasses the IPv4 pin in fetch_weights;
# on a network with a black-holed IPv6 route to the HF CDN it still hangs. The
# plain Python downloader honors the IPv4 pin, and HF_HUB_DOWNLOAD_TIMEOUT bounds
# a *stalled read* (per-read, not total time) so a hung socket fails in ~60s and
# snapshot_download resumes from the .incomplete instead of hanging forever. Big
# models can take an hour of healthy downloading; only a lack of progress trips.
_FETCH_ENV_VARS = {**_ENV_VARS, "HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_HUB_DOWNLOAD_TIMEOUT": "60"}

cpu_task_env = flyte.TaskEnvironment(
    name="imagegen-fetch",
    image=image,
    resources=flyte.Resources(cpu="4", memory="16Gi", disk="120Gi"),
    secrets=[HF_SECRET],
    env_vars=_FETCH_ENV_VARS,
)

# expandable_segments lets PyTorch's caching allocator grow/shrink segments
# instead of reserving fixed blocks, which cuts fragmentation-driven OOM. It
# matters more on the GB10: "GPU memory" is the same unified 128GB the OS, page
# cache, and other pods share (nvidia-smi even reports usage as "Not Supported"),
# so a model can transiently fail to allocate under memory pressure that a
# discrete GPU would never see. Paired with per-task retries below.
_GPU_ENV_VARS = {**_ENV_VARS, "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}

gpu_task_env = flyte.TaskEnvironment(
    name="imagegen",
    image=image,
    resources=flyte.Resources(cpu="8", memory="48Gi", gpu=1, disk="80Gi"),
    secrets=[HF_SECRET],
    env_vars=_GPU_ENV_VARS,
)

orch_env = flyte.TaskEnvironment(
    name="imagegen-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    secrets=[HF_SECRET],
    env_vars=_ENV_VARS,
    depends_on=[cpu_task_env, gpu_task_env],
)

app_gpu_pod = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                resources=V1ResourceRequirements(
                    requests={"cpu": "8", "memory": "48Gi", "ephemeral-storage": "80Gi"},
                    limits={"nvidia.com/gpu": "1"},
                ),
            )
        ]
    ),
)
