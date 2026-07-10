"""Shared config for the Magenta RealTime 2 live-music app on Flyte 2.

One GPU app: it downloads the open weights from HuggingFace at startup, loads
Magenta RT2 (JAX backend) on the GPU, and serves a Gradio UI that streams a
continuous track you steer live with text prompts.

Reality check for this box (DGX Spark, NVIDIA GB10, arm64):
  - Sub-200ms *real-time* streaming is Apple-Silicon / MLX only (see the model
    card). On NVIDIA we run the JAX path: chunked generation streamed
    near-real-time. The live-tone-shift loop (swap the style embedding between
    chunks) is identical; it just isn't guaranteed faster-than-playback.
  - `mrt2_small` (230M) leaves GPU headroom for the phase-2 Gemma-4 VLM to
    share the box. `mrt2_base` (2.4B) sounds better but is heavier.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import flyte
import flyte.app
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements


REGISTRY = "localhost:30000"
PLATFORM = ("linux/arm64",)

APP_NAME = "magenta-rt-live"
APP_PORT = 7870


@dataclass(frozen=True)
class ModelChoice:
    size: str          # mrt2_small | mrt2_base; selects the safetensors checkpoint
    checkpoint: str    # filename under the repo's checkpoints/ dir


MRT2_SMALL = ModelChoice(size="mrt2_small", checkpoint="mrt2_small.safetensors")
MRT2_BASE = ModelChoice(size="mrt2_base", checkpoint="mrt2_base.safetensors")

MODEL = MRT2_BASE if os.environ.get("MRT_SIZE") == "base" else MRT2_SMALL


# ── Image ─────────────────────────────────────────────────────────────────────
#
# magenta-rt[jax] pulls jax/flax/optax + the vendored sequence-layers. We add
# jax[cuda12] for GPU execution and gradio/hf-hub/soundfile for the app.
#
# NOTE: JAX-on-CUDA over arm64 (sbsa) is the one thing to verify on the first
# `python mrt_app.py`. If pip can't resolve the aarch64 CUDA plugin wheels on a
# plain debian base, swap `from_debian_base` for a CUDA base image (e.g.
# nvidia/cuda:12.x-cudnn-runtime) the way the vLLM app uses a vllm base; the
# rest of this file is unchanged. A CPU smoke test works with `magenta-rt[jax]`
# alone (slow, but proves download + load + UI before fighting CUDA).

mrt_image = (
    flyte.Image.from_debian_base(
        name="magenta-rt-image",
        registry=REGISTRY,
        platform=PLATFORM,
    )
    .with_apt_packages("libsndfile1", "ffmpeg")
    .with_pip_packages(
        "magenta-rt[jax]",
        # Spark = GB10 Blackwell on the cu130 stack, so the cuda13 extra. If the
        # aarch64 CUDA pip wheels don't resolve on the debian base, switch to a
        # CUDA-13 base image + "jax[cuda13-local]" (uses the image's CUDA).
        "jax[cuda13]",
        "gradio==5.42.0",
        "huggingface_hub",
        "soundfile",
        "numpy",
    )
)


# ── GPU via pod_template (not resources=) ─────────────────────────────────────
#
# On this Flyte SDK, `flyte.Resources(gpu=1)` on a plain AppEnvironment does NOT
# render `nvidia.com/gpu` into the Knative revision (the app serializer maps the
# GPU resource to a bare `gpu` name that k8s drops), so the pod schedules with no
# GPU and JAX falls back to CPU. Verified against the live revision:
#   spec.containers[0].resources == {"requests":{"cpu","memory","ephemeral-storage"}}
# Workaround: hand the app a PodTemplate whose primary container ("app") sets
# `nvidia.com/gpu` directly. When pod_template is set, the GPU-dropping container
# path is bypassed entirely (app_serde: pod_template XOR container), and the app
# serializer still propagates our image, env_vars, and port into it. We must NOT
# also pass `resources=` here, or it overwrites the container's resources with
# the same broken bare-`gpu` mapping.

_gpu_pod = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                resources=V1ResourceRequirements(
                    requests={"cpu": "8", "memory": "32Gi", "ephemeral-storage": "40Gi"},
                    limits={"nvidia.com/gpu": "1"},
                ),
            )
        ]
    ),
)


# ── App env ───────────────────────────────────────────────────────────────────
#
# The weights repo (google/magenta-realtime-2) is public (CC-BY-4.0, not gated),
# so no HF_TOKEN is required to download them; if you ever point this at a gated
# repo, add `secrets=[flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")]` and
# the app will pass it through. MAGENTA_HOME points the library's path resolver
# at a writable dir; the server downloads weights there on startup
# (ensure_weights).

env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image=mrt_image,
    pod_template=_gpu_pod,
    port=APP_PORT,
    requires_auth=False,
    env_vars={
        "MAGENTA_HOME": "/tmp/magenta",
        # Propagate a few knobs from the deploy environment if set.
        **{
            k: os.environ[k]
            for k in ("MRT_SIZE", "GRADIO_SHARE", "MRT_FRAMES_PER_CHUNK")
            if k in os.environ
        },
    },
    parameters=[
        flyte.app.Parameter(name="model_size", value=MODEL.size),
        flyte.app.Parameter(name="checkpoint", value=MODEL.checkpoint),
    ],
    # Single always-warm replica while in use; scale to zero after 15 min idle.
    # (Cold start re-downloads weights + JIT-compiles, so don't scale down fast.)
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
)
