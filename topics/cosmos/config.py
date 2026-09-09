"""Shared Flyte config for the NVIDIA Cosmos 3 world-model demo.

Runs to the `world-models` Flyte project, the same one as topics/dreamerv3, because
the two events are a pair: DreamerV3 learns a tiny world model *of one environment
from its own experience*, Cosmos 3 is a pretrained world model *of the physical
world* you condition and roll forward. The robotics demos stay in `physical-ai`.

DGX-Spark-pinned (GB10 Blackwell, arm64, cu130): aarch64 platform + the devbox-local
registry. Drop the pins for a generic Flyte 2 cluster.

── Why diffusers and not NVIDIA's own runner ───────────────────────────────────
NVIDIA's cookbook sets Cosmos 3 up with `cosmos-framework` and a single `uv sync`.
That path resolves natten and friends, which have no aarch64 wheels and build from
source against CUDA 13, and it is the same shape of problem that made the Isaac Sim
demo give up on `pip install isaacsim` entirely.

diffusers 0.39.0 ships the whole model natively: `Cosmos3OmniPipeline`,
`Cosmos3OmniTransformer`, the Wan VAE, the audio tokenizer, and
`CosmosActionCondition`. `transformer_cosmos3.py` routes attention through
diffusers' `dispatch_attention_fn`, so torch SDPA is enough and no CUDA extension
is involved. That turns this into an ordinary Debian-base + pip image, which is the
same shape as the video-generation demo and builds in minutes.

── Why the safety checker is off ───────────────────────────────────────────────
`Cosmos3OmniPipeline.from_pretrained(...)` defaults to `enable_safety_checker=True`,
which constructs a `CosmosSafetyChecker` from the separate `cosmos_guardrail`
package. That is a second model download and it pulls a GATED Llama Guard
checkpoint, so on a box whose HF token has not accepted that licence the pipeline
fails at *construction*, long before it generates anything. NVIDIA's own example
runner exposes `--disable-safety-checker` for exactly this. Every task here passes
`enable_safety_checker=False`, which means these runs have no content guardrail on
prompt or output: fine for a demo you drive yourself, not fine for anything that
takes prompts from someone else.
"""

from __future__ import annotations

import os

import flyte

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# GB10 is Blackwell (sm_121) on the cu130 stack. PyTorch publishes matching aarch64
# wheels only on this index; the plain-PyPI aarch64 wheel is CPU-only and the sole
# symptom of getting it is torch.cuda.is_available() == False at generation time.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

# The checkpoints. All three are ungated (verified against the HF API), so no licence
# click-through is needed, but HF_TOKEN still goes into the pods: unauthenticated
# pulls of a 35 GB repo are the ones that get rate-limited.
NANO = "nvidia/Cosmos3-Nano"    # 16B, ~35 GB on disk, ~30 GB resident in BF16
SUPER = "nvidia/Cosmos3-Super"  # 64B, does not fit this box; here for reference
EDGE = "nvidia/Cosmos3-Edge"    # 4B, no video-to-video transfer and no sound

HF_HOME = "/tmp/hf"
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")

# OPT-IN, and off by default. All three Cosmos 3 repos are ungated, so the token buys
# rate limits rather than access, and declaring a secret the cluster does not hold
# fails the pod at ADMISSION: the webhook denies it before any container starts, so
# the run dies in under a second with a message about secret managers and nothing in
# the logs. That is a bad trade for an optimisation.
#
# Turn it back on once `flyte create secret HF_TOKEN` has been run:
#
#     COSMOS_HF_SECRET=1 flyte run pipeline.py invert
#
# Without it the fetch is unauthenticated and can be throttled, which is survivable
# here: _ENV_VARS already disables hf_transfer and bounds a stalled read at 60s, so a
# throttled pull resumes from its .incomplete file rather than hanging the task.
USE_HF_SECRET = os.environ.get("COSMOS_HF_SECRET", "").lower() in ("1", "true", "yes")
SECRETS = [HF_SECRET] if USE_HF_SECRET else []


COSMOS_SPEC = (
    # Hard floor. Cosmos3OmniPipeline / CosmosActionCondition do not exist before
    # 0.39.0, and the model cards that tell you to install diffusers from git are
    # stale as of that release.
    "diffusers>=0.39.0",
    # Cosmos 3's vision tower is Qwen3VLVisionModel; Qwen3-VL lands in transformers 5.11.
    "transformers>=5.11",
    "accelerate>=1.10",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "pillow",
    "numpy",
    "huggingface_hub",
    "hf_transfer",
    # PyAV encodes the report mp4 and is what diffusers.utils.encode_video needs to
    # mux Cosmos 3's generated audio track. aarch64 wheels exist; imageio-ffmpeg's
    # do not reliably, the same conclusion the video-generation demo reached.
    "av",
    "imageio",
    "flyte==2.2.1",
    # 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
    "connectrpc==0.10.*",
)

image = (
    flyte.Image.from_debian_base(name="cosmos3", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("git", "ffmpeg")
    # torch on its OWN layer and from the cu130 index, before anything else can
    # resolve a plain-PyPI torch over the top of it.
    .with_pip_packages("torch", "torchvision", index_url=TORCH_INDEX)
    .with_pip_packages(*COSMOS_SPEC)
)


# ── DGX Spark tuning ────────────────────────────────────────────────────────────
#
# Carried over from the video-generation demo, where each of these was measured
# rather than guessed. Cosmos 3 is a diffusion transformer doing the same work.
#
#   CUDA_CACHE_MAXSIZE   The PTX->SASS JIT cache. A diffusion transformer JITs a lot
#                        of kernels on first use; with the default (tiny) cache they
#                        get evicted and re-JIT-ed every denoise step.
#   PYTORCH_ALLOC_CONF   expandable_segments lets the allocator grow and shrink
#                        segments instead of reserving fixed blocks. It matters more
#                        here than on a discrete GPU, because "GPU memory" IS the
#                        same unified 119.7 GiB the OS and every other pod share.
#                        (PYTORCH_CUDA_ALLOC_CONF is the older name; set both.)
#   CUDA_MODULE_LOADING  EAGER: load modules up front rather than lazily mid-step.
#
# Anti-pattern, same as next door: do NOT torch.compile the transformer. Triton does
# not emit working SASS for sm_121a yet, so it fails or silently falls back.
_SPARK_ENV = {
    "CUDA_CACHE_MAXSIZE": "4294967296",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "EAGER",
}

# hf_transfer is OFF. It is a Rust downloader that does its own DNS and ignores
# socket timeouts, so a black-holed route to the HF CDN hangs forever instead of
# erroring. The plain Python downloader plus HF_HUB_DOWNLOAD_TIMEOUT (which bounds a
# stalled *read*, not the total) fails in ~60s and resumes from the .incomplete file.
_ENV_VARS = {
    "HF_HOME": HF_HOME,
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_HUB_DOWNLOAD_TIMEOUT": "60",
}

_GPU_ENV_VARS = {**_ENV_VARS, **_SPARK_ENV}


# ── Environments ────────────────────────────────────────────────────────────────
#
# One GPU on this box, and the orchestrator is CPU-only ON PURPOSE: an orchestrator
# pod holds its resources for as long as its children run, so a GPU-holding
# orchestrator deadlocks its own GPU child on "Insufficient nvidia.com/gpu". The
# same trap as the videogen, mujoco, dreamer and Isaac Sim demos.
#
# memory=96Gi: the transformer is 16B in BF16 (~30 GB resident) and the VAE decode of
# a multi-second latent is itself a large allocation. On the GB10 that is the single
# unified 119.7 GiB pool shared with the OS and every other pod, so this sits close
# to the ceiling. If a pod goes Unschedulable, this is the knob to turn down.
#
# disk=120Gi: the snapshot is ~35 GB and it lands in the pod's own /tmp/hf, because
# there is no shared model cache across pods here. That is also why every task in
# pipeline.py loads the pipeline ONCE and generates every clip it needs from it: the
# download is the expensive part, not the denoising.
gpu_env = flyte.TaskEnvironment(
    name="cosmos",
    image=image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="120Gi"),
    secrets=SECRETS,
    env_vars=_GPU_ENV_VARS,
)

orch_env = flyte.TaskEnvironment(
    name="cosmos-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    secrets=SECRETS,
    env_vars=_ENV_VARS,
    depends_on=[gpu_env],
)
