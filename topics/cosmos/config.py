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
from kubernetes.client import (
    V1Container,
    V1HostPathVolumeSource,
    V1PodSpec,
    V1Volume,
    V1VolumeMount,
)

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

# ── The shared model cache, and why it is a hostPath ─────────────────────────────
#
# Cosmos3-Nano is 33 GB on disk. With a per-pod HF_HOME every task in this file
# re-downloads all of it before it can do anything, which costs more wall clock than
# the generation and is also the exact shape of the failure recorded in
# `reference_flyte_devbox_disk_eviction`: a big model fetch onto an already-full disk
# evicted the whole cluster, control plane included.
#
# So the weights are staged ONCE into the devbox and every pod mounts them:
#
#     docker exec flyte-devbox mkdir -p /var/lib/kubelet/hf-cache/hub
#     docker cp ~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano \
#         flyte-devbox:/var/lib/kubelet/hf-cache/hub/
#     docker exec flyte-devbox chmod -R a+rwX /var/lib/kubelet/hf-cache
#
# The path is a hostPath INSIDE THE DEVBOX, which is the part that is easy to get
# wrong. k3s runs inside the `flyte-devbox` container, so a task pod's hostPath
# resolves against that container's filesystem and not against the real host: a
# hostPath of /home/sage/.cache/huggingface mounts an empty directory and every task
# silently re-downloads. /var/lib/kubelet is a docker volume, so what is staged there
# survives `flyte stop devbox` / `flyte start devbox --gpu`.
#
# Mounted READ-WRITE on purpose. huggingface_hub takes .lock files inside the cache
# even when every blob is already present, so a read-only mount fails the resolve
# rather than serving from cache; rw also means the first pod to want a model this
# box has never seen populates the cache for every pod after it.
HF_CACHE_HOSTPATH = "/var/lib/kubelet/hf-cache"
HF_CACHE_MOUNT = "/mnt/hf"

# Off switch for a cluster that is not this devbox, where the hostPath does not exist
# and a pod that mounts it would get an empty directory:
#
#     COSMOS_SHARED_CACHE=0 flyte run pipeline.py imagine
USE_SHARED_CACHE = os.environ.get("COSMOS_SHARED_CACHE", "1").lower() not in ("0", "false", "no")

_cache_volume = V1Volume(
    name="hf-cache",
    host_path=V1HostPathVolumeSource(path=HF_CACHE_HOSTPATH, type="DirectoryOrCreate"),
)

# The container MUST be named "primary": that is the name Flyte looks for when it
# merges the task's image, command and `resources=` into the template (see
# `_get_k8s_pod` in the SDK). Resources are deliberately NOT set here -- the merge
# lets the TaskEnvironment's own Resources(gpu=1, ...) win, so the GPU request keeps
# working exactly as it did before this template existed.
hf_cache_pod = flyte.PodTemplate(
    primary_container_name="primary",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="primary",
                volume_mounts=[V1VolumeMount(name="hf-cache", mount_path=HF_CACHE_MOUNT)],
            )
        ],
        volumes=[_cache_volume],
    ),
)

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


# ── Why these are pinned exactly, and what it cost to learn ─────────────────────
#
# These used to be `diffusers>=0.39.0` and `transformers>=5.11`, floors rather than
# pins, on the reasonable-sounding grounds that a floor documents the real requirement
# and lets bug fixes in. What it actually did was let the pod's software drift away
# from the host's. Measured mid-2026: the venv `setup.sh` builds had torch 2.13.0,
# transformers 5.14.1 and diffusers 0.39.0, while an image rebuilt from the same file
# resolved to torch 2.14.0, transformers 5.16.1 and diffusers 0.40.0.
#
# That is not an abstract hygiene problem. It makes `smoke_test.py` a liar. Its whole
# claim is "if this passes on the host and the Flyte run does not, the problem is the
# pod" -- which only holds if the two are running the same software. The symptom that
# exposed it: the same greedy, do_sample=False planning question, on the same image,
# with the same weights, answered with a four-step decomposition on the host and a
# single sentence in the pod. Nothing in this repo had changed.
#
# So: exact pins, and they are the versions the results in the README were measured on.
# Bump them deliberately, rebuild, and re-run smoke_test.py plus one short task rather
# than letting a resolver decide on your behalf at 2am.
COSMOS_SPEC = (
    # Cosmos3OmniPipeline / CosmosActionCondition do not exist before 0.39.0, and the
    # model cards that tell you to install diffusers from git are stale as of it.
    "diffusers==0.40.0",
    # Cosmos 3's vision tower is Qwen3VLVisionModel (Qwen3-VL lands in 5.11), and
    # Cosmos3OmniForConditionalGeneration -- the understanding surface -- ships in the
    # same line. 5.16.1 is what the reasoning tasks were measured against.
    "transformers==5.16.1",
    "accelerate>=1.10",
    "safetensors",
    "sentencepiece",
    "protobuf",
    "pillow",
    "numpy",
    "huggingface_hub",
    # LeRobot datasets keep their non-pixel columns in parquet, which is how the
    # embodiment survey gets real actions to sit beside the real frames.
    "pyarrow",
    "hf_transfer",
    # PyAV encodes the report mp4 and is what diffusers.utils.encode_video needs to
    # mux Cosmos 3's generated audio track. aarch64 wheels exist; imageio-ffmpeg's
    # do not reliably, the same conclusion the video-generation demo reached.
    "av",
    "imageio",
    # Canny edge maps for Cosmos Transfer. The headless build on purpose: the default
    # opencv-python pulls a GUI stack that a pod has no use for and cannot start.
    "opencv-python-headless",
    # Cosmos Transfer REQUIRES its guardrail and there is no opting out: unlike
    # Cosmos3OmniPipeline, which takes enable_safety_checker=False (and whose own NVIDIA
    # runner exposes --disable-safety-checker), Cosmos2_5_TransferPipeline RAISES if the
    # checker is None and cites the NVIDIA Open Model License in the message. So it gets
    # installed rather than stubbed out. It pulls nvidia/Cosmos-1.0-Guardrail (gated) and
    # google/siglip-so400m-patch14-384 (open, 3.5 GB) at first use.
    "cosmos_guardrail",
    # nltk 3.9.1, and this one is load-bearing. cosmos_guardrail asks for nltk>=3.9.1 and
    # resolves to 3.10.x, which added `nltk.pathsec`: a hardening layer that REFUSES to
    # open a file through a symlink. The guardrail ships its blocklist tokenizer data
    # inside its HF snapshot, and a Hugging Face cache is a symlink farm (snapshots point
    # at blobs), so every guardrail load dies with
    #   Security Violation [pathsec.open]: refusing to follow a symlink at open time
    #   OSError: [Errno 40] Too many levels of symbolic links
    # There is no opt-out: NLTK_ALLOW_PROXIED_URLOPEN governs URLs, not symlinks. 3.9.1
    # predates pathsec entirely and still satisfies the guardrail's own floor.
    "nltk==3.9.1",
    # The simulator that supplies `restyle` its input, and its actions. Plain mujoco,
    # not mujoco_playground/brax/jax: the scene is a handful of primitives.
    "mujoco",
    "flyte==2.2.1",
    # 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
    "connectrpc==0.10.*",
)

image = (
    flyte.Image.from_debian_base(name="cosmos3", registry=REGISTRY, platform=PLATFORM)
    # The mesa/EGL set is what lets MuJoCo render headless in a pod, lifted from
    # topics/rl-mujoco which worked it out the hard way: MUJOCO_GL=egl needs real GL
    # libraries present, and a pod that has CUDA does NOT automatically have them.
    .with_apt_packages("git", "ffmpeg", "libegl1", "libegl-mesa0", "libgl1",
                       "libgl1-mesa-dri", "libgles2", "libglx-mesa0", "libosmesa6")
    # torch on its OWN layer and from the cu130 index, before anything else can
    # resolve a plain-PyPI torch over the top of it.
    .with_pip_packages("torch", "torchvision", index_url=TORCH_INDEX)
    .with_pip_packages(*COSMOS_SPEC)
    # Its OWN layer, and last, so adding it does not invalidate the (slow) layer
    # above. config.py builds a PodTemplate out of kubernetes.client models and is
    # imported inside the pod as well as on the host, so the package has to exist in
    # both places even though the template itself is only ever read at serialization.
    .with_pip_packages("kubernetes")
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
# MuJoCo renders through EGL in the pod. Set here rather than in mjc.py so it is true
# before anything imports OpenGL, which caches its platform at import time.
_RENDER_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}

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
    # The shared hostPath cache when it is mounted, this pod's own scratch otherwise.
    "HF_HOME": HF_CACHE_MOUNT if USE_SHARED_CACHE else HF_HOME,
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_HUB_DOWNLOAD_TIMEOUT": "60",
}

_GPU_ENV_VARS = {**_ENV_VARS, **_SPARK_ENV, **_RENDER_ENV}


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
# disk=120Gi: generous now rather than necessary. With the shared cache mounted the
# weights are not on the pod's own ephemeral disk at all, so this only has to cover
# the image and scratch. It is left high because the number that matters is what
# happens when the cache is NOT there (COSMOS_SHARED_CACHE=0), where a 33 GB snapshot
# does land in /tmp/hf and a pod that runs out of ephemeral storage is evicted rather
# than told why.
gpu_env = flyte.TaskEnvironment(
    name="cosmos",
    image=image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="120Gi"),
    secrets=SECRETS,
    env_vars=_GPU_ENV_VARS,
    pod_template=hf_cache_pod if USE_SHARED_CACHE else None,
)

orch_env = flyte.TaskEnvironment(
    name="cosmos-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    secrets=SECRETS,
    env_vars=_ENV_VARS,
    depends_on=[gpu_env],
)
