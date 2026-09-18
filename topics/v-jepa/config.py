"""Shared Flyte config for the V-JEPA 2 demo.

Runs to the `world-models` Flyte project, alongside topics/cosmos and topics/dreamerv3,
because all three are the same question asked three ways. Cosmos predicts the future in
PIXELS. DreamerV3 learns a latent world model of one small environment from its own
experience. V-JEPA 2 is the third answer: predict in REPRESENTATION space, learned
self-supervised from internet video, with no decoder anywhere in the model.

That last detail drives the whole demo. There is no pixel head to render, so "show me
what it predicted" is not a screenshot you can take. Everything visual here is either
the model's actual input (which we can show exactly) or a per-patch measurement painted
back onto that input.

DGX-Spark-pinned (GB10 Blackwell, arm64, cu130): aarch64 platform + the devbox-local
registry. Drop the pins for a generic Flyte 2 cluster.

── Why this image is boring, and that is the point ─────────────────────────────
V-JEPA 2 landed in Hugging Face Transformers proper (`VJEPA2Model`), so there is no
vendor runner, no CUDA extension, and nothing to build from source. Compare with
topics/isaac-sim, which had to give up on `pip install isaacsim` entirely, and
topics/cosmos, which had to route around natten. This is a plain Debian base plus pip,
and it builds in minutes.
"""

from __future__ import annotations

from pathlib import Path

import flyte

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# GB10 is Blackwell (sm_121) on the cu130 stack. PyTorch publishes matching aarch64
# wheels only on this index; the plain-PyPI aarch64 wheel is CPU-only and the sole
# symptom of getting it is torch.cuda.is_available() == False at encode time.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

# ── Checkpoints ─────────────────────────────────────────────────────────────────
#
# All ungated (verified against the HF API), so no licence click-through. HF_TOKEN
# still goes into the pods because unauthenticated pulls are the ones that get
# rate-limited.
#
# Every one of these is a PRETRAINED checkpoint: encoder + predictor, no task head.
# That is what we want, because the demo is about what self-supervision alone buys.
VITL = "facebook/vjepa2-vitl-fpc64-256"  # ViT-L/16,  326M, hidden 1024, 24 layers
VITH = "facebook/vjepa2-vith-fpc64-256"  # ViT-H/16,  632M, hidden 1280, 32 layers
VITG = "facebook/vjepa2-vitg-fpc64-256"  # ViT-g/16, 1035M, hidden 1408, 40 layers

# Not used here, and worth knowing why. V-JEPA 2-AC, the action-conditioned post-train
# that actually plans robot manipulation, is NOT on the Hub under facebook/ (checked:
# only the six pretrain + SSv2/Diving48 classifier repos exist). Its weights ship via
# the facebookresearch/vjepa2 repo and it has no `transformers` class. So the honest
# scope of this demo is the pretrained predictor, and `inpaint` measures exactly where
# that predictor stops being a world model. See the README.
AC_NOTE = "facebookresearch/vjepa2 (not on the Hub, no transformers class)"

# 5 classes x 20 clips, ~10s each at 15fps, train/ and val/ splits of 10 each.
# Ungated, tiny, and already the dataset the transformers V-JEPA 2 docs use.
CLIPS_REPO = "nateraw/kinetics-mini"

HF_HOME = "/tmp/hf"
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")


VJEPA_SPEC = (
    # VJEPA2Model / VJEPA2Predictor and AutoVideoProcessor's fast video path.
    "transformers>=5.11",
    "accelerate>=1.10",
    "safetensors",
    "numpy",
    "pillow",
    "huggingface_hub",
    # PyAV decodes the source clips and encodes the report mp4s. aarch64 wheels exist
    # for PyAV and do not reliably for imageio-ffmpeg, which is the conclusion the
    # video-generation, Cosmos and Isaac Sim demos all reached independently.
    "av",
    "matplotlib",
    "flyte==2.2.1",
    # 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
    "connectrpc==0.10.*",
    # ── the V-JEPA 2-AC half (ac.py, sim.py, plan.py) ──────────────────────────
    # timm and einops are what upstream's `src/models` import; there is no xformers
    # dependency despite the `vit_giant_xformers` arch name.
    "timm",
    "einops",
    "scipy",
    # The MuJoCo Franka that the world model plans in.
    # Pinned to what walk.py was verified against: the G1 stack was built on 3.11 in
    # topics/rl-mujoco, and 3.13 was checked to load and walk the same checkpoint.
    "mujoco==3.13.0",
    # The G1 humanoid (walker.py). jax is CPU-only on purpose: the GPU belongs to
    # V-JEPA, and a CPU jax wheel brings no CUDA libs to fight torch's. 0.9.2 because
    # brax 0.14.2 still calls jax.device_put_replicated, removed in 0.10.
    "jax==0.9.2",
    "mujoco-mjx==3.13.0",
    "brax",
    "playground",
    # Not a flyte dependency, and config.py is imported inside the pod too, so it
    # has to be in the image as well as the host venv (see the pod template below).
    "kubernetes",
)

# ── V-JEPA 2-AC source, pinned ──────────────────────────────────────────────────
#
# Cloned rather than pip-installed, because `pip install git+...vjepa2` installs
# nothing importable: its setup.py declares no `packages=` and no `py_modules=`, so
# pip reports success and `import src` still fails. Pinned to a commit because main
# ships a committed bug -- `VJEPA_BASE_URL = "http://localhost:8300"` in
# src/hub/backbones.py -- and we would rather find out about the next one at build
# time than mid-run. ac.py never calls the hub entrypoint anyway; it imports
# `src.models.*` and loads the checkpoint itself.
VJEPA2_REPO = "https://github.com/facebookresearch/vjepa2"
VJEPA2_SHA = "204698b45b3712590f06245fbfba32d3be539812"
VJEPA2_SRC = "/opt/vjepa2"

# Only the Panda is needed; menagerie in full is a few hundred MB of robots we will
# never load, so sparse-checkout one directory.
MENAGERIE_REPO = "https://github.com/google-deepmind/mujoco_menagerie"
MENAGERIE = "/opt/menagerie"

# MuJoCo's renderer dlopens libEGL at runtime, so the GL stack is apt packages and
# not pip. Same list as topics/rl-mujoco and topics/fruit-fly: the devbox sets
# NVIDIA_DRIVER_CAPABILITIES=compute,utility and injects no graphics driver, so EGL
# resolves to Mesa's software device. libosmesa6 is the fallback if a future devbox
# image drops the Mesa EGL platform (MUJOCO_GL=osmesa).
_GL_APT = (
    "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri",
    "libglx-mesa0", "libosmesa6", "libglib2.0-0",
)

# ── GPU rendering in pods ───────────────────────────────────────────────────────
#
# Pods get the NVIDIA compute driver only, so MuJoCo's EGL falls back to Mesa's software
# rasteriser: 3.4 fps against 580 on the GPU (measured, 256x256 G1). `./nvgfx.sh` stages
# the NVIDIA EGL libraries into nvgfx/ (gitignored) and this layers them into an image.
# Skipped when nvgfx/ is absent, which includes inside every pod: images are resolved
# from the launch-time image cache there, never rebuilt, so the spec can differ.
NVGFX = Path(__file__).resolve().parent / "nvgfx"


def _with_nvgfx(img: flyte.Image) -> flyte.Image:
    if not (NVGFX / "driver-version.txt").exists():
        return img
    return img.with_source_folder(NVGFX, "/opt/nvgfx").with_commands([
        "echo /opt/nvgfx > /etc/ld.so.conf.d/00-nvgfx.conf && ldconfig",
        "mkdir -p /usr/share/glvnd/egl_vendor.d && cp /opt/nvgfx/10_nvidia.json /usr/share/glvnd/egl_vendor.d/",
    ])


image = (
    flyte.Image.from_debian_base(name="vjepa2", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("git", "ffmpeg", *_GL_APT)
    # torch on its OWN layer and from the cu130 index, before anything else can
    # resolve a plain-PyPI torch over the top of it.
    .with_pip_packages("torch", "torchvision", index_url=TORCH_INDEX)
    .with_pip_packages(*VJEPA_SPEC)
    .with_commands(
        [
            f"git clone --filter=blob:none {VJEPA2_REPO} {VJEPA2_SRC}",
            f"git -C {VJEPA2_SRC} checkout --quiet {VJEPA2_SHA}",
            f"git clone --depth 1 --filter=blob:none --sparse {MENAGERIE_REPO} {MENAGERIE}",
            f"git -C {MENAGERIE} sparse-checkout set franka_emika_panda",
            # Playground reads the G1 XMLs from inside its OWN package directory and
            # otherwise re-clones menagerie in every pod (topics/rl-mujoco/config.py).
            "python -c 'from mujoco_playground._src import mjx_env; "
            "mjx_env.ensure_menagerie_exists()'",
        ]
    )
)
image = _with_nvgfx(image)


# ── DGX Spark tuning ────────────────────────────────────────────────────────────
#
# Carried over from the video-generation and Cosmos demos. Less load-bearing here than
# there: ViT-g peaks at 2.1 GiB, so this box is nowhere near its ceiling. Kept anyway
# because expandable_segments still matters when "GPU memory" IS the same unified
# 119.7 GiB pool the OS and every other pod share.
#
# Anti-pattern, same as next door: do NOT torch.compile the encoder. Triton does not
# emit working SASS for sm_121a yet, so it fails or silently falls back.
_SPARK_ENV = {
    "CUDA_CACHE_MAXSIZE": "4294967296",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "EAGER",
}

# hf_transfer is OFF. It is a Rust downloader that does its own DNS and ignores socket
# timeouts, so a black-holed route to the HF CDN hangs forever instead of erroring. The
# plain Python downloader plus HF_HUB_DOWNLOAD_TIMEOUT (which bounds a stalled *read*,
# not the total) fails in ~60s and resumes from the .incomplete file.
_ENV_VARS = {
    "HF_HOME": HF_HOME,
    "HF_HUB_ENABLE_HF_TRANSFER": "0",
    "HF_HUB_DOWNLOAD_TIMEOUT": "60",
    # Matplotlib writes a font cache on first import and $HOME is not writable in the pod.
    "MPLCONFIGDIR": "/tmp/mpl",
    # Where ac.py and sim.py find the cloned sources.
    "VJEPA2_SRC": VJEPA2_SRC,
    "MENAGERIE": MENAGERIE,
    # Headless MuJoCo. Read before `import mujoco`, which is why sim.py sets these
    # defaults too: a task that imports mujoco through some other path first would
    # otherwise get the default backend and fail with no display.
    "MUJOCO_GL": "egl",
    "PYOPENGL_PLATFORM": "egl",
}

_GPU_ENV_VARS = {**_ENV_VARS, **_SPARK_ENV, "JAX_PLATFORMS": "cpu"}


# ── The 11.7 GB checkpoint ──────────────────────────────────────────────────────
#
# V-JEPA 2-AC is not on the Hub in any form the torch code can load: the only
# distribution is a direct URL from the vjepa2 README. Downloading it per pod would
# dominate the run (the planning itself is a couple of minutes), so it is staged
# into the devbox once and hostPath-mounted, exactly as topics/cosmos does for
# Cosmos3-Nano. `./setup.sh --stage` does the staging; ac.py falls back to
# downloading if the mount is empty, so a fresh cluster still works, slowly.
#
# THE TRAP: k3s runs INSIDE the `flyte-devbox` container, so a pod's hostPath
# resolves against that container's filesystem and NOT the real host. Staging to
# ~/models on the host mounts an empty directory and every task silently
# re-downloads. /var/lib/kubelet is a docker volume, so it also survives
# `flyte stop devbox`.
AC_STAGE_HOST = "/var/lib/kubelet/hf-cache/vjepa2"
AC_MOUNT = "/mnt/hf/vjepa2"

try:
    from kubernetes.client import (
        V1Container,
        V1HostPathVolumeSource,
        V1PodSpec,
        V1Volume,
        V1VolumeMount,
    )

    # No resources here on purpose: the SDK MERGES the TaskEnvironment's
    # Resources(...) into the template's primary container, task keys winning, so
    # gpu=1 keeps working. Setting them here instead would silently drop the GPU.
    _ac_pod = flyte.PodTemplate(
        primary_container_name="primary",  # must be exactly "primary"
        pod_spec=V1PodSpec(
            containers=[
                V1Container(
                    name="primary",
                    volume_mounts=[V1VolumeMount(name="ac-cache", mount_path=AC_MOUNT)],
                )
            ],
            volumes=[
                V1Volume(
                    name="ac-cache",
                    host_path=V1HostPathVolumeSource(
                        path=AC_STAGE_HOST, type="DirectoryOrCreate"
                    ),
                )
            ],
        ),
    )
except ImportError:  # kubernetes missing on the host venv; the pods still have it
    _ac_pod = None


# ── Environments ────────────────────────────────────────────────────────────────
#
# One GPU on this box, and the orchestrator is CPU-only ON PURPOSE: an orchestrator pod
# holds its resources for as long as its children run, so a GPU-holding orchestrator
# deadlocks its own GPU child on "Insufficient nvidia.com/gpu". Same trap as the
# cosmos, videogen, mujoco, dreamer and Isaac Sim demos.
#
# memory=32Gi / disk=60Gi is generous for what this actually does. The largest
# checkpoint is ~4.4 GB on disk and 2.1 GiB resident, and the whole kinetics-mini
# dataset is under 200 MB. The headroom is for the 100-clip feature matrix and for
# decoding, not for the model.
gpu_env = flyte.TaskEnvironment(
    name="vjepa",
    image=image,
    resources=flyte.Resources(cpu="8", memory="32Gi", gpu=1, disk="60Gi"),
    secrets=[HF_SECRET],
    env_vars=_GPU_ENV_VARS,
)

# ── MJX on the GPU (selfwalk_train.py) ──────────────────────────────────────────
#
# `selfwalk` trains a G1 policy from scratch with Brax PPO, which needs jax ON THE GPU
# (the vjepa image's jax is CPU-only so it cannot fight torch). This is topics/rl-mujoco's
# image recipe, verified there at 200M+ steps, plus this repo's flyte/connectrpc pins.
# No torch in it: the trainer never touches V-JEPA, it reads a distilled reward net.
MJX_SPEC = (
    "jax[cuda13]==0.9.2",     # cuda13 for this driver; 0.9.x because brax 0.14.2 needs device_put_replicated
    "mujoco==3.13.0",
    "mujoco-mjx==3.13.0",
    "brax",
    "playground",
    "numpy",
    "pillow",
    "matplotlib",
    "av",
    "flyte==2.2.1",
    "connectrpc==0.10.*",
    "kubernetes",
)

mjx_image = (
    flyte.Image.from_debian_base(name="vjepa-mjx", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("git", "ffmpeg", *_GL_APT)
    .with_pip_packages(*MJX_SPEC)
    .with_commands([
        "python -c 'from mujoco_playground._src import mjx_env; "
        "mjx_env.ensure_menagerie_exists()'",
    ])
)
mjx_image = _with_nvgfx(mjx_image)

# Same numbers as rl-mujoco's g1-train env: PREALLOCATE=false is mandatory on the
# unified pool, and 96Gi is what 4096-8192 envs were verified at.
mjx_env = flyte.TaskEnvironment(
    name="vjepa-mjx",
    image=mjx_image,
    resources=flyte.Resources(cpu="8", memory="96Gi", gpu=1, disk="50Gi"),
    env_vars={
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.80",
        "CUDA_CACHE_MAXSIZE": "4294967296",
        "CUDA_MODULE_LOADING": "EAGER",
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "MPLCONFIGDIR": "/tmp/mpl",
    },
)

orch_env = flyte.TaskEnvironment(
    name="vjepa-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    secrets=[HF_SECRET],
    env_vars=_ENV_VARS,
    depends_on=[gpu_env, mjx_env],
)


# The action-conditioned tasks (`dream`, `plan`) get their own environment: the AC
# checkpoint mount, more memory for the frame bank, and the same single GPU.
#
# ViT-g encoder + AC predictor is 1.3B parameters and ~6.7 GiB resident at a CEM
# batch of 125. memory=48Gi is for the host side -- an 11.7 GB checkpoint is read
# into CPU RAM before any of it reaches the device, and the frame bank holds a few
# hundred renders at two resolutions.
ac_env = flyte.TaskEnvironment(
    name="vjepa-ac",
    image=image,
    resources=flyte.Resources(cpu="8", memory="48Gi", gpu=1, disk="80Gi"),
    secrets=[HF_SECRET],
    env_vars=_GPU_ENV_VARS,
    **({"pod_template": _ac_pod} if _ac_pod is not None else {}),
)
