"""Shared Flyte config for the DreamerV3 world-model demo.

Same shape as the MuJoCo demo next door (topics/rl-mujoco), which is deliberate: both
train a locomotion policy on MuJoCo physics on this one GB10, so the closer the two
setups look, the more honest the comparison between model-free PPO and model-based
Dreamer is.

Runs to the `world-models` Flyte project, which is new and will also house the Cosmos
and V-JEPA 2 events. The robotics demos stay in `physical-ai`.

── The two things that make this image different from upstream ─────────────────
Upstream pins `jax[cuda12]==0.4.33`. The GB10 is Blackwell, sm_121, and CUDA 12 builds
of that vintage have no kernels for it, so the pinned stack cannot use this GPU at all.
This uses `jax[cuda13]==0.9.2`, the version topics/rl-mujoco already proved here.

Moving forward five minor versions costs one incompatibility: jax made everything
after `fun` keyword-only in `jax.jit`, and upstream still calls it positionally in six
places. `patches/0001-jax-jit-keyword-only.patch` fixes exactly that, no behaviour
change, and is baked into the image at build time against a pinned upstream commit.
Measured on this box before any of this was containerised: the agent initialises on
cuda:0 (640,867 params), XLA compiles, and the reward loss falls 5.53 -> 0.58 over
4,736 steps of dmc_walker_walk.
"""

from __future__ import annotations

from pathlib import Path

import flyte

REGISTRY = "localhost:30000"
# A TUPLE, not a string. flyte passes this straight through to `docker buildx build
# --platform`, and a bare string is iterated character by character into
# `l,i,n,u,x,/,a,r,m,6,4`, which fails with `"l": unknown operating system`.
PLATFORM = ("linux/arm64",)

# Upstream, pinned. The patch is line-addressed against this commit, so a bump here
# without re-testing the patch will fail the image build rather than silently skew.
DREAMER_REPO = "https://github.com/danijar/dreamerv3"
DREAMER_COMMIT = "e3f02248693a79dc8b0ebd62c93683888ddaccfe"
DREAMER_ROOT = "/opt/dreamerv3"

# GL runtime libs are apt packages, not pip: dm_control dlopens libEGL at runtime and
# fails with a bare ImportError if they are missing. Same list as the MuJoCo demo.
#
# Rendering in a pod additionally needs the NVIDIA *graphics* driver libraries, which
# the flyte devbox does not inject by default (it runs with
# NVIDIA_DRIVER_CAPABILITIES=compute,utility). That is the bug that made every Isaac
# Sim replay come back as 300 black frames; see topics/isaac-sim/README.md. Proprio
# training needs none of it, but a pixel-space Dreamer run will.
GL_APT = (
    "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri",
    "libgles2", "libglx-mesa0", "libosmesa6",
)

SPEC = (
    # See the module docstring. cuda13, and no numpy<2 cap: upstream's own comment
    # says that cap exists for DMLab and MineRL, neither of which we run, and jax
    # 0.9.2 requires numpy>=2 anyway.
    "jax[cuda13]==0.9.2",
    # The agent and Danijar's supporting libraries, all of which dreamerv3 imports.
    "ninjax", "elements", "portal", "scope", "granular",
    "optax", "einops", "chex", "jaxtyping", "colored_traceback", "ruamel.yaml", "tqdm",
    # DeepMind Control Suite, which runs on MuJoCo. Same engine as topics/rl-mujoco.
    "dm_control", "mujoco",
    # PyAV for the replay mp4: aarch64 wheels exist, imageio-ffmpeg's do not reliably.
    # scope encodes its video columns with av too, and its image columns with pillow,
    # which is what scopevid.py reads back out to put the dream in the report.
    "av", "imageio", "pillow", "numpy",
    "flyte==2.2.1",
    # 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
    "connectrpc==0.10.*",
)

image = (
    flyte.Image.from_debian_base(name="dreamerv3", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("git", "ffmpeg", *GL_APT)
    .with_pip_packages(*SPEC)
    # The patch has to exist inside the build context before it can be applied.
    .with_source_file(
        Path(__file__).parent / "patches" / "0001-jax-jit-keyword-only.patch",
        "/opt/dv3.patch",
    )
    # Clone at the pinned commit and patch it in the image, so no task pod ever clones
    # at runtime and every pod runs byte-identical agent code. `git apply --check`
    # first: if a future commit bump breaks the patch, the BUILD fails loudly instead
    # of producing an image that dies at agent init.
    .with_commands([
        f"git clone {DREAMER_REPO} {DREAMER_ROOT}",
        f"git -C {DREAMER_ROOT} checkout {DREAMER_COMMIT}",
        f"git -C {DREAMER_ROOT} apply --check /opt/dv3.patch",
        f"git -C {DREAMER_ROOT} apply /opt/dv3.patch",
    ])
    # PYTHONPATH: dreamerv3 is a source checkout, not a pip install, so the tasks find
    # it the same way the host setup does.
    .with_env_vars({
        "PYTHONPATH": DREAMER_ROOT,
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
    })
)


# ── Environments ────────────────────────────────────────────────────────────────
#
# One GPU on this box, and the orchestrator is CPU-only ON PURPOSE: an orchestrator
# pod holds its resources for as long as its children run, so a GPU-holding
# orchestrator deadlocks its own GPU child on "Insufficient nvidia.com/gpu". The same
# trap as the videogen, mujoco and Isaac Sim demos.
#
# Dreamer is famously light on GPU memory: the whole agent is 10.5M parameters at the
# size12m preset used for pixels, and the replay buffer lives in host RAM. The memory
# ask here is for the replay buffer, not the model.
#
# A pixel run stores 64x64x3 frames instead of a state vector, so the buffer is roughly
# 12 KB per transition against 100 bytes. `replay.size` defaults to 5e6, far more than
# a 500k-step run needs, and Dreamer only allocates what it fills: measured at 1.7 GB
# after 12k steps, so ~70 GB at 500k. `--replay.size` is capped in pipeline.py instead
# of provisioning for a buffer the run will never fill.
gpu_env = flyte.TaskEnvironment(
    name="dreamer",
    image=image,
    resources=flyte.Resources(cpu="12", memory="80Gi", gpu=1, disk="80Gi"),
)

orch_env = flyte.TaskEnvironment(
    name="dreamer-orch",
    image=image,
    resources=flyte.Resources(cpu="1", memory="4Gi"),
    depends_on=[gpu_env],
)
