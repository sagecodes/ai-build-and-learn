"""Shared Flyte config for the Grokking Deep RL chapter examples.

Chapters 2-7 are tabular: numpy tables a few hundred cells big. Chapter 8+ add small
neural networks (torch) and gymnasium worlds. One image serves all of them: the pipeline
imports every chapter at the top (so `flyte run` bundles them), so every chapter's
imports must exist in it. The pip torch wheel for linux/arm64 is already the CUDA 13
build, so the same image will drive the GPU chapters later.
"""

from __future__ import annotations

import flyte

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# Pinned to what the host venv resolved, so a host run and a pod run match.
# box2d-py (LunarLander, chapter 9+) has no arm64 wheel: it builds from source, which
# needs swig and a C++ compiler in the image. LunarLander also imports pygame at module
# load even when nothing is rendered, so it has to be installed too.
# Chapter 12 adds MuJoCo (HalfCheetah) and PyAV (its videos, PyAV because imageio-ffmpeg
# has no reliable arm64 wheel). Pods have no GPU EGL, so MuJoCo renders in software
# with OSMesa; those GL libraries are apt packages that mujoco dlopens at render time.
SPEC = ("numpy", "matplotlib", "torch==2.14.1", "gymnasium[mujoco]==1.4.0", "box2d-py==2.3.8",
        "pygame==2.6.1", "mujoco==3.17.0", "av==18.1.0")
_GL_APT = ("libosmesa6", "libgl1", "libegl1", "libglx-mesa0", "libgl1-mesa-dri")

tabular_image = (
    flyte.Image.from_debian_base(name="grokking-rl", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("swig", "build-essential", *_GL_APT)
    .with_pip_packages(*SPEC)
)

# One chapter task. The tabular chapters are single-threaded Python loops and the
# networks are tiny; chapters 9-10 train their learners in parallel, one per core
# (chapter 10 runs 4 learners x 16 seeds).
#
# The pod sees all 20 host cores but its quota is 4, so every thread pool that sizes
# itself from the core count (OpenBLAS, torch's intra-op pool) spawns ~20 threads per
# process. With 4 learner processes that was 40 threads each and 26 minutes of CFS
# throttling: chapter 9 ran 3x slower than on the host. One thread per pool, set here
# so the spawned learner processes inherit it.
_ONE_THREAD = {k: "1" for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}

_ENV_VARS = {**_ONE_THREAD, "MUJOCO_GL": "osmesa", "PYOPENGL_PLATFORM": "osmesa"}

# Chapters 9-12 train many learners side by side, one per core.
cpu_env = flyte.TaskEnvironment(
    name="grokking-chapter",
    image=tabular_image,
    resources=flyte.Resources(cpu="8", memory="16Gi"),
    env_vars=_ENV_VARS,
)

# Chapters 2-8 are single-process. Asking for 8 cores each made them queue behind the
# heavy chapters in a `book` run (the node fits two 8-core pods at a time).
light_env = flyte.TaskEnvironment(
    name="grokking-chapter-light",
    image=tabular_image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    env_vars=_ENV_VARS,
)

# The orchestrator only awaits its children and stitches their reports together.
# `depends_on` so `flyte run` builds the chapter image too.
orch_env = flyte.TaskEnvironment(
    name="grokking-orch",
    image=tabular_image,
    resources=flyte.Resources(cpu="1", memory="1Gi"),
    depends_on=[cpu_env, light_env],
)
