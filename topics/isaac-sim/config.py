"""Shared Flyte config for the Isaac Sim demo, on a DGX Spark (GB10, arm64, CUDA 13).

Same shape as topics/rl-mujoco next door, with one structural difference that drives
everything here: we do NOT build from a Debian base and pip-install a simulator into
it. Isaac Sim has no aarch64 wheels. The only thing that exists is NVIDIA's container,
so the task image starts FROM that and adds Flyte to it.

── Why the container and not the source build ──────────────────────────────────
`pip install isaacsim` does not work on aarch64 at all: there is no wheel. The
supported paths on GB10 are a source build (see setup.sh, ~11 GB of build tree) or
NVIDIA's container. Baking an 11 GB build tree into a task image is the wrong shape,
and `nvcr.io/nvidia/isaac-sim:6.0.0` publishes a real arm64 manifest. It pulls
anonymously: no NGC login, no imagePullSecret.

The bare-metal build is still worth having. It is the fast iteration loop (no image
rebuild between edits) and it is the control number that `smoke_test.py` produces to
compare against the containerised one from `pipeline.py`.

── What makes this image non-obvious ───────────────────────────────────────────
THE CONTAINER HAS NO SYSTEM PYTHON, and Kit's embedded one does not find isaacsim
unless python.sh's environment is set. All of that is handled in the Dockerfile,
which is commented line by line; read it before changing anything here.

Verified on this box 2026-08-05: flyte 2.2.1, isaacsim 6.0.1 and isaaclab 6.1.17 all
import in one interpreter, and the physics checks still pass afterwards.
"""

from __future__ import annotations

from pathlib import Path

import flyte

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# Pinned deliberately. This tag is what was verified end to end; "latest" on a 17.6 GB
# pull is not something you want silently changing under a demo.
BASE_IMAGE = "nvcr.io/nvidia/isaac-sim:6.0.0"


# ── Why a Dockerfile and not a chain of with_*() layers ─────────────────────────
#
# The obvious spelling is from_base(...).with_env_vars(...).with_pip_packages(...).
# It does not work in flyte 2.2.1, and it fails twice, in this order:
#
# 1. `Cannot add additional layers to a non-extendable image.` from_base() hands back
#    a sealed image; you have to .clone(..., extendable=True) it first.
#
# 2. Then, once it builds, the kubelet refuses it:
#       no match for platform in manifest sha256:...: not found
#    which does not say "wrong architecture" but means exactly that. from_base()
#    leaves platform at the Image dataclass default of ("linux/amd64",), 2.2.1's
#    clone() takes no platform parameter (newer SDK checkouts DO, so do not copy that
#    signature out of the source tree), and dataclasses.replace is refused by the
#    Image __post_init__.
#
# from_dockerfile() is the only constructor in this version that takes an arbitrary
# base AND a platform, so that is what this uses. The cost is that it cannot carry
# additional layers, which is why every ENV and the pip install live in the Dockerfile
# rather than here.
#
# Paths must be absolute: Python resolves them from the calling directory, and the
# build context is wherever the Dockerfile lives.
image = flyte.Image.from_dockerfile(
    file=Path(__file__).parent / "Dockerfile",
    registry=REGISTRY,
    name="isaac-sim-flyte",
    platform=PLATFORM,
)

# The training image, ~25 GB. Separate from the one above because the NGC container
# ships Isaac Sim and NOTHING else: no torch, no Isaac Lab, no RL libraries. Verified
# by looking, not assuming (`import torch` -> ModuleNotFoundError inside the base).
# Adding ~8 GB of torch and Isaac Lab to the smoke-test image would make every
# plumbing check pull a quarter of the disk, so the two stay separate.
train_image = flyte.Image.from_dockerfile(
    file=Path(__file__).parent / "Dockerfile.train",
    registry=REGISTRY,
    name="isaac-lab-flyte",
    platform=PLATFORM,
)


# ── Environments ────────────────────────────────────────────────────────────────
#
# One GPU on this box. The orchestrator is CPU-only ON PURPOSE: an orchestrator pod
# holds its resources for as long as its children run, so a GPU-holding orchestrator
# deadlocks its own GPU child on "Insufficient nvidia.com/gpu". Same trap as the
# videogen and mujoco demos; cheap to avoid, expensive to debug.
#
# disk=60Gi because the image alone is 17.6 GB and Kit unpacks shader and extension
# caches on top of that on first run.
gpu_env = flyte.TaskEnvironment(
    name="isaac-sim",
    image=image,
    resources=flyte.Resources(cpu="8", memory="48Gi", gpu=1, disk="60Gi"),
)

# RL training. Measured on the host: Anymal-C flat at 4096 envs runs ~1.0s/iteration
# and 1500 iterations (a walking policy) takes 27 minutes. Memory is dominated by the
# rollout buffer, which scales with num_envs.
#
# disk=80Gi: the image is ~25 GB before Kit unpacks its shader and extension caches,
# and rsl_rl writes a checkpoint every 50 iterations on top of that.
train_env = flyte.TaskEnvironment(
    name="isaac-train",
    image=train_image,
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu=1, disk="80Gi"),
    # ── Why these are listed here and not imported ──────────────────────────────
    #
    # `flyte run` bundles code with copy_style="loaded_modules" by default
    # (_run.py:98): it walks sys.modules after importing the task file and ships only
    # what got loaded. The usual way to make a sibling module reach the pod is
    # therefore to import it at the top of pipeline.py, and that is what the note in
    # the smoke-test section of pipeline.py describes.
    #
    # It cannot work for these three. terrains.py and spark_envs.py import
    # `isaaclab.terrains` at module level, and pipeline.py is imported by BOTH task
    # environments: the orchestrator runs on the plain isaac-sim image, which has no
    # Isaac Lab in it at all. A top-level import would ship the files and break every
    # orchestrator pod with ModuleNotFoundError before it could schedule anything.
    #
    # `include` is the supported way out. It is unioned into whatever the copy style
    # discovered (_run.py:266 -> additional_files), so these ride along without anyone
    # importing them, and they are only ever imported inside the training pod's child
    # process, which does have Isaac Lab. Paths are relative to this file.
    #
    # record.py is in the list even though its top level is import-safe, because
    # nothing imports it either: it is spawned by path, as a script.
    include=("record.py", "spark_envs.py", "terrains.py"),
)

orch_env = flyte.TaskEnvironment(
    name="isaac-orch",
    image=image,
    resources=flyte.Resources(cpu="1", memory="4Gi"),
    depends_on=[gpu_env, train_env],
)
