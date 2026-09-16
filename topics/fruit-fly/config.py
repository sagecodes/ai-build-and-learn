"""Shared Flyte config for the connectome fly: a real fly's wiring, simulated, driving a body.

Same shape as topics/rl-mujoco and topics/dreamerv3 next door, so if you have read
either of those the layout is familiar:

  - connectome.py : the wiring diagram. FlyWire 783, 138,639 neurons, 15.1M edges.
  - brain.py      : the brain. A leaky integrate-and-fire network built FROM that
                    wiring, in Brian2, stepped 15 ms at a time.
  - body.py       : the body. NeuroMechFly v2 in MuJoCo, with compound eyes.
  - bridge.py     : the corpus callosum of this demo. Photoreceptor drive in one
                    direction, descending-neuron population rates back the other.
  - pipeline.py   : the Flyte orchestrators.
  - app.py        : a thin CPU Gradio launcher.

── This demo is CPU-ONLY, on purpose ───────────────────────────────────────────
Nothing here asks for the GPU, and that is not a limitation, it is the point. The
brain is a sparse event-driven spiking network (Brian2, Cython codegen) and the body
is a single fly in CPU MuJoCo. Measured on this box:

    brain   138,639 neurons / 15,091,983 edges   build 1.9 s, run 0.32x realtime
    body    NeuroMechFly v2, timestep 1e-4       2,900 steps/s = 0.29x realtime
    eyes    2 x 721 ommatidia                    2 ms per readout after warmup

A 3-second behaviour therefore costs about 30-40 seconds end to end, which is why the
Gradio app can afford to launch a run while you watch. It also means these runs never
contend with a training job for the Spark's single GPU, and the orchestrator-deadlock
trap from the videogen and rl-mujoco demos cannot fire here at all.

── The connectome data is BAKED INTO THE IMAGE ─────────────────────────────────
131 MB of it, pinned to two upstream commits. Downloading it per-pod instead would
put a network dependency inside a job that should have none, and would re-fetch the
same 100 MB parquet on every single run. See `_CONNECTOME_FETCH` below.
"""

from __future__ import annotations

import flyte
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# Runs land in `physical-ai`, the same Flyte project as rl-mujoco and isaac-sim: it is
# the box's home for things with a body. app.py reads FLYTE_PROJECT to override.
PROJECT = "physical-ai"
DOMAIN = "development"

APP_NAME = "connectome-fly-studio"
APP_PORT = 7865          # 7862 image-gen, 7863 videogen, 7864 g1-walk; don't collide


# ── Where the wiring diagram comes from ─────────────────────────────────────────
#
# Two upstream repos, both pinned to a commit so a silent re-release upstream cannot
# change what the brain is:
#
# 1. philshiu/Drosophila_brain_model - the FlyWire 783 connectivity matrix in the exact
#    form Shiu et al. 2024 (Nature) used for their leaky integrate-and-fire model:
#    a completeness table (which neurons exist) and an edge list carrying
#    `Excitatory x Connectivity`, the synapse count signed by the presynaptic
#    neurotransmitter. That signed count IS the synaptic weight; there is no training.
#
# 2. flyconnectome/flywire_annotations - Schlegel et al. 2024 (Nature), the whole-brain
#    cell typing. This is what lets us say "the sugar-sensing gustatory neurons" or
#    "DNa02, left" instead of a 19-digit root ID. Without it the connectome is 138,639
#    anonymous nodes.
#
# The two are joined on FlyWire root_id: 138,625 of the model's 138,639 neurons carry
# an annotation (99.99%).
_BRAIN_REPO = "philshiu/Drosophila_brain_model"
_BRAIN_COMMIT = "91bdd1e7dcf193f3e7ca5a8933497fcef63b7960"
_ANNOT_REPO = "flyconnectome/flywire_annotations"
_ANNOT_COMMIT = "8587524c1748ce5ef2080822a2fc890fc03bf597"

CONNECTOME_DIR = "/opt/connectome"

_CONNECTOME_FETCH = [
    f"mkdir -p {CONNECTOME_DIR}",
    f"curl -sSLf -o {CONNECTOME_DIR}/completeness.csv "
    f"https://raw.githubusercontent.com/{_BRAIN_REPO}/{_BRAIN_COMMIT}/Completeness_783.csv",
    f"curl -sSLf -o {CONNECTOME_DIR}/connectivity.parquet "
    f"https://raw.githubusercontent.com/{_BRAIN_REPO}/{_BRAIN_COMMIT}/Connectivity_783.parquet",
    f"curl -sSLf -o {CONNECTOME_DIR}/annotations.tsv "
    f"https://raw.githubusercontent.com/{_ANNOT_REPO}/{_ANNOT_COMMIT}/"
    f"supplemental_files/Supplemental_file1_neuron_annotations.tsv",
    # Fail the BUILD, not the run, if a URL rots. A 404 saved as an HTML error page
    # would otherwise sail through and die much later inside pandas.
    f"test $(stat -c%s {CONNECTOME_DIR}/connectivity.parquet) -gt 90000000",
    f"test $(stat -c%s {CONNECTOME_DIR}/annotations.tsv) -gt 30000000",
]


# ── Rendering ───────────────────────────────────────────────────────────────────
#
# MuJoCo's renderer needs a GL context and a pod has no display, so EGL is the only
# thing that works headless. Same env as the rl-mujoco render task, which is the proof
# that this works in a CPU pod on this devbox: the flyte devbox runs with
# NVIDIA_DRIVER_CAPABILITIES=compute,utility and injects no graphics driver, so EGL
# here resolves to Mesa's software device from `libegl-mesa0` + `libgl1-mesa-dri`.
# That is also why `libosmesa6` is in the apt list: MUJOCO_GL=osmesa is the fallback
# if a future devbox image drops the Mesa EGL platform.
_RENDER_ENV = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}

# Brian2 generates C++ for the integration loop and compiles it with Cython on first
# run. Measured here: 12.3 s for that first compile, then 0.3 s per 100 ms of brain.
# The cache directory must be writable and, more importantly, must NOT be shared
# between concurrently-starting pods, which is the default ($HOME/.cython) and races.
_BRIAN_ENV = {
    "CYTHON_CACHE_DIR": "/tmp/cython-brian",
    # Brian2's own "you have N cores" advisory scan shells out on import; pointless
    # in a pod with a cpu limit, and noisy in the logs.
    "BRIAN2_NO_SPEED_TEST": "1",
    # The brain is one sparse event loop, not a BLAS workload. Letting numpy fan out
    # across every core here buys nothing and fights the physics thread.
    "OMP_NUM_THREADS": "2",
    "OPENBLAS_NUM_THREADS": "2",
}

# The GL runtime libs are apt packages, not pip: `mujoco.Renderer` dlopens libEGL at
# runtime and fails with a bare ImportError if they are missing. Copied verbatim from
# topics/rl-mujoco/config.py, where it is known good on this box.
_GL_APT = (
    "libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri",
    "libgles2", "libglx-mesa0", "libosmesa6",
)

# g++ is not optional here. Brian2's default codegen target is Cython, which compiles
# the generated C++ AT RUNTIME inside the pod. Without a compiler Brian2 silently falls
# back to its numpy target, which is roughly 30x slower for a network this sparse and
# would turn a 30-second run into a quarter of an hour.
_BUILD_APT = ("git", "curl", "ffmpeg", "build-essential")

FLY_SPEC = (
    # flygym 2.1.0 = NeuroMechFly v2. Pinned exactly: the v2 rewrite (March 2026) is
    # not API-compatible with v1, and `flygym_demo.complex_terrain` (where the walking
    # controllers live) is an examples package with no stability promise at all.
    "flygym[warp]==2.1.0",
    # flygym pins mujoco>=3.9,<3.10 itself; naming it here keeps the pod and the
    # devbox venv honest about which one actually got installed.
    "mujoco==3.9.0",
    # Brian2 2.10.1 is the first release with cp312 aarch64 wheels. Older versions
    # build from source on this box, which works but adds minutes to every image.
    "brian2==2.10.1",
    "pandas",
    "pyarrow",          # the connectivity edge list is parquet
    "numpy",
    "scipy",
    "Pillow",
    "matplotlib",
    # PyAV encodes the replay mp4. Same reasoning as the videogen and rl-mujoco demos:
    # PyAV ships manylinux aarch64 wheels. flygym drags in imageio-ffmpeg==0.6.0 as a
    # hard dependency and that one does publish an aarch64 wheel, so both are present;
    # `body.py` writes frames through PyAV so the encode path matches the other demos.
    "av",
    "imageio",
    # config.py imports kubernetes.client at module top for the app pod template, and
    # task pods import config too, so every image needs it.
    "kubernetes",
)


def _fly_image(name: str) -> flyte.Image:
    return (
        flyte.Image.from_debian_base(name=name, registry=REGISTRY, platform=PLATFORM)
        .with_apt_packages(*_BUILD_APT, *_GL_APT)
        .with_pip_packages(*FLY_SPEC)
        .with_commands(_CONNECTOME_FETCH)
    )


image = _fly_image("connectome-fly-image")

# The studio app is a LAUNCHER: it submits runs and links reports, so it needs neither
# the connectome nor mujoco. Keeping it tiny means the app pod starts in seconds and
# never holds 131 MB of wiring diagram resident for the life of the app.
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
studio_app_image = (
    flyte.Image.from_debian_base(
        name="connectome-fly-studio-image", registry=REGISTRY, platform=PLATFORM
    )
    .with_pip_packages("flyte==2.2.1", "connectrpc==0.10.*", "gradio==5.42.0", "python-dotenv")
)


# ── Environments ────────────────────────────────────────────────────────────────
#
#   brain_env (fly-brain) : probe_brain and embodied_run. Holds the whole connectome.
#   orch_env  (fly-orch)  : the orchestrator. Tiny, because it only awaits children.
#
# Cross-env calls need the caller to `depends_on` the callee's env or `flyte run` won't
# build the callee's image ("Environment '…' not found in image cache").
#
# memory=24Gi: measured peak RSS for the full network is 3.2 GB, and the edge list
# costs another ~1 GB transiently while pandas materialises the two index columns.
# 24Gi is roughly 4x headroom, which matters because `probe_brain` builds a FRESH
# network per condition in the sweep and Brian2's teardown is garbage-collected rather
# than immediate, so two networks can briefly co-exist.
brain_env = flyte.TaskEnvironment(
    name="fly-brain",
    image=image,
    resources=flyte.Resources(cpu="8", memory="24Gi", disk="30Gi"),
    env_vars={**_RENDER_ENV, **_BRIAN_ENV},
)

orch_env = flyte.TaskEnvironment(
    name="fly-orch",
    image=image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    depends_on=[brain_env],
)

# AppEnvironment does NOT honor flyte.Resources(gpu=1) on this SDK: the serializer maps
# it to a bare `gpu` name that k8s drops, so the pod silently schedules CPU-only. The
# fix (verified in the magenta + imagegen + videogen demos) is a PodTemplate that sets
# nvidia.com/gpu directly, passing NO resources=. This demo never wants the GPU, so
# the template here is CPU-only and exists purely to size the launcher pod.
app_pod = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                resources=V1ResourceRequirements(
                    requests={"cpu": "2", "memory": "4Gi", "ephemeral-storage": "10Gi"},
                ),
            )
        ]
    ),
)
