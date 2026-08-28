# Isaac Sim as a Flyte task image, arm64 / GB10.
#
# This exists as a Dockerfile rather than a chain of flyte.Image.with_*() calls for
# one specific reason: in flyte 2.2.1, `from_dockerfile()` is the ONLY constructor
# that accepts both an arbitrary base image and a `platform`. `from_base()` leaves
# the platform at its dataclass default of linux/amd64, `clone()` has no platform
# parameter in this version, and `dataclasses.replace` is blocked by the Image
# __post_init__. Building without it pushes an amd64 manifest to the registry and the
# kubelet rejects it with a message that does not say "wrong architecture":
#
#     failed to pull and unpack image ... no match for platform in manifest: not found
#
# See config.py for how this file is wired in, and README.md for the whole story.

FROM nvcr.io/nvidia/isaac-sim:6.0.0

# root only for the build. We drop back to the image's own uid at the end.
USER root

# ── Everything /isaac-sim/python.sh exports, made permanent ─────────────────────
#
# The container ships NO system python. Kit's embedded CPython 3.12 at
# /isaac-sim/kit/python/bin/python3 is the only interpreter, it is not on PATH, and
# running it directly does NOT find isaacsim:
#
#   /isaac-sim/kit/python/bin/python3 -c 'import isaacsim'  ->  ModuleNotFoundError
#   /isaac-sim/python.sh              -c 'import isaacsim'  ->  works
#
# python.sh is just a wrapper that exports these and execs that interpreter. Baking
# them in means Flyte's own process starts already able to `import isaacsim`, so task
# bodies hold live simulator objects instead of shelling out to python.sh.
#
# Re-extract these if the base tag changes; do not hand-edit them:
#   docker run --rm --entrypoint bash nvcr.io/nvidia/isaac-sim:<tag> -c \
#     '/isaac-sim/python.sh -c "import os; print(os.environ[\"PYTHONPATH\"])"'
#
# The pip_prebundle entries are load-bearing: several extensions vendor their Python
# dependencies there rather than in site-packages.
ENV PYTHONPATH=/isaac-sim/kit/python/lib/python3.12:/isaac-sim/kit/python/lib/python3.12/site-packages:/isaac-sim/python_packages:/isaac-sim/exts/isaacsim.simulation_app:/isaac-sim/kit/kernel/py:/isaac-sim/kit/plugins/bindings-python:/isaac-sim/exts/isaacsim.replicator.episode_recorder/pip_prebundle:/isaac-sim/exts/isaacsim.replicator.teleop/pip_prebundle:/isaac-sim/extsDeprecated/isaacsim.robot_motion.lula/pip_prebundle:/isaac-sim/exts/isaacsim.robot_motion.cumotion/pip_prebundle:/isaac-sim/exts/isaacsim.robot_motion.pink/pip_prebundle:/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle:/isaac-sim/extscache/omni.kit.pip_archive-0.0.0+6312fa25.la64.cp312/pip_prebundle:/isaac-sim/exts/omni.isaac.core_archive/pip_prebundle:/isaac-sim/exts/omni.pip.compute/pip_prebundle:/isaac-sim/exts/omni.pip.cloud/pip_prebundle

ENV LD_LIBRARY_PATH=/isaac-sim/.:/isaac-sim/exts/isaacsim.robot.schema/plugins/lib:/isaac-sim/extsDeprecated/isaacsim.robot_motion.lula/pip_prebundle:/isaac-sim/exts/isaacsim.robot_motion.cumotion/pip_prebundle:/isaac-sim/exts/isaacsim.robot_motion.pink/pip_prebundle:/isaac-sim/exts/isaacsim.asset.exporter.urdf/pip_prebundle:/isaac-sim/kit:/isaac-sim/kit/kernel/plugins:/isaac-sim/kit/libs/iray:/isaac-sim/kit/plugins:/isaac-sim/kit/plugins/bindings-python:/isaac-sim/kit/plugins/carb_gfx:/isaac-sim/kit/plugins/rtx:/isaac-sim/kit/plugins/gpu.foundation

# Kit's loader shim. Without it Kit's native plugins cannot resolve carb symbols at
# extension startup. This is process-wide, including pip and a0, exactly as under
# python.sh.
ENV LD_PRELOAD=/isaac-sim/kit/libcarb.so
ENV CARB_APP_PATH=/isaac-sim/kit
ENV EXP_PATH=/isaac-sim/apps
ENV ISAAC_PATH=/isaac-sim

# Kit's python first, so `python3` resolves at all, and so pip's console scripts
# (notably `a0`, which IS the Flyte task entrypoint) land somewhere on PATH.
ENV PATH=/isaac-sim/kit/python/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Non-interactive form of the license prompt setup.sh makes a human answer. The
# container refuses to start headless without them.
ENV ACCEPT_EULA=Y
ENV PRIVACY_CONSENT=Y
ENV OMNI_KIT_ACCEPT_EULA=YES

# Kit writes caches under $HOME. Point it at the writable WORKDIR rather than letting
# it try /root and fail on permissions once we drop back to uid 1234.
ENV HOME=/isaac-sim

# pip will warn that flyte's pydantic (2.13.4) and typing_extensions (4.16.0) violate
# isaaclab's declared pins. Measured on this box, nothing breaks: isaacsim imports,
# isaaclab enumerates its task registry, and the physics checks still pass. Those pins
# are metadata, not runtime constraints.
#
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
RUN python3 -m pip install --no-cache-dir "flyte==2.2.1" "connectrpc==0.10.*" \
 && ln -sf /isaac-sim/kit/python/bin/python3 /isaac-sim/kit/python/bin/python \
 && a0 --help > /dev/null

# The base sets ENTRYPOINT=/isaac-sim/runheadless.sh, which launches the simulator's
# streaming app. Flyte serialises tasks with command=[] and args=["a0", ...], so k8s
# would feed those args to runheadless.sh and the pod would do something surreal
# instead of running the task. Clearing it makes args the command, which is what
# Flyte's own base image does (ENTRYPOINT=[], CMD=[python3]).
ENTRYPOINT []
CMD ["python3"]

# Back to the image's own uid. WORKDIR /isaac-sim and Kit's site-packages are both
# writable by it, so Flyte can extract its code bundle and Kit can write caches.
USER isaac-sim

# ── The graphics half of the NVIDIA driver ──────────────────────────────────────
#
# Appended at the very END of the file on purpose: ENV is a zero-byte metadata layer,
# but it still invalidates everything below it, and below this point in Dockerfile.train
# there is ~25 GB of torch and Isaac Lab. Putting it last keeps the cache.
#
# /opt/nvgfx is where nvgfx.sh stages libGLX_nvidia and the rest of the `graphics`
# capability, and where train_env's pod_template bind-mounts them. Read the header of
# nvgfx.sh for why they are not simply injected like libcuda is; the one-line version is
# that the Flyte devbox container runs with NVIDIA_DRIVER_CAPABILITIES=compute,utility
# and k3s lives inside it, so no pod can widen that.
#
# It goes at the FRONT of the path so the ICD resolves `libGLX_nvidia.so.0` here, and it
# is harmless when the directory does not exist: the loader skips missing entries, so an
# image run without the mount behaves exactly as it did before.
ENV LD_LIBRARY_PATH=/opt/nvgfx:${LD_LIBRARY_PATH}
