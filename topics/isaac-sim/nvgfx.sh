#!/usr/bin/env bash
# Stage the NVIDIA GRAPHICS driver libraries where a Flyte pod can reach them.
#
# Run this once after `flyte start devbox --gpu`, and again after any NVIDIA driver
# update. It is idempotent and it copies ~260 MB.
#
#     ./nvgfx.sh
#
# ── What this works around ──────────────────────────────────────────────────────
#
# The NVIDIA container stack splits the userspace driver into CAPABILITIES, and a
# container only gets the ones its `NVIDIA_DRIVER_CAPABILITIES` asks for. `compute`
# is libcuda and friends. `graphics` is libGLX_nvidia, libnvidia-glcore, the RT core
# and OptiX. They are separate sets and you can absolutely have one without the other.
#
# Flyte's devbox image ships:
#
#     $ docker inspect flyte-devbox --format '{{range .Config.Env}}{{println .}}{{end}}' \
#         | grep NVIDIA_DRIVER
#     NVIDIA_DRIVER_CAPABILITIES=compute,utility
#
# k3s runs INSIDE that container, so every task pod inherits that choice and there is
# no pod-level setting that can widen it: the injector reads the node's capabilities,
# not the pod's. The Isaac Sim image sets `NVIDIA_DRIVER_CAPABILITIES=all` in its own
# ENV and it makes no difference, because by then the decision has been made a level
# down. `runtimeClassName: nvidia` on the pod makes no difference either. Verified
# both ways on this box.
#
# The result is a pod where CUDA works perfectly and Vulkan does not exist:
#
#     [Error] [omni.rtx] VkResult: ERROR_INCOMPATIBLE_DRIVER
#     [Error] [omni.rtx] vkCreateInstance failed. Vulkan 1.1 is not supported
#     [Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices
#     [Error] [omni.kit.renderer.plugin] GPU Foundation is not initialized!
#
# because /etc/vulkan/icd.d/nvidia_icd.json names `libGLX_nvidia.so.0` and nothing
# mounted it. Training is completely unaffected, which is what makes this so annoying
# to diagnose: the run succeeds, the reward climbs, and only the replay is missing.
# Every CUDA error after the Vulkan one is fallout from the renderer never starting.
#
# ── Why the libraries ride in the IMAGE ─────────────────────────────────────────
#
# Two other places this could live, both tried, both rejected:
#
# A bind mount from the node, described by a Flyte `pod_template`. This is the tidy
# answer and flyte 2.2.1 accepts it without complaint: `task.pod_template` serialises,
# `_get_k8s_pod` builds a K8sPod, and the resulting pod has no volume, no mount, and a
# container still named after the action rather than `primary`. The backend drops it
# silently. Do not spend an afternoon on this one.
#
# Fixing the capability at the source, which means recreating the devbox container:
#
#     docker build -t flyte-devbox:graphics - <<'EOF'
#     FROM cr.flyte.org/flyteorg/flyte-devbox:gpu-latest
#     ENV NVIDIA_DRIVER_CAPABILITIES=all
#     EOF
#     flyte stop devbox && flyte start devbox --gpu --image flyte-devbox:graphics
#
# That is the RIGHT fix and it costs a cluster. The devbox mounts /var/lib/rancher/k3s
# from an ANONYMOUS docker volume, so recreating the container starts k3s from scratch:
# the in-cluster registry goes with it, and both Isaac images (17.6 GB and ~25 GB) have
# to be rebuilt and pushed before anything can run again. Worth doing on a quiet day.
#
# So this stages them into the build context instead and Dockerfile.train COPYs them to
# /opt/nvgfx, which is already at the front of LD_LIBRARY_PATH. 260 MB on a 25 GB image,
# a cached layer after the first build, and nothing to remember at run time.
#
# ── Why these particular files ──────────────────────────────────────────────────
#
# libGLX_nvidia is the Vulkan ICD itself. glcore/glsi/tls/glvkspirv are its direct
# dependencies (`ldd` on the host). rtcore and nvoptix are the hardware ray tracing
# path, which is the entire point of using the RTX renderer rather than a rasteriser.
# EGL is there for the offscreen contexts Kit creates when there is no display.
#
# The version suffix is deliberate. The NVIDIA userspace driver must match the loaded
# KERNEL module exactly, so these are pinned to whatever the host is running rather
# than copied as bare sonames, and the .so.0 symlinks are recreated on top.
set -euo pipefail

# Inside the build context, next to Dockerfile.train, and .gitignore'd: these are
# 260 MB of NVIDIA's binaries and they belong to the host driver, not to this repo.
DEST="${NVGFX_DEST:-$(cd "$(dirname "$0")" && pwd)/nvgfx}"
LIBDIR="${NVGFX_LIBDIR:-/usr/lib/aarch64-linux-gnu}"

# The one on disk, not the one nvidia-smi reports: nvidia-smi talks to the kernel
# module, and it is the file names we need here.
VERSION="$(basename "$(ls "${LIBDIR}"/libGLX_nvidia.so.* 2>/dev/null | grep -E '\.so\.[0-9]+\.' | head -1)" | sed 's/^libGLX_nvidia\.so\.//')"
if [[ -z "${VERSION}" ]]; then
    echo "no libGLX_nvidia.so.* in ${LIBDIR}: is the NVIDIA driver installed on this host?" >&2
    exit 1
fi
echo "host NVIDIA userspace driver: ${VERSION}"

# Already staged for this exact driver version? Then there is nothing to do, and more
# importantly nothing CHANGES: rewriting identical files would bust the Docker layer
# cache and trigger a needless 260 MB rebuild on every run. Checked against the version
# file rather than the directory, so a driver update does re-copy.
if [[ "$(cat "${DEST}/driver-version.txt" 2>/dev/null)" == "${VERSION}" ]]; then
    echo "${DEST} already has ${VERSION}, nothing to do"
    exit 0
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "${STAGE}"' EXIT

# Missing files are skipped rather than fatal: which of these a given driver release
# ships varies (there is no libnvidia-vulkan-producer on this one, for instance), and
# the set that matters is the ICD plus its ldd closure, which is always present.
for lib in libGLX_nvidia libnvidia-glcore libnvidia-glsi libnvidia-glvkspirv \
           libnvidia-tls libnvidia-rtcore libnvoptix libEGL_nvidia libnvidia-eglcore; do
    src="${LIBDIR}/${lib}.so.${VERSION}"
    if [[ -f "${src}" ]]; then
        cp "${src}" "${STAGE}/"
        # The ICD names `libGLX_nvidia.so.0`, and every one of these is looked up by
        # soname rather than by full version, so the symlinks are not optional.
        ln -sf "${lib}.so.${VERSION}" "${STAGE}/${lib}.so.0"
    else
        echo "  (skipping ${lib}, not in this driver)"
    fi
done
# Not dot-prefixed: Docker's COPY of a directory skips nothing, but a human listing the
# build context should be able to see why it is 260 MB.
echo "${VERSION}" > "${STAGE}/driver-version.txt"

echo "staging $(du -sh "${STAGE}" | cut -f1) into ${DEST}"
rm -rf "${DEST}"
mkdir -p "${DEST}"
cp -a "${STAGE}/." "${DEST}/"

echo "staged $(ls "${DEST}" | wc -l) entries. Dockerfile.train COPYs these to /opt/nvgfx."
echo "Next: flyte run pipeline.py leap   (the first build adds one ~260 MB layer)"
