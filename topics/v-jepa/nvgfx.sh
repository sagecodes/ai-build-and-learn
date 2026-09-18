#!/usr/bin/env bash
# Stage the NVIDIA EGL libraries so MuJoCo renders on the GPU inside Flyte pods.
#
#     ./nvgfx.sh          # once, and again after any NVIDIA driver update
#
# The devbox gives pods the COMPUTE half of the NVIDIA driver only
# (NVIDIA_DRIVER_CAPABILITIES=compute,utility), so EGL falls back to Mesa's software
# rasteriser. Measured on this box, same G1 clip at 256x256:
#
#     host, NVIDIA EGL           580 fps
#     Mesa llvmpipe, 8 cores       3.4 fps   (7.8 with shadows off)
#
# That made `selfwalk`'s V-JEPA labelling take hours instead of minutes, and it is also
# why `walk`'s random-play phase took ten minutes. The full story, and why a pod-level
# setting cannot fix it, is in topics/isaac-sim/nvgfx.sh. This is the EGL-only subset
# of that fix (~110 MB instead of 260): config.py copies nvgfx/ into both images and
# registers it with ldconfig, and skips the layer if this has not been run.
set -euo pipefail

DEST="$(cd "$(dirname "$0")" && pwd)/nvgfx"
LIBDIR=/usr/lib/aarch64-linux-gnu
VERSION="$(basename "$(ls "${LIBDIR}"/libEGL_nvidia.so.* | grep -E '\.so\.[0-9]+\.' | head -1)" | sed 's/^libEGL_nvidia\.so\.//')"
echo "host NVIDIA userspace driver: ${VERSION}"

if [[ "$(cat "${DEST}/driver-version.txt" 2>/dev/null)" == "${VERSION}" ]]; then
    echo "${DEST} already has ${VERSION}, nothing to do"
    exit 0
fi

rm -rf "${DEST}"
mkdir -p "${DEST}"
# libEGL_nvidia is the glvnd vendor library; the rest is its ldd closure plus glcore.
for lib in libEGL_nvidia libnvidia-eglcore libnvidia-glcore libnvidia-glsi libnvidia-tls libnvidia-gpucomp; do
    cp "${LIBDIR}/${lib}.so.${VERSION}" "${DEST}/"
    ln -sf "${lib}.so.${VERSION}" "${DEST}/${lib}.so.0"
done
cp /usr/share/glvnd/egl_vendor.d/10_nvidia.json "${DEST}/"
echo "${VERSION}" > "${DEST}/driver-version.txt"
echo "staged $(du -sh "${DEST}" | cut -f1) into ${DEST}"
