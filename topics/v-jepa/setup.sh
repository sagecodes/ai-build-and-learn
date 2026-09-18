#!/usr/bin/env bash
# Host-side setup. The pods build their own image from config.py; this is only what
# `flyte run` and smoke_test.py need locally.
#
#   ./setup.sh           venv + the V-JEPA 2-AC sources
#   ./setup.sh --ac      also fetch the 11.7 GB AC checkpoint
#   ./setup.sh --stage   also stage that checkpoint into the devbox for the pods
set -euo pipefail
cd "$(dirname "$0")"

VJEPA2_SHA=204698b45b3712590f06245fbfba32d3be539812
AC_URL=https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
AC_LOCAL=${AC_LOCAL:-$HOME/models/vjepa2/vjepa2-ac-vitg.pt}
# k3s runs INSIDE the flyte-devbox container, so a pod's hostPath resolves against
# that container's filesystem and not the real host. /var/lib/kubelet is a docker
# volume, so what we put there survives `flyte stop devbox`.
STAGE_DIR=/var/lib/kubelet/hf-cache/vjepa2

uv venv --python 3.12 .venv
# torch first, on its own, from the cu130 index. Anything else resolving torch from
# plain PyPI first gets the CPU-only aarch64 wheel and cuda silently disappears.
uv pip install --python .venv/bin/python torch torchvision \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install --python .venv/bin/python -r requirements.txt

# V-JEPA 2-AC's model code. Cloned, not pip-installed: upstream's setup.py declares
# no packages, so `pip install git+...` succeeds and imports nothing. Pinned, because
# main currently ships `VJEPA_BASE_URL = "http://localhost:8300"` in
# src/hub/backbones.py, which breaks torch.hub.load for everyone.
VJEPA2_SRC=${VJEPA2_SRC:-$PWD/.vendor/vjepa2}
if [ ! -d "$VJEPA2_SRC/src/models" ]; then
  mkdir -p "$(dirname "$VJEPA2_SRC")"
  git clone --filter=blob:none https://github.com/facebookresearch/vjepa2 "$VJEPA2_SRC"
  git -C "$VJEPA2_SRC" checkout --quiet "$VJEPA2_SHA"
fi

# The Panda. Menagerie in full is a few hundred MB of robots we never load.
MENAGERIE=${MENAGERIE:-$PWD/.vendor/menagerie}
if [ ! -d "$MENAGERIE/franka_emika_panda/assets" ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/google-deepmind/mujoco_menagerie "$MENAGERIE"
  git -C "$MENAGERIE" sparse-checkout set franka_emika_panda
fi

if [ "${1:-}" = "--ac" ] || [ "${1:-}" = "--stage" ]; then
  mkdir -p "$(dirname "$AC_LOCAL")"
  # -C - resumes a partial file rather than restarting 11.7 GB.
  curl -fSL -C - -o "$AC_LOCAL" "$AC_URL"
fi

if [ "${1:-}" = "--stage" ]; then
  docker exec flyte-devbox mkdir -p "$STAGE_DIR"
  docker cp "$AC_LOCAL" "flyte-devbox:$STAGE_DIR/vjepa2-ac-vitg.pt"
  # Read-write: anything that takes a lock in the mount fails on a read-only one.
  docker exec flyte-devbox chmod -R a+rwX "$STAGE_DIR"
  echo "staged -> $STAGE_DIR (pods see it at /mnt/hf/vjepa2)"
fi

cat <<EOF

ok: $(.venv/bin/python -c 'import torch;print(torch.__version__, "cuda", torch.cuda.is_available())')
    vjepa2 src  $VJEPA2_SRC
    menagerie   $MENAGERIE

Put these in your environment for smoke_test.py and any host-side run:
    export VJEPA2_SRC=$VJEPA2_SRC
    export MENAGERIE=$MENAGERIE
EOF
