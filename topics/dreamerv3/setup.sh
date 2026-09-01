#!/usr/bin/env bash
# Set up DreamerV3 on a DGX Spark: venv, upstream checkout, and the one patch it needs
# to run on Blackwell. Idempotent; safe to re-run.
#
# Upstream is NOT vendored into this repo. It is cloned to $DREAMER_ROOT (default
# ~/dreamerv3) at a pinned commit and patched in place, for the same reason the Isaac
# Sim demo keeps its 50 GB build tree outside git: someone else's source tree has no
# business inside ours, and pinning the commit is what makes it reproducible anyway.
set -euo pipefail

DREAMER_ROOT="${DREAMER_ROOT:-$HOME/dreamerv3}"
# Pinned. The patch below is line-addressed against this commit; a moving target would
# silently stop applying.
DREAMER_COMMIT="${DREAMER_COMMIT:-e3f02248693a79dc8b0ebd62c93683888ddaccfe}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "==> venv"
if [ ! -d "$HERE/.venv" ]; then
  uv venv --python 3.12 "$HERE/.venv"
fi
uv pip install --python "$HERE/.venv/bin/python" -r "$HERE/requirements.txt"

echo "==> upstream dreamerv3 at $DREAMER_ROOT"
if [ ! -d "$DREAMER_ROOT/.git" ]; then
  git clone https://github.com/danijar/dreamerv3 "$DREAMER_ROOT"
fi
git -C "$DREAMER_ROOT" fetch --depth=1 origin "$DREAMER_COMMIT" 2>/dev/null || git -C "$DREAMER_ROOT" fetch origin
git -C "$DREAMER_ROOT" checkout -q "$DREAMER_COMMIT"

echo "==> patch: jax.jit keyword-only arguments"
# checkout above resets the tree, so the patch always applies to a clean checkout.
# --check first so a future upstream bump fails loudly instead of half-applying.
if git -C "$DREAMER_ROOT" apply --check "$HERE/patches/0001-jax-jit-keyword-only.patch" 2>/dev/null; then
  git -C "$DREAMER_ROOT" apply "$HERE/patches/0001-jax-jit-keyword-only.patch"
  echo "    applied"
else
  echo "    already applied, or upstream moved: verify before trusting this tree"
fi

echo "==> verify"
PYTHONPATH="$DREAMER_ROOT" "$HERE/.venv/bin/python" - <<'PY'
import jax
print("  jax", jax.__version__, "devices:", jax.devices())
import dreamerv3, embodied  # noqa: F401
print("  dreamerv3 imports")
from dm_control import suite
import numpy as np
env = suite.load("walker", "walk")
env.reset()
ts = env.step(np.zeros(env.action_spec().shape))
print("  dm_control steps, reward", round(float(ts.reward), 4))
PY

cat <<EOF

Done. Train a walker:

  export PYTHONPATH=$DREAMER_ROOT
  ./.venv/bin/python \$PYTHONPATH/dreamerv3/main.py \\
    --configs dmc_proprio --task dmc_walker_walk \\
    --logdir ~/logdir/walker-\$(date +%Y%m%d-%H%M%S)

Note --run.log_every defaults to minutes, so a short run writes nothing to disk and
looks like it did nothing. Pass --run.log_every 10 when smoke testing.
EOF
