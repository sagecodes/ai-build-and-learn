"""Run the Magenta RT2 live UI directly on the Spark host (uses the host GPU).

This bypasses Flyte/Knative GPU scheduling entirely: it loads the model in this
process on the host's GB10 and opens a public gradio.live URL to play with.
Flyte-free, so it only needs the jax/magenta/gradio deps (see .venv-local).

    GRADIO_SHARE=1 .venv-local/bin/python run_local.py            # mrt2_small
    MRT_SIZE=base GRADIO_SHARE=1 .venv-local/bin/python run_local.py
"""

from __future__ import annotations

import os

from mrt_core import run_ui

_SIZES = {
    "small": ("mrt2_small", "mrt2_small.safetensors"),
    "base": ("mrt2_base", "mrt2_base.safetensors"),
}


if __name__ == "__main__":
    os.environ.setdefault("MAGENTA_HOME", "/tmp/magenta")
    size, checkpoint = _SIZES["base" if os.environ.get("MRT_SIZE") == "base" else "small"]
    run_ui(size, checkpoint)
