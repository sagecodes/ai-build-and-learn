"""Magenta RealTime 2: live, steerable music generation as a Flyte 2 app.

Thin Flyte wrapper around mrt_core (which holds the weight download, model load,
and Gradio UI). The pod downloads the open weights from HuggingFace at startup,
loads Magenta RT2 on the GPU (JAX backend), and serves the live UI.

This is phase 1 (text-prompt steering). Phase 2 points a webcam at the room,
asks the Gemma-4 vLLM app to read the "vibes", and feeds that back into the same
`apply_prompt` path, mirroring gemma4-dgx-devbox/live_camera_app.py.

To just play with it on the Spark host (host GPU, public URL, no Flyte), use
run_local.py instead.

Deploy:
    python mrt_app.py
    GRADIO_SHARE=1 python mrt_app.py     # public HTTPS tunnel for remote browsers
    MRT_SIZE=base python mrt_app.py       # bigger/better model (heavier on GPU)
"""

from __future__ import annotations

import flyte
import flyte.app

# Import config (defines the app env) and the flyte-free core at top level so
# `flyte run`/serve bundles them (see CLAUDE.md note on sibling-module bundling).
from config import env
from mrt_core import run_ui


@env.server
def mrt_server(model_size: str, checkpoint: str):
    """Run the Magenta RT2 live-music Gradio UI in the pod. Blocking."""
    import sys
    import traceback

    try:
        run_ui(model_size, checkpoint)
    except BaseException as e:  # surface the real traceback in pod logs
        print(f"!!! mrt_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Magenta RT2 live app deployed: {app.url}")
