"""Open Image-Gen Studio as a Flyte 2 GPU app.

Thin Flyte wrapper around app_ui (which holds the flyte-free Gradio UI + model
cache), mirroring magenta's mrt_app→mrt_core split. The pod loads models from
HuggingFace on first use, generates on the GPU, and shows the results side by
side in the browser.

For the batch, report-first path (a whole prompt×model grid rendered as a Flyte
report + saved PNGs), use compare_pipeline.py instead — that's the pipeline this
studio is the interactive front door to.

Deploy (on the devbox):
    python app.py
    GRADIO_SHARE=1 python app.py     # public HTTPS tunnel for a remote browser
"""

from __future__ import annotations

import os
from pathlib import Path

import flyte
import flyte.app

from config import APP_NAME, APP_PORT, HF_HOME, HF_SECRET, PLATFORM, REGISTRY, app_gpu_pod, app_image

# Bundle the flyte-free modules so the server pod can import them (same
# sibling-bundling trick the other Gradio apps use).
_bundled = app_image.with_source_file(
    [Path(__file__).parent / f for f in ("app_ui.py", "imagegen_core.py", "models.py")]
)

env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image=_bundled,
    pod_template=app_gpu_pod,       # GPU via pod_template (see config.py note)
    port=APP_PORT,
    requires_auth=False,
    secrets=[HF_SECRET],
    env_vars={
        "HF_HOME": HF_HOME,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "IMAGEGEN_PORT": str(APP_PORT),
        **({"GRADIO_SHARE": os.environ["GRADIO_SHARE"]} if "GRADIO_SHARE" in os.environ else {}),
    },
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
)


@env.server
def studio_server():
    """Run the studio UI in the pod. Blocking."""
    import sys
    import traceback

    from app_ui import launch

    try:
        launch(share=os.environ.get("GRADIO_SHARE") == "1", server_port=APP_PORT)
    except BaseException as e:  # surface the real traceback in pod logs
        print(f"!!! studio_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


if __name__ == "__main__":
    flyte.init_from_config(root_dir=Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Open Image-Gen Studio deployed: {app.url}")
