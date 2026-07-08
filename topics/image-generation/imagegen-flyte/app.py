"""Open Image-Gen Studio: a Flyte 2 app that LAUNCHES compare runs.

The studio is a thin CPU app. It loads no model and touches no GPU: every
"Generate" submits the `compare` pipeline as a Flyte run and links the report
(the prompt x model grid + saved PNGs). All GPU work happens inside the
pipeline's tasks, so the app pod stays tiny and can never pin weights in memory.

This mirrors the langgraph_agent_research tutorial's app: a Gradio front end over
`flyte.run(...)`, with a local/remote toggle.

Development progression:
  1. Local app + local pipeline:   RUN_MODE=local python app.py
  2. Local app + remote pipeline:  python app.py
  3. Deploy the pipeline, then the app:
       flyte deploy compare_pipeline.py   # register tasks + build the task image
       python app.py                      # (or deploy the app itself)
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.remote as remote

from config import APP_NAME, APP_PORT, REGISTRY, studio_app_image
from models import DEFAULT_MODELS, MODELS

RUN_MODE = os.environ.get("RUN_MODE", "remote")
# The compare task runs in project/domain flytesnacks/development on the devbox;
# override via env if you register it elsewhere.
PROJECT = os.environ.get("FLYTE_PROJECT", "flytesnacks")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
# run.url can come back as an in-cluster address; rewrite it to the console URL.
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")

# Bundle the model registry so the app image can build the model picker without
# pulling the whole torch stack.
_bundled = studio_app_image.with_source_file([Path(__file__).parent / "models.py"])

env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image=_bundled,
    resources=flyte.Resources(cpu=1, memory="1Gi"),   # a launcher, not a GPU box
    port=APP_PORT,
    requires_auth=False,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    env_vars=(
        {"GRADIO_SHARE": os.environ["GRADIO_SHARE"]} if "GRADIO_SHARE" in os.environ else {}
    ),
)

# Pre-registered compare task, fetched from the control plane at run time. Deploy
# the pipeline first (`flyte deploy compare_pipeline.py`) so this resolves. The
# ref is lazy, so importing this module without a live cluster is fine.
_compare_ref = remote.Task.get(
    "imagegen-orch.compare", project=PROJECT, domain=DOMAIN, auto_version="latest",
)


def _external_url(url) -> str:
    """Rewrite an in-cluster run URL to the browser-reachable console URL."""
    if not url:
        return ""
    s = str(url)
    if s.startswith("http") and "flyte-binary" not in s and "flyte:" not in s:
        return s
    return f"{FLYTE_UI_URL}{urlparse(s).path}"


def _compare_task():
    """The compare entrypoint: imported directly for local dev, else the ref."""
    if RUN_MODE == "local":
        from compare_pipeline import compare
        return compare
    return _compare_ref


def run_compare(prompt, model_keys, seed, steps, guidance, negative):
    """Submit a compare run, stream the run link, then confirm on completion."""
    # One prompt per line (NOT comma-split: prompts are full of commas). The grid
    # is prompts x models, and each model task loads once and renders every prompt.
    prompts = [p.strip() for p in (prompt or "").splitlines() if p.strip()]
    if not prompts:
        yield "⚠️ Enter at least one prompt (one per line).", ""
        return
    if not model_keys:
        yield "⚠️ Pick at least one model.", ""
        return

    try:
        run = flyte.run(
            _compare_task(),
            prompts=prompts,
            models=list(model_keys),
            seed=int(seed),
            steps=int(steps) if int(steps) > 0 else -1,
            guidance=float(guidance) if float(guidance) >= 0 else -1.0,
            negative_prompt=(negative or ""),
        )
    except Exception as e:
        yield f"❌ Could not launch run: {type(e).__name__}: {e}", ""
        return

    url = _external_url(getattr(run, "url", None))
    link = (
        f'<a href="{url}" target="_blank" rel="noopener">🔗 Open run '
        f'<code>{run.name}</code> on Flyte</a> — the <b>Report</b> tab has the grid.'
        if url else f"Running as <code>{run.name}</code>…"
    )
    yield (
        f"🚀 Launched compare: {len(prompts)} prompt(s) × {len(model_keys)} "
        f"model(s). Uncached weights download on first use, so the first run for "
        f"a model is slow.",
        link,
    )

    try:
        run.wait()
        yield (
            "✅ Done. Open the Report tab in the run for the side-by-side grid "
            "(click any image to zoom); full-res PNGs are the run's output.",
            link,
        )
    except Exception as e:
        yield f"⚠️ Launched, but couldn't confirm completion here: {e}", link


def create_demo():
    import gradio as gr

    def _label(k: str) -> str:
        s = MODELS[k]
        return f"{k} · {s.family}{' 🔒' if s.gated else ''}"

    choices = [(_label(k), k) for k in MODELS]

    with gr.Blocks(title="Open Image-Gen Studio") as demo:
        gr.Markdown(
            "# 🎨 Open Image-Gen Studio\n"
            "Pick the open-source models to race, hit **Generate**, and the studio "
            "launches a Flyte `compare` run. Open the linked run's **Report** tab "
            "for the side-by-side grid (🔒 = gated, needs an HF license)."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt(s) — one per line", lines=3,
                    placeholder="one prompt per line; the grid is prompts × models",
                    value="a red panda barista pouring latte art, cozy cafe, 50mm, bokeh",
                )
                negative = gr.Textbox(
                    label="Negative prompt (ignored by distilled FLUX/Z-Image/Sana)",
                    lines=1, value="",
                )
                model_sel = gr.CheckboxGroup(
                    choices=choices, value=DEFAULT_MODELS, label="Models",
                )
            with gr.Column(scale=1):
                steps = gr.Slider(0, 60, value=0, step=1, label="Steps (0 = model default)")
                guidance = gr.Slider(-1, 12, value=-1, step=0.5,
                                     label="Guidance (-1 = model default)")
                seed = gr.Number(value=1234, precision=0, label="Seed")
        gen_btn = gr.Button("Generate (launch Flyte run)", variant="primary")
        status = gr.Markdown("_idle_")
        run_link = gr.HTML()

        inputs = [prompt, model_sel, seed, steps, guidance, negative]
        gen_btn.click(run_compare, inputs=inputs, outputs=[status, run_link])

        gr.Examples(
            examples=[
                ["a woman standing in a neon Tokyo street at night, realistic"],
                ["a storefront sign that reads OPEN 24 HOURS, photograph"],
                ["a corgi astronaut floating in space, studio light"],
            ],
            inputs=prompt,
        )
    return demo


@env.server
def studio_server():
    """Serve the launcher UI from the app pod."""
    flyte.init_in_cluster(project=PROJECT, domain=DOMAIN)
    create_demo().launch(
        server_name="0.0.0.0", server_port=APP_PORT,
        share=os.environ.get("GRADIO_SHARE") == "1",
    )


if __name__ == "__main__":
    if RUN_MODE == "local":
        # Local app + local pipeline: submit against the devbox from the host.
        flyte.init_from_config(root_dir=Path(__file__).parent)
        create_demo().launch(
            server_name="0.0.0.0", server_port=APP_PORT,
            share=os.environ.get("GRADIO_SHARE") == "1",
        )
    else:
        flyte.init_from_config(root_dir=Path(__file__).parent)
        app = flyte.with_servecontext(interactive_mode=True).serve(env)
        print(f"Open Image-Gen Studio deployed: {app.url}")
