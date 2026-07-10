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
from lora_data import DATASETS, DEFAULT_DATASET
from models import DEFAULT_MODELS, MODELS

RUN_MODE = os.environ.get("RUN_MODE", "remote")

# Bases a Chroma LoRA can be loaded onto. Not chroma-flash: it's a different
# checkpoint, so an adapter trained on HD/Base doesn't belong on it.
LORA_BASE_MODELS = [k for k in MODELS if k.startswith("chroma") and k != "chroma-flash"]
# The compare task runs in project/domain flytesnacks/development on the devbox;
# override via env if you register it elsewhere.
PROJECT = os.environ.get("FLYTE_PROJECT", "flytesnacks")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
# run.url can come back as an in-cluster address; rewrite it to the console URL.
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")

# Bundle the model + dataset registries so the app image can build its pickers
# without pulling the whole torch stack. Both modules are import-light on purpose.
_here = Path(__file__).parent
_bundled = studio_app_image.with_source_file([_here / "models.py", _here / "lora_data.py"])

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
# Same deal for the LoRA generate path (`flyte deploy lora_chroma.py`).
_lora_ref = remote.Task.get(
    "imagegen-orch.generate_with_lora", project=PROJECT, domain=DOMAIN,
    auto_version="latest",
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


def _lora_task():
    """The LoRA generate entrypoint. Imported lazily: `lora_chroma` pulls in the
    task envs, which the slim app image has no business importing at module load."""
    if RUN_MODE == "local":
        from lora_chroma import generate_with_lora
        return generate_with_lora
    return _lora_ref


def _run_link(run, tail: str = "the <b>Report</b> tab has the images.") -> str:
    url = _external_url(getattr(run, "url", None))
    if not url:
        return f"Running as <code>{run.name}</code>…"
    return (f'<a href="{url}" target="_blank" rel="noopener">🔗 Open run '
            f'<code>{run.name}</code> on Flyte</a> · {tail}')


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


def run_lora_generate(lora_uri, prompt, dataset, model_key, lora_scale,
                      show_base, seed, steps, guidance):
    """Submit a generate_with_lora run against an already-trained adapter."""
    prompts = [p.strip() for p in (prompt or "").splitlines() if p.strip()]
    if not (lora_uri or "").strip():
        yield ("⚠️ Paste the LoRA Dir URI from a training run's Outputs tab "
               "(looks like `s3://flyte-data/.../chroma_lora_...`)."), ""
        return
    if not prompts:
        yield "⚠️ Enter at least one prompt (one per line).", ""
        return

    trigger = DATASETS[dataset].trigger
    missing = [p for p in prompts if trigger.lower() not in p.lower()]

    try:
        run = flyte.run(
            _lora_task(),
            lora_uri=lora_uri.strip(),
            prompts=prompts,
            dataset=dataset,
            model_key=model_key,
            lora_scale=float(lora_scale),
            show_base=bool(show_base),
            seed=int(seed),
            steps=int(steps) if int(steps) > 0 else -1,
            guidance=float(guidance) if float(guidance) >= 0 else -1.0,
        )
    except Exception as e:
        yield f"❌ Could not launch run: {type(e).__name__}: {e}", ""
        return

    link = _run_link(run)
    note = (f" ⚠️ {len(missing)} prompt(s) omit the trigger `{trigger}`, so the "
            f"LoRA may look inert there." if missing else "")
    yield (
        f"🚀 Launched LoRA generate: {len(prompts)} prompt(s) on `{model_key}`"
        f"{' with a base comparison' if show_base else ''}.{note}",
        link,
    )
    try:
        run.wait()
        yield "✅ Done. Open the Report tab for the images.", link
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
            "Every button here launches a Flyte run and links it; the app itself "
            "holds no GPU. Open the linked run's **Report** tab for the images."
        )

        with gr.Tabs():
            # ── Compare models ────────────────────────────────────────────────
            with gr.Tab("Compare models"):
                gr.Markdown(
                    "Race the open-source models on the same prompt(s). "
                    "🔒 = gated, needs an accepted HF license."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        prompt = gr.Textbox(
                            label="Prompt(s), one per line", lines=3,
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
                        steps = gr.Slider(0, 60, value=0, step=1,
                                          label="Steps (0 = model default)")
                        guidance = gr.Slider(-1, 12, value=-1, step=0.5,
                                             label="Guidance (-1 = model default)")
                        seed = gr.Number(value=1234, precision=0, label="Seed")
                gen_btn = gr.Button("Generate (launch Flyte run)", variant="primary")
                status = gr.Markdown("_idle_")
                run_link = gr.HTML()
                gen_btn.click(
                    run_compare,
                    inputs=[prompt, model_sel, seed, steps, guidance, negative],
                    outputs=[status, run_link],
                )
                gr.Examples(
                    examples=[
                        ["a woman standing in a neon Tokyo street at night, realistic"],
                        ["a storefront sign that reads OPEN 24 HOURS, photograph"],
                        ["a corgi astronaut floating in space, studio light"],
                    ],
                    inputs=prompt,
                )

            # ── Generate with a trained LoRA ──────────────────────────────────
            with gr.Tab("LoRA generate"):
                gr.Markdown(
                    "Generate from an adapter you already trained "
                    "(`flyte run lora_chroma.py train_only`). Paste the LoRA Dir "
                    "URI from that run's **Outputs** tab. Tick **Also generate "
                    "base** to render each prompt with and without the adapter, "
                    "which doubles the work but is the only way to see what the "
                    "LoRA actually changed.\n\n"
                    "Your prompt must contain the dataset's **trigger** phrase, "
                    "or the adapter has nothing to fire on."
                )
                with gr.Row():
                    with gr.Column(scale=3):
                        lora_uri = gr.Textbox(
                            label="LoRA Dir URI", lines=1,
                            placeholder="s3://flyte-data/.../chroma_lora_xxxx",
                        )
                        lora_prompt = gr.Textbox(
                            label="Prompt(s), one per line", lines=3,
                            value="a fox sitting in the snow, yarn art style",
                        )
                        with gr.Row():
                            lora_ds = gr.Dropdown(
                                choices=list(DATASETS), value=DEFAULT_DATASET,
                                label="Trained on (sets the trigger phrase)",
                            )
                            lora_model = gr.Dropdown(
                                choices=LORA_BASE_MODELS,
                                value=(LORA_BASE_MODELS[0] if LORA_BASE_MODELS else None),
                                label="Base model",
                            )
                    with gr.Column(scale=1):
                        lora_scale = gr.Slider(0, 1.5, value=1.0, step=0.05,
                                               label="LoRA scale (0 = off)")
                        show_base = gr.Checkbox(
                            value=False, label="Also generate base (before/after)",
                        )
                        lora_steps = gr.Slider(0, 60, value=0, step=1,
                                               label="Steps (0 = model default)")
                        lora_guidance = gr.Slider(-1, 12, value=-1, step=0.5,
                                                  label="Guidance (-1 = model default)")
                        lora_seed = gr.Number(value=1234, precision=0, label="Seed")
                lora_btn = gr.Button("Generate with LoRA", variant="primary")
                lora_status = gr.Markdown("_idle_")
                lora_link = gr.HTML()
                lora_btn.click(
                    run_lora_generate,
                    inputs=[lora_uri, lora_prompt, lora_ds, lora_model, lora_scale,
                            show_base, lora_seed, lora_steps, lora_guidance],
                    outputs=[lora_status, lora_link],
                )

                # The trigger lives in the data, so switching dataset should
                # rewrite the prompt box rather than leave a stale trigger behind.
                def _default_prompt(ds_key: str) -> str:
                    return DATASETS[ds_key].eval_prompts[0]

                lora_ds.change(_default_prompt, inputs=lora_ds, outputs=lora_prompt)
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
