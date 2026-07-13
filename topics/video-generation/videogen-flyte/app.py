"""Gradio studio for the video-generation demo: a thin Flyte *app* that launches runs.

The important architectural point, same as the image-gen studio: this app is a
LAUNCHER, not a worker. It has no torch, no diffusers, holds no GPU and loads no
model. Every button just submits a `flyte.run(...)` against the already-registered
tasks and hands you back a link to that run's Report tab, where the clips play.

Why it's built this way: a Gradio app pod stays alive for as long as the app is
up. If it held the GPU to generate in-process, it would hold the Spark's only GPU
forever and every pipeline task would sit Unschedulable behind it. Launching runs
means the GPU is held only while a clip is actually rendering.

Deploy (from the devbox):
    python app.py

Env:
    RUN_MODE=local     call the tasks directly instead of via remote refs
    GRADIO_SHARE=1     expose a public gradio.live URL
"""

from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.app
import flyte.remote as remote

from config import APP_NAME, APP_PORT, studio_app_image
from models import DEFAULT_MODELS, IMAGE_MODELS, MODELS

PROJECT = os.environ.get("FLYTE_PROJECT", "video-generation")
DOMAIN = os.environ.get("FLYTE_DOMAIN", "development")
FLYTE_UI_URL = os.environ.get("FLYTE_UI_URL", "http://localhost:30080")
RUN_MODE = os.environ.get("RUN_MODE", "remote")

_here = Path(__file__).parent

# Bake ONLY the import-light registry into the slim app image, so the model picker
# can be built without dragging torch/diffusers into the app pod.
_bundled = studio_app_image.with_source_file(_here / "models.py")

env = flyte.app.AppEnvironment(
    name=APP_NAME,
    image=_bundled,
    resources=flyte.Resources(cpu=1, memory="1Gi"),   # a launcher, not a GPU box
    port=APP_PORT,
    requires_auth=False,
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    env_vars=({"GRADIO_SHARE": os.environ["GRADIO_SHARE"]}
              if "GRADIO_SHARE" in os.environ else {}),
)


def _task(name: str):
    """Resolve a task: the live import locally, a remote ref in the cluster.

    Lazy on purpose, so importing this module without a cluster still works.
    """
    if RUN_MODE == "local":
        import compare_pipeline

        return getattr(compare_pipeline, name)
    return remote.Task.get(f"videogen-orch.{name}", project=PROJECT, domain=DOMAIN,
                           auto_version="latest")


def _external_url(url: str) -> str:
    """Rewrite an in-cluster run URL to one the browser can actually reach."""
    try:
        return FLYTE_UI_URL.rstrip("/") + urlparse(str(url)).path
    except Exception:
        return str(url)


def _launch(task_name: str, **kwargs):
    """Submit a run and stream status. Yields (status, link) twice: on submit, on finish."""
    prompts = [p.strip() for p in (kwargs.pop("prompts_text") or "").splitlines() if p.strip()]
    if not prompts:
        yield "⚠️ Enter at least one prompt.", ""
        return

    models = kwargs.pop("models") or DEFAULT_MODELS
    est = sum(MODELS[m].download_gb for m in models)

    run = flyte.run(_task(task_name), prompts=prompts, models=list(models), **kwargs)
    link = (f'<a href="{_external_url(run.url)}" target="_blank">Open the report for '
            f'<code>{run.name}</code></a>')
    yield (
        f"🎬 Submitted **{run.name}** · {len(prompts)} prompt(s) × {len(models)} model(s).\n\n"
        f"The clips play in the **Report** tab as each one finishes. First run for a "
        f"model also downloads its weights (up to ~{est:.0f}GB across this set), which "
        f"is cached forever after. Video generation is slow on this box: expect a few "
        f"minutes per clip.",
        link,
    )
    run.wait()
    yield f"✅ **{run.name}** finished. The clips are in the report.", link


def create_demo():
    import gradio as gr

    video_choices = [
        f"{k} · {s.family} · ~{s.download_gb:.0f}GB" + (" · ♪ audio" if s.has_audio else "")
        for k, s in MODELS.items()
    ]
    key_of = {c: k for c, k in zip(video_choices, MODELS)}
    i2v_keys = [k for k, s in MODELS.items() if s.supports_i2v]

    with gr.Blocks(title="Video-Gen Studio") as demo:
        gr.Markdown(
            "# 🎬 Video-Gen Studio\n"
            "Launch open-source video-model runs on the Spark. Each run renders your "
            "prompts and writes one side-by-side report with **playable clips**."
        )

        with gr.Tab("Text to video"):
            t_prompts = gr.Textbox(
                label="Prompts (one per line)", lines=4,
                placeholder="waves crashing on black volcanic rock, slow motion",
            )
            t_models = gr.CheckboxGroup(
                choices=video_choices,
                value=[c for c in video_choices if key_of[c] in DEFAULT_MODELS],
                label="Models",
                info="ltx2-distilled is the showpiece (22B, generates synced audio) but "
                     "its first download is ~95GB. The two Wan models are the quick default.",
            )
            with gr.Row():
                t_steps = gr.Slider(-1, 60, -1, step=1, label="Steps (-1 = model default)")
                t_frames = gr.Slider(-1, 121, -1, step=8, label="Frames (-1 = default)")
                t_seed = gr.Slider(0, 99999, 1234, step=1, label="Seed")
            t_go = gr.Button("Generate", variant="primary")
            t_status = gr.Markdown()
            t_link = gr.HTML()

            gr.Examples(
                [["a red panda barista pouring latte art, steam rising, cozy cafe, 50mm"],
                 ["waves crashing on black volcanic rock, slow motion, overcast"],
                 ["a neon-lit alley in the rain, camera slowly pushing forward"]],
                inputs=[t_prompts],
            )

            t_go.click(
                lambda p, m, s, f, sd: (yield from _launch(
                    "compare", prompts_text=p, models=[key_of[x] for x in m],
                    steps=int(s), num_frames=int(f), seed=int(sd))),
                [t_prompts, t_models, t_steps, t_frames, t_seed],
                [t_status, t_link],
            )

        with gr.Tab("Image to video"):
            gr.Markdown(
                "Generates a **first frame** with a small text-to-image model, then "
                "animates it. The report shows the frame next to the clip, so you can "
                "see what the video model kept and where it drifted."
            )
            a_prompts = gr.Textbox(label="Prompts (one per line)", lines=3,
                                   placeholder="a paper boat on a rain puddle, neon reflections")
            a_image_model = gr.Dropdown(
                choices=list(IMAGE_MODELS), value="sd-turbo", label="First-frame model",
                info="sd-turbo makes a frame in about a second. The video model is the "
                     "slow part, and the thing worth looking at.",
            )
            a_models = gr.CheckboxGroup(
                choices=i2v_keys, value=["wan22-ti2v-5b"],
                label="Video models (image-to-video capable only)",
            )
            with gr.Row():
                a_steps = gr.Slider(-1, 60, -1, step=1, label="Steps (-1 = default)")
                a_seed = gr.Slider(0, 99999, 1234, step=1, label="Seed")
            a_go = gr.Button("Animate", variant="primary")
            a_status = gr.Markdown()
            a_link = gr.HTML()

            a_go.click(
                lambda p, im, m, s, sd: (yield from _launch(
                    "animate", prompts_text=p, models=list(m), image_model=im,
                    steps=int(s), seed=int(sd))),
                [a_prompts, a_image_model, a_models, a_steps, a_seed],
                [a_status, a_link],
            )

    return demo


@env.server
def studio_server():
    flyte.init_in_cluster(project=PROJECT, domain=DOMAIN)
    create_demo().launch(
        server_name="0.0.0.0", server_port=APP_PORT,
        share=os.environ.get("GRADIO_SHARE") == "1",
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=_here)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Video-Gen Studio deployed: {app.url}")
