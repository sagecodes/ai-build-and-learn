"""Flyte-free Gradio UI for the image-generation studio.

Type a prompt, tick the models to race, and see the images appear side by side as
each model finishes. Models load lazily and stay cached (loading is the slow
part), guarded by a lock so two clicks can't load the same pipeline twice.

Used two ways, exactly like magenta's mrt_core:
  - app.py wraps `launch` in an @env.server to serve it as a Flyte GPU app, and
    passes an `on_flyte_run` callback that submits the batch compare pipeline.
  - run_local.py calls `launch(share=True)` to run it on the host GPU.

Nothing here imports flyte.
"""

from __future__ import annotations

import os
import threading
import time
from typing import Callable

from imagegen_core import generate, load_pipeline
from models import DEFAULT_MODELS, MODELS, get_spec

# Lazily-built pipeline cache: model key -> pipeline. Loading is expensive, so
# keep each warm once built. The GB10's unified memory holds several at once.
_cache: dict = {}
_cache_lock = threading.Lock()


def _get_pipe(key: str):
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    # Load outside the lock so a slow load doesn't block reads of other models;
    # double-check on the way in.
    spec = get_spec(key)
    pipe = load_pipeline(spec)
    with _cache_lock:
        _cache.setdefault(key, pipe)
        return _cache[key]


def launch(
    share: bool = False,
    server_port: int | None = None,
    on_flyte_run: Callable[[list[str], list[str], int], str] | None = None,
):
    """Build and launch the studio. Blocks forever.

    on_flyte_run(prompts, models, seed) -> report/run URL. If given, a
    "Run full grid on Flyte" button appears that hands the batch off to the
    compare pipeline and links the resulting report.
    """
    import gradio as gr

    all_keys = list(MODELS)

    def _model_label(k: str) -> str:
        s = MODELS[k]
        gate = " 🔒" if s.gated else ""
        return f"{k} · {s.family}{gate}"

    choices = [(_model_label(k), k) for k in all_keys]

    def do_generate(prompt, model_keys, steps, guidance, seed, negative, progress=gr.Progress()):
        prompt = (prompt or "").strip()
        if not prompt:
            yield [], "⚠️ Enter a prompt first."
            return
        if not model_keys:
            yield [], "⚠️ Pick at least one model."
            return

        gallery: list = []
        for idx, key in enumerate(model_keys):
            spec = get_spec(key)
            progress((idx) / len(model_keys), desc=f"loading + running {key}…")
            yield gallery, f"⏳ {key}: loading + generating ({idx + 1}/{len(model_keys)})…"
            try:
                pipe = _get_pipe(key)
                t0 = time.time()
                img = generate(
                    pipe, spec, prompt,
                    steps=None if int(steps) <= 0 else int(steps),
                    guidance=None if float(guidance) < 0 else float(guidance),
                    seed=int(seed),
                    negative_prompt=(negative or None),
                )
                secs = time.time() - t0
                gallery = gallery + [(img, f"{key} · {secs:.1f}s · {spec.steps} steps")]
                yield gallery, f"✅ {key} done in {secs:.1f}s ({idx + 1}/{len(model_keys)})"
            except Exception as e:
                yield gallery, f"❌ {key} failed: {type(e).__name__}: {e}"
        yield gallery, f"Done — {len(gallery)}/{len(model_keys)} model(s)."

    def do_flyte(prompt, model_keys, seed):
        if on_flyte_run is None:
            return "Flyte batch runs are only available in the deployed app."
        prompt = (prompt or "").strip()
        if not prompt or not model_keys:
            return "⚠️ Need a prompt and at least one model."
        try:
            url = on_flyte_run([prompt], list(model_keys), int(seed))
            return f"🚀 Launched batch compare on Flyte → [open the report]({url})"
        except Exception as e:
            return f"❌ Could not launch Flyte run: {type(e).__name__}: {e}"

    with gr.Blocks(title="Open Image-Gen Studio") as demo:
        gr.Markdown(
            "# 🎨 Open Image-Gen Studio\n"
            "Type a prompt, pick the open-source models to race, and compare the "
            "results side by side. Models load on first use (🔒 = gated, needs an "
            "HF license)."
        )
        with gr.Row():
            with gr.Column(scale=3):
                prompt = gr.Textbox(
                    label="Prompt", lines=2,
                    value="a red panda barista pouring latte art, cozy cafe, 50mm, bokeh",
                )
                negative = gr.Textbox(
                    label="Negative prompt (ignored by distilled FLUX/Z-Image)",
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
        with gr.Row():
            gen_btn = gr.Button("Generate", variant="primary")
            if on_flyte_run is not None:
                flyte_btn = gr.Button("Run full grid on Flyte (report)")
        status = gr.Markdown("_idle_")
        gallery = gr.Gallery(label="Results", columns=3, height="auto",
                             object_fit="contain", show_label=True)

        gen_btn.click(
            do_generate,
            inputs=[prompt, model_sel, steps, guidance, seed, negative],
            outputs=[gallery, status],
        )
        if on_flyte_run is not None:
            flyte_btn.click(do_flyte, inputs=[prompt, model_sel, seed], outputs=status)

    demo.queue(default_concurrency_limit=2)
    port = server_port or int(os.environ.get("IMAGEGEN_PORT", "7862"))
    _, local_url, share_url = demo.launch(
        server_name="0.0.0.0", server_port=port, share=share,
        prevent_thread_lock=True,
    )
    print(f"[studio] local URL: {local_url}", flush=True)
    print(f"[studio] PUBLIC URL: {share_url}" if share_url
          else "[studio] no share URL (set GRADIO_SHARE=1)", flush=True)
    while True:
        time.sleep(3600)
