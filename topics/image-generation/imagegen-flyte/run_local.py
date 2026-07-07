"""Generate on the Spark host GPU directly, no Flyte/Knative — for quick iteration.

Loads one model in this process on the host GB10 and either writes images to disk
or, with --serve, launches the Gradio studio locally with a public share URL.
Flyte-free, so it needs the torch/diffusers deps (see README for the local venv).

    # one model, a couple prompts, write PNGs + an HTML contact sheet to ./out
    python run_local.py --model flux1-schnell \
        --prompt "a corgi astronaut, studio light" \
        --prompt "neon cyberpunk alley in the rain"

    # compare several models on one prompt
    python run_local.py --models flux1-schnell,sdxl,qwen-image \
        --prompt "a storefront sign that reads OPEN 24 HOURS"

    # just launch the studio UI on the host GPU (public URL)
    GRADIO_SHARE=1 python run_local.py --serve
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from imagegen_core import (
    GenResult,
    load_pipeline,
    pil_to_data_uri,
    render_grid,
    timed_generate,
)
from models import DEFAULT_MODELS, get_spec, resolve_models


def main() -> None:
    ap = argparse.ArgumentParser(description="Local host-GPU image generation")
    ap.add_argument("--model", help="single model key (shortcut for --models)")
    ap.add_argument("--models", help="comma-separated model keys",
                    default=None)
    ap.add_argument("--prompt", action="append", dest="prompts", default=[],
                    help="prompt (repeatable)")
    ap.add_argument("--steps", type=int, default=-1)
    ap.add_argument("--guidance", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="out", help="output dir")
    ap.add_argument("--serve", action="store_true",
                    help="launch the Gradio studio instead of batch generating")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    if args.serve:
        from app_ui import launch  # flyte-free UI factory shared with the app
        launch(share=os.environ.get("GRADIO_SHARE") == "1")
        return

    keys = ([args.model] if args.model
            else args.models.split(",") if args.models
            else DEFAULT_MODELS)
    specs = resolve_models(keys)
    prompts = args.prompts or ["a red panda barista pouring latte art, cozy cafe, 50mm, bokeh"]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    results: list[GenResult] = []
    for spec in specs:
        print(f"\n=== {spec.key} ({spec.repo}) ===", flush=True)
        pipe = load_pipeline(spec)
        for i, prompt in enumerate(prompts):
            print(f"  [{i + 1}/{len(prompts)}] {prompt[:70]}", flush=True)
            img, secs = timed_generate(
                pipe, spec, prompt,
                steps=None if args.steps < 0 else args.steps,
                guidance=None if args.guidance < 0 else args.guidance,
                seed=args.seed,
            )
            fname = f"{spec.key}__{i:02d}.png"
            img.save(out / fname)
            print(f"      -> {out / fname}  ({secs:.1f}s)", flush=True)
            results.append(GenResult(spec.key, prompt, secs,
                                     data_uri=pil_to_data_uri(img, max_side=768)))
        del pipe
        _free_gpu()

    sheet = out / "index.html"
    sheet.write_text(render_grid(prompts, specs, results,
                                 meta=f"host-GPU · seed={args.seed}"))
    print(f"\nContact sheet: {sheet}", flush=True)


def _free_gpu() -> None:
    try:
        import gc

        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        pass


if __name__ == "__main__":
    main()
