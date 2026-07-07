"""Flyte-free core: load a diffusers pipeline, generate, and render reports.

Nothing here imports flyte, so it runs three ways unchanged:
  - inside a Flyte GPU task (compare_pipeline.py),
  - inside the Gradio app pod (app.py), and
  - directly on the Spark host (run_local.py).

The two things worth knowing:
  1. `load_pipeline` resolves the diffusers class *by name* (models.py picks it),
     falling back to AutoPipeline, so a renamed/missing class for a brand-new
     model degrades gracefully instead of exploding at import time.
  2. On the GB10 the GPU addresses the same unified 128GB as the CPU, so we just
     `.to("cuda")` — CPU offload (which streams layers host↔device) only helps on
     discrete GPUs. Set IMAGEGEN_OFFLOAD=1 to force offload on such a box.
"""

from __future__ import annotations

import base64
import html
import io
import os
import time
from dataclasses import dataclass

# Import ModelSpec lazily-safe: models.py has no heavy deps.
from models import ModelSpec


# ── Loading ─────────────────────────────────────────────────────────────────────

def _torch_dtype(name: str):
    import torch

    return getattr(torch, name)


def load_pipeline(spec: ModelSpec, device: str = "cuda", model_path: str | None = None):
    """Load `spec` into a ready-to-call diffusers pipeline on `device`.

    `model_path`, when given, is a local directory of already-downloaded weights
    (produced by the cached `fetch_weights` task) — diffusers loads straight from
    it, skipping any HuggingFace hit. Otherwise we pull `spec.repo` from the hub.

    Resolution order for the pipeline class:
      1. `getattr(diffusers, spec.pipeline)` — the intended class.
      2. `AutoPipelineForText2Image` — works for anything with a registered
         mapping (SDXL, SD3, FLUX, most others).
      3. `DiffusionPipeline` — last resort; reads the repo's own config.
    """
    import diffusers

    dtype = _torch_dtype(spec.dtype)
    source = model_path or spec.repo
    # A local path needs no token; the hub might (gated repos).
    load_kwargs = dict(torch_dtype=dtype)
    if model_path is None:
        load_kwargs["token"] = os.environ.get("HF_TOKEN")

    cls = getattr(diffusers, spec.pipeline, None)
    tried = []
    for candidate in (cls, getattr(diffusers, "AutoPipelineForText2Image", None),
                      getattr(diffusers, "DiffusionPipeline", None)):
        if candidate is None:
            continue
        try:
            print(f"[imagegen] loading {source} via {candidate.__name__}", flush=True)
            pipe = candidate.from_pretrained(source, **load_kwargs)
            break
        except Exception as e:  # class exists but can't load this repo → try next
            tried.append(f"{candidate.__name__}: {type(e).__name__}: {e}")
    else:
        raise RuntimeError(
            f"Could not load {source} with any pipeline class. Tried:\n  "
            + "\n  ".join(tried)
        )

    if os.environ.get("IMAGEGEN_OFFLOAD") == "1" and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # Cheap VRAM wins that don't change outputs.
    for opt in ("enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, opt):
            try:
                getattr(pipe, opt)()
            except Exception:
                pass
    return pipe


# ── Generating ──────────────────────────────────────────────────────────────────

@dataclass
class GenResult:
    """One (model, prompt) generation, ready to drop into a report."""
    model_key: str
    prompt: str
    seconds: float
    data_uri: str = ""          # PNG data URI for embedding in HTML/Gradio
    error: str = ""             # non-empty if generation failed


def generate(
    pipe,
    spec: ModelSpec,
    prompt: str,
    *,
    steps: int | None = None,
    guidance: float | None = None,
    seed: int = -1,
    width: int | None = None,
    height: int | None = None,
    negative_prompt: str | None = None,
):
    """Run one generation and return a PIL.Image. Sampler args default off `spec`."""
    import torch

    steps = spec.steps if steps is None else steps
    guidance = spec.guidance if guidance is None else guidance
    width = spec.width if width is None else width
    height = spec.height if height is None else height

    generator = None
    if seed is not None and seed >= 0:
        # Generator device must match the compute device (cuda), except under CPU
        # offload where the pipe expects a cpu generator.
        gen_device = "cpu" if os.environ.get("IMAGEGEN_OFFLOAD") == "1" else "cuda"
        generator = torch.Generator(device=gen_device).manual_seed(int(seed))

    kwargs = dict(
        prompt=prompt,
        num_inference_steps=int(steps),
        width=int(width),
        height=int(height),
        generator=generator,
    )
    if guidance is not None:
        kwargs["guidance_scale"] = float(guidance)
    if spec.supports_negative and negative_prompt:
        kwargs["negative_prompt"] = negative_prompt
    if spec.max_sequence_length is not None:
        kwargs["max_sequence_length"] = spec.max_sequence_length

    out = pipe(**kwargs)
    return out.images[0]


def pil_to_data_uri(img, fmt: str = "JPEG", max_side: int | None = None,
                    quality: int = 85) -> str:
    """Encode a PIL image as a base64 data URI. `max_side` downscales for reports.

    Defaults to JPEG: a base64 PNG of a 768px photo is ~1MB, so a multi-image
    grid becomes several MB in one HTML doc and the Flyte console's report iframe
    struggles to render it. JPEG at 512px is ~10x smaller and loads instantly;
    full-resolution PNGs are still saved to the task's output Dir.
    """
    if max_side:
        w, h = img.size
        if max(w, h) > max_side:
            scale = max_side / float(max(w, h))
            img = img.resize((int(w * scale), int(h * scale)))
    buf = io.BytesIO()
    fmt = fmt.upper()
    if fmt in ("JPEG", "JPG"):
        img.convert("RGB").save(buf, format="JPEG", quality=quality, optimize=True)
        mime = "image/jpeg"
    else:
        img.save(buf, format=fmt)
        mime = "image/png" if fmt == "PNG" else f"image/{fmt.lower()}"
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:{mime};base64,{b64}"


# ── Reports ─────────────────────────────────────────────────────────────────────

REPORT_CSS = """
<style>
  .ig-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
  .ig-wrap h2 { margin: 0 0 4px; }
  .ig-meta { color: #6b7280; font-size: 13px; margin-bottom: 16px; }
  .ig-grid { display: grid; gap: 14px; overflow-x: auto;
             grid-auto-flow: column; grid-auto-columns: minmax(260px, 1fr); }
  .ig-cell { border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden;
             background: #fff; display: flex; flex-direction: column; }
  .ig-cell img { width: 100%; height: auto; display: block; background: #f3f4f6; }
  .ig-cap { padding: 8px 10px; }
  .ig-model { font-weight: 600; font-size: 14px; }
  .ig-sub { color: #6b7280; font-size: 12px; margin-top: 2px; line-height: 1.35; }
  .ig-fam { display: inline-block; font-size: 11px; color: #374151;
            background: #f3f4f6; border-radius: 999px; padding: 1px 8px; margin-top: 4px; }
  .ig-err { padding: 16px; color: #b91c1c; font-size: 13px; white-space: pre-wrap; }
  .ig-prompt { background: #f9fafb; border-left: 3px solid #6366f1; padding: 8px 12px;
               border-radius: 6px; margin: 6px 0 14px; font-size: 14px; }
</style>
"""


def _cell_html(spec: ModelSpec, r: GenResult) -> str:
    header = (
        f'<div class="ig-cap"><div class="ig-model">{html.escape(spec.key)}</div>'
        f'<div class="ig-fam">{html.escape(spec.family)}</div>'
    )
    if r.error:
        body = f'<div class="ig-err">⚠️ {html.escape(r.error)}</div>'
        sub = ""
    else:
        body = f'<img src="{r.data_uri}" alt="{html.escape(spec.key)}"/>'
        sub = (f'<div class="ig-sub">{r.seconds:.1f}s · {spec.steps} steps · '
               f'{spec.license}</div>')
    return f'<div class="ig-cell">{body}{header}{sub}</div></div>'


def render_comparison(
    prompt: str,
    specs: list[ModelSpec],
    results: list[GenResult],
    *,
    meta: str = "",
) -> str:
    """One prompt across many models, side by side (models are the columns)."""
    by_key = {r.model_key: r for r in results}
    cells = "".join(
        _cell_html(s, by_key.get(s.key, GenResult(s.key, prompt, 0.0, error="no result")))
        for s in specs
    )
    return (
        f'{REPORT_CSS}<div class="ig-wrap"><h2>Model comparison</h2>'
        f'<div class="ig-meta">{html.escape(meta)}</div>'
        f'<div class="ig-prompt">🖊️ {html.escape(prompt)}</div>'
        f'<div class="ig-grid">{cells}</div></div>'
    )


def render_grid(
    prompts: list[str],
    specs: list[ModelSpec],
    results: list[GenResult],
    *,
    meta: str = "",
) -> str:
    """Full prompt×model grid: a comparison block stacked per prompt."""
    by_pair = {(r.prompt, r.model_key): r for r in results}
    blocks = []
    for p in prompts:
        by_key = {s.key: by_pair.get((p, s.key), GenResult(s.key, p, 0.0, error="no result"))
                  for s in specs}
        cells = "".join(_cell_html(s, by_key[s.key]) for s in specs)
        blocks.append(
            f'<div class="ig-prompt">🖊️ {html.escape(p)}</div>'
            f'<div class="ig-grid">{cells}</div>'
        )
    header = (f'{REPORT_CSS}<div class="ig-wrap"><h2>Image model comparison</h2>'
              f'<div class="ig-meta">{html.escape(meta)}</div>')
    return header + "".join(blocks) + "</div>"


def render_before_after(
    prompt: str,
    before: GenResult,
    after: GenResult,
    *,
    meta: str = "",
) -> str:
    """LoRA report: base model vs LoRA-tuned, side by side."""
    def one(title, r):
        if r.error:
            body = f'<div class="ig-err">⚠️ {html.escape(r.error)}</div>'
        else:
            body = f'<img src="{r.data_uri}" alt="{title}"/>'
        return (f'<div class="ig-cell">{body}<div class="ig-cap">'
                f'<div class="ig-model">{html.escape(title)}</div>'
                f'<div class="ig-sub">{r.seconds:.1f}s</div></div></div>')
    grid = one("base (no LoRA)", before) + one("LoRA-tuned", after)
    return (
        f'{REPORT_CSS}<div class="ig-wrap"><h2>LoRA fine-tune: before vs after</h2>'
        f'<div class="ig-meta">{html.escape(meta)}</div>'
        f'<div class="ig-prompt">🖊️ {html.escape(prompt)}</div>'
        f'<div class="ig-grid">{grid}</div></div>'
    )


def timed_generate(pipe, spec, prompt, **kw) -> tuple:
    """generate() wrapped with wall-clock timing; returns (PIL image, seconds)."""
    t0 = time.time()
    img = generate(pipe, spec, prompt, **kw)
    return img, time.time() - t0
