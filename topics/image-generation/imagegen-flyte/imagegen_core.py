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
import math
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
    elif getattr(spec, "quantized", False):
        # Pre-quantized (bitsandbytes 4-bit) pipeline: the 4-bit weights are already
        # placed on the GPU at load. `.to("cuda")` is the documented way to move the
        # remaining fp modules, but some bnb builds reject `.to()` on a 4-bit model,
        # so fall back to cpu-offload if it complains. (Unverified on-GPU yet.)
        try:
            pipe = pipe.to(device)
        except (ValueError, RuntimeError) as e:
            print(f"[imagegen] {spec.key}: .to({device}) rejected on 4-bit pipe "
                  f"({type(e).__name__}); using cpu offload", flush=True)
            if hasattr(pipe, "enable_model_cpu_offload"):
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


def free_gpu_memory() -> None:
    """Hand a just-dropped pipeline's memory back to the allocator/OS.

    Call this right after you drop your last reference to a pipeline (`pipe =
    None`) so the next model starts from a clean slate. On the GB10's *unified*
    memory there's no separate VRAM pool to move weights out of (CPU and GPU
    share the same 128GB), so `.to("cpu")` frees nothing; the reclaim comes from
    releasing the Python objects and then returning PyTorch's cached blocks with
    `empty_cache()`. Best-effort and import-safe: does nothing if torch/CUDA is
    absent.
    """
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass


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
  /* Wrap cells to the next row when they run out of width, instead of one long
     horizontally-scrolling row. auto-fill packs as many >=260px columns as fit. */
  .ig-grid { display: grid; gap: 14px;
             grid-template-columns: repeat(auto-fill, minmax(260px, 1fr)); }
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
  .ig-prompt code { font-size: 13px; background: none; padding: 0; }
  /* The trigger, highlighted in place: you can see at a glance whether the
     prompt can actually fire the adapter. */
  .ig-trig { background: #fde68a; border-radius: 3px; padding: 0 3px; font-weight: 600; }
  .ig-notrig { color: #b91c1c; font-weight: 600; }
  .ig-sampler { color: #6b7280; font-size: 12px; margin-top: 6px; }
  /* Click-to-zoom lightbox: thumbnails are small so many models fit; click one
     to see it big. Pure inline handlers (no <script>) so it works whether the
     console loads the report as a document or injects it via innerHTML. */
  .ig-cell img.ig-zoom { cursor: zoom-in; }
  #ig-lb { position: fixed; inset: 0; z-index: 9999; display: none; cursor: zoom-out;
           flex-direction: column; align-items: center; justify-content: center;
           gap: 12px; padding: 24px; background: rgba(0,0,0,.85); }
  #ig-lb img { max-width: 92vw; max-height: 82vh; border-radius: 8px;
               box-shadow: 0 8px 40px rgba(0,0,0,.5); }
  #ig-lb #ig-lb-cap { color: #e5e7eb; font-size: 14px;
                      font-family: system-ui, -apple-system, sans-serif; }
</style>
"""

# Opening the lightbox: set the big image + caption from the clicked thumbnail,
# then show the overlay. No user data in the handler itself, so no escaping needed.
_ZOOM_ONCLICK = (
    "document.getElementById('ig-lb-img').src=this.src;"
    "document.getElementById('ig-lb-cap').textContent=this.dataset.cap;"
    "document.getElementById('ig-lb').style.display='flex'"
)

# The overlay itself; click anywhere on it to close. Appended once per report.
_LIGHTBOX = (
    "<div id=\"ig-lb\" onclick=\"this.style.display='none'\" style=\"display:none\">"
    '<img id="ig-lb-img" src="" alt="zoomed"/><div id="ig-lb-cap"></div></div>'
)


def _zoom_img(data_uri: str, alt: str, cap: str) -> str:
    """An <img> that expands into the lightbox on click."""
    return (
        f'<img class="ig-zoom" src="{data_uri}" alt="{html.escape(alt)}" '
        f'data-cap="{html.escape(cap, quote=True)}" onclick="{_ZOOM_ONCLICK}"/>'
    )


def _cell_html(spec: ModelSpec, r: GenResult) -> str:
    header = (
        f'<div class="ig-cap"><div class="ig-model">{html.escape(spec.key)}</div>'
        f'<div class="ig-fam">{html.escape(spec.family)}</div>'
    )
    if r.error:
        body = f'<div class="ig-err">⚠️ {html.escape(r.error)}</div>'
        sub = ""
    else:
        body = _zoom_img(r.data_uri, spec.key, f"{spec.key} · {r.prompt} · {r.seconds:.1f}s")
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
        f'<div class="ig-grid">{cells}</div></div>{_LIGHTBOX}'
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
    return header + "".join(blocks) + "</div>" + _LIGHTBOX


def _lora_cell(title: str, r: GenResult) -> str:
    if r.error:
        body = f'<div class="ig-err">⚠️ {html.escape(r.error)}</div>'
    else:
        body = _zoom_img(r.data_uri, title, f"{title} · {r.prompt} · {r.seconds:.1f}s")
    return (f'<div class="ig-cell">{body}<div class="ig-cap">'
            f'<div class="ig-model">{html.escape(title)}</div>'
            f'<div class="ig-sub">{r.seconds:.1f}s</div></div></div>')


def _highlight_trigger(prompt: str, trigger: str) -> str:
    """Escape `prompt`, marking the trigger phrase, or flagging its absence.

    Which prompt reached the model is the first thing you check when a LoRA looks
    inert, so the report shows the exact string, not a paraphrase of it.
    """
    if not trigger:
        return html.escape(prompt)
    i = prompt.lower().find(trigger.lower())
    if i < 0:
        return (f"{html.escape(prompt)} "
                f'<span class="ig-notrig">← no trigger \'{html.escape(trigger)}\'</span>')
    head, hit, tail = prompt[:i], prompt[i:i + len(trigger)], prompt[i + len(trigger):]
    return (f'{html.escape(head)}<span class="ig-trig">{html.escape(hit)}</span>'
            f"{html.escape(tail)}")


def _lora_block(
    prompt: str,
    before: GenResult | None,
    after: GenResult,
    trigger: str = "",
    sampler: str = "",
) -> str:
    """One prompt's row. `before=None` renders the LoRA output on its own."""
    cells = _lora_cell("base (no LoRA)", before) if before else ""
    cells += _lora_cell("LoRA-tuned" if before else "generated", after)
    note = (
        "both columns: the identical prompt above, the same seed, one "
        "<code>load_lora_weights</code> call between them"
        if before else "prompt above, LoRA applied"
    )
    return (f'<div class="ig-prompt">🖊️ <code>{_highlight_trigger(prompt, trigger)}</code>'
            f'<div class="ig-sampler">{note}'
            + (f" · {html.escape(sampler)}" if sampler else "")
            + f'</div></div><div class="ig-grid">{cells}</div>')


def render_before_after(
    prompt: str,
    before: GenResult,
    after: GenResult,
    *,
    meta: str = "",
) -> str:
    """LoRA report: base model vs LoRA-tuned, side by side, for one prompt."""
    return (
        f'{REPORT_CSS}<div class="ig-wrap"><h2>LoRA fine-tune: before vs after</h2>'
        f'<div class="ig-meta">{html.escape(meta)}</div>'
        f'{_lora_block(prompt, before, after)}</div>{_LIGHTBOX}'
    )


def render_lora_report(
    rows: list[tuple[str, GenResult | None, GenResult]],
    *,
    title: str = "LoRA fine-tune: before vs after",
    meta: str = "",
    trigger: str = "",
    sampler: str = "",
) -> str:
    """Same thing stacked over several prompts: one base/tuned pair per row.

    One prompt can flatter a LoRA by accident. Three make it obvious whether the
    adapter learned a style or just memorized a picture.

    A row whose `before` is None renders the tuned image alone, which is what the
    plain generate path (show_base=False) emits.
    """
    blocks = "".join(
        _lora_block(p, before, after, trigger, sampler) for p, before, after in rows
    )
    return (
        f'{REPORT_CSS}<div class="ig-wrap"><h2>{html.escape(title)}</h2>'
        f'<div class="ig-meta">{html.escape(meta)}</div>'
        f'{blocks}</div>{_LIGHTBOX}'
    )


def timed_generate(pipe, spec, prompt, **kw) -> tuple:
    """generate() wrapped with wall-clock timing; returns (PIL image, seconds)."""
    t0 = time.time()
    img = generate(pipe, spec, prompt, **kw)
    return img, time.time() - t0


# ── Training report ─────────────────────────────────────────────────────────────
#
# Inline SVG, no <script> and no external assets: the Flyte console renders the
# report either as a document or via innerHTML, and a CSP would drop anything
# fetched. Hover text rides on SVG <title>, which the browser shows natively.
#
# Colors are the dataviz reference palette, validated (not eyeballed):
#   - the loss curve is an EMPHASIS chart, not two categorical series. Raw loss is
#     context (muted gray), its EMA is the point (accent blue). Accent-vs-muted
#     separates by ΔE >= 33 under protan/deutan/tritan.
#   - sigma buckets are ORDERED categories, so they take the one-hue ordinal ramp
#     (validated: monotone L, adjacent ΔL 0.19, single hue, pale end 2.06:1). The
#     pale end only just clears the 2:1 floor, so every bar carries a direct label.

_ACCENT = "#2a78d6"      # categorical slot 1 / sequential 450
_MUTED = "#898781"       # de-emphasis + axis labels
_GRID = "#e1e0d9"
_AXIS = "#c3c2b7"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_RAMP = ("#86b6ef", "#2a78d6", "#104281")   # ordinal 250 / 450 / 650

TRAIN_CSS = """
<style>
  .tr-wrap { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
             color: #0b0b0b; }
  .tr-wrap h2 { margin: 0 0 2px; }
  .tr-meta { color: #52514e; font-size: 13px; margin-bottom: 14px; }
  .tr-kpis { display: grid; gap: 10px; margin: 0 0 16px;
             grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }
  .tr-kpi { border: 1px solid #e5e7eb; border-radius: 10px; padding: 8px 10px;
            background: #fcfcfb; }
  .tr-kpi .k { font-size: 11px; color: #898781; text-transform: uppercase;
               letter-spacing: .04em; }
  .tr-kpi .v { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums;
               margin-top: 2px; }
  .tr-track { height: 8px; border-radius: 999px; background: #e1e0d9;
              overflow: hidden; margin: 0 0 18px; }
  .tr-fill { height: 100%; background: #2a78d6; border-radius: 999px; }
  .tr-card { border: 1px solid #e5e7eb; border-radius: 12px; padding: 12px 14px;
             background: #fcfcfb; margin-bottom: 14px; }
  .tr-card h3 { margin: 0 0 2px; font-size: 14px; }
  .tr-sub { color: #898781; font-size: 12px; margin: 0 0 8px; }
  .tr-legend { display: flex; gap: 14px; font-size: 12px; color: #52514e;
               margin: 6px 0 0; }
  .tr-legend span.sw { display: inline-block; width: 10px; height: 2px;
                       vertical-align: middle; margin-right: 5px; }
  .tr-wrap svg { width: 100%; height: auto; display: block; }
  .tr-wrap details { font-size: 12px; color: #52514e; margin-top: 6px; }
  .tr-wrap table { border-collapse: collapse; margin-top: 6px; font-size: 12px; }
  .tr-wrap th, .tr-wrap td { border: 1px solid #e1e0d9; padding: 2px 8px;
                             text-align: right; font-variant-numeric: tabular-nums; }
  .tr-thumbs img { height: 76px; border-radius: 6px; margin: 2px; }
  /* Callout for the one thing a reader will otherwise misread. */
  .tr-note { background: #eef4fd; border-left: 3px solid #2a78d6; padding: 8px 12px;
             border-radius: 4px; margin: 8px 0 10px; font-size: 12.5px; color: #52514e; }
  .tr-badge { display: inline-block; padding: 1px 8px; border-radius: 999px;
              font-size: 11px; font-weight: 600; background: #f3f4f6; color: #374151;
              margin-right: 6px; }
  .tr-stages { display: flex; gap: 26px; margin: 4px 0 16px; }
  .tr-stage { text-align: center; font-size: 12px; }
  .tr-stage .dot { font-size: 15px; line-height: 1.1; }
</style>
"""


def _nice_ticks(lo: float, hi: float, n: int = 4) -> list[float]:
    if not (hi > lo):
        return [lo]
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(m * mag for m in (1, 2, 2.5, 5, 10) if raw <= m * mag)
    out, v = [], math.floor(lo / step) * step
    while v <= hi + step * 1e-9:
        if v >= lo - step * 1e-9:
            out.append(round(v, 10))
        v += step
    return out


def _svg_lines(
    series: list[dict],
    max_steps: int,
    *,
    aria: str,
    height: int = 240,
    fmt: str = "{:.3f}",
) -> str:
    """One or more step-indexed lines on ONE y axis.

    `series` items: {name, color, points: [(step, value)], opacity?, end_label?}.
    Two measures of different scale get two calls, never two y axes: a dual axis
    invents whatever correlation its arbitrary alignment implies.
    """
    if not series or len(series[0]["points"]) < 2:
        return '<p class="tr-sub">Collecting the first datapoints…</p>'

    W, H = 720, height
    L, R, T, B = 46, 58, 10, 26          # right pad leaves room for the end label
    pw, ph = W - L - R, H - T - B

    ys = [v for s in series for _, v in s["points"]]
    lo, hi = min(ys), max(ys)
    pad = (hi - lo) * 0.12 or (abs(hi) * 0.1 or 0.01)
    lo, hi = lo - pad, hi + pad
    ticks = _nice_ticks(lo, hi)

    sx = lambda s: L + pw * (s / max(max_steps, 1))
    sy = lambda v: T + ph * (1 - (v - lo) / (hi - lo))

    grid = "".join(
        f'<line x1="{L}" y1="{sy(t):.1f}" x2="{L+pw}" y2="{sy(t):.1f}" '
        f'stroke="{_GRID}" stroke-width="1"/>'
        f'<text x="{L-8}" y="{sy(t)+4:.1f}" text-anchor="end" font-size="11" '
        f'fill="{_MUTED}">{fmt.format(t)}</text>'
        for t in ticks
    )
    xaxis = "".join(
        f'<text x="{sx(s):.1f}" y="{T+ph+18}" text-anchor="middle" font-size="11" '
        f'fill="{_MUTED}">{s}</text>'
        for s in ([0, max_steps // 2, max_steps] if max_steps else [0])
    )

    body = []
    for s in series:
        pts = " ".join(f"{sx(x):.1f},{sy(v):.1f}" for x, v in s["points"])
        body.append(
            f'<polyline points="{pts}" fill="none" stroke="{s["color"]}" '
            f'stroke-width="2" stroke-linejoin="round" stroke-linecap="round"'
            + (f' opacity="{s["opacity"]}"' if s.get("opacity") else "")
            + "/>"
        )
    for s in series:
        if not s.get("end_label"):
            continue
        lx, lv = s["points"][-1]
        body.append(
            f'<circle cx="{sx(lx):.1f}" cy="{sy(lv):.1f}" r="4" fill="{s["color"]}"/>'
            f'<text x="{sx(lx)+8:.1f}" y="{sy(lv)+4:.1f}" font-size="12" '
            f'font-weight="600" fill="{_INK}">{fmt.format(lv)}</text>'
        )
        # Invisible fat hit targets: the native <title> tooltip needs no JS.
        body.append("".join(
            f'<circle cx="{sx(x):.1f}" cy="{sy(v):.1f}" r="9" fill="transparent">'
            f"<title>step {x} · {s['name']} {fmt.format(v)}</title></circle>"
            for x, v in s["points"]
        ))

    # A single series needs no legend box; the card title names it.
    legend = ""
    if len(series) > 1:
        legend = '<div class="tr-legend">' + "".join(
            f'<span><span class="sw" style="background:{s["color"]}'
            + (f';opacity:{s["opacity"]}' if s.get("opacity") else "")
            + f'"></span>{html.escape(s["name"])}</span>'
            for s in series
        ) + "</div>"

    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(aria)}">{grid}'
        f'<line x1="{L}" y1="{T+ph}" x2="{L+pw}" y2="{T+ph}" stroke="{_AXIS}" '
        f'stroke-width="1"/>{xaxis}' + "".join(body) + "</svg>" + legend
    )


def _stage_indicator(stages: list[str], current: int) -> str:
    """Where in the pipeline we are. Mirrors the tutorials' step indicator."""
    out = []
    for i, label in enumerate(stages):
        if i < current:
            dot, color = "&#10003;", "#1baf7a"      # check
        elif i == current:
            dot, color = "&#9679;", _ACCENT          # filled
        else:
            dot, color = "&#9675;", _AXIS            # hollow
        out.append(
            f'<div class="tr-stage"><div class="dot" style="color:{color}">{dot}</div>'
            f'<div style="color:{color}">{html.escape(label)}</div></div>'
        )
    return f'<div class="tr-stages">{"".join(out)}</div>'


def _svg_sigma_bars(buckets: list[tuple[str, float, int]]) -> str:
    """Mean loss per noise-level bucket. Ordered categories, so: ordinal ramp."""
    rows = [b for b in buckets if b[2] > 0]
    if not rows:
        return '<p class="tr-sub">No samples yet.</p>'

    W = 720
    bar_h, gap, L, R = 26, 10, 172, 66
    H = len(rows) * (bar_h + gap) + 6
    pw = W - L - R
    top = max(v for _, v, _ in rows)

    out = []
    for i, (label, val, n) in enumerate(rows):
        y = i * (bar_h + gap)
        w = max(2.0, pw * (val / top if top else 0))
        color = _RAMP[min(i, len(_RAMP) - 1)]
        out.append(
            f'<text x="{L-10}" y="{y+bar_h/2+4:.0f}" text-anchor="end" font-size="12" '
            f'fill="{_INK2}">{html.escape(label)}</text>'
            f'<rect x="{L}" y="{y}" width="{w:.1f}" height="{bar_h}" rx="4" fill="{color}">'
            f"<title>{html.escape(label)} · mean loss {val:.4f} · {n} steps</title></rect>"
            # direct value label: the pale step only clears 2.06:1 on this surface,
            # so the number must not depend on reading the bar's color.
            f'<text x="{L+w+8:.1f}" y="{y+bar_h/2+4:.0f}" font-size="12" '
            f'font-weight="600" fill="{_INK}">{val:.3f}</text>'
        )
    return (
        f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Mean loss by noise level">'
        + "".join(out) + "</svg>"
    )


def _kpi(label: str, value: str) -> str:
    return (f'<div class="tr-kpi"><div class="k">{html.escape(label)}</div>'
            f'<div class="v">{html.escape(value)}</div></div>')


def render_training_report(
    *,
    title: str,
    meta: str,
    step: int,
    max_steps: int,
    kpis: list[tuple[str, str]],
    history: list[tuple[int, float, float, float]],
    sigma_buckets: list[tuple[str, float, int]],
    badges: list[str] = (),
    stages: list[str] = (),
    stage: int = 0,
    thumbs_html: str = "",
    footer: str = "",
) -> str:
    """The live training report: stages, KPIs, loss, grad norm, loss-by-noise-level.

    `history` is [(step, raw_loss, ema_loss, grad_norm)]; `sigma_buckets` is
    [(label, mean_loss, n_steps)] ordered low noise -> high noise.

    Loss and grad norm get their own charts rather than a shared plot with two y
    axes: a dual axis would imply whatever correlation its arbitrary alignment
    happens to draw.

    The sigma breakdown is not decoration. Flow-matching loss is dominated by
    which timestep got sampled, so the raw curve looks flat even while the model
    learns. Splitting by noise level is what makes the signal legible.
    """
    pct = 100.0 * step / max(max_steps, 1)
    table = "".join(
        f"<tr><td>{s}</td><td>{r:.4f}</td><td>{e:.4f}</td><td>{g:.3f}</td></tr>"
        for s, r, e, g in history
    )
    loss_chart = _svg_lines(
        [
            {"name": "raw loss (per 25 steps)", "color": _MUTED, "opacity": 0.55,
             "points": [(s, r) for s, r, _, _ in history]},
            {"name": "EMA (the trend)", "color": _ACCENT, "end_label": True,
             "points": [(s, e) for s, _, e, _ in history]},
        ],
        max_steps, aria="Training loss over steps",
    )
    grad_chart = _svg_lines(
        [{"name": "grad norm", "color": _ACCENT, "end_label": True,
          "points": [(s, g) for s, _, _, g in history]}],
        max_steps, aria="Gradient norm over steps", height=160,
    )
    return (
        f'{TRAIN_CSS}<div class="tr-wrap">'
        f"<h2>{html.escape(title)}</h2>"
        f'<div class="tr-meta">{html.escape(meta)}</div>'
        + ("".join(f'<span class="tr-badge">{html.escape(b)}</span>' for b in badges)
           if badges else "")
        + (_stage_indicator(list(stages), stage) if stages else "")
        + (f'<div class="tr-thumbs">{thumbs_html}</div>' if thumbs_html else "")
        + f'<div class="tr-kpis">{"".join(_kpi(k, v) for k, v in kpis)}</div>'
        f'<div class="tr-track"><div class="tr-fill" style="width:{pct:.1f}%"></div></div>'
        f'<div class="tr-note"><b>Reading this report:</b> a flat loss curve is '
        f'expected here and does not mean the run is broken. Flow-matching loss '
        f'depends far more on which noise level each step randomly drew than on how '
        f'training is going, so look at the EMA and at the per-sigma breakdown below.'
        f"</div>"
        f'<div class="tr-card"><h3>Loss</h3>'
        f'<p class="tr-sub">Raw loss is noisy by construction; watch the EMA.</p>'
        f"{loss_chart}"
        f"<details><summary>Table view</summary><table>"
        f"<tr><th>step</th><th>loss</th><th>EMA</th><th>grad norm</th></tr>"
        f"{table}</table></details>"
        f"</div>"
        f'<div class="tr-card"><h3>Gradient norm (before clipping at 1.0)</h3>'
        f'<p class="tr-sub">The stability trace. A steady band is healthy; repeated '
        f'spikes into the clip threshold mean the learning rate is too high.</p>'
        f"{grad_chart}</div>"
        f'<div class="tr-card"><h3>Mean loss by noise level (sigma)</h3>'
        f'<p class="tr-sub">Each step draws a random sigma. Near-noise steps are '
        f'simply harder, forever, so this split is where learning is actually legible.</p>'
        f"{_svg_sigma_bars(sigma_buckets)}</div>"
        + (f'<div class="tr-meta">{footer}</div>' if footer else "")
        + "</div>"
    )
