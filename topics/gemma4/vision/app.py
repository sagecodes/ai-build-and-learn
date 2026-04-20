"""
Gemma 4 vision demo: upload an image, ask a question.

Two modes:
  - Ask: free-form Q&A about the image (describe, count, read text, reason).
  - Detect: model returns normalized bounding boxes we draw on the image.

Gemma 4 is a VLM — its bounding boxes are an emergent capability, not a
dedicated detector head. Quality varies, especially on small objects. For
production detection use OWLv2 / Grounding DINO / YOLO-World.

Run (after `uv venv` + `uv pip install -r requirements.txt` + activating):
    ollama serve &
    ollama pull gemma4:31b
    python app.py

Swap model sizes without editing code:
    GEMMA_MODEL=gemma4:4b python app.py
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

import gradio as gr
import ollama
from PIL import Image, ImageDraw, ImageFont, ImageOps

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:31b")

PRESET_PROMPTS = [
    "Describe this image in detail.",
    "List every object you can see, roughly where it is in the frame.",
    "Read any text visible in the image.",
    "What's unusual or unsafe about this scene?",
    "Count the people / animals / vehicles visible.",
]

# Palette for bbox overlay. Tab10 — distinguishable at small sizes.
# Each entry is (r, g, b). We apply alpha at draw time.
BBOX_COLORS = [
    (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
    (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
    (188, 189, 34), (23, 190, 207),
]
BBOX_FILL_ALPHA = 90   # 0-255; ~35% opacity
BBOX_OUTLINE_ALPHA = 255


def _prepare_image(image_path: str) -> str:
    """Apply EXIF orientation and write a temp file we can pass to Ollama.

    Phones embed rotation in EXIF metadata; Ollama hands the raw file to the
    model without applying it, so the model sees the image sideways/upside-down.
    Pre-rotating here keeps both the model's view and our drawing consistent
    with what the user sees.
    """
    img = ImageOps.exif_transpose(Image.open(image_path))
    out = Path(tempfile.mkdtemp()) / "input.png"
    img.convert("RGB").save(out, "PNG")
    return str(out)


def list_vision_models() -> list[str]:
    """Return installed ollama models that look like gemma4 (all gemma4 variants are VLMs)."""
    try:
        resp = ollama.list()
        names = [m.model for m in resp.models]
        gemma = sorted(n for n in names if n and n.startswith("gemma4"))
        return gemma or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def ask(image_path: str | None, question: str, model: str, temperature: float):
    """Stream the answer. image_path comes from gr.Image(type='filepath')."""
    if not image_path:
        yield "Upload an image first."
        return
    if not question.strip():
        yield "Ask a question about the image."
        return

    print(f"[ask] model={model} q={question[:60]!r}", flush=True)
    yield "Thinking..."

    prepared = _prepare_image(image_path)
    stream = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": question,
            "images": [prepared],
        }],
        stream=True,
        options={"temperature": float(temperature), "num_ctx": 8192},
    )

    partial = ""
    for chunk in stream:
        partial += chunk["message"]["content"]
        if partial:  # skip empty-content chunks so "Thinking..." stays until real tokens arrive
            yield partial
    print(f"[ask] done ({len(partial)} chars)", flush=True)


# ---------------------------------------------------------------------------
# Detection mode: ask Gemma 4 for normalized bounding boxes, draw them.
# ---------------------------------------------------------------------------

DETECT_PROMPT = (
    "Detect {target} in this image. Respond with ONLY a JSON array (no prose, "
    "no markdown fence). Each detection is "
    '{{"label": "<short name>", "box_2d": [ymin, xmin, ymax, xmax]}} '
    "where coords are normalized 0–1000 (y axis down). Skip duplicates."
)


def _extract_json_array(text: str) -> list[dict]:
    """Pull the first JSON array out of the model output. Tolerates markdown fences."""
    # Strip ```json ... ``` fences if present.
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    # Find the outermost [...] block.
    start = candidate.find("[")
    end = candidate.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return []
    try:
        data = json.loads(candidate[start:end + 1])
    except json.JSONDecodeError:
        return []
    if not isinstance(data, list):
        return []
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        box = item.get("box_2d") or item.get("bbox") or item.get("box")
        label = item.get("label") or item.get("name") or ""
        if isinstance(box, list) and len(box) == 4:
            out.append({"label": str(label), "box_2d": [float(v) for v in box]})
    return out


def _draw_boxes(base: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw translucent filled bboxes + solid outlines + labels.

    `base` is the already-EXIF-corrected image so coords line up with what
    the user sees.
    """
    img = base.convert("RGBA")
    w, h = img.size

    # Separate overlay for translucent fills — we alpha-composite at the end.
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fill_draw = ImageDraw.Draw(overlay)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            size=max(14, int(min(w, h) * 0.025)),
        )
    except OSError:
        font = ImageFont.load_default()

    line_w = max(2, int(min(w, h) * 0.004))

    # Pass 1: translucent fills on the overlay.
    for i, det in enumerate(detections):
        ymin, xmin, ymax, xmax = det["box_2d"]
        x1, y1 = int(xmin / 1000 * w), int(ymin / 1000 * h)
        x2, y2 = int(xmax / 1000 * w), int(ymax / 1000 * h)
        r, g, b = BBOX_COLORS[i % len(BBOX_COLORS)]
        fill_draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, BBOX_FILL_ALPHA))

    img = Image.alpha_composite(img, overlay)

    # Pass 2: solid outlines + labels on the composited image.
    draw = ImageDraw.Draw(img)
    for i, det in enumerate(detections):
        ymin, xmin, ymax, xmax = det["box_2d"]
        x1, y1 = int(xmin / 1000 * w), int(ymin / 1000 * h)
        x2, y2 = int(xmax / 1000 * w), int(ymax / 1000 * h)
        r, g, b = BBOX_COLORS[i % len(BBOX_COLORS)]
        color = (r, g, b, BBOX_OUTLINE_ALPHA)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        label = det["label"]
        if label:
            tb = draw.textbbox((x1, y1), label, font=font)
            pad = 3
            draw.rectangle(
                [tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad],
                fill=color,
            )
            draw.text((x1, y1), label, fill=(255, 255, 255, 255), font=font)

    return img.convert("RGB")


def detect(image_path: str | None, target: str, model: str, temperature: float):
    """Returns (annotated_image, detections_json, raw_model_output)."""
    if not image_path:
        return None, "Upload an image first.", ""
    target = target.strip() or "the main objects"

    # Apply EXIF orientation once — the model sees the same image we draw on.
    base = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    tmp_path = Path(tempfile.mkdtemp()) / "detect_input.png"
    base.save(tmp_path, "PNG")

    print(f"[detect] target={target!r}", flush=True)
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": DETECT_PROMPT.format(target=target),
            "images": [str(tmp_path)],
        }],
        options={"temperature": float(temperature), "num_ctx": 8192},
    )
    raw = resp["message"]["content"]
    detections = _extract_json_array(raw)
    print(f"[detect] parsed {len(detections)} boxes", flush=True)

    if not detections:
        return base, "No detections parsed.", raw

    annotated = _draw_boxes(base, detections)
    pretty = json.dumps(detections, indent=2)
    return annotated, pretty, raw


def build_ui() -> gr.Blocks:
    models = list_vision_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Blocks(title="Gemma 4 Vision") as demo:
        gr.Markdown(
            "# Gemma 4 Vision\n"
            "Upload an image. **Ask** mode: free-form Q&A. **Detect** mode: "
            "Gemma 4 returns normalized bounding boxes that we draw on the image. "
            "Detection is emergent (not a trained head) — quality varies."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="Model", scale=2)
            temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature")

        with gr.Tabs():
            # --- Ask tab -------------------------------------------------
            with gr.Tab("Ask"):
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(type="filepath", label="Image", height=400)
                        question = gr.Textbox(
                            label="Question", lines=2,
                            placeholder="e.g. What's happening in this photo?",
                        )
                        preset = gr.Radio(PRESET_PROMPTS, label="Preset prompts", value=None)
                        submit = gr.Button("Ask", variant="primary")
                    with gr.Column():
                        answer = gr.Textbox(label="Answer", lines=20)

                preset.change(lambda p: p or "", inputs=preset, outputs=question)
                submit.click(ask, inputs=[image, question, model, temperature], outputs=answer)
                question.submit(ask, inputs=[image, question, model, temperature], outputs=answer)

            # --- Detect tab ----------------------------------------------
            with gr.Tab("Detect (bounding boxes)"):
                with gr.Row():
                    with gr.Column():
                        det_image = gr.Image(type="filepath", label="Image", height=400)
                        target = gr.Textbox(
                            label="What to detect",
                            value="the main objects",
                            placeholder="e.g. 'all people and cars', 'dogs', 'faces'",
                        )
                        det_submit = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        annotated = gr.Image(label="Annotated", height=400)
                        detections_json = gr.Code(
                            language="json", label="Detections", lines=12,
                        )
                        raw_output = gr.Textbox(label="Raw model output", lines=4)

                det_submit.click(
                    detect,
                    inputs=[det_image, target, model, temperature],
                    outputs=[annotated, detections_json, raw_output],
                )

    return demo


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build_ui().launch(server_name="0.0.0.0", server_port=7861, share=share)
