# Gemma 4 Vision

Upload an image, ask a question. Two modes:

- **Ask** — free-form Q&A about the image (describe, read text, count, reason).
- **Detect** — Gemma 4 returns normalized bounding boxes in the Gemini
  `{label, box_2d: [ymin, xmin, ymax, xmax]}` format; we draw them on the image.

Detection is an *emergent* VLM capability, not a trained detector head — it
works surprisingly well on obvious, salient objects and falls apart on small
or crowded scenes. For production detection reach for OWLv2, Grounding DINO,
or YOLO-World.

## Setup

```bash
cd topics/gemma4/vision

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

You also need Ollama running with a Gemma 4 model pulled:

```bash
ollama serve &
ollama pull gemma4:31b     # or gemma4:4b, gemma4:12b
```

## Run

```bash
python app.py
# -> http://localhost:7861
```

### Public URL (remote / forwarded-port setups)

Set `GRADIO_SHARE=1` for a public HTTPS tunnel via Gradio's servers (link good for 72 hours). Useful when SSH port-forwarding chokes on image uploads from a remote dev box.

```bash
GRADIO_SHARE=1 python app.py
```

Pick a smaller size:

```bash
GEMMA_MODEL=gemma4:4b python app.py
```

## Prompts to try

- "Describe this image in detail."
- "Read any text visible in the image."
- "What's unusual or unsafe about this scene?"
- "Count the people / animals / vehicles."
- "List every object you can see, roughly where it is in the frame."
