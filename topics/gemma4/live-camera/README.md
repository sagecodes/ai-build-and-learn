# Gemma 4 Live Camera

Webcam → Gemma 4 vision → streaming caption that describes what's happening,
in "real" time (really every few seconds — inference isn't free).

**Two modes:**

1. **Narrative** (default) — the last few captions are fed back into the
   prompt so the model focuses on *what changed*. Feels like live commentary.
2. **Independent** — each caption is a fresh description, no memory of prior
   frames.

## Setup

```bash
cd topics/gemma4/live-camera

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Ollama running with a vision-capable Gemma 4 model:

```bash
ollama serve &
ollama pull gemma4:e4b          # fast — recommended default
ollama pull gemma4:31b          # higher quality, slower per frame
```

## Run

```bash
python app.py
# -> http://localhost:7867
```

Hit **Start** to begin captioning, **Stop** to pause. Switching model or mode
mid-session takes effect on the next caption.

### Public URL (remote / forwarded-port setups)

Webcam access (`getUserMedia`) requires HTTPS in the browser, so a remote dev
box needs a public tunnel:

```bash
GRADIO_SHARE=1 python app.py
```

## Cadence

Captioning cadence is fixed at launch via `CAMERA_CADENCE` (seconds, default
`3`). Gradio's `stream_every` is baked in when the UI is built, so a runtime
slider wouldn't actually change the cadence — using an env var is honest about
that.

```bash
CAMERA_CADENCE=1 python app.py    # aggressive, needs e4b + GPU to keep up
CAMERA_CADENCE=5 python app.py    # relaxed, works with 31b
```

If inference takes longer than the cadence, `concurrency_limit=1` on the stream
drops the overlapping frames rather than piling up requests.

## Which model?

- `gemma4:e4b` — fast enough for ~2–3s cadence, quality is fine for high-level
  scene descriptions ("a person in a blue shirt is typing on a laptop").
- `gemma4:31b` — noticeably better at reading small text, counting objects,
  subtle pose/action differences, but each caption takes longer so you'll want
  a `CAMERA_CADENCE` of 5+.
- `gemma4:e2b`/`26b` — also work; same tradeoff curve.

## Swappables

```bash
GEMMA_MODEL=gemma4:31b python app.py    # default model on launch
CAMERA_CADENCE=2 python app.py          # caption every 2 seconds
```
