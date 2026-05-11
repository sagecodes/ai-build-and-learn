"""
Gemma 4 live-camera describer: webcam → Gemma 4 vision → streaming caption.

Two modes:
- independent: each caption is a fresh description of the current frame.
- narrative:   the prompt includes the last few captions so the model can
               focus on what changed — feels like live commentary.

Gradio's stream_every is static per launch, so cadence is set via env var
(CAMERA_CADENCE, in seconds) rather than a slider that wouldn't actually
rebind.

Run (after deps + `ollama serve`):
    ollama pull gemma4:e4b
    python app.py            # -> http://localhost:7867
    GRADIO_SHARE=1 python app.py   # public HTTPS (needed for webcam on remote)
"""

from __future__ import annotations

import base64
import io
import os
import threading

import gradio as gr
import ollama
from PIL import Image

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:e4b")
CADENCE_S = float(os.environ.get("CAMERA_CADENCE", "3"))
HISTORY_LEN = 4  # how many recent captions to feed the model in narrative mode
MAX_SIDE = int(os.environ.get("MAX_SIDE", "384"))          # downsample long side — smaller = faster
MAX_OUT_TOKENS = int(os.environ.get("MAX_OUT_TOKENS", "60"))  # cap reply length — hard cutoff on generation time

# Explicit backpressure. Gradio's concurrency_limit=1 on .stream() doesn't
# reliably drop overlapping frames — at webcam frame rates we end up with
# generators piling up. This flag drops any frame that arrives while a caption
# is already in flight.
_inflight = threading.Lock()

INDEPENDENT_PROMPT = (
    "Describe this image in one short sentence. Direct, specific, no preamble."
)

NARRATIVE_PROMPT = (
    "You are narrating a live camera feed. Recent captions:\n{history}\n\n"
    "Describe the new frame in one short sentence. Focus on what changed. "
    "Present tense, no preamble."
)


def list_models() -> list[str]:
    try:
        resp = ollama.list()
        gemma = sorted(m.model for m in resp.models if m.model.startswith("gemma4"))
        return gemma or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def encode_frame(frame) -> str:
    """Gradio gives us a numpy RGB array; Ollama wants base64 image bytes.
    Downsample so the payload and Gemma's vision encoder are both fast."""
    img = Image.fromarray(frame)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def format_history(history: list[str]) -> str:
    return "\n".join(f"- {h}" for h in history[-HISTORY_LEN:])


def stash_frame(frame):
    """Fast no-op handler for gr.Image.stream — just stores the latest frame
    in state. Kept trivial so Gradio's stream-cancellation doesn't matter."""
    return frame


def caption_tick(frame, model: str, mode: str, history: list[str], running: bool):
    """Called by gr.Timer on cadence. Reads the latest stashed frame, encodes,
    and streams a caption. Timer ticks don't interrupt the running generator,
    so the full ollama round-trip completes."""
    if frame is None or not running:
        return

    if not _inflight.acquire(blocking=False):
        return

    try:
        img_b64 = encode_frame(frame)

        if mode == "narrative" and history:
            prompt = NARRATIVE_PROMPT.format(history=format_history(history))
        else:
            prompt = INDEPENDENT_PROMPT

        yield f"_querying **{model}**…_", history, format_history(history)

        try:
            stream = ollama.chat(
                model=model,
                messages=[{"role": "user", "content": prompt, "images": [img_b64]}],
                stream=True,
                think=False,  # skip thinking tokens — we want a fast, direct caption
                options={"temperature": 0.4, "num_predict": MAX_OUT_TOKENS},
            )
        except Exception as e:
            yield f"**Error**: {e}", history, format_history(history)
            return

        reply = ""
        chunks = 0
        try:
            for chunk in stream:
                chunks += 1
                reply += chunk.get("message", {}).get("content", "") or ""
                yield reply, history, format_history(history)
        except Exception as e:
            yield f"**Error during streaming**: {e}\n\nPartial: {reply}", history, format_history(history)
            return

        reply = reply.strip()
        if not reply:
            yield (
                f"_(empty response — {chunks} chunks, model may have stopped "
                f"on a stop token or refused the frame)_",
                history,
                format_history(history),
            )
            return
        new_history = (history + [reply])[-HISTORY_LEN:]
        yield reply, new_history, format_history(new_history)
    finally:
        _inflight.release()


def start():
    # Activate the timer, flip running=True, reset history/caption.
    return gr.Timer(value=CADENCE_S, active=True), True, [], "_Starting…_", ""


def stop():
    return gr.Timer(active=False), False


def build_ui() -> gr.Blocks:
    models = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Blocks(title="Gemma 4 Live Camera") as demo:
        gr.Markdown(
            "# Gemma 4 Live Camera\n"
            "Webcam → Gemma 4 vision → streaming caption. "
            f"Currently captioning every **{CADENCE_S:g}s** "
            "(set `CAMERA_CADENCE` env var to change)."
        )

        with gr.Row():
            model = gr.Dropdown(models, value=default, label="Vision model")
            mode = gr.Radio(
                ["narrative", "independent"], value="narrative", label="Mode",
                info="narrative = use recent captions as context (commentary feel)",
            )

        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(
                    sources=["webcam"], streaming=True, type="numpy",
                    label="Camera", height=360,
                )
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop")
            with gr.Column(scale=1):
                caption = gr.Markdown(
                    "_Press **Start** to begin captioning…_",
                    label="Latest caption",
                )
                with gr.Accordion("Recent captions (narrative context)", open=False):
                    history_md = gr.Markdown()

        running = gr.State(False)
        history = gr.State([])
        latest_frame = gr.State(None)

        # Webcam stream writes latest frame into state. Fast, no generator,
        # so Gradio's stream-cancellation is fine here.
        cam.stream(
            stash_frame,
            inputs=cam,
            outputs=latest_frame,
            stream_every=0.2,
            concurrency_limit=None,
        )

        # Timer ticks on cadence and captions the latest stashed frame.
        # Timer handlers run to completion — no interrupt-on-new-input.
        caption_timer = gr.Timer(value=CADENCE_S, active=False)
        caption_timer.tick(
            caption_tick,
            inputs=[latest_frame, model, mode, history, running],
            outputs=[caption, history, history_md],
        )

        start_btn.click(
            start,
            outputs=[caption_timer, running, history, caption, history_md],
        )
        stop_btn.click(stop, outputs=[caption_timer, running])

    return demo


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build_ui().launch(server_name="0.0.0.0", server_port=7867, share=share)
