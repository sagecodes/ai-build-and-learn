"""
Gemma 4 chatbot with Gradio + Ollama.

Run (after `uv venv` + `uv pip install -r requirements.txt` + activating):
    ollama serve &           # if not already running
    ollama pull gemma4:31b   # or any other gemma4 size
    python app.py            # opens http://localhost:7860

Swap model sizes without editing code:
    GEMMA_MODEL=gemma4:4b python app.py
"""

from __future__ import annotations

import os

import gradio as gr
import ollama

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:31b")
DEFAULT_SYSTEM = "You are a helpful assistant."


def list_models() -> list[str]:
    """Return installed ollama models whose name starts with 'gemma4'."""
    try:
        resp = ollama.list()
        names = [m.model for m in resp.models]
        gemma = sorted(n for n in names if n and n.startswith("gemma4"))
        return gemma or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def chat(message, history, system_prompt, model, temperature, top_p):
    """Stream a response. `history` is Gradio 'messages' format: list of {role, content}."""
    msgs = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history)
    msgs.append({"role": "user", "content": message})

    stream = ollama.chat(
        model=model,
        messages=msgs,
        stream=True,
        options={"temperature": float(temperature), "top_p": float(top_p)},
    )

    partial = ""
    for chunk in stream:
        partial += chunk["message"]["content"]
        if partial:
            yield partial


def build_ui() -> gr.Blocks:
    models = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Blocks(title="Gemma 4 Chat") as demo:
        gr.Markdown("# Gemma 4 Chat\nLocal chatbot via Ollama.")
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="Model", scale=2)
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        system_prompt = gr.Textbox(
            value=DEFAULT_SYSTEM, label="System prompt", lines=2,
        )
        gr.ChatInterface(
            fn=chat,
            additional_inputs=[system_prompt, model, temperature, top_p],
            type="messages",
        )
    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)
