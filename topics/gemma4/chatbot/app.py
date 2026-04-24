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

# Rough chars-per-token heuristic for the thinking budget meter.
CHARS_PER_TOKEN = 3.5


def list_models() -> list[str]:
    """Return installed ollama models whose name starts with 'gemma4'."""
    try:
        resp = ollama.list()
        names = [m.model for m in resp.models]
        gemma = sorted(n for n in names if n and n.startswith("gemma4"))
        return gemma or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def chat(message, history, system_prompt, model, temperature, top_p, think_budget):
    """Stream a response. Yields (cleared_textbox, updated_history) on each chunk.

    think_budget: max thinking tokens (approx). 0 = unlimited. When exceeded,
    we close the thinking stream and re-prompt the model for a direct answer.
    """
    if not message or not message.strip():
        yield "", history
        return

    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
        {"role": "assistant", "content": ""},
    ]
    yield "", history

    msgs = []
    if system_prompt.strip():
        msgs.append({"role": "system", "content": system_prompt})
    msgs.extend(history[:-2])

    budget_chars = int(think_budget * CHARS_PER_TOKEN) if think_budget else 0
    stream = ollama.chat(
        model=model,
        messages=msgs,
        stream=True,
        think=True,
        options={"temperature": float(temperature), "top_p": float(top_p)},
    )

    capped = False
    try:
        for chunk in stream:
            m = chunk["message"]
            if m.get("thinking"):
                history[-2]["content"] += m["thinking"]
            if m.get("content"):
                history[-1]["content"] += m["content"]
            yield "", history

            # If we blew the thinking budget before the answer started, stop.
            if (budget_chars
                    and not history[-1]["content"]
                    and len(history[-2]["content"]) >= budget_chars):
                capped = True
                break
    finally:
        stream.close()

    if capped:
        history[-2]["content"] += f"\n\n_[capped at ~{think_budget} tokens]_"
        yield "", history

        # Second pass: take the thinking we have, no more thinking allowed, answer now.
        followup = list(msgs) + [
            {"role": "assistant", "content": history[-2]["content"]},
            {"role": "user", "content": "Stop thinking. Give your final answer now, concisely."},
        ]
        answer_stream = ollama.chat(
            model=model,
            messages=followup,
            stream=True,
            think=False,
            options={"temperature": float(temperature), "top_p": float(top_p)},
        )
        for chunk in answer_stream:
            history[-1]["content"] += chunk["message"].get("content", "")
            yield "", history


def build_ui() -> gr.Blocks:
    models = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Blocks(title="Gemma 4 Chat") as demo:
        gr.Markdown("# Gemma 4 Chat\nLocal chatbot via Ollama.")
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="Model", scale=2)
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
        with gr.Row():
            think_budget = gr.Slider(
                0, 4000, value=0, step=100,
                label="Thinking budget (tokens, 0 = unlimited)",
                info="Caps thinking tokens. When hit, we stop and force a direct answer.",
            )
        system_prompt = gr.Textbox(
            value=DEFAULT_SYSTEM, label="System prompt", lines=2,
        )

        chatbot = gr.Chatbot(type="messages", label="Conversation", height=500)
        msg = gr.Textbox(label="Your message", placeholder="Type and press Enter")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        inputs = [msg, chatbot, system_prompt, model, temperature, top_p, think_budget]
        outputs = [msg, chatbot]
        msg.submit(chat, inputs=inputs, outputs=outputs)
        send.click(chat, inputs=inputs, outputs=outputs)
        clear.click(lambda: [], outputs=chatbot)

    return demo


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=share)