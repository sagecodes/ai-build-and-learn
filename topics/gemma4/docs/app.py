"""
Gemma 4 long-context document Q&A.

Drop in a PDF / txt / md file, ask questions. The entire document goes into
the prompt (no RAG, no chunking) — gemma4:31b has a 262k context window.

Run (after `uv venv` + `uv pip install -r requirements.txt` + activating):
    ollama serve &
    ollama pull gemma4:31b
    python app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import gradio as gr
import ollama
from pypdf import PdfReader

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:31b")

# Rough tokens-per-char heuristic for the context meter. Real tokenization
# varies — this is good enough for a "are we close to the limit?" signal.
CHARS_PER_TOKEN = 3.5


def list_models() -> list[str]:
    try:
        resp = ollama.list()
        names = sorted(m.model for m in resp.models if m.model.startswith("gemma4"))
        return names or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def load_file(path: str | None) -> tuple[str, str]:
    """Read text from an uploaded file. Returns (text, status_message)."""
    if not path:
        return "", ""
    p = Path(path)
    suffix = p.suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(str(p))
            pages = [page.extract_text() or "" for page in reader.pages]
            text = "\n\n".join(pages)
            meta = f"{len(reader.pages)} pages"
        else:
            text = p.read_text(errors="replace")
            meta = f"{suffix or 'text'} file"
    except Exception as e:
        return "", f"Failed to read {p.name}: {e}"

    approx_tokens = int(len(text) / CHARS_PER_TOKEN)
    status = (
        f"Loaded **{p.name}** — {meta}, {len(text):,} chars "
        f"(~{approx_tokens:,} tokens). Context window is 262k."
    )
    return text, status


def ask(doc_text: str, question: str, model: str, temperature: float,
        think_budget: int):
    """Stream (thinking, answer). Full doc goes into the user message. If
    thinking exceeds budget before the answer starts, cancel and re-prompt
    with think=False."""
    if not doc_text.strip():
        yield "", "Load a document first."
        return
    if not question.strip():
        yield "", "Ask a question about the document."
        return

    system = (
        "You answer questions about a document. Quote relevant passages when "
        "possible. If the answer isn't in the document, say so — do not invent."
    )
    user = f"<document>\n{doc_text}\n</document>\n\nQuestion: {question}"
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]

    # Ollama defaults num_ctx=4096 which would silently truncate a long doc.
    # Size the window to fit doc + headroom for question/answer. Clamp to 256k
    # to stay under the model's 262k maximum with a small safety margin.
    approx_tokens = int(len(doc_text) / CHARS_PER_TOKEN)
    num_ctx = min(262_000, max(8_192, approx_tokens + 4_096))
    budget_chars = int(think_budget * CHARS_PER_TOKEN) if think_budget else 0

    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True,
        think=True,
        options={"temperature": float(temperature), "num_ctx": num_ctx},
    )

    thinking, answer = "", ""
    capped = False
    try:
        for chunk in stream:
            m = chunk["message"]
            if m.get("thinking"):
                thinking += m["thinking"]
            if m.get("content"):
                answer += m["content"]
            yield thinking, answer or "_streaming..._"

            if budget_chars and not answer and len(thinking) >= budget_chars:
                capped = True
                break
    finally:
        stream.close()

    if capped:
        thinking += f"\n\n_[capped at ~{think_budget} tokens]_"
        yield thinking, "_generating answer..._"

        followup = messages + [
            {"role": "assistant", "content": thinking},
            {"role": "user", "content": "Stop thinking. Give your final answer now, concisely."},
        ]
        answer_stream = ollama.chat(
            model=model,
            messages=followup,
            stream=True,
            think=False,
            options={"temperature": float(temperature), "num_ctx": num_ctx},
        )
        answer = ""
        for chunk in answer_stream:
            answer += chunk["message"].get("content", "")
            yield thinking, answer or "_generating answer..._"


def build_ui() -> gr.Blocks:
    models = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Blocks(title="Gemma 4 Docs") as demo:
        gr.Markdown(
            "# Gemma 4 Long-Context Docs\n"
            "Drop in a PDF / txt / md file and ask anything. No chunking, no RAG "
            "— the whole document goes into the 262k context window."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="Model")
            temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature")
            think_budget = gr.Slider(
                0, 4000, value=0, step=100,
                label="Thinking budget (tokens, 0 = unlimited)",
                info="Caps thinking. When hit, we force a direct answer.",
            )

        with gr.Row():
            with gr.Column():
                upload = gr.File(
                    label="Document",
                    file_types=[".pdf", ".txt", ".md"],
                    type="filepath",
                )
                status = gr.Markdown()
                doc_text = gr.Textbox(
                    label="Document text (editable)", lines=14, max_lines=30,
                )
                question = gr.Textbox(
                    label="Question", lines=2,
                    placeholder="e.g. Summarize section 3 in 5 bullets.",
                )
                submit = gr.Button("Ask", variant="primary")
            with gr.Column():
                with gr.Accordion("🧠 Thinking", open=False):
                    thinking = gr.Textbox(
                        label=None, show_label=False, lines=10,
                        placeholder="Thinking tokens stream here...",
                    )
                answer = gr.Textbox(label="Answer", lines=20)

        upload.change(load_file, inputs=upload, outputs=[doc_text, status])
        ask_inputs = [doc_text, question, model, temperature, think_budget]
        ask_outputs = [thinking, answer]
        submit.click(ask, inputs=ask_inputs, outputs=ask_outputs)
        question.submit(ask, inputs=ask_inputs, outputs=ask_outputs)

    return demo


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build_ui().launch(server_name="0.0.0.0", server_port=7863, share=share)
