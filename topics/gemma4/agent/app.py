"""
Gemma 4 agentic tool-use demo.

Gemma 4 advertises `tools` capability — the model can request tool calls,
we execute them, feed the results back, and it keeps going until it has an
answer. This is a ReAct-style loop with four tools: calculator, web_search,
current_datetime, read_file / list_files (sandboxed).

Run (after `uv venv` + `uv pip install -r requirements.txt` + activating):
    ollama serve &
    ollama pull gemma4:31b
    python app.py
"""

from __future__ import annotations

import json
import os

import gradio as gr
import ollama

from tools import SANDBOX, TOOL_REGISTRY, TOOL_SCHEMAS

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:31b")
MAX_TOOL_ROUNDS = 6  # hard cap to prevent runaway loops

SYSTEM_PROMPT = (
    "You are an assistant that can call tools. Use tools when helpful "
    "(math, current date/time, recent info via web search, files in the "
    "sandbox). When you have enough information, give a clear final answer. "
    "Don't call tools you don't need."
)


def list_models() -> list[str]:
    try:
        resp = ollama.list()
        names = sorted(m.model for m in resp.models if m.model.startswith("gemma4"))
        return names or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def run_agent(user_msg: str, model: str):
    """Generator yielding (chat_history, trace_log) tuples as the agent works."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    trace_lines: list[str] = [f"**User**: {user_msg}"]
    chat: list[dict] = [{"role": "user", "content": user_msg}]
    yield chat, "\n\n".join(trace_lines)

    for round_idx in range(MAX_TOOL_ROUNDS):
        resp = ollama.chat(
            model=model, messages=messages,
            tools=TOOL_SCHEMAS,
            options={"temperature": 0.2},
        )
        msg = resp["message"]
        tool_calls = msg.get("tool_calls") or []

        if not tool_calls:
            # Model is done — emit final answer.
            final = msg.get("content", "").strip() or "(no content)"
            chat.append({"role": "assistant", "content": final})
            trace_lines.append(f"**Final answer**:\n{final}")
            yield chat, "\n\n".join(trace_lines)
            return

        # Record the assistant's tool-calling turn so the model sees it in history.
        messages.append({
            "role": "assistant",
            "content": msg.get("content", ""),
            "tool_calls": tool_calls,
        })

        for call in tool_calls:
            fn = call.function
            name = fn.name
            args = fn.arguments if isinstance(fn.arguments, dict) else {}
            trace_lines.append(
                f"**Tool call** `{name}` args=`{json.dumps(args)}`"
            )
            yield chat, "\n\n".join(trace_lines)

            impl = TOOL_REGISTRY.get(name)
            if impl is None:
                result = f"error: unknown tool {name}"
            else:
                try:
                    result = impl(args)
                except Exception as e:
                    result = f"error: {e}"

            preview = result if len(result) < 400 else result[:400] + "..."
            trace_lines.append(f"**Tool result**:\n```\n{preview}\n```")
            yield chat, "\n\n".join(trace_lines)

            messages.append({"role": "tool", "name": name, "content": result})

    # Loop cap hit without a final answer.
    chat.append({
        "role": "assistant",
        "content": f"(stopped after {MAX_TOOL_ROUNDS} tool rounds)",
    })
    trace_lines.append(f"**Halted**: hit {MAX_TOOL_ROUNDS}-round cap")
    yield chat, "\n\n".join(trace_lines)


def build_ui() -> gr.Blocks:
    models = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    # Make sure sandbox exists with a sample file so list_files has something to show.
    SANDBOX.mkdir(exist_ok=True)
    sample = SANDBOX / "notes.txt"
    if not sample.exists():
        sample.write_text(
            "Project notes\n"
            "- The DGX Spark has 128GB unified memory and uses a GB10 GPU.\n"
            "- Gemma 4 comes in 4B, 12B, and 31B sizes (plus an E flavor).\n"
            "- Demo night: April 23rd.\n"
        )

    with gr.Blocks(title="Gemma 4 Agent") as demo:
        gr.Markdown(
            "# Gemma 4 Agent\n"
            "Ask a question that needs tools: math, current time, web search, "
            "or reading a file from the sandbox. The model decides what to call."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="Model")

        with gr.Row():
            with gr.Column():
                chat = gr.Chatbot(type="messages", label="Conversation", height=400)
                msg = gr.Textbox(
                    label="Your question",
                    placeholder="e.g. What is 847 * 293? Or: what files are in the sandbox?",
                )
                submit = gr.Button("Ask", variant="primary")
                gr.Examples(
                    examples=[
                        "What is 847 * 293 + 1024?",
                        "What's today's date and time?",
                        "List the files in the sandbox, then read the one about the project.",
                        "Search the web for the latest news on Gemma 4.",
                    ],
                    inputs=msg,
                )
            with gr.Column():
                trace = gr.Markdown(label="Tool trace", value="_trace will appear here_")

        submit.click(run_agent, inputs=[msg, model], outputs=[chat, trace])
        msg.submit(run_agent, inputs=[msg, model], outputs=[chat, trace])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7864)
