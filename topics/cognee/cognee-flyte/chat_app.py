"""Chatbot with Cognee memory — Gemma 4 vLLM front, cognee graph behind.

The simple half of the layered API: the pipeline builds the memory with
add/cognify; this app drives the *same* store with cognee's recall/remember.
Each turn:
  1. recall() pulls relevant context from the cognee graph for the user's query.
  2. Gemma 4 streams an answer with that context injected.
  3. remember() writes the exchange back into the graph (add + cognify + enrich).

Persistence is the "both" story:
  - On startup the app seeds its store from the most recent memory_pipeline run
    via RunOutput(directory) — Flyte mounts that flyte.io.Dir into the pod.
  - Knative scales the pod to zero after idle, wiping /tmp. So @on_shutdown (and
    a "Save to HF" button) tars the cognee store to a HuggingFace repo, and
    on_startup falls back to that snapshot if no pipeline run is mounted.

Deploy (Gemma 4 vLLM must already be up):
    python chat_app.py
"""

from __future__ import annotations

from pathlib import Path

import flyte
import flyte.app

from config import (
    HF_MEMORY_REPO,
    HF_MEMORY_REPO_TYPE,
    VLLM_APP_NAME,
    VLLM_MODEL_ID,
    COGNEE_PIP_PACKAGES,
)

# Fully-qualified task name Flyte resolves to the latest pipeline run.
# Must match <TaskEnvironment.name>.<function> from config.py + pipeline.py.
PIPELINE_TASK = "cognee-memory-pipeline.memory_pipeline"

# Pod-local cognee storage root. Seeded on startup, tarred to HF on shutdown.
WORK_DIR = "/tmp/cognee-mem"


chat_image = (
    flyte.Image.from_debian_base(
        name="cognee-memory-chat-image",
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    .with_pip_packages("gradio==5.42.0", *COGNEE_PIP_PACKAGES)
    .with_source_file(Path(__file__).parent / "cognee_lib.py")
    .with_source_file(Path(__file__).parent / "config.py")
)


env = flyte.app.AppEnvironment(
    name="cognee-memory-chat",
    image=chat_image,
    resources=flyte.Resources(cpu="4", memory="8Gi"),
    port=7860,
    requires_auth=False,
    secrets=[flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")],
    parameters=[
        flyte.app.Parameter(
            name="vllm_url",
            value=f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local",
            env_var="VLLM_URL",
        ),
        flyte.app.Parameter(name="model_id", value=VLLM_MODEL_ID),
        flyte.app.Parameter(name="hf_repo", value=HF_MEMORY_REPO),
        flyte.app.Parameter(name="hf_repo_type", value=HF_MEMORY_REPO_TYPE),
        # Mount the memory Dir from the most recent memory_pipeline run. Flyte
        # downloads it into the pod and exposes the local path on
        # $MEMORY_SEED_DIR. Pin a specific run with
        # MEMORY_RUN_NAME=<run> python chat_app.py.
        flyte.app.Parameter(
            name="memory_seed_dir",
            type="directory",
            value=flyte.app.RunOutput(task_name=PIPELINE_TASK, type="directory"),
            download=True,
            env_var="MEMORY_SEED_DIR",
        ),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)


# State shared across hooks (configure_cognee is process-global; we just track
# what was restored so the UI can report it).
state: dict = {}


# ──────────────────────────────────────────────────────────────────────────────
# Lifecycle: seed the store, then tar it to HF on the way down.
# ──────────────────────────────────────────────────────────────────────────────

@env.on_startup
async def load_memory(
    vllm_url: str,
    model_id: str,
    hf_repo: str,
    hf_repo_type: str,
) -> None:
    import os
    import shutil

    import cognee_lib

    # Configure cognee BEFORE importing it; this also creates WORK_DIR.
    cognee_lib.configure_cognee(WORK_DIR)

    seed = os.environ.get("MEMORY_SEED_DIR", "").strip()
    restored = ""
    if seed and os.path.isdir(seed) and any(Path(seed).iterdir()):
        shutil.copytree(seed, WORK_DIR, dirs_exist_ok=True)
        restored = f"pipeline run (mounted at {seed})"
    elif cognee_lib.pull_memory_from_hf(hf_repo, hf_repo_type, WORK_DIR):
        restored = f"HF snapshot ({hf_repo})"

    stats = cognee_lib.storage_summary(WORK_DIR)
    state["restored_from"] = restored or "empty (no prior memory found)"
    print(
        f"[cognee-mem] startup: seeded from {state['restored_from']} — "
        f"{stats['files']} files / {stats['mb']} MB",
        flush=True,
    )


@env.on_shutdown
async def save_memory_on_shutdown(
    vllm_url: str,
    model_id: str,
    hf_repo: str,
    hf_repo_type: str,
) -> None:
    import cognee_lib

    try:
        cognee_lib.push_memory_to_hf(hf_repo, hf_repo_type, WORK_DIR, note="auto-save on shutdown")
    except Exception as e:
        print(f"[cognee-mem] shutdown save failed: {type(e).__name__}: {e}", flush=True)


# ──────────────────────────────────────────────────────────────────────────────
# Gemma thinking-block parser (same as the sibling chat apps).
# ──────────────────────────────────────────────────────────────────────────────

def _split_thinking(text: str) -> tuple[str, str]:
    OPEN, OPEN_TAIL = "<|channel>", "thought\n"
    CLOSE = "<channel|>"
    j = text.find(OPEN)
    if j == -1:
        return "", text.strip()
    pre = text[:j]
    rest = text[j + len(OPEN):]
    if rest.startswith(OPEN_TAIL):
        rest = rest[len(OPEN_TAIL):]
    k = rest.find(CLOSE)
    if k == -1:
        thinking, answer = rest, pre
    else:
        thinking = rest[:k]
        answer = (pre + rest[k + len(CLOSE):])
    return thinking.strip(), answer.strip()


def _esc(s: str) -> str:
    return (s or "").replace("<", "&lt;").replace(">", "&gt;")


PANELS_CSS = """
.mem-panel { display: flex; flex-direction: column; gap: 10px; }
.mem-empty { color: var(--body-text-color-subdued); font-style: italic; padding: 10px; }
.mem-card { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 8px 10px; background: var(--background-fill-secondary); }
.mem-text { font-size: 0.9rem; line-height: 1.4; white-space: pre-wrap; word-break: break-word; }
.mem-header { font-size: 0.85rem; color: var(--body-text-color-subdued); margin-bottom: 4px; }
.mem-status { padding: 8px 12px; border-radius: 6px; background: var(--background-fill-secondary); font-size: 0.85rem; }
"""


def _render_context(snippets: list[str]) -> str:
    if not snippets:
        return '<div class="mem-empty">No memory recalled this turn.</div>'
    cards = "".join(
        f'<div class="mem-card"><div class="mem-text">{_esc(s)}</div></div>'
        for s in snippets
    )
    return f'<div class="mem-panel"><div class="mem-header">Recalled ({len(snippets)})</div>{cards}</div>'


# ──────────────────────────────────────────────────────────────────────────────
# Server.
# ──────────────────────────────────────────────────────────────────────────────

@env.server
def chat_server(vllm_url: str, model_id: str, hf_repo: str, hf_repo_type: str):
    import sys
    import traceback
    try:
        _run(vllm_url, model_id, hf_repo, hf_repo_type)
    except BaseException as e:
        print(f"!!! chat_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _run(vllm_url: str, model_id: str, hf_repo: str, hf_repo_type: str):
    import time

    import gradio as gr
    from openai import OpenAI

    import cognee
    import cognee_lib
    from cognee import SearchType

    # Idempotent: startup already configured cognee, but the server may run in a
    # context where we want to be sure the env is set before the first call.
    cognee_lib.configure_cognee(WORK_DIR)

    llm = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="not-used", timeout=120.0)
    print(f"[cognee-chat] vLLM at {vllm_url}/v1 (model={model_id})", flush=True)
    print(f"[cognee-chat] memory seeded from: {state.get('restored_from', '?')}", flush=True)

    DEFAULT_SYSTEM = (
        "You are an assistant with a persistent knowledge-graph memory. Use the "
        "MEMORY section as grounding context: it holds facts the system has "
        "ingested and remembered. Weave it in naturally; don't recite it."
    )
    MAX_TOTAL_TOKENS = 4096

    # ── cognee adapters: prefer the v1.0 recall/remember; fall back to the
    # granular search/add+cognify on older builds. Both hit the same store. ──

    async def _recall(query: str) -> list[str]:
        try:
            results = await cognee.recall(query_text=query)  # v1.0 simple API
        except AttributeError:
            results = await cognee.search(query_text=query, query_type=SearchType.CHUNKS)
        return [cognee_lib.result_text(r) for r in (results or [])][:6]

    async def _remember(text: str) -> None:
        try:
            await cognee.remember(text)  # v1.0: add + cognify + enrich
        except AttributeError:
            await cognee.add(text, dataset_name="memory")
            await cognee.cognify(datasets=["memory"])

    def recall(query: str) -> list[str]:
        import asyncio
        return asyncio.run(_recall(query))

    def remember(text: str) -> None:
        import asyncio
        asyncio.run(_remember(text))

    # ── Chat handler ──────────────────────────────────────────────────────────

    def chat(message, history, system_prompt, use_memory, enable_thinking, temperature, top_p):
        if not message or not message.strip():
            yield "", history, _render_context([]), _status_html()
            return

        snippets = recall(message) if use_memory else []
        ctx_html = _render_context(snippets)

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
            {"role": "assistant", "content": ""},
        ]
        yield "", history, ctx_html, _status_html()

        sys_text = (system_prompt or DEFAULT_SYSTEM).strip()
        if snippets:
            block = "MEMORY (recalled from the knowledge graph):\n" + "\n".join(
                f"- {s}" for s in snippets
            )
            sys_text = f"{sys_text}\n\n{block}"

        msgs = [{"role": "system", "content": sys_text}]
        for t in history[:-2]:
            if "metadata" in t:
                continue
            msgs.append({"role": t["role"], "content": t["content"]})

        stream = llm.chat.completions.create(
            model=model_id,
            messages=msgs,
            stream=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=MAX_TOTAL_TOKENS,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                "skip_special_tokens": False,
            },
        )
        buf = ""
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                thinking, answer = _split_thinking(buf)
                history[-2]["content"] = thinking
                history[-1]["content"] = answer
                yield "", history, ctx_html, _status_html()
        finally:
            stream.close()

        if not history[-2]["content"]:
            history.pop(-2)

        # Write the exchange back into the graph. Heavier than a vector write
        # (cognify rebuilds graph + embeddings), so flag it in the status line.
        final_answer = history[-1]["content"]
        yield "", history, ctx_html, _status_html(extra="💾 remembering this exchange…")
        try:
            remember(f"User said: {message}\nAssistant replied: {final_answer}")
        except Exception as e:
            print(f"[cognee-mem] remember failed: {type(e).__name__}: {e}", flush=True)
        yield "", history, ctx_html, _status_html()

    # ── Status + save ─────────────────────────────────────────────────────────

    def _status_html(extra: str = "") -> str:
        stats = cognee_lib.storage_summary(WORK_DIR)
        line = (
            f"Memory store: <b>{stats['mb']} MB</b> · seeded from "
            f"<i>{_esc(state.get('restored_from', '?'))}</i>"
        )
        if extra:
            line += f" · {extra}"
        return f'<div class="mem-status">{line}</div>'

    def save_now():
        try:
            url = cognee_lib.push_memory_to_hf(hf_repo, hf_repo_type, WORK_DIR, note="manual save from UI")
            return f'<div class="mem-status">✅ Saved to <a href="{url}" target="_blank">{hf_repo}</a></div>'
        except Exception as e:
            return f'<div class="mem-status">❌ Save failed: {_esc(type(e).__name__)}: {_esc(str(e))}</div>'

    # ── Graph render (same view as the ingest pipeline's report) ──────────────

    def render_graph():
        import asyncio

        async def _collect():
            from cognee.infrastructure.databases.graph import get_graph_engine

            graph_engine = await get_graph_engine()
            nodes, edges = await graph_engine.get_graph_data()
            viz = await cognee.visualize_graph(f"{WORK_DIR}/graph.html")
            return nodes, edges, viz

        try:
            nodes, edges, viz = asyncio.run(_collect())
        except Exception as e:
            return f'<div class="mem-status">Graph render failed: {_esc(type(e).__name__)}: {_esc(str(e))}</div>'
        header = f'<div class="mem-status">{len(nodes)} nodes · {len(edges)} relationships</div>'
        iframe = cognee_lib.embed_graph_iframe(viz)
        table = cognee_lib.graph_summary_html(nodes, edges)
        return header + iframe + "<h3>Relationships</h3>" + table

    # ── UI ──────────────────────────────────────────────────────────────────

    with gr.Blocks(title=f"Cognee Memory ({model_id})", css=PANELS_CSS) as demo:
        gr.Markdown(
            f"# Cognee Memory — graph + vectors behind Gemma 4\n"
            f"Model: `{model_id}` · Memory: cognee (SQLite + LanceDB + Ladybug) · "
            f"Backed by `{hf_repo}`"
        )
        with gr.Tab("💬 Chat"):
            with gr.Row():
                temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
                top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            with gr.Row():
                system_prompt = gr.Textbox(value=DEFAULT_SYSTEM, label="System prompt", lines=2, scale=4)
                use_memory = gr.Checkbox(value=True, label="Use memory", scale=1)
                enable_thinking = gr.Checkbox(value=True, label="Enable thinking", scale=1)
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(type="messages", label="Conversation", height=520)
                    msg = gr.Textbox(label="Your message", placeholder="Ask about what's been ingested, or tell me something to remember…")
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("Clear chat (memory stays)")
                        save_btn = gr.Button("💾 Save to HF", variant="secondary")
                with gr.Column(scale=2):
                    gr.Markdown("### Recalled this turn")
                    context_view = gr.HTML(value=_render_context([]))
                    gr.Markdown("### Status")
                    status_view = gr.HTML(value=_status_html())

            inputs = [msg, chatbot, system_prompt, use_memory, enable_thinking, temperature, top_p]
            outputs = [msg, chatbot, context_view, status_view]
            msg.submit(chat, inputs=inputs, outputs=outputs)
            send.click(chat, inputs=inputs, outputs=outputs)
            clear.click(
                lambda: ([], _render_context([]), _status_html()),
                outputs=[chatbot, context_view, status_view],
            )
            save_btn.click(save_now, outputs=status_view)

        with gr.Tab("🕸 Graph"):
            gr.Markdown(
                "The cognee knowledge graph for the **live** memory store "
                "(`/tmp/cognee-mem`), updated as you chat. Same view the ingest "
                "pipeline renders in its Flyte report. The interactive graph needs "
                "network (d3 from a CDN); the relationships table always renders."
            )
            graph_refresh = gr.Button("🔄 Render graph", variant="primary")
            graph_view = gr.HTML(value='<div class="mem-status">Click “Render graph” to draw the current memory.</div>')
            graph_refresh.click(render_graph, outputs=graph_view)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    # Keep the pod alive if launch() ever returns, so on_shutdown can fire.
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Cognee memory chat deployed: {app.url}")
