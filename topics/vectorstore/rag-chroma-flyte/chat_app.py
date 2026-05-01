"""Gradio RAG chat UI — Chroma + Gemma 4 vLLM.

Same shape as gemma4-dgx-devbox/chat_app.py, but:
  - Mounts a Chroma `PersistentClient` directory from a pipeline run via
    `flyte.app.RunOutput(task_name=PIPELINE_TASK, type="directory")`.
  - Embeds each query locally (sentence-transformers, BGE-small).
  - Retrieves top-k chunks and injects them into the system prompt.
  - Renders the retrieved chunks alongside the answer.

Deploy (after `flyte run pipeline.py rag_pipeline`):
    RAG_PIPELINE_RUN=<run-name> python chat_app.py
"""

from __future__ import annotations

import os

import flyte
import flyte.app


# ── Gemma 4 vLLM endpoint info ────────────────────────────────────────────────
# Hard-coded to match the running gemma4-dgx-devbox vLLM app. If you switched
# to the 31B variant, change these two strings.

VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"

# ── RAG knobs that the chat side needs ────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "rag_demo"
DEFAULT_TOP_K = 4

# Pipeline task name → matches `pipeline_env.name + "." + function_name`
PIPELINE_TASK = "rag-chroma-pipeline.rag_pipeline"


# ── Image ─────────────────────────────────────────────────────────────────────

chat_image = (
    flyte.Image.from_debian_base(
        name="rag-chat-image",
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    .with_pip_packages(
        # gr.Chatbot(type="messages") needs Gradio 5.x — same constraint as
        # the sibling gemma4 chat app.
        "gradio==5.42.0",
        "openai>=1.50.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
    )
)


# ── App env ───────────────────────────────────────────────────────────────────
#
# The Chroma dir is wired in via Parameter(value=RunOutput(...), download=True),
# which the Flyte serving runtime resolves to a local path and exposes as the
# CHROMA_DIR env var inside the pod (same trick fraud-detection-feast/app.py uses).
#
# Set RAG_PIPELINE_RUN=<name> to pin a specific pipeline run; otherwise the
# latest succeeded run of `rag_pipeline` is used.

_pipeline_run = os.environ.get("RAG_PIPELINE_RUN")
_chroma_run_output = (
    flyte.app.RunOutput(type="directory", run_name=_pipeline_run)
    if _pipeline_run
    else flyte.app.RunOutput(type="directory", task_name=PIPELINE_TASK)
)

env = flyte.app.AppEnvironment(
    name="rag-chat-ui",
    image=chat_image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    port=7860,
    requires_auth=False,
    parameters=[
        flyte.app.Parameter(
            name="vllm_url",
            value=f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local",
            env_var="VLLM_URL",
        ),
        flyte.app.Parameter(name="model_id", value=VLLM_MODEL_ID),
        flyte.app.Parameter(
            name="chroma_dir",
            value=_chroma_run_output,
            download=True,
            env_var="CHROMA_DIR",
        ),
        flyte.app.Parameter(name="embedding_model", value=EMBEDDING_MODEL),
        flyte.app.Parameter(name="collection_name", value=COLLECTION_NAME),
        flyte.app.Parameter(name="default_top_k", value=str(DEFAULT_TOP_K)),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=1800),
)


# ── Gemma thinking-block parser (same as the sibling chat_app.py) ─────────────

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


# ── CSS for the retrieved-chunks panel (CLAUDE.md: classes only, no inline) ───

CHUNKS_CSS = """
.chunks-panel { display: flex; flex-direction: column; gap: 12px; }
.chunks-empty {
    color: var(--body-text-color-subdued);
    font-style: italic;
    padding: 12px;
}
.chunk-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 10px 12px;
    background: var(--background-fill-secondary);
}
.chunk-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 6px;
}
.chunk-rank { font-weight: 600; }
.chunk-score { font-variant-numeric: tabular-nums; }
.chunk-text {
    font-size: 0.92rem;
    line-height: 1.45;
    white-space: pre-wrap;
    word-break: break-word;
}
.chunks-header {
    font-size: 0.85rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 4px;
}
"""


def _render_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return '<div class="chunks-empty">No chunks retrieved yet — send a message.</div>'
    cards = []
    for i, c in enumerate(chunks, 1):
        # Cosine *distance* in Chroma: 0 = identical, up to 2. Convert to a
        # similarity-ish 0–1 for display.
        sim = max(0.0, 1.0 - (c["distance"] / 2.0))
        text = (c["text"] or "").replace("<", "&lt;").replace(">", "&gt;")
        cards.append(
            '<div class="chunk-card">'
            '<div class="chunk-meta">'
            f'<span class="chunk-rank">#{i} · doc {c["doc_id"]}</span>'
            f'<span class="chunk-score">sim {sim:.3f}</span>'
            '</div>'
            f'<div class="chunk-text">{text}</div>'
            '</div>'
        )
    header = f'<div class="chunks-header">Top {len(chunks)} retrieved chunks</div>'
    return f'<div class="chunks-panel">{header}{"".join(cards)}</div>'


# ── Server ────────────────────────────────────────────────────────────────────

@env.server
def chat_server(
    vllm_url: str,
    model_id: str,
    chroma_dir: str,
    embedding_model: str,
    collection_name: str,
    default_top_k: str,
):
    import sys
    import traceback
    try:
        _run(vllm_url, model_id, chroma_dir, embedding_model, collection_name, int(default_top_k))
    except BaseException as e:
        print(f"!!! chat_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _run(
    vllm_url: str,
    model_id: str,
    chroma_dir: str,
    embedding_model: str,
    collection_name: str,
    default_top_k: int,
):
    import chromadb
    import gradio as gr
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    print(f"[chat_server] gradio version: {gr.__version__}", flush=True)
    print(f"[chat_server] vLLM at {vllm_url}/v1 (model={model_id})", flush=True)
    print(f"[chat_server] Chroma at {chroma_dir} (collection={collection_name})", flush=True)
    print(f"[chat_server] Loading encoder: {embedding_model}", flush=True)

    encoder = SentenceTransformer(embedding_model)
    chroma_client = chromadb.PersistentClient(path=chroma_dir)
    collection = chroma_client.get_collection(name=collection_name)
    print(f"[chat_server] Collection '{collection_name}' has {collection.count()} chunks", flush=True)

    indexed_model = (collection.metadata or {}).get("embedding_model")
    if indexed_model and indexed_model != embedding_model:
        # Mismatched encoders → meaningless similarity. Surface loudly rather
        # than silently retrieving garbage.
        print(
            f"WARNING: query encoder '{embedding_model}' != index encoder "
            f"'{indexed_model}'. Retrieval will be noisy.",
            flush=True,
        )

    llm = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="not-used")

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. Use the provided CONTEXT to answer. "
        "If the answer is not in the context, say you don't know — do not invent. "
        "Cite sources as [#N] where N is the chunk number."
    )
    CHARS_PER_TOKEN = 3.5
    MAX_TOTAL_TOKENS = 4096

    def retrieve(query: str, top_k: int) -> list[dict]:
        if not query.strip():
            return []
        vec = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).tolist()
        res = collection.query(
            query_embeddings=vec,
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({
                "doc_id": (meta or {}).get("doc_id", "?"),
                "text": doc,
                "distance": float(dist),
            })
        return out

    def build_context_block(chunks: list[dict]) -> str:
        if not chunks:
            return ""
        lines = ["CONTEXT:"]
        for i, c in enumerate(chunks, 1):
            lines.append(f"[#{i}] (doc {c['doc_id']})\n{c['text']}")
        return "\n\n".join(lines)

    def chat(message, history, system_prompt, use_rag, top_k,
             enable_thinking, think_budget, temperature, top_p):
        if not message or not message.strip():
            yield "", history, _render_chunks([])
            return

        chunks = retrieve(message, int(top_k)) if use_rag else []
        chunks_html = _render_chunks(chunks)

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
            {"role": "assistant", "content": ""},
        ]
        yield "", history, chunks_html

        sys_text = (system_prompt or DEFAULT_SYSTEM).strip()
        ctx_block = build_context_block(chunks)
        if ctx_block:
            sys_text = f"{sys_text}\n\n{ctx_block}"

        msgs = [{"role": "system", "content": sys_text}]
        for t in history[:-2]:
            if "metadata" in t:
                continue
            msgs.append({"role": t["role"], "content": t["content"]})

        budget_chars = int(think_budget * CHARS_PER_TOKEN) if think_budget else 0

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
        capped = False
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                thinking, answer = _split_thinking(buf)
                history[-2]["content"] = thinking
                history[-1]["content"] = answer
                yield "", history, chunks_html

                if (budget_chars and not answer and len(thinking) >= budget_chars):
                    capped = True
                    break
        finally:
            stream.close()

        if capped:
            history[-2]["content"] += f"\n\n_[capped at ~{think_budget} tokens]_"
            yield "", history, chunks_html

            followup = msgs + [
                {"role": "assistant", "content": history[-2]["content"]},
                {"role": "user", "content": "Stop thinking. Give your final answer now, concisely."},
            ]
            answer_stream = llm.chat.completions.create(
                model=model_id,
                messages=followup,
                stream=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=MAX_TOTAL_TOKENS,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "skip_special_tokens": False,
                },
            )
            buf2 = ""
            try:
                for chunk in answer_stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    buf2 += delta
                    _, ans = _split_thinking(buf2)
                    history[-1]["content"] = ans
                    yield "", history, chunks_html
            finally:
                answer_stream.close()

        if not history[-2]["content"]:
            history.pop(-2)
            yield "", history, chunks_html

    with gr.Blocks(title=f"RAG Chat ({model_id})", css=CHUNKS_CSS) as demo:
        gr.Markdown(
            f"# RAG Chat — Chroma + Gemma 4\n"
            f"Model: `{model_id}` · Collection: `{collection_name}` "
            f"({collection.count()} chunks) · Encoder: `{embedding_model}`"
        )
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            top_k = gr.Slider(1, 10, value=default_top_k, step=1, label="Top-k chunks")
            think_budget = gr.Slider(
                0, 4000, value=0, step=100,
                label="Thinking budget (tokens, 0 = unlimited)",
            )
        with gr.Row():
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM, label="System prompt", lines=2, scale=4,
            )
            use_rag = gr.Checkbox(value=True, label="Use retrieval", scale=1)
            enable_thinking = gr.Checkbox(value=True, label="Enable thinking", scale=1)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", label="Conversation", height=520)
                msg = gr.Textbox(label="Your message", placeholder="Ask something the corpus knows about…")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=2):
                gr.Markdown("### Retrieved chunks")
                chunks_view = gr.HTML(value=_render_chunks([]))

        inputs = [
            msg, chatbot, system_prompt, use_rag, top_k,
            enable_thinking, think_budget, temperature, top_p,
        ]
        outputs = [msg, chatbot, chunks_view]
        msg.submit(chat, inputs=inputs, outputs=outputs)
        send.click(chat, inputs=inputs, outputs=outputs)
        clear.click(lambda: ([], _render_chunks([])), outputs=[chatbot, chunks_view])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"RAG chat UI deployed: {app.url}")
