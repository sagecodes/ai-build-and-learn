"""RAG with a live embedding-space visualizer.

Takes the Chroma collection built by `rag-chroma-flyte/pipeline.py` and adds a
2D UMAP projection on the side. Each query is embedded, retrieved top-k from
Chroma, then projected into the same fitted UMAP space — viewers see *where*
the model went looking and *why* those neighbors landed where they did.

Architecture notes:
  - `@on_startup` does the one-time work: load Chroma, fetch all embeddings,
    fit UMAP (~10–30s on 3k rows), cache the reducer + 2D coords in module state.
  - `@server` runs Gradio. Per turn it embeds the query, retrieves top-k,
    *projects the query through the cached reducer*, redraws the Plotly
    scatter, and streams Gemma's answer with the chunks injected.
  - Plotly is SVG-rendered (CLAUDE.md exception), so colors live as Python
    constants — CSS classes can't reach into the chart.

Deploy:
    RAG_PIPELINE_RUN=<run-name> python chat_app.py
"""

from __future__ import annotations

import os

import flyte
import flyte.app
import flyte.io


# ── Endpoints + index source ──────────────────────────────────────────────────

VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "rag_demo"
DEFAULT_TOP_K = 4

PIPELINE_TASK = "rag-chroma-pipeline.rag_pipeline"


# ── Plotly colors (CLAUDE.md exception: SVG can't read CSS classes) ───────────

COLOR_BG_POINT = "#cfd5db"          # whole collection — muted gray
COLOR_RETRIEVED = [                  # top-k highlighted; rank 1 = warmest
    "#ef4444",  # 1: red-500
    "#f97316",  # 2: orange-500
    "#eab308",  # 3: yellow-500
    "#22c55e",  # 4: green-500
    "#06b6d4",  # 5: cyan-500
    "#3b82f6",  # 6: blue-500
    "#8b5cf6",  # 7: violet-500
    "#d946ef",  # 8: fuchsia-500
    "#ec4899",  # 9: pink-500
    "#10b981",  # 10: emerald-500
]
COLOR_QUERY = "#ffd700"              # gold star


# ── Image ─────────────────────────────────────────────────────────────────────

chat_image = (
    flyte.Image.from_debian_base(
        name="rag-umap-image",
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    .with_pip_packages(
        "gradio==5.42.0",
        "openai>=1.50.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
        "umap-learn>=0.5.5",
        "plotly>=5.20.0",
        "numpy",
    )
)


# ── App env ───────────────────────────────────────────────────────────────────
#
# Reuse the rag-chroma-flyte pipeline output via RunOutput. Set
# RAG_PIPELINE_RUN=<name> to pin a specific run; otherwise the latest
# succeeded run of `rag_pipeline` is used.

_pipeline_run = os.environ.get("RAG_PIPELINE_RUN")
_chroma_run_output = (
    flyte.app.RunOutput(type="directory", run_name=_pipeline_run)
    if _pipeline_run
    else flyte.app.RunOutput(type="directory", task_name=PIPELINE_TASK)
)

env = flyte.app.AppEnvironment(
    name="rag-umap-viz",
    image=chat_image,
    # Single-node devbox is tight on CPU once vLLM + sibling chat apps are
    # all ACTIVE. 1 core is enough — UMAP fit runs full-tilt for ~30s at
    # startup, then the pod is idle between queries. Bump if you have
    # spare cores.
    resources=flyte.Resources(cpu="1", memory="6Gi"),
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
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)


# ── State shared across hooks ─────────────────────────────────────────────────

state: dict = {}


# ── Startup: load Chroma, fit UMAP, cache 2D coords ───────────────────────────

@env.on_startup
async def fit_umap(
    vllm_url: str,
    model_id: str,
    chroma_dir: flyte.io.Dir,
    embedding_model: str,
    collection_name: str,
    default_top_k: str,
) -> None:
    import time

    import chromadb
    import numpy as np
    import umap
    from sentence_transformers import SentenceTransformer

    print(f"[startup] Loading encoder: {embedding_model}", flush=True)
    encoder = SentenceTransformer(embedding_model)

    print(f"[startup] Opening Chroma at {chroma_dir.path!r}", flush=True)
    client = chromadb.PersistentClient(path=chroma_dir.path)
    collection = client.get_collection(name=collection_name)
    n = collection.count()
    print(f"[startup] Collection '{collection_name}' has {n} chunks", flush=True)

    print("[startup] Fetching all embeddings…", flush=True)
    rows = collection.get(include=["embeddings", "documents", "metadatas"])
    embeddings = np.asarray(rows["embeddings"])
    documents = list(rows["documents"])
    ids = list(rows["ids"])
    doc_ids = [(m or {}).get("doc_id", "?") for m in rows["metadatas"]]

    # UMAP knobs picked for "looks clustered on a screen" rather than "best
    # ARI score": n_neighbors=15 keeps local structure tight, min_dist=0.1
    # leaves visible whitespace between clusters, cosine matches BGE.
    print(f"[startup] Fitting UMAP on {len(embeddings)} × {embeddings.shape[1]}d…", flush=True)
    t0 = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=15,
        min_dist=0.1,
        metric="cosine",
        random_state=42,
    )
    coords_2d = reducer.fit_transform(embeddings)
    print(f"[startup] UMAP fit done in {time.time() - t0:.1f}s", flush=True)

    state.update({
        "encoder": encoder,
        "collection": collection,
        "reducer": reducer,
        "coords_2d": coords_2d,           # (N, 2) np.ndarray
        "embeddings": embeddings,         # not strictly needed at runtime, kept for debug
        "documents": documents,
        "ids": ids,
        "doc_ids": doc_ids,
        "id_to_idx": {cid: i for i, cid in enumerate(ids)},
    })
    print(f"[startup] Ready. {n} chunks embedded + projected to 2D.", flush=True)


# ── Gemma thinking-block parser (same as the sibling chat apps) ───────────────

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


# ── CSS for the small status / chunks panel ───────────────────────────────────

def _rank_class_rules() -> str:
    # One CSS rule per rank — keeps the rank pip on the chunk card the same
    # color the Plotly scatter uses for that rank, without writing computed
    # inline styles (CLAUDE.md: classes only).
    return "\n".join(
        f".chunk-rank-pip.rank-{i} {{ background: {COLOR_RETRIEVED[(i - 1) % len(COLOR_RETRIEVED)]}; }}"
        for i in range(1, len(COLOR_RETRIEVED) + 1)
    )


PANELS_CSS = """
.chunks-panel { display: flex; flex-direction: column; gap: 8px; }
.chunks-empty {
    color: var(--body-text-color-subdued);
    font-style: italic;
    padding: 10px;
}
.chunk-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 8px 10px;
    background: var(--background-fill-secondary);
}
.chunk-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 4px;
}
.chunk-rank { font-weight: 600; }
.chunk-rank-pip {
    display: inline-block;
    width: 0.6em;
    height: 0.6em;
    border-radius: 50%;
    margin-right: 0.4em;
    vertical-align: middle;
}
.chunk-text {
    font-size: 0.88rem;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
}
""" + _rank_class_rules()


def _esc(s: str) -> str:
    return (s or "").replace("<", "&lt;").replace(">", "&gt;")


def _render_chunks(chunks: list[dict]) -> str:
    if not chunks:
        return '<div class="chunks-empty">No chunks retrieved yet — send a message.</div>'
    cards = []
    for i, c in enumerate(chunks, 1):
        sim = max(0.0, 1.0 - (c["distance"] / 2.0))
        rank_class = f"rank-{((i - 1) % len(COLOR_RETRIEVED)) + 1}"
        cards.append(
            '<div class="chunk-card">'
            '<div class="chunk-meta">'
            f'<span class="chunk-rank"><span class="chunk-rank-pip {rank_class}"></span>'
            f'#{i} · doc {c["doc_id"]}</span>'
            f'<span>sim {sim:.3f}</span>'
            '</div>'
            f'<div class="chunk-text">{_esc(c["text"])}</div>'
            '</div>'
        )
    return f'<div class="chunks-panel">{"".join(cards)}</div>'


# ── Server ────────────────────────────────────────────────────────────────────

@env.server
def chat_server(
    vllm_url: str,
    model_id: str,
    chroma_dir: flyte.io.Dir,
    embedding_model: str,
    collection_name: str,
    default_top_k: str,
):
    import sys
    import traceback
    try:
        _run(vllm_url, model_id, embedding_model, collection_name, int(default_top_k))
    except BaseException as e:
        print(f"!!! chat_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _figure_with_overlay(retrieved: list[dict], query_xy: tuple[float, float] | None,
                        query_text: str = ""):
    """Build a fresh Plotly figure with collection + retrieved + query traces.

    Traces are constructed with their final data — no mutation of trace
    attributes after the fact, which Plotly's serializer doesn't always
    propagate cleanly through Gradio.
    """
    import plotly.graph_objects as go

    coords = state["coords_2d"]
    docs = state["documents"]
    doc_ids = state["doc_ids"]

    bg_hover = [
        f"doc {did}<br>{_esc(d[:160])}{'…' if len(d) > 160 else ''}"
        for did, d in zip(doc_ids, docs)
    ]

    traces = [
        # Whole collection. Scattergl for the 3k-point cloud — SVG would lag.
        go.Scattergl(
            x=coords[:, 0].tolist(),
            y=coords[:, 1].tolist(),
            mode="markers",
            marker=dict(color=COLOR_BG_POINT, size=5, opacity=0.55),
            hoverinfo="text",
            hovertext=bg_hover,
            name="collection",
            showlegend=False,
        ),
    ]

    if retrieved:
        xs = [r["xy"][0] for r in retrieved]
        ys = [r["xy"][1] for r in retrieved]
        colors = [COLOR_RETRIEVED[i % len(COLOR_RETRIEVED)] for i in range(len(retrieved))]
        labels = [f"#{i + 1}" for i in range(len(retrieved))]
        hovers = [
            f"#{i + 1} · doc {r['doc_id']} · sim {max(0.0, 1.0 - r['distance'] / 2.0):.3f}<br>"
            f"{_esc(r['text'][:200])}{'…' if len(r['text']) > 200 else ''}"
            for i, r in enumerate(retrieved)
        ]
        # Plain Scatter (SVG) for the overlay — only ~10 points, and SVG
        # avoids a class of WebGL-context lifecycle issues we saw when
        # mutating Scattergl traces in place.
        traces.append(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            marker=dict(color=colors, size=16, line=dict(color="#111111", width=1.5)),
            text=labels,
            textposition="top center",
            textfont=dict(size=12),
            hoverinfo="text",
            hovertext=hovers,
            name="retrieved",
            showlegend=False,
        ))

    if query_xy is not None:
        traces.append(go.Scatter(
            x=[query_xy[0]], y=[query_xy[1]],
            mode="markers",
            marker=dict(color=COLOR_QUERY, size=24, symbol="star",
                        line=dict(color="#7a5b00", width=1.5)),
            hoverinfo="text",
            hovertext=[f"query: {_esc(query_text[:160])}"],
            name="query",
            showlegend=False,
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=10, r=10, t=10, b=10),
        height=620,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", scaleratio=1),
        dragmode="pan",
    )
    return fig


def _run(
    vllm_url: str,
    model_id: str,
    embedding_model: str,
    collection_name: str,
    default_top_k: int,
):
    import gradio as gr
    import numpy as np
    from openai import OpenAI

    encoder = state["encoder"]
    collection = state["collection"]
    reducer = state["reducer"]
    coords_2d = state["coords_2d"]
    id_to_idx = state["id_to_idx"]
    doc_ids = state["doc_ids"]

    print(f"[chat_server] gradio version: {gr.__version__}", flush=True)
    print(f"[chat_server] vLLM at {vllm_url}/v1 (model={model_id})", flush=True)

    llm = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="not-used")

    DEFAULT_SYSTEM = (
        "You are a helpful assistant. Use the provided CONTEXT to answer. "
        "If the answer is not in the context, say you don't know — do not invent. "
        "Cite sources as [#N] where N matches the chunk numbers in the chart."
    )
    MAX_TOTAL_TOKENS = 4096

    def retrieve(query: str, top_k: int) -> tuple[list[dict], tuple[float, float] | None]:
        """Retrieve top-k from Chroma and project the query into UMAP space."""
        if not query.strip():
            return [], None
        vec = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        res = collection.query(
            query_embeddings=vec.tolist(),
            n_results=max(1, top_k),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            idx = id_to_idx.get(cid)
            if idx is None:
                # Shouldn't happen — collection.query returned an id we don't know about.
                continue
            out.append({
                "id": cid,
                "doc_id": (meta or {}).get("doc_id", doc_ids[idx] if idx < len(doc_ids) else "?"),
                "text": doc,
                "distance": float(dist),
                "xy": (float(coords_2d[idx, 0]), float(coords_2d[idx, 1])),
            })

        # Project the query into the same fitted UMAP space.
        q2d = reducer.transform(vec)
        q_xy = (float(q2d[0, 0]), float(q2d[0, 1]))
        return out, q_xy

    def build_context_block(chunks: list[dict]) -> str:
        if not chunks:
            return ""
        lines = ["CONTEXT:"]
        for i, c in enumerate(chunks, 1):
            lines.append(f"[#{i}] (doc {c['doc_id']})\n{c['text']}")
        return "\n\n".join(lines)

    def chat(message, history, system_prompt, use_rag, top_k,
             enable_thinking, temperature, top_p):
        if not message or not message.strip():
            yield "", history, _render_chunks([]), gr.skip()
            return

        chunks, q_xy = retrieve(message, int(top_k)) if use_rag else ([], None)
        chunks_html = _render_chunks(chunks)
        fig = _figure_with_overlay(chunks, q_xy, message)

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
            {"role": "assistant", "content": ""},
        ]
        # First yield carries the new figure. Subsequent yields during streaming
        # use gr.skip() — re-serializing a 3k-point figure 100× per second
        # blanks the chart in the browser.
        yield "", history, chunks_html, fig

        sys_text = (system_prompt or DEFAULT_SYSTEM).strip()
        ctx_block = build_context_block(chunks)
        if ctx_block:
            sys_text = f"{sys_text}\n\n{ctx_block}"

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
                yield "", history, chunks_html, gr.skip()
        finally:
            stream.close()

        if not history[-2]["content"]:
            history.pop(-2)
            yield "", history, chunks_html, gr.skip()

    # ── UI ────────────────────────────────────────────────────────────────────

    with gr.Blocks(title=f"RAG · UMAP Visualizer ({model_id})", css=PANELS_CSS) as demo:
        gr.Markdown(
            f"# RAG with embedding-space visualizer\n"
            f"Model: `{model_id}` · Collection: `{collection_name}` "
            f"({collection.count()} chunks projected to 2D via UMAP) · Encoder: `{embedding_model}`"
        )
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            top_k = gr.Slider(1, 10, value=default_top_k, step=1, label="Top-k chunks")
        with gr.Row():
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM, label="System prompt", lines=2, scale=4,
            )
            use_rag = gr.Checkbox(value=True, label="Use retrieval", scale=1)
            enable_thinking = gr.Checkbox(value=True, label="Enable thinking", scale=1)

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages", label="Conversation", height=420)
                msg = gr.Textbox(label="Your message", placeholder="Ask the corpus something…")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
                gr.Markdown("### Retrieved chunks")
                chunks_view = gr.HTML(value=_render_chunks([]))
            with gr.Column(scale=3):
                gr.Markdown("### Embedding space (UMAP, cosine)")
                plot_view = gr.Plot(value=_figure_with_overlay([], None, ""))

        inputs = [msg, chatbot, system_prompt, use_rag, top_k,
                  enable_thinking, temperature, top_p]
        outputs = [msg, chatbot, chunks_view, plot_view]
        msg.submit(chat, inputs=inputs, outputs=outputs)
        send.click(chat, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ([], _render_chunks([]), _figure_with_overlay([], None, "")),
            outputs=[chatbot, chunks_view, plot_view],
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"RAG · UMAP visualizer deployed: {app.url}")
