"""Gradio Graph-RAG chat UI — Neo4j (HTTP) + Gemma 4 vLLM.

Three retrieval modes side by side:
  1. Vector            — pure semantic search (baseline; same as rag-chroma).
  2. Vector + Expand   — vector top-k, then 1-hop graph expansion across
                          CITES / AUTHORED_BY / IN_CATEGORY edges.
  3. Hybrid (RRF)      — vector top-k AND a separate Cypher pass for
                          most-cited papers in shared categories, fused
                          with reciprocal rank.

Same shape as topics/vectorstore/rag-chroma-flyte/chat_app.py, but talks to
Neo4j over the HTTP Cypher API (Knative is HTTP-only, so no Bolt). The right
panel shows retrieved Paper nodes plus, in modes 2 and 3, a Graph Relations
list so the audience can see what graph context the LLM got beyond raw
vector hits.

Deploy (after `python neo4j_app.py` and the Gemma vLLM is up):
    python chat_app.py
"""

from __future__ import annotations

import flyte
import flyte.app


# ── Gemma 4 vLLM endpoint info ────────────────────────────────────────────────
# Hard-coded to match the running gemma4-dgx-devbox vLLM app. If you switched
# to the 31B variant, change these two strings.

VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"

# ── Retrieval knobs ───────────────────────────────────────────────────────────

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_TOP_K = 4
EXPAND_NEIGHBOR_LIMIT = 8       # max neighbors per seed in mode 2
RRF_K = 60                       # reciprocal-rank-fusion constant

# ── Image ─────────────────────────────────────────────────────────────────────

chat_image = (
    flyte.Image.from_debian_base(
        name="graphrag-chat-image",
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    .with_pip_packages(
        # gr.Chatbot(type="messages") needs Gradio 5.x — same constraint as
        # the sibling rag-chroma chat app.
        "gradio==5.42.0",
        "openai>=1.50.0",
        "httpx>=0.27.0",
        "sentence-transformers>=3.0.0",
    )
)


# ── App env ───────────────────────────────────────────────────────────────────
#
# Unlike the rag-chroma chat app there's no `RunOutput(directory)` to mount —
# Neo4j is a long-lived service, the chat app just connects to it over HTTP
# inside the cluster. All knobs come through Parameters so the deploy doesn't
# bake URLs into the image.

env = flyte.app.AppEnvironment(
    name="graphrag-chat-ui",
    image=chat_image,
    resources=flyte.Resources(cpu="1", memory="4Gi"),
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
            name="neo4j_url",
            # Must mirror config.NEO4J_HTTP_URL — keeping it independent here
            # so the chat app can be redeployed without re-importing the
            # pipeline package.
            value=("http://graphrag-neo4j-flytesnacks-development."
                   "flyte.svc.cluster.local"),
            env_var="NEO4J_URL",
        ),
        flyte.app.Parameter(name="neo4j_user", value="neo4j"),
        flyte.app.Parameter(name="neo4j_password", value="graphrag-demo"),
        flyte.app.Parameter(name="embedding_model", value=EMBEDDING_MODEL),
        flyte.app.Parameter(name="default_top_k", value=str(DEFAULT_TOP_K)),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
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


# ── CSS for the retrieval panel (CLAUDE.md: classes only, no inline) ──────────

PANEL_CSS = """
.gr-panel { display: flex; flex-direction: column; gap: 12px; }
.gr-empty {
    color: var(--body-text-color-subdued);
    font-style: italic;
    padding: 12px;
}
.gr-section-header {
    font-size: 0.85rem;
    color: var(--body-text-color-subdued);
    margin: 6px 0 4px 0;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.gr-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 10px 12px;
    background: var(--background-fill-secondary);
}
.gr-card-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 6px;
}
.gr-rank { font-weight: 600; }
.gr-score { font-variant-numeric: tabular-nums; }
.gr-title { font-weight: 600; font-size: 0.95rem; margin-bottom: 4px; }
.gr-title a, .gr-rank a { color: inherit; text-decoration: none; border-bottom: 1px dotted var(--body-text-color-subdued); }
.gr-title a:hover, .gr-rank a:hover { color: var(--color-accent); border-bottom-color: var(--color-accent); }
.gr-snippet {
    font-size: 0.88rem;
    line-height: 1.4;
    color: var(--body-text-color);
}
.gr-source { font-style: italic; font-size: 0.75rem; color: var(--body-text-color-subdued); }
.gr-edge {
    font-size: 0.85rem;
    padding: 4px 8px;
    border-left: 3px solid var(--color-accent);
    background: var(--background-fill-primary);
    margin-bottom: 4px;
    line-height: 1.4;
}
.gr-edge-rel {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--color-accent);
    font-weight: 600;
}
"""


def _render_panel(papers: list[dict], edges: list[dict]) -> str:
    if not papers and not edges:
        return '<div class="gr-empty">No retrieval yet — send a message.</div>'

    sections = []

    if papers:
        cards = []
        for i, p in enumerate(papers, 1):
            score = p.get("score")
            score_text = f"{score:.3f}" if isinstance(score, (int, float)) else "—"
            source = p.get("source", "vector")
            snippet = (p.get("abstract") or "").strip().replace("<", "&lt;").replace(">", "&gt;")
            if len(snippet) > 280:
                snippet = snippet[:280].rsplit(" ", 1)[0] + "…"
            title = (p.get("title") or p.get("id") or "?").replace("<", "&lt;").replace(">", "&gt;")
            url = (p.get("url") or "").replace('"', "&quot;")
            # Derive a short, readable paper label from the URL: arxiv ID
            # when available, S2 short hash otherwise. Long S2 paperIds
            # like `4fe445a1df3b73c…` are noise in the meta line.
            if url.startswith("https://arxiv.org/abs/"):
                paper_label = "arXiv:" + url.rsplit("/", 1)[-1]
            elif "/paper/" in url:
                paper_label = "S2:" + url.rsplit("/", 1)[-1][:8]
            else:
                paper_label = (p.get("id") or "?")[:10]
            paper_link = (
                f'<a href="{url}" target="_blank" rel="noopener">{paper_label}</a>'
                if url else paper_label
            )
            title_html = (
                f'<a href="{url}" target="_blank" rel="noopener">{title}</a>'
                if url else title
            )
            cards.append(
                '<div class="gr-card">'
                '<div class="gr-card-meta">'
                f'<span class="gr-rank">#{i} · {paper_link} · {p.get("year", "?")}</span>'
                f'<span class="gr-score">score {score_text}</span>'
                '</div>'
                f'<div class="gr-title">{title_html}</div>'
                f'<div class="gr-snippet">{snippet}</div>'
                f'<div class="gr-source">via {source}</div>'
                '</div>'
            )
        sections.append(
            '<div class="gr-section-header">📄 Retrieved papers</div>'
            + "".join(cards)
        )

    if edges:
        edge_html = []
        for e in edges:
            rel = e["rel"]
            arrow = "→" if e.get("outgoing") else "←"
            src_id = e.get("src_id", "?")
            target = (e.get("target_label") or "?").replace("<", "&lt;").replace(">", "&gt;")
            target_kind = e.get("target_kind", "?")
            edge_html.append(
                f'<div class="gr-edge">'
                f'paper <b>{src_id}</b> '
                f'<span class="gr-edge-rel">{arrow} {rel}</span> '
                f'{target_kind}: {target}'
                '</div>'
            )
        sections.append(
            '<div class="gr-section-header">🕸 Graph relations</div>'
            + "".join(edge_html)
        )

    return f'<div class="gr-panel">{"".join(sections)}</div>'


# ── Server ────────────────────────────────────────────────────────────────────

@env.server
def chat_server(
    vllm_url: str,
    model_id: str,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    embedding_model: str,
    default_top_k: str,
):
    import sys
    import traceback
    try:
        _run(vllm_url, model_id, neo4j_url, neo4j_user, neo4j_password,
             embedding_model, int(default_top_k))
    except BaseException as e:
        print(f"!!! chat_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _run(
    vllm_url: str,
    model_id: str,
    neo4j_url: str,
    neo4j_user: str,
    neo4j_password: str,
    embedding_model: str,
    default_top_k: int,
):
    from typing import Any

    import gradio as gr
    import httpx
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    print(f"[chat_server] gradio version: {gr.__version__}", flush=True)
    print(f"[chat_server] vLLM at {vllm_url}/v1 (model={model_id})", flush=True)
    print(f"[chat_server] Neo4j at {neo4j_url}", flush=True)
    print(f"[chat_server] Loading encoder: {embedding_model}", flush=True)

    encoder = SentenceTransformer(embedding_model)
    llm = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="not-used")
    neo = httpx.Client(
        base_url=neo4j_url,
        auth=(neo4j_user, neo4j_password),
        timeout=30.0,
    )

    # Smoke-check Neo4j is reachable so we fail loud on startup, not on
    # the first user message.
    info = neo.get("/").raise_for_status().json()
    print(f"[chat_server] Neo4j {info.get('neo4j_version')} "
          f"{info.get('neo4j_edition')}", flush=True)

    DEFAULT_SYSTEM = (
        "You are a helpful assistant answering questions over a corpus of AI "
        "research papers. Use the provided CONTEXT block — papers and graph "
        "relations — to answer. If the answer is not in the context, say you "
        "don't know — do not invent. Cite sources as [#N] where N is the "
        "paper rank."
    )
    CHARS_PER_TOKEN = 3.5
    MAX_TOTAL_TOKENS = 4096

    # ── Cypher helpers ───────────────────────────────────────────────────────

    def _cypher(stmt: str, params: dict[str, Any] | None = None) -> list[list[Any]]:
        resp = neo.post("/db/neo4j/tx/commit", json={
            "statements": [{"statement": stmt, "parameters": params or {}}],
        })
        resp.raise_for_status()
        body = resp.json()
        if body.get("errors"):
            raise RuntimeError(f"Neo4j error: {body['errors']}")
        if not body.get("results"):
            return []
        return [row["row"] for row in body["results"][0].get("data", [])]

    # ── Mode 1: Vector ───────────────────────────────────────────────────────

    def _retrieve_vector(query: str, k: int) -> list[dict]:
        vec = encoder.encode([query], normalize_embeddings=True,
                             convert_to_numpy=True).tolist()[0]
        rows = _cypher(
            """
            CALL db.index.vector.queryNodes('paper_embedding_idx', $k, $vec)
            YIELD node, score
            RETURN node.id AS id, node.title AS title,
                   node.abstract AS abstract, node.year AS year,
                   node.url AS url, score
            """,
            {"k": k, "vec": vec},
        )
        return [
            {"id": r[0], "title": r[1], "abstract": r[2], "year": r[3],
             "url": r[4], "score": float(r[5]), "source": "vector"}
            for r in rows
        ]

    # ── Mode 2: Vector + 1-hop expansion ─────────────────────────────────────

    def _expand_neighbors(seed_ids: list[str]) -> list[dict]:
        if not seed_ids:
            return []
        rows = _cypher(
            f"""
            UNWIND $ids AS sid
            MATCH (s:Paper {{id: sid}})-[r]-(n)
            WHERE n:Paper OR n:Author OR n:Category
            WITH sid, r, n,
                 CASE labels(n)[0]
                     WHEN 'Paper'    THEN n.title
                     WHEN 'Author'   THEN n.name
                     WHEN 'Category' THEN n.code
                     ELSE coalesce(n.id, '?')
                 END AS target_label,
                 labels(n)[0] AS target_kind,
                 startNode(r).id = sid AS outgoing
            RETURN sid, type(r) AS rel, outgoing, target_kind, target_label,
                   coalesce(n.id, n.name, n.code) AS target_id
            LIMIT $limit
            """,
            {"ids": seed_ids, "limit": EXPAND_NEIGHBOR_LIMIT * len(seed_ids)},
        )
        return [
            {"src_id": r[0], "rel": r[1], "outgoing": bool(r[2]),
             "target_kind": r[3], "target_label": r[4], "target_id": r[5]}
            for r in rows
        ]

    def _retrieve_with_expand(query: str, k: int) -> tuple[list[dict], list[dict]]:
        seeds = _retrieve_vector(query, k)
        edges = _expand_neighbors([s["id"] for s in seeds])
        return seeds, edges

    # ── Mode 3: Hybrid (vector + category-cohort, fused via RRF) ─────────────

    def _retrieve_category_cohort(seed_ids: list[str], k: int) -> list[dict]:
        """Most-cited papers in the same Category as the seeds.

        This is the 'graph signal' half of the hybrid: papers a *graph-only*
        retriever might surface because they're authoritative within the
        topic area, even if their abstract isn't a great vector match.
        """
        if not seed_ids:
            return []
        rows = _cypher(
            """
            UNWIND $ids AS sid
            MATCH (s:Paper {id: sid})-[:IN_CATEGORY]->(c:Category)
                  <-[:IN_CATEGORY]-(other:Paper)
            WHERE NOT other.id IN $ids
            WITH other, count(DISTINCT c) AS shared_cats
            OPTIONAL MATCH ()-[r:CITES]->(other)
            WITH other, shared_cats, count(r) AS in_cites
            RETURN other.id AS id, other.title AS title,
                   other.abstract AS abstract, other.year AS year,
                   other.url AS url, shared_cats, in_cites
            ORDER BY in_cites DESC, shared_cats DESC
            LIMIT $k
            """,
            {"ids": seed_ids, "k": k},
        )
        return [
            {"id": r[0], "title": r[1], "abstract": r[2], "year": r[3],
             "url": r[4],
             "score": float(r[6]),  # in_cites count, displayed as score
             "source": f"graph (cited {r[6]}x, {r[5]} shared cats)"}
            for r in rows
        ]

    def _retrieve_hybrid(query: str, k: int) -> tuple[list[dict], list[dict]]:
        vec_hits = _retrieve_vector(query, k)
        graph_hits = _retrieve_category_cohort([h["id"] for h in vec_hits], k)

        # Reciprocal rank fusion: papers appearing in both lists get the most
        # weight; lone hits stay in but ranked lower.
        scores: dict[str, float] = {}
        record: dict[str, dict] = {}
        for rank, p in enumerate(vec_hits):
            scores[p["id"]] = scores.get(p["id"], 0.0) + 1.0 / (RRF_K + rank)
            record[p["id"]] = {**p, "source": "vector"}
        for rank, p in enumerate(graph_hits):
            scores[p["id"]] = scores.get(p["id"], 0.0) + 1.0 / (RRF_K + rank)
            if p["id"] in record:
                record[p["id"]]["source"] = "vector + graph"
            else:
                record[p["id"]] = p

        ranked_ids = sorted(scores, key=lambda i: scores[i], reverse=True)[:k]
        fused = [{**record[i], "score": scores[i]} for i in ranked_ids]

        # Show edges between fused papers so the audience sees the graph
        # structure that justified the fusion.
        edges = _expand_neighbors(ranked_ids)
        return fused, edges

    # ── Dispatcher + context block ───────────────────────────────────────────

    def retrieve(query: str, mode: str, k: int) -> tuple[list[dict], list[dict]]:
        if not query.strip():
            return [], []
        if mode == "Vector":
            return _retrieve_vector(query, k), []
        if mode == "Vector + Expand":
            return _retrieve_with_expand(query, k)
        if mode == "Hybrid (RRF)":
            return _retrieve_hybrid(query, k)
        return _retrieve_vector(query, k), []

    def build_context_block(papers: list[dict], edges: list[dict]) -> str:
        if not papers and not edges:
            return ""
        lines = ["CONTEXT:"]
        for i, p in enumerate(papers, 1):
            lines.append(
                f"\n[#{i}] (paper {p['id']}, {p.get('year', '?')}) "
                f"{p.get('title', '')}\n{(p.get('abstract') or '').strip()}"
            )
        if edges:
            lines.append("\nGRAPH RELATIONS:")
            id_to_rank = {p["id"]: i + 1 for i, p in enumerate(papers)}
            for e in edges:
                rank = id_to_rank.get(e["src_id"], "?")
                arrow = "→" if e["outgoing"] else "←"
                lines.append(
                    f"  [#{rank}] {arrow} {e['rel']} "
                    f"{e['target_kind']}: {e['target_label']}"
                )
        return "\n".join(lines)

    # ── Chat loop ────────────────────────────────────────────────────────────

    def chat(message, history, system_prompt, mode, top_k,
             enable_thinking, think_budget, temperature, top_p):
        if not message or not message.strip():
            yield "", history, _render_panel([], [])
            return

        papers, edges = retrieve(message, mode, int(top_k))
        panel_html = _render_panel(papers, edges)

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
            {"role": "assistant", "content": ""},
        ]
        yield "", history, panel_html

        sys_text = (system_prompt or DEFAULT_SYSTEM).strip()
        ctx_block = build_context_block(papers, edges)
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
                yield "", history, panel_html

                if (budget_chars and not answer and len(thinking) >= budget_chars):
                    capped = True
                    break
        finally:
            stream.close()

        if capped:
            history[-2]["content"] += f"\n\n_[capped at ~{think_budget} tokens]_"
            yield "", history, panel_html

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
                    yield "", history, panel_html
            finally:
                answer_stream.close()

        if not history[-2]["content"]:
            history.pop(-2)
            yield "", history, panel_html

    # ── UI ───────────────────────────────────────────────────────────────────

    with gr.Blocks(title=f"Graph RAG Chat ({model_id})", css=PANEL_CSS) as demo:
        gr.Markdown(
            f"# Graph RAG Chat — Neo4j + Gemma 4\n"
            f"Model: `{model_id}` · Encoder: `{embedding_model}` · "
            f"Neo4j HTTP API"
        )
        with gr.Row():
            mode = gr.Radio(
                choices=["Vector", "Vector + Expand", "Hybrid (RRF)"],
                value="Vector + Expand",
                label="Retrieval mode",
                scale=3,
            )
            top_k = gr.Slider(1, 10, value=default_top_k, step=1, label="Top-k", scale=1)
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature", scale=1)
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p", scale=1)
            think_budget = gr.Slider(
                0, 4000, value=0, step=100,
                label="Thinking budget (0 = unlimited)", scale=1,
            )
        with gr.Row():
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM, label="System prompt", lines=2, scale=4,
            )
            enable_thinking = gr.Checkbox(value=True, label="Enable thinking", scale=1)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", label="Conversation", height=560)
                msg = gr.Textbox(label="Your message",
                                 placeholder="e.g. What's the relationship between RAG and Self-RAG?")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column(scale=2):
                gr.Markdown("### Retrieval")
                panel = gr.HTML(value=_render_panel([], []))

        inputs = [
            msg, chatbot, system_prompt, mode, top_k,
            enable_thinking, think_budget, temperature, top_p,
        ]
        outputs = [msg, chatbot, panel]
        msg.submit(chat, inputs=inputs, outputs=outputs)
        send.click(chat, inputs=inputs, outputs=outputs)
        clear.click(lambda: ([], _render_panel([], [])), outputs=[chatbot, panel])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Graph RAG chat UI deployed: {app.url}")
