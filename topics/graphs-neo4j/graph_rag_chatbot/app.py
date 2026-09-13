"""
app.py — Gradio UI for the Everstorm GraphRAG chatbot.

Two tabs:
  Ingest Graph  — encode PDFs as base64 and run ingest_pipeline on Union
  Chat          — ask questions, see answer + retrieval mode badge + sources + entities

Run locally:
    python app.py

Deploy to Union:
    python app.py --deploy
"""

import base64
import json
import sys
from pathlib import Path

import flyte
import flyte.app
import gradio as gr

import config    # loads .env and calls flyte.init() for the right backend
import workflows  # imported at module level so flyte deploy bundles workflows


_CSS = """
/* ── Mode badges ──────────────────────────────────────────────────────── */
.mode-badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 12px;
    font-size: 0.8em;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}
.mode-hybrid    { background: #dbeafe; color: #1e40af; }
.mode-entity    { background: #ede9fe; color: #5b21b6; }
.mode-community { background: #d1fae5; color: #065f46; }

/* ── Source accordion ─────────────────────────────────────────────────── */
.source-accordion {
    margin-top: 10px;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.88em;
}
.source-accordion summary {
    padding: 6px 12px;
    background: rgba(255,255,255,0.07);
    cursor: pointer;
    font-weight: 600;
    color: var(--body-text-color, #e0e0e0);
    list-style: none;
}
.source-accordion summary:hover { background: rgba(255,255,255,0.13); }
.source-item {
    padding: 6px 12px;
    border-top: 1px solid rgba(255,255,255,0.07);
    color: var(--body-text-color-subdued, #aaa);
}

/* ── Entity list ──────────────────────────────────────────────────────── */
.entity-accordion {
    margin-top: 6px;
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    overflow: hidden;
    font-size: 0.88em;
}
.entity-accordion summary {
    padding: 6px 12px;
    background: rgba(255,255,255,0.07);
    cursor: pointer;
    font-weight: 600;
    color: var(--body-text-color, #e0e0e0);
    list-style: none;
}
.entity-accordion summary:hover { background: rgba(255,255,255,0.13); }
.entity-item {
    padding: 4px 12px;
    border-top: 1px solid rgba(255,255,255,0.07);
    color: var(--body-text-color-subdued, #aaa);
    font-size: 0.85em;
}

/* ── Run link ─────────────────────────────────────────────────────────── */
.run-link a {
    display: inline-block;
    padding: 6px 14px;
    background: #5865f2;
    color: #fff;
    border-radius: 6px;
    font-weight: 600;
    text-decoration: none;
}
.run-link a:hover { background: #4752c4; }

/* ── Log box ──────────────────────────────────────────────────────────── */
.log-box textarea { font-family: monospace !important; font-size: 0.85em !important; }

/* ── Sidebar divider ──────────────────────────────────────────────────── */
.tab-sidebar { border-right: 1px solid rgba(255,255,255,0.1); padding-right: 16px !important; }

/* ── Retrieval panel ──────────────────────────────────────────────────── */
.retrieval-panel {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 14px;
    margin-top: 12px;
}
.panel-header {
    font-size: 0.72em;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: rgba(255,255,255,0.35);
    margin-bottom: 8px;
}
.panel-reasoning {
    font-size: 0.92em;
    color: var(--body-text-color-subdued, #aaa);
    line-height: 1.5;
    margin: 8px 0 12px;
}
.panel-pipeline {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px;
    margin-bottom: 12px;
}
.pipeline-step {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    padding: 3px 8px;
    border-radius: 5px;
    font-size: 0.76em;
    white-space: nowrap;
    color: var(--body-text-color, #e0e0e0);
}
.pipeline-arrow {
    color: rgba(255,255,255,0.3);
    font-size: 0.76em;
}
.panel-stats {
    display: flex;
    gap: 10px;
    font-size: 0.88em;
    color: var(--body-text-color-subdued, #aaa);
}
.panel-mode-desc {
    font-size: 0.93em;
    color: rgba(255,255,255,0.7);
    font-style: italic;
    line-height: 1.5;
    margin: 6px 0 10px;
}
"""

# ── Union App deployment environment ──────────────────────────────────────────

serving_env = flyte.app.AppEnvironment(
    name="everstorm-graphrag-chatbot",
    image="docker.io/johndellenbaugh/graphrag-app:latest",
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="NEO4J_URI",         as_env_var="NEO4J_URI"),
        flyte.Secret(key="NEO4J_USERNAME",    as_env_var="NEO4J_USERNAME"),
        flyte.Secret(key="NEO4J_PASSWORD",    as_env_var="NEO4J_PASSWORD"),
    ],
    env_vars={"FLYTE_BACKEND": "cluster"},
    port=7860,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)

# ── HTML builders ──────────────────────────────────────────────────────────────

def _mode_badge(mode: str) -> str:
    label = {"hybrid": "Hybrid", "entity": "Entity", "community": "Community"}.get(mode, mode)
    return f'<span class="mode-badge mode-{mode}">{label}</span>'


def build_sources_accordion(sources: list) -> str:
    items = "".join(
        f'<div class="source-item">{s}</div>' for s in sources
    )
    return (
        f'<details class="source-accordion">'
        f'<summary>📄 {len(sources)} source document(s)</summary>'
        f'{items}'
        f'</details>'
    )


def build_entities_accordion(entities: list) -> str:
    if not entities:
        return ""
    items = "".join(f'<div class="entity-item">· {e}</div>' for e in entities)
    return (
        f'<details class="entity-accordion">'
        f'<summary>🔗 {len(entities)} entity/entities used</summary>'
        f'{items}'
        f'</details>'
    )


_PIPELINE_STEPS = {
    "hybrid":    [("📊", "Vector Search"), ("🕸️", "Graph: MENTIONS")],
    "entity":    [("🔍", "Entity Extract"), ("🕸️", "Graph: RELATED")],
    "community": [("🏘️", "Graph: Community Sim")],
}

_MODE_DESCRIPTIONS = {
    "hybrid": (
        "Vector search finds the closest text chunks by embedding similarity — "
        "the same technique used in vector RAG. "
        "The graph layer then follows MENTIONS edges from those chunks to the entities they reference, "
        "adding structured relationship context that pure vector search cannot provide."
    ),
    "entity": (
        "Named entities are extracted from your question, then matched directly to nodes in the graph. "
        "RELATED edges traverse their connections to surface how things are linked — "
        "no vector search needed because the graph index is the lookup."
    ),
    "community": (
        "Pre-computed community summaries are embedded at ingest time. "
        "Your question is embedded and matched by cosine similarity to find the closest topic cluster, "
        "then all member entities of that community are returned — useful for broad 'what does X offer' questions."
    ),
}

def build_retrieval_panel(mode: str, reasoning: str, sources: list, entities: list) -> str:
    label = {"hybrid": "Hybrid", "entity": "Entity", "community": "Community"}.get(mode, mode)
    steps = _PIPELINE_STEPS.get(mode, [])
    mode_desc = _MODE_DESCRIPTIONS.get(mode, "")
    pipeline_html = '<span class="pipeline-arrow">→</span>'.join(
        f'<span class="pipeline-step">{ico} {name}</span>'
        for ico, name in steps
    )
    return (
        f'<div class="retrieval-panel">'
        f'<div class="panel-header">Last Query Retrieval</div>'
        f'<span class="mode-badge mode-{mode}">{label}</span>'
        f'<div class="panel-reasoning">{reasoning}</div>'
        f'<div class="panel-pipeline">{pipeline_html}</div>'
        f'<div class="panel-mode-desc">{mode_desc}</div>'
        f'<div class="panel-stats">'
        f'<span>📄 {len(sources)} source(s)</span>'
        f'<span>🔗 {len(entities)} entity/entities</span>'
        f'</div>'
        f'</div>'
    )


def build_run_link(run) -> str:
    try:
        return f'<a href="{run.url}" target="_blank">🔗 View run on Union</a>'
    except Exception:
        return ""


# ── Ingest handler ─────────────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data"


def run_ingest(uploaded_files):
    log_lines: list[str] = []

    def emit(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    filenames: list[str] = []
    pdf_bytes_b64: list[str] = []

    if uploaded_files:
        yield emit(f"📂 Encoding {len(uploaded_files)} uploaded PDF(s)..."), ""
        for file_path in uploaded_files:
            src = Path(file_path)
            filenames.append(src.name)
            pdf_bytes_b64.append(base64.b64encode(src.read_bytes()).decode())
            yield emit(f"   ✅ {src.name}"), ""
    else:
        pdf_paths = sorted(_DATA_DIR.glob("*.pdf"))
        if not pdf_paths:
            yield emit("⚠️  No PDFs found — upload files or add PDFs to data/"), ""
            return
        yield emit(f"📂 Encoding {len(pdf_paths)} PDF(s) from data/..."), ""
        for p in pdf_paths:
            filenames.append(p.name)
            pdf_bytes_b64.append(base64.b64encode(p.read_bytes()).decode())
            yield emit(f"   ✅ {p.name}"), ""

    yield emit("\n🚀 Dispatching ingest_pipeline → Union cluster..."), ""

    try:
        from workflows import ingest_pipeline
        run = flyte.run(
            ingest_pipeline,
            filenames=filenames,
            pdf_bytes_b64=pdf_bytes_b64,
        )
        link = build_run_link(run)
        yield emit("⏳ Running on Union — waiting for results..."), link

        run.wait()
        summary = json.loads(run.outputs().o0)

        yield emit(
            f"\n✅ Ingest complete!\n"
            f"   Communities summarized : {summary.get('communities_summarized', '—')}"
        ), link

    except Exception as exc:
        yield emit(f"\n❌ Error: {exc}"), ""


# ── Chat handler ───────────────────────────────────────────────────────────────

def chat(question, history):
    question = question.strip()
    history = list(history or [])

    if not question:
        return history, ""

    history.append({"role": "user", "content": question})

    try:
        from workflows import query_pipeline
        run = flyte.run(query_pipeline, question=question)
        run.wait()
        result = json.loads(run.outputs().o0)

        answer         = result["answer"]
        mode           = result.get("retrieval_mode", "hybrid")
        sources        = result.get("sources", [])
        entities       = result.get("entities_used", [])
        routing_reason = result.get("routing_reason", "")

        badge    = _mode_badge(mode)
        src_html = build_sources_accordion(sources) if sources else ""
        ent_html = build_entities_accordion(entities)

        history.append({
            "role": "assistant",
            "content": f"{answer}\n\n{badge}{src_html}{ent_html}",
        })

        panel_html = build_retrieval_panel(mode, routing_reason, sources, entities)

    except Exception as exc:
        history.append({
            "role": "assistant",
            "content": f"❌ Error: {exc}",
        })
        panel_html = ""

    return history, panel_html


# ── UI layout ──────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Everstorm GraphRAG Chatbot") as app:

        gr.Markdown("# Everstorm Outfitters — GraphRAG Chatbot")
        gr.Markdown(
            "Knowledge-graph Q&A powered by Neo4j + Claude, compute running on Union."
        )

        with gr.Tabs():

            # ── Tab 1: Ingest ──────────────────────────────────────────────────
            with gr.Tab("📥 Ingest Graph"):

                with gr.Row():
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        file_upload = gr.File(
                            file_types=[".pdf"],
                            file_count="multiple",
                            label="Upload PDFs (optional)",
                        )

                    with gr.Column(scale=4):
                        gr.Markdown(
                            "Upload PDFs to add them to the graph, or leave empty to "
                            "use PDFs already in `data/`. Extracts entities and relationships, "
                            "writes the graph to Neo4j AuraDB, and generates community summaries."
                        )
                        ingest_btn = gr.Button("🚀 Run Ingest on Union", variant="primary")
                        run_link = gr.HTML(elem_classes=["run-link"])
                        status_log = gr.Textbox(
                            label="Status Log",
                            lines=14,
                            interactive=False,
                            elem_classes=["log-box"],
                        )

                ingest_btn.click(
                    fn=run_ingest,
                    inputs=[file_upload],
                    outputs=[status_log, run_link],
                )

            # ── Tab 2: Chat ────────────────────────────────────────────────────
            with gr.Tab("💬 Chat"):

                with gr.Row():
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        gr.Markdown(
                            "**Retrieval modes**\n\n"
                            "🔵 **Hybrid** — facts, rules, definitions\n\n"
                            "🟣 **Entity** — relationships between named things\n\n"
                            "🟢 **Community** — broad themes and programs"
                        )
                        clear_btn = gr.Button("🗑 Clear")
                        retrieval_panel = gr.HTML()

                    with gr.Column(scale=4):
                        query_input = gr.Textbox(
                            placeholder="Ask anything about Everstorm Outfitters...",
                            label="Question",
                            submit_btn=True,
                        )
                        chatbot = gr.Chatbot(
                            label="Everstorm GraphRAG",
                            height=480,
                        )

                panel_state = gr.State("")

                query_input.submit(
                    fn=chat,
                    inputs=[query_input, chatbot],
                    outputs=[chatbot, panel_state],
                ).then(
                    fn=lambda p: p,
                    inputs=[panel_state],
                    outputs=[retrieval_panel],
                ).then(
                    fn=lambda: "",
                    outputs=[query_input],
                )

                clear_btn.click(
                    fn=lambda: ([], "", ""),
                    outputs=[chatbot, query_input, retrieval_panel],
                )

    return app


# ── Union cluster entry point ──────────────────────────────────────────────────

@serving_env.server
def _cluster_server():
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=_CSS)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if "--deploy" in sys.argv:
        try:
            app = flyte.serve(serving_env)
            print(f"App URL: {app.url}")
        except Exception:
            print("App deployed — check console for status:")
            print("https://tryv2.hosted.unionai.cloud/v2/domain/development/project/dellenbaugh/apps/everstorm-graphrag-chatbot")
    else:
        build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=_CSS)
