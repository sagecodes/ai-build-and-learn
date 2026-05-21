"""
app.py

Gradio UI + Union AppEnvironment serving entry point for the Cognee chatbot.

Three tabs:
  Ingest  — upload PDFs (or use data/), run ingest_pipeline on Union
  Chat    — ask questions answered via Cognee search + Claude
  Graph   — generate interactive knowledge graph visualization

Deploy to Union:
  python app.py --deploy

Run locally:
  python app.py
"""

import base64
import json
import sys
from pathlib import Path

import flyte
import flyte.app
import gradio as gr

from config import UNION_ORG, UNION_PROJECT, UNION_DOMAIN

_DATA_DIR = Path(__file__).parent / "data"

_CSS = """
.tab-sidebar {
    border-right: 1px solid rgba(128, 128, 128, 0.2);
    padding-right: 16px;
}
.status-log textarea {
    font-family: monospace;
    font-size: 13px;
}
.run-link a {
    font-size: 13px;
    color: #6366f1;
}
.result-count {
    font-size: 12px;
    color: #6b7280;
}
"""

# ── Union App serving environment ─────────────────────────────────────────

serving_env = flyte.app.AppEnvironment(
    name="everstorm-cognee-chatbot",
    image="docker.io/johndellenbaugh/cognee-chatbot-app:latest",
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="DB_HOST",           as_env_var="DB_HOST"),
        flyte.Secret(key="DB_PORT",           as_env_var="DB_PORT"),
        flyte.Secret(key="DB_NAME",           as_env_var="DB_NAME"),
        flyte.Secret(key="DB_USERNAME",       as_env_var="DB_USERNAME"),
        flyte.Secret(key="DB_PASSWORD",       as_env_var="DB_PASSWORD"),
    ],
    env_vars={"FLYTE_BACKEND": "cluster"},
    port=7860,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)


# ── Helpers ───────────────────────────────────────────────────────────────

def _run_link_html(run) -> str:
    base = f"https://{UNION_ORG}.hosted.unionai.cloud"
    try:
        url = f"{base}/v2/domain/{UNION_DOMAIN}/project/{UNION_PROJECT}/executions/{run.id}"
    except Exception:
        url = base
    return f'<a href="{url}" target="_blank">View run on Union →</a>'


# ── Ingest handler ────────────────────────────────────────────────────────

def run_ingest(uploaded_files):
    log_lines: list[str] = []

    def emit(msg: str):
        log_lines.append(msg)
        return "\n".join(log_lines)

    filenames: list[str] = []
    pdf_bytes_b64: list[str] = []

    if uploaded_files:
        yield emit(f"Encoding {len(uploaded_files)} uploaded PDF(s)..."), ""
        for file_path in uploaded_files:
            src = Path(file_path)
            filenames.append(src.name)
            pdf_bytes_b64.append(base64.b64encode(src.read_bytes()).decode())
            yield emit(f"  {src.name}"), ""
    else:
        pdf_paths = sorted(_DATA_DIR.glob("*.pdf"))
        if not pdf_paths:
            yield emit("No PDFs found — upload files or add PDFs to data/"), ""
            return
        yield emit(f"Encoding {len(pdf_paths)} PDF(s) from data/..."), ""
        for p in pdf_paths:
            filenames.append(p.name)
            pdf_bytes_b64.append(base64.b64encode(p.read_bytes()).decode())
            yield emit(f"  {p.name}"), ""

    yield emit("\nDispatching ingest_pipeline to Union..."), ""

    try:
        from workflows import ingest_pipeline
        run = flyte.run(ingest_pipeline, filenames=filenames, pdf_bytes_b64=pdf_bytes_b64)
        link = _run_link_html(run)
        yield emit("Waiting for Union to complete..."), link

        run.wait()
        summary = json.loads(run.outputs().o0)

        yield emit(
            f"\nIngest complete!\n"
            f"  Documents ingested: {summary.get('documents_ingested', '—')}"
        ), link

    except Exception as exc:
        yield emit(f"\nError: {exc}"), ""


# ── Chat handler ──────────────────────────────────────────────────────────

def chat(question: str, history: list):
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

        answer = result["answer"]
        result_count = result.get("result_count", 0)
        meta = f'\n\n<span class="result-count">Retrieved {result_count} graph result(s)</span>'

        history.append({"role": "assistant", "content": answer + meta})

    except Exception as exc:
        history.append({"role": "assistant", "content": f"Error: {exc}"})

    return history, ""


# ── Graph visualization handler ───────────────────────────────────────────

def run_visualize():
    try:
        from workflows import visualize_pipeline
        run = flyte.run(visualize_pipeline)
        run.wait()
        return run.outputs().o0
    except Exception as exc:
        return f"<p style='color:red'>Error generating visualization: {exc}</p>"


# ── UI layout ─────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Everstorm Cognee Chatbot", css=_CSS) as app:

        gr.Markdown("# Everstorm Outfitters — Cognee Chatbot")
        gr.Markdown(
            "Knowledge-graph Q&A powered by Cognee + Claude, compute running on Union."
        )

        with gr.Tabs():

            # ── Tab 1: Ingest ──────────────────────────────────────────────

            with gr.Tab("Ingest"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        file_upload = gr.File(
                            file_types=[".pdf"],
                            file_count="multiple",
                            label="Upload PDFs (optional)",
                        )
                    with gr.Column(scale=4):
                        gr.Markdown(
                            "Upload PDFs or leave empty to use `data/`. "
                            "Cognee builds the knowledge graph automatically — "
                            "`cognee.add()` + `cognee.cognify()` replace the "
                            "6-step manual pipeline from the graph RAG project."
                        )
                        ingest_btn = gr.Button("Run Ingest on Union", variant="primary")
                        run_link = gr.HTML(elem_classes=["run-link"])
                        status_log = gr.Textbox(
                            label="Status",
                            lines=12,
                            interactive=False,
                            elem_classes=["status-log"],
                        )

                ingest_btn.click(
                    fn=run_ingest,
                    inputs=[file_upload],
                    outputs=[status_log, run_link],
                )

            # ── Tab 2: Chat ────────────────────────────────────────────────

            with gr.Tab("Chat"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        gr.Markdown(
                            "Ask anything about Everstorm Outfitters policies, "
                            "programs, and products. Answers are grounded in the "
                            "Cognee knowledge graph."
                        )
                        clear_btn = gr.Button("Clear")
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="Everstorm Cognee Chatbot",
                            height=460,
                        )
                        query_input = gr.Textbox(
                            placeholder="What is the return window for sale items?",
                            label="Question",
                            submit_btn=True,
                        )

                query_input.submit(
                    fn=chat,
                    inputs=[query_input, chatbot],
                    outputs=[chatbot, query_input],
                )
                clear_btn.click(
                    fn=lambda: ([], ""),
                    outputs=[chatbot, query_input],
                )

            # ── Tab 3: Graph ───────────────────────────────────────────────

            with gr.Tab("Graph"):
                with gr.Row():
                    with gr.Column(scale=1, min_width=220, elem_classes=["tab-sidebar"]):
                        gr.Markdown(
                            "Interactive visualization of the knowledge graph "
                            "built by `cognify()`. Nodes are entities; edges "
                            "are relationships extracted from the Everstorm PDFs."
                        )
                        viz_btn = gr.Button("Generate Visualization", variant="primary")
                    with gr.Column(scale=4):
                        graph_html = gr.HTML(
                            "<p>Click the button to generate the knowledge graph.</p>"
                        )

                viz_btn.click(fn=run_visualize, outputs=[graph_html])

    return app


# ── Union cluster entry point ──────────────────────────────────────────────

@serving_env.server
def _cluster_server():
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)


# ── Local / deploy entry point ─────────────────────────────────────────────

if __name__ == "__main__":
    if "--deploy" in sys.argv:
        try:
            deployed = flyte.serve(serving_env)
            print(f"App URL: {deployed.url}")
        except Exception:
            print("App deployed — check Union console:")
            print(
                f"https://{UNION_ORG}.hosted.unionai.cloud/v2/domain/{UNION_DOMAIN}"
                f"/project/{UNION_PROJECT}/apps/everstorm-cognee-chatbot"
            )
    else:
        build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)
