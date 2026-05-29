"""
app.py — Gradio comparison app.

One question fires all three RAG backends in parallel. Each panel shows
what the backend retrieved and the answer it generated. A fourth panel
has Claude evaluate all three and name a winner.
"""

import asyncio
from pathlib import Path

import gradio as gr

from backends.cognee_backend import query as cognee_query
from backends.graph import query as graph_query
from backends.shared.claude import get_client
from backends.vector import query as vector_query
from config import CLAUDE_MODEL

# ── Comparison summary ────────────────────────────────────────────────────────

_COMPARISON_SYSTEM = (
    "You are an expert evaluator of AI retrieval systems. "
    "You will receive the same question answered by three different RAG backends. "
    "Each backend uses a different strategy to retrieve context before generating its answer. "
    "Your job is to evaluate the quality of each retrieval and the resulting answer, "
    "then give a clear verdict on which approach worked best for this specific question."
)


async def _comparison_summary(
    question: str,
    v_ctx: str, v_ans: str,
    g_ctx: str, g_ans: str,
    c_ctx: str, c_ans: str,
) -> str:
    prompt = (
        f"Question: {question}\n\n"
        f"{'─' * 60}\n"
        f"VECTOR RAG\n"
        f"Strategy: Cosine similarity search over document chunks\n\n"
        f"Retrieved:\n{v_ctx}\n\n"
        f"Answer:\n{v_ans}\n\n"
        f"{'─' * 60}\n"
        f"GRAPH RAG\n"
        f"Strategy: Knowledge graph traversal with intelligent routing (hybrid / entity / community)\n\n"
        f"Retrieved:\n{g_ctx}\n\n"
        f"Answer:\n{g_ans}\n\n"
        f"{'─' * 60}\n"
        f"COGNEE\n"
        f"Strategy: Automated hybrid retrieval (vector index + knowledge graph via Cognee)\n\n"
        f"Retrieved:\n{c_ctx}\n\n"
        f"Answer:\n{c_ans}\n\n"
        f"{'─' * 60}\n\n"
        f"Structure your response exactly as follows:\n\n"
        f"## Verdict\n"
        f"A markdown table with exactly these columns: System | Performed Well | Weakness | Score (/5). "
        f"One row per system (Vector RAG, Graph RAG, Cognee). "
        f"Sort rows by score descending — winner on top. "
        f"Prefix the winning system name with 🏆 and the runner-up with 🥈. "
        f"After the table, one bold sentence naming the winner and the single biggest reason it won.\n\n"
        f"## Recommended Strategy\n"
        f"One sentence: which retrieval strategy you would recommend for questions like this one.\n\n"
        f"## System Breakdown\n"
        f"For each system evaluate:\n"
        f"1. Retrieval quality — how relevant and complete was the context it found?\n"
        f"2. Answer quality — how accurate, complete, and well-grounded is the response?\n"
        f"3. What did it miss or get wrong?"
    )

    response = await get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1500,
        system=_COMPARISON_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Backend runner with graceful error handling ───────────────────────────────

async def _safe_query(fn, question: str) -> tuple[str, str]:
    try:
        return await fn(question)
    except Exception as exc:
        msg = f"⚠ Backend error: {exc}"
        return msg, msg


# ── Main handler ──────────────────────────────────────────────────────────────

async def run_comparison(question: str):
    if not question.strip():
        empty = ("", "")
        return *empty, *empty, *empty, "Please enter a question."

    (v_ctx, v_ans), (g_ctx, g_ans), (c_ctx, c_ans) = await asyncio.gather(
        _safe_query(vector_query,  question),
        _safe_query(graph_query,   question),
        _safe_query(cognee_query,  question),
    )

    summary = await _comparison_summary(
        question,
        v_ctx, v_ans,
        g_ctx, g_ans,
        c_ctx, c_ans,
    )

    return v_ctx, v_ans, g_ctx, g_ans, c_ctx, c_ans, summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CSS = (Path(__file__).parent / "static" / "app.css").read_text()

with gr.Blocks(
    title="RAG Comparison",
    css=_CSS,
) as demo:

    gr.Markdown(
        "# RAG Comparison\n"
        "**Vector · Graph · Cognee** — same question, three retrieval strategies, one verdict.\n\n"
        "Ask anything about Everstorm Outfitters policies and see how each approach retrieves "
        "and reasons differently."
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Question",
            placeholder="e.g. What are the benefits of the Gold loyalty tier?",
            scale=5,
        )
        ask_btn = gr.Button("Ask", variant="primary", scale=1, min_width=80)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Vector RAG")
            gr.Markdown(
                "*Semantic similarity search over document chunks*",
                elem_classes="backend-subtitle",
            )
            with gr.Accordion("Retrieved context", open=False):
                vector_ctx = gr.Textbox(lines=8, interactive=False, show_label=False)
            vector_ans = gr.Markdown(label="Answer", container=True, elem_classes="answer-panel")

        with gr.Column():
            gr.Markdown("### Graph RAG")
            gr.Markdown(
                "*Knowledge graph traversal with intelligent routing*",
                elem_classes="backend-subtitle",
            )
            with gr.Accordion("Retrieved context", open=False):
                graph_ctx = gr.Textbox(lines=8, interactive=False, show_label=False)
            graph_ans = gr.Markdown(label="Answer", container=True, elem_classes="answer-panel")

        with gr.Column():
            gr.Markdown("### Cognee")
            gr.Markdown(
                "*Automated hybrid retrieval — vector index + knowledge graph*",
                elem_classes="backend-subtitle",
            )
            with gr.Accordion("Retrieved context", open=False):
                cognee_ctx = gr.Textbox(lines=8, interactive=False, show_label=False)
            cognee_ans = gr.Markdown(label="Answer", container=True, elem_classes="answer-panel")

    gr.Markdown("---")
    gr.Markdown("### Comparison Summary")
    gr.Markdown("*Claude evaluates all three answers and names a winner.*")
    summary_box = gr.Markdown(label="Evaluation", container=True, elem_classes="summary-panel")

    outputs = [vector_ctx, vector_ans, graph_ctx, graph_ans, cognee_ctx, cognee_ans, summary_box]

    ask_btn.click(fn=run_comparison, inputs=question_box, outputs=outputs)
    question_box.submit(fn=run_comparison, inputs=question_box, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
