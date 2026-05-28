"""
app.py — Gradio comparison app.

One question fires all three RAG backends in parallel. Each panel shows
what the backend retrieved and the answer it generated. A fourth panel
has Claude evaluate all three and name a winner.
"""

import asyncio

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
        f"Evaluate each system on:\n"
        f"1. Retrieval quality — how relevant and complete was the context it found?\n"
        f"2. Answer quality — how accurate, complete, and well-grounded is the response?\n"
        f"3. What did it miss or get wrong?\n\n"
        f"End with a clear verdict: which system performed best for this question, "
        f"and which retrieval strategy would you recommend for questions like this one?"
    )

    response = await get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
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

with gr.Blocks(
    title="RAG Comparison",
    theme=gr.themes.Soft(),
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
            vector_ans = gr.Textbox(label="Answer", lines=7, interactive=False)

        with gr.Column():
            gr.Markdown("### Graph RAG")
            gr.Markdown(
                "*Knowledge graph traversal with intelligent routing*",
                elem_classes="backend-subtitle",
            )
            with gr.Accordion("Retrieved context", open=False):
                graph_ctx = gr.Textbox(lines=8, interactive=False, show_label=False)
            graph_ans = gr.Textbox(label="Answer", lines=7, interactive=False)

        with gr.Column():
            gr.Markdown("### Cognee")
            gr.Markdown(
                "*Automated hybrid retrieval — vector index + knowledge graph*",
                elem_classes="backend-subtitle",
            )
            with gr.Accordion("Retrieved context", open=False):
                cognee_ctx = gr.Textbox(lines=8, interactive=False, show_label=False)
            cognee_ans = gr.Textbox(label="Answer", lines=7, interactive=False)

    gr.Markdown("---")
    gr.Markdown("### Comparison Summary")
    gr.Markdown("*Claude evaluates all three answers and names a winner.*")
    summary_box = gr.Textbox(
        label="Evaluation",
        lines=10,
        interactive=False,
        show_copy_button=True,
    )

    outputs = [vector_ctx, vector_ans, graph_ctx, graph_ans, cognee_ctx, cognee_ans, summary_box]

    ask_btn.click(fn=run_comparison, inputs=question_box, outputs=outputs)
    question_box.submit(fn=run_comparison, inputs=question_box, outputs=outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
