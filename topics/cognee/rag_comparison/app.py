"""
app.py — Gradio comparison app.

Tab 1 — RAG Comparison:
  One question fires all three RAG backends (Vector, Graph, Cognee) in parallel.
  Each panel shows retrieved context and the generated answer. Claude evaluates
  all three and names a winner.

Tab 2 — Model Comparison:
  User picks a RAG backend via dropdown. Retrieval runs once; both Claude and
  Gemma 4 generate answers from the identical context. Claude evaluates both
  answers as a pure generation quality comparison.
"""

import asyncio
from pathlib import Path

import gradio as gr

from backends.cognee_backend import query as cognee_query
from backends.cognee_backend import retrieve as cognee_retrieve
from backends.graph import query as graph_query
from backends.graph import retrieve as graph_retrieve
from backends.shared.claude import generate_answer, get_client
from backends.shared.gemma import generate_answer_gemma
from backends.vector import query as vector_query
from backends.vector import retrieve as vector_retrieve
from config import CLAUDE_MODEL

# ── RAG Comparison summary (Tab 1) ───────────────────────────────────────────

_RAG_EVAL_SYSTEM = (
    "You are an expert evaluator of AI retrieval systems. "
    "You will receive the same question answered by three different RAG backends. "
    "Each backend uses a different strategy to retrieve context before generating its answer. "
    "Your job is to evaluate the quality of each retrieval and the resulting answer, "
    "then give a clear verdict on which approach worked best for this specific question."
)


async def _rag_comparison_summary(
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
        f"Then add a second smaller table titled **Score Breakdown** with these columns: "
        f"System | Retrieval (1pt) | Accuracy (1pt) | Completeness (1pt) | Clarity (1pt) | No Hallucinations (1pt) | Total. "
        f"Score each criterion 0–1 (decimals allowed). Same row order as the verdict table above.\n\n"
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
        system=_RAG_EVAL_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Model Comparison summary (Tab 2) ─────────────────────────────────────────

_MODEL_EVAL_SYSTEM = (
    "You are an expert evaluator of large language model outputs. "
    "You will receive the same question answered by two different AI models, "
    "both given identical retrieved context. "
    "Evaluate generation quality only — retrieval is identical for both."
)


async def _model_comparison_summary(
    question: str,
    retrieved: str,
    claude_ans: str,
    gemma_ans: str,
) -> str:
    prompt = (
        f"Question: {question}\n\n"
        f"{'─' * 60}\n"
        f"Retrieved context (identical for both models):\n{retrieved}\n\n"
        f"{'─' * 60}\n"
        f"CLAUDE ({CLAUDE_MODEL})\n\n"
        f"Answer:\n{claude_ans}\n\n"
        f"{'─' * 60}\n"
        f"GEMMA 4\n\n"
        f"Answer:\n{gemma_ans}\n\n"
        f"{'─' * 60}\n\n"
        f"Structure your response exactly as follows:\n\n"
        f"## Verdict\n"
        f"A markdown table with exactly these columns: Model | Strengths | Weaknesses | Score (/5). "
        f"One row per model (Claude, Gemma 4). "
        f"Sort rows by score descending — winner on top. "
        f"Prefix the winning model name with 🏆 and the runner-up with 🥈. "
        f"After the table, one bold sentence naming the winner and the single biggest reason it won.\n\n"
        f"Then add a second smaller table titled **Score Breakdown** with these columns: "
        f"Model | Accuracy (1pt) | Completeness (1pt) | Clarity (1pt) | Context Usage (1pt) | No Hallucinations (1pt) | Total. "
        f"Score each criterion 0–1 (decimals allowed). Same row order as the verdict table above.\n\n"
        f"## Analysis\n"
        f"For each model evaluate:\n"
        f"1. Accuracy — did it correctly use the provided context?\n"
        f"2. Completeness — did it cover all relevant information from the context?\n"
        f"3. Clarity and structure of the response"
    )

    response = await get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=_MODEL_EVAL_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ── Safe wrappers ─────────────────────────────────────────────────────────────

async def _safe_query(fn, question: str) -> tuple[str, str]:
    try:
        return await fn(question)
    except Exception as exc:
        msg = f"⚠ Backend error: {exc}"
        return msg, msg


async def _safe_retrieve(fn, question: str) -> tuple[str, str]:
    try:
        return await fn(question)
    except Exception as exc:
        msg = f"⚠ Retrieval error: {exc}"
        return "", msg


async def _safe_generate(fn, question: str, context: str) -> str:
    try:
        return await fn(question, context)
    except Exception as exc:
        return f"⚠ Generation error: {exc}"


# ── Tab 1 handler ─────────────────────────────────────────────────────────────

async def run_comparison(question: str):
    if not question.strip():
        empty = ("", "")
        return *empty, *empty, *empty, "Please enter a question."

    (v_ctx, v_ans), (g_ctx, g_ans), (c_ctx, c_ans) = await asyncio.gather(
        _safe_query(vector_query,  question),
        _safe_query(graph_query,   question),
        _safe_query(cognee_query,  question),
    )

    summary = await _rag_comparison_summary(
        question,
        v_ctx, v_ans,
        g_ctx, g_ans,
        c_ctx, c_ans,
    )

    return v_ctx, v_ans, g_ctx, g_ans, c_ctx, c_ans, summary


# ── Tab 2 handler ─────────────────────────────────────────────────────────────

_RETRIEVERS = {
    "Vector RAG": vector_retrieve,
    "Graph RAG":  graph_retrieve,
    "Cognee":     cognee_retrieve,
}


async def run_model_comparison(question: str, backend: str):
    if not question.strip():
        return "Please enter a question.", "", "", ""

    retrieve_fn = _RETRIEVERS.get(backend, vector_retrieve)
    context, retrieved = await _safe_retrieve(retrieve_fn, question)

    if not context:
        msg = f"No context retrieved from {backend}."
        return retrieved, msg, msg, ""

    claude_ans, gemma_ans = await asyncio.gather(
        _safe_generate(generate_answer,       question, context),
        _safe_generate(generate_answer_gemma, question, context),
    )

    mc_summary = await _model_comparison_summary(question, retrieved, claude_ans, gemma_ans)
    return retrieved, claude_ans, gemma_ans, mc_summary


# ── Gradio UI ─────────────────────────────────────────────────────────────────

_CSS = (Path(__file__).parent / "static" / "app.css").read_text()

with gr.Blocks(
    title="RAG Comparison",
    css=_CSS,
) as demo:

    gr.Markdown(
        "# RAG Comparison\n"
        "**Vector · Graph · Cognee · Claude vs Gemma 4** — same question, different strategies and models.\n\n"
        "Ask anything about Everstorm Outfitters policies."
    )

    question_box = gr.Textbox(
        label="Question",
        placeholder="e.g. What are the benefits of the Summit loyalty tier?",
    )

    with gr.Tabs():

        # ── Tab 1: RAG Comparison ─────────────────────────────────────────────
        with gr.Tab("RAG Comparison"):

            gr.Markdown(
                "*Same question, three retrieval strategies — see how Vector, Graph, and Cognee "
                "each retrieve differently, then Claude names a winner.*"
            )

            rag_ask_btn = gr.Button("Compare RAGs", variant="primary")

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
            rag_summary = gr.Markdown(label="Evaluation", container=True, elem_classes="summary-panel")

            rag_outputs = [vector_ctx, vector_ans, graph_ctx, graph_ans, cognee_ctx, cognee_ans, rag_summary]
            rag_ask_btn.click(fn=run_comparison, inputs=question_box, outputs=rag_outputs)
            question_box.submit(fn=run_comparison, inputs=question_box, outputs=rag_outputs)

        # ── Tab 2: Model Comparison ───────────────────────────────────────────
        with gr.Tab("Model Comparison"):

            gr.Markdown(
                "*Same question, same retrieved context — see how Claude and Gemma 4 "
                "generate differently from identical information.*"
            )

            with gr.Row():
                backend_dropdown = gr.Dropdown(
                    choices=["Vector RAG", "Graph RAG", "Cognee"],
                    value="Vector RAG",
                    label="RAG Backend",
                    scale=2,
                )
                model_ask_btn = gr.Button("Compare Models", variant="primary", scale=1)

            with gr.Accordion("Retrieved context", open=False):
                mc_ctx = gr.Textbox(lines=6, interactive=False, show_label=False)

            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"### Claude")
                    gr.Markdown(
                        "*Claude Sonnet 4.6 via Anthropic API*",
                        elem_classes="backend-subtitle",
                    )
                    claude_ans = gr.Markdown(label="Answer", container=True, elem_classes="model-panel")

                with gr.Column():
                    gr.Markdown("### Gemma 4")
                    gr.Markdown(
                        "*Gemma 4 26B via Vertex AI MaaS*",
                        elem_classes="backend-subtitle",
                    )
                    gemma_ans = gr.Markdown(label="Answer", container=True, elem_classes="model-panel")

            gr.Markdown("---")
            gr.Markdown("### Model Comparison Summary")
            gr.Markdown("*Claude evaluates both answers as a pure generation quality comparison.*")
            mc_summary = gr.Markdown(label="Evaluation", container=True, elem_classes="summary-panel")

            mc_outputs = [mc_ctx, claude_ans, gemma_ans, mc_summary]
            model_ask_btn.click(fn=run_model_comparison, inputs=[question_box, backend_dropdown], outputs=mc_outputs)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
