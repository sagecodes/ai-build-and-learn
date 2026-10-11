from pathlib import Path

import gradio as gr

from app.handlers import handle_batch_eval, handle_single_eval
from eval.metrics import METRIC_NAMES

_BACKEND_NAMES = ["Vector RAG", "Graph RAG", "Cognee"]
_BATCH_COLS = ["backend"] + METRIC_NAMES


def build_demo() -> gr.Blocks:
    css_path = Path(__file__).parents[1] / "static" / "app.css"
    css = css_path.read_text() if css_path.exists() else ""

    with gr.Blocks(css=css, theme=gr.themes.Soft(), title="Ragas RAG Evaluation") as demo:
        gr.Markdown("# Ragas RAG Evaluation")
        gr.Markdown(
            "Automated evaluation of **Vector RAG**, **Graph RAG**, and **Cognee** backends "
            "using Ragas metrics: faithfulness · answer relevancy · "
            "context precision · context recall."
        )

        with gr.Tabs():
            # ── Batch Evaluation ────────────────────────────────────────────────
            with gr.TabItem("Batch Evaluation"):
                gr.Markdown(
                    "Runs all 20 testset questions against all 3 backends and returns "
                    "mean scores. Takes ~5–10 minutes."
                )
                batch_btn    = gr.Button("Run Batch Evaluation", variant="primary")
                batch_status = gr.Markdown(visible=False)
                batch_df     = gr.Dataframe(
                    headers=_BATCH_COLS,
                    label="Mean Scores by Backend",
                    interactive=False,
                )
                batch_err = gr.Textbox(label="Error", visible=False, interactive=False)

            # ── Single Question ─────────────────────────────────────────────────
            with gr.TabItem("Single Question"):
                gr.Markdown(
                    "Evaluate one question across all 3 backends in real time."
                )
                with gr.Row():
                    q_input = gr.Textbox(
                        label="Question",
                        placeholder="What is the return policy for damaged gear?",
                        scale=2,
                    )
                    gt_input = gr.Textbox(
                        label="Ground Truth (optional — needed for context_recall)",
                        placeholder="Leave blank to skip context_recall scoring.",
                        scale=1,
                    )
                single_btn = gr.Button("Evaluate", variant="primary")

                # Build per-backend output components and track them for wiring
                single_outputs: list[gr.components.Component] = []
                with gr.Row():
                    for name in _BACKEND_NAMES:
                        with gr.Column():
                            gr.Markdown(f"### {name}")
                            ctx    = gr.Textbox(label="Retrieved Context", lines=5, interactive=False)
                            answer = gr.Markdown(label="Answer")
                            scores = gr.Markdown(label="Ragas Scores")
                            single_outputs.extend([ctx, answer, scores])

        # ── Event wiring (after all components are defined) ─────────────────────
        batch_btn.click(
            fn=handle_batch_eval,
            inputs=[],
            outputs=[batch_btn, batch_df, batch_err],
            show_progress="full",
        )

        single_btn.click(
            fn=handle_single_eval,
            inputs=[q_input, gt_input],
            outputs=[single_btn] + single_outputs,
            show_progress="full",
        )

    return demo
