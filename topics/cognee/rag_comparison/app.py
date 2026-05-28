import asyncio
import gradio as gr

from backends.vector import query as vector_query
from backends.graph import query as graph_query        # graph/ package
from backends.cognee_backend import query as cognee_query


async def run_comparison(question: str):
    # Phase 4: implement parallel query + comparison summary
    raise NotImplementedError


with gr.Blocks(title="RAG Comparison") as demo:
    gr.Markdown("## RAG Comparison\nVector · Graph · Cognee — same question, three retrieval strategies.")

    question = gr.Textbox(label="Question", placeholder="Ask something about Everstorm Outfitters...")
    ask_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Vector RAG")
            vector_context = gr.Accordion("Retrieved context", open=False)
            vector_answer  = gr.Textbox(label="Answer", lines=6, interactive=False)

        with gr.Column():
            gr.Markdown("### Graph RAG")
            graph_context = gr.Accordion("Retrieved context", open=False)
            graph_answer  = gr.Textbox(label="Answer", lines=6, interactive=False)

        with gr.Column():
            gr.Markdown("### Cognee")
            cognee_context = gr.Accordion("Retrieved context", open=False)
            cognee_answer  = gr.Textbox(label="Answer", lines=6, interactive=False)

    gr.Markdown("### Comparison Summary")
    summary = gr.Textbox(label="Claude's evaluation", lines=8, interactive=False)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
