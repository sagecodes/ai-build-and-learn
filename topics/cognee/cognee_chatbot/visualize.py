"""
visualize.py

Cognee graph visualization task: generates an interactive HTML knowledge graph
and returns the HTML content as a string for rendering in the Gradio UI.
"""

import os
import tempfile

from config import task_env, configure_cognee


@task_env.task
async def visualize_pipeline() -> str:
    configure_cognee()
    import cognee

    with tempfile.TemporaryDirectory() as tmpdir:
        html_path = os.path.join(tmpdir, "graph.html")
        await cognee.visualize_graph(html_path)
        with open(html_path) as f:
            return f.read()
