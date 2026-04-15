"""
dashboard/app.py — Gradio dashboard for AutoResearch results.

Reads experiment data from Firestore and renders:
  - Stat row: current val_bpb, total experiments, kept count, success rate
  - val_bpb progression chart (Plotly)
  - Experiment log table
  - Run summary card

Auto-refreshes every 60 seconds so it stays live during an active run.

Usage (local machine):
    export GOOGLE_APPLICATION_CREDENTIALS="path/to/autoresearch-reader-key.json"
    export GCP_PROJECT="your-gcp-project-id"
    python dashboard/app.py

Then open http://localhost:7860 in your browser.
"""

import os
import sys
from pathlib import Path

import gradio as gr

# Allow imports from parent directory (firestore_logger, metrics)
sys.path.insert(0, str(Path(__file__).parent.parent))

import firestore_logger
from dashboard.ui_components import (
    app_header,
    empty_chart,
    experiment_table,
    loading_card,
    run_summary_card,
    stat_row,
    val_bpb_chart,
)

# ── Config ────────────────────────────────────────────────────────────────────

_CSS_PATH = Path(__file__).parent / "styles.css"
_CSS = _CSS_PATH.read_text(encoding="utf-8")

_GCP_PROJECT = os.getenv("GCP_PROJECT")
_REFRESH_SECONDS = 60   # dashboard auto-refresh interval

# ── Data fetching ─────────────────────────────────────────────────────────────

def _fetch_run_options() -> list[str]:
    """Return a list of run IDs for the dropdown, newest first."""
    runs = firestore_logger.list_runs(limit=10, project_id=_GCP_PROJECT)
    if not runs:
        return ["(no runs found)"]
    return [r["id"] for r in runs]


def _fetch_data(run_id: str) -> tuple[dict | None, list[dict]]:
    """
    Fetch run metadata and all experiments for the given run_id.

    Returns (run_dict, experiments_list). Both are empty/None if not found.
    """
    if not run_id or run_id == "(no runs found)":
        return None, []
    run = firestore_logger.get_run(run_id, project_id=_GCP_PROJECT)
    experiments = firestore_logger.get_experiments(run_id, project_id=_GCP_PROJECT)
    return run, experiments


# ── Render functions ──────────────────────────────────────────────────────────

def render_dashboard(run_id: str):
    """
    Fetch data for run_id and return all dashboard outputs.

    Yields in order matching the Gradio output components:
      stats_html, chart_fig, table_html, summary_html
    """
    run, experiments = _fetch_data(run_id)

    if run is None:
        yield (
            loading_card("No run selected or run not found."),
            empty_chart(),
            loading_card("Select a run to view experiments."),
            "",
        )
        return

    yield (
        stat_row(experiments, run),
        val_bpb_chart(experiments),
        experiment_table(experiments),
        run_summary_card(experiments),
    )


def refresh_run_list():
    """Refresh the run selector dropdown options."""
    return gr.Dropdown(choices=_fetch_run_options())


# ── Layout ────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(css=_CSS, title="AutoResearch Dashboard") as demo:

        gr.HTML(app_header())

        with gr.Row():
            run_selector = gr.Dropdown(
                choices=_fetch_run_options(),
                label="Run",
                elem_id="run-selector",
                scale=2,
            )
            refresh_btn = gr.Button("Refresh", scale=1, size="sm")

        stats_html = gr.HTML(loading_card())
        chart = gr.Plot(value=empty_chart(), show_label=False)
        table_html = gr.HTML(loading_card("Loading experiment log..."))
        summary_html = gr.HTML()

        # Wire up interactions
        run_selector.change(
            fn=render_dashboard,
            inputs=[run_selector],
            outputs=[stats_html, chart, table_html, summary_html],
        )
        refresh_btn.click(
            fn=refresh_run_list,
            outputs=[run_selector],
        )

        # Auto-load latest run on startup
        demo.load(
            fn=lambda: _fetch_run_options()[0] if _fetch_run_options() else "(no runs found)",
            outputs=[run_selector],
        )
        demo.load(
            fn=render_dashboard,
            inputs=[run_selector],
            outputs=[stats_html, chart, table_html, summary_html],
        )

        # Auto-refresh every 60 seconds
        gr.Timer(value=_REFRESH_SECONDS).tick(
            fn=render_dashboard,
            inputs=[run_selector],
            outputs=[stats_html, chart, table_html, summary_html],
        )

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)
