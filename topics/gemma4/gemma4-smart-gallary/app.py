"""
app.py — Gradio UI for Gemma 4 Smart Gallery.

Wires Gradio events to Flyte workflows. No API calls or business logic here.

Usage:
    Terminal 1: flyte start tui
    Terminal 2: python app.py
"""

from pathlib import Path

import gradio as gr

import ui_components as ui
import workflows

# ── Load CSS ──────────────────────────────────────────────────────────────────

_CSS = (Path(__file__).parent / "styles.css").read_text()


# ── Event handlers ────────────────────────────────────────────────────────────

def on_generate(folder_path: str) -> tuple[str, str]:
    if not folder_path or not Path(folder_path).is_dir():
        return ui.status_message("Please enter a valid folder path.", "error"), ""

    yield ui.loading_card(f"Scanning folder..."), ""

    try:
        results = workflows.run_describe_workflow(folder_path)
    except Exception as e:
        yield ui.status_message(f"Error: {e}", "error"), ""
        return

    if not results:
        yield ui.status_message("No supported images found in folder.", "warn"), ""
        return

    cards = [ui.image_card(r["path"], r["description"]) for r in results]
    grid  = ui.results_grid(cards, label="images described")
    yield ui.status_message(f"Done — {len(results)} images processed.", "success"), grid


def on_search(folder_path: str, query: str) -> tuple[str, str]:
    if not folder_path or not Path(folder_path).is_dir():
        return ui.status_message("Please enter a valid folder path.", "error"), ""

    if not query.strip():
        return ui.status_message("Please enter a search query.", "warn"), ""

    yield ui.loading_card(f'Searching for "{query}"...'), ""

    try:
        matched_paths = workflows.run_search_workflow(folder_path, query.strip())
    except Exception as e:
        yield ui.status_message(f"Error: {e}", "error"), ""
        return

    if not matched_paths:
        yield ui.status_message(f'No images matched "{query}".', "warn"), ""
        return

    cards = [ui.image_card(path, f'Matched: "{query}"') for path in matched_paths]
    grid  = ui.results_grid(cards, label=f'images matched "{query}"')
    yield ui.status_message(f'{len(matched_paths)} match(es) found.', "success"), grid


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(css=_CSS, title="Gemma 4 Smart Gallery") as demo:
    gr.HTML(ui.app_header())

    with gr.Row():
        folder_input = gr.Textbox(
            label="Image Folder Path",
            placeholder="/path/to/your/images",
            scale=4,
        )

    with gr.Row():
        generate_btn = gr.Button("Generate Descriptions", variant="primary")

    with gr.Row():
        search_input = gr.Textbox(
            label="Search",
            placeholder='e.g. ocean, dog, sunset',
            scale=3,
        )
        search_btn = gr.Button("Search", variant="secondary", scale=1)

    status_out = gr.HTML(ui.empty_state())
    results_out = gr.HTML()

    generate_btn.click(
        fn=on_generate,
        inputs=[folder_input],
        outputs=[status_out, results_out],
    )

    search_btn.click(
        fn=on_search,
        inputs=[folder_input, search_input],
        outputs=[status_out, results_out],
    )

    search_input.submit(
        fn=on_search,
        inputs=[folder_input, search_input],
        outputs=[status_out, results_out],
    )

if __name__ == "__main__":
    demo.launch()
