"""
app.py — Gradio UI for Gemma 4 Smart Gallery.

Wires Gradio events to Flyte workflows. No API calls or business logic here.

Usage:
    Terminal 1: flyte start tui
    Terminal 2: python app.py
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import gradio as gr

import ui_components as ui
import workflows

# ── Load CSS ──────────────────────────────────────────────────────────────────

_CSS = (Path(__file__).parent / "styles.css").read_text()


# ── Event handlers ────────────────────────────────────────────────────────────

def on_browse() -> str:
    """Open native OS folder picker and return selected path."""
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes("-topmost", True)
    folder = filedialog.askdirectory(title="Select Image Folder")
    root.destroy()
    return folder or ""


def on_generate(folder_path: str):
    if not folder_path or not Path(folder_path).is_dir():
        yield ui.status_message("Please select a valid folder.", "error"), ""
        return

    yield ui.loading_card("Scanning folder..."), ""

    cards = []
    try:
        for result in workflows.run_describe_workflow(folder_path):
            cards.append(ui.image_card(result["path"], result["description"]))
            grid = ui.results_grid(cards, label="images described")
            yield ui.status_message(f"Processing... {len(cards)} done", ""), grid
    except Exception as e:
        yield ui.status_message(f"Error: {e}", "error"), ""
        return

    if not cards:
        yield ui.status_message("No supported images found in folder.", "warn"), ""
        return

    grid = ui.results_grid(cards, label="images described")
    yield ui.status_message(f"Done — {len(cards)} images processed.", "success"), grid


def on_search(folder_path: str, query: str):
    if not folder_path or not Path(folder_path).is_dir():
        yield ui.status_message("Please select a valid folder.", "error"), ""
        return

    if not query.strip():
        yield ui.status_message("Please enter a search query.", "warn"), ""
        return

    yield ui.loading_card(f'Searching for "{query}"...'), ""

    try:
        matched_paths = []
        total         = 0
        for update in workflows.run_search_workflow(folder_path, query.strip()):
            total = update["total"]
            if not update["done"]:
                checked   = update["checked"]
                remaining = total - checked
                yield ui.status_message(
                    f'Searching... {checked}/{total} checked, {remaining} remaining', ""
                ), ""
            else:
                matched_paths = update["matches"]
    except Exception as e:
        yield ui.status_message(f"Error: {e}", "error"), ""
        return

    if not matched_paths:
        yield ui.status_message(f'Searched {total} image(s) — no matches for "{query}".', "warn"), ""
        return

    cards = [ui.image_card(path, f'Matched: "{query}"') for path in matched_paths]
    grid  = ui.results_grid(cards, label=f'images matched "{query}"')
    yield ui.status_message(f'Searched {total} image(s) — {len(matched_paths)} matched "{query}".', "success"), grid


# ── UI layout ─────────────────────────────────────────────────────────────────

with gr.Blocks(title="Gemma 4 Smart Gallery") as demo:
    gr.HTML(ui.app_header())

    with gr.Row():

        # ── Left sidebar ──────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=300):
            gr.HTML('<div class="sidebar">')

            # Photo Library section
            gr.HTML(ui.sidebar_label("Photo Library"))
            folder_input = gr.Textbox(
                label="Folder Path",
                placeholder="Click Browse to select a folder",
                lines=2,
                max_lines=2,
                interactive=True,
            )
            browse_btn = gr.Button("Browse...", variant="secondary", size="sm")

            gr.HTML('<hr class="sidebar-divider">')

            # Vision Powered Options section
            gr.HTML(ui.sidebar_label("Vision Powered Options"))

            gr.HTML(ui.action_button(
                "Generate Descriptions",
                "Gemma 4 analyzes each image and writes a natural language description. Results are cached for future use.",
                "generate-btn",
            ))
            generate_btn = gr.Button(
                "Generate Descriptions", variant="primary", elem_id="generate-btn",
                elem_classes=["hidden-trigger"],
            )

            search_input = gr.Textbox(
                label="Search Query",
                placeholder="e.g. ocean, dog, sunset",
                lines=1,
            )
            gr.HTML(ui.action_button(
                "Search",
                "Gemma 4 visually inspects each image in real time to find matches.",
                "search-btn",
            ))
            search_btn = gr.Button(
                "Search", variant="primary", elem_id="search-btn",
                elem_classes=["hidden-trigger"],
            )

            gr.HTML('</div>')

        # ── Main results area ─────────────────────────────────────────────────
        with gr.Column(scale=3):
            status_out  = gr.HTML(ui.empty_state())
            results_out = gr.HTML()

    # ── Event wiring ──────────────────────────────────────────────────────────

    browse_btn.click(
        fn=on_browse,
        inputs=[],
        outputs=[folder_input],
    )

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
    demo.launch(css=_CSS)
