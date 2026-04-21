"""
ui_components.py — Reusable HTML builders for Gemma 4 Smart Gallery.

All functions return HTML strings. No Gradio, no API, no DB imports here.
CSS classes reference styles.css — no inline styles.
"""

import base64
import html
from pathlib import Path


def _encode_image(image_path: str) -> str:
    """Base64-encode an image for embedding in HTML src."""
    suffix    = Path(image_path).suffix.lower()
    mime_type = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png",
                 "webp": "webp", "gif": "gif"}.get(suffix.lstrip("."), "jpeg")
    data      = Path(image_path).read_bytes()
    b64       = base64.b64encode(data).decode()
    return f"data:image/{mime_type};base64,{b64}"


def app_header() -> str:
    return (
        '<div class="app-header">'
        '<span class="app-title">Gemma 4 Smart Gallery</span>'
        '<span class="app-tagline">Vision-powered photo search · Gemma 4 · Vertex AI</span>'
        '</div>'
    )


def image_card(image_path: str, description: str) -> str:
    filename = html.escape(Path(image_path).name)
    desc     = html.escape(description)
    src      = _encode_image(image_path)
    return (
        '<div class="image-card">'
        f'<img class="image-card-thumb" src="{src}" alt="{filename}" />'
        '<div class="image-card-body">'
        f'<div class="image-card-filename">{filename}</div>'
        f'<div class="image-card-description">{desc}</div>'
        '</div>'
        '</div>'
    )


def results_grid(cards: list[str], label: str = "results") -> str:
    count = len(cards)
    if count == 0:
        return '<div class="empty-state">No images found.</div>'

    header = (
        '<div class="results-header">'
        f'<strong>{count}</strong> {label}'
        '</div>'
    )
    grid = (
        '<div class="results-grid">'
        + "".join(cards)
        + '</div>'
    )
    return header + grid


def status_message(text: str, kind: str = "") -> str:
    modifier = f" status-{kind}" if kind else ""
    return f'<div class="status-message{modifier}">{html.escape(text)}</div>'


def loading_card(message: str = "Processing...") -> str:
    return (
        '<div class="loading-card">'
        '<div class="loading-icon">&#9651;</div>'
        f'<div>{html.escape(message)}</div>'
        '</div>'
    )


def empty_state() -> str:
    return '<div class="empty-state">Select a folder and click Generate Descriptions or Search.</div>'
