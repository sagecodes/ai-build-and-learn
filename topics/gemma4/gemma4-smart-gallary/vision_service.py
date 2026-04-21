"""
vision_service.py — Business logic for Gemma 4 vision operations.

Calls gemma_client.py for all API communication.
No Flyte, no Gradio, no SQLite imports here.
"""

from gemma_client import ask_about_image

_DESCRIBE_PROMPT = (
    "Describe this image in 2-3 sentences. "
    "Focus on the main subjects, setting, colors, and mood. "
    "Be specific and descriptive."
)


def describe_image(image_path: str) -> str:
    """Return a natural language description of the image."""
    return ask_about_image(image_path, _DESCRIBE_PROMPT)


def check_image_match(image_path: str, query: str) -> bool:
    """Return True if the image visually matches the search query."""
    prompt = (
        f'Does this image contain or relate to "{query}"? '
        "Answer with only YES or NO."
    )
    response = ask_about_image(image_path, prompt)
    return response.strip().upper().startswith("YES")
