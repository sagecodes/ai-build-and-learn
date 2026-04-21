"""
gemma_client.py — Vertex AI connection and raw Gemma 4 API calls.

Uses the modern google-genai SDK with Vertex AI backend.
Owns all SDK imports and auth. No business logic here.
Swap this file to change the model provider without touching vision_service.py.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.types import Part

load_dotenv()

_GCP_PROJECT = os.environ["GCP_PROJECT"]
_GCP_REGION  = os.getenv("GCP_REGION", "global")
_GEMMA_MODEL = os.getenv("GEMMA_MODEL", "google/gemma-4-26b-a4b-it-maas")

_client = genai.Client(
    vertexai=True,
    project=_GCP_PROJECT,
    location=_GCP_REGION,
)


def ask_about_image(image_path: str, prompt: str) -> str:
    """Send an image + text prompt to Gemma 4 and return the text response."""
    image_bytes = Path(image_path).read_bytes()
    image_part  = Part.from_bytes(data=image_bytes, mime_type=_mime_type(image_path))

    response = _client.models.generate_content(
        model=_GEMMA_MODEL,
        contents=[image_part, prompt],
    )
    return response.text.strip()


def _mime_type(image_path: str) -> str:
    suffix = Path(image_path).suffix.lower()
    return {
        ".jpg":  "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png":  "image/png",
        ".webp": "image/webp",
        ".gif":  "image/gif",
    }.get(suffix, "image/jpeg")
