"""
gemma.py — Gemma 4 generation via Vertex AI MaaS.

Uses the google-genai SDK with Vertex AI backend. Auth is handled by
Application Default Credentials — on the GCP VM the instance metadata
server provides credentials automatically, no key file needed.
"""

import asyncio

from google import genai
from config import GCP_PROJECT, GCP_REGION, GEMMA_MODEL

_SYSTEM = (
    "You are a helpful customer support assistant for Everstorm Outfitters, "
    "an outdoor gear company. Answer the user's question using only the "
    "provided context. If the context doesn't fully address the question, "
    "share what you know and note the gap."
)

_client: genai.Client | None = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(vertexai=True, project=GCP_PROJECT, location=GCP_REGION)
    return _client


async def generate_answer_gemma(question: str, context: str) -> str:
    prompt = f"{_SYSTEM}\n\nContext:\n{context}\n\nQuestion: {question}"
    response = await asyncio.to_thread(
        get_client().models.generate_content,
        model=GEMMA_MODEL,
        contents=prompt,
    )
    return response.text
