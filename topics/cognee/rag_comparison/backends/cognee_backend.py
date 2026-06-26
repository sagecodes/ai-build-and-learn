"""
cognee_backend.py — Cognee GraphRAG backend using local LanceDB + SQLite storage.

Cognee automatically builds both a vector index and a knowledge graph from ingested
documents. Querying via CHUNKS search returns vector-retrieved segments backed by
the graph. Storage is entirely local — no external DB credentials required.

configure() must be called before importing cognee so environment variables are in
place when Cognee reads its configuration at import time.
"""

import os

from backends.shared.claude import generate_answer
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    COGNEE_DATA_PATH,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
)

TOP_K = 10


def _configure() -> None:
    """Set Cognee environment variables before importing cognee."""
    os.environ["LLM_PROVIDER"]         = "anthropic"
    os.environ["LLM_MODEL"]            = CLAUDE_MODEL
    os.environ["LLM_API_KEY"]          = ANTHROPIC_API_KEY
    os.environ["VECTOR_DB_PROVIDER"]   = "lancedb"
    os.environ["EMBEDDING_PROVIDER"]   = "fastembed"
    os.environ["EMBEDDING_MODEL"]      = EMBEDDING_MODEL
    os.environ["EMBEDDING_DIMENSIONS"] = str(EMBEDDING_DIMS)
    os.environ["DATA_PATH"]            = COGNEE_DATA_PATH


def _extract_text(result) -> str:
    """Extract a text string from a Cognee search result regardless of payload shape."""
    payload = getattr(result, "search_result", result)
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "content", "chunk_text", "description", "summary"):
            v = payload.get(key)
            if v and isinstance(v, str):
                return v
    for attr in ("text", "content", "chunk_text", "description", "summary"):
        v = getattr(payload, attr, None)
        if v and isinstance(v, str):
            return v
    return str(payload)


async def retrieve(question: str) -> tuple[str, str]:
    """Return (context_str, summary_str) without generating an answer."""
    _configure()

    import cognee
    from cognee.api.v1.search import SearchType

    results = await cognee.search(query_type=SearchType.CHUNKS, query_text=question)
    if not results:
        return "", "No chunks retrieved."

    texts = [t for t in (_extract_text(r) for r in results[:TOP_K]) if t]
    context = "\n\n---\n\n".join(texts)
    summary = "\n".join(f"• Chunk {i + 1}: {t[:120]}..." for i, t in enumerate(texts))
    return context, summary


async def query(question: str) -> tuple[str, str]:
    context, retrieved = await retrieve(question)
    if not context:
        return "No chunks retrieved.", "No relevant information found in Cognee's knowledge graph."
    answer = await generate_answer(question, context)
    return retrieved, answer
