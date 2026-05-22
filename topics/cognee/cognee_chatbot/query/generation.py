"""
query/generation.py

Claude synthesis call: takes Cognee search results and generates a grounded answer.
Isolated here so the prompt and model are easy to tune independently of retrieval.
"""

from config import CLAUDE_MODEL


async def generate_answer(question: str, cognee_results: list, api_key: str) -> str:
    from anthropic import Anthropic

    client = Anthropic(api_key=api_key)

    def _extract(r) -> str:
        # With ENABLE_BACKEND_ACCESS_CONTROL=false, CHUNKS search returns payload
        # dicts directly: {"text": "...", ...}. The "text" key is the chunk text.
        # We also try .search_result and common attrs as fallbacks.
        sr = getattr(r, "search_result", r)
        if isinstance(sr, str):
            return sr
        if isinstance(sr, dict):
            for key in ("text", "content", "chunk_text", "description", "summary"):
                v = sr.get(key)
                if v and isinstance(v, str):
                    return v
        for attr in ("text", "content", "chunk_text", "description", "summary"):
            v = getattr(sr, attr, None)
            if v and isinstance(v, str):
                return v
        return str(sr)

    if not cognee_results:
        context = "No relevant information found in the knowledge graph."
    else:
        texts = [_extract(r) for r in cognee_results[:10]]
        context = "\n\n---\n\n".join(t for t in texts if t)

    system = (
        "You are a helpful customer support assistant for Everstorm Outfitters, "
        "an outdoor gear company. Answer the user's question using only the "
        "provided context. If the context doesn't fully address the question, "
        "share what you know and note the gap."
    )

    prompt = f"Context from knowledge graph:\n{context}\n\nQuestion: {question}"

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )

    return response.content[0].text
