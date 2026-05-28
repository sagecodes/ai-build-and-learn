"""
routing.py — classify a question into the best graph retrieval mode.

Three modes:
  hybrid     — factual questions answered by vector search over chunks + nearby entity context
  entity     — relationship questions requiring entity lookup and neighborhood traversal
  community  — broad thematic questions spanning many documents and entity clusters
"""

from backends.shared.claude import get_client
from config import CLAUDE_MODEL

_ROUTE_TOOL = {
    "name": "classify_query",
    "description": "Classify a user question into the best graph retrieval mode.",
    "input_schema": {
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["hybrid", "entity", "community"],
                "description": (
                    "hybrid     — specific facts, rules, eligibility criteria, dates, or "
                    "numbers. Best when the answer lives in a document passage.\n"
                    "entity     — questions about named things and how they connect to each "
                    "other. Best when the answer requires traversing relationships between "
                    "programs, tiers, benefits, or policies.\n"
                    "community  — broad 'what does Everstorm offer' style questions or "
                    "overviews of a topic area. Best when no single chunk or entity contains "
                    "the full answer."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "One sentence explaining why this mode was chosen.",
            },
        },
        "required": ["mode", "reasoning"],
    },
}

_SYSTEM = (
    "You are a query router for a knowledge graph built on Everstorm Outfitters policy "
    "and product documents. Given a user question, select the retrieval mode that will "
    "surface the most relevant and complete context."
)


async def route(question: str) -> tuple[str, str]:
    """
    Classify a question into a retrieval mode.
    Returns (mode, reasoning).
    """
    response = await get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        system=_SYSTEM,
        tools=[_ROUTE_TOOL],
        tool_choice={"type": "tool", "name": "classify_query"},
        messages=[{"role": "user", "content": question}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    return tool_block.input["mode"], tool_block.input.get("reasoning", "")
