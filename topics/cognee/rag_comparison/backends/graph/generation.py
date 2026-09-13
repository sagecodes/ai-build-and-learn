"""
generation.py — format retrieved context and generate an answer for each retrieval mode.

Each mode produces a different context structure, so each gets its own formatter.
The shared generate_answer() call is identical across all three — only the context string changes.
"""

from backends.shared.claude import generate_answer


def _format_hybrid(context: dict) -> str:
    chunks = context.get("chunks", [])
    entities = context.get("entities", [])

    chunk_text = "\n\n".join(
        f"[{c['source_doc']} | score: {round(float(c['score']), 4)}]\n{c['text']}"
        for c in chunks
    ) or "No matching chunks found."

    entity_text = "\n".join(
        f"• {e['name']} ({e['type']}): {e['description']}"
        for e in entities
    ) or "None"

    return (
        f"Document chunks:\n{chunk_text}\n\n"
        f"Related entities from knowledge graph:\n{entity_text}"
    )


def _format_entity(context: dict) -> str:
    entities = context.get("entities", [])

    if not entities:
        return "No matching entities found in the knowledge graph."

    blocks = []
    for e in entities:
        neighbors = [n for n in e.get("neighbors", []) if n.get("name")]
        neighbor_str = ", ".join(
            f"{n['name']} [{n['rel_type']}]" for n in neighbors
        ) or "none"
        blocks.append(
            f"Entity: {e['name']} ({e['type']})\n"
            f"Description: {e['description']}\n"
            f"Connected to: {neighbor_str}"
        )

    return "Entity context from knowledge graph:\n\n" + "\n\n".join(blocks)


def _format_community(context: dict) -> str:
    summary = context.get("summary", "")
    members = context.get("member_entities", [])
    member_str = ", ".join(members) or "none"

    if not summary:
        return "No matching community found in the knowledge graph."

    return (
        f"Community summary:\n{summary}\n\n"
        f"Member concepts: {member_str}"
    )


_FORMATTERS = {
    "hybrid":    _format_hybrid,
    "entity":    _format_entity,
    "community": _format_community,
}


async def generate(question: str, context: dict) -> str:
    mode = context.get("mode", "hybrid")
    formatter = _FORMATTERS.get(mode, _format_hybrid)
    return await generate_answer(question, formatter(context))
