"""
graph/ — GraphRAG backend using Neo4j AuraDB Free.

Query flow:
  1. route()     — Claude classifies question into hybrid / entity / community
  2. retrieve()  — fetch context from Neo4j using the selected mode
  3. generate()  — format context and call Claude for the final answer

Retrieval modes:
  hybrid     — vector search over Chunk nodes + MENTIONS → entity expansion
  entity     — Claude extracts named entities + Neo4j neighborhood traversal
  community  — embedding similarity over Community summaries + member lookup

Public interface: query(question) -> tuple[retrieved_summary, answer]
"""

from backends.graph.generation import generate
from backends.graph.retrieval import community_retrieve, entity_retrieve, hybrid_retrieve
from backends.graph.routing import route

_RETRIEVERS = {
    "hybrid":    hybrid_retrieve,
    "entity":    entity_retrieve,
    "community": community_retrieve,
}


def _summarize(context: dict, mode: str, reasoning: str) -> str:
    lines = [f"Mode: {mode}"]
    if reasoning:
        lines.append(f"Routing reason: {reasoning}")

    if mode == "hybrid":
        chunks = context.get("chunks", [])
        entities = context.get("entities", [])
        lines.append(f"\nChunks retrieved ({len(chunks)}):")
        for c in chunks:
            lines.append(f"  • {c['source_doc']}  (score: {round(float(c['score']), 4)})")
        lines.append(f"\nEntities expanded ({len(entities)}):")
        for e in entities:
            lines.append(f"  • {e['name']} ({e['type']})")

    elif mode == "entity":
        entities = context.get("entities", [])
        lines.append(f"\nEntities found ({len(entities)}):")
        for e in entities:
            neighbors = [n for n in e.get("neighbors", []) if n.get("name")]
            lines.append(f"  • {e['name']} ({e['type']}) — {len(neighbors)} connection(s)")

    elif mode == "community":
        cid = context.get("community_id")
        members = context.get("member_entities", [])
        lines.append(f"\nCommunity: {cid}")
        lines.append(f"Member concepts ({len(members)}): {', '.join(members[:10])}")
        if len(members) > 10:
            lines.append(f"  ... and {len(members) - 10} more")

    return "\n".join(lines)


async def retrieve(question: str) -> tuple[str, str]:
    """Return (context_str, summary_str) without generating an answer."""
    from backends.graph.generation import _FORMATTERS, _format_hybrid
    mode, reasoning = await route(question)
    context_dict = await _RETRIEVERS.get(mode, hybrid_retrieve)(question)
    context_str = _FORMATTERS.get(mode, _format_hybrid)(context_dict)
    summary = _summarize(context_dict, mode, reasoning)
    return context_str, summary


async def query(question: str) -> tuple[str, str]:
    mode, reasoning = await route(question)
    retriever = _RETRIEVERS.get(mode, hybrid_retrieve)
    context = await retriever(question)
    answer = await generate(question, context)
    retrieved = _summarize(context, mode, reasoning)
    return retrieved, answer
