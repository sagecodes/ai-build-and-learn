"""
query/generation.py — generate task

Builds a RAG prompt from retrieved context and calls Claude for the answer.
"""

import json

from config import CLAUDE_MODEL, anthropic_client, task_env

_SYSTEM_PROMPT = (
    "You are a helpful assistant for Everstorm Outfitters. "
    "Answer the user's question using only the provided context. "
    "Be concise and accurate. If the context does not contain enough information "
    "to answer the question, say so — do not guess."
)


def _build_hybrid_prompt(question: str, context: dict) -> str:
    chunks = context.get("chunks", [])
    entities = context.get("entities", [])

    chunk_text = "\n\n".join(
        f"[{c['source_doc']}] {c['text']}" for c in chunks
    )
    entity_text = "\n".join(
        f"- {e['name']} ({e['type']}): {e['description']}" for e in entities
    ) or "None"

    return (
        f"Question: {question}\n\n"
        f"Document chunks:\n{chunk_text}\n\n"
        f"Related entities:\n{entity_text}\n\n"
        "Answer the question based on the document chunks above. "
        "Reference the source document name when citing a fact."
    )


def _build_entity_prompt(question: str, context: dict) -> str:
    entities = context.get("entities", [])

    entity_blocks = []
    for e in entities:
        neighbors = e.get("neighbors", [])
        neighbor_text = ", ".join(
            f"{n['name']} ({n['rel_type']})" for n in neighbors if n.get("name")
        ) or "none"
        entity_blocks.append(
            f"Entity: {e['name']} ({e['type']})\n"
            f"Description: {e['description']}\n"
            f"Connected to: {neighbor_text}"
        )

    entity_text = "\n\n".join(entity_blocks) or "No matching entities found."

    return (
        f"Question: {question}\n\n"
        f"Entity context from the knowledge graph:\n{entity_text}\n\n"
        "Explain how these entities relate to answer the question."
    )


def _build_community_prompt(question: str, context: dict) -> str:
    summary = context.get("summary", "")
    members = context.get("member_entities", [])
    member_text = ", ".join(members) or "none"

    return (
        f"Question: {question}\n\n"
        f"Community summary:\n{summary}\n\n"
        f"Member concepts: {member_text}\n\n"
        "Use the community summary to answer the question. "
        "Mention specific member concepts where relevant."
    )


_PROMPT_BUILDERS = {
    "hybrid": _build_hybrid_prompt,
    "entity": _build_entity_prompt,
    "community": _build_community_prompt,
}


@task_env.task
async def generate(question: str, context_json: str) -> str:
    """
    Generate a grounded answer from retrieved context.

    Args:
        question:     The user's original question.
        context_json: JSON string from the retrieval task.

    Returns:
        JSON — {answer, sources, retrieval_mode, entities_used}.
    """
    context = json.loads(context_json)
    mode = context.get("mode", "hybrid")

    builder = _PROMPT_BUILDERS.get(mode, _build_hybrid_prompt)
    prompt = builder(question, context)

    client = anthropic_client()
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text.strip()

    sources = list({c["source_doc"] for c in context.get("chunks", [])})
    entities_used = (
        [e["name"] for e in context.get("entities", [])]
        if mode in ("hybrid", "entity")
        else context.get("member_entities", [])
    )

    return json.dumps({
        "answer": answer,
        "sources": sources,
        "retrieval_mode": mode,
        "entities_used": entities_used,
    })
