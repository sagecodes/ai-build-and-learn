"""
retrieval.py — three retrieval modes for the graph backend.

hybrid    — vector search over Chunk nodes + MENTIONS → RELATED entity expansion
entity    — Claude extracts named entities from question + Neo4j neighborhood traversal
community — embedding similarity over Community summaries + member entity lookup
"""

import numpy as np
from neo4j import AsyncGraphDatabase

from backends.shared.claude import get_client
from backends.shared.embeddings import embed
from config import (
    CLAUDE_MODEL,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    VECTOR_INDEX_NAME,
)

_TOP_K = 3

_ENTITY_TOOL = {
    "name": "extract_entities",
    "description": "Extract named entities mentioned in the question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Entity names mentioned in the question.",
            }
        },
        "required": ["entities"],
    },
}


def _cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.array(a), np.array(b)
    return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))


async def hybrid_retrieve(question: str) -> dict:
    """
    Vector search over Chunk nodes, then expand to nearby Entity nodes via MENTIONS.
    Returns dict with mode, chunks, and entities.
    """
    query_vec = embed(question)

    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            chunk_result = await session.run(
                """
                CALL db.index.vector.queryNodes($index, $top_k, $embedding)
                YIELD node AS chunk, score
                RETURN chunk.id        AS chunk_id,
                       chunk.source_doc AS source_doc,
                       chunk.text       AS text,
                       score
                ORDER BY score DESC
                """,
                index=VECTOR_INDEX_NAME,
                top_k=_TOP_K,
                embedding=query_vec,
            )
            chunks = await chunk_result.data()

            chunk_ids = [c["chunk_id"] for c in chunks]
            entity_result = await session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.id IN $chunk_ids
                OPTIONAL MATCH (e)-[:RELATED]->(neighbor:Entity)
                RETURN DISTINCT e.name        AS name,
                                e.type        AS type,
                                e.description AS description
                LIMIT 20
                """,
                chunk_ids=chunk_ids,
            )
            entities = await entity_result.data()

    return {"mode": "hybrid", "chunks": chunks, "entities": entities}


async def entity_retrieve(question: str) -> dict:
    """
    Extract named entities from the question via Claude tool use,
    then traverse each entity's neighborhood in Neo4j.
    Returns dict with mode and entities with neighbors.
    """
    response = await get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        tools=[_ENTITY_TOOL],
        tool_choice={"type": "tool", "name": "extract_entities"},
        messages=[{"role": "user", "content": question}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    entity_names = tool_block.input.get("entities", [])

    if not entity_names:
        return {"mode": "entity", "entities": []}

    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            results = []
            for name in entity_names:
                result = await session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (e)-[r:RELATED]->(neighbor:Entity)
                    RETURN e.name        AS name,
                           e.type        AS type,
                           e.description AS description,
                           collect({
                               name:        neighbor.name,
                               rel_type:    r.type,
                               description: r.description
                           }) AS neighbors
                    LIMIT 5
                    """,
                    name=name,
                )
                results.extend(await result.data())

    return {"mode": "entity", "entities": results}


async def community_retrieve(question: str) -> dict:
    """
    Find the most relevant Community node by embedding similarity over summaries,
    then return its member entities. Community embeddings are computed on the fly.
    Returns dict with mode, community_id, summary, and member_entities.
    """
    query_vec = embed(question)

    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            comm_result = await session.run(
                "MATCH (c:Community) RETURN c.id AS id, c.summary AS summary"
            )
            communities = await comm_result.data()

            if not communities:
                return {
                    "mode": "community",
                    "community_id": None,
                    "summary": "",
                    "member_entities": [],
                }

            summary_vecs = [embed(c["summary"]) for c in communities]
            scores = [_cosine(query_vec, sv) for sv in summary_vecs]
            best = communities[int(np.argmax(scores))]

            member_result = await session.run(
                """
                MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $cid})
                RETURN e.name AS name
                """,
                cid=best["id"],
            )
            members = [r["name"] for r in await member_result.data()]

    return {
        "mode": "community",
        "community_id": best["id"],
        "summary": best["summary"],
        "member_entities": members,
    }
