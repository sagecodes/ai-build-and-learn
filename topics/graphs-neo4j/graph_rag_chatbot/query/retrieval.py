"""
query/retrieval.py — retrieval tasks (one dispatched per query based on mode)

hybrid_retrieve:    vector search + graph expansion
entity_retrieve:    named entity lookup + neighborhood traversal
community_retrieve: community summary similarity search
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from config import CLAUDE_MODEL, EMBED_MODEL, VECTOR_INDEX_NAME, anthropic_client, neo4j_driver, task_env

_TOP_K = 5
_HYBRID_SCORE_THRESHOLD = 0.75  # drop chunks below this cosine similarity


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


@task_env.task
async def hybrid_retrieve(question: str) -> str:
    """
    Mode A — vector search over Chunk nodes + graph expansion to nearby Entities.

    Returns:
        JSON — {mode, chunks: [{chunk_id, source_doc, text, score}],
                entities: [{name, type, description}]}
    """
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(question).tolist()

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            chunk_rows = session.run(
                f"""
                CALL db.index.vector.queryNodes('{VECTOR_INDEX_NAME}', $top_k, $embedding)
                YIELD node AS chunk, score
                WHERE score >= $threshold
                RETURN chunk.id AS chunk_id,
                       chunk.source_doc AS source_doc,
                       chunk.text AS text,
                       score
                ORDER BY score DESC
                """,
                top_k=_TOP_K,
                embedding=query_vec,
                threshold=_HYBRID_SCORE_THRESHOLD,
            ).data()

            chunk_ids = [r["chunk_id"] for r in chunk_rows]
            entity_rows = session.run(
                """
                MATCH (c:Chunk)-[:MENTIONS]->(e:Entity)
                WHERE c.id IN $chunk_ids
                OPTIONAL MATCH (e)-[:RELATED]->(neighbor:Entity)
                RETURN DISTINCT e.name AS name, e.type AS type,
                       e.description AS description
                LIMIT 20
                """,
                chunk_ids=chunk_ids,
            ).data()

    return json.dumps({
        "mode": "hybrid",
        "chunks": chunk_rows,
        "entities": entity_rows,
    })


_ENTITY_EXTRACT_TOOL = {
    "name": "extract_entities",
    "description": "Extract named entities mentioned in a question.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of entity names mentioned in the question.",
            }
        },
        "required": ["entities"],
    },
}


@task_env.task
async def entity_retrieve(question: str) -> str:
    """
    Mode B — extract named entities from question, traverse their neighborhood in Neo4j.

    Returns:
        JSON — {mode, entities: [{name, type, description, neighbors: [...]}]}
    """
    client = anthropic_client()

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=256,
        tools=[_ENTITY_EXTRACT_TOOL],
        tool_choice={"type": "tool", "name": "extract_entities"},
        messages=[{"role": "user", "content": question}],
    )
    tool_block = next(b for b in response.content if b.type == "tool_use")
    entity_names = tool_block.input.get("entities", [])

    if not entity_names:
        return json.dumps({"mode": "entity", "entities": []})

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            results = []
            for name in entity_names:
                rows = session.run(
                    """
                    MATCH (e:Entity)
                    WHERE toLower(e.name) CONTAINS toLower($name)
                    OPTIONAL MATCH (e)-[r:RELATED]->(neighbor:Entity)
                    RETURN e.name AS name, e.type AS type, e.description AS description,
                           collect({
                               name: neighbor.name,
                               rel_type: r.type,
                               description: r.description
                           }) AS neighbors
                    LIMIT 5
                    """,
                    name=name,
                ).data()
                results.extend(rows)

    return json.dumps({"mode": "entity", "entities": results})


@task_env.task
async def community_retrieve(question: str) -> str:
    """
    Mode C — find the most relevant Community node by embedding similarity.

    Returns:
        JSON — {mode, community_id, summary, member_entities: [name, ...]}
    """
    model = SentenceTransformer(EMBED_MODEL)
    query_vec = model.encode(question)

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            community_rows = session.run(
                "MATCH (c:Community) RETURN c.id AS id, c.summary AS summary"
            ).data()

            if not community_rows:
                return json.dumps({"mode": "community", "community_id": None,
                                   "summary": "", "member_entities": []})

            summaries = [r["summary"] for r in community_rows]
            summary_vecs = model.encode(summaries)

            scores = [_cosine_sim(query_vec, sv) for sv in summary_vecs]
            best_idx = int(np.argmax(scores))
            best = community_rows[best_idx]

            member_rows = session.run(
                """
                MATCH (e:Entity)-[:BELONGS_TO]->(c:Community {id: $cid})
                RETURN e.name AS name
                """,
                cid=best["id"],
            ).data()

    return json.dumps({
        "mode": "community",
        "community_id": best["id"],
        "summary": best["summary"],
        "member_entities": [r["name"] for r in member_rows],
    })
