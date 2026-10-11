"""
ingest/graph_loader.py — load_graph and create_vector_index tasks

load_graph:          write chunks, entities, and relationships to Neo4j.
create_vector_index: build the HNSW vector index on Chunk.embedding (idempotent).
"""

import json
from typing import List

from sentence_transformers import SentenceTransformer

from config import (
    EMBED_MODEL,
    EMBED_DIM,
    VECTOR_INDEX_NAME,
    VECTOR_SIMILARITY,
    neo4j_driver,
    task_env,
)

_MERGE_CHUNK_Q = """
MERGE (d:Document {name: $source_doc})
MERGE (c:Chunk {id: $chunk_id})
  ON CREATE SET c.text = $chunk_text,
               c.source_doc = $source_doc,
               c.chunk_index = $chunk_index
MERGE (d)-[:HAS_CHUNK]->(c)
"""

_SET_EMBEDDING_Q = """
MATCH (c:Chunk {id: $chunk_id})
SET c.embedding = $embedding
"""

_MERGE_ENTITY_Q = """
MERGE (e:Entity {name: $name})
  ON CREATE SET e.type = $type, e.description = $description
"""

_MERGE_RELATIONSHIP_Q = """
MATCH (a:Entity {name: $source})
MATCH (b:Entity {name: $target})
MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
  ON CREATE SET r.description = $description
"""

_MERGE_MENTIONS_Q = """
MATCH (c:Chunk {id: $chunk_id})
MATCH (e:Entity {name: $entity_name})
MERGE (c)-[:MENTIONS]->(e)
"""


@task_env.task
async def load_graph(extraction_results: List[str]) -> str:
    """
    Write all chunks, entities, and relationships to Neo4j.

    Returns:
        JSON summary — {chunks_written, entities_written, relationships_written}.
    """
    model = SentenceTransformer(EMBED_MODEL)
    driver = neo4j_driver()

    chunks_written = 0
    entities_written = 0
    relationships_written = 0

    with driver:
        with driver.session() as session:
            for raw in extraction_results:
                result = json.loads(raw)
                chunk_id = result["chunk_id"]
                source_doc = result["source_doc"]
                chunk_text = result["chunk_text"]
                chunk_index = int(chunk_id.split("::")[-1])
                entities = result["entities"]
                relationships = result["relationships"]

                session.run(
                    _MERGE_CHUNK_Q,
                    source_doc=source_doc,
                    chunk_id=chunk_id,
                    chunk_text=chunk_text,
                    chunk_index=chunk_index,
                )

                embedding = model.encode(chunk_text).tolist()
                session.run(_SET_EMBEDDING_Q, chunk_id=chunk_id, embedding=embedding)
                chunks_written += 1

                for ent in entities:
                    session.run(
                        _MERGE_ENTITY_Q,
                        name=ent["name"],
                        type=ent["type"],
                        description=ent["description"],
                    )
                    session.run(
                        _MERGE_MENTIONS_Q,
                        chunk_id=chunk_id,
                        entity_name=ent["name"],
                    )
                    entities_written += 1

                for rel in relationships:
                    session.run(
                        _MERGE_RELATIONSHIP_Q,
                        source=rel["source"],
                        target=rel["target"],
                        rel_type=rel["type"],
                        description=rel["description"],
                    )
                    relationships_written += 1

    return json.dumps({
        "chunks_written": chunks_written,
        "entities_written": entities_written,
        "relationships_written": relationships_written,
    })


@task_env.task
async def create_vector_index() -> str:
    """Create the HNSW vector index on Chunk.embedding (idempotent)."""
    cypher = (
        f"CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}` IF NOT EXISTS "
        f"FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{indexConfig: {{"
        f"  `vector.dimensions`: {EMBED_DIM},"
        f"  `vector.similarity_function`: '{VECTOR_SIMILARITY}'"
        f"}}}}"
    )

    driver = neo4j_driver()
    with driver:
        with driver.session() as session:
            session.run(cypher)

    return "ok"
