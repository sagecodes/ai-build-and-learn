"""
graph_ingest.py — ingest Everstorm PDFs into Neo4j AuraDB Free.

Pipeline:
  1. Extract text from each PDF and split into chunks
  2. Extract entities + relationships from each chunk via Claude tool use
     (parallel with asyncio.Semaphore to stay within rate limits)
  3. Embed chunks with fastembed
  4. Write Chunk, Entity, and Document nodes + all relationships to Neo4j
  5. Create HNSW vector index on Chunk.embedding
  6. Resolve near-duplicate Entity nodes by embedding similarity
  7. Detect entity communities via Louvain algorithm (networkx)
  8. Summarize each community with Claude and write Community nodes

Run once before launching the app:
    python ingest/graph_ingest.py
"""

import asyncio
import logging
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pymupdf
from community import best_partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from neo4j import AsyncGraphDatabase

from backends.shared.claude import get_client
from backends.shared.embeddings import embed, get_embedder
from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    DATA_DIR,
    EMBEDDING_DIMS,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USERNAME,
    VECTOR_INDEX_NAME,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE    = 800
CHUNK_OVERLAP = 100

ENTITY_TYPES = [
    "PRODUCT", "POLICY", "PROGRAM", "TIER", "BENEFIT", "CONDITION", "PROCESS",
]
RELATIONSHIP_TYPES = [
    "HAS_POLICY", "QUALIFIES_FOR", "REQUIRES", "APPLIES_TO", "PART_OF", "COVERS",
]

ENTITY_MERGE_THRESHOLD = 0.95
LOUVAIN_RESOLUTION     = 1.0
EXTRACTION_CONCURRENCY = 5   # max parallel Claude extraction calls

_EXTRACT_TOOL = {
    "name": "extract_graph",
    "description": "Extract entities and relationships from a text chunk.",
    "input_schema": {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":        {"type": "string"},
                        "type":        {"type": "string", "enum": ENTITY_TYPES},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "type", "description"],
                },
            },
            "relationships": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source":      {"type": "string"},
                        "target":      {"type": "string"},
                        "type":        {"type": "string", "enum": RELATIONSHIP_TYPES},
                        "description": {"type": "string"},
                    },
                    "required": ["source", "target", "type", "description"],
                },
            },
        },
        "required": ["entities", "relationships"],
    },
}

_EXTRACT_SYSTEM = (
    "You are a knowledge-graph extraction assistant for Everstorm Outfitters. "
    "Extract only entities and relationships that appear explicitly in the provided text. "
    "Do not infer or hallucinate facts not present in the chunk. "
    f"Entity types: {', '.join(ENTITY_TYPES)}. "
    f"Relationship types: {', '.join(RELATIONSHIP_TYPES)}."
)


# ── Step 1: Load and chunk PDFs ───────────────────────────────────────────────

def _load_chunks(data_dir: str) -> list[dict]:
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = []
    for pdf_path in pdf_paths:
        doc  = pymupdf.open(str(pdf_path))
        text = "".join(page.get_text() for page in doc)
        doc.close()
        for i, chunk_text in enumerate(splitter.split_text(text)):
            chunks.append({
                "chunk_id":    f"{pdf_path.name}::{i}",
                "source_doc":  pdf_path.name,
                "chunk_index": i,
                "chunk_text":  chunk_text,
            })
        log.info(f"  {pdf_path.name}: {len([c for c in chunks if c['source_doc'] == pdf_path.name])} chunks")
    return chunks


# ── Step 2: Extract entities and relationships via Claude ─────────────────────

async def _extract_one(chunk: dict, semaphore: asyncio.Semaphore) -> dict:
    async with semaphore:
        response = await get_client().messages.create(
            model=CLAUDE_MODEL,
            max_tokens=2048,
            system=_EXTRACT_SYSTEM,
            tools=[_EXTRACT_TOOL],
            tool_choice={"type": "tool", "name": "extract_graph"},
            messages=[{
                "role": "user",
                "content": f"Extract entities and relationships:\n\n{chunk['chunk_text']}",
            }],
        )
        tool_block = next(b for b in response.content if b.type == "tool_use")
        return {**chunk, **tool_block.input}


async def _extract_all(chunks: list[dict]) -> list[dict]:
    semaphore = asyncio.Semaphore(EXTRACTION_CONCURRENCY)
    results   = await asyncio.gather(*[_extract_one(c, semaphore) for c in chunks])
    return list(results)


# ── Step 3: Embed chunks ──────────────────────────────────────────────────────

def _embed_chunks(chunks: list[dict]) -> list[dict]:
    embedder   = get_embedder()
    texts      = [c["chunk_text"] for c in chunks]
    embeddings = list(embedder.embed(texts))
    return [{**c, "embedding": embeddings[i].tolist()} for i, c in enumerate(chunks)]


# ── Step 4: Write to Neo4j ────────────────────────────────────────────────────

async def _write_to_neo4j(enriched_chunks: list[dict]) -> None:
    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            chunks_written       = 0
            entities_written     = 0
            relationships_written = 0

            for chunk in enriched_chunks:
                await session.run(
                    """
                    MERGE (d:Document {name: $source_doc})
                    MERGE (c:Chunk {id: $chunk_id})
                      ON CREATE SET c.text        = $chunk_text,
                                    c.source_doc  = $source_doc,
                                    c.chunk_index = $chunk_index,
                                    c.embedding   = $embedding
                    MERGE (d)-[:HAS_CHUNK]->(c)
                    """,
                    source_doc=chunk["source_doc"],
                    chunk_id=chunk["chunk_id"],
                    chunk_text=chunk["chunk_text"],
                    chunk_index=chunk["chunk_index"],
                    embedding=chunk["embedding"],
                )
                chunks_written += 1

                for ent in chunk.get("entities", []):
                    await session.run(
                        """
                        MERGE (e:Entity {name: $name})
                          ON CREATE SET e.type = $type, e.description = $description
                        """,
                        name=ent["name"], type=ent["type"], description=ent["description"],
                    )
                    await session.run(
                        """
                        MATCH (c:Chunk {id: $chunk_id})
                        MATCH (e:Entity {name: $name})
                        MERGE (c)-[:MENTIONS]->(e)
                        """,
                        chunk_id=chunk["chunk_id"], name=ent["name"],
                    )
                    entities_written += 1

                for rel in chunk.get("relationships", []):
                    await session.run(
                        """
                        MATCH (a:Entity {name: $source})
                        MATCH (b:Entity {name: $target})
                        MERGE (a)-[r:RELATED {type: $rel_type}]->(b)
                          ON CREATE SET r.description = $description
                        """,
                        source=rel["source"], target=rel["target"],
                        rel_type=rel["type"], description=rel["description"],
                    )
                    relationships_written += 1

    log.info(
        f"  Chunks: {chunks_written}  |  "
        f"Entities: {entities_written}  |  "
        f"Relationships: {relationships_written}"
    )


# ── Step 5: Create vector index ───────────────────────────────────────────────

async def _create_vector_index() -> None:
    cypher = (
        f"CREATE VECTOR INDEX `{VECTOR_INDEX_NAME}` IF NOT EXISTS "
        f"FOR (c:Chunk) ON (c.embedding) "
        f"OPTIONS {{indexConfig: {{"
        f"  `vector.dimensions`: {EMBEDDING_DIMS},"
        f"  `vector.similarity_function`: 'cosine'"
        f"}}}}"
    )
    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            await session.run(cypher)
    log.info(f"  Vector index '{VECTOR_INDEX_NAME}' ready.")


# ── Step 6: Resolve near-duplicate entities ───────────────────────────────────

async def _resolve_entities() -> None:
    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            result = await session.run("MATCH (e:Entity) RETURN e.name AS name")
            names  = [r["name"] for r in await result.data()]

    if len(names) < 2:
        log.info("  No entity resolution needed.")
        return

    embedder  = get_embedder()
    vecs      = list(embedder.embed(names))
    name_vecs = {name: vecs[i] for i, name in enumerate(names)}

    def _cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    merge_pairs = [
        (a, b)
        for a, b in combinations(names, 2)
        if _cosine(name_vecs[a], name_vecs[b]) >= ENTITY_MERGE_THRESHOLD
    ]

    if not merge_pairs:
        log.info("  No near-duplicate entities found.")
        return

    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            for keep, drop in merge_pairs:
                await session.run(
                    """
                    MATCH (c:Chunk)-[:MENTIONS]->(drop:Entity {name: $drop})
                    MATCH (keep:Entity {name: $keep})
                    MERGE (c)-[:MENTIONS]->(keep)
                    """,
                    drop=drop, keep=keep,
                )
                await session.run(
                    """
                    MATCH (drop:Entity {name: $drop})-[r:RELATED]->(other:Entity)
                    MATCH (keep:Entity {name: $keep})
                    MERGE (keep)-[:RELATED {type: r.type, description: r.description}]->(other)
                    """,
                    drop=drop, keep=keep,
                )
                await session.run(
                    "MATCH (e:Entity {name: $drop}) DETACH DELETE e",
                    drop=drop,
                )

    log.info(f"  Merged {len(merge_pairs)} near-duplicate entity pair(s).")


# ── Step 7: Detect communities ────────────────────────────────────────────────

async def _detect_communities() -> None:
    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            ent_result = await session.run("MATCH (e:Entity) RETURN e.name AS name")
            rel_result = await session.run(
                "MATCH (a:Entity)-[:RELATED]->(b:Entity) "
                "RETURN a.name AS source, b.name AS target"
            )
            entity_names = [r["name"] for r in await ent_result.data()]
            rel_rows     = await rel_result.data()

    G = nx.Graph()
    G.add_nodes_from(entity_names)
    G.add_edges_from((r["source"], r["target"]) for r in rel_rows)

    partition = best_partition(G, resolution=LOUVAIN_RESOLUTION)
    n_communities = len(set(partition.values()))

    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            for entity_name, community_id in partition.items():
                await session.run(
                    "MATCH (e:Entity {name: $name}) SET e.community_id = $cid",
                    name=entity_name, cid=community_id,
                )

    log.info(f"  {n_communities} communities detected across {len(entity_names)} entities.")


# ── Step 8: Summarize communities ─────────────────────────────────────────────

async def _summarize_communities() -> None:
    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            ent_result = await session.run(
                "MATCH (e:Entity) WHERE e.community_id IS NOT NULL "
                "RETURN e.name AS name, e.type AS type, "
                "       e.description AS description, e.community_id AS cid"
            )
            rel_result = await session.run(
                "MATCH (a:Entity)-[r:RELATED]->(b:Entity) "
                "WHERE a.community_id IS NOT NULL "
                "RETURN a.name AS source, b.name AS target, "
                "       r.type AS rel_type, r.description AS description, "
                "       a.community_id AS cid"
            )
            ent_rows = await ent_result.data()
            rel_rows = await rel_result.data()

    from collections import defaultdict
    community_entities: dict = defaultdict(list)
    community_rels:     dict = defaultdict(list)
    for r in ent_rows:
        community_entities[r["cid"]].append(r)
    for r in rel_rows:
        community_rels[r["cid"]].append(r)

    async def _summarize_one(cid, entities, rels, semaphore):
        async with semaphore:
            entity_lines = "\n".join(
                f"- {e['name']} ({e['type']}): {e['description']}" for e in entities
            )
            rel_lines = "\n".join(
                f"- {r['source']} --[{r['rel_type']}]--> {r['target']}: {r['description']}"
                for r in rels
            ) or "None"

            response = await get_client().messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": (
                        "Summarize this cluster of related concepts from Everstorm Outfitters "
                        "policy documents in 2-3 concise sentences. Describe what the group "
                        "is about and how the concepts relate.\n\n"
                        f"Entities:\n{entity_lines}\n\n"
                        f"Relationships:\n{rel_lines}"
                    ),
                }],
            )
            return cid, response.content[0].text.strip(), entities

    semaphore = asyncio.Semaphore(5)
    tasks     = [
        _summarize_one(cid, entities, community_rels.get(cid, []), semaphore)
        for cid, entities in community_entities.items()
    ]
    summaries = await asyncio.gather(*tasks)

    async with AsyncGraphDatabase.driver(
        NEO4J_URI, auth=(NEO4J_USERNAME, NEO4J_PASSWORD)
    ) as driver:
        async with driver.session() as session:
            for cid, summary, entities in summaries:
                await session.run(
                    "MERGE (c:Community {id: $cid}) SET c.summary = $summary",
                    cid=cid, summary=summary,
                )
                for ent in entities:
                    await session.run(
                        """
                        MATCH (e:Entity {name: $name})
                        MATCH (c:Community {id: $cid})
                        MERGE (e)-[:BELONGS_TO]->(c)
                        """,
                        name=ent["name"], cid=cid,
                    )

    log.info(f"  {len(summaries)} communities summarized and written.")


# ── Orchestrator ──────────────────────────────────────────────────────────────

async def ingest(data_dir: str = str(DATA_DIR)) -> None:
    log.info("── Step 1: Loading and chunking PDFs")
    chunks = _load_chunks(data_dir)
    log.info(f"  Total: {len(chunks)} chunks\n")

    log.info("── Step 2: Extracting entities + relationships (parallel Claude calls)")
    enriched = await _extract_all(chunks)
    total_entities = sum(len(c.get("entities", [])) for c in enriched)
    total_rels     = sum(len(c.get("relationships", [])) for c in enriched)
    log.info(f"  Extracted {total_entities} entities, {total_rels} relationships\n")

    log.info("── Step 3: Embedding chunks")
    enriched = _embed_chunks(enriched)
    log.info(f"  {len(enriched)} chunks embedded\n")

    log.info("── Step 4: Writing to Neo4j")
    await _write_to_neo4j(enriched)
    log.info("")

    log.info("── Step 5: Creating vector index")
    await _create_vector_index()
    log.info("")

    log.info("── Step 6: Resolving near-duplicate entities")
    await _resolve_entities()
    log.info("")

    log.info("── Step 7: Detecting communities")
    await _detect_communities()
    log.info("")

    log.info("── Step 8: Summarizing communities")
    await _summarize_communities()
    log.info("")

    log.info("Graph ingest complete.")


if __name__ == "__main__":
    asyncio.run(ingest())
