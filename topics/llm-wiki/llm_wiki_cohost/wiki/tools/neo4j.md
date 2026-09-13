---
title: Neo4j
weeks: [graphs-neo4j]
---

Graph database. Stores data as nodes and typed, directed relationships (edges)
rather than rows and foreign keys. Query language is Cypher. Since version 5.11
it includes a native HNSW vector index, making it a unified store for both
graph traversal and semantic similarity search — no cross-service hops needed.

In the series, Neo4j is used in two forms: self-hosted via a Flyte
`AppEnvironment` on the DGX devbox, and Neo4j AuraDB Free (managed cloud,
no setup, sufficient for demo-scale graphs).

## Usage across the series

### Week 7 — Graph Data with Neo4j (2026-05-08)

**`graphrag-neo4j-flyte/`** — Neo4j 5 (`neo4j:5.26-community`) deployed as a
Flyte `AppEnvironment` on the devbox. A one-line `Dockerfile.neo4j` wraps the
official image; `from_dockerfile` skips the Flyte image builder's `USER flyte`
footer that breaks the Neo4j container. HTTP API on port 7474 (Bolt is blocked
by Knative's queue-proxy; HTTP supports the full Cypher surface including vector
index queries). No persistent volume by default — `snapshot.py` provides
on-demand snapshot/restore to a `flyte.io.Dir` in rustfs.

**`graph_rag_chatbot/`** — Neo4j AuraDB Free. Bolt over TLS (`neo4j+s://`)
accessible from Union cluster. Community detection runs in Python since GDS
library is unavailable on the free tier.

**Graph schema used in `graph_rag_chatbot/`:**
```
(Document)-[:HAS_CHUNK]→(Chunk)-[:MENTIONS]→(Entity)
(Entity)-[:RELATED {type, description}]→(Entity)
(Entity)-[:BELONGS_TO]→(Community)
```

**Key GraphRAG query — vector search + graph traversal in one Cypher statement:**
```cypher
CALL db.index.vector.queryNodes('chunk-embeddings', 3, $query_embedding)
YIELD node AS chunk, score
WHERE score >= 0.75
MATCH (chunk)-[:MENTIONS]->(entity:Entity)
OPTIONAL MATCH (entity)-[:RELATED]->(other:Entity)
RETURN chunk.text, entity.name, entity.type,
       collect(other.name) AS related_entities, score
ORDER BY score DESC
```

This is what pgvector cannot do: vector search and relationship traversal
combined in a single query.

**Three retrieval modes across both projects:**

| Mode | Mechanism | Wins when |
|---|---|---|
| Vector | HNSW on chunk embeddings | Direct, single-concept questions |
| Vector + 1-hop expand | Vector seeds → walk CITES / MENTIONS edges | Relationship questions; answer lives in citation structure |
| Hybrid RRF | Vector + graph-only cohort query, fused by `1/(K+rank)` | Authority/centrality matters more than abstract phrasing |

**Driver notes:** Neo4j Python driver `>=5.20.0`; must use `with` blocks
(no auto-close in destructors); create and close driver per Flyte task;
always use parameterized queries — never f-strings in Cypher.
