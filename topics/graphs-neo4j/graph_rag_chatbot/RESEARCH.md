# GraphRAG Chatbot — Research Notes

Companion project to `topics/vectorstore/vector_rag_chatbot`. Uses the same
15 Everstorm Outfitters PDFs to demonstrate when and why you need graph
retrieval on top of — or instead of — pure vector search.

---

## The Core Thesis

Neither vector RAG nor graph RAG is universally better. They answer different
question types well:

| Question type | Vector RAG | GraphRAG |
|---|---|---|
| Factual / single-chunk | ✅ | ✅ |
| Relationship-aware | ⚠️ | ✅ |
| Multi-hop / cross-document | ❌ | ✅ |
| Thematic / global | ❌ | ✅ (community summaries) |

The best production systems use both together. Neo4j enables this in a single
database: vector index on chunk embeddings + graph traversal over entity
relationships, combined in one Cypher query.

---

## Hosting Decision — Neo4j AuraDB Free

**Decision: AuraDB Free**

Evaluated three options:

| Option | Cost | GDS library | Setup time | Decision |
|---|---|---|---|---|
| AuraDB Free | $0 | ❌ | Minutes | ✅ Chosen |
| GCP VM + Neo4j Docker | ~$2-5/wk | ✅ | 30-60 min | Post-demo upgrade |
| GCP Marketplace Neo4j | ~$170-340/wk | ✅ | Hours | ❌ Too expensive |

**Why AuraDB Free:**
- Sufficient capacity for demo (200k nodes / 400k relationships limit vs our ~few thousand nodes)
- Ready immediately — no VM, no firewall rules, no SSH
- Accessible from Union cluster (AWS us-east-2) via `neo4j+s://` Bolt over TLS
- Post-demo migration to GCP VM is one line change (connection string only)

**AuraDB Free limits to know:**
- No GDS library (community detection must be done in Python)
- Vector index optimization requires > 4 GB RAM — free tier falls back to
  Lucene-backed HNSW. Performance difference is irrelevant at our chunk count.
- Auto-pauses on idle

**Connection string format:**
```
neo4j+s://<id>.databases.neo4j.io
```

---

## Neo4j Python Driver

**Version:** `>=5.20.0` (see requirements.txt)

**Breaking changes from 5.x:**
- Python 3.10+ required
- Must use `with` blocks — drivers no longer auto-close in destructors
- Create and close driver per Flyte task (tasks are separate processes)

```python
from neo4j import GraphDatabase

URI  = "neo4j+s://xxxxxxxx.databases.neo4j.io"
AUTH = ("neo4j", "<password>")

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    with driver.session() as session:
        session.run("MERGE (c:Chunk {id: $id}) SET c.text = $text",
                    id=chunk_id, text=text)
```

Always use parameterized queries — never f-strings in Cypher.

---

## Neo4j Vector Search

Native HNSW vector indexes via Apache Lucene. Available since Neo4j 5.11.
Supports cosine similarity and Euclidean distance, up to 4096 dimensions.

**Create index:**
```cypher
CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON (c.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 384,
    `vector.similarity_function`: 'cosine'
  }
}
```

**The key GraphRAG query — vector search + graph traversal in one statement:**
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

`TOP_K=3` and the `score >= 0.75` threshold work together to filter noise.
Without both, documents that share surface-level vocabulary with the query
(e.g. "Gift Cards" when asking about shipping) score above the threshold and
pollute the entity expansion with irrelevant nodes.

This is what pgvector cannot do — vector search and relationship traversal
combined in a single query.

---

## Graph Schema

### Nodes

| Label | Key Properties |
|---|---|
| `Document` | name |
| `Chunk` | id, text, embedding, chunk_index, source_doc |
| `Entity` | name, type, description, community_id |
| `Community` | id, summary |

### Relationships

| Relationship | From → To | Meaning |
|---|---|---|
| `HAS_CHUNK` | Document → Chunk | Document contains this chunk |
| `MENTIONS` | Chunk → Entity | Chunk references this entity |
| `RELATED` | Entity → Entity | Typed edge with `type` and `description` properties |
| `BELONGS_TO` | Entity → Community | Entity is member of this community |

### Everstorm Entity Ontology

**Entity types (constrained — better extraction quality than open-ended):**
`PRODUCT`, `POLICY`, `PROGRAM`, `TIER`, `BENEFIT`, `CONDITION`, `PROCESS`

**Relationship types:**
`HAS_POLICY`, `QUALIFIES_FOR`, `REQUIRES`, `APPLIES_TO`, `PART_OF`, `COVERS`

---

## Entity Extraction with Claude

Use Claude's tool use (structured output) to force valid JSON.
Never parse markdown — tool use eliminates JSON parse failures entirely.

**Output schema:**
```json
{
  "entities": [
    {
      "name": "Elite Tier",
      "type": "TIER",
      "description": "Highest loyalty tier with premium benefits"
    }
  ],
  "relationships": [
    {
      "source": "Elite Tier",
      "target": "Free Returns",
      "type": "QUALIFIES_FOR",
      "description": "Elite members receive free return shipping"
    }
  ]
}
```

**Prompt patterns that work:**
1. Define entity and relationship types upfront — constrain to domain ontology
2. Use tool use / structured output — eliminates JSON parse failures
3. Multi-shot examples in `<examples>` tags — 3-5 examples improves consistency
4. Run extraction inside `process_pdf` per PDF task — parallelism at PDF level via `asyncio.gather`

**Entity normalization (critical):**
Different chunks will refer to the same entity differently — "Everstorm",
"Everstorm Outfitters", "the Company". Without de-duplication the graph
becomes disconnected and traversal produces no benefit.

Fix: embed all extracted entity names with gte-small, merge nodes with
cosine similarity >= `ENTITY_MERGE_THRESHOLD` (0.95). Implemented in
`resolve_entities` in `ingest/enrichment.py`.

---

## Community Detection

GDS library (Leiden/Louvain) not available on AuraDB Free.

**Workaround:** Run community detection in Python using `python-louvain`
on a NetworkX graph built from the entity relationships stored in Neo4j.
Assign community IDs, use Claude to summarize each community, store
Community nodes back in Neo4j.

```python
import community as community_louvain
import networkx as nx

# Build NetworkX graph from Neo4j entity relationships
G = nx.Graph()
# ... add nodes and edges from Cypher query ...

# Detect communities
partition = community_louvain.best_partition(G, resolution=LOUVAIN_RESOLUTION)
# partition = {entity_name: community_id, ...}

# Summarize each community with Claude, store as Community nodes
```

Post-demo upgrade: migrate to GCP VM + Neo4j Docker to get native GDS
and drop this workaround.

---

## Retrieval Patterns

Three modes, selected by a Claude query router:

**Mode A — Hybrid: Vector + Graph Expansion** (factual / specific questions)
1. Embed query with gte-small → HNSW vector search finds top-3 Chunk nodes (score ≥ 0.75)
2. Follow `MENTIONS` edges → collect Entity nodes referenced by those chunks
3. Combined context (chunks + entities) sent to Claude

The threshold + TOP_K combination is critical — without it, chunks that share
surface vocabulary with the query inflate the entity list with irrelevant nodes.

**Mode B — Entity Lookup + Traversal** (relationship questions)
1. Claude tool use extracts named entities from the question
2. Neo4j pattern match on Entity names → traverse `RELATED` edges to build neighborhood subgraph
3. Entity descriptions + connected neighbors sent to Claude

**Mode C — Community Summary Search** (thematic / global questions)
1. Embed question with gte-small → cosine similarity against all Community summaries
2. Return the best-matching community summary + member entities
3. Answers "what programs does Everstorm have for X?"

---

## Python Libraries

| Library | Version | Role |
|---|---|---|
| `neo4j` | >=5.20.0 | Driver — all Cypher queries |
| `anthropic` | latest | Entity extraction, routing, generation |
| `sentence-transformers` | >=3.0.0 | gte-small embeddings (384D) |
| `python-louvain` | >=0.16 | Community detection (AuraDB workaround) |
| `networkx` | >=3.3 | Graph structure for community detection |
| `PyMuPDF` | >=1.24.0 | PDF text extraction |
| `langchain-text-splitters` | >=0.3.0 | RecursiveCharacterTextSplitter |
| `numpy` | >=1.26.0 | Cosine similarity in entity resolution |
| `gradio` | >=4.44.0 | UI |
| `flyte` | >=2.1.2 | Flyte 2.x task orchestration |

**Do not use:**
- Microsoft `graphrag` package — too opinionated, expensive LLM calls, not Neo4j native
- `neo4j-graphrag` / `langchain-neo4j` / `llama-index-neo4j` — framework overhead; direct Cypher is clearer and sufficient here

---

## Workflow Structure (Modular)

Designed modular from day one — learned from vector_rag_chatbot.
Each task has its own file so engineers can open one file and see
exactly one job. `app.py` imports only from `workflows.py`.
`flyte deploy` bundles all modules transitively.

### Project Structure

```
graph_rag_chatbot/
├── app.py                        ← Gradio UI + Union AppEnvironment + serve entry point
├── workflows.py                  ← thin re-export: ingest_pipeline + query_pipeline
├── config.py                     ← constants, secrets, TaskEnvironment, entity ontology
│
├── ingest/
│   ├── __init__.py               ← exports ingest_pipeline
│   ├── chunking.py               ← parse_and_chunk (PyMuPDF + RecursiveCharacterTextSplitter)
│   ├── extraction.py             ← extract_entities (Claude tool use → entities + relationships)
│   ├── graph_loader.py           ← load_graph + create_vector_index
│   ├── enrichment.py             ← resolve_entities + detect_communities + summarize_communities
│   └── pipeline.py               ← ingest_pipeline orchestrator
│
├── query/
│   ├── __init__.py               ← exports query_pipeline
│   ├── routing.py                ← route_query (Claude classifier)
│   ├── retrieval.py              ← hybrid_retrieve + entity_retrieve + community_retrieve
│   ├── generation.py             ← generate (mode-specific RAG prompt → Claude answer)
│   └── pipeline.py               ← query_pipeline orchestrator
│
├── requirements.txt
├── Dockerfile                    ← task image (gte-small baked in)
├── Dockerfile.app                ← app serving image
├── architecture.html             ← interactive architecture walkthrough
└── data/                         ← 15 Everstorm Outfitters PDFs
```

### Ingest Pipeline

```
ingest_pipeline  (ingest/pipeline.py — orchestrator)
    │
    ├── process_pdf × 15          ingest/pipeline.py
    │       asyncio.gather fan-out — all 15 PDFs run in parallel on Union
    │       Each task:
    │         parse_and_chunk     ingest/chunking.py
    │             PyMuPDF → text · RecursiveCharacterTextSplitter (size=800, overlap=150)
    │         extract_entities    ingest/extraction.py
    │             Claude tool use → {entities, relationships} JSON per chunk
    │
    ├── load_graph                ingest/graph_loader.py
    │       gte-small embeddings on Chunk text
    │       MERGE Document, Chunk, Entity nodes into Neo4j
    │       Create HAS_CHUNK, MENTIONS, RELATED edges
    │
    ├── create_vector_index       ingest/graph_loader.py
    │       HNSW index on Chunk.embedding (384D, cosine) — idempotent
    │
    ├── resolve_entities          ingest/enrichment.py
    │       embed entity names → merge near-duplicate nodes (cosine >= 0.95)
    │
    ├── detect_communities        ingest/enrichment.py
    │       python-louvain on NetworkX → assign community_id to each Entity
    │
    └── summarize_communities     ingest/enrichment.py
            Claude summarizes each community → Community nodes + BELONGS_TO edges
```

### Query Pipeline

```
query_pipeline  (query/pipeline.py — orchestrator)
    │
    ├── route_query               query/routing.py
    │       Claude tool use classifier → "hybrid" | "entity" | "community"
    │
    ├── [hybrid]  hybrid_retrieve     query/retrieval.py
    │       embed query → HNSW vector search → follow MENTIONS edges to Entities
    │
    ├── [entity]  entity_retrieve     query/retrieval.py
    │       Claude extracts entity names → Neo4j neighborhood traversal via RELATED
    │
    ├── [community] community_retrieve  query/retrieval.py
    │       embed question → cosine similarity to Community summaries → best match
    │
    └── generate                  query/generation.py
            mode-specific RAG prompt → Claude answer + sources + retrieval_mode + entities_used + routing_reason
```

---

## Demo Story

Same 15 Everstorm PDFs. The GraphRAG chatbot shows retrieval mode badges
(Hybrid / Entity / Community) in the chat UI so the routing is visible.

**Questions that show the graph advantage:**

| Question | Vector RAG | GraphRAG |
|---|---|---|
| "What is the return window?" | ✅ | ✅ |
| "What benefits do Elite members get on returns?" | ⚠️ | ✅ Tier→Benefit→Policy |
| "Which policies apply to international purchases?" | ❌ | ✅ Multi-hop |
| "What programs does Everstorm have for repeat customers?" | ❌ | ✅ Community summary |

**The narrative:** Vector RAG gets you to the right neighborhood.
Graph traversal tells you how everything in that neighborhood connects.
Community summaries zoom out to the full picture for thematic questions.
Production RAG needs all three.

---

## Cost Summary

| Resource | Cost |
|---|---|
| Neo4j AuraDB Free | $0 |
| Union.ai compute | Existing account |
| Anthropic API (extraction + generation) | ~$0.50-1.00 per full ingest run |
| Docker Hub | Existing account |
| Total | ~$1/run |
