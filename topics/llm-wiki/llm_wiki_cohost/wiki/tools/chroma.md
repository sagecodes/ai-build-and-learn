---
title: Chroma
weeks: [vectorstore]
---

Embeddable Python vector store. `chromadb.PersistentClient` is one line —
no server, no Docker, no migration. Writes a SQLite3 DB plus parquet shards
under one directory. In the series, Chroma is the go-to vector store for
local devbox and DGX demos; pgvector is used when cloud deployment is required.

## Usage across the series

### Week 6 — Vector Stores (2026-05-01)

Used in four of the five vectorstore projects:

**`rag-chroma-flyte/`** — Chroma as a Flyte artifact. The `embed_and_index`
task writes a `PersistentClient` directory and Flyte snapshots the whole thing
as a `flyte.io.Dir`. The chat app mounts it back at runtime via
`RunOutput(type="directory", task_name=…)`. This is the key pattern: Chroma's
persist dir is the pipeline's output artifact, not an external service.

**`agent-memory-chroma/`** — Chroma as a read+write memory store. Each turn
retrieves top-k memories by cosine similarity, then writes back new atomic
facts. The persist dir lives at `/tmp/agent_memory_chroma` inside the pod;
it's snapshotted to HuggingFace Hub on shutdown and restored on startup to
survive Knative pod evictions.

**`rag-umap-visualizer/`** — Same Chroma index as `rag-chroma-flyte/` (reuses
the `RunOutput` artifact). Adds a UMAP projection: `@on_startup` loads all
embeddings from Chroma, fits a 2D UMAP reducer, and caches the projection.
Per-query: the query vector is transformed through the same fitted reducer and
plotted as a gold star.

**`screen-context-harness/`** — Chroma stores consolidated activity outlines,
not documents. Outlines are queried by semantic similarity to find relevant
prior context ("what was I working on this morning?"). Distance threshold
`RAG_DISTANCE_THRESHOLD=0.55` controls retrieval sensitivity. Chroma DB lives
at `chroma_db/` in the project directory.
