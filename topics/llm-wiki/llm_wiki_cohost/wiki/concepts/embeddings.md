---
title: Embeddings
first_seen: topics/vectorstore/
weeks: [vectorstore]
---

Embeddings are fixed-length vectors of floats that encode the semantic meaning
of text (or images, audio, etc.). Texts with similar meanings produce vectors
that are mathematically close in embedding space. This is what makes
"search by meaning" tractable.

```
text / image / audio  →  encoder  →  vector (e.g. 384 floats)
                                           │
                                     vector store
                                     (HNSW index)

query  →  same encoder  →  query vector  →  top-k nearest neighbors
```

**Hard constraint:** the embedding model used at ingest must match the model
used at query time. Different models map to incompatible spaces — the geometry
is meaningless across them. The Chroma collection metadata records which model
built the index; mismatches should warn loudly.

## Similarity and search

**Cosine similarity** measures the angle between two vectors. Near 1.0 = same
meaning; near 0.0 = unrelated. pgvector's `<=>` operator is cosine distance
(1 − similarity), so `ORDER BY embedding <=> query_vec LIMIT k` returns the
top-k most similar chunks.

**HNSW (Hierarchical Navigable Small World)** is the index structure used by
both Chroma and pgvector. It builds a multi-layer graph; at query time it
navigates layer by layer, pruning branches moving away from the query vector.
Result: approximate nearest-neighbor in O(log n), >99% recall, vs. O(n) for
exact scan. At 800 chunks the difference is negligible; at 1M+ chunks it's
the only viable approach.

## How it appeared across the series

### Week 6 — Vector Stores (2026-05-01)

Three embedding models used across the five projects:

| Model | Dims | Size | Used in |
|---|---|---|---|
| `BAAI/bge-small-en-v1.5` | 384 | 67MB | `rag-chroma-flyte/`, `rag-umap-visualizer/` |
| `thenlper/gte-small` | 384 | 67MB | `vector_rag_chatbot/` |
| (Gemma 4 Vision captions → BGE) | 384 | — | `screen-context-harness/` |

`gte-small` was selected over `bge-small` for the pgvector demo on MTEB
retrieval benchmark scores at the 384D tier. Both are 67MB and run on CPU —
no GPU needed for embedding at this scale.

**UMAP visualization (`rag-umap-visualizer/`)** made embedding space tangible
on stream: every chunk plotted as a gray dot, top-k retrieved chunks as colored
markers, the query as a gold star. When a query lands far from the lit-up
neighbors, distances are large and the answer shouldn't be trusted. Toggling
retrieval off shows Gemma answering from parametric memory only — direct contrast
with what embeddings add.

**Chunking interacts with embedding quality.** A chunk_size=300 caused retrieval
failures for the Everstorm support docs: section headers were split from their
content, so the semantically meaningful "Password Requirements + its criteria"
was split across two chunks, each retrievable independently but neither
self-contained. Increasing to chunk_size=600 fixed retrieval precision.
`RecursiveCharacterTextSplitter` (paragraphs → sentences → words) keeps chunks
semantically coherent better than fixed-character splits.

## Open questions

- At what corpus size does 384D embedding become a retrieval bottleneck vs.
  1536D (OpenAI `text-embedding-3-small`)?
- How does the series handle embedding model versioning — if the model is
  updated, does the whole index need to be rebuilt?
