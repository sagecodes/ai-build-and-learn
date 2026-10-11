---
title: RAG (Retrieval-Augmented Generation)
first_seen: topics/vectorstore/
weeks: [vectorstore, graphs-neo4j, llm-wiki]
---

RAG answers questions by retrieving relevant documents first, then generating a
response grounded in that evidence — not relying on the LLM's training data alone.
It gives LLMs four capabilities they otherwise lack: fresh knowledge, private
knowledge, cheap updates, and grounded (citable) answers.

The pipeline has two phases:

**Indexing (offline):** docs → chunk → embed → store in a vector index.

**Retrieval + generation (online):** embed query → top-k nearest chunks →
inject as context → LLM answers citing `[#N]`.

## How it appeared across the series

### Week 6 — Vector Stores (2026-05-01)

RAG enters the wiki properly with three distinct implementations:

**Classic RAG (`rag-chroma-flyte/`)** — The baseline pattern. HuggingFace
`rag-mini-wikipedia` corpus chunked with `RecursiveCharacterTextSplitter`
(chunk_size=1200, overlap=100), embedded with `bge-small-en-v1.5`, stored in
Chroma as a `flyte.io.Dir` artifact. Gradio chat mounts the artifact via
`RunOutput` and retrieves at query time. Chunk retrieval panel shows top-k
with cosine similarity scores.

**Production RAG (`vector_rag_chatbot/`)** — Same pattern at production scale.
16 Everstorm Outfitters PDFs, chunked at 600 chars / 60 overlap (tighter for
structured support docs), embedded with `gte-small`, stored in pgvector on
Supabase. Claude Sonnet generates grounded answers with a hard grounding
instruction: answer using ONLY the provided context; if not in the docs, say so.
Ingest pipeline fans out per-PDF in parallel on Union; `embed_and_index_task`
deletes and re-inserts on re-ingest (idempotent).

**Multimodal RAG (`screen-context-harness/`)** — Documents are screenshots, not
text. Gemma 4 Vision (via Ollama) captions each frame; captions are embedded and
stored in ChromaDB. Query retrieves semantically relevant prior activity outlines.
Shows that the RAG shape (encode → store → retrieve → generate) is modality-agnostic.

### RAG variants covered this week

The `topics/vectorstore/README.md` is the most thorough treatment of RAG variants
in the series to date:

| Variant | Key idea |
|---|---|
| Naive / classic | Single embed → top-k → stuff into prompt |
| Hybrid retrieval | Dense (vector) + sparse (BM25); cross-encoder reranker |
| Query rewriting | LLM generates N query variants; union of results |
| HyDE | Embed a hypothetical answer, not the question |
| RAPTOR | Hierarchical clustering + recursive summaries |
| Graph RAG | Knowledge graph alongside vector store; multi-hop queries |
| Agentic RAG | Retrieval as a tool the LLM calls on demand |
| Self-RAG | Model decides per step whether retrieval is needed |
| CRAG | Scores retrieved passages; rewrites query if score is poor |

Graph RAG will be extended in the graphs-neo4j week (week 7).

### Week 7 — Graph Data with Neo4j (2026-05-08)

Graph RAG is the most significant extension of the RAG concept in the series.
Two implementations:

**`graphrag-neo4j-flyte/`** (academic papers) — Three retrieval modes on the
same Neo4j instance: pure vector baseline, vector + 1-hop graph expand
(seed papers → walk `CITES`/`AUTHORED_BY`/`IN_CATEGORY` edges), and hybrid
Reciprocal Rank Fusion (vector top-k fused with most-cited papers in the same
category). RRF uses `1/(K+rank)` scoring — ranks are commensurable across lists
even when raw scores aren't (cosine similarity vs. citation count), so no
normalization needed. Papers in both lists win the most weight.

**`graph_rag_chatbot/`** (Everstorm support docs) — Same documents as week 6's
`vector_rag_chatbot`, different retrieval architecture. Three modes routed by a
Claude classifier: Hybrid (vector + MENTIONS traversal to entities), Entity
(RELATED neighborhood traversal), Community (cosine similarity to pre-computed
community summaries). The retrieval mode badge in the UI makes routing visible.

**The key insight this week:** Graph RAG and vector RAG answer different question
types. Vector gets you to the right neighborhood; graph tells you how everything
in that neighborhood connects. Community summaries zoom out to the full picture
for thematic questions. The best production systems use all three.

**What pgvector cannot do:** vector search and relationship traversal in one
query. Neo4j's single Cypher statement does both:
```cypher
CALL db.index.vector.queryNodes('chunk-embeddings', 3, $query_embedding)
YIELD node AS chunk, score WHERE score >= 0.75
MATCH (chunk)-[:MENTIONS]->(entity:Entity)
OPTIONAL MATCH (entity)-[:RELATED]->(other:Entity)
RETURN chunk.text, entity.name, collect(other.name) AS related
```

### Week 8 — LLM Wiki (2026-05-15)

The LLM Wiki pattern is explicitly framed as the inversion of RAG. From
the RESEARCH.md:

> "Traditional RAG re-discovers knowledge on every query. The LLM Wiki
> pattern inverts this: the LLM reads sources once and incrementally builds
> a structured wiki that synthesizes and accumulates insight."

The key distinction: in RAG, sources are the persistent artifact and the
LLM output is ephemeral (per-query). In the LLM Wiki, the LLM output is
the persistent artifact (the wiki) and sources become input material —
read once at ingest time, not on every query.

**Practical consequence:** a wiki query reads `index.md` plus 2–4 pages
and synthesizes from structured, pre-accumulated knowledge. A RAG query
embeds, searches, retrieves, and generates from raw source chunks. At
~50–100 wiki pages, the wiki is faster and more coherent; at web-scale
corpora, RAG is the only option.

**Compounding vs. re-retrieval.** Valuable query answers get filed back as
new wiki pages. Explorations compound — each session adds to the wiki's
total knowledge. RAG queries evaporate; wiki queries accumulate. This
is the property Karpathy's pattern names most explicitly.

See [LLM Wiki Pattern](../concepts/llm-wiki-pattern.md) for the full treatment.

## Open questions

- How does the series handle RAG evaluation? (Ragas week, 2026-05-29, is the
  anticipated answer)
- What's the retrieval quality difference between Chroma and pgvector at this
  corpus scale?
- When does agentic RAG outperform classic RAG, and at what latency cost?
