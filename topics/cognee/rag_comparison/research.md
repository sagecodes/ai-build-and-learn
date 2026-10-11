# RAG Comparison — Research & Design Decisions

---

## Project Summary

**Project:** `rag_comparison`
**Location:** `topics/cognee/rag_comparison/`

### What It Does
Single Gradio app on GCP. One question fires all 3 RAG backends in parallel. Each panel shows
retrieved context + generated answer. A fourth panel has Claude compare all 3 and name a winner.

### Storage per Backend

| Backend   | Storage                     | Status                            |
|-----------|-----------------------------|-----------------------------------|
| Vector RAG | Supabase pgvector          | Existing, cloud-hosted            |
| Graph RAG  | Neo4j AuraDB Free          | Existing, needs resume before demo |
| Cognee     | LanceDB + SQLite + NetworkX | Local files on GCP VM             |

Neo4j Professional trial deletes June 1 — let it go, Free instance is sufficient.

### Shared Infrastructure
- **Embeddings:** fastembed singleton (`BAAI/bge-small-en-v1.5`, 384 dims) — loaded once, shared
  across all 3 backends. All backends use the same model so comparisons are fair.
- **Claude:** shared `generate_answer()` in `backends/shared/claude.py` — same prompt, same model
  across all 3. Retrieval is the only variable.
- **No Flyte/Union** anywhere in the query path. Ingest is a one-time script per backend.

### Repo Structure
```
topics/cognee/rag_comparison/
  data/                        ← 15 Everstorm PDFs
  backends/
    shared/
      embeddings.py            ← fastembed singleton
      claude.py                ← generate_answer()
    vector.py                  ← pgvector query
    graph.py                   ← Neo4j query
    cognee_backend.py          ← Cognee + LanceDB query
  ingest/
    vector_ingest.py
    graph_ingest.py
    cognee_ingest.py
  app.py                       ← Gradio UI
  config.py                    ← env vars, constants
  .env.example
  requirements.txt
  Dockerfile
  README.md
```

### App Layout
```
┌──────────────────────────────────────────────┐
│  [Question input]              [Ask button]  │
└──────────────────────────────────────────────┘
┌──────────────┬───────────────┬───────────────┐
│  Vector RAG  │   Graph RAG   │    Cognee     │
│  Retrieved:  │  Retrieved:   │  Retrieved:   │
│  [accordion] │  [accordion]  │  [accordion]  │
│  Answer:     │  Answer:      │  Answer:      │
└──────────────┴───────────────┴───────────────┘
┌──────────────────────────────────────────────┐
│  Comparison Summary (Claude)                 │
└──────────────────────────────────────────────┘
```

### Build Sequence

| Phase | What                                        |
|-------|---------------------------------------------|
| 1     | Scaffolding, config, `backends/shared/`     |
| 2     | 3 backend query modules                     |
| 3     | 3 ingest scripts + re-ingest all backends   |
| 4     | Gradio app + comparison summary             |
| 5     | GCP e2-medium deployment + Docker           |
| 6     | Sample questions, README, cleanup           |

### GCP
- **VM:** e2-medium (4GB RAM — headroom for fastembed + 3 active DB connections)
- **Cost:** ~$26/month against available GCP credit
- Supabase and Neo4j AuraDB are external — VM only runs the app + Cognee local storage

---

## What This Project Is

A side-by-side comparison of three RAG (Retrieval-Augmented Generation) approaches over the same
document set. One question fires all three backends in parallel. Each backend returns what it
retrieved and the answer it generated. Claude then evaluates all three and names a winner.

The goal is to show that retrieval strategy — not the LLM — determines answer quality.

---

## The Three Approaches

### Vector RAG
Embeds documents into a vector store. At query time, embeds the question and retrieves the most
semantically similar chunks via cosine similarity.

- **Strength**: Fast, simple, works well for "find me information about X" queries
- **Weakness**: Retrieves similar-sounding text, not necessarily connected facts. Misses
  relationships between entities.
- **Storage**: Supabase pgvector (cloud-hosted)

### Graph RAG
Extracts entities and relationships from documents into a graph database. At query time, finds
the nearest entity to the question and traverses connected nodes 1–2 hops out.

- **Strength**: Understands connections — "what does policy X say about scenario Y that also
  involves Z?" Graph traversal surfaces multi-hop context that vector search misses.
- **Weakness**: More complex to build and maintain. Quality depends on entity extraction quality.
  Struggles with purely semantic queries that have no clear entity anchor.
- **Storage**: Neo4j AuraDB Free (cloud-hosted)

### Cognee
Automatically builds both a vector index and a knowledge graph from documents using its own
ingestion pipeline (`cognee.add()` + `cognee.cognify()`). Query via `SearchType.CHUNKS` returns
vector-retrieved chunks backed by the knowledge graph.

- **Strength**: Hybrid retrieval without manual graph construction. Significantly less code than
  building vector + graph pipelines by hand.
- **Weakness**: Less control — the graph structure and entity extraction are opaque. Harder to
  debug when retrieval quality is poor.
- **Storage**: LanceDB (vector, local files) + SQLite (relational, local) + NetworkX (graph,
  in-memory). All local on the GCP VM. See: https://www.lancedb.com/blog/case-study-cognee
- **Embedding**: fastembed via `BAAI/bge-small-en-v1.5`

---

## Key Design Decisions

### Same embedding model across all three backends
All three use `fastembed` with `BAAI/bge-small-en-v1.5` (384 dimensions). This makes the
comparison fair — retrieval quality differences are due to strategy, not embedding model choice.
A shared singleton loads the model once at startup; all backends reference it.

### No Flyte / Union in the query path
Previous projects routed every query through Union Flyte tasks to access cloud compute for
embedding generation. This added seconds of task dispatch overhead to every user interaction
and is not a production RAG pattern.

Flyte is appropriate for batch ingestion (tracked, retryable, observable). It is not appropriate
for real-time retrieval. In this project:
- **Ingestion**: one-time Python scripts (run manually, not orchestrated)
- **Query**: direct function calls in the same process as the Gradio app

### Cognee storage is local, others are cloud
Each backend's storage reflects its natural deployment model:

| Backend   | Storage              | Why                                           |
|-----------|----------------------|-----------------------------------------------|
| Vector    | Supabase pgvector    | Already provisioned; cloud pgvector is standard |
| Graph     | Neo4j AuraDB Free    | Already provisioned; AuraDB Free is sufficient |
| Cognee    | LanceDB + SQLite     | Cognee defaults; zero external deps; demonstrates Cognee's self-contained nature |

### Shared Claude client
All three backends call the same `generate_answer(question, context)` function with the same
system prompt and model (`claude-sonnet-4-6`). The LLM is a constant — only retrieval varies.

---

## Infrastructure

### GCP VM
- **Type**: e2-medium (2 vCPU, 4GB RAM)
- **Why e2-medium**: fastembed loads a ~100MB ONNX model into memory. 2GB on e2-small leaves
  insufficient headroom when also holding 3 active DB connections and the Gradio process.
- **Cost**: ~$26/month against available GCP credit
- **Hosts**: Gradio app, fastembed model, Cognee local storage (LanceDB + SQLite)
- **Connects to**: Supabase (external), Neo4j AuraDB (external)

### Neo4j AuraDB Free
- Instance ID: c30548a7
- Pauses after ~3 days of inactivity — resume before any demo
- Professional trial (everstorm-graphrag) expired; let it delete on June 1 2026
- Free tier limits (200K nodes, 400K relationships) far exceed project needs (~1,500 nodes,
  ~5,000 relationships for 15 documents)

### Supabase
- Existing instance from vector_rag_chatbot project
- pgvector extension already enabled
- Re-ingest required (new embedding model: fastembed vs. prior model)

---

## Document Set

15 Everstorm Outfitters policy PDFs — a fictional outdoor gear company. Domain is intentionally
neutral so the audience focuses on retrieval differences, not content familiarity.

Documents cover: account security, B2B orders, warranty, gift cards, shipping, loyalty program,
member benefits, order cancellation, partner programs, privacy, product categories, promotions,
store locations, sustainability, accessibility services.

All 15 PDFs are in `data/` and must be re-ingested into all three backends using the shared
fastembed model.

---

## Comparison Summary

After all three backends return, a fourth Claude call receives:
- The original question
- All three retrieved contexts
- All three generated answers

Claude evaluates which backend retrieved the most complete and relevant context, where each fell
short, and which answer best served the question. This surfaces the retrieval differences without
the presenter having to assert them manually.

---

## What This Demonstrates

1. Understanding of three distinct retrieval strategies at an implementation level
2. Architectural judgment — knowing when orchestration adds value (batch ingest) vs. when it adds
   latency without benefit (real-time query)
3. Ability to build systems that evaluate their own outputs
4. Clean, learnable code structure — each backend is readable in isolation

---

## Phase 7 Stretch Goal — Generation Model Toggle

After the core comparison is working, add a toggle to swap the generation model:
**Claude Sonnet** | **Llama 3 via Groq**

**Why Groq instead of a local model:**
Running Ollama on a CPU-only e2-medium would be too slow for a live demo. Groq hosts open source
models (Llama 3, Mixtral) with extremely fast inference and a free API tier. Same architectural
point — generation is a pluggable component — without the compute problem.

**What this demonstrates:**
Retrieval and generation are independent concerns. The same retrieval context fed to a different
LLM produces a different answer. This reinforces the project's core thesis: retrieval strategy
is the primary quality driver, but the generation model is also a variable an agent builder
controls.

**Implementation:**
- Add `GROQ_API_KEY` to config and `.env.example`
- Add a `gr.Radio` toggle in the Gradio UI: "Generation Model"
- Pass the selected model through to `generate_answer()` — a second code path alongside the
  Anthropic client
- Keep the comparison summary on Claude regardless of toggle state (consistent evaluator)

**Note:** Do not add this until Phases 1–6 are complete and the demo is stable.
