# RAG Comparison

Side-by-side comparison of three retrieval strategies over the same document set. One question,
three answers, one verdict — all in a single Gradio app.

---

## What This Demonstrates

Your AI agent is only as good as what it can retrieve. This project shows three different ways to
give an agent memory over a document corpus, and lets you see — in real time — how each one
performs on the same question.

The core thesis: **retrieval strategy determines answer quality, not the LLM.** All three backends
use the same embedding model and the same Claude call. The only variable is how each system finds
relevant context.

---

## The Three Approaches

### Vector RAG
Embeds documents into pgvector and retrieves the most semantically similar chunks at query time.

- Fast and simple to build
- Works well for "tell me about X" questions
- Misses relationships between entities — retrieves similar-sounding text, not connected facts

### Graph RAG
Extracts entities and relationships into Neo4j. At query time, finds the nearest entity and
traverses the graph 1–2 hops to surface connected context.

- Understands connections — strong on multi-hop questions like "what policy applies when X and Y
  both apply?"
- Weaker on purely semantic queries with no clear entity anchor
- More complex to build and maintain

### Cognee
Automatically builds both a vector index and a knowledge graph from your documents via
`cognee.add()` + `cognee.cognify()`. Query via vector similarity backed by the knowledge graph.

- Hybrid retrieval without manual graph construction
- Significantly less code than building vector + graph pipelines by hand
- Less control — graph structure and entity extraction are opaque
- Uses LanceDB for local vector storage (see:
  [Cognee + LanceDB case study](https://www.lancedb.com/blog/case-study-cognee))

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │         Gradio App (app.py)      │
                    └──────────┬──────────┬────────────┘
                               │ asyncio.gather()
              ┌────────────────┼────────────────┐
              ▼                ▼                ▼
        vector.py          graph.py     cognee_backend.py
        pgvector           Neo4j         LanceDB + SQLite
        Supabase         AuraDB Free     (local on VM)
              │                │                │
              └────────────────┴────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  backends/shared/   │
                    │  embeddings.py      │  ← fastembed singleton
                    │  claude.py          │  ← shared generate_answer()
                    └─────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  Comparison Summary  │
                    │  (Claude evaluates  │
                    │   all 3 answers)    │
                    └─────────────────────┘
```

**Shared across all three backends:**
- Embedding model: `fastembed` with `BAAI/bge-small-en-v1.5` (384 dims) — loaded once at startup,
  shared via singleton. Same model ensures comparison fairness.
- Generation: `claude-sonnet-4-6` with identical system prompt. Retrieval is the only variable.

**No workflow orchestration in the query path.** All three backends are direct function calls in
the same process. This is intentional — routing real-time queries through an orchestrator
(Flyte, Airflow, Prefect) adds seconds of dispatch overhead for no benefit.

Ingestion is a different story. In a production system, ingestion would be orchestrated for
retries, observability, and scheduling. Here it is a one-time script per backend — simple enough
that orchestration would add complexity without value.

---

## Infrastructure

| Component       | Where                              |
|-----------------|------------------------------------|
| App + fastembed | GCP e2-medium (4GB RAM)            |
| Cognee storage  | Local files on GCP VM (LanceDB + SQLite) |
| Vector store    | Supabase pgvector (cloud)          |
| Graph store     | Neo4j AuraDB Free (cloud)          |
| Generation      | Anthropic API (claude-sonnet-4-6)  |

---

## Getting Started

> Setup instructions will be added after deployment (Phase 5).

**Prerequisites:**
- Python 3.11+
- Supabase project with pgvector enabled
- Neo4j AuraDB Free instance
- Anthropic API key

```bash
git clone https://github.com/johndell-914/ai-build-and-learn.git
cd topics/cognee/rag_comparison
pip install -r requirements.txt
cp .env.example .env
# fill in .env with your credentials
```

---

## Ingesting Documents

Run each ingest script once before launching the app. Each script reads the 15 Everstorm
Outfitters policy PDFs from `data/` and populates its respective backend.

```bash
python ingest/vector_ingest.py     # Supabase pgvector
python ingest/graph_ingest.py      # Neo4j AuraDB
python -m asyncio ingest/cognee_ingest.py   # Cognee (LanceDB)
```

---

## Running the App

> Launch instructions will be added after deployment (Phase 5).

```bash
python app.py
# Open http://localhost:7860
```

---

## Sample Questions

> Sample questions and expected output will be added in Phase 6.

---

## Document Set

15 fictional Everstorm Outfitters policy PDFs covering: account security, B2B orders, warranty,
gift cards, shipping, loyalty program, member benefits, order cancellation, partner programs,
privacy, product categories, promotions, store locations, sustainability, and accessibility.

The domain is intentionally neutral — simple enough that retrieval differences are obvious, not
masked by content complexity.
