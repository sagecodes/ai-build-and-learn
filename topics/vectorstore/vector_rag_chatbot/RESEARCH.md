# Vector RAG Chatbot — Research & Design

## The Event

This project was built for the AI Build & Learn vector storage stream. The theme: build a RAG (Retrieval Augmented Generation) application that uses a real vector database to answer questions grounded in a private knowledge base.

The cohort reference project used FAISS and Ollama running entirely locally. This version takes it further — production-grade vector DB, cloud compute on Union.ai, and Claude as the LLM.

---

## What is RAG?

RAG answers questions by retrieving relevant documents first, then generating a response grounded in that evidence — not relying on the LLM's training data alone.

```
User question
      ↓
  Embed question → vector
      ↓
  Search vector DB → top-k most similar chunks
      ↓
  Build prompt: [system] + [retrieved chunks] + [question]
      ↓
  LLM generates answer citing the chunks
```

The key advantage over pure LLM: the answer is anchored to your documents. The model cannot hallucinate outside the provided context because the prompt instructs it to only use what's given.

---

## Research: Vector Store Options

Evaluated five options against three criteria: managed hosting, pgvector compatibility, and free tier.

| Option | Hosting | Free tier | Notes |
|--------|---------|-----------|-------|
| FAISS | Self-hosted | ✅ | Cohort baseline. In-memory, no persistence, no cloud deployment |
| Chroma | Self-hosted | ✅ | Good local dev story, but adds another service to manage in production |
| Qdrant | Managed cloud | ✅ 1GB | Purpose-built vector DB, strong client library — console had outage during research |
| Pinecone | Managed cloud | ✅ limited | Production-grade but closed ecosystem, proprietary query API |
| Supabase pgvector | Managed Postgres | ✅ 500MB | **Selected** — open standard SQL, no proprietary API, integrates with any Postgres tooling |

**Decision: Supabase pgvector**

- `pgvector` is a Postgres extension — standard SQL with `<=>` cosine distance operator
- Supabase provides managed Postgres with `vector` extension pre-available
- No proprietary client library — just `psycopg3` and raw SQL
- Free tier with 500MB is sufficient for 16 PDFs × ~50 chunks × 384D vectors
- If we outgrow Supabase, the same SQL runs on any Postgres instance (GCP Cloud SQL, self-hosted, etc.)

---

## Research: Embedding Model

Evaluated three models for the 384D embedding space:

| Model | Dims | Size | Quality |
|-------|------|------|---------|
| `all-MiniLM-L6-v2` | 384 | 80MB | Fast, good general purpose |
| `thenlper/gte-small` | 384 | 67MB | **Selected** — MTEB benchmark top performer at 384D |
| `BAAI/bge-small-en-v1.5` | 384 | 67MB | Comparable to gte-small, slightly lower MTEB scores |

**Decision: `thenlper/gte-small`**

- Highest MTEB retrieval scores in the 384D tier
- Small enough to run on a Union cluster CPU task (no GPU needed)
- 384D keeps pgvector index size small for the free tier

---

## Research: Chunking Strategy

Customer support PDFs are short, structured documents (numbered sections, bullet points, tables). Evaluated two approaches:

| Strategy | Verdict |
|----------|---------|
| Fixed character splits | Fast but cuts mid-sentence, degrades retrieval |
| `RecursiveCharacterTextSplitter` | **Selected** — respects paragraph/sentence boundaries |

**Parameters chosen: chunk_size=300, overlap=30**

- 300 chars ≈ 2–4 sentences — enough context for Claude to answer from, small enough for precise retrieval
- 30-char overlap preserves boundary context without redundancy
- Configurable from the Gradio UI — users can experiment live

---

## Knowledge Base: Everstorm Outfitters

The cohort dataset had 4 PDFs. To make retrieval meaningful and demonstrate cross-document reasoning, the knowledge base was expanded to 16 documents.

**4 original (from cohort):**
- Payment, Refund and Security
- Product Sizing and Care Guide
- Return and Exchange Policy
- Shipping and Delivery Policy

**12 generated (Claude API + ReportLab):**
- Loyalty Program
- Gift Cards
- Order Cancellation Policy
- International Shipping Guide
- Extended Warranty
- Privacy and Data Policy
- Store Locations and Hours
- Promo and Discount Policy
- Account and Security
- Sustainability and Recycling
- B2B Corporate Orders
- Accessibility Services

Each generated PDF was prompted to match the style of the originals: numbered sections, bullet points, pipe tables. This ensures consistent chunking behavior across all 16 documents.

---

## Architecture Decisions

### Orchestration: Union.ai / Flyte

All heavy compute (PDF extraction, embedding, vector indexing, retrieval, generation) runs as Flyte tasks on the Union cluster. The Gradio UI dispatches tasks locally and waits for results.

**Why Flyte here:**
- Every task is visible as a node in the Union UI — good for the demo audience
- `cache="auto"` on load/chunk and retrieve tasks means re-ingesting the same PDF or asking the same question returns instantly from cache
- Parallel fan-out for PDF ingest via `asyncio.gather` — all PDFs embed concurrently, one task per PDF visible in Union

### Two-Pipeline Design

**Ingest pipeline** — runs once (or when docs change):
```
ingest_pipeline
  ├── load_and_chunk_task (pdf_1)  ┐
  ├── load_and_chunk_task (pdf_2)  ├─ parallel, cached per PDF
  ├── load_and_chunk_task (...)    │
  └── load_and_chunk_task (pdf_n) ┘
           ↓  merge all chunks
  embed_and_index_task  →  Supabase pgvector
```

**Query pipeline** — runs on every question:
```
query_pipeline
  ├── retrieve_task(query)         → top-k from pgvector  (cached per query)
  └── generate_task(query, chunks) → Claude RAG answer
```

Splitting into two pipelines keeps ingest and query concerns separate. Re-ingesting doesn't require touching the query path.

### Idempotent Ingest

`embed_and_index_task` deletes existing rows for `collection_name` before re-inserting. This means running ingest twice on the same collection is safe — no duplicate vectors.

### pgvector Table Schema

```sql
CREATE TABLE IF NOT EXISTS document_chunks (
    id              BIGSERIAL PRIMARY KEY,
    collection_name TEXT NOT NULL,
    source_doc      TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       vector(384) NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

HNSW index gives approximate nearest-neighbor search in O(log n) — fast enough for thousands of chunks without exact scan overhead.

**Retrieval query:**
```sql
SELECT source_doc, chunk_text, 1 - (embedding <=> %s::vector) AS score
FROM document_chunks
WHERE collection_name = %s
ORDER BY embedding <=> %s::vector
LIMIT %s
```

`<=>` is the pgvector cosine distance operator. `1 - distance` converts to a similarity score (1.0 = identical).

### LLM: Claude claude-sonnet-4-6

- Grounding instruction: answer using ONLY the provided context
- Fallback: "I don't have that information in our support docs."
- 600 token max output — sufficient for a support answer without over-generation
- Source citation requested in the system prompt

### Multi-Collection Support

Collection name is a user-configurable field in the Gradio UI (default: `everstorm_docs`). Multiple collections can coexist in the same `document_chunks` table — filtered by `WHERE collection_name = %s`. Re-testing with different chunk sizes means just changing the collection name and re-ingesting.

### Secrets

No credentials in code or `.gitignore`-bypassed files. Secrets registered in Union cluster and injected at runtime:

| Secret key | Injected as |
|-----------|-------------|
| `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY` env var |
| `DATABASE_URL` | `DATABASE_URL` env var |

```bash
flyte create secret ANTHROPIC_API_KEY --project dellenbaugh --domain development
flyte create secret DATABASE_URL --project dellenbaugh --domain development
```

---

## Final Architecture

```
topics/vectorstore/vector_rag_chatbot/
├── data/                    ← 16 Everstorm Outfitters PDFs
│   ├── Everstorm_Payment_refund_and_security.pdf
│   ├── Everstorm_Loyalty_Program.pdf
│   └── ...
├── app.py                   ← Gradio UI (ingest tab + chat tab)
├── workflows.py             ← 6 Flyte tasks, 2 pipelines
├── config.py                ← TaskEnvironment, secrets, shared constants
├── styles.css               ← all styling (no inline styles)
├── generate_docs.py         ← one-time PDF generator (already run)
├── requirements.txt
├── .env                     ← local credentials (gitignored)
└── RESEARCH.md              ← this file
```

---

## How to Run

### Prerequisites

1. Add to `.env`:
   ```
   ANTHROPIC_API_KEY=...
   DATABASE_URL=postgresql://postgres:<password>@db.<project>.supabase.co:5432/postgres
   FLYTE_BACKEND=union
   ```

2. Enable pgvector in Supabase SQL Editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. Install dependencies:
   ```bash
   pip install psycopg[binary] pgvector sentence-transformers PyMuPDF \
               langchain-text-splitters anthropic flyte gradio python-dotenv
   ```

### Run

```bash
python app.py
```

Opens at `http://localhost:7860`.

### Demo Flow

1. **Ingest tab** — select PDFs (all 16 pre-selected), click "Run Ingest on Union"
   - Watch Union UI for parallel `load_and_chunk_task` nodes per PDF
   - `embed_and_index_task` fires after all chunks are merged

2. **Chat tab** — ask any Everstorm support question
   - Each question runs `retrieve_task` → `generate_task` on the cluster
   - Response includes a collapsible accordion of the source chunks with similarity scores

### FLYTE_BACKEND Toggle

| Value | Behavior |
|-------|----------|
| `local` | Tasks run in-process, no Union needed, good for dev |
| `union` | Tasks run on Union cluster, results visible in Union UI |

---

## Supabase pgvector — Quick Reference

### Connection

```python
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(os.environ["DATABASE_URL"])
register_vector(conn)
```

### Insert vectors

```python
cur.executemany(
    "INSERT INTO document_chunks (collection_name, source_doc, chunk_index, chunk_text, embedding) VALUES (%s, %s, %s, %s, %s)",
    [(collection, doc, idx, text, embedding.tolist()) for ...]
)
```

### Query by cosine similarity

```python
cur.execute(
    "SELECT source_doc, chunk_text, 1 - (embedding <=> %s::vector) AS score FROM document_chunks WHERE collection_name = %s ORDER BY embedding <=> %s::vector LIMIT %s",
    (query_vector, collection_name, query_vector, k)
)
```

### Key facts

| | |
|---|---|
| Extension | `CREATE EXTENSION IF NOT EXISTS vector` |
| Python client | `psycopg[binary]>=3.1.0` + `pgvector>=0.3.0` |
| Distance operator | `<=>` (cosine), `<->` (L2), `<#>` (inner product) |
| Index type | HNSW (`vector_cosine_ops`) — approximate, fast |
| Score convention | `1 - distance` → 1.0 = identical, 0.0 = orthogonal |
