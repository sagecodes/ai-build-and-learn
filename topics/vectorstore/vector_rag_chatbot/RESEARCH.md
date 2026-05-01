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

## Why Chunking is Necessary

A PDF is a wall of text — anywhere from a few hundred to tens of thousands of characters. You cannot pass an entire document into an LLM prompt for two reasons:

**1. Context window limits**
LLMs have a maximum input size (measured in tokens). A single 10-page PDF can exceed that limit on its own. Sixteen PDFs definitely would. Even models with large context windows (100k+ tokens) become unreliable when the input is too long — they lose focus on the relevant parts buried in the middle.

**2. Retrieval precision**
Even if you could fit everything in, you wouldn't want to. If a user asks "what is the return window for sale items?", sending all 16 documents to Claude creates noise — the model has to mentally filter through 15 documents of irrelevant text to find the one sentence that answers the question. Smaller, focused chunks mean the retrieved context is dense with relevant signal.

**How chunking works here:**

```
Full PDF text (e.g. 8,000 chars)
      ↓  RecursiveCharacterTextSplitter(chunk_size=600, overlap=60)
[chunk_0: chars 0–600]
[chunk_1: chars 540–1140]   ← 60-char overlap preserves boundary context
[chunk_2: chars 1080–1680]
...
[chunk_n: last segment]
```

`RecursiveCharacterTextSplitter` tries to split on paragraph breaks, then sentence breaks, then word breaks — in that order. This keeps each chunk semantically coherent rather than cutting mid-sentence.

The 30-character overlap means that if an important sentence straddles a boundary, it appears in both adjacent chunks. Neither chunk loses the context needed to make it retrievable.

---

## Why Embeddings are Needed

Once text is chunked, you need a way to answer the question: *"which of these 800 chunks is most relevant to this user's question?"*

You cannot use keyword search reliably. A user asking *"how long do I have to send something back?"* will not match a chunk containing *"items must be returned within 30 days"* — the words are completely different, but the meaning is the same.

**Embeddings solve this by converting meaning into math.**

An embedding model reads a piece of text and outputs a vector — a list of numbers (384 numbers in our case) that encodes the semantic meaning of that text. Texts with similar meanings produce vectors that are mathematically close together in 384-dimensional space.

```
"how long do I have to send something back?"
      ↓  gte-small embedding model
[0.021, -0.143, 0.887, ..., 0.034]   ← 384 numbers representing the meaning

"items must be returned within 30 days"
      ↓  gte-small embedding model
[0.019, -0.138, 0.901, ..., 0.041]   ← very close to the question's vector
```

**Cosine similarity** measures the angle between two vectors. Vectors pointing in nearly the same direction have a similarity near 1.0 — they mean roughly the same thing. Unrelated texts produce vectors pointing in different directions, similarity near 0.0.

**The retriever pipeline:**

```
INGEST (run once)
  For each chunk:
    text → embedding model → 384D vector → store in pgvector

QUERY (run per question)
  User question → embedding model → 384D query vector
  Search pgvector: find the k chunks whose vectors are closest to the query vector
  Return those chunks as context to Claude
```

This is why the same embedding model must be used for both ingest and query. The vectors only "speak the same language" if they were produced by the same model. Switching models after ingesting requires re-embedding and re-indexing all chunks from scratch.

---

## Why a Vector Store is Needed

You have hundreds or thousands of chunk vectors. At query time you need to find the top-k most similar to the query vector — fast. A vector store is a database purpose-built for this problem.

**Why not just use a regular database?**

A SQL `LIKE` or full-text search cannot find semantic matches — it only matches exact words or substrings. You could store vectors in a standard Postgres `float[]` column, but finding the closest vectors would require computing the distance to *every row* in the table (a full table scan) on every query. With 800 chunks that's tolerable; with 1 million chunks it becomes unusably slow.

**What a vector store adds:**

| Feature | Benefit |
|---------|---------|
| Vector index (HNSW) | Approximate nearest-neighbor in O(log n) — no full scan |
| Distance operators | Native `<=>` cosine, `<->` L2, `<#>` inner product |
| Metadata filtering | Filter by `collection_name`, `source_doc`, etc. before or after vector search |
| Persistence | Vectors survive restarts, shareable across machines |

**How HNSW works (simplified):**

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph where each node is a vector. At query time it navigates the graph layer by layer, pruning branches that are moving away from the query vector. This gives approximate (not exact) nearest neighbors, but the approximation is extremely accurate (>99% recall) at a fraction of the cost of an exact scan.

```
Exact scan (no index):  compare query to all 800 vectors  →  O(n)
HNSW index:             navigate graph, skip most vectors  →  O(log n)
```

For this project — 16 PDFs, ~800 chunks — the difference is negligible. But the index is already in place, so if the knowledge base grows to 10,000+ chunks the query latency stays flat.

**pgvector specifically** brings all of this inside standard Postgres. The same `document_chunks` table supports both vector similarity search and standard SQL filters on `collection_name` — no second service, no separate API, no data sync between systems.

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

**Parameters chosen: chunk_size=600, overlap=60**

- 600 chars keeps a section header together with its content — critical for structured support docs where the header carries the semantic meaning (e.g. "Password Requirements" + its criteria in the same chunk)
- 60-char overlap preserves boundary context without redundancy
- Initial testing with chunk_size=300 caused retrieval failures: the B2B Corporate Orders doc outranked Account_and_Security for password questions because "account" appeared frequently in B2B chunks while the password section header was split from its content
- Configurable from the Gradio UI — users can experiment live and re-ingest

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
- `cache="auto"` on `load_and_chunk_task` means re-ingesting an unchanged PDF is a free cache hit — no re-chunking needed
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
  ├── retrieve_task(query)         → top-k from pgvector
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
| `PG_URL` | `PG_URL` env var |

```bash
flyte create secret ANTHROPIC_API_KEY --project dellenbaugh --domain development
flyte create secret PG_URL --project dellenbaugh --domain development
```

Note: `DATABASE_URL` was the original secret name but `flyte create secret` does not update existing keys — it silently does nothing. A new key name (`PG_URL`) was used to bypass this. See Deployment Issues below.

---

## Deploying the Gradio App to Union

The Gradio UI can be deployed as a persistent web app on the Union cluster — no local machine needed. This is separate from the task image (which runs the ML workloads). The app image only needs to boot Gradio and call `flyte.run()`.

### How it works

Union exposes `flyte.app.AppEnvironment` — a lightweight container that runs a long-lived server process on the cluster and gives it a public URL.

```
flyte deploy app.py serving_env
      ↓
Union bundles app.py + config.py + workflows.py → deploys pod → returns persistent URL
      ↓
https://wandering-resonance-f163a.apps.tryv2.hosted.unionai.cloud
```

### Two separate images

| Image | Contains | Used for |
|-------|----------|----------|
| Task image (`config.py env`) | psycopg, pgvector, sentence-transformers, anthropic, PyMuPDF, langchain | Flyte tasks on cluster |
| App image (`serving_env` in `app.py`) | gradio, flyte, python-dotenv | Deployed Gradio server |

The app image is intentionally minimal — it doesn't need ML dependencies because the heavy work is delegated to task runs via `flyte.run()`.

### Key patterns

**`AppEnvironment`** — defines the deployed app container:
```python
serving_env = flyte.app.AppEnvironment(
    name="everstorm-rag-chatbot",
    image="docker.io/johndellenbaugh/rag-app:latest",  # plain string URI — factory methods don't work here
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="PG_URL", as_env_var="PG_URL"),
    ],
    env_vars={"FLYTE_BACKEND": "cluster", "APP_VERSION": "4"},  # bump APP_VERSION to force redeploy
    port=7860,
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)
```

**`@serving_env.server`** — marks the function that starts the server inside the deployed pod:
```python
@serving_env.server
def _cluster_server():
    css = CSS_FILE.read_text()
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False, css=css)
```

**`flyte.init_in_cluster()`** — when `FLYTE_BACKEND=cluster`, the app is already inside Union so it initializes without an external endpoint. Added to `config.py`:
```python
elif BACKEND == "cluster":
    flyte.init_in_cluster()
```

**Deploy command:**
```bash
flyte --endpoint tryv2.hosted.unionai.cloud --org tryv2 --config flyte.yaml deploy app.py --project dellenbaugh --domain development serving_env
```

### Ingest tab: PDF upload widget

The original ingest tab used a `CheckboxGroup` backed by a local `data/` directory. This only works when running the UI on the same machine as the PDFs — not viable for a deployed app.

**Fix:** replaced with `gr.File(file_types=[".pdf"], file_count="multiple")`. Users drag-and-drop or browse to upload PDFs through the browser. Gradio saves them to a temp path; the ingest handler reads from there:

```python
for file_path in uploaded_files:
    fname = Path(file_path).name                    # original filename preserved
    b64 = base64.b64encode(Path(file_path).read_bytes()).decode()
```

This works identically in local dev and in the deployed cluster app.

---

## Deployment Issues & Solutions

Encountered during first end-to-end Union cloud run. Documented here so future projects avoid the same friction.

### 1. Gradio 6.0 Breaking Changes
Three parameters removed in Gradio 6.0:
- `css` on `gr.Blocks()` → moved to `launch(css=...)`
- `type="messages"` on `gr.Chatbot()` → removed
- `show_copy_button=True` on `gr.Chatbot()` → removed

### 2. Docker Not Running
Flyte builds the task container image locally using Docker. If Docker Desktop isn't open, the build fails immediately with a daemon connection error. Start Docker Desktop before running `python app.py` with `FLYTE_BACKEND=union`.

### 3. ghcr.io 403 on Pull
Docker couldn't pull the Flyte base image from GitHub Container Registry anonymously. Fix: authenticate with a GitHub PAT (`read:packages` scope):
```bash
echo YOUR_TOKEN | docker login ghcr.io -u YOUR_GITHUB_USERNAME --password-stdin
```

### 4. Image Push Goes to Wrong Registry by Default
The Union SDK defaults to pushing the built container image to `ghcr.io/flyteorg` — a namespace users don't own. Fix: explicitly set the registry in `config.py`:
```python
flyte.Image.from_debian_base(python_version=(3, 11), registry="docker.io/johndellenbaugh")
```
Requires a Docker Hub account. The created repo must be set to **Public** so the Union cluster can pull it.

### 5. Union Console Secrets Page Broken
The Secrets management page in the Union web UI throws "We're having trouble loading the secrets." Use the CLI instead:
```bash
flyte get secret --project dellenbaugh --domain development
flyte create secret SECRET_NAME --project dellenbaugh --domain development
```
**(Union-side bug — reported to Union team)**

### 6. `flyte create secret` Does Not Update Existing Keys
Running `flyte create secret` on an already-existing key silently does nothing — old value persists, no error returned. Fix: use a new key name (`PG_URL` instead of `DATABASE_URL`).
**(Union CLI bug — reported to Union team)**

### 7. Supabase Direct Connection is IPv6-Only
`db.<project>.supabase.co:5432` resolves to an IPv6 address. The Union cluster (AWS us-east-2) has no IPv6 connectivity. Fix: use Supabase's **Session Pooler** URL:
```
postgresql://postgres.<project>:<password>@aws-1-us-west-2.pooler.supabase.com:5432/postgres
```
Found at: Supabase project → **Connect** button → **Session pooler**.

### 8. psycopg3 Still Picks IPv6 Even With Pooler URL
Even with the session pooler hostname, psycopg3 can prefer the IPv6 address returned by DNS. Fix: added `_pg_connect()` helper in `workflows.py` that explicitly resolves to IPv4 and passes it as `hostaddr`:
```python
def _pg_connect():
    import psycopg
    url = os.environ["PG_URL"]
    p = urlparse(url)
    ipv4 = socket.getaddrinfo(p.hostname, None, socket.AF_INET)[0][4][0]
    return psycopg.connect(
        host=p.hostname, hostaddr=ipv4,
        port=p.port or 5432, dbname=p.path.lstrip("/"),
        user=p.username, password=p.password,
    )
```

### 9. `flyte.run()` is Non-Blocking on Union Backend
`flyte.run()` submits the workflow and returns immediately — it does not wait for completion. `run.outputs().o0` is `None` until the run finishes.

**Initial workaround (polling):**
```python
outputs = run.outputs()
deadline = time.time() + 180
while (outputs is None or outputs.o0 is None) and time.time() < deadline:
    time.sleep(3)
    outputs = run.outputs()
result = json.loads(outputs.o0)
```

**Proper fix (from Union "Build Apps" docs) — `run.wait()`:**
```python
run.wait()
result = json.loads(run.outputs().o0)
```
`run.wait()` blocks until the run completes, then `run.outputs()` is guaranteed to have the result. Replaced both the ingest and chat polling loops with this pattern.

### 10. `flyte deploy` CLI Requires a Config File for Project/Domain
`flyte deploy app.py --project dellenbaugh --domain development serving_env` fails with "Project must be provided to initialize the client" even though `--project` is on the command line. The CLI's own client and the Python SDK's internal client are separate — the SDK needs project info from a config file.

Fix: create `flyte.yaml` in the project directory and pass `--config flyte.yaml` to the CLI:
```yaml
admin:
  endpoint: dns:///tryv2.hosted.unionai.cloud
  insecure: false

task:
  project: dellenbaugh
  domain: development
```

Full working deploy command:
```bash
flyte --endpoint tryv2.hosted.unionai.cloud --org tryv2 --config flyte.yaml deploy app.py --project dellenbaugh --domain development serving_env
```

### 11. Module-Level Workflow Import Conflicts With `flyte deploy` CLI Discovery
When `flyte deploy` parses `app.py` to discover `serving_env`, it imports the module. A top-level `from workflows import ingest_pipeline, query_pipeline` triggered `config.py`'s `flyte.init()` which conflicted with the CLI's own client initialization. The SDK project state was overwritten, causing "Project must be provided."

Fix: lazy-import workflows inside the handler functions:
```python
def run_ingest(...):
    from workflows import ingest_pipeline  # imported on first use only
    ...

def chat(...):
    from workflows import query_pipeline
    ...
```

Then add `import config` at module level to ensure `flyte.init()` runs once with the right settings before `serving_env` is instantiated.

### 12. `flyte deploy` Bundles Only Python Files — `styles.css` Missing on Cluster
The deploy bundler follows Python imports and only packages `.py` files. `styles.css` is not bundled, so the cluster pod fails with `FileNotFoundError: '/root/styles.css'`.

Fix: inline the full CSS as a `_CSS` string constant in `app.py`, with a helper that tries the file first and falls back to the inline copy:
```python
_CSS = """..."""  # full CSS inlined

def _load_css() -> str:
    try:
        return CSS_FILE.read_text()   # works locally
    except FileNotFoundError:
        return _CSS                   # works on cluster
```

### 13. `RuntimeError: Event loop stopped before Future completed` in Cluster Server
Union's serve runner wraps `@serving_env.server` in `asyncio.run()`. Gradio's `launch()` also manages asyncio internally, causing a conflict on event loop shutdown. Investigated multiple approaches:
- `async def` + `run_in_executor`: ran `launch()` in a thread pool to isolate event loops — partially helped but event loop still errored during cleanup
- Current attempt: sync `def` + `ui.queue()` matching the pattern from Union docs

```python
@serving_env.server
def _cluster_server():
    css = _load_css()
    ui = build_ui()
    ui.queue()
    ui.launch(server_name="0.0.0.0", server_port=7860, share=False, css=css)
```
**Resolved** — sync `def` with `ui.queue()` before `ui.launch()` matches the Union docs pattern and runs cleanly.

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
   PG_URL=postgresql://postgres.<project>:<password>@aws-1-us-west-2.pooler.supabase.com:5432/postgres
   FLYTE_BACKEND=union
   ```
   Use the **Session Pooler** URL from Supabase Connect page (not the direct connection — see Issue #7).

2. Enable pgvector in Supabase SQL Editor:
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. Install dependencies:
   ```bash
   pip install psycopg[binary] pgvector sentence-transformers PyMuPDF \
               langchain-text-splitters anthropic flyte gradio python-dotenv
   ```

### Run locally

```bash
python app.py
```

Opens at `http://localhost:7860`. Docker Desktop must be running when `FLYTE_BACKEND=union`.

### Deploy to Union cluster

```bash
flyte --endpoint tryv2.hosted.unionai.cloud --org tryv2 --config flyte.yaml deploy app.py --project dellenbaugh --domain development serving_env
```

Bundles `app.py`, `config.py`, and `workflows.py` and uploads them to Union. The app image (`rag-app:latest`) provides the Python environment. After deploy, Union returns a public URL for the hosted app. No image rebuild needed unless a Dockerfile changes.

**Force redeployment** (when Union reports "No changes in App spec"): bump `APP_VERSION` in the `env_vars` dict in `serving_env` and redeploy.

### Demo Flow

1. **Ingest tab** — upload one or more PDFs (drag-and-drop or browse), click "Run Ingest on Union"
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
| `cluster` | App is running inside Union pod — use `flyte.init_in_cluster()` |

---

## Supabase pgvector — Quick Reference

### Connection

```python
import psycopg
from pgvector.psycopg import register_vector

conn = psycopg.connect(os.environ["PG_URL"])
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
