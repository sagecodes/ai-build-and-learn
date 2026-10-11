# Graph RAG Chatbot

Knowledge-graph Q&A over 15 Everstorm Outfitters policy PDFs, powered by
**Neo4j AuraDB**, **Claude Sonnet**, and **Union** (Flyte 2.x).

Companion project to [`topics/vectorstore/vector_rag_chatbot`](../../vectorstore/vector_rag_chatbot/)
— same documents, different retrieval architecture. The goal is to show _why_
and _when_ a knowledge graph outperforms pure vector similarity search.

---

## What it demonstrates

- **Graph vs. vector retrieval** — entities, relationships, and community
  structure give Claude richer context than chunk embeddings alone
- **Agentic RAG loop** — every pipeline step is a visible Union task:
  fan-out extraction, graph loading, entity resolution, community detection,
  routing, retrieval, generation
- **Intelligent query routing** — Claude classifies each question into one of
  three retrieval modes so the right strategy fires automatically
- **Neo4j as a unified store** — HNSW vector index and graph traversal in a
  single database, no cross-service hops

### Three retrieval modes

| Mode | Triggered when | How it works |
|---|---|---|
| **Hybrid** | Specific facts, rules, numbers | Vector search on Chunk embeddings → follow `MENTIONS` edges to nearby Entities |
| **Entity** | Named things and their relationships | Claude extracts entity names → Neo4j neighborhood traversal via `RELATED` edges |
| **Community** | Broad themes and program overviews | Embed question → cosine similarity to pre-computed Community summaries |

---

## Architecture

```
User → Gradio UI → flyte.run() → Union Cluster
                                       │
                    ┌──────────────────┴──────────────────┐
                    │           Ingest Pipeline            │
                    │  process_pdf × 15 (parallel)         │
                    │    └─ parse_and_chunk (PyMuPDF)      │
                    │    └─ extract_entities (Claude)      │
                    │  load_graph (gte-small + Neo4j)      │
                    │  create_vector_index (HNSW)          │
                    │  resolve_entities (cosine merge)     │
                    │  detect_communities (Louvain)        │
                    │  summarize_communities (Claude)      │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │            Query Pipeline            │
                    │  route_query (Claude classifier)     │
                    │  [hybrid | entity | community]       │
                    │  generate (Claude RAG answer)        │
                    └─────────────────────────────────────┘
                                       │
                               Neo4j AuraDB Free
                          (graph + HNSW vector index)
```

See [`architecture.html`](architecture.html) for the interactive step-by-step walkthrough.

### Graph schema

```
(Document)-[:HAS_CHUNK]→(Chunk)-[:MENTIONS]→(Entity)
(Entity)-[:RELATED {type, description}]→(Entity)
(Entity)-[:BELONGS_TO]→(Community)
```

Each `Chunk` node stores a 384D gte-small embedding indexed by Neo4j's HNSW
vector index. `Community` nodes hold Claude-generated summaries of entity clusters.

---

## Project structure

```
graph_rag_chatbot/
├── app.py               Gradio UI (Ingest + Chat tabs) + Union serve entry point
├── workflows.py         Re-exports ingest_pipeline and query_pipeline
├── config.py            Constants, secrets, TaskEnvironment, Everstorm ontology
│
├── ingest/
│   ├── pipeline.py      ingest_pipeline orchestrator (asyncio.gather fan-out)
│   ├── chunking.py      parse_and_chunk — PyMuPDF + RecursiveCharacterTextSplitter
│   ├── extraction.py    extract_entities — Claude tool use → entities + relationships
│   ├── graph_loader.py  load_graph + create_vector_index
│   └── enrichment.py   resolve_entities + detect_communities + summarize_communities
│
├── query/
│   ├── pipeline.py      query_pipeline orchestrator
│   ├── routing.py       route_query — Claude classifies to hybrid/entity/community
│   ├── retrieval.py     hybrid_retrieve, entity_retrieve, community_retrieve
│   └── generation.py   generate — mode-specific RAG prompt → Claude answer
│
├── Dockerfile           Task image (gte-small baked in, no runtime download)
├── Dockerfile.app       App serving image
├── build-images.yml     Manual GitHub Actions workflow — copy to .github/workflows/ to activate
├── architecture.html    Interactive architecture walkthrough
├── RESEARCH.md          Design decisions, schema, retrieval patterns
├── requirements.txt
└── data/                15 Everstorm Outfitters PDFs
```

---

## Setup

### 1. Neo4j AuraDB Free

1. Create a free instance at [neo4j.com/cloud/aura](https://neo4j.com/cloud/aura/)
2. Note the connection URI (`neo4j+s://...`), username, and password

### 2. Environment variables

Copy `.env.template` to `.env` and fill in:

```
ANTHROPIC_API_KEY=sk-ant-...
NEO4J_URI=neo4j+s://<id>.databases.neo4j.io
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=<your-password>
FLYTE_BACKEND=local   # or "union" for remote
```

### 3. Union secrets

Register secrets so tasks can access them on the cluster.
Use the `flyte` CLI — `union create secret` does not work on Union hosted:

```bash
flyte --endpoint tryv2.hosted.unionai.cloud --org tryv2 create secret ANTHROPIC_API_KEY --value "sk-ant-..."
flyte --endpoint tryv2.hosted.unionai.cloud --org tryv2 create secret NEO4J_URI         --value "neo4j+s://..."
flyte --endpoint tryv2.hosted.unionai.cloud --org tryv2 create secret NEO4J_USERNAME    --value "neo4j"
flyte --endpoint tryv2.hosted.unionai.cloud --org tryv2 create secret NEO4J_PASSWORD    --value "<password>"
```

### 4. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

---

## Running

### Local (tasks run in-process, no Union)

```bash
FLYTE_BACKEND=local python app.py
```

Open `http://localhost:7860`.

### Union (tasks run on the cluster)

Build and push both Docker images first (or run `build-images.yml` manually from GitHub Actions):

```bash
docker build -f Dockerfile     -t johndellenbaugh/graphrag-task:latest . && docker push johndellenbaugh/graphrag-task:latest
docker build -f Dockerfile.app -t johndellenbaugh/graphrag-app:latest  . && docker push johndellenbaugh/graphrag-app:latest
```

Deploy the app to Union:

```bash
FLYTE_BACKEND=union python app.py --deploy
```

The app URL is printed to the console and visible in the Union console under Apps.

---

## Usage

### Ingest

1. Open the **Ingest Graph** tab
2. Upload PDFs (or leave empty to use the 15 PDFs already in `data/`)
3. Click **Run Ingest on Union**

Watch the status log as the pipeline runs. Each step appears as a separate
task in the Union console. The 15 `process_pdf` tasks run in parallel.
Enrichment steps (resolve, detect, summarize) run sequentially after.

### Chat

1. Open the **Chat** tab
2. Type a question and submit
3. The answer includes a retrieval mode badge (**Hybrid** / **Entity** / **Community**),
   source documents, and entities used
4. The **Last Query Retrieval** panel (sidebar) shows Claude's routing reasoning,
   the pipeline path with graph edge types (`Graph: MENTIONS` / `Graph: RELATED` /
   `Graph: Community Sim`), and source and entity counts — useful for understanding
   why a particular retrieval strategy was chosen

**Example questions by mode:**

| Mode | Example question |
|---|---|
| Community | "What is the Everstorm loyalty program?" |
| Hybrid | "What are the point thresholds for each loyalty tier?" |
| Hybrid | "What is the return policy for damaged items?" |
| Community | "Give me an overview of Everstorm's sustainability initiatives." |
| Entity | "How do loyalty points relate to returns?" |

---

## CI/CD

`build-images.yml` builds and pushes both Docker images to Docker Hub.
It is triggered manually (`workflow_dispatch` only) — copy it to
`.github/workflows/` in your fork and run it from the GitHub Actions tab.

---

## Key design decisions

**Why `asyncio.gather` instead of Flyte `map_task`?**
Flyte 2.x eager mode dispatches Union sub-tasks via `await task()`. Wrapping
all 15 PDF tasks in `asyncio.gather` achieves true parallel fan-out visible
as separate nodes in the Union console.

**Why direct Cypher instead of `neo4j-graphrag`?**
The retrieval logic is simple enough that direct parameterized Cypher queries
are clearer and have no framework overhead. The `neo4j-graphrag` library is
worth evaluating for larger projects with more complex retrieval patterns.

**Why community summaries at ingest time?**
Pre-computing summaries during ingest means broad "overview" questions don't
require a per-query LLM call over hundreds of entities. The community summary
is retrieved in one embedding similarity lookup.

**Why AuraDB Free instead of GCP VM?**
Zero setup time for a demo. Node/relationship limits are well above what 15
PDFs produce. Community detection is handled in Python via `python-louvain`
since GDS is not available on the free tier.
