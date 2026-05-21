# Cognee Chatbot — Research & Design

## The Project

A knowledge-graph chatbot over the Everstorm Outfitters PDF corpus, powered by
Cognee as the memory and retrieval backend. Uses the same 15 fictional support
documents as `topics/vectorstore/vector_rag_chatbot` and
`topics/graphs-neo4j/graph_rag_chatbot` — making it a natural third data point
in the series' RAG progression.

**The story in one line:** the same job as the week 7 graph_rag_chatbot, in a
fraction of the code, with persistent memory the prior systems didn't have.

---

## Why Cognee

The series has built two prior approaches to RAG over the Everstorm corpus:

| Week | Project | Approach | Ingest complexity |
|---|---|---|---|
| 6 | `vector_rag_chatbot` | pgvector + Claude | ~100 lines of pipeline code |
| 7 | `graph_rag_chatbot` | Neo4j + manual entity extraction + Louvain | ~300 lines of pipeline code |
| 9 | `cognee_chatbot` | Cognee (`add()` + `cognify()`) | ~10 lines |

Week 7's ingest pipeline had 6 explicit steps: parse_and_chunk → extract_entities
(Claude tool use) → load_graph → create_vector_index → resolve_entities →
detect_communities → summarize_communities. Cognee collapses all of this into
two function calls. The graph is built automatically; entity extraction, entity
normalization, and community detection happen internally.

---

## What Cognee brings that the prior systems didn't

- **Automatic knowledge graph construction.** No manual ontology, no extraction
  pipeline. `cognify()` reads the PDFs and builds the graph using an LLM internally.
- **Persistent memory across sessions.** The knowledge graph survives restarts.
  Prior systems required re-ingesting on every fresh deployment.
- **`improve()` / Memify Pipeline.** The memory can consolidate and strengthen
  itself over time — not just grow.
- **Unified API.** `remember()` / `recall()` / `forget()` / `improve()` over the
  combined vector + graph infrastructure. No custom retrieval modes to route between.

---

## Architecture

```
User → Gradio UI → flyte.run() → Union Cluster
                                       │
                    ┌──────────────────┴──────────────────┐
                    │           Ingest Task                │
                    │  cognee.add(pdf_paths)               │
                    │  cognee.cognify()                    │
                    └──────────────────┬──────────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    │           Query Task                 │
                    │  cognee.search(query)                │
                    │  Claude generates grounded answer    │
                    └─────────────────────────────────────┘
                                       │
                               pgvector on Supabase
```

### Two tasks, not six

Week 7 had 6+ Union task nodes visible in the console. Cognee shows 2.
That's not a weakness — it's the story. The work is happening; Cognee just
does it for you. Worth naming explicitly on stream.

---

## Stack

| Layer | Technology | Notes |
|---|---|---|
| Memory / graph | Cognee | `add()` + `cognify()` + `search()` |
| Vector backend | pgvector on Supabase | Already set up from week 6 |
| LLM (answers) | Claude Sonnet (`claude-sonnet-4-6`) | Anthropic API |
| LLM (cognify) | Anthropic | Configured to use Claude internally — keeps the stack clean |
| UI | Gradio | Ingest tab + Chat tab |
| Orchestration | Union / Flyte 2 | Same app deployment pattern as weeks 6–7 |
| PDF parsing | Cognee built-in (PyPdf) | Basic text extraction — watch for table/layout issues on structured docs |

### Note on PDF parsing quality

Cognee uses PyPdfLoader for PDF extraction — basic text, struggles with complex
layouts and tables. The Everstorm PDFs are structured support docs (numbered
sections, bullet points, some tables). May affect extraction quality vs. week 6/7
which used PyMuPDF. Worth testing early.

---

## Cognee API — key patterns

```python
import cognee

# Ingest
await cognee.add("/path/to/everstorm_returns.pdf")
await cognee.cognify()   # builds knowledge graph internally

# Query
results = await cognee.search(
    query_type=SearchType.GRAPH_COMPLETION,
    query_text="What benefits do Elite members get on returns?"
)

# Persistent memory
await cognee.remember("user prefers terse answers")
context = await cognee.recall("what do I know about this user?")
```

**Note:** `add()` is the legacy API; `remember()` is recommended for new projects.
For document ingestion (PDFs), `add()` + `cognify()` is still the right pattern.
`remember()` / `recall()` are better suited for agent conversational memory.

---

## Demo story

**The thesis question:** "Can Cognee match week 7's retrieval quality with a
fraction of the code?"

**Questions to test (same as week 7's demo questions):**

| Question | Tests |
|---|---|
| "What is the return window for sale items?" | Simple factual — both systems should nail this |
| "What benefits do Elite members get on returns?" | Relationship traversal: Tier → Benefit → Policy |
| "Which policies apply to international purchases?" | Multi-hop cross-document |
| "What programs does Everstorm have for repeat customers?" | Thematic / community summary |

**Session persistence demo:** answer a question, note the response, restart the
app, ask a follow-up. Memory survived. Week 7 couldn't do this.

**Code comparison moment:** pull up `graph_rag_chatbot/ingest/pipeline.py` (6
steps, ~150 lines) next to the Cognee ingest task (~10 lines). Let it land without
over-explaining.

---

## Configuration

### LLM provider — use Claude for cognify()

Confirmed: Cognee supports Anthropic natively. Set three env vars:

```dotenv
LLM_PROVIDER="anthropic"
LLM_MODEL="claude-sonnet-4-6"
LLM_API_KEY="sk-ant-..."
```

Claude uses native tool-calling for structured output (entity extraction) — no
extra configuration needed. This keeps the entire stack on Anthropic.

### pgvector on Supabase

Install postgres extras:

```bash
pip install "cognee[postgres]"
```

Set env vars (Cognee shares these between its relational store and vector store):

```dotenv
VECTOR_DB_PROVIDER="pgvector"
DB_HOST="db.<project-ref>.supabase.co"
DB_PORT="5432"
DB_NAME="postgres"
DB_USERNAME="postgres"
DB_PASSWORD="<supabase-password>"
```

One-time setup in Supabase SQL editor (already done in week 6):

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

**Note:** Cognee takes individual `DB_*` env vars, not a connection string URL.
Use the direct connection host (`db.<project-ref>.supabase.co`), not the Session
Pooler URL. Same credentials as week 6.

### Graph visualization

`cognee.visualize_graph()` generates an interactive HTML file (nodes, edges,
weights). Embed in Gradio as a third tab:

```python
html_path = await cognee.visualize_graph("./graph.html")
with open(html_path) as f:
    html_content = f.read()
# render with gr.HTML(html_content)
```

### improve() behavior

Two modes:
- **Without session ID:** consolidates all memory globally — resolves
  contradictions, strengthens cross-document connections, adds derived retrieval
  structures.
- **With session ID:** targeted feedback loop — applies quality weights to nodes
  and edges involved in that session's queries, persists session Q&A into the
  permanent graph.

For the demo: call `cognee.improve()` after a few queries and check whether
graph edge weights changed. Visually compelling if graph visualization is wired up.

---

## Remaining open questions (require code execution)

1. **Does `cognify()` handle the Everstorm PDFs well?** PyPdf vs PyMuPDF quality
   difference on structured support docs — test early. Watch for tables and
   numbered-section parsing.
