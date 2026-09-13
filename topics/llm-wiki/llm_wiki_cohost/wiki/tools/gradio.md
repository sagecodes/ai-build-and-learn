---
title: Gradio
weeks: [mcp, tavily, openenv, autoresearch, gemma4, vectorstore, graphs-neo4j]
---

Python library for building web UIs for ML and AI applications. Minimal code
to create chat interfaces, file uploads, image displays, and interactive
components. Runs a local server and opens in the browser.

## Usage across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Used to build a chat interface (`chat_app.py`) over the data analysis MCP
server. Users type natural-language questions; the Gradio app routes them
through Claude, which calls MCP tools to load data, filter, aggregate, and
generate charts. Charts render inline in the chat window.

Runs at `http://localhost:7860`.

### Week 7 — Graph Data with Neo4j (2026-05-08)

Two chat UIs, both with graph-specific panels that make the retrieval
architecture visible:

**`graphrag-neo4j-flyte/`** — Retrieval mode radio button (Vector / Vector+Expand /
Hybrid RRF) plus a right-hand panel showing retrieved paper cards (`via vector`
or `via graph`) and a graph relations block listing neighbor titles and edge
types. Switching modes on the same question makes the difference between pure
semantic similarity and graph authority visible in real time.

**`graph_rag_chatbot/`** — Retrieval mode badge in each chat response (Hybrid /
Entity / Community), source documents, entities used, and a "Last Query Retrieval"
sidebar showing Claude's routing reasoning, the pipeline path with graph edge
types, and source/entity counts. Routing is explained, not just executed.

### Week 6 — Vector Stores (2026-05-01)

Five projects, five Gradio UIs — the densest Gradio week after Gemma 4.

**`rag-chroma-flyte/`** — Chat UI with a side panel showing retrieved chunks
and their cosine similarity scores. Top-k slider (1–10). Retrieval toggle for
direct comparison: RAG vs. plain Gemma chat.

**`agent-memory-chroma/`** — Chat with two live panels: "Retrieved this turn"
(memories injected as context) and "Written this turn" (new facts extracted).
Memory count in status bar. "Use memory" toggle for retrieval-on vs. off
comparison. Manual "💾 Save to HF" button as an explicit checkpoint.

**`rag-umap-visualizer/`** — Chat + Plotly scatter plot side-by-side. On each
query: Chroma top-k retrieved, UMAP-projected, and plotted in rank-colored
markers; query itself plotted as a gold star in the same fitted 2D space. Gray
dots = full corpus. Hover shows chunk text and similarity.

**`vector_rag_chatbot/`** — Two-tab UI: Ingest (drag-and-drop PDF upload, chunk
size/overlap config, collection name, "Run Ingest on Union") and Chat (grounded
answer + collapsible accordion of source chunks with similarity scores). Deployed
as a persistent Union app; `gr.File` widget replaces local file picker so it
works in both local dev and cluster deployments.

**`screen-context-harness/`** — Monitoring dashboard: latest screenshot capture,
LLM-generated caption, 60s consolidation summary, context chat, and a process
log showing RAG hits and LLM calls in real time.

**Gradio 6.0 breaking changes** encountered this week: `css` moved from
`gr.Blocks()` to `launch(css=…)`; `type="messages"` and `show_copy_button`
removed from `gr.Chatbot()`.

### Week 2 — Agentic Search with Tavily (2026-04-03)

Used as the UI for the LangGraph research pipeline (`langgraph_agent_research/app.py`).
Users submit a research query; the Gradio app kicks off the Flyte pipeline and
links to the Flyte run for observability. Supports both local and remote (cluster) modes via `RUN_MODE` env var.

### Week 3 — OpenEnv (2026-04-10)

Most complex Gradio UI in the series to this point. Three tabs:
- **Tab 1 (Side-by-Side)** — live Plotly chart updates per step in Local mode;
  final chart + Flyte console link in Task mode
- **Tab 2 (Agent Race)** — live scoreboard for three competing agents
- **Tab 3 (Parallel Fan-out)** — multiple questions fan out to parallel Flyte
  tasks; results stream in as tasks complete

UI components split into `ui_components.py` (Plotly chart builders, HTML card
builders) and `styles.css` (CSS classes) — first week with a dedicated UI
component module separate from `app.py`.

### Week 5 — Gemma 4 (2026-04-24)

Most Gradio-heavy week in the series — 8 separate apps, each on its own port
(7860-7865, 7867). All follow the same pattern: `python app.py` → Gradio UI
at localhost. `GRADIO_SHARE=1` enables a public HTTPS tunnel for remote dev
boxes (required for webcam access, which browsers block on non-HTTPS origins).

Notable UI patterns this week:
- **Live streaming**: `live-camera/` uses `stream_every` for webcam frame
  capture; `concurrency_limit=1` drops overlapping frames rather than queueing
- **Model picker**: `chatbot/` exposes a runtime model selector so variant
  (e2b/e4b/26b/31b) can be switched without restart
- **Side-by-side STT**: `voice/` dropdown switches between Whisper and Gemma
  native audio mid-session for A/B comparison

### Week 4 — AutoResearch (2026-04-17)

Used as a monitoring dashboard for overnight runs (`dashboard/app.py`). Unlike
prior weeks where Gradio is the primary UI, here it's a read-only viewer that
polls Google Firestore every 60 seconds. Shows val_bpb progression chart
(green=kept, red=reverted), experiment log table, stat row (current val_bpb,
total experiments, kept count, success rate), and run summary. Run dropdown
lets you toggle between `agent.py` and `flyte_workflow.py` runs stored in
the same Firestore database.
