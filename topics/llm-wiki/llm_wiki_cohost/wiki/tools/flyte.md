---
title: Flyte / Union
weeks: [mcp, tavily, openenv, autoresearch, gemma4, vectorstore, graphs-neo4j]
---

Flyte is an open-source workflow orchestration platform. Union is the managed
cloud offering. In the series, "Flyte" and "Union" are used interchangeably —
Union hosts the compute; the Flyte 2.x SDK defines tasks and apps.

Used across the series for two purposes:
1. **Task orchestration** — running ML/AI pipelines as parallelizable tasks
2. **App deployment** — hosting persistent services (Gradio UIs, MCP servers)
   via `flyte.app.AppEnvironment` + `flyte.serve()`

## Usage across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Used to deploy the FastMCP server to Union cloud. `flyte_app.py` wraps the
FastMCP server in a Starlette ASGI app and deploys it as a persistent service.
Once deployed, clients update `MCP_SERVER_URL` in `.env` to point at the remote
URL — no code changes needed since both clients read from that env var.

This was the deployment pattern for the server side only.

### Week 2 — Agentic Search with Tavily (2026-04-03)

Used for task orchestration in `langgraph_agent_research/`. Each `research_topic`
runs as a separate Flyte task with its own container, resources, and
observability. LangGraph's `Send` API fans out to the Flyte task — on a cluster,
each `Send` becomes a separate container running a full ReAct agent.

The pipeline graph accepts the Flyte task as a parameter, keeping LangGraph
(logic) and Flyte (compute) cleanly separated. This is the first use of Flyte
for parallel task fan-out rather than simple app deployment.

### Week 3 — OpenEnv (2026-04-10)

Used for three parallel execution patterns in the research agent demo:

**Side-by-side** (`run_side_by_side`) — two agents (traditional + Claude) run
as parallel sub-tasks on the same question. Results render when both complete.

**Agent race** (`run_agent_race`) — three Claude agents run in parallel.
`asyncio.as_completed` determines winner ordering.

**Parallel question fan-out (Tab 3)** — one Flyte task per research question.
Both agents run per task. Results stream as tasks complete.

New this week: **result caching**. Tasks are cached by `(query, agent_type, max_steps)`. Identical runs return instantly from cache — demonstrated live in
Tab 3. Each task pod starts its own local OpenEnv HTTP server on a random port.

### Week 5 — Gemma 4 (2026-04-24)

Used in `gemma4-smart-gallary/` for parallel vision task orchestration.
`asyncio.gather()` + `.aio` method fans out one Flyte task per image —
all visible simultaneously in the Flyte TUI. Two workflows:
- `describe_workflow`: scan_images → describe_image (parallel) → save_to_db
- `search_workflow`: load_images → check_image_match (parallel) → collect_results

`flyte.init(local_persistence=True)` — same pattern established in AutoResearch.
Results return synchronously; no polling needed.

Local-to-Union porting surfaced new constraints: cluster containers can't read
local file paths (solution: base64-encode image bytes, pass as `str`), Flyte
doesn't natively support `bytes` task inputs (falls back to PickleFile — use
`str` instead), and GCP credentials must be Union secrets not local `.env`.

### Week 7 — Graph Data with Neo4j (2026-05-08)

New pattern: **database as a Flyte `AppEnvironment`**. Neo4j 5 is deployed
alongside the chat UI as a persistent always-on service (`replicas=(1,1)`).
`from_dockerfile` is used instead of `from_base` to avoid the Flyte image
builder's `USER flyte` footer, which breaks the official Neo4j container image.

Ingest pipeline uses `asyncio.gather` to fan out `process_pdf` across all 15
PDFs in parallel (same pattern as Gemma 4 week). Post-fan-out enrichment steps
(resolve entities → detect communities → summarize) run sequentially.

**Snapshot/restore via `flyte.io.Dir`** — Neo4j has no persistent volume by
default on the devbox. `snapshot_neo4j` dumps nodes + edges + embeddings to a
`Dir` in rustfs (survives `flyte stop/start devbox`); `restore_neo4j` replays
via HTTP MERGE. Online snapshot over HTTP — no daemon stop needed, works on
community edition.

### Week 6 — Vector Stores (2026-05-01)

Two roles: pipeline orchestration and app deployment, both in the same week.

**Pipeline orchestration (`rag-chroma-flyte/`, `vector_rag_chatbot/`)** — RAG
ingest and query run as Flyte tasks. `vector_rag_chatbot/` fans out one
`load_and_chunk_task` per PDF in parallel; `embed_and_index_task` fires after
all chunks merge. `cache="auto"` on `load_and_chunk_task` makes re-ingesting an
unchanged PDF a free cache hit.

Key artifact pattern new this week: `flyte.io.Dir` as a pipeline output. The
Chroma persist directory (SQLite3 + parquet shards) is snapshotted as a `Dir`
artifact; the chat app mounts it via `RunOutput(type="directory", task_name=…)`.
The index is a first-class Flyte artifact, not a side-effect.

**App deployment (`vector_rag_chatbot/`)** — Gradio chat deployed as a persistent
Union service via `flyte.app.AppEnvironment`. Two separate images: task image
(embedding model baked in, ~ML deps) and app image (Gradio + Flyte, minimal).
The app image is intentionally thin — heavy work is delegated to task runs via
`flyte.run()`. `run.wait()` blocks until completion; `run.outputs().o0` is then
guaranteed non-None.

Notable deployment issues resolved this week (documented in `RESEARCH.md`):
Union console Secrets page broken (use CLI); `flyte create secret` silently
does nothing on update (use new key name); `flyte deploy` bundler only packages
`.py` files (inline CSS as a fallback string constant).

### Week 4 — AutoResearch (2026-04-17)

Used for per-experiment observability in `flyte_workflow.py`. Each experiment
cycle runs as 3 sequential Flyte tasks: `propose_change_task` (Claude API call,
~5-15s), `run_training_task` (write train.py + 5-min training run, ~350s),
`evaluate_task` (keep/revert + Firestore log, ~1s). If Claude returns an
unparseable response, only `propose_change_task` fails — the others are skipped,
making failures diagnosable by task.

Both modes (`agent.py` and `flyte_workflow.py`) produced equivalent ML results
(~21% success rate), confirming Flyte's orchestration overhead doesn't affect
outcomes. Flyte TUI shows a live node per iteration with status, duration, and
per-experiment HTML reports with Plotly charts.
