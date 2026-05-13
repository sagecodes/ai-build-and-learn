---
title: Flyte / Union
weeks: [mcp]
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
