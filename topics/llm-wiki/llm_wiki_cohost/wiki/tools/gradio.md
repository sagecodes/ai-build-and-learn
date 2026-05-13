---
title: Gradio
weeks: [mcp]
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

### Week 4 — AutoResearch (2026-04-17)

Used as a monitoring dashboard for overnight runs (`dashboard/app.py`). Unlike
prior weeks where Gradio is the primary UI, here it's a read-only viewer that
polls Google Firestore every 60 seconds. Shows val_bpb progression chart
(green=kept, red=reverted), experiment log table, stat row (current val_bpb,
total experiments, kept count, success rate), and run summary. Run dropdown
lets you toggle between `agent.py` and `flyte_workflow.py` runs stored in
the same Firestore database.
