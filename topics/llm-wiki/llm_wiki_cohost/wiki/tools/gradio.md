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
