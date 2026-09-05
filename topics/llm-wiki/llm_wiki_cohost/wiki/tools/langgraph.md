---
title: LangGraph
weeks: [tavily]
---

Graph-based framework for building stateful agent and pipeline workflows.
Nodes are Python functions; edges define control flow. Supports cycles
(enabling ReAct loops), conditional branching, and the `Send` API for
dynamic fan-out to parallel nodes.

Part of the LangChain ecosystem but usable independently. Tools wrapped with
LangChain's `@tool` decorator work natively in LangGraph agent nodes.

## Usage across the series

### Week 2 — Agentic Search with Tavily (2026-04-03)

Used to build two graphs in `graph.py`:

**ReAct subgraph** (`build_research_subgraph()`) — a cyclic graph with an agent
node and a tools node. Edges: agent → tools (when tool call) or agent → END
(when final answer). This is a LangGraph-native ReAct loop. Runs inside each
`research_topic` Flyte task.

**Pipeline graph** (`build_pipeline_graph()`) — a DAG with conditional edges:
plan → Send fan-out → synthesize → quality_check → (gaps: identify_gaps →
Send again) or (good: finalize). The pipeline graph accepts the `research_topic`
Flyte task as a parameter, so LangGraph controls the logic while Flyte controls
the compute.

Key design: LangGraph's `Send` API is what enables dynamic fan-out — the planner
decides how many sub-topics to research, then `Send` dispatches that many
parallel tasks at runtime. On a Flyte cluster each `Send` becomes a separate
container.
