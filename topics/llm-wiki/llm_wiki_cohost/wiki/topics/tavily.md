---
title: Agentic Search with Tavily
date: 2026-04-03
folder: topics/tavily/
concepts: [react-loop, research-pipelines, agents, tool-use]
tools: [tavily, fastmcp, langgraph, flyte, gradio]
---

Three sub-projects demonstrating Tavily as the web search backbone for AI
agents, ranging from a simple usage gallery to a full orchestrated research
pipeline with quality gates and parallel fan-out.

## What was built

**`tavily-usage-examples/`** — Runnable scripts covering all four Tavily
endpoints: search, extract, crawl, map. A reference gallery, not an agent.

**`fastmcp_agent_tavily/`** — A ReAct agent (Claude + Anthropic SDK) backed by
a FastMCP server that exposes three Tavily tools (search, extract, crawl).
The agent reasons about which tool to use and in what order, printing each tool
call to the console so the loop is visible. The first explicit ReAct
implementation in the series.

**`langgraph_agent_research/`** — A full research pipeline: LangGraph controls
the logic (plan → parallel research → synthesize → quality gate → loop or
finalize); Flyte provides the compute (each researcher runs as a separate task
with its own container). The most architecturally complex project in the series
to this point.

## Key decisions

**FastMCP as the tool layer for the Claude agent.** Reuses the MCP pattern from
week 1 — Tavily tools live on a FastMCP server, the agent connects via SSE.
Demonstrates that the MCP server is the stable tool layer; only the agent's
system prompt and ReAct loop change between projects.

**LangGraph + Flyte co-orchestration.** LangGraph owns the pipeline logic;
Flyte owns the compute. The pipeline graph accepts the Flyte task as a
parameter — LangGraph's `Send` API fans out to it. On a cluster, each `Send`
becomes a separate container. A clean separation of concerns.

**Quality gate with iterative deepening.** After synthesis, a quality checker
scores the report and identifies gaps. If gaps exist, the pipeline fans out
again to research them — then re-synthesizes. Controlled by `max_iterations`.
This is the first example in the series of a feedback loop inside a pipeline.

**OpenAI for the LangGraph project, Claude for the FastMCP project.** Different
providers used across the same week's projects, reinforcing the MCP portability
argument from week 1.

## Connections

- [ReAct Loop](../concepts/react-loop.md) — the agent reasoning pattern, first explicit implementation
- [Research Pipelines](../concepts/research-pipelines.md) — plan/fan-out/synthesize/quality-gate pattern
- [Agents](../concepts/agents.md) — web research as a concrete agent use case
- [Tool Use](../concepts/tool-use.md) — Tavily as an AI-native search tool
- [Tavily](../tools/tavily.md) — the search API powering both agent projects
- [FastMCP](../tools/fastmcp.md) — reused from week 1 as the tool server
- [LangGraph](../tools/langgraph.md) — pipeline and ReAct graph orchestration
- [Flyte / Union](../tools/flyte.md) — parallel task fan-out for the research pipeline
- [Gradio](../tools/gradio.md) — UI for the research pipeline
