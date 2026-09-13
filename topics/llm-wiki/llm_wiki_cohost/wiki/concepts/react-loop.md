---
title: ReAct Loop
first_seen: topics/tavily/
weeks: [tavily]
---

Reason + Act — the agent reasoning pattern where the model alternates between
thinking about what to do and calling a tool to do it, looping until the task
is complete. Each iteration: observe context → reason → act (tool call) →
observe result → reason again.

The key property: the model decides at runtime which tool to call next and
whether to continue looping or produce a final answer. This is what separates
an agent from a pipeline — the control flow is not predetermined.

## How it appeared across the series

### Week 2 — Agentic Search with Tavily (2026-04-03)

First explicit ReAct implementation in the series, appearing in two projects:

**`fastmcp_agent_tavily/`** — Manual ReAct loop implemented in `agent.py` using
the Anthropic SDK. The loop:
1. Send query + available tools to Claude
2. If Claude responds with `tool_use`, call each tool via FastMCP client
3. Feed results back as context
4. Repeat until Claude responds with final text (no more tool calls)

Each tool call is printed to the console — the loop is visible and inspectable.
Tools available: `tavily_search`, `tavily_extract`, `tavily_crawl`.

**`langgraph_agent_research/`** — ReAct loop implemented as a LangGraph subgraph
(`build_research_subgraph()`) for each sub-topic researcher. The subgraph
contains an agent node and a tools node with edges between them. LangGraph
manages the loop state; the `research_topic` Flyte task runs the subgraph.

Notable: the two ReAct implementations use different patterns — manual SDK loop
vs. LangGraph graph — to achieve the same behavior. LangGraph's graph
representation makes the loop structure explicit and inspectable.

### Week 5 — Gemma 4 (2026-04-24)

Third ReAct implementation in the series — the first using a local open-weight
model. The `agent/` demo runs the same observe → reason → act loop as weeks 2
and 3 but with Gemma 4 via Ollama instead of Claude or a LangGraph subgraph.

Confirms the pattern is model-agnostic: the loop structure (send context +
tools → check for tool call → execute → feed result back → repeat) is
identical regardless of provider. The model changes; the scaffolding doesn't.

## Open questions

- What's the practical difference between a manual SDK loop and a LangGraph
  ReAct subgraph for simple cases? When does LangGraph's overhead pay off?
- How do later weeks structure their agent loops?
