---
title: MCP with FastMCP
date: 2026-03-27
folder: topics/mcp/
concepts: [mcp-protocol, tool-use, agents]
tools: [fastmcp, gradio, flyte]
---

First week of the series. Built a persistent FastMCP server exposing multiple
tool types and connected to it with both OpenAI Agents SDK and Anthropic SDK
clients — demonstrating MCP's core value: same server, different model providers.

## What was built

- A persistent FastMCP server (SSE transport, port 8000) with seven tools
  spanning pure computation, file system access, external APIs, and web search
- A stateful data analysis server (`data_server.py`, port 8001) with tools
  that chain: load → filter → aggregate → chart
- Two agent clients (OpenAI and Claude) connecting to the same running server
- A Gradio chat app (`chat_app.py`) for interactive data analysis with inline charts
- Flyte deployment wrapper (`flyte_app.py`) to host the server on Union cloud

## Key decisions

**FastMCP over the official `mcp` SDK.** FastMCP's decorator pattern is more
concise — function name → tool name, docstring → description, type annotations
→ JSON Schema automatically. The official `mcp[cli]` gives more protocol-level
control but adds boilerplate for a demo.

**SSE transport.** Persistent server over Server-Sent Events so multiple clients
can connect without restarting. Each client connects, calls `tools/list`, then
`tools/call` as needed.

**Stateful data server as a second demo.** Shows that MCP tools are not
stateless by design — datasets loaded in one call persist for subsequent filter
and chart calls. The model orchestrates the chain.

**MCP vs baked-in tools tradeoff named explicitly.** The README draws the
monolith vs microservice parallel: MCP pays off when multiple agents share tools
or when you want client portability. If one agent uses one tool set, it adds
complexity for no benefit.

## Connections

- [MCP Protocol](../concepts/mcp-protocol.md) — the standard this week introduced
- [Tool Use](../concepts/tool-use.md) — how models discover and call tools
- [Agents](../concepts/agents.md) — the client side of the MCP pattern
- [FastMCP](../tools/fastmcp.md) — the library used to build the server
- [Gradio](../tools/gradio.md) — the chat interface for the data server
- [Flyte](../tools/flyte.md) — cloud deployment of the MCP server
