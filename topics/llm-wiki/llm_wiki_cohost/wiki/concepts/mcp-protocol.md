---
title: MCP Protocol
first_seen: topics/mcp/
weeks: [mcp]
---

Model Context Protocol — a standard for exposing tools and context to language
models. Instead of every framework inventing its own tool format, MCP gives
clients and servers a shared contract for listing capabilities, describing
inputs, and calling them.

The core insight: standardize the client-server interface so that the same
server works with any model client (OpenAI, Claude, Claude Code, etc.) without
modification.

## How it appeared across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Introduced as the week's central topic. The protocol flow:
1. Client calls `tools/list` → server returns JSON schemas for all tools
2. Client calls `tools/call {name, args}` → server executes and returns result

Transport options covered: SSE (persistent, multi-client) and stdio (one-shot).
The demo used SSE so multiple clients could connect to one running server.

MCP tools can wrap anything — a simple function, an external API call, or
another LLM call. The calling model only sees the tool's name, description,
and schema; the implementation is opaque.

The tradeoff named: MCP adds a network hop and deployment complexity. Worth it
when multiple agents share tools or when portability across providers matters.
Not worth it for a single agent with its own private tool set.

## Open questions

- How does MCP server discovery work in production (registries, hardcoded URLs)?
- How does authentication / authorization over MCP get handled?
- How does the series use MCP in later weeks — as a client, a server, or both?
