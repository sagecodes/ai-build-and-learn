---
title: Agents
first_seen: topics/mcp/
weeks: [mcp]
---

LLM-powered systems that use tools to take actions, not just generate text.
An agent perceives its environment (context, tool results), decides what to do
(which tool to call or what to say), acts (calls the tool), and observes the
result — repeating until the task is done.

## How it appeared across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Agents appeared as MCP clients — the consumer side of the protocol. Two clients
were built: one using the OpenAI Agents SDK, one using the Anthropic SDK. Both
connected to the same FastMCP server, demonstrating that MCP decouples the
agent (client) from the tool implementation (server).

The agent loop here is minimal: user message → model picks tools → tools execute
→ model synthesizes answer. The MCP server handles tool execution; the SDK
handles the loop.

The data analysis demo showed a multi-step agent: the model chains
load → filter → aggregate → chart across multiple tool calls in one session,
with server-side state persisting between calls.

## Open questions

- What agent frameworks does the series explore beyond the Agents SDK? (LangGraph, custom loops?)
- How does the series handle agent memory across sessions?
- What's the distinction between a "ReAct agent" and a simpler tool-calling loop?
