---
title: Tool Use
first_seen: topics/mcp/
weeks: [mcp]
---

The mechanism by which LLMs call external functions. The model receives a list
of tool schemas (name, description, JSON Schema for inputs), decides which tool
to call based on the user's request, emits a structured tool call, and receives
the result back as context for its next response.

Tool use is the foundation of agents: without it, a model can only generate
text. With it, a model can act.

## How it appeared across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Tool use appeared in two forms:

**Via MCP protocol.** The client calls `tools/list` to get schemas, then
`tools/call` when the model decides to use one. FastMCP generates the JSON
Schema automatically from Python type annotations and docstrings — the developer
writes a plain function, the protocol layer handles the rest.

**Tool types demonstrated:**
- Pure computation (add, multiply) — deterministic, no side effects
- File system (read_text_file) — stateful, reads from disk
- External API (get_weather) — side effects, network dependency
- Web tools (duck_duck_go, fetch_webpage) — unstructured external data
- Stateful chaining (data server) — load → filter → aggregate → chart in sequence

The stateful data server showed that tool calls are not inherently stateless —
server-side state can persist across a conversation turn, enabling multi-step
workflows where each tool call builds on the previous one.

## Open questions

- How does tool use differ across providers (OpenAI function calling vs
  Anthropic tool use vs MCP)? Are schemas interchangeable?
- What patterns emerge in later weeks for structuring complex multi-tool workflows?
