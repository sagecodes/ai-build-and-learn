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

### Week 2 — Agentic Search with Tavily (2026-04-03)

Tool use appeared in two integration styles:

**MCP tools (Claude agent).** Same pattern as week 1 — FastMCP server exposes
Tavily tools, agent discovers and calls them via the MCP protocol. The agent's
system prompt explicitly guides which tool to call in each situation (search
for discovery, extract for full content, crawl for documentation). Shows that
tool selection strategy belongs in the prompt, not the framework.

**LangChain `@tool` (LangGraph pipeline).** Tavily wrapped as a LangChain-
compatible tool so LangGraph's agent node can call it natively. A different
registration pattern than MCP — no separate server process, tool is a Python
function decorated with `@tool`. Simpler for a single-agent, single-pipeline
use case where portability across clients isn't needed.

The contrast with week 1 is instructive: MCP pays off when the tool server is
shared across multiple clients or needs to be hosted independently. `@tool`
is sufficient when one framework owns everything.

## Open questions

- How does tool use differ across providers (OpenAI function calling vs
  Anthropic tool use vs MCP)? Are schemas interchangeable?
- What patterns emerge in later weeks for structuring complex multi-tool workflows?
