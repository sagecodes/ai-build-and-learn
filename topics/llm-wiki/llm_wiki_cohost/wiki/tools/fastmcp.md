---
title: FastMCP
weeks: [mcp]
---

Python library for building MCP servers with minimal boilerplate. Turns Python
functions into MCP tools automatically via decorators: function name → tool
name, docstring → description, type annotations → JSON Schema.

FastMCP 1.0 was originally merged into the official `mcp` SDK as its high-level
API. FastMCP 2.0+ continued as its own package with additional features.

## Usage across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Used to build two servers:

**General server (port 8000, SSE):** Seven tools spanning computation, file
system, external APIs, and web search. Demonstrates the decorator pattern and
SSE transport for persistent multi-client connections.

**Data analysis server (port 8001, SSE):** Eight stateful tools for loading,
filtering, aggregating, and charting datasets. State (loaded datasets) persists
across tool calls within a session.

### Week 2 — Agentic Search with Tavily (2026-04-03)

Reused as the tool server layer for the Claude web research agent
(`fastmcp_agent_tavily/`). Three Tavily tools exposed on port 8001:
`tavily_search`, `tavily_extract`, `tavily_crawl`. Same SSE transport and
decorator pattern as week 1 — only the tools and port changed.

Confirms the pattern from week 1: FastMCP is a stable, reusable tool layer.
The agent (client side) changes across projects; the server pattern stays the
same.

---

Core pattern:
```python
from fastmcp import FastMCP
mcp = FastMCP("server-name")

@mcp.tool
def my_tool(x: int) -> str:
    """Description the model sees."""
    return str(x)
```
