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
