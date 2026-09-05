---
title: Tavily
weeks: [tavily]
---

AI-native search API built for LLM agents. Returns clean, structured results
(titles, content snippets, source URLs) optimized for LLM consumption — no
scraping, parsing, or post-processing. Designed for programmatic use rather
than browser-oriented rate limits.

Four endpoints:
- **Search** — query → ranked results with snippets
- **Extract** — URL(s) → full page content (batch up to 20 URLs)
- **Crawl** — root URL → follow links to a configured depth/breadth
- **Map** — root URL → list of discoverable URLs (lightweight, no content)

## Usage across the series

### Week 2 — Agentic Search with Tavily (2026-04-03)

The central tool of the week. Used in two distinct integration patterns:

**Via FastMCP (Claude agent).** Three Tavily tools exposed on a FastMCP server:
`tavily_search` (discovery), `tavily_extract` (full content from known URLs),
`tavily_crawl` (site-wide exploration). The agent chains them: search first for
discovery, extract when snippets aren't enough, crawl for documentation sites.

**Via LangChain `@tool` (LangGraph pipeline).** `web_search` wraps
`TavilyClient` as a LangChain-compatible tool so LangGraph's ReAct agent can
call it in a loop. Each researcher gets up to `max_searches` calls.

The core value proposition in both cases: Tavily's structured output means the
agent spends its token budget on reasoning rather than parsing noisy HTML.

### Week 3 — OpenEnv (2026-04-10)

Used as the action space inside the OpenEnv research environment. The three
Tavily tools (`tavily_search`, `tavily_extract`, `tavily_crawl`) are registered
as environment actions that agents discover and call via `step()`. The Claude
ReAct agent discovers all three dynamically; the traditional agent is hardcoded
to `tavily_search` only — this restriction is central to the reward hacking demo.
