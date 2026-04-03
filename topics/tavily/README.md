# Agentic Search with Tavily

[Tavily](https://tavily.com/) is a search API built for AI agents and LLM applications. Unlike general-purpose search engines, Tavily returns clean, structured results optimized for LLM consumption — no scraping, parsing, or post-processing needed.

## Why Tavily?

- **AI-native search** — Returns concise content snippets rather than raw HTML, so your agent can reason over results directly.
- **Single API call** — One request returns titles, content summaries, and source URLs. No need to chain a search API + scraper + content extractor.
- **Built-in relevance filtering** — Results are ranked and deduplicated, reducing noise for your agent.
- **Rate-limit friendly** — Designed for programmatic/agent use, unlike browser-oriented APIs that throttle automated requests.

## Tavily API Capabilities

### Search
The core endpoint. Send a query, get back ranked results with titles, content snippets, and URLs. Supports basic and advanced search depth, topic filters (general, news, finance), time ranges, domain filtering, and optional AI-generated answers. Use this when your agent needs to find information on a topic from the open web.

### Extract
Pull clean, structured content from specific URLs. Useful when your agent already knows *where* to look (e.g. from a prior search result) and needs the full page content in markdown or text. Supports batch extraction (up to 20 URLs), targeted chunks via a query parameter, and basic/advanced depth.

### Crawl
Crawl an entire website following links from a starting URL. You control depth, breadth, and can filter by path patterns. Supports natural language instructions like "Find pages about pricing." Use this when you need to explore a site systematically — for example, indexing documentation or gathering all product pages.

### Map
Discover the structure of a website without extracting content. Returns a list of URLs found by following links from the starting page. Lighter weight than crawl — use it when you just need to know what pages exist (e.g. before deciding which ones to extract or crawl in detail).

## Getting a Tavily API Key

1. Sign up at [tavily.com](https://tavily.com/)
2. Copy your API key from the dashboard
3. Add it to your `.env` file:

```
TAVILY_API_KEY=your-key-here
```

## Examples

| Example | Description |
|---------|-------------|
| [tavily-usage-examples](tavily-usage-examples/) | Core Tavily API usage — basic search, extract, and configuration options |
| [langgraph_agent_research](langgraph_agent_research/) | Full research agent pipeline using Tavily + LangGraph + Flyte with parallel sub-topic research, synthesis, and quality gates |