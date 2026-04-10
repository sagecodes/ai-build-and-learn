"""
Tavily web search tool — action: tavily_search

Wraps the Tavily search API as a discrete action in the research
RL environment. Registered on the MCPEnvironment via @mcp.tool in
research_env.py.

This is the agent's primary discovery action — use it first to find
relevant URLs and get an overview of a topic.
"""

from typing import Optional
from tavily import TavilyClient

from env.tools.common import tavily_call_with_retry


def run_search(
    client: TavilyClient,
    query: str,
    max_results: int = 5,
    search_depth: str = "basic",
    include_domains: Optional[list[str]] = None,
    exclude_domains: Optional[list[str]] = None,
) -> dict:
    """Search the web using Tavily and return ranked results with snippets.

    Use this tool first when you need to discover relevant URLs or get an
    overview of a topic. Returns titles, URLs, content snippets, and
    relevance scores.

    Args:
        query: The search query string.
        max_results: Number of results to return (default: 5, max: 20).
        search_depth: "basic" for fast overview, "advanced" for deeper
                      coverage (default: "basic").
        include_domains: Restrict results to these domains only.
        exclude_domains: Exclude results from these domains.
    """
    def _call():
        response = client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth,
            include_domains=include_domains or [],
            exclude_domains=exclude_domains or [],
        )
        return {
            "query": query,
            "results": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("url", ""),
                    "content": r.get("content", ""),
                    "score": r.get("score", 0.0),
                }
                for r in response.get("results", [])
            ],
        }

    return tavily_call_with_retry(_call, on_error={"query": query, "results": []})
