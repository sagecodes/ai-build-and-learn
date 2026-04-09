"""
Tavily content extraction tool — action: tavily_extract

Wraps the Tavily extract API as a discrete action in the research
RL environment. Registered on the MCPEnvironment via @mcp.tool in
research_env.py.

Use this action when the agent already has specific URLs from a
tavily_search step and needs full page content rather than snippets.
"""

from tavily import TavilyClient

from env.tools.common import tavily_call_with_retry


def run_extract(
    client: TavilyClient,
    urls: list[str],
    extract_depth: str = "basic",
) -> dict:
    """Extract structured full-page content from specific URLs.

    Use this tool when you already have URLs (e.g. from tavily_search) and
    need the full readable content of those pages rather than just snippets.
    More precise than crawling when you have a known list of pages to read.

    Args:
        urls: List of URLs to extract content from (1-20 URLs).
        extract_depth: "basic" for main content, "advanced" for deeper
                       extraction including tables and structured data.
    """
    def _call():
        response = client.extract(
            urls=urls,
            extract_depth=extract_depth,
        )
        return {
            "results": [
                {
                    "url": r.get("url", ""),
                    "raw_content": r.get("raw_content", ""),
                }
                for r in response.get("results", [])
            ],
            "failed_results": response.get("failed_results", []),
        }

    return tavily_call_with_retry(_call, on_error={"results": []})
