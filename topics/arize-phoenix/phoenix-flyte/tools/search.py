"""Web search tool using Tavily.

Wrapped as a LangChain @tool so the agent can call it. The OpenInference
LangChain instrumentor captures each invocation as a tool span in Phoenix, so
the search query and results show up nested under the agent's LLM spans.
"""

import logging

from langchain_core.tools import tool
from tavily import TavilyClient

log = logging.getLogger(__name__)


def create_search_tool(tavily_api_key: str):
    """Create a web_search tool bound to a Tavily API key."""
    tavily = TavilyClient(api_key=tavily_api_key)

    @tool
    async def web_search(query: str) -> str:
        """Search the web for information on a topic. Use this to find current facts, data, and sources."""
        log.info(f"Searching: {query}")
        results = tavily.search(query=query, max_results=3)
        formatted = ""
        for r in results.get("results", []):
            formatted += f"- {r['title']}: {r['content'][:300]}\n  {r['url']}\n\n"
        return formatted or "No results found."

    return web_search
