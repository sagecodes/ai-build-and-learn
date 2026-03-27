"""
Simple Agent using MCP server OpenAI
"""
import asyncio
import os
from dotenv import load_dotenv
from agents import Agent, Runner
from agents.mcp import MCPServerSse

load_dotenv()

# Note: OpenAI Agents SDK uses SSE transport, so this only works with the local server
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/sse")
print(f"Connecting to: {MCP_SERVER_URL}")


async def main():
    # Connect to the already-running FastMCP server
    server = MCPServerSse(
        params={"url": MCP_SERVER_URL}
    )

    async with server:
        agent = Agent(
            name="Assistant",
            instructions="Use the available MCP tools to help answer questions.",
            mcp_servers=[server],
        )

        result = await Runner.run(
            agent,
            "What is 12 + 30? What is 6 * 7? "
            "Greet Sage. "
            "What's the weather in Seattle? "
            "Search DuckDuckGo for 'FastMCP Python' and summarize the top result.",
        )

        print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())
