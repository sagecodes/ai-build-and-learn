"""
Simple Agent using MCP server with Claude (Anthropic SDK)
"""
import argparse
import asyncio
import json
import os
from dotenv import load_dotenv
from anthropic import Anthropic
from fastmcp import Client

load_dotenv()

LOCAL_URL = "http://localhost:8000/sse"
REMOTE_URL = "https://rapid-grass-86f44.apps.demo.hosted.unionai.cloud/mcp"

parser = argparse.ArgumentParser()
parser.add_argument("--remote", action="store_true", help="Use the deployed remote server")
args = parser.parse_args()

MCP_SERVER_URL = REMOTE_URL if args.remote else os.getenv("MCP_SERVER_URL", LOCAL_URL)
print(f"Connecting to: {MCP_SERVER_URL}")


async def main():
    # Connect to the already-running FastMCP server
    mcp_client = Client(MCP_SERVER_URL)

    async with mcp_client:
        # Discover available tools from the MCP server
        tools = await mcp_client.list_tools()

        # Convert MCP tools to Anthropic tool format
        anthropic_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

        # Set up the Anthropic client
        client = Anthropic()
        messages = [
            {
                "role": "user",
                "content": (
                    "What is 12 + 30? What is 6 * 7? "
                    "Greet Sage. "
                    "What's the weather in Seattle? "
                    "Search DuckDuckGo for 'FastMCP Python' and summarize the top result."
                ),
            }
        ]

        # Agentic loop: keep calling the model until it stops using tools
        while True:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=anthropic_tools,
                messages=messages,
            )

            # Check if the model wants to use tools
            if response.stop_reason == "tool_use":
                # Add assistant response to messages
                messages.append({"role": "assistant", "content": response.content})

                # Process each tool call
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"Calling tool: {block.name}({json.dumps(block.input)})")
                        result = await mcp_client.call_tool(block.name, block.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": str(result),
                            }
                        )

                messages.append({"role": "user", "content": tool_results})
            else:
                # No more tool calls - print final text response
                for block in response.content:
                    if hasattr(block, "text"):
                        print(block.text)
                break


if __name__ == "__main__":
    asyncio.run(main())
