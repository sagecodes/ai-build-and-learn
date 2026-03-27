"""
Claude client for the Data Analysis MCP server.
Demonstrates stateful tool chaining: load → filter → aggregate → chart.
"""
import asyncio
import os
import json
import base64
from dotenv import load_dotenv
from anthropic import Anthropic
from fastmcp import Client

load_dotenv()

DATA_SERVER_URL = os.getenv("DATA_SERVER_URL", "http://localhost:8001/sse")


async def main():
    mcp_client = Client(DATA_SERVER_URL)

    async with mcp_client:
        tools = await mcp_client.list_tools()

        anthropic_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema,
            }
            for tool in tools
        ]

        client = Anthropic()
        messages = [
            {
                "role": "user",
                "content": (
                    "Load the sample dataset. "
                    "Show me the top 5 cities by population. "
                    "What's the average temperature across all cities? "
                    "Create a bar chart of population by city."
                ),
            }
        ]

        while True:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                tools=anthropic_tools,
                messages=messages,
            )

            if response.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": response.content})

                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        print(f"Calling tool: {block.name}({json.dumps(block.input)})")
                        result = await mcp_client.call_tool(block.name, block.input)
                        result_str = str(result)

                        # Save chart images to file
                        if "data:image/png;base64," in result_str:
                            img_data = result_str.split("data:image/png;base64,")[1].strip("'\"")
                            with open("chart.png", "wb") as f:
                                f.write(base64.b64decode(img_data))
                            print("  → Chart saved to chart.png")
                            result_str = "Chart generated and saved to chart.png"

                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_str,
                            }
                        )

                messages.append({"role": "user", "content": tool_results})
            else:
                for block in response.content:
                    if hasattr(block, "text"):
                        print(block.text)
                break


if __name__ == "__main__":
    asyncio.run(main())
