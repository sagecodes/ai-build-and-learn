"""
Gradio chat interface for the Data Analysis MCP server.
Lets you interactively ask questions and see charts inline.
"""
import argparse
import os
import base64
import tempfile
import threading
import asyncio
from dotenv import load_dotenv
from anthropic import Anthropic
from fastmcp import Client
import gradio as gr

load_dotenv()

LOCAL_URL = "http://localhost:8001/sse"

parser = argparse.ArgumentParser()
parser.add_argument("--remote", type=str, help="Use a remote server URL")
args = parser.parse_args()

DATA_SERVER_URL = args.remote or os.getenv("DATA_SERVER_URL", LOCAL_URL)
print(f"Connecting to: {DATA_SERVER_URL}")

anthropic_client = Anthropic()
anthropic_tools = []

# Run a persistent event loop in a background thread for MCP
_loop = asyncio.new_event_loop()
_mcp_client = Client(DATA_SERVER_URL)


def _start_loop(loop):
    asyncio.set_event_loop(loop)
    loop.run_forever()


_thread = threading.Thread(target=_start_loop, args=(_loop,), daemon=True)
_thread.start()


def _run_async(coro):
    """Run an async coroutine on the background event loop."""
    future = asyncio.run_coroutine_threadsafe(coro, _loop)
    return future.result()


# Connect to MCP server on the background loop
_run_async(_mcp_client.__aenter__())
tools = _run_async(_mcp_client.list_tools())
anthropic_tools = [
    {
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.inputSchema,
    }
    for tool in tools
]
print(f"Connected to MCP server. Found {len(anthropic_tools)} tools.")


def chat(message, history):
    """Process a chat message using Claude + MCP tools."""
    messages = []
    for msg in history:
        if isinstance(msg, dict):
            if msg.get("role") == "user" and isinstance(msg.get("content"), str):
                messages.append({"role": "user", "content": msg["content"]})
            elif msg.get("role") == "assistant" and isinstance(msg.get("content"), str):
                messages.append({"role": "assistant", "content": msg["content"]})
    messages.append({"role": "user", "content": message})

    chart_paths = []

    while True:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2048,
            system="You are a data analysis assistant. Use the available MCP tools to help answer questions about data. When showing charts, always describe what the chart shows.",
            tools=anthropic_tools,
            messages=messages,
        )

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"Calling tool: {block.name}({block.input})")
                    result = _run_async(_mcp_client.call_tool(block.name, block.input))
                    result_str = str(result)

                    # Save chart images
                    if "data:image/png;base64," in result_str:
                        start = result_str.find("data:image/png;base64,") + len("data:image/png;base64,")
                        img_data = result_str[start:].strip("'\"")
                        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False, dir=".")
                        tmp.write(base64.b64decode(img_data))
                        tmp.close()
                        chart_paths.append(os.path.abspath(tmp.name))
                        result_str = "Chart generated successfully."

                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_str,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})
        else:
            # Collect text response
            text_parts = []
            for block in response.content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)

            # Return text + chart images as ChatMessage list
            response_messages = []
            if text_parts:
                response_messages.append(
                    gr.ChatMessage(role="assistant", content="\n".join(text_parts))
                )
            for path in chart_paths:
                response_messages.append(
                    gr.ChatMessage(
                        role="assistant",
                        content=gr.FileData(path=path, mime_type="image/png"),
                    )
                )

            return response_messages


demo = gr.ChatInterface(
    fn=chat,
    title="Data Analysis Chat",
    description="Chat with an AI data analyst powered by MCP tools.",
    examples=[
        "Load the sample dataset and describe it",
        "Show me the top 5 cities by population",
        "Filter cities with temperature above 60 and make a bar chart",
        "Create a pie chart of tech companies by city",
    ],
)

if __name__ == "__main__":
    demo.launch()
