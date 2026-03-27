# MCP with FastMCP

Build an MCP (Model Context Protocol) server using FastMCP and connect to it with the OpenAI Agents SDK.

## What is MCP?

MCP is a standard protocol for exposing tools and context to language models. Instead of every framework inventing its own tool format, MCP gives clients and servers a shared contract for listing capabilities, describing inputs, and calling them.

- Your MCP server exposes capabilities (tools)
- Your model client or agent connects to that server
- The model sees a consistent tool schema
- The model can call the tool without knowing your implementation details

```
┌─────────────────────────────────────────────────────────┐
│                      MCP Protocol                       │
│                                                         │
│  ┌─────────────┐    SSE / stdio    ┌────────────────┐   │
│  │   Client     │◄────────────────►│   MCP Server   │   │
│  │  (Agent)     │                  │   (FastMCP)    │   │
│  └──────┬──────┘                  └───────┬────────┘   │
│         │                                 │             │
│         │  1. tools/list                  │             │
│         │─────────────────────────►       │             │
│         │                                 │             │
│         │  2. tool schemas (JSON)         │             │
│         │◄─────────────────────────       │             │
│         │                                 │             │
│         │  3. tools/call {name, args}     │             │
│         │─────────────────────────►       │             │
│         │                                 │             │
│         │  4. result                      │             │
│         │◄─────────────────────────       │             │
│                                                         │
└─────────────────────────────────────────────────────────┘

┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ OpenAI Agent │   │ Claude Agent │   │ Claude Code  │
│   (client)   │   │   (client)   │   │   (client)   │
└──────┬───────┘   └──────┬───────┘   └──────┬───────┘
       │                  │                   │
       └──────────────────┼───────────────────┘
                          │
                          ▼
                ┌──────────────────┐
                │   MCP Server     │
                │    (FastMCP)     │
                │                  │
                │  - add           │
                │  - multiply      │
                │  - get_weather   │
                │  - duck_duck_go  │
                │  - fetch_webpage │
                │  - greet         │
                └──────────────────┘
```

## What we build

A persistent FastMCP server (over SSE) with several tool types:

| Tool | Type | Description |
|------|------|-------------|
| `add` | Pure computation | Add two integers |
| `multiply` | Pure computation | Multiply two integers |
| `read_text_file` | File system | Read a UTF-8 text file from disk |
| `get_weather` | External API | Get real weather data from wttr.in |
| `duck_duck_go` | Web search | Search DuckDuckGo for web results |
| `fetch_webpage` | Web scraping | Fetch and extract text from a URL |
| `greet` | Context-aware | Greet a user with MCP Context logging |

Then clients using the OpenAI Agents SDK and Anthropic SDK that connect to the running server and use the tools.

### Data Analysis Server

A second, specialized MCP server (`data_server.py`) that demonstrates stateful tools, chaining, and chart generation:

| Tool | Description |
|------|-------------|
| `load_sample_data` | Load a built-in cities dataset into memory |
| `load_csv` | Load custom CSV data |
| `list_datasets` | See what's loaded |
| `describe` | Summary stats (min/max/mean/sum) for numeric columns |
| `filter_rows` | Filter with operators (==, >, <, contains, etc.) |
| `aggregate` | Sum/avg/min/max/count on a column |
| `top_n` | Get top N rows sorted by a column |
| `create_chart` | Generate bar/line/scatter/pie chart as a PNG image |

Tools are **stateful** - datasets persist across calls, so the model can load → filter → chart in sequence.

## Setup

```bash
# Navigate to this topic
cd topics/mcp

# Create virtual environment
uv venv .venv --python 3.11

# Activate the venv
source .venv/bin/activate  # macOS/Linux
# or
.venv\Scripts\activate     # Windows

# Install dependencies
uv pip install -r requirements.txt

# Copy the example env file and add your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Running

The server runs persistently so multiple clients can connect to it.

**Terminal 1 - Start the MCP server:**

```bash
python server.py
```

The server runs on `http://localhost:8000` using SSE transport.

**Terminal 2 - Run a client:**

```bash
# OpenAI Agents SDK client
python openai_client.py

# Claude (Anthropic SDK) client
python claude_client.py
```

Both clients connect to the same running server, discover the available tools, and let the model decide which ones to call. This demonstrates MCP's portability - same server, different model providers.

### Data Analysis Server

```bash
# Terminal: Start the data server (port 8001)
python data_server.py

# Another terminal: Run the data client
python data_client.py
```

The client asks Claude to load data, find top cities, compute averages, and generate a chart. The chart is saved as `chart.png`.

### Chat Interface

A Gradio chat app (`chat_app.py`) that connects to the data analysis server. Ask questions interactively and see charts rendered inline.

```bash
# Make sure data_server.py is running, then:
python chat_app.py
```

Opens at `http://localhost:7860` with example prompts to get started.

### Deploying with Flyte

The server can be deployed to the cloud using [Flyte](https://flyte.org/) via `flyte_app.py`. This wraps the FastMCP server in a Starlette ASGI app and deploys it as a persistent service.

```bash
# Set your app name
export APP_NAME=demo-mcp-server

# Run locally with Flyte
python flyte_app.py

# Deploy to a Flyte cluster
flyte deploy flyte_app.py app_env
```

Once deployed, update `MCP_SERVER_URL` in your `.env` to point at the remote URL:

```
MCP_SERVER_URL=https://your-app-url.hosted.unionai.cloud/mcp
```

Both clients read from `MCP_SERVER_URL`, so they work against local or remote servers without code changes.

## How it works

FastMCP turns Python functions into MCP tools automatically:

- The **function name** becomes the tool name
- The **docstring** becomes the description
- The **type annotations** become the JSON Schema
- The **return value** is serialized automatically

So this function:

```python
@mcp.tool
def add(a: int, b: int) -> int:
    """Add two integers together."""
    return a + b
```

Becomes this tool schema that the model sees:

```json
{
  "name": "add",
  "description": "Add two integers together.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "a": {"type": "integer"},
      "b": {"type": "integer"}
    },
    "required": ["a", "b"]
  }
}
```

## A note on server organization

This demo bundles all tools into a single server for simplicity. In practice, you'd typically split servers by domain:

- `math_server.py` - computation tools
- `search_server.py` - DuckDuckGo search + webpage fetching
- `weather_server.py` - weather API
- `file_server.py` - file system access

MCP clients can connect to multiple servers at once, so each server stays focused and reusable across different agents and projects.

MCP tools can also wrap other agents or LLM calls. A tool is just a function - it could call another model to summarize text, spin up a sub-agent to handle a complex task, or orchestrate a whole pipeline. The calling agent only sees the tool's name, description, and schema. It doesn't know or care what's happening behind it.

## MCP vs baked-in tools: tradeoffs

MCP isn't always the right choice. It's the same monolith vs microservice tradeoff applied to agent tooling.

**When baked-in tools make more sense:**
- Tools are tightly coupled to one specific agent's logic
- You don't need to share them across agents or clients
- You want a single deployable unit with simpler debugging
- Latency matters - no network overhead on every call

**When MCP pays off:**
- Multiple agents or clients share the same tools
- Different teams own different tool sets independently
- You want to swap clients (OpenAI today, Claude tomorrow) without rewriting tools
- You want a persistent tool server that agents connect to on demand

If you only have one agent using one set of tools, MCP adds complexity for no benefit. If you have multiple consumers or want portability across model providers, the standardization is worth it.

## Resources

- [Introducing the Model Context Protocol](https://www.anthropic.com/news/model-context-protocol) - The original announcement from Anthropic
- [Introduction to MCP (Anthropic Course)](https://anthropic.skilljar.com/introduction-to-model-context-protocol) - Free course on building MCP servers and clients from scratch
- [MCP Course (Hugging Face)](https://huggingface.co/learn/mcp-course/en/unit0/introduction) - Community course on MCP

## Community Spotlight

Built something with MCP this week? Share it in [Slack](https://slack.flyte.org/) and we'll add it here!

- [Niels Bantilan Building Your Own MCP Server](https://github.com/unionai/workshops/tree/main/tutorials/mcp) - Hands-on workshop walkthrough


## Next steps

- Add a Claude Code MCP integration example
- Specialized MCP agent
- Gradio example
- Host on HF spaces
- Chat interface example
- Agent run from interface / CLI
- Specialize servers
- LLM call from server