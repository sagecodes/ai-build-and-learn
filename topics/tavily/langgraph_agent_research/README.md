# Research Agent Pipeline

A research agent pipeline that uses [Tavily](https://tavily.com/) as the web search backbone inside a LangGraph + Flyte co-orchestrated system. LangGraph controls the pipeline logic — planning, fan-out, quality gates, and iterative deepening. Flyte provides the compute — each researcher runs as a separate task with its own container, resources, and observability.

## What is Tavily?

[Tavily](https://tavily.com/) is a search API built specifically for AI agents and LLM applications. Unlike general-purpose search engines, Tavily returns clean, structured results optimized for LLM consumption — no scraping, parsing, or post-processing needed.

**Why Tavily for research agents:**

- **AI-native search** — Returns concise, relevant content snippets rather than raw HTML or ad-heavy pages, so your agent can reason over results directly.
- **Single API call** — One request returns titles, content summaries, and source URLs. No need to chain a search API + scraper + content extractor.
- **Built-in relevance filtering** — Results are ranked and deduplicated for the query, reducing the amount of noise your agent has to sift through.
- **Rate-limit friendly** — Designed for programmatic/agent use, unlike browser-oriented APIs that throttle automated requests.

**How this project uses Tavily:**

The Tavily integration lives in `tools/search.py`. It wraps the `TavilyClient` as a LangChain-compatible `@tool` so LangGraph's ReAct agent can call it in a loop:

```python
from tavily import TavilyClient
from langchain_core.tools import tool

tavily = TavilyClient(api_key="your-key")

@tool
async def web_search(query: str) -> str:
    """Search the web for information on a topic."""
    results = tavily.search(query=query, max_results=3)
    formatted = ""
    for r in results.get("results", []):
        formatted += f"- {r['title']}: {r['content'][:300]}\n  {r['url']}\n\n"
    return formatted or "No results found."
```

Each `research_topic` ReAct agent calls `web_search` in a loop — the agent decides what to search, reads the results, and either searches again or stops when it has enough information. Tavily's structured output means the agent spends its token budget on reasoning, not on parsing noisy web pages.

## Architecture

```
research_pipeline (LangGraph pipeline graph, inside a Flyte task)
  ├── plan → split query into sub-topics
  ├── research (Send fan-out → Flyte tasks)
  │     ├── research_topic("topic A")  ┐
  │     ├── research_topic("topic B")  ├── parallel Flyte tasks, each running a ReAct agent
  │     └── research_topic("topic C")  ┘
  ├── synthesize → combine into report
  ├── quality_check → score + identify gaps
  │     ├── gaps found → identify_gaps → Send fan-out → research again
  │     └── good enough → finalize
  └── finalize → final report
```

Each `research_topic` task runs a LangGraph ReAct agent that searches the web via [Tavily](https://tavily.com/) and loops until it has enough information.

## Setup

```bash
cd topics/tavily/langgraph_agent_research
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Add your API keys to `.env`:

```
OPENAI_API_KEY=your-key-here
TAVILY_API_KEY=your-key-here
```

## Run

```bash
# Local with TUI
flyte run --local --tui workflow.py research_pipeline \
  --query "Compare quantum computing approaches: superconducting vs trapped ion"

# Local without TUI & more parameters set
flyte run --local workflow.py research_pipeline \
  --query "What are the pros and cons of electric vehicles?" \
  --num_topics 2 --max_searches 1

# Remote (on a Flyte cluster)
flyte run workflow.py research_pipeline \
  --query "Compare quantum computing approaches" \
  --num_topics 5 --max_searches 3 --max_iterations 3
```

| Flag | Default | Description |
|------|---------|-------------|
| `--query` | required | Research question |
| `--num_topics` | 3 | Number of sub-topics to research in parallel |
| `--max_searches` | 2 | Max web searches per sub-topic |
| `--max_iterations` | 2 | Max quality gate iterations |

## Gradio App

A Gradio UI (`app.py`) that kicks off the pipeline and links to the Flyte run:

```bash
# Local app + local task
RUN_MODE=local python app.py

# Local app + remote task (on a Flyte cluster)
python app.py

# Deploy to cluster
flyte deploy app.py serving_env
```

## Project Structure

```
langgraph_agent_research/
├── app.py              # Gradio UI — kicks off pipeline, links to Flyte run
├── config.py           # Flyte environment, secrets, resources
├── graph.py            # LangGraph graphs — pipeline + ReAct subgraph
├── workflow.py         # Flyte tasks — research_topic + research_pipeline orchestrator
├── requirements.txt
└── tools/
    └── search.py       # Tavily web search tool
```

## How It Works

- **`graph.py`** defines two LangGraph graphs:
  - `build_research_subgraph()` — ReAct agent loop (agent ↔ tools) for a single topic
  - `build_pipeline_graph()` — pipeline graph (plan → Send fan-out → synthesize → quality check → loop)
- **`workflow.py`** defines two Flyte tasks:
  - `research_topic` — runs the ReAct subgraph on one topic (the compute unit)
  - `research_pipeline` — runs the pipeline graph, passing `research_topic` as the compute backend

The pipeline graph accepts the Flyte task as a parameter. LangGraph's `Send` API fans out work to it. On a cluster, each `Send` becomes a separate container.

See the [blog post](blog.md) for the full walkthrough.