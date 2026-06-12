# Phoenix on Flyte: Trace a LangGraph Research Pipeline

## What is Arize Phoenix?

[Arize Phoenix](https://github.com/Arize-ai/phoenix) is an open-source AI observability platform for tracing, evaluating, and debugging LLM applications. It captures the full lifecycle of your AI calls — prompts, completions, tool use, token counts, and latencies — and gives you a UI to explore them. It also supports LLM-as-a-judge evals and dataset management, so you can go from "what happened" to "how good was it" in one tool. Phoenix uses OpenTelemetry (OTLP) under the hood, so it plugs into any app that can export spans.

## How Phoenix tracing works

Tracing setup lives in [`tracing.py`](tracing.py). Two steps: **register** an OTLP exporter pointed at your Phoenix server, then **instrument** LangChain so every LLM/tool call is captured automatically.

```python
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

# 1. Point the exporter at your Phoenix collector
tracer_provider = register(
    endpoint="http://your-phoenix-server:6006/v1/traces",
    project_name="research-pipeline",
    batch=True,
    set_global_tracer_provider=False,  # keep Flyte's own OTel spans separate
)

# 2. Auto-instrument LangChain/LangGraph — all calls are now traced
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

Then in each task, wrap the agent call so spans get tagged and flushed:

```python
from tracing import setup_tracing, session_scope, flush

async def research_topic(topic, provider="openai"):
    setup_tracing()                          # register + instrument (once per process)
    graph = build_research_subgraph(...)     # plain LangGraph — no tracing code needed
    try:
        with session_scope():                # tags spans with run ID for grouping
            result = await graph.ainvoke({"topic": topic})
    finally:
        flush()                              # drain spans before the process exits
```

The agent code itself stays clean — no tracing decorators or logging needed. The instrumentor captures everything (prompts, tool calls, token counts) automatically.

## What we're building

In this tutorial we self-host Phoenix as a Flyte app and trace a multi-step LangGraph research pipeline. Two tracing views on the same run:

- **Flyte** — orchestration DAG, task compute, logs, HTML reports
- **Phoenix** — LLM prompts, tool calls, token counts, latencies, evals

```
  flyte run workflow.py research_pipeline        flyte app: phoenix-server
  ┌────────────────────────────────────┐         ┌────────────────────────────┐
  │ research_pipeline (orchestrator)    │         │ phoenix serve              │
  │   plan → research → synthesize →   │  OTLP   │  - UI            :6006     │
  │   quality_check → (loop on gaps)   │  HTTP   │  - OTLP collector          │
  │                                    │ ──────▶ │  - SQLite trace store      │
  │ each step is its own Flyte task    │         └────────────────────────────┘
  └────────────────────────────────────┘
```

## Files

| File | Purpose |
| --- | --- |
| `phoenix_app.py` | Self-hosted Phoenix server (Flyte app) |
| `workflow.py` | Pipeline tasks + orchestrator |
| `evaluate.py` | LLM-as-a-judge over captured traces |
| `graph.py` | LangGraph pipeline + ReAct research subgraph |
| `models.py` | Pydantic models |
| `llm.py` | Provider-switchable chat model (`openai` / `vllm`) |
| `tracing.py` | Phoenix OTLP setup + LangChain instrumentor |
| `tools/search.py` | Tavily web-search tool |
| `config.py` | Endpoints, image constants |

## Prerequisites

- A running [Flyte 2 devbox](https://docs.flyte.org/) (`flyte get project` to confirm)
- [uv](https://docs.astral.sh/uv/)
- API keys: [OpenAI](https://platform.openai.com/api-keys), [Tavily](https://app.tavily.com/) (free tier works)

Register secrets on the devbox (one-time, omit `--value` to be prompted):

```bash
flyte create secret OPENAI_API_KEY
flyte create secret TAVILY_API_KEY
```

## Step 1: Set up the local environment

```bash
cd topics/arize-phoenix/phoenix-flyte
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For local runs, create a `.env` with `OPENAI_API_KEY` and `TAVILY_API_KEY`.

## Step 2: Deploy the Phoenix server

```bash
python phoenix_app.py
```

Open the Phoenix UI at: `http://phoenix-server-flytesnacks-development.localhost:30081`

> **Remote access (SSH/Tailscale)?** Forward the ports:
> ```bash
> ssh -L 30081:127.0.0.1:30081 -L 30080:127.0.0.1:30080 <user>@<devbox-ip>
> ```
> Then use the same `localhost` URLs.

## Step 3: Run the pipeline

**Remote** (on the cluster):

```bash
flyte run workflow.py research_pipeline \
  --query "What is the OpenTelemetry Collector and why use it?" \
  --num_topics 2 --max_searches 1 --max_iterations 1
```

**Local** (in-process, still traces to hosted Phoenix):

```bash
flyte run --local workflow.py research_pipeline \
  --query "What is tail-based sampling in tracing?" \
  --num_topics 1 --max_searches 1
```

For local runs, set `PHOENIX_COLLECTOR_ENDPOINT` in `.env` to the host URL (in-cluster DNS isn't reachable from your machine).

## Step 4: Inspect traces in Phoenix

Open the `research-pipeline` project in Phoenix. Each task produces its own trace. Open a `research_topic` trace to see the agent loop:

```
LangGraph
├─ agent → ChatOpenAI       (LLM decides: search or summarize)
├─ tools → web_search        (Tavily call)
├─ agent → ChatOpenAI       (writes the summary)
└─ should_continue           (done)
```

Runs are grouped by **session** (keyed to the Flyte run ID), so each pipeline run appears as one entry in Phoenix's Sessions tab.

## Step 5: Evaluate traces (LLM-as-a-judge)

```bash
flyte run evaluate.py evaluate_traces
```

Scores `research_report` and `LangGraph` spans with `answer_relevance` and `answer_completeness` judges, then writes annotations back to Phoenix. Use `--provider vllm` for the in-cluster model.

## Options

**Switch LLM provider** — use in-cluster vLLM (no API key needed):
```bash
flyte run workflow.py research_pipeline --query "..." --provider vllm
```

**Tune the pipeline:**
```bash
flyte run workflow.py research_pipeline --query "..." \
  --num_topics 3 --max_searches 2 --max_iterations 2
```

**Good demo queries:**
```bash
"What is the OpenTelemetry Collector and why use it?"
"Compare OTLP over gRPC versus HTTP for sending traces."
"What is the Arize Phoenix open-source project used for?"
```

## Teardown

```bash
flyte delete app phoenix-server
```

Ctrl-C on `python phoenix_app.py` only stops the log tail, not the deployed app.

## Troubleshooting

- **"none of the secret managers injected secret"** — run `flyte get secret` and make sure both keys are present.
- **`ModuleNotFoundError` in pod** — keep `graph`/`models`/`llm`/`tracing` imports at the top of `workflow.py` (not lazy inside tasks).
- **Traces missing in Phoenix** — confirm `curl http://phoenix-server-flytesnacks-development.localhost:30081` returns 200. For local runs, check `PHOENIX_COLLECTOR_ENDPOINT` in `.env`.

## Links

- [Phoenix](https://github.com/Arize-ai/phoenix) | [Docs](https://docs.arize.com/phoenix) | [OpenInference](https://github.com/Arize-ai/openinference)
