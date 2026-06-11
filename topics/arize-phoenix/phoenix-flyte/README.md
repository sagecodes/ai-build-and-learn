# Phoenix on Flyte: trace a LangGraph agent, host the UI as a Flyte app

A small, end-to-end LLM observability tutorial. By the end you will have:

1. A self-hosted Arize Phoenix server running as a Flyte app (UI + trace collector).
2. A ReAct agent (LangGraph + a Tavily web-search tool) running as a Flyte task.
3. Every LLM call, tool call, and graph step from that agent traced into Phoenix,
   where you can open the UI and inspect the run span by span.

Phoenix is an open-source AI observability platform (tracing, evals, datasets).
Here we use the tracing half: see exactly what the agent did, and how long each
step took.

Follow the sections in order. Steps 1 to 4 are the run-through; everything after
is options, explanation, and troubleshooting.

## Architecture

```
  flyte run workflow.py research          flyte app: phoenix-server
  ┌─────────────────────────────┐         ┌────────────────────────────┐
  │ ReAct agent (LangGraph)     │  OTLP   │ phoenix serve              │
  │  agent (LLM) -> tools       │  HTTP   │  - UI            :6006      │
  │  -> agent -> answer         │ ───────▶│  - OTLP collector /v1/traces│
  │ instrumented w/ openinference│        │  - SQLite trace store      │
  └─────────────────────────────┘         └────────────────────────────┘
        client side                              server side
   arize-phoenix-otel                         arize-phoenix
```

The agent reaches the collector at the app's cluster-internal Knative DNS name
(`http://phoenix-server-flytesnacks-development.flyte.svc.cluster.local`). One
`phoenix serve` process serves both the UI and the OTLP-HTTP collector on port
6006, so a single Knative HTTP route covers both. No sidecar, no separate gRPC.

## Files

| File | What it is |
| --- | --- |
| `phoenix_app.py` | The self-hosted Phoenix server, as a Flyte app. |
| `workflow.py` | The Flyte task: wires tracing, runs the agent, flushes spans. |
| `agent.py` | The LangGraph ReAct agent (LLM + Tavily), provider-switchable. |
| `tools/search.py` | The Tavily web-search tool. |
| `config.py` | Flyte envs, the collector endpoint, vLLM + image constants. |

## Prerequisites

- A running Flyte 2 devbox you can reach (this demo targets the local DGX Spark
  devbox at `localhost:30080`, arm64, registry `localhost:30000`). Confirm with
  `flyte get project`.
- [uv](https://docs.astral.sh/uv/) for the local virtual environment.
- Two secrets registered on the devbox: `OPENAI_API_KEY` and `TAVILY_API_KEY`.
  Check with `flyte get secret`. To register a missing one:

  ```bash
  flyte create secret OPENAI_API_KEY --value sk-...
  flyte create secret TAVILY_API_KEY --value tvly-...
  ```

  (The `--provider vllm` option needs neither key; see Options.)

## Step 1: Set up the local environment

Locally you only need `flyte` plus the agent's import-time libs, so that
`flyte run` bundles `agent.py` and `tools/` into the run. The tracing and Phoenix
packages build into the remote images, not your shell.

```bash
cd topics/arize-phoenix/phoenix-flyte
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "flyte[tui]>=2.0" langgraph langchain-openai tavily-python python-dotenv
```

The agent task reads its keys from the Flyte secrets above when it runs on the
cluster. For a local `python workflow.py` run instead, copy `.env.example` to
`.env` and fill in `OPENAI_API_KEY` and `TAVILY_API_KEY`.

## Step 2: Deploy the Phoenix server

```bash
python phoenix_app.py
```

This builds the server image (`arize-phoenix`), deploys the app, and streams its
logs. It prints a console link, and the app is served at its Knative URL:

```
http://phoenix-server-flytesnacks-development.localhost:30081
```

Open that URL. You should see an empty Phoenix UI with a `default` project,
waiting for traces. Leave this process running (it just tails logs; the app is a
cluster resource and stays up on its own).

## Step 3: Run the traced agent

In a second terminal (same venv, same directory):

```bash
source .venv/bin/activate
flyte run workflow.py research --question "What is OpenTelemetry and what problem does it solve?"
```

You will see the image build (cached after the first run), a `Created Run: ...`
banner with a Flyte console URL, and the task logs as it works:

```
[research] provider=openai question='What is OpenTelemetry ...'
[agent] tool call: web_search({'query': '...'})
Searching: ...
[research] done (NNN chars)
```

The task instruments the agent, runs it, and flushes the spans to Phoenix before
the pod exits. It returns a JSON string with the question, the answer, and the
provider used.

## Step 4: Inspect the trace in Phoenix

Refresh the Phoenix UI and switch to the `langgraph-tavily-agent` project. The
run appears as a single trace; expand it to see the full ReAct loop nested, with
latency and token counts on each span:

```
LangGraph                (the graph run)
├─ agent                 (LLM decides: search or answer)
│  └─ ChatOpenAI         (the LLM call)
├─ should_continue
├─ tools
│  └─ web_search         (the Tavily call: query in, results out)
├─ agent
│  └─ ChatOpenAI
└─ should_continue       (answer ready -> end)
```

Click any span to see its inputs and outputs: the exact prompt sent to the LLM,
the tool's search query and returned text, and the final answer. That is the
whole point: the agent code has no logging in it, yet you can replay every step.

You now have the full loop working: hosted collector, traced agent, live trace.
Run the agent again with a different question and watch new traces stream in.

## Options

### Switch the agent's LLM

The agent defaults to OpenAI (`gpt-4.1-nano`). Flip it to the in-cluster gemma4
vLLM app (open-source, no API key) with a flag:

```bash
flyte run workflow.py research --question "..." --provider vllm
```

Both providers go through the same `ChatOpenAI` client, so the trace shape is
identical; only the model behind it changes.

### Tune the search budget

`--max_searches N` caps how many times the agent may call Tavily (default 3).
More searches means a longer, richer trace.

```bash
flyte run workflow.py research --question "..." --max_searches 5
```

### Good demo questions

Pick questions with a clear, searchable answer so the trace tells a clean story:

```bash
flyte run workflow.py research --question "What is OpenTelemetry and what problem does it solve?"
flyte run workflow.py research --question "What is the Arize Phoenix open-source project used for?"
flyte run workflow.py research --question "Compare OTLP over gRPC versus HTTP for sending traces."
```

(Avoid ambiguous product names: "Flyte 2" alone, for example, surfaces unrelated
sportswear in web search.)

## Persistence

This demo stores traces in SQLite under `PHOENIX_WORKING_DIR` and pins the app to
a single always-on replica, so traces survive for the whole session. The pod's
`/tmp` resets on restart, so it is session-scoped, not durable.

For production, point Phoenix at Postgres and the server becomes stateless:

```python
# in phoenix_app.py, AppEnvironment(env_vars={...})
"PHOENIX_SQL_DATABASE_URL": "postgresql://user:pass@host:5432/phoenix",
```

Then you can drop the always-on pin and let it scale to zero between sessions.

## How the tracing is wired

The client side is three lines, in `workflow.py`:

```python
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(endpoint=COLLECTOR + "/v1/traces", project_name="...", batch=True)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

`register()` points the OTLP exporter at the hosted collector. The LangChain
instrumentor patches LangChain and LangGraph (LangGraph runs on LangChain
runnables, so there is no separate LangGraph instrumentor). The agent code stays
a plain LangGraph app with no tracing calls in it. Before the task returns we
call `tracer_provider.force_flush()`, because Flyte pods are short-lived and an
un-flushed batch of spans would never reach Phoenix.

## Teardown

The agent task is one-shot; nothing to stop there. The Phoenix app keeps running
(and holding its SQLite traces) until you remove it:

```bash
flyte delete app phoenix-server
```

You can Ctrl-C the local `python phoenix_app.py` process any time; that only
stops the log tail, not the deployed app.

## Troubleshooting

- **Pod admission denied, "none of the secret managers injected secret"**: a
  requested secret does not exist on the devbox. Run `flyte get secret` and make
  sure `OPENAI_API_KEY` and `TAVILY_API_KEY` are present (note: the OpenAI secret
  is `OPENAI_API_KEY`, not `SAGE_OPENAI_API_KEY`).
- **`ModuleNotFoundError: No module named 'agent'` in the pod**: `flyte run` only
  bundles modules that are imported by the time the script loads. Keep the
  `from agent import build_agent` import at the top of `workflow.py` (not lazily
  inside the task), and keep its libs installed in the submit venv from Step 1.
- **Trace does not appear in Phoenix**: confirm the app is reachable
  (`curl -s -o /dev/null -w "%{http_code}" http://phoenix-server-flytesnacks-development.localhost:30081`
  returns `200`), and that the task log shows the `[tracing] exporting to ...`
  line. The agent must run on the cluster (remote) so it can resolve the
  collector's in-cluster DNS name.

## Links

- Phoenix: https://github.com/Arize-ai/phoenix
- Phoenix docs: https://docs.arize.com/phoenix
- OpenInference instrumentors: https://github.com/Arize-ai/openinference
