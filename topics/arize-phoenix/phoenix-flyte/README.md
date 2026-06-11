# Phoenix on Flyte: trace a LangGraph research pipeline, host the UI as a Flyte app

A hands-on LLM observability tutorial, and a side-by-side look at two kinds of
tracing on the same run. By the end you will have:

1. A self-hosted Arize Phoenix server running as a Flyte app (UI + trace collector).
2. A multi-step research pipeline (LangGraph + Tavily) where each step runs as its
   own Flyte task: plan, research (one ReAct agent per sub-topic), synthesize,
   quality-check, looping until the report is good enough.
3. Every step traced two ways at once:
   - **Flyte** stitches the whole pipeline DAG together: tasks, compute, logs,
     HTML reports, and `@flyte.trace` spans.
   - **Phoenix** captures the LLM-native detail inside each task: prompts, tool
     calls, token counts, and latencies, ready for evals.

Phoenix is an open-source AI observability platform (tracing, evals, datasets).
Comparing the two views is the point: Flyte shows you the orchestration; Phoenix
shows you what the model actually did.

Follow the sections in order. Steps 1 to 4 are the run-through; everything after
is options, the comparison, and troubleshooting.

## Architecture

```
  flyte run workflow.py research_pipeline        flyte app: phoenix-server
  ┌────────────────────────────────────┐         ┌────────────────────────────┐
  │ research_pipeline (orchestrator)    │         │ phoenix serve              │
  │   LangGraph controls the flow:      │  OTLP   │  - UI            :6006      │
  │   plan → research → synthesize →    │  HTTP   │  - OTLP collector /v1/traces│
  │   quality_check → (loop on gaps)    │ ──────▶ │  - SQLite trace store      │
  │                                     │         └────────────────────────────┘
  │ each step is its own Flyte task:    │
  │   plan_topics                       │   every task instruments LangChain,
  │   research_topic   (ReAct, ×N)      │   so its LLM + tool spans stream to
  │   synthesize                        │   the hosted Phoenix collector
  │   quality_check                     │
  └────────────────────────────────────┘
        client side                              server side
   arize-phoenix-otel                         arize-phoenix
```

The tasks reach the collector at the app's cluster-internal Knative DNS name
(`http://phoenix-server-flytesnacks-development.flyte.svc.cluster.local`). One
`phoenix serve` process serves both the UI and the OTLP-HTTP collector on port
6006, so a single Knative HTTP route covers both. No sidecar, no separate gRPC.

## Files

| File | What it is |
| --- | --- |
| `phoenix_app.py` | The self-hosted Phoenix server, as a Flyte app. |
| `workflow.py` | The four pipeline tasks + the `research_pipeline` orchestrator. |
| `graph.py` | The LangGraph pipeline graph + the ReAct research subgraph. |
| `models.py` | Pydantic models passed between tasks. |
| `llm.py` | Provider-switchable chat model (`openai` or `vllm`). |
| `tracing.py` | Phoenix setup: `register()` + the LangChain instrumentor, per task. |
| `tools/search.py` | The Tavily web-search tool (`@flyte.trace`-d too). |
| `config.py` | Flyte envs, the collector endpoint, vLLM + image constants. |

## Prerequisites

- A running Flyte 2 devbox you can reach (this demo targets the local DGX Spark
  devbox at `localhost:30080`, arm64, registry `localhost:30000`). Confirm with
  `flyte get project`.
- [uv](https://docs.astral.sh/uv/) for the local virtual environment.
- Two secrets registered on the devbox: `OPENAI_API_KEY` and `TAVILY_API_KEY`.
  List what is already there with `flyte get secret`, then add whichever are
  missing (one-time per devbox; a fresh or recreated devbox starts with none):

  ```bash
  flyte create secret OPENAI_API_KEY --value sk-...      # your OpenAI key
  flyte create secret TAVILY_API_KEY --value tvly-...    # your Tavily key
  ```

  To avoid the value landing in your shell history, omit `--value` and you will
  be prompted for it:

  ```bash
  flyte create secret OPENAI_API_KEY
  # Enter secret value: ...
  ```

  Created this way (no `--project`/`--domain`), the secret is **global**: it
  shows as `-`/`-` in `flyte get secret` and is available to every project and
  domain, including the `flytesnacks/development` where these tasks run. That is
  all this demo needs; you do not have to scope it. To scope it instead, add
  `-p flytesnacks -d development` (a scoped secret is only visible to tasks in
  that exact project/domain).

  Grab the keys here if you need them: OpenAI <https://platform.openai.com/api-keys>,
  Tavily <https://app.tavily.com/> (free tier is plenty for this demo). The
  `--provider vllm` option needs neither key; see Options.

## Step 1: Set up the local environment

Locally you only need `flyte` plus the pipeline's import-time libs, so that
`flyte run` bundles `graph.py`, `models.py`, `llm.py`, `tracing.py`, and `tools/`
into the run. The tracing exporter and Phoenix server packages build into the
remote images, not your shell.

```bash
cd topics/arize-phoenix/phoenix-flyte
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install "flyte[tui]>=2.0" langgraph langchain-openai tavily-python markdown python-dotenv
```

The tasks read their keys from the Flyte secrets above when they run on the
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

### Accessing the UIs remotely (SSH / Tailscale)

The app is served by the Knative ingress (Kourier), which routes purely by the
`...localhost` Host header on port 30081. If you are working directly on the
devbox the URL above just works. If you are remoted in (for example over
Tailscale), `localhost` points at your own machine, not the devbox, and hitting
the devbox IP directly returns 404 even with the right Host header. Forward the
port instead, then use the same hostname URL unchanged:

```bash
# from your laptop; forwards the Phoenix UI (30081) and the Flyte console (30080)
ssh -L 30081:127.0.0.1:30081 -L 30080:127.0.0.1:30080 <user>@<devbox-ip>
```

Leave that open and browse `http://phoenix-server-flytesnacks-development.localhost:30081/`
on your laptop. The `.localhost` name resolves to `127.0.0.1` locally, the tunnel
carries it to the devbox loopback, and Kourier matches the Host header. In VS Code
Remote-SSH you can instead add `30081` (and `30080`) in the Ports panel. Do not
put the devbox IP in the URL or in any config; only in the tunnel command.

## Step 3: Run the traced pipeline

In a second terminal (same venv, same directory):

```bash
source .venv/bin/activate
flyte run workflow.py research_pipeline \
  --query "What is the OpenTelemetry Collector and why use it?" \
  --num_topics 2 --max_searches 1 --max_iterations 1
```

(The small numbers keep the first run quick. The defaults are `num_topics 3`,
`max_searches 2`, `max_iterations 2`.) You will see the image build (cached after
the first run), a `Created Run: ...` banner with a Flyte console URL, and then the
pipeline fanning out across tasks. Open that console URL to watch the DAG: a
`research_pipeline` action with `plan_topics`, two `research_topic` actions,
`synthesize-1`, and `quality-1` underneath, each with its own logs and report.

Each task calls `setup_tracing()`, runs its LLM/tool work, and flushes spans to
Phoenix before its pod exits. The orchestrator returns a `PipelineResult` with the
final report, sub-reports, score, and iteration count.

## Step 4: Inspect the traces in Phoenix

Refresh the Phoenix UI and open the `research-pipeline` project. Because each step
ran in its own task pod (its own process), Phoenix shows several traces per run,
one or more per task: the `plan_topics`, `synthesize`, and `quality_check` LLM
calls, plus one ReAct trace per `research_topic`. Open a `research_topic` trace to
see the agent loop nested, with latency and token counts on each span:

```
LangGraph                (the research subgraph run)
├─ agent                 (LLM decides: search or summarize)
│  └─ ChatOpenAI         (the LLM call)
├─ should_continue
├─ tools
│  └─ web_search         (the Tavily call: query in, results out)
├─ agent
│  └─ ChatOpenAI         (writes the summary)
└─ should_continue       (done -> end)
```

Click any span to see its inputs and outputs: the exact prompt, the tool's search
query and returned text, the model's response. The pipeline code has no logging in
it, yet you can replay every model decision.

## Compare: Flyte tracing vs Phoenix tracing

This is the reason both are wired up. The same run, two lenses:

- **Flyte console** (the run URL from Step 3): the orchestration view. The pipeline
  DAG, which task ran where, how long each took, retries, the HTML reports each
  task writes (`flyte.report`), and `@flyte.trace` spans (`agent`, `web_search`)
  in each task's timeline. This is where you see compute, parallelism, and the
  shape of the workflow.
- **Phoenix** (`research-pipeline` project): the model view. The prompts and
  completions, tool inputs/outputs, token usage, and per-call latency, in a layout
  built for LLM debugging and evals. This is where you see *what the model did and
  why*, and where you would later run evals on the captured spans.

The same `web_search` and `agent` steps appear in both, because they are both
`@flyte.trace`-d and OpenInference-instrumented. Note that Phoenix and Flyte both
use OpenTelemetry; `tracing.py` registers Phoenix with
`set_global_tracer_provider=False` so Flyte's own orchestration spans stay in the
Flyte UI and do not leak into Phoenix.

## Options

### Switch the LLM provider

The pipeline defaults to OpenAI (`gpt-4.1-nano`). Flip every task to the
in-cluster gemma4 vLLM app (open-source, no API key) with one flag; it threads
through plan, research, synthesize, and quality:

```bash
flyte run workflow.py research_pipeline --query "..." --provider vllm
```

Both providers go through the same `ChatOpenAI` client, so the trace shape is
identical; only the model behind it changes.

### Tune the pipeline

```bash
flyte run workflow.py research_pipeline --query "..." \
  --num_topics 3 \      # sub-topics to fan out (more research_topic tasks)
  --max_searches 2 \    # Tavily calls allowed per sub-topic agent
  --max_iterations 2    # quality-check loops before finalizing
```

Bigger numbers mean more tasks, more spans, and a richer comparison in both UIs.

### Good demo queries

Pick questions with a clear, searchable answer so the trace tells a clean story:

```bash
flyte run workflow.py research_pipeline --query "What is the OpenTelemetry Collector and why use it?"
flyte run workflow.py research_pipeline --query "Compare OTLP over gRPC versus HTTP for sending traces."
flyte run workflow.py research_pipeline --query "What is the Arize Phoenix open-source project used for?"
```

(Avoid ambiguous product names: "Flyte 2" alone, for example, surfaces unrelated
sportswear in web search.)

## Lifecycle and persistence

The app is pinned to a single always-on replica (`Scaling(replicas=(1, 1))`), so
unlike the sibling chat apps it never scales to zero. It stays up and collecting
until you explicitly remove it (see Teardown); killing the local
`python phoenix_app.py` process only stops the log tail, not the deployed app. A
collector has to stay reachable, which is why it is pinned rather than scale-to-zero.

Traces are stored in SQLite under `PHOENIX_WORKING_DIR`. "Always on" applies to
the process, not the data: the pod will not scale down, but if it ever restarts
(devbox stop/start, node reboot, eviction) the `/tmp` store resets and earlier
traces are gone. For a single session this is exactly what you want; everything
you send accumulates and stays as long as the pod keeps running.

For production, point Phoenix at Postgres and the server becomes stateless:

```python
# in phoenix_app.py, AppEnvironment(env_vars={...})
"PHOENIX_SQL_DATABASE_URL": "postgresql://user:pass@host:5432/phoenix",
```

Then you can drop the always-on pin and let it scale to zero between sessions.

## How the tracing is wired

Each task pod is its own process, so each one sets up tracing itself. The whole
client side is in `tracing.py`:

```python
from phoenix.otel import register
from openinference.instrumentation.langchain import LangChainInstrumentor

tracer_provider = register(
    endpoint=COLLECTOR + "/v1/traces",
    project_name="research-pipeline",
    batch=True,
    set_global_tracer_provider=False,   # leave the global provider to Flyte
)
LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
```

`register()` points the OTLP exporter at the hosted collector. The LangChain
instrumentor patches LangChain and LangGraph (LangGraph runs on LangChain
runnables, so there is no separate LangGraph instrumentor); the pipeline code
stays plain, with no tracing calls in the graph itself. `setup_tracing()` is
idempotent, so container reuse is safe, and each task calls `flush()` before
returning, because Flyte pods are short-lived and an un-flushed batch of spans
would never reach Phoenix.

## Teardown

The pipeline run is one-shot; nothing to stop there. The Phoenix app keeps running
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
- **`ModuleNotFoundError` in the pod** (e.g. `No module named 'graph'`): `flyte
  run` only bundles modules that are imported by the time the script loads. Keep
  the `graph` / `models` / `llm` / `tracing` imports at the top of `workflow.py`
  (not lazily inside a task), and keep their libs installed in the submit venv
  from Step 1.
- **Traces do not appear in Phoenix**: confirm the app is reachable
  (`curl -s -o /dev/null -w "%{http_code}" http://phoenix-server-flytesnacks-development.localhost:30081`
  returns `200`), and that a task log shows the `[tracing] exporting to ...` line.
  The pipeline must run on the cluster (remote) so the tasks can resolve the
  collector's in-cluster DNS name.

## Links

- Phoenix: https://github.com/Arize-ai/phoenix
- Phoenix docs: https://docs.arize.com/phoenix
- OpenInference instrumentors: https://github.com/Arize-ai/openinference
