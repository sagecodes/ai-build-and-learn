Welcome to AI Build & Learn a weekly AI engineering stream where we pick a new topic and learn by building together.

​​​This event is about building with Cognee, an open-source memory layer for AI agents that combines vector search and graph databases into a single queryable knowledge infrastructure.

We'll explore how Cognee unifies data from multiple sources, the core API (`remember`, `recall`, `forget`, `improve`), and how persistent agent memory differs from one-shot RAG retrieval.

Some things to look up to get started:
- Cognee GitHub: https://github.com/topoteretes/cognee
- Cognee docs: https://docs.cognee.ai/


​​​Resources

​​​- GitHub: https://github.com/sagecodes/ai-build-and-learn
​​​- Events Calendar: https://luma.com/ai-builders-and-learners
​​​- Slack (Discuss during the week): https://slack.flyte.org/
​​​- Hosted by Sage Elliott: https://www.linkedin.com/in/sageelliott/


# Cognee Memory on Flyte 2

Two surfaces over one Cognee memory, running on the DGX Spark Flyte 2 devbox. Cognee's whole pitch over a plain vector store: `cognify` turns raw text into a **knowledge graph plus a vector index**, so retrieval blends graph traversal with similarity instead of nearest-neighbour chunks alone. The memory (SQLite + LanceDB + Ladybug) travels between runs as a `flyte.io.Dir`, and the LLM behind cognee's extraction + answering is the in-cluster Gemma 4 26B vLLM sibling app from `gemma4-dgx-devbox`.

```
                ┌──────────────────────────────────────────────┐
                │  cognee store  (one tar-able subtree)          │
                │   data/    SQLite (relational) + LanceDB (vec) │
                │   system/  Ladybug (graph) + bookkeeping        │
                └──────────────────────────────────────────────┘
                  ▲ carried as flyte.io.Dir       ▲ tarred to HF
                  │ (run → run)                    │ (scale-to-zero)
        ┌─────────┴──────────┐          ┌──────────┴───────────┐
        │ pipeline.py        │          │ chat_app.py          │
        │ source-ingest agent│          │ Gradio chatbot       │
        │ add → cognify      │          │ recall → Gemma       │
        │ search(SearchType) │          │ → remember           │
        └─────────┬──────────┘          └──────────┬───────────┘
                  │   granular API          simple API   │
                  └───────────────┬──────────────────────┘
                                  ▼
                       ┌──────────────────────┐
                       │  Gemma 4 26B vLLM    │  (custom OpenAI endpoint)
                       │  + local fastembed   │  (embeddings, 384-dim bge)
                       └──────────────────────┘
```

## The two surfaces (layered API)

| Surface | Cognee API | What it shows |
|---------|-----------|----------------|
| `pipeline.py` (source-ingest agent) | granular: `add` → `cognify` → `search(SearchType.GRAPH_COMPLETION)` | The mechanics. Each run loads the prior memory `Dir`, adds a source, rebuilds the graph, and queries it. Run-over-run the graph compounds. |
| `chat_app.py` (chatbot) | simple: `recall` / `remember` | The daily-driver. Recalls context for each user turn, streams a Gemma answer, then remembers the exchange back into the same graph. |

Both drive the **same** cognee store. The pipeline is the housekeeper that builds knowledge from sources; the chat app talks over it and adds to it conversationally.

## Files

| File | What it does |
|------|--------------|
| `config.py` | Flyte `TaskEnvironment` for the pipeline, vLLM connection constants, cognee storage layout, embedding + HF settings. DGX-pinned (arm64 + local registry). |
| `cognee_lib.py` | Flyte-agnostic helpers: `configure_cognee` (points cognee at vLLM + local fastembed + a storage root), tar snapshot/restore, HF push/pull, trafilatura source fetch. Imports `cognee` lazily so config lands before its first import. |
| `pipeline.py` | `init_memory`, `ingest_source`, `query_memory`, and a `memory_pipeline` orchestrator. The memory is a `flyte.io.Dir` passed task-to-task. |
| `chat_app.py` | Flyte `AppEnvironment` Gradio chat. Seeds from the latest pipeline run via `RunOutput(directory)`; checkpoints to HF on `@on_shutdown` and a save button. |
| `requirements.txt` | Local deps for `flyte run` / `python chat_app.py`. |

## How memory persists (both mechanisms)

- **Pipeline: `flyte.io.Dir`.** Each task downloads the prior memory Dir, copies it into a writable scratch path, runs cognee, and uploads the new state as a fresh Dir. Every ingest is an immutable, addressable revision in rustfs — pass `--memory flyte://flytesnacks/development/<run>/o0` from any prior run into a new task and it compounds. This survives `flyte stop devbox`.
- **Chat app: HF tarball.** Knative scales the app pod to zero after 5 min idle, wiping `/tmp/cognee-mem`. `@on_startup` first tries the mounted pipeline-run Dir (`RunOutput`), then falls back to the HF snapshot. `@on_shutdown` and the **💾 Save to HF** button tar the store back to `sagecodes/cognee-mem` so in-UI memories survive.

## How this differs from plain RAG

A vector store retrieves the top-k nearest chunks to your query and stops there. Cognee `cognify` additionally extracts entities and relationships into a graph, so `SearchType.GRAPH_COMPLETION` can follow edges between concepts the query never named, then have the LLM compose an answer over that subgraph. Same Wikipedia seed trio as the sibling `llm-wiki` demo (RAG, vector database, knowledge graph), so you can compare a knowledge-graph memory against the LLM-maintained-wiki approach on identical inputs.

## Prereqs

Same Flyte 2 devbox as the llm-wiki / graphrag projects, started with `--gpu` so the Gemma 4 vLLM app can schedule:

```bash
flyte start devbox --gpu
docker exec flyte-devbox nvidia-smi -L            # verify GB10 visible
```

Venv:

```bash
cd topics/cognee/cognee-flyte
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Flyte CLI config (shared `flytesnacks/development`):

```bash
flyte create config --endpoint localhost:30080 --project flytesnacks --domain development --builder local --insecure
```

HF token secret (chat app's checkpoint; needs **write** scope, scoped to the same project/domain):

```bash
flyte create secret HF_TOKEN -p flytesnacks -d development
```

Create the `sagecodes/cognee-mem` model repo on HF (empty is fine; first save populates it). Bring up Gemma 4 vLLM in the sibling project if it isn't already running (`topics/gemma4/gemma4-dgx-devbox`).

## 1. Run the pipeline end-to-end

```bash
flyte run pipeline.py memory_pipeline
```

Defaults: three Wikipedia seed sources and one demo query, both overridable:

```bash
flyte run pipeline.py memory_pipeline \
    --sources '["https://en.wikipedia.org/wiki/Retrieval-augmented_generation","https://en.wikipedia.org/wiki/Knowledge_graph"]' \
    --question "How does a knowledge graph improve retrieval over plain vector search?"
```

`init_memory` makes an empty store, each `ingest_source` does `add` + `cognify`, and `query_memory` runs `search(SearchType.GRAPH_COMPLETION)`. The final output is the memory `Dir` (`flyte://flytesnacks/development/<run>/o0`).

**Each `ingest_source` renders the knowledge graph in its Flyte report:** node/relationship counts, an interactive d3 force graph (cognee's `visualize_graph`, embedded as an iframe), and a static relationships table as a no-JS fallback. The interactive `graph.html` is also written into the output `Dir`, so every memory revision carries a snapshot of its graph. (The iframe loads d3 from a CDN, so it needs network when you view the report; the table always renders.)

Compound onto a prior run instead of starting fresh:

```bash
flyte run pipeline.py ingest_source \
    --memory flyte://flytesnacks/development/<prior-run>/o0 \
    --source "https://lilianweng.github.io/posts/2023-06-23-agent/"

flyte run pipeline.py query_memory \
    --memory flyte://flytesnacks/development/<run>/o0 \
    --question "What is agent memory?"
```

## 2. Deploy the chat app

```bash
python chat_app.py
# → Cognee memory chat deployed: http://cognee-memory-chat-flytesnacks-development.localhost:30081/
```

It seeds from the most recent `memory_pipeline` run via `RunOutput`. Run the pipeline first, or it cold-starts empty (then falls back to the HF snapshot if one exists). Pin a specific run with `MEMORY_RUN_NAME=<run> python chat_app.py`.

Two tabs:
- **💬 Chat** Per turn: **recall** context from the graph (shown in the right panel) → stream Gemma's answer → **remember** the exchange. The status line shows store size and where memory was seeded from.
- **🕸 Graph** Click "Render graph" to draw the live store's knowledge graph — the same `visualize_graph` view the ingest pipeline puts in its report (interactive d3 + a relationships table). Re-render after chatting to watch the graph grow as `remember` writes new exchanges in.

## Demo flow for the stream

1. `flyte run pipeline.py memory_pipeline` — watch each ingest report the store growing, then the GRAPH_COMPLETION answer.
2. `flyte run pipeline.py ingest_source --memory flyte://.../o0 --source <new url>` — show a *second* run compounding onto the first run's output. This is the "each run pulls the previous run's memory" story.
3. `python chat_app.py` — open the URL, ask *"What's been ingested so far?"*, watch the Recalled panel light up.
4. Tell it something personal (*"I'm Sage, I prefer terse answers"*), then ask about yourself next turn — it recalls from the graph.
5. **💾 Save to HF**, redeploy, ask again — memory survives the cold start.

## Known risks / gotchas

Verified locally against **cognee 1.1.0** on this aarch64 box: install, `fastembed` (the `cognee[fastembed]` extra), env-var storage redirect, and the access-control toggle all work; the `add`/`cognify`/`search`/`recall`/`remember` signatures match. The graph backend in 1.1 is cognee's embedded **Ladybug** (not Kuzu). What couldn't be checked offline is anything that hits the LLM — see below.

- **`LOG_LEVEL` must be a name, not a number.** cognee's `setup_logging()` indexes a name→level dict with `$LOG_LEVEL`, but Flyte sets `LOG_LEVEL=30` (numeric WARNING) in task pods, so `import cognee` dies with `KeyError: '30'`. `configure_cognee` normalizes the numeric value to a level name *before* cognee is imported. This only bites inside Flyte pods (a local shell has no numeric `LOG_LEVEL`), so it passes locally and fails on the cluster — keep the normalization.
- **Headless access control.** cognee 1.x defaults to multi-user access control with required auth, which expects a `User` on every call and breaks a headless pipeline. `configure_cognee` sets `ENABLE_BACKEND_ACCESS_CONTROL=false` to run single-tenant. Don't drop that.
- **cognee checks the LLM at pipeline start.** Even `add` triggers `test_llm_connection`, so the vLLM endpoint must be reachable and `LLM_API_KEY` non-empty (set to `not-used`). This is why `add`/`cognify` can't be smoke-tested without the devbox up.
- **Structured extraction on Gemma.** `cognify` asks the LLM for structured (JSON/tool) output via litellm + instructor. Gemma 4 on vLLM supports guided decoding, but if extraction errors out, the graph build degrades to chunks only. Check task logs for litellm/instructor parse failures.
- **arm64 image build.** The Flyte image installs `cognee` + `fastembed` (pulls `lancedb`, `onnxruntime`). `flyte run` builds it on first invocation. If a wheel is missing for aarch64, swap the embedding backend in `config.py` (e.g. `EMBEDDING_PROVIDER="huggingface"` with sentence-transformers).
- **`remember`/`recall` vs `add`/`cognify`/`search`.** Both exist natively in 1.1.0; the chat app still falls back to the granular API (`chat_app._recall`/`_remember`) if a future/older build drops the v1.0 names. Result entries are Pydantic models, so text is pulled via `cognee_lib.result_text` (`.search_result` / `.answer` / `.content` / `.text`), not `str()`.
- **`remember` is heavy.** Unlike the chroma demo's fast embed-and-write, cognee's `remember`/`cognify` rebuilds graph + embeddings per call (a few seconds on the GB10). The chat app streams the answer first, then writes memory with a "remembering…" status.

## Next ideas

- **Lint-driven gap filling.** Add a `lint_memory` task that asks the graph what concepts are thin or missing, hands the gaps to Tavily, and feeds results back through `ingest_source`.
- **Schedule the pipeline.** Wire `memory_pipeline` to a Flyte cron so it ingests new sources nightly; the chat app picks up the latest run on next deploy.
- **Graph in the chat app too.** The ingest pipeline now renders the graph per-report; the chat app could grow a Browse/Graph tab that shows the same `visualize_graph` view of the live `/tmp` store, or use `SearchType.INSIGHTS` to surface relationships per query.
- **Write-back from chat to a Dir.** The chat app remembers into `/tmp` + HF today; a task that promotes the HF snapshot into a canonical `flyte.io.Dir` would unify the two persistence paths.
