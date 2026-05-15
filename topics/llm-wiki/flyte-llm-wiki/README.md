Welcome to AI Build & Learn a weekly AI engineering stream where we pick a new topic and learn by building together.

​​​This event is about building an LLM-maintained Wiki, a pattern from Andrej Karpathy for turning raw sources into a persistent, compounding knowledge base instead of re-retrieving from documents on every query.

We'll explore how this differs from traditional RAG, the three-layer stack (sources, wiki, schema), and the core operations (ingest, query, lint) that keep the wiki healthy as it grows.

Some things to look up to get started:
- Karpathy's LLM Wiki gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Community implementations linked in the gist comments (SwarmVault, Kompl, Link, OmegaWiki, etc.)


​​​Resources

​​​- GitHub: https://github.com/sagecodes/ai-build-and-learn
​​​- Events Calendar: https://luma.com/ai-builders-and-learners
​​​- Slack (Discuss during the week): https://slack.flyte.org/
​​​- Hosted by Sage Elliott: https://www.linkedin.com/in/sageelliott/


# LLM Wiki on Flyte 2

A working implementation of Karpathy's LLM-maintained wiki pattern, running on the DGX Spark Flyte 2 devbox. The wiki is a `flyte.io.Dir` that gets passed from task to task; the LLM is the in-cluster Gemma 4 26B vLLM app from the sibling `gemma4-dgx-devbox` project. Two surfaces: a headless pipeline (`flyte run pipeline.py wiki_pipeline`) for a one-shot reproducible build, and a Gradio chat app that lets you ingest sources interactively and watch the wiki update.

```
┌─────────────────────────────────────────┐
│  flyte.io.Dir  (lives in rustfs)        │
│  raw/<slug>.md       source summaries   │
│  pages/<slug>.md     LLM concept pages  │
│  index.md            auto-generated     │
│  log.md              append-only log    │
│  AGENTS.md           wiki conventions   │
└─────────────────────────────────────────┘
                ▲                ▲
                │                │
        ┌───────┴───────┐ ┌──────┴────────┐
        │ pipeline.py   │ │ chat_app.py   │
        │ ingest_source │ │ Gradio UI:    │
        │ query_wiki    │ │  Ingest tab   │
        │ lint_wiki     │ │  Query tab    │
        │ wiki_pipeline │ │  Lint tab     │
        └───────┬───────┘ └──────┬────────┘
                │                │
                └────────┬───────┘
                         ▼
              ┌──────────────────────┐
              │  Gemma 4 26B vLLM    │
              │  (sibling project)   │
              └──────────────────────┘
```

## Files

| File | What it does |
|------|--------------|
| `config.py` | Flyte `TaskEnvironment` for the pipeline, vLLM connection constants, wiki layout names. |
| `wiki_lib.py` | Pure-Python helpers shared by pipeline and chat app: layout, source fetch (trafilatura), index regeneration, prompts, deterministic lint, JSON parsing. |
| `pipeline.py` | Four Flyte tasks (`init_wiki`, `ingest_source`, `query_wiki`, `lint_wiki`) plus a `wiki_pipeline` orchestrator that chains them. |
| `chat_app.py` | Flyte `AppEnvironment` running a Gradio UI with tabs for ingest, query, lint, and browse. Streams answers and lint reports from Gemma. |
| `requirements.txt` | Local deps for `flyte run` and `python chat_app.py`: `flyte[tui]`, `openai`, `httpx`, `trafilatura`, `gradio`. |

## How this differs from RAG

Most RAG systems index chunks once and re-retrieve them on every query. The wiki is different: the LLM reads each new source, writes a summary into `raw/`, then updates the concept pages in `pages/` so they incorporate the new knowledge. The retrieval surface is the wiki itself, not the raw sources. Answers come from a synthesised, cross-referenced view; corrections compound; bookkeeping cost is near zero because the LLM does the maintenance.

Three operations from the gist map directly to Flyte tasks:

| Karpathy | Flyte task | What it does |
|----------|------------|--------------|
| Ingest   | `ingest_source` | Fetch a URL (or accept pasted text); summarize to `raw/<slug>.md`; ask the LLM which concept pages to create or update; apply the edits; regenerate `index.md`; append to `log.md`. |
| Query    | `query_wiki`    | Ask the LLM to pick relevant pages from the index; then answer the question using only those pages, with `[[slug]]` citations. |
| Lint     | `lint_wiki`     | Deterministic checks (orphan pages, broken `[[links]]`) plus an LLM pass for contradictions, stale claims, missing pages and missing cross-references. |

## Why this is shaped the way it is

Two design choices worth calling out up front.

**The wiki is a `flyte.io.Dir`, not a database.** Each task downloads the latest version of the Dir, copies it into a writable scratch path, mutates files, and uploads the new version. This means: every ingest produces an immutable wiki revision in rustfs; you can pass `wiki=flyte://...` from a previous run into a fresh task; nothing hand-wired about persistence beyond what `flyte.io.Dir` already gives you. The chat app keeps a separate ephemeral copy at `/tmp/llm-wiki/` because Knative app pods restart at scale-to-zero and can't trivially mutate a `flyte.io.Dir` in place.

**`wiki_lib.py` knows nothing about Flyte.** Prompts, slugify, index regeneration, source fetch, and the deterministic lint all sit in a pure-Python module so the pipeline and the chat app share one implementation. Anywhere you see an LLM call in `pipeline.py` or `chat_app.py`, the prompts are imported from `wiki_lib` and the only thing each caller decides is whether to stream or not.

## Prereqs

Same Flyte 2 devbox as the graphrag project, started with `--gpu` so the sibling Gemma 4 vLLM app has a GPU to schedule onto:

```bash
flyte start devbox --gpu
docker exec flyte-devbox nvidia-smi -L           # verify GB10 visible
kubectl get nodes -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}'
# should print: 1
```

Set up the venv:

```bash
cd topics/llm-wiki
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Flyte CLI config (shared `flytesnacks/development`):

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

Bring up Gemma 4 vLLM in the sibling project (one-time):

```bash
cd ../gemma4/gemma4-dgx-devbox
python prefetch_model.py                                  # one-time
GEMMA_PREFETCH_RUN=<run-name> python vllm_server.py       # deploy
cd -
```

## 1. Run the pipeline end-to-end

```bash
flyte run pipeline.py wiki_pipeline
```

Defaults: three Wikipedia seed sources (RAG, vector database, large language model) and one demo question. All overridable on the CLI:

```bash
flyte run pipeline.py wiki_pipeline \
    --sources '["https://en.wikipedia.org/wiki/Retrieval-augmented_generation","https://en.wikipedia.org/wiki/Self-supervised_learning"]' \
    --question "How does self-supervised learning relate to RAG?"
```

What happens:

1. `init_wiki` creates an empty Dir with `index.md`, `log.md`, `AGENTS.md`, and empty `raw/` / `pages/`.
2. For each source, `ingest_source` fetches it (trafilatura cleans the HTML to markdown), asks Gemma for a summary page (`raw/<slug>.md`), then asks Gemma a second time for a JSON list of concept-page edits and applies them.
3. `query_wiki` runs two LLM passes: pick relevant pages from `index.md`, then answer using only those pages with `[[slug]]` citations.
4. `lint_wiki` reports orphans, broken `[[links]]`, and an LLM-driven audit.

Each task returns its own Flyte report (top-right link in the run UI). The final output is the wiki Dir itself, addressable as `flyte://flytesnacks/development/<run-name>/o0` for any follow-up task.

Run individual tasks against an existing wiki:

```bash
flyte run pipeline.py ingest_source \
    --wiki flyte://flytesnacks/development/<wiki-run>/o0 \
    --source "https://lilianweng.github.io/posts/2023-06-23-agent/"

flyte run pipeline.py query_wiki \
    --wiki flyte://flytesnacks/development/<wiki-run>/o0 \
    --question "What's the difference between RAG and an LLM wiki?"

flyte run pipeline.py lint_wiki \
    --wiki flyte://flytesnacks/development/<wiki-run>/o0
```

## 2. Deploy the chat app

```bash
python chat_app.py
```

URL is logged at the end:

```
LLM Wiki chat UI deployed: http://llm-wiki-chat-flytesnacks-development.localhost:30081/
```

By default the app **seeds itself from the most recent `wiki_pipeline` run** via `RunOutput(task_name="llm-wiki-pipeline.wiki_pipeline", type="directory")`: Flyte mounts the latest output Dir into the pod, the server copies it into `/tmp/llm-wiki/` before launching Gradio. Run the pipeline first (or you'll deploy against an empty resolver).

Pin to a specific historical run with an env var:

```bash
WIKI_RUN_NAME=<pipeline-run-name> python chat_app.py
```

Re-running the pipeline doesn't auto-refresh the live UI — redeploy the chat app to pick up a newer run.

Four tabs:

- **📥 Ingest** Paste a URL or a chunk of text; the LLM summarizes it and integrates it into concept pages. Status box shows the three steps (fetch → summarize → integrate).
- **💬 Query** Chat-style. The LLM picks relevant pages, then streams an answer citing `[[slug]]`. Pages consulted are shown above the answer.
- **🧹 Lint** One click. Deterministic stats first (pages, sources, orphans, broken links), then a streamed LLM audit underneath.
- **📂 Browse** Pick any file in the wiki Dir from the dropdown and read it.

The "Reset wiki" button wipes `/tmp/llm-wiki/` and starts fresh. Same thing happens automatically on Knative scale-to-zero (5 min idle), so for a stable demo build the wiki freshly each session, or use the pipeline for a reproducible snapshot.

## Demo prompts

Ingest a couple of sources in the chat app first, then try these queries.

**Definition questions** (single page suffices, the wiki gives a clean concise answer):

- *"What is retrieval-augmented generation?"*
- *"What is a vector database?"*
- *"What does the LLM Wiki pattern actually do differently from RAG?"*

**Relational questions** (the wiki shines when concepts cross-link):

- *"How does a vector database relate to retrieval-augmented generation?"*
- *"What does a large language model need from a retrieval system?"*

**Audit-shaped questions** (run lint, look at the report):

- After ingesting two RAG-adjacent sources, lint should flag `[[reranker]]` or `[[dense-retrieval]]` as missing pages if the LLM mentioned them in passing without giving them their own page.

## Known limitations

- **Ephemeral chat app state.** `/tmp/llm-wiki/` is wiped on pod restart and scale-to-zero. The pipeline path doesn't have this problem (rustfs survives `flyte stop devbox`). The chat app auto-seeds from the latest `wiki_pipeline` run on startup via `RunOutput`, but in-UI edits made after seeding still don't survive scale-to-zero, and the live UI doesn't auto-refresh when a newer pipeline run completes.
- **Trafilatura best-effort.** Some sites (paywalled news, heavy SPAs) extract poorly. If `fetch_to_markdown` returns junk, paste the article text directly into the Ingest tab instead.
- **Gemma 4 26B is the bottleneck on latency.** Each ingest does two LLM calls (~10-15s each on the GB10) and one query does two more. Three seed sources end-to-end takes about 90 seconds. Swap to a smaller / faster model for tight demo cycles.
- **No caching.** Re-ingesting the same URL re-fetches and re-summarizes every time. Cheap to add (`@env.task(cache="auto")` on a fetcher subtask) but not worth it for the v1 demo.

## Next ideas

- **Persist the chat app's wiki bidirectionally.** Seeding from a `RunOutput(directory)` is in place; the missing half is *writing back*. A SIGTERM handler on the chat container could upload `/tmp/llm-wiki/` as a new `flyte.io.Dir` so in-UI edits survive scale-to-zero.
- **Schema-guided ingest.** Right now every ingest uses the same prompt. Reading `AGENTS.md` into the prompt at ingest time would let users customize how the LLM structures pages per wiki (research vs personal vs team).
- **Diff view in the chat UI.** After an ingest, show a side-by-side diff of which pages changed and how, instead of just the touched list.
- **`qmd` search backend.** As the wiki grows past ~50 pages, swap "give the LLM the whole index" for `qmd`'s hybrid BM25/vector search (CLI or MCP) so the picker stays fast.
- **Re-ingest detection.** Hash the source URL/text; if the same source comes in twice, prompt the LLM to *update* its existing `raw/` summary rather than overwrite it blindly.

## Future work: a scheduled, source-driven wiki

The current setup is two manual surfaces (pipeline + chat app). A more interesting target is a wiki that maintains itself from the things you already write or save:

1. **Obsidian (or any markdown vault) as a source.** Point `ingest_source` at a local vault path or a synced cloud folder; treat each note as a source the same way URLs are treated today. A periodic scan picks up new and modified notes since the last run.
2. **Lint-driven gap filling with Tavily.** When `lint_wiki` flags a missing or thin concept page, hand the slug to Tavily web search, fetch the top result, and feed it back through `ingest_source`. The lint pass becomes an active "go learn this" step instead of just a report.
3. **Schedule it.** Wire `wiki_pipeline` to a Flyte schedule (e.g. nightly). Each run produces a fresh wiki Dir; the chat app's `WIKI_RUN_NAME` parameter gets updated to the latest run via a small redeploy job. Wake up to yesterday's notes already cross-linked into the wiki, with gaps filled overnight.
4. **Chat app as the only daily-driver surface.** Open the URL, the wiki is current, ingest anything new that comes up during the day. The pipeline becomes the housekeeper rather than the primary interface.
