# RAG with Chroma + Gemma 4 on Flyte 2

Week 1 of the vector-stores series. The pipeline is the new piece — the chat
front-end mirrors `topics/gemma4/gemma4-dgx-devbox/chat_app.py` with a
retrieval step bolted on.

```
┌────────────────────┐    flyte.io.Dir         ┌────────────────────┐
│   pipeline.py      │  (Chroma persist dir)   │   chat_app.py      │
│                    ├────────────────────────▶│                    │
│ HF dataset         │                         │ embeds query →     │
│  → chunks          │  RunOutput(             │  Chroma top-k →    │
│  → BGE-small embeds│    type="directory",    │  injects ctx →     │
│  → Chroma          │    task_name=…)         │  vLLM Gemma 4      │
└────────────────────┘                         └────────────────────┘
                                                         │
                                                         ▼
                                            already-running gemma4-26b-a4b-it
                                            vLLM app on the devbox
```

## Files

| File | What it does |
|------|--------------|
| `config.py` | Just the Flyte `TaskEnvironment` + image. DGX-Spark-pinned (arm64, local registry). |
| `pipeline.py` | Three Flyte tasks — `fetch_dataset` → `chunk_documents` → `embed_and_index` — wrapped by `rag_pipeline`. All knobs are task-arg defaults so you override on the CLI. |
| `chat_app.py` | Gradio app. `flyte.app.RunOutput` mounts the pipeline's Chroma dir at startup; the encoder loads in-process for query-side embedding. vLLM endpoint is hard-coded at the top of the file. |
| `requirements.txt` | Local deps (`flyte`, `gradio`, `chromadb`, `sentence-transformers`, `datasets`, `openai`). |

## Why a `Dir` and not a `File`

Chroma's `PersistentClient` writes a sqlite3 DB plus parquet shards under one
directory. `flyte.io.Dir` snapshots the whole thing as one artifact; the chat
app's `Parameter(value=RunOutput(type="directory"), download=True, …)` mounts
it back as a local path on pod startup.

## Prereqs

The Gemma 4 vLLM server from the sibling project should already be deployed:

```bash
cd ../../gemma4/gemma4-dgx-devbox
python vllm_server.py     # only if it's not running
```

Set up this project's venv:

```bash
cd topics/vectorstore/rag-chroma-flyte
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

The Flyte CLI config is per-directory. Create one in this dir (same cluster
as the gemma4 project — they share `flytesnacks/development`):

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

## 1. Build the index

```bash
flyte run --local --tui pipeline.py rag_pipeline
# or remote:
flyte run pipeline.py rag_pipeline
```

The `--local --tui` form is the fastest way to iterate while picking a chunk
size or dataset. Override any param on the CLI:

```bash
flyte run --local pipeline.py rag_pipeline --max_docs 500 --chunk_size 800
```

Each step writes a tiny `flyte.report` summary so you see counts (rows kept,
chunks produced, items indexed) in the Flyte UI.

When the run finishes, copy its **run name** — that's how the chat app finds
the artifact. It's printed at the end:

```
Pipeline run: <run-name>
  http://localhost:30080/v2/.../runs/<run-name>
```

## 2. Deploy the chat app

```bash
RAG_PIPELINE_RUN=<run-name> python chat_app.py
```

If you don't pin a run, the app uses the latest succeeded run of `rag_pipeline`
(via `RunOutput(task_name=…)`), which is convenient during development but
brittle in shared environments — pin once you're past iteration.

The URL is logged at the end:

```
RAG chat UI deployed: http://rag-chat-ui-flytesnacks-development.localhost:30081/
```

## What's in the UI

- **Chatbot** (left) — same streaming flow as the Gemma chat, including the
  🧠 Thinking panel and the thinking-budget slider.
- **Retrieved chunks** (right) — top-k chunks for the most recent message,
  with their cosine similarity (1 − distance/2). Disabling retrieval falls
  back to a vanilla chat.
- **Top-k slider** — 1–10. Defaults to 4.

The retrieved chunks are injected into the system prompt as a `CONTEXT:`
block; the system prompt asks the model to cite sources as `[#N]` and to
refuse if the answer isn't in context.

## Knobs worth knowing about

All defaults live as function arguments on the tasks in `pipeline.py` —
override on the CLI rather than editing the file:

```bash
flyte run pipeline.py rag_pipeline \
    --dataset_repo wikipedia --dataset_config 20220301.simple \
    --max_docs 2000 --chunk_size 800 --chunk_overlap 100
```

- **`chunk_size=1200` chars** — bge-small handles 512 tokens (~2000 chars
  English), so this leaves headroom and avoids truncation.
- **`embedding_model=BAAI/bge-small-en-v1.5`** — same model is used at index
  and query time; the collection's metadata records which model built it, and
  the chat app warns loudly on a mismatch.
- **`dataset_repo=rag-datasets/rag-mini-wikipedia`** — swap for any HF dataset
  with a text column (set `dataset_config`, `dataset_split`, `text_column`
  together).

## Troubleshooting

**`ParameterMaterializationError: No runs found for task …`** — no successful
`rag_pipeline` runs exist yet. Run the pipeline once, or pin
`RAG_PIPELINE_RUN=<run-name>`.

**Chunks panel shows "noisy" hits** — verify the chat app loaded the same
encoder the pipeline used (check the chat-pod logs for `WARNING: query encoder
… != index encoder …`). Wipe and re-run the pipeline if you bumped
`embedding_model` mid-stream.

**vLLM endpoint 404 / connection refused** — the chat app reaches the vLLM
server by its cluster-internal Knative DNS name. Confirm `VLLM_APP_NAME` at
the top of `chat_app.py` matches the deployed app (default
`gemma4-26b-a4b-it-vllm` in `flytesnacks/development`).

## Next ideas

- Swap in a doc-scraper task that crawls real docs (Flyte docs, your own
  notes) instead of the HF passage corpus.
- Add a re-ranking step (cross-encoder over the top-k) before the LLM call.
- Replace Chroma with another vector store — the pipeline's `embed_and_index`
  task is the only thing that has to change.
