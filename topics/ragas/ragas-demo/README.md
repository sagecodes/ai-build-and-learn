# Evaluating RAG with Ragas on Flyte 2

Part of the **Ragas** week of AI Build & Learn. This builds directly on
`topics/vectorstore/rag-chroma-flyte`: that project serves a RAG system (Chroma
+ BGE + gemma4 vLLM); this one **scores** it with a broad suite of Ragas metrics
and runs an A/B comparison so the evals drive a decision instead of vibes.

```
┌──────────────────────────┐
│ load_qa                  │   rag-mini-wikipedia question-answer config
│  real questions + answers│   -> {question, ground_truth}
└────────────┬─────────────┘
             │
┌────────────▼─────────────┐  reuse a rag_pipeline run (RAG_PIPELINE_RUN)
│ resolve_index            │  ...or build an identical Chroma index inline
└────────────┬─────────────┘
             │ Chroma dir + QA
┌────────────▼─────────────┐   embed query -> Chroma top-k -> gemma4 vLLM
│ run_rag                  │   -> {user_input, retrieved_contexts,
│                          │       response, reference}
└────────────┬─────────────┘
             │ results.jsonl
┌────────────▼─────────────┐   Ragas evaluate() with the metric suite
│ ragas_score              │   judge = gemma4 vLLM (or --judge openai)
│  -> Flyte scorecard      │   embeddings = BGE-small
└──────────────────────────┘
```

The **same gemma4 vLLM app** plays two roles here: the RAG answerer and the
Ragas LLM-as-judge. So the whole eval stays self-hosted on the devbox.

## Files

| File | What it does |
|------|--------------|
| `config.py` | Flyte `TaskEnvironment` + image, vLLM/dataset/embedding constants. DGX-Spark-pinned (arm64, local registry). |
| `ragas_lib.py` | Pure helpers: judge + embeddings wiring, the metric suite (`build_metrics`), `run_eval`, and the HTML scorecards. No Flyte. |
| `eval_pipeline.py` | Flyte tasks — `load_qa`, `build_index`, `run_rag`, `ragas_score` — wrapped by `ragas_eval`, `ragas_compare`, `ragas_compare_chunking`. |
| `eval_app.py` | Gradio **live eval playground** (`flyte.app.AppEnvironment`). Ask a question, watch RAG answer it, watch Ragas grade it live. |
| `requirements.txt` | Local deps (`flyte`, `ragas`, `langchain-openai`, `langchain-huggingface`, `sentence-transformers`, `chromadb`, `datasets`). |

## Dataset

[`rag-datasets/rag-mini-wikipedia`](https://huggingface.co/datasets/rag-datasets/rag-mini-wikipedia)
on the Hugging Face Hub, the same corpus the sibling `rag-chroma-flyte` project
indexes. It ships two configs and we use both:

| Config | Split | Columns | Rows | Used for |
|--------|-------|---------|-----:|----------|
| `text-corpus` | `passages` | `passage`, `id` | 3,200 | the knowledge base we chunk, embed, and index into Chroma |
| `question-answer` | `test` | `question`, `answer`, `id` | 918 | the eval test set: real questions **with** ground-truth answers |

The question-answer pairs are what make the reference-based metrics (context
recall, factual correctness, semantic similarity, noise sensitivity) possible
without hand-labeling anything. Answers are often terse (some are just "yes" /
"no"), which is worth knowing when you read the scores.

## Stack

Everything below the eval framework is self-hosted on the devbox; no external
API is required.

| Layer | What | Notes |
|-------|------|-------|
| Orchestration | **Flyte 2** | tasks, `flyte.io.Dir` artifacts, `flyte.report` scorecards, `cache="auto"` |
| Eval framework | **Ragas** (`>=0.2,<0.4`, resolves to 0.3.9) | `evaluate()` + `EvaluationDataset`, the 9-metric suite |
| LLM (RAG answerer **and** judge) | **gemma4-26b-a4b-it** via **vLLM** | OpenAI-compatible endpoint, the gemma4 sibling app; swap to OpenAI with `--judge openai` |
| Embeddings (index, query, judge) | **BAAI/bge-small-en-v1.5** | `sentence-transformers`, 384-dim, cosine; same encoder everywhere |
| Vector store | **Chroma** | `PersistentClient`, snapshotted as a `flyte.io.Dir` |
| Judge glue | **langchain-openai** + **langchain-huggingface** | wrapped by Ragas `LangchainLLMWrapper` / `LangchainEmbeddingsWrapper` |
| Hardware | **DGX Spark** (arm64) | image is platform-pinned `linux/arm64`, devbox-local registry |

## The metric suite

Run over real question-answer pairs (so the reference-based metrics have a
ground truth to compare against). Grouped in the scorecard:

| Metric | Group | Needs ground truth? | Measures |
|--------|-------|:-:|----------|
| Context Precision | Retrieval | yes | Are the retrieved chunks relevant to the answer? |
| Context Recall | Retrieval | yes | Did retrieval surface everything the answer needs? |
| Context Entity Recall | Retrieval | yes | Fraction of the reference's entities present in context. |
| Faithfulness | Generation | no | Is every claim in the answer grounded in the context? |
| Response Relevancy | Generation | no | Does the answer actually address the question? |
| Factual Correctness | Generation | yes | Do the answer's claims match the ground truth? |
| Semantic Similarity | Generation | yes | Embedding similarity of answer to ground truth. |
| Noise Sensitivity ↓ | Generation | yes | How often irrelevant context corrupts the answer (lower is better). |
| Conciseness (custom) | Custom | no | A custom `AspectCritic` you define yourself. |

That last row is the point: Ragas lets you write your own pass/fail metric in
two lines (`AspectCritic(name=..., definition=...)`), so evals can encode
whatever "good" means for your app.

## Prereqs

The gemma4 vLLM server from the sibling project should already be deployed and
ACTIVE on the devbox (`topics/gemma4/gemma4-dgx-devbox`). The eval reaches it by
its cluster-internal Knative DNS name (see `VLLM_URL` in `config.py`).

Set up the venv:

```bash
cd topics/ragas/ragas-demo
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Create the per-directory Flyte config (same cluster as the gemma4 / rag-chroma
projects):

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

## 1. Run the eval

Runs go to the devbox (remote). Local runs can't reach the in-cluster vLLM, so
don't lead with `--local`.

```bash
# quick but meaningful pass: full index, just a few questions
flyte run eval_pipeline.py ragas_eval --max_questions 5

# full eval
flyte run eval_pipeline.py ragas_eval
```

Open the run URL and look at the **report** tab on the `ragas_eval` node for the
grouped scorecard; the `ragas_score` node has the same scorecard plus a
per-question table.

> **Keep the index full.** `max_docs` shrinks the *indexed corpus*, not the
> question set. A small value (say `--max_docs 300`) means the answers aren't in
> the index, retrieval misses, and every metric cascades to ~0 — which looks like
> a broken judge but is really a starved index. Use `max_questions` (not
> `max_docs`) to control runtime. Only shrink `max_docs` to smoke-test plumbing.

A real 5-question run against the self-hosted gemma judge looks like this:

```
Retrieval   context precision 0.667 · context recall 1.000 · entity recall 0.300
Generation  faithfulness 0.800 · answer relevancy 0.644 · factual correctness 0.080
            semantic similarity 0.678 · noise sensitivity 0.600 (lower is better)
Custom      conciseness 0.800
```

(Factual correctness is low because mini-wikipedia's references are terse — often
one word like "yes" — which makes its claim-level F1 harsh. A nice reminder that
a metric's score only means something once you understand what it compares.)

Knobs (all are task-arg defaults, override on the CLI):

```bash
flyte run eval_pipeline.py ragas_eval \
    --top_k 6 --max_questions 50 --judge gemma
```

- `top_k` — retrieval depth (default 4).
- `max_questions` — eval set size (default 20). Each question runs ~9 LLM-judge
  metrics, so this is the main runtime lever; keep it small for a live pass.
- `judge` — `gemma` (default, self-hosted) or `openai` (needs `OPENAI_API_KEY`).
- `rag_run` — reuse a specific `rag_pipeline` run's Chroma index (see below).

## 2. Reuse the index from rag-chroma-flyte

Instead of building an index inline, point the eval at the Chroma artifact a
`rag_pipeline` run already produced:

```bash
RAG_PIPELINE_RUN=<run-name> flyte run eval_pipeline.py ragas_eval
# or
flyte run eval_pipeline.py ragas_eval --rag_run <run-name>
```

`resolve_index` fetches that run's output `Dir` (`o0`) via
`flyte.remote.Run`. With nothing set, it builds an identical index inline so the
demo still runs end-to-end with one command.

## 3. The feedback loop — A/B compare

This is the whole pitch: tune a hyperparameter, let the eval score it, and make a
decision from numbers instead of vibes. There are two flavors, because
hyperparameters come in two kinds.

**Query-time knob — `top_k` (retrieval depth).** Doesn't touch the index, so one
index is built and `top_k` is varied over it.

```bash
flyte run eval_pipeline.py ragas_compare --top_ks '[2,6]'
```

**Index-time knob — `chunk_size`.** Changes how documents are split before
embedding, so the index is **rebuilt for each value** (the build moves inside the
loop). Same questions throughout, so scores stay comparable; `build_index` caches
per chunk_size, so re-runs are cheap.

```bash
flyte run eval_pipeline.py ragas_compare_chunking --chunk_sizes '[300,1200]'
```

Both render a **side-by-side scorecard** with the winner highlighted per metric.
Small chunks tend to retrieve more precisely but fragment context; large chunks
carry more context per hit but blur precision. The same compare shape generalizes
to any knob: embedding model, system prompt, even the answering LLM.

## 4. Live eval playground (Gradio)

An interactive app: pick a test-set question (full 9-metric suite, since it has a
ground-truth answer) or type your own (reference-free metrics only), drag the
`top_k` slider, and watch RAG answer it and Ragas grade that single response live
with color-coded metric chips and the retrieved contexts.

It mounts the Chroma index from a standalone `build_index` run, so build one
first, then deploy pinned to it:

```bash
# 1. build (or reuse) an index as a top-level run
flyte run eval_pipeline.py build_index --max_docs 0
# -> copy the run name

# 2. deploy the app pinned to that index run
RAGAS_INDEX_RUN=<build_index-run-name> python eval_app.py
```

With no `RAGAS_INDEX_RUN`, the app falls back to the latest succeeded
`build_index` run. The deployed URL is printed at the end (a `*.localhost:30081`
Knative address, same as the other devbox apps). The app scales to zero after 5
minutes idle. It reuses `ragas_lib.evaluate_one` + `render_chips`, so the chips
match the pipeline scorecard exactly.

## Swapping the judge live

The local gemma4 model is great but smaller; some Ragas metrics lean on strict
structured output. If a metric comes back empty or noisy on stream, flip the
judge without touching code:

```bash
flyte run eval_pipeline.py ragas_eval --judge openai   # needs OPENAI_API_KEY
```

## Troubleshooting

**Every metric is ~0** — almost always a starved index, not a bad judge. If you
shrank `max_docs`, the answers aren't in the index, retrieval misses, and the
`don't-know` answers tank every downstream metric. Run with the full index
(`--max_docs 0`, the default) and only lower `max_questions` for speed. The
generation-side metrics that don't depend on retrieval (semantic similarity,
conciseness) staying non-zero while everything else is 0 is the tell.

**A single metric column is blank / all NaN** — the judge returned malformed
structured output for that one metric. Re-run with `--judge openai` to confirm
it's the model and not the data. `raise_exceptions=False` is set so one bad
metric dents a cell instead of failing the run.

**`Bad git executable`** — `import ragas` pulls in GitPython, which hard-fails
with no git binary in the container. `config.py` sets `GIT_PYTHON_REFRESH=quiet`
before any ragas import to neutralize it.

**`No QA pairs loaded`** — confirm the `question-answer` config name/split in
`config.py` matches the dataset (`QA_CONFIG` / `QA_SPLIT`).

**Retrieval looks random** — the query encoder must match the index encoder.
Both are `BAAI/bge-small-en-v1.5` here; if you reuse a `rag_pipeline` run built
with a different `embedding_model`, rebuild or align them.

**vLLM connection refused** — the eval reaches gemma4 by its cluster-internal
Knative DNS name. Confirm `VLLM_APP_NAME` in `config.py` matches the deployed
app (default `gemma4-26b-a4b-it-vllm`).

## Next ideas

- **Synthetic test data** — swap `load_qa` for Ragas `TestsetGenerator` to
  generate questions (single-hop, multi-hop, reasoning) straight from the
  corpus, no hand-labeled answers needed.
- **Wire it into a loop** — run `ragas_eval` on a Flyte schedule over fresh
  production traffic; alert when a metric regresses. That's the real "evals as a
  feedback loop" story.
- **More configs** — extend `ragas_compare` to sweep chunk size, embedding
  model, or system prompt, not just `top_k`.
