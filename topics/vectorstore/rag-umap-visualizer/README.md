# RAG with embedding-space visualizer

Same Chroma + Gemma 4 RAG demo as `rag-chroma-flyte`, with a 2D UMAP
projection of the index next to the chat. When you ask something, you see
*where* the model went looking — the query lands as a gold star, the top-k
chunks light up in rank-colored markers, and the rest of the corpus stays
muted gray. Pairs nicely with explaining what RAG actually does.

```
                     ┌──────────────────────────┐
                     │  rag-chroma-flyte run    │
                     │   chroma persist dir     │
                     └────────────┬─────────────┘
                                  │ RunOutput(task_name=…)
                                  ▼
┌───────────────────────────────────────────────────────────────────┐
│   rag-umap-viz  (Flyte AppEnvironment)                             │
│                                                                    │
│   on_startup:                                                      │
│     load Chroma → fetch all embeddings → fit UMAP → cache 2D       │
│                                                                    │
│   per turn:                                                        │
│     embed query → Chroma top-k → reducer.transform(query)          │
│     → redraw Plotly (gray collection + colored top-k + gold ⭐)    │
│     → stream Gemma answer with chunks injected                     │
└───────────────────────────────────────────────────────────────────┘
```

## What you see

- **Whole collection** — every chunk plotted as a small gray dot.
- **Retrieved top-k** — the chunks Chroma returned for this query, painted in
  rank-order colors (red = #1, orange = #2, …). Same colors used for the
  numbered chunk cards on the left, so the chart and the citations agree.
- **Query** — gold star at the projected position of `encoder.encode(query)`
  fed through the same fitted UMAP reducer.
- **Hover** — chunk text + similarity for the colored markers; truncated text
  on the gray points.

The Gemma 4 answer streams as before, citing chunks as `[#1]`, `[#2]`, …
that map to the colors on the chart.

## Why this is teaching-useful

- The "vectors" in *vector store* stop being abstract — viewers see clusters
  forming around topics.
- You can see why a retrieval succeeded or failed: if the query star lands
  far from the lit-up neighbors, similarity is poor and the answer probably
  shouldn't be trusted.
- Toggle **Use retrieval** off and re-ask: the chart goes inert, Gemma's
  answer comes from parametric memory only — direct contrast with what RAG
  adds.

## Files

| File | What it does |
|------|--------------|
| `chat_app.py` | Single-file Gradio app. `@on_startup` fits UMAP once; `@server` runs chat + Plotly viz. |
| `requirements.txt` | Same stack as `rag-chroma-flyte` plus `umap-learn` + `plotly`. |

## Prereqs

- Gemma 4 vLLM running (sibling project: `topics/gemma4/gemma4-dgx-devbox/`).
- A successful `rag-chroma-flyte` pipeline run — that's the index this app
  reads. Grab the run name from the Flyte UI or the run-name printed at
  the end of `python pipeline.py`.

## Deploy

```bash
cd topics/vectorstore/rag-umap-visualizer
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# Pin a specific pipeline run (recommended):
RAG_PIPELINE_RUN=<run-name> python chat_app.py
RAG_PIPELINE_RUN=rkvqzl9kxsz77fgbr7rl python chat_app.py
# or omit RAG_PIPELINE_RUN to use the latest succeeded run of `rag_pipeline`:
python chat_app.py
```

Cold start is slower than the plain RAG app — the on-startup UMAP fit takes
~10–30s on 3k points (default rag-mini-wikipedia corpus). Subsequent queries
are ms.

## Demo flow for the stream

1. Open the URL — chart shows the muted gray cloud of every chunk in the
   index.
2. Ask **"Who was Abraham Lincoln?"** — the query star lands inside a
   tightly-clustered region, top-k chunks light up nearby. Gemma answers
   citing them.
3. Ask **"What is photosynthesis?"** — query jumps to a different cluster.
   Drives home the "vectors group by topic" idea.
4. Ask something the corpus doesn't cover, e.g. **"Who won the 2024 World
   Series?"** — query lands in empty space, distances are large, answer
   should refuse.
5. Toggle **Use retrieval off**, re-ask the Lincoln question — chart goes
   inert, Gemma answers from training data only. Show the contrast.

## Knobs

- **Top-k chunks** — how many neighbors get highlighted. 4 is the default;
  pushing it to 10 makes the colored cluster more visible.
- **Use retrieval** — off = pure Gemma chat with no chart highlights.
- **Enable thinking** — Gemma 4's `<|channel>` thought block toggle. Off is
  faster.

## UMAP knobs (in `chat_app.py`)

- `n_components=2` — 2D projection for the plot.
- `n_neighbors=15` — local-vs-global tradeoff. Lower = tighter local clusters,
  higher = smoother global structure.
- `min_dist=0.1` — minimum spacing between points; larger values give more
  whitespace between clusters.
- `metric="cosine"` — matches BGE's normalized embeddings.
- `random_state=42` — deterministic layouts so the demo looks the same each
  cold start.

## Troubleshooting

**Cold start hangs at "Fitting UMAP…"** — UMAP's `numba` JIT compile takes
~15s the first time on a fresh container. Subsequent fits in the same pod
are fast. If it never finishes, check for arm64 numba wheels.

**Query star lands in the middle of nowhere** — UMAP `transform()` for
out-of-sample points can drift if the query is very dissimilar from the
corpus. That's not a bug, it's the truth: your question doesn't live in the
same neighborhood as your data.

**`ParameterMaterializationError: No runs found for task …`** — same fix
as the RAG demo: run `rag-chroma-flyte/pipeline.py` once, or pin
`RAG_PIPELINE_RUN`.

## Next ideas

- **Cluster discovery** — HDBSCAN on the 2D coords + a Gemma call to label
  each cluster ("'US history', 'European geography', 'Cell biology', …").
  The colored regions become legible at a glance.
- **Drift over time** — if you index a doc corpus that changes, save UMAP
  coords as a Flyte artifact per pipeline run; diff two runs to show what
  moved.
- **Plug-in mode** — drop the same on-startup hook into the agent-memory
  app and watch memories accumulate as live points on the chart.
