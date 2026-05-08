# Graph RAG with Neo4j on Flyte 2

A complete Graph-RAG demo running on the DGX Spark Flyte 2 devbox: Neo4j 5
as a Flyte app (with native vector index), a pipeline that pulls papers
from Semantic Scholar by keyword query (cached) and loads them as a
graph, and a Gradio chat UI with three retrieval modes (pure vector,
vector + 1-hop graph expand, hybrid Reciprocal Rank Fusion / RRF)
talking to Gemma 4 through vLLM.

```
┌────────────────────┐  HTTP /db/neo4j/tx/commit  ┌────────────────────┐
│   pipeline.py      │ ─────────────────────────▶ │   neo4j_app.py     │
│ S2 keyword fetch   │                            │  Flyte AppEnv      │
│  → bge-small       │   nodes, edges,            │  neo4j:5.26        │
│  → UNWIND MERGE    │   vector index             │  HTTP on 7474      │
└────────────────────┘                            └─────────┬──────────┘
                                                            │
                                                            │ Cypher
                                                            ▼
┌────────────────────┐  Bge-small encode + Cypher  ┌────────────────────┐
│  Gemma 4 26B vLLM  │ ◀────────────────────────── │   chat_app.py      │
│  (sibling project) │   chat completion stream    │  Gradio AppEnv     │
└────────────────────┘                             │  3 retrieval modes │
                                                   └────────────────────┘
```

## Files

| File | What it does |
|------|--------------|
| `config.py` | Flyte `TaskEnvironment` for the pipeline, shared Neo4j connection constants. |
| `Dockerfile.neo4j` | One-line wrapper around `neo4j:5.26-community`. Built via `flyte.Image.from_dockerfile` to skip the `USER flyte` footer that breaks the container. |
| `neo4j_app.py` | Flyte `AppEnvironment` running the neo4j image. HTTP on 7474, no persistence. |
| `pipeline.py` | Three tasks: `fetch_papers` (Semantic Scholar, cached) → `embed_papers` → `load_neo4j` (HTTP Cypher, batched via UNWIND), wrapped by `graphrag_pipeline`. |
| `chat_app.py` | Gradio `AppEnvironment` with three retrieval modes (vector, vector + expand, hybrid RRF). Streams from Gemma 4 vLLM, queries Neo4j over HTTP. |
| `snapshot.py` | Two Flyte tasks: `snapshot_neo4j` dumps the live graph to a `flyte.io.Dir` (JSONL, embeddings included); `restore_neo4j` replays it back. |
| `requirements.txt` | Local deps: `flyte[tui]`, `httpx`, `sentence-transformers`, `gradio`, `openai`, `kubernetes`. |

## Why this is shaped the way it is

Two things to know up front. They explain choices that look weird in the
code and that you will hit immediately if you try to swap things around.

**HTTP, not Bolt.** Flyte 2 deploys apps as Knative Serving services. The
queue-proxy sidecar that fronts every Knative pod only routes HTTP. Bolt
(TCP/7687) does not pass through. We use Neo4j's HTTP Cypher API on 7474
instead, which supports the full Cypher surface including the native
vector-index queries.

**`from_dockerfile`, not `from_base`.** The installed Flyte 2.2.3 image
builder appends `USER flyte` and `WORKDIR /home/flyte` to every image it
builds. The official neo4j image has no `flyte` user, so containerd fails
container creation with `no users found`. `from_dockerfile` skips that
footer entirely, which is why `Dockerfile.neo4j` is a one-liner: a bare
`FROM neo4j:5.26-community` is enough.

## Prereqs

Same Flyte 2 devbox as the vectorstore project, **started with `--gpu`** so the
sibling Gemma 4 vLLM app has a GPU to schedule onto:

```bash
flyte start devbox --gpu
docker exec flyte-devbox nvidia-smi -L           # verify GB10 visible
kubectl get nodes -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}'
# should print: 1
```

If the box was already started without `--gpu`, the only way to flip it on is
`flyte delete devbox` then `flyte start devbox --gpu` (a plain stop/start
just resumes the existing container without re-applying flags). See
`../../gemma4/gemma4-dgx-devbox/SPARK_SETUP.md` for the full GPU setup.

Set up the venv:

```bash
cd topics/graphs-neo4j/graphrag-neo4j-flyte
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Same Flyte CLI config (shared `flytesnacks/development`):

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

## 1. Deploy Neo4j

```bash
python neo4j_app.py
```

Pulls `neo4j:5.26-community` (multi-arch; arm64 picked automatically on the
DGX Spark), deploys it as a Flyte app named `graphrag-neo4j` with
`replicas=(1,1)` (always-on), exposes HTTP on 7474.

After deploy:

```
Neo4j app deployed: http://graphrag-neo4j-flytesnacks-development.localhost:30081/
  HTTP Cypher API: .../db/neo4j/tx/commit
  Browser UI:      .../browser/
  User: neo4j
  Password: graphrag-demo
```

The Neo4j browser UI loads through Knative since 7474 is HTTP, but its
connect dialog defaults to Bolt on 7687, which Knative does not route.
To actually log in, port-forward the pod (not the Knative service: those
only expose 80/443) and connect over HTTP. `--address 0.0.0.0` makes the
forward reachable from another machine on the same network (Tailscale,
LAN), which is how the demo is usually run:

```bash
# revision number drifts, so look up the live pod
kubectl -n flyte get pods | grep graphrag-neo4j

kubectl -n flyte port-forward --address 0.0.0.0 \
    pod/<that-pod-name> 7474:7474
```

Open `http://<host>:7474` in a browser, where `<host>` is the Spark's
Tailscale/LAN IP (or `localhost` if you're on the Spark itself). In the
connect dialog, pick `http://` from the protocol dropdown (not `bolt://`)
and use the **same host** as the connect URL, since the browser JS runs
on the client:

```
Connect URL:  http://<host>:7474
Username:     neo4j
Password:     graphrag-demo
```

## 2. Run the pipeline

```bash
flyte run pipeline.py graphrag_pipeline
```

Defaults: query `"retrieval augmented generation language models"`,
`max_papers=400`. Both are CLI-overridable. `fetch_papers` is cached on
those two args, so re-runs with the same query skip Semantic Scholar.
`wipe_first=True` makes Neo4j reloads idempotent.

```bash
flyte run pipeline.py graphrag_pipeline \
    --query "graph neural networks" --max_papers 200
flyte run pipeline.py graphrag_pipeline --wipe_first false
```

The fetcher makes two S2 calls: `/paper/search/bulk` (one shot, up to
1000 results, sorted by `citationCount:desc`) for paper metadata, then
`/paper/batch` (up to 500 IDs per POST) for the citation edges. Bulk
doesn't accept `references.*` in its `fields` parameter so the second
call is necessary; bulk also avoids the brutal rate-limit behavior on
the relevance-paginated `/paper/search` endpoint. Citation-sorting
yields a stronger demo corpus: the most-cited papers matching the
query, which the foundational ones (RAG, BERT, GPT-3, Self-RAG) all
land in. Both calls retry on 429 / 5xx with exponential backoff. If
the references call fails after retries, the graph still loads with
no CITES edges (modes 1 and 3 still work; mode 2 falls back to
AUTHORED_BY / IN_CATEGORY neighbors). Set `S2_API_KEY` in the env to
skip the anonymous shared limit. The successful result is cached, so
this only matters on the first run for a given query/max_papers.

Typical counts on the default query, 400 papers (varies as S2's index updates):

```
papers: ~380       # some papers in S2 don't have abstracts; we drop those
authors: ~1500
categories: ~5     # S2 fieldsOfStudy is coarse: CS, Linguistics, …
cites_edges: ~600  # only edges where both endpoints are in the corpus
authored_edges: ~1500
```

## 3. Verify the vector index

Quickest check from a Python REPL inside the venv. The Knative ingress
serves Neo4j directly, so no port-forward is needed if you set the Host
header:

```python
import httpx
from sentence_transformers import SentenceTransformer

vec = SentenceTransformer("BAAI/bge-small-en-v1.5") \
        .encode(["retrieval augmented generation"], normalize_embeddings=True) \
        .tolist()[0]

c = httpx.Client(
    base_url="http://localhost:30081",
    headers={"Host": "graphrag-neo4j-flytesnacks-development.localhost"},
    auth=("neo4j", "graphrag-demo"),
    timeout=20.0,
)
r = c.post("/db/neo4j/tx/commit", json={"statements": [{
    "statement": (
        "CALL db.index.vector.queryNodes('paper_embedding_idx', 5, $vec) "
        "YIELD node, score RETURN node.title, score"
    ),
    "parameters": {"vec": vec},
}]})
for row in r.json()["results"][0]["data"]:
    print(f"  {row['row'][1]:.3f}  {row['row'][0]}")
```

Expected: the original RAG paper (Lewis et al., 2020) ranks at the top
with a cosine score around 0.92, followed by Self-RAG, Atlas, and other
RAG-family papers that S2 returned for the seed query.

## 4. Deploy the chat app

Prereq: the Gemma 4 vLLM server from the sibling project must be running.

```bash
cd ../../gemma4/gemma4-dgx-devbox
python prefetch_model.py                                  # one-time
GEMMA_PREFETCH_RUN=<run-name> python vllm_server.py       # deploy
cd -
```

Then deploy the chat UI:

```bash
python chat_app.py
```

URL is logged at the end:

```
Graph RAG chat UI deployed: http://graphrag-chat-ui-flytesnacks-development.localhost:30081/
```

### Why graph at all

Pure vector retrieval ranks by embedding cosine, which is great for
topical overlap and bad at two specific things:

1. **Intellectual lineage.** Papers that build on each other often use
   different vocabulary. RAG and Self-RAG are tightly related, but their
   abstracts share few keywords. Vector misses the link; an explicit
   `CITES` edge captures it.
2. **Authority.** The most influential paper in a topic is rarely the
   tightest abstract match for a casual query. Vector surfaces what
   *sounds* like the question; the graph surfaces what the field built
   on.

Each edge type pulls a different lever:

- `CITES` is the strongest signal. It encodes a human judgment that two
  papers are intellectually connected. Mode 2 expansion uses it to add
  foundational dependencies and follow-on work to the LLM context.
- `IN_CATEGORY` powers cohort lookups in mode 3: "what are the most-cited
  papers in the same field as the vector hits?"
- `AUTHORED_BY` is the weakest for accuracy but matters for
  author-shaped questions ("what else has Asai worked on?").

### The three retrieval modes

The right-hand panel shows `📄 Retrieved papers` plus, in modes 2 and 3,
`🕸 Graph relations` so the audience can see exactly what graph context
the LLM got beyond raw vector hits. Each paper card has a `via …` source
label so you can tell why it surfaced.

**1. Vector.** Pure `db.index.vector.queryNodes` against the bge-small
embeddings. Baseline; same shape as the `rag-chroma` chat app. Use this
when the query is direct and well-phrased ("explain RAG").

**2. Vector + Expand.** Run the vector query, then for each seed paper
hop one edge across `CITES`, `AUTHORED_BY`, `IN_CATEGORY` and add the
neighbor titles to the LLM context as a `GRAPH RELATIONS` block (capped
at `EXPAND_NEIGHBOR_LIMIT = 8` per seed in `chat_app.py`). Wins when the
answer requires connecting papers whose abstracts don't overlap but
which sit one citation hop apart. *Question shape:* "How does X extend
Y?", "What does X cite?", "Compare A and B."

**3. Hybrid (Reciprocal Rank Fusion, RRF).** Two queries run side by
side:
- Vector top-k (same as mode 1).
- A Cypher pass for the most-cited papers `IN_CATEGORY` of the vector
  hits, excluding the hits themselves. This is the graph-only signal: a
  paper a pure-graph retriever would surface because it's authoritative
  in the topic, even if its abstract isn't a great match.

The two lists are fused with reciprocal rank fusion: for each list, a
paper's contribution is `1 / (RRF_K + rank)` (with `RRF_K = 60`), and
the per-paper scores are summed across both lists. The trick is that
ranks are commensurable across lists even when the raw scores aren't
(cosine similarity vs citation count), so no normalization is needed.
Papers in both lists win the most weight; lone hits stay in but ranked
lower. Wins when a foundational paper isn't a tight semantic match for
the query but everyone in the topic builds on it. *Question shape:*
"What are the influential papers in...?", "What's the canonical work
on...?"

### When to use which

- **Mode 1 (Vector)** for direct, well-phrased questions about a single
  paper or concept.
- **Mode 2 (Expand)** for relationship questions, where the answer lives
  in citation structure rather than abstract similarity.
- **Mode 3 (Hybrid)** when authority/centrality in the topic matters more
  than abstract phrasing.

Each retrieved paper card links to the actual paper (arXiv when
available, Semantic Scholar otherwise), so you can click through during
the demo to show the source.

### Demo prompts

Question shape determines which mode shines. Pick one per mode and flip
the radio to show what changes; the right-hand panel makes the
difference visible. Exact behavior depends on which papers your S2
query returned, but the patterns hold.

**Mode 1 (Vector): definition or single-paper questions.** These live
in one abstract, so the graph block adds little.

- *"What is RAG?"*
- *"Explain Self-RAG."*
- *"How does dense passage retrieval differ from BM25?"*

**Mode 2 (Vector + Expand): exploring around the hits.** Mode 2 walks
1-hop edges from the vector hits and dumps neighbor titles into the
GRAPH RELATIONS block, but does *not* promote neighbors to retrieved
entries. The model can mention neighbors as related work, but can't
reason about them in depth (no abstracts).

- *"For each retrieved paper, what does it cite and what cites it?"*
- *"Which authors appear across multiple retrieved papers?"*
- *"What graph-related papers are connected to the retrieved set?"*

**Mode 3 (Hybrid RRF): authority and lineage questions.** The
category-cohort lane brings foundational papers (most-cited in the
corpus) in as full retrieved entries with abstracts. Because recent
papers cite those foundational ones, lineage traces actually work in
this mode: both endpoints of a `CITES` edge typically end up retrieved,
so Gemma can walk the chain.

- *"Trace the citation lineage between the retrieved papers. What
  builds on what?"* (canonical lineage demo): typically pulls the
  original RAG paper in as `via graph (cited 100+x)` and Gemma traces
  `[#1] (Chronicles) → CITES → [#2] (RAG)` in its visible thinking.
- *"What are the most influential papers on retrieval-augmented
  generation in this corpus?"*
- *"Which papers should I read first to understand modern RAG?"*

**Mode 2 vs Mode 3 in one sentence.** Mode 2 walks edges *from* the
vector hits but doesn't bring those neighbors into the retrieved set,
so the model sees neighbor titles only. Mode 3 fuses a graph-only
ranking *alongside* the vector hits, so foundational papers become
first-class retrieved entries with abstracts and edges. That's why
Mode 3 wins both authority demos *and* lineage demos: both endpoints of
a citation tend to be retrieved, so Gemma can actually trace the chain
instead of staring at unfamiliar titles.

**Why a "What is X?" query won't show graph reasoning.** Mode 2 on a
definition question still includes the graph block, but the model
correctly ignores it because the abstract already answers the question.
Forcing it to cite edges anyway would just add noise. Graph context
earns its keep on relational questions. If you want to see graph
reasoning live, ask something where the answer requires walking from
one paper to another.

## 5. Snapshot / restore (optional)

The Neo4j pod has no persistent volume, so anything you type into the
browser between pipeline runs disappears when the pod cycles. `snapshot.py`
dumps the live graph (nodes, edges, **embeddings**) to a `flyte.io.Dir`
sitting in rustfs, which survives `flyte stop devbox` / `flyte start
devbox`. Restore replays it via HTTP MERGE.

```bash
# Take a snapshot of the current graph. Outputs a Dir (nodes.jsonl + edges.jsonl).
flyte run snapshot.py snapshot_neo4j
# → "Snapshot run: <run-name>"

# Restore that snapshot back into Neo4j. Pass the Dir from the snapshot run.
flyte run snapshot.py restore_neo4j \
    --snapshot=flyte://flytesnacks/development/<snapshot-run-name>/o0

# One-shot smoke test: snapshot, wipe, restore. Exit codes track success.
flyte run snapshot.py snapshot_then_restore
```

Notes worth knowing:

- The snapshot is **online**: pure Cypher over HTTP, no daemon stop. Works
  on Neo4j community edition (which has no online `neo4j-admin database
  dump`).
- Embeddings round-trip exactly. After a restore, querying any
  `Paper.embedding` against the index returns the same paper at score
  `1.000`.
- Snapshot is `wipe_first=True` on restore by default, so the target Neo4j
  ends up matching the snapshot exactly. Set `--wipe_first false` if you
  want to merge a snapshot on top of existing data.

## Known limitations

- **No persistence by default.** Pod restart wipes the graph. Re-run
  the pipeline to rebuild it; both `fetch_papers` (S2) and `embed_papers`
  are cached in rustfs, so re-runs with the same query are fast. For
  hand-edited graph state, take a snapshot first (see step 5).
- **HTTP API, not Bolt.** Functionally equivalent for our scale, but more
  verbose than the Bolt driver. We don't get the nice transaction objects
  or retry helpers; the loader compensates by batching with `UNWIND`.
- **Coarse categories.** S2's `fieldsOfStudy` are broad (Computer Science,
  Linguistics, …), so `IN_CATEGORY` doesn't discriminate as sharply as
  arXiv's `cs.IR` / `cs.CL`. Mode 3 (RRF) still works, just less
  selectively. Layering arXiv categories on top is a separate fetch.

## Next ideas

- **Text-to-Cypher mode.** A 4th chat mode where the LLM writes the
  Cypher itself from the question. Demoable with Gemma 4 + a tight
  prompt that includes the schema.
- **Persistence + larger corpus.** PVC-backed Neo4j and a 5–10k paper
  graph would make the demo feel weightier without changing any of the
  retrieval code.
- **Auto-snapshot on a timer.** Right now `snapshot.py` is on-demand. A
  scheduled Flyte task that snapshots every N minutes (or after each
  pipeline run) would mean any pod-cycle disaster only loses N minutes
  of browser edits.
- **Annotate-style entrypoint hook.** Today the snapshot is a separate
  task you run manually after edits. A shell entrypoint wrapping
  `/startup/docker-entrypoint.sh neo4j` could auto-restore from the
  latest snapshot on boot and best-effort save on SIGTERM. Keeps the
  current `from_dockerfile` path; the trade-off is that Knative's ~30s
  grace before SIGKILL means the shutdown save is best-effort only,
  so you'd still want the periodic timer for real durability.
