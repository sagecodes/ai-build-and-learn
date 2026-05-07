# Graph RAG with Neo4j on Flyte 2

A complete Graph-RAG demo running on the DGX Spark Flyte 2 devbox: Neo4j 5
as a Flyte app (with native vector index), a pipeline that loads a toy
corpus of AI papers, and a Gradio chat UI with three retrieval modes
(pure vector, vector + 1-hop graph expand, hybrid RRF) talking to Gemma 4
through vLLM.

```
┌────────────────────┐  HTTP /db/neo4j/tx/commit  ┌────────────────────┐
│   pipeline.py      │ ─────────────────────────▶ │   neo4j_app.py     │
│ toy AI papers      │                            │  Flyte AppEnv      │
│  → bge-small       │   nodes, edges,            │  neo4j:5.26        │
│  → MERGE Cypher    │   vector index             │  HTTP on 7474      │
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
| `pipeline.py` | Three tasks: `fetch_papers` → `embed_papers` → `load_neo4j` (HTTP Cypher), wrapped by `graphrag_pipeline`. |
| `chat_app.py` | Gradio `AppEnvironment` with three retrieval modes (vector, vector + expand, hybrid RRF). Streams from Gemma 4 vLLM, queries Neo4j over HTTP. |
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

The Neo4j browser UI loads through Knative since 7474 is HTTP. If the
browser auth flow fights with Knative's relative paths, port-forward and
hit Neo4j directly:

```bash
kubectl -n flyte port-forward \
    svc/graphrag-neo4j-flytesnacks-development-00001 7474:80
# then open http://localhost:7474 in a browser
```

## 2. Run the pipeline

```bash
flyte run --local --tui pipeline.py graphrag_pipeline
# or remote:
flyte run pipeline.py graphrag_pipeline
```

`wipe_first=True` by default so re-runs are idempotent. Override on the CLI:

```bash
flyte run --local pipeline.py graphrag_pipeline --wipe_first false
```

Expected counts on the toy corpus:

```
papers: 17
authors: 49
categories: 3
cites_edges: 26
authored_edges: 50
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

Expected ranking (top 1 should be the actual RAG paper, ~0.93 cosine):

```
  0.931  Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
  0.907  Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection
  0.890  Atlas: Few-shot Learning with Retrieval Augmented Language Models
  ...
```

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

### The three retrieval modes

The right-hand panel shows `📄 Retrieved papers` plus, in modes 2 and 3,
`🕸 Graph relations` so the audience can see exactly what graph context
the LLM got beyond raw vector hits. Each paper card has a `via …` source
label so you can tell why it surfaced.

1. **Vector** is pure `db.index.vector.queryNodes`. Baseline. Same shape
   as the `rag-chroma` chat app from the previous week.
2. **Vector + Expand** runs the vector query, then a single 1-hop
   traversal across `CITES`, `AUTHORED_BY`, `IN_CATEGORY` for every seed.
   Adds the neighbor titles into the LLM context as a `GRAPH RELATIONS`
   block.
3. **Hybrid (RRF)** runs the vector query *and* a Cypher pass for
   most-cited papers in the same `Category` as the vector hits, then
   fuses both lists with reciprocal rank. Surfaces papers that are
   authoritative in the topic but whose abstract isn't a great vector
   match.

### Demo prompts

Pick one and flip the mode radio to show what changes:

- *"What's the relationship between RAG and Self-RAG?"*: mode 2 surfaces
  the explicit `CITES` edge between them.
- *"Compare BERT and Sentence-BERT."*: mode 1 finds them via abstracts;
  mode 2 also pulls the `BERT → Sentence-BERT` citation edge.
- *"Who are the most influential authors in retrieval-augmented
  generation?"*: mode 3 promotes highly-cited papers via the graph that
  pure vector misses.

## Known limitations

- **No persistence.** Pod restart wipes the graph. The pipeline is the
  source of truth; just re-run it (`embed_papers` is cached in rustfs so
  it's ~10s). Anything you typed by hand into the Neo4j browser does not
  survive. See "Snapshot to rustfs" under Next ideas.
- **HTTP API, not Bolt.** Functionally equivalent for our scale, but more
  verbose than the Bolt driver. We don't get the nice transaction objects
  or retry helpers; queries go one statement per HTTP round trip.
- **Toy data.** 17 hardcoded AI/ML papers (`TOY_PAPERS` in `pipeline.py`).
  Real datasets are the obvious next step.

## Next ideas

- **Real dataset.** Swap `TOY_PAPERS` for an arXiv / Semantic Scholar
  fetch. S2 is the cleanest source because it carries citation edges out
  of the box; arXiv API alone has no citation data.
- **Text-to-Cypher mode.** A 4th chat mode where the LLM writes the
  Cypher itself from the question. Demoable with Gemma 4 + a tight
  prompt that includes the schema.
- **Persistence + larger corpus.** PVC-backed Neo4j and a 5–10k paper
  graph would make the demo feel less toy without changing any of the
  retrieval code.
- **Snapshot to rustfs (ephemeral compute, durable state).** The graph
  itself dies with the pod, but the Flyte object store (rustfs) survives
  devbox restarts. Add a periodic snapshot task that dumps the live graph
  via `apoc.export.cypher.all` (or `neo4j-admin database dump`) and
  uploads it as a `flyte.io.Dir`. On Neo4j startup, restore from the
  latest snapshot if one exists. Two design notes for whoever picks this
  up:
  - Don't trust shutdown hooks for the actual save: Knative gives ~30s
    grace before SIGKILL, easy to overrun on a real corpus. Snapshot on
    a timer (or after each pipeline run) and treat shutdown as
    best-effort flush.
  - Wiring this into the current `command=` based app means either
    writing a wrapped entrypoint (shell script that pulls the snapshot
    before `exec neo4j`) or switching back to an `@env.server` Python
    wrapper so the `@on_startup` / `@on_shutdown` decorators fire.
