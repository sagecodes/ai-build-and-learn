# Ragas Evaluation Project

## Context
Ragas is the June 5th AI Build & Learn topic — automated evaluation of RAG systems with structured metrics. This project extends `rag_comparison` by evaluating all three backends (Vector, Graph, Cognee) with Ragas metrics, completing the 4-week narrative: build each RAG type → compare side-by-side → evaluate objectively.

**Key constraint:** Cognee's LanceDB lives on the GCP VM filesystem. Ragas eval must run on the same VM to evaluate all three backends.

---

## Architecture

### Docker strategy: extend rag_comparison image
```dockerfile
FROM johndellenbaugh/rag-comparison:latest
# All 3 backends already at /app/backends/ — no duplication
COPY requirements.txt /tmp/ragas-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ragas-requirements.txt
COPY . /app/ragas_eval/
WORKDIR /app
EXPOSE 7861
CMD ["python", "ragas_eval/app.py"]
```
Backends are at `/app/backends/` from the base image — direct imports, no sys.path manipulation in Docker.

### Evaluation flow
```
Everstorm PDFs (15, in /app/data/)
    ↓ generate_testset.py — TestsetGenerator (Claude Haiku + fastembed)
data/testset.json — 20 questions + ground truths
    ↓ For each question:
    ├── vector_retrieve(q)  → context → generate_answer() → answer
    ├── graph_retrieve(q)   → context → generate_answer() → answer
    └── cognee_retrieve(q)  → context → generate_answer() → answer
    ↓ ragas.evaluate(question, answer, [context], ground_truth)
Scores: faithfulness · answer_relevancy · context_precision · context_recall
    ↓ Gradio dashboard (port 7861)
Backend × Metric comparison table
```

### Models
- **Claude Sonnet 4.6** — generates RAG answers (via rag_comparison's shared `generate_answer()`)
- **Claude Haiku 4.5** (`claude-haiku-4-5-20251001`) — Ragas judge calls (300-500 calls per batch eval, kept cheap)
- **fastembed BAAI/bge-small-en-v1.5** — Ragas embeddings (same model as rag_comparison)

### Deployment: second container on same GCP VM
- Image: `johndellenbaugh/rag-comparison-ragas:latest`
- Port: 7861
- Same `.env` file, same Cognee volume mount
- New systemd service: `rag-comparison-ragas.service`
- New firewall rule: `allow-rag-ragas` — tcp:7861

---

## File Structure
```
topics/ragas/ragas_eval/
├── PLAN.md                 # this file
├── requirements.txt        # ragas additions only (base image has everything else)
├── Dockerfile
├── config.py               # env vars, model IDs, paths
├── .env.example            # same vars as rag_comparison, no new secrets
├── ragas_wrappers.py       # get_ragas_llm() + get_ragas_embeddings()
├── generate_testset.py     # one-shot standalone script: PDFs → data/testset.json
│
├── eval/
│   ├── __init__.py         # public exports
│   ├── backends.py         # BACKENDS dict — imports from rag_comparison
│   ├── metrics.py          # metric list + configure_metrics()
│   ├── testset.py          # load_testset(), save_testset()
│   └── runner.py           # _eval_one_backend(), run_single_eval(), run_batch_eval()
│
├── app/
│   ├── __init__.py
│   ├── handlers.py         # Gradio event handler functions
│   └── ui.py               # Gradio layout and components
│
├── main.py                 # entry point — imports from app/ and launches
├── data/
│   └── testset.json        # generated once, committed to repo
└── static/
    └── app.css             # copied from rag_comparison/static/app.css
```

### Module responsibilities
- `eval/backends.py` — single place to register/add backends; easy to extend
- `eval/metrics.py` — metric definitions and LLM injection isolated from runner
- `eval/runner.py` — pure async eval logic, no Gradio dependencies
- `app/handlers.py` — thin sync wrappers bridging Gradio events to runner
- `app/ui.py` — layout only, zero business logic
- `main.py` — wires app together and calls `demo.launch()`

---

## Key Files

### `requirements.txt` (ragas additions only)
```
ragas>=0.2.0
langchain-anthropic>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
datasets>=2.20.0
pandas>=2.0.0
```
gradio, pymupdf, fastembed, anthropic, dotenv already in base image.

### `config.py`
```python
EVAL_LLM_MODEL  = "claude-haiku-4-5-20251001"   # cheap for many eval calls
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"       # matches rag_comparison
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
DATA_DIR = Path(__file__).parent / "data"
```
No path manipulation needed in Docker — backends are at /app/backends/.
For local dev only: `sys.path.insert(0, str(rag_comparison_dir))`.

### `ragas_wrappers.py`
```python
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import FastEmbedEmbeddings

def get_ragas_llm():
    return ChatAnthropic(model=EVAL_LLM_MODEL, api_key=ANTHROPIC_API_KEY, temperature=0)

def get_ragas_embeddings():
    return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
```

### `generate_testset.py` (run once)
```python
from ragas.testset import TestsetGenerator
# Load PDFs via PyMuPDFLoader from /app/data/ (same PDFs as rag_comparison)
generator = TestsetGenerator.from_langchain(
    generator_llm=llm, critic_llm=llm, embeddings=embeddings
)
testset = generator.generate_with_langchain_docs(
    documents, test_size=20, raise_exceptions=False
)
# Save to data/testset.json as [{question, ground_truth}, ...]
```
⚠️ **Version note:** `TestsetGenerator` API varies across ragas 0.2.x sub-releases. If `from ragas.testset.evolutions import ...` fails, omit distributions and use defaults. Verify: `python -c "import ragas; print(ragas.__version__)"`.

### `eval/backends.py`
```python
# Imports from rag_comparison via /app/backends/ (Docker) or sys.path (local dev)
from backends.vector import retrieve as vector_retrieve
from backends.graph import retrieve as graph_retrieve
from backends.cognee_backend import retrieve as cognee_retrieve
from backends.shared.claude import generate_answer

BACKENDS: dict[str, callable] = {
    "Vector RAG": vector_retrieve,
    "Graph RAG":  graph_retrieve,
    "Cognee":     cognee_retrieve,
}
```

### `eval/metrics.py`
```python
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]

def configure_metrics(llm, embeddings):
    """Inject LLM and embeddings into each metric."""
    for metric in METRICS:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
```

### `eval/testset.py`
```python
def load_testset() -> list[dict]:
    """Load [{question, ground_truth}, ...] from data/testset.json."""

def save_testset(records: list[dict]) -> None:
    """Save testset to data/testset.json."""
```

### `eval/runner.py`
```python
async def _eval_one_backend(name, retrieve_fn, question, ground_truth) -> dict:
    context_str, summary = await retrieve_fn(question)
    answer = await generate_answer(question, context_str)
    dataset = Dataset.from_dict({
        "question": [question], "answer": [answer],
        "contexts": [[context_str]],   # single-element list per question
        "ground_truth": [ground_truth],
    })
    result = evaluate(dataset=dataset, metrics=METRICS,
                      llm=get_ragas_llm(), embeddings=get_ragas_embeddings())
    scores = result.to_pandas().iloc[0].to_dict()
    return {name, summary, answer, **scores}

async def run_single_eval(question, ground_truth) -> list[dict]:
    """asyncio.gather across all backends."""

async def run_batch_eval() -> pd.DataFrame:
    """Load testset, run all questions, return aggregate mean scores per backend."""
```
⚠️ **Version note:** `evaluate()` `llm=`/`embeddings=` kwargs are v0.2.x preferred. Fall back to `metric.llm = llm` attribute injection if kwargs not accepted.

### `app/ui.py` — Gradio layout only
Two tabs, no business logic:
- **Batch Eval**: "Run Evaluation" button → `gr.Dataframe` (Backend × Metric aggregate scores)
- **Single Question**: question + optional ground_truth inputs → 3 columns (Vector / Graph / Cognee), each with retrieved context textbox, answer markdown, and Ragas scores markdown

### `app/handlers.py` — event handlers only
```python
def handle_batch_eval() -> pd.DataFrame:
    return asyncio.run(run_batch_eval())

def handle_single_eval(question, ground_truth) -> list:
    results = asyncio.run(run_single_eval(question, ground_truth))
    # format and return outputs list for Gradio
```

### `main.py` — entry point
```python
from app.ui import build_demo
demo = build_demo()
demo.launch(server_name="0.0.0.0", server_port=7861, theme=gr.themes.Soft())
```

### `Dockerfile`
```dockerfile
FROM johndellenbaugh/rag-comparison:latest
COPY requirements.txt /tmp/ragas-requirements.txt
RUN pip install --no-cache-dir -r /tmp/ragas-requirements.txt
COPY . /app/ragas_eval/
WORKDIR /app
EXPOSE 7861
CMD ["python", "ragas_eval/app.py"]
```

### systemd service (on VM)
```ini
[Service]
Restart=always
ExecStart=/usr/bin/docker run --rm --name rag-comparison-ragas \
  --env-file /home/johndellenbaugh/.env \
  -e PYTHONUNBUFFERED=1 \
  -v /home/johndellenbaugh/.cognee_system:/usr/local/lib/python3.11/site-packages/cognee/.cognee_system \
  -p 7861:7861 \
  docker.io/johndellenbaugh/rag-comparison-ragas:latest
```

---

## Error Handling & Resilience

### Backend failures during eval
- `_eval_one_backend()` wraps the retrieve + generate + ragas evaluate calls in a `try/except Exception`.
- On failure it returns a result dict with `"error": str(e)` and all metric scores as `float("nan")`.
- `run_single_eval()` uses `asyncio.gather(*tasks, return_exceptions=False)` — each task catches its own errors, so one backend failure never kills the others.
- `run_batch_eval()` collects partial results row-by-row; if every backend errors on a question, that row is still written to the output with NaN scores rather than dropped.

### Ragas evaluate() failures
- `evaluate()` is called inside `_eval_one_backend()` under the same `try/except`.
- Version compatibility: if `evaluate()` raises `TypeError` on `llm=`/`embeddings=` kwargs, fall back to attribute injection via `configure_metrics()` and retry without kwargs.
- If a metric returns `None` (ragas can do this for short contexts), coerce to `float("nan")` when building the result dict.

### LLM API failures (Anthropic rate limits / timeouts)
- `generate_answer()` from the base image already has retry logic; rely on it.
- Ragas judge calls (Haiku) are made internally by ragas — they will surface as exceptions caught by `_eval_one_backend()`.
- Do not add manual retries on top of ragas internals; let failures propagate to the NaN result path.

### Testset load/save failures
- `load_testset()` raises `FileNotFoundError` with a clear message if `data/testset.json` is missing.
- `save_testset()` writes to a temp file first, then `os.replace()` — prevents a partial write from corrupting the testset.
- Both functions validate that the loaded JSON is a non-empty list of dicts with `question` and `ground_truth` keys; raise `ValueError` with details if not.

### Backend connectivity failures at startup
- `eval/backends.py` imports backend modules at the top level; import errors (e.g. missing DB credentials) surface immediately on container start rather than silently at first eval call.
- Each backend's `retrieve()` is called lazily — connection errors appear as exceptions in `_eval_one_backend()` and flow to the NaN result path.

### fastembed / model download failures
- `get_ragas_embeddings()` in `ragas_wrappers.py` is called once per eval run, not per question.
- Wrap the `FastEmbedEmbeddings()` constructor in a `try/except` with a clear error message: "fastembed model download failed — check network and disk space."

### Gradio UI resilience
- `handle_batch_eval()` and `handle_single_eval()` in `app/handlers.py` wrap their `asyncio.run()` calls in `try/except Exception` and return a user-visible error string (or partial DataFrame) rather than letting Gradio surface a raw traceback.
- Batch eval can take several minutes — the "Run Evaluation" button is disabled while running (via `gr.update(interactive=False)`) to prevent concurrent runs.
- Single Question eval disables inputs while running for the same reason.

### Partial batch eval recovery
- `run_batch_eval()` accumulates results as it goes; if the process is killed mid-run the partial `data/results_partial.json` is written after each question via `save_testset()`.
- On next run, if `data/results_partial.json` exists, `run_batch_eval()` logs a warning and deletes it (does not resume — full re-run keeps logic simple).

### Container / environment
- `config.py` checks for required env vars at import time and raises `EnvironmentError` listing all missing vars before any backend is touched.
- Dockerfile `CMD` uses `python ragas_eval/main.py` (not a shell wrapper), so unhandled exceptions propagate to the container exit code and systemd `Restart=always` picks it up.

---

## Implementation Sequence
1. Scaffolding: `requirements.txt`, `Dockerfile`, `.env.example`, `config.py`, `ragas_wrappers.py`
2. Eval package: `eval/__init__.py`, `eval/backends.py`, `eval/metrics.py`, `eval/testset.py`, `eval/runner.py`
3. Testset: `generate_testset.py` → run inside Docker → verify `data/testset.json` has 20 rows
4. Test `eval/runner.py` with a single question before wiring the app
5. App package: `app/__init__.py`, `app/handlers.py`, `app/ui.py`
6. Entry point: `main.py` → launch and test both tabs
7. Docker build + push `rag-comparison-ragas:latest`
8. On VM: pull image, install systemd service, open port 7861
9. Run testset generation inside container (`docker exec`)
10. Run batch eval, confirm all 4 metrics for all 3 backends
11. Commit everything including `data/testset.json`

---

## No New Secrets Needed
Same `.env` as rag_comparison: ANTHROPIC_API_KEY, PG_URL, NEO4J_*, GCP_PROJECT, GCP_REGION, GEMMA_MODEL.

---

## Verification
1. `docker exec -w /app rag-comparison-ragas python ragas_eval/generate_testset.py` → 20 Q&A pairs
2. `docker exec -w /app rag-comparison-ragas python ragas_eval/evaluate.py` → scores printed
3. App at `http://VM_IP:7861` → Batch Eval tab shows 3-row table, Single Question works
