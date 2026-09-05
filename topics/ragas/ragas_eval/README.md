# Ragas RAG Evaluation

Automated evaluation of three RAG backends (Vector RAG, Graph RAG, Cognee) using [Ragas](https://docs.ragas.io/) metrics. Built on top of the `rag_comparison` project — this app adds a scoring layer that objectively measures retrieval and answer quality across all three architectures.

**Metrics:** faithfulness · answer_relevancy · context_precision · context_recall  
**UI:** Gradio on port 7861 — Batch Evaluation tab + Single Question tab  
**Judge LLM:** Claude Haiku 4.5 via Anthropic API  
**Embeddings:** BAAI/bge-small-en-v1.5 via HuggingFace Inference API (pure HTTP)

---

## Architecture

```
User (browser :7861)
  └─ Gradio UI (app/ui.py + app/handlers.py)
       └─ Eval Runner (eval/runner.py)
            ├─ eval/backends.py ──► Vector RAG  (pgvector/Supabase)
            │                  ──► Graph RAG   (Neo4j AuraDB)
            │                  ──► Cognee      (LanceDB, local)
            ├─ ragas.evaluate() ──► Claude Haiku (faithfulness judge)
            │                  ──► HuggingFace  (embeddings for answer_relevancy)
            └─ eval/testset.py ──► data/testset.json (20 Q&A pairs)
```

This project extends `johndellenbaugh/rag-comparison:latest`. That base image ships the three RAG backends at `/app/backends/`. `ragas_eval` mounts on top and adds the scoring layer without duplicating any backend code.

---

## Prerequisites

### Required services (same as `rag_comparison`)
| Service | What it's for |
|---|---|
| [Anthropic API key](https://console.anthropic.com/) | Claude Sonnet 4.6 for RAG answers; Claude Haiku 4.5 for Ragas scoring |
| [Supabase](https://supabase.com/) pgvector | Vector RAG backend |
| [Neo4j AuraDB Free](https://neo4j.com/cloud/aura/) | Graph RAG backend |
| Cognee LanceDB | Cognee backend — local on the VM, volume-mounted into Docker |
| [HuggingFace account](https://huggingface.co/settings/tokens) | Free token for embeddings API (needed for `answer_relevancy` metric) |

### Runtime environment
- Docker (tested on Docker 24+)
- GCP VM or any Linux host with at least 2 GB RAM
- The `rag_comparison` Docker image must exist and have ingested data into all three backends before this project will produce meaningful results

---

## GCP VM Setup

This project runs on a GCP Compute Engine VM that also hosts the `rag_comparison` app (port 7860). Both containers run on the same VM because Cognee's LanceDB is stored on the local filesystem and must be volume-mounted.

### VM specification

| Setting | Value |
|---|---|
| Machine type | e2-standard-2 (2 vCPU, 8 GB RAM) or larger |
| OS | Ubuntu 22.04 LTS |
| Boot disk | 30 GB SSD minimum (Cognee LanceDB grows with ingestion) |
| Region | Match your Neo4j AuraDB region to reduce latency |

> **CPU note:** GCP's default e2 and n1 Xeon CPUs do not support AVX512. This is expected — the workarounds in the Dockerfile handle it. Do not use a GPU instance; this project does not need one.

### Initial VM setup

```bash
# Install Docker
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io
sudo usermod -aG docker $USER
# Log out and back in for group change to take effect
```

### Place your .env file

```bash
# On the VM, create ~/.env with all required variables
nano ~/.env
```

The Cognee volume path must match what was used when the `rag_comparison` container ingested data. Check the existing `rag-comparison.service` on your VM to confirm the volume mount path, then use the same path here.

### Firewall rules

```bash
# Run from your local machine (gcloud authenticated)
gcloud compute firewall-rules create allow-rag-ragas \
  --allow tcp:7861 \
  --project <your-gcp-project-id>

# The rag_comparison app uses port 7860 (separate rule)
# gcloud compute firewall-rules create allow-rag-comparison --allow tcp:7860 ...
```

> **GCP project ID:** Check which project your VM belongs to with:
> ```bash
> curl -s "http://metadata.google.internal/computeMetadata/v1/project/project-id" \
>   -H "Metadata-Flavor: Google"
> ```
> The firewall rule must be in the **same project as the VM** — creating it in any other project has no effect.

### Updating a running deployment

```bash
# On the VM — pull new image and restart
docker pull <your-dockerhub-username>/rag-comparison-ragas:latest
sudo systemctl restart rag-comparison-ragas
sudo journalctl -u rag-comparison-ragas -n 20 --no-pager
```

The app is healthy when the journal shows `* Running on local URL: http://0.0.0.0:7861`.

---

## Quickstart

### 1. Clone and configure

```bash
git clone https://github.com/johndell-914/ai-build-and-learn.git
cd ai-build-and-learn/topics/ragas/ragas_eval
cp .env.example .env
# Edit .env with your actual values
```

### 2. Build and push the Docker image

```bash
docker build --no-cache -t <your-dockerhub-username>/rag-comparison-ragas:latest .
docker push <your-dockerhub-username>/rag-comparison-ragas:latest
```

> **Always use `--no-cache`** when rebuilding. Docker aggressively caches the `COPY` layer and will silently ship stale code without it.

### 3. Deploy on the VM

Create the systemd service:

```bash
sudo tee /etc/systemd/system/rag-comparison-ragas.service > /dev/null <<'EOF'
[Unit]
Description=RAG Comparison Ragas Evaluation
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStart=/usr/bin/docker run --rm --name rag-comparison-ragas \
  --dns 8.8.8.8 \
  --env-file /home/<your-user>/.env \
  -e PYTHONUNBUFFERED=1 \
  -v /home/<your-user>/.cognee_system:/usr/local/lib/python3.11/site-packages/cognee/.cognee_system \
  -p 7861:7861 \
  docker.io/<your-dockerhub-username>/rag-comparison-ragas:latest
ExecStop=/usr/bin/docker stop rag-comparison-ragas

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable rag-comparison-ragas
sudo systemctl start rag-comparison-ragas
```

> **`--dns 8.8.8.8` is required.** GCP's internal DNS (169.254.169.254) does not resolve `router.huggingface.co` which is needed for embeddings. The `--dns` flag routes DNS through Google's public resolver for that domain.

Open the firewall:

```bash
gcloud compute firewall-rules create allow-rag-ragas \
  --allow tcp:7861 \
  --project <your-gcp-project-id>
```

### 4. Generate the testset (one time)

The testset is 20 question/ground-truth pairs generated from the Everstorm PDFs. Run this once after the container is up:

```bash
docker exec -e PYTHONUNBUFFERED=1 -w /app rag-comparison-ragas \
  python ragas_eval/generate_testset.py 2>&1
```

This uses `claude-sonnet-4-6` to generate one Q&A pair per PDF chunk, samples evenly across all 15 PDFs, and saves to `data/testset.json`.

Once generated, copy it out and commit it so you don't have to regenerate on every rebuild:

```bash
docker cp rag-comparison-ragas:/app/ragas_eval/data/testset.json data/testset.json
git add data/testset.json && git commit -m "add generated testset"
```

### 5. Open the app

```
http://<vm-external-ip>:7861
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | Anthropic API key |
| `PG_URL` | ✅ | Supabase Session Pooler URL (must be IPv4 pooler, not direct connection) |
| `NEO4J_URI` | ✅ | Neo4j AuraDB connection URI |
| `NEO4J_PASSWORD` | ✅ | Neo4j password |
| `NEO4J_USERNAME` | optional | Defaults to `neo4j` |
| `HF_TOKEN` | recommended | HuggingFace token. Without it the free tier allows ~100 API calls/day — tight for a full batch eval run. Get one free at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) |

---

## Project Structure

```
ragas_eval/
├── main.py                  # Entry point — launches Gradio on 0.0.0.0:7861
├── config.py                # Env vars, model IDs, data paths
├── ragas_wrappers.py        # get_ragas_llm() and get_ragas_embeddings() factories
├── compat.py                # Stubs broken ragas/langchain deps at import time
├── generate_testset.py      # One-shot script: PDFs → data/testset.json
├── Dockerfile
├── requirements.txt
├── .env.example
│
├── eval/
│   ├── backends.py          # BACKENDS dict — imports from rag_comparison
│   ├── metrics.py           # METRICS list (4 Ragas metrics)
│   ├── testset.py           # load_testset() / save_testset()
│   └── runner.py            # _eval_one_backend(), run_single_eval(), run_batch_eval()
│
├── app/
│   ├── handlers.py          # Gradio event handlers (generator functions)
│   └── ui.py                # Gradio layout — no business logic
│
├── data/
│   └── testset.json         # 20 Q&A pairs (committed after first generation)
│
└── static/
    └── app.css
```

---

## Models

| Role | Model | Notes |
|---|---|---|
| RAG answer generation | `claude-sonnet-4-6` | Used by the rag_comparison backends |
| Ragas judge (faithfulness, precision, recall) | `claude-haiku-4-5-20251001` | ~300–500 LLM calls per batch run; Haiku keeps cost down |
| Embeddings (answer_relevancy) | `BAAI/bge-small-en-v1.5` | Served via HuggingFace Inference API at `router.huggingface.co` |
| Testset generation | `claude-sonnet-4-6` | Called once by `generate_testset.py` |

---

## Known Issues and Workarounds

These are environment-specific issues encountered on GCP VMs and documented here so you don't have to rediscover them.

### 1. ONNX crash (No AVX512 on CPU)

**Symptom:** Process exits silently after "NumExpr defaulting to 2 threads."

**Cause:** `fastembed` and several other libraries use the ONNX runtime compiled with AVX512 instructions. Older Intel Xeon CPUs on GCP (e.g., @ 2.20GHz) don't support AVX512.

**Fix (already in Dockerfile):**
- `ENV NUMEXPR_DISABLE=1` disables NumExpr SIMD
- Embeddings use the HuggingFace Inference API over HTTP instead of local fastembed
- `ragas_wrappers.py` uses `llm_factory` + `AsyncAnthropic` instead of `LangchainLLMWrapper` (which also crashed via `os._exit()` in the ragas executor)

### 2. HuggingFace DNS not resolving

**Symptom:** `socket.gaierror: [Errno -2] Name or service not known` for `router.huggingface.co`.

**Cause:** GCP's internal DNS resolver (169.254.169.254) doesn't resolve certain HuggingFace domains. The old `api-inference.huggingface.co` domain has no IPv4 A record (deprecated by HuggingFace).

**Fix:** Pass `--dns 8.8.8.8` to `docker run`. This is in the systemd service template above.

### 3. Anthropic rejects `temperature` + `top_p` together

**Symptom:** `400 Bad Request: temperature and top_p cannot both be specified for this model`

**Cause:** ragas 0.4.3's `llm_factory` sets both `temperature` and `top_p` in `model_args` by default. Anthropic's API rejects requests that specify both.

**Fix (in `ragas_wrappers.py`):**
```python
if hasattr(llm, "model_args"):
    llm.model_args.pop("top_p", None)
```

### 4. ragas imports `langchain_community.chat_models.vertexai` (removed in 0.3.x)

**Symptom:** `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

**Cause:** ragas 0.4.3 internally imports this module, which was removed in `langchain-community` 0.3.x.

**Fix (in Dockerfile):** A stub file is written into the installed package at build time:
```dockerfile
RUN python -c "import os, sys; p = next(...); p and open(os.path.join(p, 'vertexai.py'), 'w').write('class ChatVertexAI: pass\n')"
```

### 5. Config module collision

**Symptom:** `ImportError: cannot import name 'VECTOR_INDEX_NAME' from 'config'`

**Cause:** Both `ragas_eval/config.py` and `rag_comparison/config.py` exist. Python caches the first one imported as `sys.modules['config']`. The rag_comparison backends import their own config — they get the wrong one.

**Fix (in `eval/backends.py`):** The ragas_eval config is temporarily removed from `sys.modules` before importing the backends, then restored after:
```python
_our_config = sys.modules.pop("config", None)
try:
    from backends import graph as _graph
    ...
finally:
    if _our_config is not None:
        sys.modules["config"] = _our_config
```

### 6. `max_tokens` too low for faithfulness scoring

**Symptom:** `The output is incomplete due to a max_tokens length limit` — faithfulness returns N/A.

**Cause:** The faithfulness metric generates an NLI chain-of-thought response that exceeds the default 1024 token limit.

**Fix (in `ragas_wrappers.py`):** `max_tokens=4096` is passed to `llm_factory`.

---

## Batch Evaluation

The batch eval runs all 20 testset questions against all 3 backends and returns mean scores per backend. It takes ~5–10 minutes.

- Questions run **sequentially** to avoid API rate limits
- Backends run **concurrently** per question via `asyncio.gather()`
- A per-backend failure returns NaN scores and never cancels sibling tasks
- Partial results are written atomically after every question — an interrupted run can be inspected at `data/results_partial.json`

---

## Single Question Evaluation

The single question tab evaluates one question in real time across all three backends. Ground truth is optional — without it, `context_recall` scores as N/A (it requires a reference answer to compare against).

---

## Interpreting Results

| Metric | What it measures | Requires |
|---|---|---|
| `faithfulness` | Did the answer stick to the retrieved context? (1.0 = no hallucination) | question, answer, contexts |
| `answer_relevancy` | Did the answer address the question? (uses embeddings) | question, answer |
| `context_precision` | Were the retrieved chunks actually useful? | question, contexts, ground_truth |
| `context_recall` | Did retrieval find all the information needed? | question, contexts, ground_truth |

A `faithfulness` score below 1.0 means the LLM added claims not present in the retrieved context. `context_precision` = 1.0 means every retrieved chunk was relevant. `context_recall` < 1.0 means some relevant information wasn't retrieved.
