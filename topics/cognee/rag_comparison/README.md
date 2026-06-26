# RAG Comparison

> **"Your AI agent is only as good as what it can retrieve."**

Most RAG tutorials pick one approach and call it done. This project picks three — and lets you watch them compete on the same question, in real time.

Ask anything about a fictional outdoor gear company. Vector RAG, Graph RAG, and Cognee each retrieve context in completely different ways. Claude generates an answer from each, then evaluates all three and names a winner. A second tab flips the question: given the same retrieved context, does the model matter? Claude and Gemma 4 answer side-by-side so you can see for yourself.

**This is a portfolio-grade demo of three retrieval strategies and two frontier models — built to show architectural judgment, not just working code.**

---

## What You'll See

### Tab 1 — RAG Comparison
One question. Three retrieval strategies. One verdict.

| Backend | Strategy | Storage |
|---|---|---|
| Vector RAG | Cosine similarity over document chunks | Supabase pgvector |
| Graph RAG | Knowledge graph traversal with intelligent routing (hybrid / entity / community) | Neo4j AuraDB Free |
| Cognee | Automated hybrid retrieval — vector index + knowledge graph | LanceDB + SQLite on VM |

All three fire in parallel. Claude evaluates retrieval quality, answer quality, and names a winner with a scored breakdown table.

### Tab 2 — Model Comparison
Same question, same retrieved context, different models.

Pick a RAG backend from the dropdown. Retrieval runs once. Both **Claude Sonnet 4.6** and **Gemma 4 26B** (via Vertex AI MaaS) generate answers from identical context — so any difference in output is pure generation quality, not retrieval luck. Claude evaluates both and scores them across five criteria.

---

## Architecture

```
User question
      │
      ├─── Tab 1: RAG Comparison ──────────────────────────────────────────┐
      │         asyncio.gather() fires all 3 backends in parallel          │
      │         ┌──────────────┬──────────────┬─────────────────┐          │
      │         ▼              ▼              ▼                 │          │
      │    vector.py      graph/__init__  cognee_backend.py     │          │
      │    pgvector        Neo4j           LanceDB + SQLite      │          │
      │    Supabase       AuraDB Free      (local on VM)         │          │
      │         └──────────────┴──────────────┘                 │          │
      │                        │                                │          │
      │              shared/embeddings.py ← fastembed singleton │          │
      │              shared/claude.py     ← generate_answer()   │          │
      │                        │                                │          │
      │              Claude evaluates all 3 → verdict table     │          │
      └─────────────────────────────────────────────────────────┘          │
      │                                                                     │
      └─── Tab 2: Model Comparison ────────────────────────────────────────┘
                User picks RAG backend from dropdown
                retrieve() called once → context string
                asyncio.gather() runs both models in parallel
                ┌──────────────────────┬──────────────────────┐
                ▼                      ▼
          shared/claude.py       shared/gemma.py
          Claude Sonnet 4.6      Gemma 4 26B
          Anthropic API          Vertex AI MaaS
                └──────────────────────┘
                         │
              Claude evaluates both → scored breakdown
```

**Key design decisions:**
- No workflow orchestration in the query path — all backends are direct async function calls. Adding Flyte/Airflow here would add seconds of overhead for zero benefit.
- Shared fastembed singleton (`BAAI/bge-small-en-v1.5`, 384 dims) — same embeddings across all backends ensures retrieval comparison fairness.
- Each backend exposes both `query()` (retrieve + generate) and `retrieve()` (retrieve only) — Tab 2 uses `retrieve()` to get context once, then passes it to both models.

---

## Infrastructure

| Component | Where |
|---|---|
| App + all backends | GCP e2-medium VM (Docker + systemd) |
| Cognee storage | Local files on GCP VM (LanceDB + SQLite) |
| Vector store | Supabase pgvector (cloud, free tier) |
| Graph store | Neo4j AuraDB Free (cloud, free tier) |
| Claude generation | Anthropic API |
| Gemma 4 generation | Vertex AI MaaS (GCP, pay-per-token) |

---

## Prerequisites

- Python 3.11+
- Docker (for local build and GCP deployment)
- [Supabase](https://supabase.com) project with pgvector extension enabled
- [Neo4j AuraDB Free](https://neo4j.com/cloud/aura-free/) instance
- [Anthropic API key](https://console.anthropic.com)
- GCP project with Vertex AI API enabled (for Gemma 4 tab)

---

## Local Setup

```bash
git clone https://github.com/johndell-914/ai-build-and-learn.git
cd topics/cognee/rag_comparison
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

Fill in `.env` with your credentials — see `.env.example` for all required values.

### Supabase Setup
1. Create a new project at supabase.com
2. Enable the pgvector extension: **Database → Extensions → vector**
3. Copy the **Session Pooler** connection string (not direct — direct uses IPv6):
   `postgresql://postgres.<project-ref>:<password>@aws-1-us-east-1.pooler.supabase.com:5432/postgres`

### Neo4j Setup
1. Create a free instance at [console.neo4j.io](https://console.neo4j.io)
2. Copy the connection URI: `neo4j+s://<instance-id>.databases.neo4j.io`
3. Note: AuraDB Free username is the instance ID (e.g. `0d897ad6`), not `neo4j`

---

## Ingest Documents

Run each ingest script once before launching the app. Scripts read the 15 Everstorm Outfitters PDFs from `data/` and populate each backend.

```bash
python ingest/vector_ingest.py      # → Supabase pgvector (160 chunks)
python ingest/graph_ingest.py       # → Neo4j (955 entities, 855 relationships, 135 communities)
python ingest/cognee_ingest.py      # → Cognee LanceDB (15 PDFs)
```

---

## Run Locally

```bash
python app.py
# Open http://localhost:7860
```

---

## Deploy to GCP

### 1. Create the VM
```bash
gcloud compute instances create rag-comparison \
  --zone=us-west1-b \
  --machine-type=e2-medium \
  --image-family=debian-12 \
  --image-project=debian-cloud \
  --boot-disk-size=20GB \
  --scopes=cloud-platform
```

> **Important:** include `--scopes=cloud-platform` at creation time. Changing scopes later requires stopping the VM.

### 2. Open firewall
```bash
gcloud compute firewall-rules create allow-rag-comparison \
  --allow=tcp:7860 \
  --target-tags=rag-comparison
```

### 3. Grant Vertex AI access
```bash
gcloud projects add-iam-policy-binding <your-project-id> \
  --member="serviceAccount:<vm-service-account-email>" \
  --role="roles/aiplatform.user"
```

### 4. SSH and install Docker
```bash
gcloud compute ssh rag-comparison --zone=us-west1-b
# On the VM:
sudo apt-get update && sudo apt-get install -y docker.io
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
```

### 5. Create .env on VM
```bash
nano ~/.env
# Paste all credentials from your local .env
```

### 6. Create systemd service
```bash
sudo nano /etc/systemd/system/rag-comparison.service
```

```ini
[Unit]
Description=RAG Comparison Gradio App
After=docker.service
Requires=docker.service

[Service]
Restart=always
ExecStartPre=-/usr/bin/docker rm -f rag-comparison
ExecStart=/usr/bin/docker run --rm --name rag-comparison \
  --env-file /home/<your-username>/.env \
  -e PYTHONUNBUFFERED=1 \
  -v /home/<your-username>/.cognee_system:/usr/local/lib/python3.11/site-packages/cognee/.cognee_system \
  -p 7860:7860 \
  docker.io/<your-dockerhub>/rag-comparison:latest
ExecStop=/usr/bin/docker stop rag-comparison

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now rag-comparison
```

### 7. Build, push, and pull
```bash
# Local:
docker build -t <your-dockerhub>/rag-comparison:latest .
docker push <your-dockerhub>/rag-comparison:latest

# On VM:
docker pull <your-dockerhub>/rag-comparison:latest
sudo systemctl restart rag-comparison
```

### 8. Run ingest on the VM
```bash
docker exec -w /app -e PYTHONPATH=/app rag-comparison python ingest/vector_ingest.py
docker exec -w /app -e PYTHONPATH=/app rag-comparison python ingest/graph_ingest.py
docker exec -w /app -e PYTHONPATH=/app rag-comparison python ingest/cognee_ingest.py
```

> **Cognee note:** Cognee stores its databases at `/usr/local/lib/python3.11/site-packages/cognee/.cognee_system/` — not at `DATA_PATH`. The volume mount in the systemd service captures this path so data survives container restarts.

---

## Sample Questions

These questions are designed to show each backend at its best and worst:

**Factual / specific** — favors Vector RAG
- "What is the return policy for damaged gear?"
- "How many points do I earn per dollar spent at each loyalty tier?"
- "What are the requirements to qualify for a B2B corporate account?"

**Relationship / connected concepts** — favors Graph RAG
- "How do the loyalty program tiers connect to the warranty and return policies?"
- "What benefits does a Summit tier member get across all Everstorm programs?"
- "How does the referral program interact with the loyalty point system?"

**Broad / thematic** — favors Cognee
- "What is Everstorm's overall approach to sustainability?"
- "How does Everstorm handle customer data and privacy across its programs?"
- "What makes Everstorm different from a typical outdoor retailer?"

**Trick questions** — tests all three for honesty
- "What are the benefits of the Gold loyalty tier?" *(no Gold tier exists)*
- "Does Everstorm ship to Antarctica?"

---

## Document Set

15 fictional Everstorm Outfitters policy PDFs:

| Document | Topic |
|---|---|
| Accessibility Services | ADA compliance, adaptive fitting |
| Account and Security | Login, 2FA, data protection |
| B2B Corporate Orders | Bulk pricing, tax-exempt accounts |
| Extended Warranty | Summit Protect coverage tiers |
| Gift Cards | Purchase, redemption, expiry |
| International Shipping | Carriers, duties, restricted regions |
| Loyalty Program | Summit Rewards tiers (Base Camp / Summit / Everest) |
| Member Benefits Guide | Cross-program perks summary |
| Order Cancellation Policy | Windows, exceptions, refunds |
| Partner and Referral Programs | Referral links, commissions |
| Privacy and Data Policy | GDPR, data retention, opt-out |
| Product Categories and Policies | Return windows by category |
| Promo and Discount Policy | Stacking rules, exclusions |
| Store Locations and Hours | Physical stores, outlet locations |
| Sustainability and Recycling | Gear Recycle Program, B Corp status |

The domain is intentionally simple — retrieval differences stand out clearly rather than being masked by content complexity.
