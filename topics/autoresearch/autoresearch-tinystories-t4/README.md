# AutoResearch — TinyStories / GCP T4

An adaptation of [Karpathy's AutoResearch](https://github.com/karpathy/autoresearch) for a **GCP T4 GPU (16GB)** using the **TinyStories** dataset.

An AI agent runs overnight, autonomously modifying `train.py`, training a small GPT for 5 minutes, measuring `val_bpb` (validation bits per byte), and keeping or discarding each change. Results are logged to **Google Firestore** and visualized in a **local Gradio dashboard**.

---

## Project design

### Concept
Karpathy's AutoResearch asks: *what if an AI agent did ML research overnight, the same way a human researcher would — hypothesize, test, measure, iterate?*

The agent has a single objective: minimize `val_bpb` (validation bits per byte) on a language modeling task. It does this by:
1. Reading its strategy guide (`program.md`) and the current `train.py`
2. Asking Claude to propose one focused, testable change
3. Running a fixed 5-minute training budget
4. Measuring whether the change helped or hurt
5. Keeping improvements, discarding regressions
6. Repeating ~12 times per hour overnight

The result is autonomous ML experimentation — the agent explores the hyperparameter and architecture space while you sleep.

### Why this is interesting
- **Reward hacking in reverse** — instead of an agent gaming a metric, this agent is genuinely trying to improve a meaningful one
- **Approximate hill-climbing** — 5-minute runs are noisy, but signal accumulates over ~100 experiments
- **Interpretable** — every change is logged with a human-readable description and diff
- **The artifact is the code** — `train.py` at the end of the run has been iteratively improved by the agent

### Architecture overview

```
┌─────────────────────────────────────────────┐
│              GCP T4 Instance                │
│                                             │
│  agent.py                                   │
│    │                                        │
│    ├── reads program.md (strategy)          │
│    ├── reads train.py (current state)       │
│    ├── calls Claude API → proposed change   │
│    ├── writes new train.py                  │
│    ├── runs train.py (5 min)                │
│    ├── parses val_bpb from stdout           │
│    ├── keep or revert train.py              │
│    └── logs to Firestore via               │
│        firestore_logger.py                  │
│                                             │
└──────────────────┬──────────────────────────┘
                   │ writes
                   ▼
┌─────────────────────────────────────────────┐
│           Google Firestore                  │
│                                             │
│  runs/{run_id}/                             │
│    started_at, config, ended_at             │
│    experiments/{exp_id}/                    │
│      experiment_number                      │
│      change_description                     │
│      change_diff                            │
│      val_bpb_before / val_bpb_after         │
│      delta, kept                            │
│      train_loss, step_count                 │
│                                             │
└──────────────────┬──────────────────────────┘
                   │ reads
                   ▼
┌─────────────────────────────────────────────┐
│         Local Machine (your laptop)         │
│                                             │
│  dashboard/app.py (Gradio)                  │
│    ├── val_bpb progression chart (Plotly)   │
│    ├── experiment log table                 │
│    ├── stat row (bpb, kept, success rate)   │
│    └── run summary card                     │
│                                             │
│  Auto-refreshes every 60 seconds            │
│  Authenticated via gcloud ADC               │
└─────────────────────────────────────────────┘
```

### Module responsibilities

| File | Responsibility |
|------|---------------|
| `agent.py` | Main loop: reads, proposes, applies, trains, evaluates, logs |
| `train.py` | GPT model + training loop. **Only file the agent modifies** |
| `prepare.py` | One-time dataset download and tokenization. Never modified |
| `firestore_logger.py` | All Firestore reads/writes. No other I/O |
| `metrics.py` | val_bpb parsing, delta calculation, keep/revert decision. Pure functions, no I/O |
| `program.md` | Natural language strategy guide read by the agent each experiment |
| `dashboard/app.py` | Gradio layout and wiring |
| `dashboard/ui_components.py` | HTML builders and Plotly charts. No framework dependency |
| `dashboard/styles.css` | Dark theme CSS |

---

## T4 adaptations — what we changed from Karpathy's original

Karpathy's AutoResearch targets an **H100 80GB** GPU. Running on a **T4 16GB** requires six specific changes:

### 1. Flash Attention 3 → PyTorch scaled_dot_product_attention

**Original code:**
```python
cap = torch.cuda.get_device_capability()
repo = "varunneal/flash-attention-3" if cap == (9, 0) else "kernels-community/flash-attn3"
fa3 = get_kernel(repo).flash_attn_interface
# ...
out = fa3(q, k, v, ...)
```

**Why it fails on T4:** Flash Attention 3 requires compute capability 9.0 (H100). The T4 is compute capability 7.5. Even the fallback kernel targets Ampere/Hopper and is incompatible.

**Our fix:**
```python
out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```
Standard PyTorch attention — no custom kernel, fully compatible with T4 and any CUDA-capable GPU.

---

### 2. DEVICE_BATCH_SIZE: 128 → 16

**Why:** 128 sequences × 1024 tokens at float32 would require ~16GB for activations alone on an H100. The T4 has 15.3GB total VRAM. Reducing to 16 sequences fits comfortably.

**Tradeoff:** Each forward pass processes fewer tokens, so gradient accumulation steps increase to maintain the effective batch size.

---

### 3. TOTAL_BATCH_SIZE: 2\*\*19 → 2\*\*17

**Why:** 524K tokens per gradient update with a small DEVICE_BATCH_SIZE requires `524288 / (16 × 1024) = 32` gradient accumulation steps per optimizer update. That's too slow for a 5-minute budget. Reducing to 131K tokens means 8 accumulation steps — a better fit for the time constraint.

---

### 4. WINDOW_PATTERN: "SSSL" → "LLLL"

**Why:** The original pattern `"SSSL"` uses **sliding window attention** on some layers. This is implemented via a custom kernel tied to Flash Attention 3. Without FA3, there is no sliding window implementation. Setting all layers to `"L"` (full attention) removes this dependency entirely.

**Tradeoff:** Full attention is O(n²) in sequence length. For 1024-token sequences this is fine on a T4. Sliding window becomes important at much longer contexts.

---

### 5. Dataset: climbmix-400b → TinyStories

**Why:** climbmix-400b is a 400-billion token web-scale dataset. Downloading it would take hours and it's far larger than needed for 5-minute training runs. The agent would barely scratch the surface of it per experiment.

**TinyStories** ([roneneldan/TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)) is a dataset of short children's stories generated by GPT-3/4, created by Microsoft Research specifically for training and evaluating small language models.

**Why TinyStories is the right choice for this project:**

- **Size** — ~2GB total. Downloads in minutes on a GCP instance. climbmix-400b would take hours just to download and would never be fully utilized in 5-minute training runs.

- **Signal density** — TinyStories uses simple, consistent English grammar and vocabulary. This makes it much easier for a small model to learn meaningful patterns quickly. A 5-minute training run on TinyStories produces a measurable, meaningful `val_bpb` — the model is actually learning something. The same run on a noisy web-crawl dataset would produce very little signal.

- **Designed for small models** — The dataset was specifically created by Microsoft Research to study what language capabilities small models (1M–33M parameters) can and cannot learn. Our GPT (~25M parameters) is exactly the target audience. The dataset difficulty is well-matched to the model size.

- **Fast convergence** — Because the language is simple and repetitive, the model converges quickly. This is essential when each experiment only gets 5 minutes. On a more complex dataset, the model might not improve measurably within the budget, making keep/revert decisions unreliable.

- **Reproducible benchmark** — TinyStories is widely used in small model research. `val_bpb` scores on TinyStories are comparable across experiments and across different runs, making it a reliable metric for the agent's hill-climbing loop.

- **Character-level tokenization compatibility** — TinyStories' limited vocabulary (simple English words) works well with our character-level tokenizer, keeping `vocab_size` small and the model efficient.

---

### 6. MuonAdamW optimizer → standard AdamW

**Why:** The original uses a custom `MuonAdamW` optimizer combining Muon (with polar express orthogonalization) and AdamW. This is a research optimizer not available as a standard package. For T4 adaptation, we use standard `torch.optim.AdamW` which is well-understood, stable, and lets the agent focus on architecture and hyperparameter experiments rather than optimizer internals.

---

### Summary of changes

| Change | Original (H100) | This repo (T4) | Reason |
|--------|----------------|----------------|--------|
| Attention | FA3 custom kernel | `F.scaled_dot_product_attention` | FA3 requires compute cap 9.0 |
| `DEVICE_BATCH_SIZE` | 128 | 16 | VRAM limit |
| `TOTAL_BATCH_SIZE` | 2\*\*19 (524K) | 2\*\*17 (131K) | Gradient accum steps for 5-min budget |
| `WINDOW_PATTERN` | "SSSL" | "LLLL" | Sliding window requires FA3 kernel |
| Dataset | climbmix-400b | TinyStories | Download size and run budget |
| Optimizer | MuonAdamW | AdamW | Availability and simplicity |

---

## What it does

```
GCP T4 Instance
  └── agent.py (overnight loop)
        └── Asks Claude to propose one change to train.py
        └── Trains GPT on TinyStories for 5 minutes
        └── Measures val_bpb (lower = better)
        └── Keeps change if improved, reverts if not
        └── Logs result → Firestore
        └── Repeats ~12x per hour

Google Firestore
  └── One document per experiment
        └── change made, val_bpb before/after, kept/reverted

Local Machine
  └── dashboard/app.py (Gradio)
        └── Reads from Firestore
        └── Renders val_bpb progression chart
        └── Shows experiment log
```

---

## Key adaptations from Karpathy's original (H100 → T4)

| Parameter | H100 Original | T4 Adaptation |
|-----------|--------------|---------------|
| Flash Attention | FA3 custom kernel | `F.scaled_dot_product_attention` |
| `DEVICE_BATCH_SIZE` | 128 | 16 |
| `TOTAL_BATCH_SIZE` | 2\*\*19 | 2\*\*17 |
| `WINDOW_PATTERN` | "SSSL" | "LLLL" |
| Dataset | climbmix-400b (400B tokens) | TinyStories (~2GB) |

---

## Project structure

```
autoresearch-tinystories-t4/
  ├── agent.py              — main overnight loop
  ├── train.py              — GPT training script (only file agent modifies)
  ├── prepare.py            — TinyStories download + tokenization (run once)
  ├── firestore_logger.py   — all Firestore reads/writes
  ├── metrics.py            — val_bpb parsing and keep/revert logic
  ├── program.md            — agent strategy guide (natural language)
  ├── requirements.txt
  └── dashboard/
        ├── app.py          — Gradio dashboard
        ├── ui_components.py— charts, tables, stat cards
        └── styles.css      — dark theme
```

---

## Prerequisites

### GCP account
- A Google Cloud Platform account with billing enabled
- A GCP project (note your **Project ID**)

### Local machine
- [gcloud CLI](https://cloud.google.com/sdk) installed and authenticated
- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com)

---

## Part 1 — GCP Setup

### 1. Enable APIs
In the GCP console search bar, enable both:
- `Compute Engine API`
- `Cloud Firestore API`

### 2. Create Firestore database
- Go to Firestore in the GCP console
- Click **"Create Database"**
- Select **Native mode**
- Choose your preferred region (pick US region for low dashboard latency)
- Note the **database name** you give it

### 3. Create the writer service account
- Go to IAM & Admin → Service Accounts
- Create a service account named `autoresearch-writer`
- Grant it the role: `Cloud Datastore User`
- No key needed — the VM uses it automatically

### 4. Grant your Google account Firestore read access (for local dashboard)
- Go to IAM & Admin → IAM
- Click "Grant Access"
- Principal: your Google email
- Role: `Cloud Datastore Viewer`

> **Note:** If your org blocks service account key creation, use `gcloud auth application-default login` locally instead of a JSON key file. This is the recommended approach.

### 5. Check GPU quota
- Go to IAM & Admin → Quotas & System Limits
- Search `NVIDIA T4`
- Find `NVIDIA_T4_GPUS` for your target region
- If limit is `0`, request an increase to `1` (may take a few hours)

### 6. Create the T4 VM instance
- Go to Compute Engine → VM Instances → Create Instance
- **Name:** `autoresearch-t4`
- **Region/Zone:** choose a zone where you have T4 quota (e.g. `us-central1-a`)
- **Machine type:** `n1-standard-4`
- **GPU:** Add GPU → NVIDIA T4 → quantity 1
- **Boot disk:** Change → Public Images → Deep Learning on Linux → Deep Learning VM with PyTorch + CUDA → 100 GB
- **Service account:** `autoresearch-writer`
- **Access scopes:** Allow full access to all Cloud APIs
- Click **Create**

> **Note:** If the Custom Images tab can't find `deeplearning-platform-release` (org policy restriction), use the **Public Images** tab instead and search for "Deep Learning on Linux".

---

## Part 2 — T4 Instance Setup

### SSH into the instance
On Windows use the **Google Cloud SDK Shell**:
```bash
gcloud compute ssh autoresearch-t4 --zone=YOUR_ZONE
```

First time connecting, accept the prompt to install NVIDIA drivers. The instance will reboot — SSH back in after ~2 minutes.

### Verify GPU
```bash
nvidia-smi
```
Should show Tesla T4 with ~15360 MiB VRAM.

### Clone the repo
```bash
git clone https://github.com/johndell-914/ai-build-and-learn.git
cd ai-build-and-learn/topics/autoresearch/autoresearch-tinystories-t4
```

### Create virtual environment
```bash
sudo apt install python3.12-venv -y
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

> **Note:** The Deep Learning VM has PyTorch pre-installed system-wide but the venv requires an explicit install. `requirements.txt` handles this.

### Set environment variables (permanent)
```bash
echo 'export GCP_PROJECT=your-project-id' >> ~/.bashrc
echo 'export FIRESTORE_DATABASE=your-database-name' >> ~/.bashrc
echo 'export ANTHROPIC_API_KEY=your-anthropic-key' >> ~/.bashrc
echo 'source ~/ai-build-and-learn/topics/autoresearch/autoresearch-tinystories-t4/venv/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

### Verify Firestore connection
```bash
python3 -c "
from google.cloud import firestore
import os
db = firestore.Client(project=os.environ['GCP_PROJECT'], database=os.environ['FIRESTORE_DATABASE'])
db.collection('test').add({'message': 'connected'})
print('Firestore write OK')
"
```

---

## Part 3 — Prepare the dataset

Run once to download TinyStories and create tokenized training data:
```bash
python3 prepare.py
```

This creates:
- `data/train.bin` — tokenized training split
- `data/val.bin` — tokenized validation split
- `data/meta.json` — vocabulary metadata

Takes ~5 minutes depending on network speed.

---

## Part 4 — Test a single training run

Verify `train.py` runs correctly and outputs `val_bpb`:
```bash
python3 train.py
```

Runs for 5 minutes. At the end you should see:
```
val_bpb=X.XXXXXX
train_loss=X.XXXXXX
steps=XXXX
```

---

## Part 5 — Test the agent loop (short run)

Run a 30-minute test (~6 experiments) before committing to overnight:
```bash
RUN_HOURS=0.5 python3 agent.py
```

Watch for:
- `Run started: <run_id>` — Firestore run document created
- `Experiment 1 | val_bpb=X.XXXX` — agent loop running
- `KEPT` or `REVERT` — keep/revert decision after each experiment
- `Firestore write OK` — results logging to Firestore

---

## Part 6 — Overnight run

Run in a `tmux` session so it continues if your SSH connection drops:
```bash
tmux new -s autoresearch
python3 agent.py
```

Detach with `Ctrl+B` then `D`. Reattach anytime with:
```bash
tmux attach -t autoresearch
```

Default run time is **8 hours** (~96 experiments). Override with:
```bash
RUN_HOURS=12 python3 agent.py
```

---

## Part 7 — Local Gradio dashboard

### Authenticate locally
```bash
gcloud auth application-default login
```

### Set environment variables
```bash
export GCP_PROJECT=your-project-id
export FIRESTORE_DATABASE=your-database-name
```

### Install dashboard dependencies
```bash
pip install gradio plotly google-cloud-firestore
```

### Run the dashboard
```bash
cd dashboard
python3 app.py
```

Open `http://localhost:7860` in your browser.

The dashboard shows:
- **Stat row** — current val_bpb, total experiments, kept count, success rate
- **val_bpb chart** — progression over time, green=kept, red=reverted
- **Experiment log** — every experiment with change description and delta
- **Run summary** — narrative summary of the overnight run

Auto-refreshes every 60 seconds during an active run.

---

## How val_bpb works

**val_bpb = validation bits per byte** — how many bits of information the model needs to predict each byte of unseen text. Lower is better.

| Stage | Typical val_bpb |
|-------|----------------|
| Untrained | 4.0+ |
| Early training | 2.0–2.5 |
| Decent small model | 1.6–1.8 |
| Well-tuned small model | 1.4–1.6 |

After each 5-minute training run, the agent compares val_bpb before and after its change. If it improved by more than 0.001, the change is kept. Otherwise `train.py` is reverted to its previous state.

---

## The agent strategy

The agent reads `program.md` before each experiment. This file defines:
- What parameters are safe to modify
- What must not be changed
- Strategy hints for TinyStories / short training runs

You can edit `program.md` to guide the agent's behavior without touching any code.

---

## Cost estimate

| Resource | Cost |
|----------|------|
| T4 instance (n1-standard-4 + T4) | ~$0.50/hr |
| 8-hour overnight run | ~$4.00 |
| Firestore (free tier) | $0.00 |

**Stop the instance when done:**
```bash
gcloud compute instances stop autoresearch-t4 --zone=YOUR_ZONE
```

---

## Known issues and workarounds


| Issue | Fix |
|-------|-----|
| Org policy blocks JSON key creation | Use `gcloud auth application-default login` locally |
| Custom images tab can't find deeplearning-platform-release | Use Public Images tab instead |
| `python3-venv` not installed | `sudo apt install python3.12-venv -y` |
| torch not found in venv | `pip install torch` (or use requirements.txt) |
| Firestore `(default)` database not found | Set `FIRESTORE_DATABASE=your-database-name` env var |
| SSH drops during overnight run | Use `tmux` — run detaches from terminal |
