# Gemma 4 Smart Gallery — Research & Design

## The Event

This project was built for a Gemma 4 showcase event. Gemma 4 is Google's latest open-weights model family, released in April 2026. The event goal: build something that demonstrates what Gemma 4 can do.

---

## Research: What is Gemma 4?

Fetched from [deepmind.google/models/gemma/gemma-4](https://deepmind.google/models/gemma/gemma-4/).

**Key capabilities:**
- Multimodal input: vision and audio understanding
- Native function calling for agentic workflows
- 140 language support
- Fine-tuning support

**Model lineup:**

| Model | Params | Use case |
|-------|--------|----------|
| `gemma-4-e2b-it` | ~2.3B effective | Mobile / IoT |
| `gemma-4-e4b-it` | ~4.5B effective | Compact, high downloads |
| `gemma-4-26b-a4b-it` | 4B active (MoE) | Mid-range, confirmed MaaS |
| `gemma-4-31b-it` | 33B dense | Flagship, best reasoning |

The `-it` suffix means **instruction-tuned** — required for agent and chat use cases.

---

## What Makes Gemma 4 Different?

**Key differentiator:** Open weights. Unlike GPT-4o, Claude, and Gemini, Gemma 4 can be self-hosted or accessed via Vertex AI at no cost via MaaS (Model as a Service).

**Vision capability** is native — the model handles image + text input in a single API call, no separate pipeline needed.

---

## How to Access Gemma 4

We researched multiple access paths:

| Option | Verdict |
|--------|---------|
| Google AI Studio API | ❌ No managed API for Gemma 4 (Gemini only) |
| HuggingFace Inference API | ✅ Available, free tier, smaller models |
| HuggingFace transformers (local) | ❌ Requires GPU, impractical for 31B |
| Llama.cpp local server | ❌ Hardware constrained |
| LangChain + HuggingFace | ✅ Supported, vision capable |
| Vertex AI (MaaS) | ✅ **Selected** — Google's enterprise path, GCP already set up |

**Decision: Vertex AI MaaS**
- Google's official enterprise hosting for Gemma 4
- API calls only — no deployment, no GPU management
- New GCP project created: `gemma4-smart-gallery` (isolated from AutoResearch)
- Confirmed in Model Garden: `google/gemma-4-26b-a4b-it-maas` — public preview, MaaS ✅
- Region: `global` (not us-central1 — required for MaaS endpoint)
- `gemma-4-31b-it-maas` coming when confirmed

**SDK: `google-genai` (modern)**
- Google's current standard SDK — replaces older `vertexai.GenerativeModel`
- `google-cloud-aiplatform` ruled out — older SDK, not the documented path for MaaS
- Pattern: `genai.Client(vertexai=True, project=..., location=...)` + `Part.from_bytes()` for local images
- Model ID format: `google/gemma-4-26b-a4b-it-maas`

---

## Project Concept

**What it does:** A smart photo gallery powered by Gemma 4 vision.

1. **Generate Descriptions** — point the app at a folder of images, Gemma 4 describes each one, results are stored in SQLite and displayed as thumbnail + description cards
2. **Search** — type a keyword ("ocean", "dog", "sunset"), Gemma 4 looks at each image in real time and returns matches

**Why this showcases Gemma 4:**
- Vision capability is front and center
- Search is semantic — Gemma 4 understands intent, not just filenames
- Relatable to any audience, not just developers

---

## Architecture Decisions

### Storage: SQLite over JSON
JSON cache was considered but ruled out as not production quality. SQLite is file-based, ships with Python, handles 100k+ rows, and is a natural stepping stone to the future vector DB showcase project.

### Search: Live Vision over Cache Lookup
Search does not use cached descriptions. Instead, Gemma 4 inspects each image in real time per query. This keeps the two flows independent and makes Gemma 4's vision capability the star of the search experience.

### Model Setting via `.env`
```
GEMMA_MODEL=gemma-4-26b-a4b-it
```
Upgrading to 31B when confirmed requires zero code changes — just update `.env`.

### Separation of Concerns
Mirrors the AutoResearch dashboard pattern:
- `app.py` — Gradio UI only
- `vision_service.py` — all Gemma 4 / Vertex AI calls
- `workflows.py` — dispatcher only; routes to local or union backend
- `workflows_local.py` — Flyte tasks and runners for local backend
- `workflows_union.py` — Flyte tasks and runners for Union.ai remote backend
- `ui_components.py` — HTML builders, no framework dependency
- `styles.css` — all colors, layout, typography

### Flyte Orchestration
Both flows run as Flyte workflows so the full execution graph is visible in the Flyte TUI. This is intentional for the developer showcase audience — watching Flyte orchestrate Gemma 4 vision tasks in real time is part of the demo story.

**Why Flyte on both flows (not just Generate Descriptions):**
- Audience came to see Flyte + Gemma 4 working together
- Each image processed as a discrete, visible Flyte task
- Overhead is acceptable for a showcase; would reconsider for production UX

**Flyte workflows:**
- `describe_workflow` — `scan_images` → `describe_image` (per image) → `save_to_db`
- `search_workflow` — `load_images` → `check_image_match` (per image) → `collect_results`

`app.py` triggers both workflows. Gradio shows progress. Flyte TUI shows full execution graph.

---

## Final Architecture

```
topics/gemma4/gemma4-smart-gallary/
├── images/               ← user photos
├── gemma_photos.db       ← SQLite, keyed by full file path (generated)
├── app.py                ← Gradio UI
├── agent.py              ← CLI entry point
├── workflows.py          ← dispatcher: routes to local or union backend
├── workflows_local.py    ← Flyte tasks + runners for local backend
├── workflows_union.py    ← Flyte tasks + runners for Union.ai backend
├── vision_service.py     ← Gemma 4 / Vertex AI API calls
├── gemma_client.py       ← Vertex AI SDK wrapper
├── db.py                 ← SQLite cache operations
├── ui_components.py      ← HTML card builders
├── styles.css            ← all styling
├── .env                  ← GEMMA_MODEL + GCP credentials (not committed)
└── RESEARCH.md           ← this file
```

---

### Flyte Execution Mode: Local with Persistence

AutoResearch established the pattern — we follow it exactly.

```python
flyte.init(local_persistence=True)
```

- No remote Flyte server needed
- Tasks run in-process on local machine
- Results return directly and synchronously — no polling
- TUI visibility via `flyte start tui` (requires `pip install flyte[tui]`)
- `vision_service.py` is a module, never run directly — imported by `workflows.py`

### Flyte Parallel Execution

Flyte 2.x supports parallel fan-out via `asyncio.gather()` + `.aio` method. No need for `flytekit` or `map_task`.

```python
import asyncio

@env.task
async def describe_all(image_paths: list[str]) -> list[str]:
    results = await asyncio.gather(*[
        describe_image.aio(image_path=path) for path in image_paths
    ])
    return results
```

Each image processes as a parallel task — all visible in the Flyte TUI simultaneously.

**Flyte task breakdown:**

Generate Descriptions:
1. `scan_images(folder_path)` → list of image paths
2. `describe_image(image_path)` → description string (parallel, one task per image)
3. `save_descriptions(paths, descriptions)` → writes all results to SQLite

Search:
1. `load_images(folder_path)` → list of image paths
2. `check_image_match(image_path, query)` → True/False (parallel, one task per image)
3. `collect_results(paths, matches)` → list of matching paths

`app.py` calls `workflows.py` directly and receives results synchronously.

---

## How to Run the Demo

Two terminal windows:

| Terminal | Command | Purpose |
|----------|---------|---------|
| 1 | `python app.py` | Starts Gradio UI, opens in browser |
| 2 | `flyte start tui` | Shows Flyte task execution graph |

**Demo flow:**
1. Open browser → Gradio UI
2. Open Flyte TUI in second terminal
3. Trigger Generate Descriptions or Search in Gradio
4. Watch tasks light up in Flyte TUI simultaneously

`vision_service.py` is never run directly — it is a module imported by `workflows.py`.

---

## Union.ai Remote Backend

### Decision
Added support for running Gemma 4 tasks on a Union.ai managed cluster alongside the existing local backend. Controlled by `FLYTE_BACKEND` in `.env` — default is `local`, no behavior change for existing users.

### Why modular workflow files
`workflows.py` became a thin dispatcher importing `workflows_local` or `workflows_union` based on the env var. This keeps the local path completely isolated — zero risk of Union changes breaking the working local flow.

### Union-specific challenges
- **Local file access**: cluster containers can't read Windows local paths. Solution: scan folder locally (plain Python, not a Flyte task), base64-encode image bytes, pass as `str` to cluster tasks which decode to temp files.
- **`bytes` type**: Flyte doesn't natively support `bytes` as a task input — falls back to PickleFile which requires object store. Solution: base64-encode to `str` instead.
- **DB writes**: `save_descriptions_task` moved out of Flyte for the union path — results return to local machine and are written to SQLite there.
- **GCP credentials**: not available on cluster from local `.env`. Solution: Union secrets (`GCP_PROJECT`, `GCP_REGION`, `GEMMA_MODEL`) injected as env vars via `flyte.Secret()` in the `TaskEnvironment`.
- **Upload hang**: code bundle upload to Union's storage backend was hanging (39KB bundle, 2+ hrs). Root cause unresolved — likely a Union cluster storage configuration issue. Union integration is in progress pending support from Union.ai.

### Union secrets setup
```bash
union create secret GCP_PROJECT --project dellenbaugh --domain development
union create secret GCP_REGION --project dellenbaugh --domain development
union create secret GEMMA_MODEL --project dellenbaugh --domain development
```

### Status
Local backend: fully working. Union backend: code complete, upload hang blocking end-to-end validation.

---

## Gemma 4 Setup — Quick Reference

Minimal steps to connect any agent or script to Gemma 4 via Vertex AI.

### 1. GCP Project
Create a GCP project and enable billing.

### What is MaaS?
**Model as a Service** — you call the model via API without deploying it yourself. No GPUs, no servers, no containers. Google runs the infrastructure; you just send requests. This is how Gemma 4 is accessed on Vertex AI — enable the API in Model Garden and start calling it immediately.

### 2. Enable Gemma 4 on Vertex AI Model Garden
- Vertex AI → Model Garden → search **Gemma 4 26B A4B IT**
- Click **Enable API** — no deployment step, Google hosts everything

### 3. Authenticate locally
```bash
gcloud auth application-default login
```

### 4. Install the SDK
```bash
pip install google-genai
```

### 5. Call Gemma 4
```python
from google import genai
from google.genai.types import Part

client = genai.Client(
    vertexai=True,
    project="your-gcp-project-id",
    location="global",
)

# Text only
response = client.models.generate_content(
    model="google/gemma-4-26b-a4b-it-maas",
    contents=["What is the capital of France?"],
)

# Vision — image + text
image_bytes = open("photo.jpg", "rb").read()
image_part  = Part.from_bytes(data=image_bytes, mime_type="image/jpeg")

response = client.models.generate_content(
    model="google/gemma-4-26b-a4b-it-maas",
    contents=[image_part, "Describe this image."],
)

print(response.text)
```

### Key facts
| | |
|---|---|
| SDK | `google-genai` (not `google-cloud-aiplatform`) |
| Model ID | `google/gemma-4-26b-a4b-it-maas` |
| Region | `global` |
| Auth | `gcloud auth application-default login` |
| Pricing | MaaS — promotional/free during Gemma 4 launch period, verify in GCP Billing |
| Vision | Pass image as `Part.from_bytes()` alongside text prompt |
