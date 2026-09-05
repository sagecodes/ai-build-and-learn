---
title: Vertex AI
weeks: [gemma4]
---

Google Cloud's managed AI platform. In the series, used specifically for
**Model as a Service (MaaS)** — calling hosted open-weight models (Gemma 4)
via API without deploying or managing GPUs. Enable the model in Model Garden,
authenticate with `gcloud`, call via the `google-genai` SDK.

MaaS: you call the model via API; Google runs the infrastructure. No deployment
step, no containers, no GPU quota needed. Available for select models in
Vertex AI Model Garden.

**SDK:** `google-genai` (not the older `google-cloud-aiplatform`). Pattern:
```python
from google import genai
from google.genai.types import Part

client = genai.Client(vertexai=True, project="...", location="global")
response = client.models.generate_content(
    model="google/gemma-4-26b-a4b-it-maas",
    contents=[Part.from_bytes(data=image_bytes, mime_type="image/jpeg"),
              "Describe this image."],
)
```

Key facts: model ID format `google/<model-name>-maas`, region must be `global`
(not `us-central1`), auth via `gcloud auth application-default login`.

## Usage across the series

### Week 5 — Gemma 4 (2026-04-24)

Used as the vision backend for `gemma4-smart-gallary/`. Chosen over local
Ollama for the showcase context: no GPU required, reliable API, zero setup for
the audience. New GCP project created (`gemma4-smart-gallery`) isolated from
the AutoResearch project.

Vision calls: image bytes passed as `Part.from_bytes()` alongside a text
prompt in a single `generate_content()` call. The smart gallery's Union remote
backend required GCP credentials to be injected as Union secrets
(`GCP_PROJECT`, `GCP_REGION`, `GEMMA_MODEL`) since cluster containers can't
access local `.env` files.

Union upload hang (39KB bundle, 2+ hours) was a known blocker — root cause
unresolved at time of demo; local backend fully working.
