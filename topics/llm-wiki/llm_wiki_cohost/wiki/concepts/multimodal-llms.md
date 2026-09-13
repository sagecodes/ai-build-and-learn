---
title: Multimodal LLMs
first_seen: topics/gemma4/
weeks: [gemma4]
---

Language models that accept multiple input modalities — text, images, audio,
video — in a single model and a single API call. No separate pipeline needed:
one model handles understanding, one call returns the result.

The key shift from prior series weeks: models had been text-in/text-out (MCP,
Tavily, AutoResearch) or used separate vision APIs. Gemma 4 introduces native
vision and audio in the same model that generates text.

## How it appeared across the series

### Week 5 — Gemma 4 (2026-04-24)

**Vision** — all four Gemma 4 variants are vision-capable. Demonstrated in:
- `vision/`: free-form Q&A about uploaded images; emergent bounding-box
  detection (returns normalized `{label, box_2d}` without a trained detector
  head — works well on salient objects, breaks down on crowded scenes)
- `live-camera/`: webcam frames sent to Gemma every few seconds for streaming
  scene captions. Narrative mode feeds prior captions back into the prompt to
  focus on what changed
- `gemma4-smart-gallary/`: Gemma 4 describes each photo in a folder, stores
  results in SQLite; search re-queries Gemma live per image

**Audio** — only E2B and E4B ship with a native audio encoder (the larger 26B
and 31B are text+image only). Demonstrated in `voice/`: the same model that
generates the reply can also transcribe the input — no separate Whisper step.
Implementation note: Ollama routes multimodal bytes through the `images` field
regardless of modality; audio is base64-encoded WAV sent the same way.

**Deployment split:** local (Ollama) vs cloud (Vertex AI MaaS). Local gives
full control and zero API cost; MaaS gives reliability and no GPU requirement.
The smart gallery used MaaS via the `google-genai` SDK with `Part.from_bytes()`
for image input.

## Open questions

- How do other open-weight multimodal models (Llama 4, Qwen-VL) compare to
  Gemma 4 in quality and local performance?
- Does the series revisit multimodal in the context of RAG (vision + retrieval)?
- When does native audio outperform a dedicated STT model like Whisper?
