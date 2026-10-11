---
title: Ollama
weeks: [autoresearch, gemma4, vectorstore]
---

Local model serving platform. Runs open LLMs (Gemma 4, Qwen Coder, Llama, etc.)
via a local HTTP API on port 11434. Models load once into GPU VRAM and serve
requests without requiring a cloud API key or network dependency.

Key behavior to know: by default Ollama evicts the model from VRAM after 5
minutes of inactivity. In workloads that alternate between agent calls and
long training runs, this causes a ~4.5-minute reload penalty every iteration.
Fix: `curl -s http://localhost:11434/api/generate -d '{"model":"<name>","keep_alive":-1}'`
to pin the model in VRAM indefinitely.

## Usage across the series

### Week 6 — Vector Stores (2026-05-01)

Used in `screen-context-harness/` as the vision LLM for captioning screenshots.
Gemma 4 Vision (`gemma4:26b` default, swappable via `DEFAULT_MODEL`) receives
a downsampled screenshot (max 768px) every 5 seconds and returns a one-sentence
caption of the current app, file, or task.

This is the first use of Ollama as a **multimodal perception loop** rather than
a text-generation tool: the model runs in a background async task, not in
response to a user message. Images are passed as base64 bytes — same pattern
established in Gemma 4 week.

The harness also uses Ollama for consolidation: every 60 seconds, a second
Ollama call synthesizes the 12-caption rolling buffer into a "Context Outline."
Two distinct LLM roles in one app, both served by the same local Ollama endpoint.

### Week 4 — AutoResearch (2026-04-17)

Used to run Gemma 4 31B (~17GB) as the research agent in `local-llm-autoresearch/`.
The DGX Spark (128GB unified VRAM) hosts both Gemma 4 (~17GB) and the training
process (~23GB) simultaneously — no swapping between agent calls and training runs.

The local agent uses Ollama's HTTP API directly (`local_agent.py`), not a
higher-level SDK. One-shot completions per iteration — no streaming, no retry.
If the model output is malformed (no valid SEARCH/REPLACE blocks), the iteration
is logged as `discard` and the loop moves on.

Unified diffs were tried first (qwen3-coder-next: 0/3 applied). Switching to
aider-style SEARCH/REPLACE blocks got 3/3 to apply. This is why Ollama-backed
agents in the series use SEARCH/REPLACE, not diffs.

### Week 5 — Gemma 4 (2026-04-24)

Heaviest Ollama usage in the series — all seven local Gemma 4 demos run via
Ollama. Model selectable at launch via `GEMMA_MODEL` env var; all demos default
to `gemma4:31b` except `live-camera/` which defaults to `gemma4:e4b` for speed.

New multimodal usage patterns this week:
- **Vision**: images passed as base64 bytes in the `images` field
- **Audio**: E2B/E4B native audio encoder — audio WAV also passed via `images`
  field (Ollama routes all multimodal bytes the same way regardless of modality)
- **JSON-schema constraint mode**: `extract/` demo forces structured JSON output
  via Ollama's format parameter — model cannot produce invalid JSON

`num_ctx` must be set explicitly for long-document use cases — Ollama's default
of 4096 silently truncates. The `docs/` demo sizes it dynamically per document.
