---
title: Ollama
weeks: [autoresearch]
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
