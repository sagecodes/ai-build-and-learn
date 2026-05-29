---
title: Gemma 4
date: 2026-04-24
folder: topics/gemma4/
concepts: [multimodal-llms, long-context, structured-output, agents, react-loop]
tools: [ollama, gradio, vertex-ai, flyte]
---

Google's open-weight multimodal model family (Apache 2.0), explored through
eight demos spanning vision, audio, long-context docs, structured extraction,
tool-use agents, voice, live camera, and a full semantic gallery app with
Vertex AI MaaS + Flyte orchestration.

## The Gemma 4 family

| Model | What it is | Key property |
|---|---|---|
| `gemma4:e2b` | ~2.3B effective params, 128K ctx | PLE architecture + native audio encoder |
| `gemma4:e4b` | ~4.5B effective params, 128K ctx | PLE + native audio — sweet spot for local multimodal |
| `gemma4:26b` | 26B total / ~4B active (MoE), 256K ctx | 128 experts, 8 active per token — frontier quality at ~4B speed |
| `gemma4:31b` | 31B dense, 256K ctx | Most predictable, best reasoning baseline |

PLE = Per-Layer Embedding (big embedding tables off the hot path). MoE =
Mixture-of-Experts. Only E2B and E4B have the native audio encoder.

## What was built

Seven local demos via Ollama (all Gradio UIs):
- **`chatbot/`** — streaming chat, system prompt, temperature, model picker
- **`vision/`** — image Q&A + emergent bounding-box detection
- **`extract/`** — messy text → typed JSON via JSON-schema constraint mode
- **`docs/`** — PDF/txt → Q&A over full 262k context, no RAG, no chunking
- **`agent/`** — ReAct tool-use loop (calculator, web search, file read)
- **`voice/`** — Whisper or Gemma native audio STT → Gemma reasoning → Edge TTS
- **`live-camera/`** — webcam → streaming scene captions (narrative or independent mode)

One cloud demo via Vertex AI MaaS:
- **`gemma4-smart-gallary/`** — semantic photo search: scan folder → Gemma 4 describes each image → SQLite cache; search re-queries Gemma live per image. Flyte orchestrates parallel describe and search tasks. Both local and Union remote backends.

## Key decisions

**Ollama for local, Vertex AI MaaS for cloud.** All seven demo apps use Ollama
— zero API cost, no network dependency, swappable model via `GEMMA_MODEL` env
var. The smart gallery used Vertex AI MaaS (no GPU required, API-only) because
it was built for a showcase where reliability mattered more than local control.

**Long context as a RAG alternative.** The `docs/` demo deliberately avoids
chunking and retrieval — the entire document goes into the 262k context window.
Sets up the contrast with weeks 6-7 (vector RAG, graph RAG) where chunking is
necessary for larger corpora.

**JSON-schema mode for structured extraction.** Ollama's constraint mode forces
the model to match a schema — no regex fallbacks, no retry loops. The model
can't produce invalid JSON.

**Gemma native audio as STT.** E2B/E4B ship with an audio encoder — the same
model that generates the reply can also transcribe the input. The voice demo
A/Bs Whisper vs Gemma native audio at runtime.

**Smart gallery search: live vision over cached descriptions.** Search re-queries
Gemma live per image rather than searching the SQLite cache. Keeps the two flows
independent and keeps Gemma's vision capability front and center.

## Connections

- [Multimodal LLMs](../concepts/multimodal-llms.md) — vision, audio, text in one model
- [Long Context](../concepts/long-context.md) — 262k window as RAG alternative
- [Structured Output](../concepts/structured-output.md) — JSON-schema constraint mode
- [Agents](../concepts/agents.md) — ReAct tool-use loop with Gemma 4
- [ReAct Loop](../concepts/react-loop.md) — Gemma 4 agent implementation
- [Ollama](../tools/ollama.md) — local serving for all 7 demos
- [Gradio](../tools/gradio.md) — UI for all 8 sub-projects
- [Vertex AI](../tools/vertex-ai.md) — MaaS hosting for the smart gallery
- [Flyte / Union](../tools/flyte.md) — parallel vision tasks in the smart gallery
