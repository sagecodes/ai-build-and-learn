---
title: Structured Output
first_seen: topics/gemma4/
weeks: [gemma4]
---

Constraining a model's output to match a schema — JSON, a typed object, or a
specific format — so that downstream code can parse it reliably without regex
fallbacks or retry loops.

Two mechanisms appear in the series:
- **JSON-schema constraint mode** (Ollama) — the inference engine forces the
  model to produce output that matches a JSON Schema. Invalid JSON is
  structurally impossible.
- **Tool use / structured output** (Anthropic SDK) — the model is asked to call
  a tool whose input schema defines the expected shape. Claude's tool use
  eliminates JSON parse failures by construction.

## How it appeared across the series

### Week 5 — Gemma 4 (2026-04-24)

Demonstrated in `extract/`: messy text in, typed JSON out. Ollama's JSON-schema
constraint mode forces Gemma 4 to produce output matching a user-specified
schema — preset schemas in the UI (receipt, contact, calendar event, resume)
plus a free-form schema editor. No retry loops needed; the model cannot
produce invalid JSON in this mode.

This is the local-model equivalent of the Anthropic SDK tool use pattern used
in the GraphRAG project for entity extraction. The underlying principle is
identical: define the output schema at the API level, not in a parsing
post-processor.

## Open questions

- How does JSON-schema constraint mode interact with model quality? Does
  constraining output reduce the model's reasoning ability?
- When does structured output via tool use (Anthropic SDK) outperform
  JSON-schema constraint mode (Ollama)?
