---
title: Agent Memory
first_seen: topics/vectorstore/
weeks: [vectorstore]
---

Vector-store-backed memory for agents. Distinct from RAG's read-only retrieval:
the store is **read+write**. Each turn retrieves relevant past memories, uses
them as context for the answer, then extracts new atomic facts and writes them
back. The store grows over time, specific to one user, and survives across
sessions.

```
per turn:
  user message
     │
     ├─►  embed → vector store top-k → relevant past memories
     │                                        │
     │                                        ▼
     │                              inject into prompt
     │                                        │
     ├─────────────────────────────►  LLM streams answer
     │                                        │
     │                                        ▼
     └─►  extract atomic facts ──────►  embed + write back to store
```

The key difference from classic RAG: the corpus is not a static document
collection — it is the history of interactions with this specific user.

## How it appeared across the series

### Week 6 — Vector Stores (2026-05-01)

**`agent-memory-chroma/`** — First explicit agent memory implementation in the
series. Chroma as the memory store; Gemma 4 as both the answer model and the
fact-extraction model (second call per turn, thinking off, temp 0).

**Fact extraction prompt:** after every assistant reply, a small Gemma call runs:
> "Extract durable facts about the user from this exchange. Output ONLY a JSON
> array of strings. Each string is one atomic fact, preference, or decision the
> user explicitly stated or strongly implied. Skip questions, speculation, and
> trivial pleasantries. If nothing is worth remembering, output []."

A regex grabs the first `[...]` block and `json.loads` it. If parsing fails, no
memories are written; the chat still works — a graceful degradation pattern.

**Persistence across pod restarts** via HuggingFace Hub: `@on_startup` pulls
`memory.tar.gz` from a HF model repo; `@on_shutdown` (Knative SIGTERM, 30s
grace) tars and uploads it back as a new commit. Each save is a new commit —
HF provides free memory versioning. Manual "💾 Save to HF" button provides an
explicit checkpoint during live streams and guards against crash-skipped
shutdown hooks.

**Demo pattern:** turn 1 introduces the user and preferences → "Written this turn"
panel shows extracted facts → turn 2 asks "what do you know about me?" →
"Retrieved this turn" panel shows facts from turn 1 → Gemma's reply draws on
them. Optional drama: kill and redeploy; cold-start restores from HF snapshot;
same facts available.

**`screen-context-harness/`** — A second agent-memory pattern in the same week,
but the "user" is the screen itself. ChromaDB stores consolidated activity
outlines (not chat facts); queries retrieve semantically relevant prior context
("what was I working on this morning?"). Three-tier compaction (minute →
hourly → daily) mirrors how human memory consolidates over time.

## Open questions

- How does the series scale agent memory beyond a single user? (multi-user
  `entity_id` filtering is a noted next idea but not implemented in v1)
- Memory-decay pruning — removing memories not retrieved in N days — is flagged
  as a follow-up but not built. Does stale memory degrade quality?
- Cognee (week 9, 2026-05-22) is the anticipated productized take on this pattern.
  How does its memory graph differ from a flat vector store of atomic facts?
