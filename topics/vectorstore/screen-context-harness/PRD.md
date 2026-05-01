# PRD: Screen Sensory Harness

## Problem Space & Research Questions

Does an LLM following your screen activity produce useful near-realtime contextual understanding of your focus? Can it respond quickly enough (at 5s cadence) to feel live, does a 60s consolidated outline stay coherent over time, and can a vector store of past outlines provide meaningful long-term memory that enriches future consolidations?

## Hypotheses

1. A multimodal LLM can describe a screenshot in a useful sentence within the 5s capture window (fast enough to feel current, not stale).
2. A 60s consolidation cycle over rolling descriptions can maintain a coherent, compact outline of recent activity without manual curation.
3. The two-layer output (immediate vs. outline) is meaningfully different — the immediate caption captures context the outline doesn't, and vice versa.
4. Semantic retrieval of prior outlines at consolidation time produces citations that meaningfully extend context beyond the current 60s window.
5. The Context Search interface is useful as a standalone memory query — you can ask "what was I working on earlier?" and get a coherent answer.

## Validation / Success Metrics

- **Latency**: immediate caption arrives within ~5s of screen capture (streaming start visible in <2s)
- **Accuracy**: captions are specific enough to distinguish between different windows/tasks (not just "you are using a computer")
- **Outline coherence**: after 5+ minutes, the outline reads as a useful summary of the session, not noise
- **Compaction**: outline stays within max size across multiple 60s cycles without degrading
- **RAG relevance**: retrieved prior outlines are topically related to current activity (distance < 0.55 threshold produces useful hits, not noise)
- **Search utility**: a natural-language query against the store returns recognizable past activity within the top 3 results

## Prototype Tech Spec

### Core capture & caption loop
- [x] Capture primary monitor screenshot every 5s via `mss`
- [x] Downsample screenshot before sending (cap long side at 768px)
- [x] Display latest screenshot in Gradio UI immediately (independent of LLM call)
- [x] Stream immediate caption into "Right Now" panel via remote Ollama (ngrok) or localhost:11434
- [x] Accumulate last 12 captions in rolling buffer (covers ~60s of activity)
- [x] Backpressure mutex (`_caption_lock`) drops overlapping capture ticks
- [x] Module-level `_running` flag gates all work (avoids Gradio state propagation lag)

### Consolidation & RAG loop
- [x] Every 60s, compact outline if > 600 chars, then consolidate buffer into updated outline
- [x] Before consolidating: query ChromaDB for semantically relevant prior outlines (top 3, cosine distance < 0.55)
- [x] Inject retrieved prior outlines into consolidation prompt as citations
- [x] After consolidating: embed and persist new outline to ChromaDB (cosine space, persistent to `chroma_db/`)
- [x] Process log shows RAG hits with timestamps and distances
- [x] Store count updates live in Context Search section after each cycle

### Context Search
- [x] Search textbox + button queries vector store with free-text input
- [x] Results returned with timestamp, similarity score, and outline text
- [x] Returns up to 5 results ranked by semantic similarity
- [x] Live store count displayed and updated after each consolidation

### Infrastructure
- [x] ngrok URL + model selectable in UI (leave blank → localhost:11434)
- [x] Single Start/Stop control
- [x] Timestamped process log (last 50 lines) with all steps explicit for debugging
- [x] Caption buffer debug accordion

## Architecture

```
capture_timer (5s)
  └─ mss screenshot → display immediately
     └─ LLM caption (streaming) → append to rolling buffer [max 12]

consolidate_timer (60s)
  └─ compact outline if needed
     └─ ChromaDB query (captions as query text) → retrieve prior context
        └─ LLM consolidation (streaming, with prior context injected)
           └─ ChromaDB store (new outline embedded + persisted)
```

**Vector store:** ChromaDB persistent client at `./chroma_db/`. Collection: `activity_outlines`. Embedding: `all-MiniLM-L6-v2` via ChromaDB default (onnxruntime, local, ~30MB one-time download). Distance metric: cosine.

**Outline document schema:**
- `id`: `outline_{timestamp_ms}`
- `document`: outline text
- `metadata`: `{timestamp: "HH:MM:SS", ts_epoch: int}`

## Scope & Intentional Omissions

- **No multi-monitor** — primary only
- **No privacy filtering** — personal local prototype only
- **No audio** — screen only
- **No export or history replay** — search interface covers retrieval needs
- **No embedding model choice** — ChromaDB default is sufficient; swapping to Ollama embeddings is a future option
- **No session isolation** — the store accumulates across all sessions by design (long-term memory is the goal)

## Implementation Path

Gradio + `mss` + `ollama.Client` + `chromadb.PersistentClient`. Two `gr.Timer` instances always active; `_running` module-level flag gates work. ChromaDB initialized at module load, persists to `pear-harness/chroma_db/`.

Run: `uv run python app.py` → http://localhost:7868

## Evaluation Plan

1. Run a 10+ minute session across 3–4 distinct activities (coding, browsing, terminal, docs)
2. After each transition: does "Right Now" caption reflect the current window within one tick?
3. After ~5 minutes: does the outline describe the session arc accurately?
4. After 2+ consolidation cycles: do RAG hits appear in the process log? Are they topically relevant?
5. Use Context Search to ask "what was I working on earlier?" — do results match actual activity?
6. Across multiple app restarts: does the store persist and surface prior session context?

## Learnings / Next Steps

### Implementation notes (debugging log)
- `gemma4:26b` defaults to thinking mode — `think=False` required or `chunk.message.content` is always empty (thinking tokens go to `chunk.message.thinking`)
- `ollama.Client` streaming chunks are Pydantic objects, not dicts — use `chunk.message.content`, not `chunk.get("message", {}).get("content")`
- `mss.mss()` deprecated in mss 9+, use `mss.MSS()`
- Returning multiple `gr.Timer` objects from a click handler is unreliable in Gradio 6 — use a module-level `_running` flag instead of `gr.State` for control signals
- Screen capture must be independent of LLM call — yield screenshot to UI before making any network call, or a hanging LLM holds the lock and blocks captures indefinitely

### Open validation questions
- [ ] Does the remote `gemma4:26b` produce specific enough captions at 5s cadence, or is it too slow/generic?
- [ ] Is RAG distance threshold 0.55 too tight or too loose in practice?
- [ ] Does prior context injection improve outline quality, or add noise?
- [ ] Is the Context Search interface actually useful for recall, or is the outline sufficient?
