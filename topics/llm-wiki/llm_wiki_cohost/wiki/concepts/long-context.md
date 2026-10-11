---
title: Long Context
first_seen: topics/gemma4/
weeks: [gemma4]
---

Using a model's full context window to process an entire document at once,
rather than chunking it and retrieving relevant pieces. The tradeoff: simpler
pipeline and no retrieval errors, at the cost of VRAM proportional to document
length.

Long context and RAG answer the same question ("what does this document say
about X?") with opposite architectural choices:

| | Long Context | RAG |
|---|---|---|
| Pipeline | Embed everything, ask once | Chunk → embed → retrieve → generate |
| VRAM | Scales with doc size | Fixed per retrieved chunk |
| Precision | Model sees everything | Model sees only retrieved chunks |
| Miss rate | Zero (nothing omitted) | Non-zero (retrieval can miss) |
| Sweet spot | Single docs, bounded size | Large corpora, many documents |

## How it appeared across the series

### Week 5 — Gemma 4 (2026-04-24)

Demonstrated in `docs/`: drop in a PDF/txt/md file and ask questions — no
chunking, no retrieval, no vector DB. Gemma 4's 262k context window handles
most real-world documents in a single call.

Critical implementation detail: Ollama defaults `num_ctx=4096`, which silently
truncates long documents. The `docs/` app sizes `num_ctx` dynamically to fit
the document (capped at 262k). VRAM scales accordingly — a 100k-token prompt
on a 31B model needs the DGX Spark's full ~64GB unified memory.

**Setup for contrast with weeks 6-7:** The `docs/` demo explicitly avoids
RAG — the design choice is intentional. The same Everstorm Outfitters PDFs
used in weeks 6 (vector RAG) and 7 (graph RAG) would fit in Gemma 4's context
window, making the docs demo a natural baseline for the RAG comparison.

## Open questions

- At what corpus size does long context break down and RAG become necessary?
- How does answer quality compare between long-context and RAG on the same
  document set? (The series provides a natural A/B across weeks 5-7.)
- Does the 262k context window degrade in quality at the edges (lost-in-the-
  middle problem)?
