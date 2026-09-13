---
title: Vector Stores
date: 2026-05-01
folder: topics/vectorstore/
concepts: [rag, embeddings, agent-memory, multimodal-llms]
tools: [chroma, pgvector, flyte, gradio, ollama]
---

The series' first dedicated RAG week. Five projects explored vector stores from
three angles: classic RAG pipelines, long-running agent memory, and a multimodal
screen-context harness. The cohort reference used FAISS + Ollama locally; the
series extended this to production-grade cloud deployments.

## What was built

**`rag-chroma-flyte/`** — Classic RAG over a Wikipedia passage corpus. A three-task
Flyte pipeline (fetch → chunk → embed → Chroma persist dir as `flyte.io.Dir`) feeds
a Gradio chat that mounts the artifact via `RunOutput`. Uses `BAAI/bge-small-en-v1.5`
for embedding and a vLLM-hosted Gemma 4 for generation.

**`agent-memory-chroma/`** — Same Chroma + Gemma 4 stack, but the store is read+write:
the agent's long-running memory of the user, not a static doc corpus. Each turn
retrieves relevant memories, generates an answer, then extracts new atomic facts and
writes them back. HuggingFace Hub stores the Chroma snapshot (tar.gz); `@on_startup`
/ `@on_shutdown` hooks restore and persist it across pod restarts.

**`rag-umap-visualizer/`** — Adds a 2D UMAP projection of the embedding space to the
RAG chat. Each query plots as a gold star; top-k retrieved chunks light up in
rank-colored markers. Makes "vectors cluster by topic" tangible on stream.

**`vector_rag_chatbot/`** — Production RAG chatbot over 16 Everstorm Outfitters PDFs.
pgvector on Supabase (managed Postgres) replaces Chroma; Union/Flyte orchestrates
the ingest and query pipelines; Claude Sonnet generates grounded answers. Deployed
as a persistent Gradio app on Union cluster.

**`screen-context-harness/`** — Multimodal harness that captures screenshots every 5s,
captions them with Gemma 4 Vision via Ollama, consolidates 60s windows into context
outlines stored in ChromaDB, and answers natural-language queries about past activity.
Three-tier compaction hierarchy (minute → hourly → daily) maintains long-term coherence.

## Key decisions

- **Chroma for local/devbox demos; pgvector for cloud production.** Chroma's
  `PersistentClient` is one line locally but adds a service to manage in production.
  pgvector lives inside an existing Postgres instance — no extra service, standard SQL.
- **Same embedding model at ingest and query time is a hard constraint.** Different
  models map to incompatible vector spaces. The Chroma collection metadata records
  which model built it; the chat app warns loudly on a mismatch.
- **`gte-small` over `bge-small` for pgvector demo.** Marginally higher MTEB retrieval
  scores at the same 384D / 67MB size. Either works; the constraint is consistency.
- **chunk_size=600, overlap=60 for structured support PDFs.** Smaller chunks (300)
  caused retrieval failures when section headers were split from their content.
  Configurable from the Gradio UI for live experimentation.
- **Base64-encode PDFs for Union deployment.** `flyte.run()` can't pass local file
  paths to cluster tasks; `gr.File` widget saves uploads to a temp path; bytes
  encoded as base64 strings thread through safely.

## Connections

- [RAG](../concepts/rag.md) — enters the wiki properly this week; three distinct
  implementations across the five projects
- [Embeddings](../concepts/embeddings.md) — new concept page seeded from this week's
  deep coverage of embedding models, cosine similarity, and HNSW
- [Agent Memory](../concepts/agent-memory.md) — read+write vector store loop
  introduced as a distinct pattern from read-only RAG
- [Chroma](../tools/chroma.md) — primary vector store for four of the five projects
- [pgvector](../tools/pgvector.md) — production vector store choice for `vector_rag_chatbot`
- [Flyte / Union](../tools/flyte.md) — orchestrates ingest and query pipelines; hosts
  the Gradio apps as persistent Union services
- [Gradio](../tools/gradio.md) — UI layer for all five projects
- [Ollama](../tools/ollama.md) — local LLM serving for `screen-context-harness`
