Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This week's topic: **building with vector stores** for RAG and agent-context applications.

This README is a primer. The three project folders next to it are runnable Flyte 2 demos that put each idea into practice.

- [`rag-chroma-flyte/`](rag-chroma-flyte/): classic RAG over a Wikipedia passage corpus
- [`agent-memory-chroma/`](agent-memory-chroma/): the same store as long-running, read+write agent memory, with HF Hub persistence
- [`rag-umap-visualizer/`](rag-umap-visualizer/): RAG with a 2D UMAP projection of the embedding space, so you can *see* what retrieval is doing

---

## What is a vector store?

A vector store is a database whose primary index isn't on text or numbers, but on **vectors**, which are fixed-length lists of floats that summarize the meaning of a piece of content. The query operation isn't "find the row whose name equals X," it's "find the rows whose vector is *closest* to this one."

The pieces:

```
   text / image / audio  ──►  encoder  ──►  vector (e.g. 384 floats)
                                               │
                                               ▼
                                          vector store
                                          (indexed for nearest-neighbor lookup)

   query  ──►  same encoder  ──►  query vector  ──►  top-k nearest neighbors
```

Two key ideas make this work:

1. **Embedding models** map semantically similar inputs to vectors that are close together in the embedding space. "*Where is the Eiffel Tower?*" and "*Paris landmarks*" land near each other; "*Eiffel Tower*" and "*Tower of London*" land further apart even though they share words.
2. **Approximate nearest-neighbor (ANN) search**: exact nearest-neighbor in 768-dim is slow at scale, so vector stores use index structures like HNSW (graphs of close vectors), IVF (inverted files), or PQ (product quantization) that find approximate top-k in milliseconds across millions of vectors.

A vector store is what makes "search by meaning" tractable.

---

## How does the RAG pipeline work?

**Retrieval-Augmented Generation** (RAG) is the classic use case for vector stores. The LLM doesn't memorize your docs; it looks them up at query time and stuffs the relevant ones into its prompt.

The pipeline has two phases:

### Indexing (offline, run once per corpus)

```
docs ──►  load  ──►  chunk  ──►  embed  ──►  store
```

1. **Load.** Pull your source: PDFs, docs, scraped pages, a HuggingFace dataset, your own notes. The demo in `rag-chroma-flyte/` uses `rag-datasets/rag-mini-wikipedia` for repeatable runs.
2. **Chunk.** Split each document into smaller pieces. Why? Embedding models have a fixed context (~512 tokens for `bge-small`), and even when they don't, retrieval works better on focused passages than on entire documents. You don't want to inject a whole 30-page PDF into the prompt to answer one question.

   Chunking choices that matter:
    - **Size**: bigger chunks are more self-contained, smaller ones are more focused. Common range: 200–1500 chars or ~100–500 tokens.
    - **Overlap**: repeating ~10% of one chunk in the next prevents losing a sentence that straddles a boundary.
    - **Strategy**: char-based recursive splitters (paragraphs → lines → sentences → words → chars) are the simple default. Token-based splitters are more accurate but pull in a tokenizer dependency. Some pipelines split on document structure (markdown headings, code blocks) instead.

3. **Embed.** Run each chunk through an embedding model. The output is a fixed-length vector (e.g. 384 floats for `bge-small-en-v1.5`, 1024 for `bge-large`, 1536 for OpenAI `text-embedding-3-small`). The model that built the index *must* match the model used at query time. Different models map to different spaces and the geometry is meaningless across them.

4. **Store.** Write `(id, vector, original_text, metadata)` rows into the vector store. The store builds an ANN index in the background.

### Retrieval + generation (online, per query)

```
query  ──►  embed  ──►  top-k nearest  ──►  inject as context  ──►  LLM answers
```

1. **Embed the query** with the same encoder.
2. **Search** the store for the top-k closest chunks (typically k = 3–8).
3. **Build a prompt** that includes those chunks as context, plus the user's question. Standard shape:
   ```
   [SYSTEM] Use the following context to answer. If the answer isn't in
   the context, say you don't know.
   CONTEXT:
   [#1] <chunk text>
   [#2] <chunk text>
   ...
   [USER] <the actual question>
   ```
4. **Generate.** The LLM answers grounded in those chunks. Asking it to cite as `[#N]` makes the chain auditable.

That's it. Everything more sophisticated is a refinement of this loop.

---

## Why RAG matters

LLMs without retrieval are stuck with whatever was in their training data. RAG gives you four things they otherwise can't have:

- **Fresh knowledge.** A model trained in 2024 can answer questions about your meeting notes from this morning if those notes are in the index.
- **Private knowledge.** Your internal docs never have to leave your infrastructure to be useful.
- **Cheap updates.** Adding a new document means re-running the embed step, not re-training a model.
- **Grounded answers.** When the model cites `[#3]`, you can click through and verify. Hallucinations get easier to catch.

RAG vs. fine-tuning vs. long-context-window is a frequent comparison:

| Approach | Best for | Cost |
|---|---|---|
| **RAG** | Fact recall over a known corpus | Cheap; cheap to update; bounded by retrieval quality |
| **Fine-tuning** | Style, format, behavior changes | Expensive per change; doesn't add facts well |
| **Long context** | One-off "stuff the whole doc in" tasks | Token cost scales linearly; struggles past ~50k tokens |

In practice you stack them: fine-tune for tone, use RAG for facts, save long context for the hot-path conversation.

---

## Variants of RAG

The "naive RAG" pipeline above is the starting point. The field has moved fast. These are the patterns worth knowing.

### Naive / classic RAG
What `rag-chroma-flyte/` builds. Single embed → top-k → stuff into prompt. Works surprisingly well as a baseline.

### Hybrid retrieval
Combine **sparse** retrieval (BM25, keyword-based) with **dense** retrieval (vector). Sparse catches exact terms ("part number 5XQ-1180"), dense catches meaning ("the cheapest flight"). Often you stack a **cross-encoder reranker** on top: take the top-50 from each retriever, rescore with a more expensive model, keep the best 5. Big quality boost for almost every domain.

### Query rewriting / multi-query
Have the LLM reformulate the user's question into 2–N variants before retrieving. "*What did the CEO say about layoffs?*" → ["*CEO statement on layoffs*", "*executive comments workforce reductions*", …]. Retrieve for each, dedupe, send the union to the answer step.

### HyDE: Hypothetical Document Embeddings
The query and the answer often have different semantics ("*how do I reset my password?*" vs. "*click Account → Settings → Reset…*"). HyDE: ask the LLM to write a *hypothetical answer* first, embed *that*, retrieve docs similar to it. Works well for question-answer mismatches.

### Hierarchical / Tree RAG (RAPTOR)
Recursively cluster chunks. Have the LLM summarize each cluster. Embed both the raw chunks and the summaries. Retrieve from the whole tree. A question can match a high-level summary ("the chapter on photosynthesis") *or* a specific leaf ("the Calvin cycle paragraph"). Good for long documents where the right answer requires a section overview *and* a detail.

### Graph RAG
Extract entities and relationships at index time, build a knowledge graph (Neo4j, etc.) alongside the vector store. At query time, retrieve a *subgraph* around entities mentioned in the question. Strong on multi-hop questions ("Which authors have collaborated with both X and Y?") that pure vector retrieval struggles with. Microsoft's open-source GraphRAG is the well-known reference.

### Agentic RAG
Wrap retrieval as a **tool** the LLM can call. The model decides whether to search, what to search for, when it has enough, and when to ask another sub-question. Multi-hop, multi-source questions become tractable: "*Compare last quarter's product launch to the previous one*" can spawn one search per quarter, then a synthesis call.

### Self-RAG / Adaptive RAG
The model is trained (or prompted) to decide *per generation step* whether retrieval is needed. Some answers don't need any external grounding; some need it three times mid-response. The model emits special tokens ("retrieve here," "this passage is relevant") that drive the loop.

### Corrective RAG (CRAG)
Score the retrieved passages. If the score is poor, the system *corrects*: rewrite the query, fall back to web search, decompose the question. Stops the system from confidently answering with bad context.

### Multimodal RAG
Documents are images/audio/video. The encoder is CLIP (text↔image), Voxtral (audio), or similar. The store and the retrieval shape are the same; the encoder is the only thing that changes.

---

## Vector stores as agent memory

A vector store doesn't have to hold *documents*; it can hold *anything* you want to look up by meaning. One of the most useful applications: **long-running memory for an agent**.

The classic problem: chat models forget everything between sessions. Workarounds like "stuff the whole transcript into the next prompt" don't scale. Context windows are bounded and irrelevant history dilutes the signal.

Vector-store-backed memory:

```
   per turn:
     user message
        │
        ├─►  embed → vector store top-k → relevant past memories
        │                                        │
        │                                        ▼
        │                              inject into prompt
        │                                        │
        │                                        ▼
        ├─────────────────────────────►  LLM streams answer
        │                                        │
        │                                        ▼
        └─►  extract atomic facts ──────►  embed + write back to store
              (e.g. "user prefers terse answers",
               "user is building a Flyte demo")
```

This is a **read+write** loop instead of RAG's read-only one. The store grows over time, gets specific to one user (or many, with `entity_id` filters), and survives across sessions.

`agent-memory-chroma/` in this repo implements this end-to-end: each turn retrieves memories, the agent answers, a second small LLM call extracts new facts, and the whole Chroma persist dir gets snapshotted to a HuggingFace model repo so it survives pod restarts.

The same patterns from RAG variants apply here too: hybrid retrieval, summarization (memory consolidation), graph layouts (entity/relationship memory). Frameworks like Mem0 and Letta are productized takes on this loop.

---

## Vector store comparison

The popular options. They mostly do the same thing. Pick on operational fit, not features.

| Store | Shape | Why pick it |
|---|---|---|
| **Chroma** | Embeddable Python, also a server | Easiest start. Used by all three demos here. |
| **Qdrant** | Rust server, gRPC + REST | Fast, mature, great filtering DSL. Free tier in cloud. |
| **Weaviate** | Go server with built-in modules | Hybrid retrieval baked in, schema-first. |
| **Pinecone** | Hosted only | Managed, scales without thinking about it. Costs money. |
| **pgvector** | Postgres extension | Already running Postgres? Add a column, done. Slower than purpose-built at scale. |
| **Milvus / Vespa** | Big-iron servers | Billions of vectors, heavy ops. |
| **FAISS** | Library, no server | DIY. Good for research; build the persistence/API yourself. |

For week one we used Chroma because the in-process `PersistentClient` is one line. Swapping to any of the others touches only the indexing task and the query call. The chunking, embedding, and prompt logic stay put.

---

## What's in this repo

Each demo runs on a Flyte 2 devbox (a single-node k3s with Knative-served apps). They share the same Gemma 4 vLLM endpoint and the same `BAAI/bge-small-en-v1.5` encoder so they slot together.

- **`rag-chroma-flyte/`**: Flyte 2 pipeline (HF dataset → chunk → embed → Chroma persist dir as a `flyte.io.Dir`) plus a Gradio chat that mounts the Chroma artifact via `RunOutput`. The starting point.
- **`agent-memory-chroma/`**: single-app Gradio chat. `@on_startup` pulls a tarballed Chroma snapshot from a HuggingFace model repo; `@on_shutdown` (Knative SIGTERM) writes it back. Each turn retrieves memories *and* extracts new ones to write.
- **`rag-umap-visualizer/`**: same RAG pipeline output as week 1, plus a 2D UMAP projection of the index. Each query lights up its top-k as colored markers and the query itself as a gold star, in the same fitted UMAP space. Makes "vectors cluster by topic" tangible on screen.

---

## Resources

- GitHub: https://github.com/sagecodes/ai-build-and-learn
- Events Calendar: https://luma.com/ai-builders-and-learners
- Slack (discuss during the week): https://slack.flyte.org/
- Hosted by Sage Elliott: https://www.linkedin.com/in/sageelliott/
