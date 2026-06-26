---
title: pgvector
weeks: [vectorstore, graphs-neo4j]
---

A Postgres extension that adds a `vector` column type and native approximate
nearest-neighbor operators. Brings vector similarity search inside standard
Postgres — no extra service, no proprietary API, same SQL tooling. The `<=>`
operator returns cosine distance; `1 - (embedding <=> query)` converts to
similarity (1.0 = identical).

In the series, pgvector runs on Supabase (managed Postgres, free tier 500MB).
The same SQL runs on any Postgres instance (GCP Cloud SQL, self-hosted, etc.)
if Supabase is outgrown.

**Enable:** `CREATE EXTENSION IF NOT EXISTS vector;` in the Supabase SQL Editor.

## Usage across the series

### Week 6 — Vector Stores (2026-05-01)

Used in `vector_rag_chatbot/` as the production vector store for 16 Everstorm
Outfitters PDFs (~800 chunks at 384D).

**Table schema:**
```sql
CREATE TABLE document_chunks (
    id              BIGSERIAL PRIMARY KEY,
    collection_name TEXT NOT NULL,
    source_doc      TEXT NOT NULL,
    chunk_index     INTEGER NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       vector(384) NOT NULL
);
CREATE INDEX idx_chunks_embedding
    ON document_chunks USING hnsw (embedding vector_cosine_ops);
```

**Retrieval query:**
```sql
SELECT source_doc, chunk_text, 1 - (embedding <=> %s::vector) AS score
FROM document_chunks
WHERE collection_name = %s
ORDER BY embedding <=> %s::vector
LIMIT %s
```

**Deployment gotchas:**
- Supabase's direct connection (`db.<project>.supabase.co:5432`) resolves to
  IPv6. The Union cluster (AWS us-east-2) has no IPv6 — use the Session Pooler
  URL instead (found at Supabase project → Connect → Session pooler).
- Even with the pooler URL, `psycopg3` can prefer IPv6. Fix: resolve to IPv4
  explicitly via `socket.getaddrinfo(hostname, None, socket.AF_INET)` and pass
  as `hostaddr` to `psycopg.connect()`.
- `flyte create secret` does not update existing keys — it silently does nothing.
  Use a new key name if a secret needs to change.

**pgvector vs Neo4j (week 7 contrast):** same 15 Everstorm PDFs, different
retrieval architecture. pgvector excels at cosine similarity lookup with
standard SQL filtering. Neo4j adds graph traversal in the same query —
`MATCH (chunk)-[:MENTIONS]->(entity)-[:RELATED]->(other)` — which pgvector
has no equivalent for. The choice is operational: pgvector if you're already
running Postgres; Neo4j if relationship traversal is a first-class query need.

**Multi-collection support:** `collection_name` is a user-configurable field in
the Gradio UI. Multiple collections coexist in the same table, filtered by
`WHERE collection_name = %s`. Re-ingesting with different chunk sizes means just
changing the collection name.
