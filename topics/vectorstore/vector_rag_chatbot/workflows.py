"""
workflows.py — Flyte tasks for the Everstorm RAG chatbot.

Six tasks in two pipelines, each visible as a separate node in Union UI:

  INGEST PIPELINE
  ───────────────
  ingest_pipeline(filenames, pdf_bytes_b64)
      ├── load_and_chunk_task(pdf_1)  ┐
      ├── load_and_chunk_task(pdf_2)  ├─ parallel fan-out via asyncio.gather
      ├── load_and_chunk_task(...)    │
      └── load_and_chunk_task(pdf_n) ┘
                    ↓  all chunks merged
          embed_and_index_task(all_chunks)  → Supabase pgvector

  QUERY PIPELINE
  ──────────────
  query_pipeline(query)
      ├── retrieve_task(query)        → top-k chunks from pgvector
      └── generate_task(query, chunks) → Claude RAG answer
"""

import asyncio
import base64
import json
import logging
import os
import socket
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import flyte
import flyte.report

from config import COLLECTION, EMBED_DIM, EMBED_MODEL, env

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


def _pg_connect():
    """Connect to Postgres forcing IPv4 to avoid IPv6-only cluster issues."""
    import psycopg

    url = os.environ["PG_URL"]
    p = urlparse(url)
    ipv4 = socket.getaddrinfo(p.hostname, None, socket.AF_INET)[0][4][0]
    return psycopg.connect(
        host=p.hostname,
        hostaddr=ipv4,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=p.password,
    )


# ─────────────────────────────────────────────────────────────────────────────
# INGEST PIPELINE — Task 1 of 2: load_and_chunk_task
# ─────────────────────────────────────────────────────────────────────────────

@env.task(report=True, cache="auto")
async def load_and_chunk_task(
    pdf_b64: str,
    filename: str,
    chunk_size: int = 600,
    chunk_overlap: int = 60,
) -> str:
    """
    Decode a base64 PDF → extract text with PyMuPDF → split into chunks.

    One task dispatched per PDF, all running in parallel inside ingest_pipeline.
    Cached by inputs — re-ingesting an unchanged PDF is a free cache hit.

    Returns JSON: list of {source_doc, chunk_index, chunk_text}
    """
    import pymupdf
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    suffix = Path(filename).suffix or ".pdf"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(base64.b64decode(pdf_b64))
        tmp_path = f.name

    try:
        doc = pymupdf.open(tmp_path)
        full_text = "".join(page.get_text() for page in doc)
        doc.close()
    finally:
        os.unlink(tmp_path)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    raw_chunks = splitter.split_text(full_text)

    chunks = [
        {"source_doc": filename, "chunk_index": i, "chunk_text": text}
        for i, text in enumerate(raw_chunks)
    ]

    log.info(f"[load_and_chunk] {filename} → {len(chunks)} chunks")

    await flyte.report.replace.aio(
        f"<h2>load_and_chunk_task</h2>"
        f"<p><b>File:</b> {filename}</p>"
        f"<p><b>Characters extracted:</b> {len(full_text):,}</p>"
        f"<p><b>Chunks created:</b> {len(chunks)} "
        f"(size={chunk_size}, overlap={chunk_overlap})</p>"
        f"<h3>Sample Chunks</h3>"
        + "".join(
            f"<p><b>Chunk {c['chunk_index']}:</b> {c['chunk_text'][:200]}...</p>"
            for c in chunks[:3]
        )
    )
    await flyte.report.flush.aio()

    return json.dumps(chunks)


# ─────────────────────────────────────────────────────────────────────────────
# INGEST PIPELINE — Task 2 of 2: embed_and_index_task
# ─────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def embed_and_index_task(
    chunks_json: str,
    collection_name: str = COLLECTION,
) -> str:
    """
    Batch-embed all chunks with gte-small and upsert to Supabase pgvector.

    Receives the merged chunks from all parallel load_and_chunk_task results.
    Drops existing rows for the collection then re-inserts (idempotent ingest).

    Returns JSON: {collection_name, total_chunks, vectors_upserted, embed_model}
    """
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    chunks = json.loads(chunks_json)
    texts = [c["chunk_text"] for c in chunks]

    await flyte.report.replace.aio(
        f"<h2>embed_and_index_task</h2>"
        f"<p>Loading <b>{EMBED_MODEL}</b> and embedding {len(chunks)} chunks...</p>"
    )
    await flyte.report.flush.aio()

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)

    conn = _pg_connect()
    register_vector(conn)

    with conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    source_doc      TEXT NOT NULL,
                    chunk_index     INTEGER NOT NULL,
                    chunk_text      TEXT NOT NULL,
                    embedding       vector(384) NOT NULL
                )
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_chunks_embedding
                ON document_chunks USING hnsw (embedding vector_cosine_ops)
            """)
            cur.execute(
                "DELETE FROM document_chunks WHERE collection_name = %s",
                (collection_name,),
            )
            cur.executemany(
                """
                INSERT INTO document_chunks
                    (collection_name, source_doc, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        collection_name,
                        chunks[i]["source_doc"],
                        chunks[i]["chunk_index"],
                        chunks[i]["chunk_text"],
                        embeddings[i].tolist(),
                    )
                    for i in range(len(chunks))
                ],
            )

    conn.close()

    stats = {
        "collection_name":  collection_name,
        "total_chunks":     len(chunks),
        "vectors_upserted": len(chunks),
        "embed_model":      EMBED_MODEL,
        "embed_dim":        EMBED_DIM,
    }

    log.info(f"[embed_and_index] {len(chunks)} vectors → '{collection_name}'")

    source_counts: dict[str, int] = {}
    for c in chunks:
        source_counts[c["source_doc"]] = source_counts.get(c["source_doc"], 0) + 1

    rows = "".join(
        f"<tr><td>{doc}</td><td>{count}</td></tr>"
        for doc, count in sorted(source_counts.items())
    )

    await flyte.report.replace.aio(
        f"<h2>embed_and_index_task — Complete</h2>"
        f"<p><b>Collection:</b> {collection_name}</p>"
        f"<p><b>Vectors upserted:</b> {len(chunks)}</p>"
        f"<p><b>Model:</b> {EMBED_MODEL} ({EMBED_DIM}D, Cosine distance)</p>"
        f"<h3>Chunks per Document</h3>"
        f"<table border='1' cellpadding='5'>"
        f"<tr><th>Document</th><th>Chunks</th></tr>"
        f"{rows}"
        f"</table>"
    )
    await flyte.report.flush.aio()

    return json.dumps(stats)


# ─────────────────────────────────────────────────────────────────────────────
# INGEST PIPELINE — Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def ingest_pipeline(
    filenames: list[str],
    pdf_bytes_b64: list[str],
    collection_name: str = COLLECTION,
    chunk_size: int = 300,
    chunk_overlap: int = 30,
) -> str:
    """
    Orchestrates the full ingest pipeline.

    Fans out load_and_chunk_task in parallel — one Union task per PDF —
    then feeds all merged chunks into a single embed_and_index_task.

    Union UI shows this as:
      ingest_pipeline
        ├── load_and_chunk_task (x N, parallel)
        └── embed_and_index_task

    Returns JSON stats from embed_and_index_task.
    """
    await flyte.report.replace.aio(
        f"<h2>ingest_pipeline</h2>"
        f"<p>Dispatching {len(filenames)} load_and_chunk tasks in parallel...</p>"
        f"<ul>"
        + "".join(f"<li>{f}</li>" for f in filenames)
        + "</ul>"
    )
    await flyte.report.flush.aio()

    # Parallel fan-out — one task per PDF
    chunk_results = await asyncio.gather(*[
        load_and_chunk_task(
            pdf_b64=b64,
            filename=fname,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        for fname, b64 in zip(filenames, pdf_bytes_b64)
    ])

    # Merge all chunks from all PDFs into one list
    all_chunks: list[dict] = []
    for result_json in chunk_results:
        all_chunks.extend(json.loads(result_json))

    log.info(f"[ingest_pipeline] {len(all_chunks)} total chunks from {len(filenames)} PDFs")

    # Single batch embed + index
    stats_json = await embed_and_index_task(
        chunks_json=json.dumps(all_chunks),
        collection_name=collection_name,
    )
    stats = json.loads(stats_json)

    await flyte.report.replace.aio(
        f"<h2>ingest_pipeline — Complete</h2>"
        f"<p><b>PDFs processed:</b> {len(filenames)}</p>"
        f"<p><b>Total chunks:</b> {len(all_chunks)}</p>"
        f"<p><b>Vectors in pgvector:</b> {stats['vectors_upserted']}</p>"
        f"<p><b>Collection:</b> {stats['collection_name']}</p>"
    )
    await flyte.report.flush.aio()

    return stats_json


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PIPELINE — Task 1 of 2: retrieve_task
# ─────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def retrieve_task(
    query: str,
    collection_name: str = COLLECTION,
    k: int = 5,
) -> str:
    """
    Embed the query with gte-small and retrieve top-k chunks from pgvector.

    Cached by (query, collection_name, k) — repeat questions return instantly.

    Returns JSON: list of {source_doc, chunk_text, score}
    """
    from pgvector.psycopg import register_vector
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    query_vector = model.encode(query).tolist()

    conn = _pg_connect()
    register_vector(conn)

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT source_doc, chunk_text,
                       1 - (embedding <=> %s::vector) AS score
                FROM document_chunks
                WHERE collection_name = %s
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (query_vector, collection_name, query_vector, k),
            )
            db_rows = cur.fetchall()

    conn.close()

    results = [
        {
            "source_doc": row[0],
            "chunk_text": row[1],
            "score":      round(float(row[2]), 4),
        }
        for row in db_rows
    ]

    log.info(f"[retrieve] '{query[:60]}' → {len(results)} chunks")

    table_rows = "".join(
        f"<tr><td>{r['score']}</td><td>{r['source_doc']}</td>"
        f"<td>{r['chunk_text'][:150]}...</td></tr>"
        for r in results
    )

    await flyte.report.replace.aio(
        f"<h2>retrieve_task</h2>"
        f"<p><b>Query:</b> {query}</p>"
        f"<p><b>Collection:</b> {collection_name} &nbsp;|&nbsp; <b>Top-k:</b> {k}</p>"
        f"<table border='1' cellpadding='5'>"
        f"<tr><th>Score</th><th>Source</th><th>Chunk Preview</th></tr>"
        f"{table_rows}"
        f"</table>"
    )
    await flyte.report.flush.aio()

    return json.dumps(results)


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PIPELINE — Task 2 of 2: generate_task
# ─────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def generate_task(
    query: str,
    context_json: str,
) -> str:
    """
    Build a RAG prompt from retrieved chunks and call Claude for the answer.

    Returns JSON: {answer, sources}
    """
    import anthropic

    context_chunks = json.loads(context_json)

    context_text = "\n\n".join(
        f"[Source: {c['source_doc']} | Score: {c['score']}]\n{c['chunk_text']}"
        for c in context_chunks
    )

    system_prompt = (
        "You are a customer support assistant for Everstorm Outfitters, "
        "an outdoor gear e-commerce company. "
        "Answer using ONLY the information in the provided context. "
        "If the answer is not in the context, say: "
        "'I don't have that information in our support docs.' "
        "Be concise and cite the source document for key facts."
    )

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=600,
        system=system_prompt,
        messages=[{
            "role": "user",
            "content": f"CONTEXT:\n{context_text}\n\nQUESTION:\n{query}",
        }],
    )
    answer = response.content[0].text
    sources = sorted({c["source_doc"] for c in context_chunks})

    log.info(f"[generate] answer: {answer[:80]}...")

    await flyte.report.replace.aio(
        f"<h2>generate_task</h2>"
        f"<p><b>Question:</b> {query}</p>"
        f"<h3>Answer</h3>"
        f"<p>{answer}</p>"
        f"<h3>Sources Used</h3>"
        f"<ul>" + "".join(f"<li>{s}</li>" for s in sources) + "</ul>"
    )
    await flyte.report.flush.aio()

    return json.dumps({"answer": answer, "sources": sources})


# ─────────────────────────────────────────────────────────────────────────────
# QUERY PIPELINE — Orchestrator
# ─────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def query_pipeline(
    query: str,
    collection_name: str = COLLECTION,
    k: int = 5,
) -> str:
    """
    Orchestrates retrieval then generation.

    Union UI shows this as:
      query_pipeline
        ├── retrieve_task   (cached on repeat questions)
        └── generate_task

    Returns JSON: {answer, sources}
    """
    await flyte.report.replace.aio(
        f"<h2>query_pipeline</h2>"
        f"<p><b>Question:</b> {query}</p>"
        f"<p>Running retrieve_task...</p>"
    )
    await flyte.report.flush.aio()

    context_json = await retrieve_task(
        query=query,
        collection_name=collection_name,
        k=k,
    )

    result_json = await generate_task(
        query=query,
        context_json=context_json,
    )
    result = json.loads(result_json)

    await flyte.report.replace.aio(
        f"<h2>query_pipeline — Complete</h2>"
        f"<p><b>Question:</b> {query}</p>"
        f"<h3>Answer</h3>"
        f"<p>{result['answer']}</p>"
        f"<h3>Sources</h3>"
        f"<ul>" + "".join(f"<li>{s}</li>" for s in result["sources"]) + "</ul>"
    )
    await flyte.report.flush.aio()

    return result_json
