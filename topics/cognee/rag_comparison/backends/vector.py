"""
vector.py — Vector RAG backend using Supabase pgvector.

Query flow:
  1. Embed question with fastembed (shared singleton)
  2. Cosine similarity search against document_chunks table in pgvector
  3. Feed top-k chunks as context to Claude for answer generation

The Supabase Session Pooler requires IPv4 — socket.getaddrinfo forces resolution
before connecting so psycopg doesn't attempt IPv6.
"""

import asyncio
import socket
from urllib.parse import urlparse

import psycopg
from pgvector.psycopg import register_vector

from backends.shared.claude import generate_answer
from backends.shared.embeddings import embed
from config import COLLECTION, PG_URL

TOP_K = 5


def _connect_params() -> dict:
    p = urlparse(PG_URL)
    ipv4 = socket.getaddrinfo(p.hostname, None, socket.AF_INET)[0][4][0]
    return dict(
        host=p.hostname,
        hostaddr=ipv4,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=p.password,
    )


def _fetch_chunks(question: str) -> list[dict]:
    query_vec = embed(question)
    with psycopg.connect(**_connect_params()) as conn:
        register_vector(conn)
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
                (query_vec, COLLECTION, query_vec, TOP_K),
            )
            return [
                {"source_doc": r[0], "chunk_text": r[1], "score": round(float(r[2]), 4)}
                for r in cur.fetchall()
            ]


async def retrieve(question: str) -> tuple[str, str]:
    """Return (context_str, summary_str) without generating an answer."""
    chunks = await asyncio.to_thread(_fetch_chunks, question)
    if not chunks:
        return "", "No chunks retrieved."
    context = "\n\n---\n\n".join(
        f"[{c['source_doc']} | score: {c['score']}]\n{c['chunk_text']}"
        for c in chunks
    )
    summary = "\n".join(f"• {c['source_doc']}  (score: {c['score']})" for c in chunks)
    return context, summary


async def query(question: str) -> tuple[str, str]:
    context, retrieved = await retrieve(question)
    if not context:
        return "No chunks retrieved.", "No relevant information found in the vector store."
    answer = await generate_answer(question, context)
    return retrieved, answer
