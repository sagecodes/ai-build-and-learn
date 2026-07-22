"""
vector_ingest.py — ingest Everstorm PDFs into Supabase pgvector.

Pipeline:
  1. Extract text from each PDF with pymupdf
  2. Split into overlapping chunks with RecursiveCharacterTextSplitter
  3. Batch-embed all chunks with fastembed (single pass — fastest)
  4. Create table + HNSW index if they don't exist
  5. Delete existing rows for this collection (idempotent re-ingest)
  6. Insert all chunk vectors

Run once before launching the app:
    python ingest/vector_ingest.py
"""

import logging
import socket
from pathlib import Path
from urllib.parse import urlparse

import psycopg
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pgvector.psycopg import register_vector

from backends.shared.embeddings import get_embedder
from config import COLLECTION, DATA_DIR, EMBEDDING_DIMS, PG_URL

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def _connect() -> psycopg.Connection:
    p = urlparse(PG_URL)
    ipv4 = socket.getaddrinfo(p.hostname, None, socket.AF_INET)[0][4][0]
    return psycopg.connect(
        host=p.hostname,
        hostaddr=ipv4,
        port=p.port or 5432,
        dbname=p.path.lstrip("/"),
        user=p.username,
        password=p.password,
    )


def _ensure_schema(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS document_chunks (
                id              BIGSERIAL PRIMARY KEY,
                collection_name TEXT    NOT NULL,
                source_doc      TEXT    NOT NULL,
                chunk_index     INTEGER NOT NULL,
                chunk_text      TEXT    NOT NULL,
                embedding       vector({EMBEDDING_DIMS}) NOT NULL
            )
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding
            ON document_chunks USING hnsw (embedding vector_cosine_ops)
        """)
    conn.commit()


def _extract_text(pdf_path: Path) -> str:
    doc = pymupdf.open(str(pdf_path))
    text = "".join(page.get_text() for page in doc)
    doc.close()
    return text


def ingest(data_dir: str = str(DATA_DIR)) -> None:
    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    log.info(f"Found {len(pdf_paths)} PDFs")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    all_chunks: list[dict] = []
    for pdf_path in pdf_paths:
        text   = _extract_text(pdf_path)
        chunks = splitter.split_text(text)
        for i, chunk_text in enumerate(chunks):
            all_chunks.append({
                "source_doc":  pdf_path.name,
                "chunk_index": i,
                "chunk_text":  chunk_text,
            })
        log.info(f"  {pdf_path.name}: {len(chunks)} chunks")

    log.info(f"\nTotal: {len(all_chunks)} chunks — batch embedding with fastembed...")

    embedder   = get_embedder()
    texts      = [c["chunk_text"] for c in all_chunks]
    embeddings = list(embedder.embed(texts))

    log.info("Connecting to Supabase pgvector...")
    conn = _connect()
    register_vector(conn)
    _ensure_schema(conn)

    with conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM document_chunks WHERE collection_name = %s",
                (COLLECTION,),
            )
            cur.executemany(
                """
                INSERT INTO document_chunks
                    (collection_name, source_doc, chunk_index, chunk_text, embedding)
                VALUES (%s, %s, %s, %s, %s)
                """,
                [
                    (
                        COLLECTION,
                        all_chunks[i]["source_doc"],
                        all_chunks[i]["chunk_index"],
                        all_chunks[i]["chunk_text"],
                        embeddings[i].tolist(),
                    )
                    for i in range(len(all_chunks))
                ],
            )

    conn.close()
    log.info(f"\nDone. {len(all_chunks)} vectors written to collection '{COLLECTION}'.")


if __name__ == "__main__":
    ingest()
