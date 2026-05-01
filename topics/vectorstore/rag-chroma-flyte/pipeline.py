"""Flyte 2 RAG pipeline: HF dataset → chunks → embeddings → Chroma persist dir.

The final output is a `flyte.io.Dir` containing a Chroma `PersistentClient`
directory (sqlite + parquet shards). The chat app picks it up via
`flyte.app.RunOutput(task_name="rag-chroma-pipeline.rag_pipeline", type="directory")`
and downloads it to a local path on startup.

Usage:
    flyte run --local --tui pipeline.py rag_pipeline
    flyte run --local pipeline.py rag_pipeline --max_docs 500
    flyte run pipeline.py rag_pipeline                    # remote (devbox)
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from config import pipeline_env

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = pipeline_env


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — fetch dataset from HuggingFace
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def fetch_dataset(
    dataset_repo: str = "rag-datasets/rag-mini-wikipedia",
    dataset_config: str = "text-corpus",
    dataset_split: str = "passages",
    text_column: str = "passage",
    max_docs: int = 0,
) -> flyte.io.Dir:
    """Pull docs from HF and write a single jsonl with `{id, text}` per line."""
    from datasets import load_dataset

    log.info(f"Loading {dataset_repo} [{dataset_config or '-'}/{dataset_split}]")
    ds = (
        load_dataset(dataset_repo, dataset_config, split=dataset_split)
        if dataset_config
        else load_dataset(dataset_repo, split=dataset_split)
    )
    if max_docs and max_docs > 0:
        ds = ds.select(range(min(max_docs, len(ds))))

    out_dir = Path(tempfile.mkdtemp(prefix="rag_docs_"))
    docs_path = out_dir / "docs.jsonl"
    with docs_path.open("w") as f:
        for i, row in enumerate(ds):
            text = (row.get(text_column) or "").strip()
            if not text:
                continue
            f.write(json.dumps({"id": str(row.get("id", i)), "text": text}) + "\n")

    n = sum(1 for _ in docs_path.open())
    log.info(f"Wrote {n} docs to {docs_path}")
    await flyte.report.replace.aio(
        f"<h2>Fetched dataset</h2>"
        f"<p><b>Repo:</b> {dataset_repo}</p>"
        f"<p><b>Rows kept:</b> {n}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — chunk each doc with paragraph-aware char splitting
# ──────────────────────────────────────────────────────────────────────────────

def _split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive character splitter: paragraphs → lines → sentences → words → chars.

    Char-based (not token-based) so we don't pull in a tokenizer just for
    splitting. With chunk_size=1200 and bge-small's 512-token limit there's
    plenty of headroom (~2.5–3 chars per token in English).
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for sep in ("\n\n", "\n", ". ", " ", ""):
        if sep == "":
            return [text[i:i + chunk_size] for i in range(0, len(text), max(1, chunk_size - overlap))]
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        out: list[str] = []
        buf = ""
        for part in parts:
            piece = part + sep
            if len(piece) > chunk_size:
                if buf.strip():
                    out.append(buf.strip())
                    buf = ""
                out.extend(_split_text(part, chunk_size, overlap))
                continue
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf.strip():
                    out.append(buf.strip())
                tail = buf[-overlap:] if overlap and buf else ""
                buf = tail + piece
        if buf.strip():
            out.append(buf.strip())
        return [c for c in out if c.strip()]
    return [text]


@env.task(cache="auto", report=True)
async def chunk_documents(
    docs_dir: flyte.io.Dir,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> flyte.io.Dir:
    """Read docs.jsonl, split each into chunks, write chunks.jsonl."""
    in_path = Path(await docs_dir.download()) / "docs.jsonl"

    out_dir = Path(tempfile.mkdtemp(prefix="rag_chunks_"))
    chunks_path = out_dir / "chunks.jsonl"
    n_docs = 0
    n_chunks = 0
    with in_path.open() as fin, chunks_path.open("w") as fout:
        for line in fin:
            doc = json.loads(line)
            n_docs += 1
            for j, chunk in enumerate(_split_text(doc["text"], chunk_size, chunk_overlap)):
                fout.write(json.dumps({
                    "chunk_id": f"{doc['id']}::{j}",
                    "doc_id": doc["id"],
                    "text": chunk,
                }) + "\n")
                n_chunks += 1

    log.info(f"Chunked {n_docs} docs → {n_chunks} chunks (avg {n_chunks / max(n_docs, 1):.1f}/doc)")
    await flyte.report.replace.aio(
        f"<h2>Chunked documents</h2>"
        f"<p><b>Docs:</b> {n_docs}</p>"
        f"<p><b>Chunks:</b> {n_chunks} (avg {n_chunks / max(n_docs, 1):.1f}/doc)</p>"
        f"<p><b>chunk_size:</b> {chunk_size} chars · "
        f"<b>overlap:</b> {chunk_overlap}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — embed + write Chroma persistent dir
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def embed_and_index(
    chunks_dir: flyte.io.Dir,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    collection_name: str = "rag_demo",
    batch_size: int = 64,
) -> flyte.io.Dir:
    """Embed chunks and persist to a Chroma `PersistentClient` directory.

    The collection's metadata stores the embedding model name so the chat app
    can verify it loads a matching encoder for the query side.
    """
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunks_path = Path(await chunks_dir.download()) / "chunks.jsonl"
    rows = [json.loads(ln) for ln in chunks_path.open()]
    log.info(f"Embedding {len(rows)} chunks with {embedding_model}")

    encoder = SentenceTransformer(embedding_model)

    persist_dir = Path(tempfile.mkdtemp(prefix="chroma_"))
    client = chromadb.PersistentClient(path=str(persist_dir))
    coll = client.get_or_create_collection(
        name=collection_name,
        metadata={
            "embedding_model": embedding_model,
            "hnsw:space": "cosine",
        },
    )

    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [r["text"] for r in batch]
        # bge encoders prefer L2-normalized embeddings paired with cosine distance.
        vectors = encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        coll.add(
            ids=[r["chunk_id"] for r in batch],
            documents=texts,
            embeddings=vectors,
            metadatas=[{"doc_id": r["doc_id"]} for r in batch],
        )
        log.info(f"  indexed {min(start + batch_size, len(rows))}/{len(rows)}")

    log.info(f"Chroma collection '{collection_name}' has {coll.count()} chunks at {persist_dir}")
    await flyte.report.replace.aio(
        f"<h2>Embedded + indexed</h2>"
        f"<p><b>Embedding model:</b> {embedding_model}</p>"
        f"<p><b>Chunks indexed:</b> {coll.count()}</p>"
        f"<p><b>Collection:</b> {collection_name}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(persist_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline — orchestrator. Returns the Chroma dir as o0 so the chat app's
# RunOutput(task_name="...rag_pipeline", type="directory") resolves to it.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def rag_pipeline(
    dataset_repo: str = "rag-datasets/rag-mini-wikipedia",
    dataset_config: str = "text-corpus",
    dataset_split: str = "passages",
    text_column: str = "passage",
    max_docs: int = 0,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    collection_name: str = "rag_demo",
) -> flyte.io.Dir:
    await flyte.report.replace.aio("<h2>RAG pipeline</h2><p>Step 1/3 — fetching dataset…</p>")
    await flyte.report.flush.aio()
    docs = await fetch_dataset(
        dataset_repo, dataset_config, dataset_split, text_column, max_docs,
    )

    await flyte.report.replace.aio("<h2>RAG pipeline</h2><p>Step 2/3 — chunking…</p>")
    await flyte.report.flush.aio()
    chunks = await chunk_documents(docs, chunk_size, chunk_overlap)

    await flyte.report.replace.aio("<h2>RAG pipeline</h2><p>Step 3/3 — embedding + indexing…</p>")
    await flyte.report.flush.aio()
    chroma_dir = await embed_and_index(chunks, embedding_model, collection_name)

    await flyte.report.replace.aio(
        "<h2>RAG pipeline complete</h2>"
        f"<p>Chroma collection <code>{collection_name}</code> ready.</p>"
        "<p>Pass this run's name to the chat app: "
        "<code>RAG_PIPELINE_RUN=&lt;run-name&gt; python chat_app.py</code>.</p>"
    )
    await flyte.report.flush.aio()
    return chroma_dir


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(rag_pipeline)
    print(f"Pipeline run: {run.name}")
    print(f"  {run.url}")
