"""
ingest/chunking.py

parse_and_chunk: decode a PDF and split into overlapping text chunks.
Called sequentially inside ingest_pipeline for each PDF.
"""

import base64
import json

import fitz  # PyMuPDF
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


def parse_and_chunk(source_doc: str, pdf_bytes_b64: str) -> str:
    """
    Parse a PDF and split into overlapping text chunks.

    Args:
        source_doc:     Filename used as the Document node name in Neo4j.
        pdf_bytes_b64:  Base64-encoded PDF bytes.

    Returns:
        JSON string — list of {source_doc, chunk_index, chunk_text}.
    """
    raw = base64.b64decode(pdf_bytes_b64)

    with fitz.open(stream=raw, filetype="pdf") as doc:
        full_text = "\n".join(page.get_text() for page in doc)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    texts = splitter.split_text(full_text)

    chunks = [
        {"source_doc": source_doc, "chunk_index": i, "chunk_text": text}
        for i, text in enumerate(texts)
    ]

    return json.dumps(chunks)
