"""
workflows.py — flyte 2.x task entry point for graph_rag_chatbot.

Deploy to Union with:
    python app.py --deploy

Both tasks are dispatched at runtime via flyte.run() from app.py.

ingest_pipeline(filenames, pdf_bytes_b64)
    Parses PDFs, extracts entities and relationships via Claude, loads
    the full graph to Neo4j AuraDB, builds the HNSW vector index,
    resolves duplicate entities, detects communities with Louvain,
    and generates community summaries.

query_pipeline(question)
    Routes the question to the best retrieval mode (hybrid / entity /
    community), fetches relevant context from Neo4j, and generates a
    grounded answer via Claude.
"""

from ingest import ingest_pipeline
from query import query_pipeline

__all__ = ["ingest_pipeline", "query_pipeline"]
