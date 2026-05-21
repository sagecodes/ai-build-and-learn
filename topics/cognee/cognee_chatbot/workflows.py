"""
workflows.py

Module-level re-exports for all Union tasks.
Deployed alongside app.py via flyte.serve() so Union can locate the tasks.

Tasks:
  ingest_pipeline(filenames, pdf_bytes_b64)  — add() + cognify()
  query_pipeline(question)                   — cognee.search() + Claude answer
  visualize_pipeline()                       — cognee.visualize_graph() → HTML string
"""

from ingest.pipeline import ingest_pipeline
from query.pipeline import query_pipeline
from visualize import visualize_pipeline

__all__ = ["ingest_pipeline", "query_pipeline", "visualize_pipeline"]
