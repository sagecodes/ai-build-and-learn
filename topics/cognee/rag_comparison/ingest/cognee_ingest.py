"""
cognee_ingest.py — ingest Everstorm PDFs into Cognee (local LanceDB + SQLite).

Pipeline:
  1. Configure Cognee env vars (must happen before importing cognee)
  2. Call cognee.add() for each PDF — loads documents into Cognee's pipeline
  3. Call cognee.cognify() once — builds the vector index and knowledge graph

Cognee stores all data locally under .cognee_data/ (configurable via COGNEE_DATA_PATH
in .env). No external database credentials required.

Run once before launching the app:
    python ingest/cognee_ingest.py
"""

import asyncio
import logging
import os
from pathlib import Path

from config import (
    ANTHROPIC_API_KEY,
    CLAUDE_MODEL,
    COGNEE_DATA_PATH,
    DATA_DIR,
    EMBEDDING_DIMS,
    EMBEDDING_MODEL,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


def _configure() -> None:
    """Set Cognee environment variables before importing cognee."""
    os.environ["LLM_PROVIDER"]         = "anthropic"
    os.environ["LLM_MODEL"]            = CLAUDE_MODEL
    os.environ["LLM_API_KEY"]          = ANTHROPIC_API_KEY
    os.environ["VECTOR_DB_PROVIDER"]   = "lancedb"
    os.environ["EMBEDDING_PROVIDER"]   = "fastembed"
    os.environ["EMBEDDING_MODEL"]      = EMBEDDING_MODEL
    os.environ["EMBEDDING_DIMENSIONS"] = str(EMBEDDING_DIMS)
    os.environ["DATA_PATH"]            = COGNEE_DATA_PATH


async def ingest(data_dir: str = str(DATA_DIR)) -> None:
    _configure()

    import cognee

    pdf_paths = sorted(Path(data_dir).glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {data_dir}")

    log.info(f"Found {len(pdf_paths)} PDFs — adding to Cognee...")
    for pdf_path in pdf_paths:
        await cognee.add(str(pdf_path))
        log.info(f"  Added: {pdf_path.name}")

    log.info("\nRunning cognee.cognify() — building vector index and knowledge graph...")
    log.info("(This may take several minutes on first run)")
    await cognee.cognify()

    log.info(f"\nDone. Cognee data stored at: {COGNEE_DATA_PATH}")


if __name__ == "__main__":
    asyncio.run(ingest())
