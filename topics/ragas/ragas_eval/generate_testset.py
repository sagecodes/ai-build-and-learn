"""
generate_testset.py — One-shot script: Everstorm PDFs → data/testset.json

Run inside Docker after the image is built and pushed:
    docker exec -w /app rag-comparison-ragas python ragas_eval/generate_testset.py

Or locally (with ANTHROPIC_API_KEY in .env):
    python topics/ragas/ragas_eval/generate_testset.py
"""

import json
import logging
import os
import sys
from pathlib import Path

# Make ragas_eval importable whether run from repo root or its own directory
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from dotenv import load_dotenv

load_dotenv()

import config  # noqa: F401 — triggers missing-env-var check before anything else
from config import DATA_DIR
from ragas_wrappers import get_ragas_embeddings, get_ragas_llm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

# PDFs live at /app/data/ in Docker; fall back to rag_comparison/data/ for local dev
_DOCKER_DATA = Path("/app/data")
_LOCAL_DATA  = Path(__file__).parents[2] / "cognee" / "rag_comparison" / "data"
_PDF_DIR     = _DOCKER_DATA if _DOCKER_DATA.exists() else _LOCAL_DATA

TESTSET_SIZE  = 20
TESTSET_PATH  = DATA_DIR / "testset.json"


def _load_documents():
    from langchain_community.document_loaders import PyMuPDFLoader

    pdfs = sorted(_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {_PDF_DIR}")
    logger.info("Loading %d PDFs from %s", len(pdfs), _PDF_DIR)

    docs = []
    for pdf in pdfs:
        loader = PyMuPDFLoader(str(pdf))
        docs.extend(loader.load())

    logger.info("Loaded %d document pages total", len(docs))
    return docs


def _generate(docs, llm, emb):
    from ragas.testset import TestsetGenerator

    generator = TestsetGenerator.from_langchain(
        generator_llm=llm,
        critic_llm=llm,
        embeddings=emb,
    )
    try:
        return generator.generate_with_langchain_docs(
            documents=docs,
            test_size=TESTSET_SIZE,
            raise_exceptions=True,
        )
    except TypeError:
        # Older ragas 0.2.x — distributions kwarg not supported
        return generator.generate_with_langchain_docs(
            documents=docs,
            test_size=TESTSET_SIZE,
        )


def main():
    docs = _load_documents()
    llm  = get_ragas_llm()
    emb  = get_ragas_embeddings()

    logger.info("Generating testset (size=%d) — this may take several minutes...", TESTSET_SIZE)
    testset = _generate(docs, llm, emb)

    df = testset.to_pandas()
    logger.info("Generated %d rows. Columns: %s", len(df), list(df.columns))

    # Column names vary slightly across ragas sub-releases
    records = []
    for _, row in df.iterrows():
        q  = row.get("question") or row.get("user_input", "")
        gt = row.get("ground_truth") or row.get("reference", "")
        if q and gt:
            records.append({"question": str(q).strip(), "ground_truth": str(gt).strip()})

    if not records:
        raise ValueError(
            "No valid question/ground_truth pairs in generated testset. "
            f"Available columns: {list(df.columns)}"
        )

    logger.info("Saving %d Q&A pairs to %s", len(records), TESTSET_PATH)
    tmp = TESTSET_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    tmp.replace(TESTSET_PATH)
    logger.info("Done.")


if __name__ == "__main__":
    main()
