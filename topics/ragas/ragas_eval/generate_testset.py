"""
generate_testset.py — One-shot script: Everstorm PDFs → data/testset.json

Uses Claude directly to generate question/ground_truth pairs from PDF chunks,
bypassing ragas TestsetGenerator (which requires native binaries unavailable
on this VM's CPU).

Run inside Docker:
    docker exec -w /app rag-comparison-ragas python ragas_eval/generate_testset.py

Or locally:
    python topics/ragas/ragas_eval/generate_testset.py
"""

import asyncio
import json
import logging
import sys
from pathlib import Path

_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

from dotenv import load_dotenv
load_dotenv()

import compat  # noqa: F401 — stubs broken ragas deps before any ragas import
import config  # noqa: F401 — triggers env var check early
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL, DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

_DOCKER_DATA = Path("/app/data")
_LOCAL_DATA  = Path(__file__).parents[2] / "cognee" / "rag_comparison" / "data"
_PDF_DIR     = _DOCKER_DATA if _DOCKER_DATA.exists() else _LOCAL_DATA

TESTSET_SIZE = 20
TESTSET_PATH = DATA_DIR / "testset.json"

_PROMPT = """\
You are creating evaluation data for a RAG system about Everstorm Outfitters, \
an outdoor gear company.

Read the document excerpt below and generate ONE question that can be answered \
using only this excerpt, plus the ground-truth answer.

Rules:
- Question must be specific and answerable from the excerpt alone
- Ground truth must be a complete, accurate answer (1-3 sentences)
- Do not ask about page numbers, formatting, or document structure

Excerpt:
{chunk}

Respond with JSON only — no explanation, no markdown fences:
{{"question": "...", "ground_truth": "..."}}"""


def _load_chunks() -> list[str]:
    from langchain_community.document_loaders import PyMuPDFLoader

    pdfs = sorted(_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found in {_PDF_DIR}")
    logger.info("Loading PDFs from %s", _PDF_DIR)

    pages = []
    for pdf in pdfs:
        pages.extend(PyMuPDFLoader(str(pdf)).load())

    # Keep pages with enough content; sample evenly across all PDFs
    chunks = [p.page_content.strip() for p in pages if len(p.page_content.strip()) > 200]
    logger.info("%d usable pages from %d PDFs", len(chunks), len(pdfs))
    return chunks


async def _generate_pair(client, chunk: str, idx: int) -> dict | None:
    from anthropic import AsyncAnthropic
    try:
        resp = await client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=512,
            messages=[{"role": "user", "content": _PROMPT.format(chunk=chunk[:2000])}],
        )
        text = resp.content[0].text.strip()
        pair = json.loads(text)
        if pair.get("question") and pair.get("ground_truth"):
            return {"question": pair["question"].strip(), "ground_truth": pair["ground_truth"].strip()}
    except Exception as e:
        logger.warning("Skipping chunk %d: %s", idx, e)
    return None


async def _generate_all(chunks: list[str]) -> list[dict]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)

    # Sample evenly across available chunks to hit TESTSET_SIZE
    step = max(1, len(chunks) // TESTSET_SIZE)
    selected = chunks[::step][:TESTSET_SIZE * 2]  # extra in case some fail

    logger.info("Generating Q&A pairs from %d chunks...", len(selected))
    tasks = [_generate_pair(client, chunk, i) for i, chunk in enumerate(selected)]
    results = await asyncio.gather(*tasks)

    pairs = [r for r in results if r is not None]
    logger.info("Got %d valid pairs", len(pairs))
    return pairs[:TESTSET_SIZE]


def main():
    chunks = _load_chunks()
    pairs  = asyncio.run(_generate_all(chunks))

    if not pairs:
        raise ValueError("No Q&A pairs generated — check API key and PDF content.")

    logger.info("Saving %d pairs to %s", len(pairs), TESTSET_PATH)
    tmp = TESTSET_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(pairs, indent=2))
    tmp.replace(TESTSET_PATH)
    logger.info("Done.")


if __name__ == "__main__":
    main()
