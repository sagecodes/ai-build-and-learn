import asyncio
import json
import logging

import compat  # noqa: F401 — must be first; stubs broken ragas deps before import
import pandas as pd
from datasets import Dataset
from ragas import evaluate

from config import DATA_DIR
from eval.backends import BACKENDS, generate_answer
from eval.metrics import METRIC_NAMES, METRICS, configure_metrics
from eval.testset import load_testset
from ragas_wrappers import get_ragas_embeddings, get_ragas_llm

logger = logging.getLogger(__name__)

_PARTIAL_PATH = DATA_DIR / "results_partial.json"


def _run_ragas(dataset: Dataset) -> dict:
    """
    Call ragas.evaluate(), handling the kwarg API difference across 0.2.x sub-releases.
    Returns a flat dict of {metric_name: float}.
    """
    llm = get_ragas_llm()
    emb = get_ragas_embeddings()
    try:
        result = evaluate(dataset=dataset, metrics=METRICS, llm=llm, embeddings=emb)
    except TypeError:
        # Older ragas 0.2.x does not accept llm=/embeddings= kwargs
        configure_metrics(llm, emb)
        result = evaluate(dataset=dataset, metrics=METRICS)

    row = result.to_pandas().iloc[0].to_dict()
    return {
        k: float(row[k]) if row.get(k) is not None else float("nan")
        for k in METRIC_NAMES
    }


async def _eval_one_backend(
    name: str,
    retrieve_fn,
    testset_item: dict,
) -> dict:
    """
    Run retrieve → generate → ragas evaluate for a single backend + question.
    Never raises — on any failure returns NaN scores and records the error string.
    """
    question = testset_item["question"]
    ground_truth = testset_item.get("ground_truth", "")
    base = {"backend": name, "question": question}

    try:
        context_str, summary = await retrieve_fn(question)
        answer = await generate_answer(question, context_str)

        dataset = Dataset.from_dict({
            "question":     [question],
            "answer":       [answer],
            "contexts":     [[context_str]],
            "ground_truth": [ground_truth],
        })

        scores = await asyncio.to_thread(_run_ragas, dataset)

        return {
            **base,
            "answer":          answer,
            "context_summary": summary,
            "error":           None,
            **scores,
        }

    except Exception as e:
        logger.exception("Eval failed  backend=%r  question=%r", name, question)
        return {
            **base,
            "answer":          "",
            "context_summary": "",
            "error":           str(e),
            **{k: float("nan") for k in METRIC_NAMES},
        }


async def run_single_eval(question: str, ground_truth: str = "") -> list[dict]:
    """Evaluate one question across all backends concurrently."""
    item = {"question": question, "ground_truth": ground_truth}
    tasks = [
        _eval_one_backend(name, fn, item)
        for name, fn in BACKENDS.items()
    ]
    return list(await asyncio.gather(*tasks))


async def run_batch_eval() -> pd.DataFrame:
    """
    Evaluate the full testset across all backends.

    Questions run sequentially to avoid hammering APIs; backends for each
    question run concurrently. Partial results are written after every question
    so a kill mid-run can be inspected. The partial file is deleted on clean exit
    and detected + cleared on the next run.

    Returns a DataFrame of mean scores per backend.
    """
    testset = load_testset()

    if _PARTIAL_PATH.exists():
        logger.warning(
            "Stale partial results found at %s — discarding and re-running.", _PARTIAL_PATH
        )
        _PARTIAL_PATH.unlink()

    all_results: list[dict] = []

    for item in testset:
        tasks = [
            _eval_one_backend(name, fn, item)
            for name, fn in BACKENDS.items()
        ]
        row_results = list(await asyncio.gather(*tasks))
        all_results.extend(row_results)

        # Atomic partial write after each question
        tmp = _PARTIAL_PATH.with_suffix(".tmp")
        tmp.write_text(json.dumps(all_results, indent=2))
        tmp.replace(_PARTIAL_PATH)

    _PARTIAL_PATH.unlink(missing_ok=True)

    df = pd.DataFrame(all_results)
    return df.groupby("backend")[METRIC_NAMES].mean().reset_index().round(4)
