import asyncio
import logging

import gradio as gr
import pandas as pd

from eval.runner import run_batch_eval, run_single_eval
from eval.metrics import METRIC_NAMES

logger = logging.getLogger(__name__)

_EMPTY_DF = pd.DataFrame(columns=["backend"] + METRIC_NAMES)


def handle_batch_eval():
    """
    Generator — yields button-disabled state first so the UI locks immediately,
    then yields results (or error) and re-enables the button.
    """
    yield gr.update(interactive=False, value="Running... this takes 5-10 min"), _EMPTY_DF, gr.update(visible=False)
    try:
        df = asyncio.run(run_batch_eval())
        yield gr.update(interactive=True, value="Run Batch Evaluation"), df, gr.update(visible=False)
    except Exception as e:
        logger.exception("Batch eval failed")
        yield gr.update(interactive=True, value="Run Batch Evaluation"), _EMPTY_DF, gr.update(
            value=str(e), visible=True
        )


def handle_single_eval(question: str, ground_truth: str):
    """
    Generator — yields locked UI first, then per-backend results.
    Returns 10 values: [single_btn] + [ctx, answer, scores] × 3 backends.
    """
    _empty = ("", "", "")
    empty_outputs = (gr.update(interactive=False, value="Evaluating..."),) + _empty * 3
    yield empty_outputs

    if not question.strip():
        yield (gr.update(interactive=True, value="Evaluate"),) + _empty * 3
        return

    try:
        results = asyncio.run(run_single_eval(question, ground_truth))
    except Exception as e:
        logger.exception("Single eval failed")
        error_scores = f"**Error:** {e}"
        yield (gr.update(interactive=True, value="Evaluate"),) + ("", "", error_scores) * 3
        return

    outputs = [gr.update(interactive=True, value="Evaluate")]
    for r in results:
        ctx    = r.get("context_summary", "")
        answer = r.get("answer", "")
        if r.get("error"):
            scores = f"**Error:** {r['error']}"
        else:
            lines = []
            for metric in METRIC_NAMES:
                v = r.get(metric, float("nan"))
                if v != v:  # NaN check
                    lines.append(f"**{metric}:** N/A")
                else:
                    lines.append(f"**{metric}:** {v:.4f}")
            scores = "\n\n".join(lines)
        outputs.extend([ctx, answer, scores])

    yield tuple(outputs)
