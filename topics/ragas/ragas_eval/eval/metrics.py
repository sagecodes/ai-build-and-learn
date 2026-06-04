from ragas.metrics import (
    context_precision,
    context_recall,
    faithfulness,
)

# answer_relevancy requires an embeddings model (ONNX-based fastembed crashes
# on this VM's CPU instruction set). The three LLM-only metrics run fine.
METRICS = [faithfulness, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "context_precision", "context_recall"]


def configure_metrics(llm, embeddings=None) -> None:
    """Inject LLM into metrics — fallback for older ragas 0.2.x."""
    for metric in METRICS:
        metric.llm = llm
