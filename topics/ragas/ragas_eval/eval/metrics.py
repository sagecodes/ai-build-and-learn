from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)

METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]
METRIC_NAMES = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]


def configure_metrics(llm, embeddings) -> None:
    """Inject LLM and embeddings into metrics — fallback for older ragas 0.2.x."""
    for metric in METRICS:
        metric.llm = llm
        if hasattr(metric, "embeddings"):
            metric.embeddings = embeddings
