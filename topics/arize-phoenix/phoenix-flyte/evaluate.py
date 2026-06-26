"""LLM-as-a-judge over the captured traces, as a Flyte task.

This closes the observability loop entirely inside your stack: the research
pipeline writes spans to the self-hosted Phoenix (workflow.py), and this task
reads those same spans back out, scores them with an LLM judge, and writes the
scores back onto the spans as annotations. Open the Phoenix UI afterward and the
research-answer spans carry `answer_relevance` and `answer_completeness` labels.

It is the thing Flyte's tracing does NOT do: turn the trace into an evaluation.
And the judge can be OpenAI or your in-cluster gemma4 vLLM (a local, no-cost
judge), via the same --provider switch the pipeline uses.

Usage:
    # Judge the research answers captured in the research-pipeline project
    flyte run evaluate.py evaluate_traces

    # Judge with the local vLLM instead of OpenAI
    flyte run evaluate.py evaluate_traces --provider vllm

    # Point at a different project
    flyte run evaluate.py evaluate_traces --project_name research-pipeline
"""

import json
import logging

import flyte

from config import (
    eval_env,
    LLM_PROVIDER,
    OPENAI_MODEL,
    PHOENIX_BASE_URL,
    PHOENIX_PROJECT_NAME,
    VLLM_MODEL_ID,
    VLLM_URL,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

env = eval_env


# ── Span text extraction ───────────────────────────────────────────────────────
# Spans store input/output as message-JSON blobs; pull clean text out of them.

def _content_of(m) -> str:
    if isinstance(m, dict):
        if "content" in m:
            return m["content"] or ""
        if isinstance(m.get("data"), dict):
            return m["data"].get("content", "") or ""
    return ""


def _first_user(value: str) -> str:
    """The user's question from an input messages blob."""
    try:
        msgs = json.loads(value).get("messages", [])
    except Exception:
        return str(value)
    for m in msgs:
        if m.get("role") == "user" or m.get("type") == "human":
            return _content_of(m)
    return _content_of(msgs[0]) if msgs else ""


def _last_msg(value: str) -> str:
    """The final answer from an output messages blob."""
    try:
        msgs = json.loads(value).get("messages", [])
    except Exception:
        return str(value)
    return _content_of(msgs[-1]) if msgs else str(value)


# ── Judge model (provider-switchable, same as the pipeline) ────────────────────

def _build_judge(provider: str):
    from phoenix.evals.llm import LLM

    if provider == "vllm":
        # The in-cluster gemma4 vLLM app as a local, no-cost judge.
        return LLM(
            provider="openai",
            model=VLLM_MODEL_ID,
            base_url=VLLM_URL.rstrip("/") + "/v1",
            api_key="not-used",
        )
    return LLM(provider="openai", model=OPENAI_MODEL)


# ── Eval task ──────────────────────────────────────────────────────────────────

# Spans this judge scores: the full synthesized report and each per-topic answer.
#   research_report  the orchestrator's final-report span (query in, report out)
#   LangGraph        each research subgraph root (one per sub-topic)
TARGET_SPAN_NAMES = ["research_report", "LangGraph"]


@env.task
async def evaluate_traces(
    project_name: str = PHOENIX_PROJECT_NAME,
    provider: str = LLM_PROVIDER,
    limit: int = 50,
) -> str:
    """Score captured answer spans with an LLM judge; write annotations back.

    Judges both the full synthesized report (`research_report` span) and each
    per-topic research answer (`LangGraph` span), whose input is the question and
    output is the answer.
    """
    import pandas as pd
    from phoenix.client import Client
    from phoenix.evals import create_classifier, evaluate_dataframe

    px = Client(base_url=PHOENIX_BASE_URL)
    log.info(f"[eval] pulling spans from {PHOENIX_BASE_URL} (project={project_name})")
    df = px.spans.get_spans_dataframe(project_identifier=project_name)

    targets = df[df["name"].isin(TARGET_SPAN_NAMES)].copy()
    if len(targets) > limit:
        targets = targets.tail(limit)
    if targets.empty:
        msg = f"No spans named {TARGET_SPAN_NAMES} in project {project_name!r} to evaluate."
        log.warning(f"[eval] {msg}")
        return json.dumps({"evaluated": 0, "message": msg})

    targets["input"] = targets["attributes.input.value"].map(_first_user)
    targets["output"] = targets["attributes.output.value"].map(_last_msg)
    by_name = targets["name"].value_counts().to_dict()
    log.info(f"[eval] judging {len(targets)} answer span(s) {by_name} with provider={provider}")

    judge = _build_judge(provider)

    relevance = create_classifier(
        name="answer_relevance",
        llm=judge,
        prompt_template=(
            "You are grading a research assistant.\n"
            "QUESTION:\n{input}\n\nANSWER:\n{output}\n\n"
            "Is the ANSWER a relevant, on-topic, and useful response to the QUESTION?"
        ),
        choices={"relevant": 1.0, "irrelevant": 0.0},
    )
    completeness = create_classifier(
        name="answer_completeness",
        llm=judge,
        prompt_template=(
            "You are grading a research assistant.\n"
            "QUESTION:\n{input}\n\nANSWER:\n{output}\n\n"
            "Does the ANSWER thoroughly address the QUESTION, covering its main "
            "aspects rather than only part of it?"
        ),
        choices={"complete": 1.0, "incomplete": 0.0},
    )

    results = evaluate_dataframe(dataframe=targets, evaluators=[relevance, completeness])

    # Flatten each evaluator's nested score dict and log it back as annotations,
    # so the labels + explanations show up on the spans in the Phoenix UI.
    summary = {}
    for name in ("answer_relevance", "answer_completeness"):
        scores = results[f"{name}_score"]
        ann = pd.DataFrame(
            {
                "label": scores.map(lambda d: d.get("label")),
                "score": scores.map(lambda d: d.get("score")),
                "explanation": scores.map(lambda d: d.get("explanation")),
            },
            index=results.index,
        )
        ann.index.name = "span_id"
        px.spans.log_span_annotations_dataframe(
            dataframe=ann, annotation_name=name, annotator_kind="LLM", sync=True
        )
        summary[name] = {
            "mean_score": round(float(ann["score"].mean()), 3),
            "labels": ann["label"].value_counts().to_dict(),
        }
        log.info(f"[eval] {name}: {summary[name]}")

    return json.dumps({
        "evaluated": int(len(targets)),
        "project": project_name,
        "spans_by_name": by_name,
        "judge_provider": provider,
        "summary": summary,
    })


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(evaluate_traces)
    print(run.url)
