"""LLM research pipeline as a Flyte workflow, traced + judged in MLflow.

Decomposed the way you'd normally build it in Flyte — each step is its own
task, visible in the Flyte UI with its own compute, logs, and report:

    plan_topics ──> research_topic(t1) ──┐
                ├─> research_topic(t2) ──┤─> judge_research ──> research_pipeline
                └─> research_topic(tN) ──┘

Side-by-side of what each tool records:
  - Flyte  stitches the DAG: plan → research (fan-out) → judge.
  - MLflow gives the deep per-step detail via autolog: every LLM call, tool
    use, and graph step is a trace, and judge_research logs an evaluation run
    with LLM-as-a-judge scores. The agent's system prompt comes from the
    MLflow Prompt Registry, so research traces link to a prompt version.

Run remote:
    flyte run agent_tracing.py research_pipeline --query "What is OpenTelemetry?"
    flyte run agent_tracing.py register_judges        # populate the Judges UI

Run local:
    flyte run --local agent_tracing.py research_pipeline --query "What is MLflow?"
"""

from __future__ import annotations

from dataclasses import dataclass

import flyte

from config import agent_env, MLFLOW_TRACKING_URI, OPENAI_MODEL

EXPERIMENT = "llm-agent-tracing"

# ── Prompt Registry ──────────────────────────────────────────────────────────
PROMPT_NAME = "research-agent-prompt"
PROMPT_TEMPLATE = (
    "You are a meticulous research assistant. Research the topic below using "
    "the available tools, then provide a comprehensive, well-sourced answer.\n\n"
    "Topic: {{query}}"
)


def _load_or_register_prompt():
    """Fetch the agent's system prompt from the MLflow Prompt Registry.

    Registers it on first use, then loads the latest version on subsequent
    runs. Demonstrates MLflow 3's first-class, versioned prompt management.
    """
    import mlflow

    prompt = mlflow.load_prompt(PROMPT_NAME, allow_missing=True)
    if prompt is None:
        prompt = mlflow.register_prompt(
            name=PROMPT_NAME,
            template=PROMPT_TEMPLATE,
            commit_message="Research agent system prompt",
        )
    return prompt


@dataclass
class TopicReport:
    topic: str
    report: str


@dataclass
class PipelineResult:
    query: str
    topics: list[str]
    answer: str
    judge_metrics: dict


def _md_to_html(text: str) -> str:
    import markdown
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ── Task 1: plan ─────────────────────────────────────────────────────────────
@agent_env.task(report=True)
async def plan_topics(query: str, num_topics: int = 3) -> list[str]:
    """Break a research query into focused sub-topics (one traced LLM call)."""
    import json

    import mlflow
    import mlflow.langchain
    from langchain_openai import ChatOpenAI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    mlflow.langchain.autolog()

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    resp = llm.invoke(
        f"Break this research question into exactly {num_topics} focused sub-topics. "
        f"Return ONLY a JSON array of strings, nothing else.\n\nQuestion: {query}"
    )
    try:
        topics = json.loads(resp.content)[:num_topics]
    except json.JSONDecodeError:
        topics = [query]

    print(f"[plan] {query!r} -> {topics}")
    items = "".join(f"<li>{t}</li>" for t in topics)
    await flyte.report.log.aio(f"<h2>Planning</h2><p>{query}</p><ul>{items}</ul>")
    return topics


# ── Task 2: research (fan-out, one task per sub-topic) ───────────────────────
@agent_env.task(report=True)
async def research_topic(topic: str) -> TopicReport:
    """A ReAct agent (Tavily web search) researches one sub-topic. Traced by MLflow."""
    import mlflow
    import mlflow.langchain
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from tavily import TavilyClient

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    mlflow.langchain.autolog()

    prompt = _load_or_register_prompt()
    tavily = TavilyClient()

    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""
        results = tavily.search(query=query, max_results=3)
        return "\n\n".join(f"**{r['title']}**\n{r['content']}" for r in results["results"])

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    agent = create_react_agent(llm, [web_search])

    with mlflow.start_run(run_name=f"research: {topic[:40]}") as run:
        mlflow.log_param("topic", topic)
        mlflow.log_param("prompt_version", prompt.version)
        result = await agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt.format(query=topic)}]}
        )
        report = result["messages"][-1].content
        mlflow.log_metric("report_length", len(report))

    print(f"[research] {topic[:40]}... -> {len(report)} chars (run {run.info.run_id})")
    await flyte.report.log.aio(f"<h2>{topic}</h2>{_md_to_html(report)}")
    return TopicReport(topic=topic, report=report)


# ── Task 3: judge (synthesize + LLM-as-a-judge) ──────────────────────────────
@agent_env.task(report=True)
async def judge_research(query: str, reports: list[TopicReport]) -> PipelineResult:
    """Synthesize the sub-topic reports, then score the answer with judges.

    Uses all three judge types: built-in LLM judges, a custom LLM judge
    (make_judge), and a custom code judge (@scorer). Logs an MLflow evaluation
    run with per-answer scores and rationales.
    """
    import mlflow
    import mlflow.genai
    from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines, scorer
    from mlflow.genai.judges import make_judge
    from mlflow.entities import Feedback
    from langchain_openai import ChatOpenAI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    # Synthesize one answer from the sub-topic reports
    sections = "\n\n---\n\n".join(f"## {r.topic}\n\n{r.report}" for r in reports)
    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    answer = llm.invoke(
        f"Question: {query}\n\nSub-topic research:\n\n{sections}\n\n"
        f"Write a comprehensive, well-organized answer that synthesizes these findings."
    ).content

    judge_model = f"openai:/{OPENAI_MODEL}"

    conciseness = make_judge(
        name="conciseness",
        instructions="Question: {{ inputs }}\nAnswer: {{ outputs }}\nReturn true if the answer is concise.",
        model=judge_model,
        feedback_value_type=bool,
    )

    @scorer
    def substantive_answer(outputs) -> Feedback:
        ok = len(str(outputs).split()) >= 50
        return Feedback(value=ok, rationale="substantive" if ok else "too thin")

    scorers = [
        RelevanceToQuery(model=judge_model),
        Safety(model=judge_model),
        Guidelines(name="factual_tone", guidelines="Be factual; avoid speculation.", model=judge_model),
        conciseness,
        substantive_answer,
    ]

    result = mlflow.genai.evaluate(
        data=[{"inputs": {"query": query}, "outputs": answer}],
        scorers=scorers,
    )
    metrics = {k: float(v) for k, v in result.metrics.items() if isinstance(v, (int, float))}

    print(f"[judge] metrics={metrics}")
    rows = "".join(f"<tr><td>{k}</td><td>{v:.3f}</td></tr>" for k, v in sorted(metrics.items()))
    await flyte.report.log.aio(
        f"<h2>Judged Answer</h2>{_md_to_html(answer)}"
        f"<h3>Judge scores</h3><table><tr><th>Metric</th><th>Score</th></tr>{rows}</table>"
    )
    return PipelineResult(
        query=query,
        topics=[r.topic for r in reports],
        answer=answer,
        judge_metrics=metrics,
    )


# ── Orchestrator ─────────────────────────────────────────────────────────────
@agent_env.task(report=True)
async def research_pipeline(
    query: str = "What is MLflow and how does it compare to other ML tools?",
    num_topics: int = 3,
) -> PipelineResult:
    """plan → research (fan-out) → judge. Each step is its own Flyte task + MLflow trace."""
    import asyncio

    topics = await plan_topics.aio(query=query, num_topics=num_topics)
    reports = await asyncio.gather(*[research_topic.aio(topic=t) for t in topics])
    result = await judge_research.aio(query=query, reports=list(reports))

    await flyte.report.log.aio(
        f"<h2>Research Pipeline</h2>"
        f"<p><b>Query:</b> {query}</p>"
        f"<p><b>Sub-topics:</b> {', '.join(topics)}</p>"
        f"<hr/>{_md_to_html(result.answer)}"
        f"<p><small>Flyte ran {2 + len(topics)} tasks; MLflow recorded {len(topics)} "
        f"research traces + 1 evaluation run.</small></p>"
    )
    print(f"[pipeline] query={query!r} topics={len(topics)} metrics={result.judge_metrics}")
    return result


@agent_env.task
async def register_judges() -> str:
    """Register judges to the experiment so they appear in the MLflow Judges UI.

    The Judges/Scorers section of an experiment lists *registered* scorers —
    distinct from scorers passed inline to evaluate(). Registering makes a
    judge reusable and visible in the UI.

    OSS MLflow notes:
      - Built-in scorers and make_judge (LLM) judges can be registered.
      - Custom @scorer *code* judges are Databricks-only (arbitrary-code
        execution is blocked on self-hosted servers), so they can't be registered.
      - Automatic/scheduled scoring (scorer.start()) requires an MLflow AI
        Gateway model, not a raw openai:/ model — so we register but don't start.
    """
    import mlflow
    from mlflow.genai.scorers import RelevanceToQuery, Safety, Guidelines, list_scorers
    from mlflow.genai.judges import make_judge

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    eid = mlflow.set_experiment(EXPERIMENT).experiment_id
    judge_model = f"openai:/{OPENAI_MODEL}"

    judges = [
        RelevanceToQuery(model=judge_model),
        Safety(model=judge_model),
        Guidelines(
            name="factual_tone",
            guidelines="The response must be factual and avoid speculation or hedging.",
            model=judge_model,
        ),
        make_judge(
            name="conciseness",
            instructions="Question: {{ inputs }}\nAnswer: {{ outputs }}\nReturn true if the answer is concise.",
            model=judge_model,
            feedback_value_type=bool,
        ),
    ]

    existing = {s.name for s in list_scorers(experiment_id=eid)}
    newly = []
    for j in judges:
        if j.name in existing:
            continue
        j.register(experiment_id=eid)
        newly.append(j.name)

    names = [s.name for s in list_scorers(experiment_id=eid)]
    print(f"[judges] newly_registered={newly} all_registered={names}")
    return ", ".join(names)
