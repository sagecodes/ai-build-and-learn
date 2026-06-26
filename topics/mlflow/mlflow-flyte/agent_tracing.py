"""LLM agent tracing + evaluation with MLflow + Flyte.

Two tasks:
  - traced_research: trace a LangGraph research agent (every LLM call, tool
    use, and graph step) with MLflow autologging. The agent's system prompt
    is pulled from the MLflow Prompt Registry so runs link to a prompt version.
  - evaluate_agent: score answers with MLflow's LLM-as-a-judge scorers —
    built-in (Correctness, RelevanceToQuery, Guidelines, Safety) plus a
    custom judge built with make_judge — via mlflow.genai.evaluate.

Run remote:
    flyte run agent_tracing.py traced_research --query "What is MLflow?"
    flyte run agent_tracing.py evaluate_agent
"""

from __future__ import annotations

from dataclasses import dataclass

import flyte

from config import agent_env, MLFLOW_TRACKING_URI, OPENAI_MODEL

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
class AgentResult:
    query: str
    answer: str
    run_id: str


@agent_env.task(report=True)
async def traced_research(query: str = "What is MLflow and how does it compare to other ML tools?") -> AgentResult:
    """Run a research agent with MLflow tracing + a registry-managed prompt."""
    import mlflow
    import mlflow.langchain
    from langchain_openai import ChatOpenAI
    from langgraph.prebuilt import create_react_agent
    from langchain_core.tools import tool
    from tavily import TavilyClient

    # Set up MLflow tracking + LangChain autologging
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("llm-agent-tracing")
    mlflow.langchain.autolog()

    prompt = _load_or_register_prompt()

    tavily = TavilyClient()

    @tool
    def web_search(query: str) -> str:
        """Search the web for information."""
        results = tavily.search(query=query, max_results=3)
        return "\n\n".join(
            f"**{r['title']}**\n{r['content']}" for r in results["results"]
        )

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)
    agent = create_react_agent(llm, [web_search])

    with mlflow.start_run(run_name=f"research: {query[:50]}") as run:
        mlflow.log_param("query", query)
        mlflow.log_param("model", OPENAI_MODEL)
        mlflow.log_param("prompt_name", PROMPT_NAME)
        mlflow.log_param("prompt_version", prompt.version)

        # Render the registry prompt with the runtime query
        user_content = prompt.format(query=query)
        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": user_content}]
        })

        answer = result["messages"][-1].content

        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_text(answer, "answer.md")

        # Flyte report
        import markdown
        report = f"""
        <h2>Research: {query}</h2>
        <div>{markdown.markdown(answer)}</div>
        <p><small>MLflow Run ID: {run.info.run_id} | prompt: {PROMPT_NAME} v{prompt.version}</small></p>
        """
        await flyte.report.log.aio(report)

        print(f"[agent] query={query[:50]}... answer_length={len(answer)} prompt_v={prompt.version}")

        return AgentResult(query=query, answer=answer, run_id=run.info.run_id)


@agent_env.task(report=True)
async def evaluate_agent() -> str:
    """Grade LLM answers with MLflow's LLM-as-a-judge scorers.

    Runs mlflow.genai.evaluate over a small eval set using built-in judges
    plus one custom judge (make_judge). Scores show up per-row in the MLflow
    Evaluations UI with the judge's rationale.
    """
    import json

    import mlflow
    import mlflow.genai
    from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines, Safety, scorer
    from mlflow.genai.judges import make_judge
    from mlflow.entities import Feedback
    from langchain_openai import ChatOpenAI

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("llm-agent-tracing")

    # In OSS MLflow each judge needs an explicit model (no managed endpoint).
    judge_model = f"openai:/{OPENAI_MODEL}"

    # Eval set: questions + ground-truth facts the answer should contain.
    eval_data = [
        {
            "inputs": {"query": "What is MLflow?"},
            "expectations": {"expected_facts": [
                "open-source platform", "machine learning lifecycle", "experiment tracking",
            ]},
        },
        {
            "inputs": {"query": "Which company originally created MLflow?"},
            "expectations": {"expected_facts": ["Databricks"]},
        },
        {
            "inputs": {"query": "Name two things MLflow can track for an ML experiment."},
            "expectations": {"expected_facts": ["parameters", "metrics"]},
        },
    ]

    llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0)

    def predict_fn(query: str) -> str:
        """The system under test — answers get judged. Auto-traced by evaluate."""
        return llm.invoke(query).content

    # Custom judge: a domain rubric expressed in natural language.
    conciseness_judge = make_judge(
        name="conciseness",
        instructions=(
            "You are grading an assistant answer for conciseness.\n"
            "Question: {{ inputs }}\nAnswer: {{ outputs }}\n"
            "Return true if the answer is focused and free of filler, "
            "otherwise false. Explain briefly."
        ),
        model=judge_model,
        feedback_value_type=bool,  # boolean → MLflow rolls up a pass-rate metric
    )

    # Custom CODE judge: plain Python, deterministic, no LLM call (fast + free).
    # Return a Feedback object; any param (inputs/outputs/expectations) is optional.
    @scorer
    def substantive_answer(outputs) -> Feedback:
        n_words = len(str(outputs).split())
        ok = n_words >= 20
        return Feedback(
            value=ok,
            rationale=f"{n_words} words " + ("(>= 20, substantive)" if ok else "(< 20, too thin)"),
        )

    scorers = [
        Correctness(model=judge_model),          # built-in LLM judge: supported by expected_facts
        RelevanceToQuery(model=judge_model),      # built-in LLM judge: addresses the question
        Guidelines(                               # built-in LLM judge with a custom rule
            name="factual_tone",
            guidelines="The response must be factual and avoid speculation or hedging.",
            model=judge_model,
        ),
        Safety(model=judge_model),                # built-in LLM judge: no harmful content
        conciseness_judge,                        # custom LLM judge (make_judge)
        substantive_answer,                       # custom code judge (@scorer, no LLM)
    ]

    results = mlflow.genai.evaluate(
        data=eval_data,
        scorers=scorers,
        predict_fn=predict_fn,
    )

    metrics = results.metrics

    rows = "".join(
        f"<tr><td>{k}</td><td>{v:.3f}</td></tr>"
        for k, v in sorted(metrics.items()) if isinstance(v, (int, float))
    )
    report = f"""
    <h2>LLM-as-a-Judge Evaluation</h2>
    <p>Built-in LLM judges: Correctness, RelevanceToQuery, Guidelines(factual_tone), Safety.
    Custom LLM judge: conciseness (make_judge). Custom code judge: substantive_answer (@scorer).</p>
    <table><tr><th>Metric</th><th>Score</th></tr>{rows}</table>
    """
    await flyte.report.log.aio(report)

    summary = json.dumps(
        {k: v for k, v in metrics.items() if isinstance(v, (int, float))}, indent=2
    )
    print(f"[eval] judge metrics:\n{summary}")
    return summary


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
    eid = mlflow.set_experiment("llm-agent-tracing").experiment_id
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
