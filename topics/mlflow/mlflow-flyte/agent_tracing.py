"""LLM agent tracing with MLflow + Flyte.

Traces a LangGraph research agent with MLflow's LangChain autologging.
Every LLM call, tool use, and graph step is captured and visible in
the MLflow UI.

Run remote:
    flyte run agent_tracing.py traced_research --query "What is MLflow?"

Run local:
    flyte run --local agent_tracing.py traced_research --query "What is MLflow?"
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import flyte

from config import agent_env, MLFLOW_TRACKING_URI, OPENAI_MODEL


@dataclass
class AgentResult:
    query: str
    answer: str
    run_id: str


@agent_env.task(report=True)
async def traced_research(query: str = "What is MLflow and how does it compare to other ML tools?") -> AgentResult:
    """Run a research agent with MLflow tracing enabled."""
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

        result = await agent.ainvoke({
            "messages": [{"role": "user", "content": f"Research this topic and provide a comprehensive answer: {query}"}]
        })

        answer = result["messages"][-1].content

        mlflow.log_metric("answer_length", len(answer))
        mlflow.log_text(answer, "answer.md")

        # Flyte report
        import markdown
        report = f"""
        <h2>Research: {query}</h2>
        <div>{markdown.markdown(answer)}</div>
        <p><small>MLflow Run ID: {run.info.run_id}</small></p>
        """
        await flyte.report.log.aio(report)

        print(f"[agent] query={query[:50]}... answer_length={len(answer)}")

        return AgentResult(query=query, answer=answer, run_id=run.info.run_id)
