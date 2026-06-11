"""Flyte task: run the ReAct agent and trace it into the hosted Phoenix server.

This is the whole client side of the observability story:
  1. register() points the OpenTelemetry exporter at the self-hosted Phoenix
     collector (PHOENIX_COLLECTOR_ENDPOINT, set on the TaskEnvironment).
  2. LangChainInstrumentor patches LangChain/LangGraph so every LLM call, tool
     call, and graph node emits an OTLP span.
  3. The agent runs; spans stream to Phoenix over OTLP-HTTP.
  4. force_flush() guarantees the batch exporter drains before the pod exits
     (Flyte pods are short-lived, so an un-flushed batch would be lost).

Open the Phoenix app UI afterward to inspect the trace span by span.

Usage:
    # Remote (the real path: the task runs in-cluster and can reach Phoenix)
    flyte run workflow.py research --question "What is Flyte 2 and who makes it?"

    # Switch the agent LLM to the in-cluster gemma4 vLLM app
    flyte run workflow.py research --question "..." --provider vllm
"""

import json
import logging

import flyte
from langchain_core.messages import HumanMessage

from agent import build_agent
from config import (
    agent_env,
    LLM_PROVIDER,
    PHOENIX_COLLECTOR_ENDPOINT,
    PHOENIX_PROJECT_NAME,
    TAVILY_API_KEY,
)

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("agent").setLevel(logging.INFO)
logging.getLogger("tools.search").setLevel(logging.INFO)

env = agent_env


def _setup_tracing(project_name: str):
    """Wire the OpenInference LangChain instrumentor to the hosted Phoenix collector."""
    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # endpoint is the base collector URL; register() appends /v1/traces for the
    # OTLP-HTTP exporter. Reads the same value from PHOENIX_COLLECTOR_ENDPOINT if
    # omitted, but we pass it explicitly so the wiring is obvious on stream.
    tracer_provider = register(
        endpoint=PHOENIX_COLLECTOR_ENDPOINT + "/v1/traces",
        project_name=project_name,
        batch=True,           # background batch exporter (the production default)
    )
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)
    log.info(f"[tracing] exporting to {PHOENIX_COLLECTOR_ENDPOINT} (project={project_name})")
    return tracer_provider


@env.task
async def research(question: str, max_searches: int = 3, provider: str = LLM_PROVIDER) -> str:
    """Run the ReAct agent on a question, tracing every step into Phoenix.

    provider: "openai" (default) or "vllm" (the in-cluster gemma4 app).
    """
    tracer_provider = _setup_tracing(PHOENIX_PROJECT_NAME)

    log.info(f"[research] provider={provider} question={question!r}")
    agent = build_agent(TAVILY_API_KEY, provider=provider, max_searches=max_searches)

    try:
        result = await agent.ainvoke(
            {"messages": [HumanMessage(content=question)]}
        )
        answer = result["messages"][-1].content
    finally:
        # Drain the batch exporter before the pod tears down, or the trailing
        # spans never reach Phoenix.
        tracer_provider.force_flush()

    log.info(f"[research] done ({len(answer)} chars)")
    return json.dumps({"question": question, "answer": answer, "provider": provider})


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(research, question="What is Flyte 2 and who makes it?")
    print(run.url)
