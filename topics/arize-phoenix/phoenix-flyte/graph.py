"""
Research agent pipeline with Flyte-backed compute.

LangGraph controls the pipeline logic: planning, routing, quality gates, looping.
Flyte provides the compute: each step runs as a separate task with its own resources.

The pipeline graph:

    START → plan → research (fan-out via Send → Flyte tasks) → synthesize
                                                                    │
                                                              quality_check
                                                              ╱           ╲
                                                    gaps found?         good enough
                                                        │                   │
                                                  identify_gaps            END
                                                        │
                                                   research (again, with new topics)
                                                        │
                                                   synthesize → quality_check → ...

The ReAct research subgraph (runs inside each research_topic Flyte task):

    agent → (tool calls?) → tools → agent → ... → END

Both the subgraph nodes and the web_search tool are `@flyte.trace`-d (Flyte's own
tracing) AND captured by Phoenix via the LangChain instrumentor, so the same run
is visible in both tools.
"""

import logging
import operator
from typing import Annotated, TypedDict

import flyte
from langchain_core.messages import SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from llm import build_llm
from tools.search import create_search_tool

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# ReAct research subgraph (runs inside research_topic Flyte tasks)
# ------------------------------------------------------------------

def build_research_subgraph(
    tavily_api_key: str,
    provider: str = "openai",
    max_searches: int = 3,
):
    """Build a ReAct research agent that uses Tavily search."""
    web_search = create_search_tool(tavily_api_key)
    tools = [web_search]
    llm = build_llm(provider).bind_tools(tools)

    system_prompt = f"""\
You are a research agent. Your job is to thoroughly research a topic by searching the web. \
Use the web_search tool up to {max_searches} times to gather information from different angles. \
After gathering enough information, write a clear research summary with key findings and sources."""

    @flyte.trace
    async def agent(state: MessagesState) -> MessagesState:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = llm.invoke(messages)

        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                log.info(f"[Research] Tool call: {tc['name']}({tc['args']})")
        elif response.content:
            log.info(f"[Research] Response: {response.content[:200]}")

        return {"messages": [response]}

    @flyte.trace
    async def should_continue(state: MessagesState) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "tools"
        return "__end__"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", agent)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue, {
        "tools": "tools",
        "__end__": "__end__",
    })
    graph.add_edge("tools", "agent")

    return graph.compile()


# ------------------------------------------------------------------
# Research pipeline graph — nodes dispatch to Flyte tasks
# ------------------------------------------------------------------

def build_pipeline_graph(
    plan_task,
    research_task,
    synthesize_task,
    quality_check_task,
    provider: str = "openai",
):
    """
    Build the research pipeline graph.

    Each node dispatches to a Flyte task, making every step visible
    in the Flyte UI with its own compute, reports, and logs.

    Args:
        plan_task: Flyte task (query, num_topics, provider) → list[str]
        research_task: Flyte task (topic, max_searches, provider) → TopicReport
        synthesize_task: Flyte task (query, results, provider) → str
        quality_check_task: Flyte task (query, synthesis, provider) → QualityResult
        provider: LLM provider threaded into each task ("openai" or "vllm").
    """
    # Import here to avoid circular imports
    from models import TopicReport

    # State definition — kept inside the function so Flyte doesn't wrap it
    class PipelineState(TypedDict, total=False):
        query: str
        num_topics: int
        max_searches: int
        iteration: int
        max_iterations: int
        topics: list[str]
        research_results: Annotated[list[dict], operator.add]
        synthesis: str
        score: int
        gaps: list[str]
        final_report: str

    # -- Plan node → dispatches to plan_topics Flyte task ---------------
    async def plan(state: PipelineState) -> dict:
        """Split the query into focused sub-topics via Flyte task."""
        query = state["query"]
        num_topics = state.get("num_topics", 3)

        topics = await plan_task(query, num_topics, provider)
        log.info(f"[Plan] {len(topics)} sub-topics: {topics}")
        return {"topics": topics, "iteration": 1}

    # -- Fan-out to research --------------------------------------------
    def route_to_research(state: PipelineState) -> list[Send]:
        """Create a Send for each topic — each dispatches to a Flyte task."""
        topics = state.get("gaps") or state["topics"]
        max_searches = state.get("max_searches", 2)
        return [
            Send("research", {"topic": t, "max_searches": max_searches})
            for t in topics
        ]

    # -- Research node → dispatches to research_topic Flyte task --------
    async def research(state: dict) -> dict:
        """Run research on a single topic via a Flyte task."""
        topic = state["topic"]
        max_searches = state.get("max_searches", 2)
        log.info(f"[Research] Dispatching to Flyte task: {topic}")

        result = await research_task(topic, max_searches, provider)
        log.info(f"[Research] Flyte task complete: {topic}")

        return {"research_results": [{"topic": result.topic, "report": result.report}]}

    # -- Synthesize node → dispatches to synthesize Flyte task ----------
    async def synthesize_node(state: PipelineState) -> dict:
        """Combine all research results via Flyte task."""
        query = state["query"]
        results = [TopicReport(**r) for r in state["research_results"]]
        iteration = state.get("iteration", 1)

        synthesis = await synthesize_task.override(
            short_name=f"synthesize-{iteration}"
        )(query, results, provider)

        log.info(f"[Synthesize] Combined {len(results)} reports (iteration {iteration})")
        return {"synthesis": synthesis}

    # -- Quality check node → dispatches to quality_check Flyte task ----
    async def quality_check_node(state: PipelineState) -> dict:
        """Evaluate the synthesis via Flyte task."""
        query = state["query"]
        synthesis = state["synthesis"]
        iteration = state.get("iteration", 1)
        max_iterations = state.get("max_iterations", 2)

        result = await quality_check_task.override(
            short_name=f"quality-{iteration}"
        )(query, synthesis, provider)

        score = result.score
        gaps = result.gaps

        # Don't loop forever
        if iteration >= max_iterations:
            gaps = []
            log.info(f"[Quality] Max iterations reached ({max_iterations}), finishing")

        log.info(f"[Quality] Score: {score}/10, Gaps: {len(gaps)} (iteration {iteration})")
        return {"score": score, "gaps": gaps, "iteration": iteration + 1}

    # -- Routing after quality check ------------------------------------
    def after_quality_check(state: PipelineState) -> str:
        """If gaps found, research more. Otherwise, finalize."""
        if state.get("gaps"):
            log.info(f"[Quality] Gaps found, researching further: {state['gaps']}")
            return "research_more"
        return "finalize"

    # -- Identify gaps node (triggers Send fan-out on gaps) -------------
    async def identify_gaps(state: PipelineState) -> dict:
        """Pass-through node to trigger research fan-out on gaps."""
        return {}

    # -- Finalize node --------------------------------------------------
    async def finalize(state: PipelineState) -> dict:
        """Set the final report."""
        return {"final_report": state["synthesis"]}

    # -- Wire the graph -------------------------------------------------
    graph = StateGraph(PipelineState)
    graph.add_node("plan", plan)
    graph.add_node("research", research)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("identify_gaps", identify_gaps)
    graph.add_node("finalize", finalize)

    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", route_to_research, ["research"])
    graph.add_edge("research", "synthesize")
    graph.add_edge("synthesize", "quality_check")
    graph.add_conditional_edges("quality_check", after_quality_check, {
        "research_more": "identify_gaps",
        "finalize": "finalize",
    })
    graph.add_conditional_edges("identify_gaps", route_to_research, ["research"])
    graph.add_edge("finalize", END)

    return graph.compile()
