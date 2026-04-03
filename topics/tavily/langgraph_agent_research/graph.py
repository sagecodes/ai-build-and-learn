"""
Research agent pipeline with Flyte-backed compute.

This is the real LangGraph + Flyte integration story:
- LangGraph controls the pipeline logic: planning, routing, quality gates, looping
- Flyte provides the compute: each researcher runs as a separate task with its own resources

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

The ReAct research subgraph (runs inside each Flyte task):

    agent → (tool calls?) → tools → agent → ... → END
"""

import json
import operator
import logging
from typing import Annotated, TypedDict

import flyte
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Send
from tools.search import create_search_tool

log = logging.getLogger(__name__)


# ------------------------------------------------------------------
# ReAct research subgraph (runs inside Flyte tasks)
# ------------------------------------------------------------------

def build_research_subgraph(
    openai_api_key: str,
    tavily_api_key: str,
    max_searches: int = 3,
    model: str = "gpt-4.1-nano",
):
    """Build a ReAct research agent that uses Tavily search."""
    web_search = create_search_tool(tavily_api_key)
    tools = [web_search]
    llm = ChatOpenAI(model=model, api_key=openai_api_key).bind_tools(tools)

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
# Research pipeline graph
# ------------------------------------------------------------------

def build_pipeline_graph(
    openai_api_key: str,
    tavily_api_key: str,
    research_task,
    model: str = "gpt-4.1-nano",
):
    """
    Build the research pipeline graph.

    Args:
        research_task: A Flyte task function that takes (topic, max_searches)
                       and returns a JSON string with {topic, report}.
                       This is how LangGraph dispatches to Flyte compute.
    """
    llm = ChatOpenAI(model=model, api_key=openai_api_key)

    # Define state inside the function so Flyte doesn't wrap it with pydantic
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

    # -- Plan node ----------------------------------------------------------
    @flyte.trace
    async def plan(state: PipelineState) -> dict:
        """Split the query into focused sub-topics."""
        query = state["query"]
        num_topics = state.get("num_topics", 3)

        response = llm.invoke(f"""\
Break this research question into exactly {num_topics} focused sub-topics. \
Return ONLY a JSON array of strings, nothing else.

Question: {query}""")
        try:
            topics = json.loads(response.content)
        except json.JSONDecodeError:
            topics = [query]

        topics = topics[:num_topics]
        log.info(f"[Plan] {len(topics)} sub-topics: {topics}")
        return {"topics": topics, "iteration": 1}

    # -- Fan-out to research ------------------------------------------------
    def route_to_research(state: PipelineState) -> list[Send]:
        """Create a Send for each topic — each dispatches to a Flyte task."""
        topics = state.get("gaps") or state["topics"]
        max_searches = state.get("max_searches", 2)
        return [
            Send("research", {"topic": t, "max_searches": max_searches})
            for t in topics
        ]

    # -- Research node (dispatches to Flyte task) ---------------------------
    async def research(state: dict) -> dict:
        """
        Run research on a single topic via a Flyte task.

        This is the key integration point: LangGraph controls the routing,
        Flyte provides the compute. Each topic runs as a separate container.
        """
        topic = state["topic"]
        max_searches = state.get("max_searches", 2)
        log.info(f"[Research] Dispatching to Flyte task: {topic}")

        result_json = await research_task(topic, max_searches)
        result = json.loads(result_json)
        log.info(f"[Research] Flyte task complete: {topic}")

        return {"research_results": [result]}

    # -- Synthesize node ----------------------------------------------------
    @flyte.trace
    async def synthesize(state: PipelineState) -> dict:
        """Combine all research results into a report."""
        query = state["query"]
        results = state["research_results"]
        iteration = state.get("iteration", 1)

        sections = "\n\n---\n\n".join(
            f"## {r['topic']}\n\n{r['report']}" for r in results
        )

        response = llm.invoke(f"""\
You have research reports on sub-topics of this question:

{query}

Sub-topic reports:

{sections}

Write a comprehensive report that synthesizes all findings. \
Organize by theme, highlight connections between sub-topics, \
and end with key takeaways.""")
        log.info(f"[Synthesize] Combined {len(results)} reports (iteration {iteration})")
        return {"synthesis": response.content}

    # -- Quality check node -------------------------------------------------
    @flyte.trace
    async def quality_check(state: PipelineState) -> dict:
        """Evaluate the synthesis and identify any gaps."""
        query = state["query"]
        synthesis = state["synthesis"]
        iteration = state.get("iteration", 1)
        max_iterations = state.get("max_iterations", 2)

        response = llm.invoke(f"""\
Evaluate this research report for the question: {query}

Report:
{synthesis}

Rate the report quality from 1-10 and identify any gaps or missing perspectives. \
Return JSON: {{"score": <int>, "gaps": [<string>, ...]}}
If the report is comprehensive (score >= 8) or there are no significant gaps, \
return an empty gaps list.""")

        try:
            evaluation = json.loads(response.content)
            score = evaluation.get("score", 8)
            gaps = evaluation.get("gaps", [])
        except json.JSONDecodeError:
            score = 8
            gaps = []

        # Don't loop forever
        if iteration >= max_iterations:
            gaps = []
            log.info(f"[Quality] Max iterations reached ({max_iterations}), finishing")

        log.info(f"[Quality] Score: {score}/10, Gaps: {len(gaps)} (iteration {iteration})")
        return {"score": score, "gaps": gaps, "iteration": iteration + 1}

    # -- Routing after quality check ----------------------------------------
    def after_quality_check(state: PipelineState) -> str:
        """If gaps found, research more. Otherwise, finalize."""
        if state.get("gaps"):
            log.info(f"[Quality] Gaps found, researching further: {state['gaps']}")
            return "research_more"
        return "finalize"

    # -- Identify gaps node (triggers Send fan-out on gaps) ------------------
    async def identify_gaps(state: PipelineState) -> dict:
        """Pass-through node to trigger research fan-out on gaps."""
        return {}

    # -- Finalize node ------------------------------------------------------
    @flyte.trace
    async def finalize(state: PipelineState) -> dict:
        """Set the final report."""
        return {"final_report": state["synthesis"]}

    # -- Wire the graph -----------------------------------------------------
    #
    #   START → plan ──Send──→ research → synthesize → quality_check
    #                              ▲                       │
    #                              │                 gaps? │ no gaps
    #                     identify_gaps ◄───────────────────┤
    #                     (Send fan-out)                    ▼
    #                                                   finalize → END
    #
    graph = StateGraph(PipelineState)
    graph.add_node("plan", plan)
    graph.add_node("research", research)
    graph.add_node("synthesize", synthesize)
    graph.add_node("quality_check", quality_check)
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
