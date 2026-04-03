"""
Research pipeline workflow — LangGraph controls the pipeline, Flyte provides the compute.

This is the full LangGraph + Flyte integration:
- LangGraph manages the pipeline: plan → research → synthesize → quality check → loop
- Flyte runs each researcher as a separate task with its own compute
- The graph decides when to loop based on quality evaluation

Usage:
    # Local
    flyte run --local --tui workflow.py research_pipeline --query "Compare quantum computing approaches"

    # Remote
    flyte run workflow.py research_pipeline --query "Compare quantum computing approaches"
"""

import json
import base64
import logging
import markdown

import flyte
import flyte.report
from langchain_core.messages import HumanMessage
from config import base_env, OPENAI_API_KEY, TAVILY_API_KEY
from graph import build_pipeline_graph, build_research_subgraph

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("graph").setLevel(logging.INFO)
logging.getLogger("tools.search").setLevel(logging.INFO)

env = base_env
MODEL = "gpt-4.1-nano"


def md_to_html(text: str) -> str:
    """Convert markdown to HTML for Flyte reports."""
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ------------------------------------------------------------------
# Flyte task: research a single topic (called by the LangGraph graph)
# ------------------------------------------------------------------
# This is the compute unit. LangGraph's Send dispatches to this task
# for each topic. On a cluster, each runs in its own container.

@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 2) -> str:
    """Run the ReAct research agent on a single sub-topic."""
    log.info(f"[Research Task] Starting: {topic}")

    await flyte.report.replace.aio(f"<h2>Researching: {topic}</h2><p>Running searches...</p>")
    await flyte.report.flush.aio()

    graph = build_research_subgraph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        max_searches=max_searches,
        model=MODEL,
    )
    result = await graph.ainvoke({"messages": [HumanMessage(content=f"Research this topic: {topic}")]})
    report = result["messages"][-1].content
    log.info(f"[Research Task] Done: {topic}")

    await flyte.report.replace.aio(f"<h2>{topic}</h2>{md_to_html(report)}")
    await flyte.report.flush.aio()

    return json.dumps({"topic": topic, "report": report})


# ------------------------------------------------------------------
# Orchestrator: runs the LangGraph pipeline, backed by Flyte tasks
# ------------------------------------------------------------------

@env.task(report=True)
async def research_pipeline(
    query: str,
    num_topics: int = 3,
    max_searches: int = 2,
    max_iterations: int = 2,
) -> str:
    """
    Research pipeline workflow:
    1. LangGraph plans sub-topics
    2. LangGraph fans out research via Send → each dispatches to a Flyte task
    3. LangGraph synthesizes results
    4. LangGraph evaluates quality — if gaps found, loops back to step 2
    5. Repeats until quality is good or max iterations reached
    """
    log.info(f"Starting research pipeline: {query}")

    # Build the pipeline graph, passing the Flyte task as the compute backend
    pipeline = build_pipeline_graph(
        openai_api_key=OPENAI_API_KEY,
        tavily_api_key=TAVILY_API_KEY,
        research_task=research_topic,  # LangGraph dispatches to this Flyte task
        model=MODEL,
    )

    # Visualize the graphs in report tabs
    graph_tab = flyte.report.get_tab("Agent Graphs")

    png_bytes = pipeline.get_graph().draw_mermaid_png()
    img_b64 = base64.b64encode(png_bytes).decode()
    graph_tab.log(f"""\
<h2>Research Pipeline</h2>\
<img src="data:image/png;base64,{img_b64}" alt="Research pipeline" />""")

    subgraph = build_research_subgraph(OPENAI_API_KEY, TAVILY_API_KEY, max_searches, model=MODEL)
    sub_png = subgraph.get_graph().draw_mermaid_png()
    sub_b64 = base64.b64encode(sub_png).decode()
    graph_tab.log(f"""\
<h2>Research Agent (ReAct)</h2>\
<img src="data:image/png;base64,{sub_b64}" alt="ReAct research agent" />""")
    await flyte.report.flush.aio()

    # Run the pipeline — LangGraph controls the flow, Flyte runs the compute
    # All fields must be provided because Flyte wraps TypedDicts with pydantic validation.
    # The graph nodes populate the real values during execution.
    result = await pipeline.ainvoke({
        "query": query,
        "num_topics": num_topics,
        "max_searches": max_searches,
        "max_iterations": max_iterations,
        "iteration": 0,
        "topics": [],
        "research_results": [],
        "synthesis": "",
        "score": 0,
        "gaps": [],
        "final_report": "",
    })

    # Build the report
    final_report = result["final_report"]
    sub_reports = result["research_results"]
    score = result.get("score", "N/A")
    iteration = result.get("iteration", 1) - 1  # -1 because it increments after last check

    await flyte.report.replace.aio(f"""\
<h2>Research Report</h2>\
<p><b>Query:</b> {query}</p>\
<p><b>Quality:</b> {score}/10 after {iteration} iteration(s)</p>\
<hr/>{md_to_html(final_report)}""")
    await flyte.report.flush.aio()

    log.info(f"Research pipeline complete. Score: {score}/10, Iterations: {iteration}")
    return json.dumps({
        "query": query,
        "report": final_report,
        "sub_reports": sub_reports,
        "score": score,
        "iterations": iteration,
    })
