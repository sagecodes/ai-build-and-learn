"""
Research pipeline — LangGraph controls the flow, Flyte provides the compute,
Phoenix traces the LLM/tool calls.

Each pipeline step is a separate Flyte task, visible in the Flyte UI with its own
compute, report, logs, and `@flyte.trace` spans:
- plan_topics: break the query into sub-topics
- research_topic: a ReAct agent researches one sub-topic (parallel via Send)
- synthesize: combine the sub-topic reports
- quality_check: score the report and find gaps (loops back if gaps remain)

Every task also instruments LangChain with Phoenix (tracing.setup_tracing), so
the same run shows up in two places: Flyte stitches the whole DAG together across
tasks; Phoenix gives the deep per-task LLM/tool spans (prompts, tokens, latency).
That side-by-side is the point of this demo.

Usage:
    flyte run workflow.py research_pipeline --query "Compare quantum computing approaches"
    flyte run workflow.py research_pipeline --query "..." --provider vllm
    flyte run workflow.py research_pipeline --query "..." --num_topics 2 --max_iterations 1
"""

import json
import base64
import logging

import flyte
import flyte.report

from config import agent_env, LLM_PROVIDER
from models import TopicReport, QualityResult, PipelineResult
from graph import build_pipeline_graph, build_research_subgraph
from llm import build_llm
from tracing import setup_tracing, flush, get_tracer

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.getLogger("graph").setLevel(logging.INFO)
logging.getLogger("tools.search").setLevel(logging.INFO)
logging.getLogger("tracing").setLevel(logging.INFO)

env = agent_env


def md_to_html(text: str) -> str:
    """Convert markdown to HTML for Flyte reports."""
    import markdown
    return markdown.markdown(text, extensions=["tables", "fenced_code"])


# ------------------------------------------------------------------
# Flyte tasks — each step is visible in the UI while running, and each
# instruments LangChain so its LLM/tool calls land in Phoenix.
# ------------------------------------------------------------------

@env.task(report=True)
async def plan_topics(query: str, num_topics: int = 3, provider: str = LLM_PROVIDER) -> list[str]:
    """Break a research query into focused sub-topics."""
    setup_tracing()
    log.info(f"Planning {num_topics} sub-topics for: {query}")

    await flyte.report.replace.aio(
        f"<h2>Planning</h2><p>Breaking query into {num_topics} sub-topics...</p>"
    )
    await flyte.report.flush.aio()

    llm = build_llm(provider)
    try:
        response = llm.invoke(
            f"Break this research question into exactly {num_topics} focused sub-topics. "
            f"Return ONLY a JSON array of strings, nothing else.\n\nQuestion: {query}"
        )
        try:
            topics = json.loads(response.content)
        except json.JSONDecodeError:
            topics = [query]
    finally:
        flush()

    topics = topics[:num_topics]
    log.info(f"Sub-topics: {topics}")

    topic_html = "".join(f"<li>{t}</li>" for t in topics)
    await flyte.report.replace.aio(
        f"<h2>Planning</h2><p>Sub-topics:</p><ul>{topic_html}</ul>"
    )
    await flyte.report.flush.aio()

    return topics


@env.task(report=True)
async def research_topic(topic: str, max_searches: int = 2, provider: str = LLM_PROVIDER) -> TopicReport:
    """Run the ReAct research agent on a single sub-topic."""
    import os

    setup_tracing()
    log.info(f"[Research Task] Starting: {topic}")

    tavily_api_key = os.getenv("TAVILY_API_KEY")

    await flyte.report.replace.aio(f"<h2>Researching: {topic}</h2><p>Running searches...</p>")
    await flyte.report.flush.aio()

    graph = build_research_subgraph(
        tavily_api_key=tavily_api_key,
        provider=provider,
        max_searches=max_searches,
    )
    try:
        result = await graph.ainvoke(
            {"messages": [{"role": "user", "content": f"Research this topic: {topic}"}]}
        )
        report = result["messages"][-1].content
    finally:
        flush()
    log.info(f"[Research Task] Done: {topic}")

    await flyte.report.replace.aio(f"<h2>{topic}</h2>{md_to_html(report)}")
    await flyte.report.flush.aio()

    return TopicReport(topic=topic, report=report)


@env.task(report=True)
async def synthesize(query: str, results: list[TopicReport], provider: str = LLM_PROVIDER) -> str:
    """Combine sub-topic research reports into a unified synthesis."""
    setup_tracing()
    log.info(f"Synthesizing {len(results)} report(s)...")

    await flyte.report.replace.aio(
        f"<h2>Synthesis</h2><p>Combining {len(results)} reports...</p>"
    )
    await flyte.report.flush.aio()

    sections = "\n\n---\n\n".join(f"## {r.topic}\n\n{r.report}" for r in results)

    llm = build_llm(provider)
    try:
        response = llm.invoke(
            f"You have research reports on sub-topics of this question:\n\n{query}\n\n"
            f"Sub-topic reports:\n\n{sections}\n\n"
            f"Write a comprehensive report that synthesizes all findings. "
            f"Organize by theme, highlight connections between sub-topics, "
            f"and end with key takeaways."
        )
        synthesis = response.content
    finally:
        flush()
    log.info(f"Synthesis complete: {len(synthesis)} chars")

    await flyte.report.replace.aio(f"<h2>Synthesis</h2>{md_to_html(synthesis)}")
    await flyte.report.flush.aio()

    return synthesis


@env.task(report=True)
async def quality_check(query: str, synthesis: str, provider: str = LLM_PROVIDER) -> QualityResult:
    """Evaluate report quality and identify gaps."""
    setup_tracing()
    log.info("Evaluating quality...")

    await flyte.report.replace.aio(
        "<h2>Quality Check</h2><p>Evaluating report quality...</p>"
    )
    await flyte.report.flush.aio()

    llm = build_llm(provider)
    try:
        response = llm.invoke(
            f'Evaluate this research report for the question: {query}\n\n'
            f'Report:\n{synthesis}\n\n'
            f'Rate the report quality from 1-10 and identify any gaps or missing perspectives. '
            f'Return JSON: {{"score": <int>, "gaps": [<string>, ...]}}\n'
            f'If the report is comprehensive (score >= 8) or there are no significant gaps, '
            f'return an empty gaps list.'
        )
        try:
            evaluation = json.loads(response.content)
            score = evaluation.get("score", 8)
            gaps = evaluation.get("gaps", [])
        except json.JSONDecodeError:
            score = 8
            gaps = []
    finally:
        flush()

    result = QualityResult(score=score, gaps=gaps)
    log.info(f"Score: {result.score}/10, Gaps: {len(result.gaps)}")

    gap_html = "".join(f"<li>{g}</li>" for g in result.gaps) if result.gaps else "<li>None</li>"
    await flyte.report.replace.aio(
        f"<h2>Quality Check</h2>"
        f"<p><b>Score:</b> {result.score}/10</p>"
        f"<p><b>Gaps:</b></p><ul>{gap_html}</ul>"
    )
    await flyte.report.flush.aio()

    return result


# ------------------------------------------------------------------
# Orchestrator: runs the LangGraph pipeline, backed by Flyte tasks
# ------------------------------------------------------------------

@env.task(report=True)
async def research_pipeline(
    query: str,
    num_topics: int = 3,
    max_searches: int = 2,
    max_iterations: int = 2,
    provider: str = LLM_PROVIDER,
) -> PipelineResult:
    """
    Research pipeline workflow:
    1. plan_topics breaks the query into sub-topics
    2. research_topic fans out via Send → one ReAct agent per sub-topic
    3. synthesize combines the results
    4. quality_check scores the report; if gaps remain, loop back to step 2
    """
    import os

    log.info(f"Starting research pipeline: {query} (provider={provider})")
    tavily_api_key = os.getenv("TAVILY_API_KEY")

    # Build the pipeline graph, passing all Flyte tasks as compute backends
    pipeline = build_pipeline_graph(
        plan_task=plan_topics,
        research_task=research_topic,
        synthesize_task=synthesize,
        quality_check_task=quality_check,
        provider=provider,
    )

    # Visualize the graphs in a report tab. Mermaid rendering hits a remote
    # service, so keep it best-effort: a failure here must not fail the run.
    try:
        graph_tab = flyte.report.get_tab("Agent Graphs")
        png_bytes = pipeline.get_graph().draw_mermaid_png()
        img_b64 = base64.b64encode(png_bytes).decode()
        graph_tab.log(
            f'<h2>Research Pipeline</h2>'
            f'<img src="data:image/png;base64,{img_b64}" alt="Research pipeline" />'
        )
        subgraph = build_research_subgraph(tavily_api_key, provider, max_searches)
        sub_png = subgraph.get_graph().draw_mermaid_png()
        sub_b64 = base64.b64encode(sub_png).decode()
        graph_tab.log(
            f'<h2>Research Agent (ReAct)</h2>'
            f'<img src="data:image/png;base64,{sub_b64}" alt="ReAct research agent" />'
        )
        await flyte.report.flush.aio()
    except Exception as e:
        log.warning(f"Graph visualization skipped: {type(e).__name__}: {e}")

    # Run the pipeline — LangGraph controls the flow, Flyte tasks run the compute
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

    # Build the final report
    final_report = result["final_report"]
    sub_reports = [TopicReport(**r) for r in result["research_results"]]
    score = result.get("score", 0)
    iteration = result.get("iteration", 1) - 1

    # Emit a clean span for the full report (query in, final report out) so
    # Phoenix can judge the whole report as one unit, alongside the per-topic
    # research-answer spans. Best-effort: never fail the run over telemetry.
    try:
        tracer = get_tracer()
        with tracer.start_as_current_span("research_report") as span:
            span.set_attribute("openinference.span.kind", "CHAIN")
            span.set_attribute("input.mime_type", "text/plain")
            span.set_attribute("input.value", query)
            span.set_attribute("output.mime_type", "text/plain")
            span.set_attribute("output.value", final_report)
        flush()
    except Exception as e:
        log.warning(f"final-report span skipped: {type(e).__name__}: {e}")

    await flyte.report.replace.aio(
        f'<h2>Research Report</h2>'
        f'<p><b>Query:</b> {query}</p>'
        f'<p><b>Quality:</b> {score}/10 after {iteration} iteration(s)</p>'
        f'<hr/>{md_to_html(final_report)}'
    )
    await flyte.report.flush.aio()

    log.info(f"Research pipeline complete. Score: {score}/10, Iterations: {iteration}")
    return PipelineResult(
        query=query,
        report=final_report,
        sub_reports=sub_reports,
        score=score,
        iterations=iteration,
    )


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(research_pipeline, query="What is OpenTelemetry and what problem does it solve?")
    print(run.url)
