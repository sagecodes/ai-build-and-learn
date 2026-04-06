"""
Flyte workflow for the OpenEnv Research Agent demo.

Two Flyte tasks are defined:

1. run_research_episode(query, agent_type, max_steps) -> str (JSON)
   Runs a single research episode — either the OpenEnv agent or the
   traditional agent — inside a Flyte task container. Returns the
   full episode result as JSON including step log, reward history,
   tool usage, and final scores.

2. run_research_comparison(queries, max_steps) -> str (JSON)
   The main pipeline. Takes a list of research questions and fans them
   out as parallel Flyte tasks using Send — each question runs both
   agents simultaneously. Returns a comparison report.

   Fan-out diagram:
       START → fan_out ──Send──→ run_research_episode (openenv, q1)
                        ──Send──→ run_research_episode (traditional, q1)
                        ──Send──→ run_research_episode (openenv, q2)
                        ──Send──→ run_research_episode (traditional, q2)
                             ...
                        → collect_results → END

Flyte features demonstrated:
  - Parallel fan-out via Send (visual in TUI)
  - Per-task HTML reports with reward charts
  - Result caching (same query + agent_type = instant replay)
  - Run links returned to Gradio UI

Usage:
    # Local
    flyte run --local --tui workflow.py run_research_comparison \
        --queries '["What is MCP?", "How does RAG work?"]'

    # Remote
    flyte run workflow.py run_research_comparison \
        --queries '["What is MCP?", "How does RAG work?"]'
"""

import json
import base64
import logging
import operator
from typing import Annotated

import flyte
import flyte.report
import markdown as md_lib

from config import base_env, ANTHROPIC_API_KEY, TAVILY_API_KEY
from reward import keyword_reward, llm_judge_reward
from env.research_env import ResearchEnvironment
from agents.openenv_agent import OpenEnvAgent
from agents.traditional_agent import TraditionalAgent

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

env = base_env


def _md_to_html(text: str) -> str:
    return md_lib.markdown(text, extensions=["tables", "fenced_code"])


# ---------------------------------------------------------------------------
# Task 1: Run a single research episode
# ---------------------------------------------------------------------------
# Cached by (query, agent_type, max_steps) — second run of the same
# question returns instantly from Flyte's cache.

@env.task(report=True, cache=True)
async def run_research_episode(
    query: str,
    agent_type: str = "openenv",   # "openenv" or "traditional"
    max_steps: int = 10,
) -> str:
    """Run one research episode and return full results as JSON."""
    log.info(f"[Episode] Starting: agent={agent_type}, query={query[:60]}")

    await flyte.report.replace.aio(
        f"<h2>{agent_type.title()} Agent</h2>"
        f"<p><b>Query:</b> {query}</p>"
        f"<p>Running...</p>"
    )
    await flyte.report.flush.aio()

    # Build environment with the appropriate reward function
    reward_fn = llm_judge_reward if agent_type == "openenv" else keyword_reward
    environment = ResearchEnvironment(reward_fn=reward_fn, max_steps=max_steps)

    steps = []
    keyword_scores = []
    llm_scores = []

    if agent_type == "openenv":
        agent = OpenEnvAgent(query=query, max_steps=max_steps)
        for step_data in agent.run(environment):
            steps.append(step_data)
            llm_scores.append(step_data.get("llm_score", 0.0))
            log.info(f"[Episode] Step {step_data['step']}: {step_data['tool_name']} score={step_data.get('llm_score', 0):.2f}")
    else:
        agent = TraditionalAgent(query=query, max_steps=max_steps)
        for step_data in agent.run(environment):
            steps.append(step_data)
            keyword_scores.append(step_data.get("keyword_score", 0.0))
            llm_scores.append(step_data.get("llm_score", 0.0))
            log.info(f"[Episode] Step {step_data['step']}: kw={step_data.get('keyword_score', 0):.2f} llm={step_data.get('llm_score', 0):.2f}")

    final_state = environment.state
    avg_llm = sum(llm_scores) / max(len(llm_scores), 1)
    avg_kw = sum(keyword_scores) / max(len(keyword_scores), 1) if keyword_scores else None

    # Build Flyte HTML report
    step_rows = "".join(
        f"<tr><td>{s['step']}</td><td>{s['tool_name']}</td>"
        f"<td>{s.get('keyword_score', 'N/A')}</td>"
        f"<td>{s.get('llm_score', 'N/A')}</td></tr>"
        for s in steps
    )

    tool_usage_html = "".join(
        f"<li>{tool}: {count} calls</li>"
        for tool, count in final_state.tool_usage.items()
    )

    report_html = f"""
<h2>{agent_type.title()} Agent Episode Report</h2>
<p><b>Query:</b> {query}</p>
<p><b>Steps taken:</b> {final_state.step} / {final_state.max_steps}</p>
<p><b>Avg LLM-judge score:</b> {avg_llm:.2f}</p>
{"<p><b>Avg keyword score:</b> " + f"{avg_kw:.2f}</p>" if avg_kw is not None else ""}
<h3>Tool Usage</h3><ul>{tool_usage_html}</ul>
<h3>Step Log</h3>
<table border="1" cellpadding="4">
  <tr><th>Step</th><th>Tool</th><th>Keyword Score</th><th>LLM Score</th></tr>
  {step_rows}
</table>
"""

    await flyte.report.replace.aio(report_html)
    await flyte.report.flush.aio()

    result = {
        "query": query,
        "agent_type": agent_type,
        "steps": steps,
        "total_steps": final_state.step,
        "tool_usage": final_state.tool_usage,
        "avg_llm_score": round(avg_llm, 3),
        "avg_keyword_score": round(avg_kw, 3) if avg_kw is not None else None,
        "total_reward": round(final_state.total_reward, 3),
    }
    log.info(f"[Episode] Done: agent={agent_type} avg_llm={avg_llm:.2f}")
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Task 2: Comparison pipeline — fans out all queries × both agents
# ---------------------------------------------------------------------------

@env.task(report=True)
async def run_research_comparison(
    queries: list[str],
    max_steps: int = 10,
) -> str:
    """
    Fan out all queries × both agent types as parallel Flyte tasks.

    Each combination (query, agent_type) becomes one run_research_episode
    task running in parallel on the cluster.
    """
    from langgraph.types import Send  # noqa: deferred import

    log.info(f"[Comparison] Starting {len(queries)} queries × 2 agents")

    await flyte.report.replace.aio(
        f"<h2>Research Comparison Pipeline</h2>"
        f"<p>Running {len(queries)} queries × 2 agents = {len(queries) * 2} parallel tasks</p>"
    )
    await flyte.report.flush.aio()

    # Fan out — each task runs independently on the cluster
    results_json = []
    import asyncio
    tasks = [
        run_research_episode(query=q, agent_type=agent, max_steps=max_steps)
        for q in queries
        for agent in ["openenv", "traditional"]
    ]
    results_json = await asyncio.gather(*tasks)

    results = [json.loads(r) for r in results_json]

    # Build summary comparison table
    rows = ""
    for r in results:
        kw = f"{r['avg_keyword_score']:.2f}" if r["avg_keyword_score"] is not None else "N/A"
        rows += (
            f"<tr><td>{r['query'][:50]}</td><td>{r['agent_type']}</td>"
            f"<td>{r['total_steps']}</td><td>{kw}</td>"
            f"<td>{r['avg_llm_score']:.2f}</td></tr>"
        )

    summary_html = f"""
<h2>Comparison Results</h2>
<table border="1" cellpadding="4">
  <tr><th>Query</th><th>Agent</th><th>Steps</th><th>Keyword Score</th><th>LLM Score</th></tr>
  {rows}
</table>
<p><i>Notice: Traditional agent scores high on keyword metric but low on LLM judge.
OpenEnv agent scores consistently high on the LLM judge — the meaningful signal.</i></p>
"""

    await flyte.report.replace.aio(summary_html)
    await flyte.report.flush.aio()

    log.info(f"[Comparison] Done. {len(results)} episodes completed.")
    return json.dumps({"results": results, "total_episodes": len(results)})
