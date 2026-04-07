"""
Flyte workflow for the OpenEnv Research Agent demo.

Four Flyte tasks are defined:

1. run_research_episode(query, agent_type, max_steps) -> str (JSON)
   Runs a single research episode — either the OpenEnv agent or the
   traditional agent — inside a Flyte task pod. Each pod starts its own
   local OpenEnv HTTP server on a random port, then the agent connects
   to it via GenericEnvClient — same code path as local Docker development.
   Returns the full episode result as JSON including step log and scores.

2. run_research_comparison(queries, max_steps) -> str (JSON)
   The main pipeline for Tab 3. Takes a list of research questions and fans
   them out as parallel Flyte tasks — each question runs both agents
   simultaneously. Returns a comparison report.

   Fan-out diagram:
       START → run_research_comparison
                 ├─asyncio.gather──→ run_research_episode (openenv, q1)
                 ├─asyncio.gather──→ run_research_episode (traditional, q1)
                 ├─asyncio.gather──→ run_research_episode (openenv, q2)
                 └─asyncio.gather──→ run_research_episode (traditional, q2)
                 → collect results → END

3. run_side_by_side(query, max_steps) -> str (JSON)
   Runs traditional and OpenEnv agents in parallel on the same query.
   Used by Tab 1 (Side-by-Side Comparison) in Flyte mode.
   Returns kw_scores list + both final LLM scores for chart rendering.

4. run_agent_race(query, max_steps) -> str (JSON)
   Runs 3 OpenEnv agents in parallel on the same query.
   Uses asyncio.as_completed to identify the winner by completion order.
   Used by Tab 2 (Agent Race) in Flyte mode.

Flyte features demonstrated:
  - Parallel fan-out via asyncio.gather / as_completed across the cluster
  - Per-task HTML reports with step logs and reward scores
  - Result caching (same query + agent_type = instant replay)
  - Run links returned to Gradio UI

Usage:
    # Local
    flyte run --local workflow.py run_research_comparison \
        --queries '["What is MCP?", "How does RAG work?"]'

    # Remote
    flyte run workflow.py run_research_comparison \
        --queries '["What is MCP?", "How does RAG work?"]'
"""

import asyncio
import json
import logging
import socket
import threading
import time

import flyte
import flyte.report
import markdown as md_lib
import uvicorn

from config import base_env
from reward import keyword_reward, llm_judge_final_reward
from env.models import ResearchAction, ResearchObservation
from env.research_env import ResearchEnvironment
from openenv.core.env_server.http_server import create_app
from agents.openenv_agent import OpenEnvAgent
from agents.traditional_agent import TraditionalAgent

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

env = base_env


def _md_to_html(text: str) -> str:
    return md_lib.markdown(text, extensions=["tables", "fenced_code"])


# ---------------------------------------------------------------------------
# Local OpenEnv server helpers
# ---------------------------------------------------------------------------
# Each Flyte task pod starts its own OpenEnv HTTP server on a random port.
# The agent connects to it via GenericEnvClient — same code path as local
# Docker dev. The server thread is a daemon, so it dies when the task exits.

def _free_port() -> int:
    """Find a free TCP port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _start_env_server(port: int) -> None:
    """
    Start the OpenEnv HTTP server in a background daemon thread.

    The server uses keyword_reward as the per-step reward function.
    llm_judge_final_reward is called separately by agents at episode end.
    """
    app = create_app(
        env=lambda: ResearchEnvironment(reward_fn=keyword_reward),
        action_cls=ResearchAction,
        observation_cls=ResearchObservation,
        max_concurrent_envs=5,
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll /health until ready (max 30s)
    import requests
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=1,
                             proxies={"http": None, "https": None})
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Local OpenEnv server on port {port} did not start within 30s")


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

    # Start a local OpenEnv HTTP server for this task pod
    port = _free_port()
    _start_env_server(port)
    env_url = f"http://127.0.0.1:{port}"
    log.info(f"[Episode] Local OpenEnv server ready at {env_url}")

    steps = []
    keyword_scores = []
    llm_final_score = None
    tool_usage: dict[str, int] = {}

    if agent_type == "openenv":
        agent = OpenEnvAgent(query=query, max_steps=max_steps, env_url=env_url)
        for step_data in agent.run():
            steps.append(step_data)
            if step_data["tool_name"] == "final_judgment":
                llm_final_score = step_data.get("llm_final_score", 0.0)
            else:
                tool = step_data.get("tool_name", "")
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
                log.info(f"[Episode] Step {step_data['step']}: {tool}")
    else:
        agent = TraditionalAgent(query=query, max_steps=max_steps, env_url=env_url)
        for step_data in agent.run():
            steps.append(step_data)
            if step_data["tool_name"] == "final_judgment":
                llm_final_score = step_data.get("llm_final_score", 0.0)
                keyword_scores.append(step_data.get("keyword_score", 0.0))
            else:
                kw = step_data.get("keyword_score", 0.0)
                keyword_scores.append(kw)
                tool = step_data.get("tool_name", "")
                tool_usage[tool] = tool_usage.get(tool, 0) + 1
                log.info(f"[Episode] Step {step_data['step']}: kw={kw:.2f}")

    total_steps = len([s for s in steps if s["tool_name"] != "final_judgment"])
    avg_kw = sum(keyword_scores) / max(len(keyword_scores), 1) if keyword_scores else None

    # Build Flyte HTML report
    step_rows = "".join(
        f"<tr><td>{s['step']}</td><td>{s['tool_name']}</td>"
        f"<td>{s.get('keyword_score', 'N/A')}</td>"
        f"<td>{s.get('llm_final_score', 'N/A')}</td></tr>"
        for s in steps
    )
    tool_usage_html = "".join(
        f"<li>{tool}: {count} calls</li>"
        for tool, count in tool_usage.items()
    )

    report_html = f"""
<h2>{agent_type.title()} Agent Episode Report</h2>
<p><b>Query:</b> {query}</p>
<p><b>Steps taken:</b> {total_steps} / {max_steps}</p>
<p><b>Final LLM-judge score:</b> {llm_final_score:.2f}</p>
{"<p><b>Avg keyword score:</b> " + f"{avg_kw:.2f}</p>" if avg_kw is not None else ""}
<h3>Tool Usage</h3><ul>{tool_usage_html}</ul>
<h3>Step Log</h3>
<table border="1" cellpadding="4">
  <tr><th>Step</th><th>Tool</th><th>Keyword Score</th><th>LLM Final Score</th></tr>
  {step_rows}
</table>
"""

    await flyte.report.replace.aio(report_html)
    await flyte.report.flush.aio()

    result = {
        "query": query,
        "agent_type": agent_type,
        "steps": steps,
        "total_steps": total_steps,
        "tool_usage": tool_usage,
        "llm_final_score": round(llm_final_score, 3) if llm_final_score is not None else None,
        "avg_keyword_score": round(avg_kw, 3) if avg_kw is not None else None,
    }
    log.info(f"[Episode] Done: agent={agent_type} llm_final={llm_final_score:.2f}")
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
    log.info(f"[Comparison] Starting {len(queries)} queries × 2 agents")

    await flyte.report.replace.aio(
        f"<h2>Research Comparison Pipeline</h2>"
        f"<p>Running {len(queries)} queries × 2 agents"
        f" = {len(queries) * 2} parallel tasks</p>"
    )
    await flyte.report.flush.aio()

    # Fan out — each task runs independently on the cluster
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
        llm = f"{r['llm_final_score']:.2f}" if r["llm_final_score"] is not None else "N/A"
        rows += (
            f"<tr><td>{r['query'][:50]}</td><td>{r['agent_type']}</td>"
            f"<td>{r['total_steps']}</td><td>{kw}</td><td>{llm}</td></tr>"
        )

    summary_html = f"""
<h2>Comparison Results</h2>
<table border="1" cellpadding="4">
  <tr><th>Query</th><th>Agent</th><th>Steps</th><th>Keyword Score</th><th>LLM Final Score</th></tr>
  {rows}
</table>
<p><i>Traditional agent scores high on keyword metric but low on LLM judge.
OpenEnv agent scores consistently high on the LLM judge — the meaningful signal.</i></p>
"""

    await flyte.report.replace.aio(summary_html)
    await flyte.report.flush.aio()

    log.info(f"[Comparison] Done. {len(results)} episodes completed.")
    return json.dumps({"results": results, "total_episodes": len(results)})


# ---------------------------------------------------------------------------
# Task 3: Side-by-side comparison — used by Tab 1 in Flyte mode
# ---------------------------------------------------------------------------

@env.task(report=True)
async def run_side_by_side(query: str, max_steps: int = 6) -> str:
    """
    Run traditional and OpenEnv agents in parallel on the same query.

    Fans out to two run_research_episode tasks via asyncio.gather.
    Returns enough data to render the final reward chart and summaries
    in the Gradio UI without streaming — the Flyte console shows full
    per-step logs for each agent.
    """
    log.info(f"[SideBySide] Starting: query={query[:60]}")

    await flyte.report.replace.aio(
        f"<h2>Side-by-Side Comparison</h2>"
        f"<p><b>Query:</b> {query}</p>"
        f"<p>Running traditional and OpenEnv agents in parallel...</p>"
    )
    await flyte.report.flush.aio()

    trad_json, oe_json = await asyncio.gather(
        run_research_episode(query=query, agent_type="traditional", max_steps=max_steps),
        run_research_episode(query=query, agent_type="openenv", max_steps=max_steps),
    )
    trad = json.loads(trad_json)
    oe = json.loads(oe_json)

    # Extract per-step keyword scores from traditional steps for chart rendering
    kw_scores = [
        s.get("keyword_score", 0.0)
        for s in trad["steps"]
        if s.get("tool_name") != "final_judgment"
    ]

    trad_final = trad["llm_final_score"]
    oe_final = oe["llm_final_score"]
    trad_avg_kw = trad["avg_keyword_score"] or 0.0
    gap = trad_avg_kw - (trad_final or 0.0)
    oe_advantage = (oe_final or 0.0) - (trad_final or 0.0)

    trad_final_str = f"{trad_final:.2f}" if trad_final is not None else "N/A"
    oe_final_str = f"{oe_final:.2f}" if oe_final is not None else "N/A"
    trad_avg_str = f"{trad_avg_kw:.2f}"

    report_html = f"""
<h2>Side-by-Side Comparison Results</h2>
<p><b>Query:</b> {query}</p>
<table border="1" cellpadding="6">
  <tr><th>Agent</th><th>Steps</th><th>Avg Keyword Score</th><th>Final LLM Score</th></tr>
  <tr>
    <td>Traditional RL</td><td>{trad['total_steps']}</td>
    <td>{trad_avg_str}</td><td>{trad_final_str}</td>
  </tr>
  <tr>
    <td>OpenEnv Agent</td><td>{oe['total_steps']}</td>
    <td>N/A</td><td>{oe_final_str}</td>
  </tr>
</table>
<p><b>Reward hacking gap:</b> {gap:.2f} &nbsp;|&nbsp;
   <b>OpenEnv advantage:</b> +{oe_advantage:.2f}</p>
<p><i>Traditional agent scores high on keyword metric but low on LLM judge.
OpenEnv agent scores consistently higher on the meaningful signal.</i></p>
"""
    await flyte.report.replace.aio(report_html)
    await flyte.report.flush.aio()

    log.info(f"[SideBySide] Done: trad_llm={trad_final_str} oe_llm={oe_final_str} gap={gap:.2f}")
    return json.dumps({
        "query": query,
        "kw_scores": kw_scores,
        "trad_final_llm": trad_final,
        "oe_final_llm": oe_final,
        "trad_avg_kw": trad_avg_kw,
        "trad_total_steps": trad["total_steps"],
        "oe_total_steps": oe["total_steps"],
    })


# ---------------------------------------------------------------------------
# Task 4: Agent race — used by Tab 2 in Flyte mode
# ---------------------------------------------------------------------------

@env.task(report=True)
async def run_agent_race(query: str, max_steps: int = 6) -> str:
    """
    Run 3 OpenEnv agents in parallel on the same query.

    Uses asyncio.as_completed so the first task to finish on the cluster
    is identified as the winner — preserving the race semantics even when
    agents run on separate Flyte pods. Returns final scores, step counts,
    and winner for scoreboard rendering in the Gradio UI.
    """
    num_agents = 3
    log.info(f"[Race] Starting {num_agents} agents: query={query[:60]}")

    await flyte.report.replace.aio(
        f"<h2>Agent Race</h2>"
        f"<p><b>Query:</b> {query}</p>"
        f"<p>Racing {num_agents} OpenEnv agents on the cluster...</p>"
    )
    await flyte.report.flush.aio()

    async def _run(agent_id: int):
        result_json = await run_research_episode(
            query=query, agent_type="openenv", max_steps=max_steps
        )
        result = json.loads(result_json)
        result["agent_id"] = agent_id
        return result

    winner = None
    results: dict[int, dict] = {}

    # as_completed yields tasks in finish order — first completion = winner
    for coro in asyncio.as_completed([_run(i) for i in range(num_agents)]):
        result = await coro
        agent_id = result["agent_id"]
        results[agent_id] = result
        if winner is None:
            winner = agent_id
            log.info(f"[Race] Winner: Agent {winner}")

    final_scores = {i: results[i]["llm_final_score"] for i in results}
    step_counts = {i: results[i]["total_steps"] for i in results}

    scores_html = "".join(
        f"<tr><td>Agent {i}</td><td>{step_counts[i]}</td>"
        f"<td>{final_scores[i]:.2f if final_scores[i] is not None else 'N/A'}</td>"
        f"<td>{'WINNER' if i == winner else ''}</td></tr>"
        for i in sorted(results)
    )
    report_html = f"""
<h2>Agent Race Results</h2>
<p><b>Query:</b> {query}</p>
<p><b>Winner:</b> Agent {winner} (first to complete on the cluster)</p>
<table border="1" cellpadding="6">
  <tr><th>Agent</th><th>Steps</th><th>Final LLM Score</th><th>Status</th></tr>
  {scores_html}
</table>
<p><i>All 3 agents ran as separate Flyte tasks in parallel.
Winner determined by task completion order via asyncio.as_completed.</i></p>
"""
    await flyte.report.replace.aio(report_html)
    await flyte.report.flush.aio()

    log.info(f"[Race] Done. Winner=Agent {winner}, scores={final_scores}")
    return json.dumps({
        "query": query,
        "final_scores": final_scores,
        "step_counts": step_counts,
        "winner": winner,
        "num_agents": num_agents,
    })
