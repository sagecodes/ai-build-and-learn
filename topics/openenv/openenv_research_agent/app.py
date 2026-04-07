"""
Gradio UI for the OpenEnv Research Agent demo.

Three demo modes accessible via tabs:

  Tab 1 — Side-by-Side Comparison
    Run both agents on the same question simultaneously.
    Local Process mode: live chart and step logs update after every agent step.
    Flyte Task mode: both agents run as parallel Flyte tasks; final chart and
    summaries render when tasks complete. A run link opens the Flyte console.

  Tab 2 — Agent Race
    3 OpenEnv agents race on the same question with concurrent sessions.
    Local Process mode: live scoreboard ticking after each step.
    Flyte Task mode: 3 agents run as parallel Flyte tasks; winner determined
    by task completion order. Final scoreboard renders when all tasks finish.

  Tab 3 — Parallel Flyte Fan-out
    Submit multiple research questions at once.
    Each dispatches to a Flyte task. Flyte run links appear immediately.
    Results stream in as tasks complete.

Each tab has a "Run Mode" toggle:
  - Local Process — agents run in the current Python process, connecting to
    the local Docker container via GenericEnvClient. Live streaming enabled.
  - Flyte Task    — agents dispatch as Flyte tasks (local or remote cluster
    depending on RUN_MODE env var). No streaming; final results only.
    Flyte console link appears immediately for live observability.

Run modes (set via RUN_MODE env var):
    RUN_MODE=local python app.py   # fully local, Docker + local Flyte
    python app.py                  # local UI + remote Flyte cluster
    flyte deploy app.py serving_env  # fully remote

Visual components (HTML builders, chart builders, CSS) live in:
    ui_components.py  — all Plotly figures and HTML card builders
    styles.css        — CSS classes referenced by ui_components.py
"""

import json
import os

import flyte
import flyte.app
import gradio as gr
from dotenv import load_dotenv

from workflow import (
    run_research_comparison,
    run_side_by_side,
    run_agent_race,
)
from agents.openenv_agent import OpenEnvAgent
from agents.traditional_agent import TraditionalAgent
from ui_components import (
    empty_chart,
    build_reward_chart,
    trad_step_card,
    oe_step_card,
    final_score_block,
    agent_summary,
    race_scoreboard,
    race_summary,
    fanout_results_table,
)

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "remote")

_CSS_PATH = os.path.join(os.path.dirname(__file__), "styles.css")

serving_env = flyte.app.AppEnvironment(
    name="openenv-research-agent-ui",
    image=flyte.Image.from_debian_base(python_version=(3, 11)).with_pip_packages(
        "flyte>=2.1.2", "openenv-core>=0.2.3", "anthropic",
        "tavily-python", "gradio", "plotly", "python-dotenv", "markdown",
    ),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    requires_auth=False,
    port=7860,
)


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _flyte_link(run_url) -> str:
    """Return an HTML link to the Flyte console run, or empty string."""
    if run_url and str(run_url).startswith("http"):
        return (
            f'<div style="margin:8px 0;padding:8px 12px;background:#eaf4fb;'
            f'border-left:4px solid #2471a3;border-radius:4px">'
            f'<b>Flyte run:</b> '
            f'<a href="{run_url}" target="_blank" style="color:#2471a3">'
            f'View tasks, logs, and inputs/outputs &#8599;</a>'
            f'</div>'
        )
    return ""


# ---------------------------------------------------------------------------
# Tab 1: Side-by-Side Comparison
# ---------------------------------------------------------------------------

def run_comparison(query: str, max_steps: int, run_mode: str):
    """
    Run both agents on the same query and yield live updates.

    Local Process mode: yields (chart, trad_html, oe_html, "") after each step.
    Flyte Task mode: submits run_side_by_side task, shows run link immediately,
    then yields final chart and summaries when the task completes.

    Always yields 4-tuples: (reward_chart, trad_log_html, oe_log_html, link_html).
    """
    if not query.strip():
        yield empty_chart("Reward Comparison"), "Enter a research question above.", "", ""
        return

    # ── Local Process path ──────────────────────────────────────────────────
    if run_mode == "Local Process":
        trad_kw = []
        trad_html_blocks, oe_html_blocks = [], []
        trad_final_llm = None
        oe_final_llm = None

        trad_agent = TraditionalAgent(query=query, max_steps=max_steps)
        oe_agent = OpenEnvAgent(query=query, max_steps=max_steps)

        trad_gen = trad_agent.run()
        oe_gen = oe_agent.run()
        trad_done = False
        oe_done = False

        while not (trad_done and oe_done):
            if not trad_done:
                try:
                    step = next(trad_gen)
                    if step["tool_name"] == "final_judgment":
                        trad_final_llm = step.get("llm_final_score", 0.0)
                        trad_html_blocks.append(
                            final_score_block("Final LLM Judge Score", trad_final_llm, "#c0392b")
                        )
                        trad_done = True
                    else:
                        kw = step.get("keyword_score", 0.0)
                        trad_kw.append(kw)
                        trad_html_blocks.append(trad_step_card(
                            step["step"], step["tool_name"], kw,
                            step.get("query_used", ""), step.get("matched_keywords", []),
                        ))
                except StopIteration:
                    trad_done = True

            if not oe_done:
                try:
                    step = next(oe_gen)
                    if step["tool_name"] == "final_judgment":
                        oe_final_llm = step.get("llm_final_score", 0.0)
                        oe_html_blocks.append(
                            final_score_block("Final LLM Judge Score", oe_final_llm, "#1a7a4a")
                        )
                        oe_done = True
                    else:
                        oe_html_blocks.append(oe_step_card(
                            step["step"], step["tool_name"],
                            step.get("tool_args", {}), step.get("result_preview", ""),
                        ))
                except StopIteration:
                    oe_done = True

            yield (
                build_reward_chart(
                    trad_kw, trad_final_llm, oe_final_llm,
                    "Reward Comparison — Traditional RL vs OpenEnv"
                ),
                "".join(trad_html_blocks),
                "".join(oe_html_blocks),
                "",
            )

        avg_kw = sum(trad_kw) / max(len(trad_kw), 1)
        trad_llm_val = trad_final_llm or 0.0
        oe_llm_val = oe_final_llm or 0.0
        gap = avg_kw - trad_llm_val
        oe_advantage = oe_llm_val - trad_llm_val

        trad_html_blocks.append(agent_summary("Summary — Traditional RL", "#e67e22", [
            f"Avg Keyword Score: <b>{avg_kw:.2f}</b> &larr; what the agent optimized",
            f"Final LLM Judge Score: <b>{trad_llm_val:.2f}</b> &larr; actual quality",
            f"Overestimation gap: <b>{gap:.2f}</b> &larr; this is reward hacking",
        ]))
        oe_html_blocks.append(agent_summary("Summary — OpenEnv Agent", "#2471a3", [
            f"Final LLM Judge Score: <b>{oe_llm_val:.2f}</b>",
            f"OpenEnv advantage: <b>+{oe_advantage:.2f}</b> over traditional",
        ]))

        yield (
            build_reward_chart(trad_kw, trad_final_llm, oe_final_llm, "FINAL — Reward Hacking Gap"),
            "".join(trad_html_blocks),
            "".join(oe_html_blocks),
            "",
        )

    # ── Flyte Task path ─────────────────────────────────────────────────────
    else:
        yield empty_chart("Submitting to Flyte..."), "Submitting to Flyte...", "Submitting to Flyte...", ""

        result = flyte.with_runcontext(mode=RUN_MODE).run(
            run_side_by_side, query=query, max_steps=max_steps
        )
        link = _flyte_link(getattr(result, "url", None))

        yield empty_chart("Tasks running on Flyte..."), "Tasks running...", "Tasks running...", link

        result.wait()
        data = json.loads(result.outputs()[0])

        kw_scores = data["kw_scores"]
        trad_final = data["trad_final_llm"]
        oe_final = data["oe_final_llm"]
        trad_avg_kw = data["trad_avg_kw"] or 0.0
        gap = trad_avg_kw - (trad_final or 0.0)
        oe_advantage = (oe_final or 0.0) - (trad_final or 0.0)

        trad_html = (
            final_score_block("Final LLM Judge Score", trad_final or 0.0, "#c0392b")
            + agent_summary("Summary — Traditional RL", "#e67e22", [
                f"Avg Keyword Score: <b>{trad_avg_kw:.2f}</b> &larr; what the agent optimized",
                f"Final LLM Judge Score: <b>{trad_final or 0.0:.2f}</b> &larr; actual quality",
                f"Overestimation gap: <b>{gap:.2f}</b> &larr; this is reward hacking",
            ])
        )
        oe_html = (
            final_score_block("Final LLM Judge Score", oe_final or 0.0, "#1a7a4a")
            + agent_summary("Summary — OpenEnv Agent", "#2471a3", [
                f"Final LLM Judge Score: <b>{oe_final or 0.0:.2f}</b>",
                f"OpenEnv advantage: <b>+{oe_advantage:.2f}</b> over traditional",
            ])
        )

        yield (
            build_reward_chart(kw_scores, trad_final, oe_final, "FINAL — Reward Hacking Gap"),
            trad_html,
            oe_html,
            link,
        )


# ---------------------------------------------------------------------------
# Tab 2: Agent Race
# ---------------------------------------------------------------------------

def run_race(query: str, max_steps: int, run_mode: str):
    """
    Run 3 OpenEnv agents on the same query.

    Local Process mode: all 3 agents connect to the local Docker container
    via concurrent sessions (SUPPORTS_CONCURRENT_SESSIONS=True). Live
    scoreboard updates after each step. Yields (board_html, "") pairs.

    Flyte Task mode: 3 agents run as parallel Flyte tasks. Winner is the
    first task to complete on the cluster (asyncio.as_completed ordering).
    Yields (board_html, link_html) pairs — no live updates until done.
    """
    if not query.strip():
        yield "Enter a research question to start the race.", ""
        return

    # ── Local Process path ──────────────────────────────────────────────────
    if run_mode == "Local Process":
        num_agents = 3
        agents = [OpenEnvAgent(query=query, agent_id=i, max_steps=max_steps) for i in range(num_agents)]
        gens = [agent.run() for agent in agents]

        step_counts = {i: 0 for i in range(num_agents)}
        final_scores = {i: None for i in range(num_agents)}
        last_tools = {i: "" for i in range(num_agents)}
        done = {i: False for i in range(num_agents)}
        winner = None

        while not all(done.values()):
            for i, gen in enumerate(gens):
                if done[i]:
                    continue
                try:
                    step = next(gen)
                    if step["tool_name"] == "final_judgment":
                        final_scores[i] = step.get("llm_final_score", 0.0)
                        done[i] = True
                        if winner is None:
                            winner = i
                    else:
                        step_counts[i] += 1
                        last_tools[i] = step.get("tool_name", "")
                except StopIteration:
                    done[i] = True

            yield race_scoreboard(step_counts, final_scores, done, last_tools, winner), ""

        board = race_scoreboard(step_counts, final_scores, done, last_tools, winner)
        summary = race_summary(winner, final_scores)
        yield board + summary, ""

    # ── Flyte Task path ─────────────────────────────────────────────────────
    else:
        yield "Submitting race to Flyte...", ""

        result = flyte.with_runcontext(mode=RUN_MODE).run(
            run_agent_race, query=query, max_steps=max_steps
        )
        link = _flyte_link(getattr(result, "url", None))

        yield "Race running on Flyte cluster...", link

        result.wait()
        data = json.loads(result.outputs()[0])

        # JSON keys are strings — convert back to int for scoreboard functions
        final_scores = {int(k): v for k, v in data["final_scores"].items()}
        step_counts = {int(k): v for k, v in data["step_counts"].items()}
        winner = data["winner"]
        done = {i: True for i in final_scores}
        last_tools = {i: "" for i in final_scores}

        board = race_scoreboard(step_counts, final_scores, done, last_tools, winner)
        summary = race_summary(winner, final_scores)
        yield board + summary, link


# ---------------------------------------------------------------------------
# Tab 3: Parallel Flyte Fan-out
# ---------------------------------------------------------------------------

def run_flyte_fanout(queries_text: str, max_steps: int):
    """
    Submit multiple research questions to Flyte and show run links + results.

    Each question dispatches as a parallel Flyte task (both agents per question).
    Flyte caches results by (query, agent_type, max_steps) — identical runs return instantly.
    """
    queries = [q.strip() for q in queries_text.strip().splitlines() if q.strip()]
    if not queries:
        yield "<p>Enter one research question per line.</p>", ""
        return

    yield f"<p>Submitting {len(queries)} queries × 2 agents to Flyte...</p>", ""

    result = flyte.with_runcontext(mode=RUN_MODE).run(
        run_research_comparison,
        queries=queries,
        max_steps=max_steps,
    )

    link = _flyte_link(getattr(result, "url", None))
    yield f"<p>Tasks running... ({len(queries) * 2} parallel Flyte tasks)</p>", link

    result.wait()
    output = json.loads(result.outputs()[0])
    results = output.get("results", [])

    yield fanout_results_table(results), link


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_RUN_MODE_CHOICES = ["Local Process", "Flyte Task"]

def create_demo():
    css = open(_CSS_PATH).read()
    with gr.Blocks(title="OpenEnv Research Agent Demo", css=css) as demo:
        gr.Markdown(
            "# OpenEnv Research Agent Demo\n"
            "Showcasing OpenEnv + Flyte + Tavily + Claude\n\n"
            "**Traditional RL** games keyword rewards. "
            "**OpenEnv** uses LLM-as-judge for meaningful signals."
        )

        with gr.Tabs():

            # --- Tab 1: Side-by-Side Comparison ---
            with gr.Tab("Side-by-Side Comparison"):
                gr.Markdown(
                    "Run both agents on the same question. Watch keyword reward get gamed "
                    "while LLM judge stays honest.\n\n"
                    "**Local Process** — live chart updates per step. "
                    "**Flyte Task** — parallel tasks on the cluster; final results only + Flyte console link."
                )
                with gr.Row():
                    comp_query = gr.Textbox(
                        label="Research Question",
                        placeholder="What is Model Context Protocol (MCP)?",
                        scale=4,
                    )
                    comp_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps", scale=1)
                    comp_mode = gr.Radio(
                        choices=_RUN_MODE_CHOICES,
                        value="Local Process",
                        label="Run Mode",
                        scale=1,
                    )
                    comp_btn = gr.Button("Run Comparison", variant="primary", scale=0)

                comp_link = gr.HTML()
                reward_chart = gr.Plot(label="Reward Comparison")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Traditional RL Agent Steps**")
                        trad_log = gr.HTML()
                    with gr.Column():
                        gr.Markdown("**OpenEnv Agent Steps**")
                        oe_log = gr.HTML()

                comp_btn.click(
                    fn=run_comparison,
                    inputs=[comp_query, comp_steps, comp_mode],
                    outputs=[reward_chart, trad_log, oe_log, comp_link],
                )
                gr.Examples(
                    examples=[
                        ["How does retrieval-augmented generation compare to fine-tuning for production LLM applications?"],
                        ["What are the key differences between LangGraph and AutoGen for building multi-agent AI systems?"],
                        ["Compare CoreWeave vs Lambda Labs GPU cloud pricing, availability, and performance benchmarks"],
                    ],
                    inputs=comp_query,
                    label="Suggested questions (largest reward hacking gap)",
                )

            # --- Tab 2: Agent Race ---
            with gr.Tab("Agent Race"):
                gr.Markdown(
                    "3 OpenEnv agents race on the same question.\n\n"
                    "**Local Process** — concurrent sessions in the local Docker container "
                    "(SUPPORTS_CONCURRENT_SESSIONS); live scoreboard per step. "
                    "**Flyte Task** — 3 parallel tasks on the cluster; winner = first task to complete."
                )
                with gr.Row():
                    race_query = gr.Textbox(
                        label="Research Question",
                        placeholder="How does retrieval-augmented generation work?",
                        scale=4,
                    )
                    race_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps", scale=1)
                    race_mode = gr.Radio(
                        choices=_RUN_MODE_CHOICES,
                        value="Local Process",
                        label="Run Mode",
                        scale=1,
                    )
                    race_btn = gr.Button("Start Race", variant="primary", scale=0)

                race_link = gr.HTML()
                race_board = gr.HTML(label="Live Scoreboard")

                race_btn.click(
                    fn=run_race,
                    inputs=[race_query, race_steps, race_mode],
                    outputs=[race_board, race_link],
                )
                gr.Examples(
                    examples=[
                        ["How does retrieval-augmented generation compare to fine-tuning for production LLM applications?"],
                        ["What are the key differences between LangGraph and AutoGen for building multi-agent AI systems?"],
                        ["Compare CoreWeave vs Lambda Labs GPU cloud pricing, availability, and performance benchmarks"],
                    ],
                    inputs=race_query,
                    label="Suggested questions",
                )

            # --- Tab 3: Parallel Flyte Fan-out ---
            with gr.Tab("Parallel Flyte Fan-out"):
                gr.Markdown(
                    "Submit multiple research questions at once. Each dispatches as a parallel "
                    "Flyte task — both agents run simultaneously per question."
                )
                with gr.Row():
                    fanout_queries = gr.Textbox(
                        label="Research Questions (one per line)",
                        placeholder="What is MCP?\nHow does RAG work?\nWhat are AI agents?",
                        lines=4,
                        scale=3,
                    )
                    fanout_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps", scale=1)
                fanout_btn = gr.Button("Run on Flyte", variant="primary")
                fanout_link = gr.HTML()
                fanout_results = gr.HTML(label="Results")

                fanout_btn.click(
                    fn=run_flyte_fanout,
                    inputs=[fanout_queries, fanout_steps],
                    outputs=[fanout_results, fanout_link],
                )

    return demo


@serving_env.server
def app_server():
    create_demo().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    if RUN_MODE == "remote":
        flyte.init_from_config()
    create_demo().launch()

# Run modes:
#   RUN_MODE=local python app.py     — fully local
#   python app.py                    — local UI + remote Flyte
#   flyte deploy app.py serving_env  — fully remote
