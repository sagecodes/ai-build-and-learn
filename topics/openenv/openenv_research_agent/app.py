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
    fanout_narrative_summary,
    narrative_summary,  # used inside _comparison_summaries
    env_state_card,
    agent_loading_card,
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
# Shared helpers
# ---------------------------------------------------------------------------


def _comparison_summaries(
    avg_kw: float,
    trad_final: float | None,
    oe_final: float | None,
    trad_steps: int,
    oe_steps: int,
) -> tuple[str, str, str]:
    """
    Build the agent summary HTML and narrative for both comparison paths.

    Used by both the Local Process and Flyte Task paths in run_comparison()
    so the output is identical regardless of how the data was produced.

    Returns:
        (trad_summary_html, oe_summary_html, narrative_html)
    """
    trad_final = trad_final or 0.0
    oe_final = oe_final or 0.0
    gap = avg_kw - trad_final
    oe_advantage = oe_final - trad_final

    trad_html = agent_summary("Summary — Traditional RL", "#e67e22", [
        f"Avg Keyword Score: <b>{avg_kw:.2f}</b> &larr; what the agent optimized",
        f"Final LLM Judge Score: <b>{trad_final:.2f}</b> &larr; actual quality",
        f"Overestimation gap: <b>{gap:.2f}</b> &larr; this is reward hacking",
    ])
    oe_html = agent_summary("Summary — OpenEnv Agent", "#2471a3", [
        f"Final LLM Judge Score: <b>{oe_final:.2f}</b>",
        f"OpenEnv advantage: <b>+{oe_advantage:.2f}</b> over traditional",
    ])
    narr = narrative_summary(avg_kw, trad_final, oe_final, gap, oe_advantage,
                             trad_steps, oe_steps)
    return trad_html, oe_html, narr


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
        yield empty_chart("Reward Comparison"), "Enter a research question above.", "", "", ""
        return

    # ── Local Process path ──────────────────────────────────────────────────
    if run_mode == "Local Process":
        trad_kw = []
        trad_html_blocks, oe_html_blocks = [], []
        trad_final_llm = None
        oe_final_llm = None
        oe_step_count = 0

        yield (
            empty_chart("Reward Comparison — Traditional RL vs OpenEnv"),
            agent_loading_card("Traditional RL Agent", "#e67e22"),
            agent_loading_card("OpenEnv Agent", "#2471a3"),
            "", "",
        )

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
                        env_state = step.get("env_state", {})
                        if env_state:
                            oe_html_blocks.append(env_state_card(env_state))
                        oe_done = True
                    else:
                        oe_step_count += 1
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
                "",
            )

        avg_kw = sum(trad_kw) / max(len(trad_kw), 1)
        trad_summ, oe_summ, narr = _comparison_summaries(
            avg_kw, trad_final_llm, oe_final_llm, len(trad_kw), oe_step_count
        )
        trad_html_blocks.append(trad_summ)
        oe_html_blocks.append(oe_summ)

        yield (
            build_reward_chart(trad_kw, trad_final_llm, oe_final_llm, "FINAL — Reward Hacking Gap"),
            "".join(trad_html_blocks),
            "".join(oe_html_blocks),
            "",
            narr,
        )

    # ── Flyte Task path ─────────────────────────────────────────────────────
    else:
        yield empty_chart("Submitting to Flyte..."), "Submitting to Flyte...", "Submitting to Flyte...", "", ""

        result = flyte.with_runcontext(mode=RUN_MODE).run(
            run_side_by_side, query=query, max_steps=max_steps
        )
        link = _flyte_link(getattr(result, "url", None))

        yield empty_chart("Tasks running on Flyte..."), "Tasks running...", "Tasks running...", link, ""

        result.wait()
        data = json.loads(result.outputs()[0])

        kw_scores = data["kw_scores"]
        trad_final = data["trad_final_llm"]
        oe_final = data["oe_final_llm"]
        trad_avg_kw = data["trad_avg_kw"] or 0.0

        trad_summ, oe_summ, narr = _comparison_summaries(
            trad_avg_kw, trad_final, oe_final, len(kw_scores), len(kw_scores)
        )
        trad_html = final_score_block("Final LLM Judge Score", trad_final or 0.0, "#c0392b") + trad_summ
        oe_html = final_score_block("Final LLM Judge Score", oe_final or 0.0, "#1a7a4a") + oe_summ

        yield (
            build_reward_chart(kw_scores, trad_final, oe_final, "FINAL — Reward Hacking Gap"),
            trad_html,
            oe_html,
            link,
            narr,
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
        yield agent_loading_card("Agent Race — 3 OpenEnv Agents", "#2471a3"), ""

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
        yield "<p>Enter one research question per line.</p>", "", ""
        return

    yield agent_loading_card(f"Submitting {len(queries)} queries × 2 agents to Flyte...", "#2471a3"), "", ""

    result = flyte.with_runcontext(mode=RUN_MODE).run(
        run_research_comparison,
        queries=queries,
        max_steps=max_steps,
    )

    link = _flyte_link(getattr(result, "url", None))
    yield f"<p>Tasks running... ({len(queries) * 2} parallel Flyte tasks)</p>", link, ""

    result.wait()
    output = json.loads(result.outputs()[0])
    results = output.get("results", [])

    yield fanout_results_table(results), link, fanout_narrative_summary(results)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_RUN_MODE_CHOICES = ["Local Process", "Flyte Task"]

_EXAMPLE_QUESTIONS = [
    "How does retrieval-augmented generation compare to fine-tuning for production LLM applications?",
    "What are the key differences between LangGraph and AutoGen for building multi-agent AI systems?",
    "Compare CoreWeave vs Lambda Labs GPU cloud pricing, availability, and performance benchmarks",
]


def _example_list(target_elem_id: str, questions: list[str] = _EXAMPLE_QUESTIONS) -> str:
    """
    Render example questions as a clickable HTML list.

    Clicking an item finds the textarea inside the Gradio block whose
    wrapping div has id=target_elem_id, sets its value, and dispatches
    an input event so Gradio picks up the change.
    """
    def _onclick(q: str) -> str:
        escaped = q.replace("'", "\\'")
        return (
            f"(function(){{"
            f"var wrap=document.getElementById('{target_elem_id}');"
            f"var ta=wrap?wrap.querySelector('textarea'):null;"
            f"if(ta){{"
            f"var s=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;"
            f"s.call(ta,'{escaped}');"
            f"ta.dispatchEvent(new Event('input',{{bubbles:true}}));"
            f"}}"
            f"}})()"
        )

    items = "".join(
        f'<div class="ex-item" onclick="{_onclick(q)}">{q}</div>'
        for q in questions
    )
    return f'<div class="ex-list"><div class="ex-label">Example questions</div>{items}</div>'

def create_demo():
    css = open(_CSS_PATH).read()

    theme = gr.themes.Base(
        primary_hue=gr.themes.colors.orange,
        neutral_hue=gr.themes.colors.slate,
        font=gr.themes.GoogleFont("Inter"),
    ).set(
        body_background_fill="#0f0f0f",
        body_background_fill_dark="#0f0f0f",
        body_text_color="#e0e0e0",
        body_text_color_dark="#e0e0e0",
        background_fill_primary="#1a1a1a",
        background_fill_primary_dark="#1a1a1a",
        background_fill_secondary="#141414",
        background_fill_secondary_dark="#141414",
        border_color_primary="#2a2a2a",
        border_color_primary_dark="#2a2a2a",
        button_primary_background_fill="#e07b39",
        button_primary_background_fill_hover="#c96a28",
        button_primary_background_fill_dark="#e07b39",
        button_primary_background_fill_hover_dark="#c96a28",
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        input_background_fill="#1e1e1e",
        input_background_fill_dark="#1e1e1e",
        input_border_color="#2a2a2a",
        input_border_color_dark="#2a2a2a",
        block_background_fill="#1a1a1a",
        block_background_fill_dark="#1a1a1a",
        block_border_color="#2a2a2a",
        block_border_color_dark="#2a2a2a",
        block_label_text_color="#888",
        block_label_text_color_dark="#888",
        block_title_text_color="#e0e0e0",
        block_title_text_color_dark="#e0e0e0",
    )

    with gr.Blocks(title="OpenEnv Research Agent", css=css, theme=theme) as demo:

        # ── Header ────────────────────────────────────────────────────────
        gr.HTML(
            '<div class="app-header">'
            '<span class="app-title">OpenEnv Research Agent</span>'
            '<span class="app-tagline">OpenEnv &nbsp;·&nbsp; Flyte &nbsp;·&nbsp; Tavily &nbsp;·&nbsp; Claude</span>'
            '</div>'
        )

        with gr.Row(elem_id="main-layout"):

            # ── Left sidebar ──────────────────────────────────────────────
            with gr.Column(scale=1, elem_id="sidebar"):

                mode_select = gr.Radio(
                    choices=["Side-by-Side", "Agent Race", "Fan-out"],
                    value="Side-by-Side",
                    label="Demo Mode",
                    elem_classes="mode-nav",
                )

                gr.HTML('<div class="sidebar-sep"></div>')

                # Side-by-Side inputs
                with gr.Group(visible=True) as comp_panel:
                    gr.HTML('<div class="panel-heading">Run a Side-by-Side Comparison</div>'
                            '<p class="panel-desc">Run both agents on the same question. '
                            'Watch keyword reward get gamed while LLM judge stays honest.</p>')
                    comp_query = gr.Textbox(
                        label="Research Question",
                        placeholder="Type a research question or select a sample question from below",
                        lines=3,
                        elem_id="comp-query",
                    )
                    comp_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps")
                    comp_mode = gr.Radio(choices=_RUN_MODE_CHOICES, value="Local Process", label="Run Mode")
                    comp_btn = gr.Button("Run Comparison →", variant="primary")
                    gr.HTML(_example_list("comp-query", [
                        "How does retrieval-augmented generation compare to fine-tuning for production LLM applications?",
                        "What are the key differences between LangGraph and AutoGen for building multi-agent AI systems?",
                        "Compare CoreWeave vs Lambda Labs GPU cloud pricing, availability, and performance benchmarks",
                    ]))

                # Agent Race inputs
                with gr.Group(visible=False) as race_panel:
                    gr.HTML('<div class="panel-heading">Start an Agent Race</div>'
                            '<p class="panel-desc">3 OpenEnv agents race on the same question '
                            'with concurrent Docker sessions (SUPPORTS_CONCURRENT_SESSIONS).</p>')
                    race_query = gr.Textbox(
                        label="Research Question",
                        placeholder="Type a research question or select a sample question from below",
                        lines=3,
                        elem_id="race-query",
                    )
                    race_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps")
                    race_mode = gr.Radio(choices=_RUN_MODE_CHOICES, value="Local Process", label="Run Mode")
                    race_btn = gr.Button("Start Race →", variant="primary")
                    gr.HTML(_example_list("race-query"))

                # Fan-out inputs
                with gr.Group(visible=False) as fanout_panel:
                    gr.HTML('<div class="panel-heading">Parallel Flyte Fan-out</div>'
                            '<p class="panel-desc">Submit multiple research questions at once '
                            'as parallel Flyte tasks — both agents run per question.</p>')
                    fanout_queries = gr.Textbox(
                        label="Research Questions (one per line)",
                        placeholder="Type a research question or select a sample question from below",
                        lines=5,
                    )
                    fanout_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps")
                    fanout_btn = gr.Button("Run on Flyte →", variant="primary")

            # ── Right content area ────────────────────────────────────────
            with gr.Column(scale=2, elem_id="content-area"):

                # Side-by-Side outputs
                with gr.Group(visible=True) as comp_out:
                    comp_link = gr.HTML()
                    reward_chart = gr.Plot(label="Reward Comparison")
                    with gr.Row():
                        with gr.Column():
                            gr.HTML('<div class="col-label">Traditional RL Agent</div>')
                            trad_log = gr.HTML()
                        with gr.Column():
                            gr.HTML('<div class="col-label">OpenEnv Agent</div>')
                            oe_log = gr.HTML()
                    narrative_html = gr.HTML()

                # Agent Race outputs
                with gr.Group(visible=False) as race_out:
                    race_link = gr.HTML()
                    race_board = gr.HTML()

                # Fan-out outputs
                with gr.Group(visible=False) as fanout_out:
                    fanout_link = gr.HTML()
                    fanout_results = gr.HTML()
                    fanout_narrative = gr.HTML()

        # ── Mode switching ────────────────────────────────────────────────
        def switch_mode(mode):
            is_comp   = mode == "Side-by-Side"
            is_race   = mode == "Agent Race"
            is_fanout = mode == "Fan-out"
            return (
                gr.update(visible=is_comp),
                gr.update(visible=is_race),
                gr.update(visible=is_fanout),
                gr.update(visible=is_comp),
                gr.update(visible=is_race),
                gr.update(visible=is_fanout),
            )

        mode_select.change(
            fn=switch_mode,
            inputs=[mode_select],
            outputs=[comp_panel, race_panel, fanout_panel, comp_out, race_out, fanout_out],
        )

        # ── Button wiring ─────────────────────────────────────────────────
        comp_btn.click(
            fn=run_comparison,
            inputs=[comp_query, comp_steps, comp_mode],
            outputs=[reward_chart, trad_log, oe_log, comp_link, narrative_html],
        )
        race_btn.click(
            fn=run_race,
            inputs=[race_query, race_steps, race_mode],
            outputs=[race_board, race_link],
        )
        fanout_btn.click(
            fn=run_flyte_fanout,
            inputs=[fanout_queries, fanout_steps],
            outputs=[fanout_results, fanout_link, fanout_narrative],
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
