"""
Gradio UI for the OpenEnv Research Agent demo.

Three demo modes accessible via tabs:

  Tab 1 — Side-by-Side Comparison
    Run both agents on the same question simultaneously.
    Shows live reward curves (Plotly) and step logs updating in real time.
    The reward hacking gap is visible here.

  Tab 2 — Agent Race
    3 OpenEnv agents race on the same question with concurrent sessions.
    Live scoreboard and step counts. First to finish wins.

  Tab 3 — Parallel Flyte Fan-out
    Submit multiple research questions at once.
    Each dispatches to a Flyte task. Flyte run links appear immediately.
    Results stream in as tasks complete.

Run modes (same pattern as langgraph_agent_research):
    RUN_MODE=local python app.py   # fully local, Docker + local Flyte
    python app.py                  # local UI + remote Flyte cluster
    flyte deploy app.py serving_env  # fully remote
"""

import json
import os
import threading

import flyte
import flyte.app
import gradio as gr
import plotly.graph_objects as go
from dotenv import load_dotenv

from workflow import run_research_episode, run_research_comparison
from reward import keyword_reward
from env.research_env import ResearchEnvironment
from agents.openenv_agent import OpenEnvAgent
from agents.traditional_agent import TraditionalAgent

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "remote")

serving_env = flyte.app.AppEnvironment(
    name="openenv-research-agent-ui",
    image=flyte.Image.from_debian_base(python_version=(3, 11)).with_pip_packages(
        "flyte>=2.1.2", "openenv-core>=0.2.3", "anthropic",
        "tavily-python", "gradio", "plotly", "python-dotenv",
        "markdown", "unionai-reuse",
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
# Reward chart helpers
# ---------------------------------------------------------------------------

def _empty_chart(title: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Reward (0-1)",
        yaxis=dict(range=[0, 1]),
        height=300,
    )
    return fig


def _build_chart(
    keyword_scores: list[float],
    llm_scores: list[float],
    title: str,
) -> go.Figure:
    steps = list(range(1, len(llm_scores) + 1))
    fig = go.Figure()
    if keyword_scores:
        fig.add_trace(go.Scatter(
            x=steps[:len(keyword_scores)],
            y=keyword_scores,
            mode="lines+markers",
            name="Keyword Reward (gameable)",
            line=dict(color="orange", dash="dash"),
        ))
    fig.add_trace(go.Scatter(
        x=steps,
        y=llm_scores,
        mode="lines+markers",
        name="LLM Judge Reward (meaningful)",
        line=dict(color="royalblue"),
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Reward (0-1)",
        yaxis=dict(range=[0, 1]),
        height=300,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def _build_final_chart(
    kw_scores: list[float],
    trad_final: float | None,
    oe_final: float | None,
    title: str,
) -> go.Figure:
    """
    Bar chart comparing keyword reward vs final LLM judge scores.
    Makes the reward hacking gap visually obvious.
    """
    fig = go.Figure()

    # Per-step keyword scores as a line (traditional only)
    if kw_scores:
        steps = list(range(1, len(kw_scores) + 1))
        fig.add_trace(go.Scatter(
            x=steps,
            y=kw_scores,
            mode="lines+markers",
            name="Per-step Keyword Score (Traditional)",
            line=dict(color="orange", dash="dash"),
        ))

    # Final scores as bars for direct comparison
    labels, values, colors = [], [], []
    if trad_final is not None:
        labels.append("Traditional\nFinal LLM Score")
        values.append(trad_final)
        colors.append("tomato")
    if oe_final is not None:
        labels.append("OpenEnv\nFinal LLM Score")
        values.append(oe_final)
        colors.append("mediumseagreen")

    if labels:
        fig.add_trace(go.Bar(
            x=labels,
            y=values,
            name="Final LLM Judge Score",
            marker_color=colors,
            width=0.4,
        ))

    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 1], title="Score (0-1)"),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        barmode="group",
    )
    return fig


# ---------------------------------------------------------------------------
# Tab 1: Side-by-Side Comparison
# ---------------------------------------------------------------------------

def run_comparison(query: str, max_steps: int):
    """
    Run both agents on the same query and yield live updates.
    Uses Gradio's generator pattern for real-time chart updates.
    """
    if not query.strip():
        yield (
            _empty_chart("Traditional RL Agent"),
            _empty_chart("OpenEnv Agent"),
            "Enter a research question above.",
            "Enter a research question above.",
        )
        return

    trad_kw = []
    trad_log, oe_log = [], []
    trad_final_llm = None
    oe_final_llm = None

    # Build environments
    trad_env = ResearchEnvironment(reward_fn=keyword_reward, max_steps=max_steps)
    oe_env = ResearchEnvironment(reward_fn=keyword_reward, max_steps=max_steps)  # reward_fn unused for final scoring

    trad_agent = TraditionalAgent(query=query, max_steps=max_steps)
    oe_agent = OpenEnvAgent(query=query, max_steps=max_steps)

    trad_gen = trad_agent.run(trad_env)
    oe_gen = oe_agent.run(oe_env)

    trad_done = False
    oe_done = False

    while not (trad_done and oe_done):
        if not trad_done:
            try:
                step = next(trad_gen)
                if step["tool_name"] == "final_judgment":
                    trad_final_llm = step.get("llm_final_score", 0.0)
                    trad_log.append(f"\nFINAL LLM JUDGE SCORE: {trad_final_llm:.2f}")
                    trad_done = True
                else:
                    kw = step.get("keyword_score", 0.0)
                    trad_kw.append(kw)
                    trad_log.append(
                        f"Step {step['step']}: {step['tool_name']} | KW={kw:.2f}"
                    )
                if step["done"]:
                    trad_done = True
            except StopIteration:
                trad_done = True

        if not oe_done:
            try:
                step = next(oe_gen)
                if step["tool_name"] == "final_judgment":
                    oe_final_llm = step.get("llm_final_score", 0.0)
                    oe_log.append(f"\nFINAL LLM JUDGE SCORE: {oe_final_llm:.2f}")
                    oe_done = True
                else:
                    oe_log.append(
                        f"Step {step['step']}: {step['tool_name']}"
                    )
                if step["done"]:
                    oe_done = True
            except StopIteration:
                oe_done = True

        # Chart: traditional shows per-step keyword scores as a bar
        # Final LLM scores shown as horizontal reference lines when available
        trad_chart = _build_final_chart(
            trad_kw, trad_final_llm, oe_final_llm,
            "Traditional RL vs OpenEnv — Reward Comparison"
        )
        oe_chart = _build_final_chart(
            [], trad_final_llm, oe_final_llm,
            "Final LLM Judge Scores"
        )

        yield (
            trad_chart,
            oe_chart,
            "\n".join(trad_log[-15:]),
            "\n".join(oe_log[-15:]),
        )

    # Final summary
    avg_kw = sum(trad_kw) / max(len(trad_kw), 1)
    gap = avg_kw - (trad_final_llm or 0)
    oe_advantage = (oe_final_llm or 0) - (trad_final_llm or 0)

    trad_log.append(
        f"\n--- SUMMARY ---\n"
        f"Avg Keyword Score:        {avg_kw:.2f}  ← what the agent optimized\n"
        f"Final LLM Judge Score:    {trad_final_llm:.2f}  ← actual quality\n"
        f"Overestimation gap:       {gap:.2f}  ← this is reward hacking"
    )
    oe_log.append(
        f"\n--- SUMMARY ---\n"
        f"Final LLM Judge Score:    {oe_final_llm:.2f}\n"
        f"OpenEnv advantage:        +{oe_advantage:.2f} over traditional"
    )

    yield (
        _build_final_chart(trad_kw, trad_final_llm, oe_final_llm, "FINAL — Traditional RL vs OpenEnv"),
        _build_final_chart([], trad_final_llm, oe_final_llm, "FINAL — LLM Judge Scores"),
        "\n".join(trad_log),
        "\n".join(oe_log),
    )


# ---------------------------------------------------------------------------
# Tab 2: Agent Race
# ---------------------------------------------------------------------------

def run_race(query: str, max_steps: int):
    """
    Run 3 OpenEnv agents concurrently on the same query.
    Yields a scoreboard string updated after each step.
    """
    if not query.strip():
        yield "Enter a research question to start the race."
        return

    num_agents = 3
    agents = [OpenEnvAgent(query=query, agent_id=i, max_steps=max_steps) for i in range(num_agents)]
    envs = [ResearchEnvironment(reward_fn=llm_judge_reward, max_steps=max_steps) for _ in range(num_agents)]
    gens = [agent.run(env) for agent, env in zip(agents, envs)]

    scores = {i: [] for i in range(num_agents)}
    done = {i: False for i in range(num_agents)}
    winner = None

    while not all(done.values()):
        for i, gen in enumerate(gens):
            if done[i]:
                continue
            try:
                step = next(gen)
                scores[i].append(step.get("llm_score", 0.0))
                if step["done"]:
                    done[i] = True
                    if winner is None:
                        winner = i
            except StopIteration:
                done[i] = True

        board = f"{'Agent':<10} {'Steps':<8} {'Avg Score':<12} {'Status'}\n"
        board += "-" * 45 + "\n"
        for i in range(num_agents):
            avg = sum(scores[i]) / max(len(scores[i]), 1)
            status = "DONE" if done[i] else "running..."
            if winner == i:
                status = "WINNER"
            board += f"Agent {i:<5} {len(scores[i]):<8} {avg:<12.3f} {status}\n"

        yield board

    yield board + f"\nRace complete. Winner: Agent {winner}"


# ---------------------------------------------------------------------------
# Tab 3: Parallel Flyte Fan-out
# ---------------------------------------------------------------------------

def run_flyte_fanout(queries_text: str, max_steps: int):
    """
    Submit multiple research questions to Flyte and show run links + results.
    """
    queries = [q.strip() for q in queries_text.strip().splitlines() if q.strip()]
    if not queries:
        yield "Enter one research question per line.", ""
        return

    yield f"Submitting {len(queries)} queries × 2 agents to Flyte...", ""

    result = flyte.with_runcontext(mode=RUN_MODE).run(
        run_research_comparison,
        queries=queries,
        max_steps=max_steps,
    )

    run_url = getattr(result, "url", None)
    link_html = ""
    if run_url and str(run_url).startswith("http"):
        link_html = f'<a href="{run_url}" target="_blank">View run on Flyte</a>'

    yield f"Tasks running... ({len(queries) * 2} parallel Flyte tasks)", link_html

    result.wait()
    output = json.loads(result.outputs()[0])
    results = output.get("results", [])

    # Build comparison table
    lines = [f"{'Query':<50} {'Agent':<12} {'Steps':<8} {'KW Score':<10} {'LLM Score'}"]
    lines.append("-" * 100)
    for r in results:
        kw = f"{r['avg_keyword_score']:.2f}" if r["avg_keyword_score"] is not None else "N/A"
        lines.append(
            f"{r['query'][:48]:<50} {r['agent_type']:<12} "
            f"{r['total_steps']:<8} {kw:<10} {r['avg_llm_score']:.2f}"
        )

    yield "\n".join(lines), link_html


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

def create_demo():
    with gr.Blocks(title="OpenEnv Research Agent Demo") as demo:
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
                    "while LLM judge stays honest."
                )
                with gr.Row():
                    comp_query = gr.Textbox(
                        label="Research Question",
                        placeholder="What is Model Context Protocol (MCP)?",
                        scale=3,
                    )
                    comp_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps", scale=1)
                comp_btn = gr.Button("Run Comparison", variant="primary")

                with gr.Row():
                    trad_chart = gr.Plot(label="Traditional RL Agent")
                    oe_chart = gr.Plot(label="OpenEnv Agent")

                with gr.Row():
                    trad_log = gr.Textbox(label="Traditional Agent Steps", lines=8, interactive=False)
                    oe_log = gr.Textbox(label="OpenEnv Agent Steps", lines=8, interactive=False)

                comp_btn.click(
                    fn=run_comparison,
                    inputs=[comp_query, comp_steps],
                    outputs=[trad_chart, oe_chart, trad_log, oe_log],
                )

            # --- Tab 2: Agent Race ---
            with gr.Tab("Agent Race"):
                gr.Markdown(
                    "3 OpenEnv agents race on the same question using concurrent sessions. "
                    "Demonstrates OpenEnv's `SUPPORTS_CONCURRENT_SESSIONS`."
                )
                with gr.Row():
                    race_query = gr.Textbox(
                        label="Research Question",
                        placeholder="How does retrieval-augmented generation work?",
                        scale=3,
                    )
                    race_steps = gr.Slider(minimum=3, maximum=15, value=6, step=1, label="Max Steps", scale=1)
                race_btn = gr.Button("Start Race", variant="primary")
                race_board = gr.Textbox(label="Live Scoreboard", lines=10, interactive=False)

                race_btn.click(
                    fn=run_race,
                    inputs=[race_query, race_steps],
                    outputs=[race_board],
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
                fanout_results = gr.Textbox(label="Results", lines=15, interactive=False)

                fanout_btn.click(
                    fn=run_flyte_fanout,
                    inputs=[fanout_queries, fanout_steps],
                    outputs=[fanout_results, fanout_link],
                )

        gr.Examples(
            examples=[
                ["What is Model Context Protocol (MCP)?"],
                ["How does retrieval-augmented generation work?"],
                ["What are the key differences between OpenEnv and OpenAI Gym?"],
            ],
            inputs=comp_query,
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
