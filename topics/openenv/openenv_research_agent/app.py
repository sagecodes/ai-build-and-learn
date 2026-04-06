"""
Gradio UI for the OpenEnv Research Agent demo.

Three demo modes accessible via tabs:

  Tab 1 — Side-by-Side Comparison
    Run both agents on the same question simultaneously.
    Shows a unified reward chart (Plotly) and HTML step logs updating in real time.
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
from agents.openenv_agent import OpenEnvAgent
from agents.traditional_agent import TraditionalAgent

load_dotenv()

RUN_MODE = os.getenv("RUN_MODE", "remote")

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
# HTML step log helpers
# ---------------------------------------------------------------------------

def _score_badge(score: float, color: str) -> str:
    return (
        f'<span style="background:{color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-weight:bold;font-size:0.85em">{score:.2f}</span>'
    )


def _trad_step_html(step: int, tool: str, kw: float, query_used: str, matched: list[str]) -> str:
    badge = _score_badge(kw, "#e67e22")
    q = query_used[:90] + ("..." if len(query_used) > 90 else "")
    kw_tags = " ".join(
        f'<span style="background:#fdebd0;color:#784212;padding:1px 5px;'
        f'border-radius:3px;font-size:0.8em">{k}</span>'
        for k in matched
    )
    return (
        f'<div style="border-left:4px solid #e67e22;padding:6px 10px;margin:6px 0;background:#fdf6ec">'
        f'<div style="font-weight:bold">Step {step}: {tool} &nbsp; {badge}</div>'
        f'<div style="color:#555;font-size:0.85em;margin-top:3px">&#128269; &nbsp;<em>{q}</em></div>'
        f'{"<div style=\'margin-top:4px\'>Matched: " + kw_tags + "</div>" if matched else ""}'
        f'</div>'
    )


def _oe_step_html(step: int, tool: str, tool_args: dict, preview: str) -> str:
    detail = ""
    if tool == "tavily_search":
        q = tool_args.get("query", "")
        if q:
            detail = f'<div style="color:#555;font-size:0.85em;margin-top:3px">&#128269; &nbsp;<em>{q[:90]}{"..." if len(q) > 90 else ""}</em></div>'
    elif tool == "tavily_extract":
        urls = tool_args.get("urls", [])
        if urls:
            u = urls[0]
            extra = f" <span style='color:#888'>+{len(urls)-1} more</span>" if len(urls) > 1 else ""
            detail = f'<div style="color:#555;font-size:0.85em;margin-top:3px">&#128196; &nbsp;<a href="{u}" target="_blank" style="color:#2471a3">{u[:80]}</a>{extra}</div>'
    elif tool == "tavily_crawl":
        url = tool_args.get("url", "")
        if url:
            detail = f'<div style="color:#555;font-size:0.85em;margin-top:3px">&#128375; &nbsp;<a href="{url}" target="_blank" style="color:#2471a3">{url[:80]}</a></div>'

    found = ""
    if preview:
        p = preview[:120] + ("..." if len(preview) > 120 else "")
        found = f'<div style="color:#666;font-size:0.8em;margin-top:3px;font-style:italic">Found: {p}</div>'

    tool_color = {"tavily_search": "#2980b9", "tavily_extract": "#8e44ad", "tavily_crawl": "#16a085"}.get(tool, "#555")
    tool_badge = f'<span style="background:{tool_color};color:white;padding:1px 7px;border-radius:4px;font-size:0.82em">{tool}</span>'

    return (
        f'<div style="border-left:4px solid #2471a3;padding:6px 10px;margin:6px 0;background:#eaf4fb">'
        f'<div style="font-weight:bold">Step {step}: &nbsp;{tool_badge}</div>'
        f'{detail}{found}'
        f'</div>'
    )


def _summary_html(title: str, color: str, lines: list[str]) -> str:
    content = "".join(f'<div style="margin:2px 0">{l}</div>' for l in lines)
    return (
        f'<div style="border-left:4px solid {color};padding:8px 12px;'
        f'margin:10px 0;background:#f9f9f9;border-radius:4px">'
        f'<div style="font-weight:bold;margin-bottom:4px">{title}</div>'
        f'{content}</div>'
    )


def _final_score_html(label: str, score: float, color: str) -> str:
    return (
        f'<div style="text-align:center;padding:12px;margin:8px 0;'
        f'background:{color};border-radius:8px;color:white">'
        f'<div style="font-size:0.9em">{label}</div>'
        f'<div style="font-size:2em;font-weight:bold">{score:.2f}</div>'
        f'</div>'
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
    Step chart showing per-step keyword scores as a line, with final LLM
    judge scores for both agents as horizontal reference lines.
    Makes the reward hacking gap visually obvious on a single axis.
    """
    fig = go.Figure()

    max_x = max(len(kw_scores), 1)

    # Per-step keyword scores as an orange dashed line
    if kw_scores:
        steps = list(range(1, len(kw_scores) + 1))
        fig.add_trace(go.Scatter(
            x=steps,
            y=kw_scores,
            mode="lines+markers",
            name="Keyword Score per step",
            line=dict(color="#e67e22", dash="dash", width=2),
            marker=dict(size=7),
        ))
        max_x = len(kw_scores)

    # Traditional final LLM score as a solid red horizontal line
    if trad_final is not None:
        fig.add_hline(
            y=trad_final,
            line=dict(color="#c0392b", width=2.5, dash="dot"),
            annotation_text=f"Traditional Final LLM: {trad_final:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#c0392b", size=11),
        )

    # OpenEnv final LLM score as a solid green horizontal line
    if oe_final is not None:
        fig.add_hline(
            y=oe_final,
            line=dict(color="#1a7a4a", width=2.5, dash="dot"),
            annotation_text=f"OpenEnv Final LLM: {oe_final:.2f}",
            annotation_position="bottom left",
            annotation_font=dict(color="#1a7a4a", size=11),
        )

    fig.update_layout(
        title=title,
        xaxis=dict(title="Step", dtick=1, range=[0.5, max_x + 0.5]),
        yaxis=dict(range=[0, 1.05], title="Score (0-1)"),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
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
            _empty_chart("Reward Comparison"),
            "Enter a research question above.",
            "Enter a research question above.",
        )
        return

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
                        _final_score_html("Final LLM Judge Score", trad_final_llm, "#c0392b")
                    )
                    trad_done = True
                else:
                    kw = step.get("keyword_score", 0.0)
                    trad_kw.append(kw)
                    trad_html_blocks.append(_trad_step_html(
                        step["step"],
                        step["tool_name"],
                        kw,
                        step.get("query_used", ""),
                        step.get("matched_keywords", []),
                    ))
            except StopIteration:
                trad_done = True

        if not oe_done:
            try:
                step = next(oe_gen)
                if step["tool_name"] == "final_judgment":
                    oe_final_llm = step.get("llm_final_score", 0.0)
                    oe_html_blocks.append(
                        _final_score_html("Final LLM Judge Score", oe_final_llm, "#1a7a4a")
                    )
                    oe_done = True
                else:
                    oe_html_blocks.append(_oe_step_html(
                        step["step"],
                        step["tool_name"],
                        step.get("tool_args", {}),
                        step.get("result_preview", ""),
                    ))
            except StopIteration:
                oe_done = True

        chart = _build_final_chart(
            trad_kw, trad_final_llm, oe_final_llm,
            "Reward Comparison — Traditional RL vs OpenEnv"
        )

        yield (
            chart,
            "".join(trad_html_blocks),
            "".join(oe_html_blocks),
        )

    # Final summary
    avg_kw = sum(trad_kw) / max(len(trad_kw), 1)
    trad_llm_val = trad_final_llm if trad_final_llm is not None else 0.0
    oe_llm_val = oe_final_llm if oe_final_llm is not None else 0.0
    gap = avg_kw - trad_llm_val
    oe_advantage = oe_llm_val - trad_llm_val

    trad_html_blocks.append(_summary_html(
        "Summary — Traditional RL", "#e67e22", [
            f"Avg Keyword Score: <b>{avg_kw:.2f}</b> &larr; what the agent optimized",
            f"Final LLM Judge Score: <b>{trad_llm_val:.2f}</b> &larr; actual quality",
            f"Overestimation gap: <b>{gap:.2f}</b> &larr; this is reward hacking",
        ]
    ))
    oe_html_blocks.append(_summary_html(
        "Summary — OpenEnv Agent", "#2471a3", [
            f"Final LLM Judge Score: <b>{oe_llm_val:.2f}</b>",
            f"OpenEnv advantage: <b>+{oe_advantage:.2f}</b> over traditional",
        ]
    ))

    yield (
        _build_final_chart(trad_kw, trad_final_llm, oe_final_llm, "FINAL — Reward Hacking Gap"),
        "".join(trad_html_blocks),
        "".join(oe_html_blocks),
    )


# ---------------------------------------------------------------------------
# Tab 2: Agent Race
# ---------------------------------------------------------------------------

def run_race(query: str, max_steps: int):
    """
    Run 3 OpenEnv agents concurrently on the same query.
    Yields an HTML scoreboard table updated after each agent step,
    followed by a summary card with narrative, scores, and session stats.
    """
    if not query.strip():
        yield "Enter a research question to start the race."
        return

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

        # Build HTML scoreboard
        rows = ""
        for i in range(num_agents):
            if winner == i:
                status = '<span style="background:#1a7a4a;color:white;padding:2px 8px;border-radius:4px;font-weight:bold">WINNER &#127942;</span>'
                row_bg = "#eafaf1"
            elif done[i]:
                status = '<span style="color:#888">done</span>'
                row_bg = "#f9f9f9"
            else:
                status = f'<span style="color:#2471a3">running... ({last_tools[i]})</span>'
                row_bg = "#ffffff"

            score_str = f"{final_scores[i]:.2f}" if final_scores[i] is not None else "-"
            rows += (
                f'<tr style="background:{row_bg}">'
                f'<td style="padding:8px 12px;font-weight:bold;color:#111">Agent {i}</td>'
                f'<td style="padding:8px 12px;color:#333">{step_counts[i]}</td>'
                f'<td style="padding:8px 12px;color:#333">{score_str}</td>'
                f'<td style="padding:8px 12px">{status}</td>'
                f'</tr>'
            )

        board = (
            f'<div style="color:#111">'
            f'<table style="width:100%;border-collapse:collapse;font-family:monospace;color:#111">'
            f'<tr style="background:#2c3e50;color:white">'
            f'<th style="padding:8px 12px;text-align:left;color:white">Agent</th>'
            f'<th style="padding:8px 12px;text-align:left;color:white">Steps</th>'
            f'<th style="padding:8px 12px;text-align:left;color:white">Final LLM Score</th>'
            f'<th style="padding:8px 12px;text-align:left;color:white">Status</th>'
            f'</tr>{rows}</table></div>'
        )

        yield board

    winner_str = f"Agent {winner}" if winner is not None else "unknown"
    all_scores = {i: s for i, s in final_scores.items() if s is not None}
    best_agent = max(all_scores, key=lambda i: all_scores[i]) if all_scores else winner
    best_score = all_scores.get(best_agent, 0.0)
    winner_score = all_scores.get(winner, 0.0)
    avg_score = sum(all_scores.values()) / max(len(all_scores), 1)

    # Narrative — adapts based on whether winner == best scorer
    scores_range = max(all_scores.values()) - min(all_scores.values()) if all_scores else 0
    if best_agent != winner:
        narrative = (
            f"Agent {winner} crossed the finish line first with a score of {winner_score:.2f}, "
            f"but Agent {best_agent} produced the strongest research at {best_score:.2f}. "
            f"Speed vs quality &mdash; the classic concurrent agent trade-off."
        )
    elif scores_range < 0.05:
        narrative = (
            f"Agent {winner} won the race with a score of {winner_score:.2f}. "
            f"All three agents finished with nearly identical quality scores &mdash; "
            f"a tight race from start to finish."
        )
    else:
        narrative = (
            f"Agent {winner} won the race AND scored highest at {winner_score:.2f} &mdash; "
            f"finishing first with the best research quality."
        )

    def _score_color(i):
        if i == best_agent:
            return "#1a7a4a"
        elif i == winner:
            return "#2471a3"
        else:
            return "#888"

    score_cells = "".join(
        f'<td style="padding:8px 16px;text-align:center;color:#111;border-right:1px solid #ddd">'
        f'<div style="font-size:1.6em;font-weight:bold;color:{_score_color(i)}">{all_scores[i]:.2f}</div>'
        f'<div style="font-size:0.85em;color:#555;margin-top:2px">Agent {i}'
        f'{"&nbsp;&#127942;" if i == winner else ""}'
        f'{"&nbsp;&#11088;" if i == best_agent and best_agent != winner else ""}'
        f'</div></td>'
        for i in sorted(all_scores)
    )

    summary = (
        f'<div style="margin-top:14px;border:1px solid #a9dfbf;border-radius:6px;overflow:hidden">'

        f'<div style="background:#1a7a4a;color:white;padding:10px 16px;font-weight:bold;font-size:1.05em">'
        f'Race Complete &mdash; {winner_str} wins'
        f'</div>'

        f'<div style="background:#eafaf1;padding:12px 16px;color:#111;line-height:1.7;border-bottom:1px solid #a9dfbf">'
        f'{narrative}'
        f'</div>'

        f'<div style="background:#f9f9f9;padding:10px 0;border-bottom:1px solid #ddd">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr>'
        f'<td style="padding:6px 16px;color:#333;font-size:0.9em;font-weight:bold;border-right:1px solid #ddd;white-space:nowrap">Final LLM Scores</td>'
        f'{score_cells}'
        f'<td style="padding:8px 16px;color:#111;font-size:0.92em;white-space:nowrap">'
        f'Avg: <b style="color:#111">{avg_score:.2f}</b><br>'
        f'Sessions: <b style="color:#111">3 concurrent</b>'
        f'</td>'
        f'</tr>'
        f'</table>'
        f'</div>'

        f'<div style="background:#2c3e50;padding:12px 16px;color:white;font-size:0.92em;line-height:1.6">'
        f'<b>OpenEnv Concurrent Sessions:</b> All 3 agents ran simultaneously inside a single '
        f'Docker container, each with its own fully isolated session '
        f'(SUPPORTS_CONCURRENT_SESSIONS=True). No shared state, no interference between agents.'
        f'</div>'

        f'</div>'
    )

    yield board + summary


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
    lines = [f"{'Query':<50} {'Agent':<12} {'Steps':<8} {'Keyword Score':<15} {'Final LLM Score'}"]
    lines.append("-" * 105)
    for r in results:
        kw = f"{r['avg_keyword_score']:.2f}" if r["avg_keyword_score"] is not None else "N/A"
        llm = f"{r['llm_final_score']:.2f}" if r["llm_final_score"] is not None else "N/A"
        lines.append(
            f"{r['query'][:48]:<50} {r['agent_type']:<12} "
            f"{r['total_steps']:<8} {kw:<15} {llm}"
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
                    inputs=[comp_query, comp_steps],
                    outputs=[reward_chart, trad_log, oe_log],
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
                race_board = gr.HTML(label="Live Scoreboard")

                race_btn.click(
                    fn=run_race,
                    inputs=[race_query, race_steps],
                    outputs=[race_board],
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
                fanout_results = gr.Textbox(label="Results", lines=15, interactive=False)

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
