"""
UI component builders for the OpenEnv Research Agent demo.

All HTML and Plotly chart construction lives here, keeping app.py focused
on Gradio wiring and agent orchestration.

Sections:
  - Chart builders  — Plotly figures for the reward comparison chart
  - Step card builders — HTML cards for Tab 1 live step logs
  - Score blocks — HTML final score displays
  - Summary builders — Tab 1 agent summary callouts
  - Race builders — Tab 2 scoreboard table and post-race summary card
  - Fanout builders — Tab 3 results table
"""

import plotly.graph_objects as go

# CSS class names mirror styles.css — changes to visual style go there,
# not here (unless adding new structure).

# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def empty_chart(title: str) -> go.Figure:
    """Return a blank placeholder chart before any agent has run."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis_title="Step",
        yaxis_title="Score (0-1)",
        yaxis=dict(range=[0, 1]),
        height=320,
    )
    return fig


def build_reward_chart(
    kw_scores: list[float],
    trad_final: float | None,
    oe_final: float | None,
    title: str,
) -> go.Figure:
    """
    Unified reward comparison chart for Tab 1.

    Shows:
    - Orange dashed line: Traditional agent's per-step keyword scores
    - Red dotted hline: Traditional agent's final LLM judge score
    - Green dotted hline: OpenEnv agent's final LLM judge score

    All on one axis so the reward hacking gap is immediately visible.
    """
    fig = go.Figure()
    max_x = max(len(kw_scores), 1)

    if kw_scores:
        steps = list(range(1, len(kw_scores) + 1))
        fig.add_trace(go.Scatter(
            x=steps,
            y=kw_scores,
            mode="lines+markers",
            name="Keyword Score per step (Traditional)",
            line=dict(color="#e67e22", dash="dash", width=2),
            marker=dict(size=7),
        ))
        max_x = len(kw_scores)

    if trad_final is not None:
        fig.add_hline(
            y=trad_final,
            line=dict(color="#c0392b", width=2.5, dash="dot"),
            annotation_text=f"Traditional Final LLM: {trad_final:.2f}",
            annotation_position="top left",
            annotation_font=dict(color="#c0392b", size=11),
        )

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
# Step card builders
# ---------------------------------------------------------------------------

def trad_step_card(step: int, tool: str, kw: float, query_used: str, matched: list[str]) -> str:
    """
    HTML card for one Traditional agent step.
    Orange left border. Shows keyword score badge, stuffed query, matched keywords.
    """
    badge = f'<span class="badge badge-trad">{kw:.2f}</span>'
    q = query_used[:90] + ("..." if len(query_used) > 90 else "")
    kw_tags = " ".join(
        f'<span class="kw-tag">{k}</span>' for k in matched
    )
    matched_row = f'<div style="margin-top:4px">Matched: {kw_tags}</div>' if matched else ""
    return (
        f'<div class="step-card step-card-trad">'
        f'<div class="step-title">Step {step}: {tool} &nbsp; {badge}</div>'
        f'<div class="step-detail">&#128269; &nbsp;<em>{q}</em></div>'
        f'{matched_row}'
        f'</div>'
    )


def oe_step_card(step: int, tool: str, tool_args: dict, preview: str) -> str:
    """
    HTML card for one OpenEnv agent step.
    Blue left border. Color-coded tool badge, query/URL detail, result preview.
    """
    tool_class = {
        "tavily_search": "tool-search",
        "tavily_extract": "tool-extract",
        "tavily_crawl": "tool-crawl",
    }.get(tool, "")
    tool_badge = f'<span class="tool-badge {tool_class}">{tool}</span>'

    detail = ""
    if tool == "tavily_search":
        q = tool_args.get("query", "")
        if q:
            detail = (
                f'<div class="step-detail">&#128269; &nbsp;'
                f'<em>{q[:90]}{"..." if len(q) > 90 else ""}</em></div>'
            )
    elif tool == "tavily_extract":
        urls = tool_args.get("urls", [])
        if urls:
            extra = f' <span style="color:#888">+{len(urls)-1} more</span>' if len(urls) > 1 else ""
            detail = (
                f'<div class="step-detail">&#128196; &nbsp;'
                f'<a href="{urls[0]}" target="_blank" style="color:#2471a3">'
                f'{urls[0][:80]}</a>{extra}</div>'
            )
    elif tool == "tavily_crawl":
        url = tool_args.get("url", "")
        if url:
            detail = (
                f'<div class="step-detail">&#128375; &nbsp;'
                f'<a href="{url}" target="_blank" style="color:#2471a3">'
                f'{url[:80]}</a></div>'
            )

    found = ""
    if preview:
        p = preview[:120] + ("..." if len(preview) > 120 else "")
        found = f'<div class="step-preview">Found: {p}</div>'

    return (
        f'<div class="step-card step-card-oe">'
        f'<div class="step-title">Step {step}: &nbsp;{tool_badge}</div>'
        f'{detail}{found}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Score blocks
# ---------------------------------------------------------------------------

def final_score_block(label: str, score: float, color: str) -> str:
    """Large centered score block shown at end of each agent's log."""
    return (
        f'<div class="final-score-block" style="background:{color}">'
        f'<div class="final-score-label">{label}</div>'
        f'<div class="final-score-value">{score:.2f}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Summary builders (Tab 1)
# ---------------------------------------------------------------------------

def agent_summary(title: str, color: str, lines: list[str]) -> str:
    """Callout box summarising one agent's episode results."""
    content = "".join(f'<div style="margin:2px 0;color:#111">{l}</div>' for l in lines)
    return (
        f'<div class="summary-box" style="border-color:{color}">'
        f'<div class="summary-title" style="color:{color}">{title}</div>'
        f'{content}'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Race builders (Tab 2)
# ---------------------------------------------------------------------------

def race_scoreboard(
    step_counts: dict[int, int],
    final_scores: dict[int, float | None],
    done: dict[int, bool],
    last_tools: dict[int, str],
    winner: int | None,
) -> str:
    """Live HTML scoreboard table for the Agent Race tab."""
    rows = ""
    for i in sorted(step_counts):
        if winner == i:
            row_bg = "#eafaf1"
            status = '<span class="badge badge-oe">WINNER &#127942;</span>'
        elif done[i]:
            row_bg = "#f9f9f9"
            status = '<span style="color:#888">done</span>'
        else:
            row_bg = "#ffffff"
            status = f'<span style="color:#2471a3">running... ({last_tools[i]})</span>'

        score_str = f"{final_scores[i]:.2f}" if final_scores[i] is not None else "-"
        rows += (
            f'<tr style="background:{row_bg}">'
            f'<td style="padding:8px 12px;font-weight:bold;color:#111">Agent {i}</td>'
            f'<td style="padding:8px 12px;color:#333">{step_counts[i]}</td>'
            f'<td style="padding:8px 12px;color:#333">{score_str}</td>'
            f'<td style="padding:8px 12px">{status}</td>'
            f'</tr>'
        )

    return (
        f'<div style="color:#111">'
        f'<table class="race-table">'
        f'<tr>'
        f'<th>Agent</th><th>Steps</th><th>Final LLM Score</th><th>Status</th>'
        f'</tr>{rows}'
        f'</table></div>'
    )


def race_summary(
    winner: int | None,
    final_scores: dict[int, float | None],
) -> str:
    """
    Post-race summary card for the Agent Race tab.

    Includes an adaptive narrative (speed-vs-quality or dominant-win),
    per-agent score columns, aggregate stats, and a footer explaining
    OpenEnv's concurrent session isolation.
    """
    all_scores = {i: s for i, s in final_scores.items() if s is not None}
    if not all_scores:
        return ""

    winner_str = f"Agent {winner}" if winner is not None else "unknown"
    best_agent = max(all_scores, key=lambda i: all_scores[i])
    best_score = all_scores[best_agent]
    winner_score = all_scores.get(winner, 0.0)
    avg_score = sum(all_scores.values()) / len(all_scores)
    scores_range = max(all_scores.values()) - min(all_scores.values())

    # Adaptive narrative
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

    def _score_color(i: int) -> str:
        if i == best_agent:
            return "#1a7a4a"
        if i == winner:
            return "#2471a3"
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

    return (
        f'<div class="race-card">'

        f'<div class="race-card-header">Race Complete &mdash; {winner_str} wins</div>'

        f'<div class="race-card-narrative">{narrative}</div>'

        f'<div class="race-card-scores">'
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr>'
        f'<td style="padding:6px 16px;color:#333;font-size:0.9em;font-weight:bold;'
        f'border-right:1px solid #ddd;white-space:nowrap">Final LLM Scores</td>'
        f'{score_cells}'
        f'<td style="padding:8px 16px;color:#111;font-size:0.92em;white-space:nowrap">'
        f'Avg: <b style="color:#111">{avg_score:.2f}</b><br>'
        f'Sessions: <b style="color:#111">3 concurrent</b>'
        f'</td>'
        f'</tr>'
        f'</table>'
        f'</div>'

        f'<div class="race-card-footer">'
        f'<b>OpenEnv Concurrent Sessions:</b> All 3 agents ran simultaneously inside a single '
        f'Docker container, each with its own fully isolated session '
        f'(SUPPORTS_CONCURRENT_SESSIONS=True). No shared state, no interference between agents.'
        f'</div>'

        f'</div>'
    )


# ---------------------------------------------------------------------------
# Fanout results table (Tab 3)
# ---------------------------------------------------------------------------

def fanout_results_table(results: list[dict]) -> str:
    """
    HTML results table for the Parallel Flyte Fan-out tab.

    Rows are grouped in pairs (openenv + traditional) per query.
    Alternating query groups get a subtle background tint so related rows
    read as a unit. Agent type is shown as a colour-coded badge.
    Scores are colour-coded: green = good (≥0.6), amber = mid (≥0.4), red = low.
    """
    def _score_color(score: float | None) -> str:
        if score is None:
            return "#888"
        if score >= 0.6:
            return "#1a7a4a"
        if score >= 0.4:
            return "#b7770d"
        return "#c0392b"

    # Group into (openenv, traditional) pairs keyed by query
    pairs: dict[str, dict] = {}
    for r in results:
        pairs.setdefault(r["query"], {})[r["agent_type"]] = r

    rows = ""
    for idx, (query, agents) in enumerate(pairs.items()):
        row_bg = "#f4f6f7" if idx % 2 == 0 else "#ffffff"
        for agent_type in ("openenv", "traditional"):
            r = agents.get(agent_type)
            if r is None:
                continue

            badge_class = "badge-oe" if agent_type == "openenv" else "badge-trad"
            kw_str = f"{r['avg_keyword_score']:.2f}" if r["avg_keyword_score"] is not None else "—"
            llm = r["llm_final_score"]
            llm_str = f"{llm:.2f}" if llm is not None else "—"
            llm_color = _score_color(llm)

            # Only show query text on the first row of each pair
            query_cell = (
                f'<td style="padding:8px 12px;color:#111;vertical-align:top">{query}</td>'
                if agent_type == "openenv"
                else '<td style="padding:8px 12px"></td>'
            )

            rows += (
                f'<tr style="background:{row_bg}">'
                f'{query_cell}'
                f'<td style="padding:8px 12px">'
                f'<span class="badge {badge_class}">{agent_type}</span></td>'
                f'<td style="padding:8px 12px;color:#333;text-align:center">{r["total_steps"]}</td>'
                f'<td style="padding:8px 12px;color:#333;text-align:center">{kw_str}</td>'
                f'<td style="padding:8px 12px;font-weight:bold;text-align:center;'
                f'color:{llm_color}">{llm_str}</td>'
                f'</tr>'
            )

    return (
        f'<div style="color:#111">'
        f'<table class="race-table" style="font-family:inherit">'
        f'<tr>'
        f'<th style="width:45%">Query</th>'
        f'<th>Agent</th>'
        f'<th style="text-align:center">Steps</th>'
        f'<th style="text-align:center">Keyword Score</th>'
        f'<th style="text-align:center">Final LLM Score</th>'
        f'</tr>'
        f'{rows}'
        f'</table>'
        f'</div>'
    )
