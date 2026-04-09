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
# Color palette — one place to change the dark theme across all HTML output
# ---------------------------------------------------------------------------

# Text
_TEXT    = "#ccc"      # primary body text
_TEXT2   = "#aaa"      # secondary / dimmer text
_MUTED   = "#888"      # disabled / placeholder
_DIM     = "#555"      # very muted labels and captions
_BRIGHT  = "#e0e0e0"   # highlighted / bright text

# Backgrounds
_BG      = "#1a1a1a"   # card / surface background
_BG2     = "#141414"   # deeper / alternate row background
_BG3     = "#1e1e1e"   # section header background

# Borders
_BORDER  = "#2a2a2a"   # standard border / divider

# Semantic score colors
_GOOD    = "#1a7a4a"   # good score / OpenEnv win (green)
_WARN    = "#b7770d"   # mid score / caution (amber)
_BAD     = "#c0392b"   # low score / Traditional (red)

# Agent identity colors
_TRAD    = "#e67e22"   # Traditional RL agent (orange)
_OE      = "#2471a3"   # OpenEnv agent (blue)
_ACCENT  = "#e07b39"   # primary UI accent (orange)

# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

_DARK = dict(
    paper_bgcolor=_BG,
    plot_bgcolor=_BG2,
    font=dict(color=_TEXT2),
)

_AXIS = dict(gridcolor=_BORDER, zerolinecolor=_BORDER, color=_TEXT2)


def empty_chart(title: str) -> go.Figure:
    """Return a blank placeholder chart before any agent has run."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccc")),
        xaxis=dict(title="Step", **_AXIS),
        yaxis=dict(title="Score (0-1)", range=[0, 1], **_AXIS),
        height=320,
        **_DARK,
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
            line=dict(color=_TRAD, dash="dash", width=2),
            marker=dict(size=7),
        ))
        max_x = len(kw_scores)

    if trad_final is not None:
        fig.add_hline(
            y=trad_final,
            line=dict(color=_BAD, width=2.5, dash="dot"),
            annotation_text=f"Traditional Final LLM: {trad_final:.2f}",
            annotation_position="top left",
            annotation_font=dict(color=_BAD, size=11),
        )

    if oe_final is not None:
        fig.add_hline(
            y=oe_final,
            line=dict(color=_GOOD, width=2.5, dash="dot"),
            annotation_text=f"OpenEnv Final LLM: {oe_final:.2f}",
            annotation_position="bottom left",
            annotation_font=dict(color=_GOOD, size=11),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccc")),
        xaxis=dict(title="Step", dtick=1, range=[0.5, max_x + 0.5], **_AXIS),
        yaxis=dict(title="Score (0-1)", range=[0, 1.05], **_AXIS),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(color="#aaa"), bgcolor="rgba(0,0,0,0)"),
        **_DARK,
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
            extra = f' <span style="color:{_MUTED}">+{len(urls)-1} more</span>' if len(urls) > 1 else ""
            detail = (
                f'<div class="step-detail">&#128196; &nbsp;'
                f'<a href="{urls[0]}" target="_blank" style="color:{_OE}">'
                f'{urls[0][:80]}</a>{extra}</div>'
            )
    elif tool == "tavily_crawl":
        url = tool_args.get("url", "")
        if url:
            detail = (
                f'<div class="step-detail">&#128375; &nbsp;'
                f'<a href="{url}" target="_blank" style="color:{_OE}">'
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
    content = "".join(f'<div style="margin:2px 0;color:{_TEXT}">{l}</div>' for l in lines)
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
            row_bg = "#0f2018"
            status = '<span class="badge badge-oe">WINNER &#127942;</span>'
        elif done[i]:
            row_bg = _BG
            status = '<span style="color:{_DIM}">done</span>'
        else:
            row_bg = _BG2
            status = f'<span style="color:{_OE}">running... ({last_tools[i]})</span>'

        score_str = f"{final_scores[i]:.2f}" if final_scores[i] is not None else "-"
        rows += (
            f'<tr style="background:{row_bg}">'
            f'<td style="padding:8px 12px;font-weight:bold;color:{_TEXT}">Agent {i}</td>'
            f'<td style="padding:8px 12px;color:{_TEXT2}">{step_counts[i]}</td>'
            f'<td style="padding:8px 12px;color:{_TEXT2}">{score_str}</td>'
            f'<td style="padding:8px 12px">{status}</td>'
            f'</tr>'
        )

    return (
        f'<div style="color:{_TEXT}">'
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
            return _GOOD
        if i == winner:
            return _OE
        return "#888"

    score_cells = "".join(
        f'<td style="padding:8px 16px;text-align:center;color:#ccc;border-right:1px solid #333">'
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
        f'<td style="padding:6px 16px;color:#aaa;font-size:0.9em;font-weight:bold;'
        f'border-right:1px solid #333;white-space:nowrap">Final LLM Scores</td>'
        f'{score_cells}'
        f'<td style="padding:8px 16px;color:#ccc;font-size:0.92em;white-space:nowrap">'
        f'Avg: <b style="color:{_TEXT}">{avg_score:.2f}</b><br>'
        f'Sessions: <b style="color:{_TEXT}">3 concurrent</b>'
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
            return _GOOD
        if score >= 0.4:
            return _WARN
        return _BAD

    # Group into (openenv, traditional) pairs keyed by query
    pairs: dict[str, dict] = {}
    for r in results:
        pairs.setdefault(r["query"], {})[r["agent_type"]] = r

    rows = ""
    for idx, (query, agents) in enumerate(pairs.items()):
        row_bg = _BG if idx % 2 == 0 else _BG2
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
                f'<td style="padding:8px 12px;color:#ccc;vertical-align:top;background:{row_bg}">{query}</td>'
                if agent_type == "openenv"
                else f'<td style="padding:8px 12px;background:{row_bg}"></td>'
            )

            rows += (
                f'<tr style="background:{row_bg}">'
                f'{query_cell}'
                f'<td style="padding:8px 12px">'
                f'<span class="badge {badge_class}">{agent_type}</span></td>'
                f'<td style="padding:8px 12px;color:#aaa;text-align:center">{r["total_steps"]}</td>'
                f'<td style="padding:8px 12px;color:#aaa;text-align:center">{kw_str}</td>'
                f'<td style="padding:8px 12px;font-weight:bold;text-align:center;'
                f'color:{llm_color}">{llm_str}</td>'
                f'</tr>'
            )

    return (
        f'<div style="color:{_TEXT}">'
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


# ---------------------------------------------------------------------------
# Environment state card (Tab 1) — output of client.state
# ---------------------------------------------------------------------------

def env_state_card(state: dict) -> str:
    """
    HTML card showing the OpenEnv episode state returned by client.state.

    Displays tool_usage counts, total accumulated reward, and step progress.
    Called after the final step so the full episode record is visible.
    """
    if not state:
        return ""

    tool_usage = state.get("tool_usage", {})
    total_reward = state.get("total_reward", 0.0)
    step = state.get("step", 0)
    max_steps = state.get("max_steps", "?")
    done = state.get("done", False)

    tool_rows = "".join(
        f'<div style="display:flex;justify-content:space-between;padding:3px 0;'
        f'border-bottom:1px solid {_BORDER}">'
        f'<span style="color:{_TEXT2}">{tool}</span>'
        f'<span style="color:#e0e0e0;font-weight:600">{count}x</span>'
        f'</div>'
        for tool, count in tool_usage.items()
    )

    status_color = _GOOD if done else _WARN
    status_text = "complete" if done else "in progress"

    return (
        f'<div style="margin:10px 0;border:1px solid {_BORDER};border-radius:6px;overflow:hidden">'
        f'<div style="background:#1e1e1e;padding:8px 12px;font-size:0.78em;font-weight:700;'
        f'text-transform:uppercase;letter-spacing:0.07em;color:#555;border-bottom:1px solid {_BORDER}">'
        f'&#128202; client.state &mdash; Episode Snapshot</div>'
        f'<div style="background:#141414;padding:10px 12px">'
        f'<div style="display:flex;gap:20px;margin-bottom:10px">'
        f'<div style="text-align:center">'
        f'<div style="font-size:1.4em;font-weight:700;color:{_BRIGHT}">{step}/{max_steps}</div>'
        f'<div style="font-size:0.75em;color:#555;margin-top:2px">Steps</div>'
        f'</div>'
        f'<div style="text-align:center">'
        f'<div style="font-size:1.4em;font-weight:700;color:#e07b39">{total_reward:.2f}</div>'
        f'<div style="font-size:0.75em;color:#555;margin-top:2px">Total Reward</div>'
        f'</div>'
        f'<div style="text-align:center">'
        f'<div style="font-size:1.4em;font-weight:700;color:{status_color}">{status_text}</div>'
        f'<div style="font-size:0.75em;color:#555;margin-top:2px">Status</div>'
        f'</div>'
        f'</div>'
        f'<div style="font-size:0.82em;color:#666;margin-bottom:4px;font-weight:600;'
        f'text-transform:uppercase;letter-spacing:0.05em">Tool Usage</div>'
        f'{tool_rows}'
        f'</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Fan-out narrative summary (Tab 3)
# ---------------------------------------------------------------------------

def fanout_narrative_summary(results: list[dict]) -> str:
    """
    Dynamic cross-question narrative for the Parallel Flyte Fan-out tab.

    Aggregates results across all questions to build a pattern-level argument
    for OpenEnv — harder to dismiss than a single run.
    """
    oe = [r for r in results if r["agent_type"] == "openenv" and r["llm_final_score"] is not None]
    trad = [r for r in results if r["agent_type"] == "traditional" and r["llm_final_score"] is not None]

    if not oe or not trad:
        return ""

    n = len(oe)
    avg_oe_llm = sum(r["llm_final_score"] for r in oe) / n
    avg_trad_llm = sum(r["llm_final_score"] for r in trad) / max(len(trad), 1)

    trad_kw_scores = [r["avg_keyword_score"] for r in trad if r["avg_keyword_score"] is not None]
    avg_trad_kw = sum(trad_kw_scores) / max(len(trad_kw_scores), 1)
    kw_range = max(trad_kw_scores) - min(trad_kw_scores) if len(trad_kw_scores) > 1 else 0

    overall_advantage = avg_oe_llm - avg_trad_llm
    kw_overestimate = avg_trad_kw - avg_trad_llm

    # Per-question breakdown
    query_gaps = []
    for oe_r in oe:
        trad_r = next((r for r in trad if r["query"] == oe_r["query"]), None)
        if trad_r:
            query_gaps.append({
                "query": oe_r["query"],
                "oe_llm": oe_r["llm_final_score"],
                "trad_llm": trad_r["llm_final_score"],
                "trad_kw": trad_r["avg_keyword_score"] or 0.0,
                "gap": oe_r["llm_final_score"] - trad_r["llm_final_score"],
            })

    n_wins = sum(1 for q in query_gaps if q["gap"] > 0)
    n_ties = sum(1 for q in query_gaps if abs(q["gap"]) < 0.05)
    best = max(query_gaps, key=lambda q: q["gap"]) if query_gaps else None
    worst = min(query_gaps, key=lambda q: q["gap"]) if query_gaps else None

    # Consistency framing for keyword scores
    if kw_range < 0.03:
        kw_consistency = (
            f"The Traditional agent's keyword score was nearly identical across all {n} questions "
            f"(<b>{min(trad_kw_scores):.2f}</b>–<b>{max(trad_kw_scores):.2f}</b>). "
            f"Different topics, same score — a clear sign the agent learned to game the metric "
            f"rather than adapt its research to the question."
        )
    else:
        kw_consistency = (
            f"The Traditional agent's keyword scores ranged from "
            f"<b>{min(trad_kw_scores):.2f}</b> to <b>{max(trad_kw_scores):.2f}</b> across questions, "
            f"but still overestimated actual research quality by an average of <b>{kw_overestimate:.2f}</b>."
        )

    # Win rate framing
    if n_wins == n:
        win_note = f"OpenEnv outscored Traditional on <b>every question</b> ({n}/{n})."
    elif n_wins > n // 2:
        win_note = f"OpenEnv outscored Traditional on <b>{n_wins} of {n} questions</b>."
    else:
        win_note = f"Results were mixed — OpenEnv won <b>{n_wins} of {n} questions</b>."

    # Per-question rows
    q_rows = ""
    for q in query_gaps:
        short = q["query"][:70] + ("..." if len(q["query"]) > 70 else "")
        gap_color = _GOOD if q["gap"] > 0.05 else _WARN if q["gap"] >= 0 else _BAD
        gap_str = f"+{q['gap']:.2f}" if q["gap"] >= 0 else f"{q['gap']:.2f}"
        q_rows += (
            f'<tr>'
            f'<td style="padding:6px 12px;color:#aaa;font-size:0.88em">{short}</td>'
            f'<td style="padding:6px 12px;text-align:center;color:#1a7a4a;font-weight:600">{q["oe_llm"]:.2f}</td>'
            f'<td style="padding:6px 12px;text-align:center;color:#c0392b;font-weight:600">{q["trad_llm"]:.2f}</td>'
            f'<td style="padding:6px 12px;text-align:center;color:{gap_color};font-weight:700">{gap_str}</td>'
            f'</tr>'
        )

    per_question_table = (
        f'<table style="width:100%;border-collapse:collapse;margin-top:10px">'
        f'<tr>'
        f'<th style="padding:6px 12px;text-align:left;color:#555;font-size:0.75em;'
        f'text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid {_BORDER}">Question</th>'
        f'<th style="padding:6px 12px;text-align:center;color:#555;font-size:0.75em;'
        f'text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid {_BORDER}">OpenEnv</th>'
        f'<th style="padding:6px 12px;text-align:center;color:#555;font-size:0.75em;'
        f'text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid {_BORDER}">Traditional</th>'
        f'<th style="padding:6px 12px;text-align:center;color:#555;font-size:0.75em;'
        f'text-transform:uppercase;letter-spacing:0.06em;border-bottom:1px solid {_BORDER}">Advantage</th>'
        f'</tr>'
        f'{q_rows}'
        f'</table>'
    )

    body = (
        f"{win_note} "
        f"Averaged across all {n} questions, OpenEnv scored <b>{avg_oe_llm:.2f}</b> vs "
        f"Traditional's <b>{avg_trad_llm:.2f}</b> on the LLM judge &mdash; "
        f"a <b>+{overall_advantage:.2f}</b> advantage that held consistently across different topics."
        f"<br><br>"
        f"{kw_consistency}"
        f"<br><br>"
        f"The keyword metric overestimated Traditional's research quality by an average of "
        f"<b>{kw_overestimate:.2f}</b> per question. "
        f"An agent evaluated only on keyword scores would appear to be performing well "
        f"while delivering research that is significantly worse than an OpenEnv agent. "
        f"Across {n} different questions, that gap never closed."
        f"{per_question_table}"
    )

    openenv_how = (
        f"Each question was submitted as an independent Flyte task — "
        f"{n * 2} tasks running in parallel, two agents per question. "
        f"Inside each task, the OpenEnv environment ran its full episode contract: "
        f"<b>reset()</b> to start a clean session, <b>step()</b> for each tool call, "
        f"and <b>client.state</b> to capture the final episode record. "
        f"Flyte caches results by <em>(query, agent_type, max_steps)</em> — "
        f"rerunning the same questions at the same step count returns instantly from cache, "
        f"which is why changing max steps by even 1 forces a fresh execution."
    )

    return (
        f'<div class="narrative-card">'
        f'<div class="narrative-header">&#128202; What the pattern shows</div>'
        f'<div class="narrative-body">{body}</div>'
        f'<div class="narrative-header narrative-header-alt">&#9881; How OpenEnv + Flyte works</div>'
        f'<div class="narrative-body">{openenv_how}</div>'
        f'</div>'
    )


# ---------------------------------------------------------------------------
# Narrative summary (Tab 1)
# ---------------------------------------------------------------------------

def narrative_summary(
    avg_kw: float,
    trad_llm: float,
    oe_llm: float,
    gap: float,
    oe_advantage: float,
    trad_steps: int,
    oe_steps: int,
) -> str:
    """
    Dynamic plain-English summary of the Side-by-Side run.

    Always frames the result in terms of why OpenEnv's LLM-as-judge
    reward produces better outcomes than the traditional keyword metric,
    using the actual numbers from the run to make the argument concrete.
    """

    # Gap severity language
    if gap >= 0.5:
        gap_desc = "a stark"
        gap_verb = "completely misrepresents"
    elif gap >= 0.3:
        gap_desc = "a significant"
        gap_verb = "badly overestimates"
    elif gap >= 0.15:
        gap_desc = "a clear"
        gap_verb = "overestimates"
    else:
        gap_desc = "a modest"
        gap_verb = "slightly overestimates"

    # OpenEnv quality language
    if oe_llm >= 0.75:
        oe_quality = "high-quality, substantive research"
    elif oe_llm >= 0.55:
        oe_quality = "solid, well-directed research"
    elif oe_llm >= 0.35:
        oe_quality = "reasonable research with some depth"
    else:
        oe_quality = "research with room for improvement"

    # Tool diversity note (OpenEnv uses more steps = more tool types)
    if oe_steps > trad_steps:
        tool_note = (
            f"OpenEnv used {oe_steps} steps vs {trad_steps} for Traditional. "
            f"Rather than repeating the same search with keyword variations, it chained tools "
            f"purposefully: <b>tavily_search</b> to find candidate pages, "
            f"<b>tavily_extract</b> to pull full content from specific URLs, "
            f"and <b>tavily_crawl</b> to explore related sections &mdash; "
            f"each call informed by what the previous one found. "
        )
    else:
        tool_note = (
            f"Both agents used {oe_steps} steps, but OpenEnv's tool choices were purposeful: "
            f"<b>tavily_search</b> to find URLs, then <b>tavily_extract</b> to read actual page "
            f"content &mdash; not the same query repeated with keyword stuffing. "
        )

    # Advantage framing
    if oe_advantage > 0:
        advantage_note = (
            f"The LLM judge correctly awarded OpenEnv <b>{oe_llm:.2f}</b> vs "
            f"Traditional's <b>{trad_llm:.2f}</b> — "
            f"a <b>+{oe_advantage:.2f}</b> advantage that reflects real research quality, "
            f"not keyword frequency."
        )
    else:
        advantage_note = (
            f"Both agents scored similarly on the LLM judge (<b>{oe_llm:.2f}</b> vs "
            f"<b>{trad_llm:.2f}</b>), but the keyword metric still "
            f"{gap_verb} Traditional's actual quality by <b>{gap:.2f}</b>."
        )

    body = (
        f"The Traditional RL agent scored <b>{avg_kw:.2f}</b> on the keyword metric "
        f"by stuffing its queries with topic words on every step &mdash; "
        f"a textbook example of reward hacking. "
        f"Yet the LLM judge, evaluating the <em>actual research quality</em>, "
        f"scored it just <b>{trad_llm:.2f}</b>. "
        f"That is {gap_desc} gap of <b>{gap:.2f}</b> &mdash; the keyword reward "
        f"{gap_verb} the true value of the research. "
        f"<br><br>"
        f"{tool_note}"
        f"{advantage_note}"
        f"<br><br>"
        f"This is why OpenEnv uses LLM-as-judge rewards: keyword counts "
        f"measure surface-level matching, not understanding. "
        f"An agent that optimises for keyword scores will always game the metric "
        f"rather than answer the question. "
        f"The <b>client.state</b> snapshot above shows exactly what the OpenEnv environment "
        f"recorded &mdash; tool usage counts, accumulated reward, and episode status &mdash; "
        f"making every reward signal inspectable and trustworthy."
    )

    openenv_how = (
        f"OpenEnv wraps each research episode in a standard RL environment contract. "
        f"This demo uses all three core methods:"
        f"<br><br>"
        f"<b>reset(query=...)</b> &mdash; called once at the start of every episode. "
        f"It clears all state (step counter, tool usage, reward history) and sets the "
        f"research question. This is what makes scores comparable across agents and runs &mdash; "
        f"every episode starts from the same clean slate, with no carried-over context."
        f"<br><br>"
        f"<b>step(action)</b> &mdash; called once per tool invocation. It executes the tool, "
        f"records the result in episode history, computes the per-step reward, and returns "
        f"an observation containing the result and a <em>done</em> flag. "
        f"This is the core feedback loop: the agent sees what each tool call actually found "
        f"before deciding what to do next &mdash; search, then extract a specific URL, "
        f"then crawl deeper if needed."
        f"<br><br>"
        f"<b>client.state</b> &mdash; called at episode end to retrieve the full environment "
        f"record: tool usage counts, accumulated reward, step count, and completion status. "
        f"This is the observability layer &mdash; you can inspect exactly what the agent did "
        f"and what the environment scored, making the reward signal debuggable and trustworthy. "
        f"The snapshot above is the live output of this call."
    )

    return (
        f'<div class="narrative-card">'
        f'<div class="narrative-header">&#128214; What just happened</div>'
        f'<div class="narrative-body">{body}</div>'
        f'<div class="narrative-header narrative-header-alt">&#9881; How OpenEnv works</div>'
        f'<div class="narrative-body">{openenv_how}</div>'
        f'</div>'
    )
