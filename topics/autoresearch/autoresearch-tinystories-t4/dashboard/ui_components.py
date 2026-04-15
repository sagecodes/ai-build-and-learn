"""
ui_components.py — Reusable HTML and chart builders for the AutoResearch dashboard.

All functions return either an HTML string or a Plotly figure.
No Gradio imports here — this module has no UI framework dependency.
No Firestore imports here — receives data as plain dicts.

Color constants are defined once at the top and reused throughout.
"""

import plotly.graph_objects as go
from typing import Optional

# ── Color constants ───────────────────────────────────────────────────────────

_TEXT   = "#ccc"
_TEXT2  = "#aaa"
_MUTED  = "#888"
_DIM    = "#555"
_BRIGHT = "#e0e0e0"

_BG     = "#1a1a1a"
_BG2    = "#141414"
_BG3    = "#1e1e1e"
_BORDER = "#2a2a2a"

_GOOD   = "#1a7a4a"
_WARN   = "#b7770d"
_BAD    = "#8b2020"
_INFO   = "#2471a3"

_GREEN  = "#2ecc71"
_RED    = "#e74c3c"

# Plotly dark layout shared across all charts
_CHART_LAYOUT = dict(
    paper_bgcolor="#0f0f0f",
    plot_bgcolor="#0f0f0f",
    font=dict(color=_TEXT2, size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    legend=dict(bgcolor="#141414", bordercolor=_BORDER, borderwidth=1),
)
_AXIS = dict(gridcolor=_BORDER, zerolinecolor=_BORDER, color=_TEXT2)


# ── Header ────────────────────────────────────────────────────────────────────

def app_header() -> str:
    """Render the top header bar."""
    return (
        '<div class="app-header">'
        '<span class="app-title">AutoResearch</span>'
        '<span class="app-tagline">Overnight GPT training · TinyStories · T4</span>'
        '</div>'
    )


# ── Stat cards ────────────────────────────────────────────────────────────────

def stat_card(label: str, value: str, sub: str = "", style: str = "stat-neut") -> str:
    """
    Render a single metric stat card.

    Args:
        label : Short uppercase label (e.g. "Val BPB")
        value : Large display value (e.g. "1.742")
        sub   : Small subtitle line (e.g. "↓ 0.041 from baseline")
        style : CSS modifier class — stat-good, stat-warn, stat-info, stat-neut
    """
    return (
        f'<div class="stat-card {style}">'
        f'<div class="stat-label">{label}</div>'
        f'<div class="stat-value">{value}</div>'
        f'{"<div class=stat-sub>" + sub + "</div>" if sub else ""}'
        f'</div>'
    )


def stat_row(experiments: list[dict], run: Optional[dict] = None) -> str:
    """
    Render the top stat row from a list of experiment dicts.

    Shows: current val_bpb, total experiments, kept count, success rate.
    """
    if not experiments:
        return (
            '<div class="stat-row">'
            + stat_card("Val BPB", "—", "no data yet", "stat-neut")
            + stat_card("Experiments", "0", "", "stat-neut")
            + stat_card("Kept", "0", "", "stat-neut")
            + stat_card("Success Rate", "—", "", "stat-neut")
            + '</div>'
        )

    total = len(experiments)
    kept = sum(1 for e in experiments if e.get("kept"))
    success_rate = kept / total if total > 0 else 0.0

    baseline = experiments[0]["val_bpb_before"]
    current = experiments[-1]["val_bpb_after"]
    improvement = baseline - current

    bpb_sub = f"↓ {improvement:.4f} from {baseline:.4f}" if improvement > 0 else f"baseline {baseline:.4f}"
    bpb_style = "stat-good" if improvement > 0.005 else "stat-warn" if improvement > 0 else "stat-neut"

    rate_style = "stat-good" if success_rate >= 0.4 else "stat-warn" if success_rate >= 0.2 else "stat-neut"

    return (
        '<div class="stat-row">'
        + stat_card("Current Val BPB", f"{current:.4f}", bpb_sub, bpb_style)
        + stat_card("Experiments", str(total), f"{run['config'].get('run_hours', '?')}h budget" if run else "", "stat-info")
        + stat_card("Kept", str(kept), f"{total - kept} reverted", "stat-good" if kept > 0 else "stat-neut")
        + stat_card("Success Rate", f"{success_rate:.0%}", "changes that improved bpb", rate_style)
        + '</div>'
    )


# ── Val BPB chart ─────────────────────────────────────────────────────────────

def val_bpb_chart(experiments: list[dict]) -> go.Figure:
    """
    Plotly line chart showing val_bpb progression across experiments.

    - Green dots: kept experiments (improvement)
    - Red dots: reverted experiments (regression or no change)
    - Dashed line: baseline val_bpb
    """
    fig = go.Figure()

    if not experiments:
        fig.update_layout(
            title="val_bpb Progression",
            xaxis=dict(title="Experiment", **_AXIS),
            yaxis=dict(title="val_bpb", **_AXIS),
            **_CHART_LAYOUT,
        )
        return fig

    xs = [e["experiment_number"] for e in experiments]
    ys_after = [e["val_bpb_after"] for e in experiments]

    kept_xs = [e["experiment_number"] for e in experiments if e.get("kept")]
    kept_ys = [e["val_bpb_after"] for e in experiments if e.get("kept")]

    reverted_xs = [e["experiment_number"] for e in experiments if not e.get("kept")]
    reverted_ys = [e["val_bpb_after"] for e in experiments if not e.get("kept")]

    baseline = experiments[0]["val_bpb_before"]

    # Connecting line
    fig.add_trace(go.Scatter(
        x=xs, y=ys_after,
        mode="lines",
        line=dict(color=_BORDER, width=1),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Kept experiments — green
    fig.add_trace(go.Scatter(
        x=kept_xs, y=kept_ys,
        mode="markers",
        name="Kept",
        marker=dict(color=_GREEN, size=8, symbol="circle"),
        hovertemplate="Exp %{x}<br>val_bpb=%{y:.6f}<extra>Kept</extra>",
    ))

    # Reverted experiments — red
    fig.add_trace(go.Scatter(
        x=reverted_xs, y=reverted_ys,
        mode="markers",
        name="Reverted",
        marker=dict(color=_RED, size=6, symbol="x"),
        hovertemplate="Exp %{x}<br>val_bpb=%{y:.6f}<extra>Reverted</extra>",
    ))

    # Baseline dashed line
    fig.add_hline(
        y=baseline,
        line_dash="dash",
        line_color=_DIM,
        annotation_text=f"baseline {baseline:.4f}",
        annotation_font_color=_DIM,
    )

    fig.update_layout(
        title="val_bpb Progression",
        xaxis=dict(title="Experiment #", **_AXIS),
        yaxis=dict(title="val_bpb (lower = better)", **_AXIS),
        **_CHART_LAYOUT,
    )
    return fig


# ── Experiment log table ──────────────────────────────────────────────────────

def experiment_table(experiments: list[dict]) -> str:
    """
    Render the full experiment log as an HTML table.

    Columns: #, Status, val_bpb before→after, delta, duration, description.
    Most recent experiments appear first.
    """
    if not experiments:
        return '<p style="color:#555;font-size:0.88em">No experiments logged yet.</p>'

    rows = []
    for exp in reversed(experiments):
        kept = exp.get("kept", False)
        badge = '<span class="badge badge-kept">KEPT</span>' if kept else '<span class="badge badge-reverted">REVERTED</span>'
        delta = exp.get("delta", 0.0)
        delta_class = "exp-delta-good" if delta < 0 else "exp-delta-bad"
        delta_str = f"{delta:+.4f}"
        duration = exp.get("duration_seconds", 0)
        desc = exp.get("change_description", "")[:120]

        rows.append(
            f"<tr>"
            f'<td class="exp-num">{exp.get("experiment_number", "")}</td>'
            f"<td>{badge}</td>"
            f'<td style="color:{_MUTED}">{exp.get("val_bpb_before", 0):.4f} → {exp.get("val_bpb_after", 0):.4f}</td>'
            f'<td class="{delta_class}">{delta_str}</td>'
            f'<td style="color:{_DIM}">{duration:.0f}s</td>'
            f'<td class="exp-desc">{desc}</td>'
            f"</tr>"
        )

    return (
        '<table class="exp-table">'
        "<thead><tr>"
        "<th>#</th><th>Status</th><th>val_bpb</th><th>Delta</th><th>Duration</th><th>Change</th>"
        "</tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody>"
        "</table>"
    )


# ── Run summary card ──────────────────────────────────────────────────────────

def run_summary_card(experiments: list[dict]) -> str:
    """
    Render a narrative summary card for the completed run.

    Shows starting/ending val_bpb, total improvement, best single experiment.
    """
    if not experiments:
        return ""

    total = len(experiments)
    kept = [e for e in experiments if e.get("kept")]
    baseline = experiments[0]["val_bpb_before"]
    final = experiments[-1]["val_bpb_after"]
    improvement = baseline - final

    best = min(experiments, key=lambda e: e.get("delta", 0))
    best_desc = best.get("change_description", "")[:100]

    trend = "improved" if improvement > 0.005 else "stayed flat" if abs(improvement) <= 0.005 else "got slightly worse"
    trend_color = _GREEN if improvement > 0.005 else _MUTED if abs(improvement) <= 0.005 else _RED

    return (
        '<div class="summary-card">'
        '<div class="summary-header">Run Summary</div>'
        '<div class="summary-body">'
        f'<b>val_bpb {trend}</b> overnight — '
        f'started at <b>{baseline:.4f}</b>, ended at '
        f'<b style="color:{trend_color}">{final:.4f}</b> '
        f'({"↓" if improvement > 0 else "↑"} {abs(improvement):.4f}).'
        "<ul>"
        f"<li>Ran <b>{total}</b> experiments · <b>{len(kept)}</b> kept · <b>{total - len(kept)}</b> reverted</li>"
        f"<li>Success rate: <b>{len(kept)/total:.0%}</b></li>"
        f'<li>Best single change: <b>{best.get("delta", 0):+.4f}</b> — {best_desc}</li>'
        "</ul>"
        "</div></div>"
    )


# ── Loading placeholder ───────────────────────────────────────────────────────

def loading_card(message: str = "Loading data from Firestore...") -> str:
    """Placeholder shown while Firestore data is being fetched."""
    return (
        f'<div style="text-align:center;padding:48px 16px;color:{_DIM}">'
        f'<div style="font-size:1.4em;margin-bottom:10px">&#9651;</div>'
        f'<div style="font-size:0.92em">{message}</div>'
        f'</div>'
    )


def empty_chart(title: str = "val_bpb Progression") -> go.Figure:
    """Return an empty dark-themed Plotly chart as a placeholder."""
    fig = go.Figure()
    fig.update_layout(
        title=title,
        xaxis=dict(title="Experiment #", **_AXIS),
        yaxis=dict(title="val_bpb", **_AXIS),
        **_CHART_LAYOUT,
    )
    return fig
