"""HTML for the Flyte report: the fly, its brain, and the controls that keep it honest.

Same palette and helpers as topics/rl-mujoco/reports.py and topics/dreamerv3/reports.py,
so a viewer moving between the demos is not re-learning the colours. Charts are
hand-rolled SVG rather than matplotlib: the page already carries video, and a polyline
needs no dependency.

── The order of the page is the argument it is making ──────────────────────────
The clip goes first, and the clip contains its own control: body on top, brain
underneath, so "the fly turned" and "the brain was doing something when it turned" are
one image rather than two claims. Underneath that is the trajectory, which is the only
evidence that the turning went anywhere.

The control runs come next and they are not an appendix. A connectome fly walking to a
pillar is not evidence of anything on its own, because a fly that curves gently to the
left will eventually walk into something on its left. The shuffled-wiring run is what
turns the clip into a measurement.
"""

from __future__ import annotations

import base64

import numpy as np

_BG = "#0f0f23"
_PANEL = "#1a1a2e"
_TEXT = "#ccc"
_ACCENT = "#00b894"
_HILITE = "#fdcb6e"
_MUTED = "#888"
_WARN = "#e17055"
_LEFT = "#6cb0ff"
_RIGHT = "#ff8c6e"

_TITLE = "A simulated brain, based on a real fly's wiring"


def table(rows: list[tuple[str, str]]) -> str:
    body = ""
    for i, (k, v) in enumerate(rows):
        border = "border-bottom:1px solid #333;" if i < len(rows) - 1 else ""
        body += (
            f'<tr><td style="padding:6px;{border}white-space:nowrap;">{k}</td>'
            f'<td style="padding:6px;{border}color:{_ACCENT};text-align:right;">{v}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;width:100%;font-size:13px;">{body}</table>'


def panel(title: str, inner: str) -> str:
    return (
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};'
        f'padding:20px;border-radius:8px;margin-bottom:16px;">'
        f'<h3 style="color:{_ACCENT};margin-top:0;">{title}</h3>{inner}</div>'
    )


def heading(text: str) -> str:
    return f'<h3 style="color:{_HILITE};font-family:monospace;margin-bottom:6px;">{text}</h3>'


def note(text: str) -> str:
    return (
        f'<p style="color:{_MUTED};font-family:monospace;font-size:12px;'
        f'line-height:1.6;max-width:900px;">{text}</p>'
    )


def video_html(mp4: bytes, caption: str = "", width: int = 480) -> str:
    """A self-contained inline clip. No javascript, no external asset, no player."""
    b64 = base64.b64encode(mp4).decode()
    cap = f'<div style="color:{_MUTED};font-size:12px;margin-top:6px;">{caption}</div>' if caption else ""
    return (
        f'<video src="data:video/mp4;base64,{b64}" controls autoplay loop muted '
        f'playsinline style="width:{width}px;max-width:100%;border-radius:6px;'
        f'background:#000;"></video>{cap}'
    )


def image_html(rgb: np.ndarray, caption: str = "", width: int | None = None) -> str:
    """An RGB array as an inline PNG."""
    import io

    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.asarray(rgb, np.uint8)).save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    style = f"width:{width}px;" if width else "max-width:100%;"
    cap = f'<div style="color:{_MUTED};font-size:12px;margin-top:6px;">{caption}</div>' if caption else ""
    return f'<img src="data:image/png;base64,{b64}" style="{style}border-radius:6px;"/>{cap}'


# ── Charts ──────────────────────────────────────────────────────────────────────


def _poly(values, x0, y0, w, h, lo, hi, color, width=1.6) -> str:
    if len(values) < 2:
        return ""
    span = max(hi - lo, 1e-9)
    pts = " ".join(
        f"{x0 + w * i / (len(values) - 1):.1f},"
        f"{y0 + h - h * (min(max(v, lo), hi) - lo) / span:.1f}"
        for i, v in enumerate(values)
    )
    return (
        f'<polyline points="{pts}" fill="none" stroke="{color}" '
        f'stroke-width="{width}"/>'
    )


def trace_chart(
    series: list[tuple[str, list[float], str]],
    title: str,
    height: int = 130,
    width: int = 760,
    zero_line: bool = False,
) -> str:
    """Several time series on shared axes, labelled, in SVG."""
    flat = [v for _, values, _ in series for v in values]
    if not flat:
        return ""
    lo, hi = min(flat), max(flat)
    if hi - lo < 1e-9:
        lo, hi = lo - 0.5, hi + 0.5
    pad = (hi - lo) * 0.08
    lo, hi = lo - pad, hi + pad

    pl, pt = 54, 22
    w, h = width - pl - 12, height - pt - 18
    svg = [
        f'<svg width="{width}" height="{height}" style="background:{_PANEL};border-radius:6px;">',
        f'<text x="8" y="14" fill="{_HILITE}" font-family="monospace" font-size="11">{title}</text>',
        f'<rect x="{pl}" y="{pt}" width="{w}" height="{h}" fill="none" stroke="#333"/>',
        f'<text x="{pl - 6}" y="{pt + 9}" fill="{_MUTED}" font-family="monospace" '
        f'font-size="10" text-anchor="end">{hi:.2f}</text>',
        f'<text x="{pl - 6}" y="{pt + h}" fill="{_MUTED}" font-family="monospace" '
        f'font-size="10" text-anchor="end">{lo:.2f}</text>',
    ]
    if zero_line and lo < 0 < hi:
        y = pt + h - h * (0 - lo) / (hi - lo)
        svg.append(
            f'<line x1="{pl}" y1="{y:.1f}" x2="{pl + w}" y2="{y:.1f}" '
            f'stroke="#444" stroke-dasharray="3,3"/>'
        )
    legend_x = pl + 6
    for label, values, color in series:
        svg.append(_poly(values, pl, pt, w, h, lo, hi, color))
        svg.append(
            f'<text x="{legend_x}" y="{pt + h + 14}" fill="{color}" '
            f'font-family="monospace" font-size="10">{label}</text>'
        )
        legend_x += 9 * len(label) + 16
    svg.append("</svg>")
    return "".join(svg)


def trajectory_chart(
    runs: list[tuple[str, list[tuple[float, float, float]], str]],
    target_xy: tuple[float, float],
    target_r: float,
    size: int = 300,
) -> str:
    """Bird's-eye view of where each fly actually went. The real scoreboard."""
    pts = [(x, y) for _, traj, _ in runs for x, y, _ in traj]
    pts += [target_xy, (0.0, 0.0)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    margin = target_r + 3
    lo = min(min(xs), min(ys)) - margin
    hi = max(max(xs), max(ys)) + margin
    span = max(hi - lo, 1e-6)

    def sx(x):
        return 26 + (size - 40) * (x - lo) / span

    def sy(y):
        # +y (the fly's left) points UP in the image, as on a map.
        return size - 22 - (size - 40) * (y - lo) / span

    svg = [
        f'<svg width="{size}" height="{size}" style="background:{_PANEL};border-radius:6px;">',
        f'<text x="8" y="14" fill="{_HILITE}" font-family="monospace" font-size="11">'
        f'trajectories (mm, bird\'s eye)</text>',
        f'<circle cx="{sx(target_xy[0]):.1f}" cy="{sy(target_xy[1]):.1f}" '
        f'r="{max((size - 40) * target_r / span, 3):.1f}" fill="#555" stroke="{_HILITE}"/>',
        f'<circle cx="{sx(0):.1f}" cy="{sy(0):.1f}" r="3" fill="{_TEXT}"/>',
    ]
    for label, traj, color in runs:
        if len(traj) < 2:
            continue
        pts = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y, _ in traj)
        svg.append(
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
            f'opacity="0.9"/>'
        )
        x, y, _ = traj[-1]
        svg.append(f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.5" fill="{color}"/>')
    svg.append("</svg>")
    return "".join(svg)


MODE_COLORS = {
    "connectome": _ACCENT,
    "shuffled": _WARN,
    "blind": _LEFT,
    "nobrain": _MUTED,
}

MODE_BLURB = {
    "connectome": "the real FlyWire wiring, eyes connected",
    "shuffled": "same neurons, same out-degrees, same weights, partners permuted",
    "blind": "real wiring running, photoreceptors held at tonic",
    "nobrain": "no brain at all, constant forward drive",
}


# ── Page sections ───────────────────────────────────────────────────────────────


def header(subtitle: str) -> str:
    return (
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};padding:20px;'
        f'border-radius:8px;margin-bottom:16px;">'
        f'<h2 style="color:{_ACCENT};margin:0 0 6px 0;">{_TITLE}</h2>'
        f'<div style="color:{_MUTED};font-size:13px;">{subtitle}</div></div>'
    )


def connectome_panel(describe: dict, composition: list[tuple[str, str]]) -> str:
    rows = [
        ("neurons", f"{describe['neurons']:,}"),
        ("connected pairs", f"{describe['edges']:,}"),
        ("synapses", f"{describe['synapses']:,}"),
        ("excitatory / inhibitory pairs",
         f"{describe['excitatory_edges']:,} / {describe['inhibitory_edges']:,}"),
        ("mean out-degree", f"{describe['mean_out_degree']}"),
    ]
    return panel(
        "the brain in this run",
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;">'
        f'<div style="flex:1;min-width:300px;">{table(rows)}</div>'
        f'<div style="flex:1;min-width:260px;">{table(composition)}</div></div>'
        + note(
            "FlyWire 783, the complete adult female brain. Every weight is the measured "
            "synapse count for that pair, signed by the presynaptic neuron's predicted "
            "neurotransmitter, times one global scale. Nothing here was trained."
        ),
    )


def run_panel(result, mp4: bytes, calibration_rows: list[tuple[str, str]] | None) -> str:
    m = result.metrics
    rows = [
        ("mode", f"{result.mode} - {MODE_BLURB.get(result.mode, '')}"),
        ("fly time", f"{result.duration_s:.2f} s ({result.ticks} ticks of 15 ms)"),
        ("wall clock", f"{result.wall_s:.0f} s"),
        ("spikes", f"{result.total_spikes:,}"),
        ("reached the pillar",
         f"YES, in {m['time_to_reach_s']:.2f} s of fly time" if m.get("reached")
         else "no"),
        ("start distance", f"{m['start_distance']:.1f} mm"),
        ("closest approach", f"{m['closest_distance']:.1f} mm"),
        ("final distance", f"{m['final_distance']:.1f} mm"),
        ("net approach", f"{m['approach']:+.1f} mm"),
        ("|bearing| error, last third", f"{m['mean_abs_bearing_late']:.0f} deg"),
        ("path length", f"{m['path_length']:.1f} mm"),
        ("straightness", f"{m['straightness']:.2f}"),
    ]
    cal = ""
    if calibration_rows:
        cal = (
            f'<div style="flex:1;min-width:320px;">'
            f'{heading("calibration (measured, not chosen)")}{table(calibration_rows)}</div>'
        )
    return panel(
        f"run: {result.mode}",
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div>{video_html(mp4, "body above, brain below. the bars are the descending command actually sent to the legs.")}</div>'
        f'<div style="flex:1;min-width:320px;">{table(rows)}</div>'
        f'{cal}</div>',
    )


def traces_panel(traces: dict[str, list[float]]) -> str:
    charts = [
        trace_chart(
            [("left eye", traces["eye_left"], _LEFT), ("right eye", traces["eye_right"], _RIGHT)],
            "what each compound eye sees (fraction of ommatidia darkened vs empty arena)",
        ),
        trace_chart(
            [("DN left", traces["dn_left"], _LEFT), ("DN right", traces["dn_right"], _RIGHT)],
            "descending population rate (Hz, summed over 645 + 646 neurons)",
        ),
        trace_chart(
            [("imbalance", traces["imbalance"], _ACCENT)],
            "smoothed descending imbalance = the steering signal",
            zero_line=True,
        ),
        trace_chart(
            [("bearing to pillar", traces["bearing"], _HILITE)],
            "bearing error (degrees, + means the pillar is to the fly's left). "
            "this is read from the physics, where the brain cannot reach it",
            zero_line=True,
        ),
    ]
    named = []
    for cell_type in ("DNa02", "DNa01"):
        if f"{cell_type}_left" in traces:
            named.append(
                trace_chart(
                    [
                        (f"{cell_type} left", traces[f"{cell_type}_left"], _LEFT),
                        (f"{cell_type} right", traces[f"{cell_type}_right"], _RIGHT),
                    ],
                    f"{cell_type}, the textbook steering pair, one cell per side",
                )
            )
    return panel(
        "inside the loop",
        "".join(f'<div style="margin-bottom:10px;">{c}</div>' for c in charts)
        + (
            heading("and the single cells, for comparison")
            + "".join(f'<div style="margin-bottom:10px;">{c}</div>' for c in named)
            + note(
                "If one of these traces is pinned high on one side and flat on the "
                "other regardless of what the eyes are doing, that is the measurement "
                "that forced this demo onto the population readout. A single cell in an "
                "untuned model is not a steering signal."
            )
            if named
            else ""
        ),
    )


def comparison_panel(results, target_xy, target_r) -> str:
    """The controls, side by side. This is the part that makes the clip mean something."""
    rows = []
    header_cells = "".join(
        f'<th style="padding:6px;text-align:right;color:{MODE_COLORS.get(r.mode, _TEXT)};">'
        f"{r.mode}</th>"
        for r in results
    )
    metrics = [
        ("reached the pillar", "reached", "{:.0f}", False),
        ("time to reach (s of fly time)", "time_to_reach_s", "{:.2f}", True),
        ("closest approach (mm)", "closest_distance", "{:.1f}", True),
        ("net approach (mm)", "approach", "{:+.1f}", False),
        ("final distance (mm)", "final_distance", "{:.1f}", True),
        ("|bearing| last third (deg)", "mean_abs_bearing_late", "{:.0f}", True),
        ("path length (mm)", "path_length", "{:.1f}", False),
        ("straightness", "straightness", "{:.2f}", False),
    ]
    for label, key, fmt, lower_better in metrics:
        values = [r.metrics[key] for r in results]
        best = min(values) if lower_better else max(values)
        cells = ""
        for r, v in zip(results, values):
            weight = "bold" if abs(v - best) < 1e-9 else "normal"
            color = _ACCENT if abs(v - best) < 1e-9 else _TEXT
            cells += (
                f'<td style="padding:6px;text-align:right;font-weight:{weight};'
                f'color:{color};border-top:1px solid #333;">{fmt.format(v)}</td>'
            )
        rows.append(
            f'<tr><td style="padding:6px;border-top:1px solid #333;">{label}</td>{cells}</tr>'
        )
    grid = (
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:6px;text-align:left;">metric</th>{header_cells}</tr>'
        f'{"".join(rows)}</table>'
    )
    traj = trajectory_chart(
        [(r.mode, r.trajectory, MODE_COLORS.get(r.mode, _TEXT)) for r in results],
        target_xy,
        target_r,
    )
    return panel(
        "the controls",
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div>{traj}</div><div style="flex:1;min-width:380px;">{grid}</div></div>'
        + note(
            "The claim is not that the connectome fly reaches the pillar. It is that it "
            "reaches the pillar and the shuffled one does not, with the same neurons, "
            "the same number of connections, the same weight distribution and the same "
            "body. If those two columns match, this demo found nothing."
        ),
    )


def probe_panel(conditions: list[dict]) -> str:
    """Brain-only sweep: which sensory channels lateralise at all."""
    rows = ""
    for c in conditions:
        delta = c["dn_right"] - c["dn_left"]
        color = _ACCENT if abs(delta) > 100 else _MUTED
        rows += (
            f'<tr><td style="padding:6px;border-top:1px solid #333;">{c["label"]}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{c["dn_left"]:,.0f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{c["dn_right"]:,.0f}</td>'
            f'<td style="padding:6px;text-align:right;color:{color};'
            f'border-top:1px solid #333;">{delta:+,.0f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{c["active"]:,}</td></tr>'
        )
    return panel(
        "brain only: does this sensory channel lateralise?",
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:6px;text-align:left;">stimulus</th>'
        f'<th style="padding:6px;text-align:right;">DN left (Hz)</th>'
        f'<th style="padding:6px;text-align:right;">DN right (Hz)</th>'
        f'<th style="padding:6px;text-align:right;">R - L</th>'
        f'<th style="padding:6px;text-align:right;">active neurons</th></tr>'
        f"{rows}</table>"
        + note(
            "Each row is a fresh brain, driven for 300 ms with nothing else running. A "
            "channel is usable for steering only if the left and right rows flip sign. "
            "Vision and touch do; smell does not, which is why this fly has its eyes "
            "wired up and not its antennae."
        ),
    )


def leaderboard_panel(conn, total_counts, duration_s: float, top: int = 14) -> str:
    """Which cell types actually did the work, ranked by spikes over the whole run.

    This is the panel that makes the connectome feel like a brain rather than a matrix.
    The names are real: T4 and T5 are the fly's elementary motion detectors, Tm and Mi
    are medulla interneurons, LC cells are the lobula columnar types that carry visual
    features out of the optic lobe, and if a DN shows up here it means a descending
    neuron was among the busiest cells in the animal while it was turning.
    """
    import numpy as np
    import pandas as pd

    counts = np.asarray(total_counts, float)
    ann = conn.ann
    per_neuron = counts[ann["bi"].to_numpy()]
    frame = pd.DataFrame(
        {
            "cell_type": ann["cell_type"].replace("", np.nan).fillna(ann["cell_class"]),
            "super_class": ann["super_class"],
            "spikes": per_neuron,
        }
    )
    grouped = (
        frame.groupby(["cell_type", "super_class"], sort=False)
        .agg(spikes=("spikes", "sum"), cells=("spikes", "size"))
        .reset_index()
        .sort_values("spikes", ascending=False)
        .head(top)
    )
    total = max(counts.sum(), 1.0)
    rows = ""
    for _, r in grouped.iterrows():
        if r["spikes"] <= 0:
            continue
        share = r["spikes"] / total
        hz = r["spikes"] / max(r["cells"], 1) / max(duration_s, 1e-9)
        bar = (
            f'<div style="background:{_ACCENT};height:8px;width:{max(share * 100, 0.5):.1f}%;'
            f'border-radius:2px;"></div>'
        )
        rows += (
            f'<tr><td style="padding:5px;border-top:1px solid #333;">{r["cell_type"]}</td>'
            f'<td style="padding:5px;border-top:1px solid #333;color:{_MUTED};">'
            f'{r["super_class"]}</td>'
            f'<td style="padding:5px;text-align:right;border-top:1px solid #333;">'
            f'{int(r["cells"]):,}</td>'
            f'<td style="padding:5px;text-align:right;border-top:1px solid #333;">'
            f'{hz:,.0f} Hz</td>'
            f'<td style="padding:5px;width:38%;border-top:1px solid #333;">{bar}</td></tr>'
        )
    return panel(
        "which cells did the work",
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:5px;text-align:left;">cell type</th>'
        f'<th style="padding:5px;text-align:left;">class</th>'
        f'<th style="padding:5px;text-align:right;">cells</th>'
        f'<th style="padding:5px;text-align:right;">mean rate</th>'
        f'<th style="padding:5px;text-align:left;">share of all spikes</th></tr>'
        f"{rows}</table>"
        + note(
            "Ranked by total spikes over the run. These are real named cell types from "
            "the FlyWire annotation, not clusters found by this demo: T4 and T5 are the "
            "fly's elementary motion detectors, Tm and Mi are medulla interneurons, and "
            "LC types carry visual features out of the optic lobe toward the descending "
            "neurons."
        ),
    )


def brain_still_panel(still, caption: str = "") -> str:
    """One frame of the glowing brain, large, for the top of the page."""
    return panel(
        "the brain, mid-run",
        image_html(still, caption or
                   "138,639 neurons at their real positions in the FAFB volume. Frontal "
                   "view: the fly's left is on the right of frame. The two big lobes are "
                   "the optic lobes, which are 77,541 of the neurons in this brain."),
    )


def feeding_panel(results) -> str:
    """Sugar against bitter: the proboscis trace, and the walk halting under it."""
    charts = ""
    for r in results:
        traces = r.traces
        if "feeding" not in traces:
            continue
        charts += (
            heading(f"{r.mode}")
            + trace_chart(
                [
                    ("proboscis drive", traces["feeding"], _HILITE),
                    ("walk, left", traces["drive_left"], _LEFT),
                    ("walk, right", traces["drive_right"], _RIGHT),
                ],
                f"{r.mode}: feeding motor neurons and the walking command",
            )
            + trace_chart(
                [("distance to the drop (mm)", traces["distance"], _ACCENT)],
                f"{r.mode}: distance",
            )
        )
    rows = [
        (
            r.mode,
            f"peak proboscis {r.metrics.get('peak_feeding', 0.0):.2f}, "
            f"{int(r.metrics.get('ticks_feeding', 0))} ticks feeding, "
            f"closest {r.metrics['closest_distance']:.1f} mm",
        )
        for r in results
    ]
    return panel(
        "the feeding decision",
        table(rows)
        + charts
        + note(
            "Sugar drives 30 of the 110 feeding motor neurons in this brain, including "
            "MN10. Bitter drives none of them. Neither fact is written anywhere in this "
            "repo: both fall out of which gustatory receptor neurons the contact "
            "happens to drive and where their axons go. The walking command is scaled "
            "by the proboscis drive, which is the one line that turns the brain's "
            "decision into behaviour."
        ),
    )


# ── The swat ────────────────────────────────────────────────────────────────────


def swat_panel(results) -> str:
    """Looming against receding against static: what the summed descending rate did."""
    rows = [
        (
            r.mode,
            f"baseline {r.metrics.get('baseline_dn_total', 0):,.0f} Hz -> peak "
            f"{r.metrics.get('peak_dn_total', 0):,.0f} Hz "
            f"(x{r.metrics.get('startle_ratio', 0):.2f}), "
            f"peak startle {r.metrics.get('peak_startle', 0):.2f}",
        )
        for r in results
    ]
    charts = ""
    for r in results:
        traces = r.traces
        if "dn_total" not in traces:
            continue
        charts += (
            heading(f"{r.mode}")
            + trace_chart(
                [("summed descending rate (Hz)", traces["dn_total"], _ACCENT)],
                f"{r.mode}: the COMMON MODE, which is the channel a symmetric stimulus "
                f"lives in",
            )
            + trace_chart(
                [
                    ("left eye", traces["eye_left"], _LEFT),
                    ("right eye", traces["eye_right"], _RIGHT),
                ],
                f"{r.mode}: what the eyes saw (fraction of ommatidia darkened)",
            )
            + trace_chart(
                [
                    ("startle", traces["startle"], _HILITE),
                    ("walk, left", traces["drive_left"], _LEFT),
                ],
                f"{r.mode}: the startle signal and the walking command it scaled",
            )
        )
    return panel(
        "the swat, in the body",
        table(rows)
        + charts
        + note(
            "The looming and receding runs sweep the same retinal sizes in opposite "
            "time order. A brain that responds to APPROACH rather than to SIZE has to "
            "tell them apart. Whether this one does is measured in the next panel, "
            "where there is no body to add noise to the answer."
        ),
    )


def looming_panel(data: dict) -> str:
    """The brain-only autopsy: where the signal dies, and the time-reversal test."""
    stimulus = data.get("stimulus", [])
    inner = ""
    if stimulus:
        inner += trace_chart(
            [
                ("left eye", [l for l, _ in stimulus], _LEFT),
                ("right eye", [r for _, r in stimulus], _RIGHT),
            ],
            "the stimulus: a real sphere flown at a real fly, captured from the physics",
        )

    for gain, blob in data.get("gains", {}).items():
        reversal = blob.get("time_reversal_corr", float("nan"))
        ceiling = blob.get("noise_ceiling_corr", float("nan"))
        stim_corr = blob.get("stimulus_corr", float("nan"))
        back_corr = blob.get("stimulus_corr_backward", float("nan"))
        sat_f = blob.get("saturation_forward", 0.0)
        sat_b = blob.get("saturation_backward", 0.0)
        # The reversal is scored against how well the brain agrees with ITSELF on a
        # repeat of the same stimulus, not against 1.0. Anything at or above that
        # ceiling is indistinguishable from a repeat.
        #
        # Before any of that can be read, BOTH runs have to be tracking their own
        # stimulus: a network that has ignited and pinned itself near its ceiling
        # produces a flat trace that correlates with nothing, which looks like
        # selectivity and is not. That, and not the saturation fraction, is the right
        # gate. Saturation is reported alongside as the explanation when the gate trips,
        # because a fraction-above-threshold statistic on a Poisson trace is too blunt
        # to decide anything on its own.
        if min(stim_corr, back_corr) < 0.6:
            verdict = (
                f"inconclusive at this gain. The response tracks its own stimulus at "
                f"only r = {min(stim_corr, back_corr):+.2f} in one direction, and sits "
                f"near its ceiling for {100 * max(sat_f, sat_b):.0f}% of the run, so the "
                f"network is saturated rather than selective and the reversal number "
                f"cannot be read as either."
            )
        elif reversal >= ceiling - 0.1:
            verdict = (
                "a receding sphere is as good a match to the looming response as a "
                "REPEAT of the looming stimulus is. This brain cannot tell an approach "
                "from a retreat: it is reading size, not motion."
            )
        else:
            verdict = (
                "the reversal falls below the repeat ceiling while both runs still "
                "track their own stimulus, so something in here responds to the "
                "direction of change and not only to size."
            )
        inner += heading(f"w_syn = {gain} mV")
        inner += trace_chart(
            [
                ("looming", blob["forward"], _ACCENT),
                ("looming again, new seed", blob.get("repeat", []), _MUTED),
                ("receding, time-reversed", blob["backward_reversed"], _WARN),
            ],
            "summed descending rate, three replays of the same retinal sizes",
        )
        inner += note(
            f"reversal <b>r = {reversal:+.3f}</b> against a same-stimulus noise ceiling "
            f"of <b>r = {ceiling:+.3f}</b>. The response tracks the stimulus itself at "
            f"r = {stim_corr:+.3f} looming and r = {back_corr:+.3f} receding, and sits "
            f"near its peak for {100 * sat_f:.0f}% / {100 * sat_b:.0f}% of the run. "
            f"{verdict}"
        )
        # Where the signal dies.
        stage_rows = ""
        for label, counts in blob.get("stages", {}).items():
            active, cells = counts["active"], counts["cells"]
            share = active / max(cells, 1)
            colour = _ACCENT if active else _WARN
            bar = (
                f'<div style="background:{colour};height:8px;'
                f'width:{max(share * 100, 0.4):.1f}%;border-radius:2px;"></div>'
            )
            stage_rows += (
                f'<tr><td style="padding:5px;border-top:1px solid #333;">{label}</td>'
                f'<td style="padding:5px;text-align:right;border-top:1px solid #333;'
                f'color:{colour};">{active:,} / {cells:,}</td>'
                f'<td style="padding:5px;width:45%;border-top:1px solid #333;">{bar}</td></tr>'
            )
        inner += (
            f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
            f'<tr><th style="padding:5px;text-align:left;">stage of the visual pathway</th>'
            f'<th style="padding:5px;text-align:right;">cells that fired</th>'
            f'<th style="padding:5px;text-align:left;"></th></tr>{stage_rows}</table>'
        )
        escape_rows = [
            (label, f"{hz:,.0f} Hz" if hz else "silent")
            for label, hz in blob.get("escape", {}).items()
        ]
        if escape_rows:
            inner += heading("the escape cells, at their loudest moment")
            inner += table(escape_rows)

    return panel(
        "the swat, brain only: where does a looming stimulus die?",
        inner
        + note(
            "The autopsy reads down the pathway a looming signal has to travel. The "
            "photoreceptors fire and the lamina fires; whether anything past that does "
            "is the whole question. LPLC2 is the fly's canonical looming detector and "
            "DNp01 is the giant fibre, the largest axon in the animal and the trigger "
            "for the escape jump. If those rows say silent, this brain saw a large dark "
            "thing arrive and had no circuit to call it a threat."
        ),
    )


# ── The compass ─────────────────────────────────────────────────────────────────


def ring_embedding_chart(ring: dict, size: int = 260) -> str:
    """The recovered ring itself: each cell at its place in the spectral embedding.

    The point of showing this is that the cells were not SORTED onto a circle, they
    landed on one. If the scatter is a blob, the ordering underneath every other compass
    chart is meaningless and the reader can see that for themselves.
    """
    x = np.asarray(ring["x"], float)
    y = np.asarray(ring["y"], float)
    if len(x) == 0:
        return ""
    span = max(np.abs(x).max(), np.abs(y).max()) * 1.15 or 1.0

    def sx(v):
        return size / 2 + (size / 2 - 24) * v / span

    def sy(v):
        return size / 2 - (size / 2 - 24) * v / span

    dots = "".join(
        f'<circle cx="{sx(px):.1f}" cy="{sy(py):.1f}" r="4" fill="{_ACCENT}" '
        f'opacity="0.85"/>'
        for px, py in zip(x, y)
    )
    return (
        f'<svg width="{size}" height="{size}" style="background:{_PANEL};border-radius:6px;">'
        f'<text x="8" y="14" fill="{_HILITE}" font-family="monospace" font-size="11">'
        f'{ring["name"]}: {len(x)} cells, spectral embedding</text>'
        f'<circle cx="{size/2}" cy="{size/2}" r="{size/2-24:.0f}" fill="none" '
        f'stroke="#333" stroke-dasharray="3,3"/>{dots}'
        f'<text x="8" y="{size-8}" fill="{_MUTED}" font-family="monospace" font-size="10">'
        f'similarity vs circular distance r={ring["quality"]:+.2f}, radius CV '
        f'{ring["radius_cv"]:.2f}, eigengap {ring.get("eigengap", 0):.2f}</text></svg>'
    )


def ring_activity_chart(records: list[dict], readout: str, width: int = 470) -> str:
    """Bump position down the page, ring position across it, firing rate as brightness.

    A compass makes a bright band that marches diagonally across this grid: move the
    bump, the response moves with it. A population that is merely switched on makes
    vertical stripes, and a dead one makes an empty box.
    """
    rows = [r["readouts"][readout]["profile"] for r in records if readout in r["readouts"]]
    if not rows:
        return ""
    grid = np.asarray(rows, float)
    top = float(grid.max()) or 1.0
    n_rows, n_cols = grid.shape
    cell_w = (width - 90) / n_cols
    cell_h = 20
    height = n_rows * cell_h + 44

    cells = ""
    for i in range(n_rows):
        for j in range(n_cols):
            level = grid[i, j] / top
            if level <= 0.01:
                continue
            # Same amber-to-white ramp as the brain canvas, so the two read as one demo.
            shade = int(60 + 195 * min(level, 1.0) ** 0.6)
            colour = f"rgb({shade},{int(shade*0.72)},{int(shade*0.26)})"
            cells += (
                f'<rect x="{84 + j * cell_w:.1f}" y="{28 + i * cell_h}" '
                f'width="{cell_w + 0.6:.1f}" height="{cell_h - 2}" fill="{colour}"/>'
            )
    labels = "".join(
        f'<text x="78" y="{28 + i * cell_h + 14}" fill="{_MUTED}" font-family="monospace" '
        f'font-size="10" text-anchor="end">{records[i]["driven_deg"]:+.0f}</text>'
        for i in range(n_rows)
    )
    return (
        f'<svg width="{width}" height="{height}" style="background:{_PANEL};border-radius:6px;">'
        f'<text x="8" y="14" fill="{_HILITE}" font-family="monospace" font-size="11">'
        f'{readout} firing rate around its ring</text>'
        f'<rect x="84" y="28" width="{n_cols * cell_w:.1f}" height="{n_rows * cell_h - 2}" '
        f'fill="#0b0b18"/>{cells}{labels}'
        f'<text x="8" y="{height - 6}" fill="{_MUTED}" font-family="monospace" font-size="10">'
        f'each row is one bump position (deg, left); across is position on the '
        f'{readout} ring</text></svg>'
    )


def compass_tracking_chart(records, fit, readout, size: int = 300) -> str:
    """Where the bump was driven against where the response landed. Slope is the claim."""
    pts = [
        (r["driven_deg"], r["readouts"][readout]["peak_deg"])
        for r in records
        if readout in r["readouts"] and np.isfinite(r["readouts"][readout]["peak_deg"])
    ]
    pad = 42

    def sx(v):
        return pad + (size - pad - 14) * (v + 180) / 360

    def sy(v):
        return size - pad - (size - pad - 22) * (v + 180) / 360

    dots = "".join(
        f'<circle cx="{sx(a):.1f}" cy="{sy(b):.1f}" r="4" fill="{_ACCENT}"/>' for a, b in pts
    )
    line = ""
    if np.isfinite(fit.get("slope", float("nan"))):
        slope, offset = fit["slope"], fit["offset_deg"]
        segment = []
        for a in np.linspace(-180, 180, 181):
            b = (slope * a + offset + 180) % 360 - 180
            segment.append((a, b))
        # Break the polyline where the fitted line wraps, so it does not draw a
        # spurious vertical stroke across the whole chart.
        runs, current = [], [segment[0]]
        for prev, nxt in zip(segment, segment[1:]):
            if abs(nxt[1] - prev[1]) > 180:
                runs.append(current)
                current = []
            current.append(nxt)
        runs.append(current)
        for run in runs:
            if len(run) < 2:
                continue
            path = " ".join(f"{sx(a):.1f},{sy(b):.1f}" for a, b in run)
            line += (
                f'<polyline points="{path}" fill="none" stroke="{_HILITE}" '
                f'stroke-width="1.4" stroke-dasharray="4,3"/>'
            )
    return (
        f'<svg width="{size}" height="{size}" style="background:{_PANEL};border-radius:6px;">'
        f'<text x="8" y="14" fill="{_HILITE}" font-family="monospace" font-size="11">'
        f'bump position vs {readout} response</text>'
        f'<rect x="{pad}" y="22" width="{size-pad-14}" height="{size-pad-22}" fill="none" '
        f'stroke="#333"/>{line}{dots}'
        f'<text x="{pad-6}" y="26" fill="{_MUTED}" font-family="monospace" font-size="10" '
        f'text-anchor="end">+180</text>'
        f'<text x="{pad-6}" y="{size-pad}" fill="{_MUTED}" font-family="monospace" '
        f'font-size="10" text-anchor="end">-180</text>'
        f'<text x="{size/2}" y="{size-8}" fill="{_MUTED}" font-family="monospace" '
        f'font-size="10" text-anchor="middle">bump driven (deg)</text></svg>'
    )


def compass_panel(data: dict) -> str:
    """The whole compass result: the recovered ring, the rotation, and the control."""
    rings = data.get("rings", {})
    embeddings = "".join(
        f'<div>{ring_embedding_chart(ring)}</div>' for ring in rings.values()
    )
    inner = (
        heading("the ring, recovered from connectivity alone")
        + f'<div style="display:flex;gap:12px;flex-wrap:wrap;">{embeddings}</div>'
        + note(
            "The FlyWire annotation labels all 47 EPG cells the same way and gives no "
            "wedge number, so the order around the ring is not in the metadata. It is "
            "recovered here by spectral embedding of who each cell shares Delta7 "
            "partners with. The number under each circle is the check: in the recovered "
            "order, connectivity similarity has to fall off with distance around the "
            "ring, and it does."
        )
    )

    for condition, block in data.get("conditions", {}).items():
        records = block.get("records", [])
        fits = block.get("fits", {})
        inner += heading(f"{condition}")
        charts = ""
        for readout, fit in fits.items():
            charts += (
                f'<div style="display:flex;gap:12px;flex-wrap:wrap;'
                f'align-items:flex-start;margin-bottom:10px;">'
                f'<div>{ring_activity_chart(records, readout)}</div>'
                f'<div>{compass_tracking_chart(records, fit, readout)}</div></div>'
            )
            if np.isfinite(fit.get("residual_deg", float("nan"))):
                charts += note(
                    f"<b>{readout}</b>: slope {fit['slope']:+.0f}, residual "
                    f"<b>{fit['residual_deg']:.1f} deg RMS</b>, circular correlation "
                    f"{fit['circular_corr']:+.3f}, mean bump sharpness "
                    f"{fit['mean_vector_length']:.2f} "
                    f"(1.0 is a perfect bump, 0.0 is the population evenly lit)."
                )
            else:
                charts += note(
                    f"<b>{readout}</b>: an average of {fit.get('mean_active', 0):.1f} of "
                    f"{fit.get('cells', 0)} cells fired across the bump positions, which "
                    f"is not enough to fit an angle to. This is what the shuffled "
                    f"control is supposed to look like."
                )
        inner += charts

    return panel(
        "the compass: a ring attractor, recovered and then driven",
        inner
        + note(
            "A slope of +1 or -1 means the downstream response ROTATES with the bump: "
            "move the compass and its output moves the same amount. The sign is "
            "arbitrary because the two rings are embedded independently and each has "
            "its own handedness; the residual is the part that is a claim. The shuffled "
            "control keeps every neuron, every out-degree and the whole weight "
            "distribution, and permutes only who talks to whom."
        ),
    )


# ── The drum ────────────────────────────────────────────────────────────────────


def drum_panel(results) -> str:
    """Optomotor: does the steering command flip sign with the drum's direction?"""
    rows = ""
    for r in results:
        m = r.metrics
        mean = m.get("mean_imbalance", 0.0)
        corr = m.get("azimuth_corr", float("nan"))
        colour = _ACCENT if abs(mean) > 2 * m.get("imbalance_sd", 0.0) / max(1, 1) else _TEXT
        rows += (
            f'<tr><td style="padding:6px;border-top:1px solid #333;">{r.mode}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{colour};">{mean:+.4f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{m.get("imbalance_sd", 0.0):.4f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{corr:+.3f}</td></tr>'
        )
    table_html = (
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:6px;text-align:left;">condition</th>'
        f'<th style="padding:6px;text-align:right;">mean steering command</th>'
        f'<th style="padding:6px;text-align:right;">its SD</th>'
        f'<th style="padding:6px;text-align:right;">vs bar azimuth</th></tr>{rows}</table>'
    )

    # The verdict, computed rather than asserted. Two questions, and the second is what
    # keeps the first honest: does the MOTION stimulus separate its two directions, and
    # does the POSITION stimulus separate its two, on the same rig in the same run?
    by_mode = {r.mode: r.metrics for r in results}

    def contrast(label_a: str, label_b: str):
        a, b = by_mode.get(label_a), by_mode.get(label_b)
        if not a or not b:
            return None
        mean_a, mean_b = a.get("mean_imbalance", 0.0), b.get("mean_imbalance", 0.0)
        noise = max(a.get("imbalance_sd", 0.0), b.get("imbalance_sd", 0.0))
        return mean_a, mean_b, abs(mean_a - mean_b), noise

    verdict = ""
    motion = contrast("12 bars, counter-clockwise", "12 bars, clockwise")
    position = contrast("1 bar, counter-clockwise", "1 bar, clockwise")
    if motion:
        mean_a, mean_b, gap, noise = motion
        verdict += note(
            f"<b>Motion (the full drum):</b> counter-clockwise gives {mean_a:+.4f} and "
            f"clockwise {mean_b:+.4f}. They differ by <b>{gap:.4f}</b> against a "
            f"within-run SD of {noise:.4f}, so the two directions are "
            + (
                "distinguishable, which is an optomotor response."
                if gap > noise
                else f"<b>{noise / max(gap, 1e-9):.0f} times closer together than the "
                f"noise in a single run</b>. There is no optomotor response here."
            )
        )
    if position:
        mean_a, mean_b, gap, noise = position
        verdict += note(
            f"<b>Position (one bar):</b> the same rig, the same brain, a stimulus the "
            f"encoder can carry. Sweeping left gives {mean_a:+.4f} and sweeping right "
            f"{mean_b:+.4f}, differing by <b>{gap:.4f}</b>, "
            + (
                f"<b>{gap / max(motion[2], 1e-9):.0f} times the separation the rotating "
                f"drum managed</b>. "
                if motion
                else ""
            )
            + "So the rig works and the fly is steerable; it is motion specifically that "
            "this model cannot see. Note that the two sweep directions are not mirror "
            "images of each other, which is the same left-right asymmetry the single-fly "
            "demo measures in the descending populations."
        )

    charts = ""
    for r in results:
        if "imbalance" not in r.traces:
            continue
        charts += heading(r.mode) + trace_chart(
            [
                ("steering command", r.traces["imbalance"], _ACCENT),
                ("nearest bar azimuth (deg)", r.traces["bearing"], _HILITE),
            ],
            f"{r.mode}: the descending imbalance against where the nearest bar is",
            zero_line=True,
        )

    return panel(
        "the drum: optomotor, and what it takes to see motion at all",
        table_html
        + verdict
        + charts
        + note(
            "A full drum is a motion stimulus and a single bar is a position stimulus, "
            "and this repo's retina reports one number per eye: the fraction of "
            "ommatidia darker than an empty arena. That encoder cannot represent motion "
            "at all, so the twelve-bar rows fail before the connectome is consulted. "
            "The brain-only autopsy in the swat report says the same thing from the "
            "other end: T4 and T5, the fly's elementary motion detectors, do not fire "
            "here at any synaptic gain tried, because L1 is glutamatergic in this "
            "dataset and dominates Mi1's input at -122,574 signed synapses. Both halves "
            "of the pathway are broken, and the single-bar rows show what the same rig "
            "does when the stimulus is one the encoder can actually carry."
        ),
    )


# ── Two flies ───────────────────────────────────────────────────────────────────

_FLY_COLORS = {"a": _LEFT, "b": _RIGHT}


def duet_panel(result, mp4: bytes) -> str:
    """One two-fly run: the clip, both flies' numbers, and how much they crowded."""
    rows = []
    for name, m in result.metrics.items():
        rows.append(
            (
                f"fly {name} ({result.modes[name]})",
                f"closest {m['closest_distance']:.1f} mm, net approach "
                f"{m['approach']:+.1f} mm, "
                + (
                    f"reached in {m['time_to_reach_s']:.2f} s"
                    if m["reached"]
                    else "did not reach"
                ),
            )
        )
    rows.append(
        (
            "closest they came to each other",
            f"{min(result.separation):.1f} mm, crowded (under 5 mm) for "
            f"{int(next(iter(result.metrics.values()))['ticks_crowded'])} ticks",
        )
    )
    charts = trace_chart(
        [("separation between the two flies (mm)", result.separation, _HILITE)],
        "how close the two animals got. a fly is 2.5 mm long, so under 5 mm they are "
        "genuinely in each other's way",
    )
    for name, traces in result.traces.items():
        charts += trace_chart(
            [
                (f"fly {name}: distance to pillar (mm)", traces["distance"],
                 _FLY_COLORS.get(name, _ACCENT)),
            ],
            f"fly {name} ({result.modes[name]}): closing on the target",
        )
    return panel(
        f"two flies: {' vs '.join(result.modes.values())}",
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div>{video_html(mp4, "both points of view, both brains, one map.", width=560)}</div>'
        f'<div style="flex:1;min-width:320px;">{table(rows)}</div></div>'
        + charts,
    )


def duet_comparison_panel(blobs: list[dict], target_xy, target_r) -> str:
    """Both conditions side by side, with every fly's track on one map."""
    runs, rows = [], ""
    for blob in blobs:
        for name, traj in blob["trajectories"].items():
            mode = blob["modes"][name]
            colour = _ACCENT if mode == "connectome" else _WARN
            runs.append((f"{name} ({mode})", [tuple(p) for p in traj], colour))
        for name, m in blob["metrics"].items():
            rows += (
                f'<tr><td style="padding:6px;border-top:1px solid #333;">'
                f'{blob["label"]}</td>'
                f'<td style="padding:6px;border-top:1px solid #333;">fly {name}, '
                f'{blob["modes"][name]}</td>'
                f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
                f'{m["closest_distance"]:.1f}</td>'
                f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
                f'{m["approach"]:+.1f}</td>'
                f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
                f'{"yes" if m["reached"] else "no"}</td>'
                f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
                f'{int(m["ticks_crowded"])}</td></tr>'
            )
    grid = (
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:6px;text-align:left;">condition</th>'
        f'<th style="padding:6px;text-align:left;">fly</th>'
        f'<th style="padding:6px;text-align:right;">closest (mm)</th>'
        f'<th style="padding:6px;text-align:right;">approach (mm)</th>'
        f'<th style="padding:6px;text-align:right;">reached</th>'
        f'<th style="padding:6px;text-align:right;">ticks crowded</th></tr>{rows}</table>'
    )
    return panel(
        "both conditions",
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div>{trajectory_chart(runs, target_xy, target_r, size=340)}</div>'
        f'<div style="flex:1;min-width:360px;">{grid}</div></div>'
        + note(
            "The second condition is the control the single-fly demo has to run "
            "sequentially: a real connectome and a shuffled one in the SAME arena at "
            "the SAME moment, so they cannot differ by lighting, floor, spawn or luck. "
            "Note that the two flies are solid to one another, so a shuffled fly "
            "wandering into the real one's path is a real effect on the real one's run "
            "and not noise."
        ),
    )


# ── The poke ────────────────────────────────────────────────────────────────────


def poke_panel(results) -> str:
    """Touch one antenna: does the steering command flip with the side?"""
    rows = ""
    for r in results:
        m = r.metrics
        sep = m.get("imbalance_separation_sd", 0.0)
        colour = _ACCENT if sep >= 2.0 else _WARN
        rows += (
            f'<tr><td style="padding:6px;border-top:1px solid #333;">{r.mode}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{_LEFT};">{m.get("imbalance_left", 0):+.3f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{_RIGHT};">{m.get("imbalance_right", 0):+.3f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{_MUTED};">{m.get("imbalance_quiet", 0):+.3f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{colour};font-weight:bold;">{sep:.2f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{m.get("turn_after_left_poke", float("nan")):+.0f} / '
            f'{m.get("turn_after_right_poke", float("nan")):+.0f}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{_MUTED};">{m.get("turn_unpoked", float("nan")):.0f}</td></tr>'
        )
    grid = (
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:6px;text-align:left;">wiring</th>'
        f'<th style="padding:6px;text-align:right;">poked left</th>'
        f'<th style="padding:6px;text-align:right;">poked right</th>'
        f'<th style="padding:6px;text-align:right;">not poked</th>'
        f'<th style="padding:6px;text-align:right;">apart (SD)</th>'
        f'<th style="padding:6px;text-align:right;">turn L / R (deg)</th>'
        f'<th style="padding:6px;text-align:right;">wander (deg)</th></tr>{rows}</table>'
    )

    charts = ""
    for r in results:
        if "poke" not in r.traces:
            continue
        charts += heading(r.mode) + trace_chart(
            [
                ("which antenna (-1 left, +1 right)", r.traces["poke"], _HILITE),
                ("steering command", r.traces["imbalance"], _ACCENT),
            ],
            f"{r.mode}: the poke and what the brain did about it",
            zero_line=True,
        )
    return panel(
        "the poke: the strongest reaction in this model",
        grid
        + charts
        + note(
            "The first three columns are the descending imbalance, the number that "
            "steers the fly, averaged over the ticks when each antenna was being poked "
            "and over the ticks when neither was. A touch response means the left and "
            "right columns sit on OPPOSITE sides of the unpoked value and are far apart "
            "compared to the scatter within each condition, which is the SD column. "
            "Heading is shown too and is deliberately the weaker number: a walking fly "
            "wanders about 10 degrees per 180 ms on its own, so the behaviour is noisier "
            "than the command driving it. Shuffled wiring keeps every neuron, every "
            "out-degree and the whole weight distribution, and still has 2,656 "
            "mechanosensory neurons being driven just as hard."
        ),
    )


# ── Learning ────────────────────────────────────────────────────────────────────


def specificity_panel(stages: list[dict]) -> str:
    """Where along the olfactory pathway odour identity survives, and where it dies."""
    rows = ""
    for stage in stages:
        jac = stage["jaccard"]
        # Low overlap means the odours are distinct, which is what learning needs.
        colour = _ACCENT if (jac == jac and jac < 0.5) else _WARN
        counts = " / ".join(f"{v:,}" for v in stage["active"].values())
        # NaN means fewer than two channels fired, so there is no overlap to report.
        shown = f"{jac:.3f}" if jac == jac else "n/a"
        cell_colour = colour if jac == jac else _MUTED
        rows += (
            f'<tr><td style="padding:6px;border-top:1px solid #333;">{stage["stage"]}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{_MUTED};">{stage["cells"]:,}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">{counts}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{stage["odors_firing"]}</td>'
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;'
            f'color:{cell_colour};font-weight:bold;">{shown}</td></tr>'
        )
    return panel(
        "can this brain tell two odours apart?",
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:6px;text-align:left;">stage of the olfactory pathway</th>'
        f'<th style="padding:6px;text-align:right;">cells</th>'
        f'<th style="padding:6px;text-align:right;">active per odour</th>'
        f'<th style="padding:6px;text-align:right;">odours firing</th>'
        f'<th style="padding:6px;text-align:right;">overlap</th></tr>{rows}</table>'
        + note(
            "Overlap is the mean Jaccard index between the sets of cells different "
            "odours activate, computed only over the odour channels that actually fire: "
            "a silent channel scores 0 against everything and would read as perfect "
            "decorrelation. Low is good. Odour-specific learning needs the Kenyon cell "
            "row to be low, because the learning rule depresses exactly the cells that "
            "were active. If one glomerulus drives most of the projection neurons, every "
            "odour looks the same from the mushroom body onward and punishing one "
            "punishes all of them."
        ),
    )


def learning_panel(conditions: dict) -> str:
    """Before and after training, for the real wiring and the shuffled control."""
    inner = ""
    for label, result in conditions.items():
        t, c = result["trained_odor"], result["control_odor"]
        report = result["training"]
        rows = [
            ("trained odour", t),
            ("control odour (never punished)", c),
            ("Kenyon cells active during training",
             f"{report['active_kc']:,} of 5,177 ({100 * report['kc_fraction']:.1f}%)"),
            ("MBONs the dopamine reaches", f"{report['taught_mbons']} of 96"),
            ("synapses depressed",
             f"{report['edges_depressed']:,} connections "
             f"({report['synapses_depressed']:,.0f} synapses) by "
             f"{100 * report['learning_rate']:.0f}%"),
            (f"MBON output for {t}",
             f"{result['before'][t]['mbon_hz']:,.0f} -> "
             f"{result['after'][t]['mbon_hz']:,.0f} Hz "
             f"({result['trained_drop_pct']:+.1f}%)"),
            (f"MBON output for {c}",
             f"{result['before'][c]['mbon_hz']:,.0f} -> "
             f"{result['after'][c]['mbon_hz']:,.0f} Hz "
             f"({result['control_drop_pct']:+.1f}%)"),
            ("specificity (trained drop minus control drop)",
             f"{result['specificity_pct']:+.1f} points"),
        ]
        inner += heading(label) + table(rows)
    return panel(
        "the training",
        inner
        + note(
            "The rule is dopamine-gated depression, which is what the animal uses: a "
            "Kenyon cell active at the moment dopamine arrives has its synapse onto that "
            "compartment's output neuron weakened. Nothing here is Hebbian and nothing is "
            "trained by gradient descent. The shuffled control is the honest one: with "
            "the wiring permuted, the odour activates almost no Kenyon cells, so there is "
            "nothing to depress and the same procedure changes nothing. Note also what "
            "the specificity number is doing, and read it next to the panel above: a "
            "memory that is not odour-specific is a memory about odours in general."
        ),
    )


def learning_video_panel(mp4: bytes, odor: str, result: dict) -> str:
    """The clip: the same odour, the same brain, before and after it was punished."""
    drop = result.get("trained_drop_pct", 0.0)
    control_drop = result.get("control_drop_pct", 0.0)
    return panel(
        "watching it learn",
        video_html(
            mp4,
            f"the same odour ({odor}) presented to the same brain twice: on the left "
            f"before training, on the right after the odour was paired with dopamine. "
            f"Top row is the whole brain, bottom row is the mushroom body on its own.",
            width=720,
        )
        + note(
            f"The mushroom body output falls {drop:.1f}% for this odour. Watch the "
            f"bottom right panel go dark while the top row barely changes: the rest of "
            f"the brain is doing exactly what it did before, because only the Kenyon "
            f"cell synapses onto the output neurons were touched. The colour scale is "
            f"fixed from the BEFORE run on both sides, so the darkening is a real "
            f"difference and not the autoscale rebalancing. "
            f"The odour the fly was never punished for falls {control_drop:.1f}%, which "
            f"is the problem, and the panel after next is about why."
        ),
    )


# ── Neurochemistry ──────────────────────────────────────────────────────────────

_NT_COLORS = {
    "acetylcholine": "#00b894",   # the excitatory workhorse
    "glutamate": "#6cb0ff",       # inhibitory in the fly
    "gaba": "#a29bfe",            # inhibitory
    "dopamine": "#fdcb6e",        # the modulators, warm
    "serotonin": "#ff8c6e",
    "octopamine": "#e84393",
}


def chemistry_panel(traces: dict[str, list[float]], title: str = "") -> str:
    """What the brain released, tick by tick, split by transmitter.

    Two charts, because the six transmitters differ by two orders of magnitude in how
    much of the brain releases them and one shared axis would flatten the modulators
    into the baseline.
    """
    fast = [
        (name, traces[f"nt:{name}"], _NT_COLORS[name])
        for name in ("acetylcholine", "glutamate", "gaba")
        if traces.get(f"nt:{name}")
    ]
    slow = [
        (name, traces[f"nt:{name}"], _NT_COLORS[name])
        for name in ("dopamine", "serotonin", "octopamine")
        if traces.get(f"nt:{name}")
    ]
    if not fast and not slow:
        return ""

    charts = ""
    if fast:
        charts += trace_chart(
            fast,
            "fast transmitters: synaptic events released per tick, in thousands",
        )
    if slow:
        charts += trace_chart(
            slow,
            "the modulators: dopamine, serotonin, octopamine, same units",
        )
    balance = []
    if traces.get("nt:excitatory"):
        balance.append(("excitatory", traces["nt:excitatory"], _ACCENT))
    if traces.get("nt:inhibitory"):
        balance.append(("inhibitory", traces["nt:inhibitory"], _WARN))
    if balance:
        charts += trace_chart(balance, "excitation against inhibition, in thousands of events")
    if traces.get("nt:ei_ratio"):
        charts += trace_chart(
            [("E / I", traces["nt:ei_ratio"], _HILITE)],
            "the ratio. above 1 the brain is net driving itself, below 1 net damping it",
        )

    rows = []
    for name in _NT_COLORS:
        values = traces.get(f"nt:{name}")
        if values:
            rows.append((name, f"{np.mean(values):,.1f}k events per tick, peak "
                               f"{np.max(values):,.1f}k"))
    return panel(
        f"what it released{(' - ' + title) if title else ''}",
        "".join(f'<div style="margin-bottom:10px;">{c}</div>' for c in [charts])
        + table(rows)
        + note(
            "A spike is not a spike. Every neuron carries a predicted transmitter and an "
            "outgoing synapse budget, so a single cell firing delivers that many synaptic "
            "events of that one transmitter: 354 for the average cholinergic neuron, 658 "
            "for the average GABAergic one, 656 for an octopaminergic one. These traces "
            "are those budgets summed over whoever actually fired, which is closer to "
            "what a head would be full of than a raster is. "
            "<b>The caveat matters:</b> this is release, not effect. The model collapses "
            "all six transmitters onto a +1/-1 sign, so the dopamine trace is a real "
            "count of what a real fly would be releasing, drawn next to a simulation "
            "that does not implement what that release does."
        ),
    )


def learning_chemistry_panel(chem: dict[str, dict[str, list[float]]]) -> str:
    """What the same odour released before and after the fly was punished."""
    if not chem:
        return ""
    order = [k for k in ("before training", "after training") if k in chem]
    colours = {"before training": _LEFT, "after training": _RIGHT}
    charts = ""
    for name in ("acetylcholine", "gaba", "glutamate", "dopamine"):
        series = [
            (f"{label}", chem[label][f"nt:{name}"], colours.get(label, _ACCENT))
            for label in order
            if chem[label].get(f"nt:{name}")
        ]
        if series:
            charts += trace_chart(
                series,
                f"{name} released per tick, in thousands of synaptic events",
            )
    rows = []
    for name in ("acetylcholine", "gaba", "glutamate", "dopamine", "serotonin", "octopamine"):
        cells = []
        for label in order:
            values = chem[label].get(f"nt:{name}")
            cells.append(f"{np.mean(values):,.1f}k" if values else "n/a")
        change = ""
        if len(cells) == 2 and "n/a" not in cells:
            was = np.mean(chem[order[0]][f"nt:{name}"])
            now = np.mean(chem[order[1]][f"nt:{name}"])
            if was > 0:
                change = f"  ({100 * (now - was) / was:+.1f}%)"
        rows.append((name, " -> ".join(cells) + change))
    return panel(
        "what training changed chemically",
        "".join(f'<div style="margin-bottom:10px;">{c}</div>' for c in [charts])
        + table(rows)
        + note(
            "Kenyon cells are cholinergic, and training weakens their synapses onto the "
            "output neurons by 90%. So the acetylcholine trace falls for two reasons at "
            "once: the same cells are firing, and each of their spikes now delivers "
            "fewer synaptic events. Every other transmitter should be close to "
            "unchanged, because nothing else was touched, and a large move in the GABA "
            "or dopamine rows would mean the edit reached further than intended."
        ),
    )


# ── The fly riding a robot ──────────────────────────────────────────────────────


def ride_panel(result, mp4: bytes) -> str:
    """One ride: the clip, the numbers, and what the fly was telling the robot."""
    m = result.metrics
    rows = [
        ("mode", f"{result.mode} - {MODE_BLURB.get(result.mode, '')}"),
        ("robot time", f"{result.duration_s:.2f} s ({result.ticks} ticks of 15 ms)"),
        ("wall clock", f"{result.wall_s:.0f} s"),
        ("spikes", f"{result.total_spikes:,}"),
        ("reached the pillar",
         f"YES, in {m['time_to_reach_s']:.2f} s" if m.get("reached") else "no"),
        ("start distance", f"{m['start_distance']:.0f} mm"),
        ("closest approach", f"{m['closest_distance']:.0f} mm"),
        ("net approach", f"{m['approach']:+.0f} mm"),
        ("|bearing| error, last third", f"{m['mean_abs_bearing_late']:.0f} deg"),
        ("path length", f"{m['path_length']:.0f} mm"),
        ("still upright", "yes" if m.get("upright") else "NO, it fell over"),
    ]
    charts = trace_chart(
        [("turn command sent to the robot", result.traces["turn_cmd"], _ACCENT)],
        "what the fly asked for. positive is one way, negative the other",
        zero_line=True,
    ) + trace_chart(
        [("bearing to the pillar (deg)", result.traces["bearing"], _HILITE)],
        "where the pillar actually is, read from the physics. this is the error the fly "
        "is trying to drive to zero",
        zero_line=True,
    ) + trace_chart(
        [("distance to the pillar (mm)", result.traces["distance"], _LEFT)],
        "and whether it is getting there",
    ) + trace_chart(
        [("left eye", result.traces["eye_left"], _LEFT),
         ("right eye", result.traces["eye_right"], _RIGHT)],
        "what the compound eyes see from the saddle",
    )
    return panel(
        f"the ride: {result.mode}",
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div>{video_html(mp4, "the robot, the brain driving it, and the map.", width=640)}</div>'
        f'<div style="flex:1;min-width:300px;">{table(rows)}</div></div>'
        + charts
        + note(
            "Nothing about the brain changed for this. Same connectome, same 721-facet "
            "retina, same bridge, same calibration as the walking demo. The entire "
            "interface between a fly and a four-legged robot is four lines in "
            "<code>rider.ride_command</code>: the mean of the fly's two descending "
            "drives becomes the robot's speed and their difference becomes its yaw. "
            "The one thing that had to be measured rather than assumed is the SIGN, "
            "because a fly turns toward the side that pushes less and this robot turns "
            "toward the side that strides further, so the command means opposite things "
            "on the two bodies."
        ),
    )


def ride_comparison_panel(results, target_xy, target_r) -> str:
    """The ride against its controls."""
    rows = ""
    metrics = [
        ("reached the pillar", "reached", "{:.0f}"),
        ("closest approach (mm)", "closest_distance", "{:.0f}"),
        ("net approach (mm)", "approach", "{:+.0f}"),
        ("|bearing| last third (deg)", "mean_abs_bearing_late", "{:.0f}"),
        ("path length (mm)", "path_length", "{:.0f}"),
        ("mean turn command", "mean_turn_cmd", "{:+.3f}"),
        ("still upright", "upright", "{:.0f}"),
    ]
    header_cells = "".join(
        f'<th style="padding:6px;text-align:right;color:{MODE_COLORS.get(r.mode, _TEXT)};">'
        f"{r.mode}</th>" for r in results
    )
    for label, key, fmt in metrics:
        cells = "".join(
            f'<td style="padding:6px;text-align:right;border-top:1px solid #333;">'
            f'{fmt.format(r.metrics.get(key, float("nan")))}</td>' for r in results
        )
        rows += f'<tr><td style="padding:6px;border-top:1px solid #333;">{label}</td>{cells}</tr>'
    grid = (
        f'<table style="border-collapse:collapse;width:100%;font-size:13px;">'
        f'<tr><th style="padding:6px;text-align:left;">metric</th>{header_cells}</tr>{rows}</table>'
    )
    traj = trajectory_chart(
        [(r.mode, r.trajectory, MODE_COLORS.get(r.mode, _TEXT)) for r in results],
        target_xy, target_r, size=340,
    )
    return panel(
        "the ride, against its controls",
        f'<div style="display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start;">'
        f'<div>{traj}</div><div style="flex:1;min-width:360px;">{grid}</div></div>'
        + note(
            "The controls are the same ones the walking demo uses and they mean the same "
            "thing here. <b>shuffled</b> keeps every neuron, every out-degree and the "
            "whole weight distribution and permutes only who talks to whom; "
            "<b>nobrain</b> sends a constant command, which on this robot means walking "
            "in a straight line forever. A pillar 40 degrees off the nose is missed by "
            "96 mm by anything that does not turn, so reaching it is the measurement."
        ),
    )
