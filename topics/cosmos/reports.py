"""HTML for the Flyte reports: generated clips, frame strips, and run facts.

Same palette and helpers as topics/rl-mujoco/reports.py, topics/isaac-sim/reports.py
and topics/dreamerv3/reports.py, so a viewer moving between the world-model demos is
not re-learning the colours. Everything is inline HTML and inline data: URIs; no
JavaScript, no external assets, no object-store round trip.
"""

from __future__ import annotations

import html

_BG = "#0f0f23"
_PANEL = "#1a1a2e"
_TEXT = "#ccc"
_ACCENT = "#00b894"
_HILITE = "#fdcb6e"
_MUTED = "#888"

_TITLE = "NVIDIA Cosmos 3 - a world model you can roll forward"


def _table(rows: list[tuple[str, str]]) -> str:
    body = ""
    for i, (k, v) in enumerate(rows):
        border = "border-bottom:1px solid #333;" if i < len(rows) - 1 else ""
        body += (
            f'<tr><td style="padding:6px;{border}white-space:nowrap;">{k}</td>'
            f'<td style="padding:6px;{border}color:{_ACCENT};">{v}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;width:100%;">{body}</table>'


def _panel(title: str, inner: str) -> str:
    return (
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};'
        f'padding:20px;border-radius:8px;">'
        f'<h3 style="color:{_ACCENT};margin-top:0;">{title}</h3>{inner}</div>'
    )


def _heading(text: str) -> str:
    return f'<h3 style="color:{_HILITE};font-family:monospace;">{text}</h3>'


def _note(text: str) -> str:
    return (
        f'<p style="color:{_MUTED};font-family:monospace;font-size:12px;'
        f'line-height:1.5;">{text}</p>'
    )


def _details(summary: str, body: str) -> str:
    return (
        f'<details><summary style="cursor:pointer;color:{_MUTED};font-family:monospace;">'
        f"{summary}</summary>"
        f'<pre style="font-size:11px;color:{_TEXT};background:{_PANEL};padding:12px;'
        f'border-radius:4px;overflow-x:auto;white-space:pre-wrap;">'
        f"{html.escape(body)}</pre></details>"
    )


# Public aliases: pipeline.py assembles the comparison cells itself, so it needs
# these two. Exported by name rather than reached for as `reports._note`, which is
# the kind of thing that quietly breaks when this file is tidied.
note = _note
details = _details


def progress_html(stage: str, detail: str, rows: list[tuple[str, str]]) -> str:
    """Painted while the pod is still working, so the report is never blank.

    Worth having here specifically: this task spends its first several minutes
    downloading 35 GB and its next few loading a 16B transformer, and a report that
    stays empty through all of that is indistinguishable from a hung pod.
    """
    return (
        f"<h2>{_TITLE}</h2>"
        + _panel(stage, _table(rows) + _note(detail))
    )


def clip_block(
    title: str,
    video_html: str,
    strip_html: str = "",
    probe: str = "",
    prompt: str = "",
) -> str:
    """One generated clip: the video, its frame strip, and what produced it."""
    out = _heading(title) + video_html
    if strip_html:
        out += (
            _note("The same clip laid out in time. A world model that has collapsed to "
                  "a single repeated frame looks fine as a loop and obvious as a strip.")
            + strip_html
        )
    if probe:
        out += _note(probe)
    if prompt:
        out += _details("prompt", prompt)
    return out + "<br/>"


def side_by_side(cells: list[tuple[str, str]]) -> str:
    """Lay several clip blocks out in a responsive row.

    flex-wrap rather than a grid: Flyte reports render at a width you do not control,
    and two 560px clips side by side have to be allowed to become two stacked clips.
    """
    inner = ""
    for label, body in cells:
        inner += (
            f'<div style="flex:1 1 380px;min-width:320px;">'
            f'<h4 style="color:{_HILITE};font-family:monospace;margin:0 0 8px;">'
            f"{label}</h4>{body}</div>"
        )
    return f'<div style="display:flex;gap:16px;flex-wrap:wrap;">{inner}</div>'


def action_traces(
    truth: list[list[float]],
    pred: list[list[float]],
    labels: list[str] | None = None,
    dims: list[int] | None = None,
    caption: str = "",
    width: int = 520,
    height: int = 84,
) -> str:
    """Overlay recovered actions on ground truth, one small chart per channel.

    Inline SVG rather than a plotting library. matplotlib is not in the image spec and
    is not worth adding for two polylines, and an <img> of a chart cannot be zoomed in
    a Flyte report the way vector text can.

    Each channel gets its own y-scale taken from both series together. Sharing one
    scale across channels would be the honest thing for commensurable numbers and is
    the wrong thing here: the `av` action packs translation next to a rotation basis
    resting near 1.0, so a shared axis flattens every channel that matters.
    """
    n = min(len(truth), len(pred))
    if n < 2:
        return _note("Not enough steps to plot.")
    ncols = len(truth[0])
    dims = list(range(ncols)) if dims is None else [d for d in dims if d < ncols]

    out = ""
    for d in dims:
        a = [row[d] for row in truth[:n]]
        b = [row[d] for row in pred[:n]]
        lo, hi = min(min(a), min(b)), max(max(a), max(b))
        span = (hi - lo) or 1.0
        pad = span * 0.08
        lo, hi = lo - pad, hi + pad
        span = hi - lo

        def path(series: list[float]) -> str:
            step = width / (n - 1)
            pts = [
                f"{i * step:.1f},{height - (v - lo) / span * height:.1f}"
                for i, v in enumerate(series)
            ]
            return " ".join(pts)

        name = labels[d] if labels and d < len(labels) else f"channel {d}"
        out += (
            f'<div style="margin:0 0 10px;">'
            f'<div style="color:{_MUTED};font-family:monospace;font-size:11px;'
            f'margin-bottom:2px;">{name} '
            f'<span style="color:{_ACCENT};">&#9473; truth</span> '
            f'<span style="color:{_HILITE};">&#9473; recovered</span></div>'
            f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
            f'preserveAspectRatio="none" style="background:{_BG};border-radius:4px;">'
            f'<polyline fill="none" stroke="{_ACCENT}" stroke-width="1.6" '
            f'points="{path(a)}"/>'
            f'<polyline fill="none" stroke="{_HILITE}" stroke-width="1.6" '
            f'stroke-dasharray="4 3" points="{path(b)}"/>'
            f"</svg></div>"
        )
    return out + (_note(caption) if caption else "")


def final_html(
    subtitle: str,
    rows: list[tuple[str, str]],
    body: str,
    explainer: str = "",
    logs: list[str] | None = None,
) -> str:
    """The finished report: run facts, the generated media, and what to look at."""
    out = f"<h2>{_TITLE}</h2>" + _panel(subtitle, _table(rows))
    if explainer:
        out += _panel("What to look for", _note(explainer))
    out += "<br/>" + body
    if logs:
        out += _panel("Logs", _details("run tail", "\n".join(logs[-30:])))
    return out


# ── Explainers ──────────────────────────────────────────────────────────────────
#
# Kept as text in the report rather than only in the README. The report is the
# artifact people scroll past on the stream, and a clip with no claim attached to it
# is just a pretty video.

IMAGINE_EXPLAINER = (
    "This is the generator surface: text (and optionally one image) in, video out. "
    "Judge it as physics, not as cinematography. Do surfaces in contact stay in "
    "contact? Does anything accelerate without a cause, or come to rest without "
    "touching something? Do objects keep their shape and volume across frames, and "
    "does what passes behind something reappear on the other side? Those are the "
    "questions a world model is supposed to answer, and they are the ones a video "
    "model that has only learned appearance gets wrong."
)

ROLLOUT_EXPLAINER = (
    "This is the part that makes it a world model rather than a video generator. "
    "The model is given ONE observed frame and a sequence of raw robot actions, and "
    "asked what happens next: p(future video | first frame, actions). Nothing about "
    "the future is described in words. If the predicted motion tracks the commanded "
    "actions, the model has learned the dynamics of that embodiment, which is what "
    "lets it stand in for a simulator when you are generating training data or "
    "evaluating a policy without a robot. Compare that with topics/dreamerv3, where "
    "the agent learns exactly this same p(next state | state, action) from scratch "
    "for one environment, and with topics/isaac-sim, where the dynamics are a hand-"
    "written physics engine instead of a learned model."
)

COMPARE_EXPLAINER = (
    "Same scene, same seed, two prompts. The left one is the sentence you would "
    "hand any other video model. The right one is the structured JSON caption Cosmos "
    "3 was actually trained on, naming subjects, lighting, camera and a beat-by-beat "
    "temporal description of the physical event. NVIDIA's guidance is to upsample the "
    "short form into the long one with an LLM before generation; this is what that "
    "step buys, without a second model in the loop."
)


INVERT_EXPLAINER = (
    "Forward dynamics asks what happens next. This asks the opposite: here is a "
    "video, what actions produced it? Cosmos denoises the action channel with the "
    "same machinery it uses for pixels, so running the model backwards is a mode "
    "flag rather than a second model. The clip and the answer key both ship inside "
    "the checkpoint, so the dashed line is the model's guess and the solid line is "
    "what actually happened, and the gap between them is a number rather than an "
    "impression. This is also the sharpest contrast with topics/dreamerv3: that "
    "world model maps (state, action) to the next state and has no path in the "
    "other direction at all, so a Dreamer agent can dream a future but can never "
    "watch a video and say what was done in it."
)
