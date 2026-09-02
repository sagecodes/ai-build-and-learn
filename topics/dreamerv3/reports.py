"""HTML for the Flyte report: the dream, the reality, and the curves underneath.

Same palette and helpers as topics/rl-mujoco/reports.py and topics/isaac-sim/reports.py,
so a viewer moving between the demos is not re-learning the colours. Charts are
hand-rolled SVG rather than matplotlib: the image is already large, and a polyline
needs no dependency.

── The order of the page is the argument it is making ──────────────────────────
The dream goes first, above the numbers. A world model is a claim about prediction,
and the strongest available evidence for it is the prediction itself sitting next to
what actually happened. A reward curve is a much weaker claim that happens to be
easier to plot, which is why almost every RL report leads with one.

Second is the real rollout, and it is there to be distrusted on purpose. Score climbing
does not prove the walker is walking, so the report shows the body moving in the world
and, next to it, metres actually travelled, which arena.py logs from the physics where
the policy cannot reach it.
"""

from __future__ import annotations

_BG = "#0f0f23"
_PANEL = "#1a1a2e"
_TEXT = "#ccc"
_ACCENT = "#00b894"
_HILITE = "#fdcb6e"
_MUTED = "#888"
_WARN = "#e17055"

_TITLE = "DreamerV3 - learning a world model of MuJoCo"


def table(rows: list[tuple[str, str]]) -> str:
    body = ""
    for i, (k, v) in enumerate(rows):
        border = "border-bottom:1px solid #333;" if i < len(rows) - 1 else ""
        body += (
            f'<tr><td style="padding:6px;{border}white-space:nowrap;">{k}</td>'
            f'<td style="padding:6px;{border}color:{_ACCENT};">{v}</td></tr>'
        )
    return f'<table style="border-collapse:collapse;width:100%;">{body}</table>'


def panel(title: str, inner: str) -> str:
    return (
        f'<div style="font-family:monospace;background:{_BG};color:{_TEXT};'
        f'padding:20px;border-radius:8px;">'
        f'<h3 style="color:{_ACCENT};margin-top:0;">{title}</h3>{inner}</div>'
    )


def heading(text: str) -> str:
    return f'<h3 style="color:{_HILITE};font-family:monospace;">{text}</h3>'


def note(text: str) -> str:
    return (
        f'<p style="color:{_MUTED};font-family:monospace;font-size:12px;'
        f'line-height:1.6;max-width:820px;">{text}</p>'
    )


def curve(
    points: list[tuple[float, float]],
    label: str,
    colour: str = _ACCENT,
    w: int = 760,
    h: int = 220,
) -> str:
    """(x, y) as an SVG polyline. x is environment steps, y whatever is plotted."""
    if len(points) < 2:
        return note(f"{label}: not enough points yet")

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    xspan = (x1 - x0) or 1.0
    yspan = (y1 - y0) or 1.0
    pad = 40
    pts = " ".join(
        f"{pad + (x - x0) / xspan * (w - 2 * pad):.1f},"
        f"{h - pad - (y - y0) / yspan * (h - 2 * pad):.1f}"
        for x, y in points
    )
    return (
        f'<div style="background:{_BG};padding:12px 16px;border-radius:8px;">'
        f'<svg viewBox="0 0 {w} {h}" style="width:100%;max-width:{w}px;background:{_PANEL};'
        f'border-radius:4px;" font-family="monospace">'
        f'<polyline points="{pts}" fill="none" stroke="{colour}" stroke-width="2"/>'
        f'<text x="{pad}" y="16" fill="{_HILITE}" font-size="12">{label}</text>'
        f'<text x="{pad}" y="{h - 12}" fill="{_MUTED}" font-size="11">'
        f"{y0:.3g} to {y1:.3g}</text>"
        f'<text x="{w - pad}" y="{h - 12}" fill="{_MUTED}" font-size="11" '
        f'text-anchor="end">{x1:,.0f} steps</text>'
        f"</svg></div>"
    )


def video_html(mp4: bytes, caption: str, max_width: int = 820) -> str:
    import base64

    b64 = base64.b64encode(mp4).decode()
    cap = note(caption) if caption else ""
    return (
        f'<div style="background:{_BG};padding:16px;border-radius:8px;">'
        f'<video src="data:video/mp4;base64,{b64}" controls autoplay loop muted '
        f'playsinline style="max-width:{max_width}px;width:100%;border:2px solid #333;'
        f'border-radius:4px;display:block;image-rendering:pixelated;">'
        f"</video>{cap}</div>"
    )


def filmstrip(stills: list[tuple[int, bytes]], height: int = 190) -> str:
    """Stills across training, oldest left. This is where progress becomes visible.

    `image-rendering:pixelated` matters: these are 64 px frames blown up, and letting
    the browser smooth them would paint in detail the model never predicted.
    """
    import base64

    if len(stills) < 2:
        return ""
    cells = ""
    for step, data in stills:
        b64 = base64.b64encode(data).decode()
        cells += (
            f'<div style="flex:0 0 auto;text-align:center;">'
            f'<img src="data:image/png;base64,{b64}" style="height:{height}px;'
            f'border:1px solid #333;border-radius:3px;display:block;'
            f'image-rendering:pixelated;"/>'
            f'<span style="color:{_MUTED};font-size:10px;font-family:monospace;">'
            f"{step / 1000:.0f}k</span></div>"
        )
    return (
        f'<div style="background:{_BG};padding:12px 16px;border-radius:8px;'
        f'overflow-x:auto;"><div style="display:flex;gap:6px;align-items:flex-end;">'
        f"{cells}</div></div>"
    )


_DREAM_LEGEND = (
    "Six sequences side by side. Top row is what really happened, middle row is the "
    "world model's reconstruction, bottom row is the difference. The border is "
    "<b style='color:#0f0'>green</b> while the model is still being shown the real "
    "frames, and turns <b style='color:#f33'>red</b> the moment the images are taken "
    "away and it has to predict the rest from actions alone. Everything after the red "
    "line is imagination. Early in training the imagined half dissolves within a few "
    "frames; a working world model keeps the walker, the posts and the ball where "
    "physics would have put them."
)

_ROLLOUT_LEGEND = (
    "The policy in the real environment, at the 64x64 resolution the agent actually "
    "sees, taken from the training loop's own episode recording. The camera tracks the "
    "walker, so the walker itself barely moves in frame and the posts streaming past "
    "are the evidence of real forward travel. A policy that has learned to score "
    "without going anywhere looks completely still against that background."
)


def _dream_block(film) -> str:
    got = film.latest.get("dream")
    strip = filmstrip(film.thinned("dream"))
    if not got:
        return heading("The dream") + note(
            "No open-loop prediction yet. Dreamer writes one every "
            "<code>run.report_every</code> seconds once the replay buffer has enough "
            "data to sample a batch. If this never fills, the run is on "
            "<code>dmc_proprio</code>: the agent has no image observation, so the "
            "decoder has no image to reconstruct and there is nothing to dream."
        )
    body = heading("The dream: what the model thinks happens next")
    body += note(_DREAM_LEGEND)
    body += video_html(got["mp4"], f"step {got['step']:,} &middot; {got['probe']}")
    if strip:
        body += "<br/>" + note(
            "The last imagined frame of the first sequence, one per report, oldest on "
            "the left. This strip is the world model learning."
        ) + filmstrip(film.thinned("dream"))
    return body


def _rollout_block(film) -> str:
    got = film.latest.get("rollout")
    if not got:
        return ""
    body = heading("The reality: the policy in the environment")
    body += note(_ROLLOUT_LEGEND)
    body += video_html(got["mp4"], f"step {got['step']:,} &middot; {got['probe']}", 420)
    strip = film.thinned("rollout")
    if len(strip) >= 2:
        body += "<br/>" + filmstrip(strip, height=150)
    return body


def _honesty_block(data: dict) -> str:
    """Score and metres travelled, side by side, and what it means if they disagree."""
    score, dist = data["score"], data["distance"]
    body = heading("Is it actually walking?")
    body += note(
        "The left curve is what the agent is paid. The right is how far the torso got "
        "from where the episode started, in metres, read from the physics and logged "
        "under a <code>log/</code> key so it never enters the observation and the "
        "policy has no way to influence it except by moving. The walker reward pays "
        "for a mix of standing upright and moving at 1 m/s, so a policy that learns "
        "the standing half and skips the moving half can sit near a return of 300 "
        "forever. Score up and distance flat is that failure, and it is visible here "
        "long before it is visible in the video."
    )
    body += curve(score, "episode return", _ACCENT)
    body += "<br/>" + curve(dist, "metres travelled (max per episode)", _HILITE)
    if score and dist:
        s, d = score[-1][1], dist[-1][1]
        verdict = (
            f"latest episode: return {s:.0f}, travelled {d:.1f} m."
            + (
                f" Return is climbing but the walker is not travelling; check the "
                f"rollout video above."
                if s > 250 and d < 1.5
                else ""
            )
        )
        body += note(verdict)
    return body


def _losses_block(losses: dict[str, list[tuple[float, float]]]) -> str:
    """The world model's own losses, which is where you see it learning the dynamics.

    Named so the report explains itself: `dyn` and `rep` are the two sides of the KL
    between what the model predicted the next latent would be and what the encoder
    actually saw, `image` is the decoder reconstructing pixels, `rew` and `con` are the
    reward and continuation heads, and `policy`/`value` are the actor and critic
    trained entirely inside imagination.
    """
    meaning = {
        "dyn": "dynamics KL: the predictor moving toward the encoder",
        "rep": "representation KL: the encoder moving toward something predictable",
        "image": "decoder: reconstructing 64x64 pixels from the latent",
        "rew": "reward head",
        "con": "continuation head",
        "policy": "actor, trained on imagined rollouts only",
        "value": "critic, trained on imagined rollouts only",
        "repval": "critic regularised toward the replayed returns",
    }
    order = ["image", "dyn", "rep", "rew", "con", "policy", "value", "repval"]
    keys = [k for k in order if k in losses] + [k for k in losses if k not in order]
    blocks = ""
    for key in keys:
        pts = losses[key]
        if len(pts) < 2:
            continue
        blocks += curve(pts, f"loss/{key} ({meaning.get(key, '')})", _HILITE) + "<br/>"
    return blocks or note("no loss points logged yet")


def _summary_rows(task, config, size, data, extra) -> list[tuple[str, str]]:
    fps = data["fps"]
    rows = [
        ("Task", task),
        ("Agent", f"DreamerV3, {config} {size}"),
        (
            "Observation",
            "64x64 pixels" if "vision" in config else "proprioceptive state vector",
        ),
    ]
    rows += extra
    if fps:
        rows.append(
            ("Throughput", ", ".join(f"{k} {v:,.0f}" for k, v in sorted(fps.items())))
        )
    if data["ram"]:
        rows.append(("Replay buffer", f"{data['ram'][-1][1]:.1f} GB"))
    return rows


def progress_html(task, config, size, step, total, data, film, secs) -> str:
    pct = (100.0 * step / total) if total else 0.0
    eta = ""
    if step and secs > 60:
        remain = (total - step) * (secs / step)
        eta = f", about {remain / 3600:.1f} h left"
    score = data["score"]
    rows = _summary_rows(task, config, size, data, [
        ("Progress", f"{step:,} / {total:,} env steps ({pct:.0f}%){eta}"),
        ("Elapsed", f"{secs / 60:.0f} min"),
        ("Episodes scored", f"{len(score)}"),
        ("Latest return", f"{score[-1][1]:.1f}" if score else "no episode finished yet"),
        (
            "Furthest travelled",
            f"{max((y for _, y in data['distance']), default=0.0):.1f} m"
            if data["distance"] else "n/a",
        ),
    ])
    return (
        f"<h2>{_TITLE}</h2>"
        + panel("Training", table(rows))
        + "<br/>"
        + _dream_block(film)
        + "<br/>"
        + _rollout_block(film)
        + "<br/>"
        + _honesty_block(data)
        + "<br/>"
        + heading("World model")
        + panel("Losses", _losses_block(data["losses"]))
    )


def final_html(
    task, config, size, steps, secs, data, film, params, tail,
    video: str = "", clip_probe: str = "",
) -> str:
    score = data["score"]
    best = max((y for _, y in score), default=0.0)
    far = max((y for _, y in data["distance"]), default=0.0)
    rows = _summary_rows(task, config, size, data, [
        ("Environment", "DeepMind Control Suite (dm_control on MuJoCo)"),
        ("Env steps", f"{steps:,}"),
        ("Wall clock", f"{secs / 3600:.2f} h"),
        ("Agent size", params or "n/a"),
        ("Episodes scored", f"{len(score)}"),
        ("Return, first", f"{score[0][1]:.1f}" if score else "n/a"),
        ("Return, final", f"{score[-1][1]:.1f}" if score else "n/a"),
        ("Return, best", f"{best:.1f}" if score else "n/a"),
        ("Furthest travelled", f"{far:.1f} m" if data["distance"] else "n/a"),
    ])
    logs = (
        "\n".join(tail[-30:])
        .replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    replay_block = ""
    if video:
        replay_block = (
            heading("The trained policy, re-rendered")
            + note(
                "The same policy as the rollout above, filmed again at 480x480 so the "
                "arena is legible. The agent still acts on the 64x64 observation it "
                "trained on; only the camera resolution changed, so this is a "
                "recording of the policy rather than a different run."
            )
            + video
            + "<br/>"
        )

    return (
        f"<h2>{_TITLE}</h2>"
        + panel("Run summary", table(rows))
        + "<br/>"
        + _dream_block(film)
        + "<br/>"
        + replay_block
        + _rollout_block(film)
        + "<br/>"
        + _honesty_block(data)
        + "<br/>"
        + heading("World model")
        + panel("Losses", _losses_block(data["losses"]))
        + "<br/>"
        + panel(
            "Logs",
            f'<details><summary style="cursor:pointer;color:{_MUTED};">training tail'
            f"</summary>"
            f'<pre style="font-size:11px;color:{_TEXT};background:{_PANEL};padding:12px;'
            f'border-radius:4px;overflow-x:auto;">{logs}</pre></details>',
        )
    )
