"""HTML building blocks for the Flyte reports.

The reports are meant to be readable at a glance by someone who has not read the
chapter yet, so most of this file is about explaining, not drawing:

  intro()          - a story card: who the agent is, what it wants, what goes wrong
  legend()         - what the icons, arrows, numbers and colours mean
  explain()        - a "how to read this / what to notice" box under a visual
  takeaways()      - the chapter's points in plain words
  grid()           - one frame of a corridor or lake: icons, values, arrows, the agent
  decision_card()  - every move from one square, with where the ice might send it:
                     the Bellman equation, drawn
  slider()         - any list of HTML frames behind a scrubber and a play button
  replay()         - record an episode and play it back, with the agent's reasoning
                     at every step in plain words

`pct=True` shows values as "chance of reaching the goal". That is exactly what a value
means when gamma = 1 and the only reward is +1 at the goal, and it is far easier to
read than 0.558.

Everything is self-contained HTML with inline JS (the same approach as the openenv
maze replays), so it renders in a Flyte report with no extra assets.
"""

import base64
import html
import itertools
import json
from pathlib import Path

import numpy as np

import envs

_BG, _PANEL, _TEXT, _ACCENT, _HILITE = "#0f0f23", "#1a1a2e", "#ccc", "#00b894", "#fdcb6e"
_HOLE, _GOAL, _AGENT = "#4a1d2b", "#1e6f50", "#fdcb6e"
AGENT, GOAL, HOLE = "🤖", "🎁", "🕳️"
_ids = itertools.count()

STYLE = (
    f"<style>body{{background:{_BG};color:{_TEXT};font-family:sans-serif;line-height:1.45}}"
    f"h2{{color:{_HILITE}}}h3{{color:{_ACCENT};margin:22px 0 4px}}"
    f"pre{{background:{_PANEL};padding:12px;overflow-x:auto;font-size:13px;line-height:1.3}}"
    ".gx td{width:48px;height:48px;text-align:center;font-size:12px;color:#eee;"
    "border:1px solid #0f0f23;position:relative;padding:0}"
    ".gx .ar{font-size:18px;line-height:18px;display:block}"
    ".gx .ic{font-size:22px;line-height:24px;display:block}"
    ".gx .st{position:absolute;top:1px;left:3px;font-size:9px;color:#aaa}"
    ".gx .tie{display:grid;grid-template-columns:repeat(3,12px);grid-template-rows:repeat(3,10px);"
    "justify-content:center;font-size:11px;line-height:10px;margin-bottom:1px}"
    ".row{display:flex;gap:24px;flex-wrap:wrap;align-items:flex-start}"
    f".card{{background:{_PANEL};border-left:4px solid {_ACCENT};padding:10px 14px;"
    "border-radius:4px;margin:8px 0;max-width:900px}"
    f".card.note{{border-left-color:{_HILITE}}}"
    ".card b{color:#fff}"
    "table.dc td,table.dc th{padding:4px 10px;text-align:left;border-bottom:1px solid #333;"
    "font-size:13px}</style>"
)


# ── words ───────────────────────────────────────────────────────────────────────

_INTROS = {
    "lake": (
        f"<b>The frozen lake.</b> {AGENT} starts in the top-left corner and wants the "
        f"{GOAL} in the bottom-right (+1 point, and the game ends). The {HOLE} squares are "
        "holes in the ice: step on one and the game ends with nothing. "
        "<b>The catch: the ice is slippery.</b> When the agent picks a direction it only goes "
        "that way 1 time in 3. The rest of the time it slides to one side or the other "
        "(never backwards). Walking into the edge of the board just leaves it where it was."
    ),
    "walk": (
        f"<b>The slippery walk.</b> A narrow icy bridge. {AGENT} starts in the middle. Off the "
        f"right end is the {GOAL} (+1 point); off the left end is a {HOLE} (nothing). It can only "
        "choose ← or →. It goes the way it chose half the time, stays put a third of the time, "
        "and slides the wrong way the remaining sixth."
    ),
    "rw": (
        f"<b>The random walk.</b> The same icy bridge, but the agent has no control at all: "
        f"every step is a coin flip, left or right. Reaching the {GOAL} on the right end is +1."
    ),
    "bandit": (
        f"<b>The bandit walk.</b> The shortest possible bridge: one square, a {HOLE} on the left, "
        f"a {GOAL} on the right. One decision, then it is over."
    ),
}


def ice_rules():
    """The slip rule as a picture: choose ↑, and where you can end up."""
    c = "width:96px;height:54px;text-align:center;font-size:12px;border:1px solid #333"
    you = f"background:#2a3550;{c}"
    return (
        '<div class="row" style="align-items:center;margin:6px 0">'
        '<table style="border-collapse:collapse">'
        f'<tr><td style="{c}"></td><td style="background:#1e3a2f;{c}">1 in 3<br>goes ↑ as chosen</td><td style="{c}"></td></tr>'
        f'<tr><td style="background:#3a2f1e;{c}">1 in 3<br>slides ←</td>'
        f'<td style="{you}"><span style="font-size:20px">{AGENT}</span><br>chooses ↑</td>'
        f'<td style="background:#3a2f1e;{c}">1 in 3<br>slides →</td></tr>'
        f'<tr><td style="{c}"></td><td style="{c};color:#888">never<br>backwards</td><td style="{c}"></td></tr>'
        "</table>"
        '<div style="max-width:420px;font-size:13px"><b>How the ice works.</b> Whatever direction '
        "the agent chooses, it only goes that way 1 time in 3; otherwise it slides to one of the two "
        "sides, at right angles. <b>Sliding or walking into the edge of the board means you stay "
        "put.</b> That last rule is the agent's best friend: point at a wall and some of the "
        "slides turn into harmless bumps.</div></div>"
    )


def intro(name, extra=""):
    """A story card for an environment: what the agent wants and what goes wrong."""
    kind = {"fl4": "lake", "fl8": "lake", "swf": "walk", "sws": "walk", "rw": "rw",
            "bw": "bandit", "bsw": "bandit"}[name]
    card = f'<div class="card">{_INTROS[kind]} {extra}</div>'
    return card + (ice_rules() if kind == "lake" else "")


def legend(pct=True, arrows=True):
    num = ("number and colour = <b>chance of reaching the " + GOAL + " from that square</b> "
           "(brighter = better)") if pct else "number and colour = how good the agent thinks the square is"
    arr = (" · arrow = the move the plan makes on that square (several arrows = those moves "
           "are equally good there; hover a square to see every move's value)") if arrows else ""
    return (f'<div style="font-size:13px;margin:4px 0 10px">{AGENT} agent · {GOAL} goal (+1, game over) · '
            f"{HOLE} hole (game over, nothing){arr} · {num}</div>")


def explain(how="", notice=""):
    parts = []
    if how:
        parts.append(f"<b>How to read this:</b> {how}")
    if notice:
        parts.append(f"<b>What to notice:</b> {notice}")
    return '<div class="card note">' + "<br>".join(parts) + "</div>"


def takeaways(points):
    items = "".join(f"<li>{p}</li>" for p in points)
    return f'<h3>The takeaways</h3><div class="card"><ul style="margin:4px 0">{items}</ul></div>'


def details(summary, body):
    return f"<details><summary>{html.escape(summary)}</summary>{body}</details>"


def _fmt(v, pct):
    if pct and 0 < v < 0.005:
        return "<1%"  # tiny but not zero: "0%" would make a real difference look like a tie
    return f"{v:.0%}" if pct else f"{v:.2f}"


def _odds(p):
    for k in (2, 3, 6):
        if abs(p - 1 / k) < 1e-6:
            return f"1 in {k}"
    if abs(p - 2 / 3) < 1e-6:
        return "2 in 3"
    return "always" if p > 1 - 1e-9 else f"{p:.0%}"


# ── drawing ─────────────────────────────────────────────────────────────────────

def layout_for(name):
    """A character map for any env: lakes have one, corridors get H F...F G."""
    P, start, _, layout = envs.ENVS[name]
    if layout is not None:
        return layout
    return ["H" + "F" * (start - 1) + "S" + "F" * (len(P) - 2 - start) + "G"]


TIE_TOL = 1e-6
# Where each lake action sits in the 3x3 tie marker (row, column): left, down, right, up.
_TIE_POS = {0: (2, 1), 1: (3, 2), 2: (2, 3), 3: (1, 2)}


def tied(Q, s, tol=TIE_TOL):
    """Every action within tol of the best one: the moves that are equally good."""
    return [int(a) for a in np.flatnonzero(Q[s] >= Q[s].max() - tol)]


def _arrows_html(s, policy, Q, arrows):
    if policy is None:
        return ""
    acts = tied(Q, s) if Q is not None else [policy[s]]
    if len(acts) == 1:
        return f'<span class="ar">{arrows[acts[0]]}</span>'
    if len(arrows) == 2:  # corridor: just list them
        return f'<span class="ar" style="font-size:13px">{"".join(arrows[a] for a in acts)}</span>'
    cells = "".join(f'<span style="grid-area:{_TIE_POS[a][0]}/{_TIE_POS[a][1]}">{arrows[a]}</span>'
                    for a in acts)
    return f'<span class="tie">{cells}</span>'


def _shade(v, vmax):
    t = 0.0 if vmax <= 0 else float(np.clip(v / vmax, 0, 1))
    lo, hi = np.array([0x22, 0x2a, 0x44]), np.array([0x00, 0xb8, 0x94])
    r, g, b = (lo + (hi - lo) * t).astype(int)
    return f"#{r:02x}{g:02x}{b:02x}"


def grid(layout, values=None, policy=None, agent=None, trail=(), vmax=None, caption="",
         pct=False, Q=None, glow=None):
    """One board. Pass Q as well as policy and squares where several moves tie show
    all of the tied arrows, so an arbitrary tie-break never looks like a decision."""
    rows, cols = len(layout), len(layout[0])
    arrows = "←→" if rows == 1 else "←↓→↑"
    if values is not None and vmax is None:
        vmax = max(float(np.max(values)), 1e-9)
    trail = set(trail)
    out = ['<table class="gx" style="border-collapse:collapse">']
    for r in range(rows):
        out.append("<tr>")
        for c in range(cols):
            s = r * cols + c
            cell = layout[r][c]
            start = '<span class="st">start</span>' if cell == "S" else ""
            if cell == "H":
                bg, body = _HOLE, f'<span class="ic">{HOLE}</span>'
            elif cell == "G":
                bg, body = _GOAL, f'<span class="ic">{GOAL}</span>'
            else:
                bg = _shade(values[s], vmax) if values is not None else "#2a3550"
                arrow = _arrows_html(s, policy, Q, arrows)
                num = _fmt(values[s], pct) if values is not None else ""
                body = start + arrow + num
            if s == agent:
                body = f'<span class="ic">{AGENT}</span>' + (start if cell == "S" else "")
            style = f"background:{bg}"
            tip = ""
            if Q is not None and cell not in "HG":
                best = tied(Q, s)
                tip = ' title="' + "  ".join(
                    f"{arrows[a]} {_fmt(Q[s, a], pct)}{' ★' if a in best else ''}"
                    for a in range(Q.shape[1])) + '"'
                style += ";cursor:help"
            if glow is not None and glow[s] > 0.01 and s != agent:
                style += f";box-shadow:inset 0 0 0 {1 + 5 * glow[s]:.0f}px rgba(253,203,110,{0.25 + 0.75 * glow[s]:.2f})"
            elif s in trail and s != agent:
                style += ";box-shadow:inset 0 0 0 2px #fdcb6e55"
            if s == agent:
                style += f";box-shadow:inset 0 0 0 3px {_AGENT}"
            out.append(f'<td data-s="{s}" style="{style}"{tip}>{body}</td>')
        out.append("</tr>")
    out.append("</table>")
    if caption:
        out.append(f'<div style="font-size:12px;margin-top:4px;max-width:260px">{caption}</div>')
    return "".join(out)


def slider(frames, captions=None, title="", interval_ms=250):
    """Scrub or play through pre-rendered HTML frames."""
    uid = f"sl{next(_ids)}"
    captions = captions or [""] * len(frames)
    return f"""
<div style="background:{_PANEL};padding:12px;border-radius:6px;display:inline-block;margin:6px 0">
  <div style="color:{_ACCENT};font-weight:bold;margin-bottom:6px">{html.escape(title)}</div>
  <input type="range" id="{uid}-r" min="0" max="{len(frames) - 1}" value="0" style="width:280px">
  <button id="{uid}-b" style="margin-left:8px;padding:3px 12px;background:{_ACCENT};
    color:#fff;border:none;border-radius:4px;cursor:pointer">Play</button>
  <div id="{uid}-c" style="margin-top:6px;font-size:13px;min-height:18px"></div>
  <div id="{uid}-f" style="margin-top:8px"></div>
</div>
<script>
(function() {{
  const F = {json.dumps(frames)}, C = {json.dumps(captions)};
  const r = document.getElementById("{uid}-r"), b = document.getElementById("{uid}-b");
  let timer = null;
  function show() {{
    document.getElementById("{uid}-f").innerHTML = F[r.value];
    document.getElementById("{uid}-c").innerHTML = C[r.value];
  }}
  r.oninput = show;
  b.onclick = function() {{
    if (timer) {{ clearInterval(timer); timer = null; b.textContent = "Play"; return; }}
    if (+r.value >= F.length - 1) r.value = 0;
    b.textContent = "Pause";
    timer = setInterval(function() {{
      if (+r.value >= F.length - 1) {{ clearInterval(timer); timer = null; b.textContent = "Play"; return; }}
      r.value = +r.value + 1; show();
    }}, {interval_ms});
  }};
  show();
}})();
</script>"""


# ── the agent's reasoning ───────────────────────────────────────────────────────

def _where(name, s):
    layout = layout_for(name)
    if len(layout) == 1:
        return f"square {s} of the bridge"
    r, c = divmod(s, len(layout[0]))
    return f"row {r + 1}, column {c + 1}"


def _direction(name, s, s2):
    """Which way the agent actually moved, in words."""
    layout = layout_for(name)
    cols = len(layout[0])
    d = s2 - s
    if d == 0:
        return "stays put"
    if len(layout) == 1:
        return "moves →" if d > 0 else "moves ←"
    return {-1: "slides ←", 1: "slides →", cols: "slides ↓", -cols: "slides ↑"}.get(d, "moves")


def outcome(name, s, a, s2):
    """What one move did, in words: as chosen, a slide, or a bump."""
    arrows = "←→" if len(layout_for(name)) == 1 else "←↓→↑"
    if _moved_as_intended(name, s, a, s2):
        return f"goes {arrows[a]} as chosen"
    moved = _direction(name, s, s2)
    return "bumps the edge and stays put" if moved == "stays put" else f"the ice {moved.replace('slides', 'slides it')}"


def _cell(name, s):
    layout = layout_for(name)
    cols = len(layout[0])
    return layout[s // cols][s % cols]


def _outcomes(name, s, a, V, gamma):
    """Merged (probability, next state, value of landing there) for one move."""
    P = envs.ENVS[name][0]
    merged = {}
    for p, s2, rew, done in P[s][a]:
        val = rew + gamma * (0.0 if done else V[s2])
        q, _ = merged.get(s2, (0.0, val))
        merged[s2] = (q + p, val)
    return [(p, s2, val) for s2, (p, val) in merged.items()]


def _landing(name, s2, val, pct, stayed=False):
    if stayed:
        return f"(bumps the edge), worth {_fmt(val, pct)}"
    cell = _cell(name, s2)
    if cell == "H":
        return f"into a {HOLE} hole (0)"
    if cell == "G":
        return f"onto the {GOAL} (+1)"
    return f"to a square worth {_fmt(val, pct)}"


def decision_card(name, s, V, gamma=1.0, pct=True, title=""):
    """Every move from square s, and where the slippery ice might send each one.

    This is the Bellman equation drawn as a table: a move's value is the
    probability-weighted average of where it can land.
    """
    layout = layout_for(name)
    arrows = "←→" if len(layout) == 1 else "←↓→↑"
    nA = len(envs.ENVS[name][0][s])
    rows, totals = [], []
    for a in range(nA):
        outs = _outcomes(name, s, a, V, gamma)
        total = sum(p * v for p, _, v in outs)
        totals.append(total)
        where = "<br>".join(f"{_odds(p)}: {_direction(name, s, s2).replace('slides', 'goes')} "
                            f"{_landing(name, s2, v, pct, s2 == s)}" for p, s2, v in outs)
        rows.append((a, where, total))
    best = [a for a in range(nA) if totals[a] >= max(totals) - TIE_TOL]
    mark = " ★ best" if len(best) == 1 else " ★ tied best"
    hi = ' style="background:#1e3a2f"'
    body = "".join(
        f"<tr{hi if a in best else ''}><td style='font-size:18px'>"
        f"{arrows[a]}</td><td>{where}</td><td><b>{_fmt(t, pct)}</b>{mark if a in best else ''}</td></tr>"
        for a, where, t in rows
    )
    table = (f'<table class="dc" style="border-collapse:collapse"><tr><th>move</th>'
             f"<th>where the ice might send it</th><th>worth (the average)</th></tr>{body}</table>")
    board = grid(layout, V, agent=s, vmax=1.0 if pct else None, pct=pct)
    head = f"<div style='color:{_ACCENT};font-weight:bold;margin-bottom:6px'>{title}</div>" if title else ""
    return (f'<div style="background:{_PANEL};padding:12px;border-radius:6px;display:inline-block">'
            f"{head}{row(board, table)}</div>")


def observation_panel(name, s, Q=None, V=None, gamma=1.0, chosen=None, pct=False):
    """What the agent sees on square s, how it rates each move, and why, in plain words."""
    layout = layout_for(name)
    arrows = "←→" if len(layout) == 1 else "←↓→↑"
    cell = _cell(name, s)
    out = [f"{AGENT} is on <b>{_where(name, s)}</b>"]
    if cell == "H":
        return _panel(out + [f"{HOLE} Fell through the ice. Game over, nothing scored."])
    if cell == "G":
        return _panel(out + [f"{GOAL} Made it! +1, game over."])
    ties = []
    if Q is not None:
        ties = tied(Q, s)
        label = f"chance of reaching the {GOAL}" if pct else "how good it thinks each move is"
        qs = " &nbsp; ".join(
            f"<span style='color:{_HILITE if a in ties else _TEXT}'>{arrows[a]} {_fmt(Q[s, a], pct)}"
            f"{' ★' if a in ties else ''}</span>" for a in range(Q.shape[1])
        )
        out += [f"Each move, rated ({label}):", f"&nbsp;&nbsp;{qs}"]
    if chosen is not None:
        if len(ties) > 1 and chosen in ties:
            others = " ".join(arrows[a] for a in ties)
            out.append(f"It picks <b>{arrows[chosen]}</b>, but {others} are all equally good "
                       "here, so which one it picks does not matter.")
        else:
            out.append(f"It picks <b>{arrows[chosen]}</b>.")
    if V is not None and chosen is not None:
        outs = _outcomes(name, s, chosen, V, gamma)
        total = sum(p * v for p, _, v in outs)
        out.append(f"Why is {arrows[chosen]} worth {_fmt(total, pct)}? The ice decides where it really goes:")
        out += [f"&nbsp;&nbsp;{_odds(p)}: {_direction(name, s, s2)} {_landing(name, s2, v, pct, s2 == s)}"
                for p, s2, v in outs]
        out.append(f"&nbsp;&nbsp;average: <b>{_fmt(total, pct)}</b>")
        out.append("<span style='color:#888'>(That averaging is the Bellman equation: "
                   "Q(s,a) = Σ p · (r + γ·V(s')).)</span>")
    return _panel(out)


def _panel(lines):
    return (f'<div style="font-size:13px;line-height:1.7;background:{_BG};padding:10px 14px;'
            f'border-radius:4px;min-width:340px;max-width:460px">' + "<br>".join(lines) + "</div>")


def record_episode(P, start, max_steps, policy, seed=0):
    """Run one episode; returns the visited states, the actions and the rewards."""
    env = envs.TabularEnv(P, start, max_steps, seed)
    s, done, trunc = env.reset(), False, False
    states, actions, rewards = [s], [], []
    while not (done or trunc):
        a = int(policy(s, env.rng))
        s, r, done, trunc = env.step(a)
        states.append(s)
        actions.append(a)
        rewards.append(r)
    return states, actions, rewards


def replay(name, policy, title, seed=0, values=None, greedy_policy=None, Q=None, gamma=1.0,
           pct=False, panels=True):
    """Record one episode of `policy` on env `name` and play it back.

    `values`/`greedy_policy` shade the board and draw arrows behind the agent.
    `Q` and `values` add the reasoning panel next to the board (panels=False drops it,
    for compact side-by-side reels).
    """
    P, start, max_steps, _ = envs.ENVS[name]
    states, actions, rewards = record_episode(P, start, max_steps, policy, seed)
    caps = step_captions(name, states, actions)
    side = None
    if panels and (Q is not None or values is not None):
        side = [observation_panel(name, s, Q, values, gamma,
                                  actions[t] if t < len(actions) else None, pct)
                for t, s in enumerate(states)]
    board = grid(layout_for(name), values, greedy_policy, vmax=1.0 if pct else None, pct=pct,
                 Q=Q if greedy_policy is not None else None)
    outcome = "reached the goal" if rewards and rewards[-1] > 0 else "did not make it"
    return agent_player(board, states, caps, f"{title}: {outcome} in {len(actions)} steps", side)


def step_captions(name, states, actions):
    """One line per step: what the agent tried and what the ice actually did."""
    arrows = "←→" if len(layout_for(name)) == 1 else "←↓→↑"
    caps = ["press Play, or drag the slider to step through"]
    for t in range(1, len(states)):
        prev, a, s = states[t - 1], actions[t - 1], states[t]
        moved = _direction(name, prev, s)
        if _moved_as_intended(name, prev, a, s):
            what = f"tried {arrows[a]} and it worked"
        elif moved == "stays put":
            what = f"tried {arrows[a]} but stayed put"
        else:
            what = f"tried {arrows[a]}, but the ice <b>{moved}</b> instead"
        end = {"H": f" → fell in a {HOLE}", "G": f" → reached the {GOAL}!"}.get(_cell(name, s), "")
        caps.append(f"step {t}: {what}{end}")
    return caps


def agent_player(board, states, captions, title, panels=None, interval_ms=300):
    """A static board with the agent moved by script: tiny even for 200-step episodes."""
    uid = f"ap{next(_ids)}"
    side = f'<div id="{uid}-p"></div>' if panels else ""
    return f"""
<div style="background:{_PANEL};padding:12px;border-radius:6px;display:inline-block;margin:6px 0">
  <div style="color:{_ACCENT};font-weight:bold;margin-bottom:6px">{html.escape(title)}</div>
  <input type="range" id="{uid}-r" min="0" max="{len(states) - 1}" value="0" style="width:280px">
  <button id="{uid}-b" style="margin-left:8px;padding:3px 12px;background:{_ACCENT};
    color:#fff;border:none;border-radius:4px;cursor:pointer">Play</button>
  <div id="{uid}-c" style="margin-top:6px;font-size:13px;min-height:18px;max-width:560px"></div>
  <div class="row" style="margin-top:8px"><div id="{uid}-g">{board}</div>{side}</div>
</div>
<script>
(function() {{
  const S = {json.dumps([int(x) for x in states])}, C = {json.dumps(captions)}, PN = {json.dumps(panels)};
  const r = document.getElementById("{uid}-r"), b = document.getElementById("{uid}-b");
  const cells = {{}}, orig = {{}};
  document.getElementById("{uid}-g").querySelectorAll("td[data-s]").forEach(function(td) {{
    cells[td.dataset.s] = td; orig[td.dataset.s] = [td.innerHTML, td.style.boxShadow];
  }});
  let timer = null;
  function show() {{
    const t = +r.value;
    for (const k in cells) {{ cells[k].innerHTML = orig[k][0]; cells[k].style.boxShadow = orig[k][1]; }}
    for (let i = 0; i < t; i++) cells[S[i]].style.boxShadow = "inset 0 0 0 2px #fdcb6e55";
    const a = cells[S[t]];
    a.innerHTML = '<span class="ic">{AGENT}</span>';
    a.style.boxShadow = "inset 0 0 0 3px {_AGENT}";
    document.getElementById("{uid}-c").innerHTML = C[t];
    if (PN) document.getElementById("{uid}-p").innerHTML = PN[t];
  }}
  r.oninput = show;
  b.onclick = function() {{
    if (timer) {{ clearInterval(timer); timer = null; b.textContent = "Play"; return; }}
    if (+r.value >= S.length - 1) r.value = 0;
    b.textContent = "Pause";
    timer = setInterval(function() {{
      if (+r.value >= S.length - 1) {{ clearInterval(timer); timer = null; b.textContent = "Play"; return; }}
      r.value = +r.value + 1; show();
    }}, {interval_ms});
  }};
  show();
}})();
</script>"""


def reel(name, checkpoints, P_eval=None, seed=0):
    """"Watch it learn": the greedy plan at several points in training, side by side.

    checkpoints = [(label, Q)]. Every replay uses the same dice (seed), so the only thing
    that differs between them is the plan. Each is captioned with the plan's exact
    chance of reaching the goal, because one lucky game is not a good plan.
    """
    from ch03_planning.planning import policy_evaluation  # here, not at the top: ch03 imports viz

    P, start, _, _ = envs.ENVS[name]
    out = []
    for label, Q in checkpoints:
        pi = Q.argmax(axis=1)
        ok = policy_evaluation(pi, P, 1.0, theta=1e-8, max_sweeps=20_000)[start]
        pol = lambda s, rng, pi=pi: pi[s]
        out.append(replay(name, pol, f"{label} (plan wins {ok:.0%})", seed,
                          values=Q.max(axis=1), greedy_policy=pi, Q=Q, panels=False))
    return row(*out)


def _moved_as_intended(name, s, a, s2):
    layout = layout_for(name)
    rows, cols = len(layout), len(layout[0])
    if rows == 1:
        target = s + (1 if a == envs.RIGHT else -1)
    else:
        dr, dc = envs.LAKE_MOVES[a]
        r, c = divmod(s, cols)
        target = min(max(r + dr, 0), rows - 1) * cols + min(max(c + dc, 0), cols - 1)
    return s2 == target


def find_seed(name, policy, success=True, tries=200):
    """A seed whose episode ends the way we want, so a replay can show both outcomes."""
    P, start, max_steps, _ = envs.ENVS[name]
    for seed in range(tries):
        _, _, rewards = record_episode(P, start, max_steps, policy, seed)
        if (bool(rewards) and rewards[-1] > 0) == success:
            return seed
    return 0


# ── misc ────────────────────────────────────────────────────────────────────────

def figure(path):
    b64 = base64.b64encode(Path(path).read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" style="max-width:100%">'


def text(s):
    return f"<pre>{html.escape(s)}</pre>"


def row(*parts):
    return '<div class="row">' + "".join(parts) + "</div>"
