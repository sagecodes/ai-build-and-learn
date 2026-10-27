"""Report pieces for the deep RL chapters (8+): continuous worlds and the networks
that play them.

  cartpole_intro()   - the story card and a labelled drawing of the four inputs
  cartpole_player()  - a game replayed on a canvas from the recorded states
  network_player()   - the game next to the network, every unit lighting up with its
                       activation on that frame, the strongest weights drawn as lines
  q_heatmap()        - what the network prefers, over pole angle x pole spin

Like viz.py: self-contained HTML + inline JS, so it renders inside a Flyte report.
"""

import base64
import html
import io
import itertools
import json

import numpy as np

import viz

_ids = itertools.count()
_PANEL, _ACCENT = "#1a1a2e", "#00b894"

_CART_JS = """
function drawCart(cv, s, a) {
  const g = cv.getContext("2d"), W = cv.width, H = cv.height;
  g.clearRect(0, 0, W, H);
  const y = H * 0.72, scale = W / 5.6, cx = W / 2 + s[0] * scale;
  g.fillStyle = "#2a3550"; g.fillRect(0, y + 14, W, 4);                      // track
  g.fillStyle = "#4a1d2b"; g.fillRect(W / 2 - 2.4 * scale - 3, y - 4, 3, 22);  // ends
  g.fillRect(W / 2 + 2.4 * scale, y - 4, 3, 22);
  g.fillStyle = "#ccc"; g.fillRect(cx - 24, y - 6, 48, 20);                  // cart
  const L = H * 0.55, tx = cx + L * Math.sin(s[2]), ty = y - 6 - L * Math.cos(s[2]);
  const danger = Math.min(1, Math.abs(s[2]) / 0.21);
  g.strokeStyle = `rgb(${200 + 55 * danger},${200 - 120 * danger},${110 - 60 * danger})`;
  g.lineWidth = 7; g.lineCap = "round";
  g.beginPath(); g.moveTo(cx, y - 6); g.lineTo(tx, ty); g.stroke();       // pole
  if (a !== null && a !== undefined) {                                    // the push
    g.fillStyle = "#00b894"; g.font = "bold 22px sans-serif";
    g.fillText(a === 1 ? "→" : "←", a === 1 ? cx + 30 : cx - 52, y + 10);
  }
}
function cartWords(s) {
  const deg = s[2] * 180 / Math.PI, side = deg >= 0 ? "right" : "left";
  const fall = s[3] >= 0 ? "right" : "left";
  return `pole leaning ${Math.abs(deg).toFixed(1)}° ${side}, swinging ${fall}; ` +
         `cart ${Math.abs(s[0]).toFixed(2)} m ${s[0] >= 0 ? "right" : "left"} of centre`;
}
"""


def cartpole_intro():
    return (
        '<div class="card"><b>CartPole.</b> A cart on a short track with a pole balanced on '
        "top, hinged at the bottom. Every step the agent pushes the cart left or right (no "
        "\"do nothing\"). It scores +1 for every step the pole stays up. The game ends if the "
        "pole tilts more than 12°, if the cart runs off either end of the track, or after "
        "500 steps, which is a perfect game. <b>The agent sees four numbers</b>, not a "
        "picture: cart position, cart speed, pole angle, and pole spin (how fast the angle is "
        "changing). An untrained agent drops the pole in about 10 steps.</div>"
    )


def _player_shell(uid, title, n, body, interval_ms):
    return f"""
<div style="background:{_PANEL};padding:12px;border-radius:6px;display:inline-block;margin:6px 0">
  <div style="color:{_ACCENT};font-weight:bold;margin-bottom:6px">{html.escape(title)}</div>
  <input type="range" id="{uid}-r" min="0" max="{n - 1}" value="0" style="width:260px">
  <button id="{uid}-b" style="margin-left:8px;padding:3px 12px;background:{_ACCENT};
    color:#fff;border:none;border-radius:4px;cursor:pointer">Play</button>
  <div id="{uid}-c" style="margin-top:6px;font-size:13px;min-height:18px;max-width:640px"></div>
  {body}
</div>"""


def _player_js(uid, frames, render, interval_ms):
    return f"""
<script>
(function() {{
  {_CART_JS}
  const F = {json.dumps(frames, separators=(",", ":"))};
  const r = document.getElementById("{uid}-r"), b = document.getElementById("{uid}-b");
  let timer = null;
  function show() {{ const t = +r.value, f = F[t]; {render} }}
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


def cartpole_player(frames, title, small=False, interval_ms=40):
    """frames = [{"s": [x, v, theta, omega], "a": 0 or 1}, ...]"""
    uid = f"cp{next(_ids)}"
    w, h = (260, 150) if small else (420, 220)
    body = f'<div style="margin-top:6px"><canvas id="{uid}-cv" width="{w}" height="{h}"></canvas></div>'
    n = len(frames)
    render = (f'drawCart(document.getElementById("{uid}-cv"), f.s, f.a);'
              f'document.getElementById("{uid}-c").textContent = `step ${{t + 1}} of {n}: ` + cartWords(f.s);')
    return _player_shell(uid, title, n, body, interval_ms) + _player_js(uid, frames, render, interval_ms)


def _layer_xy(n, x, top, height):
    gap = height / max(n, 1)
    return [(x, top + gap * (i + 0.5)) for i in range(n)]


def _network_svg(uid, net, inputs, actions, keep=(48, 96, 32)):
    """Static drawing: nodes get ids so the player can colour them per frame; only the
    strongest weights of each layer are drawn, or it is a solid block of lines."""
    W = [net.l1.weight.detach().numpy(), net.l2.weight.detach().numpy(), net.out.weight.detach().numpy()]
    sizes = [W[0].shape[1], W[0].shape[0], W[1].shape[0], W[2].shape[0]]
    xs = [70, 190, 290, 400]
    cols = [_layer_xy(n, x, 20, 330) for n, x in zip(sizes, xs)]
    parts = []
    for li, (w, k) in enumerate(zip(W, keep)):
        flat = np.abs(w).ravel()
        cut = np.sort(flat)[-min(k, flat.size)]
        wmax = flat.max()
        for j, i in zip(*np.nonzero(np.abs(w) >= cut)):
            (x1, y1), (x2, y2) = cols[li][i], cols[li + 1][j]
            col = "#e17055" if w[j, i] > 0 else "#74b9ff"
            op = 0.15 + 0.6 * abs(w[j, i]) / wmax
            parts.append(f'<line x1="{x1}" y1="{y1:.1f}" x2="{x2}" y2="{y2:.1f}" stroke="{col}" '
                         f'stroke-opacity="{op:.2f}" stroke-width="1"/>')
    names = ["in", "h1", "h2", "q"]
    for li, col in enumerate(cols):
        r = 9 if li in (0, 3) else 4.5
        for i, (x, y) in enumerate(col):
            parts.append(f'<circle id="{uid}-{names[li]}{i}" cx="{x}" cy="{y:.1f}" r="{r}" '
                         f'fill="#2a3550" stroke="#555"/>')
    for i, (x, y) in enumerate(cols[0]):
        parts.append(f'<text x="{x - 14}" y="{y + 4:.1f}" text-anchor="end" font-size="11" fill="#ccc">{inputs[i]}</text>'
                     f'<text id="{uid}-inv{i}" x="{x - 14}" y="{y + 17:.1f}" text-anchor="end" font-size="11" fill="#fdcb6e"></text>')
    for i, (x, y) in enumerate(cols[3]):
        parts.append(f'<text x="{x + 16}" y="{y - 4:.1f}" font-size="12" fill="#ccc">Q({actions[i]})</text>'
                     f'<text id="{uid}-qv{i}" x="{x + 16}" y="{y + 12:.1f}" font-size="13" font-weight="bold" fill="#fff"></text>')
    labels = [("4 inputs", xs[0]), ("hidden layer 1", xs[1]), ("hidden layer 2", xs[2]), ("2 outputs", xs[3])]
    parts += [f'<text x="{x}" y="365" text-anchor="middle" font-size="11" fill="#888">{t}</text>' for t, x in labels]
    return f'<svg width="520" height="372" style="background:#0f0f23;border-radius:4px">{"".join(parts)}</svg>'


def network_player(frames, net, inputs, actions, title, interval_ms=40):
    """The game on the left, the network on the right, both stepping together.

    frames = [{"s", "a", "h1", "h2", "q"}, ...] from a recorded greedy game.
    """
    uid = f"np{next(_ids)}"
    n = len(frames)
    h1max = max(max(f["h1"]) for f in frames) or 1.0
    h2max = max(max(f["h2"]) for f in frames) or 1.0
    scale = [2.4, 2.0, 0.21, 2.0]
    body = (f'<div class="row" style="margin-top:6px;align-items:center">'
            f'<canvas id="{uid}-cv" width="380" height="210"></canvas>'
            f'{_network_svg(uid, net, inputs, actions)}</div>'
            f'<div id="{uid}-why" style="font-size:13px;margin-top:6px;max-width:900px"></div>')
    render = f"""
    drawCart(document.getElementById("{uid}-cv"), f.s, f.a);
    document.getElementById("{uid}-c").textContent = `step ${{t + 1}} of {n}: ` + cartWords(f.s);
    const heat = (v, m) => {{ const k = Math.max(0, Math.min(1, v / m));
      return `rgb(${{42 + 211 * k}},${{53 + 150 * k}},${{80 + 30 * k}})`; }};
    const sc = {json.dumps(scale)};
    f.s.forEach((v, i) => {{
      const k = Math.min(1, Math.abs(v) / sc[i]);
      document.getElementById("{uid}-in" + i).setAttribute("fill", v >= 0 ? `rgba(225,112,85,${{0.2 + 0.8 * k}})` : `rgba(116,185,255,${{0.2 + 0.8 * k}})`);
      document.getElementById("{uid}-inv" + i).textContent = v.toFixed(2);
    }});
    f.h1.forEach((v, i) => document.getElementById("{uid}-h1" + i).setAttribute("fill", heat(v, {h1max})));
    f.h2.forEach((v, i) => document.getElementById("{uid}-h2" + i).setAttribute("fill", heat(v, {h2max})));
    f.q.forEach((v, i) => {{
      document.getElementById("{uid}-q" + i).setAttribute("fill", i === f.a ? "#00b894" : "#2a3550");
      document.getElementById("{uid}-qv" + i).textContent = v.toFixed(1) + (i === f.a ? "  ★ chosen" : "");
    }});
    const gap = Math.abs(f.q[1] - f.q[0]);
    const pick = f.a === 1 ? "→" : "←";
    document.getElementById("{uid}-why").innerHTML = gap < 0.5
      ? `Both pushes look about equally good (${{f.q[0].toFixed(1)}} vs ${{f.q[1].toFixed(1)}}): nothing urgent, it pushes ${{pick}}.`
      : `<b>Push ${{pick}}</b> is clearly better here (${{Math.max(...f.q).toFixed(1)}} vs ${{Math.min(...f.q).toFixed(1)}}): the network expects the other push to drop the pole sooner.`;
    """
    return _player_shell(uid, title, n, body, interval_ms) + _player_js(uid, frames, render, interval_ms)


def q_heatmap(net, inputs, size=(5.2, 3.6)):
    """The network's preferred push over pole angle (x) and pole spin (y), with the
    cart still and centred: the 'table' as a smooth picture."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    th = np.linspace(-0.21, 0.21, 121)
    om = np.linspace(-2.0, 2.0, 121)
    T, O = np.meshgrid(th, om)
    X = np.stack([np.zeros_like(T), np.zeros_like(T), T, O], axis=-1).reshape(-1, 4)
    with torch.no_grad():
        q = net(torch.as_tensor(X, dtype=torch.float32)).numpy()
    diff = (q[:, 1] - q[:, 0]).reshape(T.shape)
    lim = max(np.percentile(np.abs(diff), 95), 1e-6)
    fig, ax = plt.subplots(figsize=size, facecolor="#0f0f23")
    ax.set_facecolor("#0f0f23")
    im = ax.imshow(diff, origin="lower", extent=[th[0] * 57.3, th[-1] * 57.3, om[0], om[-1]],
                   aspect="auto", cmap="coolwarm", vmin=-lim, vmax=lim)
    ax.axvline(0, color="#888", lw=0.5)
    ax.axhline(0, color="#888", lw=0.5)
    ax.set_xlabel("pole angle (degrees; + = leaning right)", color="#ccc")
    ax.set_ylabel("pole spin (+ = falling right)", color="#ccc")
    ax.tick_params(colors="#888")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("prefers push →   (Q→ − Q←)", color="#ccc")
    cb.ax.tick_params(colors="#888")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="#0f0f23")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


# ── LunarLander ─────────────────────────────────────────────────────────────────

def lander_intro():
    return (
        '<div class="card"><b>LunarLander.</b> A lander starts high above the moon with a random '
        "push and has to settle gently on the pad between the two flags. Each step it picks one "
        "of four actions: do nothing, fire the left engine, fire the main engine, or fire the "
        "right engine (side engines tilt and nudge it). Moving towards the pad and slowing down "
        "earns points, each leg touching down earns +10, firing costs a little fuel, crashing "
        "costs −100 and coming to rest earns +100. A score of <b>200</b> or more is a clean "
        "landing. The agent sees eight numbers: position, velocity, tilt, spin, and whether "
        "each leg is touching the ground.</div>"
    )


_LANDER_JS = """
function drawLander(cv, scene, f, small) {
  const g = cv.getContext("2d"), W = cv.width, H = cv.height, sx = W / 20, sy = H / 13.333;
  const X = x => x * sx, Y = y => H - y * sy;
  g.fillStyle = "#0f0f23"; g.fillRect(0, 0, W, H);
  g.fillStyle = "#3a3f55"; g.beginPath(); g.moveTo(0, H);
  scene.terrain.forEach(p => g.lineTo(X(p[0]), Y(p[1]))); g.lineTo(W, H); g.closePath(); g.fill();
  const [x1, x2, py] = scene.pad;
  [x1, x2].forEach(x => { g.strokeStyle = "#ccc"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(X(x), Y(py)); g.lineTo(X(x), Y(py) - 22 * (small ? 0.6 : 1)); g.stroke();
    g.fillStyle = "#fdcb6e"; g.fillRect(X(x), Y(py) - 22 * (small ? 0.6 : 1), 10 * (small ? 0.6 : 1), 6); });
  g.save(); g.translate(X(f.x), Y(f.y)); g.rotate(-f.ang);
  const k = sx / 30, body = [[-14,17],[-17,0],[-17,-10],[17,-10],[17,0],[14,17]];
  if (f.a === 2) { g.fillStyle = "#e17055"; g.beginPath(); g.moveTo(-6*k, 10*k); g.lineTo(6*k, 10*k);
    g.lineTo(0, (24 + 8 * Math.random()) * k); g.closePath(); g.fill(); }
  if (f.a === 1 || f.a === 3) { const s = f.a === 1 ? -1 : 1; g.fillStyle = "#e17055";
    g.beginPath(); g.moveTo(s*17*k, -4*k); g.lineTo(s*17*k, 2*k); g.lineTo(s*28*k, -1*k); g.closePath(); g.fill(); }
  g.strokeStyle = "#aaa"; g.lineWidth = 2;
  [[-1, 0], [1, 1]].forEach(([s, i]) => { g.strokeStyle = f.legs[i] ? "#00b894" : "#aaa";
    g.beginPath(); g.moveTo(s*12*k, 8*k); g.lineTo(s*20*k, 22*k); g.stroke(); });
  g.fillStyle = "#c8c8e8"; g.beginPath(); body.forEach((p, i) => (i ? g.lineTo : g.moveTo).call(g, p[0]*k, -p[1]*k));
  g.closePath(); g.fill();
  g.restore();
}
"""


def lander_player(scene, frames, title, actions, small=False, interval_ms=30):
    """A flight on a canvas; with Q-values and hidden activations if the frames have them."""
    uid = f"ll{next(_ids)}"
    w, h = (300, 200) if small else (480, 320)
    has_net = "q" in frames[0]
    policy = "pi" in frames[0]
    duel = "v" in frames[0] and not policy
    side = ""
    if policy:
        vals = [f["v"] for f in frames]
        vlo, vhi = min(vals), max(vals)
        bars = "".join(
            f'<div style="font-size:12px;margin:3px 0">{html.escape(actions[i])}'
            f'<div style="background:#222;width:220px;height:12px"><div id="{uid}-p{i}" '
            f'style="height:12px;width:0"></div></div><span id="{uid}-pv{i}" style="font-size:11px"></span></div>'
            for i in range(len(actions)))
        side = (f'<div style="min-width:260px"><div style="font-size:12px;color:#888">the actor\'s odds '
                f'for each action</div>{bars}<div style="font-size:12px;color:#888;margin-top:10px">'
                f'the critic: V(s), how well it expects this flight to go</div>'
                f'<div style="background:#222;width:220px;height:16px;margin:3px 0">'
                f'<div id="{uid}-vb" style="height:16px;background:#fdcb6e"></div></div>'
                f'<div id="{uid}-vv" style="font-size:13px;font-weight:bold"></div></div>')
    if duel:
        vals = [f["v"] for f in frames]
        vlo, vhi = min(vals), max(vals)
        amax = max(max(abs(x) for x in f["adv"]) for f in frames) or 1.0
        bars = "".join(
            f'<div style="font-size:12px;margin:3px 0">{html.escape(actions[i])}'
            f'<div style="position:relative;background:#222;width:220px;height:12px">'
            f'<div style="position:absolute;left:110px;width:1px;height:12px;background:#666"></div>'
            f'<div id="{uid}-a{i}" style="position:absolute;height:12px"></div></div>'
            f'<span id="{uid}-av{i}" style="font-size:11px"></span></div>' for i in range(len(actions)))
        side = (f'<div style="min-width:260px"><div style="font-size:12px;color:#888">V(s): how good is '
                f'this moment?</div><div style="background:#222;width:220px;height:16px;margin:3px 0">'
                f'<div id="{uid}-vb" style="height:16px;background:#fdcb6e"></div></div>'
                f'<div id="{uid}-vv" style="font-size:13px;font-weight:bold"></div>'
                f'<div style="font-size:12px;color:#888;margin-top:10px">A(s, a): how much better than '
                f'average is each action?</div>{bars}</div>')
    if has_net and not duel:
        bars = "".join(
            f'<div style="font-size:12px;margin:3px 0">{html.escape(actions[i])}'
            f'<div style="background:#222;width:220px;height:12px"><div id="{uid}-b{i}" '
            f'style="height:12px;width:0"></div></div><span id="{uid}-v{i}" style="font-size:11px"></span></div>'
            for i in range(len(actions)))
        nh = len(frames[0]["h"])
        strip = "".join(f'<span id="{uid}-h{i}" style="display:inline-block;width:7px;height:7px;margin:0 1px 1px 0"></span>'
                        + ("<br>" if (i + 1) % 32 == 0 else "") for i in range(nh))
        side = (f'<div style="min-width:260px"><div style="font-size:12px;color:#888">Q-value per action</div>{bars}'
                f'<div style="font-size:12px;color:#888;margin-top:8px">last hidden layer ({nh} units)</div>'
                f'<div style="line-height:7px">{strip}</div></div>')
    body = (f'<div class="row" style="margin-top:6px;align-items:flex-start">'
            f'<canvas id="{uid}-cv" width="{w}" height="{h}"></canvas>{side}</div>')
    hmax = max((max(f["h"]) for f in frames), default=1.0) if has_net else 1.0
    net_js = ""
    if policy:
        net_js = f"""
        f.pi.forEach((p, i) => {{
          const b = document.getElementById("{uid}-p" + i);
          b.style.width = (220 * p) + "px";
          b.style.background = i === f.a ? "#00b894" : "#555";
          document.getElementById("{uid}-pv" + i).textContent = (100 * p).toFixed(0) + "%" + (i === f.a ? "  ← the dice picked this" : "");
        }});
        document.getElementById("{uid}-vb").style.width = (8 + 212 * (f.v - {vlo}) / Math.max({vhi} - {vlo}, 1e-6)) + "px";
        document.getElementById("{uid}-vv").textContent = "V = " + f.v.toFixed(1);
        """
    if duel:
        net_js = f"""
        document.getElementById("{uid}-vb").style.width = (8 + 212 * (f.v - {vlo}) / Math.max({vhi} - {vlo}, 1e-6)) + "px";
        document.getElementById("{uid}-vv").textContent = "V = " + f.v.toFixed(1);
        f.adv.forEach((x, i) => {{
          const b = document.getElementById("{uid}-a" + i), w = 108 * Math.abs(x) / {amax};
          b.style.width = w + "px"; b.style.left = (x >= 0 ? 110 : 110 - w) + "px";
          b.style.background = i === f.a ? "#00b894" : (x >= 0 ? "#e17055" : "#74b9ff");
          document.getElementById("{uid}-av" + i).textContent = (x >= 0 ? "+" : "") + x.toFixed(2) + (i === f.a ? "  ★ chosen" : "");
        }});
        """
    if has_net and not duel:
        net_js = f"""
        const lo = Math.min(...f.q), hi = Math.max(...f.q), span = Math.max(hi - lo, 1e-6);
        f.q.forEach((v, i) => {{
          const b = document.getElementById("{uid}-b" + i);
          b.style.width = (20 + 200 * (v - lo) / span) + "px";
          b.style.background = i === f.a ? "#00b894" : "#555";
          document.getElementById("{uid}-v" + i).textContent = v.toFixed(1) + (i === f.a ? "  ★ chosen" : "");
        }});
        f.h.forEach((v, i) => {{ const k = Math.min(1, v / {hmax});
          document.getElementById("{uid}-h" + i).style.background = `rgb(${{42 + 211 * k}},${{53 + 150 * k}},${{80 + 30 * k}})`; }});
        """
    acts = json.dumps(actions)
    render = f"""
    drawLander(document.getElementById("{uid}-cv"), SC, f, {str(small).lower()});
    const alt = (f.y - SC.pad[2]).toFixed(1), tilt = (f.ang * 180 / Math.PI).toFixed(0);
    document.getElementById("{uid}-c").textContent =
      `step ${{t + 1}} of {len(frames)}: ${{alt}} m above the pad, tilted ${{tilt}}°, score so far ${{f.score}}; ` +
      {acts}[f.a] + (f.legs[0] || f.legs[1] ? " (legs down)" : "");
    {net_js}
    """
    js = _player_js(uid, frames, render, interval_ms).replace(
        "{_CART_JS}", "").replace("const F =", f"{_LANDER_JS}\n  const SC = {json.dumps(scene)};\n  const F =", 1)
    return _player_shell(uid, title, len(frames), body, interval_ms) + js


# ── Pendulum ────────────────────────────────────────────────────────────────────

def pendulum_player(frames, title, interval_ms=50):
    """frames = [{"th": angle (0 = upright), "u": torque, "score", optional "mean"/"std"
    (SAC's bell curve) or "q": [q1, q2] (TD3's twin critics)}]"""
    uid = f"pd{next(_ids)}"
    extra = "std" in frames[0] or "q" in frames[0]
    body = (f'<div style="margin-top:6px"><canvas id="{uid}-cv" width="230" height="230"></canvas>'
            f'<div id="{uid}-x" style="font-size:12px;min-height:{34 if extra else 0}px"></div></div>')
    render = f"""
    const cv = document.getElementById("{uid}-cv"), g = cv.getContext("2d"), W = cv.width, c = W / 2, L = W * 0.36;
    g.fillStyle = "#0f0f23"; g.fillRect(0, 0, W, W);
    g.strokeStyle = "#2a3550"; g.lineWidth = 1; g.beginPath(); g.arc(c, c, L, 0, 2 * Math.PI); g.stroke();
    const arc = (u, col, w) => {{ const span = Math.PI * 0.8 * u / 2;
      g.strokeStyle = col; g.lineWidth = w; g.beginPath();
      g.arc(c, c, 26, -Math.PI / 2, -Math.PI / 2 - span, span > 0); g.stroke(); }};
    if (f.std !== undefined) {{
      const lo = Math.max(-2, f.mean - 2 * f.std * 2), hi = Math.min(2, f.mean + 2 * f.std * 2);
      g.globalAlpha = 0.3; arc(hi, "#fdcb6e", 14); arc(lo, "#fdcb6e", 14); g.globalAlpha = 1; }}
    arc(f.u, "#e17055", 7);
    const x = c + L * Math.sin(f.th), y = c - L * Math.cos(f.th);
    const up = Math.cos(f.th);
    g.strokeStyle = up > 0.95 ? "#00b894" : "#c8c8e8"; g.lineWidth = 8; g.lineCap = "round";
    g.beginPath(); g.moveTo(c, c); g.lineTo(x, y); g.stroke();
    g.fillStyle = "#ccc"; g.beginPath(); g.arc(c, c, 6, 0, 2 * Math.PI); g.fill();
    const deg = (f.th * 180 / Math.PI).toFixed(0);
    document.getElementById("{uid}-c").textContent =
      `step ${{t + 1}}/{len(frames)}: pole at ${{deg}}° from upright, torque ${{f.u >= 0 ? "+" : ""}}${{f.u.toFixed(2)}}` +
      (Math.abs(f.u) > 1.9 ? " (full power: pumping)" : "") + (up > 0.95 ? " (balanced)" : "");
    const X = document.getElementById("{uid}-x");
    if (f.std !== undefined) X.textContent = `bell curve: centre ${{f.mean.toFixed(2)}}, spread ±${{f.std.toFixed(2)}} (before squashing)`;
    if (f.q !== undefined) X.textContent = `twin critics on this torque: Q1 = ${{f.q[0]}}, Q2 = ${{f.q[1]}}; it trusts the lower, ${{Math.min(...f.q).toFixed(1)}}`;
    """
    js = _player_js(uid, frames, render, interval_ms).replace("{_CART_JS}", "")
    return _player_shell(uid, title, len(frames), body, interval_ms) + js
