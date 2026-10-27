"""Chapter 11: policy-gradient and actor-critic methods, on LunarLander.

Chapters 8-10 learned values and acted greedily on them. Policy-gradient methods learn
the policy itself: a network that outputs a probability for each action, nudged so
good actions become more likely. Four learners, each adding one idea:

  * REINFORCE      - after each game, raise log pi(a|s) for every move in proportion to
                     the return G that followed it. Unbiased, but G swings wildly from
                     game to game, so learning is slow and noisy.
  * VPG            - subtract a learned baseline V(s): weight each move by G - V(s),
                     "how much better than expected did this go?". Same expected
                     gradient, far less noise. (Plus an entropy bonus to keep exploring.)
  * A2C            - an actor-critic: 8 landers at once, update every 5 steps, and
                     replace the rest of the game with the critic's guess:
                     advantage = r_t + ... + gamma^5 V(s_t+5) - V(s_t)
  * A2C + GAE      - generalized advantage estimation: a lambda-weighted blend of every
                     n-step advantage (TD(lambda) from chapter 5, applied to advantages),
                     over 128-step windows

What this run found (and the report says): on LunarLander with this budget, 5-step A2C
learns to ~+140 and then COLLAPSES on every seed. Not a bug: rollouts of 32 and 128
steps with the same code are stable. With 5 real steps per target, almost all the signal
is the critic's guess, and an immature critic steers the actor off a cliff. GAE's lambda
is the dial: 0.95 (look-ahead ~17 steps) still fails, 0.99 learns, 1.0 learns too. The
lambda-dial experiment in the report measures exactly that.

    python -m ch11_policy_gradient.pg       # from topics/grokking-rl
"""

import os

for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")  # see ch09_dqn/dqn.py: one thread per learner process

import argparse
import copy
import multiprocessing as mp
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

import asciichart
import deepviz
import viz
from ch09_dqn.dqn import ACTIONS, ENV_ID, usable_cores

OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))
torch.set_num_threads(1)


def mlp(n_out):
    return nn.Sequential(nn.Linear(8, 128), nn.Tanh(), nn.Linear(128, 128), nn.Tanh(), nn.Linear(128, n_out))


def clip(net):
    """Clip the actor and the critic separately: the critic's loss is in squared score
    units (hundreds squared), and one shared clip let its gradients swamp the actor's."""
    nn.utils.clip_grad_norm_(net.actor.parameters(), 0.5)
    nn.utils.clip_grad_norm_(net.critic.parameters(), 0.5)


class ActorCritic(nn.Module):
    """Two separate networks: the actor (policy: action logits) and the critic (V(s))."""

    def __init__(self):
        super().__init__()
        self.actor, self.critic = mlp(4), mlp(1)

    def dist(self, s):
        return torch.distributions.Categorical(logits=self.actor(s))

    def value(self, s):
        return self.critic(s).squeeze(-1)


VARIANTS = ["REINFORCE", "VPG (+ baseline)", "A2C (5-step critic)", "A2C + GAE (λ=0.99)"]
LR = {v: 7e-4 for v in VARIANTS}
LAMBDA_DIAL = (0.9, 0.95, 0.98, 0.99, 1.0)


class Tracker:
    """Training-game scores against environment steps, for curves on a common x-axis."""

    def __init__(self, every):
        self.every, self.scores, self.curve, self.next = every, [], [], every

    def add(self, score):
        self.scores.append(score)

    def tick(self, steps):
        while steps >= self.next:
            recent = self.scores[-20:]
            self.curve.append((self.next, float(np.mean(recent)) if recent else float("nan")))
            self.next += self.every


def train(variant, budget, seed=0, gamma=0.99, lr=None, ent=0.01, curve_every=10_000, snap_at=(),
          norm_adv=None, lam=0.99):
    if lr is None:
        lr = LR.get(variant, 7e-4)
    if norm_adv is None:
        norm_adv = variant.startswith("A2C")
    torch.manual_seed(seed)
    net = ActorCritic()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # Step size decays linearly to zero over the budget (for every learner, to keep the
    # race fair). At a constant rate A2C's ~25,000 small noisy updates learned to ~+140
    # and then wandered off and collapsed to ~-120 on every seed.
    opt.decay = lambda steps: [g.__setitem__("lr", lr * max(0.0, 1 - steps / budget)) for g in opt.param_groups]
    track, snaps = Tracker(curve_every), {}
    stats = {"weights": []}  # gradient weights (G or advantage) seen in updates, sampled
    snap_at = sorted(snap_at)
    if variant in ("REINFORCE", "VPG (+ baseline)"):
        _train_episodic(variant, net, opt, budget, seed, gamma, ent, track, snaps, snap_at, stats)
    else:
        _train_a2c(variant, net, opt, budget, seed, gamma, ent, track, snaps, snap_at, stats, norm_adv,
                   lam=lam)
    snaps[budget] = copy.deepcopy(net.state_dict())
    return track.curve, snaps, stats


def _train_episodic(variant, net, opt, budget, seed, gamma, ent, track, snaps, snap_at, stats):
    """REINFORCE and VPG: play a whole game, then one update from its returns."""
    env = gym.make(ENV_ID)
    steps, ep = 0, 0
    while steps < budget:
        s, _ = env.reset(seed=seed * 1_000_000 + ep)
        S, A, R, done = [], [], [], False
        while not done:
            with torch.no_grad():
                a = int(net.dist(torch.as_tensor(s, dtype=torch.float32)).sample())
            s2, r, term, trunc, _ = env.step(a)
            S.append(s), A.append(a), R.append(r)
            s, done = s2, term or trunc
        steps += len(R)
        ep += 1
        track.add(sum(R))
        track.tick(steps)
        G = np.zeros(len(R), np.float32)
        acc = 0.0
        for t in reversed(range(len(R))):
            acc = R[t] + gamma * acc
            G[t] = acc
        S_t, A_t, G_t = torch.as_tensor(np.array(S)), torch.as_tensor(A), torch.as_tensor(G)
        dist = net.dist(S_t)
        logp = dist.log_prob(A_t)
        if variant == "REINFORCE":
            weight = G_t
            loss = -(logp * weight).mean()
        else:
            V = net.value(S_t)
            weight = (G_t - V).detach()
            loss = -(logp * weight).mean() + 0.5 * ((G_t - V) ** 2).mean() - ent * dist.entropy().mean()
        if ep % 5 == 0:
            stats["weights"].extend(weight[:: max(1, len(R) // 20)].tolist())
        opt.decay(steps)
        opt.zero_grad()
        loss.backward()
        clip(net)
        opt.step()
        while snap_at and steps >= snap_at[0]:
            snaps[snap_at.pop(0)] = copy.deepcopy(net.state_dict())


def _train_a2c(variant, net, opt, budget, seed, gamma, ent, track, snaps, snap_at, stats,
               norm_adv=True, n_envs=8, n_steps=5, lam=0.95):
    """A2C: 8 landers in lockstep, an update every n_steps, bootstrapping from the critic."""
    gae = "GAE" in variant
    if gae:
        n_steps = 128  # GAE blends over a long window; lambda does the weighting
    envs = gym.make_vec(ENV_ID, num_envs=n_envs, vectorization_mode="sync",
                        vector_kwargs={"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP})
    s, _ = envs.reset(seed=[seed * 1_000_000 + i for i in range(n_envs)])
    running = np.zeros(n_envs)
    steps, updates = 0, 0
    while steps < budget:
        S, A, R, D, V = [], [], [], [], []
        for _ in range(n_steps):
            st = torch.as_tensor(s, dtype=torch.float32)
            with torch.no_grad():
                a = net.dist(st).sample().numpy()
                v = net.value(st).numpy()
            s2, r, term, trunc, info = envs.step(a)
            # A game cut off by the time limit is not a failure: bootstrap from where it
            # really ended (SAME_STEP autoreset puts that state in final_obs).
            boot = np.zeros(n_envs, np.float32)
            if trunc.any():
                fo = np.stack([info["final_obs"][i] if trunc[i] else s2[i] for i in range(n_envs)])
                with torch.no_grad():
                    boot = np.where(trunc & ~term, net.value(torch.as_tensor(fo, dtype=torch.float32)).numpy(), 0.0)
            S.append(s), A.append(a), R.append(r + gamma * boot), D.append(term | trunc), V.append(v)
            running += r
            for i in np.flatnonzero(term | trunc):
                track.add(running[i])
                running[i] = 0.0
            s = s2
            steps += n_envs
        track.tick(steps)
        with torch.no_grad():
            last_v = net.value(torch.as_tensor(s, dtype=torch.float32)).numpy()
        R, D, V = np.array(R), np.array(D, np.float32), np.array(V)
        adv = np.zeros_like(R)
        if gae:
            g = np.zeros(n_envs)
            for t in reversed(range(n_steps)):
                nv = last_v if t == n_steps - 1 else V[t + 1]
                delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
                g = delta + gamma * lam * (1 - D[t]) * g
                adv[t] = g
            ret = adv + V
        else:
            ret = np.zeros_like(R)
            acc = last_v
            for t in reversed(range(n_steps)):
                acc = R[t] + gamma * acc * (1 - D[t])
                ret[t] = acc
            adv = ret - V
        S_t = torch.as_tensor(np.array(S).reshape(-1, 8), dtype=torch.float32)
        A_t = torch.as_tensor(np.array(A).reshape(-1))
        adv_t = torch.as_tensor(adv.reshape(-1), dtype=torch.float32)
        updates += 1
        if updates % 25 == 0:  # raw advantages, before rescaling, so spreads are comparable
            stats["weights"].extend(adv_t[::8].tolist())
        if norm_adv:  # rescale each batch's advantages: only their order and relative size matter
            adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        ret_t = torch.as_tensor(ret.reshape(-1), dtype=torch.float32)
        dist = net.dist(S_t)
        v_pred = net.value(S_t)
        loss = (-(dist.log_prob(A_t) * adv_t).mean() + 0.5 * ((ret_t - v_pred) ** 2).mean()
                - ent * dist.entropy().mean())
        opt.decay(steps)
        opt.zero_grad()
        loss.backward()
        clip(net)
        opt.step()
        while snap_at and steps >= snap_at[0]:
            snaps[snap_at.pop(0)] = copy.deepcopy(net.state_dict())


def _late(curve):
    v = [c[1] for c in curve if np.isfinite(c[1])]
    return float(np.mean(v[len(v) * 2 // 3:])) if v else float("nan")


def _verdict(st, base, base_name):
    diff = st["mean"] - base["mean"]
    noise = 2 * np.sqrt(st["sem"] ** 2 + base["sem"] ** 2)
    if diff > noise:
        return f"clearly better than {base_name} (+{diff:.0f}, noise ±{noise:.0f})"
    if diff < -noise:
        return f"clearly worse than {base_name} ({diff:.0f}, noise ±{noise:.0f})"
    return f"no clear difference from {base_name} ({diff:+.0f}, noise ±{noise:.0f})"


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000, help="environment steps per learner")
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--dial-seeds", type=int, default=4, help="seeds per lambda in the dial")
    args = ap.parse_args(argv)

    snap_at = tuple(int(args.steps * f) for f in (0.05, 0.25, 0.5, 0.75))
    t = time.perf_counter()
    workers = usable_cores()
    jobs = [(name, seed) for seed in range(args.seeds) for name in VARIANTS]
    dial_jobs = [(lam, seed) for seed in range(args.dial_seeds) for lam in LAMBDA_DIAL]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {(n, s): ex.submit(train, n, args.steps, s, snap_at=snap_at if s == 0 else ())
                   for n, s in jobs}
        dial_f = {(l, s): ex.submit(train, "A2C + GAE", args.steps, 100 + s, lam=l) for l, s in dial_jobs}
        runs = {k: f.result() for k, f in futures.items()}
        dial = {k: f.result()[0] for k, f in dial_f.items()}
    print(f"trained {len(jobs) + len(dial_jobs)} runs ({len(VARIANTS)} learners x {args.seeds} seeds + "
          f"a {len(LAMBDA_DIAL)}-point lambda dial x {args.dial_seeds} seeds, {args.steps:,} steps each) "
          f"on {workers} cores in {time.perf_counter() - t:.0f}s\n")

    stats = {}
    for name in VARIANTS:
        late = np.array([_late(runs[name, s][0]) for s in range(args.seeds)])
        curves = np.array([[c[1] for c in runs[name, s][0]] for s in range(args.seeds)])
        w = np.array(runs[name, 0][2]["weights"])
        stats[name] = {"late": late, "mean": float(np.nanmean(late)),
                       "sem": float(np.nanstd(late, ddof=1) / np.sqrt(len(late))),
                       "curves": curves, "x": [c[0] for c in runs[name, 0][0]],
                       "w_std": float(w.std()) if len(w) else float("nan")}
    prev = None
    for name, st in stats.items():
        st["verdict"] = "baseline" if prev is None else _verdict(st, stats[prev], prev)
        print(f"{name:>24}: training score, last third {st['mean']:7.1f} ± {st['sem']:5.1f}   "
              f"gradient-weight spread {st['w_std']:6.1f}   {st['verdict']}")
        prev = name
    dial_stats = {}
    for lam in LAMBDA_DIAL:
        v = np.array([_late(dial[lam, s]) for s in range(args.dial_seeds)])
        dial_stats[lam] = (float(v.mean()), float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)
    print("\nGAE lambda dial (effective look-ahead ~ 1 / (1 - gamma * lambda) steps):")
    print(asciichart.bars({f"λ={l} (~{1 / (1 - 0.99 * l):.0f} steps)": m for l, (m, _) in dial_stats.items()},
                          title="last-third training score", fmt="{:+.0f}"))
    print()
    print(asciichart.line({k: np.nanmean(st["curves"], axis=0) for k, st in stats.items()},
                          ymin=-300, ymax=300, title=f"training score, mean of {args.seeds} seeds",
                          xmax=args.steps, xlabel="environment steps"))

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    for i, (k, st) in enumerate(stats.items()):
        m = np.nanmean(st["curves"], axis=0)
        lo, hi = np.nanpercentile(st["curves"], [25, 75], axis=0)
        line, = ax[0].plot(st["x"], m, label=k)
        ax[0].fill_between(st["x"], lo, hi, color=line.get_color(), alpha=0.12)
        jitter = np.random.default_rng(0).uniform(-0.15, 0.15, len(st["late"]))
        ax[1].scatter(i + jitter, st["late"], s=14, color=line.get_color())
        ax[1].errorbar(i, st["mean"], yerr=2 * st["sem"], color="k", capsize=6, lw=2)
    ax[0].axhline(200, color="k", ls="--", lw=1, label="solved")
    ax[0].set(xlabel="environment steps", ylabel="training score (last 20 games)", ylim=(-400, 320),
              title=f"mean of {args.seeds} seeds (band: middle half of seeds)")
    ax[0].legend(fontsize=9)
    ax[1].set_xticks(range(len(stats)), [k.replace(" (", "\n(") for k in stats], fontsize=8)
    ax[1].set(ylabel="last-third training score", title="every seed (bar: mean ± 2 standard errors)")
    fig.suptitle("Chapter 11: from REINFORCE to actor-critic, on LunarLander")
    fig.tight_layout()
    path = OUT / "pg.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")
    return _report(runs, stats, dial_stats, path, args)


# ── the report ──────────────────────────────────────────────────────────────────

def _load(state):
    net = ActorCritic()
    net.load_state_dict(state)
    return net


def record_policy_flight(net, seed, max_steps=1000, sample_seed=0):
    """One flight sampled from the policy, with its probabilities and the critic's V."""
    env = gym.make(ENV_ID)
    s, _ = env.reset(seed=seed)
    gen = torch.Generator().manual_seed(sample_seed)
    u = env.unwrapped
    terrain = [[round(p[0][0], 2), round(p[0][1], 2)] for p in u.sky_polys] + \
              [[round(u.sky_polys[-1][1][0], 2), round(u.sky_polys[-1][1][1], 2)]]
    frames, done, total = [], False, 0.0
    while not done and len(frames) < max_steps:
        st = torch.as_tensor(s, dtype=torch.float32)
        with torch.no_grad():
            p = net.dist(st).probs
            v = float(net.value(st))
        a = int(torch.multinomial(p, 1, generator=gen))
        L = u.lander
        frames.append({"x": round(L.position[0], 3), "y": round(L.position[1], 3), "ang": round(L.angle, 3),
                       "legs": [int(l.ground_contact) for l in u.legs], "a": a, "score": round(total, 1),
                       "pi": [round(float(x), 3) for x in p], "v": round(v, 1)})
        s, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    outcome = "landed" if total >= 200 else "touched down" if total > 0 else "crashed or drifted off"
    return {"terrain": terrain, "pad": [u.helipad_x1, u.helipad_x2, u.helipad_y]}, frames, total, outcome


def _weights_picture(runs):
    fig, ax = plt.subplots(figsize=(7.5, 3.4), facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")
    for name in VARIANTS:
        w = np.array(runs[name, 0][2]["weights"])
        if len(w):
            ax.hist(np.clip(w, -300, 300), bins=80, histtype="step", density=True, lw=1.6,
                    label=f"{name}: spread {w.std():.0f}")
    ax.set_xlabel("weight on log π(a|s) in an update (G for REINFORCE, the advantage otherwise)", color="#ccc")
    ax.set_yscale("log")
    ax.tick_params(colors="#888")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ccc")
    fig.tight_layout()
    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="#0f0f23")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _dial_picture(dial_stats, a2c):
    fig, ax = plt.subplots(figsize=(7, 3.6), facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")
    lams = list(dial_stats)
    m = [dial_stats[l][0] for l in lams]
    e = [2 * dial_stats[l][1] for l in lams]
    xs = [1 / (1 - 0.99 * l) for l in lams]
    ax.errorbar(xs, m, yerr=e, color="#fdcb6e", marker="o", capsize=4, label="A2C + GAE, by λ")
    for x, l, y in zip(xs, lams, m):
        ax.annotate(f"λ={l}", (x, y), textcoords="offset points", xytext=(6, -14), color="#ccc", fontsize=9)
    ax.scatter([5], [a2c], color="#e17055", s=50, zorder=3, label="A2C, 5-step (mean of all seeds)")
    ax.set_xscale("log")
    ax.set_xlabel("how far ahead it looks before trusting the critic (steps, log scale)", color="#ccc")
    ax.set_ylabel("last-third training score", color="#ccc")
    ax.tick_params(colors="#888")
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ccc")
    fig.tight_layout()
    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="#0f0f23")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _report(runs, stats, dial_stats, path, args):
    best = "A2C + GAE (λ=0.99)"
    _, snaps, _ = runs[best, 0]
    net = _load(snaps[args.steps])
    scene, frames, total, outcome = record_policy_flight(net, 4242)
    # Measured, not assumed: how random the policy is, and what the critic expects, by altitude.
    bands = {"high up": [], "middle": [], "near the ground": []}
    for sd in range(4242, 4250):
        sc, fr, _, _ = record_policy_flight(net, sd)
        for f in fr:
            alt = (f["y"] - sc["pad"][2]) / 6.67
            p = np.array(f["pi"])
            k = "high up" if alt > 0.8 else "near the ground" if alt < 0.25 else "middle"
            bands[k].append((float(-(p * np.log(p + 1e-9)).sum() / np.log(4)), f["v"]))
    band = {k: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for k, v in bands.items() if v}
    band_txt = "; ".join(f"{k}: {r:.0%} random, V ≈ {v:+.0f}" for k, (r, v) in band.items())
    rows = "".join(f"<tr><td>{k}</td><td>{st['mean']:.0f} ± {st['sem']:.0f}</td>"
                   f"<td>{st['w_std']:.0f}</td><td>{st['verdict']}</td></tr>" for k, st in stats.items())
    reel = []
    for step in sorted(snaps):
        sc, fr, tot, out = record_policy_flight(_load(snaps[step]), 4242)
        for f in fr:
            f.pop("pi"), f.pop("v")
        reel.append(deepviz.lander_player(sc, fr, f"after {step:,} steps: {out} ({tot:.0f})", ACTIONS, small=True))
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> instead of estimating how good every "
        "action is and picking the best, learn the policy directly: a network that outputs "
        "probabilities, and make the moves that turned out well more likely.</p>",
        deepviz.lander_intro(),
        '<div class="card">Same lander as chapters 9 and 10, but a different kind of learner. '
        "A value method (DQN) asks \"how good is each action?\" and takes the best one. A "
        "<b>policy-gradient</b> method asks the network directly \"what should I do?\" and gets "
        "back odds, like <i>main engine 70%, left 20%, nothing 8%, right 2%</i>, then rolls the "
        "dice. Learning means shifting those odds towards moves that worked out.</div>",

        "<h3>1. A policy at work: odds, not scores</h3>",
        deepviz.lander_player(scene, frames, f"A2C + GAE, end of training: {outcome} ({total:.0f})",
                              ACTIONS),
        viz.explain(
            how="The bars are the <b>actor's</b> odds for each action on this frame (they add up "
                "to 100%); the agent rolls the dice with them, and the green bar is what came up. "
                "The meter is the <b>critic's</b> estimate V(s): how well it expects the flight to go "
                "from here. The critic never chooses anything; it only grades.",
            notice=f"Measured over 8 flights ({band_txt}; 100% = a uniform coin over all four): "
                   "the policy <b>never fully commits</b>. Even after a million steps it keeps a lot "
                   "of randomness everywhere, partly from the entropy bonus that rewards it for "
                   "exploring. That is a big reason its scores sit below chapter 9 and 10's value "
                   "learners, which act greedily. The critic is most hopeful high up, with the whole "
                   "landing bonus still ahead, and least hopeful low down, where much of the time "
                   "goes on hovering and burning fuel."),

        "<h3>2. Four learners, one idea added at a time</h3>",
        viz.figure(path),
        f"<table class='dc'><tr><th>learner</th><th>training score, last third (mean ± s.e.)</th>"
        f"<th>spread of update weights</th><th>vs the learner above it</th></tr>{rows}</table>",
        "<pre>REINFORCE   weight = G_t                     the whole rest of the game's score\n"
        "VPG         weight = G_t − V(s_t)            better or worse than the critic expected?\n"
        "A2C         weight = r_t + … + γ⁵·V(s_t+5) − V(s_t)   5 real steps, then the critic's guess\n"
        "GAE         weight = Σ (γλ)^k · δ_t+k        a λ-blend of every n-step version</pre>",
        viz.explain(
            how=f"Each learner had the same budget of {args.steps:,} environment steps and was "
                f"trained {args.seeds} times. The verdict compares each learner with the one "
                "directly above it, and only calls a difference real if it beats the noise between "
                "seeds. The <b>weight</b> is how hard an update pushes on a move's probability; its "
                "spread is how noisy those pushes are.",
            notice=f"REINFORCE pushes every move by the whole game's score (spread "
                   f"{stats['REINFORCE']['w_std']:.0f}), so one lucky or unlucky ending shoves "
                   "hundreds of moves the same way. Subtracting the critic's expectation (VPG) leaves "
                   f"only the surprise (spread {stats['VPG (+ baseline)']['w_std']:.0f}). "
                   "<b>Then the surprise:</b> 5-step A2C, the textbook upgrade, does worst of all. "
                   "Section 3 shows why."),
        _weights_picture(runs),
        viz.explain(how="The distribution of those weights during training (seed 0, log scale, "
                        "before A2C's per-batch rescaling). Narrower means less noisy gradients."),

        "<h3>3. Why A2C failed: the bias-variance dial</h3>",
        _dial_picture(dial_stats, stats["A2C (5-step critic)"]["mean"]),
        viz.explain(
            how="Every point is A2C + GAE with a different λ (4 seeds each, bars ±2 standard "
                "errors), placed by how many steps of real reward it effectively uses before "
                "trusting the critic's guess, about 1 / (1 − γλ). The red dot is 5-step A2C.",
            notice="5-step A2C learns to about +140 and then collapses on every seed. It is not a "
                   "bug: the same code with 32- or 128-step rollouts is stable. With only 5 real steps "
                   "per target, nearly all the learning signal is the critic's guess, and early on "
                   "that guess is wrong: the actor follows it off a cliff. This is chapter 5's "
                   "bias-variance trade-off, live. Lean on the critic (short look-ahead): low "
                   "variance, high bias. Lean on real rewards (long look-ahead): unbiased, noisier. "
                   "Here, with a young critic, the long end wins; with a better critic (more training, "
                   "a bigger network, or PPO's safeguards in chapter 12) the sweet spot moves left."),

        "<h3>4. Watch it learn</h3>",
        viz.row(*reel),
        viz.explain(how="A2C + GAE's policy at points in training, on the same terrain, sampled with "
                        "the same dice."),
        viz.takeaways([
            "A policy-gradient agent learns the odds of each action directly.",
            "REINFORCE: make moves more likely in proportion to the return that followed. Unbiased, noisy.",
            "A baseline (the critic's V) removes most of that noise without changing what it learns.",
            "Actor-critic: the critic grades every few steps, so the actor need not wait for the game "
            "to end. But a critic is only as good as its training: bootstrapping from a bad one "
            "steers the actor wrong.",
            "GAE's λ is a dial between trusting the critic and trusting real rewards, the same idea "
            "as chapter 5's TD(λ). Where to set it depends on how good the critic is.",
            "Next: continuous actions (how hard to fire, not which engine), chapter 12.",
        ]),
    ]


if __name__ == "__main__":
    main()
