"""Chapter 10: sample-efficient value-based methods. Dueling DDQN and prioritized replay.

Chapter 9's Double DQN works, but it is wasteful in two ways:

  * one output per action has to learn everything: how good this moment is AND how
    much each engine matters, even though in most moments the actions barely differ
  * it replays every memory equally often, so a rare crash or landing (where all the
    learning is) gets the same attention as 1,000 frames of drifting through space

Two fixes, each changing one part of chapter 9's learner:

  * Dueling network      - split the head in two: V(s), how good this spot is, and
                           A(s, a), how much better each action is than average;
                           Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
  * Prioritized replay   - replay memories in proportion to how surprising they were
    (PER)                  (|TD error| ^ alpha), and down-weight them by importance
                           sampling so the average update stays honest

Four learners on chapter 9's LunarLander, all with Double DQN targets:

    Double DQN  ->  + dueling  ->  + PER  ->  dueling + PER

    python -m ch10_dueling_per.dueling_per      # from topics/grokking-rl
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
from ch09_dqn.dqn import ACTIONS, ENV_ID, Memory, QNet, act, evaluate, usable_cores

OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))
torch.set_num_threads(1)


class DuelingNet(nn.Module):
    """A shared body, then two heads: V(s) (one number) and A(s, a) (one per action)."""

    def __init__(self, n_in=8, n_out=4, hidden=128):
        super().__init__()
        self.l1 = nn.Linear(n_in, hidden)
        self.v1, self.v = nn.Linear(hidden, hidden), nn.Linear(hidden, 1)
        self.a1, self.a = nn.Linear(hidden, hidden), nn.Linear(hidden, n_out)

    def forward(self, x, keep=False):
        h = torch.relu(self.l1(x))
        v = self.v(torch.relu(self.v1(h)))
        adv = self.a(torch.relu(self.a1(h)))
        # Subtracting the mean pins down the split: otherwise "+5 to V, -5 to every A"
        # gives the same Q and the two heads drift.
        q = v + adv - adv.mean(dim=-1, keepdim=True)
        return (q, v.squeeze(-1), adv) if keep else q


class PrioritizedMemory(Memory):
    """Chapter 9's ring buffer plus a priority per memory.

    Sampling is proportional to priority via a cumulative sum and a binary search:
    O(N) per batch, fine at 100k. (At millions you would use a sum-tree.)
    """

    def __init__(self, size, alpha=0.6, eps=1e-3):
        super().__init__(size)
        self.P = np.zeros(size, np.float64)
        self.replayed = np.zeros(size, bool)  # new memories carry max priority until replayed once
        self.alpha, self.eps = alpha, eps

    def add(self, s, a, r, s2, t):
        # New memories get the current maximum priority, so each is replayed at least once soon.
        self.P[self.i] = self.P[: self.n].max() if self.n else 1.0
        self.replayed[self.i] = False
        super().add(s, a, r, s2, t)

    def sample(self, batch, beta, rng):
        p = self.P[: self.n]
        cum = np.cumsum(p)
        idx = np.minimum(np.searchsorted(cum, rng.random(batch) * cum[-1]), self.n - 1)
        prob = p[idx] / cum[-1]
        w = (self.n * prob) ** (-beta)
        return idx, torch.as_tensor(w / w.max(), dtype=torch.float32)

    def update(self, idx, td):
        self.P[idx] = (np.abs(td) + self.eps) ** self.alpha
        self.replayed[idx] = True


VARIANTS = {
    # name: (dueling network?, prioritized replay?)
    "Double DQN": (False, False),
    "+ dueling": (True, False),
    "+ prioritized replay": (False, True),
    "dueling + prioritized": (True, True),
}


def train(variant, episodes, seed=0, gamma=0.99, lr=5e-4, batch=64, buffer=100_000,
          target_every=1000, warmup=1000, train_every=4, eval_every=10, probe=None, snap_at=()):
    dueling, per = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = gym.make(ENV_ID)
    net = DuelingNet() if dueling else QNet()
    target = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mem = PrioritizedMemory(buffer) if per else Memory(buffer)
    hist, snaps, steps, total_steps_est = [], {}, 0, episodes * 300
    for ep in range(episodes):
        eps = max(0.02, 1.0 - ep / (0.5 * episodes))
        beta = min(1.0, 0.4 + 0.6 * ep / episodes)  # importance-sampling correction, annealed to 1
        s, _ = env.reset(seed=seed * 1_000_000 + ep)
        done, total = False, 0.0
        while not done:
            a = int(rng.integers(4)) if rng.random() < eps else act(net, s)
            s2, r, term, trunc, _ = env.step(a)
            mem.add(s, a, r, s2, float(term))
            s, total, done, steps = s2, total + r, term or trunc, steps + 1
            if mem.n >= warmup and steps % train_every == 0:
                if per:
                    idx, w = mem.sample(batch, beta, rng)
                else:
                    idx, w = rng.integers(mem.n, size=batch), torch.ones(batch)
                S, A, R, S2, T = mem.get(idx)
                with torch.no_grad():
                    a2 = net(S2).argmax(dim=1, keepdim=True)          # Double DQN:
                    y = R + gamma * target(S2).gather(1, a2).squeeze(1) * (1 - T)  # pick, then score
                q = net(S).gather(1, A[:, None]).squeeze(1)
                td = y - q
                loss = (w * nn.functional.smooth_l1_loss(q, y, reduction="none")).mean()
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10)
                opt.step()
                if per:
                    mem.update(idx, td.detach().numpy())
            if steps % target_every == 0:
                target.load_state_dict(net.state_dict())
        if ep % eval_every == 0 or ep == episodes - 1:
            score, _ = evaluate(net, gamma=gamma)
            hist.append((ep + 1, total, score))
        if ep + 1 in snap_at:
            snaps[ep + 1] = copy.deepcopy(net.state_dict())
    snaps[episodes] = copy.deepcopy(net.state_dict())
    replay = _replay_summary(mem, rng) if per else None
    return hist, snaps, replay, dueling


def _replay_summary(mem, rng, n=4000):
    """What prioritized replay pays attention to, computed in the worker so 100k memories
    are not shipped back between processes.

    Memories never replayed yet are left out of the "most replayed" table and the map:
    they all carry the current maximum priority by construction (so each gets replayed
    soon), which says nothing about what PER has learned to care about.
    """
    p = mem.P[: mem.n] / mem.P[: mem.n].sum()
    R, T, S = mem.R[: mem.n], mem.T[: mem.n], mem.S[: mem.n]
    seen = mem.replayed[: mem.n]
    big = (np.abs(R) >= 50)  # crash (-100) or coming to rest (+100): the episode-deciding moments
    rank = np.where(seen, p, -1.0)
    top = np.argsort(-rank)[:10]
    top50 = np.argsort(-rank)[:50]
    ids = np.flatnonzero(seen)
    pick = rng.choice(ids, size=min(n, len(ids)), replace=False)
    return {
        "n": int(mem.n),
        "unreplayed": float(1 - seen.mean()),
        "big_share_of_memories": float(big.mean()),
        "big_share_of_replays": float(p[big].sum()),
        "top50_big": float(big[top50].mean()),
        "top50_low": float((S[top50, 1] < 0.1).mean()),
        "top": [{"x": float(S[i, 0]), "y": float(S[i, 1]), "vx": float(S[i, 2]), "vy": float(S[i, 3]),
                 "a": int(mem.A[i]), "r": float(R[i]), "end": bool(T[i]), "times": float(p[i] * mem.n)}
                for i in top],
        "scatter": {"x": S[pick, 0].tolist(), "y": S[pick, 1].tolist(), "p": (p[pick] * mem.n).tolist()},
    }


def _late(hist):
    return float(np.mean([h[2] for h in hist[len(hist) * 2 // 3:]]))


def _first_100(hist):
    """Sample efficiency: the episode where the 3-evaluation moving average first passes 100."""
    sc = np.array([h[2] for h in hist])
    hit = np.flatnonzero(np.convolve(sc, np.ones(3) / 3, mode="valid") >= 100)
    return int(hist[hit[0] + 2][0]) if len(hit) else None


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=16)
    args = ap.parse_args(argv)

    # One seed is not enough to compare these: LunarLander runs vary a lot, and a
    # single-seed ranking of these four came out baseline-first. 16 seeds per learner.
    t = time.perf_counter()
    workers = usable_cores()
    jobs = [(name, seed) for seed in range(args.seeds) for name in VARIANTS]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {(name, seed): ex.submit(train, name, args.episodes, seed,
                                           snap_at=(50, 150, 300) if seed == 0 else ())
                   for name, seed in jobs}
        runs = {key: f.result() for key, f in futures.items()}
    print(f"trained {len(jobs)} runs ({len(VARIANTS)} learners x {args.seeds} seeds) on "
          f"{workers} cores in {time.perf_counter() - t:.0f}s\n")

    stats = {}
    for name in VARIANTS:
        late = np.array([_late(runs[name, s][0]) for s in range(args.seeds)])
        first = [_first_100(runs[name, s][0]) for s in range(args.seeds)]
        reached = [f for f in first if f is not None]
        curves = np.array([[h[2] for h in runs[name, s][0]] for s in range(args.seeds)])
        stats[name] = {"late": late, "mean": late.mean(), "sem": late.std(ddof=1) / np.sqrt(len(late)),
                       "first_median": float(np.median(reached)) if reached else None,
                       "reached": len(reached), "curves": curves,
                       "x": [h[0] for h in runs[name, 0][0]]}
    base = stats["Double DQN"]
    for name, st in stats.items():
        st["verdict"] = _verdict(st, base) if name != "Double DQN" else "baseline"
        fm = f"{st['first_median']:.0f}" if st["first_median"] else "never"
        print(f"{name:>24}: last-third average {st['mean']:6.1f} ± {st['sem']:4.1f}  "
              f"steady 100+ (median episode) {fm:>5} [{st['reached']}/{args.seeds} seeds]  {st['verdict']}")
    rep = runs["dueling + prioritized", 0][2]
    print(f"\nprioritized replay (seed 0): crash/rest moments are {rep['big_share_of_memories']:.1%} of "
          f"memories but get {rep['big_share_of_replays']:.1%} of replays")
    print()
    print(asciichart.line({k: st["curves"].mean(axis=0) for k, st in stats.items()}, ymin=-300, ymax=300,
                          title=f"greedy score, mean of {args.seeds} seeds", xmax=args.episodes,
                          xlabel="episode"))

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [2, 1]})
    for k, st in stats.items():
        m = st["curves"].mean(axis=0)
        lo, hi = np.percentile(st["curves"], [25, 75], axis=0)
        line, = ax[0].plot(st["x"], m, label=k)
        ax[0].fill_between(st["x"], lo, hi, color=line.get_color(), alpha=0.12)
        ax[1].scatter(np.full(len(st["late"]), list(stats).index(k)) + np.random.default_rng(0).uniform(-0.15, 0.15, len(st["late"])),
                      st["late"], s=14, color=line.get_color())
        ax[1].errorbar(list(stats).index(k), st["mean"], yerr=2 * st["sem"], color="k", capsize=6, lw=2)
    ax[0].axhline(200, color="k", ls="--", lw=1, label="solved")
    ax[0].set(xlabel="episode", ylabel="greedy score", ylim=(-400, 320),
              title=f"mean of {args.seeds} seeds (band: middle half of seeds)")
    ax[0].legend(fontsize=9)
    ax[1].set_xticks(range(len(stats)), [k.replace(" + ", "\n+ ").replace("dueling + ", "dueling\n+ ") for k in stats], fontsize=8)
    ax[1].set(ylabel="last-third average score", title="every seed (bar: mean ± 2 standard errors)")
    fig.suptitle("Chapter 10: dueling heads and prioritized replay, on LunarLander")
    fig.tight_layout()
    path = OUT / "dueling_per.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")
    return _report(runs, stats, path, args)


def _verdict(st, base):
    """Better / worse / no clear difference, judged against the noise (about 2 standard
    errors of the difference), never on the raw gap alone."""
    diff = st["mean"] - base["mean"]
    noise = 2 * np.sqrt(st["sem"] ** 2 + base["sem"] ** 2)
    if diff > noise:
        return f"clearly better than Double DQN (+{diff:.0f}, noise ±{noise:.0f})"
    if diff < -noise:
        return f"clearly worse than Double DQN ({diff:.0f}, noise ±{noise:.0f})"
    return f"no clear difference from Double DQN ({diff:+.0f}, noise ±{noise:.0f})"


# ── the report ──────────────────────────────────────────────────────────────────

def _load(state, dueling):
    net = DuelingNet() if dueling else QNet()
    net.load_state_dict(state)
    return net


def record_dueling(net, seed, max_steps=1000):
    """One greedy flight with V and A recorded on every frame."""
    env = gym.make(ENV_ID)
    s, _ = env.reset(seed=seed)
    u = env.unwrapped
    terrain = [[round(p[0][0], 2), round(p[0][1], 2)] for p in u.sky_polys] + \
              [[round(u.sky_polys[-1][1][0], 2), round(u.sky_polys[-1][1][1], 2)]]
    frames, done, total = [], False, 0.0
    while not done and len(frames) < max_steps:
        with torch.no_grad():
            q, v, adv = net(torch.as_tensor(s, dtype=torch.float32), keep=True)
        a = int(q.argmax())
        L = u.lander
        adv = adv - adv.mean()
        frames.append({"x": round(L.position[0], 3), "y": round(L.position[1], 3), "ang": round(L.angle, 3),
                       "legs": [int(l.ground_contact) for l in u.legs], "a": a, "score": round(total, 1),
                       "v": round(float(v), 1), "adv": [round(float(x), 2) for x in adv]})
        s, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    outcome = "landed" if total >= 200 else "touched down" if total > 0 else "crashed or drifted off"
    return {"terrain": terrain, "pad": [u.helipad_x1, u.helipad_x2, u.helipad_y]}, frames, total, outcome


def _priority_picture(rep):
    sc = rep["scatter"]
    order = np.argsort(sc["p"])
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")
    pts = ax.scatter(np.array(sc["x"])[order], np.array(sc["y"])[order], c=np.array(sc["p"])[order],
                     s=6, cmap="inferno", norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=max(sc["p"])))
    cb = fig.colorbar(pts, ax=ax)
    cb.set_label("replayed ×  as often as average", color="#ccc")
    cb.ax.tick_params(colors="#888")
    ax.set_xlabel("lander x", color="#888")
    ax.set_ylabel("lander height", color="#888")
    ax.tick_params(colors="#666")
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-0.2, 1.6)
    fig.tight_layout()
    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="#0f0f23")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _report(runs, stats, path, args):
    hist, snaps, rep, _ = runs["dueling + prioritized", 0]
    net = _load(snaps[args.episodes], True)
    scene, frames, total, outcome = record_dueling(net, 4242)
    def row(k, st):
        first = f"{st['first_median']:.0f}" if st["first_median"] else "never"
        return (f"<tr><td>{k}</td><td>{st['mean']:.0f} ± {st['sem']:.0f}</td>"
                f"<td>{first} ({st['reached']}/{args.seeds} seeds)</td><td>{st['verdict']}</td></tr>")

    rows = "".join(row(k, st) for k, st in stats.items())
    top = "".join(
        f"<tr><td>{t['times']:.0f}×</td><td>{t['y']:.2f}</td><td>{t['vy']:+.2f}</td>"
        f"<td>{ACTIONS[t['a']]}</td><td>{t['r']:+.1f}</td>"
        f"<td>{('yes: flew off the edge' if abs(t['x']) >= 0.99 else 'yes: crashed') if t['end'] and t['r'] < 0 else 'yes: came to rest' if t['end'] else 'no'}</td></tr>"
        for t in rep["top"])
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> learn more from the same experience, by "
        "giving the network a better shape (dueling) and by spending more replay time on the "
        "memories that still surprise it (prioritized replay).</p>",
        deepviz.lander_intro(),
        '<div class="card">This chapter keeps chapter 9\'s lander and its best learner, Double DQN, '
        "and changes one part at a time. The question is not \"can it land?\" (chapter 9 "
        "showed it can) but <b>how fast it learns to</b>, from how many flights.</div>",

        "<h3>1. Four learners, one change at a time</h3>",
        viz.figure(path),
        f"<table class='dc'><tr><th>learner</th><th>last-third average (mean ± standard error)</th>"
        f"<th>first steady 100+ (median episode)</th><th>verdict</th></tr>{rows}</table>",
        viz.explain(
            how=f"All four use chapter 9's Double DQN targets and the same budget of games, and each "
                f"was trained <b>{args.seeds} times with different random seeds</b>. Left: the "
                "average learning curve, with a band covering the middle half of the seeds. Right: "
                "every seed's result as a dot. \"First steady 100+\" is when three evaluations in a "
                "row first average above 100: <b>sample efficiency</b>, how few games it needs.",
            notice="The verdict column only calls a difference real when it is bigger than the "
                   "noise between seeds (about two standard errors). One seed is not enough: a "
                   "single run of these four ranked the plain baseline first. Read the dots "
                   "before the averages: the spread between seeds of the <i>same</i> learner is "
                   "often bigger than the gap between learners."),

        "<h3>2. The dueling network: two questions instead of one</h3>",
        "<pre>chapter 9:   state → body → Q(s, ←engine), Q(s, main), Q(s, →engine), Q(s, nothing)\n"
        "dueling:     state → body ─┬→ V(s)       how good is this moment, whatever I do?\n"
        "                           └→ A(s, a)    how much better is each action than average?\n"
        "             Q(s, a) = V(s) + A(s, a) − mean(A)</pre>",
        deepviz.lander_player(scene, frames, f"dueling + prioritized, end of training: {outcome} ({total:.0f})",
                              ACTIONS),
        viz.explain(
            how="Left: the flight. Right: the network's two heads on this frame. <b>V</b> is its "
                "verdict on the moment itself (a big number = it expects a good landing from here). "
                "The bars are <b>A</b>: how much better or worse each action is than the average "
                "action, right now.",
            notice="High up and drifting, the A bars are short: no single move matters much, so the "
                   "network learns almost everything through V, which every flight updates. Near the "
                   "ground they spread out: the choice of engine now decides crash or landing. A "
                   "plain Q network has to learn \"this moment is good\" separately for every action; "
                   "the dueling one learns it once."),

        "<h3>3. Prioritized replay: study the surprising moments</h3>",
        viz.row(_priority_picture(rep),
                f'<div class="card" style="max-width:360px">Of its last {rep["n"]:,} memories, '
                f"<b>{rep['big_share_of_memories']:.1%}</b> are the big moments (a crash or coming to "
                f"rest, ±100 points). Uniform replay would study them {rep['big_share_of_memories']:.1%} "
                f"of the time. Prioritized replay studies them <b>{rep['big_share_of_replays']:.1%}</b> "
                "of the time.</div>"),
        f"<table class='dc'><tr><th>replayed</th><th>height</th><th>falling speed</th><th>action</th>"
        f"<th>reward</th><th>game over?</th></tr>{top}</table>",
        viz.explain(
            how="Each dot is a memory at the lander's position, coloured by how often prioritized "
                "replay picks it compared with average (1× = what uniform replay would give every "
                "memory). The table is its ten most-replayed memories. Brand-new memories are left "
                "out of both: PER gives every new memory the current top priority so it gets "
                "replayed at least once, which says nothing about what it has learned to care about.",
            notice=f"Of its 50 most-replayed memories, <b>{rep['top50_big']:.0%}</b> are a crash or "
                   f"coming to rest, and {rep['top50_low']:.0%} are on the ground. Those are the "
                   "moments its predictions keep getting wrong, so it keeps studying them, instead of "
                   "letting thousands of uneventful frames of drifting drown them out. The "
                   "importance-sampling weights shrink each of those extra updates, so the average "
                   "lesson is not distorted.<br><b>Did it help here?</b> Not clearly: see the verdicts "
                   "in section 1. Prioritized replay pays off most when the important moments are "
                   "rare <i>and</i> hard; on LunarLander, uniform replay already sees enough landings "
                   "by the time it matters. Dueling is the change that made the measurable difference."),

        viz.takeaways([
            "Sample efficiency is about learning more from each game, not just learning eventually.",
            "Dueling splits Q into V (how good is this moment) and A (how much each action matters), "
            "so the shared part is learned once.",
            "Prioritized replay studies memories in proportion to how surprising they were.",
            "Importance-sampling weights keep that biased sampling from biasing the result.",
            "Next: stop estimating values for every action and learn the policy directly "
            "(chapter 11, policy gradients).",
        ]),
    ]


if __name__ == "__main__":
    main()
