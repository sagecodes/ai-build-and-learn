"""Chapter 9: more stable value-based methods. DQN and Double DQN on LunarLander.

Chapter 8's NFQ wobbled because two things fight it:

  * correlated data  - consecutive moves are nearly identical (the same two seconds of
                       the same flight), so a batch of them teaches one narrow lesson and
                       overwrites the last one
  * a moving target  - the target r + gamma * max Q(s', .) comes from the very network
                       being trained, so every update moves the goalposts

DQN adds one fix for each, and Double DQN fixes a bias the max leaves behind:

  * replay buffer   - keep the last 100,000 moves; every 4 steps, train on 64 picked
                      at random
  * target network  - compute targets with a frozen copy, refreshed every 1,000 steps
  * Double DQN      - the online network picks the best next move, the frozen copy
                      scores it; the max of noisy estimates no longer rounds up

Four learners, same network and budget, each adding one idea:

    no replay, no target   ->  + replay buffer  ->  + target network (= DQN)  ->  Double DQN

    python -m ch09_dqn.dqn                 # from topics/grokking-rl; writes dqn.png here
"""

import os

# One thread per pool, set before numpy/torch load (and inherited by the learner
# processes). Four learners each spawning a 20-thread OpenBLAS pool fought over the cores:
# 437 s on the host, 1,620 s in a 4-core pod; with this, 77 s.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

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

OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))
ENV_ID = "LunarLander-v3"
ACTIONS = ["do nothing", "fire left engine", "fire main engine", "fire right engine"]
torch.set_num_threads(1)  # one learner per core, see main()


class QNet(nn.Module):
    def __init__(self, n_in=8, n_out=4, hidden=128):
        super().__init__()
        self.l1, self.l2, self.out = nn.Linear(n_in, hidden), nn.Linear(hidden, hidden), nn.Linear(hidden, n_out)

    def forward(self, x, keep=False):
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        q = self.out(h2)
        return (q, h1, h2) if keep else q


VARIANTS = {
    # name: (sample from a shuffled replay buffer?, frozen target network?, double?)
    "no replay, no target": (False, False, False),
    "+ replay buffer": (True, False, False),
    "DQN (+ target network)": (True, True, False),
    "Double DQN": (True, True, True),
}


def act(net, s):
    with torch.no_grad():
        return int(net(torch.as_tensor(s, dtype=torch.float32)).argmax())


def evaluate(net, episodes=5, seed=50_000, gamma=0.99):
    """Mean score of the greedy policy, plus (predicted, actual) discounted value at each
    game's first state: how optimistic the network is about its own play."""
    env = gym.make(ENV_ID)
    scores, pairs = [], []
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i)
        with torch.no_grad():
            predicted = float(net(torch.as_tensor(s, dtype=torch.float32)).max())
        done, total, disc, k = False, 0.0, 0.0, 0
        while not done:
            s, r, term, trunc, _ = env.step(act(net, s))
            total += r
            disc += gamma ** k * r
            k += 1
            done = term or trunc
        scores.append(total)
        pairs.append((predicted, disc))
    return float(np.mean(scores)), pairs


class Memory:
    """Ring buffer of moves as NumPy arrays (random access into a deque is O(n))."""

    def __init__(self, size, n_obs=8):
        self.S, self.S2 = np.zeros((size, n_obs), np.float32), np.zeros((size, n_obs), np.float32)
        self.A, self.R, self.T = np.zeros(size, np.int64), np.zeros(size, np.float32), np.zeros(size, np.float32)
        self.size, self.i, self.n = size, 0, 0

    def add(self, s, a, r, s2, t):
        self.S[self.i], self.A[self.i], self.R[self.i], self.S2[self.i], self.T[self.i] = s, a, r, s2, t
        self.i, self.n = (self.i + 1) % self.size, min(self.n + 1, self.size)

    def last(self, k):
        return (self.i - 1 - np.arange(k)[::-1]) % self.size

    def get(self, idx):
        return (torch.from_numpy(x[idx]) for x in (self.S, self.A, self.R, self.S2, self.T))


def train(variant, episodes, seed=0, gamma=0.99, lr=5e-4, batch=64, buffer=100_000,
          target_every=1000, warmup=1000, train_every=4, eval_every=10, probe=None, snap_at=()):
    replay, frozen, double = VARIANTS[variant]
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = gym.make(ENV_ID)
    net = QNet()
    target = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    mem = Memory(buffer)
    hist, probes, overest, snaps, steps = [], [], [], {}, 0
    for ep in range(episodes):
        eps = max(0.02, 1.0 - ep / (0.5 * episodes))
        s, _ = env.reset(seed=seed * 1_000_000 + ep)
        done, total = False, 0.0
        while not done:
            a = int(rng.integers(4)) if rng.random() < eps else act(net, s)
            s2, r, term, trunc, _ = env.step(a)
            mem.add(s, a, r, s2, float(term))
            s, total, done, steps = s2, total + r, term or trunc, steps + 1
            if mem.n >= warmup and steps % train_every == 0:
                if replay:
                    idx = rng.integers(mem.n, size=batch)
                else:  # the last `batch` moves, in order: one stretch of one flight
                    idx = mem.last(batch)
                S, A, R, S2, T = mem.get(idx)
                with torch.no_grad():
                    scorer = target if frozen else net
                    if double:
                        a2 = net(S2).argmax(dim=1, keepdim=True)
                        nxt = scorer(S2).gather(1, a2).squeeze(1)
                    else:
                        nxt = scorer(S2).max(dim=1).values
                    y = R + gamma * nxt * (1 - T)
                q = net(S).gather(1, A[:, None]).squeeze(1)
                loss = nn.functional.smooth_l1_loss(q, y)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 10)
                opt.step()
            if frozen and steps % target_every == 0:
                target.load_state_dict(net.state_dict())
            if probe is not None and steps % 500 == 0:
                with torch.no_grad():
                    probes.append((steps, float(net(probe).max(dim=1).values.mean())))
        if ep % eval_every == 0 or ep == episodes - 1:
            score, pairs = evaluate(net, gamma=gamma)
            hist.append((ep + 1, total, score))
            overest.append((ep + 1, float(np.mean([p for p, _ in pairs])), float(np.mean([a for _, a in pairs]))))
        if ep + 1 in snap_at:
            snaps[ep + 1] = copy.deepcopy(net.state_dict())
    snaps[episodes] = copy.deepcopy(net.state_dict())
    return net, hist, probes, overest, snaps


def usable_cores():
    """Cores this process may actually use. In a pod, os.cpu_count() reports the whole
    host (20) while the cgroup quota is what the task asked for."""
    try:
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except OSError:
        pass
    return len(os.sched_getaffinity(0))


def probe_states(n=256, seed=123):
    """A fixed set of states from random flights: the same questions asked of every
    learner throughout training, to watch how its answers move."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s, _ = env.reset(seed=int(rng.integers(1 << 30)))
        done = False
        while not done and len(out) < n:
            out.append(s)
            s, _, term, trunc, _ = env.step(int(rng.integers(4)))
            done = term or trunc
    return torch.as_tensor(np.array(out), dtype=torch.float32)


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    probe = probe_states()
    snap_at = (25, 100, 200, 300)
    # The four learners are independent: train them side by side, one per core.
    t = time.perf_counter()
    workers = min(len(VARIANTS), usable_cores())
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        futures = {name: ex.submit(train, name, args.episodes, args.seed, probe=probe, snap_at=snap_at)
                   for name in VARIANTS}
        results = {name: f.result() for name, f in futures.items()}
    print(f"trained {len(VARIANTS)} learners on {workers} cores in {time.perf_counter() - t:.0f}s\n")
    for name, (_, hist, *_rest) in results.items():
        late = np.mean([h[2] for h in hist[len(hist) * 2 // 3:]])
        print(f"{name:>24}: last-third average score {late:7.1f}, final {hist[-1][2]:7.1f}")
    print("\n(200 = solved: a clean landing on the pad)\n")
    print(asciichart.line({k: [h[2] for h in v[1]] for k, v in results.items()}, ymin=-300, ymax=300,
                          title="greedy score (mean of 5 flights)", xmax=args.episodes, xlabel="episode"))

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(1, 3, figsize=(18, 4.8))
    colors = {}
    for name, (_, hist, probes, over, _) in results.items():
        line, = ax[0].plot([h[0] for h in hist], [h[2] for h in hist], label=name)
        colors[name] = line.get_color()
        ax[1].plot([p[0] for p in probes], [p[1] for p in probes], label=name, lw=1, color=colors[name])
    for name in ("DQN (+ target network)", "Double DQN"):
        over = results[name][3]
        ax[2].plot([o[0] for o in over], [o[1] for o in over], color=colors[name], label=f"{name}: predicted")
        ax[2].plot([o[0] for o in over], [o[2] for o in over], ls="--", color=colors[name],
                   label=f"{name}: actual")
    ax[2].set_ylim(-60, 130)  # the first few games dip to -350 and flatten everything else
    ax[0].axhline(200, color="k", ls="--", lw=1, label="solved")
    ax[0].set(title="greedy score", xlabel="episode", ylim=(-400, 320))
    ax[1].set(title="average Q over 256 fixed probe states", xlabel="training step")
    ax[2].set(title="predicted vs actual value of the first state", xlabel="episode")
    for a in ax:
        a.legend(fontsize=8)
    fig.suptitle("Chapter 9: what each DQN idea buys, on LunarLander")
    fig.tight_layout()
    path = OUT / "dqn.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    print(f"\nwrote {path}")
    return _report(results, path, args)


# ── the report ──────────────────────────────────────────────────────────────────

def _load(state):
    net = QNet()
    net.load_state_dict(state)
    return net


def record_flight(net, seed, keep=True, max_steps=1000):
    """One greedy flight, recorded from the simulator for the canvas player."""
    env = gym.make(ENV_ID)
    s, _ = env.reset(seed=seed)
    u = env.unwrapped
    terrain = [[round(p[0][0], 2), round(p[0][1], 2)] for p in u.sky_polys] + \
              [[round(u.sky_polys[-1][1][0], 2), round(u.sky_polys[-1][1][1], 2)]]
    frames, done, total = [], False, 0.0
    while not done and len(frames) < max_steps:
        with torch.no_grad():
            q, h1, h2 = net(torch.as_tensor(s, dtype=torch.float32), keep=True)
        a = int(q.argmax())
        L = u.lander
        f = {"x": round(L.position[0], 3), "y": round(L.position[1], 3), "ang": round(L.angle, 3),
             "legs": [int(l.ground_contact) for l in u.legs], "a": a, "score": round(total, 1)}
        if keep:
            f["q"] = [round(float(v), 1) for v in q]
            f["h"] = [round(float(v), 1) for v in h2]
        frames.append(f)
        s, r, term, trunc, _ = env.step(a)
        total += r
        done = term or trunc
    outcome = ("landed" if total >= 200 else "touched down" if total > 0 else "crashed or drifted off")
    return {"terrain": terrain, "pad": [u.helipad_x1, u.helipad_x2, u.helipad_y]}, frames, total, outcome


def _batch_picture(seed=0):
    """Where 64 training moves come from: one stretch of one flight vs the whole buffer."""
    env = gym.make(ENV_ID)
    rng = np.random.default_rng(seed)
    mem = []
    for ep in range(60):
        s, _ = env.reset(seed=ep)
        done = False
        while not done:
            mem.append(s[:2])
            s, _, term, trunc, _ = env.step(int(rng.integers(4)))
            done = term or trunc
    mem = np.array(mem)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), facecolor="#0f0f23")
    for a, idx, title in ((ax[0], np.arange(len(mem) - 64, len(mem)), "no replay: the last 64 moves"),
                          (ax[1], rng.integers(len(mem), size=64), "replay buffer: 64 random moves")):
        a.set_facecolor("#1a1a2e")
        a.scatter(mem[:, 0], mem[:, 1], s=2, color="#2a3550")
        a.scatter(mem[idx, 0], mem[idx, 1], s=14, color="#fdcb6e")
        a.set_title(title, color="#ccc", fontsize=10)
        a.set_xlim(-1.1, 1.1)
        a.set_ylim(-0.2, 1.6)
        a.tick_params(colors="#666")
        a.set_xlabel("lander x", color="#888")
        a.set_ylabel("lander height", color="#888")
    fig.tight_layout()
    import base64, io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="#0f0f23")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">'


def _report(results, path, args):
    dqn = results["Double DQN"]
    net = _load(dqn[4][args.episodes])
    seed = 4242
    scene, frames, total, outcome = record_flight(net, seed)
    late = {k: float(np.mean([h[2] for h in v[1][len(v[1]) * 2 // 3:]])) for k, v in results.items()}
    rows = "".join(f"<tr><td>{k}</td><td>{v:.0f}</td></tr>" for k, v in late.items())
    # Over-optimism = predicted minus actual value of a flight's first state, averaged
    # over the last third of training (one evaluation is 5 flights: noisy).
    gap = {}
    for k in ("DQN (+ target network)", "Double DQN"):
        o = results[k][3][len(results[k][3]) * 2 // 3:]
        gap[k] = float(np.mean([p - a for _, p, a in o]))
    g_dqn, g_ddqn = gap["DQN (+ target network)"], gap["Double DQN"]
    verdict = ("Double DQN's predictions are the more honest of the two here." if g_ddqn < g_dqn - 2 else
               "On this run the two are about equally honest: the max's bias is small when rewards "
               "are this noisy, and it shows more clearly on longer training.")
    reel = []
    for ep in sorted(dqn[4]):
        sc, fr, tot, out = record_flight(_load(dqn[4][ep]), seed, keep=False)
        reel.append(deepviz.lander_player(sc, fr, f"after {ep} games: {out} ({tot:.0f})", ACTIONS, small=True))
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> two small tricks make deep Q-learning "
        "stable: learn from a shuffled memory of past moves instead of the last few, and aim at "
        "targets from a frozen copy of the network instead of the one you are changing.</p>",
        deepviz.lander_intro(),

        "<h3>1. Four learners, one idea added at a time</h3>",
        viz.figure(path),
        f"<table class='dc'><tr><th>learner</th><th>average score, last third of training</th></tr>{rows}</table>",
        viz.explain(
            how="Left: each learner's greedy score (200 = a clean landing). Middle: the average "
                "Q-value each network gives the same 256 fixed states during training, i.e. how "
                "its opinion moves. Right: for DQN and Double DQN, the value the network "
                "<i>predicts</i> for the first moment of a flight vs what it actually went on to "
                "score from there.",
            notice="Read the middle chart for stability: the learners without a frozen target "
                   "have opinions that drift and jump, because every update moves their own "
                   "targets. Read the right chart for honesty: the gap between solid (predicted) "
                   "and dashed (actual) is over-optimism."),

        "<h3>2. Fix one: the replay buffer breaks up correlated data</h3>",
        _batch_picture(),
        viz.explain(
            how="Grey: lander positions from 60 random flights. Yellow: one training batch.",
            notice="Without replay, the 64 moves in a batch are 64 consecutive frames: one short "
                   "stretch of one flight, nearly identical. Every update teaches the network one "
                   "narrow lesson and overwrites the last. A shuffled sample from the buffer covers "
                   "the whole sky, so each update is a fair summary of everything it has seen."),

        "<h3>3. Fix two: a frozen target network stops the goalposts moving</h3>",
        "<pre>without:  target = r + γ · max Q_net(s', ·)       ← Q_net changes every step, so does the target\n"
        "with:     target = r + γ · max Q_frozen(s', ·)    ← Q_frozen is a copy, refreshed every 1,000 steps\n"
        "Double:   target = r + γ · Q_frozen(s', argmax Q_net(s', ·))   ← one picks, the other scores</pre>",
        viz.explain(notice=f"Over-optimism (predicted minus actual value of a flight's first moment, "
                           f"last third of training): DQN <b>{g_dqn:+.0f}</b>, Double DQN "
                           f"<b>{g_ddqn:+.0f}</b>. Taking the max over noisy estimates picks the "
                           "luckiest one, so plain DQN tends to round up; Double DQN lets one "
                           f"network choose and the other judge. {verdict}"),

        "<h3>4. Watch it fly, and what the network is thinking</h3>",
        deepviz.lander_player(scene, frames, f"Double DQN, end of training: {outcome} ({total:.0f})", ACTIONS),
        viz.explain(
            how="The bars are the network's Q-value for each of the 4 actions on this frame (the "
                "chosen one is green). The strip below is the 128 units of its last hidden layer, "
                "brighter = more active. Flames show which engine is firing.",
            notice="High up, the four Q-values are close: nothing is urgent. Near the ground they "
                   "spread out; the wrong move now costs a crash."),

        "<h3>5. Watch it learn</h3>",
        viz.row(*reel),
        viz.explain(how="Double DQN's network after that many games, all on the same terrain and "
                        "starting push."),
        viz.takeaways([
            "Deep Q-learning on raw experience is unstable: consecutive moves are correlated and "
            "the targets chase the network.",
            "A replay buffer shuffles experience so each batch is a fair sample.",
            "A target network freezes the targets for a while so learning has something fixed to aim at.",
            "Double DQN separates choosing from judging, removing the max's optimism.",
            "Together that is DQN: the recipe that first learned Atari from pixels.",
        ]),
    ]


if __name__ == "__main__":
    main()
