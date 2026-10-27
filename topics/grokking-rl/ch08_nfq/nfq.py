"""Chapter 8: introduction to value-based deep RL. NFQ on CartPole.

Chapters 3-7 kept a note Q(s, a) for every square of a tiny grid. CartPole has no
squares: its state is four real numbers (cart position, cart speed, pole angle, pole
spin), so no table can hold a note for every state. Two ways out:

  * chop each number into a few bins and keep a table anyway (chapter 6's Q-learning
    on a 3 x 3 x 8 x 8 grid of bins): the honest baseline
  * NFQ (neural fitted Q iteration): replace the table with a small neural network that
    takes the four numbers and outputs one Q-value per action, and fit it like a
    regression problem on batches of experience

NFQ's loop (Riedmiller 2005), the skeleton every later deep value method keeps:

    play one game with the current network (epsilon-greedy), add its moves to the
    moves seen so far (the most recent 20,000)
    repeat 20 times, over that whole dataset:
        target = r + gamma * max_a' Q(s', a')      (from the network, recomputed)
        one gradient step pulling Q(s, a) towards the targets

    python -m ch08_nfq.nfq                 # from topics/grokking-rl; writes nfq.png here
"""

import argparse
import copy
import os
import time
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
ENV_ID = "CartPole-v1"
INPUTS = ["cart position", "cart speed", "pole angle", "pole spin"]
ACTIONS = ["push ←", "push →"]
torch.set_num_threads(1)  # tiny network: threads cost more than they save


class QNet(nn.Module):
    """4 numbers in, one Q-value per action out. Small on purpose, so every unit fits
    on screen in the report."""

    def __init__(self, hidden=(32, 32)):
        super().__init__()
        self.l1 = nn.Linear(4, hidden[0])
        self.l2 = nn.Linear(hidden[0], hidden[1])
        self.out = nn.Linear(hidden[1], 2)

    def forward(self, x, keep=False):
        h1 = torch.relu(self.l1(x))
        h2 = torch.relu(self.l2(h1))
        q = self.out(h2)
        return (q, h1, h2) if keep else q


def greedy(net, s):
    with torch.no_grad():
        return int(net(torch.as_tensor(s, dtype=torch.float32)).argmax())


def evaluate(policy, episodes=3, seed=10_000):
    """Mean steps balanced by a policy, on fixed seeds so checkpoints are comparable."""
    env = gym.make(ENV_ID)
    total = 0
    for i in range(episodes):
        s, _ = env.reset(seed=seed + i)
        done = False
        while not done:
            s, r, term, trunc, _ = env.step(policy(s))
            total += r
            done = term or trunc
    return total / episodes


def fit(net, opt, batch, gamma, epochs):
    """One round of fitted Q iteration on every move collected so far.

    The targets are computed ONCE, then the network is fit to them like fixed labels.
    Recomputing them every epoch (chasing a target that moves with each step) let the
    Q-values run away to ~1,800 when the honest ceiling is 1 / (1 - gamma) = 100.
    """
    S, A, R, S2, T = (torch.as_tensor(np.array(x), dtype=torch.float32) for x in zip(*batch))
    A = A.long()
    with torch.no_grad():
        target = R + gamma * net(S2).max(dim=1).values * (1 - T)
    for _ in range(epochs):
        q = net(S).gather(1, A[:, None]).squeeze(1)
        loss = ((q - target) ** 2).mean() / 2
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss.detach())


def train_nfq(episodes, seed=0, epochs=20, gamma=0.99, lr=1e-3, max_data=20_000,
              eval_every=10, snap_every=None):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    env = gym.make(ENV_ID)
    net = QNet()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    data, history, snaps, fits = [], [], [(0, copy.deepcopy(net.state_dict()))], 0
    best = (-1.0, 0, None)  # (score, episode, weights): NFQ wobbles, so keep its best moment
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.5 * episodes))  # explore a lot early, little late
        s, _ = env.reset(seed=seed * 100_000 + ep)
        done, steps = False, 0
        while not done:
            a = int(rng.integers(2)) if rng.random() < eps else greedy(net, s)
            s2, r, term, trunc, _ = env.step(a)
            # Truncation at 500 steps is a time limit, not a failure: keep bootstrapping.
            data.append((s, a, r, s2, float(term)))
            s, done, steps = s2, term or trunc, steps + 1
        # Refit after every game, like the original NFQ: early games are short and cheap,
        # so the network gets many updates exactly when it knows least.
        data = data[-max_data:]
        fit(net, opt, data, gamma, epochs)
        fits += 1
        if ep % eval_every == 0 or ep == episodes - 1:
            score = evaluate(lambda x: greedy(net, x))
            history.append((ep + 1, steps, score))
            if score > best[0]:
                best = (score, ep + 1, copy.deepcopy(net.state_dict()))
        if snap_every and (ep + 1) % snap_every == 0:
            snaps.append((ep + 1, copy.deepcopy(net.state_dict())))
    return net, history, snaps, fits, best


# ── the baseline: chapter 6's Q-learning on a table of bins ─────────────────────

BINS = [np.linspace(-2.4, 2.4, 4)[1:-1],      # cart position: 3 bins
        np.linspace(-2, 2, 4)[1:-1],          # cart speed: 3 bins
        np.linspace(-0.21, 0.21, 9)[1:-1],    # pole angle: 8 bins
        np.linspace(-2, 2, 9)[1:-1]]          # pole spin: 8 bins


def binned(s):
    return tuple(int(np.digitize(v, b)) for v, b in zip(s, BINS))


def train_table(episodes, seed=0, gamma=0.99, eval_every=10):
    rng = np.random.default_rng(seed)
    env = gym.make(ENV_ID)
    Q = np.zeros([len(b) + 1 for b in BINS] + [2])
    history = []
    for ep in range(episodes):
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        alpha = max(0.05, 0.5 * (1 - ep / episodes))
        s, _ = env.reset(seed=seed * 100_000 + ep)
        done, steps = False, 0
        while not done:
            b = binned(s)
            a = int(rng.integers(2)) if rng.random() < eps else int(np.argmax(Q[b]))
            s2, r, term, trunc, _ = env.step(a)
            b2 = binned(s2)
            Q[b + (a,)] += alpha * (r + gamma * Q[b2].max() * (not term) - Q[b + (a,)])
            s, done, steps = s2, term or trunc, steps + 1
        if ep % eval_every == 0 or ep == episodes - 1:
            history.append((ep + 1, steps, evaluate(lambda x: int(np.argmax(Q[binned(x)])))))
    return Q, history


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=300)
    ap.add_argument("--seed", type=int, default=2)
    args = ap.parse_args(argv)

    t = time.perf_counter()
    net, hist, snaps, fits, best = train_nfq(args.episodes, args.seed, snap_every=max(1, args.episodes // 20))
    t_nfq = time.perf_counter() - t
    t = time.perf_counter()
    Qt, hist_t = train_table(args.episodes, args.seed)
    t_tab = time.perf_counter() - t

    cells = int(np.prod(Qt.shape[:-1]))
    seen = int((np.abs(Qt).sum(axis=-1) > 0).sum())
    late = lambda h: float(np.mean([x[2] for x in h[len(h) * 2 // 3:]]))  # one 3-game check can get lucky
    best_nfq, best_tab = late(hist), late(hist_t)
    print(f"{ENV_ID}: balance for up to 500 steps (+1 per step)\n")
    print(f"NFQ        : {fits} refits, last-third average {best_nfq:.0f}, "
          f"final {hist[-1][2]:.0f}   ({t_nfq:.0f}s)")
    print(f"binned table: {cells} cells ({seen} ever visited), last-third average {best_tab:.0f}, "
          f"final {hist_t[-1][2]:.0f}   ({t_tab:.0f}s)")
    print()
    print(asciichart.line({"NFQ": [h[2] for h in hist], "binned table": [h[2] for h in hist_t]},
                          ymin=0, ymax=500, title="steps balanced by the greedy policy",
                          xmax=args.episodes, xlabel="episode"))

    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot([h[0] for h in hist], [h[2] for h in hist], label="NFQ (neural network)")
    ax.plot([h[0] for h in hist_t], [h[2] for h in hist_t], label="Q-learning on a binned table")
    ax.axhline(500, color="k", ls="--", lw=1, label="perfect (500 steps)")
    ax.set(xlabel="episode", ylabel="steps balanced (greedy policy, mean of 3)",
           title="Chapter 8: a network instead of a table, on CartPole")
    ax.legend()
    fig.tight_layout()
    path = OUT / "nfq.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"\nwrote {path}")
    return _report(net, hist, hist_t, snaps, best, path, cells, seen, args, best_nfq, best_tab)


# ── the report ──────────────────────────────────────────────────────────────────

def _load(state):
    net = QNet()
    net.load_state_dict(state)
    return net


def record(net, seed, max_steps=500, keep=True):
    """One greedy game; every state, and every layer's activations, for the player."""
    env = gym.make(ENV_ID)
    s, _ = env.reset(seed=seed)
    frames, done = [], False
    while not done and len(frames) < max_steps:
        with torch.no_grad():
            q, h1, h2 = net(torch.as_tensor(s, dtype=torch.float32), keep=True)
        a = int(q.argmax())
        frames.append({"s": [round(float(v), 4) for v in s], "a": a,
                       **({"h1": [round(float(v), 2) for v in h1], "h2": [round(float(v), 2) for v in h2],
                           "q": [round(float(v), 2) for v in q]} if keep else {})})
        s, r, term, trunc, _ = env.step(a)
        done = term or trunc
    return frames, len(frames)


def _report(net, hist, hist_t, snaps, best, path, cells, seen, args, late_nfq, late_tab):
    final_score = hist[-1][2]
    best_score, best_ep, best_state = best
    net = _load(best_state)  # show the network at its best; the text says so
    frames, n = record(net, seed=777)
    reel = []
    for ep, state in [snaps[0]] + [snaps[i] for i in (2, 5, 10)] + [snaps[-1]]:
        f, k = record(_load(state), seed=777, keep=False)
        label = "untrained network" if ep == 0 else f"after {ep} episodes"
        reel.append(deepviz.cartpole_player(f, f"{label}: balanced {k} steps", small=True))
    heat_frames, heat_caps = [], []
    for ep, state in snaps[:: max(1, len(snaps) // 10)] + [snaps[-1]]:
        heat_frames.append(deepviz.q_heatmap(_load(state), INPUTS))
        heat_caps.append(f"after {ep} episodes" if ep else "untrained: random weights")
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> when the world has too many states "
        "for a table, replace the table with a small neural network that <i>guesses</i> the "
        "notes for any state, and train it on batches of experience like any other "
        "regression problem.</p>",
        deepviz.cartpole_intro(),

        "<h3>1. Why the table stops working</h3>",
        viz.explain(
            how="CartPole's state is four real numbers, so there are no squares to keep notes "
                "for. The only way to use chapter 6's table is to chop every number into bins: "
                f"here 3 × 3 × 8 × 8 = <b>{cells} cells</b>. Every cell has to be visited (it "
                f"only ever saw {seen}) and learns nothing from its neighbours.",
            notice="A network sees that 'pole leaning 4° right' and 'pole leaning 5° right' "
                   "are nearly the same situation, so one experience improves its guesses for "
                   "all the states nearby. That is <b>generalization</b>, and it is the whole "
                   "point of deep RL."),
        viz.figure(path),
        viz.explain(notice=f"Over the last third of training: network {late_nfq:.0f} steps on "
                           f"average, table {late_tab:.0f}. The table occasionally lands a lucky "
                           "500, but it cannot hold onto it; it has no way to share what it learned "
                           "in one bin with the bin next door."),

        "<h3>2. Inside the network, while it plays</h3>",
        deepviz.network_player(frames, net, INPUTS, ACTIONS,
                               f"the network at its best (episode {best_ep}): balanced {n} steps"),
        viz.explain(
            how="Left: the game. Right: the network on this exact frame. The four inputs are "
                "what it sees. Each little square in the two middle columns is one hidden unit, "
                "brighter = more active. The two outputs are its Q-values: how many more "
                "steps (discounted) it expects to keep the pole up if it pushes ← or → now. "
                "The lines are the strongest weights (orange = positive, blue = negative).",
            notice="While the pole is well balanced the two Q-values are close (either push is "
                   "fine), and they split apart when it starts to fall: then one push is clearly "
                   "the rescue. With γ = 0.99, a Q of 100 would mean 'I expect never to fall' "
                   "(1 + 0.99 + 0.99² + ... = 100). This network says about 115-120: it is "
                   "<i>over-optimistic</i>, because taking the max of its own noisy guesses "
                   "always rounds up. Chapter 9's Double DQN is built to fix exactly that."),

        "<h3>3. The notes as a picture: what the network learned</h3>",
        viz.slider(heat_frames, heat_caps, "the network's preferred push, by pole angle and spin", 700),
        viz.explain(
            how="Each map holds the cart still in the middle and asks the network about every "
                "pole angle (across) and spin (up). Orange means it prefers pushing →, blue "
                "pushing ←. This is the table from earlier chapters, but smooth: every point "
                "has an answer, even ones it never visited.",
            notice="Training draws a diagonal boundary. Mostly the network watches the <b>spin</b>: "
                   "if the pole is falling right, push → to get the cart back under it. The "
                   "<b>angle</b> shifts the line: a pole already leaning right gets the → push a "
                   "little earlier. Nobody told it any physics."),

        "<h3>4. Watch it learn</h3>",
        viz.row(*reel),
        viz.explain(how="The same starting position each time; only the network's weights differ."),

        "<h3>5. The NFQ loop</h3>",
        "<pre>play one game (some moves random, to explore) and add its moves to everything seen so far\n"
        "repeat 20 times, over all of it:\n"
        "    target = reward + γ · max over next moves of Q(next state)   ← same target as Q-learning\n"
        "    nudge the network so Q(state, move) gets closer to target    ← one gradient step\n"
        "repeat</pre>",
        viz.explain(
            notice="This is chapter 6's Q-learning update turned into a regression problem: the "
                   "targets are the labels, and the network is the model being fit. Its weakness: "
                   "the targets come from the same network that is changing, which makes "
                   "training wobbly: in this run it scored a perfect "
                   f"{best_score:.0f} at episode {best_ep} and finished at {final_score:.0f}. "
                   "That is why the network shown above is a saved copy of its best moment. "
                   "Chapter 9's DQN fixes exactly this."),
        viz.takeaways([
            "Real problems have too many states for a table.",
            "A neural network can stand in for the table: state in, one Q-value per action out.",
            "It generalizes: one experience improves its guesses for similar states.",
            "NFQ fits the network to Q-learning targets on batches of experience.",
            f"Over the last third of training the network averaged {late_nfq:.0f} of 500 steps; "
            f"the binned table averaged {late_tab:.0f}. (Either can land a lucky single game, so "
            "compare averages, not best moments.)",
        ]),
    ]


if __name__ == "__main__":
    main()
