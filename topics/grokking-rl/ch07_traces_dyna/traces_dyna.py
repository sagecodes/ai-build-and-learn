"""Chapter 7: achieving goals more effectively and efficiently.

Two ways to squeeze more learning out of every step than chapter 6's one-step
Q-learning does:

  * eligibility traces - SARSA(lambda) and Watkins's Q(lambda) push each TD (temporal difference) error back
                         along the recent path, not just into the last (s, a)
  * learn a model      - Dyna-Q records what the world did, then replays imagined
                         transitions from that model between real steps; trajectory
                         sampling plans forward from where the agent is right now

Run on the 8x8 frozen lake, where the reward is far away and rare enough that
one-step Q-learning struggles.

    python -m ch07_traces_dyna.traces_dyna     # from topics/grokking-rl; writes traces_dyna_fl8.png here
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import asciichart
import envs
import viz
from ch03_planning.planning import policy_evaluation, to_arrays, value_iteration
from ch05_prediction.prediction import decay_schedule
from ch06_control.control import (eps_greedy, learning_reel, plans_row, q_learning,
                                  train_and_score)

# Figures land next to this file (the README embeds them); Flyte points it at scratch.
OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))


def sarsa_lambda(lam):
    def algo(env, Q, alphas, epsilons, gamma, on_episode):
        for e in range(len(alphas)):
            E = np.zeros_like(Q)
            s, done, trunc = env.reset(), False, False
            a = eps_greedy(Q, s, epsilons[e], env.rng)
            while not (done or trunc):
                s2, r, done, trunc = env.step(a)
                a2 = eps_greedy(Q, s2, epsilons[e], env.rng)
                delta = r + gamma * Q[s2, a2] * (not done) - Q[s, a]
                E[s] = 0  # replacing trace: clear the other actions in s,
                E[s, a] = 1  # then mark the one taken
                Q += alphas[e] * delta * E
                E *= gamma * lam
                s, a = s2, a2
            on_episode(e, Q)
    return algo


def watkins_q_lambda(lam):
    def algo(env, Q, alphas, epsilons, gamma, on_episode):
        for e in range(len(alphas)):
            E = np.zeros_like(Q)
            s, done, trunc = env.reset(), False, False
            a = eps_greedy(Q, s, epsilons[e], env.rng)
            while not (done or trunc):
                s2, r, done, trunc = env.step(a)
                a2 = eps_greedy(Q, s2, epsilons[e], env.rng)
                next_is_greedy = Q[s2, a2] == Q[s2].max()
                delta = r + gamma * Q[s2].max() * (not done) - Q[s, a]
                E[s] = 0
                E[s, a] = 1
                Q += alphas[e] * delta * E
                # An exploratory action breaks the "this is what greedy would do" chain,
                # so the trace can no longer credit earlier steps through it.
                E = E * gamma * lam if next_is_greedy else np.zeros_like(Q)
                s, a = s2, a2
            on_episode(e, Q)
    return algo


class Model:
    """Tabular world model: counts of (s, a) -> s', and the rewards seen there."""

    def __init__(self, nS, nA):
        self.counts = np.zeros((nS, nA, nS))
        self.rewards = np.zeros((nS, nA, nS))
        self.terminal = np.zeros(nS, dtype=bool)
        self.seen = []

    def update(self, s, a, r, s2, done):
        if not self.counts[s, a].any():
            self.seen.append((s, a))
        self.counts[s, a, s2] += 1
        self.rewards[s, a, s2] += r
        self.terminal[s2] |= done

    def sample(self, s, a, rng):
        c = self.counts[s, a]
        cum = c.cumsum()
        s2 = int(np.searchsorted(cum, rng.random() * cum[-1], side="right"))
        return self.rewards[s, a, s2] / c[s2], s2, self.terminal[s2]


def dyna_q(n_planning, trajectory_sampling=False):
    def algo(env, Q, alphas, epsilons, gamma, on_episode):
        model = Model(*Q.shape)
        for e in range(len(alphas)):
            alpha = alphas[e]
            s, done, trunc = env.reset(), False, False
            while not (done or trunc):
                a = eps_greedy(Q, s, epsilons[e], env.rng)
                s2, r, done, trunc = env.step(a)
                Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
                model.update(s, a, r, s2, done)

                if trajectory_sampling:
                    # Imagine a short greedy rollout forward from where we just landed.
                    ps = s2
                    for _ in range(n_planning):
                        if model.terminal[ps]:
                            break
                        pa = eps_greedy(Q, ps, 0.0, env.rng)  # random tie-break, not argmax
                        if not model.counts[ps, pa].any():
                            break
                        pr, ps2, pdone = model.sample(ps, pa, env.rng)
                        Q[ps, pa] += alpha * (pr + gamma * Q[ps2].max() * (not pdone) - Q[ps, pa])
                        ps = ps2
                else:
                    # Classic Dyna-Q: replay random (s, a) pairs we have ever tried.
                    for _ in range(n_planning):
                        ps, pa = model.seen[env.rng.integers(len(model.seen))]
                        pr, ps2, pdone = model.sample(ps, pa, env.rng)
                        Q[ps, pa] += alpha * (pr + gamma * Q[ps2].max() * (not pdone) - Q[ps, pa])
                s = s2
            on_episode(e, Q)
    return algo


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="fl8", choices=["sws", "fl4", "fl8"])
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--planning", type=int, default=5, help="imagined updates per real step")
    ap.add_argument("--every", type=int, default=100)
    args = ap.parse_args(argv)

    algos = {
        "Q-learning (ch 6 baseline)": q_learning,
        f"SARSA(lambda={args.lam})": sarsa_lambda(args.lam),
        f"Q(lambda={args.lam})": watkins_q_lambda(args.lam),
        f"Dyna-Q ({args.planning} plans/step)": dyna_q(args.planning),
        f"trajectory sampling ({args.planning})": dyna_q(args.planning, trajectory_sampling=True),
    }

    P, start, max_steps, layout = envs.ENVS[args.env]
    V_star, pi_star, _ = value_iteration(P, args.gamma)
    R, T = to_arrays(P)
    Q_star = R + args.gamma * T @ V_star
    # The ceiling for "P(reach goal)" is the gamma=1 optimum: a gamma<1 optimum trades a
    # little safety for speed, so a learner can legitimately beat it on this metric.
    best = value_iteration(P, 1.0)[0][start]
    print(f"{args.env}: the best any policy can do is reach the goal {best:.1%} of the time\n")

    curves, learned = {}, {}
    for name, algo in algos.items():
        runs = [
            train_and_score(algo, P, start, max_steps, args.episodes, args.gamma, seed,
                            Q_star, args.every)
            for seed in range(args.seeds)
        ]
        err = np.mean([r[1] for r in runs], axis=0)
        succ = np.mean([r[2] for r in runs], axis=0)
        curves[name] = (err, succ)
        learned[name] = (runs[0][0], runs[0][3])
        # First checkpoint where the greedy policy is within 10% of optimal.
        hit = np.flatnonzero(succ >= 0.9 * best)
        when = f"episode {hit[0] * args.every}" if len(hit) else "never"
        print(f"{name:>30}: final success {succ[-1]:6.1%}  |Q-Q*| {err[-1]:.4f}  "
              f">=90% of optimal at {when}")

    print()
    print(asciichart.line({k: v[1] for k, v in curves.items()}, ymin=0, ymax=best,
                          title=f"P(reach goal) of the greedy policy (top = best possible, {best:.1%})",
                          xmax=args.episodes, xlabel="episode"))

    OUT.mkdir(exist_ok=True)
    x = np.minimum(np.arange(len(err)) * args.every, args.episodes - 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, (err, succ) in curves.items():
        axes[0].plot(x, err, label=name)
        axes[1].plot(x, succ, label=name)
    axes[0].set(title="mean |Q - Q*| over live states", xlabel="episode", yscale="log")
    axes[1].axhline(best, color="k", ls="--", lw=1, label="best possible (gamma=1 optimum)")
    axes[1].set(title="P(reach goal) of the greedy policy (exact)", xlabel="episode")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=8)
    fig.suptitle(f"Chapter 7: traces and planning vs one-step Q-learning on {args.env} "
                 f"({args.seeds} seeds)")
    fig.tight_layout()
    path = OUT / f"traces_dyna_{args.env}.png"
    fig.savefig(path, dpi=120)
    print(f"\nwrote {path}")
    return _report(args, path, learned, curves, best)



# ── the report ──────────────────────────────────────────────────────────────────
#
# The two narrated games re-implement one game of SARSA(lambda) and of Dyna-Q with
# every number kept; same rules as the training functions above.

ARROWS = "←↓→↑"


def _pretrain(algo, name, k, episodes, gamma):
    P, start, max_steps, _ = envs.ENVS[name]
    alphas = decay_schedule(0.5, 0.01, 0.5, episodes)
    epsilons = decay_schedule(1.0, 0.1, 0.9, episodes)
    Q = np.zeros((len(P), len(P[0])))
    algo(envs.TabularEnv(P, start, max_steps, 0), Q, alphas[:k], epsilons[:k], gamma,
         lambda e, Q: None)
    return Q, float(alphas[k]), float(epsilons[k])


def _game_seed(name, Q, eps, min_steps=8):
    P, start, max_steps, _ = envs.ENVS[name]
    for seed in range(300):
        env = envs.TabularEnv(P, start, max_steps, seed)
        s, done, trunc, n = env.reset(), False, False, 0
        while not (done or trunc):
            s, r, done, trunc = env.step(eps_greedy(Q, s, eps, env.rng))
            n += 1
        if n >= min_steps and r > 0:
            return seed
    return 0


def narrate_sarsa_lambda(name, Q, alpha, eps, gamma, lam, max_frames=60):
    """One game of SARSA(lambda), traces glowing. The dice are chosen by dry-running the
    narration itself (the agent learns as it plays, so its moves depend on the updates):
    we want a game that ends at the goal, where one big surprise lights up the whole path."""
    for seed in range(300):
        frames, caps, won = _sarsa_lambda_game(name, Q, alpha, eps, gamma, lam, seed, max_frames)
        if won and len(frames) >= 9:
            break
    return viz.slider(frames, caps, f"one game of SARSA(λ={lam}), traces glowing: press Play", 1400)


def _sarsa_lambda_game(name, Q, alpha, eps, gamma, lam, seed, max_frames):
    P, start, max_steps, _ = envs.ENVS[name]
    layout = viz.layout_for(name)
    env = envs.TabularEnv(P, start, max_steps, seed)
    Q, E = Q.copy(), np.zeros_like(Q)
    s, done, trunc, r = env.reset(), False, False, 0.0
    a = eps_greedy(Q, s, eps, env.rng)
    frames = [viz.row(viz.grid(layout, Q.max(axis=1), Q.argmax(axis=1), agent=s, Q=Q, vmax=1.0),
                      viz._panel([f"{viz.AGENT} starts a game. No trace yet: nothing visited."]))]
    caps = ["press Play. The yellow glow is the memory trace: brighter = visited more recently."]
    while not (done or trunc) and len(frames) < max_frames:
        s2, r, done, trunc = env.step(a)
        a2 = eps_greedy(Q, s2, eps, env.rng)
        delta = r + gamma * Q[s2, a2] * (not done) - Q[s, a]
        E[s] = 0
        E[s, a] = 1
        before = Q.copy()
        Q += alpha * delta * E
        changed = sorted(((abs(Q[x, b] - before[x, b]), x, b) for x, b in zip(*np.nonzero(E))), reverse=True)
        lines = [f"{viz.AGENT} tried {ARROWS[a]} on {viz._where(name, s)}: {viz.outcome(name, s, a, s2)}"
                 + {"H": f", into a {viz.HOLE}", "G": f", onto the {viz.GOAL} (+1)"}.get(viz._cell(name, s2), "") + ".",
                 f"Surprise (TD error) δ = reward + γ·note(next move) − note(this move) = "
                 f"{r:g} + {gamma}×{before[s2, a2] * (not done):.2f} − {before[s, a]:.2f} = <b>{delta:+.3f}</b>",
                 f"<b>Every move still glowing</b> gets α × δ × its trace (α = {alpha:.2f}). "
                 "Biggest changes:"]
        lines += [f"&nbsp;&nbsp;{viz._where(name, x)} {ARROWS[b]}: trace {E[x, b]:.2f}, note "
                  f"{before[x, b]:.3f} → {Q[x, b]:.3f}" for _, x, b in changed[:6]]
        if len(changed) > 6:
            lines.append(f"&nbsp;&nbsp;... and {len(changed) - 6} more")
        lines.append(f"Then every trace fades by γλ = {gamma * lam:.2f}.")
        glow = E.max(axis=1).copy()
        E *= gamma * lam
        frames.append(viz.row(viz.grid(layout, Q.max(axis=1), Q.argmax(axis=1), agent=s2, Q=Q,
                                       vmax=1.0, glow=glow), viz._panel(lines)))
        caps.append(f"step {len(frames) - 1}: δ = {delta:+.3f}, shared by {len(changed)} remembered "
                    f"move{'s' if len(changed) != 1 else ''}"
                    + (" <b>: the win lights up the whole path</b>" if r > 0 else ""))
        s, a = s2, a2
    return frames, caps, r > 0


def narrate_dyna(name, Q, alpha, eps, gamma, n_planning, pretrain_episodes, max_frames=40):
    """Dyna-Q: one real step, then n imagined ones from its learned model, all shown."""
    P, start, max_steps, _ = envs.ENVS[name]
    layout = viz.layout_for(name)
    # Rebuild the model from the same warm-up games the notes came from.
    model = Model(*Q.shape)
    Q = Q.copy()
    _dyna_warmup(name, Q, model, alpha, eps, gamma, n_planning, pretrain_episodes)
    env = envs.TabularEnv(P, start, max_steps, _game_seed(name, Q, eps))
    dream = np.random.default_rng(1234)  # imagination gets its own dice, so the real game
    s, done, trunc = env.reset(), False, False  # plays out the way _game_seed found it
    frames = [viz.row(viz.grid(layout, Q.max(axis=1), Q.argmax(axis=1), agent=s, Q=Q, vmax=1.0),
                      viz._panel([f"{viz.AGENT} has played {pretrain_episodes} games and kept a "
                                  f"<b>model</b>: for every move it tried, a tally of where it landed."]))]
    caps = ["press Play: after every real step it imagines more."]
    while not (done or trunc) and len(frames) < max_frames:
        a = eps_greedy(Q, s, eps, env.rng)
        s2, r, done, trunc = env.step(a)
        old = Q[s, a]
        Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
        model.update(s, a, r, s2, done)
        lines = [f"<b>Real step:</b> tried {ARROWS[a]} on {viz._where(name, s)}, "
                 f"{viz.outcome(name, s, a, s2)}. Note {old:.2f} → {Q[s, a]:.2f}. Tally updated.",
                 f"<b>Then it imagines {n_planning} moves</b> it has made before, using its tallies "
                 f"instead of the real ice:"]
        for _ in range(n_planning):
            ps, pa = model.seen[dream.integers(len(model.seen))]
            pr, ps2, pdone = model.sample(ps, pa, dream)
            c = model.counts[ps, pa]
            pold = Q[ps, pa]
            Q[ps, pa] += alpha * (pr + gamma * Q[ps2].max() * (not pdone) - Q[ps, pa])
            land = {"H": viz.HOLE, "G": viz.GOAL}.get(viz._cell(name, ps2), viz._where(name, ps2))
            lines.append(f"&nbsp;&nbsp;💭 on {viz._where(name, ps)}, tries {ARROWS[pa]}: lands on "
                         f"{land} (its tally: {int(c[ps2])} of {int(c.sum())} past tries). "
                         f"Note {pold:.2f} → {Q[ps, pa]:.2f}")
        frames.append(viz.row(viz.grid(layout, Q.max(axis=1), Q.argmax(axis=1), agent=s2, Q=Q,
                                       vmax=1.0), viz._panel(lines)))
        caps.append(f"step {len(frames) - 1}: 1 real update + {n_planning} imagined ones")
        s = s2
    return viz.slider(frames, caps, f"one game of Dyna-Q: real steps and imagined ones", 1600)


def _dyna_warmup(name, Q, model, alpha, eps, gamma, n_planning, episodes):
    P, start, max_steps, _ = envs.ENVS[name]
    env = envs.TabularEnv(P, start, max_steps, 0)
    for _ in range(episodes):
        s, done, trunc = env.reset(), False, False
        while not (done or trunc):
            a = eps_greedy(Q, s, eps, env.rng)
            s2, r, done, trunc = env.step(a)
            Q[s, a] += alpha * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
            model.update(s, a, r, s2, done)
            for _ in range(n_planning):
                ps, pa = model.seen[env.rng.integers(len(model.seen))]
                pr, ps2, pdone = model.sample(ps, pa, env.rng)
                Q[ps, pa] += alpha * (pr + gamma * Q[ps2].max() * (not pdone) - Q[ps, pa])
            s = s2


def _report(args, path, learned, curves, best):
    small = "fl4"
    k = 200
    Qs, a_s, e_s = _pretrain(sarsa_lambda(args.lam), small, k, 10_000, args.gamma)
    Qd = np.zeros((16, 4))
    reels = []
    for n, (_, snaps) in learned.items():
        reels += [f"<div style='margin-top:10px'><b>{n}</b></div>",
                  learning_reel(args.env, snaps, picks=(100, 1000, 3000, None))]
    rows = "".join(f"<tr><td>{n}</td><td>{succ[-1]:.0%}</td><td>{err[-1]:.3f}</td></tr>"
                   for n, (err, succ) in curves.items())
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> one-step Q-learning learns slowly "
        "because each game only updates the last move; remembering the whole recent path, or "
        "practising in your imagination, squeezes far more learning out of every game.</p>",
        viz.intro(args.env, "This chapter's race runs on the <b>8x8 lake</b>: the goal is far "
                  "away, so the first win takes a long time and one-step learning crawls. The "
                  "step-by-step games use the small 4x4 lake so every number fits on screen."),
        viz.legend(pct=False),

        "<h3>1. Memory traces: credit the whole path, not just the last step</h3>",
        narrate_sarsa_lambda(small, Qs, a_s, e_s, args.gamma, args.lam),
        viz.explain(
            how=f"A game of SARSA(λ) after {k} warm-up games. Every move it makes leaves a "
                "<b>trace</b> (the yellow glow) that fades by γλ each step. When a step brings a "
                "surprise δ (better or worse than its note expected), every move still glowing "
                "gets a share of it, in proportion to its trace.",
            notice="Ordinary Q-learning would only update the single move just made. Here one "
                   "surprise near the goal reaches back along the whole path at once. λ (lambda) "
                   "sets how long the memory lasts: 0 is one-step learning, 1 remembers the whole "
                   "game (like Monte Carlo). Q(λ) is the same idea, but it wipes the trace after a "
                   "random exploring move, because that move was not part of its best plan."),

        "<h3>2. Imagination: Dyna-Q practises with a model it learned itself</h3>",
        narrate_dyna(small, Qd, 0.5, 0.3, args.gamma, args.planning, 300),
        viz.explain(
            how="Dyna-Q keeps a <b>tally</b> of where every move it tried actually landed: that "
                "is a learned model of the ice, built from experience (chapter 3 was given one). "
                f"After each real step it replays {args.planning} past moves in its head (💭), "
                "drawing the landing square from its tallies, and updates its notes as if they had "
                "really happened.",
            notice="The tallies converge on the real 1-in-3 slips without anyone telling it. Every "
                   f"real step becomes {args.planning + 1} updates, so the notes become accurate "
                   "with far fewer real games. (Trajectory sampling is the same, but it imagines "
                   "forward from where it is now, along its own plan.)"),

        "<h3>3. The race on the 8x8 lake: watch each method learn</h3>",
        viz.explain(how="One row per method: its plan after 100, 1,000 and 3,000 games and at "
                        "the end, replayed with the same dice. The titles give each plan's real "
                        "chance of winning."),
        *reels,
        "<h3>4. Where they ended up</h3>",
        plans_row(args.env, learned, args.gamma),
        viz.figure(path),
        f"<table class='dc'><tr><th>method</th><th>final plan wins</th><th>gap to the true notes</th></tr>{rows}</table>",
        viz.explain(notice="Dyna-Q's notes end up closest to the truth (its imagination gives it "
                           "many more updates per real game). Q(λ) and trajectory sampling get a "
                           "good plan soonest. SARSA(λ) learns a cautious plan that happens to "
                           "suit a slippery lake well."),

        viz.takeaways([
            "One-step learning wastes experience: only the last move learns.",
            "Eligibility traces spread each surprise back along the recent path (λ sets how far).",
            "A learned model lets the agent practise in its head: more updates per real game.",
            "Planning (chapter 3) and learning (chapters 5-6) combine: learn the model, then plan with it.",
            "Next: the tables get too big, so the notes become a neural network (deep RL, chapter 8).",
        ]),
    ]

if __name__ == "__main__":
    main()
