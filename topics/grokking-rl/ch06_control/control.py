"""Chapter 6: improving agents' behaviours.

Chapter 5 estimated the value of a fixed policy. Control closes the loop: estimate
Q(s, a), act (mostly) greedily on it, repeat. The four classics differ only in the
target they move Q(s, a) towards:

MC = Monte Carlo. SARSA = State, Action, Reward, next State, next Action: the
five things its update uses.

  * MC control   - the full sampled return, after the episode ends
  * SARSA        - r + gamma * Q(s', a'), a' being the action actually taken next (on-policy)
  * Q-learning   - r + gamma * max_a' Q(s', a'), whatever the agent does next (off-policy)
  * Double Q     - two tables; one picks argmax a', the other scores it (fixes max bias)

Every learner is scored against chapter 3's value iteration, which knows the answer.

    python -m ch06_control.control                                   # frozen lake 4x4
    python -m ch06_control.control --env sws --episodes 3000         # slippery walk seven
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

# Figures land next to this file (the README embeds them); Flyte points it at scratch.
OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))


def eps_greedy(Q, s, eps, rng):
    if rng.random() < eps:
        return int(rng.integers(Q.shape[1]))
    q = Q[s]
    return int(rng.choice(np.flatnonzero(q == q.max())))


def mc_control(env, Q, alphas, epsilons, gamma, on_episode):
    for e in range(len(alphas)):
        s, done, trunc, traj = env.reset(), False, False, []
        while not (done or trunc):
            a = eps_greedy(Q, s, epsilons[e], env.rng)
            s2, r, done, trunc = env.step(a)
            traj.append((s, a, r))
            s = s2
        G, seen, updates = 0.0, set(), []
        for s, a, r in reversed(traj):
            G = r + gamma * G
            updates.append((s, a, G))
        for s, a, G in reversed(updates):  # first visit = earliest occurrence
            if (s, a) not in seen:
                seen.add((s, a))
                Q[s, a] += alphas[e] * (G - Q[s, a])
        on_episode(e, Q)


def sarsa(env, Q, alphas, epsilons, gamma, on_episode):
    for e in range(len(alphas)):
        s, done, trunc = env.reset(), False, False
        a = eps_greedy(Q, s, epsilons[e], env.rng)
        while not (done or trunc):
            s2, r, done, trunc = env.step(a)
            a2 = eps_greedy(Q, s2, epsilons[e], env.rng)
            Q[s, a] += alphas[e] * (r + gamma * Q[s2, a2] * (not done) - Q[s, a])
            s, a = s2, a2
        on_episode(e, Q)


def q_learning(env, Q, alphas, epsilons, gamma, on_episode):
    for e in range(len(alphas)):
        s, done, trunc = env.reset(), False, False
        while not (done or trunc):
            a = eps_greedy(Q, s, epsilons[e], env.rng)
            s2, r, done, trunc = env.step(a)
            Q[s, a] += alphas[e] * (r + gamma * Q[s2].max() * (not done) - Q[s, a])
            s = s2
        on_episode(e, Q)


def double_q_learning(env, Q, alphas, epsilons, gamma, on_episode):
    Q1, Q2 = Q.copy(), Q.copy()
    for e in range(len(alphas)):
        s, done, trunc = env.reset(), False, False
        while not (done or trunc):
            Q[:] = (Q1 + Q2) / 2
            a = eps_greedy(Q, s, epsilons[e], env.rng)
            s2, r, done, trunc = env.step(a)
            A, B = (Q1, Q2) if env.rng.random() < 0.5 else (Q2, Q1)
            a_star = int(np.argmax(A[s2]))
            A[s, a] += alphas[e] * (r + gamma * B[s2, a_star] * (not done) - A[s, a])
            s = s2
        Q[:] = (Q1 + Q2) / 2
        on_episode(e, Q)


ALGOS = {
    "MC control": mc_control,
    "SARSA": sarsa,
    "Q-learning": q_learning,
    "Double Q-learning": double_q_learning,
}


def train_and_score(algo, P, start, max_steps, episodes, gamma, seed, Q_star, every):
    """Run one learner; every `every` episodes, record |Q - Q*| and the exact
    success probability of its current greedy policy."""
    env = envs.TabularEnv(P, start, max_steps, seed)
    arrays = to_arrays(P)
    live = np.array([not all(t[3] and t[1] == s for t in P[s][0]) for s in range(len(P))])
    alphas = decay_schedule(0.5, 0.01, 0.5, episodes)
    epsilons = decay_schedule(1.0, 0.1, 0.9, episodes)
    q_err, success, snaps = [], [], []

    def on_episode(e, Q):
        if e % every == 0 or e == episodes - 1:
            q_err.append(np.abs(Q[live] - Q_star[live]).mean())
            pi = Q.argmax(axis=1)
            V = policy_evaluation(pi, P, 1.0, theta=1e-7, max_sweeps=20_000, arrays=arrays)
            success.append(V[start])
            snaps.append((e, Q.copy()))

    Q = np.zeros((len(P), len(P[0])))
    algo(env, Q, alphas, epsilons, gamma, on_episode)
    return Q, np.array(q_err), np.array(success), snaps


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="fl4", choices=["swf", "sws", "fl4", "fl8"])
    ap.add_argument("--episodes", type=int, default=10_000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--every", type=int, default=25)
    args = ap.parse_args(argv)

    P, start, max_steps, layout = envs.ENVS[args.env]
    V_star, pi_star, _ = value_iteration(P, args.gamma)
    Q_star = to_arrays(P)[0] + args.gamma * to_arrays(P)[1] @ V_star
    # The ceiling for "P(reach goal)" is the gamma=1 optimum: a gamma<1 optimum trades a
    # little safety for speed, so a learner can legitimately beat it on this metric.
    best = value_iteration(P, 1.0)[0][start]
    print(f"{args.env}: the best any policy can do is reach the goal {best:.1%} of the time")
    print(envs.show(V_star, layout, pi_star))

    curves, learned = {}, {}
    for name, algo in ALGOS.items():
        runs = [
            train_and_score(algo, P, start, max_steps, args.episodes, args.gamma, seed,
                            Q_star, args.every)
            for seed in range(args.seeds)
        ]
        curves[name] = (np.mean([r[1] for r in runs], axis=0),
                        np.mean([r[2] for r in runs], axis=0))
        Q = runs[0][0]
        learned[name] = (Q, runs[0][3])
        print(f"\n{name}: final |Q - Q*| {curves[name][0][-1]:.4f}, "
              f"greedy success {curves[name][1][-1]:.1%} (mean of {args.seeds} seeds)")
        print(envs.show(Q.max(axis=1), layout, Q.argmax(axis=1)))

    print("\nMC control and SARSA are on-policy: with epsilon floored at 0.1 they learn what the\n"
          "*exploring* policy is worth, so their Q sits below Q*. Q-learning targets the greedy\n"
          "policy directly, which is why its error goes much lower on the same experience.")

    print()
    print(asciichart.line({k: v[1] for k, v in curves.items()}, ymin=0, ymax=best,
                          title=f"P(reach goal) of the greedy policy (top = best possible, {best:.1%})",
                          xmax=args.episodes, xlabel="episode"))

    OUT.mkdir(exist_ok=True)
    x = np.minimum(np.arange(len(next(iter(curves.values()))[0])) * args.every,
                   args.episodes - 1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for name, (err, succ) in curves.items():
        axes[0].plot(x, err, label=name)
        axes[1].plot(x, succ, label=name)
    axes[0].set(title="mean |Q - Q*| over live states", xlabel="episode", yscale="log")
    axes[1].axhline(best, color="k", ls="--", lw=1, label="best possible (gamma=1 optimum)")
    axes[1].set(title="P(reach goal) of the greedy policy (exact)", xlabel="episode")
    axes[0].legend()
    axes[1].legend()
    fig.suptitle(f"Chapter 6: model-free control on {args.env} "
                 f"(gamma={args.gamma}, {args.seeds} seeds)")
    fig.tight_layout()
    path = OUT / f"control_{args.env}.png"
    fig.savefig(path, dpi=120)
    print(f"\nwrote {path}")
    return _report(args, path, learned, curves, best)


# ── the report ──────────────────────────────────────────────────────────────────
#
# Shared with chapter 7: plans_row() and learning_reel(). The step-by-step explainer
# re-implements the Q-learning update for one narrated game; it is the same rule as
# q_learning() above, kept separate so the training loops stay uncluttered.

def plans_row(name, learned, gamma):
    """Each learner's final greedy plan, scored exactly, next to the true best plan."""
    P, start, _, _ = envs.ENVS[name]
    layout = viz.layout_for(name)
    boards = []
    for n, (Q, _) in learned.items():
        pi = Q.argmax(axis=1)
        V = policy_evaluation(pi, P, 1.0, theta=1e-8, max_sweeps=20_000)
        boards.append(viz.grid(layout, V, pi, Q=Q, pct=True, vmax=1.0,
                               caption=f"<b>{n}</b>: its plan reaches the {viz.GOAL} {V[start]:.0%} of the time"))
    V1, _, _ = value_iteration(P, 1.0)
    from ch03_planning.planning import q_from_v
    Q1 = q_from_v(V1, P, 1.0)
    boards.append(viz.grid(layout, V1, Q1.argmax(axis=1), Q=Q1, pct=True, vmax=1.0,
                           caption=f"<b>the true best plan</b> (chapter 3, from the rulebook): {V1[start]:.0%}"))
    return viz.row(*boards)


def learning_reel(name, snaps, picks=(0, 100, 500, 2000, None)):
    """The greedy plan at a few points in training, all replayed with the same dice."""
    chosen = []
    for e in picks:
        k = len(snaps) - 1 if e is None else min(range(len(snaps)), key=lambda i: abs(snaps[i][0] - e))
        ep, Q = snaps[k]
        games = f"{ep + 1:,} game{'s' if ep else ''}"
        chosen.append((f"after {games}" if e is not None else f"end of training ({games})", Q))
    # Same dice for every board; pick dice the final plan wins with, so the reel shows the
    # path from losing to winning. Each title still gives the plan's true odds.
    final = chosen[-1][1].argmax(axis=1)
    seed = viz.find_seed(name, lambda s, rng: final[s], True)
    return viz.reel(name, chosen, seed=seed)


def _narratable_seed(name, Q, eps, min_steps=8):
    """Dice for a game long enough to be worth narrating (the agent explores with eps)."""
    P, start, max_steps, _ = envs.ENVS[name]
    for seed in range(300):
        env = envs.TabularEnv(P, start, max_steps, seed)
        s, done, trunc, n = env.reset(), False, False, 0
        while not (done or trunc):
            explore = env.rng.random() < eps
            a = int(env.rng.integers(Q.shape[1])) if explore else int(env.rng.choice(viz.tied(Q, s)))
            s, r, done, trunc = env.step(a)
            n += 1
        if n >= min_steps and r > 0:
            return seed
    return 0


def narrate_q_learning(name, Q, alpha, eps, gamma, seed=None, max_frames=60):
    """One game of Q-learning, every update written out, the board updating as it learns."""
    P, start, max_steps, _ = envs.ENVS[name]
    if seed is None:
        seed = _narratable_seed(name, Q, eps)
    layout = viz.layout_for(name)
    arrows = "←↓→↑" if len(layout) > 1 else "←→"
    env = envs.TabularEnv(P, start, max_steps, seed)
    Q = Q.copy()
    s, done, trunc = env.reset(), False, False
    frames = [viz.row(viz.grid(layout, Q.max(axis=1), Q.argmax(axis=1), agent=s, Q=Q, vmax=1.0),
                      viz._panel([f"{viz.AGENT} starts a new game with what it has learned so far."]))]
    caps = ["press Play. The numbers on the board are the agent's notes: its best move's value on each square."]
    trail = [s]
    while not (done or trunc) and len(frames) < max_frames:
        explore = env.rng.random() < eps
        a = int(env.rng.integers(len(arrows))) if explore else int(env.rng.choice(viz.tied(Q, s)))
        notes = " &nbsp; ".join(f"{arrows[b]} {Q[s, b]:.2f}" for b in range(Q.shape[1]))
        s2, r, done, trunc = env.step(a)
        best_next = 0.0 if done else Q[s2].max()
        target = r + gamma * best_next
        old = Q[s, a]
        Q[s, a] += alpha * (target - Q[s, a])
        moved = viz.outcome(name, s, a, s2)
        land = {"H": f"into a {viz.HOLE} (game over, 0)", "G": f"onto the {viz.GOAL} (+1!)"}.get(
            viz._cell(name, s2), f"on {viz._where(name, s2)}")
        panel = viz._panel([
            f"{viz.AGENT} on <b>{viz._where(name, s)}</b>. Its notes for this square: {notes}",
            (f"Coin says <b>explore</b> (ε = {eps:.2f}): random move {arrows[a]}." if explore else
             f"Coin says <b>use what it knows</b>: best noted move {arrows[a]}."),
            f"Result: {moved}, landing {land}.",
            f"<b>Q-learning update</b> for ({viz._where(name, s)}, {arrows[a]}):",
            f"&nbsp;&nbsp;target = reward + γ × best note at the new square",
            f"&nbsp;&nbsp;&nbsp;&nbsp;= {r:g} + {gamma} × {best_next:.2f} = <b>{target:.2f}</b>",
            f"&nbsp;&nbsp;error = target − old note = {target:.2f} − {old:.2f} = {target - old:+.2f}",
            f"&nbsp;&nbsp;new note = old + α × error = {old:.2f} + {alpha:.2f} × {target - old:+.2f} "
            f"= <b>{Q[s, a]:.2f}</b>",
        ])
        trail.append(s2)
        frames.append(viz.row(viz.grid(layout, Q.max(axis=1), Q.argmax(axis=1), agent=s2,
                                       trail=trail[:-1], Q=Q, vmax=1.0), panel))
        caps.append(f"step {len(frames) - 1}: tried {arrows[a]}, {moved}; note for that move "
                    f"{old:.2f} → <b>{Q[s, a]:.2f}</b>")
        s = s2
    return viz.slider(frames, caps, "one game of Q-learning, every update shown: press Play", 1300)


def _report(args, path, learned, curves, best):
    name = args.env
    P, start, max_steps, _ = envs.ENVS[name]
    # The narrated game: train plain Q-learning for a while, then pause and narrate one more.
    k = min(300, args.episodes - 1)
    alphas = decay_schedule(0.5, 0.01, 0.5, args.episodes)
    epsilons = decay_schedule(1.0, 0.1, 0.9, args.episodes)
    Q = np.zeros((len(P), len(P[0])))
    q_learning(envs.TabularEnv(P, start, max_steps, 0), Q, alphas[:k], epsilons[:k], args.gamma,
               lambda e, Q: None)
    rows = "".join(f"<tr><td>{n}</td><td>{succ[-1]:.0%}</td><td>{err[-1]:.3f}</td></tr>"
                   for n, (err, succ) in curves.items())
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> with no rulebook at all, the agent "
        "plays thousands of games, keeps a note of how good every move looked on every square, "
        "and slowly turns those notes into a plan.</p>",
        viz.intro(name, "Unlike chapter 3, <b>nobody tells the agent the odds</b>. It starts "
                  "knowing nothing, and all it ever sees is where it landed and what it scored."),
        viz.legend(pct=False),

        "<h3>1. How it learns: one game, every update shown</h3>",
        narrate_q_learning(name, Q, float(alphas[k]), float(epsilons[k]), args.gamma),
        viz.explain(
            how=f"This is game {k + 1} of Q-learning's training. The panel shows the agent's "
                "<b>notes</b> (its Q-values: how good it thinks each move is on this square), the "
                "coin flip that decides whether it explores, what the ice did, and the update. "
                "The board shows, for each square, its best note and that move's arrow; you can "
                "watch the number on the square it just left change.",
            notice="The target borrows from the agent's notes on the <i>next</i> square, the "
                   "same bootstrapping as chapter 5's TD. Good news from the goal leaks backwards "
                   "one move at a time, game after game. γ = 0.99 makes each step back worth a "
                   "little less, so the notes read roughly as \"chance of winning, a bit less for "
                   "every step it takes\"."),

        "<h3>2. Watch it learn: the same dice, at different points in training</h3>",
        learning_reel(name, learned["Q-learning"][1]),
        viz.explain(
            how="Each board replays the agent's plan (always its best-noted move) as it was after "
                "that many games. Every replay uses exactly the same dice, so the only thing that "
                "differs is the plan. The title gives the plan's real chance of winning, worked "
                "out exactly, because one lucky game says little.",
            notice="Early plans point anywhere and fall in fast. Later ones discover the "
                   "chapter 3 trick on their own, from nothing but experience: point at walls."),

        "<h3>3. Four learners, four plans</h3>",
        plans_row(name, learned, args.gamma),
        viz.explain(
            how="Each learner's final plan. The number on a square is now the plan's real chance "
                "of reaching the goal from there; hover a square to see the learner's own notes.",
            notice=f"Only Q-learning matches the true best plan ({best:.0%}). See the next section "
                   "for why."),

        "<h3>4. Why they differ: what each one learns from</h3>",
        "<table class='dc'><tr><th>learner</th><th>moves its note towards...</th><th>so it learns the value of...</th></tr>"
        "<tr><td>Monte Carlo control</td><td>the real final score of the game</td><td>the plan <i>including</i> its random exploring moves</td></tr>"
        "<tr><td>SARSA</td><td>reward + note for the move it <i>actually</i> makes next</td><td>the plan including exploring (on-policy)</td></tr>"
        "<tr><td>Q-learning</td><td>reward + <b>best</b> note at the next square, whatever it does</td><td>the best plan, even while exploring (off-policy)</td></tr>"
        "<tr><td>Double Q-learning</td><td>like Q-learning, but one set of notes picks the best move and a second set scores it</td><td>the best plan, without Q-learning's habit of being over-optimistic</td></tr></table>",
        viz.explain(notice="The agent never stops exploring completely (1 move in 10 stays random). "
                           "On a lake full of holes those random moves are dangerous, so learners "
                           "that judge the plan <i>with</i> them (MC, SARSA) end up valuing squares "
                           "lower and playing more cautiously than the true best plan."),
        viz.figure(path),
        f"<table class='dc'><tr><th>learner</th><th>final plan wins</th><th>gap to the true notes (error)</th></tr>{rows}</table>",
        viz.explain(how="Averages over several training runs. Left chart: how far each learner's "
                        "notes are from the true values. Right: its plan's real chance of winning "
                        "as training goes on."),

        viz.takeaways([
            "No rulebook needed: learn a note Q(square, move) for every move from experience.",
            "Each update moves a note towards reward + the value of where you landed (bootstrapping).",
            "Exploring is necessary, and costly on dangerous ground.",
            "On-policy learners (SARSA, MC) learn the value of how they actually play; "
            "off-policy Q-learning learns the best plan while still exploring.",
            "Next chapter: learn faster from the same experience, with memory traces and imagination.",
        ]),
    ]

if __name__ == "__main__":
    main()
