"""Chapter 5: evaluating agents' behaviours.

Same question as chapter 3's policy evaluation (what is this policy worth?), but
now without the transition table: the agent only gets sampled episodes.

The random walk is the classic testbed: 5 non-terminal states, a fair coin moves
you left or right, +1 for leaving on the right. The true values are known in closed
form (1/6, 2/6, ... 5/6), so we can watch each estimator converge to them:

MC = Monte Carlo, TD = temporal difference.

  * first-visit MC / every-visit MC - wait for the episode to end, use the real return
  * TD(0)                           - bootstrap on the next state's estimate after one step
  * n-step TD                       - real rewards for n steps, then bootstrap
  * TD(lambda)                      - a decaying blend of every n, via eligibility traces

    python -m ch05_prediction.prediction   # from topics/grokking-rl; writes prediction.png here
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
from ch03_planning.planning import policy_evaluation

# Figures land next to this file (the README embeds them); Flyte points it at scratch.
OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))


def decay_schedule(start, end, decay_frac, n):
    """Exponential decay from start to end over the first decay_frac of n, then flat."""
    k = max(1, int(decay_frac * n))
    sched = np.full(n, end)
    sched[:k] = start * (end / start) ** (np.arange(k) / max(k - 1, 1))
    return sched


def generate_episode(env, policy):
    s, done, trunc, traj = env.reset(), False, False, []
    while not (done or trunc):
        s2, r, done, trunc = env.step(policy(s))
        traj.append((s, r, s2, done))
        s = s2
    return traj


def mc_update(V, traj, alpha, gamma, first_visit):
    G, returns = 0.0, []
    for s, r, _, _ in reversed(traj):
        G = r + gamma * G
        returns.append((s, G))
    returns.reverse()
    seen = set()
    for s, G in returns:
        if first_visit and s in seen:
            continue
        seen.add(s)
        V[s] += alpha * (G - V[s])


def td_update(V, traj, alpha, gamma):
    for s, r, s2, done in traj:
        V[s] += alpha * (r + gamma * V[s2] * (not done) - V[s])


def ntd_update(V, traj, alpha, gamma, n):
    # Applying the updates in time order after the episode gives the same result as
    # doing them online here, because the policy (a coin flip) never reads V.
    T = len(traj)
    for tau in range(T):
        end = min(tau + n, T)
        G = sum(gamma ** (k - tau) * traj[k][1] for k in range(tau, end))
        if end < T:
            G += gamma ** n * V[traj[end][0]]
        elif not traj[-1][3]:  # truncated, not terminated: still bootstrap
            G += gamma ** (end - tau) * V[traj[-1][2]]
        V[traj[tau][0]] += alpha * (G - V[traj[tau][0]])


def td_lambda_update(V, traj, alpha, gamma, lam):
    E = np.zeros_like(V)
    for s, r, s2, done in traj:
        delta = r + gamma * V[s2] * (not done) - V[s]
        E[s] += 1  # accumulating trace
        V += alpha * delta * E
        E *= gamma * lam


METHODS = {
    "first-visit MC": lambda V, tr, a, g: mc_update(V, tr, a, g, first_visit=True),
    "every-visit MC": lambda V, tr, a, g: mc_update(V, tr, a, g, first_visit=False),
    "TD(0)": td_update,
    "3-step TD": lambda V, tr, a, g: ntd_update(V, tr, a, g, n=3),
    "TD(lambda=0.5)": lambda V, tr, a, g: td_lambda_update(V, tr, a, g, lam=0.5),
}


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args(argv)

    P, start, max_steps, _ = envs.ENVS["rw"]
    nS = len(P)
    inner = slice(1, nS - 1)
    true_V = policy_evaluation(np.zeros(nS, dtype=int), P, args.gamma)
    print("true values:", envs.show(true_V))

    alphas = decay_schedule(0.5, 0.01, 0.5, args.episodes)
    rms = {m: np.zeros((args.seeds, args.episodes)) for m in METHODS}
    trace = {m: np.zeros(args.episodes) for m in METHODS}  # seed 0's estimate of V(s5)
    snaps = {m: [] for m in METHODS}  # seed 0's whole V after each episode, for the report

    for seed in range(args.seeds):
        env = envs.TabularEnv(P, start, max_steps, seed)
        episodes = [generate_episode(env, lambda s: 0) for _ in range(args.episodes)]
        for m, update in METHODS.items():
            V = np.full(nS, 0.5)
            V[0] = V[-1] = 0.0
            for e, traj in enumerate(episodes):
                update(V, traj, alphas[e], args.gamma)
                rms[m][seed, e] = np.sqrt(np.mean((V[inner] - true_V[inner]) ** 2))
                if seed == 0:
                    trace[m][e] = V[nS - 2]
                    snaps[m].append(V.copy())

    print(f"\nRMS error vs truth, averaged over {args.seeds} seeds (same episodes for every method):")
    checkpoints = [9, 49, 99, args.episodes - 1]
    print(f"{'':>16}" + "".join(f"  ep {c + 1:>4}" for c in checkpoints))
    for m in METHODS:
        mean = rms[m].mean(axis=0)
        print(f"{m:>16}" + "".join(f"  {mean[c]:7.4f}" for c in checkpoints))

    print()
    print(asciichart.line({m: rms[m].mean(axis=0) for m in METHODS}, logy=True,
                          title="RMS error vs episode (log scale)",
                          xmax=args.episodes, xlabel="episode"))

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for m in METHODS:
        axes[0].plot(rms[m].mean(axis=0), label=m)
        axes[1].plot(trace[m], label=m, lw=1)
    axes[0].set(title=f"RMS error over states (mean of {args.seeds} seeds)",
                xlabel="episode", yscale="log")
    axes[1].axhline(true_V[nS - 2], color="k", ls="--", lw=1, label="true V(s5)")
    axes[1].set(title="one seed's estimate of V(s5): MC is noisy, TD is smooth but lags",
                xlabel="episode")
    axes[0].legend()
    axes[1].legend(fontsize=8)
    fig.suptitle("Chapter 5: Monte Carlo vs temporal-difference prediction on the random walk")
    fig.tight_layout()
    path = OUT / "prediction.png"
    fig.savefig(path, dpi=120)
    print(f"\nwrote {path}")
    return _report(path, snaps, true_V, args.episodes, rms)


def _pick_episode(P, start, max_steps):
    """Five warm-up games, then a game worth narrating: it ends at the goal, with some
    back and forth on the way."""
    env = envs.TabularEnv(P, start, max_steps, seed=3)
    warmup = [generate_episode(env, lambda s: 0) for _ in range(5)]
    for _ in range(500):
        traj = generate_episode(env, lambda s: 0)
        if traj[-1][1] > 0 and 6 <= len(traj) <= 10:
            return warmup, traj
    return warmup, traj


def _warm_start(warmup, alpha, update):
    """Guesses after a few earlier games, so the narrated game has something to change."""
    V = np.full(len(viz.layout_for("rw")[0]), 0.5)
    V[0] = V[-1] = 0.0
    for tr in warmup:
        update(V, tr, alpha, 1.0)
    return V


def _walk_frames_td(traj, alpha, layout, V):
    """TD(0) on one game: an update after every single step, arithmetic shown."""
    V = V.copy()
    frames = [viz.grid(layout, V, agent=traj[0][0], pct=True, vmax=1.0)]
    caps = ["these are its guesses after 5 earlier games (every square started at 50%). Press Play."]
    for t, (s, r, s2, done) in enumerate(traj):
        nxt = 0.0 if done else V[s2]
        target = r + nxt
        old = V[s]
        V[s] += alpha * (target - V[s])
        where = {0: f"into the {viz.HOLE}", len(V) - 1: f"onto the {viz.GOAL} (+1)"}.get(
            s2, f"to square {s2} (guess {_p(nxt)})")
        caps.append(
            f"step {t + 1}: the coin moves it {'→' if s2 > s else '←'} {where}. "
            f"<b>TD update for square {s}:</b> target = reward + guess at the new square = "
            f"{r:g} + {_p(nxt)} = {_p(target)}; it was {_p(old)}, so move halfway (α = {alpha}) "
            f"towards the target: <b>{_p(old)} → {_p(V[s])}</b>")
        frames.append(viz.grid(layout, V, agent=s2, trail=[x[0] for x in traj[:t + 1]], pct=True,
                               vmax=1.0))
    return frames, caps


def _walk_frames_mc(traj, alpha, layout, V):
    """First-visit MC on the same game: nothing until the end, then everything at once."""
    V = V.copy()
    frames = [viz.grid(layout, V, agent=traj[0][0], pct=True, vmax=1.0)]
    caps = ["these are its guesses after the same 5 earlier games. Press Play."]
    for t, (s, r, s2, done) in enumerate(traj):
        caps.append(f"step {t + 1}: the coin moves it {'→' if s2 > s else '←'}. "
                    "<b>No update yet:</b> Monte Carlo has to wait until the game is over to "
                    "know how it ended.")
        frames.append(viz.grid(layout, V, agent=s2, trail=[x[0] for x in traj[:t + 1]], pct=True,
                               vmax=1.0))
    G = sum(r for _, r, _, _ in traj)  # gamma = 1
    for s in dict.fromkeys(x[0] for x in traj):  # first visits, in order
        old = V[s]
        V[s] += alpha * (G - V[s])
        caps.append(f"game over, final score <b>{G:g}</b>. Every square it visited moves halfway "
                    f"(α = {alpha}) towards that score: <b>square {s}: {_p(old)} → {_p(V[s])}</b>")
        frames.append(viz.grid(layout, V, agent=s, pct=True, vmax=1.0))
    return frames, caps


def _p(v):
    return f"{v:.0%}"


def _estimates_slider(snaps, true_V, episodes, layout):
    """Every method's current V on the bridge, one row each, episode by episode."""
    shown = sorted(set(np.unique(np.geomspace(1, episodes, 40).astype(int)) - 1))
    frames = []
    for e in shown:
        rows = [f"<tr><td style='padding-right:10px'><b>the true answer</b></td><td>"
                f"{viz.grid(layout, true_V, vmax=1.0, pct=True)}</td></tr>"]
        rows += [f"<tr><td style='padding-right:10px'>{m}</td><td>"
                 f"{viz.grid(layout, s[e], vmax=1.0, pct=True)}</td></tr>" for m, s in snaps.items()]
        frames.append("<table>" + "".join(rows) + "</table>")
    caps = [f"after {e + 1} games" for e in shown]
    return viz.slider(frames, caps, "every method's guesses, game by game: press Play", 350)


def _report(path, snaps, true_V, episodes, rms):
    P, start, max_steps, _ = envs.ENVS["rw"]
    layout = viz.layout_for("rw")
    warmup, traj = _pick_episode(P, start, max_steps)
    td_f, td_c = _walk_frames_td(traj, 0.5, layout, _warm_start(warmup, 0.5, td_update))
    mc_f, mc_c = _walk_frames_mc(traj, 0.5, layout, _warm_start(warmup, 0.5, METHODS["first-visit MC"]))
    rows = "".join(
        f"<tr><td>{m}</td>" + "".join(f"<td>{rms[m].mean(axis=0)[c]:.3f}</td>" for c in (9, 99, episodes - 1))
        + "</tr>" for m in METHODS)
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> without a rulebook, you can still learn "
        "how good each square is by watching games, either by waiting to see how each game ends "
        "(Monte Carlo) or by updating after every step from your next guess (temporal difference).</p>",
        viz.intro("rw", "The agent's moves do not matter here: the question is only <b>how likely "
                  f"is a game to end at the {viz.GOAL}, starting from each square?</b> We can "
                  "work out the true answer (17%, 33%, 50%, 67%, 83%), but the learners are not "
                  "told it: they only get to watch games. That makes this the perfect test: we can "
                  "see exactly how wrong each learner is at every moment."),
        viz.grid(layout, true_V, pct=True, vmax=1.0, caption="the true answer, which the learners "
                 "never see"),
        viz.legend(arrows=False),

        "<h3>1. One game, learned two ways</h3>"
        "<p>(In the very first game TD learns almost nothing: every target is another 50% "
        "guess, so only the square next to the end moves. This is game 6, after 5 warm-up "
        "games, so both learners already have opinions.)</p>",
        viz.row(viz.slider(td_f, td_c, "temporal difference, TD(0): learn after every step", 1200),
                viz.slider(mc_f, mc_c, "Monte Carlo (MC): learn when the game ends", 1200)),
        viz.explain(
            how="Both boards watch the <i>same</i> game. The numbers are each learner's current "
                "guess for that square, and the text says exactly how each guess changed. α (alpha, "
                "the step size) = 0.5 means \"move halfway from the old guess towards the target\".",
            notice="TD updates on every step using its <i>own next guess</i> as the target "
                   "(\"bootstrapping\"); early on, most of those targets are just 50% guesses, so "
                   "only the squares near the ends really move. MC waits, then pushes every square "
                   "it visited towards the real final score. TD's targets are biased (they are "
                   "guesses) but calm; MC's are honest but swing wildly from game to game."),

        "<h3>2. Hundreds of games: who gets the right answer fastest?</h3>",
        _estimates_slider(snaps, true_V, episodes, layout),
        viz.explain(
            how="Same idea over many games. The top row is the true answer; each row below is "
                "one learner's guesses after that many games. α starts at 0.5 and shrinks, so "
                "later games nudge the guesses less.",
            notice="MC's guesses jump with every game's result. TD's change smoothly from both "
                   "ends inwards: the ends are the only places a real result comes from. "
                   "n-step TD and TD(λ) sit in between."),
        viz.figure(path),
        f"<table class='dc'><tr><th>method</th><th>error after 10 games</th><th>100 games</th>"
        f"<th>{episodes} games</th></tr>{rows}</table>",
        viz.explain(
            how="The error is the RMS (root mean square) gap between a learner's guesses and the "
                "true answer, averaged over 100 repeats with the same games fed to every learner.",
            notice="TD(0) is ahead at every point on this small, noisy bridge: low variance beats "
                   "low bias here. Every-visit MC is worst; a square visited 5 times in one game "
                   "gets 5 big pushes from the same lucky or unlucky result."),

        viz.takeaways([
            "You can learn how good a square is from experience alone, no rulebook needed.",
            "Monte Carlo waits for the real outcome: unbiased, but noisy.",
            "Temporal difference updates every step from its own next guess: biased early, "
            "but much calmer, and often faster.",
            "n-step TD and TD(λ) blend the two, trading bias for variance.",
            "Next chapter: use this to learn not just how good a plan is, but the plan itself.",
        ]),
    ]

if __name__ == "__main__":
    main()
