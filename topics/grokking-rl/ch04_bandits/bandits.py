"""Chapter 4: balancing gathering and using information.

The 10-armed Gaussian bandit testbed: one state, ten slot machines, each paying
N(mean_k, 1) with mean_k ~ N(0, 1). Every strategy below is the same loop
(pick an arm, pull it, update a running average); only the pick differs.
UCB = upper confidence bound; Thompson sampling draws from a posterior per arm.

All runs are vectorised: `Q` is (runs, arms), so 2000 independent bandits learn
in lockstep and the curves are averages over all of them.

    python -m ch04_bandits.bandits      # from topics/grokking-rl; writes bandits.png here
"""

import argparse
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import asciichart
import viz

# Figures land next to this file (the README embeds them); Flyte points it at scratch.
OUT = Path(os.environ.get("GROKKING_OUT", Path(__file__).parent))


def greedy(Q, rng):
    # Tiny noise breaks ties at random instead of always picking arm 0.
    return (Q + rng.random(Q.shape) * 1e-9).argmax(axis=1)


def sample_from(probs, rng):
    cum = probs.cumsum(axis=1)
    u = rng.random((len(probs), 1)) * cum[:, -1:]
    return np.minimum((cum < u).sum(axis=1), probs.shape[1] - 1)


def pure_exploitation(Q, N, t, T, rng):
    return greedy(Q, rng)


def pure_exploration(Q, N, t, T, rng):
    return rng.integers(Q.shape[1], size=len(Q))


def epsilon_greedy(eps):
    def select(Q, N, t, T, rng):
        explore = rng.random(len(Q)) < eps
        return np.where(explore, rng.integers(Q.shape[1], size=len(Q)), greedy(Q, rng))
    return select


def decaying_epsilon(start=1.0, end=0.01, frac=0.3):
    def select(Q, N, t, T, rng):
        eps = max(end, start - (start - end) * t / (frac * T))
        return epsilon_greedy(eps)(Q, N, t, T, rng)
    return select


def softmax(start_temp=1.0, end_temp=0.01, frac=0.6):
    def select(Q, N, t, T, rng):
        temp = max(end_temp, start_temp - (start_temp - end_temp) * t / (frac * T))
        z = (Q - Q.max(axis=1, keepdims=True)) / temp
        return sample_from(np.exp(z), rng)
    return select


def ucb(c=2.0):
    def select(Q, N, t, T, rng):
        with np.errstate(divide="ignore", invalid="ignore"):
            bonus = c * np.sqrt(np.log(t + 1) / N)
        return greedy(np.where(N == 0, np.inf, Q + bonus), rng)
    return select


def thompson(Q, N, t, T, rng):
    # Gaussian posterior with a N(0, 1) prior and known reward variance 1:
    # mean = sum / (n + 1), std = 1 / sqrt(n + 1). Draw once per arm, play the best draw.
    post_mean = Q * N / (N + 1)
    return greedy(rng.normal(post_mean, 1 / np.sqrt(N + 1)), rng)


# name -> (select, initial Q, initial count). Optimistic init works by pretending
# every arm has already paid out q0 a few times, so the agent is "disappointed"
# into trying each one.
STRATEGIES = {
    "pure exploitation": (pure_exploitation, 0.0, 0),
    "pure exploration": (pure_exploration, 0.0, 0),
    "epsilon-greedy 0.1": (epsilon_greedy(0.1), 0.0, 0),
    "decaying epsilon": (decaying_epsilon(), 0.0, 0),
    "optimistic (q0=5, n0=5)": (pure_exploitation, 5.0, 5),
    "softmax (decaying temp)": (softmax(), 0.0, 0),
    "UCB c=2": (ucb(2.0), 0.0, 0),
    "Thompson sampling": (thompson, 0.0, 0),
}


def run(select, q0, n0, means, steps, rng):
    runs, arms = means.shape
    Q = np.full((runs, arms), q0, dtype=float)
    N = np.full((runs, arms), n0, dtype=float)
    idx = np.arange(runs)
    best = means.argmax(axis=1)
    reward = np.zeros((steps, runs))
    optimal = np.zeros((steps, runs))
    regret = np.zeros((steps, runs))
    for t in range(steps):
        a = select(Q, N, t, steps, rng)
        r = rng.normal(means[idx, a], 1.0)
        N[idx, a] += 1
        Q[idx, a] += (r - Q[idx, a]) / N[idx, a]
        reward[t] = r
        optimal[t] = a == best
        regret[t] = means[idx, best] - means[idx, a]
    return reward.mean(axis=1), optimal.mean(axis=1), regret.cumsum(axis=0).mean(axis=1)


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--arms", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    means = rng.normal(0, 1, (args.runs, args.arms))
    print(f"{args.runs} bandits x {args.arms} arms x {args.steps} pulls")
    print(f"best possible average reward: {means.max(axis=1).mean():.3f}\n")

    results = {}
    for name, (select, q0, n0) in STRATEGIES.items():
        results[name] = run(select, q0, n0, means, args.steps, rng)
        r, opt, reg = results[name]
        print(f"{name:>24}: reward {r.mean():6.3f}   best arm {opt[-100:].mean():6.1%} "
              f"(last 100)   total regret {reg[-1]:7.1f}")

    print()
    print(asciichart.bars({k: v[2][-1] for k, v in results.items()},
                          title="total regret after all pulls (lower is better)"))
    print()
    keep = ["pure exploitation", "epsilon-greedy 0.1", "UCB c=2", "Thompson sampling"]
    print(asciichart.line({k: results[k][1] for k in keep}, ymin=0, ymax=1,
                          title="fraction of runs pulling the best arm",
                          xmax=args.steps, xlabel="pull"))

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for name, (r, opt, reg) in results.items():
        smooth = np.convolve(r, np.ones(10) / 10, mode="valid")
        axes[0].plot(smooth, label=name, lw=1.2)
        axes[1].plot(opt, label=name, lw=1.2)
        axes[2].plot(reg, label=name, lw=1.2)
    axes[0].set(title="average reward (10-step smoothing)", xlabel="pull")
    axes[1].set(title="% of runs pulling the best arm", xlabel="pull", ylim=(0, 1))
    axes[2].set(title="cumulative regret (lower is better)", xlabel="pull", yscale="log")
    axes[2].legend(fontsize=8, loc="lower right")
    fig.suptitle("Chapter 4: exploration strategies on the 10-armed Gaussian bandit")
    fig.tight_layout()
    path = OUT / "bandits.png"
    fig.savefig(path, dpi=120)
    print(f"\nwrote {path}")
    return _report(path, results, means.max(axis=1).mean())



# ── the report: one bandit, pull by pull ────────────────────────────────────────

def _explain_one(kind, mu, pulls=1000, detailed=40, seed=1):
    """Play one bandit with one strategy, narrating every early pull.

    Same update rule as `run()`, but for a single bandit and with the reasoning kept.
    Returns (frames, captions).
    """
    rng = np.random.default_rng(seed)
    K = len(mu)
    Q, N = np.zeros(K), np.zeros(K)
    best, regret = int(np.argmax(mu)), 0.0
    frames, caps = [], []
    shown = set(range(detailed)) | {99, 249, 499, pulls - 1}
    for t in range(pulls):
        extra, label = None, ""
        if kind == "eps":
            if rng.random() < 0.1:
                a = int(rng.integers(K))
                why = f"the 1-in-10 coin says <b>explore</b>, so it picks arm {a + 1} at random"
            else:
                a = int(np.argmax(Q + rng.random(K) * 1e-9))
                ties = int((Q == Q.max()).sum())
                why = (f"<b>exploits</b>: arm {a + 1} has the best estimate so far" if ties == 1 else
                       f"<b>exploits</b>, but {ties} machines tie for the best estimate "
                       f"({Q.max():.2f}), so it picks arm {a + 1} among them")
        elif kind == "ucb":
            with np.errstate(divide="ignore", invalid="ignore"):
                bonus = 2.0 * np.sqrt(np.log(t + 1) / N)
            score = np.where(N == 0, np.inf, Q + bonus)
            a = int(np.argmax(score + rng.random(K) * 1e-9))
            extra, label = np.where(N == 0, np.inf, bonus), "uncertainty bonus"
            why = (f"arm {a + 1} has never been tried, so its score is infinite" if N[a] == 0 else
                   f"arm {a + 1} has the highest score: estimate {Q[a]:.2f} + bonus {bonus[a]:.2f} "
                   f"(the bonus is big for arms tried only a few times)")
        else:
            draws = rng.normal(Q * N / (N + 1), 1 / np.sqrt(N + 1))
            a = int(np.argmax(draws))
            extra, label = draws, "imagined payout this pull"
            why = (f"it imagines a payout for every arm from what it has seen so far; arm {a + 1}'s "
                   f"imagined payout ({draws[a]:.2f}) is the highest, so it pulls that one")
        r = rng.normal(mu[a], 1.0)
        old = Q[a]
        N[a] += 1
        Q[a] += (r - Q[a]) / N[a]
        regret += mu[best] - mu[a]
        if t in shown:
            frames.append(_arms_table(mu, Q, N, a, extra, label))
            caps.append(f"pull {t + 1}: {why}. It pays <b>{r:.2f}</b>; arm {a + 1}'s estimate "
                        f"{old:.2f} → <b>{Q[a]:.2f}</b> (the average of its {int(N[a])} "
                        f"{'pull' if N[a] == 1 else 'pulls'}). "
                        f"Reward lost so far vs always pulling the best arm: {regret:.1f}"
                        + ("" if t < detailed else "  <i>(skipping ahead)</i>"))
    return frames, caps


def _arms_table(mu, Q, N, pulled, extra=None, label=""):
    lo, hi, w = -3.0, 3.0, 150
    x = lambda v: (np.clip(v, lo, hi) - lo) / (hi - lo) * w
    best = int(np.argmax(mu))
    head = f"<th>{label}</th>" if extra is not None else ""
    rows = []
    for k in range(len(mu)):
        est = (f"<div style='position:relative;width:{w}px;height:12px;background:#222'>"
               f"<div style='position:absolute;left:{x(min(0, Q[k])):.0f}px;width:{abs(x(Q[k]) - x(0)):.0f}px;"
               f"height:12px;background:#fdcb6e'></div>"
               f"<div style='position:absolute;left:{x(mu[k]):.0f}px;width:2px;height:12px;background:#fff'></div></div>")
        ex = ""
        if extra is not None:
            ex = "<td>∞</td>" if np.isinf(extra[k]) else f"<td>{extra[k]:.2f}</td>"
        style = " style='background:#1e3a2f'" if k == pulled else ""
        rows.append(f"<tr{style}><td>arm {k + 1}{' 👑' if k == best else ''}</td><td>{est}</td>"
                    f"<td>{Q[k]:.2f}</td><td>{int(N[k])}</td>{ex}</tr>")
    return (f"<table class='dc'><tr><th>machine</th><th>estimate (yellow) vs secret average "
            f"(white tick)</th><th>estimate</th><th>pulls</th>{head}</tr>{''.join(rows)}</table>")


def _report(path, results, best_avg):
    mu = np.random.default_rng(7).normal(0, 1, 10)
    sliders = []
    for kind, title in (("eps", "ε-greedy (explore 1 pull in 10)"),
                        ("ucb", "UCB: optimism about the unknown"),
                        ("thompson", "Thompson sampling: act on a guess")):
        frames, caps = _explain_one(kind, mu)
        sliders.append(viz.slider(frames, caps, title, 700))
    rows = "".join(
        f"<tr><td>{name}</td><td>{r.mean():.2f}</td><td>{opt[-100:].mean():.0%}</td>"
        f"<td>{reg[-1]:.0f}</td></tr>" for name, (r, opt, reg) in results.items())
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> to find the best option you have to "
        "try the others, and every try of a worse one costs you; the trick is deciding how much "
        "to explore.</p>",
        '<div class="card"><b>The slot machines.</b> There are 10 machines (the "arms" of a '
        '"multi-armed bandit"). Each one pays a random amount around its own <i>secret average</i>; '
        "some averages are good, some are negative. You get 1,000 pulls and want the biggest "
        "total. There is no map and no ice here: just one decision, made over and over, and the "
        "only way to learn about a machine is to pull it. Every strategy keeps the same notes "
        "(the average payout it has seen from each machine) and differs only in how it picks "
        "the next pull.</div>",

        "<h3>1. Watch three strategies play the same 10 machines</h3>",
        viz.explain(
            how="Each row is a machine. The white tick is its secret average (the agent cannot see "
                "it; 👑 marks the best machine); the yellow bar is the agent's current estimate. "
                "The highlighted row is the machine it just pulled. The text above says why it "
                "chose it and how the estimate changed. The first 40 pulls are shown one by one, "
                "then it skips to pulls 100, 250, 500 and 1000."),
        *sliders,
        viz.explain(notice="ε-greedy keeps wasting 1 pull in 10 on random machines forever, even "
                           "after it has found the best. UCB tries every machine once, then keeps "
                           "re-checking the ones it is unsure about. Thompson sampling's guesses "
                           "get sharper as it learns, so it explores a lot at first and almost "
                           "never at the end, with no schedule to tune."),

        "<h3>2. The same strategies on 2,000 different sets of machines</h3>",
        viz.figure(path),
        viz.explain(
            how="Left: average payout per pull (the best possible is "
                f"{best_avg:.2f}). Middle: how often each strategy pulls the best machine. "
                "Right: <b>regret</b>, the total payout lost compared with always pulling the "
                "best machine (lower is better, log scale)."),
        f"<table class='dc'><tr><th>strategy</th><th>average payout</th><th>pulls the best machine "
        f"(last 100)</th><th>total regret</th></tr>{rows}</table>",
        viz.explain(notice="Pure exploitation locks onto whichever machine looked good first and is "
                           "stuck with it a third of the time it is wrong. Pure exploration never "
                           "uses what it learns. Thompson sampling loses the least by a wide "
                           "margin."),

        viz.takeaways([
            "Exploring costs reward now; not exploring costs reward forever.",
            "ε-greedy is simple but never stops paying for random pulls.",
            "UCB explores where it is uncertain: optimism in the face of the unknown.",
            "Thompson sampling explores in proportion to its own uncertainty, and wins here.",
            "Next chapter: back to worlds with states, learning a plan's value from play.",
        ]),
    ]

if __name__ == "__main__":
    main()
