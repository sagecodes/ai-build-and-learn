"""Chapter 2: mathematical foundations of reinforcement learning.

An MDP (Markov decision process) is just a table: states, actions, and for each (state, action) a list of
(probability, next state, reward, done). This script prints those tables for the
book's walks, then checks by sampling that the environment really behaves the way
its table says, and shows what discounting does to a return.

    python -m ch02_mdps.mdps            # from topics/grokking-rl
"""

from collections import Counter

import numpy as np

import envs
import viz


def act(P, a):
    return "LR"[a] if len(P[0]) == 2 else envs.LAKE_ARROWS[a]


def print_mdp(name, P):
    print(f"\n{name}: {len(P)} states, {len(P[0])} actions")
    for s in P:
        for a in P[s]:
            moves = ", ".join(
                f"{p:.2f}->s{s2}{' r=+1' if r else ''}{' (end)' if d else ''}"
                for p, s2, r, d in P[s][a]
            )
            print(f"  s{s} {act(P, a)}: {moves}")


def empirical_transitions(name, s, a, n=100_000, seed=0):
    P, start, _, _ = envs.ENVS[name]
    env = envs.TabularEnv(P, s, seed=seed)
    counts = Counter()
    for _ in range(n):
        env.reset()
        s2, *_ = env.step(a)
        counts[s2] += 1
    table = {s2: p for p, s2, _, _ in P[s][a]}
    print(f"\n{name.upper()} from s{s}, action {act(P, a)} ({n:,} samples):")
    for s2 in sorted(table):
        print(f"  s{s2}: table {table[s2]:.3f}  sampled {counts[s2] / n:.3f}")
    return table, counts, n


def episode_returns(name, policy, episodes=20_000, gammas=(1.0, 0.99, 0.9), seed=0):
    env = envs.make(name, seed)
    returns = {g: [] for g in gammas}
    lengths = []
    for _ in range(episodes):
        s, done, trunc, rewards = env.reset(), False, False, []
        while not (done or trunc):
            s, r, done, trunc = env.step(policy(s, env.rng))
            rewards.append(r)
        lengths.append(len(rewards))
        for g in gammas:
            returns[g].append(sum(r * g**t for t, r in enumerate(rewards)))
    return returns, np.array(lengths)


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    print_mdp("Bandit walk (BW)", envs.BW)
    print_mdp("Bandit slippery walk (BSW)", envs.BSW)
    print_mdp("Slippery walk five (SWF)", envs.SWF)

    dice_walk = empirical_transitions("swf", 3, envs.RIGHT)
    dice_lake = empirical_transitions("fl4", 1, 3)  # top edge, above the hole, choose up

    print("\nSame environment, two policies, three discount factors (SWF, start s3):")
    policies = {
        "random": lambda s, rng: rng.integers(2),
        "always right": lambda s, rng: envs.RIGHT,
    }
    summary = {}
    for name, pol in policies.items():
        returns, lengths = episode_returns("swf", pol)
        summary[name] = ({k: float(np.mean(v)) for k, v in returns.items()}, float(lengths.mean()))
        g = "  ".join(f"G(gamma={k})={np.mean(v):.3f}" for k, v in returns.items())
        print(f"  {name:>12}: mean length {lengths.mean():5.1f} steps  {g}")
    print("gamma=1 only asks whether you arrive. gamma<1 also asks how fast, so the slower random\n"
          "policy loses a bigger share of its return as gamma drops.")

    return _report(policies, dice_walk, dice_lake, summary)


def _dice_bars(name, s, a, dice, what):
    """The table's probabilities next to what 100,000 real tries did."""
    table, counts, n = dice
    rows = []
    for s2 in sorted(table, key=lambda k: -table[k]):
        where = viz._direction(name, s, s2)
        if viz._cell(name, s2) in "HG":
            where += {"H": f" into the {viz.HOLE}", "G": f" onto the {viz.GOAL}"}[viz._cell(name, s2)]
        p, q = table[s2], counts[s2] / n
        bar = lambda v, col: (f"<div style='background:{col};height:10px;width:{v * 300:.0f}px;"
                              "display:inline-block'></div>")
        rows.append(f"<tr><td>{where}</td><td>{bar(p, '#00b894')} {p:.1%}</td>"
                    f"<td>{bar(q, '#fdcb6e')} {q:.1%}</td></tr>")
    return (f"<div style='background:#1a1a2e;padding:12px;border-radius:6px;display:inline-block'>"
            f"<b>{what}</b><table class='dc'><tr><th>what happened</th><th>the rulebook says</th>"
            f"<th>{n:,} real tries</th></tr>{''.join(rows)}</table></div>")


def _report(policies, dice_walk, dice_lake, summary):
    """The Flyte report: written to be understood without reading the chapter first."""
    rand = policies["random"]
    right = policies["always right"]
    lake_rand = lambda s, rng: rng.integers(4)
    gam = "".join(
        f"<tr><td>{name}</td><td>{length:.1f} steps</td>"
        + "".join(f"<td>{g:.2f}</td>" for g in returns.values()) + "</tr>"
        for name, (returns, length) in summary.items()
    )
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> a reinforcement learning problem is "
        "a world you can write down as a table: where you are, what you can do, where each "
        "choice might take you, and what you score.</p>",
        '<div class="card">That table is called a <b>Markov decision process (MDP)</b>. '
        "<i>Markov</i> means the next step depends only on where you are now and what you choose, "
        "not on how you got there. The whole book runs on a few tiny worlds like these, so it is "
        "worth meeting them properly.</div>",

        "<h3>1. The worlds</h3>",
        viz.intro("swf"),
        viz.grid(viz.layout_for("swf"), caption="the slippery walk: 5 squares of ice"),
        viz.intro("fl4"),
        viz.grid(envs.FL4, caption="the 4x4 frozen lake"),
        viz.legend(arrows=False),

        "<h3>2. Is the world really that random? Roll the dice 100,000 times</h3>",
        viz.row(
            _dice_bars("swf", 3, envs.RIGHT, dice_walk, "slippery walk, middle square, choose →"),
            _dice_bars("fl4", 1, 3, dice_lake, "lake, top edge above a hole, choose ↑"),
        ),
        viz.explain(
            how="Green is what the MDP's table says should happen; yellow is what actually "
                "happened when we made that one move 100,000 times from the same square.",
            notice="They match: the table <i>is</i> the world. On the lake, choosing ↑ into the "
                   "top wall bumps the edge and stays put 1 time in 3, and slides along the edge "
                   "the other 2. It never goes down into the hole."),

        "<h3>3. Agents that know nothing: random moves</h3>",
        viz.row(
            viz.replay("swf", rand, "slippery walk, random", viz.find_seed("swf", rand, True)),
            viz.replay("swf", rand, "slippery walk, random", viz.find_seed("swf", rand, False)),
        ),
        viz.row(
            viz.replay("fl4", lake_rand, "lake, random", viz.find_seed("fl4", lake_rand, True)),
            viz.replay("fl4", lake_rand, "lake, random", viz.find_seed("fl4", lake_rand, False)),
        ),
        viz.explain(
            how="Press Play. The line above each board says what the agent chose and what the "
                "ice actually did. The faint yellow squares are where it has been.",
            notice="Even on the same world, the same (random) agent sometimes wins and sometimes "
                   "loses. That is why every number later in the book is an average over many "
                   "games, or computed exactly from the table."),

        "<h3>4. Scoring: the return, and how patient the agent is (γ)</h3>",
        f"<table class='dc'><tr><th>policy on the slippery walk</th><th>average game length</th>"
        f"<th>score with γ = 1</th><th>γ = 0.99</th><th>γ = 0.9</th></tr>{gam}</table>",
        viz.explain(
            how="The <b>return</b> is the total score of one game. With a discount γ (gamma), a "
                "point scored t steps from now only counts γ<sup>t</sup>: with γ = 0.9, a point "
                "10 steps away is worth 0.9<sup>10</sup> ≈ 0.35. The table averages 20,000 games.",
            notice="With γ = 1 the score only asks <i>whether</i> you reach the goal (random: half "
                   "the time). Lower γ also asks <i>how fast</i>, and the slow random agent loses "
                   "a much bigger share of its score than the direct one."),

        viz.takeaways([
            "An RL problem is a table: states, actions, the odds of where each action leads, "
            "and rewards. That table is the MDP.",
            "Choosing an action and getting the outcome you wanted are different things on ice.",
            "The return adds up rewards, discounted by γ: γ is how much the agent cares about later.",
            "Next chapter: given the table, work out the best plan without playing at all.",
        ]),
    ]

if __name__ == "__main__":
    main()
