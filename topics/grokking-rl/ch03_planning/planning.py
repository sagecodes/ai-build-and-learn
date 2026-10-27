"""Chapter 3: balancing immediate and long-term goals.

Dynamic programming (DP) with the full model in hand:
  * policy evaluation   - what is this policy worth, in every state?
  * policy improvement  - act greedily on those values to get a better policy
  * policy iteration    - alternate the two until nothing changes
  * value iteration     - fold improvement into every evaluation sweep

The later chapters import `policy_evaluation` and `value_iteration` from here:
the optimal values are the ground truth every model-free learner is scored against.

    python -m ch03_planning.planning    # from topics/grokking-rl
"""

import argparse
import time

import numpy as np

import envs
import viz


def to_arrays(P):
    """R[s, a] = expected immediate reward; T[s, a, s'] = prob of continuing into s'.

    Transitions that end the episode are left out of T, so `R + gamma * T @ V`
    is the one-step Bellman backup with V(terminal) = 0 built in.
    """
    nS, nA = len(P), len(P[0])
    R = np.zeros((nS, nA))
    T = np.zeros((nS, nA, nS))
    for s in range(nS):
        for a in range(nA):
            for p, s2, r, done in P[s][a]:
                R[s, a] += p * r
                if not done:
                    T[s, a, s2] += p
    return R, T


def policy_evaluation(pi, P, gamma=1.0, theta=1e-10, max_sweeps=100_000, arrays=None):
    """Iterative policy evaluation for a deterministic policy (pi[s] = action).

    Pass `arrays=to_arrays(P)` when calling this in a loop to skip rebuilding them.
    """
    R, T = arrays if arrays is not None else to_arrays(P)
    idx = np.arange(len(P))
    Rpi, Tpi = R[idx, pi], T[idx, pi]
    V = np.zeros(len(P))
    for _ in range(max_sweeps):
        V_new = Rpi + gamma * Tpi @ V
        if np.max(np.abs(V_new - V)) < theta:
            return V_new
        V = V_new
    return V


def q_from_v(V, P, gamma=1.0):
    R, T = to_arrays(P)
    return R + gamma * T @ V


def policy_improvement(V, P, gamma=1.0):
    return np.argmax(q_from_v(V, P, gamma), axis=1)


def policy_iteration(P, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pi = rng.integers(len(P[0]), size=len(P))
    rounds = 0
    while True:
        rounds += 1
        V = policy_evaluation(pi, P, gamma)
        new_pi = policy_improvement(V, P, gamma)
        # Compare values, not actions: ties can flip actions forever without changing V.
        if np.allclose(q_from_v(V, P, gamma)[np.arange(len(P)), new_pi], V, atol=1e-9):
            return V, new_pi, rounds
        pi = new_pi


def value_iteration(P, gamma=1.0, theta=1e-10, max_sweeps=100_000):
    R, T = to_arrays(P)
    V = np.zeros(len(P))
    for sweep in range(1, max_sweeps + 1):
        Q = R + gamma * T @ V
        V_new = Q.max(axis=1)
        if np.max(np.abs(V_new - V)) < theta:
            return V_new, Q.argmax(axis=1), sweep
        V = V_new
    return V, Q.argmax(axis=1), max_sweeps


def value_iteration_history(P, gamma, sweeps):
    """V and greedy policy after each sweep, so a report can animate them."""
    R, T = to_arrays(P)
    V, hist = np.zeros(len(P)), []
    for _ in range(max(sweeps) + 1):
        Q = R + gamma * T @ V
        hist.append((V.copy(), Q.argmax(axis=1)))
        V = Q.max(axis=1)
    return [hist[k] for k in sweeps]


def policy_iteration_history(P, gamma=1.0, seed=0):
    rng = np.random.default_rng(seed)
    pi = rng.integers(len(P[0]), size=len(P))
    hist = []
    while True:
        V = policy_evaluation(pi, P, gamma)
        hist.append((V, pi.copy()))
        new_pi = policy_improvement(V, P, gamma)
        if np.allclose(q_from_v(V, P, gamma)[np.arange(len(P)), new_pi], V, atol=1e-9):
            return hist
        pi = new_pi


def rollout_success(P, pi, start, max_steps, episodes=10_000, seed=0):
    """Empirical check of V(start): how often does pi actually reach the goal?"""
    env = envs.TabularEnv(P, start, max_steps, seed)
    wins = 0
    for _ in range(episodes):
        s, done, trunc = env.reset(), False, False
        while not (done or trunc):
            s, r, done, trunc = env.step(pi[s])
        wins += r > 0
    return wins / episodes


def main(argv=None):
    """Prints the results; returns HTML parts for the report."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=10_000)
    args = ap.parse_args(argv)

    print("=" * 72)
    print("Slippery walk five: how good are 'always left' and 'always right'?")
    print("=" * 72)
    P, start, _, layout = envs.ENVS["swf"]
    for name, a in (("always left", envs.LEFT), ("always right", envs.RIGHT)):
        pi = np.full(len(P), a)
        V = policy_evaluation(pi, P, gamma=1.0)
        print(f"{name:>13}: {envs.show(V)}")
    print("Even 'always right' only reaches the goal from the start "
          f"{policy_evaluation(np.full(len(P), 1), P)[start]:.1%} of the time: the ice pushes you back.")

    print()
    print("=" * 72)
    print("Frozen lake 4x4: a hand-written 'careful' policy vs the optimal one")
    print("=" * 72)
    P, start, max_steps, layout = envs.ENVS["fl4"]
    L, D, R_, U = 0, 1, 2, 3
    # Hand-written: hug the walls so a slip pushes you into a wall, not into a hole.
    careful = np.array([L, U, U, U, L, L, R_, L, U, D, L, L, L, R_, D, L])
    gofor = np.array([R_, R_, D, L, D, L, D, L, R_, R_, D, L, L, R_, R_, L])

    for name, pi in (("go-get-it", gofor), ("careful", careful)):
        V = policy_evaluation(pi, P, gamma=1.0)
        emp = rollout_success(P, pi, start, max_steps, args.episodes)
        print(f"\n{name}: P(success) = {V[start]:.3f} exact, {emp:.3f} sampled "
              f"(sampled runs stop at {max_steps} steps)")
        print(envs.show(V, layout, pi))

    t = time.perf_counter()
    V_pi, pi_pi, rounds = policy_iteration(P, gamma=1.0)
    t_pi = time.perf_counter() - t
    t = time.perf_counter()
    V_vi, pi_vi, sweeps = value_iteration(P, gamma=1.0)
    t_vi = time.perf_counter() - t
    print(f"\npolicy iteration: {rounds} improvement rounds, {t_pi*1e3:.1f} ms")
    print(f"value iteration:  {sweeps} sweeps, {t_vi*1e3:.1f} ms")
    print(f"same values: {np.allclose(V_pi, V_vi, atol=1e-6)}")
    emp = rollout_success(P, pi_vi, start, max_steps, args.episodes)
    print(f"\noptimal: P(success) = {V_vi[start]:.3f} exact, {emp:.3f} sampled")
    print("The careful policy was already (almost) optimal. Wall-hugging is the whole trick.")
    print(envs.show(V_vi, layout, pi_vi))

    print()
    print("=" * 72)
    print("What gamma does: the same lake, solved with less patience")
    print("=" * 72)
    for gamma in (1.0, 0.99, 0.9, 0.5):
        V, pi, _ = value_iteration(P, gamma)
        exact = policy_evaluation(pi, P, gamma=1.0)[start]
        print(f"\ngamma={gamma}: greedy policy reaches the goal {exact:.1%} of the time")
        print(envs.show(V, layout, pi))
    print("\nLow gamma trades safety for speed: it takes risky shortcuts past the holes.")

    parts = _report(P, start, layout, gofor, careful, V_vi, pi_vi)
    return parts


def _report(P, start, layout, gofor, careful, V_opt, pi_opt):
    """The Flyte report: written to be understood without reading the chapter first."""
    name = "fl4"
    pct = dict(pct=True, vmax=1.0)
    V_go, V_careful = policy_evaluation(gofor, P), policy_evaluation(careful, P)

    sweeps = list(range(0, 16)) + [20, 25, 30, 40, 50, 75, 100, 200, 400, 800]
    vi = value_iteration_history(P, 1.0, sweeps)
    vi_frames = [viz.grid(layout, V, pi, Q=q_from_v(V, P, 1.0), **pct) for V, pi in vi]
    vi_caps = [f"after round {k}: from the start square the agent now knows it can reach the "
               f"{viz.GOAL} {V[start]:.0%} of the time" for k, (V, _) in zip(sweeps, vi)]

    pi_hist = policy_iteration_history(P, 1.0)
    # Q of the plan being scored: on its last frame these are the optimal values.
    pi_frames = [viz.grid(layout, V, pi, Q=q_from_v(V, P, 1.0), **pct) for V, pi in pi_hist]
    pi_caps = [f"plan {i + 1}: reaches the {viz.GOAL} {V[start]:.0%} of the time from the start"
               + (" (can't improve any further: this is the best plan)" if i == len(pi_hist) - 1 else "")
               for i, (V, _) in enumerate(pi_hist)]

    gammas = []
    for gamma, words in ((1.0, "patient: a point later is as good as a point now"),
                         (0.9, "a point one step later is worth 90%"),
                         (0.5, "impatient: a point one step later is worth half")):
        Vg, pi, _ = value_iteration(P, gamma)
        V = policy_evaluation(pi, P, 1.0)
        gammas.append(viz.grid(layout, V, pi, Q=q_from_v(Vg, P, gamma), caption=f"<b>γ = {gamma}</b>, {words}. "
                               f"This plan reaches the {viz.GOAL} {V[start]:.0%} of the time.", **pct))

    Q_opt = q_from_v(V_opt, P, 1.0)
    opt = lambda s, rng: pi_opt[s]
    go = lambda s, rng: gofor[s]
    return [
        "<p style='font-size:16px'><b>In one sentence:</b> if you know exactly how the world "
        "behaves, you can work out the best move from every square without ever playing.</p>",
        viz.intro(name, "Here the agent has a perfect rulebook: for every square and every move, "
                  "it knows the odds of where it will end up. The question is how to turn that "
                  "rulebook into a plan. <b>No playing happens in this chapter:</b> everything "
                  "is worked out on paper from the odds. Learning by playing starts in chapter 5."),
        viz.legend(),
        viz.grid(layout, caption="the 4x4 lake"),

        "<h3>1. How good is a plan?</h3>",
        viz.row(
            viz.grid(layout, V_go, gofor, caption="<b>go-get-it</b>: head straight for the goal",
                     **pct),
            viz.grid(layout, V_careful, careful, caption="<b>careful</b>: always point at a wall",
                     **pct),
        ),
        viz.explain(
            how=f"A plan is an arrow on every square. The number on a square is the chance of "
                f"reaching the {viz.GOAL} if you start there and follow the arrows. It is computed "
                "exactly from the rulebook (this is <i>policy evaluation</i>), no games played.",
            notice=f"Going straight for the goal works only {V_go[start]:.0%} of the time from the "
                   f"start: the slides keep pushing you into holes. The careful plan gets "
                   f"{V_careful[start]:.0%}."),

        "<h3>2. Why pointing at a wall works: the value of a move is an average</h3>",
        viz.row(viz.decision_card(name, 1, V_opt, title="on the top edge, just above a hole"),
                viz.decision_card(name, 6, V_opt, title="between two holes")),
        viz.explain(
            how="Each row is one move. Because of the ice, a move has three possible results; "
                "the move is worth the average of the squares it might land on. "
                "That averaging rule is the <b>Bellman equation</b>, the idea the whole book is "
                "built on: <code>value of a move = Σ chance × (reward + value of where you land)</code>.",
            notice=f"Above the hole, pointing ↑ <i>into the wall</i> is worth 82% and every other "
                   "move only 55%. Pointing at the wall, the worst the ice can do is bump you in "
                   "place or slide you along the edge; it can never push you down into the hole. "
                   "Between two holes there is no safe move: the best you can do is a 1 in 3 risk."),

        "<h3>3. Finding the best plan: value iteration</h3>",
        viz.slider(vi_frames, vi_caps, "value iteration: press Play", 350),
        viz.explain(
            how="Each round, every square looks at its neighbours' numbers and updates its own "
                "with the averaging rule from section 2, then points its arrow at its best move.",
            notice=f"Round 0 knows nothing (all 0%), so every arrow ties. The squares next to the {viz.GOAL} light up "
                   "first, and the good news spreads backwards one square per round until it "
                   "reaches the start. Keep going and the numbers settle: that is the best "
                   "possible plan."),

        "<h3>4. The other way: policy iteration</h3>",
        viz.slider(pi_frames, pi_caps, "policy iteration: press Play", 1200),
        viz.explain(
            how="Start with random arrows. Score the plan (section 1), then on every square switch "
                "the arrow to whichever move scores best. Repeat.",
            notice=f"It needs only {len(pi_hist)} plans to land on the same answer value iteration "
                   "found, and it ends up pointing at walls, just like the careful plan. Squares "
                   "with several arrows are ties: the start square shows all four because every "
                   "move from it is worth the same 82% (each one either bumps a wall or lands on "
                   "another 82% square), so there it does not matter what you pick. "
                   "Other squares also say 82% but show one arrow: there only that move reaches "
                   "82%, the rest are worse. Hover any square to see all four."),

        "<h3>5. Patience: the discount factor γ (gamma)</h3>",
        viz.row(*gammas),
        viz.explain(
            how="γ says how much the agent cares about later. With γ = 0.5, a point one step away "
                "is worth half a point now, two steps away a quarter, and so on. Each board is the "
                "best plan for that γ; the numbers are its real chance of reaching the goal.",
            notice="The impatient agent takes risky shortcuts past the holes because a safe but "
                   "slow route is worth almost nothing to it."),

        "<h3>6. Watch the best plan play, and what the agent is thinking</h3>"
        "<p>The plan was finished before this game started; the agent is not learning here, "
        "just following its arrows so you can see the plan work (and the ice fight it).</p>",
        viz.replay(name, opt, "best plan", viz.find_seed(name, opt, True),
                   values=V_opt, greedy_policy=pi_opt, Q=Q_opt, gamma=1.0, pct=True),
        viz.replay(name, go, "go-get-it plan", viz.find_seed(name, go, False),
                   values=V_go, greedy_policy=gofor, Q=q_from_v(V_go, P), gamma=1.0, pct=True),
        viz.explain(
            how="Step through with the slider. The line above the board says what the agent "
                "tried and what the ice actually did. The panel shows how it rates every move from "
                "where it stands, and the averaging behind the move it picks.",
            notice="The best plan often stays put (bumping a wall on purpose) and still gets "
                   "there; the go-get-it plan moves faster and falls in."),

        viz.takeaways([
            "With a perfect rulebook (a <i>model</i>), you never need to play to find the best "
            "plan. That is <i>planning</i>, or dynamic programming.",
            "A move is worth the average of where it might land: the Bellman equation.",
            "Value iteration and policy iteration both repeat that averaging until nothing "
            "changes, and both find the same best plan.",
            "On slippery ice the best plan points at walls so slides cannot throw it into holes.",
            "γ is patience; an impatient agent gambles.",
            "Next chapter: what if there is no rulebook, and you have to learn by trying?",
        ]),
    ]

if __name__ == "__main__":
    main()
