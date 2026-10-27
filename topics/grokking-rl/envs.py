"""The small tabular MDPs (Markov decision processes) the book uses, as plain transition tables.

Every environment is a dict with the same shape as Gym's `env.unwrapped.P`:

    P[state][action] = [(probability, next_state, reward, done), ...]

Having the full table is what makes chapters 2-3 possible (planning needs the
model). Chapters 4-7 pretend they cannot see it and only sample from it through
`TabularEnv`, which is exactly the model-free setting.
"""

import numpy as np

LEFT, RIGHT = 0, 1


def walk(n_states, p_forward, p_stay, p_backward):
    """A 1-D corridor: hole at state 0, goal at the far right, +1 for reaching it.

    n_states counts the non-terminal cells; the corridor has n_states + 2 cells.
    """
    nS = n_states + 2
    goal = nS - 1
    P = {}
    for s in range(nS):
        P[s] = {}
        for a in (LEFT, RIGHT):
            if s in (0, goal):
                P[s][a] = [(1.0, s, 0.0, True)]
                continue
            step = -1 if a == LEFT else 1
            outcomes = [(p_forward, s + step), (p_stay, s), (p_backward, s - step)]
            P[s][a] = [
                (p, s2, float(s2 == goal), s2 in (0, goal))
                for p, s2 in outcomes
                if p > 0
            ]
    return P


# Chapter 2: the bandit walk (BW) is deterministic, the bandit slippery walk (BSW) is not.
BW = walk(1, 1.0, 0.0, 0.0)
BSW = walk(1, 0.8, 0.0, 0.2)
# Slippery walk five/seven (SWF/SWS): 50% intended, 33.3% stay put, 16.7% go backwards.
SWF = walk(5, 0.5, 1 / 3, 1 / 6)
SWS = walk(7, 0.5, 1 / 3, 1 / 6)
# Chapter 5's random walk (RW): the action does nothing, it is a coin flip each step.
RW = walk(5, 0.5, 0.0, 0.5)


# Frozen lake 4x4 / 8x8 (FL4/FL8). S start, F frozen (safe), H hole, G goal.
FL4 = ["SFFF", "FHFH", "FFFH", "HFFG"]
FL8 = [
    "SFFFFFFF",
    "FFFFFFFF",
    "FFFHFFFF",
    "FFFFFHFF",
    "FFFHFFFF",
    "FHHFFFHF",
    "FHFFHFHF",
    "FFFHFFFG",
]
# Frozen lake action order matches Gym: left, down, right, up.
LAKE_MOVES = [(0, -1), (1, 0), (0, 1), (-1, 0)]
LAKE_ARROWS = "<v>^"


def frozen_lake(desc=FL4, slippery=True):
    """Gym's FrozenLake: on ice you go where you meant 1/3 of the time,
    and slide to either perpendicular direction the other 2/3."""
    rows, cols = len(desc), len(desc[0])
    P = {}
    for r in range(rows):
        for c in range(cols):
            s = r * cols + c
            P[s] = {}
            for a in range(4):
                if desc[r][c] in "HG":
                    P[s][a] = [(1.0, s, 0.0, True)]
                    continue
                dirs = [(a - 1) % 4, a, (a + 1) % 4] if slippery else [a]
                P[s][a] = []
                for d in dirs:
                    dr, dc = LAKE_MOVES[d]
                    nr = min(max(r + dr, 0), rows - 1)
                    nc = min(max(c + dc, 0), cols - 1)
                    cell = desc[nr][nc]
                    P[s][a].append(
                        (1 / len(dirs), nr * cols + nc, float(cell == "G"), cell in "HG")
                    )
    return P


FL4_P = frozen_lake(FL4)
FL8_P = frozen_lake(FL8)

# name -> (P, start state, step limit, grid layout or None for corridors)
ENVS = {
    "bw": (BW, 1, 100, None),
    "bsw": (BSW, 1, 100, None),
    "swf": (SWF, 3, 100, None),
    "sws": (SWS, 4, 100, None),
    "rw": (RW, 3, 100, None),
    "fl4": (FL4_P, 0, 100, FL4),
    "fl8": (FL8_P, 0, 200, FL8),
}


class TabularEnv:
    """Samples from P without exposing it: the agent only sees (s', r, done)."""

    def __init__(self, P, start, max_steps=100, seed=None):
        self.P = P
        self.start = start
        self.max_steps = max_steps
        self.nS = len(P)
        self.nA = len(P[0])
        self.rng = np.random.default_rng(seed)
        self._cum = {
            s: {a: np.cumsum([t[0] for t in P[s][a]]) for a in P[s]} for s in P
        }

    def reset(self):
        self.s = self.start
        self.t = 0
        return self.s

    def step(self, a):
        """Returns (next_state, reward, terminated, truncated)."""
        cum = self._cum[self.s][a]
        i = min(int(np.searchsorted(cum, self.rng.random() * cum[-1])), len(cum) - 1)
        _, s2, r, done = self.P[self.s][a][i]
        self.s = s2
        self.t += 1
        return s2, r, done, (not done) and self.t >= self.max_steps


def make(name, seed=None):
    P, start, max_steps, _ = ENVS[name]
    return TabularEnv(P, start, max_steps, seed)


def show(values, layout=None, policy=None, fmt="{:6.3f}"):
    """Text rendering of a value function (and optionally a policy) on the env's shape."""
    if layout is None:
        cells = [fmt.format(v) for v in values]
        if policy is not None:
            cells = [
                f"{c} {'<>'[policy[s]]}" if 0 < s < len(values) - 1 else f"{c}  "
                for s, c in enumerate(cells)
            ]
        return " | ".join(cells)
    cols = len(layout[0])
    width = len(fmt.format(0.0)) + (2 if policy is not None else 0)
    lines = []
    for r, row in enumerate(layout):
        cells = []
        for c, cell in enumerate(row):
            s = r * cols + c
            if cell in "HG":
                cells.append(("HOLE" if cell == "H" else "GOAL").center(width) + " ")
            else:
                arrow = f" {LAKE_ARROWS[policy[s]]}" if policy is not None else ""
                cells.append(fmt.format(values[s]) + arrow + " ")
        lines.append("".join(cells))
    return "\n".join(lines)
