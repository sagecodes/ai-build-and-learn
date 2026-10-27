# Chapter 3: Planning with a model

**Book chapter:** *Balancing immediate and long-term goals*
**Run:** `flyte run pipeline.py ch03_planning`  ·  locally `python -m ch03_planning.planning`

## The core idea

If you know `P` (the transition table from chapter 2), you can compute how good a policy π (pi, the rule for which action to take in each state) is without ever running it. Computing answers from a known model like this is called **dynamic programming** (DP). Everything rests on the **Bellman equation**: the value V(s) of a state is the expected immediate reward plus the discounted value of wherever you land. V^π is that value when following π; V\* is the value under the best possible policy.

Acronyms and symbols are all in the [glossary](../README.md#glossary).

```
V^π(s)  =  Σ_s'  P(s' | s, π(s)) · [ r + γ · V^π(s') ]          (evaluate a policy)

V*(s)   =  max_a  Σ_s'  P(s' | s, a) · [ r + γ · V*(s') ]       (the best possible)
```

One backup, drawn for a frozen-lake cell where the agent chooses "right":

```
                 (s, →)
          ┌────────┼────────┐
        1/3       1/3      1/3          the ice: as chosen, or either side
         ▼         ▼        ▼
       s_up     s_right   s_down
     r+γV(s')  r+γV(s')  r+γV(s')      Q(s,→) = average of the three
```

Turn that equation into an update and sweep it over every state until nothing changes, and you have the chapter's three algorithms:

| Algorithm | Loop |
|---|---|
| **Policy evaluation** | apply the V^π backup to every state, repeat until it stops changing |
| **Policy iteration** | evaluate π fully → make π greedy on those values → repeat until π is stable |
| **Value iteration** | apply the V* backup (with the max) every sweep; evaluation and improvement fused into one |

## How the code does it

[`planning.py`](planning.py) converts `P` into two arrays once:

```python
R[s, a]     = expected immediate reward
T[s, a, s'] = probability of continuing into s'   (episode-ending transitions left out)
```

so a full Bellman backup over every state at once is one line, `Q = R + gamma * T @ V`, and leaving terminal transitions out of `T` is what builds `V(terminal) = 0` in. Policy evaluation, policy iteration and value iteration are each a few lines on top of that. Later chapters import `policy_evaluation` and `value_iteration` from here as their answer key.

Policy iteration stops when the greedy policy's *values* stop improving, not when the actions stop changing. Two actions with equal value can flip forever without anything getting better.

## What it shows

```
FL4 (frozen lake 4x4), P(reach the goal) from the start (γ = 1):

   go-get-it policy (straight for the goal)   0.037
   careful policy (hug the walls)             0.824
   optimal (policy iteration)                 0.824

   policy iteration: 6 rounds      value iteration: 806 sweeps      same answer
```

The careful policy is optimal: choose a direction that points *into a wall*, so the two slip outcomes either bump you back in place or slide you along it, and never into a hole. On FL4 the best any policy can do is 82.4%.

**γ changes the answer, not just the numbers.** Solving the same lake with less patience gives policies that take risks:

```
   γ = 1.0   →  reaches the goal 82.4%
   γ = 0.99  →  82.4%
   γ = 0.9   →  78.0%
   γ = 0.5   →  45.1%    (a reward 6 steps away is worth 1/64 now, so it gambles)
```

The sampled success rate comes out a bit lower than the exact one (0.75 vs 0.82) because sampled episodes are cut off at 100 steps; the careful policy is safe but slow.

## In the report

Each square shows the chance of reaching the 🎁 from there, and hovering a square shows all four moves' values. Squares where several moves tie show every tied arrow. The sections:

1. **How good is a plan?** go-get-it (4%) vs careful (82%).
2. **Why pointing at a wall works.** Decision cards list every move from a square and where the ice can send it: the Bellman equation drawn as a table.
3. **Value iteration**, round by round: good news spreads back from the goal.
4. **Policy iteration**, plan by plan.
5. **Patience (γ)**: the impatient agent's plans take risky shortcuts.
6. **The best plan playing**, with the agent's reasoning on every step in plain words.
