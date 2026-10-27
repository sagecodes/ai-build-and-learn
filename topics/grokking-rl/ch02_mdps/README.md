# Chapter 2: Mathematical foundations of RL

**Book chapter:** *Mathematical foundations of reinforcement learning*
**Run:** `flyte run pipeline.py ch02_mdps`  ·  locally `python -m ch02_mdps.mdps`

## The core idea

A reinforcement learning (RL) problem is an **MDP**, a *Markov decision process*: a set of states, a set of actions, and a transition function that says, for every (state, action), what can happen next and how likely it is. Add a reward on each transition and a discount factor γ (gamma, between 0 and 1), and the problem is fully specified.

"Markov" is the key assumption: what happens next depends only on the current state and action, not on how you got there. That is what lets a single table describe the whole world.

Acronyms and symbols are all in the [glossary](../README.md#glossary).

```
          agent
       ┌─────────┐   action a_t
       │ policy  │ ───────────────┐
       └─────────┘                ▼
          ▲   ▲            ┌─────────────┐
   state  │   │ reward     │ environment │   P(s', r | s, a)
   s_t+1  │   │ r_t+1      │  (the MDP)  │
          └───┴────────────┴─────────────┘
```

Three things in this chapter are worth seeing with your own eyes rather than taking on faith:

1. **The transition function is just a table.** Nothing more mysterious than `P[s][a] = [(p, s', r, done), ...]`.
2. **Stochasticity is real.** On the slippery walk, choosing "right" moves you right only half the time. The agent's choice and the outcome are different things.
3. **The return G depends on γ.** `G = r_1 + γ r_2 + γ² r_3 + ...`. With γ = 1 a reward is worth the same whenever it arrives; with γ < 1, a reward that takes longer to reach is worth less.

## The worlds

```
BW  (bandit walk)            H [1] G     deterministic: you go where you choose
BSW (bandit slippery walk)   H [1] G     80% as chosen, 20% the opposite way
SWF (slippery walk five)     H [1][2][3][4][5] G     50% as chosen, 33% stay, 17% backwards
FL4 (frozen lake 4x4)        S . . .     1/3 as chosen, 1/3 each perpendicular
                             . H . H
                             . . . H
                             H . . G
```

H = hole (episode ends, reward 0), G = goal (episode ends, reward +1), S = start, [n] = an ordinary cell.

## How the code does it

- [`envs.py`](../envs.py) `walk()` builds any corridor from three numbers (p forward, p stay, p backward). `frozen_lake()` mirrors Gym's slip rule. Both return the plain `P` dict.
- `TabularEnv.step()` samples from `P[s][a]` with one uniform draw against the cumulative probabilities. This is the only way chapters 4 to 7 are allowed to touch the world.
- [`mdps.py`](mdps.py) prints the tables, then samples 100,000 transitions from one (state, action) and puts the empirical frequencies next to the table's probabilities, then runs two policies on SWF and scores the same episodes under three discount factors.

## What it shows

```
SWF from s3, action R (100,000 samples):     FL4 from s0, action > (100,000 samples):
  s2: table 0.167  sampled 0.165               s0: table 0.333  sampled 0.334
  s3: table 0.333  sampled 0.334               s1: table 0.333  sampled 0.335
  s4: table 0.500  sampled 0.501               s4: table 0.333  sampled 0.332

               mean length   G(γ=1.0)   G(γ=0.99)   G(γ=0.9)
      random      13.5 steps    0.500      0.443       0.192
always right       8.3 steps    0.966      0.899       0.507
```

At γ = 1 the return only asks *whether* you arrive. At γ = 0.9 it also asks *how fast*, and the slow random policy loses 62% of its return to discounting while "always right" loses 48%.

## In the report

Written to make sense without the book open: a story card for each world, a picture of how the ice works, and then:

1. **The worlds**, drawn with icons (🤖 agent, 🎁 goal, 🕳️ hole).
2. **The dice experiment**: the MDP table's odds next to what 100,000 real tries did, as bars.
3. **Random agents playing**, with every step captioned ("tried ↑, but the ice slides it → instead").
4. **The discount table**, explaining γ as patience.
