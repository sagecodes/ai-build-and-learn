---
title: Reinforcement Learning
first_seen: topics/openenv/
weeks: [openenv]
---

A training paradigm where an agent learns by interacting with an environment:
observe state → take action → receive reward → update policy. The agent
maximizes cumulative reward over time without being told explicitly what to do.

Key vocabulary:
- **Episode** — one complete run from start to terminal state
- **Step** — one action-observation-reward cycle within an episode
- **Reward** — scalar signal telling the agent how good its last action was
- **Policy** — the agent's strategy (what action to take given a state)
- **Exploration vs exploitation** — trying new things (explore) vs using what
  you know works (exploit). Epsilon controls this tradeoff in DQN.

## How it appeared across the series

### Week 3 — OpenEnv (2026-04-10)

Two RL algorithms demonstrated on the same maze environment:

**DQN (Deep Q-Network) — classic RL.**
Learns a Q-function: Q(state, action) = expected future reward. At each step,
picks the action with the highest Q-value. Key mechanisms: replay buffer (breaks
correlation between consecutive steps), target network (frozen copy of Q-network
for stable Bellman targets), epsilon decay (1.0 → 0.05, random → learned).
Solves a fixed 12x12 maze in ~40 training steps (~1,280 total episodes).
Natural fit for discrete actions and dense reward shaping.

**GRPO (Group Relative Policy Optimization) — LLM fine-tuning.**
Trains SmolLM2-135M to output actions as text ("UP", "DOWN", "LEFT", "RIGHT").
Groups episodes, computes relative advantages within each group (above-average
reward → positive advantage), updates generation probabilities. Same family as
RLHF. Slower than DQN for spatial tasks; shines when the task benefits from
language reasoning (code generation, tool use, text-based games).

**The key lesson:** RL algorithm choice depends on the task. Classic RL (DQN/PPO)
is faster and more efficient for numeric/spatial problems. LLM-based RL (GRPO/
RLHF) is better when observations are language and actions require reasoning.
OpenEnv supports both with the same environment API.

### Week 4 — AutoResearch (2026-04-17)

Autoresearch is RL applied to ML research. The mapping is direct:
- **State** — current `train.py` + recent `results.tsv`
- **Action** — one proposed code change
- **Reward** — `val_bpb_before - val_bpb_after` (positive = improvement)
- **Policy** — the agent's strategy, shaped by `program.md`
- **Episode** — one propose → train → evaluate cycle

Key difference from classic RL: the "episode" takes 5 real-world minutes. The
agent cannot explore millions of states — it must propose good changes based on
prior results logged to `results.tsv`. The context window is the replay buffer.

The ~21% success rate (1 in 5 changes improves val_bpb) matches the expected
difficulty of random-walk hill-climbing near a local minimum. Both T4 overnight
runs reached this rate independently, suggesting it's a property of the problem
space rather than a particular agent.

## Open questions

- How does GRPO compare to PPO for LLM fine-tuning in production?
- Does the series revisit RL training with larger models or GPU resources?
  (AutoResearch week may touch this.)
