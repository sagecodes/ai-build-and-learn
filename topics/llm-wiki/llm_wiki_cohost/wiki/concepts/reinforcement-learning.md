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

## Open questions

- How does GRPO compare to PPO for LLM fine-tuning in production?
- Does the series revisit RL training with larger models or GPU resources?
  (AutoResearch week may touch this.)
