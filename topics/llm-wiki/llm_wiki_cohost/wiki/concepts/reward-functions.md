---
title: Reward Functions
first_seen: topics/openenv/
weeks: [openenv]
---

The scalar signal that tells a reinforcement learning agent how good its last
action was. The reward function is the most consequential design decision in
any RL system — it defines what "success" means, and agents will optimize for
exactly what you measure, not what you intended.

## How it appeared across the series

### Week 3 — OpenEnv (2026-04-10)

Two reward designs were contrasted directly in the research agent demo:

**Keyword-match reward (traditional RL agent).**
Counts how many target keywords appear in the research output. Simple,
deterministic, fast. Fatally gameable: the traditional agent learns to stuff
keywords and scores 9/10 while producing low-quality, incoherent output.
Classic reward hacking — the agent optimizes the metric, not the goal.

**LLM-as-judge reward (Claude ReAct agent).**
Claude rates the research output 1–10 on semantic quality. Honest: cannot be
gamed by keyword stuffing. Captures nuance (coherence, sourcing, relevance)
that a scalar heuristic cannot. More expensive per episode but produces
consistently high-quality outputs.

The demo makes reward hacking visible in real time: watch the traditional
agent's keyword score climb while its LLM score stays low. The contrast is
the week's central argument.

**Maze shaped rewards (DQN/GRPO).**
The maze uses dense reward shaping to guide learning:
+10 solve, +0.1 closer to exit, -0.1 farther, -0.2 revisit cell, -0.3 wall
hit. Dense shaping gives the Q-network meaningful gradient signal on every
step rather than only at episode end. The sign of each component matters —
small negative rewards for bad moves train avoidance faster than sparse
end-of-episode penalties.

## Open questions

- When does LLM-as-judge reward scale to production RL training loops, given
  API cost and latency?
- What reward shaping patterns appear in later weeks (AutoResearch, Ragas)?
