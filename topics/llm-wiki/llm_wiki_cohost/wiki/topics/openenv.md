---
title: OpenEnv — Environments for RL Training
date: 2026-04-10
folder: topics/openenv/
concepts: [reinforcement-learning, reward-functions, agents, research-pipelines]
tools: [openenv, flyte, gradio, tavily]
---

Three sub-projects exploring OpenEnv (Meta's production-ready RL environment
framework) across two paradigms: classic RL and LLM-as-agent. The week's
central demo proves that keyword-match reward functions are gameable, and LLM-
as-judge is the honest alternative.

## What was built

**`maze-rl/`** — A custom 12x12 maze environment with two completely different
agents trained on it: a DQN neural network (classic RL, learns in ~40 steps)
and SmolLM2-135M fine-tuned with GRPO (LLM-as-agent). Same environment,
different training algorithms. Both orchestrated as Flyte pipelines with
visual reports.

**`atari/`** — A minimal demo using a prebuilt OpenEnv Atari environment (Pong,
Breakout, Space Invaders). Shows that OpenEnv ships ready-made environments
alongside custom ones. Same `reset()`/`step()` API regardless.

**`openenv_research_agent/`** — The main demo. A research environment where
two agents compete side-by-side: a traditional agent (fixed actions, keyword
reward) vs a Claude ReAct agent (dynamic tool discovery, LLM-as-judge reward).
The traditional agent games the keyword score while earning a low LLM quality
score — reward hacking made visible in real time.

## Key decisions

**Same environment, different agents.** The maze environment (`maze_env/`) is
identical for DQN and GRPO. OpenEnv's clean `reset()`/`step()` API separates
environment from agent — this is the core architectural value of the framework.

**LLM-as-judge reward.** The research agent demo makes reward hacking
observable: the traditional agent scores 9/10 on keyword count while producing
garbage. Claude rates it honestly. The contrast is the demo's thesis.

**Docker isolation for episodes.** OpenEnv runs environments as HTTP/WebSocket
servers in containers. `SUPPORTS_CONCURRENT_SESSIONS = True` lets one container
host multiple isolated agent sessions simultaneously. `GenericEnvClient.from_docker_image()` is the heavier, fully-isolated alternative.

**Flyte result caching.** Parallel research tasks are cached by `(query, agent_type, max_steps)`. Running the same query twice returns instantly — shown
as a demo feature in Tab 3.

## Connections

- [Reinforcement Learning](../concepts/reinforcement-learning.md) — DQN, GRPO, episodes, rewards
- [Reward Functions](../concepts/reward-functions.md) — keyword reward vs LLM-as-judge, reward hacking
- [Agents](../concepts/agents.md) — traditional fixed-policy vs Claude ReAct agent
- [Research Pipelines](../concepts/research-pipelines.md) — parallel fan-out across research questions
- [OpenEnv](../tools/openenv.md) — the environment framework
- [Flyte / Union](../tools/flyte.md) — parallel task fan-out, result caching
- [Gradio](../tools/gradio.md) — multi-tab UI with live charts
- [Tavily](../tools/tavily.md) — web research tools inside the environment
