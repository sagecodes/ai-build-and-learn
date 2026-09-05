---
title: OpenEnv
weeks: [openenv]
---

Meta's framework for building, deploying, and interacting with isolated
execution environments for agentic reinforcement learning. Gymnasium API
(`reset()`, `step()`, `state()`) over HTTP/WebSocket — the same interface
works whether the environment is local, in Docker, or on a remote cluster.

Key properties vs Gymnasium:
- **Client-server** — environment runs as a microservice, agent connects over WebSocket
- **Docker-isolated** — each episode can run in its own container
- **Concurrent sessions** — one container can host multiple isolated sessions
- **LLM-native** — built-in `AnthropicClient`, `MCPClient`, tool discovery
- **Type-safe** — Pydantic models for Action, Observation, State

Three components to define a custom environment:
1. **Models** — Pydantic `Action`, `Observation`, `State` types
2. **Environment** — `reset()`, `step()`, `state` property
3. **Server** — one line: `create_app(MyEnv, MyAction, MyObservation)`

## Usage across the series

### Week 3 — OpenEnv (2026-04-10)

Used in all three sub-projects:

**Maze RL** — Custom `MazeEnvironment` with shaped rewards, shared between a
DQN agent and a GRPO LLM agent. Demonstrates the "same environment, different
agents" pattern that is OpenEnv's core value proposition.

**Atari** — Prebuilt ALE-backed environment (Pong, Breakout, etc.) used with
zero custom environment code. Shows OpenEnv ships a library of ready-made
environments alongside the custom environment API.

**Research Agent** — `ResearchEnvironment` with Tavily tools as actions.
`SUPPORTS_CONCURRENT_SESSIONS = True` allows one Docker container to host
multiple isolated agent sessions. `GenericEnvClient` connects over HTTP.
Each Flyte task pod starts its own local OpenEnv server on a random port.
