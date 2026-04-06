# OpenEnv Research Agent

A demo project showcasing [OpenEnv](https://github.com/meta-pytorch/OpenEnv) — Meta's framework for isolated agentic RL environments — combined with [Flyte](https://flyte.org/) for compute orchestration, [Tavily](https://tavily.com/) for web research tools, and Anthropic's Claude as the agent brain.

The demo proves a core point: **traditional RL reward functions fail for language tasks**. A keyword-match reward is easily gamed — an agent can score 9/10 while producing garbage. An LLM-as-judge reward is honest. This project shows both, side by side, in real time.

---

## What is OpenEnv?

[OpenEnv](https://github.com/meta-pytorch/OpenEnv) is a unified framework for building, deploying, and interacting with isolated execution environments for agentic reinforcement learning. Unlike OpenAI Gym (local, in-process, discrete), OpenEnv is:

- **HTTP/WebSocket-based** — environments run as microservices, not in-process
- **Docker-isolated** — environments run inside containers; agents connect via `GenericEnvClient`
- **LLM-native** — built-in `AnthropicClient`, `MCPClient`, tool discovery
- **Concurrent** — one container can host multiple independent sessions simultaneously
- **Minimal API** — just `reset()`, `step()`, `state`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Gradio UI (app.py)                   │
│   Tab 1: Side-by-Side  │  Tab 2: Agent Race  │  Tab 3: Flyte│
└────────────┬───────────────────┬─────────────────┬──────────┘
             │                   │                 │
    ┌────────▼────────┐ ┌────────▼───────┐  ┌──────▼──────────┐
    │ TraditionalAgent│ │  OpenEnvAgent  │  │  Flyte Cluster  │
    │  fixed actions  │ │  Claude ReAct  │  │  parallel tasks │
    │  keyword reward │ │  LLM judge     │  │  cached results │
    └────────┬────────┘ └────────┬───────┘  └──────┬──────────┘
             │  GenericEnvClient │  GenericEnvClient│
             │  HTTP/WebSocket   │  HTTP/WebSocket  │
             └───────────────────┴─────────────────┘
                                 │
                    ┌────────────▼────────────────────────┐
                    │   Docker Container (port 8000)      │
                    │   OpenEnv HTTP server               │
                    │                                     │
                    │   Session 1: TraditionalAgent ep.   │
                    │   Session 2: OpenEnvAgent ep.       │
                    │   Session 3+: Race agents ...       │
                    │                                     │
                    │   Each session = isolated           │
                    │   ResearchEnvironment instance      │
                    │   ├── tavily_search                 │
                    │   ├── tavily_extract                │
                    │   └── tavily_crawl                  │
                    └─────────────────────────────────────┘
```

> **One container, many sessions.** Agents do not each get their own container in this demo. Instead, `ResearchEnvironment` sets `SUPPORTS_CONCURRENT_SESSIONS = True` and the server is configured with `max_concurrent_envs=10`. Each agent connects via `GenericEnvClient(base_url="http://localhost:8000").sync()` and gets its own isolated session (its own episode state, step counter, and tool usage) inside the same running container. OpenEnv also supports spinning up a fresh container per agent via `GenericEnvClient.from_docker_image()` — that's the heavier, fully-isolated alternative.

### The Two Agents

| | Traditional RL Agent | OpenEnv Agent |
|---|---|---|
| Action space | Fixed — `tavily_search` only | Dynamic — discovers all 3 tools |
| Strategy | Keyword stuffing | Reason → search → extract → crawl |
| Reward | Keyword match count | LLM-as-judge (Claude rates 1-10) |
| Environment | Isolated session in Docker container | Isolated session in Docker container |
| Connection | `GenericEnvClient` via HTTP | `GenericEnvClient` via HTTP |
| Result | High keyword score, low LLM score | Consistently high LLM score |

---

## Project Structure

```
topics/openenv/openenv_research_agent/
├── env/
│   ├── tools/
│   │   ├── search.py        # tavily_search action
│   │   ├── extract.py       # tavily_extract action
│   │   └── crawl.py         # tavily_crawl action
│   ├── research_env.py      # OpenEnv Environment — step/reset/state
│   ├── models.py            # Pydantic Action/Observation/State types
│   └── Dockerfile           # Container for episode isolation
├── agents/
│   ├── openenv_agent.py     # Claude agent via Anthropic SDK
│   └── traditional_agent.py # Fixed-policy keyword-stuffing agent
├── reward.py                # keyword_reward, keyword_reward_with_detail, llm_judge_final_reward
├── system_prompt.py         # Claude's research instructions
├── workflow.py              # Flyte tasks + parallel pipeline
├── config.py                # Flyte TaskEnvironment + secrets
├── app.py                   # Gradio UI (3 demo tabs)
├── requirements.txt         # Full project dependencies
└── requirements_env.txt     # Docker container dependencies only
```

---

## Prerequisites

- Python 3.11+
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) — required for OpenEnv container isolation
- A Flyte cluster — required for remote mode only; local mode works without one
- API keys: Anthropic, Tavily

---

## Installation

```bash
cd topics/openenv/openenv_research_agent
python -m venv .venv

# macOS/Linux
source .venv/bin/activate

# Windows
source .venv/Scripts/activate

pip install -r requirements.txt
```

Create a `.env` file in the project directory:

```
ANTHROPIC_API_KEY=your-anthropic-key-here
TAVILY_API_KEY=your-tavily-key-here
```

Build the OpenEnv Docker container:

```bash
docker build -t research-env:latest -f env/Dockerfile .
```

---

## Running the Demo

### Run Modes

| Mode | Command | Flyte | Docker |
|---|---|---|---|
| Fully local | `RUN_MODE=local python app.py` | Local process | Local Docker |
| Local UI + remote tasks | `python app.py` | Remote cluster | Cluster |
| Fully remote | `flyte deploy app.py serving_env` | Remote cluster | Cluster |

### Local (no cluster needed)

```bash
RUN_MODE=local python app.py
```

Open `http://localhost:7860` in your browser.

### Remote (Flyte cluster)

```bash
flyte init_from_config   # configure cluster connection once
python app.py
```

---

## Demo Tabs

### Tab 1 — Side-by-Side Comparison

Enter a research question. Both agents run simultaneously. Watch:

- The traditional agent's **keyword score climb** while its **LLM score stays low** — reward hacking in action
- The OpenEnv agent chain tools intelligently and earn a **consistently high LLM score**

The live Plotly chart updates after every step so the gap is visible in real time.

### Tab 2 — Agent Race

Three OpenEnv agents race on the same question using OpenEnv's `SUPPORTS_CONCURRENT_SESSIONS`. A live scoreboard updates after each step. First agent to finish wins. All three agents connect to the **same Docker container** — each gets its own isolated session with independent episode state. No interference between sessions despite sharing one process.

### Tab 3 — Parallel Flyte Fan-out

Enter multiple research questions (one per line). Each dispatches as a parallel Flyte task — both agents run per question. A Flyte run link appears immediately. Results stream in as tasks complete. Run the same question twice to see Flyte's **result cache** return instantly.

Each Flyte task pod starts its own local OpenEnv HTTP server on a random port, then agents connect to it via `GenericEnvClient` — the same code path used in local Docker development. Flyte provides the task isolation; OpenEnv provides the HTTP interface inside each pod.

---

## Key Concepts

### Why OpenEnv beats traditional RL for language tasks

Traditional RL requires a fixed, discrete action space and a scalar reward. For open-ended language tasks this breaks down:

- You can't enumerate all possible queries as discrete actions
- Keyword-count rewards are trivially gameable
- Episodes can't be isolated from each other without significant infrastructure

OpenEnv solves all three:

- Actions are natural language tool calls, discovered dynamically via MCP
- Rewards are computed by an LLM judge — semantic quality, not keyword frequency
- Each episode runs in an isolated Docker container

### Flyte's role

Flyte is the compute orchestrator. It operates independently from OpenEnv — it doesn't know or care what's running inside its tasks.

In this project Flyte:

- Fans out multiple research tasks in parallel across a cluster
- Caches results by `(query, agent_type, max_steps)` — identical runs return instantly
- Generates per-task HTML reports with step logs and reward data
- Returns run links to the Gradio UI for live observability

### How the layers fit together

```
Your PC
├── Gradio UI          — always local
├── Flyte TUI          — watches remote jobs
└── Docker Desktop     — local dev and testing only

Remote Flyte Cluster
├── Task: research("quantum computing", openenv)
│   └── OpenEnv Docker container  ← runs here
├── Task: research("quantum computing", traditional)
│   └── OpenEnv Docker container  ← runs here
└── ...
```

---

## API Keys

| Key | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) |
