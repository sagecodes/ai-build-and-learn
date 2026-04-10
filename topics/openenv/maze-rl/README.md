# Reinforcement Learning with OpenEnv — Maze Demo

Train agents to navigate mazes using [OpenEnv](https://github.com/meta-pytorch/OpenEnv), demonstrating that the same environment works with completely different training approaches — a DQN neural network and an LLM fine-tuned with GRPO.

Both pipelines are orchestrated with [Flyte](https://flyte.org/) and generate visual reports with interactive maze replays.

## The Maze Environment

- **12x12** randomly generated maze (DFS recursive backtracker + loop creation)
- `#` wall, `.` open path, `A` agent, `E` exit
- Shaped rewards: +10 solve, +0.1 closer, -0.1 farther, -0.2 revisit, -0.3 wall hit
- Max 100 steps per episode

## Setup

```bash
cd topics/openenv/maze-rl

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify GPU: `python -c "import torch; print(torch.cuda.is_available())"`

## Two Agents, Same Environment

### DQN Agent (`maze_rl_dqn.py`)

A small neural network trained with [Deep Q-Network (DQN)](https://arxiv.org/abs/1312.5602). Learns to solve a fixed maze in ~40 training steps, going from 0% to 100% solve rate.

```bash
# Run with defaults (40 steps, 32 episodes/step, seed=42)
flyte run --local maze_rl_dqn.py pipeline

# Random mazes (harder — needs more training)
flyte run --local maze_rl_dqn.py pipeline --training_steps 100 --maze_seed null
```

#### Key terminology

- **Episode** — one complete attempt at solving the maze. The agent starts at `A`, takes actions until it reaches `E` (solved) or hits the step limit (failed). Each episode produces a sequence of (state, action, reward) transitions that get stored in the replay buffer.
- **Training step** — one round of the training loop. During each step, the agent plays `episodes_per_step` episodes to collect experience, then trains the neural network on random batches sampled from the replay buffer. Early steps use high epsilon (mostly random exploration); later steps use low epsilon (mostly learned policy).
- **Epsilon** — the exploration rate. With epsilon=1.0 the agent picks random actions (exploring). With epsilon=0.05 it almost always picks the action with the highest Q-value (exploiting what it learned). Epsilon decays linearly from `epsilon_start` to `epsilon_end` over the training steps.

So with the defaults (40 steps, 32 episodes/step), the agent plays **1,280 total episodes** across training — each one generating dozens of transitions that teach the network which actions lead to good outcomes.

#### Why DQN works well here

DQN is a natural fit for maze navigation because the problem has the exact properties it was designed for:

- **Discrete actions** — only 4 choices (UP, DOWN, LEFT, RIGHT). DQN learns a Q-value for each action and picks the best one. No need for continuous action spaces.
- **Small, fixed observation** — the 12x12 grid flattens to 148 numbers (144 grid cells + 4 position values). A 2-layer MLP can process this in microseconds, so the agent plays thousands of episodes quickly.
- **Dense reward shaping** — the environment gives +0.1 for moving closer and -0.3 for hitting walls, so the Q-values get meaningful gradient signal on every step, not just at the end.
- **Replay buffer** — DQN stores past experiences and trains on random batches, which breaks correlation between consecutive steps and makes learning stable.
- **Target network** — a frozen copy of the Q-network provides stable targets for the Bellman equation, preventing the "moving goalpost" problem.

The agent learns a function Q(state, action) that estimates "how much total future reward will I get if I take this action?" After enough training, it simply picks the action with the highest Q-value at each step.

#### Why the LLM struggles with the same task

The LLM approach (GRPO on SmolLM2-135M) faces several disadvantages:

- **Observation format** — the grid must be rendered as text, making the input hundreds of tokens instead of 68 numbers. The model spends capacity just parsing the format.
- **Output format** — instead of directly outputting 1 of 4 actions, the LLM generates free-form text that must be parsed for a direction. It often produces "To find the shortest path from A to..." instead of just "RIGHT".
- **No replay buffer** — GRPO is an on-policy method. It can only learn from episodes it just played, so it needs far more rollouts to see the same amount of experience.
- **Model size** — 135M parameters is massive overkill for choosing between 4 directions, but simultaneously too small to develop real spatial reasoning from text.

The DQN solves the maze in minutes. The LLM would need orders of magnitude more training to match it. But that's the point — **OpenEnv doesn't care**. Same environment, same API, different agents.

| Parameter | Default | Description |
|---|---|---|
| `training_steps` | `40` | Number of DQN training iterations |
| `episodes_per_step` | `32` | Episodes collected per step |
| `eval_episodes` | `20` | Episodes for each evaluation |
| `lr` | `1e-3` | Learning rate |
| `gamma` | `0.99` | Discount factor |
| `epsilon_start` | `1.0` | Initial exploration rate |
| `epsilon_end` | `0.05` | Final exploration rate |
| `maze_seed` | `42` | Fixed maze seed (set to `null` for random) |

### LLM Agent (`maze_rl_llm.py`)

SmolLM2-135M fine-tuned with GRPO (Group Relative Policy Optimization). Shows how OpenEnv works with LLM training — needs more compute to see improvement.

```bash
# Quick run
flyte run --local maze_rl_llm.py pipeline --training_steps 5

# Longer training
flyte run --local maze_rl_llm.py pipeline --training_steps 30 --rollouts_per_step 16
```

## Architecture

```
maze_rl_dqn.py      # DQN Flyte pipeline (fast learner)
maze_rl_llm.py      # LLM GRPO Flyte pipeline

maze_env/           # Shared OpenEnv environment
  models.py         # Pydantic models (Action, Observation, State)
  server/
    environment.py  # MazeEnvironment class
    app.py          # FastAPI server (for standalone use)
```

The key point: **`maze_env/` is identical for both agents.** OpenEnv provides the environment API — the agent and training algorithm are up to you.
