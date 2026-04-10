# Reinforcement Learning with OpenEnv — Maze Demo

Train agents to navigate mazes using [OpenEnv](https://github.com/meta-pytorch/OpenEnv), demonstrating that the same environment works with completely different training approaches — a DQN neural network and an LLM fine-tuned with GRPO.

Both pipelines are orchestrated with [Flyte](https://flyte.org/) and generate visual reports with interactive maze replays.

## The Maze Environment

- **12x12** randomly generated maze (DFS recursive backtracker + loop creation)
- `#` wall, `.` open path, `A` agent, `E` exit
- Shaped rewards: +10 solve, +0.1 closer, -0.1 farther, -0.2 revisit, -0.3 wall hit
- Max 200 steps per episode

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

| Parameter | Default | Description |
|---|---|---|
| `training_steps` | `80` | Number of DQN training iterations |
| `episodes_per_step` | `32` | Episodes collected per step |
| `eval_episodes` | `20` | Episodes for each evaluation |
| `lr` | `5e-4` | Learning rate |
| `gamma` | `0.99` | Discount factor |
| `epsilon_start` | `1.0` | Initial exploration rate |
| `epsilon_end` | `0.05` | Final exploration rate |
| `maze_seed` | `42` | Fixed maze seed (set to `null` for random) |

### LLM + GRPO Agent (`maze_rl_llm.py`)

[GRPO (Group Relative Policy Optimization)](https://arxiv.org/abs/2402.03300) fine-tunes an LLM to act as the agent. Instead of learning Q-values, the model learns to *generate* the right action as text — it reads the maze grid and outputs "UP", "DOWN", "LEFT", or "RIGHT".

```bash
# Run with defaults (seed=42)
flyte run --local maze_rl_llm.py pipeline --training_steps 10

# More training
flyte run --local maze_rl_llm.py pipeline --training_steps 30 --rollouts_per_step 16
```

#### How GRPO works

GRPO is a policy gradient method designed for LLM fine-tuning:

1. **Play episodes in groups** — the model plays several maze episodes, generating text actions at each step
2. **Compute group-relative advantages** — within each group, episodes with above-average reward get positive advantage, below-average get negative
3. **Update the policy** — increase the probability of token sequences that led to good outcomes, decrease those that led to bad ones

Unlike DQN which learns a value function, GRPO directly optimizes the model's generation probabilities. This is the same family of algorithms used for RLHF (reinforcement learning from human feedback) in production LLM training.

#### When to use LLM agents vs classic RL

LLM-based agents shine in environments where the task benefits from **language understanding and reasoning** — things like:

- Code generation environments (write code, run tests, iterate)
- Tool-use tasks (search, API calls, multi-step workflows)
- Text-based games and dialogue systems
- Tasks where the observation is naturally language (documents, instructions)

For pure spatial/numeric tasks like maze navigation, classic RL (DQN, PPO) is faster and more efficient. But OpenEnv supports both equally — the environment doesn't change, only the agent does.

| Parameter | Default | Description |
|---|---|---|
| `training_steps` | `5` | Number of GRPO iterations |
| `rollouts_per_step` | `8` | Episodes per training step |
| `group_size` | `4` | Episodes per advantage group |
| `eval_episodes` | `20` | Episodes for each evaluation |
| `lr` | `1e-5` | Learning rate |
| `maze_seed` | `42` | Fixed maze seed (set to `null` for random) |

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
