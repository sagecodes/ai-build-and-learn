"""Train a DQN agent to navigate mazes using OpenEnv + Flyte.

Demonstrates that OpenEnv is model-agnostic: same maze environment,
but trained with a simple DQN (2-layer MLP) instead of an LLM.
The DQN agent learns fast and visibly improves within minutes.

Run locally:
  flyte run --local maze_rl_dqn.py pipeline --training_steps 20

Run on a cluster:
  flyte run maze_rl_dqn.py pipeline --training_steps 50 --episodes_per_step 32
"""

import base64
import io
import json
import os
import random
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import flyte
import flyte.report
from flyte.io import File
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server import Action, Observation, State
from pydantic import Field as PydanticField

# ---------------------------------------------------------------------------
# Models (shared with maze_env server)
# ---------------------------------------------------------------------------


class MazeAction(Action):
    direction: str = "RIGHT"


class MazeObservation(Observation):
    grid: list[list[str]] = PydanticField(default_factory=list)
    agent_pos: tuple[int, int] = (1, 1)
    exit_pos: tuple[int, int] = (5, 5)
    steps_taken: int = 0


class MazeState(State):
    maze_seed: int = 0
    optimal_path_length: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MazeEnv(EnvClient[MazeAction, MazeObservation, MazeState]):
    def _step_payload(self, action: MazeAction) -> dict:
        return {"direction": action.direction}

    def _parse_result(self, payload: dict) -> StepResult[MazeObservation]:
        obs_data = payload.get("observation", payload)
        observation = MazeObservation(
            grid=obs_data.get("grid", []),
            agent_pos=tuple(obs_data.get("agent_pos", (1, 1))),
            exit_pos=tuple(obs_data.get("exit_pos", (6, 6))),
            steps_taken=obs_data.get("steps_taken", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> MazeState:
        return MazeState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            maze_seed=payload.get("maze_seed", 0),
            optimal_path_length=payload.get("optimal_path_length", 0),
        )


# ---------------------------------------------------------------------------
# Flyte environment
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="maze_rl_dqn",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch",
        "openenv-core",
        "matplotlib",
        "uvicorn",
    ).with_source_folder(Path(__file__).parent / "maze_env"),
    resources=flyte.Resources(cpu=2, memory="4Gi", gpu=1),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]
GRID_SIZE = 12

# Map grid characters to numeric values for the neural net
CELL_MAP = {"#": 0.0, ".": 1.0, "A": 2.0, "E": 3.0}


# ---------------------------------------------------------------------------
# Episode recording for visual replay
# ---------------------------------------------------------------------------


@dataclass
class EpisodeFrame:
    step: int
    grid: list[list[str]]
    action: str
    reward: float


@dataclass
class EpisodeRecording:
    frames: list[EpisodeFrame] = field(default_factory=list)
    total_reward: float = 0.0
    solved: bool = False
    length: int = 0
    label: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def get_device():
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def cleanup_memory():
    import gc
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fig_to_html(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<img src="data:image/png;base64,{b64}" />'


def grid_to_tensor(grid, agent_pos, exit_pos, device):
    """Convert maze grid + positions into a flat tensor for the DQN."""
    import torch

    # Flatten grid to numeric values
    flat = []
    for row in grid:
        for cell in row:
            flat.append(CELL_MAP.get(cell, 1.0))

    # Add normalized agent and exit positions
    flat.extend([
        agent_pos[0] / GRID_SIZE,
        agent_pos[1] / GRID_SIZE,
        exit_pos[0] / GRID_SIZE,
        exit_pos[1] / GRID_SIZE,
    ])

    return torch.tensor(flat, dtype=torch.float32, device=device).unsqueeze(0)


# State size: 12x12 grid + 4 position features
STATE_SIZE = GRID_SIZE * GRID_SIZE + 4
ACTION_SIZE = 4


# ---------------------------------------------------------------------------
# Co-located env server
# ---------------------------------------------------------------------------


def start_env_server(port: int = 8000):
    import uvicorn
    from openenv.core.env_server import create_app
    from maze_env.models import MazeAction as MA, MazeObservation as MO
    from maze_env.server.environment import MazeEnvironment

    app = create_app(
        MazeEnvironment, MA, MO,
        env_name="maze", max_concurrent_envs=8,
    )
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(2)
    print(f"  Env server running on localhost:{port}")
    return server


def create_client(port: int = 8000):
    async_client = MazeEnv(
        base_url=f"http://localhost:{port}",
        connect_timeout_s=30.0,
        message_timeout_s=300.0,
    )
    client = async_client.sync()
    client.connect()
    return client


# ---------------------------------------------------------------------------
# DQN Model
# ---------------------------------------------------------------------------


def build_dqn(device):
    """3-layer MLP for Q-value estimation."""
    import torch.nn as nn

    model = nn.Sequential(
        nn.Linear(STATE_SIZE, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, ACTION_SIZE),
    ).to(device)
    return model


# ---------------------------------------------------------------------------
# Replay Buffer
# ---------------------------------------------------------------------------


class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Episode playing
# ---------------------------------------------------------------------------


def play_episode(client, model, device, epsilon=0.1, record=False, maze_seed=None):
    """Play one maze episode with epsilon-greedy DQN policy.

    Returns (transitions, total_reward, solved, recording_or_None).
    """
    import torch

    reset_kwargs = {"seed": maze_seed} if maze_seed is not None else {}
    try:
        result = client.reset(**reset_kwargs)
    except Exception:
        client.connect()
        result = client.reset(**reset_kwargs)

    obs = result.observation
    state = grid_to_tensor(obs.grid, obs.agent_pos, obs.exit_pos, device)

    transitions = []
    recording = EpisodeRecording() if record else None
    total_reward = 0.0

    if recording:
        recording.frames.append(EpisodeFrame(
            step=0, grid=obs.grid, action="START", reward=0.0,
        ))

    step_num = 0
    while not result.done and step_num < 200:
        # Epsilon-greedy action selection
        if random.random() < epsilon:
            action_idx = random.randint(0, ACTION_SIZE - 1)
        else:
            with torch.no_grad():
                q_values = model(state)
                action_idx = q_values.argmax(dim=1).item()

        direction = DIRECTIONS[action_idx]
        result = client.step(MazeAction(direction=direction))

        step_num += 1
        reward = result.reward or 0.0
        total_reward += reward

        obs = result.observation
        next_state = grid_to_tensor(obs.grid, obs.agent_pos, obs.exit_pos, device)
        done = result.done

        transitions.append((state, action_idx, reward, next_state, done))
        state = next_state

        if recording:
            recording.frames.append(EpisodeFrame(
                step=step_num, grid=obs.grid, action=direction, reward=reward,
            ))

    solved = result.done and (result.reward or 0) >= 10.0

    if recording:
        recording.total_reward = total_reward
        recording.solved = solved
        recording.length = step_num

    return transitions, total_reward, solved, recording


def play_episode_baseline(client, policy="random", maze_seed=None):
    """Play one maze episode with a baseline policy. Returns recording."""
    reset_kwargs = {"seed": maze_seed} if maze_seed is not None else {}
    try:
        result = client.reset(**reset_kwargs)
    except Exception:
        client.connect()
        result = client.reset(**reset_kwargs)

    recording = EpisodeRecording()
    obs = result.observation
    recording.frames.append(EpisodeFrame(
        step=0, grid=obs.grid, action="START", reward=0.0,
    ))

    total_reward = 0.0
    step_num = 0
    visited = set()

    while not result.done and step_num < 200:
        obs = result.observation
        agent_r, agent_c = obs.agent_pos
        exit_r, exit_c = obs.exit_pos
        grid = obs.grid

        if policy == "wall_follower":
            visited.add((agent_r, agent_c))
            best_dir, best_dist = None, float("inf")
            for d in DIRECTIONS:
                dr, dc = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[d]
                nr, nc = agent_r + dr, agent_c + dc
                if 0 <= nr < len(grid) and 0 <= nc < len(grid[0]) and grid[nr][nc] != "#":
                    dist = abs(nr - exit_r) + abs(nc - exit_c)
                    if (nr, nc) in visited:
                        dist += 5
                    if dist < best_dist:
                        best_dist = dist
                        best_dir = d
            direction = best_dir or random.choice(DIRECTIONS)
        else:
            direction = random.choice(DIRECTIONS)

        result = client.step(MazeAction(direction=direction))
        step_num += 1
        reward = result.reward or 0.0
        total_reward += reward

        obs = result.observation
        recording.frames.append(EpisodeFrame(
            step=step_num, grid=obs.grid, action=direction, reward=reward,
        ))

    solved = result.done and (result.reward or 0) >= 10.0
    recording.total_reward = total_reward
    recording.solved = solved
    recording.length = step_num
    return recording


# ---------------------------------------------------------------------------
# HTML Replay Generator
# ---------------------------------------------------------------------------


def generate_replay_html(recordings: list[EpisodeRecording], title: str = "Maze Replay") -> str:
    CELL_COLORS = {
        "#": "#2d2d2d",
        ".": "#e8e8e8",
        "A": "#4CAF50",
        "E": "#FF5722",
    }

    episodes_json = []
    for rec in recordings:
        frames = []
        for f in rec.frames:
            frames.append({
                "step": f.step,
                "grid": f.grid,
                "action": f.action,
                "reward": round(f.reward, 3),
            })
        episodes_json.append({
            "frames": frames,
            "total_reward": round(rec.total_reward, 3),
            "solved": rec.solved,
            "length": rec.length,
            "label": rec.label,
        })

    return f"""
    <div style="font-family: monospace; background: #0f0f23; color: #ccc; padding: 20px; border-radius: 8px;">
      <h3 style="color: #4CAF50; margin-top: 0;">{title}</h3>
      <div style="margin-bottom: 10px;">
        <label>Episode:
          <select id="ep-select" onchange="changeEpisode()" style="background:#1a1a2e;color:#ccc;padding:4px;border:1px solid #333;">
          </select>
        </label>
        <span id="ep-info" style="margin-left: 15px;"></span>
      </div>
      <div style="margin-bottom: 10px;">
        <label>Step: <span id="step-label">0</span></label><br>
        <input type="range" id="step-slider" min="0" max="0" value="0" oninput="renderFrame()"
               style="width: 300px;">
        <button onclick="playReplay()" id="play-btn"
                style="margin-left:10px;padding:4px 12px;background:#4CAF50;color:#fff;border:none;border-radius:4px;cursor:pointer;">
          Play
        </button>
      </div>
      <div style="display:flex; gap: 20px; align-items: flex-start;">
        <canvas id="maze-canvas" width="360" height="360"
                style="border: 2px solid #333; border-radius: 4px;"></canvas>
        <div id="frame-info" style="font-size: 14px; line-height: 1.6;"></div>
      </div>
    </div>

    <script>
    const EPISODES = {json.dumps(episodes_json)};
    const COLORS = {json.dumps(CELL_COLORS)};
    let currentEp = 0;
    let playInterval = null;

    function init() {{
      const sel = document.getElementById('ep-select');
      EPISODES.forEach((ep, i) => {{
        const opt = document.createElement('option');
        opt.value = i;
        opt.text = (ep.label ? ep.label : 'Episode ' + (i+1)) + (ep.solved ? ' (SOLVED)' : ' (failed)');
        sel.appendChild(opt);
      }});
      changeEpisode();
    }}

    function changeEpisode() {{
      currentEp = parseInt(document.getElementById('ep-select').value);
      const ep = EPISODES[currentEp];
      const slider = document.getElementById('step-slider');
      slider.max = Math.max(ep.frames.length - 1, 0);
      slider.value = 0;
      document.getElementById('ep-info').textContent =
        (ep.solved ? 'SOLVED' : 'Failed') +
        ' | Steps: ' + ep.length +
        ' | Reward: ' + ep.total_reward;
      renderFrame();
    }}

    function renderFrame() {{
      const ep = EPISODES[currentEp];
      const idx = parseInt(document.getElementById('step-slider').value);
      const frame = ep.frames[idx];
      if (!frame) return;

      document.getElementById('step-label').textContent = frame.step;

      const canvas = document.getElementById('maze-canvas');
      const ctx = canvas.getContext('2d');
      const grid = frame.grid;
      const rows = grid.length;
      const cols = grid[0].length;
      const cellW = canvas.width / cols;
      const cellH = canvas.height / rows;

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      for (let r = 0; r < rows; r++) {{
        for (let c = 0; c < cols; c++) {{
          ctx.fillStyle = COLORS[grid[r][c]] || '#e8e8e8';
          ctx.fillRect(c * cellW, r * cellH, cellW - 1, cellH - 1);
        }}
      }}

      document.getElementById('frame-info').innerHTML =
        'Action: <b>' + frame.action + '</b><br>' +
        'Reward: ' + frame.reward;
    }}

    function playReplay() {{
      if (playInterval) {{
        clearInterval(playInterval);
        playInterval = null;
        document.getElementById('play-btn').textContent = 'Play';
        return;
      }}
      document.getElementById('play-btn').textContent = 'Pause';
      const slider = document.getElementById('step-slider');
      playInterval = setInterval(() => {{
        let val = parseInt(slider.value);
        if (val >= parseInt(slider.max)) {{
          clearInterval(playInterval);
          playInterval = null;
          document.getElementById('play-btn').textContent = 'Play';
          return;
        }}
        slider.value = val + 1;
        renderFrame();
      }}, 200);
    }}

    init();
    </script>
    """


# ---------------------------------------------------------------------------
# Flyte tasks
# ---------------------------------------------------------------------------


@env.task
async def eval_baselines(num_episodes: int = 50, maze_seed: int | None = None) -> str:
    """Run random and wall-follower baselines."""
    start_env_server()
    client = create_client()

    results = {}
    best_recordings = {}

    for policy in ["random", "wall_follower"]:
        solve_count = 0
        rewards = []
        solve_steps = []
        best_rec = None
        best_reward = -float("inf")

        for _ in range(num_episodes):
            rec = play_episode_baseline(client, policy=policy, maze_seed=maze_seed)
            rewards.append(rec.total_reward)
            if rec.solved:
                solve_count += 1
                solve_steps.append(rec.length)
            if rec.total_reward > best_reward:
                best_reward = rec.total_reward
                best_rec = rec

        solve_rate = solve_count / num_episodes
        avg_steps = sum(solve_steps) / len(solve_steps) if solve_steps else 0

        results[policy] = {
            "solve_rate": solve_rate,
            "avg_steps": avg_steps,
            "avg_reward": sum(rewards) / len(rewards),
        }
        if best_rec:
            best_recordings[policy] = {
                "frames": [{"step": f.step, "grid": f.grid,
                            "action": f.action, "reward": f.reward}
                           for f in best_rec.frames],
                "total_reward": best_rec.total_reward,
                "solved": best_rec.solved,
                "length": best_rec.length,
            }
        print(f"  {policy}: solve_rate={solve_rate:.2f}, avg_steps={avg_steps:.1f}")

    client.close()
    return json.dumps({"results": results, "recordings": best_recordings})


@env.task(report=True)
async def train_dqn(
    training_steps: int = 80,
    episodes_per_step: int = 32,
    eval_episodes: int = 20,
    batch_size: int = 128,
    lr: float = 5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    target_update_freq: int = 5,
    replay_capacity: int = 100000,
    maze_seed: int | None = None,
) -> tuple[File, str]:
    """Full DQN training loop with live report updates."""
    import torch
    import torch.nn.functional as F
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    start_env_server()
    client = create_client()
    device = get_device()
    print(f"  Device: {device}")

    q_net = build_dqn(device)
    target_net = build_dqn(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    optimizer = torch.optim.Adam(q_net.parameters(), lr=lr)
    replay = ReplayBuffer(replay_capacity)

    all_metrics = []
    all_eval_results = []

    async def update_report():
        """Update the Flyte report with current training progress."""
        if not all_eval_results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ax = axes[0][0]
        es = [e["step"] for e in all_eval_results]
        ax.plot(es, [e["solve_rate"] for e in all_eval_results], "b-o", linewidth=2, label="DQN Agent")
        ax.set_xlabel("Training Step"); ax.set_ylabel("Solve Rate")
        ax.set_title("Solve Rate (Evaluation)")
        ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 1.05)

        ax = axes[0][1]
        if all_metrics:
            ts = [m["step"] for m in all_metrics]
            ax.plot(ts, [m["avg_reward"] for m in all_metrics], "m-", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Training Step"); ax.set_ylabel("Avg Reward")
        ax.set_title("Training Reward"); ax.grid(True, alpha=0.3)

        ax = axes[1][0]
        if all_metrics:
            ax.plot(ts, [m["loss"] for m in all_metrics], "r-", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Training Step"); ax.set_ylabel("Loss")
        ax.set_title("DQN Loss"); ax.grid(True, alpha=0.3)

        ax = axes[1][1]
        ax2 = ax.twinx()
        if all_metrics:
            ax.plot(ts, [m["epsilon"] for m in all_metrics], "c-", linewidth=1.5, label="Epsilon")
            ax2.plot(ts, [m["solve_rate"] for m in all_metrics], "orange", linewidth=1.5, alpha=0.7, label="Train Solve Rate")
        ax.set_xlabel("Training Step"); ax.set_ylabel("Epsilon", color="c")
        ax2.set_ylabel("Train Solve Rate", color="orange")
        ax.set_title("Exploration vs Exploitation"); ax.grid(True, alpha=0.3)
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, loc="center right")

        plt.tight_layout()
        charts_html = fig_to_html(fig)
        plt.close(fig)

        latest = all_eval_results[-1]
        step_num = all_metrics[-1]["step"] if all_metrics else 0
        await flyte.report.replace.aio(
            f"<h2>DQN Training — Step {step_num}/{training_steps}</h2>"
            f"<p>Eval solve rate: <b>{latest['solve_rate']:.2f}</b> | "
            f"Avg steps: <b>{latest['avg_steps']:.0f}</b></p>"
            f"{charts_html}"
        )
        await flyte.report.flush.aio()

    # Evaluate untrained
    print("\n  === Evaluating untrained ===")
    eval_result = _evaluate(client, q_net, device, eval_episodes, record_best=True, maze_seed=maze_seed)
    all_eval_results.append({"step": 0, **eval_result})
    print(f"    Untrained: solve_rate={eval_result['solve_rate']:.2f}")
    await update_report()

    epsilon_decay = (epsilon_start - epsilon_end) / max(training_steps, 1)

    for step in range(1, training_steps + 1):
        epsilon = max(epsilon_start - epsilon_decay * step, epsilon_end)
        step_rewards = []
        step_solved = []

        for _ in range(episodes_per_step):
            transitions, total_reward, solved, _ = play_episode(
                client, q_net, device, epsilon=epsilon, record=False, maze_seed=maze_seed,
            )
            for t in transitions:
                replay.push(*t)
            step_rewards.append(total_reward)
            step_solved.append(solved)

        # Train on replay buffer — 32 gradient updates per step
        total_loss = 0.0
        num_updates = min(len(replay) // batch_size, 32)

        for _ in range(num_updates):
            batch = replay.sample(batch_size)
            states = torch.cat([s for s, _, _, _, _ in batch])
            actions = torch.tensor([a for _, a, _, _, _ in batch], device=device)
            rewards_t = torch.tensor([r for _, _, r, _, _ in batch], dtype=torch.float32, device=device)
            next_states = torch.cat([ns for _, _, _, ns, _ in batch])
            dones = torch.tensor([d for _, _, _, _, d in batch], dtype=torch.float32, device=device)

            q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

            # Double DQN: q_net selects, target_net evaluates
            with torch.no_grad():
                best_actions = q_net(next_states).argmax(dim=1)
                next_q = target_net(next_states).gather(1, best_actions.unsqueeze(1)).squeeze(1)
                target = rewards_t + gamma * next_q * (1 - dones)

            loss = F.smooth_l1_loss(q_values, target)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        if step % target_update_freq == 0:
            target_net.load_state_dict(q_net.state_dict())

        solve_rate = sum(step_solved) / len(step_solved)
        avg_reward = sum(step_rewards) / len(step_rewards)
        avg_loss = total_loss / max(num_updates, 1)

        all_metrics.append({
            "step": step,
            "solve_rate": solve_rate,
            "avg_reward": avg_reward,
            "loss": avg_loss,
            "epsilon": epsilon,
            "replay_size": len(replay),
        })
        print(f"    Step {step}/{training_steps}: solve={solve_rate:.2f} reward={avg_reward:.2f} eps={epsilon:.2f} loss={avg_loss:.4f}")

        # Evaluate and update report periodically
        if step % max(training_steps // 8, 1) == 0 or step == training_steps:
            print(f"    Evaluating...")
            eval_result = _evaluate(client, q_net, device, eval_episodes, record_best=True, maze_seed=maze_seed)
            all_eval_results.append({"step": step, **eval_result})
            print(f"    Eval: solve_rate={eval_result['solve_rate']:.2f} avg_steps={eval_result['avg_steps']:.1f}")
            await update_report()

    client.close()

    # Save checkpoint
    save_dir = os.path.join("checkpoints", "dqn_final")
    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "q_net": q_net.state_dict(),
        "target_net": target_net.state_dict(),
    }, os.path.join(save_dir, "model.pt"))

    tar_path = os.path.join("checkpoints", "dqn_final.tar.gz")
    shutil.make_archive(os.path.join("checkpoints", "dqn_final"), "gztar", ".", save_dir)
    checkpoint = await File.from_local(tar_path)

    del q_net, target_net, optimizer
    cleanup_memory()

    return checkpoint, json.dumps({
        "train_metrics": all_metrics,
        "eval_results": all_eval_results,
    })


def _evaluate(client, model, device, num_episodes, record_best=False, maze_seed=None):
    """Evaluate the DQN model. Returns dict with stats + optional best replay."""
    solve_count = 0
    rewards = []
    solve_steps = []
    best_rec = None
    best_reward = -float("inf")

    for _ in range(num_episodes):
        _, total_reward, solved, rec = play_episode(
            client, model, device, epsilon=0.0, record=record_best, maze_seed=maze_seed,
        )
        rewards.append(total_reward)
        if solved:
            solve_count += 1
            solve_steps.append(rec.length if rec else 0)
        if rec and total_reward > best_reward:
            best_reward = total_reward
            best_rec = rec

    solve_rate = solve_count / num_episodes
    avg_steps = sum(solve_steps) / len(solve_steps) if solve_steps else 0
    avg_reward = sum(rewards) / len(rewards)

    result = {
        "solve_rate": solve_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
    }

    if best_rec:
        result["best_replay"] = {
            "frames": [{"step": f.step, "grid": f.grid,
                        "action": f.action, "reward": f.reward}
                       for f in best_rec.frames],
            "total_reward": best_rec.total_reward,
            "solved": best_rec.solved,
            "length": best_rec.length,
        }

    return result


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@env.task(report=True)
async def pipeline(
    training_steps: int = 80,
    episodes_per_step: int = 32,
    eval_episodes: int = 20,
    batch_size: int = 128,
    lr: float = 5e-4,
    gamma: float = 0.99,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    target_update_freq: int = 5,
    maze_seed: int | None = 42,
) -> tuple[str, File]:
    """Full DQN maze pipeline: baselines -> train -> eval -> visual report.

    Set maze_seed to train on a fixed maze (faster learning).
    Set maze_seed to None for random mazes each episode (harder).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = get_device()
    seed_label = f"seed={maze_seed}" if maze_seed is not None else "random mazes"
    print(f"Device: {device}")
    print(f"Config: {training_steps} steps, {episodes_per_step} episodes/step, {seed_label}\n")

    # 1. Baselines
    print("=== Evaluating baselines ===")
    baselines_json = json.loads(await eval_baselines(num_episodes=50, maze_seed=maze_seed))
    baselines = baselines_json["results"]
    baseline_recordings = baselines_json.get("recordings", {})
    print(f"  Random:        solve_rate={baselines['random']['solve_rate']:.2f}")
    print(f"  Wall-follower: solve_rate={baselines['wall_follower']['solve_rate']:.2f}")

    # 2. Train DQN
    print("\n=== Training DQN ===")
    checkpoint, metrics_json = await train_dqn(
        training_steps=training_steps,
        episodes_per_step=episodes_per_step,
        eval_episodes=eval_episodes,
        batch_size=batch_size,
        lr=lr,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        target_update_freq=target_update_freq,
        maze_seed=maze_seed,
    )
    all_data = json.loads(metrics_json)
    train_metrics = all_data["train_metrics"]
    eval_results = all_data["eval_results"]

    # 3. Generate report
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Chart 1: Solve rate (eval)
    ax = axes[0][0]
    eval_steps = [e["step"] for e in eval_results]
    ax.plot(eval_steps, [e["solve_rate"] for e in eval_results], "b-o", linewidth=2, label="DQN Agent")
    ax.axhline(baselines["random"]["solve_rate"], color="r", linestyle="--",
               label=f"Random ({baselines['random']['solve_rate']:.2f})")
    ax.axhline(baselines["wall_follower"]["solve_rate"], color="g", linestyle="--",
               label=f"Wall-follower ({baselines['wall_follower']['solve_rate']:.2f})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Solve Rate")
    ax.set_title("Solve Rate (Evaluation)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Chart 2: Training reward
    ax = axes[0][1]
    t_steps = [m["step"] for m in train_metrics]
    ax.plot(t_steps, [m["avg_reward"] for m in train_metrics], "m-", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Episode Reward")
    ax.set_title("Training Reward")
    ax.grid(True, alpha=0.3)

    # Chart 3: Loss
    ax = axes[1][0]
    ax.plot(t_steps, [m["loss"] for m in train_metrics], "r-", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Loss")
    ax.set_title("DQN Loss")
    ax.grid(True, alpha=0.3)

    # Chart 4: Epsilon + solve rate
    ax = axes[1][1]
    ax2 = ax.twinx()
    ax.plot(t_steps, [m["epsilon"] for m in train_metrics], "c-", linewidth=1.5, label="Epsilon")
    ax2.plot(t_steps, [m["solve_rate"] for m in train_metrics], "orange", linewidth=1.5, alpha=0.7, label="Train Solve Rate")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Epsilon", color="c")
    ax2.set_ylabel("Train Solve Rate", color="orange")
    ax.set_title("Exploration vs Exploitation")
    ax.grid(True, alpha=0.3)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.tight_layout()
    charts_html = fig_to_html(fig)
    plt.close(fig)

    # Build replay HTML
    replay_recs = []
    for policy_name, rec_data in baseline_recordings.items():
        label = "Random" if policy_name == "random" else "Wall-follower"
        replay_recs.append(EpisodeRecording(
            frames=[EpisodeFrame(**f) for f in rec_data["frames"]],
            total_reward=rec_data["total_reward"],
            solved=rec_data["solved"],
            length=rec_data["length"],
            label=f"Baseline: {label}",
        ))
    for e in eval_results:
        replay_data = e.get("best_replay")
        if replay_data:
            step_label = f"DQN step {e['step']}" if e["step"] > 0 else "DQN untrained"
            replay_recs.append(EpisodeRecording(
                frames=[EpisodeFrame(**f) for f in replay_data["frames"]],
                total_reward=replay_data["total_reward"],
                solved=replay_data["solved"],
                length=replay_data["length"],
                label=step_label,
            ))

    replay_html = generate_replay_html(replay_recs, title="Maze Replays — DQN Agent")

    final = eval_results[-1]
    await flyte.report.replace.aio(
        f"<h2>Maze RL Training Report — DQN</h2>"
        f"<h3>Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><th>Policy</th><th>Solve Rate</th><th>Avg Steps</th><th>Avg Reward</th></tr>"
        f"<tr><td>Random</td><td>{baselines['random']['solve_rate']:.2f}</td>"
        f"<td>{baselines['random']['avg_steps']:.0f}</td>"
        f"<td>{baselines['random']['avg_reward']:.2f}</td></tr>"
        f"<tr><td>Wall-follower</td><td>{baselines['wall_follower']['solve_rate']:.2f}</td>"
        f"<td>{baselines['wall_follower']['avg_steps']:.0f}</td>"
        f"<td>{baselines['wall_follower']['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>DQN (untrained)</b></td><td>{eval_results[0]['solve_rate']:.2f}</td>"
        f"<td>{eval_results[0]['avg_steps']:.0f}</td>"
        f"<td>{eval_results[0]['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>DQN (final)</b></td><td><b>{final['solve_rate']:.2f}</b></td>"
        f"<td><b>{final['avg_steps']:.0f}</b></td>"
        f"<td><b>{final['avg_reward']:.2f}</b></td></tr>"
        f"</table>"
        f"<h3>Training Progress</h3>{charts_html}"
        f"<h3>Visual Replay</h3>{replay_html}"
        f"<h3>Config</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td>Algorithm</td><td>DQN (3-layer MLP, 256-256-128)</td></tr>"
        f"<tr><td>Training Steps</td><td>{training_steps}</td></tr>"
        f"<tr><td>Episodes/Step</td><td>{episodes_per_step}</td></tr>"
        f"<tr><td>Batch Size</td><td>{batch_size}</td></tr>"
        f"<tr><td>Learning Rate</td><td>{lr}</td></tr>"
        f"<tr><td>Gamma</td><td>{gamma}</td></tr>"
        f"<tr><td>Epsilon</td><td>{epsilon_start} -> {epsilon_end}</td></tr>"
        f"<tr><td>Maze Seed</td><td>{maze_seed if maze_seed is not None else 'Random'}</td></tr>"
        f"<tr><td>Device</td><td>{device}</td></tr>"
        f"</table>"
    )
    await flyte.report.flush.aio()

    summary = (
        f"DQN final solve_rate: {final['solve_rate']:.2f} "
        f"(random: {baselines['random']['solve_rate']:.2f}, "
        f"wall-follower: {baselines['wall_follower']['solve_rate']:.2f})"
    )
    print(f"\n{summary}")

    cleanup_memory()

    return summary, checkpoint
