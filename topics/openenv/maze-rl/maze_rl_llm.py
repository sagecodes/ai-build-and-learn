"""Train an LLM to navigate mazes via GRPO using OpenEnv + Flyte.

Demonstrates:
- Building a custom OpenEnv environment (8x8 maze)
- Co-located env server (starts inside each task, zero network overhead)
- GRPO training loop with reward shaping
- Visual HTML replay with frame slider in Flyte reports

Run locally:
  flyte run --local maze_rl.py pipeline --training_steps 5

Run on a cluster:
  flyte run maze_rl.py pipeline --training_steps 20 --rollouts_per_step 16
"""

import base64
import io
import json
import os
import random
import shutil
import subprocess
import sys
import threading
import time
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
    """Action: choose a direction."""
    direction: str = "RIGHT"


class MazeObservation(Observation):
    """What the agent sees after each step."""
    grid: list[list[str]] = PydanticField(default_factory=list)
    agent_pos: tuple[int, int] = (1, 1)
    exit_pos: tuple[int, int] = (5, 5)
    steps_taken: int = 0


class MazeState(State):
    """Metadata about the current maze episode."""
    maze_seed: int = 0
    optimal_path_length: int = 0


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class MazeEnv(EnvClient[MazeAction, MazeObservation, MazeState]):
    """Client that connects to a Maze OpenEnv server."""

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
    name="maze_rl",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "torch",
        "transformers",
        "openenv-core",
        "matplotlib",
        "uvicorn",
    ).with_source_folder(Path(__file__).parent / "maze_env"),
    resources=flyte.Resources(cpu=2, memory="8Gi", gpu=1),
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DIRECTIONS = ["UP", "DOWN", "LEFT", "RIGHT"]

SYSTEM_PROMPT = (
    "You are navigating a 12x12 maze.\n"
    "The grid uses: # = wall, . = open path, A = you (agent), E = exit\n"
    "You MUST respond with exactly one word: UP, DOWN, LEFT, or RIGHT.\n"
    "Strategy: Find the shortest path from A to E while avoiding walls (#)."
)


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


def format_observation(grid, agent_pos, exit_pos, steps_taken):
    rows = [" ".join(row) for row in grid]
    grid_text = "\n".join(rows)
    return (
        f"Maze:\n{grid_text}\n\n"
        f"Your position: row={agent_pos[0]}, col={agent_pos[1]}\n"
        f"Exit position: row={exit_pos[0]}, col={exit_pos[1]}\n"
        f"Steps taken: {steps_taken}\n"
        "Which direction? Reply UP, DOWN, LEFT, or RIGHT."
    )


def parse_direction(text: str) -> str:
    upper = text.strip().upper()
    for d in DIRECTIONS:
        if d in upper:
            return d
    return random.choice(DIRECTIONS)


# ---------------------------------------------------------------------------
# Co-located env server (starts inside each task)
# ---------------------------------------------------------------------------


def start_env_server(port: int = 8000):
    """Start the maze env server in a background thread. Returns the server."""
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
    """Create a connected sync maze env client."""
    async_client = MazeEnv(
        base_url=f"http://localhost:{port}",
        connect_timeout_s=30.0,
        message_timeout_s=300.0,
    )
    client = async_client.sync()
    client.connect()
    return client


# ---------------------------------------------------------------------------
# HTML Replay Generator
# ---------------------------------------------------------------------------


def generate_replay_html(recordings: list[EpisodeRecording], title: str = "Maze Replay") -> str:
    """Generate HTML with a slider to step through recorded maze episodes."""

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
        opt.text = 'Episode ' + (i+1) + (ep.solved ? ' (SOLVED)' : ' (failed)');
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
# Game-playing helpers
# ---------------------------------------------------------------------------


def play_episode_record(client, model, tokenizer, device, temperature=0.7, maze_seed=None):
    """Play one maze episode with the LLM. Returns (trajectory, total_reward, recording)."""
    import torch

    reset_kwargs = {"seed": maze_seed} if maze_seed is not None else {}
    try:
        result = client.reset(**reset_kwargs)
    except Exception:
        client.connect()
        result = client.reset(**reset_kwargs)

    trajectory = []
    recording = EpisodeRecording()
    total_reward = 0.0

    obs = result.observation
    recording.frames.append(EpisodeFrame(
        step=0, grid=obs.grid, action="START", reward=0.0,
    ))

    step_num = 0
    while not result.done and step_num < 200:
        obs = result.observation
        user_prompt = format_observation(obs.grid, obs.agent_pos, obs.exit_pos, obs.steps_taken)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)
        prompt_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=8,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-4),
                return_dict_in_generate=True,
                output_scores=True,
            )

        gen_ids = outputs.sequences[0, prompt_len:]
        gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)

        log_probs = []
        for i, score in enumerate(outputs.scores):
            if i < len(gen_ids):
                lp = torch.log_softmax(score[0], dim=-1)
                log_probs.append(lp[gen_ids[i]].item())

        direction = parse_direction(gen_text)
        result = client.step(MazeAction(direction=direction))

        step_num += 1
        reward = result.reward or 0.0
        total_reward += reward

        trajectory.append({
            "prompt_ids": inputs["input_ids"][0].tolist(),
            "completion_ids": gen_ids.tolist(),
            "log_probs": log_probs,
            "action": gen_text.strip(),
        })

        obs = result.observation
        recording.frames.append(EpisodeFrame(
            step=step_num, grid=obs.grid, action=direction, reward=reward,
        ))

    solved = result.done and (result.reward or 0) >= 10.0
    recording.total_reward = total_reward
    recording.solved = solved
    recording.length = step_num

    return trajectory, total_reward, recording


def play_episode_baseline(client, policy="random", maze_seed=None):
    """Play one maze episode with a simple policy. Returns recording."""
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
# Flyte tasks
# ---------------------------------------------------------------------------


@env.task
async def eval_baselines(num_episodes: int = 50, maze_seed: int | None = None) -> str:
    """Run random and wall-follower baselines. Returns JSON with stats + best replays."""
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
        avg_reward = sum(rewards) / len(rewards)

        results[policy] = {
            "solve_rate": solve_rate,
            "avg_steps": avg_steps,
            "avg_reward": avg_reward,
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


@env.task
async def train_step(
    model_path: str,
    num_rollouts: int = 8,
    group_size: int = 4,
    lr: float = 1e-5,
    step_idx: int = 0,
    checkpoint_file: File | None = None,
    use_bfloat16: bool = True,
    gradient_checkpointing: bool = True,
    maze_seed: int | None = None,
) -> tuple[File, str]:
    """One GRPO training iteration. Co-locates env server. Returns checkpoint + metrics."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if checkpoint_file is not None:
        local_tar = await checkpoint_file.download()
        extract_dir = os.path.join("checkpoints", f"prev_checkpoint_{step_idx}")
        shutil.unpack_archive(local_tar, extract_dir)
        model_path = os.path.join(extract_dir, f"checkpoint_step_{step_idx - 1}")

    device = get_device()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    print(f"  Step {step_idx} | device={device} | dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.train()
    if gradient_checkpointing:
        model.gradient_checkpointing_enable()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    start_env_server()
    client = create_client()

    all_rewards = []
    all_solved = []
    total_loss = 0.0
    num_groups = max(num_rollouts // group_size, 1)

    try:
        for g in range(num_groups):
            group_trajs = []
            group_rewards = []

            for _ in range(group_size):
                traj, reward, rec = play_episode_record(client, model, tokenizer, device, maze_seed=maze_seed)
                group_trajs.append(traj)
                group_rewards.append(reward)
                all_solved.append(rec.solved)

            all_rewards.extend(group_rewards)

            mean_r = sum(group_rewards) / len(group_rewards)
            std_r = (sum((r - mean_r) ** 2 for r in group_rewards) / len(group_rewards)) ** 0.5
            std_r = max(std_r, 1e-8)
            advantages = [(r - mean_r) / std_r for r in group_rewards]

            optimizer.zero_grad()
            group_loss = 0.0

            for traj, adv in zip(group_trajs, advantages):
                for step_data in traj:
                    if not step_data["completion_ids"]:
                        continue
                    prompt_t = torch.tensor([step_data["prompt_ids"]], device=device)
                    comp_t = torch.tensor([step_data["completion_ids"]], device=device)
                    full_ids = torch.cat([prompt_t, comp_t], dim=1)

                    out = model(full_ids)
                    logits = out.logits[0, prompt_t.shape[1] - 1 : -1]
                    lp = torch.log_softmax(logits, dim=-1)
                    token_lp = lp.gather(1, comp_t[0].unsqueeze(1)).squeeze(1)
                    step_loss = -token_lp.sum() * adv
                    step_loss.backward()
                    group_loss += step_loss.item()

                    del out, logits, lp, token_lp, step_loss, prompt_t, comp_t, full_ids

            optimizer.step()
            total_loss += group_loss
            cleanup_memory()
    finally:
        client.close()

    solve_rate = sum(all_solved) / len(all_solved) if all_solved else 0
    avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0

    os.makedirs("checkpoints", exist_ok=True)
    save_dir = os.path.join("checkpoints", f"checkpoint_step_{step_idx}")
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)

    tar_path = os.path.join("checkpoints", f"checkpoint_step_{step_idx}.tar.gz")
    shutil.make_archive(os.path.join("checkpoints", f"checkpoint_step_{step_idx}"), "gztar", ".", save_dir)
    checkpoint = await File.from_local(tar_path)

    metrics = json.dumps({
        "step": step_idx,
        "solve_rate": solve_rate,
        "avg_reward": avg_reward,
        "loss": total_loss / num_groups,
    })
    print(f"  Step {step_idx} | solve_rate={solve_rate:.2f} avg_reward={avg_reward:.2f}")

    del model, optimizer
    cleanup_memory()

    return checkpoint, metrics


@env.task
async def eval_model(
    model_path: str,
    num_episodes: int = 20,
    step_idx: int = 0,
    checkpoint_file: File | None = None,
    use_bfloat16: bool = True,
    maze_seed: int | None = None,
) -> str:
    """Evaluate model on mazes. Co-locates env server. Returns JSON with stats + best replay."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if checkpoint_file is not None:
        local_tar = await checkpoint_file.download()
        extract_dir = os.path.join("checkpoints", f"eval_checkpoint_{step_idx}")
        shutil.unpack_archive(local_tar, extract_dir)
        model_path = os.path.join(extract_dir, f"checkpoint_step_{step_idx}")

    device = get_device()
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype).to(device)
    model.eval()

    start_env_server()
    client = create_client()

    solve_count = 0
    rewards = []
    solve_steps = []
    direction_counts = {d: 0 for d in DIRECTIONS}
    best_rec = None
    best_reward = -float("inf")

    try:
        for _ in range(num_episodes):
            traj, reward, rec = play_episode_record(client, model, tokenizer, device, temperature=0.0, maze_seed=maze_seed)
            rewards.append(reward)
            if rec.solved:
                solve_count += 1
                solve_steps.append(rec.length)
            if reward > best_reward:
                best_reward = reward
                best_rec = rec
            for s in traj:
                d = parse_direction(s["action"])
                direction_counts[d] += 1
    finally:
        client.close()

    solve_rate = solve_count / num_episodes
    avg_reward = sum(rewards) / len(rewards)
    avg_steps = sum(solve_steps) / len(solve_steps) if solve_steps else 0

    best_replay = None
    if best_rec:
        best_replay = {
            "frames": [{"step": f.step, "grid": f.grid,
                        "action": f.action, "reward": f.reward}
                       for f in best_rec.frames],
            "total_reward": best_rec.total_reward,
            "solved": best_rec.solved,
            "length": best_rec.length,
        }

    del model
    cleanup_memory()

    return json.dumps({
        "step": step_idx,
        "solve_rate": solve_rate,
        "avg_steps": avg_steps,
        "avg_reward": avg_reward,
        "direction_distribution": direction_counts,
        "best_replay": best_replay,
    })


# ---------------------------------------------------------------------------
# Pipeline — orchestrates everything and generates the report
# ---------------------------------------------------------------------------


@env.task(report=True)
async def pipeline(
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct",
    training_steps: int = 5,
    rollouts_per_step: int = 8,
    group_size: int = 4,
    eval_episodes: int = 20,
    lr: float = 1e-5,
    use_bfloat16: bool = True,
    gradient_checkpointing: bool = True,
    maze_seed: int | None = 42,
) -> tuple[str, File]:
    """Full Maze RL pipeline: baselines -> GRPO training -> eval -> visual report.

    Set maze_seed to train on a fixed maze. Set to None for random mazes.
    The env server starts co-located inside each task.
    """
    device = get_device()
    seed_label = f"seed={maze_seed}" if maze_seed is not None else "random mazes"
    print(f"Device: {device}")
    print(f"Model:  {model_id}")
    print(f"Config: {training_steps} steps, {rollouts_per_step} rollouts/step, group_size={group_size}, {seed_label}\n")

    # 1. Baselines
    print("=== Evaluating baselines ===")
    baselines_json = json.loads(await eval_baselines(num_episodes=50, maze_seed=maze_seed))
    baselines = baselines_json["results"]
    baseline_recordings = baselines_json.get("recordings", {})
    print(f"  Random:        solve_rate={baselines['random']['solve_rate']:.2f}")
    print(f"  Wall-follower: solve_rate={baselines['wall_follower']['solve_rate']:.2f}")

    # 2. Evaluate untrained model
    print("\n=== Evaluating untrained model ===")
    eval_results = [json.loads(
        await eval_model(model_id, eval_episodes, 0, use_bfloat16=use_bfloat16, maze_seed=maze_seed)
    )]
    print(f"  Untrained: solve_rate={eval_results[0]['solve_rate']:.2f}")

    # 3. Training loop
    prev_checkpoint = None
    train_metrics = []

    for step in range(1, training_steps + 1):
        print(f"\n=== Training step {step}/{training_steps} ===")

        checkpoint_file, metrics_json = await train_step(
            model_id,
            num_rollouts=rollouts_per_step,
            group_size=group_size,
            lr=lr,
            step_idx=step,
            checkpoint_file=prev_checkpoint,
            use_bfloat16=use_bfloat16,
            gradient_checkpointing=gradient_checkpointing,
            maze_seed=maze_seed,
        )
        train_metrics.append(json.loads(metrics_json))

        eval_json = await eval_model(
            model_id, eval_episodes, step,
            checkpoint_file=checkpoint_file,
            use_bfloat16=use_bfloat16,
            maze_seed=maze_seed,
        )
        eval_results.append(json.loads(eval_json))
        prev_checkpoint = checkpoint_file
        print(f"  Eval: solve_rate={eval_results[-1]['solve_rate']:.2f}")

    # 4. Generate report
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps_list = [e["step"] for e in eval_results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Chart 1: Solve rate
    ax = axes[0]
    ax.plot(steps_list, [e["solve_rate"] for e in eval_results], "b-o", linewidth=2, label="GRPO Agent")
    ax.axhline(baselines["random"]["solve_rate"], color="r", linestyle="--",
               label=f"Random ({baselines['random']['solve_rate']:.2f})")
    ax.axhline(baselines["wall_follower"]["solve_rate"], color="g", linestyle="--",
               label=f"Wall-follower ({baselines['wall_follower']['solve_rate']:.2f})")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Solve Rate")
    ax.set_title("Solve Rate Over Training")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # Chart 2: Avg reward
    ax = axes[1]
    ax.plot(steps_list, [e["avg_reward"] for e in eval_results], "m-o", linewidth=2)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Avg Reward")
    ax.set_title("Evaluation Reward")
    ax.grid(True, alpha=0.3)

    # Chart 3: Direction distribution
    ax = axes[2]
    for d in DIRECTIONS:
        fracs = []
        for e in eval_results:
            dist = e.get("direction_distribution", {})
            total = sum(dist.values())
            fracs.append(dist.get(d, 0) / max(total, 1))
        ax.plot(steps_list, fracs, "-o", markersize=4, label=d)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Fraction")
    ax.set_title("Direction Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    charts_html = fig_to_html(fig)
    plt.close(fig)

    # Build replay HTML from best episodes
    replay_recs = []
    for policy_name, rec_data in baseline_recordings.items():
        rec = EpisodeRecording(
            frames=[EpisodeFrame(**f) for f in rec_data["frames"]],
            total_reward=rec_data["total_reward"],
            solved=rec_data["solved"],
            length=rec_data["length"],
        )
        replay_recs.append(rec)
    for e in eval_results:
        replay_data = e.get("best_replay")
        if replay_data:
            rec = EpisodeRecording(
                frames=[EpisodeFrame(**f) for f in replay_data["frames"]],
                total_reward=replay_data["total_reward"],
                solved=replay_data["solved"],
                length=replay_data["length"],
            )
            replay_recs.append(rec)

    replay_html = generate_replay_html(replay_recs, title="Maze Replays")

    final = eval_results[-1]
    await flyte.report.replace.aio(
        f"<h2>Maze RL Training Report</h2>"
        f"<h3>Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><th>Policy</th><th>Solve Rate</th><th>Avg Steps</th><th>Avg Reward</th></tr>"
        f"<tr><td>Random</td><td>{baselines['random']['solve_rate']:.2f}</td>"
        f"<td>{baselines['random']['avg_steps']:.0f}</td>"
        f"<td>{baselines['random']['avg_reward']:.2f}</td></tr>"
        f"<tr><td>Wall-follower</td><td>{baselines['wall_follower']['solve_rate']:.2f}</td>"
        f"<td>{baselines['wall_follower']['avg_steps']:.0f}</td>"
        f"<td>{baselines['wall_follower']['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>GRPO (untrained)</b></td><td>{eval_results[0]['solve_rate']:.2f}</td>"
        f"<td>{eval_results[0]['avg_steps']:.0f}</td>"
        f"<td>{eval_results[0]['avg_reward']:.2f}</td></tr>"
        f"<tr><td><b>GRPO (final)</b></td><td><b>{final['solve_rate']:.2f}</b></td>"
        f"<td><b>{final['avg_steps']:.0f}</b></td>"
        f"<td><b>{final['avg_reward']:.2f}</b></td></tr>"
        f"</table>"
        f"<h3>Training Progress</h3>{charts_html}"
        f"<h3>Visual Replay</h3>{replay_html}"
        f"<h3>Config</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><td>Model</td><td>{model_id}</td></tr>"
        f"<tr><td>Training Steps</td><td>{training_steps}</td></tr>"
        f"<tr><td>Rollouts/Step</td><td>{rollouts_per_step}</td></tr>"
        f"<tr><td>Group Size</td><td>{group_size}</td></tr>"
        f"<tr><td>Learning Rate</td><td>{lr}</td></tr>"
        f"<tr><td>Device</td><td>{device}</td></tr>"
        f"</table>"
    )
    await flyte.report.flush.aio()

    summary = (
        f"Final solve_rate: {final['solve_rate']:.2f} "
        f"(random: {baselines['random']['solve_rate']:.2f}, "
        f"wall-follower: {baselines['wall_follower']['solve_rate']:.2f})"
    )
    print(f"\n{summary}")

    del prev_checkpoint
    cleanup_memory()

    return summary, checkpoint_file
