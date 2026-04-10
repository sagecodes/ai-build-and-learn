"""Play Atari games using a prebuilt OpenEnv environment.

Demonstrates using OpenEnv with a prebuilt environment — no custom
env code needed. Just connect a client and play.

The Atari env wraps the Arcade Learning Environment (ALE) behind
the standard OpenEnv reset/step/state API.

Run locally:
  flyte run --local atari_demo.py pipeline --game_name pong --num_episodes 3

Try different games:
  flyte run --local atari_demo.py pipeline --game_name breakout
  flyte run --local atari_demo.py pipeline --game_name space_invaders
"""

import base64
import io
import json
import random
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import flyte
import flyte.report
from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server import Action, Observation, State
from pydantic import Field as PydanticField

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class AtariAction(Action):
    action_id: int = 0
    game_name: str = "pong"
    obs_type: str = "rgb"
    full_action_space: bool = False


class AtariObservation(Observation):
    screen: List[int] = PydanticField(default_factory=list)
    screen_shape: List[int] = PydanticField(default_factory=list)
    legal_actions: List[int] = PydanticField(default_factory=list)
    lives: int = 0
    episode_frame_number: int = 0
    frame_number: int = 0


class AtariState(State):
    game_name: str = "pong"
    obs_type: str = "rgb"
    full_action_space: bool = False
    frameskip: int = 4


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AtariEnv(EnvClient[AtariAction, AtariObservation, AtariState]):
    def _step_payload(self, action: AtariAction) -> dict:
        return {
            "action_id": action.action_id,
            "game_name": action.game_name,
            "obs_type": action.obs_type,
            "full_action_space": action.full_action_space,
        }

    def _parse_result(self, payload: dict) -> StepResult[AtariObservation]:
        obs_data = payload.get("observation", payload)
        observation = AtariObservation(
            screen=obs_data.get("screen", []),
            screen_shape=obs_data.get("screen_shape", []),
            legal_actions=obs_data.get("legal_actions", []),
            lives=obs_data.get("lives", 0),
            episode_frame_number=obs_data.get("episode_frame_number", 0),
            frame_number=obs_data.get("frame_number", 0),
            done=payload.get("done", False),
            reward=payload.get("reward"),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> AtariState:
        return AtariState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            game_name=payload.get("game_name", "pong"),
            obs_type=payload.get("obs_type", "rgb"),
            full_action_space=payload.get("full_action_space", False),
            frameskip=payload.get("frameskip", 4),
        )


# ---------------------------------------------------------------------------
# Flyte environment
# ---------------------------------------------------------------------------

env = flyte.TaskEnvironment(
    name="atari_demo",
    image=flyte.Image.from_debian_base().with_pip_packages(
        "openenv-core",
        "ale-py>=0.8.0",
        "numpy>=1.24.0",
        "matplotlib",
        "uvicorn",
    ).with_source_folder(Path(__file__).parent / "atari_env"),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def screen_to_image(screen: list[int], shape: list[int]):
    """Convert flattened screen pixels to a numpy array."""
    import numpy as np
    arr = np.array(screen, dtype=np.uint8).reshape(shape)
    return arr


def image_to_b64(img) -> str:
    """Convert numpy image array to base64 PNG."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 5))
    if img.ndim == 2:
        ax.imshow(img, cmap="gray")
    else:
        ax.imshow(img)
    ax.axis("off")
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


@dataclass
class EpisodeResult:
    total_reward: float = 0.0
    steps: int = 0
    final_lives: int = 0
    frames_b64: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Co-located env server
# ---------------------------------------------------------------------------


def start_env_server(port: int = 8000):
    import uvicorn
    from openenv.core.env_server import create_app
    from atari_env.models import AtariAction as AA, AtariObservation as AO
    from atari_env.server.environment import AtariEnvironment

    app = create_app(
        AtariEnvironment, AA, AO,
        env_name="atari", max_concurrent_envs=4,
    )
    config = uvicorn.Config(app, host="0.0.0.0", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    time.sleep(2)
    print(f"  Atari env server running on localhost:{port}")
    return server


def create_client(port: int = 8000):
    async_client = AtariEnv(
        base_url=f"http://localhost:{port}",
        connect_timeout_s=30.0,
        message_timeout_s=300.0,
    )
    client = async_client.sync()
    client.connect()
    return client


# ---------------------------------------------------------------------------
# Flyte tasks
# ---------------------------------------------------------------------------


@env.task
async def play_random(
    game_name: str = "pong",
    num_episodes: int = 3,
    max_steps: int = 1000,
    capture_every: int = 50,
) -> str:
    """Play random episodes of an Atari game. Returns JSON with results + frame captures."""
    start_env_server()
    client = create_client()

    results = []
    for ep in range(num_episodes):
        result = client.reset(game_name=game_name)
        total_reward = 0.0
        steps = 0
        frames_b64 = []

        # Capture initial frame
        obs = result.observation
        if obs.screen and obs.screen_shape:
            img = screen_to_image(obs.screen, obs.screen_shape)
            frames_b64.append(image_to_b64(img))

        while not result.done and steps < max_steps:
            action_id = random.choice(obs.legal_actions) if obs.legal_actions else 0
            result = client.step(AtariAction(
                action_id=action_id,
                game_name=game_name,
            ))
            obs = result.observation
            total_reward += result.reward or 0.0
            steps += 1

            # Capture frame periodically
            if steps % capture_every == 0 and obs.screen and obs.screen_shape:
                img = screen_to_image(obs.screen, obs.screen_shape)
                frames_b64.append(image_to_b64(img))

        # Capture final frame
        if obs.screen and obs.screen_shape:
            img = screen_to_image(obs.screen, obs.screen_shape)
            frames_b64.append(image_to_b64(img))

        results.append({
            "episode": ep + 1,
            "total_reward": total_reward,
            "steps": steps,
            "final_lives": obs.lives,
            "num_frames": len(frames_b64),
            "frames_b64": frames_b64,
        })
        print(f"  Episode {ep + 1}: reward={total_reward:.0f}, steps={steps}, lives={obs.lives}")

    client.close()

    # Get game info
    state_client = create_client()
    state_info = client.state() if hasattr(client, 'state') else None
    state_client.close()

    return json.dumps({
        "game_name": game_name,
        "num_episodes": num_episodes,
        "results": results,
    })


@env.task(report=True)
async def pipeline(
    game_name: str = "pong",
    num_episodes: int = 3,
    max_steps: int = 1000,
    capture_every: int = 50,
) -> str:
    """Play Atari games and generate a visual report with frame captures."""
    print(f"Game: {game_name}")
    print(f"Episodes: {num_episodes}\n")

    # Play random episodes
    print("=== Playing random episodes ===")
    data = json.loads(await play_random(
        game_name=game_name,
        num_episodes=num_episodes,
        max_steps=max_steps,
        capture_every=capture_every,
    ))
    results = data["results"]

    # Build results table
    rows = ""
    for r in results:
        rows += (
            f"<tr><td>{r['episode']}</td>"
            f"<td>{r['total_reward']:.0f}</td>"
            f"<td>{r['steps']}</td>"
            f"<td>{r['final_lives']}</td></tr>"
        )

    avg_reward = sum(r["total_reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)

    # Build frame gallery — show frames from the best episode
    best_ep = max(results, key=lambda r: r["total_reward"])
    frames_html = ""
    for i, b64 in enumerate(best_ep["frames_b64"]):
        label = "Start" if i == 0 else ("End" if i == len(best_ep["frames_b64"]) - 1 else f"Step {i * capture_every}")
        frames_html += (
            f'<div style="display:inline-block;text-align:center;margin:8px;">'
            f'<img src="data:image/png;base64,{b64}" style="border:1px solid #333;border-radius:4px;" />'
            f'<br><span style="font-size:12px;color:#888;">{label}</span></div>'
        )

    await flyte.report.replace.aio(
        f"<h2>Atari Demo — {game_name.replace('_', ' ').title()}</h2>"
        f"<p>Playing <b>{num_episodes}</b> random episodes using OpenEnv's Atari environment.</p>"
        f"<p>This uses a <b>prebuilt environment</b> — no custom env code needed. "
        f"Just connect a client and call <code>reset()</code> / <code>step()</code>.</p>"
        f"<h3>Results</h3>"
        f"<table border='1' cellpadding='8' cellspacing='0' style='border-collapse:collapse;'>"
        f"<tr><th>Episode</th><th>Total Reward</th><th>Steps</th><th>Lives Left</th></tr>"
        f"{rows}"
        f"<tr><td><b>Average</b></td><td><b>{avg_reward:.0f}</b></td>"
        f"<td><b>{avg_steps:.0f}</b></td><td>-</td></tr>"
        f"</table>"
        f"<h3>Frame Captures (Best Episode)</h3>"
        f"<div style='background:#111;padding:16px;border-radius:8px;overflow-x:auto;'>"
        f"{frames_html}"
        f"</div>"
        f"<h3>How It Works</h3>"
        f"<pre style='background:#1a1a2e;color:#ccc;padding:16px;border-radius:8px;'>"
        f"# Connect to the Atari OpenEnv server\n"
        f"client = AtariEnv(base_url='http://localhost:8000').sync()\n"
        f"client.connect()\n\n"
        f"# Play an episode\n"
        f"result = client.reset(game_name='{game_name}')\n"
        f"while not result.done:\n"
        f"    action = random.choice(result.observation.legal_actions)\n"
        f"    result = client.step(AtariAction(action_id=action))\n"
        f"    print(f'Reward: {{result.reward}}, Lives: {{result.observation.lives}}')\n"
        f"</pre>"
    )
    await flyte.report.flush.aio()

    summary = f"{game_name}: avg_reward={avg_reward:.0f} over {num_episodes} episodes"
    print(f"\n{summary}")
    return summary
