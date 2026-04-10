# Atari with OpenEnv — Prebuilt Environment Demo

Play classic Atari games using a prebuilt [OpenEnv](https://github.com/meta-pytorch/OpenEnv) environment. No custom environment code needed — just connect a client and play.

This demonstrates that OpenEnv has a library of **ready-made environments** you can use out of the box, alongside custom ones like the maze demo.

## Setup

```bash
cd topics/openenv/atari

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run

```bash
# Play Pong with random actions
flyte run --local atari_demo.py pipeline --game_name pong --num_episodes 3

# Try other games
flyte run --local atari_demo.py pipeline --game_name breakout
flyte run --local atari_demo.py pipeline --game_name space_invaders
```

The report shows a results table and frame captures from the best episode.

## Available Games

Some popular Atari games supported by ALE:

| Game | Name |
|---|---|
| Pong | `pong` |
| Breakout | `breakout` |
| Space Invaders | `space_invaders` |
| Ms. Pac-Man | `ms_pacman` |
| Asteroids | `asteroids` |
| Freeway | `freeway` |
| Frostbite | `frostbite` |

See the [ALE docs](https://ale.farama.org/) for the full list.

## How It Works

The Atari environment runs as a co-located OpenEnv server — same architecture as the maze demo, but using ALE (Arcade Learning Environment) instead of a custom grid world.

```python
# Same OpenEnv API as the maze — reset/step/state
client = AtariEnv(base_url="http://localhost:8000").sync()
client.connect()

result = client.reset(game_name="pong")
while not result.done:
    action = random.choice(result.observation.legal_actions)
    result = client.step(AtariAction(action_id=action))
    # result.observation.screen -> pixel array (210x160x3 RGB)
    # result.reward -> game score change
    # result.observation.lives -> remaining lives
```

The key point: **same API, different environment.** Whether it's a maze, Atari, or anything else, the client code looks the same.
