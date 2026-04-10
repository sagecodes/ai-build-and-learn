"""FastAPI app serving the Atari environment via OpenEnv protocol.

Run directly:
    python -m atari_env.server.app
"""

import uvicorn
from openenv.core.env_server import create_app

from atari_env.models import AtariAction, AtariObservation
from atari_env.server.environment import AtariEnvironment

app = create_app(
    AtariEnvironment,
    AtariAction,
    AtariObservation,
    env_name="atari",
    max_concurrent_envs=4,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
