"""
OpenEnv environment server entrypoint.

Uses create_app() from openenv.core.env_server.http_server to build a
FastAPI application wrapping ResearchEnvironment, then runs it with uvicorn.

This is the correct way to start an OpenEnv environment server — create_app()
takes an environment factory, the action class, and the observation class,
and wires them into HTTP + WebSocket endpoints that EnvClient connects to.
"""

import uvicorn
from openenv.core.env_server.http_server import create_app

from env.models import ResearchAction, ResearchObservation
from env.research_env import ResearchEnvironment
from reward import keyword_reward


def env_factory() -> ResearchEnvironment:
    """Factory function — OpenEnv calls this to create a fresh environment per session."""
    return ResearchEnvironment(reward_fn=keyword_reward)


app = create_app(
    env=env_factory,
    action_cls=ResearchAction,
    observation_cls=ResearchObservation,
    max_concurrent_envs=10,
)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
