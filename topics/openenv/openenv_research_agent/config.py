"""
Flyte task environment configuration.

Defines the shared TaskEnvironment used by all Flyte tasks in workflow.py.
Secrets are injected at runtime by the Flyte cluster — never hardcoded.
"""

import os
from dotenv import load_dotenv
import flyte

load_dotenv()

base_env = flyte.TaskEnvironment(
    name="openenv-research-agent-env",
    image=flyte.Image.from_debian_base(python_version=(3, 11)).with_pip_packages(
        "flyte>=2.1.2",
        "openenv-core>=0.2.3",
        "anthropic",
        "tavily-python",
        "python-dotenv",
        "gradio",
        "plotly",
        "markdown",
    ),
    resources=flyte.Resources(cpu=2, memory="2Gi"),
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
