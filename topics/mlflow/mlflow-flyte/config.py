"""Flyte env + shared constants for the MLflow-on-Flyte demo.

Two parts:
  - Classic ML: train sklearn models, log params/metrics/artifacts to MLflow
  - LLM tracing: trace a LangChain agent with MLflow, run evals

Both parts log to a self-hosted MLflow server running as a Flyte app.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
import flyte

load_dotenv()

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# ── MLflow server (self-hosted as a Flyte app) ──────────────────────────────
MLFLOW_APP_NAME = "mlflow-server"
MLFLOW_PORT = 5000

MLFLOW_CLUSTER_URL = (
    f"http://{MLFLOW_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local"
)

# For local runs, override in .env:
#   MLFLOW_TRACKING_URI=http://mlflow-server-flytesnacks-development.localhost:30081
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", MLFLOW_CLUSTER_URL)

# ── API keys ─────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
OPENAI_MODEL = "gpt-4.1-nano"

# ── Classic ML task environment ──────────────────────────────────────────────
ML_PIP_PACKAGES = (
    "mlflow>=3.1",
    "scikit-learn",
    "pandas",
    "matplotlib",  # confusion matrix / metric plots from mlflow.evaluate
    "shap",        # feature-importance explanations from mlflow.evaluate
    "python-dotenv",
    "unionai-reuse",
)

ml_env = flyte.TaskEnvironment(
    name="mlflow-ml",
    image=flyte.Image.from_debian_base(
        name="mlflow-ml-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(*ML_PIP_PACKAGES),
    env_vars={"MLFLOW_TRACKING_URI": MLFLOW_CLUSTER_URL},
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)

# ── LLM agent task environment ──────────────────────────────────────────────
AGENT_PIP_PACKAGES = (
    "mlflow>=3.1",
    "langgraph>=1.0.7",
    "langchain",
    "langchain-openai",
    "tavily-python",
    "litellm",  # backs mlflow.genai LLM-as-a-judge scorers for openai:/ models
    "python-dotenv",
    "markdown",
    "unionai-reuse",
)

agent_env = flyte.TaskEnvironment(
    name="mlflow-agent",
    image=flyte.Image.from_debian_base(
        name="mlflow-agent-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(*AGENT_PIP_PACKAGES),
    secrets=[
        flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    env_vars={"MLFLOW_TRACKING_URI": MLFLOW_CLUSTER_URL},
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)
