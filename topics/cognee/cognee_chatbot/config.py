"""
config.py

Flyte backend initialization, TaskEnvironment, and Cognee configuration helpers.
Imported by every task module and by app.py.

Secret resolution order:
  1. Flyte secrets (when running on Union cluster)
  2. Environment variables / .env (local development)
"""

import os

import flyte
from dotenv import load_dotenv

load_dotenv()

# ── Union / Flyte backend ─────────────────────────────────────────────────

UNION_ORG      = "tryv2"
UNION_ENDPOINT = f"{UNION_ORG}.hosted.unionai.cloud"
UNION_PROJECT  = "dellenbaugh"
UNION_DOMAIN   = "development"

BACKEND = os.getenv("FLYTE_BACKEND", "local")

if BACKEND == "union":
    flyte.init(endpoint=UNION_ENDPOINT, project=UNION_PROJECT, domain=UNION_DOMAIN)
elif BACKEND == "cluster":
    flyte.init_in_cluster(org=UNION_ORG, project=UNION_PROJECT, domain=UNION_DOMAIN)
else:
    flyte.init(local_persistence=True)

# ── Task environment ──────────────────────────────────────────────────────

task_env = flyte.TaskEnvironment(
    name="everstorm-cognee-tasks",
    image="docker.io/johndellenbaugh/cognee-chatbot:latest",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="DB_HOST",           as_env_var="DB_HOST"),
        flyte.Secret(key="DB_PORT",           as_env_var="DB_PORT"),
        flyte.Secret(key="DB_NAME",           as_env_var="DB_NAME"),
        flyte.Secret(key="DB_USERNAME",       as_env_var="DB_USERNAME"),
        flyte.Secret(key="DB_PASSWORD",       as_env_var="DB_PASSWORD"),
    ],
)

# ── Constants ─────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-6"


# ── Secret helper ─────────────────────────────────────────────────────────

def _secret(key: str) -> str:
    """Return secret from Flyte context when on cluster, env var otherwise."""
    try:
        from flytekit import current_context
        return current_context().secrets.get(key=key)
    except Exception:
        return os.environ[key]


# ── Cognee configuration ──────────────────────────────────────────────────

def configure_cognee() -> str:
    """
    Set Cognee env vars from resolved secrets and return the Anthropic API key.

    Must be called inside every task, before importing cognee, so the env vars
    are in place when Cognee reads its configuration at import time.
    """
    api_key = _secret("ANTHROPIC_API_KEY")

    os.environ["LLM_PROVIDER"]       = "anthropic"
    os.environ["LLM_MODEL"]          = CLAUDE_MODEL
    os.environ["LLM_API_KEY"]        = api_key
    os.environ["VECTOR_DB_PROVIDER"] = "pgvector"

    # DB_* vars are already injected by as_env_var in task_env secrets,
    # but we set them explicitly here so local .env also works.
    for key in ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USERNAME", "DB_PASSWORD"):
        os.environ[key] = _secret(key)

    return api_key


# ── Anthropic client ──────────────────────────────────────────────────────

def anthropic_client():
    """Return an Anthropic client using the resolved API key."""
    from anthropic import Anthropic
    return Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))
