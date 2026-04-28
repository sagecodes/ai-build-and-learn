"""
Flyte task environment for vector_rag_chatbot.

Defines the shared TaskEnvironment used by all tasks in workflows.py.
Secrets are injected at runtime by the Union cluster — never read from .env
on the cluster side. The local .env is only used for local development.
"""

import os
from pathlib import Path

import flyte
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── Flyte backend ─────────────────────────────────────────────────────────────
# Set FLYTE_BACKEND=union in .env to target Union cloud.
# Default is local (tasks run in-process, good for dev/testing).

BACKEND = os.getenv("FLYTE_BACKEND", "local")

UNION_ORG      = "tryv2"
UNION_ENDPOINT = f"{UNION_ORG}.hosted.unionai.cloud"
UNION_PROJECT  = "dellenbaugh"
UNION_DOMAIN   = "development"

if BACKEND == "union":
    flyte.init(endpoint=UNION_ENDPOINT, project=UNION_PROJECT, domain=UNION_DOMAIN)
elif BACKEND == "cluster":
    flyte.init_in_cluster(org=UNION_ORG, project=UNION_PROJECT, domain=UNION_DOMAIN)
else:
    flyte.init(local_persistence=True)

# ── Shared task environment ───────────────────────────────────────────────────

env = flyte.TaskEnvironment(
    name="vector-rag-chatbot",
    image=flyte.Image.from_debian_base(python_version=(3, 11), registry="docker.io/johndellenbaugh").with_pip_packages(
        "flyte>=2.1.2",
        "psycopg[binary]>=3.1.0",
        "pgvector>=0.3.0",
        "sentence-transformers>=3.0.0",
        "anthropic>=0.40.0",
        "PyMuPDF>=1.24.0",
        "langchain-text-splitters>=0.3.0",
        "python-dotenv>=1.0.0",
    ),
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="PG_URL",             as_env_var="PG_URL"),
    ],
)

# ── Constants shared across tasks ─────────────────────────────────────────────

COLLECTION  = "everstorm_docs"
EMBED_MODEL = "thenlper/gte-small"
EMBED_DIM   = 384
