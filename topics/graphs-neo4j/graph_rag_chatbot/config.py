"""
config.py

Central constants and connection helpers for graph_rag_chatbot.
Loaded by every ingest and query task.

Secret resolution order:
  1. Flyte secrets (when running on Union cluster)
  2. Environment variables / .env (local development)
"""

import os

import flyte
from dotenv import load_dotenv

load_dotenv()

# ── Flyte / Union backend ─────────────────────────────────────────────────────

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

task_env = flyte.TaskEnvironment(
    name="everstorm-graphrag-tasks",
    image="docker.io/johndellenbaugh/graphrag-app:latest",
    resources=flyte.Resources(cpu=2, memory="4Gi"),
    secrets=[
        flyte.Secret(key="ANTHROPIC_API_KEY", as_env_var="ANTHROPIC_API_KEY"),
        flyte.Secret(key="NEO4J_URI",         as_env_var="NEO4J_URI"),
        flyte.Secret(key="NEO4J_USERNAME",    as_env_var="NEO4J_USERNAME"),
        flyte.Secret(key="NEO4J_PASSWORD",    as_env_var="NEO4J_PASSWORD"),
    ],
)


def _secret(key: str) -> str:
    """Return secret from Flyte context when on cluster, env var otherwise."""
    try:
        from flytekit import current_context
        return current_context().secrets.get(key=key)
    except Exception:
        return os.environ[key]


def neo4j_driver():
    """Return an open Neo4j driver. Caller must close it (use with driver: ...)."""
    from neo4j import GraphDatabase
    uri = _secret("NEO4J_URI")
    user = _secret("NEO4J_USERNAME")
    password = _secret("NEO4J_PASSWORD")
    return GraphDatabase.driver(uri, auth=(user, password))


def anthropic_client():
    """Return an Anthropic client."""
    from anthropic import Anthropic
    return Anthropic(api_key=_secret("ANTHROPIC_API_KEY"))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

CLAUDE_MODEL = "claude-sonnet-4-6"
EMBED_MODEL = "thenlper/gte-small"  # 384D, same model as vector RAG project
EMBED_DIM = 384

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

CHUNK_SIZE = 800        # characters
CHUNK_OVERLAP = 150     # characters

# ---------------------------------------------------------------------------
# Graph schema — Everstorm ontology
# ---------------------------------------------------------------------------

ENTITY_TYPES = [
    "PRODUCT",
    "POLICY",
    "PROGRAM",
    "TIER",
    "BENEFIT",
    "CONDITION",
    "PROCESS",
]

RELATIONSHIP_TYPES = [
    "HAS_POLICY",
    "QUALIFIES_FOR",
    "REQUIRES",
    "APPLIES_TO",
    "PART_OF",
    "COVERS",
]

# ---------------------------------------------------------------------------
# Vector index
# ---------------------------------------------------------------------------

VECTOR_INDEX_NAME = "chunk-embeddings"
VECTOR_SIMILARITY = "cosine"

# ---------------------------------------------------------------------------
# Entity resolution
# ---------------------------------------------------------------------------

ENTITY_MERGE_THRESHOLD = 0.95   # cosine similarity above which two entities are merged

# ---------------------------------------------------------------------------
# Community detection
# ---------------------------------------------------------------------------

LOUVAIN_RESOLUTION = 1.0        # higher = more, smaller communities
