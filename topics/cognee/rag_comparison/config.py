import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"

# ── Models ────────────────────────────────────────────────────────────────────

CLAUDE_MODEL       = "claude-sonnet-4-6"
EMBEDDING_MODEL    = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMS     = 384

# ── Anthropic ─────────────────────────────────────────────────────────────────

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

# ── Supabase / pgvector (Vector RAG) ──────────────────────────────────────────

PG_HOST     = os.environ["PG_HOST"]
PG_PORT     = os.environ.get("PG_PORT", "5432")
PG_DB       = os.environ["PG_DB"]
PG_USER     = os.environ["PG_USER"]
PG_PASSWORD = os.environ["PG_PASSWORD"]

# ── Neo4j AuraDB Free (Graph RAG) ─────────────────────────────────────────────

NEO4J_URI      = os.environ["NEO4J_URI"]
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]

# ── Cognee (local LanceDB + SQLite) ───────────────────────────────────────────
# No credentials needed — Cognee stores data locally under .cognee_data/
# COGNEE_DATA_PATH can be set to override the default storage location.

COGNEE_DATA_PATH = os.environ.get(
    "COGNEE_DATA_PATH", str(ROOT_DIR / ".cognee_data")
)
