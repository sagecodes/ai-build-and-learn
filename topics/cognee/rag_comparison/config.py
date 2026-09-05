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
# Use the Session Pooler URL (not direct connection — Supabase direct uses IPv6)
# Format: postgresql://postgres.<project-ref>:<password>@<pooler-host>:5432/postgres

PG_URL     = os.environ["PG_URL"]
COLLECTION = "everstorm_docs"

# ── Neo4j AuraDB Free (Graph RAG) ─────────────────────────────────────────────

NEO4J_URI         = os.environ["NEO4J_URI"]
NEO4J_USERNAME    = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD    = os.environ["NEO4J_PASSWORD"]
VECTOR_INDEX_NAME = "chunk-embeddings"

# ── Cognee (local LanceDB + SQLite) ───────────────────────────────────────────
# No credentials needed — Cognee stores data locally under .cognee_data/
# COGNEE_DATA_PATH can be set to override the default storage location.

COGNEE_DATA_PATH = os.environ.get(
    "COGNEE_DATA_PATH", str(ROOT_DIR / ".cognee_data")
)

# ── Google Vertex AI / Gemma 4 (Model Comparison tab) ────────────────────────
# Auth via Application Default Credentials — no key file needed on GCP VM.

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")
GCP_REGION  = os.environ.get("GCP_REGION", "global")
GEMMA_MODEL = os.environ.get("GEMMA_MODEL", "publishers/google/models/gemma-4-26b-a4b-it-maas")
