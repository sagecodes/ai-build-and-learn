import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Models ─────────────────────────────────────────────────────────────────────

EVAL_LLM_MODEL  = "claude-haiku-4-5-20251001"   # cheap for ~300-500 ragas judge calls
CLAUDE_MODEL    = "claude-sonnet-4-6"            # RAG answer generation
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"       # matches rag_comparison

# ── Env vars ───────────────────────────────────────────────────────────────────

_REQUIRED = ["ANTHROPIC_API_KEY", "PG_URL", "NEO4J_URI", "NEO4J_PASSWORD"]
_missing = [v for v in _REQUIRED if not os.environ.get(v)]
if _missing:
    raise EnvironmentError(f"Missing required environment variables: {', '.join(_missing)}")

ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
PG_URL            = os.environ["PG_URL"]
NEO4J_URI         = os.environ["NEO4J_URI"]
NEO4J_PASSWORD    = os.environ["NEO4J_PASSWORD"]
NEO4J_USERNAME    = os.environ.get("NEO4J_USERNAME", "neo4j")

# ── Google Cloud ───────────────────────────────────────────────────────────────
# Used for Vertex AI embeddings (text-embedding-004). VM uses ADC — no key needed.

GCP_PROJECT = os.environ.get("GCP_PROJECT", "")

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
