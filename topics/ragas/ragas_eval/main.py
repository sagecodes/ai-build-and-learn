"""
main.py — Entry point for the Ragas RAG Evaluation app.

Docker:  python ragas_eval/main.py  (WORKDIR /app)
Local:   python topics/ragas/ragas_eval/main.py
"""

import logging
import sys
from pathlib import Path

# Ensure ragas_eval/ is importable when run from repo root (Docker uses WORKDIR /app,
# so ragas_eval/ is already a package under /app/ragas_eval/).
_here = Path(__file__).parent
if str(_here) not in sys.path:
    sys.path.insert(0, str(_here))

import compat  # noqa: F401 — must be before ragas; stubs broken deps

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)

from app.ui import build_demo  # noqa: E402 — must come after sys.path setup

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861)
