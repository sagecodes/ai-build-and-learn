"""
backends.py — Registry of retrieve functions for each RAG backend.

In Docker, WORKDIR is /app and backends/ lives at /app/backends/ — but /app
is not automatically on sys.path when Python is launched from a subdirectory.
For local dev, rag_comparison is in topics/cognee/rag_comparison/.
"""

import sys
from pathlib import Path
from typing import Callable, Awaitable

_DOCKER_BASE = Path("/app")
_LOCAL_BASE  = Path(__file__).parents[3] / "cognee" / "rag_comparison"

# Add the right base directory so 'from backends import ...' resolves.
_base = _DOCKER_BASE if _DOCKER_BASE.joinpath("backends").exists() else _LOCAL_BASE
if str(_base) not in sys.path:
    sys.path.insert(0, str(_base))

from backends import graph as _graph                  # type: ignore
from backends import vector as _vector                # type: ignore
from backends import cognee_backend as _cognee        # type: ignore
from backends.shared.claude import generate_answer    # type: ignore

# Each value is async (question: str) -> tuple[str, str] (context, summary)
BACKENDS: dict[str, Callable[[str], Awaitable[tuple[str, str]]]] = {
    "Vector RAG": _vector.retrieve,
    "Graph RAG":  _graph.retrieve,
    "Cognee":     _cognee.retrieve,
}

__all__ = ["BACKENDS", "generate_answer"]
