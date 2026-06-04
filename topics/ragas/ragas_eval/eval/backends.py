"""
backends.py — Registry of retrieve functions for each RAG backend.

In Docker (WORKDIR /app), /app/backends/ is on sys.path already.
For local dev, falls back to importing from the rag_comparison source tree.
"""

import sys
from pathlib import Path
from typing import Callable, Awaitable

# Try Docker path first; fall back to local dev path on ImportError.
try:
    from backends import graph as _graph
    from backends import vector as _vector
    from backends import cognee_backend as _cognee
    from backends.shared.claude import generate_answer
except ImportError:
    _rag_path = Path(__file__).parents[3] / "cognee" / "rag_comparison"
    if str(_rag_path) not in sys.path:
        sys.path.insert(0, str(_rag_path))
    from backends import graph as _graph          # type: ignore
    from backends import vector as _vector        # type: ignore
    from backends import cognee_backend as _cognee  # type: ignore
    from backends.shared.claude import generate_answer  # type: ignore

# Each value is async (question: str) -> tuple[str, str] (context, summary)
BACKENDS: dict[str, Callable[[str], Awaitable[tuple[str, str]]]] = {
    "Vector RAG": _vector.retrieve,
    "Graph RAG":  _graph.retrieve,
    "Cognee":     _cognee.retrieve,
}

__all__ = ["BACKENDS", "generate_answer"]
