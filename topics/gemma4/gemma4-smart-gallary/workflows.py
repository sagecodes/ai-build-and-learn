"""
workflows.py — Public entry point for Gemma 4 Smart Gallery workflows.

Reads FLYTE_BACKEND from .env and dispatches to the appropriate backend module:
  FLYTE_BACKEND=local  (default) → workflows_local.py
  FLYTE_BACKEND=union            → workflows_union.py

app.py and agent.py import only from this file.
"""

import os

import flyte
from dotenv import load_dotenv

load_dotenv()

_FLYTE_BACKEND = os.getenv("FLYTE_BACKEND", "local")


def _init_flyte() -> None:
    if _FLYTE_BACKEND == "union":
        flyte.init(
            endpoint="tryv2.hosted.unionai.cloud",
            project="dellenbaugh",
            domain="development",
        )
    else:
        flyte.init(local_persistence=True)


def run_describe_workflow(folder_path: str):
    """Describe all images in folder. Yields one {path, description} per image."""
    _init_flyte()
    if _FLYTE_BACKEND == "union":
        import workflows_union as backend
    else:
        import workflows_local as backend
    yield from backend.run_describe(folder_path)


def run_search_workflow(folder_path: str, query: str):
    """Search images by query. Yields progress dicts; final dict includes matches."""
    _init_flyte()
    if _FLYTE_BACKEND == "union":
        import workflows_union as backend
    else:
        import workflows_local as backend
    yield from backend.run_search(folder_path, query)
