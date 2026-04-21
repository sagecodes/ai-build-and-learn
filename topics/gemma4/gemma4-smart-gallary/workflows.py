"""
workflows.py — Flyte tasks and parallel workflows for Gemma 4 Smart Gallery.

Each image is processed as a discrete Flyte task, visible in the Flyte TUI.
Parallel execution happens inside coordinator tasks via asyncio.gather() + .aio —
.aio must be called from within a Flyte task context, not from regular functions.

Usage:
    flyte start tui          # terminal 1 — watch tasks execute
    python app.py            # terminal 2 — Gradio UI triggers these workflows
"""

import asyncio
from pathlib import Path

import flyte

import db
import vision_service

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

env = flyte.TaskEnvironment(name="gemma4-smart-gallery")


# ── Shared tasks ──────────────────────────────────────────────────────────────

@env.task
async def scan_folder(folder_path: str) -> list[str]:
    """Return all supported image paths in the folder."""
    folder = Path(folder_path)
    return [
        str(p) for p in sorted(folder.iterdir())
        if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]


# ── Describe workflow tasks ───────────────────────────────────────────────────

@env.task
async def describe_image_task(image_path: str) -> dict:
    """Describe a single image via Gemma 4 vision."""
    description = vision_service.describe_image(image_path)
    return {"path": image_path, "description": description}


@env.task
async def describe_all_task(image_paths: list[str]) -> list[dict]:
    """Fan out describe_image_task across all images in parallel."""
    results = await asyncio.gather(*[
        describe_image_task.aio(image_path=path)
        for path in image_paths
    ])
    return list(results)


@env.task
async def save_descriptions_task(results: list[dict]) -> int:
    """Persist all descriptions to SQLite. Returns count saved."""
    db.init_db()
    for result in results:
        db.save_description(result["path"], result["description"])
    return len(results)


# ── Search workflow tasks ─────────────────────────────────────────────────────

@env.task
async def check_match_task(image_path: str, query: str) -> dict:
    """Check if a single image matches the search query."""
    matched = vision_service.check_image_match(image_path, query)
    return {"path": image_path, "matched": matched}


@env.task
async def check_all_task(image_paths: list[str], query: str) -> list[dict]:
    """Fan out check_match_task across all images in parallel."""
    results = await asyncio.gather(*[
        check_match_task.aio(image_path=path, query=query)
        for path in image_paths
    ])
    return list(results)


@env.task
async def collect_matches_task(results: list[dict]) -> list[str]:
    """Filter and return paths of images that matched the query."""
    return [r["path"] for r in results if r["matched"]]


# ── Workflow runners ──────────────────────────────────────────────────────────

def run_describe_workflow(folder_path: str) -> list[dict]:
    """
    Process all images in folder_path with Gemma 4 and cache descriptions.
    Returns list of {path, description} dicts for UI rendering.
    """
    flyte.init(local_persistence=True)

    scan_run    = flyte.run(scan_folder, folder_path=folder_path)
    image_paths = scan_run.outputs().o0

    if not image_paths:
        return []

    describe_run = flyte.run(describe_all_task, image_paths=image_paths)
    results      = describe_run.outputs().o0

    flyte.run(save_descriptions_task, results=results)

    return results


def run_search_workflow(folder_path: str, query: str) -> list[str]:
    """
    Check every image in folder_path against query using Gemma 4 vision.
    Returns list of matching image paths.
    """
    flyte.init(local_persistence=True)

    scan_run    = flyte.run(scan_folder, folder_path=folder_path)
    image_paths = scan_run.outputs().o0

    if not image_paths:
        return []

    check_run = flyte.run(check_all_task, image_paths=image_paths, query=query)
    results   = check_run.outputs().o0

    collect_run = flyte.run(collect_matches_task, results=results)
    return collect_run.outputs().o0
