"""
workflows_local.py — Flyte tasks and runners for local backend.

Tasks use local file paths. Flyte runs with local_persistence=True.
All file I/O and DB writes happen on the local machine.
"""

from pathlib import Path

import flyte

import db
import vision_service

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

env = flyte.TaskEnvironment(name="gemma4-smart-gallery")


# ── Tasks ─────────────────────────────────────────────────────────────────────

@env.task
async def scan_folder(folder_path: str) -> list[str]:
    folder = Path(folder_path)
    return [
        str(p) for p in sorted(folder.iterdir())
        if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]


@env.task
async def describe_image_task(image_path: str) -> dict:
    description = vision_service.describe_image(image_path)
    return {"path": image_path, "description": description}


@env.task
async def save_descriptions_task(results: list[dict]) -> int:
    db.init_db()
    for result in results:
        db.save_description(result["path"], result["description"])
    return len(results)


@env.task
async def check_match_task(image_path: str, query: str) -> dict:
    matched = vision_service.check_image_match(image_path, query)
    return {"path": image_path, "matched": matched}


@env.task
async def collect_matches_task(results: list[dict]) -> list[str]:
    return [r["path"] for r in results if r["matched"]]


# ── Runners ───────────────────────────────────────────────────────────────────

def run_describe(folder_path: str):
    """Yields one {path, description} dict per image as it completes."""
    scan_run    = flyte.run(scan_folder, folder_path=folder_path)
    image_paths = scan_run.outputs().o0

    if not image_paths:
        return

    results = []
    for path in image_paths:
        run    = flyte.run(describe_image_task, image_path=path)
        result = run.outputs().o0
        results.append(result)
        yield result

    flyte.run(save_descriptions_task, results=results)


def run_search(folder_path: str, query: str):
    """Yields progress dicts per image; final dict includes matches list."""
    scan_run    = flyte.run(scan_folder, folder_path=folder_path)
    image_paths = scan_run.outputs().o0

    if not image_paths:
        yield {"checked": 0, "total": 0, "matches": [], "done": True}
        return

    total   = len(image_paths)
    results = []
    for i, path in enumerate(image_paths):
        run    = flyte.run(check_match_task, image_path=path, query=query)
        result = run.outputs().o0
        results.append(result)
        yield {"checked": i + 1, "total": total, "done": False}

    collect_run = flyte.run(collect_matches_task, results=results)
    matches     = collect_run.outputs().o0
    yield {"checked": total, "total": total, "matches": matches, "done": True}
