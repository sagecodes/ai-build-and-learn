"""
workflows_union.py — Flyte tasks and runners for Union.ai remote backend.

Images are read locally, base64-encoded, and passed as strings to cluster tasks.
GCP credentials are injected via Union secrets — not read from local .env.
DB writes happen locally after results return from the cluster.
"""

import base64
import os
import tempfile
from pathlib import Path

import flyte

import db
import vision_service

_SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

env = flyte.TaskEnvironment(
    name="gemma4-smart-gallery",
    secrets=[
        flyte.Secret(key="GCP_PROJECT", as_env_var="GCP_PROJECT"),
        flyte.Secret(key="GCP_REGION",  as_env_var="GCP_REGION"),
        flyte.Secret(key="GEMMA_MODEL", as_env_var="GEMMA_MODEL"),
    ],
)


# ── Tasks ─────────────────────────────────────────────────────────────────────

@env.task
async def describe_image_bytes_task(image_b64: str, image_name: str) -> dict:
    """Decode base64 image, write to temp file, describe via Gemma 4."""
    suffix = Path(image_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(base64.b64decode(image_b64))
        temp_path = f.name
    try:
        description = vision_service.describe_image(temp_path)
    finally:
        os.unlink(temp_path)
    return {"name": image_name, "description": description}


@env.task
async def check_match_bytes_task(image_b64: str, image_name: str, query: str) -> dict:
    """Decode base64 image, write to temp file, check match via Gemma 4."""
    suffix = Path(image_name).suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(base64.b64decode(image_b64))
        temp_path = f.name
    try:
        matched = vision_service.check_image_match(temp_path, query)
    finally:
        os.unlink(temp_path)
    return {"name": image_name, "matched": matched}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _scan_local(folder_path: str) -> list[str]:
    """Scan folder on local machine — cluster tasks never touch local paths."""
    folder = Path(folder_path)
    return [
        str(p) for p in sorted(folder.iterdir())
        if p.suffix.lower() in _SUPPORTED_EXTENSIONS
    ]


# ── Runners ───────────────────────────────────────────────────────────────────

def run_describe(folder_path: str):
    """Yields one {path, description} dict per image. DB written locally."""
    image_paths = _scan_local(folder_path)

    if not image_paths:
        return

    db.init_db()
    for path in image_paths:
        image_b64 = base64.b64encode(Path(path).read_bytes()).decode()
        run       = flyte.run(
            describe_image_bytes_task,
            image_b64=image_b64,
            image_name=Path(path).name,
        )
        result = run.outputs().o0
        db.save_description(path, result["description"])
        yield {"path": path, "description": result["description"]}


def run_search(folder_path: str, query: str):
    """Yields progress dicts per image; final dict includes matches list."""
    image_paths = _scan_local(folder_path)

    if not image_paths:
        yield {"checked": 0, "total": 0, "matches": [], "done": True}
        return

    total   = len(image_paths)
    results = []
    for i, path in enumerate(image_paths):
        image_b64 = base64.b64encode(Path(path).read_bytes()).decode()
        run       = flyte.run(
            check_match_bytes_task,
            image_b64=image_b64,
            image_name=Path(path).name,
            query=query,
        )
        result = run.outputs().o0
        results.append({"path": path, "matched": result["matched"]})
        yield {"checked": i + 1, "total": total, "done": False}

    matches = [r["path"] for r in results if r["matched"]]
    yield {"checked": total, "total": total, "matches": matches, "done": True}
