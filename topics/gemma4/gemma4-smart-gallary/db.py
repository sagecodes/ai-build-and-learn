"""
db.py — SQLite operations for the image description cache.

All database reads and writes live here.
Swap this file to change storage backend without touching other modules.
"""

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).parent / "gemma_photos.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS images (
    filepath     TEXT PRIMARY KEY,
    filename     TEXT NOT NULL,
    description  TEXT NOT NULL,
    processed_at TEXT NOT NULL
);
"""


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(_SCHEMA)


def is_cached(filepath: str) -> bool:
    with _connect() as conn:
        row = conn.execute(
            "SELECT 1 FROM images WHERE filepath = ?", (filepath,)
        ).fetchone()
    return row is not None


def save_description(filepath: str, description: str) -> None:
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO images (filepath, filename, description, processed_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(filepath) DO UPDATE SET
                description  = excluded.description,
                processed_at = excluded.processed_at
            """,
            (
                filepath,
                Path(filepath).name,
                description,
                datetime.now(timezone.utc).isoformat(),
            ),
        )


def get_all() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT filepath, filename, description FROM images ORDER BY filename"
        ).fetchall()
    return [dict(row) for row in rows]


def clear() -> None:
    with _connect() as conn:
        conn.execute("DELETE FROM images")
