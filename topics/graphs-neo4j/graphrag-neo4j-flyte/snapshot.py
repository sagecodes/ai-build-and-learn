"""Snapshot / restore the live Graph-RAG Neo4j into rustfs as a flyte.io.Dir.

Why this exists: every Neo4j pod restart wipes /data. The pipeline is one
source of truth (the Semantic Scholar fetch is cached, so re-running is
cheap), but anything you typed into the Neo4j browser between pipeline
runs would be gone. These two tasks let you snapshot the live graph to
the Flyte object store and replay it back later. The snapshot is a
`flyte.io.Dir` artifact, so it lives in the same rustfs as the pipeline
outputs and the prefetched Gemma weights, and it survives `flyte stop
devbox` / `flyte start devbox`.

Snapshot layout (one Dir):
    nodes.jsonl   one JSON per node: {"label": "...", "props": {...}}
                  Paper nodes include their `embedding` array.
    edges.jsonl   one JSON per edge: {"rel": "...",
                                      "src": [label, key, value],
                                      "dst": [label, key, value]}

Snapshot is online: we run pure Cypher over HTTP, no daemon stop required.
That's why this works on community edition (which has no online dump).

Usage:
    # Take a snapshot of the current graph.
    flyte run snapshot.py snapshot_neo4j

    # Replay it (run name from the snapshot CLI output, e.g. "abc1234..."):
    flyte run snapshot.py restore_neo4j \\
        --snapshot=flyte://flytesnacks/development/<snapshot-run-name>/o0

    # Smoke-test round trip (snapshot, immediately replay):
    flyte run snapshot.py snapshot_then_restore
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import flyte
import flyte.io
import flyte.report

from config import (
    NEO4J_HTTP_URL,
    NEO4J_PASSWORD,
    NEO4J_USER,
    pipeline_env,
)
from pipeline import EMBEDDING_DIM, _run_cypher

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = pipeline_env

# Stable natural keys per label, used to identify nodes across snapshot/restore.
# Matches the constraints created by load_neo4j (`pipeline.py`).
NATURAL_KEYS: dict[str, str] = {
    "Paper": "id",
    "Author": "name",
    "Category": "code",
}


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot: read live graph, write JSONL, return as a flyte.io.Dir.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def snapshot_neo4j(
    http_url: str = NEO4J_HTTP_URL,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
) -> flyte.io.Dir:
    """Dump the current Neo4j graph to JSONL files inside a flyte.io.Dir."""
    import httpx

    out_dir = Path(tempfile.mkdtemp(prefix="neo4j_snapshot_"))
    nodes_path = out_dir / "nodes.jsonl"
    edges_path = out_dir / "edges.jsonl"
    counts: dict[str, int] = {}

    with httpx.Client(base_url=http_url, auth=(user, password), timeout=60.0) as client:
        # Confirm we're talking to a real Neo4j and log version for the report.
        info = client.get("/").raise_for_status().json()
        log.info(f"Snapshotting Neo4j {info.get('neo4j_version')} "
                 f"{info.get('neo4j_edition')} at {http_url}")

        # Nodes: one MATCH per known label, including all properties.
        # `properties(n)` includes the 384-dim Paper.embedding array; bge-small
        # vectors are JSON-serializable lists of float so this just works.
        with nodes_path.open("w") as f:
            for label in NATURAL_KEYS:
                rows = _run_cypher(
                    client,
                    f"MATCH (n:{label}) RETURN properties(n) AS props",
                )
                for r in rows:
                    f.write(json.dumps({"label": label, "props": r[0]}) + "\n")
                counts[f"{label}_nodes"] = len(rows)

        # Edges: pull each known relation type with the natural keys of the
        # endpoints, so restore can MATCH them without elementId() (which is
        # not stable across DB instances).
        edge_specs = [
            ("CITES",        "Paper",  "id",   "Paper",    "id"),
            ("AUTHORED_BY",  "Paper",  "id",   "Author",   "name"),
            ("IN_CATEGORY",  "Paper",  "id",   "Category", "code"),
        ]
        with edges_path.open("w") as f:
            for rel, src_label, src_key, dst_label, dst_key in edge_specs:
                rows = _run_cypher(client, f"""
                    MATCH (a:{src_label})-[:{rel}]->(b:{dst_label})
                    RETURN a.{src_key} AS src, b.{dst_key} AS dst
                """)
                for r in rows:
                    f.write(json.dumps({
                        "rel": rel,
                        "src": [src_label, src_key, r[0]],
                        "dst": [dst_label, dst_key, r[1]],
                    }) + "\n")
                counts[f"{rel}_edges"] = len(rows)

    log.info(f"Snapshot written to {out_dir}: {counts}")
    rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in counts.items())
    await flyte.report.replace.aio(
        f"<h2>Neo4j snapshot</h2>"
        f"<p><b>Source:</b> <code>{http_url}</code></p>"
        f"<table border=1 cellpadding=4>"
        f"<tr><th>Entity</th><th>Count</th></tr>{rows_html}</table>"
        f"<p>Wrote <code>nodes.jsonl</code> + <code>edges.jsonl</code> to a "
        f"<code>flyte.io.Dir</code>. Pass this run's output to "
        f"<code>restore_neo4j</code> to replay.</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Restore: replay a snapshot Dir back into Neo4j.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def restore_neo4j(
    snapshot: flyte.io.Dir,
    http_url: str = NEO4J_HTTP_URL,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    wipe_first: bool = True,
) -> dict[str, int]:
    """MERGE every node + edge from a snapshot Dir back into Neo4j."""
    import httpx

    snap_path = Path(await snapshot.download())
    nodes_path = snap_path / "nodes.jsonl"
    edges_path = snap_path / "edges.jsonl"
    if not nodes_path.exists() or not edges_path.exists():
        raise FileNotFoundError(
            f"Snapshot dir {snap_path} missing nodes.jsonl or edges.jsonl"
        )

    counts: dict[str, int] = {"nodes": 0, "edges": 0}
    with httpx.Client(base_url=http_url, auth=(user, password), timeout=60.0) as client:
        if wipe_first:
            _run_cypher(client, "MATCH (n) DETACH DELETE n")
            _run_cypher(client, "DROP INDEX paper_embedding_idx IF EXISTS")
            log.info("Wiped existing graph + vector index")

        # Constraints first; otherwise MERGEs fall back to seq-scan and dupe.
        _run_cypher(client, "CREATE CONSTRAINT paper_id IF NOT EXISTS "
                            "FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        _run_cypher(client, "CREATE CONSTRAINT author_name IF NOT EXISTS "
                            "FOR (a:Author) REQUIRE a.name IS UNIQUE")
        _run_cypher(client, "CREATE CONSTRAINT category_code IF NOT EXISTS "
                            "FOR (c:Category) REQUIRE c.code IS UNIQUE")

        # Nodes.
        for line in nodes_path.open():
            row = json.loads(line)
            label = row["label"]
            props = row["props"]
            if label not in NATURAL_KEYS:
                raise ValueError(f"Unknown node label in snapshot: {label}")
            key = NATURAL_KEYS[label]
            _run_cypher(
                client,
                # Label and key are from a closed whitelist (NATURAL_KEYS), so
                # f-stringing them in is safe. The actual values are still
                # parameterized.
                f"MERGE (n:{label} {{{key}: $key_val}}) SET n = $props",
                {"key_val": props[key], "props": props},
            )
            counts["nodes"] += 1

        # Edges.
        for line in edges_path.open():
            row = json.loads(line)
            rel = row["rel"]
            src_label, src_key, src_val = row["src"]
            dst_label, dst_key, dst_val = row["dst"]
            # Same whitelist argument applies to label/key/rel.
            _run_cypher(
                client,
                f"""
                MATCH (a:{src_label} {{{src_key}: $sv}}),
                      (b:{dst_label} {{{dst_key}: $dv}})
                MERGE (a)-[:{rel}]->(b)
                """,
                {"sv": src_val, "dv": dst_val},
            )
            counts["edges"] += 1

        # Re-create the vector index. The embeddings were restored as part of
        # the Paper.embedding property; the index just needs to be told where
        # to scan.
        _run_cypher(client, f"""
            CREATE VECTOR INDEX paper_embedding_idx IF NOT EXISTS
            FOR (p:Paper) ON (p.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

    log.info(f"Restored: {counts}")
    rows_html = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in counts.items())
    await flyte.report.replace.aio(
        f"<h2>Neo4j restore</h2>"
        f"<p><b>Target:</b> <code>{http_url}</code></p>"
        f"<table border=1 cellpadding=4>"
        f"<tr><th>Entity</th><th>Count</th></tr>{rows_html}</table>"
    )
    await flyte.report.flush.aio()
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Smoke test: snapshot then immediately restore. Useful for catching regressions
# in the JSONL schema before relying on it across a pod restart.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def snapshot_then_restore() -> dict[str, int]:
    snap = await snapshot_neo4j()
    return await restore_neo4j(snap)


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(snapshot_neo4j)
    print(f"Snapshot run: {run.name}")
    print(f"  {run.url}")
