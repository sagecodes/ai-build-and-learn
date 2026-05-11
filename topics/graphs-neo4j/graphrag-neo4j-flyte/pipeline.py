"""Flyte 2 Graph RAG pipeline: real AI-paper graph → Neo4j with vector index.

Fetches papers from Semantic Scholar by relevance for a keyword query, embeds
abstracts with bge-small, and loads nodes + edges into Neo4j over the HTTP
Cypher API. The fetch task is cached on (query, max_papers) so re-runs are
free until you change either argument.

The pipeline writes:
  - (Paper {id, title, abstract, year, url, embedding})  embedding = 384-dim bge-small
  - (Author {name})
  - (Category {code})
  - (:Paper)-[:AUTHORED_BY]->(:Author)
  - (:Paper)-[:IN_CATEGORY]->(:Category)
  - (:Paper)-[:CITES]->(:Paper)
  - VECTOR INDEX paper_embedding_idx ON (:Paper.embedding)  cosine, 384 dims

Usage:
    flyte run pipeline.py graphrag_pipeline                 # remote (devbox)
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
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

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = pipeline_env

EMBEDDING_DIM = 384  # bge-small-en-v1.5
S2_BASE = "https://api.semanticscholar.org/graph/v1"


# ──────────────────────────────────────────────────────────────────────────────
# Task 1 — fetch papers from Semantic Scholar for a keyword query.
#
# Uses the /paper/search/bulk endpoint (one call, up to 1000 results) sorted
# by citationCount desc. The relevance-ranked /paper/search endpoint paginates
# at 100 per page and gets aggressively rate-limited on the anonymous tier;
# bulk gives us the whole result in a single request and tends to clear the
# rate gate on the first try. Citation-sorted is also a stronger demo signal:
# the corpus is anchored on the most-cited papers matching the query.
#
# Cached on (query, max_papers): re-runs are free until you change either.
# Set S2_API_KEY in the env to skip the anonymous shared rate limit.
# ──────────────────────────────────────────────────────────────────────────────


def _paper_url(paper: dict[str, Any]) -> str:
    """Prefer arXiv → openAccessPdf → S2 paper page."""
    ext = paper.get("externalIds") or {}
    if arxiv_id := ext.get("ArXiv"):
        return f"https://arxiv.org/abs/{arxiv_id}"
    if oa := (paper.get("openAccessPdf") or {}).get("url"):
        return oa
    return f"https://www.semanticscholar.org/paper/{paper['paperId']}"


def _s2_request(
    client: "httpx.Client",
    method: str,
    url: str,
    *,
    params: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    max_attempts: int = 6,
) -> "httpx.Response":
    """HTTP request with retry on 429 / 5xx. Honors Retry-After when present.

    S2's unauthenticated rate limit is shared across all anonymous callers,
    so 429s are routine without an API key. Set S2_API_KEY in the env to
    avoid the queue.
    """
    for attempt in range(max_attempts):
        r = client.request(method, url, params=params, json=json_body)
        if r.status_code == 429 or 500 <= r.status_code < 600:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
            log.info(
                f"S2 {r.status_code} on attempt {attempt + 1}/{max_attempts}, "
                f"sleeping {wait:.1f}s"
            )
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r
    # Final attempt: let the exception propagate.
    r.raise_for_status()
    return r


@env.task(cache="auto", report=True)
async def fetch_papers(
    query: str = "retrieval augmented generation language models",
    max_papers: int = 400,
) -> flyte.io.File:
    """Fetch up to `max_papers` papers from Semantic Scholar for `query`.

    Citation edges are kept only when both endpoints are inside the result
    set, so the graph is internally consistent. Papers without an abstract
    are dropped (we can't embed them).

    Returns a `flyte.io.File` (JSONL, one paper per line) so the payload
    rides through Flyte's object store. A pickled `list[dict]` would blow
    past the default 2MB output cap once we're past ~150 papers.
    """
    import httpx

    headers: dict[str, str] = {}
    if api_key := os.environ.get("S2_API_KEY"):
        headers["x-api-key"] = api_key

    # Step 1: bulk search for papers + metadata. The bulk endpoint does NOT
    # accept `references.*` in `fields` (returns 400), so references come
    # through a separate /paper/batch call below.
    bulk_fields = ",".join([
        "paperId", "title", "abstract", "year",
        "authors.name", "fieldsOfStudy",
        "externalIds", "openAccessPdf",
        "citationCount",
    ])
    # Bulk caps at 1000 per page; pull headroom because we'll drop papers
    # missing abstracts. For larger corpora we'd follow the response `token`
    # for the next page.
    bulk_limit = min(max(max_papers * 2, max_papers), 1000)
    with httpx.Client(timeout=60.0, headers=headers) as client:
        r = _s2_request(
            client, "GET",
            f"{S2_BASE}/paper/search/bulk",
            params={
                "query": query,
                "fields": bulk_fields,
                "sort": "citationCount:desc",
                "limit": bulk_limit,
            },
        )
        raw = r.json().get("data") or []

        # Drop papers we can't use (no abstract → can't embed) and trim.
        # The in-set lookup is built *after* the trim so citation edges to
        # dropped neighbors are filtered out downstream.
        valid = [p for p in raw if p.get("paperId") and p.get("abstract") and p.get("title")]
        valid = valid[:max_papers]
        in_set = {p["paperId"] for p in valid}

        # Step 2: batch fetch references.paperId for the trimmed set. The
        # batch endpoint accepts up to 500 IDs per POST. If this fails (rate
        # limit, transient error), we still ship a graph — just without
        # CITES edges. Modes 1 / 2 still work via abstracts, AUTHORED_BY,
        # IN_CATEGORY; mode 3 RRF still works via the category cohort.
        refs_by_id: dict[str, list[str]] = {}
        ids = [p["paperId"] for p in valid]
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            try:
                rr = _s2_request(
                    client, "POST",
                    f"{S2_BASE}/paper/batch",
                    params={"fields": "references.paperId"},
                    json_body={"ids": chunk},
                )
                for pd in rr.json() or []:
                    if pd and pd.get("paperId"):
                        refs_by_id[pd["paperId"]] = [
                            ref["paperId"]
                            for ref in (pd.get("references") or [])
                            if ref and ref.get("paperId")
                        ]
            except Exception as e:
                log.warning(
                    f"References batch ({len(chunk)} ids) failed after retries: "
                    f"{e}. Continuing without CITES edges for that batch."
                )

    papers: list[dict[str, Any]] = []
    for p in valid:
        pid = p["paperId"]
        cites = [
            r for r in refs_by_id.get(pid, [])
            if r in in_set and r != pid
        ]
        papers.append({
            "id": pid,
            "title": p["title"],
            "abstract": p["abstract"],
            "year": p.get("year") or 0,
            "authors": [a["name"] for a in (p.get("authors") or []) if a.get("name")],
            "categories": p.get("fieldsOfStudy") or [],
            "cites": cites,
            "url": _paper_url(p),
        })

    n_papers = len(papers)
    n_authors = len({a for p in papers for a in p["authors"]})
    n_cats = len({c for p in papers for c in p["categories"]})
    n_cites = sum(len(p["cites"]) for p in papers)

    log.info(
        f"S2 query '{query}': {n_papers} papers, {n_authors} authors, "
        f"{n_cats} categories, {n_cites} in-set citations"
    )

    out_path = Path(tempfile.mkdtemp(prefix="graphrag_papers_")) / "papers.jsonl"
    with out_path.open("w") as f:
        for p in papers:
            f.write(json.dumps(p) + "\n")

    await flyte.report.replace.aio(
        f"<h2>Fetched papers (Semantic Scholar)</h2>"
        f"<p><b>Query:</b> <code>{query}</code></p>"
        f"<ul>"
        f"<li><b>Papers:</b> {n_papers}</li>"
        f"<li><b>Authors:</b> {n_authors}</li>"
        f"<li><b>Categories:</b> {n_cats}</li>"
        f"<li><b>Citation edges (in-set):</b> {n_cites}</li>"
        f"</ul>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.File.from_local(str(out_path))


# ──────────────────────────────────────────────────────────────────────────────
# Task 2 — embed each paper's abstract with bge-small.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def embed_papers(
    papers_file: flyte.io.File,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
) -> flyte.io.File:
    """Add an `embedding` (list[float], len=384) field to each paper.

    I/O is JSONL via `flyte.io.File` to stay under the 2MB output cap.
    With 384-dim float embeddings, the embedded JSONL is ~3-4× the input
    size, so even at 200 papers a pickled return would exceed the limit.
    """
    from sentence_transformers import SentenceTransformer

    local_in = await papers_file.download()
    papers = [json.loads(line) for line in Path(local_in).open()]

    log.info(f"Embedding {len(papers)} abstracts with {embedding_model}")
    encoder = SentenceTransformer(embedding_model)
    texts = [f"{p['title']}. {p['abstract']}" for p in papers]
    # bge-small expects L2-normalized vectors paired with cosine similarity.
    vectors = encoder.encode(
        texts, normalize_embeddings=True, convert_to_numpy=True
    ).tolist()
    dim = len(vectors[0]) if vectors else 0
    log.info(f"Produced {len(vectors)} embeddings of dim {dim}")

    out_path = Path(tempfile.mkdtemp(prefix="graphrag_embedded_")) / "papers_embedded.jsonl"
    with out_path.open("w") as f:
        for p, v in zip(papers, vectors):
            f.write(json.dumps({**p, "embedding": v}) + "\n")

    await flyte.report.replace.aio(
        f"<h2>Embedded abstracts</h2>"
        f"<p><b>Model:</b> {embedding_model}</p>"
        f"<p><b>Dimensions:</b> {dim}</p>"
        f"<p><b>Vectors:</b> {len(vectors)}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.File.from_local(str(out_path))


# ──────────────────────────────────────────────────────────────────────────────
# Task 3 — write nodes/edges/vector-index to Neo4j over the HTTP Cypher API.
#
# Why HTTP and not Bolt: Flyte 2 fronts apps with Knative Serving's queue-proxy,
# which is HTTP-only. Bolt (TCP/7687) doesn't pass through. Neo4j's HTTP API on
# 7474 supports the full Cypher surface, including vector index queries.
# ──────────────────────────────────────────────────────────────────────────────

def _run_cypher(
    client: "httpx.Client",
    cypher: str,
    params: dict[str, Any] | None = None,
) -> list[list[Any]]:
    """POST one Cypher statement to /db/neo4j/tx/commit and return the rows.

    Returns the `data[*].row` arrays from the first result. Raises if Neo4j
    reports any error in the response payload (HTTP 200 with `errors[]` set).
    """
    payload = {"statements": [{"statement": cypher, "parameters": params or {}}]}
    resp = client.post("/db/neo4j/tx/commit", json=payload)
    resp.raise_for_status()
    body = resp.json()
    if body.get("errors"):
        raise RuntimeError(f"Neo4j error: {body['errors']}")
    results = body.get("results", [])
    if not results:
        return []
    return [row["row"] for row in results[0].get("data", [])]


@env.task(report=True)
async def load_neo4j(
    papers_file: flyte.io.File,
    http_url: str = NEO4J_HTTP_URL,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    wipe_first: bool = True,
) -> dict[str, int]:
    """MERGE nodes + edges into Neo4j and (re)create the vector index."""
    import httpx

    local = await papers_file.download()
    papers = [json.loads(line) for line in Path(local).open()]

    log.info(f"Connecting to Neo4j HTTP API at {http_url} as {user}")
    # The Knative service URL points to port 80 → queue-proxy → user port 7474.
    # Basic auth header on every request.
    with httpx.Client(
        base_url=http_url,
        auth=(user, password),
        timeout=30.0,
    ) as client:
        # Smoke-check: discovery doc at GET /. Confirms auth, version, routing.
        info = client.get("/").raise_for_status().json()
        log.info(f"Neo4j {info.get('neo4j_version')} {info.get('neo4j_edition')}")

        if wipe_first:
            _run_cypher(client, "MATCH (n) DETACH DELETE n")
            _run_cypher(client, "DROP INDEX paper_embedding_idx IF EXISTS")
            log.info("Wiped existing graph + vector index")

        # Constraints make MERGE-by-key fast and prevent dupes.
        _run_cypher(client, "CREATE CONSTRAINT paper_id IF NOT EXISTS "
                            "FOR (p:Paper) REQUIRE p.id IS UNIQUE")
        _run_cypher(client, "CREATE CONSTRAINT author_name IF NOT EXISTS "
                            "FOR (a:Author) REQUIRE a.name IS UNIQUE")
        _run_cypher(client, "CREATE CONSTRAINT category_code IF NOT EXISTS "
                            "FOR (c:Category) REQUIRE c.code IS UNIQUE")

        # All upserts go through UNWIND so each entity type is one HTTP round
        # trip regardless of corpus size. With 400 papers this drops total
        # round trips from ~6k to ~6.
        paper_rows = [
            {
                "id": p["id"],
                "title": p["title"],
                "abstract": p["abstract"],
                "year": p["year"],
                "url": p.get("url", ""),
                "embedding": p["embedding"],
            }
            for p in papers
        ]
        _run_cypher(client, """
            UNWIND $rows AS row
            MERGE (p:Paper {id: row.id})
            SET p.title = row.title,
                p.abstract = row.abstract,
                p.year = row.year,
                p.url = row.url,
                p.embedding = row.embedding
        """, {"rows": paper_rows})

        author_rows = [
            {"name": a, "pid": p["id"]} for p in papers for a in p["authors"]
        ]
        if author_rows:
            _run_cypher(client, """
                UNWIND $rows AS row
                MERGE (a:Author {name: row.name})
                WITH a, row
                MATCH (p:Paper {id: row.pid})
                MERGE (p)-[:AUTHORED_BY]->(a)
            """, {"rows": author_rows})

        category_rows = [
            {"code": c, "pid": p["id"]} for p in papers for c in p["categories"]
        ]
        if category_rows:
            _run_cypher(client, """
                UNWIND $rows AS row
                MERGE (c:Category {code: row.code})
                WITH c, row
                MATCH (p:Paper {id: row.pid})
                MERGE (p)-[:IN_CATEGORY]->(c)
            """, {"rows": category_rows})

        cite_rows = [
            {"src": p["id"], "dst": cited}
            for p in papers
            for cited in p["cites"]
        ]
        if cite_rows:
            _run_cypher(client, """
                UNWIND $rows AS row
                MATCH (src:Paper {id: row.src}), (dst:Paper {id: row.dst})
                MERGE (src)-[:CITES]->(dst)
            """, {"rows": cite_rows})

        # Native vector index — Neo4j 5.11+. Cosine matches the L2-normalized
        # bge-small embeddings produced upstream.
        _run_cypher(client, f"""
            CREATE VECTOR INDEX paper_embedding_idx IF NOT EXISTS
            FOR (p:Paper) ON (p.embedding)
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: {EMBEDDING_DIM},
                `vector.similarity_function`: 'cosine'
            }}}}
        """)

        counts = {
            "papers": _run_cypher(client, "MATCH (p:Paper) RETURN count(p)")[0][0],
            "authors": _run_cypher(client, "MATCH (a:Author) RETURN count(a)")[0][0],
            "categories": _run_cypher(client, "MATCH (c:Category) RETURN count(c)")[0][0],
            "cites_edges": _run_cypher(
                client, "MATCH ()-[r:CITES]->() RETURN count(r)"
            )[0][0],
            "authored_edges": _run_cypher(
                client, "MATCH ()-[r:AUTHORED_BY]->() RETURN count(r)"
            )[0][0],
        }

    log.info(f"Loaded into Neo4j: {counts}")
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in counts.items())
    await flyte.report.replace.aio(
        f"<h2>Loaded into Neo4j</h2>"
        f"<p><b>HTTP URL:</b> <code>{http_url}</code></p>"
        f"<table border=1 cellpadding=4>"
        f"<tr><th>Entity</th><th>Count</th></tr>{rows}</table>"
        f"<p>Vector index <code>paper_embedding_idx</code> "
        f"({EMBEDDING_DIM} dims, cosine) created.</p>"
    )
    await flyte.report.flush.aio()
    return counts


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline orchestrator.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def graphrag_pipeline(
    query: str = "retrieval augmented generation language models",
    max_papers: int = 400,
    embedding_model: str = "BAAI/bge-small-en-v1.5",
    http_url: str = NEO4J_HTTP_URL,
    user: str = NEO4J_USER,
    password: str = NEO4J_PASSWORD,
    wipe_first: bool = True,
) -> dict[str, int]:
    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline</h2><p>Step 1/3 — fetching papers…</p>"
    )
    await flyte.report.flush.aio()
    papers_file = await fetch_papers(query, max_papers)

    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline</h2><p>Step 2/3 — embedding abstracts…</p>"
    )
    await flyte.report.flush.aio()
    embedded_file = await embed_papers(papers_file, embedding_model)

    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline</h2><p>Step 3/3 — loading Neo4j…</p>"
    )
    await flyte.report.flush.aio()
    counts = await load_neo4j(embedded_file, http_url, user, password, wipe_first)

    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in counts.items())
    await flyte.report.replace.aio(
        "<h2>Graph RAG pipeline complete</h2>"
        f"<table border=1 cellpadding=4>"
        f"<tr><th>Entity</th><th>Count</th></tr>{rows}</table>"
    )
    await flyte.report.flush.aio()
    return counts


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(graphrag_pipeline)
    print(f"Pipeline run: {run.name}")
    print(f"  {run.url}")
