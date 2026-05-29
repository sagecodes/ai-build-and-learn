"""Flyte-agnostic helpers shared by pipeline.py and chat_app.py.

Nothing here imports `flyte`; `cognee` is imported lazily *after*
configure_cognee() has set the env vars it reads at import time. That split is
deliberate: the pipeline passes cognee's storage as a flyte.io.Dir and the chat
app tars it to HF, but both drive cognee through the exact same calls.

Cognee storage model (what we snapshot):
    <work>/data     DATA_ROOT_DIRECTORY    SQLite (relational) + LanceDB (vectors)
    <work>/system   SYSTEM_ROOT_DIRECTORY  Ladybug (graph) + bookkeeping
Tar the <work> dir and you have the whole memory; untar it elsewhere, point
cognee at the same two subdirs, and the graph + vectors come back.
"""

from __future__ import annotations

import os
import tarfile
from pathlib import Path

from config import (
    COGNEE_DATA_SUBDIR,
    COGNEE_SYSTEM_SUBDIR,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    HF_MEMORY_FILENAME,
    VLLM_MODEL_ID,
    VLLM_URL,
)


# ──────────────────────────────────────────────────────────────────────────────
# Cognee configuration. Call BEFORE `import cognee` anywhere in the process.
# ──────────────────────────────────────────────────────────────────────────────

def configure_cognee(work_dir: str) -> tuple[str, str]:
    """Point cognee at the in-cluster vLLM + a local fastembed encoder + a
    storage root under ``work_dir``. Returns ``(data_dir, system_dir)``.

    cognee reads these env vars when its config module is first imported, so
    this must run before the first ``import cognee`` in the process. The
    pipeline tasks and the chat-app server both call it as their first line.
    """
    data_dir = os.path.join(work_dir, COGNEE_DATA_SUBDIR)
    system_dir = os.path.join(work_dir, COGNEE_SYSTEM_SUBDIR)
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(system_dir, exist_ok=True)

    # cognee's setup_logging() indexes a name->level dict with $LOG_LEVEL and
    # crashes (KeyError) if it's numeric. Flyte sets LOG_LEVEL=30 (WARNING) in
    # task pods, so normalize to a level *name* before cognee is imported.
    _level_names = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}
    _numeric_to_name = {
        "50": "CRITICAL", "40": "ERROR", "30": "WARNING",
        "20": "INFO", "10": "DEBUG", "0": "NOTSET",
    }
    _lvl = os.environ.get("LOG_LEVEL", "")
    if _lvl in _numeric_to_name:
        os.environ["LOG_LEVEL"] = _numeric_to_name[_lvl]
    elif _lvl.upper() not in _level_names:
        os.environ["LOG_LEVEL"] = "INFO"

    # LLM: Gemma 4 vLLM as a custom OpenAI-compatible endpoint. The "openai/"
    # prefix tells litellm (cognee's LLM layer) to speak the OpenAI protocol;
    # LLM_ENDPOINT becomes api_base. The key is unused by vLLM but litellm
    # rejects an empty one.
    os.environ["LLM_PROVIDER"] = "custom"
    os.environ["LLM_MODEL"] = f"openai/{VLLM_MODEL_ID}"
    os.environ["LLM_ENDPOINT"] = VLLM_URL.rstrip("/") + "/v1"
    os.environ.setdefault("LLM_API_KEY", "not-used")

    # Run headless. cognee 1.x defaults to multi-user access control +
    # required auth, which expects a User object on every call; disabling it
    # lets the pipeline tasks and the app server operate as a single tenant.
    os.environ["ENABLE_BACKEND_ACCESS_CONTROL"] = "false"

    # Embeddings: local fastembed, no second endpoint. If only the LLM were
    # configured, cognee silently falls back to OpenAI embeddings (and 401s
    # without an OpenAI key), so we set the embedding side explicitly.
    os.environ["EMBEDDING_PROVIDER"] = EMBEDDING_PROVIDER
    os.environ["EMBEDDING_MODEL"] = EMBEDDING_MODEL
    os.environ["EMBEDDING_DIMENSIONS"] = str(EMBEDDING_DIMENSIONS)

    # Storage roots -> one tar-able subtree per memory.
    os.environ["DATA_ROOT_DIRECTORY"] = data_dir
    os.environ["SYSTEM_ROOT_DIRECTORY"] = system_dir

    return data_dir, system_dir


# ──────────────────────────────────────────────────────────────────────────────
# Snapshot / restore: local tar round-trip. flyte.io.Dir and HF both build on
# these — the pipeline wraps a Dir around `work_dir`, the chat app tars it.
# ──────────────────────────────────────────────────────────────────────────────

def tar_work_dir(work_dir: str, tar_path: str) -> str:
    """Tar the cognee storage root (``data/`` + ``system/``) to ``tar_path``."""
    if not os.path.isdir(work_dir):
        raise RuntimeError(f"Cannot snapshot: {work_dir} doesn't exist")
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(work_dir, arcname=".")
    return tar_path


def untar_into(tar_path: str, work_dir: str) -> None:
    """Extract a snapshot tarball into ``work_dir`` (created if missing)."""
    os.makedirs(work_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(work_dir)


# ──────────────────────────────────────────────────────────────────────────────
# HuggingFace checkpoint (chat app scale-to-zero survival). Lifted from the
# chroma agent-memory demo; 404 / missing-file is treated as "start fresh".
# ──────────────────────────────────────────────────────────────────────────────

def pull_memory_from_hf(hf_repo: str, repo_type: str, work_dir: str) -> bool:
    """Download + extract the HF snapshot into ``work_dir``. Returns True if a
    prior snapshot was restored, False on first run / missing file."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    os.makedirs(work_dir, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=hf_repo, repo_type=repo_type, filename=HF_MEMORY_FILENAME
        )
    except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
        print(f"[cognee-mem] No prior snapshot at {hf_repo} — starting fresh.", flush=True)
        return False
    except Exception as e:
        msg = str(e)
        if "404" in msg or "not found" in msg.lower():
            print(f"[cognee-mem] No prior snapshot ({type(e).__name__}) — starting fresh.", flush=True)
            return False
        raise

    untar_into(path, work_dir)
    print(f"[cognee-mem] Restored snapshot from {hf_repo} -> {work_dir}", flush=True)
    return True


def push_memory_to_hf(hf_repo: str, repo_type: str, work_dir: str, note: str = "") -> str:
    """Tar ``work_dir`` and upload it as the HF snapshot. Returns the commit URL."""
    import datetime as _dt
    import tempfile
    from huggingface_hub import upload_file

    fd, tar_path = tempfile.mkstemp(prefix="cognee_memory_", suffix=".tar.gz")
    os.close(fd)
    tar_work_dir(work_dir, tar_path)

    msg = f"cognee-memory snapshot {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z"
    if note:
        msg += f" — {note}"
    commit = upload_file(
        path_or_fileobj=tar_path,
        path_in_repo=HF_MEMORY_FILENAME,
        repo_id=hf_repo,
        repo_type=repo_type,
        commit_message=msg,
    )
    os.remove(tar_path)
    url = getattr(commit, "commit_url", str(commit))
    print(f"[cognee-mem] Pushed snapshot to {hf_repo} ({url})", flush=True)
    return url


# ──────────────────────────────────────────────────────────────────────────────
# Source fetch: URL -> clean markdown (trafilatura), or pass text through.
# Mirrors wiki_lib.fetch_to_markdown so behaviour matches the sibling demo.
# ──────────────────────────────────────────────────────────────────────────────

def fetch_to_text(source: str) -> tuple[str, str, str]:
    """Return ``(title, text, source_url)`` for a URL or a pasted text blob.

    URLs are fetched and run through trafilatura; anything that doesn't look
    like a URL is treated as already-clean text (source_url == "").
    """
    source = source.strip()
    if not (source.startswith("http://") or source.startswith("https://")):
        title = source.splitlines()[0][:80] if source else "pasted text"
        return title, source, ""

    import trafilatura

    downloaded = trafilatura.fetch_url(source)
    if not downloaded:
        raise RuntimeError(f"Could not fetch {source}")
    text = trafilatura.extract(downloaded, include_links=False, include_comments=False) or ""
    meta = trafilatura.extract_metadata(downloaded)
    title = (getattr(meta, "title", None) if meta else None) or source
    if not text.strip():
        raise RuntimeError(f"trafilatura extracted no text from {source}")
    return title, text, source


# ──────────────────────────────────────────────────────────────────────────────
# Storage introspection for reports/UI. Best-effort, never raises.
# ──────────────────────────────────────────────────────────────────────────────

def result_text(entry) -> str:
    """Pull readable text out of a cognee search/recall result entry.

    cognee 1.x returns Pydantic models, not strings:
      - cognee.search -> SearchResult(search_result=...)
      - cognee.recall -> ResponseQAEntry(answer=...), ResponseGraphContextEntry
        (content=...), ResponseGraphEntry(text=...), ...
    This flattens any of them to a string so callers can join + render.
    """
    if hasattr(entry, "search_result"):
        sr = entry.search_result
        if isinstance(sr, (list, tuple)):
            return "\n".join(result_text(x) for x in sr)
        return str(sr)
    for attr in ("answer", "content", "text"):
        val = getattr(entry, attr, None)
        if val:
            return str(val)
    return str(entry)


def _esc(s) -> str:
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def embed_graph_iframe(viz_html: str, height: int = 620) -> str:
    """Wrap cognee's standalone graph HTML in a base64 data: iframe.

    cognee.visualize_graph() returns a full <!DOCTYPE html> document that loads
    d3 from a CDN. We can't drop a whole document into the report HTML, so embed
    it as an iframe via a base64 data URI (no quote-escaping headaches). No
    sandbox attribute, so the d3 script runs. The report viewer's browser
    fetches d3 from the CDN, so this needs network at view time.
    """
    import base64

    b64 = base64.b64encode(viz_html.encode("utf-8")).decode("ascii")
    return (
        f'<iframe src="data:text/html;base64,{b64}" '
        f'style="width:100%;height:{height}px;border:1px solid #ccc;border-radius:8px;" '
        f'title="cognee knowledge graph"></iframe>'
    )


def graph_summary_html(nodes, edges, limit: int = 25) -> str:
    """Static (no-JS) relationship table from cognee get_graph_data() output.

    Always renders, so it's the fallback if the interactive iframe is stripped.
    nodes: list of (node_id, node_info{name,type,...});
    edges: list of (source, target, relation, edge_info).
    """
    name = {}
    for nid, info in nodes or []:
        name[str(nid)] = (info or {}).get("name") or str(nid)

    rows = []
    for e in (edges or [])[:limit]:
        src, tgt, rel = str(e[0]), str(e[1]), e[2]
        rows.append(
            f"<tr><td>{_esc(name.get(src, src))}</td>"
            f"<td><i>{_esc(rel)}</i></td>"
            f"<td>{_esc(name.get(tgt, tgt))}</td></tr>"
        )
    more = "" if len(edges or []) <= limit else f"<p><i>… and {len(edges) - limit} more relationships</i></p>"
    if not rows:
        return "<p><i>No relationships extracted yet.</i></p>"
    return (
        "<table style='border-collapse:collapse;font-size:0.85rem;'>"
        "<thead><tr><th style='text-align:left;padding:2px 10px'>Source</th>"
        "<th style='text-align:left;padding:2px 10px'>Relationship</th>"
        "<th style='text-align:left;padding:2px 10px'>Target</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>{more}"
    )


def storage_summary(work_dir: str) -> dict:
    """Cheap on-disk stats so reports show the memory growing without us having
    to crack open Ladybug. Counts files + total bytes under the storage root."""
    n_files = 0
    total_bytes = 0
    for p in Path(work_dir).rglob("*"):
        if p.is_file():
            n_files += 1
            try:
                total_bytes += p.stat().st_size
            except OSError:
                pass
    return {"files": n_files, "bytes": total_bytes, "mb": round(total_bytes / 1e6, 2)}
