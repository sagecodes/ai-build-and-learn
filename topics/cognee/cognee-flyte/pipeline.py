"""Flyte 2 source-ingest agent backed by Cognee, on a devbox.

Cognee's memory (SQLite + LanceDB + Ladybug) lives inside a `flyte.io.Dir`. Each
task downloads the prior memory, runs cognee against it, and uploads the new
state as a fresh Dir. So every ingest produces an immutable, addressable
revision of the memory in rustfs, and you can compound runs:

    run A  ──o0──▶  run B (--memory flyte://.../A/o0)  ──o0──▶  run C ...

The LLM behind cognee's extraction + answering is the in-cluster Gemma 4 vLLM
sibling app (config.VLLM_*); embeddings run locally via fastembed. This is the
"granular" half of the layered API — add -> cognify -> search(SearchType) —
so the graph build is explicit. The chat app uses cognee's simpler
remember/recall over the same store.

Usage:
    flyte run pipeline.py memory_pipeline                                # full demo
    flyte run pipeline.py init_memory
    flyte run pipeline.py ingest_source --memory <dir> --source <url>
    flyte run pipeline.py query_memory  --memory <dir> --question "..."
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import flyte
import flyte.io
import flyte.report

import cognee_lib
from config import pipeline_env

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = pipeline_env

# Default seed sources for memory_pipeline. Same RAG-adjacent Wikipedia trio as
# the llm-wiki demo so you can compare a knowledge-graph memory (cognee) against
# the LLM-maintained-wiki approach on identical inputs.
SEED_SOURCES: list[str] = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Vector_database",
    "https://en.wikipedia.org/wiki/Knowledge_graph",
]

# cognee groups data into named datasets; one is enough for the demo.
DATASET = "memory"


# ──────────────────────────────────────────────────────────────────────────────
# Shared per-task setup: fresh work dir, restore prior memory, configure cognee.
# ──────────────────────────────────────────────────────────────────────────────

async def _open_memory(memory: flyte.io.Dir | None, prefix: str) -> str:
    """Make a writable work dir, restore a prior memory Dir into it if given,
    and configure cognee to use it. Returns the local work-dir path.

    Order matters: configure_cognee() sets the env vars cognee reads at import,
    so it runs before the first `import cognee` below.
    """
    work = tempfile.mkdtemp(prefix=prefix)
    if memory is not None:
        restored = Path(await memory.download())
        shutil.copytree(restored, work, dirs_exist_ok=True)
    cognee_lib.configure_cognee(work)
    return work


# ──────────────────────────────────────────────────────────────────────────────
# init_memory: empty cognee store, pruned clean.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def init_memory() -> flyte.io.Dir:
    """Create a fresh, empty cognee memory and return it as a flyte.io.Dir."""
    work = await _open_memory(None, "cognee_init_")

    import cognee

    # Start from a clean slate so re-runs don't inherit a stray local store.
    await cognee.prune.prune_data()
    await cognee.prune.prune_system(metadata=True)

    stats = cognee_lib.storage_summary(work)
    await flyte.report.replace.aio(
        "<h2>Cognee memory initialized</h2>"
        f"<p>Empty store under <code>data/</code> + <code>system/</code> "
        f"({stats['files']} files, {stats['mb']} MB).</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(work)


# ──────────────────────────────────────────────────────────────────────────────
# ingest_source: add a source, cognify it into the graph, return new memory.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def ingest_source(
    memory: flyte.io.Dir,
    source: str,
    title_override: str = "",
) -> flyte.io.Dir:
    """Ingest one source (URL or pasted text) into the memory graph.

    add() ingests + chunks the raw text; cognify() builds embeddings, extracts
    entities/relationships, and writes the knowledge graph. Returns the updated
    memory as a new flyte.io.Dir.
    """
    work = await _open_memory(memory, "cognee_ingest_")

    import cognee

    title, text, source_url = cognee_lib.fetch_to_text(source)
    if title_override:
        title = title_override
    log.info(f"Ingesting: {title} ({source_url or 'pasted text'}) — {len(text)} chars")

    await cognee.add(text, dataset_name=DATASET)
    await cognee.cognify(datasets=[DATASET])

    # Pull the knowledge graph for the report. Best-effort: a viz failure must
    # not fail the ingest. The interactive HTML is also written into the work
    # dir so it travels inside the output flyte.io.Dir.
    n_nodes = n_edges = 0
    graph_table = "<p><i>Graph render unavailable.</i></p>"
    graph_iframe = ""
    try:
        from cognee.infrastructure.databases.graph import get_graph_engine

        graph_engine = await get_graph_engine()
        nodes, edges = await graph_engine.get_graph_data()
        n_nodes, n_edges = len(nodes), len(edges)
        graph_table = cognee_lib.graph_summary_html(nodes, edges)

        viz_html = await cognee.visualize_graph(str(Path(work) / "graph.html"))
        graph_iframe = cognee_lib.embed_graph_iframe(viz_html)
    except Exception as e:
        log.warning(f"Graph render failed (continuing): {type(e).__name__}: {e}")

    stats = cognee_lib.storage_summary(work)
    log.info(f"Memory now {n_nodes} nodes / {n_edges} edges, {stats['mb']} MB")
    await flyte.report.replace.aio(
        f"<h2>Ingested: {title}</h2>"
        f"<p><b>Source:</b> {source_url or '(pasted text)'}</p>"
        f"<p><b>Characters added:</b> {len(text):,} · "
        f"<b>Graph:</b> {n_nodes} nodes, {n_edges} relationships · "
        f"<b>Store:</b> {stats['mb']} MB (SQLite + LanceDB + Ladybug)</p>"
        f"<h3>Knowledge graph</h3>{graph_iframe}"
        f"<h3>Relationships extracted</h3>{graph_table}"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(work)


# ──────────────────────────────────────────────────────────────────────────────
# query_memory: ask the graph a question (graph + vector blended completion).
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def query_memory(memory: flyte.io.Dir, question: str) -> str:
    """Answer a question against the memory and return the text answer.

    SearchType.GRAPH_COMPLETION blends graph traversal with vector similarity,
    then asks the LLM to compose an answer — the thing cognee does that a plain
    vector store can't.
    """
    await _open_memory(memory, "cognee_query_")

    import cognee
    from cognee import SearchType

    results = await cognee.search(
        query_text=question, query_type=SearchType.GRAPH_COMPLETION
    )
    answer = (
        "\n\n".join(cognee_lib.result_text(r) for r in results)
        if results
        else "(no answer)"
    )

    await flyte.report.replace.aio(
        f"<h2>Query</h2><blockquote>{question}</blockquote>"
        f"<h3>Answer (GRAPH_COMPLETION)</h3><pre>{answer}</pre>"
    )
    await flyte.report.flush.aio()
    return answer


# ──────────────────────────────────────────────────────────────────────────────
# memory_pipeline: orchestrator. init, ingest each seed, query.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def memory_pipeline(
    sources: list[str] = SEED_SOURCES,
    question: str = "How do vector databases and knowledge graphs each support retrieval-augmented generation?",
) -> flyte.io.Dir:
    """End-to-end demo: init the memory, ingest each source, query the graph."""
    await flyte.report.replace.aio(
        "<h2>Cognee memory pipeline</h2><p>Step 1: initialising memory…</p>"
    )
    await flyte.report.flush.aio()
    memory = await init_memory()

    for i, src in enumerate(sources, 1):
        await flyte.report.replace.aio(
            f"<h2>Cognee memory pipeline</h2>"
            f"<p>Step {i + 1}: ingesting source {i}/{len(sources)}…</p>"
            f"<pre>{src}</pre>"
        )
        await flyte.report.flush.aio()
        memory = await ingest_source(memory, src)

    await flyte.report.replace.aio(
        f"<h2>Cognee memory pipeline</h2><p>Querying memory…</p>"
        f"<blockquote>{question}</blockquote>"
    )
    await flyte.report.flush.aio()
    answer = await query_memory(memory, question)

    await flyte.report.replace.aio(
        f"<h2>Cognee memory pipeline complete</h2>"
        f"<p><b>Sources ingested:</b> {len(sources)}</p>"
        f"<h3>Demo query</h3><blockquote>{question}</blockquote>"
        f"<pre>{answer}</pre>"
    )
    await flyte.report.flush.aio()
    return memory


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(memory_pipeline)
    print(f"Pipeline run: {run.name}")
    print(f"  {run.url}")
