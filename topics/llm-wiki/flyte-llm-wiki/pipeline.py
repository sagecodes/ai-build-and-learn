"""Flyte 2 LLM Wiki pipeline: Karpathy's pattern, on a devbox.

The wiki is a `flyte.io.Dir` that lives in rustfs and gets passed from task to
task. Each ingest produces a new version of the Dir; the LLM is the only
thing that writes to the `pages/` subtree, and the LLM is the in-cluster
Gemma 4 vLLM sibling app from `topics/gemma4/gemma4-dgx-devbox`.

Layout inside the Dir:
    raw/<slug>.md     immutable source summaries
    pages/<slug>.md   LLM-maintained concept/entity pages
    index.md          auto-generated catalog
    log.md            append-only chronological log
    AGENTS.md         schema/conventions (you and the LLM co-evolve this)

Three operations follow Karpathy's gist:
    ingest_source   pull a source in, summarize it, integrate into pages
    query_wiki      ask a question, get a synthesized answer with citations
    lint_wiki       health-check: orphans, broken links, contradictions

`wiki_pipeline` chains init → ingest each seed source → query → lint for a
single reproducible `flyte run`.

Usage:
    flyte run pipeline.py wiki_pipeline                              # full demo
    flyte run pipeline.py init_wiki
    flyte run pipeline.py ingest_source --wiki <dir> --source <url>
    flyte run pipeline.py query_wiki --wiki <dir> --question "..."
    flyte run pipeline.py lint_wiki --wiki <dir>
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import flyte
import flyte.io
import flyte.report

import wiki_lib
from config import VLLM_MODEL_ID, VLLM_URL, pipeline_env

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = pipeline_env

# Default seed sources for wiki_pipeline. Wikipedia is stable, gets clean
# markdown out of trafilatura, and three RAG-adjacent topics are enough to
# show pages cross-referencing each other after the second ingest.
SEED_SOURCES: list[str] = [
    "https://en.wikipedia.org/wiki/Retrieval-augmented_generation",
    "https://en.wikipedia.org/wiki/Vector_database",
    "https://en.wikipedia.org/wiki/Large_language_model",
]


# ──────────────────────────────────────────────────────────────────────────────
# LLM client + small wrappers.
# ──────────────────────────────────────────────────────────────────────────────

def _llm():
    """OpenAI-compatible client pointed at the in-cluster Gemma 4 vLLM."""
    from openai import OpenAI

    return OpenAI(
        base_url=VLLM_URL.rstrip("/") + "/v1",
        api_key="not-used",
        timeout=120.0,
    )


def _chat(
    llm,
    messages: list[dict],
    *,
    temperature: float = 0.2,
    max_tokens: int = 4096,
    response_format: dict | None = None,
) -> str:
    kwargs = {
        "model": VLLM_MODEL_ID,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format
    r = llm.chat.completions.create(**kwargs)
    return r.choices[0].message.content or ""


# ──────────────────────────────────────────────────────────────────────────────
# init_wiki: empty wiki Dir with scaffolding.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def init_wiki() -> flyte.io.Dir:
    """Create an empty wiki Dir with index/log/schema scaffolding."""
    root = Path(tempfile.mkdtemp(prefix="wiki_init_"))
    wiki_lib.init_layout(root)
    wiki_lib.regenerate_index(root)
    wiki_lib.append_log(root, f"## [{wiki_lib.now_utc()}] init")

    layout = "\n".join(f"- <code>{p.name}</code>" for p in sorted(root.iterdir()))
    await flyte.report.replace.aio(
        f"<h2>Wiki initialized</h2>"
        f"<p>Scaffolding written to a fresh <code>flyte.io.Dir</code>:</p>"
        f"<ul>{layout}</ul>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(root))


# ──────────────────────────────────────────────────────────────────────────────
# ingest_source: read a source, summarize it, integrate into the wiki.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def ingest_source(
    wiki: flyte.io.Dir,
    source: str,
    title_override: str = "",
) -> flyte.io.Dir:
    """Ingest one source (URL or text blob) and return the updated wiki Dir."""
    src_path = Path(await wiki.download())
    root = Path(tempfile.mkdtemp(prefix="wiki_ingest_"))
    shutil.copytree(src_path, root, dirs_exist_ok=True)
    wiki_lib.init_layout(root)

    title, markdown, source_url = wiki_lib.fetch_to_markdown(source)
    if title_override:
        title = title_override
    slug = wiki_lib.slugify(title)
    log.info(f"Ingesting source: {title} ({source_url or 'pasted text'}) → raw/{slug}.md")

    llm = _llm()

    # Pass 1: write a raw/<slug>.md summary of the source itself.
    summary_md = _chat(
        llm,
        wiki_lib.prompt_source_summary(title, source_url, markdown),
        temperature=0.2,
    )
    (root / wiki_lib.WIKI_RAW_DIR / f"{slug}.md").write_text(
        summary_md.rstrip() + "\n"
    )

    # Pass 2: ask the LLM to update concept pages based on the new summary.
    # JSON mode + an explicit schema in the prompt keeps the parse robust.
    pages = wiki_lib.read_pages(root)
    index_md = (root / wiki_lib.WIKI_INDEX_FILE).read_text()
    pages_dump = wiki_lib.dump_pages_for_prompt(pages)
    integration_raw = _chat(
        llm,
        wiki_lib.prompt_integrate(summary_md, index_md, pages_dump),
        temperature=0.2,
        max_tokens=4096,
        response_format={"type": "json_object"},
    )

    try:
        ops = wiki_lib.parse_json_blob(integration_raw).get("ops", []) or []
    except Exception as e:
        log.warning(
            f"Could not parse integration JSON ({e}); no concept pages updated. "
            f"Raw output head: {integration_raw[:200]!r}"
        )
        ops = []

    touched = wiki_lib.apply_page_ops(root, ops)
    log.info(f"Pages touched: {touched or '(none)'}")

    wiki_lib.regenerate_index(root)
    wiki_lib.append_log(
        root,
        f"## [{wiki_lib.now_utc()}] ingest | {title}\n"
        f"- Source: {source_url or '(pasted text)'}\n"
        f"- Raw summary: [[raw/{slug}]]\n"
        f"- Pages touched: "
        + (", ".join(f"[[{s}]]" for s in touched) if touched else "_none_"),
    )

    touched_html = (
        "".join(f"<li><code>pages/{s}.md</code></li>" for s in touched)
        or "<li><i>none</i></li>"
    )
    await flyte.report.replace.aio(
        f"<h2>Ingested: {title}</h2>"
        f"<p><b>Source:</b> {source_url or '(pasted text)'}</p>"
        f"<p><b>Raw summary:</b> <code>raw/{slug}.md</code></p>"
        f"<p><b>Pages touched:</b></p><ul>{touched_html}</ul>"
    )
    await flyte.report.flush.aio()

    return await flyte.io.Dir.from_local(str(root))


# ──────────────────────────────────────────────────────────────────────────────
# query_wiki: answer a question using the wiki, with citations.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def query_wiki(
    wiki: flyte.io.Dir,
    question: str,
) -> str:
    """Answer a question against the wiki and return the markdown answer."""
    src_path = Path(await wiki.download())
    root = Path(tempfile.mkdtemp(prefix="wiki_query_"))
    shutil.copytree(src_path, root, dirs_exist_ok=True)
    wiki_lib.init_layout(root)

    llm = _llm()
    index_md = (root / wiki_lib.WIKI_INDEX_FILE).read_text()

    pick_raw = _chat(
        llm,
        wiki_lib.prompt_pick_pages(question, index_md),
        temperature=0.0,
        max_tokens=512,
        response_format={"type": "json_object"},
    )
    try:
        slugs = wiki_lib.parse_json_blob(pick_raw).get("slugs", []) or []
    except Exception:
        slugs = []

    pages = wiki_lib.read_pages(root)
    selected = {s: pages[s] for s in slugs if s in pages}
    if not selected:
        # Fallback when the model didn't pick anything useful: include all
        # pages so the answer pass at least has something to work with.
        selected = pages
    pages_dump = wiki_lib.dump_pages_for_prompt(selected, per_page_chars=3000)

    answer = _chat(
        llm, wiki_lib.prompt_answer(question, pages_dump), temperature=0.3
    )

    wiki_lib.append_log(
        root,
        f"## [{wiki_lib.now_utc()}] query | {question[:80]}\n"
        f"- Pages consulted: "
        + (", ".join(f"[[{s}]]" for s in sorted(selected.keys())) or "_none_"),
    )

    pages_html = ", ".join(f"<code>{s}</code>" for s in sorted(selected.keys()))
    await flyte.report.replace.aio(
        f"<h2>Query</h2><blockquote>{question}</blockquote>"
        f"<h3>Answer</h3><pre>{answer}</pre>"
        f"<p><b>Pages consulted:</b> {pages_html}</p>"
    )
    await flyte.report.flush.aio()
    return answer


# ──────────────────────────────────────────────────────────────────────────────
# lint_wiki: deterministic + LLM-driven health check.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def lint_wiki(wiki: flyte.io.Dir) -> str:
    """Audit the wiki and return a markdown report.

    Combines two passes:
      1. deterministic: orphan pages, broken `[[links]]`
      2. LLM: contradictions, stale claims, missing pages / cross-refs
    """
    src_path = Path(await wiki.download())
    root = Path(tempfile.mkdtemp(prefix="wiki_lint_"))
    shutil.copytree(src_path, root, dirs_exist_ok=True)
    wiki_lib.init_layout(root)

    det = wiki_lib.deterministic_lint(root)

    llm = _llm()
    index_md = (root / wiki_lib.WIKI_INDEX_FILE).read_text()
    pages = wiki_lib.read_pages(root)
    pages_dump = wiki_lib.dump_pages_for_prompt(pages, per_page_chars=2000)
    llm_report = _chat(
        llm, wiki_lib.prompt_lint(index_md, pages_dump), temperature=0.3
    )

    parts = [
        f"# Lint report: {wiki_lib.now_utc()}",
        "",
        "## Stats",
        f"- Pages: {det['n_pages']}",
        f"- Raw summaries: {det['n_raw']}",
        f"- Orphans: {len(det['orphans'])}",
        f"- Broken links: {len(det['broken_links'])}",
        "",
    ]
    if det["orphans"]:
        parts.append("## Orphans")
        parts.extend(f"- [[{s}]]" for s in det["orphans"])
        parts.append("")
    if det["broken_links"]:
        parts.append("## Broken links")
        parts.extend(
            f"- [[{src}]] → `[[{tgt}]]` (no such page)"
            for src, tgt in det["broken_links"]
        )
        parts.append("")
    parts.append(llm_report)
    report = "\n".join(parts)

    wiki_lib.append_log(
        root,
        f"## [{wiki_lib.now_utc()}] lint\n"
        f"- Orphans: {len(det['orphans'])}, "
        f"broken links: {len(det['broken_links'])}",
    )

    await flyte.report.replace.aio(f"<h2>Lint report</h2><pre>{report}</pre>")
    await flyte.report.flush.aio()
    return report


# ──────────────────────────────────────────────────────────────────────────────
# wiki_pipeline: orchestrator. init, ingest each seed, query, lint.
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def wiki_pipeline(
    sources: list[str] = SEED_SOURCES,
    question: str = "What is retrieval-augmented generation and why is it useful?",
) -> flyte.io.Dir:
    """End-to-end demo run: init the wiki, ingest each source, query, lint."""
    await flyte.report.replace.aio(
        "<h2>LLM Wiki pipeline</h2><p>Step 1: initialising wiki…</p>"
    )
    await flyte.report.flush.aio()
    wiki = await init_wiki()

    for i, src in enumerate(sources, 1):
        await flyte.report.replace.aio(
            f"<h2>LLM Wiki pipeline</h2>"
            f"<p>Step {i + 1}: ingesting source {i}/{len(sources)}…</p>"
            f"<pre>{src}</pre>"
        )
        await flyte.report.flush.aio()
        wiki = await ingest_source(wiki, src)

    await flyte.report.replace.aio(
        f"<h2>LLM Wiki pipeline</h2><p>Querying wiki…</p>"
        f"<blockquote>{question}</blockquote>"
    )
    await flyte.report.flush.aio()
    answer = await query_wiki(wiki, question)

    await flyte.report.replace.aio(
        "<h2>LLM Wiki pipeline</h2><p>Linting wiki…</p>"
    )
    await flyte.report.flush.aio()
    report = await lint_wiki(wiki)

    await flyte.report.replace.aio(
        f"<h2>LLM Wiki pipeline complete</h2>"
        f"<p><b>Sources ingested:</b> {len(sources)}</p>"
        f"<h3>Demo query</h3>"
        f"<blockquote>{question}</blockquote>"
        f"<pre>{answer}</pre>"
        f"<h3>Lint report</h3><pre>{report}</pre>"
    )
    await flyte.report.flush.aio()
    return wiki


if __name__ == "__main__":
    flyte.init_from_config()
    run = flyte.run(wiki_pipeline)
    print(f"Pipeline run: {run.name}")
    print(f"  {run.url}")
