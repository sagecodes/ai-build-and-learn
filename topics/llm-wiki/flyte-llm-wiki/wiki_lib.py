"""Pure-Python helpers for the LLM Wiki.

Used by both `pipeline.py` (Flyte tasks) and `chat_app.py` (Gradio app), so it
deliberately knows nothing about Flyte or OpenAI clients. It operates on a
local `Path` and returns prompt messages for the caller to send to whatever
LLM they're using.

Wiki layout under `root`:
    raw/<slug>.md     immutable source summaries, one per ingested source
    pages/<slug>.md   LLM-maintained concept/entity pages
    index.md          auto-generated catalog
    log.md            append-only chronological log
    AGENTS.md         schema/conventions
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

# Layout
WIKI_PAGES_DIR = "pages"
WIKI_RAW_DIR = "raw"
WIKI_INDEX_FILE = "index.md"
WIKI_LOG_FILE = "log.md"
WIKI_SCHEMA_FILE = "AGENTS.md"

# Prompt sizing. Gemma 4 26B has plenty of context, but ingest does two LLM
# passes per source so keeping prompts tight keeps the demo snappy. Bump these
# if you switch to a bigger model.
MAX_SOURCE_CHARS = 12_000
MAX_PAGE_DUMP_CHARS = 8_000

SLUG_RE = re.compile(r"[^a-z0-9]+")
# Match [[slug]] or [[slug|label]]; strip pipe-aliases and section anchors.
LINK_RE = re.compile(r"\[\[([^\]\|#]+)(?:\|[^\]]+)?(?:#[^\]]+)?\]\]")


SEED_SCHEMA = """# Wiki schema and conventions

This wiki is maintained by an LLM. Humans add sources; the LLM writes and
updates pages.

## Layout
- `raw/<slug>.md` immutable source summaries, one per ingested source
- `pages/<slug>.md` concept/entity pages, owned and maintained by the LLM
- `index.md` auto-generated catalog of every page
- `log.md` append-only log of ingests, queries, and lint passes

## Page conventions
- Every page starts with `# <Title>`
- Use `[[slug]]` to link to other concept pages
- Use `[[raw/<slug>]]` to cite a source summary
- End pages with `## Sources` listing the raw summary pages that informed them

## Operations
- **Ingest**: read a source, write a `raw/` summary, update affected `pages/`
- **Query**: answer a question using the wiki (not the raw sources directly)
- **Lint**: report contradictions, orphans, missing pages, stale claims
"""


# ──────────────────────────────────────────────────────────────────────────────
# Layout + IO
# ──────────────────────────────────────────────────────────────────────────────

def slugify(s: str) -> str:
    out = SLUG_RE.sub("-", s.lower()).strip("-")
    return out[:80] or "page"


def now_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def init_layout(root: Path) -> None:
    """Idempotently create the wiki scaffolding under `root`."""
    (root / WIKI_PAGES_DIR).mkdir(parents=True, exist_ok=True)
    (root / WIKI_RAW_DIR).mkdir(parents=True, exist_ok=True)
    if not (root / WIKI_SCHEMA_FILE).exists():
        (root / WIKI_SCHEMA_FILE).write_text(SEED_SCHEMA)
    if not (root / WIKI_INDEX_FILE).exists():
        (root / WIKI_INDEX_FILE).write_text(
            "# Wiki index\n\n_Empty; ingest a source to get started._\n"
        )
    if not (root / WIKI_LOG_FILE).exists():
        (root / WIKI_LOG_FILE).write_text("# Wiki log\n\n")


def read_pages(root: Path, subdir: str = WIKI_PAGES_DIR) -> dict[str, str]:
    pages: dict[str, str] = {}
    d = root / subdir
    if d.exists():
        for p in sorted(d.glob("*.md")):
            pages[p.stem] = p.read_text()
    return pages


def read_raw_summaries(root: Path) -> dict[str, str]:
    return read_pages(root, WIKI_RAW_DIR)


def _first_heading(md: str) -> str:
    for line in md.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "(no title)"


def _one_liner(md: str, limit: int = 140) -> str:
    for line in md.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("-"):
            continue
        return s[:limit]
    return ""


def regenerate_index(root: Path) -> str:
    """Rebuild `index.md` from the current contents on disk and return its text."""
    pages = read_pages(root, WIKI_PAGES_DIR)
    raw = read_pages(root, WIKI_RAW_DIR)

    lines = [
        "# Wiki index",
        "",
        f"_Generated {now_utc()}._",
        "",
        f"## Pages ({len(pages)})",
        "",
    ]
    if pages:
        for slug, content in sorted(pages.items()):
            lines.append(
                f"- [[{slug}]] **{_first_heading(content)}**: {_one_liner(content)}"
            )
    else:
        lines.append("_None yet._")
    lines += ["", f"## Sources ({len(raw)})", ""]
    if raw:
        for slug, content in sorted(raw.items()):
            lines.append(
                f"- [raw/{slug}](raw/{slug}.md) **{_first_heading(content)}**: "
                f"{_one_liner(content)}"
            )
    else:
        lines.append("_None yet._")
    lines.append("")

    text = "\n".join(lines)
    (root / WIKI_INDEX_FILE).write_text(text)
    return text


def append_log(root: Path, entry: str) -> None:
    p = root / WIKI_LOG_FILE
    existing = p.read_text() if p.exists() else "# Wiki log\n\n"
    p.write_text(existing.rstrip() + "\n\n" + entry.rstrip() + "\n")


# ──────────────────────────────────────────────────────────────────────────────
# Source fetch: URL to markdown, or pass-through pasted text.
# ──────────────────────────────────────────────────────────────────────────────

def fetch_to_markdown(source: str) -> tuple[str, str, str]:
    """Return `(title, markdown, source_url)`.

    `source` is either an http(s) URL or a raw text blob. For URLs we use
    trafilatura, which handles the bulk of readability/boilerplate stripping
    and outputs markdown directly.
    """
    import httpx
    import trafilatura

    s = source.strip()
    if s.startswith(("http://", "https://")):
        with httpx.Client(
            follow_redirects=True,
            timeout=30.0,
            headers={"User-Agent": "llm-wiki/0.1 (+https://github.com/sagecodes/ai-build-and-learn)"},
        ) as client:
            r = client.get(s)
            r.raise_for_status()
            html = r.text
        md = trafilatura.extract(
            html, output_format="markdown", include_links=True
        )
        if not md:
            raise RuntimeError(f"trafilatura could not extract content from {s}")
        meta = trafilatura.extract_metadata(html)
        title = (meta.title if (meta and meta.title) else s).strip()
        return title, md, s

    lines = [ln for ln in s.splitlines() if ln.strip()]
    title = lines[0][:120] if lines else "Pasted note"
    return title, s, ""


# ──────────────────────────────────────────────────────────────────────────────
# Prompts. Caller decides which LLM and whether to stream.
# ──────────────────────────────────────────────────────────────────────────────

def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "\n\n[...truncated]"


def prompt_source_summary(
    title: str, source_url: str, markdown: str
) -> list[dict]:
    body = _truncate(markdown, MAX_SOURCE_CHARS)
    user = f"""You are maintaining a personal wiki. A new source was just added. Write a markdown summary page for it.

The page must follow this exact shape:

# {title}

## Summary
(2-4 short paragraphs covering the core ideas of the source)

## Key takeaways
- (5-8 concise bullet points)

## Concepts to integrate
- (3-8 concepts/entities mentioned in the source that should have, or already have, their own wiki page; short noun phrases)

## Source
- URL: {source_url or "(pasted text)"}

Return ONLY the markdown. No preamble. No code fences.

SOURCE CONTENT:
{body}
"""
    return [
        {"role": "system", "content": "You write concise, accurate markdown summaries of source documents."},
        {"role": "user", "content": user},
    ]


def prompt_integrate(
    source_summary: str, index_md: str, pages_dump: str
) -> list[dict]:
    body = _truncate(pages_dump, MAX_PAGE_DUMP_CHARS)
    user = f"""You are maintaining a personal wiki. A new source was just summarized. Now update the rest of the wiki to integrate it.

Decide which CONCEPT pages to create or update so that:
- New entities/concepts mentioned in the source get their own page (if not already present)
- Existing pages that the source extends, refines, or contradicts get updated
- Each page is concise: a few short paragraphs of prose, with [[slug]] cross-references
- Slugs are lowercase kebab-case (e.g. `retrieval-augmented-generation`, `vector-database`)

Be conservative. A typical ingest touches 2-5 pages. Do not edit raw source summaries.

Return STRICTLY this JSON shape. No prose, no code fences, no markdown.

{{
  "ops": [
    {{
      "slug": "retrieval-augmented-generation",
      "title": "Retrieval-Augmented Generation",
      "content": "# Retrieval-Augmented Generation\\n\\nRAG is...\\n\\n## Sources\\n- [[raw/some-slug]]\\n"
    }}
  ]
}}

Each `content` must:
- Start with `# <title>`
- Use `[[other-slug]]` for concept-page links
- End with a `## Sources` section listing `[[raw/<slug>]]` entries for the source summaries that informed the page

NEW SOURCE SUMMARY:
{source_summary}

CURRENT INDEX:
{index_md}

EXISTING PAGES:
{body or "(none)"}
"""
    return [
        {"role": "system", "content": "You return strictly valid JSON matching the requested shape. No prose, no markdown fences."},
        {"role": "user", "content": user},
    ]


def prompt_pick_pages(question: str, index_md: str) -> list[dict]:
    user = f"""Given the question and the wiki index, pick the 3-7 most relevant page slugs to read.

Return STRICTLY this JSON. No prose, no fences.

{{"slugs": ["slug-a", "slug-b"]}}

QUESTION: {question}

INDEX:
{index_md}
"""
    return [
        {"role": "system", "content": "You return strictly valid JSON."},
        {"role": "user", "content": user},
    ]


def prompt_answer(question: str, pages_dump: str) -> list[dict]:
    body = _truncate(pages_dump, MAX_PAGE_DUMP_CHARS * 2)
    user = f"""Answer the question using only the wiki pages below. Cite pages inline as [[slug]]. If the answer is not in the wiki, say so explicitly; do not invent.

QUESTION: {question}

WIKI PAGES:
{body or "(empty)"}
"""
    return [
        {"role": "system", "content": "You answer questions strictly from the provided wiki pages. You cite [[slug]] inline."},
        {"role": "user", "content": user},
    ]


def prompt_lint(index_md: str, pages_dump: str) -> list[dict]:
    body = _truncate(pages_dump, MAX_PAGE_DUMP_CHARS * 2)
    user = f"""You are auditing a personal wiki. Read the index and pages and report issues.

Group the report under these headings (omit a heading entirely if you found nothing for it):
- ## Contradictions
- ## Stale or weakly supported claims
- ## Missing pages (concepts mentioned across pages but lacking their own page)
- ## Missing cross-references (pages that should link to each other but don't)

Be specific. Cite [[slug]] where applicable. Keep it concise.

INDEX:
{index_md}

PAGES:
{body or "(empty)"}
"""
    return [
        {"role": "system", "content": "You write concise audit reports of personal wikis."},
        {"role": "user", "content": user},
    ]


# ──────────────────────────────────────────────────────────────────────────────
# Deterministic helpers (no LLM)
# ──────────────────────────────────────────────────────────────────────────────

def deterministic_lint(root: Path) -> dict:
    """Compute structural issues without an LLM: orphans and broken `[[links]]`."""
    pages = read_pages(root, WIKI_PAGES_DIR)
    inbound: dict[str, set[str]] = {slug: set() for slug in pages}
    broken: list[tuple[str, str]] = []
    for slug, content in pages.items():
        for m in LINK_RE.finditer(content):
            target = m.group(1).strip()
            if target.startswith("raw/"):
                continue
            if target in pages:
                if target != slug:
                    inbound[target].add(slug)
            else:
                broken.append((slug, target))
    orphans = sorted(
        s for s, refs in inbound.items() if not refs and len(pages) > 1
    )
    return {
        "n_pages": len(pages),
        "n_raw": len(read_raw_summaries(root)),
        "orphans": orphans,
        "broken_links": broken,
    }


def apply_page_ops(root: Path, ops: list[dict]) -> list[str]:
    """Apply LLM-returned page ops to disk. Returns the slugs that were touched."""
    pages_dir = root / WIKI_PAGES_DIR
    pages_dir.mkdir(parents=True, exist_ok=True)
    touched: list[str] = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        slug_seed = op.get("slug") or op.get("title") or ""
        slug = slugify(slug_seed)
        if not slug:
            continue
        content = (op.get("content") or "").strip()
        if not content:
            continue
        (pages_dir / f"{slug}.md").write_text(content + "\n")
        touched.append(slug)
    return touched


def dump_pages_for_prompt(
    pages: dict[str, str], per_page_chars: int = 1500
) -> str:
    parts: list[str] = []
    for slug, content in sorted(pages.items()):
        body = content
        if len(body) > per_page_chars:
            body = body[:per_page_chars] + "\n\n[...truncated]"
        parts.append(f"---\n### {slug}\n{body}")
    return "\n\n".join(parts)


def parse_json_blob(text: str) -> dict:
    """Best-effort JSON parse: strips code fences and surrounding prose."""
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[1] if "\n" in t else t
        if t.endswith("```"):
            t = t.rsplit("```", 1)[0]
        if t.lstrip().startswith("json"):
            t = t.lstrip()[4:]
    # If the model wrapped JSON in prose, grab the outermost braces.
    i, j = t.find("{"), t.rfind("}")
    if i != -1 and j != -1 and j > i:
        t = t[i : j + 1]
    return json.loads(t)
