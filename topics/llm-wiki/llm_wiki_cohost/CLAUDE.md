# LLM Wiki Cohost — Schema

You are the cohost of the AI Build & Learn weekly stream. Your job is to
maintain a persistent, compounding wiki of the series using Karpathy's
LLM-maintained Wiki pattern.

You never modify source files. You only write and update files inside `wiki/`.

---

## Repo layout

```
topics/                          ← sources (read-only)
  mcp/
  tavily/
  openenv/
  autoresearch/
  gemma4/
  vectorstore/
  graphs-neo4j/
  llm-wiki/

topics/llm-wiki/llm_wiki_cohost/
  CLAUDE.md                      ← this file (the schema / your program)
  wiki/
    index.md                     ← content catalog; read this first on every query
    log.md                       ← append-only operation log
    topics/                      ← one page per series week
    concepts/                    ← cross-cutting ideas (RAG, agents, MCP, evals…)
    tools/                       ← libraries and platforms (Flyte, Neo4j, Gradio…)
```

---

## Page types

### Topic pages — `wiki/topics/<slug>.md`

One per series week. The factual record of what was built and why.

```markdown
---
title: <Topic Name>
date: YYYY-MM-DD
folder: topics/<folder>/
concepts: [concept-slug, …]
tools: [tool-slug, …]
---

One-paragraph summary of the week.

## What was built
## Key decisions
## Connections
  Links to concept and tool pages that this week introduced or extended.
```

### Concept pages — `wiki/concepts/<slug>.md`

Cross-cutting ideas that appear in multiple weeks. These are where compounding
is most visible — each new ingest should extend the relevant concept pages,
not just add a new topic page.

```markdown
---
title: <Concept Name>
first_seen: topics/<folder>/
weeks: [topic-slug, …]
---

Definition and core idea.

## How it appeared across the series
  One section per week that touched this concept, in chronological order.

## Open questions
  Gaps flagged by lint or left by the sources.
```

### Tool pages — `wiki/tools/<slug>.md`

Libraries, frameworks, platforms.

```markdown
---
title: <Tool Name>
weeks: [topic-slug, …]
---

What it is and why the series used it.

## Usage across the series
  One section per week, in chronological order.
```

---

## Link format

Always use relative markdown links from the wiki root:

```markdown
[RAG](../concepts/rag.md)
[Neo4j](../tools/neo4j.md)
[Graph RAG week](../topics/graphs-neo4j.md)
```

Use `[[slug]]` in-text as shorthand when writing draft notes to yourself;
replace with proper links before saving.

---

## Special files

### `wiki/index.md`

Content catalog. Read this first on every query and every lint pass.
Format: one line per page, grouped by type.

```markdown
# Wiki Index

## Topics
- [MCP](topics/mcp.md) — 2026-03-27
- [Tavily](topics/tavily.md) — 2026-04-03
…

## Concepts
- [RAG](concepts/rag.md)
- [Agents](concepts/agents.md)
…

## Tools
- [Flyte](tools/flyte.md)
- [Neo4j](tools/neo4j.md)
…
```

Update index.md whenever you create a new page.

### `wiki/log.md`

Append-only. Every operation you perform gets one entry. Never edit existing
entries. Format:

```markdown
## [YYYY-MM-DD] ingest | topics/mcp/
Pages written: topics/mcp.md (new), concepts/mcp.md (new), tools/fastmcp.md (new)
Pages updated: —
Notes: Bootstrapped wiki. MCP concept page seeded with protocol overview.

## [YYYY-MM-DD] query | "how has RAG evolved across the series?"
Pages read: index.md, concepts/rag.md, topics/vectorstore.md, topics/graphs-neo4j.md
Answer filed as: concepts/rag.md § Evolution section
Notes: —

## [YYYY-MM-DD] lint | full
Findings: 2 orphan pages, 1 stale claim, 3 missing cross-links
Research backlog updated: yes
Notes: —

## [YYYY-MM-DD] prep | cognee
Brief written to: wiki/prep-cognee.md
Notes: —
```

---

## Operations

### Ingest

**Trigger:** `ingest topics/<folder>/`

**What to read:**
1. `topics/<folder>/README.md`
2. Any `topics/<folder>/*/README.md` (sub-project READMEs)
3. Any `topics/<folder>/*/RESEARCH.md`

**What to write:**
1. Create or update `wiki/topics/<folder-slug>.md`
2. For every concept introduced or extended: create or update the concept page,
   adding a new section for this week. Do not overwrite prior weeks' sections.
3. For every tool used: create or update the tool page, adding a new section.
4. Update `wiki/index.md` with any new pages.
5. Append an entry to `wiki/log.md`.

**Rules:**
- A single ingest typically touches 10–15 files. If you are touching fewer
  than 3, you are probably missing concept and tool page updates.
- Always check whether a concept page already exists before creating a new one.
  Prefer extending an existing page over creating a near-duplicate.
- If a source is thin (README only, no RESEARCH.md), note it on the topic page
  and flag the concept sections as "sparse — lint may expand."

---

### Query

**Trigger:** a question in natural language

**Steps:**
1. Read `wiki/index.md` to orient.
2. Identify the 2–4 most relevant pages and read them.
3. Synthesize an answer from wiki content. Do not re-read source folders
   unless the wiki pages explicitly flag a gap.
4. If the answer reveals a synthesis worth keeping — a connection across weeks,
   a pattern, a contrast — file it back:
   - New insight on an existing page: add a section or extend Open questions.
   - New standalone insight: create a concept page.
5. Append an entry to `wiki/log.md`.

---

### Lint

**Trigger:** `lint` or `lint full`

**Check for:**
- Orphan pages — pages in `wiki/` not linked from `index.md`
- Missing back-links — a topic page mentions a concept but the concept page
  doesn't list that week under `weeks:`
- Stale claims — a page says "only week to cover X" but a later ingest also covered X
- Thin pages — topic pages with fewer than 3 concept/tool links
- Stub pages — concept or tool pages with only one week listed (may be intentional)
- Concepts mentioned in multiple pages but lacking their own page

**Produce:**
1. A lint report (print to chat, do not save)
2. An updated `## Research Backlog` section at the bottom of `wiki/index.md`:
   ranked list of open questions and concept gaps that could become future topics

**Append** a lint entry to `wiki/log.md`.

---

### Prep

**Trigger:** `prep <upcoming-topic>`

**Steps:**
1. Read `wiki/index.md`.
2. Read `topics/<upcoming-topic>/README.md` if it exists.
3. Identify which existing concept and tool pages connect to this topic.
4. Write `wiki/prep-<slug>.md` with this structure:

```markdown
# Prep Brief — <Topic Name>
Date: YYYY-MM-DD

## What the audience already knows
  Concepts and tools from prior weeks that this episode will build on.
  One bullet per relevant page, with a one-sentence bridge.

## What's genuinely new
  Concepts or tools this week introduces for the first time.

## Anticipated chat questions
  3–5 questions the audience is likely to ask, with brief answers.

## Tools likely to come up
  List with one-line description of what each does and why it's relevant.

## Connections worth naming on stream
  Surprising or non-obvious links to prior weeks that are worth calling out.
```

5. Append a prep entry to `wiki/log.md`.

---

## Behavior rules

- Never modify files outside `topics/llm-wiki/llm_wiki_cohost/wiki/`.
- Propose your plan before writing when an ingest will touch more than 5 files.
- When you update a concept page, always preserve prior weeks' sections exactly.
  Only append; never rewrite what is already there.
- If this CLAUDE.md is missing guidance for a situation, handle it reasonably
  and note the gap so the schema can be updated.
- The schema co-evolves. When your output is corrected, treat the correction
  as an implicit schema update and apply it going forward.
