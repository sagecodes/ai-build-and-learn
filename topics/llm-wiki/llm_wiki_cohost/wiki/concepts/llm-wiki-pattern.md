---
title: LLM Wiki Pattern
first_seen: topics/llm-wiki/
weeks: [llm-wiki]
---

Karpathy's LLM-maintained Wiki pattern. The inversion of RAG: instead of
re-discovering knowledge on every query, the LLM reads sources once and
incrementally builds a structured wiki that synthesizes and accumulates insight.
The wiki is the artifact; sources become input material.

**Traditional RAG:** sources stay raw; LLM re-derives knowledge per query.

**LLM Wiki:** LLM reads sources once; writes a wiki; future queries read the
wiki, not the sources. Explorations compound — a query that produces a useful
synthesis gets filed back as a new wiki page, not discarded.

## Three layers

| Layer | What it is | In this repo |
|---|---|---|
| **Sources** | Immutable inputs; the LLM reads but never modifies | `topics/*/` |
| **Wiki** | LLM-written markdown pages; the compounding artifact | `wiki/` |
| **Schema** | Defines structure, conventions, operations | `CLAUDE.md` |

**The schema is the program.** CLAUDE.md specifies page types, frontmatter
conventions, link format, the four operation workflows, logging format, and
behavior rules. The LLM executes the schema; correction is the programming
interface — fix CLAUDE.md once, behavior persists.

## Four operations (from the gist, extended)

**Ingest** — reads sources, writes topic page, updates all concept and tool
pages (typically 10–15 files per week). The most visible operation: pages
gain new sections; cross-links form; the wiki grows denser.

**Query** — reads index and relevant pages, synthesizes an answer. Valuable
answers are filed back into the wiki so explorations compound. The LLM does
not re-read source folders unless a wiki page explicitly flags a gap.

**Lint** — periodic health check: orphan pages, stale claims, missing
cross-links, thin pages. Produces a ranked research backlog — concepts
mentioned across multiple sources but lacking their own page. The wiki
proposes its own next moves.

**Prep** (series extension) — produces a host brief for an upcoming episode:
what prior weeks connect, what's genuinely new for the audience, anticipated
chat questions, tools likely to come up. Turns the wiki from a passive
knowledge store into an active cohost.

## How it appeared across the series

### Week 8 — LLM Wiki (2026-05-15)

The pattern is implemented as the wiki you are reading. Key design choices that
differ from a naive interpretation of the gist:

**`index.md` as poor-man's RAG.** At ~50–100 pages, a content catalog is
sufficient for orientation. The LLM reads `index.md` first on every operation,
then drills into 2–4 relevant pages. No vector search needed at this scale.

**`log.md` as append-only record.** Every ingest, query, lint, and prep gets
one log entry — date, operation, pages written, pages updated, notes. The log
makes the wiki's own history queryable.

**Three page types, kept tight.** Topic pages (one per week), concept pages
(cross-cutting ideas), tool pages (libraries and platforms). No "pattern" or
"meta" page types in v1. They can emerge if useful.

**Forward-looking ingest.** The schema seeds stub pages for upcoming episodes
based on the series schedule. The wiki has a model of where the series is going,
not just where it's been. When the self-referential ingest runs (this week), the
wiki notices structural overlap with upcoming Cognee week and the research
backlog gains a new entry.

**The self-referential moment.** When the series ingests `topics/llm-wiki/`,
the wiki writes a page about the pattern used to build it. The wiki becomes
aware of its own construction. This is the demo's narrative payoff — not a
technical trick, but a conceptual closure.

## Open questions

- Does the pattern scale past ~100 pages before index.md becomes unwieldy?
  The gist doesn't address this — at what point does poor-man's RAG need
  to become real RAG?
- How does the schema co-evolution work long-term? What's the right cadence
  for deliberate schema review vs. implicit correction?
- Cognee (week 9) is a productized memory layer. How much of the LLM Wiki
  pattern could Cognee replace or accelerate?
