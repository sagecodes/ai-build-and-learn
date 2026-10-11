# LLM Wiki Cohost — Project Brief

A repo-scoped implementation of Karpathy's LLM-maintained Wiki pattern, built to
serve the AI Build & Learn weekly stream as a persistent, compounding cohost.

Reference: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f

---

## What we're building

A wiki of the AI Build & Learn series itself. The LLM (Claude Code) ingests
each week's topic folder as a source, writes and maintains markdown pages
that synthesize the series across episodes, and acts as an active cohost — it
prepares the host (Sage) for upcoming episodes, surfaces gaps that become
candidate future topics, and grows denser with every week.

---

## Karpathy's pattern (the foundation we're keeping)

Traditional RAG re-discovers knowledge on every query. The LLM Wiki pattern
inverts this: the LLM reads sources once and incrementally builds a structured
wiki that synthesizes and accumulates insight. The wiki is the artifact;
sources become input material.

Karpathy describes the gist as an "idea file" meant to be copy-pasted into an
LLM coding agent which then builds the specifics with you. There is no shipped
implementation — the *pattern* is the deliverable.

### Three layers

| Layer | What it is | Where it lives in our repo |
|---|---|---|
| **Sources** | Immutable inputs the LLM reads but never modifies | `topics/*/` — each prior week's folder |
| **Wiki** | LLM-written markdown pages — the compounding artifact | `topics/llm-wiki/llm_wiki_cohost/wiki/` |
| **Schema** | Config that defines structure, conventions, workflows | `topics/llm-wiki/llm_wiki_cohost/CLAUDE.md` |

### Three core operations (from the gist)

**Ingest.** Drop a source in; the LLM reads it, writes a summary page, updates
relevant entity and concept pages across the wiki (often 10-15 files in one
pass), and appends an entry to the log.

**Query.** Ask the wiki a question. The LLM reads the index, drills into
relevant pages, synthesizes an answer. Valuable answers get filed back as new
wiki pages so explorations compound, not just ingestions.

**Lint.** Periodic health check. Flags contradictions, stale claims, orphan
pages, missing cross-references, and concepts mentioned but lacking their own
page.

### The principle that makes the pattern work

The human curates sources, directs analysis, and asks good questions. The LLM
does everything else — writing, updating, linking, summarizing, auditing.
The LLM is the worker; the schema file is the program.

---

## What we're adding (the cohost extensions)

Three additions layered on top of Karpathy's foundation. Each is small to
implement and visually punchy on stream.

**Forward-looking ingest.** Beyond reading aired episodes, the LLM also reads
upcoming weeks' topic descriptions from the root README and seeds stub pages
for concepts those future episodes will touch (e.g., "memory layer" for the
Cognee week, "RAG evaluation" for the Ragas week). The wiki has a model of
where the series is going, not just where it's been. Stream payoff: when we
ingest the LLM-Wiki week itself, the wiki notices structural overlap with
Cognee and writes a stub linking them.

**Research backlog from lint.** Standard lint flags contradictions and
orphans. Ours also produces a ranked list of "open questions" — gaps where
multiple sources gesture at something none of them define. These become
candidate topics for future episodes. The wiki proposes its own future.

**`prep` workflow as a first-class operation.** A fourth operation alongside
ingest/query/lint. `prep <upcoming-topic>` produces a structured host brief:
what prior weeks connect to this topic, what's genuinely new for the audience,
anticipated chat questions, tools likely to come up. Turns the wiki from a
passive knowledge store into an active cohost. Demo finale: run
`prep cognee` and show the brief Sage will actually use Thursday night.

---

## Architecture decisions (locked)

**Stack: canonical Karpathy pattern.** No custom Python orchestration, no
Gradio, no web app. Claude Code is the agent. VS Code's file tree is the wiki
browser on stream. The CLAUDE.md schema is the program.

**Wiki location:** `topics/llm-wiki/llm_wiki_cohost/wiki/`.

**Schema location:** `topics/llm-wiki/llm_wiki_cohost/CLAUDE.md`. This is the
next artifact to draft.

**Sources stay in place.** No copying into a `raw/` folder. The LLM reads
existing READMEs directly from `topics/*/`. The schema declares which paths
count as sources.

**Page types — three, kept tight.**
- **Topic pages** — one per week. The factual record of what we built and why.
- **Concept pages** — RAG, agents, MCP, evals, memory, embeddings,
  multimodal, RL, local-LLMs, knowledge-graphs. These are where the
  compounding shows.
- **Tool/library pages** — Flyte, Gradio, Chroma, Neo4j, Tavily, Gemma,
  Voxtral, FastMCP, LangGraph.

No "pattern" or "meta" page types in v1. They can emerge in v2 if useful.

**Two special files inside the wiki:**
- `wiki/index.md` — content catalog, read first during queries. Karpathy's
  poor-man's-RAG at this scale.
- `wiki/log.md` — chronological append-only record of ingests, queries,
  lint passes, and preps. Date-prefixed headings so unix tools can parse it
  (e.g., `## [2026-05-15] ingest | topics/mcp/`).

---

## Source set — eight ingests, chronological

| Order | Date | Topic folder | Notes |
|---|---|---|---|
| 1 | 2026-03-27 | `topics/mcp/` | MCP + FastMCP. Bootstraps the wiki. |
| 2 | 2026-04-03 | `topics/tavily/` | First compounding moment — FastMCP page gets updated, not just referenced. |
| 3 | 2026-04-10 | `topics/openenv/` | Mostly new concept pages (RL, environments). |
| 4 | 2026-04-17 | `topics/autoresearch/` | Karpathy entity page is born. |
| 5 | 2026-04-24 | `topics/gemma4/` | Big ingest — seven sub-projects. |
| 6 | 2026-05-01 | `topics/vectorstore/` | RAG enters the wiki properly. |
| 7 | 2026-05-08 | `topics/graphs-neo4j/` | RAG page mutates — graph-RAG section. |
| 8 | 2026-05-15 | `topics/llm-wiki/` | Self-referential ingest. Karpathy page hits two mentions. |

**Per-week inputs.** The LLM reads the topic-level README plus any RESEARCH.md
plus any sub-project READMEs. The schema should make this explicit.

**Source coverage is uneven by design.** Some weeks have rich RESEARCH.md
files (this one), some have only a README. The wiki is built to handle that —
denser sources produce richer pages. Lint flags thin pages so we know what
the wiki knows it doesn't know.

---

## User experience

There is no app. The interface is conversation with Claude Code, running in
VS Code's terminal pane, on the `ai-build-and-learn` repo.

Concretely:
- You open VS Code at the repo root.
- Claude Code automatically loads `CLAUDE.md` on startup, so it knows the
  wiki's structure and workflows without being told.
- You type natural-language instructions in the chat: *"Ingest topics/mcp/"*,
  *"Run lint"*, *"Prep me for the Cognee episode."*
- Claude proposes the edits it'll make, you approve, and files appear in
  the VS Code file tree as Claude writes them.
- Every action is logged to `wiki/log.md`.

The schema co-evolves through correction. When Claude does something
suboptimal, you fix it once in CLAUDE.md and the behavior persists across
sessions.

---

## Demo plan — May 15 stream (60 min)

**Pre-stream (off-camera):** Pre-ingest weeks 1-5 (MCP through Gemma 4) so
the wiki has a baseline. Pre-write the CLAUDE.md schema. Verify that running
each operation produces the expected shape.

**On stream:**

1. *Open with the gist (5 min).* Karpathy's premise, what it is, why it's
   different from RAG. Acknowledge it's an "idea file," not a tool.
2. *Walk through the CLAUDE.md (10 min).* The schema is the interesting
   artifact. Show the page types, the ingest workflow, the lint checklist,
   the prep template. This is the part viewers can take home.
3. *Show the wiki state (5 min).* The five pre-ingested weeks. Open a
   concept page (RAG, MCP) so viewers see what a built-up page looks like.
4. *Live ingest the RAG arc (15 min).* In order: vectorstore →
   graphs-neo4j → llm-wiki itself. Audience watches the RAG concept page
   gain new sections at each step. The self-referential moment lands on
   the third ingest.
5. *Query (10 min).* Run two: *"trace how RAG has been handled across
   the series"* and *"what's Karpathy contributed to this series."*
   Show the synthesis. File the first answer back as a new wiki page.
6. *Lint + research backlog (5 min).* Run lint. Show the findings. Read
   the research backlog out loud — these are candidate future topics.
7. *Prep finale (5 min).* `prep cognee`. Show the brief. Tell the
   audience: this is what I'll actually use Thursday night.
8. *Close (5 min).* Recap, mention prior art, RSVP link for next week.

The narrative arc: RAG-introduction → RAG-extension → RAG-inversion →
the wiki proposing its own next moves.

---

## Prior art worth referencing

From the gist's comments. Listed in order of relevance to our scope.

- **zhurudong/andrej-karpathy-llm-wiki** — most literal implementation. Two
  layers + one CLAUDE.md, no vector DB. Symlinks CLAUDE.md to AGENTS.md for
  cross-tool compatibility. Closest reference for our schema.
- **theafh/ai-modules** (`plugins/knowledge_management`) — per-repo wiki
  scoped to a single codebase. Ships as a skill. Has a deterministic linter
  plus a cleanup agent that splits oversized pages and repairs links.
  Closest spiritual match for our framing.
- **AgriciDaniel/best-practices** — codifies a "six-axiom kernel" for the
  schema layer. Useful as a reference for schema design philosophy.
- **tuirk/Kompl** — full-stack tool, but worth borrowing tactics from:
  deterministic regex-injected wikilinks, comparison pages auto-generated
  when sources disagree across 3+ sources.
- **skyllwt/OmegaWiki** — for an example of how Claude Code skills can be
  used as the operation layer.

---

## Out of scope for v1

- No vector search. The `index.md` is enough at our scale (~10 weeks of
  sources, ~50-100 pages).
- No embedding pipeline, no qmd, no external search tool.
- No web app or Gradio UI. VS Code is the only surface.
- No agentic-loop framing. Operations are conversational instructions; the
  schema defines what each one does.
- No Flyte wrapping (despite three prior weeks using Flyte). Not core to
  the wiki idea and adds setup time.
- No multi-LLM ingest, no twin-reconciliation, no audience-driven live
  ingest. All interesting follow-ups, none in v1.

---

## Next concrete actions

1. **Draft `CLAUDE.md`** — the schema/program. Defines page types,
   frontmatter conventions, link format, the four operation workflows
   (ingest, query, lint, prep), and the logging format. This is the single
   most important artifact; everything else follows from it.
2. **Seed empty `wiki/` directory** with placeholder `index.md` and
   `log.md` so Claude Code has a known starting state.
3. **Dress-rehearse a single ingest** — run the MCP ingest end-to-end and
   confirm the output matches expectations. Adjust CLAUDE.md until it does.
4. **Pre-ingest weeks 1-5** to build up the demo baseline.
5. **Pre-write the demo script** — exact prompts to type on stream, in
   order, with timing.
