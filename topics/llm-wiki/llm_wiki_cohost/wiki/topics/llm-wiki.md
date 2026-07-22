---
title: Karpathy's LLM Wiki
date: 2026-05-15
folder: topics/llm-wiki/
concepts: [llm-wiki-pattern, rag, agents]
tools: [claude-code]
---

The series' self-referential week. The topic is the wiki itself — the LLM Wiki
pattern from Andrej Karpathy, implemented as a persistent cohost for the AI
Build & Learn stream. The artifact built during this week is the file you are
currently reading.

This is the only week where the ingest and the subject of the ingest are the
same thing.

## What was built

**`llm_wiki_cohost/`** — An implementation of Karpathy's LLM-maintained Wiki
pattern. No Python app, no Gradio UI, no orchestration framework. The stack is:

- **Sources** — immutable inputs; the prior weeks' topic folders (`topics/*/`)
- **Wiki** — LLM-written markdown pages; the compounding artifact (`wiki/`)
- **Schema** — `CLAUDE.md`; the program that defines structure and workflows

Claude Code is the agent. `CLAUDE.md` is the program. VS Code's file tree is
the wiki browser. Every action — ingest, query, lint, prep — is a natural
language instruction in the chat pane; Claude reads the schema, executes the
operation, and writes files directly.

**Four operations:**

| Operation | Trigger | Output |
|---|---|---|
| Ingest | `ingest topics/<folder>/` | Topic page + updated concept/tool pages (10–15 files) |
| Query | Natural language question | Answer synthesized from wiki; filed back if worth keeping |
| Lint | `lint` | Orphan pages, stale claims, missing cross-links, research backlog |
| Prep | `prep <upcoming-topic>` | Host brief: prior connections, what's new, anticipated questions |

**Three cohost extensions on Karpathy's base:**
- **Forward-looking ingest** — seeds stub pages for upcoming episodes
- **Research backlog from lint** — gaps become ranked candidate future topics
- **`prep` as a first-class operation** — turns the wiki from a passive store into an active cohost

## Key decisions

- **No app.** Canonical Karpathy pattern: no custom orchestration, no web UI.
  The CLAUDE.md schema is the only artifact beyond the wiki itself.
- **Sources stay in place.** The LLM reads existing READMEs from `topics/*/`
  directly. No staging folder, no copying.
- **Schema co-evolves.** When output is corrected, the correction is treated as
  an implicit schema update. The schema is the program; correction is the
  programming interface.
- **The wiki proposes its own next moves.** Lint produces a ranked research
  backlog of open questions — gaps where multiple sources gesture at something
  none of them define. The wiki becomes the source of future episode ideas.

## Connections

- [LLM Wiki Pattern](../concepts/llm-wiki-pattern.md) — the concept this week
  implements; the inversion of RAG
- [RAG](../concepts/rag.md) — LLM Wiki is explicitly positioned as the
  alternative to re-retrieval; sources become input material, not the artifact
- [Agents](../concepts/agents.md) — Claude Code as the agent; CLAUDE.md as
  the program; a new agent pattern in the series
- [Claude Code](../tools/claude-code.md) — the agent driving the entire wiki
