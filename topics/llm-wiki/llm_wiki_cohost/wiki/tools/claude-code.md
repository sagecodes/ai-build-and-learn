---
title: Claude Code
weeks: [llm-wiki]
---

Anthropic's official CLI for Claude. Runs in VS Code's terminal pane, loads
`CLAUDE.md` automatically on startup, and writes files directly into the
repository. In the LLM Wiki pattern, Claude Code is the agent; `CLAUDE.md`
is the program; the file tree is the wiki browser.

Key property for the LLM Wiki use case: Claude Code operates on the local
filesystem directly — no API integration layer, no custom orchestration. The
entire wiki is driven by natural language instructions in the chat pane.

## Usage across the series

### Week 8 — LLM Wiki (2026-05-15)

The sole agent for all wiki operations. Every ingest, query, lint, and prep
is a natural language instruction; Claude Code reads the schema, executes
the operation, and writes files. The workflow:

1. Claude Code starts; auto-loads `CLAUDE.md` from the project directory
2. User types an operation trigger (`ingest topics/vectorstore/`, `lint`, `prep cognee`)
3. Claude Code reads sources, proposes a plan for ingests touching >5 files
4. User approves; Claude Code writes all files
5. Every action is logged to `wiki/log.md`

**Schema as program.** `CLAUDE.md` is the complete specification of Claude
Code's behavior for this project: page types, frontmatter conventions, link
format, operation workflows, behavior rules. Correcting Claude Code's output
is the mechanism for updating the program — fix CLAUDE.md once, the behavior
persists across sessions.

**The correction loop.** When Claude Code does something suboptimal, the fix
goes into CLAUDE.md not into a conversation. The schema co-evolves. This is
the primary difference from a one-off prompt: the schema accumulates the
refinements of every session.
