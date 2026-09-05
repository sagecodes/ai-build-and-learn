# LLM Wiki Cohost

An implementation of [Karpathy's LLM-maintained Wiki pattern](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) built to serve as a persistent, compounding cohost for the AI Build & Learn weekly stream.

The LLM (Claude Code) ingests each week's topic folder, writes and maintains interconnected markdown pages, and acts as an active cohost — preparing the host for upcoming episodes, surfacing gaps that become candidate future topics, and growing denser with every week.

---

## How it works

Traditional RAG re-discovers knowledge on every query. The LLM Wiki pattern inverts this: the LLM reads sources once and incrementally builds a structured wiki that synthesizes and accumulates insight. The wiki is the artifact; sources become input material.

| Layer | What it is | Location |
|---|---|---|
| **Sources** | Immutable inputs — each week's topic folder | `topics/*/` |
| **Wiki** | LLM-written markdown pages — the compounding artifact | `wiki/` |
| **Schema** | The program that defines structure, conventions, and workflows | `CLAUDE.md` |

---

## Four operations

| Command | What it does |
|---|---|
| `ingest topics/<folder>/` | Reads the week's sources, writes a topic page, updates all relevant concept and tool pages (typically 10–15 files) |
| `<question in natural language>` | Reads the index and relevant pages, synthesizes an answer, files valuable answers back into the wiki |
| `lint` | Health check: orphan pages, stale claims, missing cross-links, thin pages, concepts mentioned but lacking their own page |
| `prep <upcoming-topic>` | Produces a structured host brief: what prior weeks connect, what's genuinely new, anticipated chat questions, tools likely to come up |

---

## Wiki structure

```
wiki/
  index.md          ← content catalog; read first on every operation
  log.md            ← append-only record of every ingest, query, lint, and prep
  topics/           ← one page per series week
  concepts/         ← cross-cutting ideas (RAG, agents, MCP, RL…)
  tools/            ← libraries and platforms (Flyte, Neo4j, Tavily, FastMCP…)
```

**Current state:** 5 topics · 12 concepts · 8 tools

**Ingested (weeks 1–5):**

| # | Date | Topic |
|---|---|---|
| 1 | 2026-03-27 | MCP with FastMCP |
| 2 | 2026-04-03 | Agentic Search with Tavily |
| 3 | 2026-04-10 | Reinforcement Learning with OpenEnv |
| 4 | 2026-04-17 | AutoResearch |
| 5 | 2026-04-24 | Gemma 4 |

**Pending ingest (weeks 6–8 — demo live on stream):**

| # | Date | Topic |
|---|---|---|
| 6 | 2026-05-01 | Vector Stores |
| 7 | 2026-05-08 | Graph Data with Neo4j |
| 8 | 2026-05-15 | Karpathy's LLM Wiki ← *self-referential ingest* |

**Coming up (future episodes — stub pages seeded by forward-looking ingest):**

| Date | Topic | What it adds to the wiki |
|---|---|---|
| 2026-05-22 | Cognee: Memory Layer for Agents | Persistent agent memory, memory-as-graph |
| 2026-05-29 | Ragas: Evals for RAG & Memory | RAG evaluation metrics, retrieval quality |

---

## Usage

There is no app. The interface is a conversation with Claude Code in VS Code's terminal pane.

```
# Ingest a week's source material
ingest topics/vectorstore/

# Ask the wiki a question
how has RAG evolved across the series?

# Run a health check
lint

# Prepare for an upcoming episode
prep cognee
```

Claude Code loads `CLAUDE.md` automatically, so it knows the wiki's structure and workflows without being told. Every action is logged to `wiki/log.md`.

---

## Source set

| # | Date | Topic | Notes |
|---|---|---|---|
| 1 | 2026-03-27 | `topics/mcp/` | MCP + FastMCP. Bootstraps the wiki. |
| 2 | 2026-04-03 | `topics/tavily/` | First compounding moment — FastMCP page updated, not just referenced. |
| 3 | 2026-04-10 | `topics/openenv/` | RL and environments enter the wiki. |
| 4 | 2026-04-17 | `topics/autoresearch/` | Autonomous ML research, Karpathy's experiments. |
| 5 | 2026-04-24 | `topics/gemma4/` | Vision and multimodal LLMs. |
| 6 | 2026-05-01 | `topics/vectorstore/` | RAG enters the wiki properly. |
| 7 | 2026-05-08 | `topics/graphs-neo4j/` | RAG page mutates — graph-RAG section added. |
| 8 | 2026-05-15 | `topics/llm-wiki/` | Self-referential ingest. |

---

## Design decisions

- **No app, no vector DB.** `wiki/index.md` is enough poor-man's RAG at this scale (~50–100 pages). VS Code's file tree is the wiki browser.
- **No custom Python orchestration.** `CLAUDE.md` is the program. Claude Code is the agent.
- **Sources stay in place.** The LLM reads existing READMEs directly from `topics/*/` — no copying into a staging folder.
- **The schema co-evolves.** When output is corrected, the correction is treated as an implicit schema update to `CLAUDE.md`.

---

## Prior art

- [zhurudong/andrej-karpathy-llm-wiki](https://github.com/zhurudong/andrej-karpathy-llm-wiki) — most literal implementation; closest reference for the schema design
- [theafh/ai-modules](https://github.com/theafh/ai-modules) — per-repo wiki scoped to a single codebase; ships as a skill
- [Karpathy's original gist](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f) — the pattern this project implements
