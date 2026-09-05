---
title: Agents
first_seen: topics/mcp/
weeks: [mcp, tavily, openenv, autoresearch, gemma4, vectorstore, graphs-neo4j, llm-wiki]
---

LLM-powered systems that use tools to take actions, not just generate text.
An agent perceives its environment (context, tool results), decides what to do
(which tool to call or what to say), acts (calls the tool), and observes the
result — repeating until the task is done.

## How it appeared across the series

### Week 1 — MCP with FastMCP (2026-03-27)

Agents appeared as MCP clients — the consumer side of the protocol. Two clients
were built: one using the OpenAI Agents SDK, one using the Anthropic SDK. Both
connected to the same FastMCP server, demonstrating that MCP decouples the
agent (client) from the tool implementation (server).

The agent loop here is minimal: user message → model picks tools → tools execute
→ model synthesizes answer. The MCP server handles tool execution; the SDK
handles the loop.

The data analysis demo showed a multi-step agent: the model chains
load → filter → aggregate → chart across multiple tool calls in one session,
with server-side state persisting between calls.

### Week 2 — Agentic Search with Tavily (2026-04-03)

Agents appeared in two forms, both as web research assistants:

**Claude + FastMCP ReAct agent** — A manual ReAct loop where the agent chooses
between `tavily_search`, `tavily_extract`, and `tavily_crawl` based on the
query. The agent's system prompt specifies when to use each tool and quality
standards (cite sources, cross-verify claims, iterate on poor results). The
loop is visible in the console — each tool call is printed.

**LangGraph ReAct sub-agent** — Each sub-topic researcher in the pipeline is a
LangGraph ReAct subgraph running inside a Flyte task. The subgraph calls
`web_search` (Tavily) in a loop until it has enough information or reaches
`max_searches`. Multiple researchers run in parallel across the fan-out.

Key new idea: a system prompt as a first-class artifact. `system_prompt.py` is
a standalone module imported by `agent.py` — the agent's strategy (which tool
to use when, quality standards) is explicit and separately maintainable.

### Week 3 — OpenEnv (2026-04-10)

Two agents contrasted directly in the research demo:

**Traditional RL agent** — fixed action space (`tavily_search` only), keyword-
stuffing strategy, scalar keyword-match reward. High keyword score, low quality.
Demonstrates what happens when you optimize the wrong metric.

**Claude ReAct agent** — dynamic tool discovery (finds all 3 Tavily tools via
the environment's tool registry), reason-then-act loop, LLM-as-judge reward.
Consistently high quality score. Shows Claude acting as an RL agent inside a
structured environment loop, not just as a chatbot.

New pattern: agents running as parallel competitors in an "agent race" — three
OpenEnv agents on the same question, first to finish wins. Each gets an isolated
session in the same Docker container via `SUPPORTS_CONCURRENT_SESSIONS`.

### Week 4 — AutoResearch (2026-04-17)

Research agents — a new agent type in the series. The agent's action space is
code edits to a single file; the reward is a measurable metric from running
that code. Two agents compared directly:

**Claude Sonnet** — first instinct is structural (depth axis). Diagnosed real
GPU compatibility issues autonomously on a brand-new GB10 platform: Triton
ptxas version mismatch, Flash Attention 3 kernel absence, MFU bottleneck —
all diagnosed and worked around without human help across the first 4 iterations,
before any ML research happened. val_bpb 1.819 → 1.395 in 7 kept changes.

**Gemma 4 31B (via Ollama)** — first instinct is optimizer tuning (LR axis).
Without a diversity prompt, never touched model depth across 10 iterations.
With a diversity prompt, found the depth trick on its own as its first change
and ran 78 iterations to val_bpb 1.239 — best result across all experiments.

Key insight: same harness, genuinely different research "personalities." The
model is not just an interchangeable text predictor in this context —
its training distribution shapes what hypothesis it forms first.

### Week 5 — Gemma 4 (2026-04-24)

Local open-weight agent via Ollama. The `agent/` demo runs a ReAct loop with
Gemma 4 (31B by default) and five tools: calculator (AST-safe arithmetic),
current_datetime, web_search (DuckDuckGo), list_files, and read_file (sandbox-
scoped). Files dropped into `./sandbox/` become accessible to the agent.

Key contrast with prior weeks: the agent runs entirely locally — no Anthropic
API key, no cloud dependency. Demonstrates that the ReAct pattern is model-
agnostic; the same loop structure works with Claude, OpenAI, or a local Gemma.

The `gemma4-smart-gallary/` also behaves as an agent: it reasons about each
image in a folder, decides whether it matches a search query, and returns
structured results — but without an explicit ReAct loop. Vision + reasoning
as a single API call rather than a tool-calling loop.

### Week 8 — LLM Wiki (2026-05-15)

A new agent pattern in the series: **the schema-driven file agent**. Prior
weeks used agents to call tools, run research loops, or execute ML experiments.
This week the agent's action space is the local filesystem and its program
is a markdown file.

**Claude Code as agent.** No SDK, no framework, no orchestration layer.
Claude Code runs in VS Code's terminal pane, loads `CLAUDE.md` on startup,
and writes wiki files directly. The agent loop is: read schema → read sources
→ plan → write files → log. Natural language is the only interface.

**CLAUDE.md as program.** The schema is the complete specification of agent
behavior — not a system prompt discarded after the session, but a persistent
file that accumulates corrections. When output is wrong, you fix CLAUDE.md.
The next session inherits the fix. This is the programming model: the human
curates the schema; the agent executes it.

**The correction loop as programming.** Every prior agent pattern used a
fixed prompt or framework. This week's pattern makes the prompt a first-class
artifact that co-evolves with the agent's output quality. The schema is
version-controlled alongside the wiki it governs.

This pattern is the answer to "how does the series handle agents across
sessions?" — not a memory store, not a vector DB, but a schema file that
accumulates domain knowledge about how to do the task.

### Week 7 — Graph Data with Neo4j (2026-05-08)

Graph as agent context — a distinct pattern from the memory-store loop of
week 6. Three shapes:

**Structured knowledge traversal.** An agent with access to a knowledge graph
can answer multi-hop questions that flat vector retrieval can't: "which
policies apply to international Elite-tier customers?" requires walking
`RELATED` and `APPLIES_TO` edges, not just finding similar text chunks.
The `entity_retrieve` mode in `graph_rag_chatbot/` demonstrates this directly.

**Intelligent query routing as an agent decision.** `route_query` in
`graph_rag_chatbot/` is a Claude tool-use classifier that decides which
retrieval strategy to fire — Hybrid, Entity, or Community — based on the
question shape. This is agentic RAG: the agent decides *how* to retrieve,
not just *what* to retrieve.

**Memory of past actions.** The topic README notes graphs for agent context:
nodes as tool calls and observations, edges as `PROMPTED_BY`/`RESULTED_IN`.
This lets an agent answer "when did I last try X for this user?" — richer
than a flat conversation log when the agent must reason about its own history.
Not implemented this week, but it's the bridge to Cognee (week 9).

### Week 6 — Vector Stores (2026-05-01)

Two new agent patterns this week, both distinct from prior ReAct / research loops:

**Memory agent (`agent-memory-chroma/`)** — The agent's action space is not
tool calls or code edits; it's read+write access to a vector memory store. Each
turn: retrieve relevant past memories → answer grounded in memories → extract
new facts → write back. The "environment" is the agent's own accumulated
knowledge of one user. See [Agent Memory](../concepts/agent-memory.md) for the
full pattern.

**Perceptual agent (`screen-context-harness/`)** — No user messages drive this
agent; it runs autonomously on a timer. Perceives: screenshot every 5s. Acts:
caption → embed → store. Consolidates: every 60s, synthesizes 12 captions into
a context outline. The agent builds a model of the user's focus over time — a
passive observer rather than a task executor.

Both patterns answer the open question from prior weeks: "How does the series
handle agent memory across sessions?" The answer is a vector store, not a
database or context window.

## Open questions

- What agent frameworks does the series explore beyond the Agents SDK? (LangGraph, custom loops?)
- How does the series handle agent memory across sessions?
- What's the distinction between a "ReAct agent" and a simpler tool-calling loop?
