# Wiki Log

Append-only. One entry per operation. Never edit existing entries.

<!-- format:
## [YYYY-MM-DD] <operation> | <subject>
Pages written: …
Pages updated: …
Notes: …
-->

## [2026-05-15] prep | cognee
Brief written to: wiki/prep-cognee.md
Pages read: index.md, concepts/agent-memory.md, concepts/knowledge-graphs.md, concepts/rag.md, concepts/llm-wiki-pattern.md, tools/chroma.md, tools/neo4j.md, topics/cognee/README.md
Notes: Cognee lands as synthesis of weeks 6-7-8. Five anticipated chat questions drafted. Arc connection (W6 manual memory → W7 graph → W8 RAG inversion → W9 library) is the key narrative thread.

## [2026-05-15] ingest | topics/llm-wiki/
Pages written: topics/llm-wiki.md (new), concepts/llm-wiki-pattern.md (new), tools/claude-code.md (new)
Pages updated: concepts/rag.md, concepts/agents.md, concepts/autonomous-ml-research.md, index.md
Notes: 3 source files read (topic README + llm_wiki_cohost/RESEARCH.md + llm_wiki_cohost/CLAUDE.md). 8 files touched. Self-referential ingest — the wiki writes about the pattern used to build it. RAG concept page gains the RAG inversion framing. Karpathy's second contribution lands on the autonomous-ml-research page. Agents page now spans all 8 weeks.

## [2026-05-15] ingest | topics/graphs-neo4j/
Pages written: topics/graphs-neo4j.md (new), tools/neo4j.md (new), concepts/knowledge-graphs.md (new)
Pages updated: concepts/rag.md, concepts/agents.md, tools/flyte.md, tools/gradio.md, tools/pgvector.md, index.md
Notes: 4 source files read (topic README + 2 sub-project READMEs + 1 RESEARCH.md). 10 files touched. Graph RAG section added to RAG concept page — the most significant RAG extension in the series. pgvector page updated with direct comparison. Same 15 Everstorm PDFs as week 6 makes the vector vs. graph tradeoff concrete.

## [2026-05-15] ingest | topics/vectorstore/
Pages written: topics/vectorstore.md (new), concepts/rag.md (new), concepts/embeddings.md (new), concepts/agent-memory.md (new), tools/chroma.md (new), tools/pgvector.md (new)
Pages updated: concepts/agents.md, tools/flyte.md, tools/gradio.md, tools/ollama.md, index.md
Notes: 6 source files read (topic README + 4 sub-project READMEs + 1 RESEARCH.md). 12 files touched. RAG enters the wiki properly with 3 implementations. Agent memory introduced as a distinct read+write pattern. RAG concept page seeds the graph-RAG extension for week 7.

## [2026-05-12] ingest | topics/gemma4/
Pages written: topics/gemma4.md (new), concepts/multimodal-llms.md (new), concepts/long-context.md (new), concepts/structured-output.md (new), tools/vertex-ai.md (new)
Pages updated: concepts/agents.md, concepts/react-loop.md, tools/ollama.md, tools/gradio.md, tools/flyte.md, index.md
Notes: 8 sub-project sources read (README.md × 7 + RESEARCH.md × 1). 12 files touched. Biggest week yet. Long-context page seeds the RAG contrast for weeks 6-7.

## [2026-05-12] ingest | topics/autoresearch/
Pages written: topics/autoresearch.md (new), concepts/autonomous-ml-research.md (new), concepts/prompt-steering.md (new), tools/ollama.md (new)
Pages updated: concepts/reinforcement-learning.md, concepts/reward-functions.md, concepts/agents.md, tools/flyte.md, tools/gradio.md, index.md
Notes: 4 source files read (README + 2 sub-project READMEs + RESULTS.md). 11 files touched. Karpathy entity page born. Prompt-steering concept captures week's deepest insight.

## [2026-05-12] ingest | topics/openenv/
Pages written: topics/openenv.md (new), concepts/reinforcement-learning.md (new), concepts/reward-functions.md (new), tools/openenv.md (new)
Pages updated: concepts/agents.md, concepts/research-pipelines.md, tools/flyte.md, tools/gradio.md, tools/tavily.md, index.md
Notes: 4 sub-project READMEs read. 11 files touched. New concepts: RL training paradigms, reward hacking, LLM-as-judge.

## [2026-05-12] ingest | topics/tavily/
Pages written: topics/tavily.md (new), concepts/react-loop.md (new), concepts/research-pipelines.md (new), tools/tavily.md (new), tools/langgraph.md (new)
Pages updated: concepts/agents.md, concepts/tool-use.md, tools/fastmcp.md, tools/flyte.md, tools/gradio.md, index.md
Notes: First compounding ingest — 5 existing pages extended with week 2 sections. 3 sub-project READMEs read. 12 files touched.

## [2026-05-12] ingest | topics/mcp/
Pages written: topics/mcp.md (new), concepts/mcp-protocol.md (new), concepts/tool-use.md (new), concepts/agents.md (new), tools/fastmcp.md (new), tools/gradio.md (new), tools/flyte.md (new)
Pages updated: index.md
Notes: Bootstrapped wiki from MCP week. Single source file (README only, no RESEARCH.md). 7 new pages + index = 8 files touched. Dress-rehearsal ingest.
