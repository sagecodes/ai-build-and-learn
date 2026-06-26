---
title: Graph Data with Neo4j
date: 2026-05-08
folder: topics/graphs-neo4j/
concepts: [knowledge-graphs, rag, agents]
tools: [neo4j, flyte, gradio, pgvector]
---

Two projects explored graph databases for RAG and agent context: a paper
citation network on a DGX devbox (`graphrag-neo4j-flyte/`) and a production
knowledge-graph chatbot over the same 15 Everstorm PDFs from vector week
(`graph_rag_chatbot/`). The side-by-side with week 6 makes the graph
vs. vector tradeoff concrete.

## What was built

**`graphrag-neo4j-flyte/`** — Graph RAG over ~400 academic papers fetched
from Semantic Scholar. Neo4j 5 deployed as a Flyte `AppEnvironment`; a
three-task pipeline (fetch → embed → load) builds a graph of papers, authors,
categories, and citation edges. A Gradio chat UI offers three retrieval modes:
pure vector, vector + 1-hop expand, and hybrid Reciprocal Rank Fusion (RRF).
Gemma 4 via vLLM answers queries with graph context included.

**`graph_rag_chatbot/`** — Production GraphRAG chatbot over 15 Everstorm
Outfitters PDFs. Neo4j AuraDB Free (cloud, no setup) replaces the devbox
instance. A six-stage ingest pipeline (chunk → extract entities with Claude
tool use → load graph → create HNSW index → resolve entities → detect
communities → summarize with Claude) builds a full knowledge graph. Three
retrieval modes routed by a Claude classifier: Hybrid (vector + MENTIONS
traversal), Entity (RELATED neighborhood), Community (summary similarity).
Deployed as a persistent Union app.

## Key decisions

- **AuraDB Free over GCP VM.** Zero setup, sufficient for demo-scale graphs.
  Community detection runs in Python (`python-louvain` + NetworkX) since
  GDS library isn't available on the free tier.
- **Direct Cypher over framework abstractions.** `neo4j-graphrag`,
  `langchain-neo4j`, and similar libraries add overhead without benefit at
  this scale. Parameterized Cypher queries are clearer and sufficient.
- **Claude tool use for entity extraction.** Structured output via tool use
  eliminates JSON parse failures entirely. Open-ended entity extraction
  produces too much noise; constraining to a domain ontology
  (`PRODUCT`, `POLICY`, `PROGRAM`, `TIER`, `BENEFIT`, `CONDITION`, `PROCESS`)
  dramatically improves extraction consistency.
- **Entity normalization is critical.** Without it, "Everstorm", "Everstorm
  Outfitters", and "the Company" become three disconnected nodes. Fix:
  embed all entity names, merge nodes with cosine similarity ≥ 0.95.
- **HTTP API not Bolt for devbox Neo4j.** Knative's queue-proxy sidecar
  only routes HTTP; Bolt (TCP/7687) doesn't pass through. Full Cypher
  surface including vector index queries works over the HTTP API.

## Connections

- [Knowledge Graphs](../concepts/knowledge-graphs.md) — new concept page;
  graph schema, entity extraction, community detection all introduced here
- [RAG](../concepts/rag.md) — Graph RAG section added; the most significant
  RAG extension in the series
- [Agents](../concepts/agents.md) — graph as agent context: memory of past
  actions, structured knowledge traversal
- [Neo4j](../tools/neo4j.md) — new tool page
- [Flyte / Union](../tools/flyte.md) — Neo4j as a Flyte AppEnvironment;
  parallel ingest fan-out
- [Gradio](../tools/gradio.md) — retrieval mode selector + graph context panels
- [pgvector](../tools/pgvector.md) — direct comparison: same documents,
  different retrieval architecture
