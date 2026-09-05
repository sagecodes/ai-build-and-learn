---
title: Knowledge Graphs
first_seen: topics/graphs-neo4j/
weeks: [graphs-neo4j]
---

A knowledge graph stores entities as nodes and relationships as typed,
directed edges. The value lives in the edge types — they encode human
judgments about how things connect that vector embeddings can't capture.
The modeling exercise: enumerate three or four edge types you care about
(`CITES`, `REPORTS_TO`, `DEPENDS_ON`, `MENTIONS`) and ask which questions
become trivial Cypher and which remain hard.

**When graphs beat vectors:**

| Question type | Vector RAG | Graph RAG |
|---|---|---|
| Factual / single-chunk | ✅ | ✅ |
| Relationship-aware | ⚠️ | ✅ |
| Multi-hop / cross-document | ❌ | ✅ |
| Thematic / global | ❌ | ✅ (community summaries) |

**When not to reach for a graph:** plain text corpora with no structured
relationships gain little; if you can't enumerate three edge types users
will ask about, modeling will drift; graph traversal adds visible latency
to the hot path of streaming chat.

## How it appeared across the series

### Week 7 — Graph Data with Neo4j (2026-05-08)

**Graph schema** for `graph_rag_chatbot/` (Everstorm policy documents):

```
(Document)-[:HAS_CHUNK]→(Chunk)-[:MENTIONS]→(Entity)
(Entity)-[:RELATED {type, description}]→(Entity)
(Entity)-[:BELONGS_TO]→(Community)
```

Entity ontology constrained to a domain-specific set (`PRODUCT`, `POLICY`,
`PROGRAM`, `TIER`, `BENEFIT`, `CONDITION`, `PROCESS`) — open-ended extraction
produces too much noise. Relationship types: `HAS_POLICY`, `QUALIFIES_FOR`,
`REQUIRES`, `APPLIES_TO`, `PART_OF`, `COVERS`.

**Entity extraction** via Claude tool use (structured output). Tool use
eliminates JSON parse failures entirely vs. prompting for raw JSON.
Multi-shot examples in `<examples>` tags improve consistency. Runs per-PDF
in parallel inside `process_pdf` tasks.

**Entity normalization** (critical): without it, "Everstorm", "Everstorm
Outfitters", and "the Company" become three disconnected nodes. Fix: embed
all extracted entity names with `gte-small`, merge nodes with cosine
similarity ≥ 0.95 (`resolve_entities` in `ingest/enrichment.py`).

**Community detection**: groups entities into clusters. Ideal tool is Neo4j
GDS (Leiden/Louvain), unavailable on AuraDB Free. Workaround: export entity
relationships to NetworkX, run `python-louvain`, write community IDs back
to Neo4j, then have Claude summarize each community. Pre-computed at ingest
time so broad "overview" questions don't require a per-query LLM call.

**Academic paper graph** in `graphrag-neo4j-flyte/`:
```
(Paper)-[:CITES]→(Paper)
(Paper)-[:AUTHORED_BY]→(Author)
(Paper)-[:IN_CATEGORY]→(Category)
```
Papers fetched from Semantic Scholar sorted by `citationCount:desc` —
foundational papers (original RAG paper, BERT, GPT-3) land in the corpus.
`CITES` edges encode intellectual lineage that vector similarity misses
entirely.

## Open questions

- How does this graph schema evolve when Cognee (week 9) builds a memory
  graph? Entity nodes vs. atomic fact nodes — are they the same pattern?
- Text-to-Cypher (LLM writes the query itself from a question) is flagged
  as a next idea but not built. Where does it break vs. the fixed-mode
  routing approach?
- Community detection quality is limited by AuraDB Free's lack of GDS.
  How much does Louvain in Python vs. Leiden in GDS change community
  coherence at this scale?
