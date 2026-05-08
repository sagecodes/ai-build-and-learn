Welcome to AI Build & Learn a weekly AI engineering stream where we pick a new topic and learn by building together.

​​​This event is about building with Neo4j, a graph database, for RAG or Agent context applications.

We'll explore how graph databases differ from vector databases, when to reach for knowledge graphs, and how to combine graph traversal with semantic search (GraphRAG) to give agents richer context.

Some things to look up to get started:
- Neo4j: https://neo4j.com/
- Neo4j GraphRAG: https://neo4j.com/labs/genai-ecosystem/graphrag/

**Working demo in this repo:** [`graphrag-neo4j-flyte/`](./graphrag-neo4j-flyte/)
deploys Neo4j as a Flyte 2 app, loads ~400 papers from Semantic Scholar
into a graph (papers, authors, categories, citations), and serves a
Gradio chat with three retrieval modes (vector, vector+expand, hybrid
Reciprocal Rank Fusion / RRF) wired to Gemma 4 over vLLM.

## Why graphs for RAG and agents

Vector retrieval is great at one thing: finding text whose embedding is
close to the query. That covers a lot of "what does this document say?"
questions, but it falls short whenever the answer lives in
*relationships* rather than topic similarity.

Three places pure vector misses:

1. **Intellectual lineage.** Two papers can be tightly related (one
   cites the other, one extends the other) without their abstracts
   sharing keywords. RAG and Self-RAG are a textbook case. Vector misses
   the link; an explicit `CITES` edge captures it.
2. **Authority.** The most-cited paper in a topic is rarely the tightest
   abstract match for a casual query. Graphs let you ask "what's central
   in this neighborhood?" instead of "what's similar to this string?"
3. **Compositional structure.** Org charts, dependency trees, supply
   chains, conversations, codebases: the *shape* of the data is the
   answer. A vector store flattens it; a graph keeps it.

The value lives in the edge types. A useful modeling exercise: list the
three or four edge types you care about (`CITES`, `REPORTS_TO`,
`DEPENDS_ON`, `FOLLOWED_BY`, …) and ask which questions become trivial
Cypher and which remain hard. If your domain has none of those, a graph
probably isn't the right tool.

## How a minimal Graph RAG works

The simplest GraphRAG is just vector RAG with one extra hop. Side by
side:

```
Vector RAG (baseline)
─────────────────────

  query ─▶ embed ─▶ ANN ─▶ top-k chunks ─▶ LLM ─▶ answer


Graph RAG (vector + 1-hop expand)
─────────────────────────────────

  query ─▶ embed ─▶ ANN ─▶ top-k chunks ──┐
                                          │
                  ┌───────────────────────┘
                  ▼
          1-hop walk across
          CITES, AUTHORED_BY,
          IN_CATEGORY
                  │
                  ▼
          chunks + neighbor                  answer with
          titles + edges ─▶ LLM ─▶          lineage cited
```

The graph half costs one extra Cypher query and a few hundred extra
tokens of context. The payoff is that the LLM now sees the *structure*
around the retrieved chunks, not just the chunks themselves.

## Common GraphRAG patterns

Worth holding in your head when you decide how much graph to bring:

**Vector + 1-hop expand.**

```
  query ─▶ vector top-k ─▶ 1-hop walk ─▶ chunks + edges ─▶ LLM
```

Run a vector query, then walk one edge from each hit and add the
neighbors to the LLM context. Cheap, predictable, surfaces direct
dependencies (citations, prerequisites, "see also"s). The lightest
pattern to add to an existing vector RAG.

**Hybrid retrieval (Reciprocal Rank Fusion, RRF).**

```
            ┌─▶ vector top-k     ─┐
  query ────┤                     ├─▶ RRF fuse ─▶ LLM
            └─▶ graph cohort      ─┘
                (e.g. most-cited
                 in category)
```

Run vector AND a graph-only query (e.g. "most-cited papers in the same
category"), then fuse with reciprocal-rank fusion: for each list, score
a doc as `1 / (K + rank)`, sum across lists. Top items in either list
win, shared items win bigger, and raw scores on different scales
(cosine vs citation count) don't have to be normalized. Surfaces
authoritative-but-not-keyword-matching results. Good when authority
matters as much as topic.

**Text-to-Cypher.**

```
  query ─▶ LLM (with schema) ─▶ Cypher ─▶ Neo4j ─▶ rows ─▶ LLM ─▶ answer
```

The LLM writes the Cypher query itself from the user's question, given
the schema. Powerful for structured questions ("how many engineers
report to Alice?") but brittle: schema drift, query failures,
hallucinated edge types. Pair with a fallback mode.

**Community-summary GraphRAG (Microsoft style).**

```
  docs ─▶ LLM extract ─▶ knowledge graph
                              │
                              ▼  Leiden / community detection
                         communities
                              │
                              ▼  LLM summary per community
                         summaries
                              │
   query ──────────────────▶  ▼
                              LLM ─▶ answer
```

Build the graph from documents using an LLM, run community detection,
summarize each community, and answer global questions ("what are the
main themes?") off the summaries. Heavy lift, big payoff for narrative
corpora.

## Graphs for agent context

Agents are orchestration over state, and most of the state you care
about is relational:

- **Memory of past actions.** "Which tools have I called for this user,
  with what arguments, in what order?" One node per action, edges back
  to the prompts and outputs that fed it. Beats a flat conversation log
  when the agent has to reason about its own history.
- **Knowledge over structured data.** Org charts, codebases, products,
  customer accounts. A vector store can find the right sentence but
  can't answer "everyone two hops from this person" or "all services
  depending on this library."
- **Planning over a state graph.** Represent goals, subgoals, and
  prerequisites as a DAG. The agent walks it forward, pruning branches
  as new evidence comes in. Easier to debug than freeform prompt chains.

The pragmatic split: vector store for "what was said?", graph for "how
does it connect?". Most production GraphRAG systems use both.

## Use cases: where graphs shine in practice

A graph is the right tool when you can name the relationships up front.
Concrete domains where this pays off, with the kinds of nodes and edges
you'd typically model:

- **Citation networks** (academic papers, legal precedents). Nodes:
  papers, authors. Edges: `CITES`, `AUTHORED_BY`. Surfaces lineage and
  authority. *(This is what the demo in this repo builds.)*
- **Code intelligence.** Nodes: files, functions, classes. Edges:
  `CALLS`, `IMPORTS`, `INHERITS`. Powers "what breaks if I change
  this?" agents and impact-analysis tools.
- **Org charts and access control.** Nodes: people, roles, teams.
  Edges: `REPORTS_TO`, `MEMBER_OF`, `OWNS`. Answers "everyone who can
  approve this" without parsing prose policies.
- **Product knowledge graphs.** Nodes: products, features, customers,
  tickets. Edges: `HAS_FEATURE`, `BOUGHT`, `RAN_INTO`. Customer-support
  agents that route by topology, not just keyword match.
- **Drug discovery and biomedical research.** Nodes: compounds,
  proteins, diseases. Edges: `BINDS_TO`, `TREATS`, `INTERACTS_WITH`.
  Multi-hop reasoning ("compounds that bind X via Y are linked to
  disease Z") that flat embeddings can't do.
- **Supply chain and logistics.** Nodes: parts, suppliers, factories.
  Edges: `SUPPLIED_BY`, `ASSEMBLED_AT`, `SHIPS_TO`. "What breaks if
  this supplier goes down for two weeks?"
- **Recommendation systems.** Nodes: users, items, sessions. Edges:
  `VIEWED`, `PURCHASED`, `FOLLOWED_BY`. Heterogeneous graphs beat
  co-occurrence matrices in sparse domains.
- **Healthcare records.** Nodes: patients, conditions, treatments.
  Edges: `DIAGNOSED_WITH`, `TREATED_BY`, `RESPONDED_TO`. Differential
  diagnosis, outcome paths, and cohort lookup.
- **Security and fraud detection.** Nodes: accounts, transactions,
  devices. Edges: `TRANSFERRED_TO`, `LOGGED_IN_FROM`, `SHARED_WITH`.
  Multi-hop fraud rings invisible to flat tables.
- **Conversation memory for agents.** Nodes: turns, tool calls,
  observations. Edges: `PROMPTED_BY`, `RESULTED_IN`, `REFERENCES`.
  Lets the agent answer "when did I last try X for this user?"

What these have in common:

- You can enumerate three or more edge types up front.
- The interesting questions are multi-hop: "two steps from X",
  "everything connected to Y through Z", "all paths from A to B".
- A vector embedding can find the topic but not the structure. The
  *shape* of the data is part of the answer.

## When not to reach for a graph

- **The data isn't relational.** Plain text corpora with no structured
  relationships gain little.
- **You can't name the edge types yet.** If you can't enumerate three
  edge types your users will ask about, modeling will drift and queries
  will be ad hoc.
- **Latency budget is tight.** Graph traversal in Neo4j is fast but
  never as fast as a single ANN lookup. Mixing a multi-hop traversal
  into the hot path of a streaming chat reply costs visible tokens of
  latency.

​​​Resources

​​​- GitHub: https://github.com/sagecodes/ai-build-and-learn
​​​- Events Calendar: https://luma.com/ai-builders-and-learners
​​​- Slack (Discuss during the week): https://slack.flyte.org/
​​​- Hosted by Sage Elliott: https://www.linkedin.com/in/sageelliott/
