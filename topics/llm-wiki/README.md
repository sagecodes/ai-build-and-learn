Welcome to AI Build & Learn a weekly AI engineering stream where we pick a new topic and learn by building together.

​​​This event is about building an LLM-maintained Wiki, a pattern from Andrej Karpathy for turning raw sources into a persistent, compounding knowledge base instead of re-retrieving from documents on every query.

We'll explore how this differs from traditional RAG, the three-layer stack (sources, wiki, schema), and the core operations (ingest, query, lint) that keep the wiki healthy as it grows.

Some things to look up to get started:
- Karpathy's LLM Wiki gist: https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f
- Community implementations linked in the gist comments (SwarmVault, Kompl, Link, OmegaWiki, etc.)


## What is the LLM Wiki pattern?

The idea is to treat an LLM as a long-running *editor* of a small knowledge base instead of a one-shot answerer over re-retrieved chunks. Every time you bring in a new source (a URL, a note, a paper), the LLM reads it, writes a brief immutable summary into `raw/`, and then revises a set of human-readable concept pages in `pages/` so they incorporate the new knowledge. Over time the wiki becomes a curated, cross-linked view of everything you have read, and queries are answered against that view rather than against the raw documents.

Three operations keep the wiki alive, all of them backed by the same LLM:

- **Ingest** pull a source in, summarize it, integrate it into existing pages, create new pages where needed.
- **Query** pick the relevant pages from the wiki's own index, then answer with `[[slug]]` citations into those pages.
- **Lint** audit the wiki for contradictions, orphan pages, broken links, and missing cross-references; report what needs human attention.

The schema is co-evolved: an `AGENTS.md` file lives inside the wiki itself and tells the LLM what page shape and conventions to follow. You change the schema by editing that file; the next ingest picks it up.

## How it works

```
            sources                              question
            (urls, notes, text)                  (chat)
                  │                                 │
                  ▼                                 ▼
            ┌─────────────┐                   ┌─────────────┐
            │     LLM     │   reads pages     │     LLM     │
            │   INGEST    │ ◄──────────────── │    QUERY    │
            └──────┬──────┘                   └──────┬──────┘
              writes                                 │
                  │                                  ▼
                  │                          answer + [[slug]] citations
                  ▼
   ┌──────────────────────────────────────────────────┐
   │                    The Wiki                      │
   │                                                  │
   │   raw/<slug>.md      immutable source summaries  │
   │   pages/<slug>.md    LLM-maintained concept pages│
   │   index.md           auto-generated TOC          │
   │   log.md             append-only history         │
   │   AGENTS.md          schema and conventions      │
   └──────────────────────┬───────────────────────────┘
                          │
                          ▼
                    ┌─────────────┐
                    │     LLM     │       orphans, broken links,
                    │    LINT     │ ────► contradictions, missing
                    └─────────────┘       pages and cross-refs
```

Every box marked `LLM` is the same model speaking the same conventions; the only thing that changes between operations is the prompt and which slice of the wiki is shown to it.

## How this differs from RAG

A typical RAG stack indexes chunks of raw documents once, then runs vector search at query time and asks the LLM to synthesise an answer from the retrieved chunks. The wiki flips this: the synthesis is done at *ingest* time and saved, so the LLM at query time reads pre-distilled concept pages instead of raw chunks. Three practical consequences:

1. The retrieval surface (concept pages) is small and human-readable, so the picker step is cheap and the result is auditable.
2. Corrections compound: editing a page once fixes every future answer that consults it, instead of re-ranking different bad chunks each query.
3. Costs flip from query-time to ingest-time, which is the right side to spend on for slow-changing knowledge bases.

The implementation in this repo lives under [`flyte-llm-wiki/`](./flyte-llm-wiki/): a Flyte 2 pipeline that builds the wiki end-to-end, plus a Gradio chat app that lets you ingest sources interactively against the same underlying Dir.


​​​Resources

​​​- GitHub: https://github.com/sagecodes/ai-build-and-learn
​​​- Events Calendar: https://luma.com/ai-builders-and-learners
​​​- Slack (Discuss during the week): https://slack.flyte.org/
​​​- Hosted by Sage Elliott: https://www.linkedin.com/in/sageelliott/