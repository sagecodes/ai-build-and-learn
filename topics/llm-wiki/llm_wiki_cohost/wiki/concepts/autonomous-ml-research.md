---
title: Autonomous ML Research
first_seen: topics/autoresearch/
weeks: [autoresearch, llm-wiki]
---

A closed-loop agent pattern for running ML experiments unattended: the agent
reads a strategy guide, proposes one focused change to a training script, runs
a fixed-budget training job, measures whether the metric improved, keeps or
reverts, and repeats. The constraints are the insight — a tight sandbox makes
experiments comparable and uncheatable.

Karpathy's formulation (autoresearch):
- One editable file (`train.py`)
- One fixed-budget training run per experiment (5 minutes, wall clock)
- One metric (`val_bpb`, lower is better)
- One strategy guide (`program.md`, natural language)
- One logbook (`results.tsv`, append-only)

The surprising empirical claim: with these constraints, current frontier models
can run closed-loop ML research and produce real gains, unattended, overnight.

## How it appeared across the series

### Week 4 — AutoResearch (2026-04-17)

Two implementations run in parallel, each demonstrating different aspects of
the pattern:

**GCP T4 overnight runs (autoresearch-tinystories-t4).** Two independent runs
(plain Python loop vs Flyte orchestration) both reached ~21% success rate and
discovered the same dominant lever without human guidance: halving
`TOTAL_BATCH_SIZE` gives more optimizer steps per 5-minute budget and dominates
all other changes. Run 1: val_bpb 3.4 → 1.2 (64% improvement, 80 experiments).
Run 2: val_bpb 3.7 → 1.4 (61%, 58 experiments). The independent rediscovery
of the same insight across two runs is the key validation.

**Multi-model comparison (local-llm-autoresearch).** Four experiments on a DGX
Spark comparing Claude and Gemma 4 as research agents. Best result: Gemma 4
with a diversity prompt ran 78 iterations and reached val_bpb 1.239 — beating
all shorter runs by combining depth reduction (Claude's strategy) with optimizer
tuning (Gemma's default) and window pattern exploration.

The pattern generalizes past `train.py`. The same loop — fixed single-file
edit surface, fixed budget, one measurable metric — can drive research in any
domain where an agent can propose testable changes and measure an outcome.

### Week 8 — LLM Wiki (2026-05-15)

Karpathy's second contribution to the series. The LLM Wiki pattern is a
different expression of the same underlying idea as AutoResearch: give the
LLM a tight sandbox, a persistent artifact, and a schema that defines what
"good work" looks like — then let it run.

**AutoResearch:** sandbox = one `train.py` file; metric = `val_bpb`; schema =
`program.md`; artifact = the improved training script + results log.

**LLM Wiki:** sandbox = one wiki directory; metric = coverage + coherence;
schema = `CLAUDE.md`; artifact = the compounding wiki itself.

The structural parallel is exact. Both patterns: fixed single-artifact edit
surface, measurable improvement criterion, natural language program, appendonly
log. The difference is the domain — ML experiments vs. knowledge synthesis —
and the human's role: AutoResearch runs overnight unattended; the LLM Wiki
is interactive, with the human directing operations.

Karpathy's insight across both: **constrained sandboxes make agents tractable**.
An agent with an unbounded action space produces unpredictable results. An
agent with a well-defined edit surface, a clear quality criterion, and a
schema that specifies expected behavior produces consistent, improvable work.

## Open questions

- At what scale does the fixed-budget constraint break down (longer experiments,
  larger models, slower metrics)?
- Does the series revisit autoresearch with a non-ML task (e.g., optimizing
  code performance, tuning a prompt)?
- How does `program.md` evolve as the agent runs more experiments —
  does it get updated based on what's been tried?
