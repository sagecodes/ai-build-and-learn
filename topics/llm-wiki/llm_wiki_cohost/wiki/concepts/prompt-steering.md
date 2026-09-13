---
title: Prompt Steering
first_seen: topics/autoresearch/
weeks: [autoresearch]
---

The observation that the prompt (or strategy guide) is the primary lever for
directing an agent's behavior — not the code. Same code, different markdown,
different behavior. Changing a model's instructions reshapes what it does more
than changing the framework around it.

Karpathy's framing: "`program.md` as the program." Research org code lives
in markdown, not Python. Today's `program.md` is bare; tomorrow's encodes
taste, multiple agent roles, branch strategies, paper-reading subloops.

## How it appeared across the series

### Week 4 — AutoResearch (2026-04-17)

The starkest demonstration in the series so far. Three instruction files,
same code, dramatically different research behavior:

**`karpathy.md`** (default) — Claude's first instinct is structural surgery.
Depth axis: model size 8 → 3 layers. val_bpb 1.395 in 7 iterations.

**`karpathy_verbose.md`** — Adds explicit search/replace format rules and
common-mistake guidance. Default for local models, which are less reliable
at producing correctly-formatted edits.

**`overnight.md`** (diversity prompt) — Adds an explicit axis list (model
size, batch size, LRs, schedule, architecture, optimizer), an anti-fixation
rule ("if your last 3 experiments were on the same axis, switch"), and a nudge
toward architecture changes.

Without the diversity prompt: Gemma never touched model depth across 10
iterations (experiment 3). With it: depth was Gemma's *first* change
(experiment 4). Same model, same harness — the prompt was the only variable.

The combined result: Gemma with the diversity prompt (val_bpb 1.239) beat
Claude with the default prompt (1.395) and Gemma with Claude's history (1.296).
The right prompt on one model outperformed two models with the wrong prompt.

This connects directly to the LLM Wiki Cohost project: `CLAUDE.md` is the
schema that defines how Claude Code behaves when running wiki operations.
The principle is identical — the markdown file is the program.

## Open questions

- How does prompt steering interact with model capability? Does a stronger
  model respond more predictably to steering, or is it harder to constrain?
- What's the relationship between prompt steering and fine-tuning? At what
  point does baking behavior into weights outperform prompting it in?
