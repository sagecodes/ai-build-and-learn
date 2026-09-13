---
title: AutoResearch
date: 2026-04-17
folder: topics/autoresearch/
concepts: [autonomous-ml-research, prompt-steering, reinforcement-learning, reward-functions, agents]
tools: [flyte, gradio, ollama]
---

Karpathy's autonomous ML research loop — an AI agent edits `train.py`, trains
for 5 minutes, checks whether `val_bpb` improved, keeps or reverts, and repeats
overnight. Two sub-projects: a GCP T4 adaptation and a local-LLM variant that
ran four experiments comparing Claude vs Gemma 4 as research agents.

## What was built

**`autoresearch-tinystories-t4/`** — Karpathy's loop adapted for a GCP T4 GPU
(16GB, vs H100 target). Six T4-specific changes: SDPA instead of Flash
Attention 3, reduced batch sizes, LLLL window pattern, TinyStories instead of
climbmix-400b, AdamW instead of MuonAdamW. Two orchestration modes: plain
`agent.py` loop and `flyte_workflow.py` (3 tasks per experiment: propose →
train → evaluate). Results logged to Google Firestore, visualized in a Gradio
dashboard. Two overnight runs proved the concept: val_bpb 3.4 → 1.2 (64%)
and 3.7 → 1.4 (61%), ~21% success rate in both runs.

**`local-llm-autoresearch/`** — Same loop with Gemma 4 31B (via Ollama)
instead of Claude, on a DGX Spark (128GB unified VRAM). Four experiments:

| Experiment | Agent | Iterations | Best val_bpb | Strategy |
|---|---|---|---|---|
| 1 | Claude Sonnet | 7 | 1.395 | depth axis (8 to 3) |
| 2 | Gemma + Claude's history | Gemma 4 31B | ~15 | 1.296 | optimizer (built on Claude's structural wins) |
| 3 | Gemma clean slate | Gemma 4 31B | 10 | 1.547 | optimizer only (never touched depth) |
| 4 | Gemma overnight + diversity prompt | Gemma 4 31B | 78 | **1.239** | everything (depth + optimizer + window + batch) |

## Key decisions

**`program.md` as the lever.** Same code, different markdown file, different
research behavior. Gemma never touched model depth without the diversity prompt
(`overnight.md`). With it, depth was its first change. The brief is the
programmable surface, not the code.

**Different models, different strategies.** Claude's first instinct: structural
surgery (shrink the model). Gemma's first instinct: optimizer tuning (LR axis).
Same harness, genuinely different research personalities. The combined strategy
(Claude's depth wins + Gemma's optimizer tuning) outperformed either alone.

**TOTAL_BATCH_SIZE as the dominant lever for time-budgeted training.** Both T4
overnight runs independently discovered the same thing: halving TOTAL_BATCH_SIZE
gives more optimizer steps per 5-minute budget, which dominates everything else.
The agent found this by experiment 9-12 in both independent runs.

**Flyte for per-experiment observability.** `flyte_workflow.py` breaks each
experiment into 3 visible tasks (propose → train → evaluate). When Claude returns
an unparseable response, only `propose_change_task` fails; the others are skipped.
Makes failure diagnosis instant.

## Connections

- [Autonomous ML Research](../concepts/autonomous-ml-research.md) — the autoresearch pattern
- [Prompt Steering](../concepts/prompt-steering.md) — program.md as the programming surface
- [Reinforcement Learning](../concepts/reinforcement-learning.md) — autoresearch as closed-loop RL
- [Reward Functions](../concepts/reward-functions.md) — val_bpb as the reward signal
- [Agents](../concepts/agents.md) — Claude and Gemma as research agents with different strategies
- [Flyte / Union](../tools/flyte.md) — per-experiment task visibility, Flyte TUI
- [Gradio](../tools/gradio.md) — Firestore-backed monitoring dashboard
- [Ollama](../tools/ollama.md) — local model serving for Gemma 4
