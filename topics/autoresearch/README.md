# AutoResearch

[Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch) is an
autonomous LLM-pretraining research loop. You give an AI agent a small but
real GPT training setup; it edits `train.py`, trains for ~5 minutes, checks
whether `val_bpb` improved, keeps or reverts, and repeats. Wake up the next
morning to a log of experiments and (hopefully) a better model.

```
The autoresearch loop:

  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │   ┌──────────┐    ┌───────────┐    ┌──────────┐  │
  │   │  Agent   │───>│ train.py  │───>│ Train    │  │
  │   │ proposes │    │  edited   │    │ 5 min    │  │
  │   │ a change │    │           │    │ budget   │  │
  │   └──────────┘    └───────────┘    └────┬─────┘  │
  │        ^                                │        │
  │        │         ┌──────────────┐       │        │
  │        │         │  val_bpb     │       │        │
  │        └─────────│  improved?   │<──────┘        │
  │                  │              │                 │
  │                  │  yes: keep   │                 │
  │                  │  no: revert  │                 │
  │                  └──────────────┘                 │
  │                                                  │
  └──────────────────────────────────────────────────┘
               repeat until stopped
```

## What is this really doing?

Stripped of the hype, autoresearch is small:

- A fixed training harness (`prepare.py` + a read-only eval).
- One file the agent is allowed to edit (`train.py`).
- One markdown file (`program.md`) telling an off-the-shelf coding agent what
  to do.
- A logbook convention (`results.tsv` + per-experiment git commits).
- A `while true:` loop with permission prompts disabled.

There is no novel agent framework, no clever ML, no new optimization
technique. The agent is Claude (or Codex, or a local model) in a loop,
hitting a metric.

So why does anyone care? Three reasons, in decreasing order of substance:

1. **The constraints are the insight, not the code.** Single editable file +
   read-only eval + fixed 5-minute budget = experiments are *comparable* and
   *uncheatable*. Most "AI does science" demos handwave this and the agent
   ends up gaming the metric or growing the surface area until results aren't
   apples-to-apples. The discipline is the thing. Autoresearch is a
   methodology demo dressed up as a tool.

2. **It actually works.** Karpathy's `progress.png` shows monotonic `val_bpb`
   improvement over ~100 iterations. That is the surprising empirical claim:
   with a tight enough sandbox, current frontier models can run closed-loop
   ML research and produce real gains, today, unattended. Not a 2027
   prediction. A thing that runs overnight on one GPU.

3. **`program.md` as the program.** Karpathy's bet is that "research org
   code" lives in markdown, not Python. Today's `program.md` is bare;
   tomorrow's encodes taste, multiple agent roles, branch strategies,
   paper-reading subloops. The repo is a wedge for "you program research
   orgs by writing prompts." Whether that pays off is a different question.

The takeaway most people miss: this isn't really about LLM pretraining.
It's a 700-line lesson in how to bound an agent so it can do useful
closed-loop work. The shape generalizes well past `train.py`.

## Dataset

The model trains on **karpathy's ClimbMix** (`karpathy/climbmix-400b-shuffle`),
a curated 400B-token mix of high-quality web text, code, and educational
content. Not TinyStories. The model won't produce coherent text at this scale;
the exercise is purely about minimizing the `val_bpb` metric.

## Hardware

Karpathy's defaults assume a single H100. The DGX Spark (GB10) works, but
expect different absolute val_bpb numbers. The 5-minute budget is fair within
your machine, not across machines. If you're on smaller hardware, check the
**Platform support** section of `upstream/README.md` for tuning hints
(`DEPTH`, `MAX_SEQ_LEN`, `WINDOW_PATTERN=L`, `TOTAL_BATCH_SIZE`).

**GB10 platform fix:** karpathy's default `train.py` uses Flash Attention 3,
which has no prebuilt kernel for Blackwell (sm_121a). Before running on GB10,
you need two changes to `train.py`:
1. Add `os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda/bin/ptxas"` (Triton
   bundles CUDA 12.8 ptxas; GB10 needs CUDA 13)
2. Replace the `fa3.flash_attn_func(...)` call with
   `F.scaled_dot_product_attention(...)` (PyTorch SDPA works everywhere)

## One-time setup

```bash
# Clone karpathy's repo as upstream/ (gitignored from build-learn).
cd topics/autoresearch
git clone https://github.com/karpathy/autoresearch.git upstream

# Install dependencies via uv
cd upstream
uv sync

# One-time data prep (~2 min)
uv run prepare.py

# Sanity check: one baseline training run (~5 min)
uv run train.py
```

## Running with Claude Code

The bare karpathy flow. From inside `upstream/`:

```bash
cd upstream
claude --permission-mode bypassPermissions
```

Then prompt:

> Hi, have a look at program.md and let's kick off a new experiment. Let's do
> the setup first.

The agent creates a branch, runs the baseline, and starts looping until you
stop it (Ctrl-C). `bypassPermissions` means the agent can read, edit, run git,
and train without asking for approval each time. Best for an unattended
overnight run.

## Running with a local LLM

See [`local-llm-autoresearch/`](local-llm-autoresearch/) for a wrapper that
runs the same loop with open models (Gemma 4, Qwen Coder, etc.) via Ollama,
orchestrated with Flyte for visibility. Includes experiment results, charts,
and instructions for overnight runs.

## Where results land

- `upstream/results.tsv` - the canonical log
  (`commit  val_bpb  memory_gb  status  description`). Untracked by
  upstream's git on purpose; it is a per-machine artifact.
- `upstream/run.log` - last training run's stdout/stderr.
- Git history on branch `autoresearch/<tag>` inside `upstream/` - one
  commit per kept experiment.

## Resetting a run

```bash
cd upstream
git checkout master
git branch -D autoresearch/<tag>
rm -f results.tsv run.log
```

## Notes

- `upstream/` is its own git repo. Agent commits never pollute `build-learn`.
- Karpathy's `program.md` has an explicit **NEVER STOP** rule: once the
  experiment loop begins, the agent will not pause to ask if it should
  continue. Ctrl-C to stop it.
- Training always runs for a fixed 5-minute time budget (wall clock,
  excluding startup/compilation), regardless of the model configuration.
  The metric is `val_bpb` (validation bits per byte), lower is better.
