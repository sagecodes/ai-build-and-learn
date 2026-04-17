# Local-LLM AutoResearch

Run karpathy's autoresearch loop with local open models (Gemma 4, Qwen Coder)
instead of Claude, orchestrated with Flyte for per-iteration visibility.

```
local-llm-autoresearch/
├── README.md                <- you are here
├── driver.py                <- one iteration = propose -> train -> keep/revert
├── local_agent.py           <- Ollama HTTP client + search/replace edit + git apply
├── workflow.py              <- Flyte wrapper, renders HTML reports + Plotly chart
├── plot_progress.py         <- standalone Plotly chart generator
├── requirements.txt         <- all deps (training + flyte + plotly)
├── instructions/
│   ├── karpathy.md          <- default brief (copy of upstream/program.md)
│   ├── karpathy_verbose.md  <- verbose variant for small models
│   └── overnight.md         <- diverse-exploration prompt for long runs
└── saved_runs/              <- archived results + charts
    ├── gemma-clean-slate.tsv/.html
    └── gemma-with-history.tsv/.html
```

All modes share the same `upstream/` clone one directory up. The local agent
edits `upstream/train.py`, trains there, and logs to `upstream/results.tsv`.

---

## Our experiments

We ran autoresearch on a DGX Spark (NVIDIA GB10, Blackwell, CUDA 13,
128 GB unified VRAM) with three different setups. Each one told a
different story.

### Experiment 1: Claude Code on a brand-new GPU

The first time we pointed Claude Code at karpathy's repo on GB10, the
first ~4 iterations weren't research at all. They were platform
shakedown. The agent diagnosed and worked around real hardware issues
entirely on its own:

1. Triton's bundled `ptxas` was CUDA 12.8; GB10 needs 13.0. The agent
   set `TRITON_PTXAS_PATH` to the system binary. Commit, retry.
2. `kernels-community/flash-attn3` ships no prebuilt `sm_121a` kernel.
   The agent swapped FA3 for `flex_attention`. Commit, retry.
3. `flex_attention` ran but at **1.21% MFU** (vs ~40% on H100). The
   agent correctly diagnosed kernel overhead as the bottleneck and
   tried SDPA next.

This is the *same* read-log, form-hypothesis, make-one-change, measure,
keep-or-revert loop that program.md describes for ML research, applied to
hardware compatibility instead. Watching it debug a GPU you yourself
don't fully understand is, frankly, a more compelling demo than watching
it tune `n_embd`.

Once the kernel situation stabilized, the loop took val_bpb from
**1.819 to 1.395** (-23% relative) across six kept commits, plus a
clean discard that proved the reject-and-revert path works:

| # | commit  | val_bpb  | VRAM   | status   | description                                          |
|---|---------|----------|--------|----------|------------------------------------------------------|
| 0 | 567db9f | 1.818913 | 44.0 G | keep     | baseline (FA3 to flex_attention for sm_121a compat)   |
| 1 | e1181b8 | 1.821681 | 44.0 G | keep     | SDPA instead of flex, drop sliding window. *Kept despite +0.003 regression because the code was simpler and SDPA is faster downstream* |
| 2 | 9dfa100 | 1.793186 | 87.2 G | keep     | DEVICE_BATCH_SIZE 128 to 256 (grad_accum 2 to 1)     |
| 3 | 3b9d79d | **1.467645** | 27.9 G | keep     | DEPTH 8 to 4 (50M to 11.5M params; **110 steps vs 39**) |
| 4 | 4d93489 | 1.424601 | 13.1 G | keep     | DEPTH 4 to 2 (3.5M params; 247 steps)                |
| 5 | 29c26a9 | **1.395246** | 23.2 G | keep     | DEPTH 2 to 3 (10.7M params; 132 steps). Found the sweet spot |
| 6 | 5a815c4 | 1.503361 | 30.8 G | **discard** | ASPECT_RATIO 64 to 128 (17.9M params, only 96 steps). Reverted |

Key observations:

1. **The biggest single jump** (-0.32 val_bpb at iter 3) came from
   *shrinking* the model. The agent figured out that on GB10's throughput
   budget the model was severely undertrained (only 39 steps in 5 minutes),
   so a smaller model that takes more steps wins. Karpathy notes this exact
   tradeoff in the upstream README, but the agent rediscovered it empirically
   from the run logs without being told.

2. **Iters 4-5 are a clean two-step hill climb.** Iter 4 pushed the
   smaller-is-better hypothesis further (DEPTH 4 to 2: still wins). Iter 5
   *backed off* (2 to 3) and discovered the local optimum sat between them.
   Binary-search behaviour over a single hyperparameter, executed across
   two commits.

3. **Iter 6 is the first discard.** Widening (ASPECT_RATIO 64 to 128) lost
   the same step-budget tradeoff. Run logged, branch reset, val_bpb unchanged.

### Experiment 2: Gemma 4 with Claude's history

Next we ran Gemma 4 (31B, via Ollama) as the agent, but starting from the
same branch with Claude's `results.tsv` still in the prompt. Gemma could
*see* what Claude had already tried.

Result: Gemma never touched depth (it could see depth=3 was already the
sweet spot from Claude's log). Instead it went pure optimizer:
WARMDOWN_RATIO, EMBEDDING_LR, MATRIX_LR hill-climbs, then a batch size
reduction. Best val_bpb: **1.296**, beating Claude's 1.395.

The lesson: Gemma built on Claude's structural wins with optimizer tuning
Claude never tried. The *combined* strategy (architecture from Claude +
optimizer from Gemma) outperformed either model alone.

### Experiment 3: Gemma 4 from a clean slate

To test whether Gemma's optimizer focus was its own "personality" or
just because it read Claude's history, we ran it from scratch. Fresh
master, empty results.tsv, karpathy defaults (depth=8).

| # | val_bpb | status | description |
|---|---------|--------|-------------|
| 0 | 1.821 | baseline | depth=8 (karpathy default) |
| 1 | -- | crash | depth 8 to 10 (tried bigger, TIMEOUT) |
| 2 | 1.853 | discard | MATRIX_LR down 0.04 to 0.02 (wrong direction) |
| 3 | **1.634** | keep | TOTAL_BATCH_SIZE halved (more optimizer steps) |
| 4 | **1.594** | keep | WARMDOWN_RATIO 0.5 to 0.2 |
| 5-7 | **1.593 to 1.561** | keep x3 | MATRIX_LR hill-climb: 0.04 to 0.05 to 0.06 to 0.08 |
| 8 | 1.572 | discard | MATRIX_LR 0.08 to 0.1 (overshot, correctly rejected) |
| 9 | **1.547** | keep | EMBEDDING_LR 0.6 to 0.8 |
| 10 | 1.548 | discard | EMBEDDING_LR 0.8 to 1.0 (overshot again, rejected) |

**Gemma never touched depth.** Not once in 10 iterations. It went straight
to optimizer tuning and stayed there. Claude's first instinct was to shrink
the model; Gemma's was to crank the learning rate. Same problem, same
harness, genuinely different research strategies.

Gemma's best (1.547) is worse than Claude's (1.395) because it never
discovered the depth trick. But its optimizer hill-climbing was clean:
pushed MATRIX_LR through four steps, correctly rejected the overshoot at
0.1, pivoted to a different knob (EMBEDDING_LR), repeated the pattern.

### Experiment 4: overnight run

We ran Gemma 4 overnight with `overnight.md`, a prompt that explicitly
nudges the model to explore diverse axes (architecture, optimizer, batch
size, attention patterns) instead of fixating on one.

```bash
tmux new -s overnight
flyte run --local --tui workflow.py run_autoresearch \
    --tag gemma-overnight --iterations 100 \
    --agent local --model gemma4:31b \
    --instructions instructions/overnight.md
```

The question: with the diversity prompt, does Gemma find the depth trick
on its own? Check `saved_runs/` for results.

### What we learned

1. **Same harness, different model, different strategy.** Claude did
   structural surgery (depth axis). Gemma did optimizer tuning (LR axis).
   Neither tried the other's approach. The model's "research personality"
   is a real variable.

2. **The combined strategy wins.** Gemma building on Claude's depth wins
   (1.296) beat either model alone (Claude: 1.395, Gemma solo: 1.547).
   Multi-model research pipelines aren't just a cute idea.

3. **The prompt steers the research.** The `overnight.md` prompt nudges
   toward diversity. If it works, it proves karpathy's thesis: program.md
   is the real programming surface. Same code, different markdown,
   different research behavior.

---

## Setup

```bash
# From the build-learn repo root:
uv venv .venv --python 3.11
source .venv/bin/activate

# --index-strategy lets uv fall back to PyPI for packages that also exist
# on the PyTorch cu128 index (e.g. requests).
uv pip install --index-strategy unsafe-best-match \
    -r topics/autoresearch/local-llm-autoresearch/requirements.txt

# Clone upstream if not already done (see parent README)
cd topics/autoresearch
git clone https://github.com/karpathy/autoresearch.git upstream
cd upstream && python prepare.py && cd ..
```

### Ollama setup

```bash
# Install: https://ollama.com/download
# Pull a model
ollama pull gemma4:31b        # ~17 GB, general reasoning
ollama pull qwen3-coder-next  # ~50 GB, strongest open coder
```

**Keep the model loaded (important).** By default Ollama evicts the model
from GPU after 5 minutes of inactivity. Since autoresearch alternates
between agent calls and training runs, the model gets evicted every
iteration and reloading from disk takes ~4.5 minutes. Fix:

```bash
# Tell Ollama to keep the model loaded indefinitely (no restart needed)
curl -s http://localhost:11434/api/generate -d '{"model":"gemma4:31b","keep_alive":-1}'

# Warm the model (first load takes ~4 min, subsequent calls are instant)
ollama run gemma4:31b "say hello"
```

The DGX Spark has 128 GB unified VRAM, more than enough for both the
local model (~17 GB) and training (~23 GB) to coexist.

If Ollama is installed as a system service, you can't `pkill` it without
sudo. Use `sudo systemctl stop ollama` if you need to restart, or just
use the `keep_alive` curl on the running instance.

---

## Running

All commands run from inside this directory (`local-llm-autoresearch/`).

### Headless (Mode B)

```bash
# Default: Claude Sonnet as the agent
python driver.py --tag demo --iterations 3

# Local model
python driver.py --tag local-demo --iterations 3 \
    --agent local --model gemma4:31b

# Custom instructions
python driver.py --tag overnight --iterations 100 \
    --agent local --model gemma4:31b \
    --instructions instructions/overnight.md
```

### Flyte TUI (Mode C)

Same loop, but each iteration is a Flyte task with a live HTML report
(including an interactive Plotly progress chart). The `--tui` flag pops
the live status panel:

```bash
flyte run --local --tui workflow.py run_autoresearch \
    --tag demo --iterations 3 --agent local --model gemma4:31b
```

Use `tmux` so the TUI survives SSH/tunnel disconnects:

```bash
tmux new -s autoresearch
flyte run --local --tui workflow.py run_autoresearch \
    --tag demo --iterations 3 --agent local --model gemma4:31b
# Ctrl-B, D to detach. tmux attach -t autoresearch to reconnect.
```

The Flyte TUI shows:

- A node per iteration (baseline + N iterations) executing sequentially
- Per-iteration HTML report: status badge (BASELINE / KEEP / DISCARD /
  CRASH), val_bpb, peak VRAM, change description, log tail
- Rolling summary with Plotly chart + table on the workflow node,
  updating after each iteration

---

## How the local agent edits code

The local model gets ONE shot per iteration to output aider-style
`<<<<<<< SEARCH / ======= / >>>>>>> REPLACE` blocks. We do literal
substitution against `train.py`. No unified diffs (small models can't
reliably compute `@@ -line,count @@` headers). No retry or reflection.
If the output is malformed or doesn't match, the iteration is logged as
`discard` with a descriptive `[TAG]` and the loop moves on.

We tried unified diffs first. Result on qwen3-coder-next: **0/3 applied**
("corrupt patch", "patch failed at line 599"). The reasoning in the
descriptions was correct ("reduce DEPTH from 3 to 2"), but the format was
broken. Switching to search/replace got 3/3 to apply on the next test.
This is also why aider, Cursor, and similar tools use this format.

### Failure tags in results.tsv

| Tag | Meaning |
|-----|---------|
| `[OLLAMA UNREACHABLE]`  | Can't reach Ollama. Check `ollama serve` is running. |
| `[INVALID OUTPUT: no SEARCH/REPLACE blocks]` | Model emitted prose instead of following the format. |
| `[SEARCH NOT FOUND ...]` | SEARCH text doesn't appear in `train.py`. Usually whitespace drift. |
| `[AMBIGUOUS MATCH ...]`  | SEARCH text appears multiple times. Add more context. |
| `[TIMEOUT]` | Training exceeded the 10-minute hard cap. |

---

## Customizing the agent's brief

`instructions/` is the customization surface. Drop in a new markdown file
and point the driver at it with `--instructions path/to/yours.md`. Three
ship by default:

- `karpathy.md` - verbatim copy of `upstream/program.md`. Default for
  `--agent claude`.
- `karpathy_verbose.md` - explicit search/replace format rules, concrete
  edit examples, "common mistakes" list. Default for `--agent local`.
- `overnight.md` - all of the above plus diversity guidance: explicit axis
  list (model size, batch size, LRs, schedule, architecture, optimizer),
  anti-fixation rule ("if your last 3 experiments were on the same axis,
  switch"), and a nudge toward architecture changes.

The brief is the actual lever. Same code, different markdown, different
research behavior.

---

## Generating charts

`plot_progress.py` generates interactive Plotly charts from `results.tsv`,
similar to karpathy's `progress.png` but interactive (hover for details,
zoom, dark theme):

```bash
# From the current run
python plot_progress.py --title "My Run"

# From a saved run
python plot_progress.py --file saved_runs/gemma-clean-slate.tsv \
    --title "Gemma 4 Clean Slate"

# Hide VRAM bars (auto-hidden when VRAM barely varies)
python plot_progress.py --no-vram
```

Charts are also embedded in the Flyte workflow report and update live
after each iteration.

---

## Notes

- The driver passes the recent `results.tsv` rows to the agent each
  iteration so the agent has memory of what's been tried, even though
  each call is a fresh session.
- Hard caps: agent has 10 min to respond (configurable via
  `AUTORESEARCH_AGENT_TIMEOUT`), training has 10 min before it's killed.
- Configurable env vars: `OLLAMA_URL` (default `http://localhost:11434`),
  `AUTORESEARCH_LOCAL_MODEL` (default `qwen3-coder-next`),
  `AUTORESEARCH_AGENT_TIMEOUT` (default `600` seconds),
  `AUTORESEARCH_REPORT_PORT` (default `8080` for Flyte report links).
