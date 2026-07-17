# Open-source video generation on Flyte (DGX Spark)

Render prompts across several open-source video models and get **one side-by-side
report with clips that actually play in the browser**, plus the full-resolution
`.mp4`s as artifacts.

```
                    ┌─ fetch wan21-t2v-1.3b (29GB, cached) ─┐   ┌─ generate ─┐
  prompts ──────────┤                                        ├───┤            ├──> report
                    └─ fetch wan22-ti2v-5b  (34GB, cached) ─┘   └─ generate ─┘   (clips play)

  image-to-video:   prompt ──(sd-turbo)──> first frame ──(wan22-ti2v-5b)──> clip
```

Sibling of the image-generation demo in `topics/image-generation`; if you've read
that one, this will feel familiar. The three things that are genuinely new here are
**playable video in a Flyte report**, **selective model downloads** (the difference
between a 53GB pull and a 372GB one), and **memory guardrails** (on this box an
oversized model doesn't OOM, it hangs the whole machine).

---

## Quickstart

```bash
# 1. GPU devbox
flyte start devbox --gpu

# 2. CLI venv (on the host)
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python flyte==2.2.1 kubernetes 'connectrpc==0.10.0'

# 3. Point at the cluster. (.flyte/ is gitignored, so a fresh clone needs this.)
mkdir -p .flyte && cat > .flyte/config.yaml <<'YAML'
admin:
  endpoint: dns:///localhost:30080
  insecure: true
image:
  builder: local
task:
  domain: development
  project: video-generation
YAML

# 4. HF token (only needed for gated repos; every default model here is ungated)
flyte create secret HF_TOKEN --value hf_xxx

# 5. Generate. First run also builds the image and downloads ~63GB of weights,
#    both cached from then on.
.venv/bin/flyte run compare_pipeline.py compare \
    --prompts '["a red panda barista pouring latte art, steam rising, cozy cafe, 50mm"]'

# 6. Pull the full-resolution clips down
.venv/bin/python download_outputs.py <run_name>
```

Faster first signal (one model, ~29GB instead of ~63GB):

```bash
.venv/bin/flyte run compare_pipeline.py generate_one \
    --model_key wan21-t2v-1.3b --prompts '["a corgi astronaut, studio light"]'
```

**Image-to-video** (make a first frame, then animate it) is a first-class entrypoint
too. See [Image to video](#2-image-to-video):

```bash
.venv/bin/flyte run compare_pipeline.py animate \
    --prompts '["a red fox trotting through tall grass, morning light"]'
```

Open the run's **Report** tab. The clips play inline.

---

## What's here

| file | what it does |
|---|---|
| `models.py` | the model registry: repos, pipeline classes, sampler defaults, **download patterns**, memory estimates |
| `config.py` | Flyte images, environments, GPU resources, and the DGX Spark env vars |
| `videogen_core.py` | Flyte-free: load, generate, encode mp4, **render the report** (this is where playback lives) |
| `compare_pipeline.py` | the pipeline: `fetch_weights` (cached), `generate_for_model`, `compare`, `animate` |
| `long_video.py` | **long video by chaining**: generate a clip, feed its last frame into the next |
| `vace.py` | **video-to-video**: VACE restyle (edge-map control) + first-last-frame |
| `app.py` | Gradio studio: a thin CPU launcher that submits runs and links the report |
| `run_local.py` | host GPU, no Flyte. The fastest way to check a model loads at all |
| `download_outputs.py` | pull the `.mp4` artifacts out of the devbox blob store |

---

## The experiments

Each is a section below. Status is honest: **verified** means a clip came out and we
looked at it, not that a task went green.

| # | experiment | entrypoint | status | the headline |
|---|---|---|---|---|
| 1 | [Text to video](#1-text-to-video-the-model-comparison) | `compare` | ✅ verified | the grid: size vs licence vs architecture |
| 2 | [Image to video](#2-image-to-video) | `animate` | ✅ verified | **the first frame is the bottleneck**: 6.2x less noise from one argument |
| 3 | [Motion prompts](#the-i2v-model-is-prompted-too---motion_prompts) | `animate --motion_prompts` | ✅ verified | the prompt is a real lever: **2.5x** motion energy from the same frame |
| 4 | [LoRA to animate](#4-lora-to-animate) | `animate --lora papercut` | ⚠️ built, **unrun** | fine-tuned look, moving, with no video training |
| 5 | [Length & consistency: 4 approaches](#5-length-and-consistency-four-approaches-measured) | `long_video.py`, `vace.py` | ✅ verified | chain / renorm / bookend / anchored. **The beats mattered more than any algorithm** (+8.9 -> +1.6, free) |
| 5b | [`bookend`](#bookend-the-fix-for-the-trajectory-the-chain-wont-advance) | `vace.py bookend` | ✅ verified | endpoints pinned: the only run where the subject actually **recedes** |
| 5c | [`anchored`](#anchored-the-experiment-that-tests-the-actual-claim) | `vace.py anchored` | ✅ verified | `reference_images` holds identity through the turn that broke the chain |
| 5d | [`refine`](#refine-chain-first-then-re-render-against-one-anchor) | `vace.py refine` | ❌ **negative result** | edge control discards appearance; made a good clip worse |
| 6 | [SkyReels 14B](#6-skyreels-14b-long-video-done-properly) | `generate_one` | ⏳ overnight | loads + runs, but **283 s/step, ~4h/clip**. Latent history vs our RGB hand-off |
| 7 | [Video to video (VACE)](#7-video-to-video-vace) | `vace.py restyle` | ✅ verified | the only model that takes video IN. 19GB, 323s/clip. **Name the subject** or it invents one |
| 8 | [Krea v2v](#8-krea-realtime-video-researched-not-built) | — | 📋 researched | 14B Apache-2.0 causal. 51.8GB with an allowlist, 137.6GB without |

The through-line, if you want one: **#5 breaks, #6 is the principled fix, #7 has the
tool (`reference_images`) for the part #6 doesn't fix.** See
[exposure bias](#what-we-hit-has-a-name-exposure-bias---renorm).

---

## The knobs (what to turn when it looks wrong)

Every one of these is a CLI flag. Defaults are chosen to be honest rather than
flattering: `--renorm` ships **off** so the drift stays visible, and the sampler
defaults are demo-sized rather than the model cards' (see [the models](#the-models)).

| knob | where | default | range | what it does |
|---|---|---|---|---|
| `--strength` | `long_video.py polish` | `0.4` | 0.2-0.8 | **The most important one.** True v2v: how far to move off the source frames. Too low = nothing changes; too high = each window drifts off alone and the seams come back. |
| `--renorm` | `long_video.py long_video` | `0.0` (off) | 0-1 | Re-anchors each hand-off frame's per-channel mean/std to chunk 1's. Helps in proportion to how **stationary** the scene is: candle **-65%** contrast creep, boat -46%, walking person **no help at all**. |
| `--keep_pct` | `vace.py restyle` | `8.0` | 4-15 | Edge-map density (percentile threshold). Lower = only strong contours, VACE freer to invent; higher = more structure, VACE more constrained. **8% is tuned**: a fixed threshold gave a 79% white map that VACE couldn't read. |
| `--conditioning_scale` | `vace.py restyle` | `1.0` | 0.5-1.5 | How hard VACE is pulled toward the control video. |
| `--use_anchor` / `--no-use_anchor` | `vace.py anchored` | on | flag | Pass `reference_images` to every chunk. `--no-use_anchor` is the control arm. |
| `--lora_scale` | `compare_pipeline.py animate` | `1.0` | 0-1.5 | LoRA fusion strength. |
| `--motion_prompts` | `compare_pipeline.py animate` | reuses `--prompts` | list | Separates the **motion** prompt from the **frame** prompt. Measured **2.5x** motion energy from "leaps out of the water" vs "drifts gently", same frame. |
| `--steps` / `--guidance` | most entrypoints | `-1` = the spec's | — | `-1` sentinels mean "use the model's default". |
| `flow_shift` | `models.py` (spec field) | `0` = don't touch | 3.0 / 5.0 | Wan's flow-matching schedule shift. **3.0 for 480P, 5.0 for 720P** per the VACE card. Left at 0 for models verified on their shipped scheduler. |
| `extra_call_kwargs` | `models.py` (spec field) | `()` | — | Family-specific `__call__` params. SkyReels' `ar_step` / `overlap_history` live here, and **without them the long-video path is unreachable**. |

**Two that will surprise you:**

- **`ar_step=5` makes SkyReels run 50 iterations, not the 30 you asked for.** Async mode
  expands the schedule, so real cost is ~1.7x the step count. Budget on 50.
- **A LoRA trigger word is a knob.** `papercut`, `pixel art` — without the token in the
  prompt the adapter loads, fuses, and does nothing. Registry LoRAs prepend it for you;
  a raw repo id does not.

---

## The models

| key | repo | download | resident (bf16) | license | i2v | audio |
|---|---|---|---|---|---|---|
| `wan21-t2v-1.3b` | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` | 29GB | ~15GB | Apache-2.0 | no | no |
| `wan22-ti2v-5b` **(default)** | `Wan-AI/Wan2.2-TI2V-5B-Diffusers` | 34GB | ~26GB | Apache-2.0 | **yes** | no |
| `ltx2-distilled` **(showpiece)** | `diffusers/LTX-2.3-Distilled-Diffusers` | 95GB | ~72GB | LTX-2 Community | **yes** | **yes** |
| `wan22-t2v-a14b` | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | 126GB | ~68GB | Apache-2.0 | no | no |
| `cogvideox-5b` | `zai-org/CogVideoX-5b` | 22GB | ~20GB | CogVideoX | yes | no |

All ungated: no license clicks, no token needed.

**The default pair** is the two Wan models, because together they're a 63GB pull and
each clip lands in a couple of minutes, so a cold first run finishes *during* the
stream rather than after it.

**The headline comparison** is Wan vs LTX-2: Apache-2.0-and-small against
frontier-with-audio. It isn't the default only because LTX-2 is a 95GB first
download.

```bash
flyte run compare_pipeline.py compare \
    --prompts '["waves crashing on black volcanic rock, slow motion"]' \
    --models '["wan22-ti2v-5b","ltx2-distilled"]'
```

LTX-2.3 is the only open model here that generates a **synced audio track**. It's
worth unmuting the player in the report; nothing else in open source does this.

### A note on what is and isn't real

There is a lot of SEO blogspam claiming "Wan 2.5 / 2.7, Apache-2.0, open weights."
Those repos **do not exist** on Hugging Face. The newest official Wan open weights
are 2.2 (Aug 2025); 2.5+ is API-only. Similarly, several model cards still tell you
to install `diffusers` from `main`; that's stale, everything here is in the stable
`diffusers>=0.39.0` release.

---

## Playing video in a Flyte report

This was the interesting problem. The constraint is tight: the console renders a
report either as a document or by injecting the HTML via `innerHTML`, and a CSP
drops anything fetched from outside. So **no `<script>` tags, no external assets**.

The way through is that an HTML5 `<video>` element needs no JavaScript at all:

```html
<video controls loop muted autoplay playsinline src="data:video/mp4;base64,...">
```

`controls` gives you scrub and play/pause for free, `loop` makes a 3-second clip
readable, `muted autoplay` is allowed to start without a user gesture, and the
base64 data URI carries the bytes inside the HTML. You get **real playback**, not a
poster frame.

Two consequences worth knowing:

**The report clip is not the artifact clip.** Base64 inflates by 4/3 and every clip
in a grid lands in one HTML document, so report clips are re-encoded down to a 512px
long side (`REPORT_MAX_SIDE`). The full-resolution originals, audio intact, are in
the task's output `Dir`; `download_outputs.py` pulls them.

**Every cell also carries a frame strip.** Six frames sampled across the clip,
composited into a single JPEG, under the player. It's the fallback if playback ever
fails, but it earns its place anyway: temporal drift and identity collapse are
obvious in a strip and genuinely easy to miss while a 3-second clip loops past you.
If a clip somehow exceeds the embed budget, the player is dropped and the strip is
shown alone, so a report always renders.

---

## Memory guardrails (do not skip this on a Spark)

On a discrete GPU, asking for too much memory raises `OutOfMemoryError` and you move
on. **On GB10 the GPU allocates from the same DRAM the OS lives in.** There is
nothing to OOM *into*: the driver spins on `NV_ERR_NO_MEMORY`, the kernel reclaims
from everything else, and the box locks up instead of raising. You lose the machine,
not just the run.

Video models are the biggest thing you'll run here (LTX-2.3 is ~72GB resident
against a 119.7GiB pool shared with the OS, rustfs, and every other pod), so
`videogen_core.prepare_gpu()` runs before every load, in this order:

1. **Free** anything still held. A previous pipeline in the same process (the app
   switching models, a local loop) keeps its weights in PyTorch's cache until
   dropped; loading a 72GB model on top of a 26GB corpse is how you wedge the box.
2. **Cap** the process with `set_per_process_memory_fraction` (default 0.90,
   `VIDEOGEN_MEM_FRACTION` to override). Past the cap PyTorch raises a normal
   `OutOfMemoryError`, which Flyte's retry can act on, instead of hanging the host.
   **This is the guardrail that matters.**
3. **Check** the model actually fits, and fail with an actionable message *before*
   spending 20 minutes downloading and loading 95GB to find out.

Each report cell shows the **peak GPU allocation** for that clip, so you can see how
close to the ceiling a model really ran.

### The double-copy trap (this one actually bit us)

The obvious way to load a pipeline is `from_pretrained(...).to("cuda")`. On a
discrete GPU that's correct: weights land in host RAM, get copied across PCIe into
separate VRAM, host copy is freed. **Two different pools.**

On GB10 there is only one pool. `from_pretrained` builds the model in system RAM and
`.to("cuda")` allocates a *second full copy* in the very same 119.7GiB. Peak demand
is 2x the model, so LTX-2.3 (~72GB) needs ~144GB it will never have. What we
actually saw:

```
CUDA out of memory. Tried to allocate 32.00 MiB.
GPU 0 has a total capacity of 119.70 GiB of which 527.23 MiB is free.
this process has 52.68 GiB memory in use. 107.73 GiB allowed
```

Read that carefully: PyTorch itself held only 52.68GB, well under its 107.73GB
allowance, and yet the GPU had 527MiB free. The missing ~66GB was **the host-side
copy of the same model.** No amount of `empty_cache()` helps, because nothing is
stale; the load is legitimately holding both copies at once.

The fix is `device_map="balanced"`, which makes accelerate dispatch each module
straight to the device as it reads the checkpoint, so the full host-side copy never
exists. Only models over `DEVICE_MAP_MIN_GB` (40GB) take this path; below that the
transient 2x fits fine and the plain load is simpler. After that change LTX loaded
and ran at peak 73GB, and the weight load got *faster* (9s vs 63s) because the extra
copy was pure waste.

One non-obvious detail: **you must pass `max_memory` explicitly.** Accelerate sizes
its budget from `torch.cuda.mem_get_info()`, and on unified memory "free" reads as
near-zero (the OS page cache holds it; reclaimable, but not counted). We measured
"5.4GB free" on a box that then happily ran a 16GB model. Left to itself, accelerate
concludes the GPU is full and offloads everything back to CPU, recreating the exact
problem you were trying to solve. Mind the units too: `mem_get_info` returns decimal
GB and `max_memory` wants GiB, which differ by ~7%.

The same reason is why we do **not** warn when `est_vram_gb > free`: on this box that
warning would fire on literally every run.

Two more knobs are host-level, worth doing before a long session:

```bash
sudo swapoff -a                 # swap thrash on unified memory is a silent freeze
sudo nvidia-smi -lgc 300,2100   # the hard crashes on this box are POWER spikes, not OOM
```

And one anti-pattern: **do not `torch.compile` the transformer.** Triton doesn't emit
working SASS for `sm_121a` yet. The diffusers Wan docs recommend
`torch.compile(mode="max-autotune")`; ignore that here.

---

## Selective downloads (why every spec has `allow_patterns`)

Video repos ship several mutually redundant copies of the same model, so a naive
`snapshot_download` is catastrophic rather than merely wasteful:

- `tencent/HunyuanVideo-1.5` is **372GB** because it packs eleven separate 33GB
  transformers (480p/720p × t2v/i2v × distilled/not). You want one.
- `Lightricks/LTX-2.3` is a flat 157GB repo where `-dev`, `-distilled` and
  `-distilled-1.1` are each 46GB and mutually exclusive. (We use the
  `diffusers/`-layout mirror instead; the native repo won't load with
  `from_pretrained` at all.)
- Wan's `-Diffusers` repos are the ones to use. The non-`-Diffusers` variants ship
  the original `.pth` layout that `WanPipeline` can't read.

So the patterns in `models.py` are a correctness requirement, not an optimization,
and `download_gb` in the table above is the *measured* size of what we actually pull.

### It bites the image models too (this one shipped broken)

The first-frame models (`IMAGE_MODELS`) had **no patterns at all** until 2026-07-16,
and it cost us a real run: an `--image_model sdxl-turbo` job sat at 47GB and climbing
before stalling. `stabilityai/sdxl-turbo` is **55.5GB**, and ships *four* redundant
copies of one model:

| | size | usable by `from_pretrained`? |
|---|---|---|
| `sd_xl_turbo_1.0.safetensors` | 13.9GB | no (single-file fp32) |
| `sd_xl_turbo_1.0_fp16.safetensors` | 6.9GB | no (single-file fp16) |
| `unet/model.onnx_data` + friends | ~13.6GB | no (ONNX export) |
| `unet/diffusion_pytorch_model.safetensors` + friends | ~13.8GB | yes (fp32 diffusers) |
| `unet/diffusion_pytorch_model.fp16.safetensors` + friends | **6.9GB** | **yes, and it's what we load** |

We load fp16, so `_FP16_DIFFUSERS` allowlists the last set: **6.9GB instead of 55.5GB**.
`sd-turbo` had the same disease (13.0GB naive -> **2.6GB**), and both specs' old
`download_gb` values (14.4 and 5.2) were simply wrong. Verified after the fix: the pull
reports `Fetching 20 files` and stops at 6.9GB.

**The gotcha: `allow_patterns` and `variant` are a pair.** Keeping only
`*.fp16.safetensors` means `from_pretrained` must be told `variant="fp16"`, or it goes
looking for the fp32 filenames you deliberately didn't download and dies. If you add an
image model, set both or neither.

**How the image-gen demo does it, and why we differ.** `topics/image-generation` solves
the same problem with one *global denylist* in `fetch_weights` (drop `*.onnx`, `*.bin`,
`*.fp16.safetensors`, `sd_xl_*.safetensors`, `flux1-*.safetensors`, ...), keeping the
**fp32** components and casting to bf16 on load. That works, and it's less typing per
model. The trade is that a denylist has to enumerate every new repo's junk by name, so
a model whose single-file checkpoint isn't in the list slips through silently and you
eat the extra copy. An allowlist fails the other way: it fails *loudly* at load time,
which on a box where a 55GB mistake costs 11 minutes is the direction you want to fail.

`fetch_weights` is `cache="auto"`, keyed on `(model_key, task version)`, so a 95GB
download happens **once, ever**. It runs on a CPU pod so no GPU idles for the hour a
cold LTX-2 pull can take, and it wraps `snapshot_download` in a throughput watchdog
that kills and resumes the transfer when bytes flatline (resuming from the
`.incomplete` files, so a restart keeps every byte so far).

The watchdog is not theoretical: on the very first run here, the Wan 2.1 pull ran at
~100MB/s to 27.6GB, then **flatlined at +0 MB for five consecutive 30s windows**.
The watchdog killed it, resumed from the `.incomplete` files, and it finished at
30.5GB. Without it that run just hangs, because neither `HF_HUB_DOWNLOAD_TIMEOUT`
nor a socket timeout ever fires on a stalled-but-open stream, and a Flyte retry
would restart the whole download from zero.

---

## 1. Text to video: the model comparison

The original demo, and still the backbone: one prompt, N models, one grid report with
clips that play inline.

```bash
.venv/bin/flyte run compare_pipeline.py compare \
    --prompts '["a red panda barista pouring latte art, steam rising, cozy cafe, 50mm"]' \
    --models '["wan22-ti2v-5b","ltx2-distilled"]'
```

Each model is its own GPU task, so it loads once and renders every prompt before the
next model starts; on the single-GPU Spark they serialize at the scheduler, on a
multi-GPU box they fan out for free. The interesting axes are in [the models](#the-models)
table: size (1.3B -> 22B), licence (Apache-2.0 vs community vs non-commercial), and
architecture (dense DiT vs MoE vs linear attention vs diffusion forcing).

The headline pairing is `wan22-ti2v-5b` vs `ltx2-distilled`: Apache-2.0-and-small
against frontier-with-synced-audio. And the most surprising number is in
[Speed, honestly](#speed-honestly) — the 22B model beats the 1.3B by ~6x on wall clock,
purely because it's distilled to 8 steps. Size is not the thing that costs you time;
step count is.

---

## 2. Image to video

Rather than make you find a source image, `animate` generates the first frame and
then animates it:

```bash
flyte run compare_pipeline.py animate \
    --prompts '["a paper boat on a rain puddle, neon reflections"]'
```

```
prompt ──(sd-turbo, ~1s)──> first frame ──(wan22-ti2v-5b)──> clip
```

The report shows the frame next to the clip, so you can see what the video model
kept and where it drifted. Only I2V-capable models are allowed (`wan22-ti2v-5b`,
`ltx2-distilled`, `cogvideox-5b`); a text-only model fails fast with a message
naming the ones that work.

**Verified on the Spark** (2026-07-16, `sd-turbo` -> `wan22-ti2v-5b`, defaults,
2 prompts): the whole run took **10m52s** end to end and produced two 832x480,
49-frame @24fps clips. Both were coherent: the fox prompt gave a real trotting
animal with a panning camera and intact grass texture.

**The first frame is the bottleneck, and it is not close.** In that same run the
"paper boat on a rain puddle, neon reflections" prompt came back structurally sound
(the boats stay boats, the ripples read as ripples) but the water was harsh
high-frequency noise and the neon a flat pink streak. That is not the video model
failing; it is `sd-turbo` handing it a weak 512px frame, and `wan22-ti2v-5b`
faithfully animating whatever it is given, flaws included.

Re-running the **same prompt, same video model, same seed**, changing only the
first-frame model:

```bash
.venv/bin/flyte run compare_pipeline.py animate \
    --image_model sdxl-turbo \
    --prompts '["a paper boat drifting on a rain puddle at night, neon reflections rippling"]'
```

| first frame | result | HF noise* |
|---|---|---|
| `sd-turbo` (512px) | harsh black-and-white noise for water, flat green boats | **34.03** |
| `sdxl-turbo` | yellow origami boat, real depth of field, reflection, bokeh rain rings | **5.45** |

<sub>*mean abs difference from a Gaussian-blurred copy of each frame, averaged over
the clip. Lower is cleaner. Measured 2026-07-16, both runs at the shipped defaults.</sub>

A **6.2x** drop in noise, from one changed argument. So: if a clip looks bad, **look
at the frame before you blame the video model** (the report puts them side by side
for exactly this reason). The video model is rarely the thing that's broken.

### The I2V model is prompted too (`--motion_prompts`)

Easy to miss: image-to-video is image-conditioned **and** text-guided. The pipeline
gets the starting frame *and* a prompt, and that prompt is the only lever you have
over what actually moves.

By default `animate` reuses the frame's prompt for the animation, which is
convenient and slightly wasteful: the composition is already settled by the time the
video model runs, so spending its prompt on scenery ("neon reflections, 50mm") buys
nothing while the motion goes unspecified. The two prompts want different things.
A frame prompt describes a **static composition**; an I2V prompt describes **motion**.

`--motion_prompts` separates them (it must line up 1:1 with `--prompts`, or the run
fails fast):

```bash
.venv/bin/flyte run compare_pipeline.py animate --image_model sdxl-turbo \
    --prompts        '["a paper boat on a rain puddle at night, neon reflections"]' \
    --motion_prompts '["the paper boat leaps up out of the water, spray flying"]'
```

The clean way to see this is to hold the frame prompt and seed fixed and vary only
the motion prompt, so the starting image is identical and the only variable is text.
**Verified on the Spark** (2026-07-16, `sdxl-turbo` -> `wan22-ti2v-5b`, one shared
boat frame, seed 1234):

| motion prompt | what happened | motion energy* |
|---|---|---|
| "the paper boat **leaps up out of the water**, spray flying" | it launches: airborne by frame 32, spray streaking, leaving the top of the frame by 48, camera tracking up | **22.36** |
| "the paper boat **spins rapidly in place**, water swirling" | holds position, water churns around it | 8.76 |
| "the paper boat **drifts gently right** as rain rings spread" | nearly serene, barely moves | 9.12 |

<sub>*mean abs luma difference between consecutive frames. Higher = more is moving.</sub>

The motion prompt is a **real lever, not a suggestion**: "jump" carries ~2.5x the
motion energy of the other two from an identical starting frame. This is the cheapest
way to find out how much motion control a model actually gives you, and it's a good
segment: one image, three sentences, three different physics.

One measurement trap worth avoiding if you extend this: don't compare the *clips'*
frame 0 to check the inputs matched. I2V re-renders frame 0 through the VAE and the
prompt influences it, so it differs across runs even when the input image is
byte-identical. Compare the first-frame data URIs instead.

This is also exactly what `long_video.py` is built on: each **beat** there is a
motion prompt, with the image carrying the composition forward between chunks.

**On reusing the image-gen project's models:** you can't, quite. Flyte's cache is
keyed on `(project, task, version, inputs)`, and this demo runs in the
`video-generation` project, so the image-generation project's already-cached SDXL
does **not** carry over. That's why the first-frame models here are deliberately
cheap: `sd-turbo` is a 5GB pull that makes a frame in about a second, paid once.
(If you'd rather share the cache, point `.flyte/config.yaml` at the same project the
image demo uses.)

---

## 4. LoRA to animate

*(Built, **not yet run**. The one assumption is called out at the end.)*


Training a LoRA on a **video** model is an overnight job. Training one on an **image**
model takes minutes (or you can just download one at a few hundred MB), and on the
image-to-video path that turns out to be enough: the LoRA styles the **first frame**,
and the video model animates whatever it is handed. The LoRA rides on the frame, not
the motion, so it costs **nothing at video time**. That's the whole trick.

```bash
# no training, runs today: a paper boat rendered as layered papercut, then animated
.venv/bin/flyte run compare_pipeline.py animate \
    --image_model sdxl-turbo --lora papercut \
    --prompts        '["a paper boat on a rain puddle at night"]' \
    --motion_prompts '["the boat drifts slowly right as rain rings spread"]'
```

`--lora` takes three forms:

| form | example | notes |
|---|---|---|
| a built-in key | `--lora papercut` | see `models.LORAS`: `papercut`, `pixel-art`, `3d-render`. Ungated, 85-340MB |
| **your own** | `--lora s3://flyte-data/.../sdxl_lora_xxxx` | the Dir from the image-gen demo's `lora_finetune.py train_lora`. Read **cross-project** via `Dir.from_existing_remote`: both projects share the `s3://flyte-data` bucket, so nothing is re-trained or re-uploaded |
| any hub repo | `--lora nerijs/pixel-art-xl` | loaded as-is |

Two ways this silently does nothing, both handled:

- **Trigger words are load-bearing.** A style LoRA fires on a specific token
  (`papercut`, `pixel art`). Without it in the prompt the adapter loads, fuses, and
  changes almost nothing, which reads as "the LoRA is broken" when it's really "you
  didn't say the magic word". Registry LoRAs prepend their trigger automatically and
  log it; a raw repo id has no trigger of its own, so put it in the prompt yourself.
- **Base mismatch.** These are SDXL adapters; the default `sd-turbo` is **SD2.1**, so
  they cannot load onto it. `animate` refuses up front naming `sdxl-turbo`, rather
  than dying on a shape error deep inside diffusers after a 3GB download.

The adapter is **fused** (`fuse_lora`) rather than kept live, so generation costs
exactly what the base model costs and no call site threads a `scale` around.

⚠️ **Untested**: these are base-SDXL LoRAs fused onto the **turbo** checkpoint.
sdxl-turbo *is* SDXL (few-step distilled) so the shapes match and it loads, but expect
the style to land weaker than on stock SDXL. Needs a run to confirm.

---

## 5. Length and consistency: four approaches, measured

All four try to beat the same wall — **every model here has its clip length baked into
its latent shape**, so one call buys ~2-5 seconds. They fail in different ways, and the
differences are the interesting part. Detail sections follow; this is the map.

| approach | where | length | what it fixes | what it costs | verdict |
|---|---|---|---|---|---|
| **chain** | `long_video.py long_video` | **unbounded** (8s tested) | nothing by itself; it's the baseline | drifts; won't advance a trajectory | ✅ works if the beats behave |
| **chain + `--renorm`** | `long_video.py long_video --renorm 1.0` | unbounded | **statistical** drift (contrast/colour creep) | nothing | ✅ on stationary scenes (candle **-65%**), ❌ on a moving subject |
| **bookend** | `vace.py bookend` | **49 frames only** | **the trajectory** — endpoints pinned, so it can't run away | you give up unbounded length | ✅ the only one that made the subject actually recede |
| **anchored chain** | `vace.py anchored` | unbounded | **identity** — the subject stays the same subject | contrast slightly worse; VACE is 1.3B | ✅ the only fix for *semantic* drift |
| **refine after** | `vace.py refine` | n/a (post-pass) | intended: appearance, globally | seams pop 2.7-3.4x; the coat went black | ❌ wrong tool (see below) |
| **polish after** | `long_video.py polish` | n/a (post-pass) | true v2v with a `strength` knob | needs a **Wan 2.1** checkpoint (VAE mismatch) | ⚠️ built, unrun |

**How they actually relate**, which took all night to work out:

1. **The chain is fine.** Its famous collapse (a person turning around and walking into
   the lens) was **our beats**, not the method. Removing one geometry-inverting beat
   took drift from +8.9 to **+1.6**. Prompt design beat every algorithm we tried, for
   free.
2. **Chaining fails at three different things, and each needs a different tool.**
   Statistical creep -> `--renorm`. Identity -> an anchor (`reference_images`).
   Trajectory -> pinned endpoints (`bookend`). No single knob does all three, and
   using the wrong one looks like the method failing.
3. **A post-pass can't repair a collapse.** `refine` re-renders from the chain's
   *output*, so if the chain already dissolved, the control signal is already garbage.
   An anchor has to be present *while* the damage is done — which is exactly why
   `anchored` works and `refine` doesn't.
4. **Watch the right number.** Contrast drift is the right probe for statistical
   degradation and **blind** to semantic drift: the anchored run kept the red coat for
   8 seconds *while* scoring slightly worse on contrast. Numbers for exposure; eyes for
   identity.

---

## 5a. Long video by chaining (`long_video.py`)

Every model here has its clip length baked into the latent shape, so one call buys
~2-5 seconds and no more. `long_video.py` gets around that with the obvious trick:
generate a chunk, keep its **last frame**, feed it back as the next chunk's first
frame. Each beat gets its own prompt, so the clip can tell a small story rather than
loop one motion, which is something no single-shot call to these models can do.

```
beat 1 ──> [chunk 1] ──last frame──> [chunk 2] ──last frame──> [chunk 3] ──> concat
```

```bash
.venv/bin/flyte run long_video.py long_video --beats '[
  "a red fox trotting through tall grass, morning light",
  "the fox slows and stops, ears twitching",
  "the fox turns its head toward the camera",
  "the fox looks up at the sky"
]'
```

It needs **no new model and no new download**: it reuses `wan22-ti2v-5b`'s existing
I2V path. The report shows every chunk plus `chained.mp4`, the whole thing
concatenated.

**Why this is a Flyte demo and not a for-loop.** The alternative is one fragile
20-minute task. Chunking makes each unit independently retryable, so chunk 4 OOMing
doesn't throw away chunks 1-3, and on this box OOM is often transient (something
else grabbed the unified pool). Note the deliberate exception: the chunks share
**one resident pipeline inside one task** rather than a task per chunk. Loading
`wan22-ti2v-5b` costs ~60s for 26GB, so paying that per chunk would dominate the
runtime, and on a single-GPU box each chunk pod would serialize behind the previous
one's teardown anyway.

### Verified: it works, and the drift is one number, not two

First real run (2026-07-16, `sdxl-turbo` first frame -> `wan22-ti2v-5b`, 4 beats,
defaults): **193 frames @24fps = 8.0s**, from a model whose per-call budget is 49
frames (~2s). Each chunk took ~135s, so ~9 minutes for the clip.

| chunk | beat | last-frame luma | contrast (std) |
|---|---|---|---|
| 1 | boat on a dark rain puddle | 56.4 | 48.2 |
| 2 | drifts right, rain rings spread | 53.9 | 52.6 |
| 3 | rocks as the rain grows heavier | 57.4 | 60.4 |
| 4 | rain eases, water settles | 54.3 | 63.6 |
| | **drift over the chain** | **-2.1** | **+15.4** |

Findings from this run:

- **Contrast accumulates**: 48.2 -> 52.6 -> 60.4 -> 63.6, **+32%** across four hops,
  never once going down. That's VAE round-trip error compounding, and you can see it:
  the bokeh highlights in the late chunks sparkle harder than in chunk 1.
- **Brightness does NOT drift** (-2.1 net, non-monotonic). We predicted saturation
  creep too; it isn't there.

(The second run below tempers both of these. Contrast creep is real but **not**
reliably monotonic, and the boat's clean curve was n=1.)

### Then we tried a human, and it fell apart (the useful run)

Same 4-beat structure, subject = a person in a red coat on a neon street: "turns to
look over one shoulder", then "walks **away** from the camera, receding, growing
smaller". Also 193 frames / 8.0s.

It did the opposite, and then ran away with it. Frames 0-48: walks away, back to
camera, as asked. Frame 72: turned fully around, facing camera. Frames 96-144: walking
*toward* camera, larger each chunk. Frame 192: **the subject is gone**, dissolved into
out-of-focus bokeh.

> **Follow-up (same day): it was the beat, and the fix is free.** Re-running with the
> turn removed — "walking away ... **seen from behind**" in every beat, nothing else
> changed — the chain holds all 8 seconds with **no collapse**, and contrast drift
> falls from **+8.9 to +1.6**, essentially flat. Details in
> [the hierarchy of fixes](#the-hierarchy-of-fixes-cheapest-first) below.

**The failure mode is a feedback loop, and it is inherent to last-frame chaining.**
Beat 2 said "look over one shoulder"; the model turned the person all the way around.
From there the frame and the prompt disagreed about which way the subject faced, and
**the frame won** (I2V follows image geometry over text). So "walking" became walking
*toward* the camera. Each chunk then ended with the person slightly closer, and the
next chunk faithfully continued walking from that closer frame. Nothing can correct
it: each chunk sees exactly **one RGB frame**, with no memory of trajectory or intent.
Four hops took it from a clean shot to destroyed.

Why the boat survived and the person didn't: the boat's motion is **bounded** (drifting
in a puddle has nowhere to go), so error averages out. Walking is **unbounded**, so
error compounds. That's the rule of thumb: chain bounded motion, don't chain a
trajectory.

The numbers, and how they revise the boat's story:

| | boat | human |
|---|---|---|
| contrast drift | +15.4, monotonic | **+8.9, non-monotonic** (40.8 -> 46.4 -> 40.6 -> 49.7) |
| brightness drift | -2.1 (noise) | **-11.4** (the dark coat grows to fill the frame: *subject*-driven, not scene-driven) |
| seams | 0.63x / 0.50x / 0.74x (all smooth) | 0.75x / 0.52x / **1.03x** (the last one stops smoothing, right where the subject breaks) |

So: **contrast creep is real but not reliably monotonic**, and a seam ratio drifting to
~1.0x is a decent automatic tell that the chain is coming apart.

### What we hit has a name: exposure bias (`--renorm`)

Chaining fails the way autoregressive generation always fails. The model is trained on
clean history and fed its own imperfect output at inference; that distribution shift is
**exposure bias**, and per-hop error compounds into what the literature calls semantic
collapse. Which is a fair description of a person dissolving into bokeh by frame 192.

The fixes in the literature are all forms of **re-anchoring instead of trusting the
drifted frame**:

| approach | idea | do we have it? |
|---|---|---|
| [StreamingT2V](https://arxiv.org/html/2403.14773) | an Appearance Preservation Module re-injects features from an **anchor frame** so identity doesn't drift | the same idea is VACE's `reference_images` (`vace.py`) |
| [FramePack](https://arxiv.org/html/2504.12626v1) | **anti-drifting sampling**: generate the endpoints first, interpolate the middle with bidirectional context, so error can't compound one-way | `vace.py flf2v` is exactly this primitive |
| SkyReels `addnoise_condition=20` | add noise to the conditioning so the model can't over-trust a drifted history | already set, from the model card |
| `--renorm` (here) | match the hand-off frame's per-channel mean/std back to chunk 1's | implemented below |

`--renorm 0..1` is the cheapest member of that family, and it's off by default so the
drift stays visible and the A/B against the +32% baseline above stays honest:

```bash
.venv/bin/flyte run long_video.py long_video --renorm 1.0 --beats '[...]'
```

**Verified on the real drifted frames** from the boat run (chunk 1 frame 0 is the
anchor: the only frame in the chain that never went through a VAE round trip):

| frame | luma |
|---|---|
| anchor | 58.2±47.0 |
| drifted (after 4 hops) | 54.0±**63.3** |
| renormalized | 57.6±**47.1** |

#### Measured: four scenes, both arms, same seeds (2026-07-16)

Every pair below is one variable: identical beats, identical seed, `--renorm` on vs off.

| scene | motion | contrast drift OFF | contrast drift ON | verdict |
|---|---|---|---|---|
| **candle** on a table | tiny, fixed camera | +9.5 | **+3.3** | ✅ best case: **-65%** |
| **paper boat** on a puddle | bounded drift | +15.4 | **+8.3** | ✅ **-46%** |
| **beach, sunset -> night** | the scene is *meant* to change | +2.0 | -0.5 | ➖ barely matters |
| **person walking** away | unbounded trajectory | +8.9 | **-16.7** | ❌ different failure, not a fix |

**The rule: renorm helps in proportion to how stationary the scene is.** A candle on a
table is its ideal case; a person walking a trajectory is outside its remit entirely.

#### Three things this got wrong, which is why we ran it

1. **"It only corrects moments, not meaning" was too strong.** It does only correct
   moments — but the baseline boat's *two-boat hallucination* is largely **absent** in
   the renorm run. Cleaner statistics at the hand-off = a less degraded input = the
   model invents less. The statistical fix bought a semantic benefit indirectly.
2. **We predicted renorm would FIGHT the sunset** (forcing frame 192's brightness back
   to a sunlit anchor when you asked for night). It barely did: -8.5 -> -7.4. Because
   renorm only touches the *hand-off frame*, and the next chunk's prompt ("night
   settles") simply re-darkens it. **The beats override the correction every chunk**,
   so renorm can't fight your intent. Reassuring, and not what we expected.
3. **The drift is not IN the statistics.** This is the real finding. The hand-off
   correction is flawless — every chunk starts at the anchor's stats within 0.1:

   | hand-off | before | after |
   |---|---|---|
   | 1 -> 2 | 56.4±48.2 | **58.0±46.9** |
   | 2 -> 3 | 55.3±51.8 | **58.1±46.9** |
   | 3 -> 4 | 59.5±52.7 | **58.0±46.8** |

   And yet each chunk drifts *further* from that identical clean start: **+4.9, +5.8,
   +9.5**. Per-chunk drift **accelerates** even from statistically identical
   conditions. So the thing accumulating is the **content** — structure the histogram
   can't see. Statistics were the symptom, not the disease.

**The split, restated honestly:**

- **statistical drift** (contrast/colour creep) -> `--renorm`, cheap, real, and worth
  it on a stationary scene
- **semantic drift** (identity, geometry, "walks away" becoming "walks at you") ->
  untouched by renorm at any strength. Needs an anchor that understands *content*:
  `reference_images`, or FramePack-style endpoint pinning (`vace.py bookend`)

### The hierarchy of fixes (cheapest first)

Measured on the human scene, 4 beats, 193 frames, same seed throughout:

| fix | contrast drift | collapse by frame 192? | cost |
|---|---|---|---|
| baseline ("turns to look over one shoulder") | +8.9 | **yes**, subject dissolves | — |
| `--renorm 1.0` | -16.7 | **yes** (flat/mushy instead of harsh) | free |
| **remove the turn beat** | **+1.6** | **no** — holds all 8s | **free** |
| remove the turn beat + `--renorm 1.0` | -7.7 | no | free |

**Writing beats that don't contradict themselves beat every algorithmic fix we tried,
and cost nothing.** That is the headline. Renorm bought -46% on a stationary boat;
deleting one word-level geometry inversion took the human from total collapse to
essentially flat drift.

The rule: **never chain a beat that inverts your subject's geometry.** "Turns to look
over one shoulder" makes the *image* say "facing camera" while the *prompt* still says
"walks away", and the image always wins (I2V follows geometry over text). From that
frame on, every chunk faithfully walks the subject **toward** the lens. Keep the
geometry consistent ("seen from behind" in every beat) and the failure disappears.

Two things the prompt fix does **not** buy, both visible in the clip:

- **The subject never recedes.** Beats 3-4 ask for "further away, smaller in frame";
  the person stays the same size for 8 seconds. The chain preserves what is *in* frame
  but will not advance a *trajectory* — the same one-frame-of-memory limit, in a
  politer form. Pinning the endpoint (`vace.py bookend`) is the fix for that.
- **Chunk 4 still softens.** Not the old runaway (contrast wobbles 48.1 -> 44.0 ->
  53.5 -> 49.6 rather than climbing), but it's the weakest chunk.

Sources: [FramePack](https://arxiv.org/html/2504.12626v1) ·
[Packing/forcing memory for long-form consistency](https://arxiv.org/html/2510.01784v1)

**No visible seam.** The frame-to-frame delta *at* each boundary is **lower** than the
clip's average (0.63x, 0.50x, 0.74x at frames 49/97/145), i.e. the hand-off is smoother
than ordinary motion rather than a hitch, because the next chunk starts from the exact
frame we just showed. Subject identity survives all 8 seconds: the boat is still the
same boat, the scene is still the same scene.

The honest limit is that contrast creep sets the ceiling on how long you can chain, and
at +32% over 4 hops you would not want to run 20. That's the real contrast with
`skyreels-v2-df-1.3b`, which does diffusion forcing and keeps a real **latent history**
across chunks instead of a single RGB frame, so it should not accumulate this way. This
is the cheap trick that needs no new model; SkyReels is the principled version. Showing
both together, with this table, is the interesting segment.

Two details worth knowing: each later chunk's frame 0 is dropped (it's a re-render of
the frame we fed in, so keeping it duplicates the seam and reads as a hitch), and the
seed is varied per chunk (one fixed seed makes every chunk resolve toward the same
motion from a near-identical frame, and the chain comes out a stutter loop).

---

## 6. SkyReels 14B: long video done properly

*(Weights fetched and verified at 80.4GB. First run in flight.)*

This is the **control** for experiment #5, and the reason it's interesting is exactly
the thing chaining can't do: **hold composure across a long clip**.

Our chain hands the next chunk one **decoded RGB frame**. That single frame is its
entire memory, which is why the walking person could turn around and no later chunk
could object, and why VAE round-trip error compounds into +32% contrast. Diffusion
forcing hands the next chunk **latents**. It's not a claim from a blog post; it's a
line in the diffusers source
(`pipeline_skyreels_v2_diffusion_forcing.py`):

```python
prefix_video_latents = video_latents[:, :, -overlap_history_latent_frames:]
```

It never leaves latent space between chunks, so there is no round trip to accumulate,
and it carries *many* frames of history rather than one — enough context to hold a
trajectory instead of only a snapshot.

**Run it against the case that BROKE**, not the one that worked. The boat chain
survived (bounded motion); the walking person collapsed. So the overnight run uses the
human, at the same 193 frames, so the comparison is against a measured failure:

```bash
# ~4 hours. The direct answer to "does latent history hold a trajectory?"
.venv/bin/flyte run compare_pipeline.py generate_one \
    --model_key skyreels-v2-df-14b --num_frames 193 \
    --prompts '["full body shot of a person in a long red coat walking away from the camera down a wet neon city street at night, receding into the distance, cinematic, 35mm"]'
```

Note diffusion forcing takes **one prompt for the whole clip**, not per-chunk beats:
the history is latents, not a re-prompted frame. That's the difference in one line.
The chain needed a beat per chunk *because* a single RGB frame couldn't carry intent.

**Why the 14B and not the 1.3B we already had**: the 1.3B is the model that made long
video look bad here, and size was the likeliest reason. The 14B is the same
architecture and the same pipeline class, so it's a drop-in registry row: only the
download differs (80.4GB).

**The knobs are not optional.** Without `extra_call_kwargs` this is just another
49-frame T2V model, and the long path is unreachable: the pipeline *raises* if
`overlap_history` is missing once `num_frames > base_num_frames`. We set `ar_step=5`,
`causal_block_size=5`, `base_num_frames=97`, `overlap_history=17`, and
`addnoise_condition=20` (the card's consistency knob, and itself an anti-drift trick:
it stops the model over-trusting a drifted history).

What to compare against #5, on one chart: contrast drift per hop. Ours crept **+32%**
over 4 hops. If SkyReels stays flat, latent history is the whole story.

### ⚠️ It works, and it costs ~4 hours a clip (measured)

First run, 2026-07-16, 960x544 x 193 frames, 30 steps, on the Spark. It loads, the
long-video path activates, and it denoises at a steady rate. That rate is the problem:

```
1/50 [04:42<3:51:00, 282.87s/it]
```

- **283 s/step**, 96% GPU the whole time. Not hung, just enormous.
- **50 iterations, not 30.** `ar_step=5` (asynchronous mode) expands the schedule, so
  the real cost is ~1.7x what `--steps` suggests. Budget on 50.
- **~3h51m for one 8s clip.** For scale, `wan22-ti2v-5b` does 3.3s/step: this is a 14B
  (2.8x params) at 1.3x the pixels over 3.9x the frames, on a box whose ceiling is
  memory bandwidth (~273GB/s). ~85x slower per step is the honest number.

So this is an **overnight artifact, not a live demo**. Options if you need it
interactive: drop to `--num_frames 97` (one `base_num_frames` chunk, but then the
long-video hand-off never triggers and you're not testing the thing you came for),
lower the resolution, or set `ar_step=0` for synchronous mode and fewer iterations.

Also measured: **~85GB resident**, not the `est_vram_gb=42` computed from parameter
counts. It fits the 119.7GiB pool, but only just, and only with rustfs freshly
restarted. The sampler defaults still come from the 1.3B's card, not measurement.

---

## 7. Video to video (VACE)

*(✅ **Verified** 2026-07-16: 49 frames restyled in **323s** on the 1.3B. Fast enough to
be a live demo, unlike #6.)*

Every other model here goes text -> video or image -> video. **VACE is the only one
that takes video IN**, which makes it the one new *axis* rather than another quality
datapoint. It's also the smallest model in the registry (19GB), which is a pleasing
inversion: the new capability is the cheap one.

```bash
# generate a source clip, derive an edge-map control video, restyle it
.venv/bin/flyte run vace.py restyle \
    --style_prompt "the same scene as a pen and ink sketch, cross-hatched, monochrome"

# first-last-frame: give it two images, it invents the motion between
.venv/bin/flyte run vace.py flf2v --prompt "the boat drifts across the puddle"
```

### The one that cost a run: **name the subject in the style prompt**

First attempt used `--style_prompt "the same scene as a pen and ink sketch, heavy
cross-hatching, monochrome, white paper"`. It produced a beautiful pen-and-ink
**landscape**. The boat was gone.

The control map was fine (hull, mast, ripples all clearly there). The prompt was the
bug: **"the same scene" means nothing to the model** — it has no idea what the previous
scene was. Given sparse edges plus a prompt describing only a *medium*, it invented a
plausible subject for that medium and drew a field.

VACE's control video is **structural guidance, not a subject specification.** Say what
the thing is:

```bash
--style_prompt "a paper boat floating on a puddle, pen and ink sketch, heavy cross-hatching, monochrome line drawing on white paper"
```

| | motion* | result |
|---|---|---|
| source clip | 5.48 | the original |
| restyle, subject **not** named | 13.62 | a landscape. Flickering: inventing content per frame |
| restyle, subject named | **5.56** | the boat, in ink, tracking the control geometry |

<sub>*mean abs luma delta between consecutive frames.</sub>

A restyle that tracks its control should land near the **source's** motion number, and
5.56 vs 5.48 is what "it followed the video" looks like numerically. 13.62 is what
"it ignored the video" looks like.

Three more things, learned before writing a line of it, all in `vace.py`'s docstring:

1. **The mask semantics are backwards from intuition**: black (0) = *keep* this frame,
   white (255) = *generate* it.
2. **For control tasks the `video` is not footage** — it's a control signal (depth,
   pose, scribble). Passing raw RGB with an all-white mask silently degenerates into
   plain text-to-video, which is the easiest way to get VACE wrong.
3. **Pick your source clip for its edges**, and **look at the control map before you
   blame the model**. Two bugs came out of doing exactly that, neither visible in the
   output clip:

   - *Threshold*: a fixed cutoff gave a **79% white** map on a fox-in-grass frame (the
     fox had no outline; the control was noise). Now: blur, then keep the top
     `keep_pct`% by percentile, so density is bounded (~8%) whatever the scene.
   - *Colourblindness*: the Sobel ran on **luma**, so a red coat on a teal street — a
     huge *colour* contrast, a weak *luma* one — made the person vanish while every
     neon sign lit up. Now the gradient is per-RGB-channel, max across channels.

   Even fixed, the ranking is: **boat on a dark puddle (great) > person on a neon
   street (workable) > fox in grass (hopeless)**. In tall grass the texture **is** the
   gradient. That's a property of edge control, not of VACE, and a depth map wouldn't
   have it (at the cost of another model download).

`flf2v` is here as the low-risk path (it's the diffusers example verbatim), but it's
also the interesting one: **it's FramePack's anti-drifting primitive** — fix the
endpoints, interpolate the middle. And VACE's `reference_images` is the identity anchor
that [renormalization can't provide](#what-we-hit-has-a-name-exposure-bias---renorm).
So #7 holds the tools for the part of #5 that #6 may not fix.

### `bookend`: the fix for the trajectory the chain won't advance

✅ **Verified 2026-07-16.** The chain's residual failure (after the beat fix) is that
the subject **never recedes**: you ask for "further away, smaller in frame" and get a
treadmill for 8 seconds. It can't do otherwise — each chunk sees one frame and has no
idea where the shot is supposed to *end*.

`bookend` inverts that. Generate BOTH endpoints first, then interpolate:

```bash
.venv/bin/flyte run vace.py bookend --image_model sdxl-turbo \
  --first_prompt "a person in a long red coat close to the camera on a wet neon street at night, seen from behind" \
  --last_prompt  "a wet neon street at night, a small distant figure in a red coat far down the street, seen from behind" \
  --prompt       "the person walks away from the camera down the street, receding into the distance"
```

The trajectory is **bounded by construction**: the model cannot walk the subject into
the lens, because it has been told up front that the last frame is a small figure in
the distance. Result: a coherent figure, seen from behind the whole way, that **actually
gets smaller** — which no chained run managed at any renorm setting.

The trade is honest: **49 frames (~2s), not an unbounded chain.** Bounded is the point.
Chaining buys length and loses the trajectory; bookending buys the trajectory and loses
length. Pick per shot.

### `refine`: chain first, then re-render against one anchor

The natural next move, and it targets the thing renorm couldn't. Chaining is good at
motion and bad at holding appearance; a **global second pass** re-renders appearance
with the whole clip in view:

```bash
.venv/bin/flyte run vace.py refine \
    --source_clip_uri s3://flyte-data/.../chain_wan22-ti2v-5b_xxxx \
    --style_prompt "a person in a long red coat walking away down a wet neon street at night, seen from behind, cinematic"
```

**Why this can beat `--renorm`:** the windows **do not chain**. Each 49-frame window's
input is its own slice of the ORIGINAL clip plus the same anchor (source frame 0), so
there is no hand-off for error to accumulate through. Renorm corrected every hop and
drift still grew (+4.9, +5.8, +9.5 from statistically identical starts) *because the
hops were still serial*. Remove the serial dependency and accumulation has nowhere to
live. `reference_images` carries appearance; the edge maps carry structure — which is
the division of labour VACE was designed for.

**The trade, which is the mirror image of the chain's:** independent windows can't
drift, but they aren't tied to each other either, so expect **style pops at the
49-frame boundaries** where the chain had unusually smooth seams (0.5-0.75x of average
frame delta). A slow monotonic degradation, traded for possible discontinuities.

⚠️ And the obvious risk: VACE is **1.3B**; the chain came from a **5B**. A refinement
pass can plausibly make things worse. If it does, `Wan-AI/Wan2.1-VACE-14B-diffusers` is
the same pipeline class at 75.1GB — a one-line spec change, not an integration.

#### ❌ Result: it doesn't work, and the reason is structural

Ran 2026-07-16 on the no-turn chain: 193 frames, 4 windows, 1384s. **It made a good
clip worse.**

| | source (no-turn chain) | refined |
|---|---|---|
| contrast drift | **+1.5** | -12.9 |
| boundary 49 | 1.48x | **3.41x** |
| boundary 98 | 0.42x | **2.71x** |
| boundary 147 | 0.32x | **2.79x** |
| the red coat | red | **near-black** |

Three separate failures, and the middle one is the interesting one:

1. **The seams pop, exactly as predicted.** 2.7-3.4x the average frame delta at each
   window boundary, where the chain's seams were *below* average. No drift, but visible
   jumps. That trade is real and it is not subtle.
2. **The anchor did not hold appearance — the coat went black.** Preserving appearance
   was the entire point. The cause is a design error: **edge maps discard all
   appearance by construction**, so we handed VACE structure-only control and expected
   one `reference_images` frame to rebuild the scene's whole look. It cannot. Identity
   anchoring is not scene reconstruction.
3. **Wrong patient.** We refined the clip that had *nothing wrong with it* (+1.6 drift
   after the beat fix), so the pass could only cost. Refine a *drifted* clip if you
   retry this.

**The generalisable lesson: VACE control mode is a RESTYLE tool, not a REFINE tool.**
It regenerates from structure. The mask is binary (keep / generate), so there is no
low-strength "touch this up" knob the way an img2img pass at strength 0.3 would give
you. A refinement stage needs a mechanism VACE does not have.

The idea is still sound — a global pass *is* the right shape for fixing drift that
per-hop correction can't. It just needs a tool that starts from the pixels rather than
from their edges. See [`anchored`](#anchored-the-experiment-that-tests-the-actual-claim)
for the version that keeps the real frame.

### `anchored`: the experiment that tests the actual claim

Everything measured says the chain's real failure is **semantic, not statistical**:

- `--renorm` corrects the hand-off statistics *perfectly* (58.0±46.9, every hop) and
  drift still **accelerates** (+4.9, +5.8, +9.5).
- A person who turns around walks into the lens and dissolves, at any renorm strength.
- Fixing the **beats** helped 5x more than any algorithm (+8.9 -> +1.6), for free.

So: does an anchor that understands *content* fix what histograms couldn't? That's
StreamingT2V's Appearance Preservation Module in one sentence, and `reference_images`
is the closest thing we have to it.

```bash
# the A/B: identical everything, one variable
.venv/bin/flyte run vace.py anchored --image_model sdxl-turbo --use_anchor    --beats '[...]'
.venv/bin/flyte run vace.py anchored --image_model sdxl-turbo --no-use_anchor --beats '[...]'
```

#### The two arms, precisely

Both arms run the **identical** code path: VACE in image-to-video shape, where each
chunk is handed `video = [prev_last_frame, grey, grey, ...]` with mask
`[BLACK, white, white, ...]` — remember **black = keep, white = generate** — so frame 0
is the previous chunk's last frame and the other 48 are invented. Same model, same mask
shape, same beats, same seeds. **One variable:**

| | `--use_anchor` (**ON**) | `--no-use_anchor` (**OFF**, the control) |
|---|---|---|
| what each chunk gets | prev chunk's last frame **+ `reference_images=[anchor]`** | prev chunk's last frame **only** |
| memory of the subject | the **original frame 0**, re-supplied every chunk, forever | whatever survived N hops of re-encoding |
| the anchor | frame 0 of chunk 1 — the least-drifted frame in the clip | none |
| this is | StreamingT2V's Appearance Preservation Module, roughly | plain last-frame chaining, VACE edition |

**OFF is not a straw man** — it's exactly what `long_video.py` does, reimplemented on
VACE so that the *only* difference from the ON arm is the anchor. Without it, any
improvement could just be "VACE behaves differently from wan22" rather than "the anchor
works."

So ON keeps a permanent, un-drifted reference to who the subject *is*, while OFF only
ever knows the subject through a frame that has been through the VAE N times. If
semantic drift is really the disease, ON should hold identity where OFF morphs it —
and that is precisely what happened.

Two design points that matter, both learned from `refine` failing:

- **The real RGB frame carries appearance**, not an edge map. `refine` handed VACE
  edges and asked one reference image to rebuild a whole scene; the coat went black.
  Here appearance flows through the actual frame and the anchor only has to hold
  *identity*. That's a job `reference_images` was built for.
- **`refine` cannot do this job at all.** It re-renders from the chain's *output*, so
  if the chain already collapsed into bokeh, refine gets bokeh as its control signal.
  An anchor has to be present while the damage is being done, not afterwards.

Run on the **turn beats** deliberately — the known-broken case. The no-turn beats
already work (+1.6), so an anchor would have nothing to prove there. `--no-use_anchor`
is the control arm: same model, same mask shape, same seeds, so any difference is the
anchor and not the model swap.

#### ✅ Result: the anchor holds identity — and contrast was the wrong yardstick

Run 2026-07-16, turn beats, 193 frames, both arms:

| | anchor OFF | anchor ON |
|---|---|---|
| contrast drift | **+15.2** | +18.3 |
| the red coat at frame 128 | **morphed** into a different jacket + jeans | **still the red coat** |
| the coat at frame 192 | a dark coat entirely | dark, but the same garment |
| the turn (beat 2) | not really executed | **executed at ~f64, recovered by f128** |
| collapse into bokeh | no | no |

**The anchor did exactly its job and the metric couldn't see it.** `reference_images`
kept the subject the *same subject* for 8 seconds through the beat that destroyed the
wan22 chain — it turns, and then it *recovers* and keeps walking away. Meanwhile
contrast drift got slightly **worse**.

Those aren't in tension: contrast measures **exposure**, not **identity**. We'd been
leaning on it as a proxy for "is this degrading", and here it is blind to the only
thing that matters. That's a lesson about the measurement, not the method:

- **statistical drift** -> contrast is the right probe, `--renorm` is the lever
- **semantic drift** -> contrast is *useless*; you have to look, and the lever is an
  anchor (`reference_images`)

**This is the first mechanism here that addresses semantic drift at all.** Renorm
couldn't (by construction), better beats avoided the problem rather than solving it,
and `refine` operated too late to help.

⚠️ One confound worth stating: both arms are **VACE-1.3B**, while the chain that
collapsed was **wan22-5B**. The ON/OFF comparison is clean (same model, same seeds), but
"VACE chains more stably than wan22" is *not* isolated by this experiment.

---

## 8. Krea realtime video (researched, not built)

The obvious next swing if #6 disappoints: 14B, **Apache-2.0**, ungated, causal with a
KV cache, and it advertises video-to-video. Not built because it would be the fourth
unrun thing; the research is done and parked in [Still TODO](#still-todo).

The one number worth carrying here, because it's this repo's recurring lesson: Krea
ships **only** the transformer and pulls its text encoder / VAE / tokenizer from
`Wan-AI/Wan2.1-T2V-14B-Diffusers`. Naive = **137.6GB**. With an allowlist = **51.8GB**.
The difference is one duplicated 28.6GB single-file checkpoint plus a 57.2GB Wan
transformer that Krea replaces and we would never load.

---

## Speed, honestly

All measured on the Spark, at the shipped defaults:

| | `wan21-t2v-1.3b` | `ltx2-distilled` |
|---|---|---|
| download (once) | 30.5GB | 101.7GB |
| clip | 832x480, 49f @16fps | 768x512, 57f @24fps |
| steps | 30 | **8** (distilled) |
| per step | 5.9s | **2.3s** |
| **generate** | **210s** | **33s** |
| peak GPU | 16GB | 73GB |
| audio | no | **yes** (AAC stereo, generated) |

**The 22B model is 6x faster than the 1.3B one.** That is the single most
counterintuitive result here and it is worth pausing on during the stream: 8
distilled steps beat 30 regular steps by far more than the 17x parameter
difference costs. Step count dominates model size. It also produces visibly better
video *and* a synced audio track.

The caveat is loading, not generating. LTX's 8 denoise steps take 33 seconds; getting
101.7GB of fp32 weights off disk and onto the device takes several minutes. Once
it's warm, it's the fastest thing here.

For context, the Spark is bandwidth-bound (~273GB/s against ~672GB/s on a mid-range
discrete card), and community reports have Wan 2.2 **I2V** at 15 to 30 minutes for a
5s 720p clip. Few-step distillation is the whole game on this box.

So the defaults in `models.py` are **demo defaults, not the model card's**: shorter
clips, fewer steps, 480p, tuned for a ~2 to 5 minute cell. Each spec's `native` field
records what the card actually recommends. To get there:

```bash
flyte run compare_pipeline.py compare --prompts '["..."]' \
    --models '["wan22-ti2v-5b"]' --steps 50 --num_frames 121 --width 1280 --height 704
```

Budget half an hour for that one.

---

## Try it without a cluster

```bash
python run_local.py --model wan21-t2v-1.3b --prompt "a corgi astronaut, studio light"
```

Writes an `.mp4` and a standalone `.html` using the same renderer the Flyte report
uses, so if playback works there it works in the console. This is the fastest way to
find out whether a model loads on this box before committing to a cluster run.

---

## The overnight run

One command: 8 models x the 8-prompt capability suite = **64 clips**.

```bash
.venv/bin/flyte run compare_pipeline.py compare --suite overnight \
    --models '["wan21-t2v-1.3b","wan22-ti2v-5b","ltx2-distilled","hunyuan-1.5-t2v","kandinsky5-lite","skyreels-v2-df-1.3b","motif-video-2b","sana-video-2b"]'
```

It's a multi-hour job: the cells serialize on the single GPU, and the five new models
are ~137GB of first-time download (cached forever after). A model that fails renders
as an error cell and the rest carry on, so one bad model can't sink the night.

### The prompt suite (`prompts.py`)

The image demo has "prompts to try, and what each stresses". Video needs its own axes,
because a model can render eight beautiful frames and still fail: **what's being
tested is what happens between them.**

| # | axis | what to look for |
|---|---|---|
| 1 | fluid motion (the control) | Every model manages this. Water has no fixed shape to be inconsistent about. If a model fails here it's broken. |
| 2 | rhythmic motion + **audio sync** | The LTX-2 prompt. Does the clang land ON the hammer strike? Sparks appearing *before* impact = no causal model, just "blacksmith" texture. |
| 3 | **identity consistency** | The hardest common failure. Same face in frame 1 and frame 6? Faces morphing mid-clip is the classic open-model tell. |
| 4 | camera motion + parallax | Does the *camera* move, or does the scene just shimmer? Real parallax = near trunks sweep past faster than far ones. |
| 5 | physics + causality | Expect failure, interestingly. Does the glass tip or vibrate? Does wine leave the glass before it falls? |
| 6 | motion magnitude | Wan drifts toward static clips (its long default negative prompt exists to fight exactly this). Does the car actually cross the frame? |
| 7 | **text stability** | Brutal, and the clearest separator. Legible text is a win; text that stays the *same* text across frames is a bigger one. |
| 8 | object permanence + a scripted event | Three boats in frame 1 *and* frame 6? And does the specific event (the middle one sinks) happen, or just generic boats? |

The **frame strip** in each report cell is the surface that makes most of these
judgeable: identity drift and object-count errors are obvious across six frames side
by side, and genuinely easy to miss while a 3-second clip loops past you.

`python prompts.py` prints the full suite with the failure mode for each.

### Still TODO

- **Video-to-video via Wan VACE.** The cheapest real capability left, and the only
  new *axis* on this list. `Wan-AI/Wan2.1-VACE-1.3B-diffusers` is **19GB** (the
  smallest model we'd carry, vs 75GB for the 14B) and `WanVACEPipeline` has been in
  diffusers since 0.34, so 0.39 has it. It does restyle, control (depth/pose) and
  reference-to-video. Not a drop-in registry entry though: its `__call__` takes
  `video` / `mask` / `reference_images`, so it needs its own task rather than a
  `VideoModelSpec` row. Diffusers 0.39 also exports `WanVideoToVideoPipeline`,
  `SkyReelsV2DiffusionForcingVideoToVideoPipeline` and `LTX2ConditionPipeline` if we
  want a v2v comparison grid.
- **Krea realtime video** (`krea/krea-realtime-video`): 14B, **Apache-2.0**, ungated,
  causal/autoregressive with a KV cache (`kv_cache_num_frames: 3`,
  `num_frames_per_block: 3`), and it advertises **video-to-video**. The obvious swing
  if `skyreels-v2-df-14b` disappoints. Researched 2026-07-16; three things to know
  before starting, none of them blocking:

  1. **It's Modular Diffusers, but that's fine.** No `model_index.json`; it ships
     `modular_model_index.json` naming `_class_name: WanModularPipeline` and
     `_blocks_class_name: WanRTBlocks`. `WanModularPipeline` **is exported by
     diffusers 0.39**, and `WanRTBlocks` lives in the repo's own `modular_blocks.py`,
     so this is `trust_remote_code`, not a reimplementation. (An earlier version of
     this note called it un-loadable. That was wrong.)
  2. **It needs TWO repos, and an allowlist is mandatory.** Krea ships *only* the
     transformer; the scheduler, tokenizer, text encoder and VAE are all pulled from
     `Wan-AI/Wan2.1-T2V-14B-Diffusers`. Naive = **137.6GB**, needed = **51.8GB**:

     | repo | total | needed | the trap |
     |---|---|---|---|
     | `krea/krea-realtime-video` | 57.2GB | `transformer/` **28.6GB** | root `krea-realtime-video-14b.safetensors` is an exact 28.6GB duplicate |
     | `Wan-AI/Wan2.1-T2V-14B-Diffusers` | 80.4GB | text_encoder+vae+tokenizer **23.3GB** | its `transformer/` is 57.2GB we never load: Krea replaces it |

  3. **Most of it may already be on disk.** `Wan2.1-T2V-14B-Diffusers` has identical
     component sizes to `skyreels-v2-df-14b` (transformer 57.15 / text_encoder 22.72 /
     vae 0.51) because SkyReels V2 is Wan-based. Same UMT5-XXL and Wan VAE. If those
     Dirs can be shared, Krea's marginal cost is its 28.6GB transformer.
- **Long video by chaining clips** (generate a clip, take its last frame, feed it
  back as the next clip's first frame). Worth trying precisely because it maps onto
  Flyte so naturally: each chunk is a cached task and the chain is just a loop, so a
  30s clip becomes six 5s tasks you can retry independently instead of one fragile
  20-minute task. Uses `wan22-ti2v-5b`'s existing I2V path, so it needs **no new
  model** and no new download. Expect drift and contrast accumulation across hops
  (each chunk re-encodes the previous chunk's output); that degradation *is* the
  interesting result, and it's the honest counterweight to diffusion forcing, which
  keeps a real latent history instead of a single frame.
- **The lightx2v 4-step distill LoRAs** (`lightx2v/Wan2.2-Distill-Loras`, Apache-2.0).
  Repo checked: it carries rank-64 4-step high/low-noise pairs for **both t2v and
  i2v** (four files), so this would speed up the animate path too, not just t2v. The highest-leverage item left. We carry `wan22-t2v-a14b` but keep
  it out of every lineup because it's 126GB and ~15-30 min/clip undistilled; these
  LoRAs cut 30 steps to 4 and make the best-quality Wan demoable. diffusers 0.39 has
  `WanLoraLoaderMixin.load_into_transformer_2`, so the high/low-noise pair maps onto
  the MoE's two experts. Untested. It would also add a **distillation axis** to the
  grid, which is the most interesting question this demo has surfaced: see the speed
  table above, where the 22B model beats the 1.3B one by 6x purely on step count.
- The five newly-added models have **never been run here**. Their sampler defaults come
  from model cards, not measurement, and `est_vram_gb` is computed from parameter
  counts rather than observed. Expect to tune after the first night.

Two things worth **not** re-litigating, both verified: `Wan 2.5/2.6/2.7` open weights
**do not exist** (SEO blogspam insists otherwise; the newest official Wan open weights
are 2.2), and Cosmos-Predict 2.5 is gated, non-diffusers-layout, and a robotics world
model that would look bad on creative prompts. Cosmos3 (shipped 2026-07-09) is ungated
and interesting, but its `model_index.json` names `Cosmos3OmniDiffusersPipeline`, which
does not exist in 0.39.0 (the library exports `Cosmos3OmniPipeline`). A spike, not a
registry entry.

## The things to verify first

- **arm64 + CUDA 13 torch.** Same risk as every demo on this box. If the cu130
  aarch64 wheels don't resolve, build on an NGC PyTorch base instead (see the note
  at the top of `config.py`).
- **PyAV, not imageio-ffmpeg.** Video only becomes an mp4 via an encoder, and
  diffusers' `encode_video` (the only path that muxes LTX-2's audio) hard-raises
  without PyAV. PyAV ships aarch64 wheels; imageio-ffmpeg doesn't reliably.
- **Wan's VAE must stay fp32.** Loading the whole pipeline in bf16 takes the VAE with
  it and the decode comes out washed-out and banded. `load_pipeline` passes an fp32
  `AutoencoderKLWan` in explicitly.
- **No CPU offload.** Every HF example calls `enable_sequential_cpu_offload()`
  because it assumes a 24GB discrete card. On 128GB of unified memory it moves
  nothing and just adds syncs. We keep models resident and use VAE tiling instead
  (which *does* matter: the VAE decode of a 121-frame latent is the single largest
  allocation in a run and OOMs long before the transformer does).
- **Getting artifacts out / blank report tabs.** Same rustfs story as the image demo:
  if the report tab is blank, forward port 30002. See that README's troubleshooting
  section.
