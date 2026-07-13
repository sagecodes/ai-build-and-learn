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

Open the run's **Report** tab. The clips play inline.

---

## What's here

| file | what it does |
|---|---|
| `models.py` | the model registry: repos, pipeline classes, sampler defaults, **download patterns**, memory estimates |
| `config.py` | Flyte images, environments, GPU resources, and the DGX Spark env vars |
| `videogen_core.py` | Flyte-free: load, generate, encode mp4, **render the report** (this is where playback lives) |
| `compare_pipeline.py` | the pipeline: `fetch_weights` (cached), `generate_for_model`, `compare`, `animate` |
| `app.py` | Gradio studio: a thin CPU launcher that submits runs and links the report |
| `run_local.py` | host GPU, no Flyte. The fastest way to check a model loads at all |
| `download_outputs.py` | pull the `.mp4` artifacts out of the devbox blob store |

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

## Image to video

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

**On reusing the image-gen project's models:** you can't, quite. Flyte's cache is
keyed on `(project, task, version, inputs)`, and this demo runs in the
`video-generation` project, so the image-generation project's already-cached SDXL
does **not** carry over. That's why the first-frame models here are deliberately
cheap: `sd-turbo` is a 5GB pull that makes a frame in about a second, paid once.
(If you'd rather share the cache, point `.flyte/config.yaml` at the same project the
image demo uses.)

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
