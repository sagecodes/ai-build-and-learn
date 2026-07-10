# Open-source image generation on Flyte: compare models + fine-tune LoRA

Generate images with the top open-source text-to-image models, compare them
side by side in a Flyte report, drive them from a Gradio studio, and fine-tune
one with a LoRA. Everything runs on a Flyte 2 devbox with a GPU; weights are
pulled from HuggingFace at runtime (nothing baked into the image).

```
                       ┌─ FLUX.1-schnell ─┐
   prompt(s) ──►       ├─ SDXL ───────────┤ ──► side-by-side Flyte report
                       └─ Qwen-Image ─────┘      (+ full-res PNGs saved as a Dir)

   5 photos of a subject ──► SDXL LoRA   (U-Net, epsilon loss) ──► base-vs-tuned report
   18 images of a style  ──► Chroma LoRA (DiT, flow matching)  ──► base-vs-tuned report
```

## Quickstart

```bash
# 1. GPU devbox (Flyte 2 cluster with host GPUs)
flyte start devbox --gpu

# 2. CLI venv (submits runs; separate from the heavy task image)
cd topics/image-generation/imagegen-flyte
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python flyte==2.2.1 kubernetes 'connectrpc==0.10.0'
export PATH="$PWD/.venv/bin:$PATH"

# 3. HF token secret (needed for gated models; avoids rate limits on the rest)
flyte create secret HF_TOKEN --value hf_xxxxxxxxxxxx

# 4. Compare models -> side-by-side Flyte report (open the run URL it prints)
flyte run compare_pipeline.py compare \
  --models '["sdxl","zimage-turbo","flux1-schnell"]' \
  --prompts '["a red panda barista pouring latte art, cozy cafe, 50mm, bokeh"]'

# 5. Pull the generated images to ./downloads/
python download_outputs.py <run_name>
```

The first run downloads weights (cached after, so later runs are instant); big
models like Qwen are ~50GB. Weights fetch on a CPU pod and log a live progress
heartbeat (`X GB so far (+Y MB/30s)`) in the task logs, with a socket timeout +
retries so a stalled connection self-heals instead of hanging. Sections below cover
the Gradio studio, LoRA fine-tune, host-GPU mode, pulling artifacts, and
troubleshooting.

## What's here

| File                  | What it is                                                        |
|-----------------------|-------------------------------------------------------------------|
| `models.py`           | Registry of models (repo, pipeline class, steps, license, family).|
| `config.py`           | Torch/diffusers image, GPU task env + app pod template, HF secret.|
| `imagegen_core.py`    | Flyte-free: load a pipeline, generate, render the report HTML.     |
| `compare_pipeline.py` | The batch pipeline: prompts × models → one side-by-side report.   |
| `app_ui.py`           | Flyte-free Gradio studio (prompt + model picker + gallery).       |
| `app.py`              | Serves the studio as a Flyte GPU app.                             |
| `lora_finetune.py`    | DreamBooth-LoRA on SDXL (subject), then a base-vs-tuned report.   |
| `lora_chroma.py`      | Style-LoRA on Chroma (flow matching); train, sample, generate.    |
| `lora_data.py`        | Small captioned datasets for LoRA training + preprocessing.       |
| `run_local.py`        | Run any of it on the host GPU, no Flyte (quick iteration).         |

## The models

All are open weights, pulled from HuggingFace at runtime. `models.py` is the
source of truth; the default comparison set is the non-gated spread
`flux1-schnell, sdxl, qwen-image`.

| Key             | Repo                                   | Family              | License            | Gated |
|-----------------|----------------------------------------|---------------------|--------------------|-------|
| `flux1-schnell` | black-forest-labs/FLUX.1-schnell       | DiT / rectified flow| Apache-2.0         | yes¹  |
| `zimage-turbo`  | Tongyi-MAI/Z-Image-Turbo               | DiT / distilled     | Apache-2.0         | no    |
| `sdxl`          | stabilityai/stable-diffusion-xl-base-1.0 | U-Net LDM         | OpenRAIL++         | no    |
| `qwen-image`    | Qwen/Qwen-Image                        | MM-DiT              | Apache-2.0         | no    |
| `sd35-large`    | stabilityai/stable-diffusion-3.5-large | MM-DiT              | Stability (gated)  | yes   |
| `flux1-dev`     | black-forest-labs/FLUX.1-dev           | DiT / rectified flow| non-commercial     | yes   |
| `flux2-dev`     | black-forest-labs/FLUX.2-dev           | DiT (next-gen)      | non-commercial     | yes   |

¹ FLUX.1-schnell is Apache-2.0 but its **repo** is still click-through gated:
accept the license once at the model page or the download 403s even with a valid
token. Every FLUX repo is gated this way.

"Diffusion vs transformer": SDXL is the classic **U-Net** latent-diffusion
denoiser; FLUX / Z-Image are **DiT / rectified-flow** transformers; SD3.5 and
Qwen-Image are **MM-DiT** (multimodal diffusion transformers). All sample by
denoising; the backbone is what differs. Gated repos need a HuggingFace token
that has accepted the model's license (see below).

> **Ungated out of the box:** `sdxl`, `qwen-image`, `zimage-turbo` need no
> license click. If you just want a green run immediately, use those; accept the
> FLUX / SD3.5 licenses (links below) to unlock the rest.

## Prerequisites

### 1. A GPU devbox

```bash
flyte start devbox --gpu          # local Flyte cluster with host GPUs
```

UI at http://localhost:30080/v2, image registry at localhost:30000.

### 2. A local venv (for the CLI / to submit runs)

```bash
cd topics/image-generation/imagegen-flyte
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python flyte==2.2.1 kubernetes 'connectrpc==0.10.0'
```

> **Pin `connectrpc==0.10.0`.** flyte 2.2.1 calls `ctx.request_headers()`, but
> connectrpc 0.11 turned `request_headers` into a property, so a fresh install
> fails every run with `SelectCluster failed: 'Headers' object is not callable`
> (surfaced only as "Failed to get signed url"). 0.10.x keeps it a method.

`.flyte/config.yaml` points the CLI at the devbox (`localhost:30080`, insecure,
local image builder, project `flytesnacks` / domain `development`).

### 3. A HuggingFace token secret

The tasks/app mount an `HF_TOKEN`. It's required for the **gated** models
(SD3.5, FLUX.1/2 dev) and lets you avoid rate limits on the open ones. Create it
once on the devbox (use a token with **gated-repo access** if you want the gated
models):

```bash
flyte create secret HF_TOKEN --value hf_xxxxxxxxxxxxxxxxxxxx
# verify:
flyte get secret            # HF_TOKEN → FULLY_PRESENT
```

To rotate it (e.g. swap in a token that has gated access), delete then recreate:

```bash
flyte delete secret HF_TOKEN
flyte create secret HF_TOKEN --value hf_yyyy
```

Accept each gated model's license on its HuggingFace page first, or the download
still 401s even with a valid token.

## Compare models (the report)

```bash
cd topics/image-generation/imagegen-flyte
export PATH="$PWD/.venv/bin:$PATH"

# default 3-model set, one prompt
flyte run compare_pipeline.py compare \
  --prompts '["a red panda barista pouring latte art, cozy cafe, 50mm, bokeh"]'

# choose the models and add prompts (the grid is prompts × models)
flyte run compare_pipeline.py compare \
  --prompts '["neon cyberpunk alley in the rain","a storefront sign that reads OPEN 24 HOURS"]' \
  --models '["sdxl","flux1-schnell","qwen-image"]'

# single model, quick smoke
flyte run compare_pipeline.py generate_one \
  --model_key flux1-schnell --prompts '["a corgi astronaut, studio light"]'
```

Open the run URL it prints. The **report** tab shows the images side by side
(models are columns, prompts rows) with per-image timing; the full-resolution
PNGs are saved as each task's output directory.

### Prompts to try (and what each stresses)

A small eval sweep that surfaces where models diverge. In the studio, paste one
per line; on the CLI, pass them as a JSON array (escape the inner quotes in any
prompt that has them, like the menu and poster ones: `\"TODAY: PUMPKIN LATTE $5\"`).

- **Fine detail, materials, macro light:** a hedgehog knight in handmade acorn armor guarding a glowing blue mushroom, misty forest floor at dawn, cinematic macro, golden rim light
- **Text rendering**: a hand-lettered chalkboard menu outside a cafe that reads "TODAY: PUMPKIN LATTE $5"
- **Hands + shallow depth of field** (hands are the classic tell): close-up of a barista's hands pulling an espresso shot, steam, shallow depth of field, 85mm
- **Translucency + glow on black:** a translucent glass hummingbird perched on a neon sign, bioluminescent, dark background, macro
- **People in a busy scene** (candid face, hands, crowd coherence): a woman laughing while hailing a taxi on a busy New York sidewalk, golden hour, candid street photography, 50mm
- **Portrait realism** (skin, freckles, soft light): portrait of a woman with freckles and windswept hair on a rooftop at sunset, city skyline behind her, 85mm, soft light
- **Dynamic pose + motion blur** (full-body anatomy): a man skateboarding through a sunlit city plaza, motion blur, dynamic pose, street photography
- **Prompt adherence: counting + object binding** (exact counts are a classic failure): exactly three red apples and one green pear in a row on a wooden table, studio light
- **Multiple interacting people** (two faces, hands, interaction): two chefs high-fiving in a busy restaurant kitchen, flames and steam, candid photo
- **Heavy typography + graphic design:** a retro mid-century travel poster for MARS with the headline "VISIT THE RED PLANET", bold flat design
- **Architecture: perspective + repetition:** a symmetrical grand library interior, rows of arched bookshelves receding into the distance, warm light
- **Reflection + physical consistency:** a still alpine lake at dawn perfectly mirroring a snow-capped mountain and pink clouds, ultra sharp
- **Art-style fidelity:** a fox in a snowy forest, Japanese ukiyo-e woodblock print style
- **Lighting extremes / low light:** a lone lantern-lit ramen stall on a dark rainy street, warm glow against deep shadow, cinematic
- **Human + animal: cradling** (arms, contact, scale): a farmer cradling a newborn lamb in both arms, golden-hour field, documentary photo
- **Human + animals: hands + small subjects** (birds on outstretched hands): an old man feeding pigeons on a park bench, birds landing on his outstretched hands, candid street photo
- **Human + animal in motion** (two subjects, mid-action): a woman playing fetch with her golden retriever on a beach at sunset, mid-motion, candid
- **Negation** (models struggle to leave things out): an empty city street at dawn with absolutely no cars and no people
- **Spatial relations + compositionality** (position, not just count): a small red cube on top of a large blue sphere, a green pyramid to the left, plain studio backdrop
- **Reflective / chrome materials** (environment reflected in a curved surface): a chrome robot standing in a desert, the landscape reflected in its polished body, midday sun
- **Liquid / fluid physics** (splash coherence): a splash of coffee frozen mid-air being poured into a white cup, high-speed photography

**How it's wired:** a `fetch_weights` task (CPU, `cache="auto"`) snapshots each
model's HuggingFace repo into a `Dir` in the blob store, then one GPU
`generate_for_model` task per model loads from that cached Dir. So weights
download **once**: every later run is a cache hit served in-cluster, and no GPU
sits idle during the download. On a single-GPU devbox the generate tasks
serialize at the scheduler (one holds the GPU at a time).

**Downloads run one model at a time, on purpose.** Fetching all the models in
parallel opens dozens of concurrent sockets to the HuggingFace CDN, and on a
lossy uplink (like the Spark over Wi-Fi) that congestion black-holes a transfer
mid-stream and hangs the fetch with no error. Serializing keeps the socket count
low and the downloads reliable. In production on a real cluster with a fat,
reliable pipe, fetch in parallel instead: swap the serial loop in
`compare_pipeline.py` back to
`weights = await asyncio.gather(*[fetch_weights(s.key) for s in specs])`.

> **TODO — skip the single-file weight dupes.** Many repos ship a root-level
> single-file checkpoint (`Chroma1-HD.safetensors`, `Chroma1-HD-Flash.safetensors`,
> `raw.safetensors`, `turbo.safetensors`, the `flux*-*.safetensors`) that just
> duplicates the `transformer/` folder `from_pretrained` actually loads. The fetch
> still downloads it, wasting ~18–24 GB per big model. Add these globs to
> `ignore_patterns` in `compare_pipeline.py:fetch_weights`. (Changing that function
> re-keys its `cache="auto"`, so doing it re-downloads every model once — batch it
> with any other fetch change and do it deliberately, not mid-run.)

> **TODO — durable, model-keyed weight store (seedable from local).** `fetch_weights`
> leans on Flyte's `cache="auto"`, which keys on a code hash: editing the task
> re-downloads every model, and there's no clean way to hand it pre-existing
> weights. But the task already returns a `flyte.io.Dir` that lives in the blob
> store, so the weights are right there — the fix is to write them to a **stable,
> model-keyed path** (e.g. `s3://flyte-data/weights/<model_key>/repo`) and look
> there first instead of re-caching. `fetch_weights` checks that path, returns it
> if present (never re-downloads, survives code edits), else pulls from HuggingFace
> and writes it there. Wins: (1) immune to fetch-code changes; (2) a local folder
> can be uploaded to that same path to *seed* Flyte with no HF hit — the inverse of
> the download/export script, giving a full local↔devbox round-trip; (3) weights
> become a real addressable artifact, not an opaque cache entry. Pairs with a
> `download_outputs.py --weights` mode (pull the `/repo` Dirs it currently skips)
> for the export direction.

## The Gradio studio

A front door to the pipeline, with two tabs:

- **Compare models** launches a Flyte `compare` run and links its report (the
  prompt x model grid + saved PNGs).
- **LoRA generate** launches a `generate_with_lora` run against an adapter you
  already trained. Paste the LoRA Dir URI, pick the dataset it was trained on
  (that sets the trigger phrase), and go.

The app is a thin CPU launcher: it loads no model and touches no GPU, so it
can't pin weights in memory. All the GPU work happens in the pipeline's tasks.
This mirrors the langgraph_agent_research tutorial's "Gradio over `flyte.run`"
pattern.

```bash
# register the pipelines first so the app can reference their tasks
flyte deploy compare_pipeline.py
flyte deploy lora_chroma.py      # only needed for the LoRA generate tab

python app.py                    # local app -> remote (devbox) pipeline
RUN_MODE=local python app.py     # local app -> local pipeline (host dev)
GRADIO_SHARE=1 python app.py     # add a public HTTPS tunnel for a remote browser
```

🔒 in the picker marks gated models. For the interactive, in-process path on a
host GPU (no Flyte), use `run_local.py` instead.

## Fine-tune a LoRA

Two trainers, on purpose. They teach the same trick (freeze the model, train a
few MB of adapter) on the two backbone families the model table splits on, and
reading them side by side is most of the lesson.

|                  | `lora_finetune.py` (SDXL)   | `lora_chroma.py` (Chroma)               |
|------------------|-----------------------------|-----------------------------------------|
| backbone         | U-Net latent diffusion      | DiT, rectified flow                     |
| teaches          | a **subject** (`sks dog`)   | a **style** (`yarn art style`)          |
| text encoder     | two CLIPs, plus a pooled embed | one T5-XXL, no pooled embed          |
| training target  | the noise (epsilon)         | the velocity, `noise - latents`         |
| timestep sampling| uniform over 1000           | logit-normal, weighted to the middle    |
| latent layout    | a 4-channel image grid      | 16 channels, packed into 2x2 patches    |

### Subject LoRA on SDXL

```bash
# end-to-end: train on the 5-image dog set, then render before/after
flyte run lora_finetune.py lora_demo --max_steps 400

# just train and keep the adapter
flyte run lora_finetune.py train_lora --max_steps 400 --rank 8
```

Point `--dataset_repo` at any HF image dataset (or your own photos) and change
`--instance_prompt` to teach a different subject.

### Style LoRA on Chroma

Chroma is the ungated, Apache-2.0 de-distill of FLUX.1-schnell. The default set
is 18 yarn-art images and trains in about 37 minutes at 512px on a GB10.

```bash
# end-to-end: fetch weights, train the yarn-art LoRA, render before/after
flyte run lora_chroma.py chroma_lora_demo

# a different dataset, longer, at native resolution
flyte run lora_chroma.py chroma_lora_demo --dataset tarot --max_steps 1200 --resolution 1024

# train only; the adapter comes back as a Dir artifact
flyte run lora_chroma.py train_only --dataset 3d-icon --rank 16
```

### Generating from a trained adapter

Training returns the LoRA as a Dir artifact. Grab its URI from the run's
**Outputs** tab and generate against it without retraining:

```bash
# pictures only (fast): one image per prompt, LoRA applied
flyte run lora_chroma.py generate_with_lora \
  --lora_uri s3://flyte-data/.../chroma_lora_xxxx \
  --prompts '["a fox sitting in the snow, yarn art style"]'

# evaluating a fresh adapter: render each prompt with AND without it
flyte run lora_chroma.py generate_with_lora \
  --lora_uri s3://... --show_base true --lora_scale 0.8 \
  --prompts '["a fox sitting in the snow, yarn art style"]'
```

`--show_base` is the knob that separates the two jobs the report can do. On, you
get a base/tuned pair per prompt and can actually tell what the adapter changed;
it's the default in `chroma_lora_demo` for exactly that reason. Off, you get one
image per prompt at half the GPU time, which is what you want once the adapter is
known good. The base pass always runs *before* `load_lora_weights`, because the
adapter fuses into the live pipeline and there's no "base" left afterwards.

Same thing from the studio's **LoRA generate** tab, with `--show_base` as the
"Also generate base" checkbox and a `lora_scale` slider (0 turns the adapter off
entirely, which is a nice way to sanity-check that it's doing anything at all).

Two things to keep straight:

- **The prompt must contain the trigger.** A prompt without it measures the
  adapter's bleed into unrelated prompts, not its effect. The report prints a
  warning when prompts omit it rather than leaving you with a puzzling null result.
- **The trigger lives in the data, not the dataset name.** The tarot set's
  captions use the rare token `trtcrd`, not "tarot card". `lora_data.py` records
  the real one for each set.

### How big a before/after to expect (measured)

A trained adapter can only teach the base model something it doesn't already
know, and **how much it teaches depends entirely on how rare the trigger is.**

Run `r7gxz7bswkvswldgrsvs` trained the yarn-art LoRA (18 images, rank 16, 800
steps, 512px) and rendered base vs tuned at 1024px:

| prompt | mean abs pixel Δ | near-black pixels, base → tuned |
|---|---|---|
| a fox sitting in the snow | 119 / 255 | 2.0% → 60.2% |
| an astronaut riding a horse | 135 / 255 | 12.0% → 63.6% |
| the Golden Gate Bridge at sunset | 55 / 255 | 20.2% → 21.4% |

The adapter changes a *lot*, but not in the way you might advertise: **base Chroma
already renders "yarn art style" beautifully**, because that phrase is plain
English it has seen in training. What the LoRA actually learned is the *dataset's
photographic setup* — the dark studio backdrop and loose stray strands of those 18
particular photos. Hence near-black backgrounds tripling.

So:

- **Natural-language triggers** (`yarn art style`) sharpen a style the model
  already has. The before/after is real but subtle, and skews toward the training
  set's incidental qualities. Dial it back with `--lora_scale 0.6`.
- **Rare-token triggers** (`trtcrd`, `3dicon`, `sks dog`) carry no prior at all, so
  the base model simply cannot produce them and the before/after is dramatic. This
  is why DreamBooth invented `sks`, and why the tarot and 3d-icon sets are the
  better choice when you want an unmistakable demo.
- 800 steps on 18 images is well into memorizing the set. Fewer steps, or a lower
  `lora_scale`, buys back generality.

Datasets live in `lora_data.py`, each with a trigger phrase and eval prompts that
show the effect. All four were checked against the Hub:

| `--dataset` | Repo | Images | Trigger | License |
|-------------|------|--------|---------|---------|
| `yarn-art` (default) | `Norod78/Yarn-art-style` | 18 | `yarn art style` | unspecified |
| `tarot`     | `multimodalart/1920-raider-waite-tarot-public-domain` | 78 | `trtcrd` | public domain |
| `3d-icon`   | `linoyts/3d_icon` | 23 | `3dicon` | Unsplash |
| `dog`       | `diffusers/dog-example` | 5 | `sks dog` | unspecified |

The run is three tasks, split so nothing expensive holds the GPU:

```
fetch_weights(chroma-hd)  ─┐                        (CPU, cached)
                           ├─► train_chroma_lora ─► sample_chroma_lora
fetch_dataset(yarn-art) ──┘    (GPU)                (GPU)
```

Both fetches are cached CPU tasks keyed on their inputs, so the GPU pod starts
with the weights and the images already on local disk and does nothing but train.
`fetch_dataset` writes a boring folder (`0000.png` … plus `captions.jsonl`), so
reading it back needs no `datasets` dependency and you can eyeball what the model
actually saw.

### The training report

Live, and it keeps its charts after the task finishes:

- a stage indicator (fetch → train → sample) and badges, same shape as the
  `workshops/tutorials` reports
- a KPI row (step, EMA loss, grad norm, steps/sec, elapsed, ETA, trainable params)
- the loss curve: raw per-25-step loss in gray, its EMA in blue
- the gradient norm, on **its own chart**, not a second y axis on the loss plot
- **mean loss bucketed by noise level (sigma)**
- a table view under each chart, and hover tooltips via SVG `<title>` (the report
  is injected as HTML, so it carries no `<script>` and no external assets)

That last chart is the one that matters. Flow-matching loss depends far more on
which sigma a step happened to draw than on how training is going, so the raw
curve looks flat from step 1 to step 800 even when the LoRA is learning fine.
Splitting by noise level makes it legible: near-image steps sit low, near-noise
steps sit high, permanently. **Do not read a flat loss curve as a broken run.**

Notes worth knowing before you change things:

- **Train on `chroma-hd` or `chroma-base`, never `chroma-flash`.** Flash is
  speed-distilled, and a plain flow-matching loss walks the weights off the
  distilled manifold. `chroma_lora_demo` refuses it outright.
- **The same trap rules out Z-Image-Turbo**, which is DMD-distilled. The trainers
  that do support it (ai-toolkit, SimpleTuner) carry a Turbo-specific "assistant
  adapter" to hold the distillation in place while the LoRA learns. Chroma needs
  no such trick, which is exactly why it's the one here. The non-distilled
  `Tongyi-MAI/Z-Image` base would also train fine, at the cost of another ~20GB pull.
- **`--resolution` defaults to 512**, not Chroma's native 1024, because image
  tokens grow with the square (1024 vs 4096 per step) and the demo should be
  watchable. Use 1024 for an adapter you intend to keep.
- The encoders are loaded and freed one at a time (T5, then the VAE, then the
  transformer), so peak memory is one big model rather than three. That is what
  lets an 8.9B transformer train next to a 4.7B text encoder on one box.
- Latents are cached once, so there's no per-step random crop. `--flip_augment`
  (on by default) compensates by encoding each image twice.
- `--limit` subsets the dataset, and it lives on `fetch_dataset`, so changing it
  re-keys that task's cache and re-materializes the folder. That's cheap here
  (the largest set is 78 images), unlike `fetch_weights`.

Both loops are intentionally compact (no prior-preservation, EMA, or SNR
weighting). See diffusers' `train_dreambooth_lora_sdxl.py` and
`train_dreambooth_lora_flux.py` for the full production recipes; note that
neither diffusers nor kohya ships a Chroma trainer, so `lora_chroma.py` is the
FLUX recipe with Chroma's three deviations applied (no `pooled_projections`, no
`guidance` kwarg, and an attention mask that keeps one padding token).

## Host-GPU (no Flyte)

For fast iteration, generate directly on the Spark GPU:

```bash
# needs the torch/diffusers deps locally (a separate, heavier venv than the CLI one)
python run_local.py --models flux1-schnell,sdxl --prompt "a corgi astronaut, studio light"
GRADIO_SHARE=1 python run_local.py --serve         # just launch the studio
```

## The things to verify first

- **arm64 (sbsa) + CUDA 13 torch install.** The image pulls `torch`/`torchvision`
  from the cu130 wheel index (`config.py: TORCH_INDEX`). This resolves on the
  DGX Spark (GB10 Blackwell). If those aarch64 wheels ever fail to resolve, swap
  `config._image` to a CUDA/PyTorch base image (see the note at the top of
  `config.py`) — torch is then already present and you only pip-install diffusers.
- **New-model pipeline classes.** The exact diffusers class for the newest models
  (FLUX.2, Z-Image, Qwen-Image) depends on your diffusers version. The loader
  falls back to `AutoPipelineForText2Image`, but if a model won't load, bump
  `diffusers` in `config.DIFFUSERS_SPEC` or drop it from the default set.
- **GPU on tasks vs apps.** Tasks get the GPU via `flyte.Resources(gpu=1)`. The
  app can't (this SDK drops the GPU from the Knative revision), so it uses a
  `PodTemplate` that sets `nvidia.com/gpu` directly — see the note in `config.py`.
- **IPv6 to the HuggingFace CDN can black-hole weight downloads.** On a dual-stack
  network the OS prefers IPv6, and if the IPv6 route to HF's CDN is broken the
  download does not error, it just **hangs mid-shard at 0 bytes forever** (no
  timeout). Symptom: a `fetch_weights` pod stuck for ages, disk size frozen, a
  `.incomplete` file at 0 bytes. Confirm with `curl -6 -L` vs `curl -4 -L` on a
  weight URL (broken IPv6 = 0 bytes; IPv4 = full speed). `fetch_weights` pins name
  resolution to IPv4 to avoid this (a `socket.getaddrinfo` filter); the same guard
  protects future large pulls (Qwen, FLUX, video weights).

## Getting artifacts (images / videos) out of the devbox

Task outputs (`File`/`Dir`) live in rustfs, the in-cluster S3-compatible store
(bucket `flyte-data`, static creds `rustfs` / `rustfsstorage`). Right-clicking an
image in a report does not scale to videos or bulk, so use the helper:

```bash
python download_outputs.py <run_name>                 # -> ./downloads/
python download_outputs.py <run_name> --dest ./out
# from a laptop (not on the Spark), hit the Tailscale IP to skip the 30002 forward:
python download_outputs.py <run_name> --endpoint http://<spark-tailscale-ip>:30002
```

It walks **every action** in the run (so a `compare` run's per-model images, which
live in the child `generate_for_model` actions, all come down, not just the root
output), decodes msgpack dataclass literals like `ModelRun` to find the blob URIs,
and pulls them via the Flyte download API. By default it keeps only media files and
skips weight-fetch `/repo` dirs, so `.safetensors` never come down; use `--all` for
everything, `--ext .mp4,.png` to customize, or `--skip-uri` to change what's
excluded.

**Auth, and why the client needs it.** `File.download()` / `Dir.download()` hit the
blob store *directly* using storage credentials (uploads use presigned URLs;
downloads do not). Inside a task this is automatic: the pod is injected with
`FLYTE_AWS_ACCESS_KEY_ID` / `FLYTE_AWS_SECRET_ACCESS_KEY` (= `rustfs` /
`rustfsstorage`) and the in-cluster endpoint, so `await mr.images.download()` just
works. From your **laptop / devbox host** the client has *no* storage creds, so you
must supply them; the SDK reads these env vars (which `download_outputs.py` sets for
you):

```bash
export FLYTE_AWS_ENDPOINT=http://<spark-tailscale-ip>:30002   # or localhost:30002 on the Spark
export FLYTE_AWS_ACCESS_KEY_ID=rustfs
export FLYTE_AWS_SECRET_ACCESS_KEY=rustfsstorage
export FLYTE_AWS_S3_ADDRESSING_STYLE=path                     # rustfs/minio need path-style
```

Then `flyte.io.Dir(path="s3://flyte-data/...").download("./out")` works from the
host. On the devbox this is the *simple* case (static creds); a real cloud
deployment instead hands clients time-limited **presigned URLs** (no local creds) or
uses IAM in-cluster. To get a URI by hand: `flyte get io <run_name>`, or decode the
output literal as the script does. `mc` / `aws` with the same creds work too and are
handy for very large files (resumable, parallel).

## Troubleshooting: the report tab won't load (blank / spinner)

> **First rule when working against a remote devbox (e.g. the Spark): shut down
> any local devbox AND quit Docker Desktop on your laptop.** A local `flyte-devbox`
> publishes `*:30002` on your laptop and wins `localhost:30002`, so the report's
> presigned URLs hit the wrong (or dead) rustfs and the tab never loads. This is
> the single most common cause of the blank/spinner report and it is easy to
> forget. `flyte stop devbox` then quit Docker Desktop before you connect.

Reports are images served from rustfs (the in-cluster S3) via **presigned URLs**.
The devbox signs those URLs with the host `http://localhost:30002` by design: the
devbox is a Docker container (`flyte-devbox`) that publishes `30002:30002`, and the
signed host is matched to that publish. So whatever browser opens the report
**must be able to reach `localhost:30002`**.

- Browser running **on the Spark itself**: works out of the box (Docker publishes
  the port on the host).
- Browser on a **laptop** (the usual case): needs a forward from the laptop to the
  Spark's `30002`. VSCode's remote port-forward does this, but it goes stale on a
  reconnect, a dropped SSH, or a port collision (a second devbox, e.g. one on a
  Mac, grabbing `localhost:30002` shows up as `NoSuchKey` instead of a blank tab).

**Ground truth first (run in a laptop terminal, not the remote one):**

```bash
curl -sm5 localhost:30002/ | head
# 403 AccessDenied XML -> forward is live; hard-refresh the report tab
# hang / timeout       -> forward is dead; rebuild it (below)
```

If a laptop `ssh -N -L 30002:localhost:30002 sage@<spark-ip>` fails with
`bind [127.0.0.1]:30002: Address already in use`, something local is squatting the
port (most often a lingering Docker/devbox on the laptop publishing `*:30002`, plus
a stale VSCode forward). Find and clear it, then rebuild the forward:

```bash
lsof -nP -iTCP:30002 -sTCP:LISTEN     # shows e.g. com.docker / Code Helper on 30002
docker ps --filter publish=30002      # if Docker: docker stop <name>, or quit Docker Desktop
```

**Fix: rebuild the path to `localhost:30002`.** Fastest is to reconnect / reload
the VSCode remote window, which tears down and rebuilds its port-forward. For a
forward that does not depend on VSCode's flaky panel, run a dedicated tunnel from
the laptop and leave it open:

```bash
ssh -N -L 30002:localhost:30002 sage@<spark-tailscale-ip>
```

**Why you cannot just re-point the signed URL to the Tailscale IP.** The endpoint
lives in ConfigMap `flyte-binary-config` (`100-inline-config.yaml` ->
`storage.signedURL.stowConfigOverride.endpoint`), but it is **not** editable in any
durable way:

- k3s runs a wrangler drift-controller that reverts any live edit to the ConfigMap
  back to `localhost:30002` within seconds (verified: it even re-adopts the object
  if you strip its `objectset.rio.cattle.io/*` ownership markers).
- The k3s Addon's source manifest (`/var/lib/rancher/k3s/server/manifests/flyte.yaml`)
  does not exist on disk; the desired state is cached in etcd, so there is no file
  to `sed`.
- `flyte start devbox` regenerates this config on every start, so even a change
  that stuck would be wiped on the next `flyte stop/start devbox`.

Note: because Docker publishes `30002` on all host interfaces, rustfs is directly
reachable over Tailscale (`curl -sm5 http://<spark-tailscale-ip>:30002/` returns
`403 AccessDenied` = healthy). That does not help the browser on its own, because
the signed URL still says `localhost`; a durable switch to the Tailscale host would
have to happen in the devbox tooling that generates the config, not in the running
cluster.
