# Open-source image generation on Flyte: compare models + fine-tune LoRA

Generate images with the top open-source text-to-image models, compare them
side by side in a Flyte report, drive them from a Gradio studio, and fine-tune
one with a LoRA. Everything runs on a Flyte 2 devbox with a GPU; weights are
pulled from HuggingFace at runtime (nothing baked into the image).

```
                       ┌─ FLUX.1-schnell ─┐
   prompt(s) ──►       ├─ SDXL ───────────┤ ──► side-by-side Flyte report
                       └─ Qwen-Image ─────┘      (+ full-res PNGs saved as a Dir)

   5 photos of a subject ──► SDXL LoRA (a few hundred steps) ──► base-vs-tuned report
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
| `lora_finetune.py`    | DreamBooth-LoRA on SDXL, then a base-vs-tuned report.             |
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

## The Gradio studio

A front door to the pipeline: type a prompt, tick the models to race, hit
Generate, and the studio **launches a Flyte `compare` run** and links its report
(the prompt x model grid + saved PNGs). The app is a thin CPU launcher: it loads
no model and touches no GPU, so it can't pin weights in memory. All the GPU work
happens in the pipeline's tasks. This mirrors the langgraph_agent_research
tutorial's "Gradio over `flyte.run`" pattern.

```bash
# register the pipeline first so the app can reference the compare task
flyte deploy compare_pipeline.py

python app.py                    # local app -> remote (devbox) pipeline
RUN_MODE=local python app.py     # local app -> local pipeline (host dev)
GRADIO_SHARE=1 python app.py     # add a public HTTPS tunnel for a remote browser
```

🔒 in the picker marks gated models. For the interactive, in-process path on a
host GPU (no Flyte), use `run_local.py` instead.

## Fine-tune a LoRA

DreamBooth-style: teach SDXL a new subject from ~5 photos tied to a rare token
(`sks dog`), producing a small LoRA adapter, then compare base vs tuned.

```bash
# end-to-end: train on the 5-image dog set, then render before/after
flyte run lora_finetune.py lora_demo --max_steps 400

# just train and keep the adapter
flyte run lora_finetune.py train_lora --max_steps 400 --rank 8
```

Point `--dataset_repo` at any HF image dataset (or your own photos) and change
`--instance_prompt` to teach a different subject. The loop is intentionally
compact (no prior-preservation / EMA / SNR weighting); see diffusers'
`train_dreambooth_lora_sdxl.py` for the full production recipe.

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
