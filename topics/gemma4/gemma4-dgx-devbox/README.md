# Gemma 4 Chat on Flyte 2

Port of the original Ollama+Gradio Gemma 4 demos to a Flyte 2 devbox running on a GPU host (tested on DGX Spark, NVIDIA GB10, aarch64). One vLLM backend + three Gradio frontends:

- **vLLM model server** (`vllm_server.py`) — serves Gemma 4 IT via vLLM's OpenAI-compatible API. Streams safetensors directly from Flyte's object store to GPU. Autoscales to zero.
- **Gradio chat UI** (`chat_app.py`) — text chat. Pickled into a separate Flyte app. Talks to vLLM over the cluster-internal Knative DNS. Has a thinking-mode toggle and a thinking-budget slider.
- **Gradio vision UI** (`vision_app.py`) — image upload → free-form Q&A or bounding-box detection (`box_2d` JSON, drawn on the image).
- **Gradio live-camera UI** (`live_camera_app.py`) — webcam → vision caption every few seconds. Optionally exposes a public HTTPS URL via Gradio's tunnel.

The chat and vision apps preserve the 🧠 Thinking panel from the originals — Gemma 4 IT's thinking is wrapped in `<|channel>...<channel|>` special-token markers, which we keep visible by setting `skip_special_tokens=False` and parse client-side.

## Files

| File | What it does |
|------|--------------|
| `config.py` | Model + GPU choice. Default is `gemma-4-26B-A4B-it`; flip via `GEMMA_VARIANT=31b`. |
| `prefetch_model.py` | One-shot `flyte.prefetch.hf_model` — downloads HF weights into Flyte object store. |
| `vllm_server.py` | `VLLMAppEnvironment` for the chosen Gemma. `__main__` runs prefetch + deploys. |
| `chat_app.py` | Gradio chat `AppEnvironment` + `@env.server`. |
| `vision_app.py` | Gradio vision Q&A + bounding-box detection app. |
| `live_camera_app.py` | Gradio webcam vision-caption `AppEnvironment` + `@env.server`. |
| `requirements.txt` | Local-side deps (no vllm — that runs in the Flyte container). |
| `SPARK_SETUP.md` | Quick-start setup guide specific to DGX Spark. |

## Setup

```bash
cd tutorials/gemma4-chat

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Start the devbox

```bash
flyte start devbox --gpu
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

The `--gpu` flag is critical — without it, the devbox container is started without `--gpus all` and workload pods can't see the GPU. It also swaps the default image to `cr.flyte.org/flyteorg/flyte-devbox:gpu-latest`, which has the NVIDIA runtime hooks baked in. Verify with:

```bash
docker exec flyte-devbox nvidia-smi -L
```

## Add your HF token

Gemma is gated. Create a Flyte secret with a HuggingFace token that has accepted the Gemma license:

```bash
flyte create secret HF_TOKEN
# paste your hf_xxx token
```

## Deploy

Order matters — the frontends hardcode the cluster-internal vLLM URL by name, so vLLM must be deployed first (the URL resolves lazily, so the apps tolerate a cold backend).

```bash
# 1. Serve Gemma 4 via vLLM. Prefetches weights on first run.
python vllm_server.py
# → vLLM app deployed: https://...

# 2. Pick whichever frontends you want — each is independent.
python chat_app.py                          # text chat
GRADIO_SHARE=1 python vision_app.py         # image upload Q&A + bbox detection
python live_camera_app.py                   # webcam → caption every few seconds
```

If a prior prefetch run already wrote the weights to the object store and you don't want to re-download:

```bash
GEMMA_PREFETCH_RUN=<run-name> python vllm_server.py
```

Get the run name from the Flyte UI (`http://localhost:30080/v2`) under successful `prefetch-hf-model` runs, or from the `Prefetch run:` line printed by an earlier successful invocation.

Open the frontend URL. First request spins up the vLLM replica (cold start ~6 min on first deploy, ~30–60s after); subsequent requests are warm.

## Switching models

```bash
GEMMA_VARIANT=31b python vllm_server.py
GEMMA_VARIANT=31b python chat_app.py
```

| Variant | Params | `gpu` in `config.py` | Notes |
|---|---|---|---|
| `gemma-4-26B-A4B` (default) | 26B total / 4B active (MoE) | `1` | Fast — only 4B active params per forward pass. Comfortable on 80GB+ GPUs and on GB10 unified memory. |
| `gemma-4-31B` | 31B dense | `1` (bump to `2` for TP=2) | Dense bf16 ≈ 62GB; needs TP=2 on a multi-GPU box. On a single-GPU GB10 it's tight even with unified memory — try `--max-model-len 4096` and watch for OOMs. |

`gpu` is a plain count (any GPU type). On real clusters you can use Flyte's `"<accelerator>:<count>"` form to match a node label, but the GB10 devbox node isn't labeled `H100`/`L40s`/etc. and a typed mismatch silently drops the request — see SPARK_SETUP.md.

## Architecture

```
┌────────────────────┐
│   chat_app.py      │ ─┐
├────────────────────┤  │   cluster-internal Knative DNS    ┌──────────────────────┐
│   vision_app.py    │ ─┼──▶  http://<vllm-app>.flyte.svc   ▶│ gemma4-26b-a4b-vllm  │
├────────────────────┤  │      .cluster.local               │ (vLLM, 1 GPU)        │
│ live_camera_app.py │ ─┘    (passed as a Parameter)        │ port 8080            │
│  (Gradio, CPU)     │                                       └──────────────────────┘
└────────────────────┘                                                  ▲
        ▲                                                               │ stream safetensors
        │ user (browser)                                                │
        │                                                          Flyte object store
        ▼                                                          (prefetched HF weights)
   public Knative URL
   (or gradio.live HTTPS tunnel for webcam)
```

The frontends pass the cluster-internal URL as a string Parameter rather than wiring through `AppEndpoint` + `depends_on` — see SPARK_SETUP.md for why.

## Why vLLM (not Ollama)?

Flyte 2's first-class GPU serving is `flyteplugins.vllm.VLLMAppEnvironment` (and `flyteplugins.sglang.SGLangAppEnvironment`). Both expose an OpenAI-compatible API, handle GPU resources, autoscale, and stream model weights directly from blob store to GPU. Ollama would mean managing a sidecar process, manual model pulls inside the container, and no scale-to-zero.

vLLM over SGLang for chat: simpler image (no Rust/CUDA toolkit install at build time), broader model support today. SGLang wins for structured/JSON output — swap by importing `SGLangAppEnvironment` and adjusting `extra_args` (`--context-length` instead of `--max-model-len`, `--tp` instead of `--tensor-parallel-size`).

## Troubleshooting

**`Repository google/gemma-4-26B-A4B does not exist in HuggingFace`** — your HF token hasn't accepted the Gemma license, or the repo path drifted. Visit the model page and click "Acknowledge license", then retry.

**vLLM pod OOMs at startup** — drop `--max-model-len` in `config.py`, or move to the larger GPU spec.

**`<think>` tags showing inline in the answer** — Gemma chose not to produce a thinking block for that prompt; or the tag name differs. Check what the model actually emits via vLLM's `/docs` UI, then update `OPEN`/`CLOSE` in `chat_app.py:_split_thinking`.

**Chat UI shows the URL but `/v1/chat/completions` fails** — vLLM replica is still cold-starting. Wait ~6 min on first request after idle (image pull + safetensors stream + Inductor compile + CUDA-graph capture). Watch the vLLM app logs in the Flyte UI at http://localhost:30080/v2.

**vLLM image build fails on aarch64 / Blackwell (GB10) host** — vanilla `vllm==0.11.0` + `flashinfer` wheels are x86_64-only and even when they build, they don't recognize Gemma 4. Use NVIDIA's prebuilt `vllm/vllm-openai:gemma4-cu130` image instead (already wired up in `vllm_server.py`). See [build.nvidia.com/spark/vllm](https://build.nvidia.com/spark/vllm).

## Why this is the way it is — gotchas we hit

The DGX-Spark-specific knobs (`--gpu` flag, `gpu=1` over typed, `localhost:30000` registry, `linux/arm64` pin, `gemma4-cu130` image, `flyteplugins-vllm` install target, `gpu-memory-utilization=0.85`, cluster-internal URL) are documented in **SPARK_SETUP.md**'s cheat sheet. The notes here are about the apps themselves.

1. **`from_base()` returns a frozen, non-extendable image** — to layer `flyteplugins-vllm` on top, call `.clone(extendable=True)`. Setting `platform` also requires `object.__setattr__` since `Image` is a frozen dataclass and `from_base()` doesn't expose a platform kwarg.

2. **Gradio version matters** — `gr.Chatbot(type="messages")` (the metadata-titled 🧠 Thinking panel) needs Gradio 5.x. 6.x dropped that kwarg, 4.x doesn't have messages format. We pin `gradio==5.42.0`.

3. **Knative scales to zero by default** — first request after idle pays the full ~6 min cold start. We bump `scaledown_after` to 1800s (30 min) on the vLLM server so an active session doesn't trip over it; the frontends use shorter idle windows since their cold starts are quick.

4. **Webcam needs HTTPS or localhost** — `getUserMedia` is blocked over plain HTTP from a non-localhost origin. Two ways to handle it for `live_camera_app.py` and `vision_app.py` (image upload also needs a secure context for the file picker on some browsers):

    - On the Spark itself, browse to `http://localhost:30081/...` — `localhost` is exempt and webcam works.
    - From a remote/Tailscaled machine, deploy with `GRADIO_SHARE=1`. Gradio inside the pod opens an outbound TLS tunnel to `gradio.live` and gets a public `https://<random>.gradio.live` URL that proxies back to the pod's gradio port.

5. **Gradio share traffic bypasses Knative's queue-proxy** — when using `GRADIO_SHARE=1`, requests come in via the gradio.live tunnel directly to gradio's port inside the pod, **not** through Knative's ingress + queue-proxy on `:8012`. So Knative sees zero activity and scales the pod (and tunnel) down at `scaledown_after`. The vision and live-camera apps work around this with an in-pod keep-alive thread that pokes `:8012` while there's recent user activity. To pin always-on instead, set `replicas=(1, 1)` on the AppEnvironment.

6. **Gemma 4 IT thinking is wrapped in special tokens, not `<think>`** — the IT model emits its chain-of-thought between `<|channel>thought\n` and `<channel|>` markers, which vLLM normally strips. Set `skip_special_tokens=False` on the request so they reach the client, then split client-side (`_split_thinking` in `chat_app.py` / `vision_app.py`). When the thinking-budget cap fires, we second-pass with `enable_thinking=False` and feed the truncated thought as priming.

7. **Detection is emergent, not a head** — `vision_app.py`'s "Detect" tab just prompts the model for `box_2d` JSON in the same normalized 0–1000 coordinate format Gemini uses. Quality varies; small/clustered objects in particular drop out. There's no detector head to tune.
