# DGX Spark setup for the Gemma 4 chat tutorial

This is the minimal, ordered checklist to go from a freshly imaged **NVIDIA DGX Spark** (Grace-Blackwell GB10, aarch64, ~119 GiB unified memory) to the two Flyte 2 apps in this directory running and serving chat.

Read this before `README.md` — `README.md` documents the *what* and the *why-it-broke-and-how-we-fixed-it*. This doc is just *what to actually run*.

## 0. Host prerequisites

You need on the Spark host:

```bash
# NVIDIA driver visible
nvidia-smi -L
# → GPU 0: NVIDIA GB10 (UUID: ...)

# Docker daemon up, with nvidia-container-toolkit hooks installed
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.0-base-ubuntu22.04 nvidia-smi -L

# uv (https://docs.astral.sh/uv/)
uv --version

# Architecture sanity check — must be aarch64
uname -m
```

If `docker run --gpus all` fails, install/configure the NVIDIA container runtime first ([docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)).

If you'll be SSH'ing/Tailscaling in from elsewhere, install Tailscale on the Spark and bring it up (`tailscale up`) — the devbox UI binds to `0.0.0.0:30080` so reaching it via the Spark's Tailscale IP just works.

## 1. HuggingFace: token + Gemma 4 license

Gemma 4 weights are gated. On HuggingFace:

1. Visit `https://huggingface.co/google/gemma-4-26B-A4B-it` and click **Acknowledge license**.
2. Generate a read-only token at `https://huggingface.co/settings/tokens`.

Keep that token handy — you'll paste it into a Flyte secret in step 5.

## 2. Clone + venv + Flyte CLI

```bash
cd tutorials/gemma4-chat

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

# `flyte` is a console script provided by the `flyte` package
flyte --version
# → Flyte SDK version: 2.x.y
```

## 3. Start the devbox **with `--gpu`**

This is the single most-missed step. Without `--gpu`, the devbox container is started with `runc` (no GPU) and your workload pods schedule but can't actually use the GPU.

```bash
flyte start devbox --gpu
```

Verify the GPU is visible from inside the devbox container:

```bash
docker exec flyte-devbox nvidia-smi -L
# → GPU 0: NVIDIA GB10 (UUID: ...)
```

Also confirm k3s has the device plugin and the node has `nvidia.com/gpu: 1`:

```bash
docker exec flyte-devbox kubectl get node -o jsonpath='{.items[0].status.capacity.nvidia\.com/gpu}'
# → 1
```

## 4. Configure the Flyte CLI to point at the devbox

```bash
flyte create config \
    --endpoint localhost:30080 \
    --project flytesnacks \
    --domain development \
    --builder local \
    --insecure
```

This writes `.flyte/config.yaml` in the current directory.

## 5. Create the HF_TOKEN secret

```bash
flyte create secret HF_TOKEN
# paste the hf_xxx token from step 1 (input is hidden)
```

Verify:

```bash
flyte get secret | grep HF_TOKEN
# → HF_TOKEN regular ... FULLY_PRESENT
```

## 6. Open the Flyte UI

- Locally on the Spark: <http://localhost:30080/v2>
- From a Tailscale-connected machine: `http://<spark-tailscale-ip>:30080/v2`
  (find the IP with `tailscale ip` on the Spark)

You'll watch the deploys land here in steps 7–8.

## 7. Deploy the vLLM model server

This single command does everything: prefetches the model from HF into Flyte's object store (one-shot, ~15-20 min the first time), builds the vLLM image, pushes it to the devbox-local registry, and registers the app with Knative.

```bash
python vllm_server.py
```

Things to know:

- **Model**: `google/gemma-4-26B-A4B-it`. ~52 GB safetensors. The IT (instruction-tuned) variant — needed for chat format and thinking-mode support.
- **Image**: `vllm/vllm-openai:gemma4-cu130` as the base, layered with `flyteplugins-vllm` (provides `vllm-fserve` for streaming weights from object store to GPU). NVIDIA's gemma4 fork — vanilla vLLM doesn't recognize Gemma 4's architecture.
- **GPU**: `gpu=1` (any GPU, not typed). The local devbox node is labeled GB10 and a typed `"H100:1"` request would silently match nothing.
- **Memory**: `--gpu-memory-utilization 0.85`. GB10's 119.7 GiB unified memory has ~106 GiB free at startup; the default 0.9 ratio overshoots by ~1 GiB and crashes with `ValueError: Free memory < desired`.
- **Cold start**: ~6 min — image pull (cached after first), safetensors stream from object store, Inductor compile, CUDA-graph capture across 50+ batch sizes.
- **Autoscale**: scales to 0 after 30 min idle. First message after that pays the full cold start.

If the prefetch step hangs on HuggingFace timeouts (it sometimes does), let it retry — it's run-cached so re-running is cheap. If you have a known-good prefetch run name, skip the re-prefetch:

```bash
GEMMA_PREFETCH_RUN=<run-name-from-flyte-ui> python vllm_server.py
```

## 8. Deploy the Gradio chat UI

```bash
python chat_app.py
```

This is a much smaller image (just `gradio` + `openai`); build is fast. The chat app talks to the vLLM app over the **cluster-internal Knative DNS** (`http://<app>.flyte.svc.cluster.local`), not the public `.localhost` URL — that one only resolves on the host.

The chat URL is logged at the end:

```
Chat UI deployed: http://gemma4-chat-ui-flytesnacks-development.localhost:30081/
```

## 9. Open the chat

From the Spark itself: paste the URL above into a browser.

From a remote/Tailscaled machine: Knative routes by `Host` header, so add this to **your local** `/etc/hosts` (the machine you're browsing from):

```
<spark-tailscale-ip>  gemma4-chat-ui-flytesnacks-development.localhost gemma4-26b-a4b-it-vllm-flytesnacks-development.localhost
```

Then `http://gemma4-chat-ui-flytesnacks-development.localhost:30081/` works.

Type a message. The 🧠 Thinking panel fills with the model's reasoning, then the answer appears below it.

## DGX-Spark-specific values cheat sheet

| Setting | Value | Why |
|---|---|---|
| `flyte start devbox` | `--gpu` | adds `--gpus all` + uses `flyte-devbox:gpu-latest` |
| `gpu` in `flyte.Resources` | `1` | typed labels (`H100`, `L40s`) don't match the GB10 node |
| `--gpu-memory-utilization` | `0.85` | only ~106 GiB free of 119.7 GiB unified memory |
| `Image.from_*(registry=...)` | `"localhost:30000"` | otherwise SDK pushes to `ghcr.io/flyteorg` and 403s |
| `Image.platform` | `("linux/arm64",)` | host is aarch64; QEMU-emulated amd64 segfaults on `uv venv` |
| vLLM base image | `vllm/vllm-openai:gemma4-cu130` | Gemma 4 architecture support + Blackwell sm_120 kernels |
| `flyteplugins-vllm` install | via `with_commands(["/usr/bin/python3 -m pip install --pre flyteplugins-vllm"])` | base image's torch/vllm live in system Python, not Flyte's `/opt/venv` |
| HF model | `google/gemma-4-26B-A4B-it` (IT variant) | base model has no chat template; IT ships `chat_template.jinja` |
| inter-app URL | `http://<app>-flytesnacks-development.flyte.svc.cluster.local` | Knative `.localhost` only resolves on the host, not from sibling pods |

## Tear down

```bash
# stop the devbox (preserves data volume — apps and prefetched models survive a restart)
flyte stop devbox

# fully delete the devbox cluster + its volume
flyte delete devbox --volume
```
