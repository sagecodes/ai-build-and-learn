# Voxtral TTS

Local text-to-speech demo using [mistralai/Voxtral-4B-TTS-2603](https://huggingface.co/mistralai/Voxtral-4B-TTS-2603) — an open-weights TTS model from Mistral.

- 9 languages, 20 preset voices, 24 kHz output
- Runs as a vLLM server with an OpenAI-compatible `/audio/speech` endpoint
- Needs a single GPU with ≥16 GB memory (Mistral tested on H200)
- License: **CC BY-NC 4.0** (non-commercial)

## Setup

```bash
cd topics/voxtral

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

On an H100/H200/A100 x86 box, that's enough. On **DGX Spark (GB10 / aarch64)** see the section below — the default aarch64 torch wheel is CPU-only, so you need a few extra steps.

## Run

Start the server (first run downloads ~8 GB of weights):

```bash
vllm-omni serve mistralai/Voxtral-4B-TTS-2603 --omni \
  --stage-configs-path ./voxtral_tts_spark.yaml
```

The `--omni` flag routes through vllm-omni's multi-stage engine (TTS acoustic transformer + audio tokenizer). Without it, vllm loads the base Mistral LLM and rejects the TTS weights.

The stage config (`voxtral_tts_spark.yaml`) caps stage-0 memory at `0.5` so the two stages fit in the Spark's 120 GiB unified memory. Raise it back to `0.8` on a dedicated 80/141 GB GPU.

### Gradio UI

With the server running, start the Gradio front-end in a second shell:

```bash
python tts_gradio.py             # local only, http://localhost:7860
GRADIO_SHARE=1 python tts_gradio.py   # adds a public *.gradio.live URL (~72h)
```

Text input + voice dropdown (voices fetched live from `/v1/audio/voices`) + audio player.

### Command-line client

```bash
python tts_client.py "Paris is a beautiful city!" --voice casual_male --out paris.wav

# Generate one file per preset voice
python tts_client.py "Hello from Voxtral." --voice all
```

## DGX Spark (GB10 / aarch64 / CUDA 13) notes

The default `uv pip install vllm` on aarch64 pulls in CPU-only torch, so GPU inference silently breaks. Install GPU torch from the PyTorch index first, then vllm on top:

```bash
uv pip install torch==2.10.0+cu130 torchaudio==2.11.0+cu130 \
  --index-url https://download.pytorch.org/whl/cu130
uv pip install "vllm==0.18.*" "vllm-omni==0.18.*" \
  "mistral-common>=1.10.0" httpx soundfile gradio

# vllm's _C extension is compiled against CUDA 12 — shim it:
uv pip install nvidia-cuda-runtime-cu12 nvidia-cublas-cu12 \
  nvidia-cuda-nvrtc-cu12 nvidia-cusparse-cu12 nvidia-cusolver-cu12 \
  nvidia-cufft-cu12 nvidia-curand-cu12 nvidia-cudnn-cu12 nvidia-nccl-cu12

# Put the cu12 shim libs on LD_LIBRARY_PATH before launching vllm-omni:
SITE=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cufft/lib:$SITE/nvidia/curand/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/nccl/lib:$LD_LIBRARY_PATH"
```

torch will warn that GB10 is sm_121 (above its pre-built max of sm_120). That's fine — PTX JIT handles it.

## Files

```
tts_client.py           # CLI that calls /v1/audio/speech and writes a .wav
tts_gradio.py           # Gradio UI (text + voice dropdown + audio player)
voxtral_tts_spark.yaml  # stage config with Spark-friendly memory caps
requirements.txt        # vllm, vllm-omni, mistral-common, httpx, soundfile, gradio
```
