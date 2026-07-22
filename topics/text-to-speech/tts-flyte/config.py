"""Shared Flyte config for the open-source text-to-speech comparison demo.

Same shape as the video-generation demo next door (topics/video-generation), so
if you've read that one you already know the layout:

  - compare_pipeline.py : GPU *tasks* that read one script across TTS models and
    emit a single side-by-side Flyte report whose clips play inline.
  - app.py              : a thin CPU Gradio *app* that launches runs and links the
    report. It holds no GPU and loads no model.

DGX-Spark-pinned (GB10 Blackwell, arm64, cu130 stack): aarch64 platform + the
devbox-local registry. Drop the pins for a generic Flyte 2 cluster.

── The one big difference from the video demo: PER-MODEL IMAGES ─────────────────
The video demo ran every model on ONE image because they all load through the
same `diffusers`. TTS does not work that way. Each open model ships its own
inference package with its own, mutually hostile, hard pins:

  qwen-tts      pins  transformers==4.57.3, accelerate==1.12.0
  chatterbox    pins  transformers==5.2.0, torch==2.6.0, diffusers==0.29.0, numpy<2
  kokoro        wants misaki + espeak-ng, a recent transformers
  dia (via HF)  wants a recent transformers with the Dia architecture

You cannot resolve those into one environment. So every adapter gets its OWN
image and its OWN GPU TaskEnvironment, and the orchestrator dispatches each model
to the matching task. The two Qwen models share the `qwen` image (same package),
which is why the images are keyed by *adapter*, not by model.

Chatterbox is the awkward one: its `torch==2.6.0` pin has no cu130 arm64 wheel, so
we install `chatterbox-tts` with `--no-deps` on top of the Spark's cu130 torch and
bring its real runtime deps in by hand. That's the `extra_args="--no-deps"` below.
"""

from __future__ import annotations

import flyte
from kubernetes.client import V1Container, V1PodSpec, V1ResourceRequirements

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# GB10 is Blackwell on the cu130 stack; PyTorch publishes matching aarch64 wheels.
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"

APP_NAME = "tts-studio"
APP_PORT = 7864          # 7862 image-gen, 7863 videogen; don't collide

HF_HOME = "/tmp/hf"
HF_SECRET = flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")


# ── Deps every image needs ───────────────────────────────────────────────────────
#
# soundfile writes the wavs; matplotlib renders the waveform + spectrogram PNG that
# is the visual comparison surface (the audio analogue of the video demo's frame
# strip); the hf/kubernetes bits are for the download task and the pod templates.
# We deliberately do NOT put librosa here: matplotlib's own specgram is enough for
# the report and keeps the images lighter.
_COMMON = (
    "soundfile",
    "numpy",
    "matplotlib",
    "huggingface_hub",
    "hf_transfer",
    "kubernetes",     # config.py imports kubernetes.client at module top
)


def _base(name: str) -> flyte.Image:
    """Debian base + cu130 torch/torchaudio + the common report deps.

    ffmpeg for the system codecs; espeak-ng because Kokoro's G2P (misaki) shells out
    to it and a missing espeak-ng is a *runtime* failure, not an import one; git
    because a couple of TTS repos vendor code via git.
    """
    return (
        flyte.Image.from_debian_base(name=name, registry=REGISTRY, platform=PLATFORM)
        .with_apt_packages("git", "ffmpeg", "espeak-ng")
        .with_pip_packages("torch", "torchaudio", index_url=TORCH_INDEX)
        .with_pip_packages(*_COMMON)
    )


# ── Per-adapter images ────────────────────────────────────────────────────────────

# Qwen3-TTS. The package pulls its own pinned transformers/accelerate; let it.
qwen_image = _base("tts-qwen").with_pip_packages("qwen-tts")

# Kokoro. misaki[en] is the English G2P; espeak-ng (apt, above) is its fallback.
kokoro_image = _base("tts-kokoro").with_pip_packages("kokoro>=0.9.4", "misaki[en]>=0.9.4")

# Chatterbox. Install the package WITHOUT its deps (its torch==2.6.0 pin has no cu130
# arm64 wheel and would break the build), then bring its actual runtime deps in on
# top of the cu130 torch already in the base. transformers/diffusers versions are the
# ones chatterbox pins; they're isolated to this image so they can't fight Qwen's.
chatterbox_image = (
    _base("tts-chatterbox")
    .with_pip_packages(
        "transformers==5.2.0", "diffusers==0.29.0", "numpy<2.0.0",
        "librosa==0.11.0", "s3tokenizer", "resemble-perth", "conformer==0.3.2",
        "safetensors", "omegaconf", "pyloudnorm", "einops",
    )
    .with_pip_packages("chatterbox-tts", extra_args="--no-deps")
)

# Dia AND Sesame CSM both load through transformers-native classes
# (DiaForConditionalGeneration, CsmForConditionalGeneration), and 4.57+ carries both
# architectures plus the DAC/Mimi audio decoders. So they SHARE one image: same object
# => Flyte builds it once and both GPU envs pull it.
transformers_image = _base("tts-transformers").with_pip_packages("transformers>=4.57.3", "accelerate")

# Parler-TTS pins transformers==4.46.1 (older than everyone else here), so it needs its
# own image; the package pulls that transformers plus its descript-audio-codec deps.
# Two dep fights to win here, both isolated to this image:
#  1. The numba/llvmlite floor is load-bearing: without it uv backtracks through
#     Parler's Descript audio deps to the ancient llvmlite==0.36.0, which has NO wheel
#     and fails to compile on Python 3.12. Flooring both picks llvmlite>=0.48 (prebuilt
#     aarch64 wheels), so the image builds.
#  2. Parler's descript-audiotools-unofficial pins protobuf<5 (only for its tensorboard
#     logging, which inference never touches), but the Flyte runtime injected into every
#     task image needs protobuf>=6.30.1 to serialize a task's return value. Without the
#     bump the task RUNS but fails converting its ModelRun output ("Struct has no 'type'
#     field"). A FINAL, separate layer forces protobuf back up (last write wins), which
#     is safe because the synth path (ParlerTTS + the DAC codec) never imports tensorboard.
parler_image = (
    _base("tts-parler")
    .with_pip_packages("numba>=0.60", "llvmlite>=0.43", "parler-tts")
    .with_pip_packages("protobuf>=6.30.1")
)


# Voxtral (Mistral). The ONLY served model: it runs as a vLLM-omni server the task
# talks to over HTTP, not a from_pretrained load. Its image is the gnarliest, straight
# from the voxtral/ README's hard-won Spark recipe:
#   - torch pinned to 2.10.0+cu130 (what vllm 0.18 wants), not the floating cu130 torch;
#   - vllm + vllm-omni provide the --omni TTS engine;
#   - vllm's _C extension is compiled against CUDA 12, so the cu12 runtime libs are
#     installed as a shim and put on LD_LIBRARY_PATH (set in the env below) before the
#     server launches, or it fails to load on the cu130 box.
_VOXTRAL_SITE = "/opt/venv/lib/python3.12/site-packages"
_VOXTRAL_CU12 = (
    "nvidia-cuda-runtime-cu12", "nvidia-cublas-cu12", "nvidia-cuda-nvrtc-cu12",
    "nvidia-cusparse-cu12", "nvidia-cusolver-cu12", "nvidia-cufft-cu12",
    "nvidia-curand-cu12", "nvidia-cudnn-cu12", "nvidia-nccl-cu12",
)
_VOXTRAL_LD = ":".join(
    f"{_VOXTRAL_SITE}/nvidia/{d}/lib" for d in
    ("cuda_runtime", "cublas", "cudnn", "cuda_nvrtc", "cufft", "curand", "cusolver", "cusparse", "nccl")
)
voxtral_image = (
    flyte.Image.from_debian_base(name="tts-voxtral", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("git", "ffmpeg")
    .with_pip_packages("torch==2.10.0", "torchaudio==2.11.0", index_url=TORCH_INDEX)
    .with_pip_packages("vllm==0.18.*", "vllm-omni==0.18.*", "mistral-common>=1.10.0", "httpx")
    .with_pip_packages(*_VOXTRAL_CU12)
    .with_pip_packages(*_COMMON)
)


ADAPTER_IMAGES: dict[str, flyte.Image] = {
    "qwen": qwen_image,
    "kokoro": kokoro_image,
    "chatterbox": chatterbox_image,
    "dia": transformers_image,
    "csm": transformers_image,
    "parler": parler_image,
    "voxtral": voxtral_image,
}

# Per-adapter env extras and resource overrides (most adapters need neither).
_ADAPTER_ENV_EXTRA: dict[str, dict[str, str]] = {
    "voxtral": {"LD_LIBRARY_PATH": _VOXTRAL_LD, "VLLM_WORKER_MULTIPROC_METHOD": "spawn"},
}
_ADAPTER_MEM: dict[str, str] = {"voxtral": "64Gi"}  # two vLLM stages + server overhead

# The download task is model-agnostic: it only needs huggingface_hub, so it rides the
# lightest image (kokoro's happens to be small and dep-light).
fetch_image = _base("tts-fetch")

# The studio app is a thin LAUNCHER: it submits runs and links the report, so it needs
# no torch and no TTS package, just flyte + gradio + the registry for the model picker.
# connectrpc pinned to 0.10.x: 0.11 breaks flyte 2.2.1 runs ('Headers' not callable).
studio_app_image = (
    flyte.Image.from_debian_base(
        name="tts-studio-image", registry=REGISTRY, platform=PLATFORM
    )
    .with_pip_packages("flyte==2.2.1", "connectrpc==0.10.*", "gradio==5.42.0", "python-dotenv")
)


# ── DGX Spark tuning ────────────────────────────────────────────────────────────
#
# TTS models are tiny next to video (the biggest here, Qwen 1.7B, is ~4.5GB), so the
# memory-pressure knobs the video demo needed are mostly irrelevant. We keep the JIT
# cache sizing (first-call kernel compiles still happen) and the expandable allocator
# (cheap, and the GB10's "GPU memory" is the same unified pool the OS shares), and
# skip the rest.
_SPARK_ENV = {
    "CUDA_CACHE_MAXSIZE": "2147483648",
    "PYTORCH_ALLOC_CONF": "expandable_segments:True",
    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    "CUDA_MODULE_LOADING": "EAGER",
}

_ENV_VARS = {"HF_HOME": HF_HOME, "HF_HUB_ENABLE_HF_TRANSFER": "1"}

# The fetch task turns hf_transfer OFF on purpose: it's a Rust downloader that does its
# own DNS and bypasses the timeout knobs, so a stalled socket hangs forever. The plain
# Python downloader honors HF_HUB_DOWNLOAD_TIMEOUT (a per-read bound), so a hung read
# fails in ~60s and snapshot_download resumes from the .incomplete file.
_FETCH_ENV_VARS = {**_ENV_VARS, "HF_HUB_ENABLE_HF_TRANSFER": "0", "HF_HUB_DOWNLOAD_TIMEOUT": "60"}

_GPU_ENV_VARS = {**_ENV_VARS, **_SPARK_ENV}


# ── Environments ────────────────────────────────────────────────────────────────
#
#   cpu_env    (tts-fetch)      : fetch_weights, an HF download, no GPU.
#   GPU_ENVS   (tts-gen-<adapter>) : one GPU env per adapter/image; the synth work.
#   orch_env   (tts-orch)       : the orchestrator. CPU-only ON PURPOSE, because an
#                                 orchestrator pod stays alive holding its resources
#                                 while awaiting children, so if it held the box's one
#                                 GPU its own GPU children would deadlock forever.

cpu_env = flyte.TaskEnvironment(
    name="tts-fetch",
    image=fetch_image,
    resources=flyte.Resources(cpu="4", memory="8Gi", disk="40Gi"),
    secrets=[HF_SECRET],
    env_vars=_FETCH_ENV_VARS,
)


def _gpu_env(adapter: str) -> flyte.TaskEnvironment:
    return flyte.TaskEnvironment(
        name=f"tts-gen-{adapter}",
        image=ADAPTER_IMAGES[adapter],
        resources=flyte.Resources(cpu="8", memory=_ADAPTER_MEM.get(adapter, "32Gi"), gpu=1, disk="60Gi"),
        secrets=[HF_SECRET],
        env_vars={**_GPU_ENV_VARS, **_ADAPTER_ENV_EXTRA.get(adapter, {})},
    )


GPU_ENVS: dict[str, flyte.TaskEnvironment] = {a: _gpu_env(a) for a in ADAPTER_IMAGES}

orch_env = flyte.TaskEnvironment(
    name="tts-orch",
    image=fetch_image,
    resources=flyte.Resources(cpu="2", memory="4Gi", disk="20Gi"),
    secrets=[HF_SECRET],
    env_vars=_ENV_VARS,
    depends_on=[cpu_env, *GPU_ENVS.values()],
)

# Kept for parity with the video demo's in-pod experiments; unused while the studio is
# a pure launcher. AppEnvironment does NOT honor flyte.Resources(gpu=1) on this SDK
# (it serializes to a bare `gpu` key k8s drops and the pod silently schedules CPU-only);
# a PodTemplate that sets nvidia.com/gpu directly is the fix.
app_gpu_pod = flyte.PodTemplate(
    primary_container_name="app",
    pod_spec=V1PodSpec(
        containers=[
            V1Container(
                name="app",
                resources=V1ResourceRequirements(
                    requests={"cpu": "8", "memory": "32Gi", "ephemeral-storage": "60Gi"},
                    limits={"nvidia.com/gpu": "1"},
                ),
            )
        ]
    ),
)
