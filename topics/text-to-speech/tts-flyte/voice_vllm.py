"""The LLM half of the voice app's `vllm` deployment: a vLLM server on Flyte.

Deliberately a near-copy of `topics/gemma4/gemma4-dgx-devbox/vllm_server.py`, because
that pattern already works on this box: the frozen-dataclass arm64 fix, installing the
plugin into the base image's system Python, `flyte.prefetch.hf_model` + `RunOutput` so
the weights are fetched once instead of on every cold start.

Deploy:
    python voice_vllm.py
    # then point the voice app at the URL it prints:
    LLM_BACKEND=vllm VLLM_URL=<url> python voice_app.py

── Why a QUANTIZED checkpoint here and not bf16 ─────────────────────────────────
The Spark is memory-bandwidth-bound, so tokens/sec tracks bytes-read-per-token, and a
voice assistant has to stay ahead of speech (~3 tok/s of talking) with margin. The
gemma4 devbox app serves bf16 (~52GB), which is the right call for batch throughput and
the wrong one here. NVFP4 puts the same 26B-A4B at ~16.5GB: smaller than ollama's Q4
(18GB) and on vLLM's kernels, so this should be the fastest option available on the box.

RISK, untested at the time of writing: NVFP4 kernels want Blackwell, and while the GB10
IS Blackwell, it is sm_121 rather than the B200's sm_100. This repo has been bitten by
sm_121 gaps repeatedly (CUDA-graph capture hangs, torch.compile failures, the whole
voxtral deferral). If NVFP4 will not load, fall back to FP8 and then to bf16 by flipping
CHECKPOINT below; the voice app does not care which one is behind the endpoint.
"""

from __future__ import annotations

import os

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

# Ordered best-to-safest. NVFP4 is the smallest and should be fastest; FP8 is the
# conservative Blackwell choice; the bf16 base always loads but reads ~3x more per token.
CHECKPOINTS = {
    "nvfp4": ("nvidia/Gemma-4-26B-A4B-NVFP4", "gemma-4-26b-a4b-nvfp4"),
    "fp8":   ("RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic", "gemma-4-26b-a4b-fp8"),
    "bf16":  ("google/gemma-4-26B-A4B-it", "gemma-4-26b-a4b-it"),
}
CHECKPOINT = os.environ.get("VOICE_LLM_QUANT", "nvfp4")
HF_REPO, MODEL_ID = CHECKPOINTS[CHECKPOINT]

APP_NAME = f"tts-voice-llm-{CHECKPOINT}"

# vLLM budgets `util * total` of the pool and refuses to start unless that much is free.
# The OS holds ~20Gi of the 119.7Gi, so anything above ~0.83 can never be satisfied; 0.75
# is the value the gemma4 app settled on and the same one the GB10 memory notes recommend.
GPU_MEMORY_UTILIZATION = os.environ.get("VOICE_GPU_MEM_UTIL", "0.75")

# from_base() returns a FROZEN dataclass pinned to linux/amd64 and clone() exposes no
# platform kwarg, so the freeze is bypassed to set arm64. Straight from vllm_server.py.
_base = flyte.Image.from_base("vllm/vllm-openai:gemma4-cu130")
object.__setattr__(_base, "platform", ("linux/arm64",))

image = (
    _base.clone(registry="localhost:30000", name="tts-voice-vllm-image", extendable=True)
    # Into the base image's system Python, where vllm + torch already live: installing
    # into Flyte's /opt/venv leaves vllm-fserve unable to import torch.
    .with_commands([
        "/usr/bin/python3 -m pip install --no-cache-dir --pre flyteplugins-vllm"
    ])
)

vllm_app = VLLMAppEnvironment(
    name=APP_NAME,
    image=image,
    model_hf_path=HF_REPO,      # replaced by model_path=RunOutput(...) on deploy
    model_id=MODEL_ID,
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu=1, disk="40Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        # Cold starts are minutes, and this app holds the box's only GPU while up, so the
        # window is a balance: long enough that a conversation doesn't pay a reload,
        # short enough that the compare/clone pipelines get the GPU back.
        scaledown_after=1800,
    ),
    requires_auth=False,
    extra_args=[
        # Voice replies are short; a big context just costs KV cache we would rather
        # leave free on a shared pool.
        "--max-model-len", "8192",
        "--trust-remote-code",
        "--gpu-memory-utilization", GPU_MEMORY_UTILIZATION,
        # sm_121 hangs during CUDA-graph capture on this box (same failure that deferred
        # voxtral). Eager skips capture at a small throughput cost.
        "--enforce-eager",
    ],
)


if __name__ == "__main__":
    flyte.init_from_config()

    # VOICE_PREFETCH_RUN=<run-name> reuses a known-good prefetch instead of re-fetching.
    if existing := os.environ.get("VOICE_PREFETCH_RUN"):
        run_name = existing
        print(f"Reusing prefetched weights from run: {run_name}")
    else:
        import flyte.prefetch
        from flyte.remote import Run

        print(f"Prefetching {HF_REPO} …")
        run: Run = flyte.prefetch.hf_model(repo=HF_REPO)
        run.wait()
        print(f"Prefetch run: {run.url}")
        run_name = run.name

    print(f"Deploying vLLM for {MODEL_ID} ({CHECKPOINT}) …")
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
            model_hf_path=None,
        )
    )
    print(f"vLLM app deployed: {app.url}")
    print()
    print("Point the voice app at it:")
    print(f"  LLM_BACKEND=vllm VLLM_URL={app.url} VLLM_MODEL_ID={MODEL_ID} python voice_app.py")
