"""Shared config for the Gemma 4 chat app on Flyte 2.

Switch between the MoE 26B-A4B and the dense 31B by flipping `MODEL`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import flyte
import flyte.app


@dataclass(frozen=True)
class ModelChoice:
    hf_repo: str
    model_id: str          # name vLLM exposes over its OpenAI API
    app_name: str          # Flyte app name (DNS-safe, lowercase)
    gpu: int | str         # int = any GPU type; str like "H100:1" = typed
    max_model_len: int


GEMMA_4_26B_A4B = ModelChoice(
    hf_repo="google/gemma-4-26B-A4B-it",
    model_id="gemma-4-26b-a4b-it",
    app_name="gemma4-26b-a4b-it-vllm",
    gpu=1,
    max_model_len=32768,
)

GEMMA_4_31B = ModelChoice(
    hf_repo="google/gemma-4-31B-it",
    model_id="gemma-4-31b-it",
    app_name="gemma4-31b-it-vllm",
    gpu=1,                 # bump to 2 for TP=2 on a multi-GPU box
    max_model_len=8192,
)

MODEL = GEMMA_4_31B if os.environ.get("GEMMA_VARIANT") == "31b" else GEMMA_4_26B_A4B

# vLLM budgets `util * total` GPU memory and refuses to start unless that much is
# free. On the Spark's unified memory the OS holds ~20Gi of the 119.7Gi pool, so
# anything above ~0.83 can never be satisfied.
GPU_MEMORY_UTILIZATION = os.environ.get("GEMMA_GPU_MEM_UTIL", "0.75")

CHAT_APP_NAME = "gemma4-chat-ui"
