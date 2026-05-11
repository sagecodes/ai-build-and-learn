"""Prefetch a Gemma 4 model from Hugging Face into the Flyte object store.

Standalone helper — useful when you want to download/cache before deploying,
or to verify the model + HF_TOKEN secret work. The deploy in `vllm_server.py`
also calls this same prefetch (it's run-cached, so calling it twice is cheap).

Usage:
    flyte create secret HF_TOKEN  # one-time, paste your HF token
    python prefetch_model.py      # 26B-A4B by default
    GEMMA_VARIANT=31b python prefetch_model.py
"""

from __future__ import annotations

import flyte
import flyte.prefetch
from flyte.remote import Run

from config import MODEL


def prefetch() -> Run:
    flyte.init_from_config()
    print(f"Prefetching {MODEL.hf_repo} → Flyte object store…")
    run: Run = flyte.prefetch.hf_model(
        repo=MODEL.hf_repo,
        # Default `hf_token_key="HF_TOKEN"` — create the secret with
        #   flyte create secret HF_TOKEN
        # before running this.
    )
    run.wait()
    print(f"Prefetch run: {run.url}")
    print(f"Run name (use this in vllm_server.py if deploying separately): {run.name}")
    return run


if __name__ == "__main__":
    prefetch()
