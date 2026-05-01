"""vLLM model-serving app for Gemma 4.

This defines the `vllm_app` AppEnvironment (importable by `chat_app.py`) and,
when run as `__main__`, prefetches the model from HF and deploys the server
to the Flyte 2 devbox. The vLLM server speaks the OpenAI-compatible API on
port 8080; `/v1/chat/completions`, `/docs`, etc.

Deploy:
    python vllm_server.py
    # or, for the dense 31B variant:
    GEMMA_VARIANT=31b python vllm_server.py
    or:
    flyte serve vllm_server.py vllm_app
"""

from __future__ import annotations

from flyteplugins.vllm import VLLMAppEnvironment

import flyte
import flyte.app

from config import MODEL


# `from_base()` returns a frozen dataclass with platform=linux/amd64 and
# clone() doesn't expose a platform kwarg — bypass the freeze to set arm64.
_base = flyte.Image.from_base("vllm/vllm-openai:gemma4-cu130")
object.__setattr__(_base, "platform", ("linux/arm64",))

image = (
    _base.clone(
        registry="localhost:30000",
        name="gemma4-vllm-image",
        extendable=True,
    )
    # Install into the base image's system Python (where vllm + torch already
    # live), not Flyte's /opt/venv — otherwise vllm-fserve can't import torch.
    .with_commands([
        "/usr/bin/python3 -m pip install --no-cache-dir --pre flyteplugins-vllm"
    ])
)

# model_hf_path is a placeholder; deploy below overrides it with model_path=RunOutput(...).
vllm_app = VLLMAppEnvironment(
    name=MODEL.app_name,
    image=image,
    model_hf_path=MODEL.hf_repo,
    model_id=MODEL.model_id,
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu=MODEL.gpu, disk="20Gi"),
    stream_model=True,
    scaling=flyte.app.Scaling(
        replicas=(0, 1),
        scaledown_after=1800,   # cold starts are ~6 min, so amortize over a generous idle window
    ),
    requires_auth=False,
    extra_args=[
        "--max-model-len", str(MODEL.max_model_len),
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.85",
    ],
)


if __name__ == "__main__":
    import os

    flyte.init_from_config()

    # GEMMA_PREFETCH_RUN=<run-name> skips the prefetch step and reuses a known-good run.
    existing_run = os.environ.get("GEMMA_PREFETCH_RUN")
    if existing_run:
        run_name = existing_run
        print(f"Reusing prefetched model from run: {run_name}")
    else:
        import flyte.prefetch
        from flyte.remote import Run

        print(f"Prefetching {MODEL.hf_repo}…")
        run: Run = flyte.prefetch.hf_model(repo=MODEL.hf_repo)
        run.wait()
        print(f"Prefetch run: {run.url}")
        run_name = run.name

    print(f"Deploying vLLM server for {MODEL.model_id} on {MODEL.gpu}…")
    app = flyte.serve(
        vllm_app.clone_with(
            name=vllm_app.name,
            model_path=flyte.app.RunOutput(type="directory", run_name=run_name),
            model_hf_path=None,
        )
    )
    print(f"vLLM app deployed: {app.url}")
    print(f"  OpenAI base URL: {app.url}/v1")
    print(f"  OpenAPI docs:    {app.url}/docs")
