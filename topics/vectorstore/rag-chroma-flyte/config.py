"""Flyte env for the RAG pipeline.

DGX-Spark-pinned: aarch64 platform + devbox-local registry. Drop the pins if
you ever target a generic Flyte 2 cluster.
"""

from __future__ import annotations

from dotenv import load_dotenv
import flyte

load_dotenv()

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

pipeline_env = flyte.TaskEnvironment(
    name="rag-chroma-pipeline",
    image=flyte.Image.from_debian_base(
        name="rag-pipeline-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(
        "datasets>=3.0.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
    ),
    resources=flyte.Resources(cpu="4", memory="8Gi"),
)
