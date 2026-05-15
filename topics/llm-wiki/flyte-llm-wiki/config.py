"""Flyte env + shared constants for the LLM Wiki pipeline.

DGX-Spark-pinned: aarch64 platform + devbox-local registry. Drop the pins if
you ever target a generic Flyte 2 cluster.
"""

from __future__ import annotations

from dotenv import load_dotenv
import flyte

load_dotenv()

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# Same Gemma 4 vLLM sibling app as the graphrag-neo4j-flyte project. The
# pipeline + chat app both hit it over the cluster-internal Knative URL.
# Change these two strings if you switched to the 31B variant.
VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"
VLLM_URL = (
    f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local"
)

# Wiki layout inside the flyte.io.Dir. The pipeline owns this layout; the
# chat app mirrors it on local disk inside its pod.
WIKI_PAGES_DIR = "pages"
WIKI_RAW_DIR = "raw"
WIKI_INDEX_FILE = "index.md"
WIKI_LOG_FILE = "log.md"
WIKI_SCHEMA_FILE = "AGENTS.md"

pipeline_env = flyte.TaskEnvironment(
    name="llm-wiki-pipeline",
    image=flyte.Image.from_debian_base(
        name="llm-wiki-pipeline-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(
        "openai>=1.50.0",
        "httpx>=0.27.0",
        "trafilatura>=1.12.0",
        "python-dotenv>=1.0.0",
    ),
    resources=flyte.Resources(cpu="2", memory="4Gi"),
)
