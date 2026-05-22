"""Flyte env + shared constants for the Cognee-on-Flyte memory demos.

DGX-Spark-pinned: aarch64 platform + devbox-local registry. Drop the pins if
you ever target a generic Flyte 2 cluster.

Two surfaces share this module:
  - pipeline.py     the source-ingest agent (TaskEnvironment below)
  - chat_app.py     the Gradio chatbot (its own AppEnvironment, but reuses the
                    vLLM + storage constants here)

Cognee itself is configured in cognee_lib.configure_cognee(), not here, because
those env vars must be set inside the pod/task before `import cognee`.
"""

from __future__ import annotations

from dotenv import load_dotenv
import flyte

load_dotenv()

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# Same Gemma 4 vLLM sibling app as the llm-wiki / graphrag projects. Cognee
# talks to it as a custom OpenAI-compatible LLM (chat + structured extraction);
# the chat app streams from it directly. Change these two strings if you
# switched to the 31B variant.
VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"
VLLM_URL = (
    f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local"
)

# Cognee writes a relational DB (SQLite), a vector store (LanceDB), and a graph
# DB (Ladybug) under these two roots. We point them at a single working dir per
# task so the whole memory state is one tar-able subtree -> one flyte.io.Dir.
#   <root>/data     DATA_ROOT_DIRECTORY    (relational + vector files)
#   <root>/system   SYSTEM_ROOT_DIRECTORY  (graph + bookkeeping)
COGNEE_DATA_SUBDIR = "data"
COGNEE_SYSTEM_SUBDIR = "system"

# Embeddings run locally (no second endpoint beside the chat-only vLLM).
# fastembed ships onnxruntime wheels for arm64; bge-small matches the encoder
# the sibling chroma agent-memory demo uses, at 384 dims.
EMBEDDING_PROVIDER = "fastembed"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIMENSIONS = 384

# HF model repo for the chat app's scale-to-zero checkpoint (the "both"
# persistence story: pipeline uses flyte.io.Dir, chat app also tars here).
HF_MEMORY_REPO = "sagecodes/cognee-mem"
HF_MEMORY_REPO_TYPE = "model"
HF_MEMORY_FILENAME = "cognee_memory.tar.gz"

# Shared pip set so the pipeline image and chat image stay in lockstep.
COGNEE_PIP_PACKAGES = (
    "cognee>=0.1.40",
    "fastembed>=0.3.0",
    "openai>=1.50.0",
    "httpx>=0.27.0",
    "trafilatura>=1.12.0",
    "huggingface_hub>=0.24.0",
    "python-dotenv>=1.0.0",
)

pipeline_env = flyte.TaskEnvironment(
    name="cognee-memory-pipeline",
    image=flyte.Image.from_debian_base(
        name="cognee-memory-pipeline-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(*COGNEE_PIP_PACKAGES),
    # cognify does chunking + embedding + graph build; give it real headroom.
    resources=flyte.Resources(cpu="4", memory="8Gi"),
    secrets=[flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")],
)
