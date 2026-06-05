"""Flyte env + shared constants for the Ragas-on-Flyte eval demo.

DGX-Spark-pinned: aarch64 platform + devbox-local registry. Drop the pins if
you ever target a generic Flyte 2 cluster.

This demo evaluates the same RAG system the sibling
`topics/vectorstore/rag-chroma-flyte` project serves: same mini-wikipedia
corpus, same BGE encoder, same Chroma collection, same gemma4 vLLM app. The
eval pipeline can either reuse that project's Chroma artifact (set
RAG_PIPELINE_RUN) or build an identical index inline.
"""

from __future__ import annotations

import os

# `import ragas` pulls in GitPython, which hard-fails at import when there's no
# git binary in the container ("Bad git executable"). ragas only uses git for
# optional run tracking, so quiet the refresh. Set before any module imports
# ragas (config.py is imported first by both ragas_lib and eval_pipeline).
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

from dotenv import load_dotenv
import flyte

load_dotenv()

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# ── gemma4 vLLM sibling app ────────────────────────────────────────────────────
# Same app the rag-chroma-flyte / cognee projects talk to. Here it plays two
# roles: the RAG answerer (run_rag) and, by default, the Ragas LLM-as-judge.
# Change these two strings if you switched to the 31B variant.
VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"
VLLM_URL = (
    f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local"
)

# ── Dataset ─────────────────────────────────────────────────────────────────────
# rag-mini-wikipedia ships two configs: the passage corpus we index, and a
# question-answer set with real ground-truth answers. The QA set is the eval
# test set, so reference-based metrics (context recall, factual correctness,
# semantic similarity, noise sensitivity) have something to compare against.
DATASET_REPO = "rag-datasets/rag-mini-wikipedia"

CORPUS_CONFIG = "text-corpus"
CORPUS_SPLIT = "passages"
CORPUS_TEXT_COLUMN = "passage"

QA_CONFIG = "question-answer"
QA_SPLIT = "test"
QA_QUESTION_COLUMN = "question"
QA_ANSWER_COLUMN = "answer"

# ── RAG knobs (must match the sibling project so we evaluate the same system) ──
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "rag_demo"

# Sibling pipeline task name = pipeline_env.name + "." + function_name. Used to
# resolve a prior rag_pipeline run's Chroma Dir when RAG_PIPELINE_RUN is unset
# but you still want the latest succeeded build.
PIPELINE_TASK = "rag-chroma-pipeline.rag_pipeline"

# ── Image + task env ────────────────────────────────────────────────────────────
# ragas pinned to the stable 0.2/0.3 line (resolves to 0.3.9): in 0.4 the classic
# metric classes are deprecated out of `ragas.metrics`, and the evaluate() +
# EvaluationDataset API we use here is rock solid pre-0.4.
#
# The langchain stack is capped to the 0.3 generation: ragas 0.3.x eagerly
# imports `langchain_community.chat_models.vertexai`, a module LangChain 1.x
# deleted. Without the caps, uv pulls langchain-core 1.x and ragas fails to load.
RAGAS_PIP_PACKAGES = (
    "ragas>=0.2,<0.4",
    "langchain-core>=0.3,<0.4",
    "langchain-community>=0.3,<0.4",
    "langchain-openai>=0.2,<0.4",
    "langchain-huggingface>=0.1,<0.4",
    "sentence-transformers>=3.0.0",
    "chromadb>=0.5.0",
    # Force a modern opentelemetry: chromadb's loose lower bound otherwise lets
    # the resolver backtrack opentelemetry-proto to 1.11.1, whose generated
    # _pb2 is incompatible with modern protobuf and breaks `import chromadb`.
    "opentelemetry-proto>=1.27",
    "opentelemetry-exporter-otlp-proto-grpc>=1.27",
    "datasets>=3.0.0",
    "pandas",
)

eval_env = flyte.TaskEnvironment(
    name="ragas-eval",
    image=flyte.Image.from_debian_base(
        name="ragas-eval-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(*RAGAS_PIP_PACKAGES),
    # The judge metrics fan out a lot of concurrent LLM calls and we load BGE
    # plus a few langchain stacks in-process, so give it more headroom than the
    # sibling indexing pipeline.
    resources=flyte.Resources(cpu="4", memory="12Gi"),
)
