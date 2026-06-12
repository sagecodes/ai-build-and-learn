"""Flyte env + shared constants for the Phoenix-on-Flyte tracing demo.

DGX-Spark-pinned: aarch64 platform + devbox-local registry. Drop the pins if
you ever target a generic Flyte 2 cluster.

Two surfaces share this module:
  - workflow.py     the traced LangGraph + Tavily agent (TaskEnvironment below)
  - phoenix_app.py  the self-hosted Phoenix server/UI (its own AppEnvironment,
                    but reuses the collector-endpoint + vLLM constants here)

The agent sends OTLP spans to the Phoenix app over the cluster-internal Knative
DNS name; the app serves the UI and the OTLP-HTTP collector on the same port.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
import flyte

load_dotenv()

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# ── Phoenix server (self-hosted as a Flyte app) ────────────────────────────────
# One `phoenix serve` process exposes BOTH the UI and the OTLP-HTTP collector on
# PHOENIX_PORT. Knative routes a single HTTP port cleanly, so the agent ships
# spans over OTLP-HTTP to this same port (no separate gRPC route needed).
PHOENIX_APP_NAME = "phoenix-server"
PHOENIX_PORT = 6006

# Cluster-internal address the agent task uses to reach the collector. Same
# Knative DNS shape the cognee/ragas demos use for the vLLM app; the app's
# port 6006 is fronted by the Knative service on port 80, so no port suffix.
PHOENIX_COLLECTOR_ENDPOINT = (
    f"http://{PHOENIX_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local"
)

# Phoenix UI groups traces by project. Every pipeline task tags its spans with
# this, so plan/research/synthesize/quality traces land together.
PHOENIX_PROJECT_NAME = "research-pipeline"

# The Phoenix REST base URL (used by the eval task's phoenix.client to pull spans
# and write annotations). Same host as the OTLP collector: Knative fronts the
# app's port 6006 on port 80, so the base URL is the bare DNS name, no path.
PHOENIX_BASE_URL = PHOENIX_COLLECTOR_ENDPOINT

# ── gemma4 vLLM sibling app (the LLM_PROVIDER=vllm path) ────────────────────────
# Same in-cluster app the rag-chroma-flyte / cognee / ragas demos talk to. It is
# OpenAI-compatible, so the agent reaches it through ChatOpenAI with a base_url.
# Change these two strings if you switched to the 31B variant.
VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"
VLLM_URL = (
    f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local"
)

# ── Agent LLM ───────────────────────────────────────────────────────────────────
# Default is OpenAI (cloud), exactly like the sibling tavily demo. Flip to the
# in-cluster gemma4 vLLM with LLM_PROVIDER=vllm (no API key, fully OSS path).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")
OPENAI_MODEL = "gpt-4.1-nano"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# ── Shared pip set so the agent task image stays in lockstep ───────────────────
# Tracing client side: arize-phoenix-otel (the light exporter wrapper, NOT the
# full arize-phoenix platform) plus the OpenInference LangChain instrumentor,
# which auto-traces LangGraph too (LangGraph runs on LangChain runnables).
AGENT_PIP_PACKAGES = (
    "langgraph>=1.0.7",
    "langchain-openai",
    "tavily-python",
    "arize-phoenix-otel>=0.16",
    "openinference-instrumentation-langchain>=0.1.66",
    "markdown",            # Flyte report HTML
    "python-dotenv",
    "unionai-reuse",
)

agent_env = flyte.TaskEnvironment(
    name="phoenix-agent",
    image=flyte.Image.from_debian_base(
        name="phoenix-agent-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(*AGENT_PIP_PACKAGES),
    secrets=[
        flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY"),
        flyte.Secret(key="TAVILY_API_KEY", as_env_var="TAVILY_API_KEY"),
    ],
    # Point the OpenInference exporter at the hosted Phoenix collector. Set here
    # so it lands in the pod env before workflow.py calls register().
    env_vars={"PHOENIX_COLLECTOR_ENDPOINT": PHOENIX_COLLECTOR_ENDPOINT},
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)


# ── Eval task (LLM-as-a-judge over captured traces) ────────────────────────────
# A separate, lean image: it talks to Phoenix over REST (pull spans, write
# annotations) and calls an LLM judge. It does NOT need langgraph/tavily.
EVAL_PIP_PACKAGES = (
    "arize-phoenix-client>=2",
    "arize-phoenix-evals>=3",     # the post-v14 evals API (create_classifier, evaluate_dataframe)
    "openai>=1.50.0",
    "pandas",
    "python-dotenv",
    "unionai-reuse",
)

eval_env = flyte.TaskEnvironment(
    name="phoenix-eval",
    image=flyte.Image.from_debian_base(
        name="phoenix-eval-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(*EVAL_PIP_PACKAGES),
    # The judge defaults to OpenAI; the vLLM judge path needs no key.
    secrets=[flyte.Secret(key="OPENAI_API_KEY", as_env_var="OPENAI_API_KEY")],
    resources=flyte.Resources(cpu=2, memory="2Gi"),
)
