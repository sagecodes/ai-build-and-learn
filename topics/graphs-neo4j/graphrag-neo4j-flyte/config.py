"""Flyte env for the Graph RAG pipeline.

DGX-Spark-pinned: aarch64 platform + devbox-local registry. Drop the pins if
you ever target a generic Flyte 2 cluster.
"""

from __future__ import annotations

from dotenv import load_dotenv
import flyte

load_dotenv()

PLATFORM = ("linux/arm64",)
REGISTRY = "localhost:30000"

# Neo4j is deployed as a Flyte AppEnvironment (see neo4j_app.py). Knative
# Serving fronts every Flyte app and only routes HTTP, so we use Neo4j's
# HTTP Cypher API on 7474, not Bolt on 7687.
NEO4J_APP_NAME = "graphrag-neo4j"
NEO4J_HTTP_PORT = 7474
# Cluster-internal URL — Knative service in the `flyte` namespace, named
# `<app>-<project>-<domain>`. Pipeline pods reach it on port 80 (Knative
# always exposes the user port behind 80 on the cluster service).
NEO4J_HTTP_URL = (
    f"http://{NEO4J_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local"
)
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "graphrag-demo"

pipeline_env = flyte.TaskEnvironment(
    name="graphrag-neo4j-pipeline",
    image=flyte.Image.from_debian_base(
        name="graphrag-neo4j-pipeline-image",
        registry=REGISTRY,
        platform=PLATFORM,
    ).with_pip_packages(
        "httpx>=0.27.0",
        "sentence-transformers>=3.0.0",
        "python-dotenv>=1.0.0",
    ),
    resources=flyte.Resources(cpu="2", memory="4Gi"),
)
