"""Neo4j 5 Community as a Flyte 2 app, exposing the HTTP Cypher API on 7474.

Why HTTP and not Bolt: Flyte 2 deploys apps as Knative Serving services. The
queue-proxy sidecar that fronts every Knative pod is HTTP-only, so Bolt
(TCP/7687) won't pass through. Neo4j's HTTP API on 7474 is fully featured
(any Cypher, including vector index queries):
    GET  /                    → discovery doc, 200 OK (probe-friendly)
    POST /db/neo4j/tx/commit  → run Cypher

Two image-build gotchas this avoids:
  - `flyte.Image.from_base("neo4j:...")` appends `USER flyte` + `WORKDIR /home/flyte`
    on top of the neo4j image. neo4j has no `flyte` user → containerd "no users
    found" at runtime. We use `from_dockerfile` instead, which skips the footer.
  - `pod_template` on AppEnvironment looks supported in the SDK but the Flyte 2.2.3
    server rejects it ("K8sPod app payload is not yet supported"). So we stay on
    the regular image-based path and just hand Flyte a plain neo4j image.

Deploy:
    python neo4j_app.py
"""

from __future__ import annotations

import pathlib

import flyte
import flyte.app

from config import (
    NEO4J_APP_NAME,
    NEO4J_HTTP_PORT,
    NEO4J_PASSWORD,
    NEO4J_USER,
    PLATFORM,
    REGISTRY,
)


# `from_dockerfile` is the only `flyte.Image` constructor that doesn't
# tack on the `USER flyte` + `WORKDIR /home/flyte` footer.
neo4j_image = flyte.Image.from_dockerfile(
    file=pathlib.Path(__file__).parent / "Dockerfile.neo4j",
    registry=REGISTRY,
    name="graphrag-neo4j-image",
    platform=PLATFORM,
)


neo4j_app = flyte.app.AppEnvironment(
    name=NEO4J_APP_NAME,
    image=neo4j_image,
    # Replace Flyte's default Python entrypoint with neo4j's own startup
    # script. CMD `neo4j` runs the daemon in the foreground (PID 1).
    command=["/startup/docker-entrypoint.sh", "neo4j"],
    port=flyte.app.Port(port=NEO4J_HTTP_PORT),
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    # Always-on. Cold start (~10s) would force every pipeline run to wait.
    scaling=flyte.app.Scaling(replicas=(1, 1)),
    requires_auth=False,
    env_vars={
        # Initial password set on first boot (NEO4J_AUTH=user/pass).
        "NEO4J_AUTH": f"{NEO4J_USER}/{NEO4J_PASSWORD}",
        # Listen on all pod interfaces so Knative can reach us.
        "NEO4J_server_default__listen__address": "0.0.0.0",
        "NEO4J_server_http_listen__address": f"0.0.0.0:{NEO4J_HTTP_PORT}",
        # Modest heap for the toy graph. Tune up for real datasets.
        "NEO4J_server_memory_heap_initial__size": "512m",
        "NEO4J_server_memory_heap_max__size": "1g",
        "NEO4J_server_memory_pagecache_size": "512m",
    },
)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    # Plain `flyte.serve` (not `with_servecontext(interactive_mode=True)`):
    # interactive mode forces a pkl bundle that requires `@app.server`. Our
    # app has no Python entrypoint to bundle. Same shape as the n8n example
    # in the Flyte SDK.
    app = flyte.serve(neo4j_app)
    print(f"Neo4j app deployed: {app.url}")
    print(f"  HTTP Cypher API: {app.url}/db/neo4j/tx/commit")
    print(f"  Browser UI:      {app.url}/browser/")
    print(f"  User: {NEO4J_USER}")
    print(f"  Password: {NEO4J_PASSWORD}")
