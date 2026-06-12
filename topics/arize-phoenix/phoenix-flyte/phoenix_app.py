"""Self-host the Arize Phoenix server as a Flyte app.

One `phoenix serve` process exposes BOTH surfaces on PHOENIX_PORT (6006):
  - the Phoenix UI (traces, projects, evals)
  - the OTLP-HTTP collector at POST /v1/traces

Knative fronts that single port, so the agent task (workflow.py) ships spans to
this app over plain HTTP at its cluster-internal DNS name. No separate gRPC
route, no sidecar collector.

Persistence (demo-grade): SQLite under PHOENIX_WORKING_DIR, with the app pinned
to a single always-on replica so traces survive for the whole session. The pod's
/tmp is wiped on restart, so this is session-scoped, not durable. For production,
set PHOENIX_SQL_DATABASE_URL to a Postgres instance and the app becomes stateless
(see the README).

Deploy:
    python phoenix_app.py
    # prints the app URL; open it to watch traces land live.
"""

from __future__ import annotations

import pathlib

import flyte
import flyte.app

from config import PHOENIX_APP_NAME, PHOENIX_PORT, PLATFORM, REGISTRY

# Phoenix's SQLite DB + working files live here. A real, writable path inside the
# pod; session-scoped because the Knative pod's /tmp resets on restart.
WORKING_DIR = "/tmp/phoenix"


phoenix_image = flyte.Image.from_debian_base(
    name="phoenix-server-image",
    registry=REGISTRY,
    platform=PLATFORM,
).with_pip_packages(
    # The full platform (server, UI, collector, evals). Ships the `phoenix` CLI.
    # Only the SERVER image needs this; the agent image uses arize-phoenix-otel.
    "arize-phoenix>=17,<18",
)


env = flyte.app.AppEnvironment(
    name=PHOENIX_APP_NAME,
    image=phoenix_image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    port=PHOENIX_PORT,
    requires_auth=False,
    env_vars={
        "PHOENIX_HOST": "0.0.0.0",
        "PHOENIX_PORT": str(PHOENIX_PORT),
        "PHOENIX_WORKING_DIR": WORKING_DIR,
    },
    # A collector must stay reachable for the agent to ship spans, so pin one
    # always-on replica (no scale-to-zero) instead of the chat apps' (0, 1).
    scaling=flyte.app.Scaling(replicas=(1, 1)),
)


@env.server
def phoenix_server():
    """Launch `phoenix serve`; it serves the UI and the OTLP collector on PHOENIX_PORT."""
    import os
    import subprocess

    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"[phoenix] serving UI + OTLP collector on :{PHOENIX_PORT} (store={WORKING_DIR})", flush=True)
    # Blocks for the life of the pod; keeps the app process alive.
    subprocess.run(["phoenix", "serve"], check=True)


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Phoenix server deployed: {app.url}")
