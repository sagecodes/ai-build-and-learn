"""Self-host an MLflow tracking server as a Flyte app.

Serves the MLflow UI and tracking API on MLFLOW_PORT (5000). Tasks log
experiments, metrics, and artifacts here. SQLite-backed (session-scoped);
for production, point at Postgres.

Deploy:
    python mlflow_app.py
"""

from __future__ import annotations

import pathlib

import flyte
import flyte.app

from config import MLFLOW_APP_NAME, MLFLOW_PORT, PLATFORM, REGISTRY

WORKING_DIR = "/tmp/mlflow"

mlflow_image = flyte.Image.from_debian_base(
    name="mlflow-server-image",
    registry=REGISTRY,
    platform=PLATFORM,
).with_pip_packages(
    "mlflow>=3.1",
)

env = flyte.app.AppEnvironment(
    name=MLFLOW_APP_NAME,
    image=mlflow_image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    port=MLFLOW_PORT,
    requires_auth=False,
    env_vars={
        "MLFLOW_HOST": "0.0.0.0",
        "MLFLOW_PORT": str(MLFLOW_PORT),
    },
    scaling=flyte.app.Scaling(replicas=(1, 1)),
)


@env.server
def mlflow_server():
    """Launch `mlflow server` with SQLite backend."""
    import os
    import subprocess

    os.makedirs(WORKING_DIR, exist_ok=True)
    print(f"[mlflow] serving UI + tracking API on :{MLFLOW_PORT} (store={WORKING_DIR})", flush=True)
    subprocess.run(
        [
            "mlflow", "server",
            "--host", "0.0.0.0",
            "--port", str(MLFLOW_PORT),
            "--backend-store-uri", f"sqlite:///{WORKING_DIR}/mlflow.db",
            "--default-artifact-root", f"{WORKING_DIR}/artifacts",
            "--allowed-hosts", "*",
            "--cors-allowed-origins", "*",
        ],
        check=True,
    )


if __name__ == "__main__":
    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"MLflow server deployed: {app.url}")
