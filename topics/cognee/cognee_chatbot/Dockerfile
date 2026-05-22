# Task image — used by TaskEnvironment pods on Union.
# No gradio; no COPY (tasks receive code via Flyte).

FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "flyte>=2.1.2" \
    "cognee[postgres-binary]>=0.1.0" \
    "anthropic>=0.40.0" \
    "fastembed>=0.8.0" \
    "pyvis>=0.3.2" \
    "python-dotenv>=1.0.0"
