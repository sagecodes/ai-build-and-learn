FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "gradio>=6.0.0" \
    "flyte>=2.1.2" \
    "python-dotenv>=1.0.0"
