# App serving image — used by AppEnvironment on Union.
# Includes gradio and the full project source.

FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "flyte>=2.1.2" \
    "union>=0.1.194" \
    "cognee[postgres-binary]>=0.1.0" \
    "anthropic>=0.40.0" \
    "gradio>=4.44.0" \
    "python-dotenv>=1.0.0"

WORKDIR /app
COPY . /app

EXPOSE 7860
CMD ["python", "app.py"]
