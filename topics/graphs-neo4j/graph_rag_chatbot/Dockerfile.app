FROM python:3.11-slim

RUN pip install --no-cache-dir \
    "flyte>=2.1.2" \
    "union>=0.1.194" \
    "neo4j>=5.20.0" \
    "sentence-transformers>=3.0.0" \
    "anthropic>=0.40.0" \
    "PyMuPDF>=1.24.0" \
    "langchain-text-splitters>=0.3.0" \
    "networkx>=3.3" \
    "python-louvain>=0.16" \
    "numpy>=1.26.0" \
    "python-dotenv>=1.0.0" \
    "gradio>=4.44.0"

# Pre-download embedding model so the app pod doesn't re-fetch it on startup
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('thenlper/gte-small')"

WORKDIR /app
COPY . /app

EXPOSE 7860
CMD ["python", "app.py"]
