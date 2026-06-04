from langchain_anthropic import ChatAnthropic
from langchain_google_vertexai import VertexAIEmbeddings

from config import ANTHROPIC_API_KEY, EVAL_LLM_MODEL, GCP_PROJECT

# Google's text-embedding-004 — no ONNX, runs via Vertex AI REST API.
# VM has Application Default Credentials so no key file needed.
_VERTEX_EMBEDDING_MODEL = "text-embedding-004"


def get_ragas_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=EVAL_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )


def get_ragas_embeddings() -> VertexAIEmbeddings:
    return VertexAIEmbeddings(
        model_name=_VERTEX_EMBEDDING_MODEL,
        project=GCP_PROJECT,
    )
