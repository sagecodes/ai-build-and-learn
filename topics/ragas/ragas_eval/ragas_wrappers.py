from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings

from config import ANTHROPIC_API_KEY, EMBEDDING_MODEL, EVAL_LLM_MODEL


class _FastEmbedWrapper(Embeddings):
    """
    Thin LangChain-compatible wrapper around fastembed.TextEmbedding.
    Avoids the ONNX segfault from langchain_community.FastEmbedEmbeddings by
    explicitly specifying CPUExecutionProvider.
    fastembed is already present in the base image — no extra dependency needed.
    """

    def __init__(self, model_name: str) -> None:
        try:
            from fastembed import TextEmbedding
            self._model = TextEmbedding(
                model_name=model_name,
                providers=["CPUExecutionProvider"],
            )
        except Exception as e:
            raise RuntimeError(
                f"fastembed model load failed — check network and disk space. "
                f"Original error: {e}"
            ) from e

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [emb.tolist() for emb in self._model.embed(texts)]

    def embed_query(self, text: str) -> List[float]:
        return next(iter(self._model.embed([text]))).tolist()


def get_ragas_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=EVAL_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )


def get_ragas_embeddings() -> _FastEmbedWrapper:
    return _FastEmbedWrapper(model_name=EMBEDDING_MODEL)
