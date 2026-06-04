import json
import os
import time
import urllib.request
from typing import List

from langchain_anthropic import ChatAnthropic
from langchain_core.embeddings import Embeddings

from config import ANTHROPIC_API_KEY, EVAL_LLM_MODEL

_HF_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_HF_URL   = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{_HF_MODEL}"


class _HuggingFaceInferenceEmbeddings(Embeddings):
    """
    Real neural embeddings via the HuggingFace Inference API (pure HTTP).
    No ONNX, no gRPC, no native binaries — works on any CPU.

    Set HF_TOKEN in .env for higher rate limits (free tier, login at
    huggingface.co). Without a token the free tier allows ~100 calls/day,
    which is tight for batch eval. With a token it's several thousand.
    """

    def __init__(self) -> None:
        token = os.environ.get("HF_TOKEN", "")
        self._headers = {
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        }

    def _post(self, texts: List[str], retries: int = 3) -> List[List[float]]:
        body = json.dumps({
            "inputs": texts,
            "options": {"wait_for_model": True},
        }).encode()

        for attempt in range(retries):
            req = urllib.request.Request(_HF_URL, data=body, headers=self._headers)
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    result = json.loads(resp.read())
                    # Single text returns a flat list; wrap it
                    if result and isinstance(result[0], float):
                        return [result]
                    return result
            except urllib.error.HTTPError as e:
                if e.code == 503 and attempt < retries - 1:
                    # Model loading — wait and retry
                    time.sleep(10 * (attempt + 1))
                    continue
                raise RuntimeError(
                    f"HuggingFace Inference API error {e.code}: {e.read().decode()}"
                ) from e

        raise RuntimeError("HuggingFace Inference API failed after retries")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._post(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._post([text])[0]


def get_ragas_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=EVAL_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )


def get_ragas_embeddings() -> _HuggingFaceInferenceEmbeddings:
    return _HuggingFaceInferenceEmbeddings()
