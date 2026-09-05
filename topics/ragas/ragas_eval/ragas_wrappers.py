import asyncio
import json
import os
import time
import urllib.request
from typing import List

from pydantic import Field
from ragas.embeddings import BaseRagasEmbeddings
from ragas.run_config import RunConfig

from config import ANTHROPIC_API_KEY, EVAL_LLM_MODEL

_HF_MODEL = "BAAI/bge-small-en-v1.5"
_HF_URL   = f"https://router.huggingface.co/hf-inference/models/{_HF_MODEL}"


def _post_hf(texts: List[str], retries: int = 3) -> List[List[float]]:
    token = os.environ.get("HF_TOKEN", "")
    headers = {
        "Content-Type": "application/json",
        **({"Authorization": f"Bearer {token}"} if token else {}),
    }
    body = json.dumps({
        "inputs": texts,
        "options": {"wait_for_model": True},
    }).encode()

    for attempt in range(retries):
        req = urllib.request.Request(_HF_URL, data=body, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read())
                if result and isinstance(result[0], float):
                    return [result]
                return result
        except urllib.error.HTTPError as e:
            if e.code == 503 and attempt < retries - 1:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(
                f"HuggingFace Inference API error {e.code}: {e.read().decode()}"
            ) from e

    raise RuntimeError("HuggingFace Inference API failed after retries")


class HuggingFaceRagasEmbeddings(BaseRagasEmbeddings):
    """
    Pure-HTTP HuggingFace embeddings, ragas 0.4.x native (extends BaseRagasEmbeddings).
    No ONNX, no gRPC — compatible with any CPU.
    """
    run_config: RunConfig = Field(default_factory=RunConfig)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return _post_hf(texts)

    def embed_query(self, text: str) -> List[float]:
        return _post_hf([text])[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await asyncio.to_thread(_post_hf, texts)

    async def aembed_query(self, text: str) -> List[float]:
        return (await asyncio.to_thread(_post_hf, [text]))[0]


def get_ragas_llm():
    """ragas 0.4.x native LLM via llm_factory + AsyncAnthropic."""
    from anthropic import AsyncAnthropic
    from ragas.llms import llm_factory
    llm = llm_factory(
        model=EVAL_LLM_MODEL,
        provider="anthropic",
        client=AsyncAnthropic(api_key=ANTHROPIC_API_KEY),
        max_tokens=4096,  # faithfulness NLI reasoning needs more than the default 1024
    )
    # Anthropic rejects requests with both temperature and top_p.
    # InstructorLLM includes top_p in model_args by default — remove it.
    if hasattr(llm, "model_args"):
        llm.model_args.pop("top_p", None)
    return llm


def get_ragas_embeddings() -> HuggingFaceRagasEmbeddings:
    return HuggingFaceRagasEmbeddings()
