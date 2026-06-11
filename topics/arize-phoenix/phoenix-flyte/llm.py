"""Provider-switchable chat model, shared by every pipeline step.

Default is OpenAI (cloud), like the sibling tavily workshop. Flip the whole
pipeline to the in-cluster gemma4 vLLM app (OSS, no API key) by passing
provider="vllm" (the `--provider vllm` flag threads it through every task).
Both go through ChatOpenAI, so the traces look the same; only the model differs.
"""

from langchain_openai import ChatOpenAI

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    VLLM_MODEL_ID,
    VLLM_URL,
)


def build_llm(provider: str = LLM_PROVIDER, **kwargs):
    """Build the chat model for the chosen provider ("openai" or "vllm")."""
    if provider == "vllm":
        # The gemma4 vLLM app speaks the OpenAI API; api_key is required by the
        # client but unused by the in-cluster server.
        return ChatOpenAI(
            model=VLLM_MODEL_ID,
            base_url=VLLM_URL.rstrip("/") + "/v1",
            api_key="not-used",
            **kwargs,
        )
    return ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, **kwargs)
