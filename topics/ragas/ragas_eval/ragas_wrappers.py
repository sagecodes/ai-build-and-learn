from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import FastEmbedEmbeddings

from config import ANTHROPIC_API_KEY, EMBEDDING_MODEL, EVAL_LLM_MODEL


def get_ragas_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=EVAL_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )


def get_ragas_embeddings() -> FastEmbedEmbeddings:
    try:
        return FastEmbedEmbeddings(model_name=EMBEDDING_MODEL)
    except Exception as e:
        raise RuntimeError(
            f"fastembed model download failed — check network and disk space. "
            f"Original error: {e}"
        ) from e
