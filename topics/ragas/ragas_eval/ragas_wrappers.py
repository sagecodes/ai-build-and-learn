from langchain_anthropic import ChatAnthropic

from config import ANTHROPIC_API_KEY, EVAL_LLM_MODEL


def get_ragas_llm() -> ChatAnthropic:
    return ChatAnthropic(
        model=EVAL_LLM_MODEL,
        api_key=ANTHROPIC_API_KEY,
        temperature=0,
    )
