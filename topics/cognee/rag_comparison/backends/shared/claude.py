from anthropic import AsyncAnthropic
from config import ANTHROPIC_API_KEY, CLAUDE_MODEL

_client: AsyncAnthropic | None = None

SYSTEM_PROMPT = (
    "You are a helpful customer support assistant for Everstorm Outfitters, "
    "an outdoor gear company. Answer the user's question using only the "
    "provided context. If the context doesn't fully address the question, "
    "share what you know and note the gap."
)


def get_client() -> AsyncAnthropic:
    global _client
    if _client is None:
        _client = AsyncAnthropic(api_key=ANTHROPIC_API_KEY)
    return _client


async def generate_answer(question: str, context: str) -> str:
    response = await get_client().messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
    )
    return response.content[0].text
