async def query(question: str) -> tuple[str, str]:
    """
    Query the pgvector backend.
    Returns (retrieved_context, answer).
    """
    raise NotImplementedError
