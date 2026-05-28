async def query(question: str) -> tuple[str, str]:
    """
    Query the Neo4j graph backend.
    Returns (retrieved_context, answer).
    """
    raise NotImplementedError
