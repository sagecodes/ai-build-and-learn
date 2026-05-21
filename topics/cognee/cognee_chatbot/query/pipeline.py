"""
query/pipeline.py

Cognee query task: search the knowledge graph and generate a grounded answer.
Cognee handles retrieval routing internally — no manual mode selection needed.
"""

import json

from config import task_env, configure_cognee
from query.generation import generate_answer


@task_env.task
async def query_pipeline(question: str) -> str:
    # configure_cognee() must run before importing cognee so env vars are set
    api_key = configure_cognee()
    import cognee
    from cognee.api.v1.search import SearchType

    results = await cognee.search(
        query_type=SearchType.GRAPH_COMPLETION,
        query_text=question,
    )

    answer = await generate_answer(
        question=question,
        cognee_results=results or [],
        api_key=api_key,
    )

    return json.dumps({
        "answer": answer,
        "result_count": len(results) if results else 0,
    })
