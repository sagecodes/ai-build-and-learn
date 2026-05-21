"""
ingest/pipeline.py

Cognee ingest task: decode PDFs, add to Cognee, run cognify().

This is the "10 lines vs 300 lines" moment — cognee.add() + cognee.cognify()
replace the 6-step pipeline from graph_rag_chatbot (parse → extract → load →
index → resolve → detect communities → summarize).
"""

import base64
import json
import tempfile
from pathlib import Path

from config import task_env, configure_cognee


@task_env.task
async def ingest_pipeline(filenames: list[str], pdf_bytes_b64: list[str]) -> str:
    # configure_cognee() must run before importing cognee so env vars are set
    configure_cognee()
    import cognee

    with tempfile.TemporaryDirectory() as tmpdir:
        for name, b64 in zip(filenames, pdf_bytes_b64):
            path = Path(tmpdir) / name
            path.write_bytes(base64.b64decode(b64))
            await cognee.add(str(path))

    await cognee.cognify()

    return json.dumps({
        "status": "ok",
        "documents_ingested": len(filenames),
        "filenames": filenames,
    })
