"""Live Ragas eval playground (Gradio on Flyte 2).

Pick a question from the test set (full 9-metric suite, since it has a
ground-truth answer) or type your own (reference-free metrics only). The app
retrieves top-k chunks from the mounted Chroma index, answers with the gemma4
vLLM app, then scores that single response with Ragas live and shows the metric
chips + retrieved contexts.

Same AppEnvironment shape as the sibling rag-chroma-flyte/chat_app.py:
the Chroma index is mounted from a `build_index` run via `flyte.app.RunOutput`.

Deploy (after `flyte run eval_pipeline.py build_index --max_docs 0`):
    RAGAS_INDEX_RUN=<build_index-run-name> python eval_app.py
"""

from __future__ import annotations

import os
from pathlib import Path

import flyte
import flyte.app
import flyte.io

from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    RAGAS_PIP_PACKAGES,
    REGISTRY,
    PLATFORM,
    VLLM_APP_NAME,
    VLLM_MODEL_ID,
)

DEFAULT_TOP_K = 4
# build_index, run standalone, exposes the Chroma dir as o0.
INDEX_TASK = "ragas-eval.build_index"


# ── Image ─────────────────────────────────────────────────────────────────────

app_image = (
    flyte.Image.from_debian_base(
        name="ragas-app-image",
        registry=REGISTRY,
        platform=PLATFORM,
    )
    .with_pip_packages(
        "gradio==5.42.0",
        "openai>=1.50.0",
        *RAGAS_PIP_PACKAGES,
    )
    # Bundle the sibling modules into the app pod (same trick the cognee chat app
    # uses); otherwise `import config` / `import ragas_lib` fail in the server.
    .with_source_file(Path(__file__).parent / "config.py")
    .with_source_file(Path(__file__).parent / "ragas_lib.py")
)


# ── App env ───────────────────────────────────────────────────────────────────
#
# Mount the Chroma index from a build_index run (pin with RAGAS_INDEX_RUN, else
# the latest succeeded build_index run). The vLLM endpoint is reached by its
# cluster-internal Knative DNS name, same as the chat apps.

_index_run = os.environ.get("RAGAS_INDEX_RUN")
_chroma_run_output = (
    flyte.app.RunOutput(type="directory", run_name=_index_run)
    if _index_run
    else flyte.app.RunOutput(type="directory", task_name=INDEX_TASK)
)

env = flyte.app.AppEnvironment(
    name="ragas-eval-app",
    image=app_image,
    # Single-query retrieval + a handful of LLM-judge calls. Give it a little
    # more room than the chat app since ragas loads langchain + BGE in-process.
    resources=flyte.Resources(cpu="2", memory="6Gi"),
    port=7860,
    requires_auth=False,
    parameters=[
        flyte.app.Parameter(
            name="vllm_url",
            value=f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local",
            env_var="VLLM_URL",
        ),
        flyte.app.Parameter(name="model_id", value=VLLM_MODEL_ID),
        flyte.app.Parameter(
            name="chroma_dir",
            value=_chroma_run_output,
            download=True,
            env_var="CHROMA_DIR",
        ),
        flyte.app.Parameter(name="embedding_model", value=EMBEDDING_MODEL),
        flyte.app.Parameter(name="collection_name", value=COLLECTION_NAME),
        flyte.app.Parameter(name="judge", value="gemma"),
        flyte.app.Parameter(name="default_top_k", value=str(DEFAULT_TOP_K)),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)


APP_CSS = """
.ev-answer { border: 1px solid var(--border-color-primary); border-radius: 8px;
    padding: 12px 14px; background: var(--background-fill-secondary);
    color: var(--body-text-color); line-height: 1.5; white-space: pre-wrap; }
.ev-ref { color: var(--body-text-color-subdued); font-size: 0.85rem; margin-top: 6px; }
.ev-empty { color: var(--body-text-color-subdued); font-style: italic; padding: 8px; }
/* Override the report CSS's hardcoded light colors so the retrieved-context
   cards are readable on the app's (dark) theme. */
.sc-ctx > summary { color: var(--body-text-color); }
.sc-ctx-item { background: var(--background-fill-primary);
    color: var(--body-text-color); border: 1px solid var(--border-color-primary); }
.sc-ctx-n { color: var(--body-text-color-subdued); }
/* The chips keep their pale green/amber/red backgrounds, but Gradio's dark
   theme forces the text white (unreadable on pale). Pin dark text on the chip
   and its inner spans with !important so it wins the cascade. */
.sc-chip-good, .sc-chip-good * { color: #1c7a3e !important; }
.sc-chip-mid, .sc-chip-mid * { color: #9a6b12 !important; }
.sc-chip-bad, .sc-chip-bad * { color: #b23030 !important; }
.sc-chip-na, .sc-chip-na * { color: #555 !important; }
"""


# ── Server ────────────────────────────────────────────────────────────────────

@env.server
def eval_server(
    vllm_url: str,
    model_id: str,
    chroma_dir: flyte.io.Dir,
    embedding_model: str,
    collection_name: str,
    judge: str,
    default_top_k: str,
):
    import sys
    import traceback
    try:
        _run(vllm_url, model_id, chroma_dir.path, embedding_model,
             collection_name, judge, int(default_top_k))
    except BaseException as e:
        print(f"!!! eval_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _run(vllm_url, model_id, chroma_path, embedding_model,
         collection_name, judge, default_top_k):
    import chromadb
    import gradio as gr
    from datasets import load_dataset
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    import config as C
    from ragas_lib import SCORECARD_CSS, evaluate_one, render_chips, _esc

    print(f"[eval_server] vLLM at {vllm_url}/v1 (model={model_id})", flush=True)
    print(f"[eval_server] Chroma at {chroma_path!r} (collection={collection_name})", flush=True)

    encoder = SentenceTransformer(embedding_model)
    collection = chromadb.PersistentClient(path=chroma_path).get_collection(name=collection_name)
    print(f"[eval_server] Collection '{collection_name}' has {collection.count()} chunks", flush=True)
    llm = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="not-used")

    # Test-set questions for the dropdown; map question -> ground-truth answer so
    # a picked question runs the full reference-based suite.
    qa = load_dataset(C.DATASET_REPO, C.QA_CONFIG, split=C.QA_SPLIT).shuffle(seed=42)
    qa_pairs = [
        (r[C.QA_QUESTION_COLUMN].strip(), r[C.QA_ANSWER_COLUMN].strip())
        for r in qa.select(range(min(40, len(qa))))
        if r.get(C.QA_QUESTION_COLUMN) and r.get(C.QA_ANSWER_COLUMN)
    ]
    qa_map = dict(qa_pairs)
    question_choices = [q for q, _ in qa_pairs]

    RAG_SYSTEM_PROMPT = (
        "You are a helpful assistant. Use the provided CONTEXT to answer the "
        "question. If the answer is not in the context, say you don't know; do "
        "not invent. Answer concisely."
    )

    def _answer(question, contexts):
        ctx_block = "\n\n".join(f"[#{i}] {c}" for i, c in enumerate(contexts, 1))
        return (llm.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": f"CONTEXT:\n{ctx_block}\n\nQUESTION: {question}"},
            ],
            temperature=0.0,
            max_tokens=512,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ).choices[0].message.content or "").strip()

    def _contexts_html(contexts):
        if not contexts:
            return ""
        items = "".join(
            f'<div class="sc-ctx-item"><span class="sc-ctx-n">#{i}</span>{_esc(c)}</div>'
            for i, c in enumerate(contexts, 1)
        )
        return (f'<details class="sc-ctx" open><summary>Retrieved contexts '
                f'({len(contexts)})</summary>{items}</details>')

    def evaluate_cb(question, top_k):
        q = (question or "").strip()
        if not q:
            yield ('<div class="ev-empty">Pick or type a question, then Evaluate.</div>',
                   "", "")
            return

        reference = qa_map.get(q)
        ref_note = (f'<div class="ev-ref">Ground truth: <b>{_esc(reference)}</b> '
                    f'· full 9-metric suite</div>') if reference else (
                    '<div class="ev-ref">No ground truth (typed question) '
                    '· reference-free metrics only</div>')

        vec = encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True).tolist()
        res = collection.query(query_embeddings=vec, n_results=max(1, int(top_k)),
                               include=["documents"])
        contexts = res["documents"][0] if res["documents"] else []
        ctx_html = _contexts_html(contexts)

        # Stream the answer first, then the (slower) judge scores.
        answer = _answer(q, contexts)
        answer_html = f'<div class="ev-answer">{_esc(answer)}</div>{ref_note}'
        yield answer_html, '<div class="ev-empty">Scoring with Ragas…</div>', ctx_html

        specs, record, metric_cols = evaluate_one(q, contexts, answer, reference, judge)
        chips = render_chips(specs, record, metric_cols)
        yield answer_html, chips, ctx_html

    with gr.Blocks(title="Ragas eval playground", css=SCORECARD_CSS + APP_CSS) as demo:
        gr.Markdown(
            f"# Ragas eval playground\n"
            f"RAG answerer + judge: `{model_id}` · Encoder: `{embedding_model}` · "
            f"Collection: `{collection_name}` ({collection.count()} chunks)"
        )
        with gr.Row():
            question = gr.Dropdown(
                choices=question_choices, label="Question (pick from the test set, or type your own)",
                allow_custom_value=True, scale=4,
            )
            top_k = gr.Slider(1, 10, value=default_top_k, step=1, label="top_k", scale=1)
        evaluate = gr.Button("Evaluate", variant="primary")

        gr.Markdown("### Answer")
        answer_out = gr.HTML(value='<div class="ev-empty">Pick or type a question, then Evaluate.</div>')
        gr.Markdown("### Ragas metrics")
        chips_out = gr.HTML()
        gr.Markdown("### Retrieved context")
        ctx_out = gr.HTML()

        evaluate.click(evaluate_cb, inputs=[question, top_k],
                       outputs=[answer_out, chips_out, ctx_out])

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    import time
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Ragas eval playground deployed: {app.url}")
