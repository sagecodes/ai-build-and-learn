"""Flyte 2 Ragas eval pipeline: score the rag-chroma-flyte RAG system with a
broad Ragas metric suite, then A/B compare configs.

Flow:
    resolve_index ─┐                     (reuse a rag_pipeline run, or build inline)
                   ├─> run_rag ─> ragas_score ─> scorecard report
    load_qa ───────┘

The same gemma4 vLLM app is both the RAG answerer and the LLM-as-judge.

Usage:
    # quick local smoke (small + fast)
    flyte run --local eval_pipeline.py ragas_eval --max_questions 5 --max_docs 200

    # full eval, remote on the devbox
    flyte run eval_pipeline.py ragas_eval

    # reuse the Chroma index from an existing rag_pipeline run
    RAG_PIPELINE_RUN=<run-name> flyte run eval_pipeline.py ragas_eval

    # swap the judge to OpenAI if the local model trips a metric
    flyte run eval_pipeline.py ragas_eval --judge openai

    # the feedback loop: compare retrieval depths side by side
    flyte run eval_pipeline.py ragas_compare --top_ks '[2,6]'
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from config import (
    COLLECTION_NAME,
    CORPUS_CONFIG,
    CORPUS_SPLIT,
    CORPUS_TEXT_COLUMN,
    DATASET_REPO,
    EMBEDDING_MODEL,
    QA_ANSWER_COLUMN,
    QA_CONFIG,
    QA_QUESTION_COLUMN,
    QA_SPLIT,
    VLLM_MODEL_ID,
    VLLM_URL,
    eval_env,
)
from ragas_lib import build_metrics, render_compare, render_scorecard, run_eval, split_text

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = eval_env


@dataclass
class EvalResult:
    """One eval pass. `aggregate` maps each Ragas metric column to its mean."""
    judge: str
    n_questions: int
    top_k: int
    aggregate: dict[str, float] = field(default_factory=dict)
    results: flyte.io.Dir | None = None


# ──────────────────────────────────────────────────────────────────────────────
# Task — load the eval test set (real questions + ground-truth answers)
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def load_qa(max_questions: int = 20, seed: int = 42) -> flyte.io.Dir:
    """Pull the question-answer config and write qa.jsonl of {question, ground_truth}."""
    from datasets import load_dataset

    ds = load_dataset(DATASET_REPO, QA_CONFIG, split=QA_SPLIT)
    ds = ds.shuffle(seed=seed)
    if max_questions and max_questions > 0:
        ds = ds.select(range(min(max_questions, len(ds))))

    out_dir = Path(tempfile.mkdtemp(prefix="ragas_qa_"))
    qa_path = out_dir / "qa.jsonl"
    n = 0
    with qa_path.open("w") as f:
        for row in ds:
            q = (row.get(QA_QUESTION_COLUMN) or "").strip()
            a = (row.get(QA_ANSWER_COLUMN) or "").strip()
            if not q or not a:
                continue
            f.write(json.dumps({"question": q, "ground_truth": a}) + "\n")
            n += 1

    log.info(f"Loaded {n} QA pairs from {DATASET_REPO} [{QA_CONFIG}]")
    await flyte.report.replace.aio(
        f"<h2>Eval test set</h2>"
        f"<p><b>Source:</b> {DATASET_REPO} [{QA_CONFIG}/{QA_SPLIT}]</p>"
        f"<p><b>Questions kept:</b> {n} (real ground-truth answers)</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Task — build the Chroma index inline (identical to the sibling rag pipeline)
# ──────────────────────────────────────────────────────────────────────────────

@env.task(cache="auto", report=True)
async def build_index(
    max_docs: int = 0,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
) -> flyte.io.Dir:
    """Fetch the passage corpus, chunk, embed with BGE, persist a Chroma dir.

    Same dataset, splitter, encoder, and collection as
    topics/vectorstore/rag-chroma-flyte, so we evaluate the same RAG system.
    """
    import chromadb
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    ds = load_dataset(DATASET_REPO, CORPUS_CONFIG, split=CORPUS_SPLIT)
    if max_docs and max_docs > 0:
        ds = ds.select(range(min(max_docs, len(ds))))

    rows: list[dict] = []
    for i, row in enumerate(ds):
        text = (row.get(CORPUS_TEXT_COLUMN) or "").strip()
        if not text:
            continue
        doc_id = str(row.get("id", i))
        for j, chunk in enumerate(split_text(text, chunk_size, chunk_overlap)):
            rows.append({"chunk_id": f"{doc_id}::{j}", "doc_id": doc_id, "text": chunk})

    log.info(f"Embedding {len(rows)} chunks with {EMBEDDING_MODEL}")
    encoder = SentenceTransformer(EMBEDDING_MODEL)

    persist_dir = Path(tempfile.mkdtemp(prefix="chroma_"))
    client = chromadb.PersistentClient(path=str(persist_dir))
    coll = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"embedding_model": EMBEDDING_MODEL, "hnsw:space": "cosine"},
    )
    batch_size = 64
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        texts = [r["text"] for r in batch]
        vectors = encoder.encode(
            texts, normalize_embeddings=True, convert_to_numpy=True
        ).tolist()
        coll.add(
            ids=[r["chunk_id"] for r in batch],
            documents=texts,
            embeddings=vectors,
            metadatas=[{"doc_id": r["doc_id"]} for r in batch],
        )

    log.info(f"Chroma collection '{COLLECTION_NAME}' has {coll.count()} chunks")
    await flyte.report.replace.aio(
        f"<h2>Built Chroma index</h2>"
        f"<p><b>Embedding model:</b> {EMBEDDING_MODEL}</p>"
        f"<p><b>Chunks indexed:</b> {coll.count()}</p>"
        f"<p><b>Collection:</b> {COLLECTION_NAME}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(persist_dir))


async def resolve_index(
    rag_run: str,
    max_docs: int,
    chunk_size: int,
    chunk_overlap: int,
) -> flyte.io.Dir:
    """Reuse a prior rag_pipeline run's Chroma Dir, else build one inline.

    A run name (CLI `--rag_run`, or the RAG_PIPELINE_RUN env var) wins; that's
    the "build on the previous project" path. Otherwise we build an identical
    index so the demo runs end-to-end with a single command.
    """
    run_name = rag_run or os.environ.get("RAG_PIPELINE_RUN", "")
    if run_name:
        from flyte.remote import Run

        log.info(f"Reusing Chroma index from rag_pipeline run '{run_name}'")
        run = await Run.get.aio(run_name)
        outputs = await run.outputs.aio()
        return outputs[0]  # rag_pipeline returns the Chroma Dir as o0
    log.info("No RAG_PIPELINE_RUN set; building an index inline")
    return await build_index(max_docs, chunk_size, chunk_overlap)


# ──────────────────────────────────────────────────────────────────────────────
# Task — run the RAG system over every question (retrieve + generate)
# ──────────────────────────────────────────────────────────────────────────────

RAG_SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the provided CONTEXT to answer the "
    "question. If the answer is not in the context, say you don't know; do not "
    "invent. Answer concisely."
)


@env.task(report=True)
async def run_rag(
    chroma_dir: flyte.io.Dir,
    qa_dir: flyte.io.Dir,
    top_k: int = 4,
) -> flyte.io.Dir:
    """For each question: BGE encode -> Chroma top-k -> gemma4 answer.

    Writes results.jsonl in exactly the shape Ragas wants:
    {user_input, retrieved_contexts, response, reference}.
    """
    import chromadb
    from openai import OpenAI
    from sentence_transformers import SentenceTransformer

    chroma_path = await chroma_dir.download()
    qa_path = Path(await qa_dir.download()) / "qa.jsonl"
    questions = [json.loads(ln) for ln in qa_path.open()]

    encoder = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=chroma_path)
    coll = client.get_collection(name=COLLECTION_NAME)
    llm = OpenAI(base_url=VLLM_URL.rstrip("/") + "/v1", api_key="not-used")

    out_dir = Path(tempfile.mkdtemp(prefix="ragas_results_"))
    results_path = out_dir / "results.jsonl"
    with results_path.open("w") as fout:
        for idx, item in enumerate(questions, 1):
            q = item["question"]
            vec = encoder.encode([q], normalize_embeddings=True, convert_to_numpy=True).tolist()
            res = coll.query(query_embeddings=vec, n_results=max(1, top_k), include=["documents"])
            contexts = res["documents"][0] if res["documents"] else []

            ctx_block = "\n\n".join(f"[#{i}] {c}" for i, c in enumerate(contexts, 1))
            answer = llm.chat.completions.create(
                model=VLLM_MODEL_ID,
                messages=[
                    {"role": "system", "content": RAG_SYSTEM_PROMPT},
                    {"role": "user", "content": f"CONTEXT:\n{ctx_block}\n\nQUESTION: {q}"},
                ],
                temperature=0.0,
                max_tokens=512,
                # Suppress gemma's <|channel>thought…</channel> reasoning tokens
                # so the answer is clean prose for the judge to score.
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            ).choices[0].message.content or ""

            fout.write(json.dumps({
                "user_input": q,
                "retrieved_contexts": contexts,
                "response": answer.strip(),
                "reference": item["ground_truth"],
            }) + "\n")
            if idx % 5 == 0:
                log.info(f"  answered {idx}/{len(questions)}")

    log.info(f"Ran RAG over {len(questions)} questions (top_k={top_k})")
    await flyte.report.replace.aio(
        f"<h2>Ran RAG system</h2>"
        f"<p><b>Questions:</b> {len(questions)}</p>"
        f"<p><b>top_k:</b> {top_k} · <b>model:</b> {VLLM_MODEL_ID}</p>"
    )
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Task — score with the Ragas metric suite + render the scorecard
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def ragas_score(
    results_dir: flyte.io.Dir,
    judge: str = "gemma",
    top_k: int = 4,
) -> EvalResult:
    """Run the metric suite over the RAG outputs and emit the scorecard report."""
    results_path = Path(await results_dir.download()) / "results.jsonl"
    samples = [json.loads(ln) for ln in results_path.open()]
    log.info(f"Scoring {len(samples)} samples with judge={judge}")

    specs, records, aggregate = run_eval(samples, judge=judge)

    meta = {
        "questions": len(samples),
        "top_k": top_k,
        "judge": judge,
        "model": VLLM_MODEL_ID,
    }
    await flyte.report.replace.aio(render_scorecard(specs, records, aggregate, meta))
    await flyte.report.flush.aio()

    for col, score in aggregate.items():
        log.info(f"  {col}: {score:.3f}")
    return EvalResult(
        judge=judge,
        n_questions=len(samples),
        top_k=top_k,
        aggregate=aggregate,
        results=results_dir,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrators
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def ragas_eval(
    top_k: int = 4,
    max_questions: int = 20,
    max_docs: int = 0,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    judge: str = "gemma",
    rag_run: str = "",
) -> EvalResult:
    """End-to-end: resolve index -> load QA -> run RAG -> score -> scorecard."""
    await flyte.report.replace.aio("<h2>Ragas eval</h2><p>Step 1/3 — preparing index + test set…</p>")
    await flyte.report.flush.aio()
    chroma = await resolve_index(rag_run, max_docs, chunk_size, chunk_overlap)
    qa = await load_qa(max_questions)

    await flyte.report.replace.aio(f"<h2>Ragas eval</h2><p>Step 2/3 — running RAG (top_k={top_k})…</p>")
    await flyte.report.flush.aio()
    results = await run_rag(chroma, qa, top_k)

    await flyte.report.replace.aio("<h2>Ragas eval</h2><p>Step 3/3 — scoring with Ragas…</p>")
    await flyte.report.flush.aio()
    res = await ragas_score(results, judge, top_k)

    # Re-render the scorecard at the top-level node so it's the first thing you
    # see on the run (aggregate only; the per-question table lives on the
    # ragas_score node). build_metrics() is cheap and makes no LLM calls.
    specs = build_metrics()
    meta = {"questions": res.n_questions, "top_k": top_k, "judge": judge, "model": VLLM_MODEL_ID}
    await flyte.report.replace.aio(render_scorecard(specs, [], res.aggregate, meta))
    await flyte.report.flush.aio()
    return res


@env.task(report=True)
async def ragas_compare(
    top_ks: list[int] = [2, 6],
    max_questions: int = 20,
    max_docs: int = 0,
    chunk_size: int = 1200,
    chunk_overlap: int = 150,
    judge: str = "gemma",
    rag_run: str = "",
) -> list[EvalResult]:
    """The feedback loop: score the same questions at several retrieval depths.

    Build the index + test set once, then run + score per top_k, and render a
    side-by-side scorecard so you can see which config the metrics prefer.
    """
    chroma = await resolve_index(rag_run, max_docs, chunk_size, chunk_overlap)
    qa = await load_qa(max_questions)

    results: list[EvalResult] = []
    for k in top_ks:
        await flyte.report.replace.aio(
            f"<h2>Ragas A/B</h2><p>Evaluating top_k={k} "
            f"({len(results) + 1}/{len(top_ks)})…</p>"
        )
        await flyte.report.flush.aio()
        rag_out = await run_rag(chroma, qa, k)
        results.append(await ragas_score(rag_out, judge, k))

    specs = build_metrics()
    by_config = [(f"top_k={r.top_k}", r.aggregate) for r in results]
    meta = {"questions": results[0].n_questions if results else 0, "judge": judge, "model": VLLM_MODEL_ID}
    await flyte.report.replace.aio(render_compare(by_config, specs, meta))
    await flyte.report.flush.aio()
    return results


@env.task(report=True)
async def ragas_compare_chunking(
    chunk_sizes: list[int] = [300, 1200],
    top_k: int = 4,
    max_questions: int = 20,
    max_docs: int = 0,
    chunk_overlap: int = 150,
    judge: str = "gemma",
) -> list[EvalResult]:
    """The other kind of feedback loop: sweep an INDEX-time hyperparameter.

    Unlike ragas_compare (which reuses one index and varies the query-time
    top_k), chunk_size changes how documents are split before embedding, so the
    index is rebuilt for each value. Same questions throughout, so the scores are
    comparable. build_index caches per chunk_size, so re-runs are cheap.
    """
    qa = await load_qa(max_questions)

    results: list[EvalResult] = []
    for cs in chunk_sizes:
        await flyte.report.replace.aio(
            f"<h2>Ragas A/B (chunking)</h2><p>Rebuilding index + evaluating "
            f"chunk_size={cs} ({len(results) + 1}/{len(chunk_sizes)})…</p>"
        )
        await flyte.report.flush.aio()
        chroma = await build_index(max_docs, cs, chunk_overlap)
        rag_out = await run_rag(chroma, qa, top_k)
        results.append(await ragas_score(rag_out, judge, top_k))

    specs = build_metrics()
    by_config = [(f"chunk={cs}", r.aggregate) for cs, r in zip(chunk_sizes, results)]
    meta = {
        "questions": results[0].n_questions if results else 0,
        "top_k": top_k,
        "judge": judge,
        "model": VLLM_MODEL_ID,
    }
    await flyte.report.replace.aio(render_compare(by_config, specs, meta))
    await flyte.report.flush.aio()
    return results


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(ragas_eval)
    print(f"Ragas eval run: {run.name}")
    print(f"  {run.url}")
