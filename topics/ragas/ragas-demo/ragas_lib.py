"""Pure helpers for the Ragas eval demo: judge/embeddings wiring, the metric
suite, the evaluate() call, and the HTML scorecards.

No Flyte in here on purpose. Everything that actually scores text lives in this
module so `eval_pipeline.py` stays a thin orchestration layer (same split as the
sibling cognee project's `cognee_lib.py`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import EMBEDDING_MODEL, VLLM_MODEL_ID, VLLM_URL


# ──────────────────────────────────────────────────────────────────────────────
# Chunking — copied verbatim from rag-chroma-flyte/pipeline.py so the index this
# demo builds inline is byte-for-byte the same RAG system we serve there.
# ──────────────────────────────────────────────────────────────────────────────

def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Recursive character splitter: paragraphs -> lines -> sentences -> words -> chars."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    for sep in ("\n\n", "\n", ". ", " ", ""):
        if sep == "":
            return [text[i:i + chunk_size] for i in range(0, len(text), max(1, chunk_size - overlap))]
        parts = text.split(sep)
        if len(parts) == 1:
            continue
        out: list[str] = []
        buf = ""
        for part in parts:
            piece = part + sep
            if len(piece) > chunk_size:
                if buf.strip():
                    out.append(buf.strip())
                    buf = ""
                out.extend(split_text(part, chunk_size, overlap))
                continue
            if len(buf) + len(piece) <= chunk_size:
                buf += piece
            else:
                if buf.strip():
                    out.append(buf.strip())
                tail = buf[-overlap:] if overlap and buf else ""
                buf = tail + piece
        if buf.strip():
            out.append(buf.strip())
        return [c for c in out if c.strip()]
    return [text]


# ──────────────────────────────────────────────────────────────────────────────
# Judge LLM + judge embeddings
# ──────────────────────────────────────────────────────────────────────────────

def build_judge(judge: str = "gemma"):
    """Return a Ragas LLM wrapper for the LLM-as-judge metrics.

    `gemma` (default) points at the self-hosted gemma4 vLLM app over its
    OpenAI-compatible endpoint, so the whole eval stays on the devbox. `openai`
    swaps in gpt-4o-mini for rock-solid structured output (needs OPENAI_API_KEY)
    when the smaller local model trips up a metric live.
    """
    from langchain_openai import ChatOpenAI
    from ragas.llms import LangchainLLMWrapper

    if judge == "openai":
        chat = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    elif judge == "gemma":
        chat = ChatOpenAI(
            model=VLLM_MODEL_ID,
            base_url=VLLM_URL.rstrip("/") + "/v1",
            api_key="not-used",
            temperature=0,
            timeout=180,
            max_retries=2,
            # gemma4 emits <|channel>thought…</channel> reasoning tokens by
            # default, which corrupt the JSON Ragas's judge parser expects and
            # collapse every structured metric to 0. Turn thinking off so the
            # judge returns clean structured output.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    else:
        raise ValueError(f"unknown judge {judge!r}; use 'gemma' or 'openai'")
    return LangchainLLMWrapper(chat)


def build_embeddings():
    """BGE-small as the judge embeddings (answer relevancy + semantic similarity).

    Same encoder the index uses, so similarity is measured in one vector space.
    """
    from langchain_huggingface import HuggingFaceEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    hf = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(hf)


# ──────────────────────────────────────────────────────────────────────────────
# Metric suite — the showcase. Grouped so the scorecard reads as
# "retrieval quality" vs "generation quality" vs "your own metric".
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MetricSpec:
    label: str           # human label in the scorecard
    group: str           # Retrieval | Generation | Custom
    needs_ref: bool      # does it compare against the ground-truth answer?
    higher_better: bool  # score direction (noise sensitivity is lower-better)
    blurb: str           # one-line "what does this measure"
    instance: object = field(repr=False, default=None)
    col: str = ""        # the column name Ragas emits for it (set at build time)


def build_metrics(reference: bool = True) -> list[MetricSpec]:
    """Instantiate the metric suite. The judge LLM + embeddings are injected by
    evaluate(), so metrics that need them take no args here.

    With `reference=False` (e.g. a free-typed question with no ground truth),
    the reference-based metrics are dropped, leaving the reference-free ones
    (faithfulness, response relevancy, conciseness).
    """
    from ragas.metrics import (
        AspectCritic,
        ContextEntityRecall,
        FactualCorrectness,
        Faithfulness,
        LLMContextPrecisionWithReference,
        LLMContextRecall,
        NoiseSensitivity,
        ResponseRelevancy,
        SemanticSimilarity,
    )

    specs = [
        # Retrieval quality — did we fetch the right context?
        MetricSpec(
            "Context Precision", "Retrieval", True, True,
            "Are the retrieved chunks actually relevant to the answer?",
            LLMContextPrecisionWithReference(),
        ),
        MetricSpec(
            "Context Recall", "Retrieval", True, True,
            "Did retrieval surface everything the reference answer needs?",
            LLMContextRecall(),
        ),
        MetricSpec(
            "Context Entity Recall", "Retrieval", True, True,
            "Fraction of the reference's entities present in the context.",
            ContextEntityRecall(),
        ),
        # Generation quality — is the answer good given that context?
        MetricSpec(
            "Faithfulness", "Generation", False, True,
            "Is every claim in the answer grounded in the retrieved context?",
            Faithfulness(),
        ),
        MetricSpec(
            "Response Relevancy", "Generation", False, True,
            "Does the answer actually address the question asked?",
            ResponseRelevancy(),
        ),
        MetricSpec(
            "Factual Correctness", "Generation", True, True,
            "Do the answer's claims match the ground-truth answer?",
            FactualCorrectness(),
        ),
        MetricSpec(
            "Semantic Similarity", "Generation", True, True,
            "Embedding similarity between the answer and the ground truth.",
            SemanticSimilarity(),
        ),
        MetricSpec(
            "Noise Sensitivity", "Generation", True, False,
            "How often irrelevant retrieved context corrupts the answer (lower is better).",
            NoiseSensitivity(),
        ),
        # Custom — a metric you define yourself, to show Ragas is extensible.
        MetricSpec(
            "Conciseness (custom)", "Custom", False, True,
            "Custom AspectCritic: is the answer concise and direct, no hedging or filler?",
            AspectCritic(
                name="conciseness",
                definition="Is the answer concise and direct, without hedging or filler?",
            ),
        ),
    ]
    if not reference:
        specs = [s for s in specs if not s.needs_ref]
    for s in specs:
        s.col = getattr(s.instance, "name", s.label)
    return specs


# ──────────────────────────────────────────────────────────────────────────────
# Run evaluate()
# ──────────────────────────────────────────────────────────────────────────────

INPUT_COLS = {"user_input", "retrieved_contexts", "response", "reference"}


def run_eval(samples: list[dict], judge: str = "gemma", max_workers: int = 8,
             reference: bool = True):
    """Score `samples` with the metric suite.

    `reference=False` drops the ground-truth metrics (for free-typed questions).
    Returns (specs, per_sample_records, aggregate) where aggregate maps each
    emitted metric column to its mean over the questions.
    """
    import pandas as pd
    from ragas import EvaluationDataset, evaluate
    from ragas.run_config import RunConfig

    specs = build_metrics(reference=reference)
    judge_llm = build_judge(judge)
    emb = build_embeddings()

    dataset = EvaluationDataset.from_list(samples)
    result = evaluate(
        dataset=dataset,
        metrics=[s.instance for s in specs],
        llm=judge_llm,
        embeddings=emb,
        # Local gemma is slower than a hosted API; cap concurrency and give each
        # judge call generous time so we get scores rather than timeouts.
        run_config=RunConfig(timeout=180, max_workers=max_workers),
        # One flaky judge call should dent a cell, not blow up the whole run.
        raise_exceptions=False,
    )

    df = result.to_pandas()
    metric_cols = [c for c in df.columns if c not in INPUT_COLS]
    aggregate: dict[str, float] = {}
    for c in metric_cols:
        vals = pd.to_numeric(df[c], errors="coerce").dropna()
        if len(vals):
            aggregate[c] = float(vals.mean())

    records = df.to_dict(orient="records")
    return specs, records, aggregate


def evaluate_one(question: str, contexts: list[str], answer: str,
                 reference: str | None = None, judge: str = "gemma"):
    """Score a single RAG response. Powers the live Gradio playground.

    With a `reference` (a test-set question), runs the full suite; without one
    (a free-typed question), runs only the reference-free metrics.
    Returns (specs, record, metric_cols) where record carries the metric scores.
    """
    sample = {
        "user_input": question,
        "retrieved_contexts": list(contexts or []),
        "response": answer or "",
    }
    has_ref = bool(reference and str(reference).strip())
    if has_ref:
        sample["reference"] = reference
    specs, records, aggregate = run_eval([sample], judge=judge, reference=has_ref)
    record = records[0] if records else {}
    return specs, record, list(aggregate.keys())


def spec_for_col(col: str, specs: list[MetricSpec]) -> MetricSpec | None:
    """Best-effort map a Ragas output column back to its MetricSpec (the emitted
    column name can carry a suffix, e.g. factual_correctness(mode=f1))."""
    for s in specs:
        if s.col == col:
            return s
    for s in specs:
        if s.col and (s.col in col or col in s.col):
            return s
    return None


# ──────────────────────────────────────────────────────────────────────────────
# HTML scorecards for flyte.report
# ──────────────────────────────────────────────────────────────────────────────

SCORECARD_CSS = """
<style>
.sc-wrap { font-family: system-ui, sans-serif; max-width: 900px; }
.sc-meta { color: #666; font-size: 0.9rem; margin-bottom: 16px; }
.sc-meta code { background: #f0f0f0; padding: 1px 5px; border-radius: 4px; }
.sc-group { margin: 18px 0 6px; font-size: 1.05rem; font-weight: 700; }
.sc-table { border-collapse: collapse; width: 100%; margin-bottom: 8px; }
.sc-table td, .sc-table th { padding: 7px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: middle; }
.sc-table th { font-size: 0.8rem; color: #888; text-transform: uppercase; letter-spacing: .03em; }
.sc-name { font-weight: 600; }
.sc-blurb { color: #777; font-size: 0.82rem; }
.sc-ref { font-size: 0.7rem; color: #b06; background: #fbeaf2; padding: 1px 6px; border-radius: 8px; margin-left: 6px; }
.sc-score { font-variant-numeric: tabular-nums; font-weight: 700; width: 64px; }
.sc-bar { background: #eee; border-radius: 6px; height: 10px; width: 160px; overflow: hidden; }
.sc-bar > div { height: 100%; background: linear-gradient(90deg,#7c9,#5b8); }
.sc-win { color: #2a7; font-weight: 700; }
.sc-lose { color: #999; }
.sc-q { font-size: 0.85rem; }
.sc-q td, .sc-q th { padding: 5px 8px; }
.sc-q .truncate { max-width: 380px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sc-qcard { border: 1px solid #e5e5e5; border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; background: #fff; }
.sc-qq { font-weight: 600; font-size: 0.98rem; margin-bottom: 10px; }
.sc-qn { display: inline-block; background: #eef; color: #446; border-radius: 6px; padding: 1px 7px; margin-right: 8px; font-size: 0.8rem; }
.sc-qa { display: flex; gap: 10px; margin: 6px 0; font-size: 0.9rem; }
.sc-qlabel { flex: 0 0 70px; color: #999; text-transform: uppercase; font-size: 0.7rem; letter-spacing: .04em; padding-top: 2px; }
.sc-qtext { flex: 1; line-height: 1.45; white-space: pre-wrap; word-break: break-word; }
.sc-qref { color: #777; }
.sc-chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 10px; }
.sc-chip { display: inline-flex; align-items: center; gap: 5px; border-radius: 999px; padding: 2px 9px; font-size: 0.78rem; border: 1px solid transparent; }
.sc-chip-k { opacity: .85; }
.sc-chip-v { font-weight: 700; font-variant-numeric: tabular-nums; }
.sc-chip-good { background: #e7f6ec; color: #1c7a3e; border-color: #bfe6cd; }
.sc-chip-mid { background: #fff6e5; color: #9a6b12; border-color: #f0dcae; }
.sc-chip-bad { background: #fdecec; color: #b23030; border-color: #f3c9c9; }
.sc-chip-na { background: #f0f0f0; color: #999; }
.sc-ctx { margin-top: 10px; font-size: 0.85rem; }
.sc-ctx > summary { cursor: pointer; color: #557; user-select: none; }
.sc-ctx-item { margin: 8px 0; padding: 8px 10px; background: #f7f7f9; border-radius: 6px; line-height: 1.4; white-space: pre-wrap; word-break: break-word; }
.sc-ctx-n { display: inline-block; font-weight: 700; color: #88a; margin-right: 6px; }
</style>
"""

GROUP_ORDER = ["Retrieval", "Generation", "Custom", "Other"]


def _fmt(v) -> str:
    try:
        return f"{float(v):.3f}"
    except (TypeError, ValueError):
        return "—"


def _bar(v, higher_better=True) -> str:
    try:
        f = max(0.0, min(1.0, float(v)))
    except (TypeError, ValueError):
        return ""
    pct = (f if higher_better else 1.0 - f) * 100
    return f'<div class="sc-bar"><div style="width:{pct:.0f}%"></div></div>'


def _esc(s) -> str:
    return (str(s) if s is not None else "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _chip_class(score, higher_better=True) -> str:
    """Color a metric chip green/amber/red, flipping direction for lower-is-better."""
    try:
        f = float(score)
    except (TypeError, ValueError):
        return "sc-chip sc-chip-na"
    if not higher_better:
        f = 1.0 - f
    if f >= 0.7:
        return "sc-chip sc-chip-good"
    if f >= 0.4:
        return "sc-chip sc-chip-mid"
    return "sc-chip sc-chip-bad"


def render_chips(specs, record: dict, metric_cols) -> str:
    """A row of color-coded metric chips for one scored sample.

    Shared by the per-question scorecard cards and the live Gradio playground.
    """
    chips = []
    for c in metric_cols:
        spec = spec_for_col(c, specs)
        label = spec.label if spec else c
        hb = spec.higher_better if spec else True
        chips.append(
            f'<span class="{_chip_class(record.get(c), hb)}">'
            f'<span class="sc-chip-k">{label}</span>'
            f'<span class="sc-chip-v">{_fmt(record.get(c))}</span></span>'
        )
    return f'<div class="sc-chips">{"".join(chips)}</div>'


def render_scorecard(specs, records, aggregate, meta: dict) -> str:
    """Aggregate scorecard (grouped) + a compact per-question table."""
    grouped: dict[str, list] = {g: [] for g in GROUP_ORDER}
    for col, score in aggregate.items():
        spec = spec_for_col(col, specs)
        g = spec.group if spec else "Other"
        grouped[g].append((spec, col, score))

    meta_bits = " · ".join(
        f"<b>{k}:</b> <code>{v}</code>" for k, v in meta.items()
    )
    html = [SCORECARD_CSS, '<div class="sc-wrap">', '<h2>Ragas scorecard</h2>',
            f'<div class="sc-meta">{meta_bits}</div>']

    for g in GROUP_ORDER:
        rows = grouped.get(g) or []
        if not rows:
            continue
        html.append(f'<div class="sc-group">{g} quality</div>' if g != "Other"
                    else '<div class="sc-group">Other metrics</div>')
        html.append('<table class="sc-table"><tr>'
                    '<th>Metric</th><th>Score</th><th></th></tr>')
        for spec, col, score in rows:
            label = spec.label if spec else col
            blurb = spec.blurb if spec else ""
            ref = '<span class="sc-ref">needs ground truth</span>' if (spec and spec.needs_ref) else ""
            arrow = "" if (spec and spec.higher_better) else " ↓"
            hb = spec.higher_better if spec else True
            html.append(
                "<tr>"
                f'<td><span class="sc-name">{label}{arrow}</span>{ref}'
                f'<br><span class="sc-blurb">{blurb}</span></td>'
                f'<td class="sc-score">{_fmt(score)}</td>'
                f'<td>{_bar(score, hb)}</td>'
                "</tr>"
            )
        html.append("</table>")

    # Per-question detail (scrollable, all metric columns). Skipped on the
    # aggregate-only render (orchestrator node passes records=[]).
    if not records:
        html.append("</div>")
        return "".join(html)
    # One card per question: full question + answer + reference, the metric
    # scores as color-coded chips, and the retrieved contexts in a collapsed
    # expander. Reads like a graded answer sheet.
    metric_cols = list(aggregate.keys())
    html.append('<div class="sc-group">Per-question</div>')
    for i, r in enumerate(records, 1):
        ans = _esc(r.get("response")) or "<em>(empty)</em>"
        chips_html = render_chips(specs, r, metric_cols)
        ctxs = list(r.get("retrieved_contexts") or [])
        ctx_items = "".join(
            f'<div class="sc-ctx-item"><span class="sc-ctx-n">#{j}</span>{_esc(c)}</div>'
            for j, c in enumerate(ctxs, 1)
        )
        details = (
            f'<details class="sc-ctx"><summary>Retrieved contexts ({len(ctxs)})</summary>'
            f'{ctx_items}</details>'
        ) if ctxs else ""
        html.append(
            '<div class="sc-qcard">'
            f'<div class="sc-qq"><span class="sc-qn">Q{i}</span>{_esc(r.get("user_input"))}</div>'
            f'<div class="sc-qa"><span class="sc-qlabel">Answer</span>'
            f'<div class="sc-qtext">{ans}</div></div>'
            f'<div class="sc-qa"><span class="sc-qlabel">Reference</span>'
            f'<div class="sc-qtext sc-qref">{_esc(r.get("reference"))}</div></div>'
            f'{chips_html}'
            f'{details}'
            '</div>'
        )
    html.append("</div>")
    return "".join(html)


def render_compare(results_by_config: list[tuple[str, dict]], specs, meta: dict) -> str:
    """Side-by-side aggregate table across configs (the eval feedback loop).

    `results_by_config` is a list of (config_label, aggregate_dict). The best
    cell per metric row is highlighted, honoring each metric's score direction.
    """
    # Union of metric columns, ordered by group then suite order.
    all_cols: list[str] = []
    for _, agg in results_by_config:
        for c in agg:
            if c not in all_cols:
                all_cols.append(c)

    def sort_key(c):
        spec = spec_for_col(c, specs)
        g = spec.group if spec else "Other"
        return (GROUP_ORDER.index(g) if g in GROUP_ORDER else 99,
                next((i for i, s in enumerate(specs) if s.col == c), 99))
    all_cols.sort(key=sort_key)

    meta_bits = " · ".join(f"<b>{k}:</b> <code>{v}</code>" for k, v in meta.items())
    html = [SCORECARD_CSS, '<div class="sc-wrap">',
            "<h2>Ragas A/B comparison</h2>",
            f'<div class="sc-meta">{meta_bits}</div>',
            '<table class="sc-table"><tr><th>Metric</th>'
            + "".join(f"<th>{lbl}</th>" for lbl, _ in results_by_config) + "</tr>"]

    last_group = None
    for c in all_cols:
        spec = spec_for_col(c, specs)
        group = spec.group if spec else "Other"
        if group != last_group:
            span = len(results_by_config) + 1
            html.append(f'<tr><td colspan="{span}" class="sc-group">{group} quality</td></tr>')
            last_group = group
        label = spec.label if spec else c
        arrow = "" if (spec and spec.higher_better) else " ↓"
        hb = spec.higher_better if spec else True
        vals = [agg.get(c) for _, agg in results_by_config]
        numeric = [v for v in vals if isinstance(v, (int, float))]
        best = (max(numeric) if hb else min(numeric)) if numeric else None
        cells = ""
        for v in vals:
            cls = "sc-win" if (best is not None and v == best and len(numeric) > 1) else "sc-lose"
            cells += f'<td class="sc-score {cls}">{_fmt(v)}</td>'
        html.append(f'<tr><td class="sc-name">{label}{arrow}</td>{cells}</tr>')
    html.append("</table></div>")
    return "".join(html)
