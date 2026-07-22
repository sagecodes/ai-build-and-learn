"""Voice cloning: one reference clip, five models, and two numbers per clip.

Usage (runs on the DGX Spark devbox, not local Python):

    # the default: 5 cloners x the 5-script clone suite, scored
    flyte run clone_pipeline.py clone \
        --ref_audio refs/sage.wav --ref_text "$(cat refs/sage.txt)"

    # a fast two-script pass on two models
    flyte run clone_pipeline.py clone \
        --ref_audio refs/sage.wav --ref_text "..." \
        --suite clone-quick --models '["chatterbox","qwen3-1.7b-clone"]'

    # does this model clone this voice at all?
    flyte run clone_pipeline.py clone_one \
        --ref_audio refs/sage.wav --ref_text "..." --model_key chatterbox

Shape (the compare demo's shape, plus a scoring stage):

    clone ─┬─ fetch_weights(model)        ·· CPU, cached: one HF download per model
           ├─ CLONE_TASKS[adapter](...)   ·· GPU: load once, say every line in the voice
           └─ score_clones(ref, runs)     ·· GPU: WavLM x-vectors + Whisper over every clip

The interesting property: this adds NO new image to the build. The two Qwen -Base
checkpoints ride the existing qwen image, Dia and CSM ride the shared transformers
image, Chatterbox its own, and the scoring task reuses the transformers image because
both scorers were chosen to be transformers-native. Only the weights are new.

Why scoring is its own task rather than folded into each generation task: loading
Whisper and WavLM once and scoring every model's clips together costs one model load
instead of five, and it puts every clip through one identical scorer, which is the only
way the numbers are comparable across models.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass, field, replace
from pathlib import Path

import flyte
import flyte.io
import flyte.report
import soundfile as sf

import metrics
import tts_core
from config import GPU_ENVS, clone_orch_env, cpu_env, metrics_env
from compare_pipeline import fetch_weights
from models import get_spec, resolve_cloners
from prompts import get_suite

log = logging.getLogger("tts.clone")
logging.basicConfig(level=logging.INFO)


# ── Data crossing the task boundary ──────────────────────────────────────────────
#
# Same discipline as the compare demo: metadata + a Dir of wavs, never the base64 data
# URIs. The parent re-reads the wavs to build the report.

@dataclass
class CloneItem:
    text: str
    filename: str = ""            # wav in the model's Dir; "" if this line failed
    seconds: float = 0.0          # synth wall-clock
    audio_seconds: float = 0.0
    sample_rate: int = 0
    peak_gb: float = 0.0
    # "" = the image had no detector, "yes"/"no" = it ran. A plain string rather than
    # an Optional[bool] because the three states must survive serialization distinctly:
    # "we could not check" and "there is no watermark" are different claims.
    watermark: str = ""
    error: str = ""


@dataclass
class CloneRun:
    model_key: str
    items: list[CloneItem] = field(default_factory=list)
    clips: flyte.io.Dir | None = None


@dataclass
class ClipScore:
    model_key: str
    text: str
    similarity: float = 0.0
    wer: float = 0.0
    transcript: str = ""
    error: str = ""


@dataclass
class ScoreReport:
    clips: list[ClipScore] = field(default_factory=list)
    # The control: the reference scored against ITSELF (first half vs second half).
    # This is what makes a raw cosine interpretable, so it is a first-class output.
    ceiling: float = 0.0
    error: str = ""


# ── Loading the reference inside a task ──────────────────────────────────────────

async def _load_ref(ref_audio: flyte.io.File, ref_text: str) -> tts_core.RefVoice:
    local = await ref_audio.download()
    return tts_core.RefVoice.from_file(local, ref_text, name="reference")


# ── The per-model work, shared by every adapter task ─────────────────────────────

async def _run_clone(model_key: str, weights: flyte.io.Dir, texts: list[str],
                     ref_audio: flyte.io.File, ref_text: str) -> CloneRun:
    """Load one cloner once, say every line in the reference voice, return a CloneRun."""
    spec = get_spec(model_key)
    if not spec.clone_capable:
        raise ValueError(f"{model_key} cannot clone; use compare_pipeline.py for it.")

    # Point HF at the pre-fetched Dir BEFORE any loader imports huggingface_hub.
    local = await weights.download()
    os.environ["HF_HUB_CACHE"] = str(local)
    os.environ["HF_HOME"] = str(local)

    ref = await _load_ref(ref_audio, ref_text)
    log.info(f"[{model_key}] reference: {ref.seconds:.1f}s @ {ref.sample_rate}Hz")
    for w in ref.warnings():
        log.warning(f"[{model_key}] reference: {w}")

    out_dir = Path(tempfile.mkdtemp(prefix=f"clone_{model_key}_"))
    items: list[CloneItem] = []
    results: list[tts_core.AudioResult] = []
    meta = f"{spec.repo} · cloning {ref.seconds:.1f}s of reference · {spec.license}"

    handle = None
    try:
        tts_core.prepare_gpu()
        log.info(f"[{model_key}] loading via '{spec.adapter}' adapter for cloning")
        handle = tts_core.load_model(spec)

        for i, text in enumerate(texts):
            try:
                tts_core.reset_peak_memory()
                wav, sr, secs = tts_core.synth_clone(handle, spec, text, ref)
                peak = tts_core.peak_memory_gb()
                fn = f"{model_key}__{i:02d}.wav"
                tts_core.write_wav(wav, sr, out_dir / fn)
                r = tts_core.build_audio_result(spec, text, wav, sr, secs, peak)
                results.append(r)

                # Detect the watermark HERE, in the generation image: Chatterbox's image
                # is the only one with `perth` installed (it ships as one of that
                # package's deps), and metrics.detect_watermark returns None everywhere
                # else, so this costs nothing in the other four images.
                mark = metrics.detect_watermark(wav, sr)
                items.append(CloneItem(
                    text=text, filename=fn, seconds=secs, audio_seconds=r.audio_seconds,
                    sample_rate=sr, peak_gb=peak,
                    watermark="" if mark is None else ("yes" if mark else "no"),
                ))
                log.info(f"[{model_key}] {i+1}/{len(texts)}: {secs:.1f}s -> "
                         f"{r.audio_seconds:.1f}s audio ({r.speedup:.1f}x RT)")
            except Exception as e:  # one bad line must not kill the model's other lines
                log.exception(f"[{model_key}] line {i} failed")
                results.append(tts_core.AudioResult(model_key, text, error=repr(e)))
                items.append(CloneItem(text=text, error=repr(e)))

            await flyte.report.replace.aio(
                tts_core.render_grid(texts[: i + 1], [spec], results,
                                     title=f"{model_key} · cloning", meta=meta))
            await flyte.report.flush.aio()
    finally:
        try:
            if handle is not None:
                tts_core.close_model(spec, handle)
        except Exception:
            log.exception(f"[{model_key}] close_model failed")
        handle = None
        tts_core.free_gpu_memory()

    clips = await flyte.io.Dir.from_local(str(out_dir))
    return CloneRun(model_key=model_key, items=items, clips=clips)


# ── One task per adapter (one image/env each). Body is shared via _run_clone ──────
#
# Four, not seven: Kokoro and Parler cannot clone, and Voxtral is deferred. Both Qwen
# -Base checkpoints flow through clone_qwen, exactly as the compare demo's two Qwen
# models share generate_qwen.

@GPU_ENVS["qwen"].task(report=True, retries=1)
async def clone_qwen(model_key: str, weights: flyte.io.Dir, texts: list[str],
                     ref_audio: flyte.io.File, ref_text: str) -> CloneRun:
    return await _run_clone(model_key, weights, texts, ref_audio, ref_text)


@GPU_ENVS["chatterbox"].task(report=True, retries=1)
async def clone_chatterbox(model_key: str, weights: flyte.io.Dir, texts: list[str],
                           ref_audio: flyte.io.File, ref_text: str) -> CloneRun:
    return await _run_clone(model_key, weights, texts, ref_audio, ref_text)


@GPU_ENVS["dia"].task(report=True, retries=1)
async def clone_dia(model_key: str, weights: flyte.io.Dir, texts: list[str],
                    ref_audio: flyte.io.File, ref_text: str) -> CloneRun:
    return await _run_clone(model_key, weights, texts, ref_audio, ref_text)


@GPU_ENVS["csm"].task(report=True, retries=1)
async def clone_csm(model_key: str, weights: flyte.io.Dir, texts: list[str],
                    ref_audio: flyte.io.File, ref_text: str) -> CloneRun:
    return await _run_clone(model_key, weights, texts, ref_audio, ref_text)


CLONE_TASKS = {
    "qwen": clone_qwen,
    "chatterbox": clone_chatterbox,
    "dia": clone_dia,
    "csm": clone_csm,
}


# ── Scoring: every clip from every model, through one scorer ─────────────────────

@metrics_env.task(report=True, retries=1)
async def score_clones(ref_audio: flyte.io.File, ref_text: str,
                       runs: list[CloneRun]) -> ScoreReport:
    """Speaker similarity + WER for every clip, plus the reference-vs-itself ceiling.

    Runs after all the generation tasks so both scoring models load exactly once, and
    so every clip is measured by the same weights: comparing a similarity from one
    scorer against a similarity from another would be meaningless.
    """
    out = ScoreReport()
    try:
        ref = await _load_ref(ref_audio, ref_text)
        scorer = metrics.Scorer().load()
        scorer.set_reference(ref.wav, ref.sample_rate)

        # The ceiling. Scoring the reference against itself would trivially return 1.0
        # and tell you nothing, so we split it in half and score half A against half B:
        # the same person, different audio, through the identical embedding path. That
        # is the realistic top of this scale, and every model bar is read against it.
        half = ref.wav.size // 2
        if half > ref.sample_rate // 2:      # need >0.5s per half to mean anything
            import torch
            a = scorer.embed(ref.wav[:half], ref.sample_rate)
            b = scorer.embed(ref.wav[half:], ref.sample_rate)
            out.ceiling = float(torch.nn.CosineSimilarity(dim=-1)(a[0], b[0]))
            log.info(f"reference self-similarity (ceiling): {out.ceiling:.3f}")

        for run in runs:
            # A model that failed every line returns an empty Dir, and downloading an
            # empty Dir raises in the parent, so check before touching it.
            usable = [it for it in run.items if it.filename and not it.error]
            if not run.clips or not usable:
                out.clips.extend(
                    ClipScore(run.model_key, it.text, error=it.error or "no output")
                    for it in run.items)
                continue

            local = Path(await run.clips.download())
            for it in run.items:
                if it.error or not it.filename:
                    out.clips.append(ClipScore(run.model_key, it.text,
                                               error=it.error or "no output"))
                    continue
                try:
                    wav, sr = sf.read(str(local / it.filename), dtype="float32")
                    s = scorer.score(tts_core.to_mono_float32(wav), int(sr), it.text)
                    out.clips.append(ClipScore(
                        run.model_key, it.text, similarity=s.similarity, wer=s.wer,
                        transcript=s.transcript, error=s.error))
                    log.info(f"[{run.model_key}] sim={s.similarity:.3f} wer={s.wer:.3f}")
                except Exception as e:
                    out.clips.append(ClipScore(run.model_key, it.text, error=repr(e)))
    except Exception as e:      # scoring must never take the audio down with it
        log.exception("scoring failed")
        out.error = repr(e)
    return out


# ── Re-derive report objects from a run's wavs, in the parent ────────────────────

async def _to_results(spec, run: CloneRun) -> list[tts_core.AudioResult]:
    if not run.clips or not any(it.filename and not it.error for it in run.items):
        return [tts_core.AudioResult(spec.key, it.text, error=it.error or "no output")
                for it in run.items]

    local = Path(await run.clips.download())
    out: list[tts_core.AudioResult] = []
    for it in run.items:
        if it.error or not it.filename:
            out.append(tts_core.AudioResult(spec.key, it.text, error=it.error or "no output"))
            continue
        try:
            wav, sr = sf.read(str(local / it.filename), dtype="float32")
            out.append(tts_core.build_audio_result(spec, it.text, wav, sr, it.seconds, it.peak_gb))
        except Exception as e:
            out.append(tts_core.AudioResult(spec.key, it.text,
                                            error=f"could not read {it.filename}: {e}"))
    return out


# ── The orchestrator ─────────────────────────────────────────────────────────────

@clone_orch_env.task(report=True)
async def clone(
    ref_audio: flyte.io.File,
    ref_text: str,
    texts: list[str] | None = None,
    suite: str = "clone",
    models: list[str] | None = None,
) -> ScoreReport:
    """Clone one voice across the cloning models, score every clip, render one report.

    `ref_audio` is a local wav path on the CLI (flyte run uploads it) and `ref_text` is
    its exact transcript: Qwen, Dia and CSM all condition on the transcript, so a wrong
    one degrades three of the five models. `texts` wins over `suite`.

    Returns the ScoreReport rather than the runs, because the scores ARE the result
    here; the audio lives in the report.
    """
    # Rejects a non-cloner before any fetch or GPU pod (see resolve_cloners).
    specs = resolve_cloners(models)
    if texts is None:
        texts = get_suite(suite)

    await flyte.report.replace.aio(tts_core.render_status(
        "Voice cloning comparison",
        f"{len(specs)} models × {len(texts)} scripts, from one reference clip. "
        f"Fetching weights, then cloning, then scoring. "
        f"Models: {', '.join(s.key for s in specs)}."))
    await flyte.report.flush.aio()

    # Fetch serially (parallel HF pulls just fight for the uplink), tolerating a
    # per-model failure so a gated repo becomes an error column, not a dead run.
    weights: dict[str, flyte.io.Dir] = {}
    fetch_errors: dict[str, str] = {}
    for s in specs:
        try:
            weights[s.key] = await fetch_weights.override(short_name=f"fetch {s.key}")(s.key)
        except Exception as e:
            log.exception(f"[{s.key}] fetch failed")
            fetch_errors[s.key] = f"weights fetch failed: {e}"

    # Clone in parallel; one GPU means the scheduler serializes them anyway.
    launch = [s for s in specs if s.key in weights]
    raw = await asyncio.gather(*[
        CLONE_TASKS[s.adapter].override(short_name=f"clone {s.key}")(
            s.key, weights[s.key], texts, ref_audio, ref_text)
        for s in launch
    ], return_exceptions=True)

    runs: list[CloneRun] = []
    all_results: list[tts_core.AudioResult] = []
    done = {s.key: res for s, res in zip(launch, raw)}
    for s in specs:
        res = done.get(s.key)
        if s.key in fetch_errors:
            all_results.extend(tts_core.AudioResult(s.key, t, error=fetch_errors[s.key])
                               for t in texts)
        elif isinstance(res, Exception):
            all_results.extend(tts_core.AudioResult(s.key, t, error=f"clone failed: {res}")
                               for t in texts)
        else:
            runs.append(res)
            all_results.extend(await _to_results(s, res))

    # Score everything that survived, in one pass, with one scorer.
    report = ScoreReport()
    if runs:
        try:
            report = await score_clones.override(short_name="score")(ref_audio, ref_text, runs)
        except Exception as e:
            log.exception("scoring task failed")
            report = ScoreReport(error=repr(e))

    # The watermark verdict lives on the generation items, the scores on the score
    # report; join them here for the renderer.
    marks = {(it.text, r.model_key): it.watermark for r in runs for it in r.items}
    scores: dict[tuple[str, str], tts_core.CloneScore] = {}
    for c in report.clips:
        mark = marks.get((c.text, c.model_key), "")
        scores[(c.text, c.model_key)] = tts_core.CloneScore(
            similarity=c.similarity, wer=c.wer, transcript=c.transcript,
            watermarked=None if mark == "" else (mark == "yes"),
            error=c.error,
        )

    # Rebuild the reference for the report header (the orchestrator never loaded it).
    # build_audio_result only reads spec.key, so a stand-in spec keyed "reference" is
    # enough and keeps the reference from being labelled with some model's name.
    ref_result, ref_warnings = None, []
    try:
        ref = await _load_ref(ref_audio, ref_text)
        ref_warnings = ref.warnings()
        ref_result = tts_core.build_audio_result(
            replace(specs[0], key="reference"), ref_text, ref.wav, ref.sample_rate, 0.0)
    except Exception:
        log.exception("could not load the reference for the report header")

    note = f" · scoring error: {report.error}" if report.error else ""
    meta = (f"{len(specs)} models · {len(texts)} scripts · one reference voice · "
            f"similarity vs WER, same scorer for every clip{note}")
    await flyte.report.replace.aio(tts_core.render_clone_grid(
        texts, specs, all_results, scores,
        ref_result=ref_result, ref_transcript=ref_text, ref_warnings=ref_warnings,
        ceiling=report.ceiling,
        title="Voice cloning: one reference, five models", meta=meta))
    await flyte.report.flush.aio()
    return report


@clone_orch_env.task(report=True)
async def clone_one(
    ref_audio: flyte.io.File,
    ref_text: str,
    model_key: str = "chatterbox",
    text: str = "I was going to call you back, but the whole afternoon got away from me.",
) -> CloneRun:
    """Single-model smoke test: does this model clone this voice at all?

    Fetch and clone one line, NO scoring: it deliberately skips the scoring task so it
    stays the cheap 'does this load and produce audio' check. Mirrors
    compare_pipeline.generate_one. Use `clone` when you want the numbers.
    """
    spec = get_spec(model_key)
    if not spec.clone_capable:
        raise ValueError(f"{model_key} cannot clone. See models.CLONE_MODELS.")
    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)
    return await CLONE_TASKS[spec.adapter].override(short_name=f"clone {model_key}")(
        model_key, w, [text], ref_audio, ref_text)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    here = pathlib.Path(__file__).parent
    wav, txt = here / "refs" / "sage.wav", here / "refs" / "sage.txt"
    if not wav.exists():
        raise SystemExit(f"no reference clip at {wav}; record one first (see the README)")
    # from_local_sync, not from_local: we're outside the event loop here.
    run = flyte.run(clone, ref_audio=flyte.io.File.from_local_sync(str(wav)),
                    ref_text=txt.read_text().strip(), suite="clone-quick")
    print(f"Clone run: {run.name}")
    print(f"  {run.url}")
