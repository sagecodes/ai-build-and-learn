"""Run one TTS model directly on the host GPU, no Flyte, no cluster.

The fastest way to answer "does this model even load and speak on this box?" before a
cluster round-trip. Writes a .wav plus a standalone .html report (the SAME renderer
the Flyte report uses, so if the audio plays here it plays there).

    # the flagship, one line
    python run_local.py --model qwen3-1.7b \
        --text "The morning light came slow and gold across the harbor."

    # the tiny fast one
    python run_local.py --model kokoro-82m --text "Hello from a very small model."

    # the dialogue model on a two-speaker line
    python run_local.py --model dia-1.6b \
        --text "[S1] Did it work? [S2] It worked on the first try. (laughs)"

CAVEAT: only ONE model's package fits in a given Python env (that's the whole reason
the Flyte demo uses per-adapter images). So a local venv can drive whichever model's
package you've installed; it can't drive all five at once. Install per model, e.g.
`pip install qwen-tts` for the Qwen models, `pip install kokoro`, etc.

Weights land in the normal HF cache (~/.cache/huggingface), shared with anything else
on the host but NOT with the Flyte tasks (those cache their own copy in the blob store).
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from models import MODELS, get_spec
from tts_core import (
    build_audio_result,
    free_gpu_memory,
    peak_memory_gb,
    prepare_gpu,
    render_grid,
    reset_peak_memory,
    write_wav,
)


@click.command()
@click.option("--model", "model_key", default="qwen3-1.7b",
              type=click.Choice(list(MODELS)), help="Which TTS model to run.")
@click.option("--text", required=True, help="What to say.")
@click.option("--voice", default="", help="Override the default voice (Kokoro pack / "
              "Qwen speaker / Parler description). Empty = the model's default.")
@click.option("--ref-audio", default="", type=click.Path(exists=True, dir_okay=False),
              help="Reference wav: switches this to VOICE CLONING. Needs a clone-capable "
                   "model (see models.CLONE_MODELS).")
@click.option("--ref-text", default="", help="Exact transcript of --ref-audio. Defaults "
              "to the sibling .txt (refs/sage.wav -> refs/sage.txt). Qwen, Dia and CSM "
              "all condition on it.")
@click.option("--out", default="./downloads", help="Where to write the .wav and report.")
def main(model_key, text, voice, ref_audio, ref_text, out):
    """Synthesize one line on the host GPU and write a .wav + an .html report."""
    from dataclasses import replace
    spec = get_spec(model_key)
    if voice:
        spec = replace(spec, voice=voice, voice_label=voice)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    import tts_core

    # Cloning mode. Resolve the transcript before loading 5GB of weights, so a missing
    # one is an instant error rather than one discovered two minutes in.
    ref = None
    if ref_audio:
        if not spec.clone_capable:
            raise click.ClickException(
                f"{model_key} cannot clone. Clone-capable: "
                + ", ".join(k for k, s in MODELS.items() if s.clone_capable))
        sidecar = Path(ref_audio).with_suffix(".txt")
        if not ref_text and sidecar.exists():
            ref_text = sidecar.read_text().strip()
        if not ref_text and spec.adapter != "chatterbox":
            raise click.ClickException(
                f"{model_key} conditions on the reference transcript; pass --ref-text "
                f"or put it in {sidecar}.")
        ref = tts_core.RefVoice.from_file(ref_audio, ref_text)
        click.echo(f"  cloning {ref.seconds:.1f}s reference @ {ref.sample_rate}Hz")
        for w in ref.warnings():
            click.echo(f"  ! {w}")

    click.echo(f"[{model_key}] {spec.repo}  ({spec.params}, {spec.license})")
    click.echo(f"  ~{spec.download_gb:.1f}GB download (cached after the first run)")

    prepare_gpu()
    t0 = time.time()
    handle = tts_core.load_model(spec)
    click.echo(f"  loaded in {time.time() - t0:.0f}s")

    try:
        reset_peak_memory()
        if ref is not None:
            wav, sr, secs = tts_core.synth_clone(handle, spec, text, ref)
        else:
            wav, sr, secs = tts_core.synth_one(handle, spec, text)
        peak = peak_memory_gb()
    finally:
        handle = None
        free_gpu_memory()

    stem = f"{model_key}-clone" if ref is not None else model_key
    wav_path = out_dir / f"{stem}.wav"
    write_wav(wav, sr, wav_path)

    r = build_audio_result(spec, text, wav, sr, secs, peak)
    html_path = out_dir / f"{stem}.html"
    html_path.write_text(render_grid(
        [text], [spec], [r],
        title=f"{stem} (host GPU)",
        meta=f"{spec.repo} · {secs:.1f}s synth · {r.audio_seconds:.1f}s audio "
             f"({r.speedup:.1f}x real-time)"))

    click.echo(f"\n  {secs:.1f}s to synth · {r.audio_seconds:.1f}s audio "
               f"· {r.speedup:.1f}x real-time · peak {peak:.1f}GB")
    click.echo(f"  wav:    {wav_path}")
    click.echo(f"  report: {html_path}  (open it in a browser; the clip plays inline)")

    # Score it too, if the scorers happen to be importable here. Only transformers is
    # needed, which the transformers-based venvs (Dia, CSM) already have, so this often
    # works for free; it stays optional because the Qwen and Chatterbox venvs pin their
    # own transformers and loading Whisper there is not worth a dependency fight.
    if ref is not None:
        try:
            import metrics
            scorer = metrics.Scorer().load()
            scorer.set_reference(ref.wav, ref.sample_rate)
            sc = scorer.score(wav, sr, text)
            click.echo(f"  similarity: {sc.similarity:.3f} · WER {sc.wer * 100:.1f}% "
                       f"· heard: “{sc.transcript}”")
        except Exception as e:
            click.echo(f"  (scoring skipped: {type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
