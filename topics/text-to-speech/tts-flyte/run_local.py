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
@click.option("--out", default="./downloads", help="Where to write the .wav and report.")
def main(model_key, text, voice, out):
    """Synthesize one line on the host GPU and write a .wav + an .html report."""
    from dataclasses import replace
    spec = get_spec(model_key)
    if voice:
        spec = replace(spec, voice=voice, voice_label=voice)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[{model_key}] {spec.repo}  ({spec.params}, {spec.license})")
    click.echo(f"  ~{spec.download_gb:.1f}GB download (cached after the first run)")

    import tts_core

    prepare_gpu()
    t0 = time.time()
    handle = tts_core.load_model(spec)
    click.echo(f"  loaded in {time.time() - t0:.0f}s")

    try:
        reset_peak_memory()
        wav, sr, secs = tts_core.synth_one(handle, spec, text)
        peak = peak_memory_gb()
    finally:
        handle = None
        free_gpu_memory()

    wav_path = out_dir / f"{model_key}.wav"
    write_wav(wav, sr, wav_path)

    r = build_audio_result(spec, text, wav, sr, secs, peak)
    html_path = out_dir / f"{model_key}.html"
    html_path.write_text(render_grid(
        [text], [spec], [r],
        title=f"{model_key} (host GPU)",
        meta=f"{spec.repo} · {secs:.1f}s synth · {r.audio_seconds:.1f}s audio "
             f"({r.speedup:.1f}x real-time)"))

    click.echo(f"\n  {secs:.1f}s to synth · {r.audio_seconds:.1f}s audio "
               f"· {r.speedup:.1f}x real-time · peak {peak:.1f}GB")
    click.echo(f"  wav:    {wav_path}")
    click.echo(f"  report: {html_path}  (open it in a browser; the clip plays inline)")


if __name__ == "__main__":
    main()
