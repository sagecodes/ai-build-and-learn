"""Run a video model directly on the host GPU, no Flyte, no cluster.

The fastest way to answer "does this model even load on this box?" before you
spend an hour on a cluster round-trip. Writes an .mp4 plus a standalone .html
report (the same renderer the Flyte report uses, so if playback works here it
works there).

    # smallest model, one clip
    python run_local.py --model wan21-t2v-1.3b \
        --prompt "a red panda barista pouring latte art, steam rising"

    # the one with audio (95GB first download)
    python run_local.py --model ltx2-distilled --prompt "rain on a tin roof, cozy"

    # trim it down further if the box is busy
    python run_local.py --model wan21-t2v-1.3b --prompt "..." --steps 15 --num-frames 25

Weights land in the normal HF cache (~/.cache/huggingface), so this shares
downloads with anything else on the host, but NOT with the Flyte tasks (those
cache their own copy in the blob store).
"""

from __future__ import annotations

import time
from pathlib import Path

import click

from models import MODELS, get_spec
from videogen_core import (
    build_clip_result,
    free_gpu_memory,
    prepare_gpu,
    render_grid,
    timed_generate,
    write_mp4,
)


@click.command()
@click.option("--model", "model_key", default="wan21-t2v-1.3b",
              type=click.Choice(list(MODELS)), help="Which video model to run.")
@click.option("--prompt", required=True, help="What to generate.")
@click.option("--steps", default=-1, type=int, help="-1 = the model's default.")
@click.option("--num-frames", default=-1, type=int, help="-1 = the model's default.")
@click.option("--width", default=-1, type=int)
@click.option("--height", default=-1, type=int)
@click.option("--seed", default=1234, type=int)
@click.option("--out", default="./downloads", help="Where to write the .mp4 and report.")
def main(model_key, prompt, steps, num_frames, width, height, seed, out):
    """Generate one clip on the host GPU and write an mp4 + an html report."""
    spec = get_spec(model_key)
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)

    click.echo(f"[{model_key}] {spec.repo}")
    click.echo(f"  ~{spec.download_gb:.0f}GB download (cached after the first run), "
               f"~{spec.est_vram_gb:.0f}GB resident in bf16")

    # Same guardrails the Flyte task uses: clean the pool, cap this process so an
    # overshoot RAISES instead of locking the box, and refuse a model that can't fit.
    prepare_gpu(spec)

    from videogen_core import load_pipeline

    t0 = time.time()
    pipe = load_pipeline(spec, model_path=None)
    click.echo(f"  loaded in {time.time() - t0:.0f}s")

    def _on_step(k, total):
        click.echo(f"  step {k}/{total}", nl=False)
        click.echo("\r", nl=False)

    try:
        frames, audio, sr, secs = timed_generate(
            pipe, spec, prompt,
            steps=None if steps < 0 else steps,
            num_frames=None if num_frames < 0 else num_frames,
            width=None if width < 0 else width,
            height=None if height < 0 else height,
            seed=seed, on_step=_on_step,
        )
    finally:
        pipe = None
        free_gpu_memory()

    mp4 = out_dir / f"{model_key}.mp4"
    write_mp4(frames, spec.fps, mp4, audio=audio, sample_rate=sr)

    r = build_clip_result(spec, prompt, frames, audio, sr, secs)
    html_path = out_dir / f"{model_key}.html"
    html_path.write_text(render_grid(
        [prompt], [spec], [r],
        title=f"{model_key} (host GPU)",
        meta=f"{spec.repo} · seed={seed} · {secs:.0f}s · peak {r.peak_gb:.0f}GB",
    ))

    click.echo(f"\n  {secs:.0f}s to generate · peak {r.peak_gb:.0f}GB GPU")
    click.echo(f"  clip:   {mp4}")
    click.echo(f"  report: {html_path}  (open it in a browser; the clip plays inline)")
    if audio is not None:
        click.echo("  the clip has a generated audio track; unmute the player to hear it")


if __name__ == "__main__":
    main()
