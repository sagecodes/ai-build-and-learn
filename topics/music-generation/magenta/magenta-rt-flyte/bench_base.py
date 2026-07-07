"""Measure steady-state real-time factor (RTF) for a Magenta RT2 model.

RTF = seconds of compute per second of audio. < 1.0 keeps ahead of playback.
Set XLA_FLAGS before running to test perf knobs, e.g.:
    XLA_FLAGS="--xla_gpu_graph_min_graph_size=1" MRT_BENCH_SIZE=mrt2_base python bench_base.py
"""

from __future__ import annotations

import os
import time

import jax

from mrt_core import MODELS, ensure_weights

SIZE = os.environ.get("MRT_BENCH_SIZE", "mrt2_base")
FRAMES = int(os.environ.get("MRT_FRAMES_PER_CHUNK", "50"))  # 50 = 2s @ 25fps
WARMUP = 3
MEASURE = 6


def main():
    os.environ.setdefault("MAGENTA_HOME", "/tmp/magenta")
    print(f"jax backend={jax.default_backend()} devices={jax.devices()}", flush=True)
    print(f"XLA_FLAGS={os.environ.get('XLA_FLAGS','')!r}", flush=True)

    ensure_weights(MODELS[SIZE])
    from magenta_rt import MagentaRT2Jax

    t0 = time.time()
    mrt = MagentaRT2Jax(size=SIZE, checkpoint=MODELS[SIZE],
                        temperature=1.3, top_k=40, cfg_musiccoca=3.0, cfg_notes=1.0)
    print(f"loaded {SIZE} in {time.time()-t0:.1f}s", flush=True)

    emb = mrt.embed_style("driving techno, acid bassline", use_mapper=True)
    audio_sec = FRAMES / 25.0

    st = None
    for _ in range(WARMUP):
        _, st = mrt.generate(style=emb, frames=FRAMES, state=st)  # generate() syncs (device_get)

    times = []
    for _ in range(MEASURE):
        t = time.time()
        _, st = mrt.generate(style=emb, frames=FRAMES, state=st)
        times.append(time.time() - t)

    times.sort()
    med = times[len(times) // 2]
    print(f"\n=== {SIZE}: {audio_sec:g}s chunk ===", flush=True)
    print(f"per-chunk seconds (sorted): {[round(x,3) for x in times]}", flush=True)
    print(f"median chunk: {med:.3f}s  ->  RTF={med/audio_sec:.3f} "
          f"({'REAL-TIME OK' if med < audio_sec else 'TOO SLOW'})", flush=True)


if __name__ == "__main__":
    main()
