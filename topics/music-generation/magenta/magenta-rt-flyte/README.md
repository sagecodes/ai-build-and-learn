# Magenta RealTime 2 on Flyte: live, steerable music

Serve Google's [Magenta RealTime 2](https://huggingface.co/google/magenta-realtime-2)
as a Flyte 2 GPU app with a Gradio UI. Press **Start**, type a vibe, and the
track morphs to match without a seam. The weights are pulled from HuggingFace
at startup; nothing is baked into the image.

Apache-2.0 code, CC-BY-4.0 weights.

## What it does

Magenta RT2 generates a *continuous* stream of music in short chunks. Each chunk
is conditioned on:

1. the previous audio (carried forward in a `state` object), and
2. a **style embedding** (a 768-dim MusicCoCa vector from your text prompt).

Steering is just swapping that style embedding between chunks. So a live "tone
shift" is: keep the same `state` (for seamless audio), change the style, and the
next chunk glides toward the new vibe. This app exposes exactly that loop.

```
text prompt ──► MusicCoCa.embed ──► style embedding ┐
                                                      ├─► generate(state, style) ──► 2s chunk ──► browser
previous chunk ───────────────────► state ───────────┘            ▲
                                                                   └── loop, swapping style live
```

## Reality check: real-time on this box

The model card's headline ~200ms real-time streaming is **Apple-Silicon / MLX
only**. On NVIDIA (this DGX Spark, GB10, arm64) we run the **JAX path**: chunked
generation streamed near-real-time. The steering loop is identical; it just is
not guaranteed faster-than-playback, so on a slow chunk the audio may buffer
briefly. Two levers if playback stutters:

- `MRT_SIZE` unset uses `mrt2_small` (230M), the fast default. `MRT_SIZE=base`
  uses `mrt2_base` (2.4B): better sound, heavier.
- `MRT_FRAMES_PER_CHUNK` (default 50 = 2s). Smaller chunks lower latency per
  yield; larger chunks amortize fixed per-call overhead.

## Files

| File         | Purpose                                                        |
|--------------|----------------------------------------------------------------|
| `config.py`  | Image, GPU app env, `HF_TOKEN` secret, model choice.           |
| `mrt_app.py` | Weight download, model load, the live generation loop + UI.    |

## Prerequisites

- A Flyte devbox with a GPU (`flyte start devbox --gpu`).
- An `HF_TOKEN` secret on the devbox with access to `google/magenta-realtime-2`
  (accept the license on the model page first). This is the same secret the
  other demos use.

## Deploy

```bash
cd topics/magenta/magenta-rt-flyte

# default: mrt2_small, localhost-only
python mrt_app.py

# public HTTPS tunnel (needed for a browser on another machine to reach it)
GRADIO_SHARE=1 python mrt_app.py

# bigger model
MRT_SIZE=base python mrt_app.py
```

On first start the app downloads `resources/musiccoca`, `resources/spectrostream`,
and the one checkpoint into `$MAGENTA_HOME/magenta-rt-v2`, then JIT-compiles the
JAX graph (a one-time cost). Watch the pod logs for `[mrt] model ready`.

## Use it

1. **Start** begins the stream (silence until the first chunk lands).
2. Type a vibe in the prompt box and **Apply** (or press Enter).
3. **Morph over N chunks** controls the glide: `0` switches instantly, `4` eases
   over ~8s. The audio `state` is preserved either way, so transitions are
   seamless.
4. Temperature / Top-k / Style-guidance (CFG) tune the sampler live.

Prompt ideas: `driving techno, acid bassline`, `ambient drone, deep reverb`,
`new orleans brass band`, `warm lo-fi hip hop, dusty drums`.

## The one thing to verify first

`magenta-rt[jax]` + `jax[cuda12]` on **arm64 (sbsa)** is the install risk. If the
image build cannot resolve the aarch64 CUDA plugin wheels on the plain debian
base, switch `config.py`'s `from_debian_base` to a CUDA base image (for example
`nvidia/cuda:12.x-cudnn-runtime`), the same way the vLLM app builds on a vLLM
base. A CPU-only smoke test (drop `jax[cuda12]`) proves download + load + UI
before fighting CUDA; it is slow but end-to-end.

## Phase 2 (stretch): webcam vibes via Gemma 4

The plan, not yet wired: point a webcam at the room, ask the existing
`gemma4-dgx-devbox` vLLM app to read the "vibes" of the frame, and feed its
answer into this app's `apply_prompt` path. The webcam + timed-VLM-call
machinery already exists in `topics/gemma4/gemma4-dgx-devbox/live_camera_app.py`
(`gr.Image(sources=["webcam"], streaming=True)` + a `gr.Timer` tick hitting the
vLLM endpoint), so phase 2 is mostly: reuse that capture loop, change the prompt
to "describe the mood of this scene as a short music style", and route the reply
into `apply_prompt` instead of a caption box. The two GPU apps share the single
GB10: Magenta runs the steady loop, Gemma fires in bursts on each frame.

Open item before building it: confirm the deployed Gemma 4 vLLM has vision
enabled in its serving config.
