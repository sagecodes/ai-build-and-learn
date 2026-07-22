# Open-source TTS, side by side (on Flyte + DGX Spark)

Read the same script through several state-of-the-art open TTS models and get back a
single Flyte report where every clip **plays inline**, next to a waveform and
spectrogram, with speed (real-time factor) on each card. The point is to *hear* them
against each other, on the same words, in one place.

This is the audio sibling of the `video-generation/videogen-flyte` demo and shares its
shape: cached HF downloads on CPU pods, GPU synthesis tasks, one aggregated report.

## The models

Seven models, all commercial-safe (Apache-2.0 / MIT). Same text, each in its own
built-in voice(s).

| key | model | params | license | why it's here |
|---|---|---|---|---|
| `qwen3-1.7b` | Qwen3-TTS 1.7B | 1.7B | Apache-2.0 | Alibaba's Jan-2026 flagship: expressive, multilingual. The one to beat. |
| `qwen3-0.6b` | Qwen3-TTS 0.6B | 0.6B | Apache-2.0 | Same family, a third the size. The size/speed contrast. |
| `chatterbox` | Chatterbox | 0.5B | MIT | Beat ElevenLabs in a blind test. The natural-speech bar. |
| `kokoro-82m` | Kokoro-82M | 82M | Apache-2.0 | Tiny, CPU-capable, 54 fixed voices. The efficiency floor. |
| `dia-1.6b` | Dia 1.6B | 1.6B | Apache-2.0 | Best open **dialogue** voice; shines on the two-speaker script. |
| `parler-mini` | Parler-TTS mini v1 | 880M | Apache-2.0 | Voice controlled by a text **description**. A different UX entirely. |
| `csm-1b` | Sesame CSM-1B | 1B | Apache-2.0 | Very natural conversational voice. **Gated** (see below). |

`csm-1b` is a **gated** repo: accept the license once at
[hf.co/sesame/csm-1b](https://huggingface.co/sesame/csm-1b) with the account tied to
your `HF_TOKEN`, or its fetch 403s (the run degrades to an error column, it doesn't crash).

### Male / female voices

Models with named built-in voices render M/F as separate columns. `--voices` picks:
`all` (every variant), `female`, `male`, or `default` (one column per model).

| model | female | male |
|---|---|---|
| `kokoro-82m` | af_heart | am_michael |
| `qwen3-1.7b` | Vivian | Ryan |
| `qwen3-0.6b` | Serena | Eric |
| `parler-mini` | "Laura's voice…" | "Gary's voice…" |

Chatterbox, Dia and CSM are zero-shot / random-speaker models: they have no named
built-in voices, so controllable M/F needs a reference clip, which is the **voice
cloning** task (separate, not built here). This one is pure generation.

## The one thing that's different from the video demo: per-model images

The video models all loaded through `diffusers`, so they shared one image. TTS models
do not: each ships its own package with mutually hostile pins.

```
qwen-tts    pins transformers==4.57.3, accelerate==1.12.0
chatterbox  pins transformers==5.2.0, torch==2.6.0, diffusers==0.29.0, numpy<2
kokoro      wants misaki + espeak-ng
dia + csm   via transformers (Dia/CsmForConditionalGeneration) — SHARE one image
parler-tts  pins transformers==4.46.1 + old protobuf; needs numba/llvmlite floors
```

So every adapter gets its **own image and its own GPU `TaskEnvironment`**, and the
orchestrator dispatches each model to the task whose image has its package
(`config.GPU_ENVS`, `compare_pipeline.GEN_TASKS`). The two Qwen models share the `qwen`
image; Dia and CSM share the `transformers` image (built once). Three images needed
hand-work, all isolated to that one image and documented in `config.py`:

- **Chatterbox** `torch==2.6.0` has no cu130 arm64 wheel, so it's installed `--no-deps`
  on the Spark's cu130 torch with its real deps by hand.
- **Parler** backtracks to `llvmlite==0.36.0` (no py3.12 wheel) without a `numba`/
  `llvmlite` floor; and its `descript-audiotools` caps `protobuf<5`, which the Flyte
  runtime (needs `>=6.30.1` to serialize task outputs) can't live with, so a final layer
  forces protobuf back up.

## Run it

On the DGX Spark devbox (not local Python; images build on the cluster):

```bash
# default: all 7 models, M+F voice columns, the 5-script suite
flyte run compare_pipeline.py compare

# one column per model (skip the M/F expansion)
flyte run compare_pipeline.py compare --voices default

# only the female voices, quick 3-script pass on a couple of models
flyte run compare_pipeline.py compare --suite quick --voices female \
    --models '["qwen3-1.7b","kokoro-82m"]'

# your own lines
flyte run compare_pipeline.py compare --texts '["The quick brown fox jumps over the lazy dog."]'

# single-model smoke test (does it even load?), optionally a specific voice
flyte run compare_pipeline.py generate_one --model_key kokoro-82m --voice am_michael
```

> Note: `flyte run` needs its own local `.venv` (flyte + a few light deps) because it
> imports the task module on the host to discover tasks. Drive runs with
> `.venv/bin/flyte run …`. The heavy model packages stay lazy inside the adapters, so
> they are only ever installed in the per-adapter images, never locally.

Open the run's **Report** tab. Scripts are rows, models are columns; play a row
top-to-bottom to compare the same words across models.

### Fast local iteration

`run_local.py` drives one model on the host GPU (no cluster) and writes a `.wav` plus a
standalone `.html` using the identical renderer, so if it plays there it plays in the
Flyte report. Note only one model's package fits a given venv (the reason the cluster
uses per-adapter images), so install per model:

```bash
pip install qwen-tts && python run_local.py --model qwen3-1.7b --text "Hello there."
pip install kokoro   && python run_local.py --model kokoro-82m --text "Hello there."
```

## The scripts

Five scripts, each stressing one axis you can only judge by ear (see `prompts.py`):
naturalness/prosody, **text normalization** (numbers, dates, currency, a URL, an
acronym: the clearest quality separator), questions + emphasis, hard phonemes, and a
two-speaker dialogue with a nonverbal (Dia's home turf). For single-voice models the
`[S1]`/`[S2]` tags and `(laughs)` are stripped so they read one clean narrator.

## Files

- `config.py` — Flyte images/envs, per-adapter, Spark-pinned (arm64, cu130, local registry).
- `models.py` — the model registry (`TTSModelSpec`, one `adapter` per model).
- `prompts.py` — the read-aloud scripts and what each one tests.
- `tts_core.py` — Flyte-free: the per-adapter loaders/synth, audio + spectrogram
  embedding, and the HTML report renderer. Shared by the task, run_local, and (later) the app.
- `compare_pipeline.py` — the Flyte tasks: `fetch_weights`, four adapter `generate_*`
  tasks, and the `compare` orchestrator.
- `run_local.py` — one model on the host GPU, for quick checks.

## Adding a model later

One entry in `models.py` (`key`, `repo`, `adapter`, defaults; add a `voices` tuple for
M/F variants). If it needs a new package, add an adapter branch in
`tts_core.load_model`/`synth_one`, an image + GPU env in `config.py`, and a `generate_*`
task in `compare_pipeline.py`. If it reuses an existing package (e.g. another
transformers-native model), point its adapter at the shared image and you skip the new
image. Candidates parked for later: Voxtral (Mistral, needs a vLLM server), CosyVoice2
(painful install: Matcha-TTS + ttsfrd submodules), IndexTTS-2, VibeVoice.
