# Open-source TTS, side by side (on Flyte + DGX Spark)

Read the same script through several state-of-the-art open TTS models and get back a
single Flyte report where every clip **plays inline**, next to a waveform and
spectrogram, with speed (real-time factor) on each card. The point is to *hear* them
against each other, on the same words, in one place.

This is the audio sibling of the `video-generation/videogen-flyte` demo and shares its
shape: cached HF downloads on CPU pods, GPU synthesis tasks, one aggregated report.

## The models

All commercial-safe (Apache-2.0 / MIT). Same text, each in its own default voice.

| key | model | params | license | why it's here |
|---|---|---|---|---|
| `qwen3-1.7b` | Qwen3-TTS 1.7B | 1.7B | Apache-2.0 | Alibaba's Jan-2026 flagship: expressive, multilingual. The one to beat. |
| `qwen3-0.6b` | Qwen3-TTS 0.6B | 0.6B | Apache-2.0 | Same family, a third the size. The size/speed contrast. |
| `chatterbox` | Chatterbox | 0.5B | MIT | Beat ElevenLabs in a blind test. The natural-speech bar. |
| `kokoro-82m` | Kokoro-82M | 82M | Apache-2.0 | Tiny, CPU-capable, 54 fixed voices. The efficiency floor. |
| `dia-1.6b` | Dia 1.6B | 1.6B | Apache-2.0 | Best open **dialogue** voice; shines on the two-speaker script. |

Voice cloning is a **separate task** (it needs a reference clip and a different
fairness setup), so there's none of that plumbing here. This is pure generation.

## The one thing that's different from the video demo: per-model images

The video models all loaded through `diffusers`, so they shared one image. TTS models
do not: each ships its own package with mutually hostile pins.

```
qwen-tts    pins transformers==4.57.3, accelerate==1.12.0
chatterbox  pins transformers==5.2.0, torch==2.6.0, diffusers==0.29.0, numpy<2
kokoro      wants misaki + espeak-ng
dia         via transformers (DiaForConditionalGeneration)
```

So every adapter gets its **own image and its own GPU `TaskEnvironment`**, and the
orchestrator dispatches each model to the task whose image has its package
(`config.GPU_ENVS`, `compare_pipeline.GEN_TASKS`). The two Qwen models share the one
`qwen` image. Chatterbox's `torch==2.6.0` pin has no cu130 arm64 wheel, so it's
installed `--no-deps` on top of the Spark's cu130 torch with its real deps by hand.

## Run it

On the DGX Spark devbox (not local Python; images build on the cluster):

```bash
# default: all 5 models × the 5-script suite
flyte run compare_pipeline.py compare

# quick 3-script pass on a couple of models
flyte run compare_pipeline.py compare --suite quick --models '["qwen3-1.7b","kokoro-82m"]'

# your own lines
flyte run compare_pipeline.py compare --texts '["The quick brown fox jumps over the lazy dog."]'

# single-model smoke test (does it even load?)
flyte run compare_pipeline.py generate_one --model_key dia-1.6b
```

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

One entry in `models.py` (`key`, `repo`, `adapter`, defaults). If it needs a new
package, add an adapter branch in `tts_core.load_model`/`synth_one`, an image + GPU env
in `config.py`, and a `generate_*` task in `compare_pipeline.py`. Candidates parked for
later: Voxtral (Mistral, needs a vLLM server), CosyVoice2, IndexTTS-2, VibeVoice.
