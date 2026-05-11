# Gemma 4 Voice Assistant

Speak into your mic → STT → Gemma 4 thinks → Edge TTS speaks the reply.

**Two ways to transcribe, pickable at runtime from a dropdown:**

1. **Whisper** (faster-whisper, local CTranslate2) — the classic baseline.
2. **Gemma 4 itself** — the E2B and E4B edge variants ship with a native audio
   encoder, so the same model family that generates the reply can also decode
   your voice. No separate STT model in the stack.

Switch between them mid-session to A/B quality and latency without a restart.

## Stack

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base.en`, `small.en`, `medium.en`) **or** Gemma 4 native audio (`gemma4:e2b`, `gemma4:e4b`) via Ollama
- **LLM**: Gemma 4 via Ollama
- **TTS**: [edge-tts](https://github.com/rany2/edge-tts) (Microsoft Edge TTS, free, online, no API key)

## Setup

```bash
cd topics/gemma4/voice

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

You also need Ollama running with a Gemma 4 model pulled:

```bash
ollama serve &
ollama pull gemma4:31b          # reasoning
ollama pull gemma4:e4b          # optional — native-audio STT
```

## Run

```bash
python app.py
# -> http://localhost:7865
```

First run downloads the Whisper model (~140MB for `base.en`).

### Public URL (remote / forwarded-port setups)

Set `GRADIO_SHARE=1` for a public HTTPS tunnel via Gradio's servers (link good for 72 hours). Useful when SSH port-forwarding is flaky from a remote dev box — and necessary for the mic to work in most remote setups, since browsers block `getUserMedia` on non-HTTPS origins.

```bash
GRADIO_SHARE=1 python app.py
```

## Gemma 4 as STT

Pick `gemma4:e2b` or `gemma4:e4b` in the **Speech-to-text** dropdown and
Whisper is skipped entirely — the Gemma model decodes the audio itself.
Implementation note: Ollama currently routes multimodal bytes through the
`images` field regardless of modality, so the app base64-encodes the WAV and
sends it that way; the audio encoder inside the model handles it.

Only the E2B and E4B edge variants have the audio encoder. The larger
`gemma4:26b` / `gemma4:31b` are text+image only and won't work as STT.

You can also mix: pick `gemma4:e4b` for STT and `gemma4:31b` for reasoning to
get native audio decoding *and* the bigger model's replies.

Caveat: the Ollama runner currently has an
[intermittent crash on sustained audio requests](https://github.com/ollama/ollama/issues/15333)
(roughly every 2–4 turns). If it dies mid-conversation, restart `ollama serve`
and try again, or fall back to Whisper for the rest of the session.

## Swappables

```bash
GEMMA_MODEL=gemma4:e4b python app.py      # smaller/faster LLM
WHISPER_SIZE=small.en python app.py       # default STT on launch (overridable in UI)
WHISPER_DEVICE=cuda python app.py         # run whisper on GPU
TTS_VOICE=en-GB-RyanNeural python app.py  # different voice
```

## Fully-local TTS

`edge-tts` needs the internet. To go fully offline, swap the `synthesize()`
function for [kokoro](https://github.com/thewh1teagle/kokoro-onnx) or
[piper-tts](https://github.com/rhasspy/piper). Both need `espeak-ng` installed
at the system level:

```bash
sudo apt install espeak-ng
```
