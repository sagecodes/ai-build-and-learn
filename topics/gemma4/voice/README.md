# Gemma 4 Voice Assistant

Speak into your mic → Whisper transcribes → Gemma 4 thinks → Edge TTS speaks the reply.

## Stack

- **STT**: [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`base.en` by default)
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
ollama pull gemma4:31b
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

## Swappables

```bash
GEMMA_MODEL=gemma4:4b python app.py       # smaller/faster LLM
WHISPER_SIZE=small.en python app.py       # larger/slower STT
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
