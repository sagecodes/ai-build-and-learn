"""
Gemma 4 voice assistant: mic → Whisper STT → Gemma 4 → Edge TTS → audio out.

- STT: faster-whisper (local, CTranslate2). Downloads the model on first run.
- LLM: Gemma 4 via Ollama.
- TTS: Microsoft Edge TTS (edge-tts). Online and free, no API key.

Swap TTS for a fully-local one (kokoro, piper) if you want — see README.

Run (after `uv venv` + `uv pip install -r requirements.txt` + activating):
    ollama serve &
    ollama pull gemma4:31b
    python app.py
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path

import edge_tts
import gradio as gr
import ollama
from faster_whisper import WhisperModel

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:31b")
WHISPER_SIZE = os.environ.get("WHISPER_SIZE", "base.en")
TTS_VOICE = os.environ.get("TTS_VOICE", "en-US-AriaNeural")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Keep answers conversational and concise "
    "— typically 1-3 sentences. No markdown, no lists, no code blocks. Just "
    "speak naturally like a person."
)

# Load whisper once. int8 on CPU is fast enough for demo-length utterances;
# if you have a CUDA build of ctranslate2 set WHISPER_DEVICE=cuda.
_whisper: WhisperModel | None = None


def get_whisper() -> WhisperModel:
    global _whisper
    if _whisper is None:
        device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute = "int8" if device == "cpu" else "float16"
        _whisper = WhisperModel(WHISPER_SIZE, device=device, compute_type=compute)
    return _whisper


def list_models() -> list[str]:
    try:
        resp = ollama.list()
        names = sorted(m.model for m in resp.models if m.model.startswith("gemma4"))
        return names or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def transcribe(audio_path: str) -> str:
    segments, _info = get_whisper().transcribe(audio_path, beam_size=1)
    return "".join(seg.text for seg in segments).strip()


async def _tts_to_file(text: str, voice: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize(text: str, voice: str) -> str:
    """Return a path to an mp3 of the spoken text."""
    out = Path(tempfile.mkdtemp()) / "reply.mp3"
    asyncio.run(_tts_to_file(text, voice, str(out)))
    return str(out)


def converse(audio_path: str | None, history: list, model: str, voice: str):
    """One round-trip: transcribe user audio, run LLM, synthesize reply."""
    if not audio_path:
        return history, None, "No audio received."

    user_text = transcribe(audio_path)
    if not user_text:
        return history, None, "Didn't catch that — try again."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})

    resp = ollama.chat(
        model=model, messages=messages,
        options={"temperature": 0.6},
    )
    reply = resp["message"]["content"].strip()

    audio_out = synthesize(reply, voice)
    new_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    transcript = f"**You**: {user_text}\n\n**Gemma**: {reply}"
    return new_history, audio_out, transcript


def clear_history():
    return [], None, ""


def build_ui() -> gr.Blocks:
    models = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    voices = [
        "en-US-AriaNeural", "en-US-GuyNeural", "en-US-JennyNeural",
        "en-GB-RyanNeural", "en-GB-SoniaNeural",
    ]

    with gr.Blocks(title="Gemma 4 Voice") as demo:
        gr.Markdown(
            "# Gemma 4 Voice Assistant\n"
            "Speak → Whisper STT → Gemma 4 → Edge TTS → hear the reply."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="LLM")
            voice = gr.Dropdown(voices, value=TTS_VOICE, label="TTS voice")
            clear = gr.Button("Clear conversation")

        history_state = gr.State([])

        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
                send = gr.Button("Send", variant="primary")
            with gr.Column():
                reply_audio = gr.Audio(label="Reply", autoplay=True)
                transcript = gr.Markdown()

        send.click(
            converse,
            inputs=[mic, history_state, model, voice],
            outputs=[history_state, reply_audio, transcript],
        )
        clear.click(clear_history, outputs=[history_state, reply_audio, transcript])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7865)
