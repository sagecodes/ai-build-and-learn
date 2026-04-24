"""
Gemma 4 voice assistant: mic → STT (Whisper or Gemma 4 native) → Gemma 4 → Edge TTS.

- STT: choose at runtime — faster-whisper (local, CTranslate2) or Gemma 4's
  native audio encoder (E2B/E4B edge variants, via Ollama).
- LLM: Gemma 4 via Ollama.
- TTS: Microsoft Edge TTS (edge-tts). Online and free, no API key.

Swap TTS for a fully-local one (kokoro, piper) if you want — see README.

Run (after `uv venv` + `uv pip install -r requirements.txt` + activating):
    ollama serve &
    ollama pull gemma4:31b      # reasoning
    ollama pull gemma4:e4b      # optional, for native-audio STT
    python app.py
"""

from __future__ import annotations

import asyncio
import base64
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

WHISPER_PREFIX = "whisper:"
GEMMA_AUDIO_PREFIX = "gemma4:"

# STT models to expose in the UI. Whisper entries are faster-whisper sizes;
# gemma4: entries are Ollama model tags with a native audio encoder (E2B/E4B).
STT_CHOICES = [
    f"{WHISPER_PREFIX}base.en",
    f"{WHISPER_PREFIX}small.en",
    f"{WHISPER_PREFIX}medium.en",
    "gemma4:e2b",
    "gemma4:e4b",
]
DEFAULT_STT = f"{WHISPER_PREFIX}{WHISPER_SIZE}"
if DEFAULT_STT not in STT_CHOICES:
    STT_CHOICES.insert(0, DEFAULT_STT)

GEMMA_STT_PROMPT = (
    "Transcribe the spoken audio verbatim. Output only the transcription text — "
    "no quotes, no commentary, no labels."
)

# Rough chars-per-token heuristic for the thinking-budget cutoff.
CHARS_PER_TOKEN = 3.5

DEFAULT_ROLE = (
    "Role: Helpful voice assistant.\n"
    "Constraints: Conversational, concise (1-3 sentences), no markdown, "
    "no lists, no code blocks, natural speech."
)

# Cache whisper models by size. int8 on CPU is fast enough for demo-length
# utterances; if you have a CUDA build of ctranslate2 set WHISPER_DEVICE=cuda.
_whisper_cache: dict[str, WhisperModel] = {}


def get_whisper(size: str) -> WhisperModel:
    if size not in _whisper_cache:
        device = os.environ.get("WHISPER_DEVICE", "cpu")
        compute = "int8" if device == "cpu" else "float16"
        _whisper_cache[size] = WhisperModel(size, device=device, compute_type=compute)
    return _whisper_cache[size]


def list_models() -> list[str]:
    try:
        resp = ollama.list()
        names = sorted(m.model for m in resp.models if m.model.startswith("gemma4"))
        return names or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def transcribe_whisper(audio_path: str, size: str) -> str:
    segments, _info = get_whisper(size).transcribe(audio_path, beam_size=1)
    return "".join(seg.text for seg in segments).strip()


def transcribe_gemma(audio_path: str, model: str) -> str:
    """Transcribe using Gemma 4's native audio encoder via Ollama.
    Ollama currently accepts audio bytes through the `images` field — same
    multimodal channel, different encoder selected by the model's modality.
    """
    with open(audio_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()
    resp = ollama.chat(
        model=model,
        messages=[{
            "role": "user",
            "content": GEMMA_STT_PROMPT,
            "images": [audio_b64],
        }],
        options={"temperature": 0.0},
    )
    return resp["message"]["content"].strip()


def transcribe(audio_path: str, stt_choice: str) -> str:
    if stt_choice.startswith(WHISPER_PREFIX):
        return transcribe_whisper(audio_path, stt_choice[len(WHISPER_PREFIX):])
    return transcribe_gemma(audio_path, stt_choice)


async def _tts_to_file(text: str, voice: str, out_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_path)


def synthesize(text: str, voice: str) -> str:
    """Return a path to an mp3 of the spoken text."""
    out = Path(tempfile.mkdtemp()) / "reply.mp3"
    asyncio.run(_tts_to_file(text, voice, str(out)))
    return str(out)


def converse(audio_path: str | None, history: list, model: str, voice: str,
             role: str, think_budget: int, stt_choice: str):
    """One round-trip: transcribe, stream LLM (with optional thinking budget),
    then synthesize. Yields (history, audio, transcript, thinking) tuples."""
    if not audio_path:
        yield history, None, "No audio received.", ""
        return

    yield history, None, f"**You**: _transcribing with {stt_choice}..._", ""
    user_text = transcribe(audio_path, stt_choice)
    if not user_text:
        yield history, None, "Didn't catch that — try again.", ""
        return

    transcript = f"**You**: {user_text}\n\n**Gemma**: _thinking..._"
    yield history, None, transcript, ""

    system_text = role.strip() or DEFAULT_ROLE
    messages = [{"role": "system", "content": system_text}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    budget_chars = int(think_budget * CHARS_PER_TOKEN) if think_budget else 0

    stream = ollama.chat(
        model=model, messages=messages,
        stream=True, think=True,
        options={"temperature": 0.6},
    )

    thinking, reply = "", ""
    capped = False
    try:
        for chunk in stream:
            m = chunk["message"]
            if m.get("thinking"):
                thinking += m["thinking"]
            if m.get("content"):
                reply += m["content"]
            transcript = f"**You**: {user_text}\n\n**Gemma**: {reply or '_thinking..._'}"
            yield history, None, transcript, thinking

            if budget_chars and not reply and len(thinking) >= budget_chars:
                capped = True
                break
    finally:
        stream.close()

    if capped:
        thinking += f"\n\n_[capped at ~{think_budget} tokens]_"
        transcript = f"**You**: {user_text}\n\n**Gemma**: _generating reply..._"
        yield history, None, transcript, thinking

        followup = messages + [
            {"role": "assistant", "content": thinking},
            {"role": "user", "content": "Stop thinking. Give your short spoken reply now."},
        ]
        answer_stream = ollama.chat(
            model=model, messages=followup,
            stream=True, think=False,
            options={"temperature": 0.6},
        )
        reply = ""
        for chunk in answer_stream:
            reply += chunk["message"].get("content", "")
            transcript = f"**You**: {user_text}\n\n**Gemma**: {reply or '_generating reply..._'}"
            yield history, None, transcript, thinking

    reply = reply.strip() or "(no reply)"
    audio_out = synthesize(reply, voice)
    new_history = history + [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": reply},
    ]
    transcript = f"**You**: {user_text}\n\n**Gemma**: {reply}"
    yield new_history, audio_out, transcript, thinking


def clear_history():
    return [], None, "", ""


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
            "Speak → STT (Whisper or Gemma 4 native) → Gemma 4 → Edge TTS → hear the reply."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="LLM")
            stt = gr.Dropdown(
                STT_CHOICES, value=DEFAULT_STT, label="Speech-to-text",
                info="whisper:* = faster-whisper. gemma4:e2b/e4b = native audio encoder.",
            )
            voice = gr.Dropdown(voices, value=TTS_VOICE, label="TTS voice")
            think_budget = gr.Slider(
                0, 2000, value=200, step=50,
                label="Thinking budget (tokens, 0 = unlimited)",
                info="Voice wants snappy replies — a tight cap keeps latency low.",
            )
            clear = gr.Button("Clear conversation")

        role = gr.Textbox(
            value=DEFAULT_ROLE, label="Role / system prompt", lines=3,
        )

        history_state = gr.State([])

        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
                send = gr.Button("Send", variant="primary")
            with gr.Column():
                reply_audio = gr.Audio(label="Reply", autoplay=True)
                transcript = gr.Markdown()
                with gr.Accordion("🧠 Thinking", open=False):
                    thinking = gr.Textbox(
                        label=None, show_label=False, lines=8,
                        placeholder="Thinking tokens stream here...",
                    )

        send.click(
            converse,
            inputs=[mic, history_state, model, voice, role, think_budget, stt],
            outputs=[history_state, reply_audio, transcript, thinking],
        )
        clear.click(clear_history, outputs=[history_state, reply_audio, transcript, thinking])

    return demo


if __name__ == "__main__":
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    build_ui().launch(server_name="0.0.0.0", server_port=7865, share=share)
