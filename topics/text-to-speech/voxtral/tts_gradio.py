"""Gradio UI for the Voxtral TTS server.

Start the server first:
    vllm-omni serve mistralai/Voxtral-4B-TTS-2603 --omni \
        --stage-configs-path ./voxtral_tts_spark.yaml

Then run this:
    python tts_gradio.py
"""
import io

import gradio as gr
import httpx
import numpy as np
import soundfile as sf

BASE_URL = "http://localhost:8000/v1"
MODEL = "mistralai/Voxtral-4B-TTS-2603"


def list_voices() -> list[str]:
    try:
        r = httpx.get(f"{BASE_URL}/audio/voices", timeout=10.0)
        r.raise_for_status()
        return sorted(r.json().get("voices", []))
    except Exception as e:
        print(f"[warn] could not fetch voices: {e}")
        return ["casual_male", "casual_female", "neutral_male", "neutral_female"]


def synthesize(text: str, voice: str) -> tuple[int, np.ndarray]:
    if not text.strip():
        raise gr.Error("Enter some text first.")
    payload = {
        "input": text,
        "model": MODEL,
        "response_format": "wav",
        "voice": voice,
    }
    r = httpx.post(f"{BASE_URL}/audio/speech", json=payload, timeout=180.0)
    r.raise_for_status()
    audio, sr = sf.read(io.BytesIO(r.content), dtype="float32")
    return sr, audio


VOICES = list_voices()

with gr.Blocks(title="Voxtral TTS") as demo:
    gr.Markdown("# Voxtral TTS\nType text, pick a voice, hit generate.")
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                label="Text",
                value="Hello from the DGX Spark. Voxtral TTS is alive.",
                lines=4,
            )
            voice = gr.Dropdown(
                label="Voice",
                choices=VOICES,
                value="casual_male" if "casual_male" in VOICES else VOICES[0],
            )
            go = gr.Button("Generate", variant="primary")
        with gr.Column():
            audio = gr.Audio(label="Output", type="numpy")

    go.click(synthesize, inputs=[text, voice], outputs=audio)

if __name__ == "__main__":
    import os
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    demo.launch(server_name="0.0.0.0", server_port=7860, share=share)
