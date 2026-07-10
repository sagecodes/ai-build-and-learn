"""Minimal Voxtral TTS client.

Requires a running Voxtral server:
    vllm serve mistralai/Voxtral-4B-TTS-2603 --omni
"""
import argparse
import io
import sys

import httpx
import soundfile as sf

MODEL = "mistralai/Voxtral-4B-TTS-2603"

VOICES = [
    "casual_male", "casual_female",
    "narrator_male", "narrator_female",
]


def synthesize(text: str, voice: str, out_path: str, base_url: str) -> None:
    payload = {
        "input": text,
        "model": MODEL,
        "response_format": "wav",
        "voice": voice,
    }
    resp = httpx.post(f"{base_url}/audio/speech", json=payload, timeout=120.0)
    resp.raise_for_status()

    audio, sr = sf.read(io.BytesIO(resp.content), dtype="float32")
    sf.write(out_path, audio, sr)
    print(f"[{voice}] {len(audio)} samples @ {sr} Hz -> {out_path}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("text", nargs="?", default="Paris is a beautiful city!")
    p.add_argument("--voice", default="casual_male", choices=VOICES + ["all"])
    p.add_argument("--out", default="out.wav")
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    args = p.parse_args()

    if args.voice == "all":
        for v in VOICES:
            synthesize(args.text, v, f"out_{v}.wav", args.base_url)
    else:
        synthesize(args.text, args.voice, args.out, args.base_url)
    return 0


if __name__ == "__main__":
    sys.exit(main())