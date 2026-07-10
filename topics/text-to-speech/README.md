# Open-Source Text-to-Speech: "Natural" Voices

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event is about turning text into speech with AI. As with the other generation events, there's no single model we're locked into — the point is to explore what's out there and actually try a few. We'll focus on open-source models, but you're welcome to bring commercial ones (ElevenLabs and friends) if you want to compare — and it's worth seeing how close the best open models now get.

We'll look at the main axes that matter in practice: fast, lightweight narration versus the most natural, expressive voices, plus **voice cloning** and multilingual support. I'll research and try some of the best open-source options ahead of the stream, and we'll talk through the tradeoffs: quality, real-time latency, cloning, languages, model size / hardware, and — importantly — **licensing** (several strong models are non-commercial).

Some things to look up to get started:

**Open-source models:**
- Kokoro-82M: tiny and fast — runs on CPU, Apache-2.0; 54 built-in voices but no cloning
- Chatterbox (Resemble AI): natural speech with voice cloning, MIT; the Turbo variant is lightweight and beat ElevenLabs in a blind listening test
- XTTS v2 (Coqui): broad multilingual voice cloning (~17 languages) — note: non-commercial license
- F5-TTS / Orpheus 3B: strong research-grade voice cloning (F5-TTS weights are non-commercial; Orpheus is Apache/MIT)
- Piper: ultra-light, great for CPU / Raspberry Pi and offline or edge use

**Tooling:**
- Hugging Face TTS Arena — a crowd-ranked leaderboard: https://huggingface.co/spaces/TTS-AGI/TTS-Arena
- Coqui TTS toolkit: https://github.com/coqui-ai/TTS
