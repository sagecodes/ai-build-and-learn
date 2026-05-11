Welcome to AI Build & Learn — a weekly AI engineering stream where we pick a new topic and learn by building together.

This event is about building with the Gemma 4 models Google just released.
The family covers everything from on-device edge models to a 31B dense flagship, all multimodal and all open-weight under Apache 2.0.

https://deepmind.google/models/gemma/gemma-4/

## The Gemma 4 family

Four variants, each with its own design tradeoff:

| Model | What it actually is | Why it's interesting |
|---|---|---|
| **`gemma4:e2b`** | ~5.1B total / ~2.3B "effective" params, 128K ctx | Per-Layer Embedding (PLE) architecture — big embedding tables live off the hot path, so you can load and run a capable multimodal model on a phone or laptop. **E2B and E4B also ship with a native audio encoder** (speech-to-text without a separate Whisper step). |
| **`gemma4:e4b`** | ~8B total / ~4.5B effective, 128K ctx | Same PLE trick as E2B but bigger — the sweet spot for local multimodal: vision, audio, and decent reasoning. |
| **`gemma4:26b`** | 26B total / ~4B **active** per token (MoE), 256K ctx | Mixture-of-Experts: 128 expert sub-networks, the router picks 8 per token. You pay 26B-sized RAM to load it, but inference runs at ~4B speed. This is the one to watch for local dev boxes — frontier-ish quality at small-model latency. |
| **`gemma4:31b`** | 31B dense, 256K ctx | Classic dense transformer. Slower than 26B MoE and ~same RAM footprint, but the most "predictable" behavior — good baseline. |

"E" = *effective* params (PLE models). "A" in specs like "26B A4B" = *active* params per token (MoE). Both are ways to make a model feel smaller at inference time than its weight file suggests.

Only **E2B and E4B** have the native audio encoder. All four are vision-capable.

Sources:
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4)
- [HF: Welcome Gemma 4](https://huggingface.co/blog/gemma4)
- [A Visual Guide to Gemma 4 — Maarten Grootendorst](https://newsletter.maartengrootendorst.com/p/a-visual-guide-to-gemma-4)

## Examples in this folder

All examples use Ollama to serve Gemma 4 locally and default to `gemma4:31b`
(where it makes sense — `live-camera/` defaults to `e4b` for speed).
Set `GEMMA_MODEL=gemma4:<variant>` to try a different one without editing code.

| Folder | Port | What it shows |
|---|---|---|
| [`chatbot/`](./chatbot)         | 7860 | Streaming chat UI — system prompt, temp, model picker |
| [`vision/`](./vision)           | 7861 | Upload an image, ask anything about it |
| [`extract/`](./extract)         | 7862 | Messy text → typed JSON via JSON-schema mode |
| [`docs/`](./docs)               | 7863 | Drop in a PDF, ask questions — 262k context, no RAG |
| [`agent/`](./agent)             | 7864 | Tool-use: calculator, web search, file read |
| [`voice/`](./voice)             | 7865 | Speak → STT (Whisper **or** Gemma 4 native audio) → Gemma → Edge TTS |
| [`live-camera/`](./live-camera) | 7867 | Webcam → Gemma 4 vision → streaming caption of what's happening in "real" time |

## Prereqs

```bash
ollama serve &
ollama pull gemma4:31b     # or gemma4:4b, gemma4:12b
```

Each example has its own `requirements.txt` and `README.md` with setup instructions. The pattern for every demo is:

```bash
cd topics/gemma4/<example>

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt

python app.py
```

## Resources

- GitHub: https://github.com/sagecodes/ai-build-and-learn
- Events Calendar: https://luma.com/ai-builders-and-learners
- Slack (Discuss during the week): https://slack.flyte.org/
- Hosted by Sage Elliott: https://www.linkedin.com/in/sageelliott/
