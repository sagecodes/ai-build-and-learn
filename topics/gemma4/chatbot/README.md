# Gemma 4 Chatbot

Gradio chat UI over a local Gemma 4 model served by Ollama.

## Setup

```bash
cd topics/gemma4/chatbot

uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

You also need Ollama running with a Gemma 4 model pulled:

```bash
ollama serve &
ollama pull gemma4:31b     # or gemma4:4b, gemma4:12b
```

## Run

```bash
python app.py
# -> http://localhost:7860
```

### Public URL (remote / forwarded-port setups)

Set `GRADIO_SHARE=1` for a public HTTPS tunnel via Gradio's servers (link good for 72 hours). Useful when SSH port-forwarding chokes on large uploads or behaves weirdly.

```bash
GRADIO_SHARE=1 python app.py
```

Use a different size without editing code:

```bash
GEMMA_MODEL=gemma4:4b python app.py
```

The dropdown is populated from `ollama list` (anything starting with `gemma4`).

## Playing with Temperature and Top-p

Both sliders change how the next token is picked from the model's probability distribution. They do different things — people mix them up constantly.

### Temperature (0.0 – 1.5)

Controls **how peaked or flat** that distribution is before sampling.

- **0.0** — deterministic. Always picks the single highest-probability token. Same input → same output. Good for factual Q&A, code, extraction.
- **0.3–0.5** — slightly loosened. Still focused, but with some variation. Good default for helpful-assistant vibes.
- **0.7** (default here) — balanced. Creative without going off the rails.
- **1.0+** — flattens the distribution. Rarer words get picked more often. Good for brainstorming, fiction.
- **1.3+** — starts producing weird, incoherent, or looping output. Fun to break things with.

Try asking the same question at `0.0`, `0.7`, and `1.3` to feel the difference.

### Top-p (0.1 – 1.0) — "nucleus sampling"

Instead of reshaping the distribution, top-p **truncates the tail**. It keeps only the smallest set of tokens whose cumulative probability adds up to `p`, then samples from those.

- **1.0** — no truncation; any token can be picked (subject to temperature).
- **0.95** (default here) — drops the weirdest 5% of tails. Very common default.
- **0.5** — only tokens in the top-50% probability mass are eligible. Tight, safe, repetitive.
- **0.1** — basically forces the top choice. Similar feel to temperature 0.

### Using them together

The two stack. Common combinations:

| Goal | Temp | Top-p |
|---|---|---|
| Deterministic / factual | 0.0 | 1.0 |
| Default chat | 0.7 | 0.95 |
| Creative writing | 1.0 | 0.95 |
| Tight but not robotic | 0.3 | 0.9 |
| Chaos | 1.4 | 1.0 |

Rule of thumb: **tune one at a time**. Pick temperature for the vibe (focused vs. creative), leave top-p at 0.9–0.95, and only touch top-p if you're getting weird tail outputs.
