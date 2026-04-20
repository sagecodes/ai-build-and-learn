# Gemma 4 Long-Context Docs

Drop in a PDF / txt / md file and ask anything. No chunking, no retrieval,
no vector DB — the whole document goes straight into Gemma 4's 262k context
window.

## Setup

```bash
cd topics/gemma4/docs

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
# -> http://localhost:7863
```

## Gotcha

Ollama defaults `num_ctx=4096`, which silently truncates long docs. This app
sizes `num_ctx` dynamically to fit your doc (capped at 262k). VRAM scales
with context size — a 100k-token prompt on a 31B model will need the Spark's
full ~64GB unified memory.

## Things to try

- Drop in an academic paper, ask "what's the key claim and what's the
  weakest part of their argument?"
- Long transcript → "bullet-point action items"
- Codebase concat → "trace how `foo()` is called across the repo"
