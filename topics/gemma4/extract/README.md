# Gemma 4 Structured Extraction

Messy text in, typed JSON out. Uses Ollama's JSON-schema constraint mode — the
model is forced to produce output that matches the schema, so you don't need
regex fallbacks or retry loops.

Preset schemas in the UI: receipt, contact, calendar event, resume. Edit the
schema field freely to extract whatever shape you want.

## Setup

```bash
cd topics/gemma4/extract

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
# -> http://localhost:7862
```
