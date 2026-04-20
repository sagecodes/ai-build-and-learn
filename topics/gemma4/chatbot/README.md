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

Use a different size without editing code:

```bash
GEMMA_MODEL=gemma4:4b python app.py
```

The dropdown is populated from `ollama list` (anything starting with `gemma4`).
