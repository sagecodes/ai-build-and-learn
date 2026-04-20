Welcome to AI Build & Learn — a weekly AI engineering stream where we pick a new topic and learn by building together.

This event is about building with the Gemma 4 models Google just released!
They come in different sizes and can be used in different applications, such as chat, agents, or visual understanding.

https://deepmind.google/models/gemma/gemma-4/

## Examples in this folder

All examples use Ollama to serve Gemma 4 locally and default to `gemma4:31b`.
Set `GEMMA_MODEL=gemma4:<size>` to try a different size without editing code.

| Folder | Port | What it shows |
|---|---|---|
| [`chatbot/`](./chatbot) | 7860 | Streaming chat UI — system prompt, temp, model picker |
| [`vision/`](./vision)   | 7861 | Upload an image, ask anything about it |
| [`extract/`](./extract) | 7862 | Messy text → typed JSON via JSON-schema mode |
| [`docs/`](./docs)       | 7863 | Drop in a PDF, ask questions — 262k context, no RAG |
| [`agent/`](./agent)     | 7864 | Tool-use: calculator, web search, file read |
| [`voice/`](./voice)     | 7865 | Speak → Whisper → Gemma → Edge TTS → reply |

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
