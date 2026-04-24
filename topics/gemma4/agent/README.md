# Gemma 4 Agent

ReAct-style tool-use loop. The model receives a user question, decides
which tools to call (if any), we execute them, feed results back, and it
continues until it has an answer.

## Tools

- **calculator** — AST-safe arithmetic (+, -, *, /, //, %, **)
- **current_datetime** — ISO 8601 timestamp
- **web_search** — DuckDuckGo via `ddgs`, 5 results
- **list_files** — list files in `./sandbox/`
- **read_file** — read a file from `./sandbox/` (path traversal blocked)

Drop any files into `./sandbox/` to make them available to the agent.

## Setup

```bash
cd topics/gemma4/agent

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
# -> http://localhost:7864
```

### Public URL (remote / forwarded-port setups)

Set `GRADIO_SHARE=1` for a public HTTPS tunnel via Gradio's servers (link good for 72 hours). Useful when SSH port-forwarding is flaky from a remote dev box.

```bash
GRADIO_SHARE=1 python app.py
```

## Things to try

- "What is 847 * 293 + 1024?" — forces calculator
- "Search for the latest news on Gemma 4" — web search
- "List the files in the sandbox, then read the one about the project." — multi-step tool chain
