# Screen Context Harness

A multimodal LLM-powered application that monitors your screen activity to provide near-real-time contextual understanding and long-term memory. It captures screenshots, generates descriptions, consolidates them into persistent outlines, and enables natural-language retrieval of your activity history.

## Purpose

The Screen Context Harness explores whether an LLM can produce useful, near-real-time contextual understanding of a user's focus. It aims to answer:
- Can a 5-second capture cadence feel "live"?
- Does a rolling 60-second consolidation stay coherent?
- Can a vector store of past activity provide meaningful long-term memory?

## How It Works

The application operates through three primary asynchronous loops:

### 1. Capture & Caption Loop (5s)
- **Capture:** Grabs a screenshot of the primary monitor using `mss`.
- **Downsample:** Resizes the image (max 768px) to minimize latency.
- **Analyze:** Sends the image to a multimodal LLM (e.g., `gemma4:26b` via Ollama).
- **Caption:** Generates a one-sentence description of the current app, file, or task.
- **Buffer:** Stores the last 12 captions (covering ~60s) in a rolling buffer.

### 2. Consolidation & RAG Loop (60s)
- **Retrieve:** Queries ChromaDB for semantically relevant prior outlines (RAG).
- **Consolidate:** Uses an LLM to synthesize the current caption buffer and any relevant prior context into a cohesive "Context Outline."
- **Persist:** Embeds and saves the new outline into a local vector store (`chroma_db/`).

### 3. Compaction Hierarchy (Maintenance)
To maintain long-term coherence, the system periodically compacts older entries:
- **Minute:** Raw consolidated outlines (kept for ~10 mins).
- **Hourly:** Summaries of minute entries (kept for ~72 hours).
- **Daily:** High-level summaries of hourly entries (kept permanently).

## User Interface

The Gradio-based UI provides:
- **Latest Capture:** Real-time view of what the harness sees.
- **Current Focus:** The latest LLM-generated caption.
- **Recent Summary:** The most recent 60-second consolidation.
- **Context Chat:** A natural-language interface to query your activity history ("What was I working on this morning?").
- **Process Log:** A transparent view of the underlying LLM calls, RAG hits, and system status.

## Installation & Setup

### Prerequisites
- **Python 3.10+** (managed via `uv` is recommended)
- **Ollama:** A running instance (local or remote via ngrok) with a multimodal model like `gemma4:26b` or `llava`.
- **Permissions:** (macOS) Grant "Screen Recording" permissions to your terminal or IDE.

### Running the App
1. Navigate to the project directory:
   ```bash
   cd topics/vectorstore/screen-context-harness
   ```
2. Run the application:
   ```bash
   uv sync && uv run python app.py
   ```
3. Open the UI in your browser: `http://localhost:7868`

## Configuration

Tweak timing, model choices, and RAG sensitivity directly in `app.py` (see the `── CONFIGURATION ──` section):
- `CAPTURE_CADENCE_S`: Frequency of screenshots (default: 5s).
- `CONSOLIDATE_CADENCE_S`: Frequency of consolidation (default: 60s).
- `RAG_DISTANCE_THRESHOLD`: Sensitivity for context retrieval (default: 0.55).
- `DEFAULT_MODEL`: The Ollama model to use.

## Tech Stack
- **UI:** [Gradio](https://gradio.app/)
- **Vision/LLM:** [Ollama](https://ollama.com/) (Gemma 4 Vision)
- **Vector Store:** [ChromaDB](https://www.trychroma.com/)
- **Capture:** `mss` & `Pillow`
