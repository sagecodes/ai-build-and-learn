"""
Gemma 4 structured extraction: messy text in, typed JSON out.

Uses Ollama's JSON schema mode (`format=<json_schema>`) so the model is
constrained to produce valid JSON matching your schema — no regex parsing,
no retry loops.

Run (after `uv venv` + `uv pip install -r requirements.txt` + activating):
    ollama serve &
    ollama pull gemma4:31b
    python app.py
"""

from __future__ import annotations

import json
import os

import gradio as gr
import ollama

DEFAULT_MODEL = os.environ.get("GEMMA_MODEL", "gemma4:31b")

# ---------------------------------------------------------------------------
# Schemas — each is a (name, schema, sample input) triple shown in the UI.
# ---------------------------------------------------------------------------

RECEIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "merchant": {"type": "string"},
        "date": {"type": "string", "description": "ISO 8601 date, YYYY-MM-DD"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "qty": {"type": "number"},
                    "price": {"type": "number"},
                },
                "required": ["name", "price"],
            },
        },
        "subtotal": {"type": "number"},
        "tax": {"type": "number"},
        "total": {"type": "number"},
    },
    "required": ["merchant", "total", "items"],
}

RECEIPT_SAMPLE = """BLUE BOTTLE COFFEE
1 Ferry Building, SF
04/17/2026  10:42

Cortado          1  4.75
Almond croissant 1  5.50
Drip coffee      2  7.00

Subtotal      17.25
Tax (8.5%)     1.47
TOTAL         18.72
VISA ****1234"""


CONTACT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "phone": {"type": "string"},
        "company": {"type": "string"},
        "title": {"type": "string"},
    },
    "required": ["name"],
}

CONTACT_SAMPLE = """Hey, it's Jamie Chen from Anthropic — I lead the
developer-experience team. Best way to reach me is jamie@anthropic.com
or (415) 555-0147 if it's urgent. Let's chat next week!"""


EVENT_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "start": {"type": "string", "description": "ISO 8601 datetime"},
        "end": {"type": "string", "description": "ISO 8601 datetime"},
        "location": {"type": "string"},
        "attendees": {"type": "array", "items": {"type": "string"}},
        "notes": {"type": "string"},
    },
    "required": ["title", "start"],
}

EVENT_SAMPLE = """Confirming our Gemma 4 demo session Thursday Apr 23 from
2-3pm at the Mission Bay office, conference room B. Sage, Priya and Marco
will be there. Bring the DGX Spark."""


RESUME_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "email": {"type": "string"},
        "skills": {"type": "array", "items": {"type": "string"}},
        "experience": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {"type": "string"},
                    "title": {"type": "string"},
                    "start": {"type": "string"},
                    "end": {"type": "string"},
                },
                "required": ["company", "title"],
            },
        },
    },
    "required": ["name", "experience"],
}

RESUME_SAMPLE = """Alex Kowalski — alex.k@example.com
ML Engineer with 6 years building recommender systems.

Experience
  Netflix, Senior ML Engineer, 2022 - present
  Airbnb, ML Engineer, 2019 - 2022
  Stripe, Data Scientist, 2018 - 2019

Skills: Python, PyTorch, Ray, Kubernetes, SQL, feature engineering"""


PRESETS = {
    "Receipt": (RECEIPT_SCHEMA, RECEIPT_SAMPLE),
    "Contact": (CONTACT_SCHEMA, CONTACT_SAMPLE),
    "Calendar event": (EVENT_SCHEMA, EVENT_SAMPLE),
    "Resume": (RESUME_SCHEMA, RESUME_SAMPLE),
}


def list_models() -> list[str]:
    try:
        resp = ollama.list()
        names = sorted(m.model for m in resp.models if m.model.startswith("gemma4"))
        return names or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]


def extract(text: str, schema_json: str, model: str):
    """Call Gemma 4 with the given JSON schema as the format constraint."""
    if not text.strip():
        return "Paste some text first.", ""
    try:
        schema = json.loads(schema_json)
    except json.JSONDecodeError as e:
        return f"Invalid schema JSON: {e}", ""

    system = (
        "You extract structured data from text into JSON matching the given schema. "
        "Follow every field description (e.g. date formats). Include all optional "
        "fields when the information is clearly present in the text. Only put items "
        "in an 'items' array if they are actual items — never totals, taxes, or "
        "metadata. Output JSON only, no prose."
    )
    resp = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        format=schema,
        options={"temperature": 0.0},
    )
    raw = resp["message"]["content"]
    try:
        pretty = json.dumps(json.loads(raw), indent=2)
    except json.JSONDecodeError:
        pretty = raw
    return pretty, raw


def load_preset(name: str):
    schema, sample = PRESETS[name]
    return sample, json.dumps(schema, indent=2)


def build_ui() -> gr.Blocks:
    models = list_models()
    default = DEFAULT_MODEL if DEFAULT_MODEL in models else models[0]

    with gr.Blocks(title="Gemma 4 Extract") as demo:
        gr.Markdown(
            "# Gemma 4 Structured Extraction\n"
            "Messy text → typed JSON. Powered by Ollama's JSON schema mode."
        )
        with gr.Row():
            model = gr.Dropdown(models, value=default, label="Model")
            preset = gr.Dropdown(
                list(PRESETS.keys()), value="Receipt", label="Preset schema",
            )

        with gr.Row():
            with gr.Column():
                text = gr.Textbox(
                    value=RECEIPT_SAMPLE, label="Input text", lines=14,
                )
                schema = gr.Code(
                    value=json.dumps(RECEIPT_SCHEMA, indent=2),
                    language="json", label="JSON schema (edit freely)", lines=14,
                )
                submit = gr.Button("Extract", variant="primary")
            with gr.Column():
                output = gr.Code(language="json", label="Output", lines=20)
                raw = gr.Textbox(label="Raw model output", lines=4, visible=False)

        preset.change(load_preset, inputs=preset, outputs=[text, schema])
        submit.click(extract, inputs=[text, schema, model], outputs=[output, raw])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7862)
