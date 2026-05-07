"""
Screen Context Harness — screenshot → Gemma 4 → caption + RAG outline.

Run:  uv run python app.py   →  http://localhost:7868
macOS: grant Screen Recording permission to the terminal before running.
"""

from __future__ import annotations

import base64
import io
import threading
import time
from datetime import datetime
from pathlib import Path

import chromadb
import gradio as gr
import mss
import ollama
from PIL import Image

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# Model settings
DEFAULT_MODEL = "gemma4:26b"

# Capture & caption loop
CAPTURE_CADENCE_S    = 5     # how often to grab a screenshot and caption it
MAX_SIDE             = 768   # downsample long-edge to this before sending to LLM (px)
MAX_OUT_TOKENS       = 80    # hard cap on caption token output
BUFFER_MAX           = 12    # rolling caption buffer size (~60s at 5s cadence)

# Consolidation loop
CONSOLIDATE_CADENCE_S = 60   # how often to run the outline consolidation cycle
OUTLINE_MAX_CHARS     = 600  # trigger inline compaction when outline exceeds this

# Compaction hierarchy
COMPACT_CADENCE_S  = 300         # how often the compaction check runs (5 min)
TTL_MINUTE_S       = 600         # 10 minutes — promote minute entries to hourly
TTL_HOURLY_S       = 259_200     # 72 hours  — promote hourly entries to daily
MIN_COMPACT_COUNT  = 3           # minimum expired entries needed to trigger compaction

LEVEL_MINUTE = "minute"
LEVEL_HOURLY = "hourly"
LEVEL_DAILY  = "daily"

# RAG retrieval defaults
RAG_TOP_K              = 3     # candidates pulled from vector store at consolidation time
RAG_DISTANCE_THRESHOLD = 0.55  # cosine distance cutoff — lower = stricter match

# Vector store
CHROMA_PATH     = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "activity_outlines"

# UI
LOG_MAX_LINES = 50   # process log keeps this many lines before trimming

# ── APPLICATION STATE & INIT ──────────────────────────────────────────────────
_caption_lock     = threading.Lock()
_consolidate_lock = threading.Lock()
_compact_lock     = threading.Lock()

_chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
_store  = _chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


def _backfill_level_field() -> int:
    """Tag legacy records that predate the level field as 'minute'."""
    results = _store.get(include=["metadatas"])
    needs = [
        (id_, meta)
        for id_, meta in zip(results["ids"], results["metadatas"])
        if "level" not in meta
    ]
    if needs:
        _store.update(
            ids=[id_ for id_, _ in needs],
            metadatas=[{**meta, "level": LEVEL_MINUTE} for _, meta in needs],
        )
    return len(needs)


_migrated = _backfill_level_field()

CAPTION_PROMPT = (
    "You are watching a live screen recording. "
    "Describe what the person is doing RIGHT NOW in one direct sentence. "
    "Be specific: name the app, file, or content visible. "
    "No preamble. Present tense."
)

CONSOLIDATE_PROMPT = (
    "Recent screen activity (last ~60s):\n{captions}\n\n"
    "Current context outline:\n{outline}\n\n"
    "{prior_context}"
    "Update the outline to reflect the current session. "
    "Keep it under 500 characters. If prior context is relevant, briefly reference it. "
    "Focus on: what app/task is active, what they've been working on, any recent transitions. "
    "Remove stale or redundant detail. No preamble."
)

COMPACT_PROMPT = (
    "Compress this context outline to under 350 characters. "
    "Preserve the most important recent activity:\n\n{outline}"
)

HOURLY_COMPACT_PROMPT = (
    "Summarize these minute-by-minute screen activity notes into a single hourly summary.\n\n"
    "{entries}\n\n"
    "Write a cohesive summary under 400 characters. "
    "Cover the main tasks, tools used, and significant transitions. No preamble."
)

DAILY_COMPACT_PROMPT = (
    "Summarize these hourly screen activity summaries into a single daily log entry.\n\n"
    "{entries}\n\n"
    "Write a cohesive daily summary under 600 characters. "
    "Cover what was accomplished, tools/projects used, and notable patterns. No preamble."
)

CHAT_SYSTEM_PROMPT = (
    "You are a personal context assistant with access to a log of the user's screen activity. "
    "The log contains entries at different granularities: minute-level outlines, hourly summaries, "
    "and daily logs — all captured by a screen sensory harness.\n\n"
    "{context_block}"
    "Answer the user's question using the activity log. Be specific and direct. "
    "If the log doesn't contain enough information to answer, say so clearly."
)

CHAT_SYSTEM_PROMPT_EMPTY = (
    "You are a personal context assistant. The user's screen activity log is currently empty — "
    "no outlines have been stored yet. Let the user know and offer to help once sessions have run."
)


def extract_text(content) -> str:
    """Normalize Gradio message content — may arrive as a string or list of content blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(block.get("text", "") for block in content if isinstance(block, dict))
    return str(content)


def resolve_host(url: str) -> str:
    host = url.strip().rstrip("/")
    return host if host else "http://localhost:11434"


def make_client(url: str) -> ollama.Client:
    return ollama.Client(host=resolve_host(url))


def host_status(url: str) -> str:
    host = resolve_host(url)
    if "localhost" in host or "127.0.0.1" in host:
        return f"🟡 Local — {host}"
    return "🟢 Remote — connected"


def ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def append_log(log: str, line: str) -> str:
    lines = log.splitlines() if log else []
    lines.append(f"[{ts()}] {line}")
    return "\n".join(lines[-LOG_MAX_LINES:])


def capture_screen() -> tuple[str, Image.Image]:
    with mss.MSS() as sct:
        raw = sct.grab(sct.monitors[1])
    img = Image.frombytes("RGB", raw.size, raw.bgra, "raw", "BGRX")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode(), img


def save_outline(outline: str, level: str = LEVEL_MINUTE) -> int:
    _store.add(
        ids=[f"{level}_{int(time.time() * 1000)}"],
        documents=[outline],
        metadatas=[{"timestamp": ts(), "ts_epoch": int(time.time()), "level": level}],
    )
    return _store.count()


def latest_outline() -> tuple[str, dict | None]:
    if _store.count() == 0:
        return "", None
    results = _store.get(include=["documents", "metadatas"])
    pairs = sorted(
        zip(results["metadatas"], results["documents"]),
        key=lambda x: x[0].get("ts_epoch", 0),
        reverse=True,
    )
    meta, doc = pairs[0]
    return doc, meta


def search_store(query: str) -> list[dict]:
    n = _store.count()
    if n == 0:
        return []
    results = _store.query(
        query_texts=[query],
        n_results=min(RAG_TOP_K, n),
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "text": doc,
            "timestamp": meta["timestamp"],
            "level": meta.get("level", LEVEL_MINUTE),
            "distance": round(dist, 3),
        }
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        )
        if dist < RAG_DISTANCE_THRESHOLD
    ]


def format_rag_hits(hits: list[dict]) -> str:
    if not hits:
        return ""
    lines = "\n".join(f"  [{h['timestamp']} {h['level']}] {h['text']}" for h in hits)
    return f"Relevant prior context (cite if applicable):\n{lines}\n\n"


def store_summary() -> str:
    results = _store.get(include=["metadatas"])
    total = len(results["ids"])
    if total == 0:
        return "**0** outlines stored"
    counts = {LEVEL_MINUTE: 0, LEVEL_HOURLY: 0, LEVEL_DAILY: 0}
    for meta in results["metadatas"]:
        lv = meta.get("level", LEVEL_MINUTE)
        if lv in counts:
            counts[lv] += 1
    breakdown = " · ".join(f"{v} {k}" for k, v in counts.items() if v > 0)
    return f"**{total}** stored — {breakdown}"


def caption_tick(host_url: str, model: str, caption_buffer: list[str], log: str):
    if not _running:
        return

    if not _caption_lock.acquire(blocking=False):
        log = append_log(log, "⏭ Caption skipped — previous call in flight")
        yield gr.skip(), gr.skip(), caption_buffer, log, gr.skip()
        return

    try:
        log = append_log(log, "📸 Capturing screen...")
        yield gr.skip(), gr.skip(), caption_buffer, log, gr.skip()

        t0 = time.time()
        try:
            b64, img = capture_screen()
        except Exception as e:
            log = append_log(log, f"✗ Capture failed: {e}")
            yield gr.skip(), gr.skip(), caption_buffer, log, gr.skip()
            return

        kb = len(b64) * 3 // 4 // 1024
        log = append_log(log, f"  {img.size[0]}×{img.size[1]}px ~{kb}KB — querying {model} @ {host_status(host_url)}")
        yield img, gr.skip(), caption_buffer, log, gr.skip()

        try:
            stream = make_client(host_url).chat(
                model=model,
                messages=[{"role": "user", "content": CAPTION_PROMPT, "images": [b64]}],
                stream=True,
                think=False,
                options={"temperature": 0.3, "num_predict": MAX_OUT_TOKENS},
            )
        except Exception as e:
            log = append_log(log, f"✗ LLM error: {e}")
            yield gr.skip(), f"**Error:** {e}", caption_buffer, log, gr.skip()
            return

        reply = ""
        for chunk in stream:
            reply += chunk.message.content or ""
            yield gr.skip(), reply, caption_buffer, log, gr.skip()

        reply = reply.strip()
        elapsed = time.time() - t0

        if not reply:
            log = append_log(log, f"✗ Empty response ({elapsed:.1f}s)")
            yield gr.skip(), "_Empty response._", caption_buffer, log, gr.skip()
            return

        log = append_log(log, f"✓ {elapsed:.1f}s: \"{reply[:80]}{'…' if len(reply) > 80 else ''}\"")
        new_buffer = (caption_buffer + [reply])[-BUFFER_MAX:]
        buffer_md = "\n".join(f"{i+1}. {c}" for i, c in enumerate(new_buffer))
        yield gr.skip(), reply, new_buffer, log, buffer_md

    finally:
        _caption_lock.release()


def consolidate_tick(host_url: str, model: str, caption_buffer: list[str], outline: str, log: str):
    if not _running:
        return

    if not _consolidate_lock.acquire(blocking=False):
        log = append_log(log, "⏭ Consolidation skipped — previous cycle running")
        yield gr.skip(), outline, caption_buffer, log, gr.skip()
        return

    try:
        if not caption_buffer:
            log = append_log(log, "⏭ Consolidation skipped — no captions yet")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            return

        if len(outline) > OUTLINE_MAX_CHARS:
            log = append_log(log, f"⚙ Outline {len(outline)} chars — compacting...")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            try:
                resp = make_client(host_url).chat(
                    model=model,
                    messages=[{"role": "user", "content": COMPACT_PROMPT.format(outline=outline)}],
                    stream=False,
                    think=False,
                    options={"temperature": 0.2, "num_predict": 120},
                )
                outline = (resp.message.content or outline).strip()
                log = append_log(log, f"✓ Compacted to {len(outline)} chars")
            except Exception as e:
                log = append_log(log, f"✗ Compaction failed: {e}")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()

        captions_text = "\n".join(f"- {c}" for c in caption_buffer)
        log = append_log(log, f"🔍 Querying vector store ({_store.count()} outlines)...")
        yield gr.skip(), outline, caption_buffer, log, gr.skip()

        hits = search_store(captions_text)
        prior_context = format_rag_hits(hits)
        if hits:
            log = append_log(log, f"  Found {len(hits)} relevant: " +
                             ", ".join(f"[{h['timestamp']} {h['level']} d={h['distance']}]" for h in hits))
        else:
            log = append_log(log, "  No relevant prior context found")
        yield gr.skip(), outline, caption_buffer, log, gr.skip()

        log = append_log(log, f"🗂 Consolidating {len(caption_buffer)} captions @ {host_status(host_url)}")
        yield gr.skip(), outline, caption_buffer, log, gr.skip()

        try:
            stream = make_client(host_url).chat(
                model=model,
                messages=[{
                    "role": "user",
                    "content": CONSOLIDATE_PROMPT.format(
                        captions=captions_text,
                        outline=outline or "(empty)",
                        prior_context=prior_context,
                    ),
                }],
                stream=True,
                think=False,
                options={"temperature": 0.3, "num_predict": 200},
            )
        except Exception as e:
            log = append_log(log, f"✗ Consolidation LLM error: {e}")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            return

        new_outline = ""
        for chunk in stream:
            new_outline += chunk.message.content or ""
            yield new_outline, new_outline, caption_buffer, log, gr.skip()

        new_outline = new_outline.strip()
        if not new_outline:
            log = append_log(log, "✗ Empty consolidation response")
            yield gr.skip(), outline, caption_buffer, log, gr.skip()
            return

        count = save_outline(new_outline, level=LEVEL_MINUTE)
        log = append_log(log, f"✓ Outline updated ({len(new_outline)} chars) · 💾 #{count} stored as minute")
        yield new_outline, new_outline, [], log, store_summary()

    finally:
        _consolidate_lock.release()


def compact_tick(host_url: str, model: str, log: str):
    """Runs independently of _running — compaction is maintenance, not capture."""
    if not _compact_lock.acquire(blocking=False):
        return log, gr.skip()

    try:
        now = time.time()
        results = _store.get(include=["documents", "metadatas"])
        entries = list(zip(results["ids"], results["metadatas"], results["documents"]))
        stats_changed = False

        expired_minute = [
            (id_, meta, doc) for id_, meta, doc in entries
            if meta.get("level") == LEVEL_MINUTE
            and now - meta.get("ts_epoch", now) > TTL_MINUTE_S
        ]

        if len(expired_minute) >= MIN_COMPACT_COUNT:
            log = append_log(log, f"⚗ Compacting {len(expired_minute)} minute entries → hourly...")
            entries_text = "\n\n".join(f"[{m['timestamp']}] {d}" for _, m, d in expired_minute)
            try:
                resp = make_client(host_url).chat(
                    model=model,
                    messages=[{"role": "user", "content": HOURLY_COMPACT_PROMPT.format(entries=entries_text)}],
                    stream=False,
                    think=False,
                    options={"temperature": 0.2, "num_predict": 150},
                )
                summary = (resp.message.content or "").strip()
                if summary:
                    save_outline(summary, level=LEVEL_HOURLY)
                    _store.delete(ids=[id_ for id_, _, _ in expired_minute])
                    log = append_log(log, f"✓ {len(expired_minute)} minute → 1 hourly stored")
                    stats_changed = True
                else:
                    log = append_log(log, "✗ Empty hourly compaction response — skipping")
            except Exception as e:
                log = append_log(log, f"✗ minute→hourly failed: {e}")

        # re-fetch after minute compaction before checking hourly
        results = _store.get(include=["documents", "metadatas"])
        entries = list(zip(results["ids"], results["metadatas"], results["documents"]))

        expired_hourly = [
            (id_, meta, doc) for id_, meta, doc in entries
            if meta.get("level") == LEVEL_HOURLY
            and now - meta.get("ts_epoch", now) > TTL_HOURLY_S
        ]

        if len(expired_hourly) >= MIN_COMPACT_COUNT:
            log = append_log(log, f"⚗ Compacting {len(expired_hourly)} hourly entries → daily...")
            entries_text = "\n\n".join(f"[{m['timestamp']}] {d}" for _, m, d in expired_hourly)
            try:
                resp = make_client(host_url).chat(
                    model=model,
                    messages=[{"role": "user", "content": DAILY_COMPACT_PROMPT.format(entries=entries_text)}],
                    stream=False,
                    think=False,
                    options={"temperature": 0.2, "num_predict": 200},
                )
                summary = (resp.message.content or "").strip()
                if summary:
                    save_outline(summary, level=LEVEL_DAILY)
                    _store.delete(ids=[id_ for id_, _, _ in expired_hourly])
                    log = append_log(log, f"✓ {len(expired_hourly)} hourly → 1 daily stored")
                    stats_changed = True
                else:
                    log = append_log(log, "✗ Empty daily compaction response — skipping")
            except Exception as e:
                log = append_log(log, f"✗ hourly→daily failed: {e}")

        return log, store_summary() if stats_changed else gr.skip()

    finally:
        _compact_lock.release()


def context_chat(
    message: str,
    history: list,
    host_url: str,
    model: str,
    top_k: int,
    min_similarity: float,
    temperature: float,
    max_tokens: int,
):
    if not message.strip():
        return history, "", ""

    store_count = _store.count()

    if store_count > 0:
        results = _store.query(
            query_texts=[message],
            n_results=min(top_k, store_count),
            include=["documents", "metadatas", "distances"],
        )
        context_lines = []
        debug_lines = [f"**Retrieved {min(top_k, store_count)} of {store_count} stored outlines** (min similarity: {min_similarity}):\n"]

        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = round(1 - dist, 3)
            passed = similarity >= min_similarity
            level = meta.get("level", LEVEL_MINUTE)
            debug_lines.append(
                f"{'✓' if passed else '✗ filtered'} **[{meta['timestamp']}]** `{level}` similarity `{similarity}`  \n> {doc}"
            )
            if passed:
                context_lines.append(f"[{meta['timestamp']} {level}] (similarity {similarity}) {doc}")

        retrieved_md = "\n\n".join(debug_lines)
        if not context_lines:
            retrieved_md += "\n\n_All results filtered out by min similarity — LLM receives no context._"
            system = CHAT_SYSTEM_PROMPT_EMPTY
        else:
            context_block = f"Activity log ({len(context_lines)} entries):\n" + "\n".join(context_lines) + "\n\n"
            system = CHAT_SYSTEM_PROMPT.format(context_block=context_block)
    else:
        system = CHAT_SYSTEM_PROMPT_EMPTY
        retrieved_md = "_No outlines in store yet._"

    messages = [{"role": "system", "content": system}]
    for turn in history:
        messages.append({"role": turn["role"], "content": extract_text(turn["content"])})
    messages.append({"role": "user", "content": message})

    history = history + [{"role": "user", "content": message}]
    yield history, "", retrieved_md

    try:
        stream = make_client(host_url).chat(
            model=model,
            messages=messages,
            stream=True,
            think=False,
            options={"temperature": temperature, "num_predict": max_tokens},
        )
    except Exception as e:
        yield history + [{"role": "assistant", "content": f"**Error:** {e}"}], "", retrieved_md
        return

    reply = ""
    for chunk in stream:
        reply += chunk.message.content or ""
        yield history + [{"role": "assistant", "content": reply}], "", retrieved_md

    history = history + [{"role": "assistant", "content": reply.strip()}]
    yield history, "", retrieved_md


def start(log: str):
    global _running
    _running = True

    doc, meta = latest_outline()
    if doc and meta:
        seed_ts = meta.get("timestamp", "unknown time")
        level = meta.get("level", LEVEL_MINUTE)
        log = append_log(log, f"▶ Started · seeded from {seed_ts} ({level})")
        display = f"_Seeded {seed_ts} ({level})_\n\n{doc}"
    else:
        doc = ""
        log = append_log(log, "▶ Started · no prior context in store")
        display = "_Waiting for first consolidation (~60s)…_"

    log = append_log(log, f"  {store_summary()}")
    if _migrated:
        log = append_log(log, f"  Migrated {_migrated} records → level=minute")

    return [], doc, "_Starting — first capture in ~5s…_", display, log


def stop(log: str):
    global _running
    _running = False
    return append_log(log, "■ Session stopped")


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Screen Context Harness") as demo:
        gr.Markdown(
            "# Screen Context Harness\n"
            f"Primary monitor → Gemma 4 vision → caption every **{CAPTURE_CADENCE_S}s**, "
            f"outline consolidated every **{CONSOLIDATE_CADENCE_S}s** with RAG from prior sessions. "
            f"Context compacted every **{COMPACT_CADENCE_S//60}min** (minute→hourly→daily)."
        )

        with gr.Row():
            host_url = gr.Textbox(
                label="ngrok URL",
                placeholder="https://your-ngrok-url.ngrok-free.app  (leave blank → localhost:11434)",
                scale=4,
                type="password",
            )
            model_select = gr.Dropdown(
                label="Model",
                choices=[DEFAULT_MODEL],
                value=DEFAULT_MODEL,
                scale=2,
            )
            start_btn = gr.Button("Start", variant="primary", scale=1)
            stop_btn  = gr.Button("Stop", scale=1)

        status_md = gr.Markdown(host_status(""))
        host_url.change(host_status, inputs=[host_url], outputs=[status_md])

        with gr.Row():
            with gr.Column(scale=1):
                screenshot_img = gr.Image(
                    label="Latest Capture",
                    type="pil",
                    interactive=False,
                    height=360,
                )
            with gr.Column(scale=1):
                gr.Markdown("### Current Focus")
                focus_md = gr.Markdown("_Press **Start** to begin…_")
                gr.Markdown("### Recent Summary")
                summary_md = gr.Markdown("_Appears after first consolidation cycle (~60s)…_")

        log_box = gr.Textbox(
            label="Process Log",
            lines=10,
            max_lines=10,
            interactive=False,
            autoscroll=True,
        )

        with gr.Accordion("Caption Buffer (debug)", open=False):
            buffer_md = gr.Markdown("_No captions yet._")

        gr.Markdown("---")
        gr.Markdown("## Context Chat")

        with gr.Row():
            stats_md = gr.Markdown(store_summary())
            clear_btn = gr.Button("Clear chat", size="sm", scale=0)

        with gr.Accordion("RAG Parameters", open=True):
            gr.Markdown(
                "Tune retrieval and generation. **Retrieved Context** below shows exactly "
                "what the LLM sees after each query."
            )
            with gr.Row():
                slider_top_k = gr.Slider(
                    minimum=1, maximum=10, value=5, step=1,
                    label="Top K — chunks retrieved from store",
                    info="How many past outlines to pull before filtering. Higher = more candidates.",
                )
                slider_min_similarity = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.0, step=0.05,
                    label="Min similarity — relevance filter",
                    info="Cosine similarity (0=any, 1=near-identical). Raise to filter loosely related chunks.",
                )
            with gr.Row():
                slider_temperature = gr.Slider(
                    minimum=0.0, maximum=1.0, value=0.5, step=0.05,
                    label="Temperature — LLM creativity",
                    info="0 = deterministic, 1 = creative. Lower for factual recall, higher for synthesis.",
                )
                slider_max_tokens = gr.Slider(
                    minimum=100, maximum=800, value=400, step=50,
                    label="Max tokens — response length",
                    info="Hard cap on LLM output. Raise if answers are getting cut off.",
                )

        chat_history = gr.State([])
        chatbot = gr.Chatbot(label="Ask about your activity", height=400)

        with gr.Row():
            chat_input = gr.Textbox(
                label=None,
                placeholder="Ask a question about your screen activity…",
                scale=5,
                show_label=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1)

        with gr.Accordion("Retrieved Context (debug)", open=False):
            retrieved_md = gr.Markdown("_Retrieved chunks appear here after your first query._")

        chat_inputs  = [chat_input, chat_history, host_url, model_select,
                        slider_top_k, slider_min_similarity, slider_temperature, slider_max_tokens]
        chat_outputs = [chatbot, chat_input, retrieved_md]

        send_btn.click(context_chat, inputs=chat_inputs, outputs=chat_outputs)
        chat_input.submit(context_chat, inputs=chat_inputs, outputs=chat_outputs)

        chatbot.change(lambda h: h, inputs=[chatbot], outputs=[chat_history])
        clear_btn.click(
            lambda: ([], [], "_Cleared._"),
            outputs=[chatbot, chat_history, retrieved_md],
        )

        caption_buffer = gr.State([])
        outline_state  = gr.State("")

        capture_timer     = gr.Timer(value=CAPTURE_CADENCE_S,     active=True)
        consolidate_timer = gr.Timer(value=CONSOLIDATE_CADENCE_S, active=True)
        compact_timer     = gr.Timer(value=COMPACT_CADENCE_S,     active=True)

        capture_timer.tick(
            caption_tick,
            inputs=[host_url, model_select, caption_buffer, log_box],
            outputs=[screenshot_img, focus_md, caption_buffer, log_box, buffer_md],
        )

        consolidate_timer.tick(
            consolidate_tick,
            inputs=[host_url, model_select, caption_buffer, outline_state, log_box],
            outputs=[summary_md, outline_state, caption_buffer, log_box, stats_md],
        )

        compact_timer.tick(
            compact_tick,
            inputs=[host_url, model_select, log_box],
            outputs=[log_box, stats_md],
        )

        start_btn.click(
            start,
            inputs=[log_box],
            outputs=[caption_buffer, outline_state, focus_md, summary_md, log_box],
        )

        stop_btn.click(stop, inputs=[log_box], outputs=[log_box])

    return demo


if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7868, share=False)
