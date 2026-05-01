"""Agent with persistent memory — Chroma + Gemma 4 vLLM, backed by HF Hub.

The vector store is the agent's long-running context. Each turn:
  1. Embed the user query, retrieve top-k memories.
  2. Stream Gemma's answer with those memories injected as system context.
  3. Make a second small Gemma call to extract atomic facts/preferences from
     the exchange.
  4. Embed those facts and write them back to Chroma.

Memory persists across pod restarts via a tarballed snapshot of the Chroma
persist dir, stored in a HuggingFace model repo. `@env.on_startup` pulls and
extracts; `@env.on_shutdown` (Knative SIGTERM, ~30s grace) packs and pushes.
A "Save to HF" UI button does the same on demand.

Deploy:
    python chat_app.py
"""

from __future__ import annotations

import flyte
import flyte.app


# ── Endpoint + repo defaults ──────────────────────────────────────────────────
# Mirror of gemma4-dgx-devbox/config.py — copy the two strings, not an import.

VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
COLLECTION_NAME = "agent_memory"
DEFAULT_TOP_K = 5

# HF repo holding the memory tarball. Plain `<user>/<name>` → repo_type="model".
HF_MEMORY_REPO = "sagecodes/agent-mem"
HF_MEMORY_REPO_TYPE = "model"
HF_MEMORY_FILENAME = "memory.tar.gz"

# Local path inside the pod where Chroma persists. The startup hook untars
# into here; the shutdown hook tars from here.
LOCAL_CHROMA_DIR = "/tmp/agent_memory_chroma"


# ── Image ─────────────────────────────────────────────────────────────────────

chat_image = (
    flyte.Image.from_debian_base(
        name="agent-memory-image",
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    .with_pip_packages(
        # gr.Chatbot(type="messages") needs Gradio 5.x.
        "gradio==5.42.0",
        "openai>=1.50.0",
        "chromadb>=0.5.0",
        "sentence-transformers>=3.0.0",
        "huggingface_hub>=0.24.0",
    )
)


# ── App env ───────────────────────────────────────────────────────────────────

env = flyte.app.AppEnvironment(
    name="agent-memory-chat",
    image=chat_image,
    # Idle pod is just doing single-query BGE encoding + chat-stream proxying
    # to the vLLM endpoint. 1 CPU is plenty and keeps the single-node devbox
    # from going into "Insufficient cpu" when sibling apps are also ACTIVE.
    resources=flyte.Resources(cpu="1", memory="4Gi"),
    port=7860,
    requires_auth=False,
    secrets=[flyte.Secret(key="HF_TOKEN", as_env_var="HF_TOKEN")],
    parameters=[
        flyte.app.Parameter(
            name="vllm_url",
            value=f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local",
            env_var="VLLM_URL",
        ),
        flyte.app.Parameter(name="model_id", value=VLLM_MODEL_ID),
        flyte.app.Parameter(name="embedding_model", value=EMBEDDING_MODEL),
        flyte.app.Parameter(name="collection_name", value=COLLECTION_NAME),
        flyte.app.Parameter(name="default_top_k", value=str(DEFAULT_TOP_K)),
        flyte.app.Parameter(name="hf_repo", value=HF_MEMORY_REPO),
        flyte.app.Parameter(name="hf_repo_type", value=HF_MEMORY_REPO_TYPE),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)


# ── State shared across hooks ─────────────────────────────────────────────────
# Populated by on_startup, used by server, flushed by on_shutdown / save button.

state: dict = {}


# ── Helpers (importable both at startup and from the save button) ─────────────

def _pull_memory_tarball(hf_repo: str, repo_type: str, dest_dir: str) -> None:
    """Download memory.tar.gz from HF and extract into dest_dir.

    On 404 / first-run / missing file we just create the empty dir — the
    collection will be created fresh.
    """
    import os
    import tarfile
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import EntryNotFoundError, RepositoryNotFoundError

    os.makedirs(dest_dir, exist_ok=True)
    try:
        path = hf_hub_download(
            repo_id=hf_repo,
            repo_type=repo_type,
            filename=HF_MEMORY_FILENAME,
        )
    except (EntryNotFoundError, RepositoryNotFoundError, FileNotFoundError):
        print(f"[memory] No prior tarball at {hf_repo}:{HF_MEMORY_FILENAME} — starting fresh.", flush=True)
        return
    except Exception as e:
        # Treat 404-ish errors as "no prior memory" rather than failing the boot.
        msg = str(e)
        if "404" in msg or "not found" in msg.lower():
            print(f"[memory] No prior tarball ({type(e).__name__}) — starting fresh.", flush=True)
            return
        raise

    with tarfile.open(path, "r:gz") as tar:
        tar.extractall(dest_dir)
    print(f"[memory] Restored Chroma snapshot from {hf_repo} → {dest_dir}", flush=True)


def _push_memory_tarball(hf_repo: str, repo_type: str, src_dir: str, note: str = "") -> str:
    """Tar src_dir and upload as memory.tar.gz to the HF repo. Returns the commit URL."""
    import datetime as _dt
    import os
    import tarfile
    import tempfile
    from huggingface_hub import upload_file

    if not os.path.isdir(src_dir):
        raise RuntimeError(f"Cannot save: {src_dir} doesn't exist")

    fd, tar_path = tempfile.mkstemp(prefix="agent_memory_", suffix=".tar.gz")
    os.close(fd)
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(src_dir, arcname=".")

    msg = f"agent-memory snapshot {_dt.datetime.utcnow().isoformat(timespec='seconds')}Z"
    if note:
        msg += f" — {note}"
    commit = upload_file(
        path_or_fileobj=tar_path,
        path_in_repo=HF_MEMORY_FILENAME,
        repo_id=hf_repo,
        repo_type=repo_type,
        commit_message=msg,
    )
    print(f"[memory] Pushed snapshot to {hf_repo} ({commit.commit_url if hasattr(commit, 'commit_url') else commit})", flush=True)
    os.remove(tar_path)
    return getattr(commit, "commit_url", str(commit))


# ── Lifecycle hooks ───────────────────────────────────────────────────────────

@env.on_startup
async def load_memory(
    vllm_url: str,
    model_id: str,
    embedding_model: str,
    collection_name: str,
    default_top_k: str,
    hf_repo: str,
    hf_repo_type: str,
) -> None:
    """Pull the snapshot from HF, untar, build Chroma client + encoder."""
    import chromadb
    from sentence_transformers import SentenceTransformer

    print(f"[memory] Loading encoder: {embedding_model}", flush=True)
    encoder = SentenceTransformer(embedding_model)

    print(f"[memory] Restoring snapshot from {hf_repo}…", flush=True)
    _pull_memory_tarball(hf_repo, hf_repo_type, LOCAL_CHROMA_DIR)

    chroma_client = chromadb.PersistentClient(path=LOCAL_CHROMA_DIR)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        metadata={"embedding_model": embedding_model, "hnsw:space": "cosine"},
    )
    print(f"[memory] Collection '{collection_name}' loaded with {collection.count()} memories", flush=True)

    state["encoder"] = encoder
    state["collection"] = collection
    state["hf_repo"] = hf_repo
    state["hf_repo_type"] = hf_repo_type


@env.on_shutdown
async def save_memory_on_shutdown(
    vllm_url: str,
    model_id: str,
    embedding_model: str,
    collection_name: str,
    default_top_k: str,
    hf_repo: str,
    hf_repo_type: str,
) -> None:
    """Knative scale-down hook: pack and push the current Chroma dir to HF."""
    try:
        _push_memory_tarball(hf_repo, hf_repo_type, LOCAL_CHROMA_DIR, note="auto-save on shutdown")
    except Exception as e:
        print(f"[memory] Shutdown save failed: {type(e).__name__}: {e}", flush=True)


# ── Gemma thinking-block parser (same as the sibling chat apps) ───────────────

def _split_thinking(text: str) -> tuple[str, str]:
    OPEN, OPEN_TAIL = "<|channel>", "thought\n"
    CLOSE = "<channel|>"
    j = text.find(OPEN)
    if j == -1:
        return "", text.strip()
    pre = text[:j]
    rest = text[j + len(OPEN):]
    if rest.startswith(OPEN_TAIL):
        rest = rest[len(OPEN_TAIL):]
    k = rest.find(CLOSE)
    if k == -1:
        thinking, answer = rest, pre
    else:
        thinking = rest[:k]
        answer = (pre + rest[k + len(CLOSE):])
    return thinking.strip(), answer.strip()


# ── CSS for the side panels (CLAUDE.md: classes only, no inline styles) ───────

PANELS_CSS = """
.mem-panel { display: flex; flex-direction: column; gap: 10px; }
.mem-empty {
    color: var(--body-text-color-subdued);
    font-style: italic;
    padding: 10px;
}
.mem-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 8px;
    padding: 8px 10px;
    background: var(--background-fill-secondary);
}
.mem-card-new {
    border: 1px solid var(--color-accent);
    border-radius: 8px;
    padding: 8px 10px;
    background: var(--background-fill-secondary);
}
.mem-meta {
    display: flex;
    justify-content: space-between;
    font-size: 0.75rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 4px;
}
.mem-rank { font-weight: 600; }
.mem-score { font-variant-numeric: tabular-nums; }
.mem-text {
    font-size: 0.9rem;
    line-height: 1.4;
    white-space: pre-wrap;
    word-break: break-word;
}
.mem-header {
    font-size: 0.85rem;
    color: var(--body-text-color-subdued);
    margin-bottom: 4px;
}
.mem-status {
    padding: 8px 12px;
    border-radius: 6px;
    background: var(--background-fill-secondary);
    font-size: 0.85rem;
}
"""


def _esc(s: str) -> str:
    return (s or "").replace("<", "&lt;").replace(">", "&gt;")


def _render_retrieved(memories: list[dict]) -> str:
    if not memories:
        return '<div class="mem-empty">No memories retrieved this turn.</div>'
    cards = []
    for i, m in enumerate(memories, 1):
        sim = max(0.0, 1.0 - (m["distance"] / 2.0))
        cards.append(
            '<div class="mem-card">'
            f'<div class="mem-meta"><span class="mem-rank">#{i}</span>'
            f'<span class="mem-score">sim {sim:.3f}</span></div>'
            f'<div class="mem-text">{_esc(m["text"])}</div>'
            '</div>'
        )
    return f'<div class="mem-panel"><div class="mem-header">Retrieved (top {len(memories)})</div>{"".join(cards)}</div>'


def _render_written(memories: list[str]) -> str:
    if not memories:
        return '<div class="mem-empty">No new memories written this turn.</div>'
    cards = [
        f'<div class="mem-card-new"><div class="mem-text">+ {_esc(m)}</div></div>'
        for m in memories
    ]
    return f'<div class="mem-panel"><div class="mem-header">Wrote {len(memories)} new</div>{"".join(cards)}</div>'


# ── Server ────────────────────────────────────────────────────────────────────

@env.server
def chat_server(
    vllm_url: str,
    model_id: str,
    embedding_model: str,
    collection_name: str,
    default_top_k: str,
    hf_repo: str,
    hf_repo_type: str,
):
    import sys
    import traceback
    try:
        _run(vllm_url, model_id, embedding_model, collection_name,
             int(default_top_k), hf_repo, hf_repo_type)
    except BaseException as e:
        print(f"!!! chat_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _run(
    vllm_url: str,
    model_id: str,
    embedding_model: str,
    collection_name: str,
    default_top_k: int,
    hf_repo: str,
    hf_repo_type: str,
):
    import json
    import re
    import time
    import uuid

    import gradio as gr
    from openai import OpenAI

    encoder = state["encoder"]
    collection = state["collection"]
    llm = OpenAI(base_url=vllm_url.rstrip("/") + "/v1", api_key="not-used")

    print(f"[chat_server] gradio version: {gr.__version__}", flush=True)
    print(f"[chat_server] vLLM at {vllm_url}/v1 (model={model_id})", flush=True)

    DEFAULT_SYSTEM = (
        "You are an assistant with persistent memory of the user across "
        "conversations. Use the MEMORIES section as context about the user "
        "(their preferences, projects, prior decisions). Stay natural — don't "
        "list memories at the user; weave them in only when relevant."
    )

    EXTRACTION_SYSTEM = (
        "You extract durable facts about the user from a single exchange. "
        "Output ONLY a JSON array of strings. Each string is one atomic fact, "
        "preference, or decision the user explicitly stated or strongly implied. "
        "Skip questions, speculation, and trivial pleasantries. If nothing is "
        "worth remembering, output []."
    )
    EXTRACTION_EXAMPLES = (
        '["User\'s name is Sage.", '
        '"User is building an AI demo for a Thursday livestream.", '
        '"User prefers terse responses without filler."]'
    )

    MAX_TOTAL_TOKENS = 4096

    # ── Per-turn helpers ──────────────────────────────────────────────────────

    def retrieve(query: str, top_k: int) -> list[dict]:
        if not query.strip() or collection.count() == 0:
            return []
        vec = encoder.encode([query], normalize_embeddings=True, convert_to_numpy=True).tolist()
        res = collection.query(
            query_embeddings=vec,
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        out = []
        for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            out.append({
                "text": doc,
                "distance": float(dist),
                "ts": (meta or {}).get("ts", ""),
            })
        return out

    def build_context_block(mems: list[dict]) -> str:
        if not mems:
            return ""
        lines = ["MEMORIES (most relevant first):"]
        for i, m in enumerate(mems, 1):
            lines.append(f"[#{i}] {m['text']}")
        return "\n".join(lines)

    def extract_memories(user_msg: str, assistant_msg: str) -> list[str]:
        """Second small LLM call. Thinking off; expect a JSON array."""
        prompt = (
            f"Last user message: {user_msg!r}\n"
            f"Last assistant message: {assistant_msg!r}\n\n"
            f"Example output:\n{EXTRACTION_EXAMPLES}\n\n"
            "Output ONLY the JSON array. No preface, no explanation."
        )
        try:
            resp = llm.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": EXTRACTION_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
                max_tokens=512,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "skip_special_tokens": True,
                },
            )
            raw = (resp.choices[0].message.content or "").strip()
        except Exception as e:
            print(f"[memory] extraction call failed: {e}", flush=True)
            return []

        # Find the first JSON array in the response — Gemma sometimes prefaces.
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if not match:
            return []
        try:
            facts = json.loads(match.group(0))
        except json.JSONDecodeError:
            return []
        return [f.strip() for f in facts if isinstance(f, str) and f.strip()]

    def write_memories(facts: list[str]) -> None:
        if not facts:
            return
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        vectors = encoder.encode(facts, normalize_embeddings=True, convert_to_numpy=True).tolist()
        collection.add(
            ids=[str(uuid.uuid4()) for _ in facts],
            documents=facts,
            embeddings=vectors,
            metadatas=[{"ts": ts} for _ in facts],
        )

    # ── Chat handler ──────────────────────────────────────────────────────────

    def chat(message, history, system_prompt, use_memory, top_k,
             enable_thinking, temperature, top_p):
        if not message or not message.strip():
            yield "", history, _render_retrieved([]), _render_written([]), _status_html()
            return

        mems = retrieve(message, int(top_k)) if use_memory else []
        retrieved_html = _render_retrieved(mems)

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
            {"role": "assistant", "content": ""},
        ]
        yield "", history, retrieved_html, _render_written([]), _status_html()

        sys_text = (system_prompt or DEFAULT_SYSTEM).strip()
        ctx = build_context_block(mems)
        if ctx:
            sys_text = f"{sys_text}\n\n{ctx}"

        msgs = [{"role": "system", "content": sys_text}]
        for t in history[:-2]:
            if "metadata" in t:
                continue
            msgs.append({"role": t["role"], "content": t["content"]})

        stream = llm.chat.completions.create(
            model=model_id,
            messages=msgs,
            stream=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=MAX_TOTAL_TOKENS,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                "skip_special_tokens": False,
            },
        )

        buf = ""
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                thinking, answer = _split_thinking(buf)
                history[-2]["content"] = thinking
                history[-1]["content"] = answer
                yield "", history, retrieved_html, _render_written([]), _status_html()
        finally:
            stream.close()

        if not history[-2]["content"]:
            history.pop(-2)

        # Memory-extraction pass
        final_answer = history[-1]["content"]
        new_facts = extract_memories(message, final_answer)
        write_memories(new_facts)
        yield "", history, retrieved_html, _render_written(new_facts), _status_html()

    # ── Save / status helpers ─────────────────────────────────────────────────

    def _status_html() -> str:
        return f'<div class="mem-status">Memories in store: <b>{collection.count()}</b></div>'

    def save_now():
        try:
            url = _push_memory_tarball(hf_repo, hf_repo_type, LOCAL_CHROMA_DIR, note="manual save from UI")
            return f'<div class="mem-status">✅ Saved to <a href="{url}" target="_blank">{hf_repo}</a> ({collection.count()} memories)</div>'
        except Exception as e:
            return f'<div class="mem-status">❌ Save failed: {_esc(type(e).__name__)}: {_esc(str(e))}</div>'

    # ── Gradio UI ─────────────────────────────────────────────────────────────

    with gr.Blocks(title=f"Agent Memory ({model_id})", css=PANELS_CSS) as demo:
        gr.Markdown(
            f"# Agent Memory — Chroma + Gemma 4\n"
            f"Model: `{model_id}` · Encoder: `{embedding_model}` · "
            f"Backed by `{hf_repo}` ({hf_repo_type} repo)"
        )
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            top_k = gr.Slider(1, 10, value=default_top_k, step=1, label="Top-k memories")
        with gr.Row():
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM, label="System prompt", lines=2, scale=4,
            )
            use_memory = gr.Checkbox(value=True, label="Use memory", scale=1)
            enable_thinking = gr.Checkbox(value=True, label="Enable thinking", scale=1)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(type="messages", label="Conversation", height=520)
                msg = gr.Textbox(label="Your message", placeholder="Tell me about yourself…")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear chat (memory stays)")
                    save_btn = gr.Button("💾 Save to HF", variant="secondary")
            with gr.Column(scale=2):
                gr.Markdown("### Retrieved this turn")
                retrieved_view = gr.HTML(value=_render_retrieved([]))
                gr.Markdown("### Written this turn")
                written_view = gr.HTML(value=_render_written([]))
                gr.Markdown("### Status")
                status_view = gr.HTML(value=_status_html())

        inputs = [msg, chatbot, system_prompt, use_memory, top_k,
                  enable_thinking, temperature, top_p]
        outputs = [msg, chatbot, retrieved_view, written_view, status_view]
        msg.submit(chat, inputs=inputs, outputs=outputs)
        send.click(chat, inputs=inputs, outputs=outputs)
        clear.click(
            lambda: ([], _render_retrieved([]), _render_written([]), _status_html()),
            outputs=[chatbot, retrieved_view, written_view, status_view],
        )
        save_btn.click(save_now, outputs=status_view)

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    # Fallback if launch() ever stops blocking — keep the pod alive so Knative
    # doesn't reap it before on_shutdown gets a chance to fire.
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    app = flyte.with_servecontext(interactive_mode=True).serve(env)
    print(f"Agent memory chat deployed: {app.url}")
