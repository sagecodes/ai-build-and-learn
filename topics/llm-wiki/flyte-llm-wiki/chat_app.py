"""Gradio UI for the LLM Wiki: ingest / query / lint over Gemma 4 vLLM.

The wiki state lives at `/tmp/llm-wiki/` inside this pod. Knative scales the
app to zero after 5 min of idle time, so a long-running tab survives but a
fresh visit tomorrow morning will get an empty wiki; re-ingest the sources
you care about, or use the pipeline (`flyte run pipeline.py wiki_pipeline`)
for a reproducible build.

Same shape as topics/graphs-neo4j/graphrag-neo4j-flyte/chat_app.py, but talks
to Gemma 4 directly (no Neo4j) and reuses the prompt helpers from
`wiki_lib.py`.

Deploy (Gemma 4 vLLM must already be up):
    python chat_app.py
"""

from __future__ import annotations

from pathlib import Path

import flyte
import flyte.app


# Match the in-cluster Gemma 4 vLLM sibling app, same as the pipeline.
VLLM_APP_NAME = "gemma4-26b-a4b-it-vllm"
VLLM_MODEL_ID = "gemma-4-26b-a4b-it"

# Fully-qualified task name used by Flyte to resolve the latest pipeline run.
# Must match `<TaskEnvironment.name>.<function-name>` from config.py + pipeline.py.
PIPELINE_TASK = "llm-wiki-pipeline.wiki_pipeline"


chat_image = (
    flyte.Image.from_debian_base(
        name="llm-wiki-chat-image",
        registry="localhost:30000",
        platform=("linux/arm64",),
    )
    .with_pip_packages(
        # gr.Chatbot(type="messages") needs Gradio 5.x; same pin as the
        # sibling graphrag chat app.
        "gradio==5.42.0",
        "openai>=1.50.0",
        "httpx>=0.27.0",
        "trafilatura>=1.12.0",
    )
    .with_source_file(Path(__file__).parent / "wiki_lib.py")
)


env = flyte.app.AppEnvironment(
    name="llm-wiki-chat",
    image=chat_image,
    resources=flyte.Resources(cpu="2", memory="4Gi"),
    port=7860,
    requires_auth=False,
    parameters=[
        flyte.app.Parameter(
            name="vllm_url",
            value=f"http://{VLLM_APP_NAME}-flytesnacks-development.flyte.svc.cluster.local",
            env_var="VLLM_URL",
        ),
        flyte.app.Parameter(name="model_id", value=VLLM_MODEL_ID),
        flyte.app.Parameter(name="wiki_root", value="/tmp/llm-wiki"),
        # Mount the wiki Dir from the most recent wiki_pipeline run. Flyte
        # downloads it inside the pod and exposes the local path on
        # $WIKI_SEED_DIR. Override with WIKI_RUN_NAME=<run> python chat_app.py
        # to pin to a specific run.
        flyte.app.Parameter(
            name="wiki_seed_dir",
            type="directory",
            value=flyte.app.RunOutput(
                task_name=PIPELINE_TASK, type="directory"
            ),
            download=True,
            env_var="WIKI_SEED_DIR",
        ),
    ],
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=300),
)


@env.server
def chat_server(vllm_url: str, model_id: str, wiki_root: str):
    import sys
    import traceback
    try:
        _run(vllm_url, model_id, wiki_root)
    except BaseException as e:
        print(f"!!! chat_server crashed: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        raise


def _run(vllm_url: str, model_id: str, wiki_root: str):
    import logging
    import os
    import shutil
    import time
    from pathlib import Path

    import gradio as gr
    from openai import OpenAI

    import wiki_lib

    log = logging.getLogger("llm-wiki-chat")
    logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
    log.info(f"vLLM at {vllm_url}/v1 (model={model_id})")
    log.info(f"Wiki root: {wiki_root}")

    llm = OpenAI(
        base_url=vllm_url.rstrip("/") + "/v1",
        api_key="not-used",
        timeout=120.0,
    )

    root = Path(wiki_root)
    seed = os.environ.get("WIKI_SEED_DIR", "").strip()
    if seed:
        seed_path = Path(seed)
        if seed_path.is_dir():
            log.info(f"Seeding wiki from mounted pipeline output: {seed_path}")
            root.mkdir(parents=True, exist_ok=True)
            shutil.copytree(seed_path, root, dirs_exist_ok=True)
        else:
            log.warning(
                f"WIKI_SEED_DIR={seed_path} is not a directory; starting empty."
            )
    wiki_lib.init_layout(root)
    wiki_lib.regenerate_index(root)

    # ── Read-side helpers ────────────────────────────────────────────────────

    def status_md() -> str:
        pages = wiki_lib.read_pages(root)
        raw = wiki_lib.read_raw_summaries(root)
        return (
            "### Wiki status\n"
            f"- **Pages:** {len(pages)}\n"
            f"- **Sources:** {len(raw)}\n"
            f"- **Root:** `{root}`\n"
            "\n_State lives in the pod; wiped on Knative scale-to-zero "
            "(after 5 min idle)._"
        )

    def file_list() -> list[str]:
        files = [
            wiki_lib.WIKI_INDEX_FILE,
            wiki_lib.WIKI_LOG_FILE,
            wiki_lib.WIKI_SCHEMA_FILE,
        ]
        for p in sorted((root / wiki_lib.WIKI_RAW_DIR).glob("*.md")):
            files.append(f"raw/{p.name}")
        for p in sorted((root / wiki_lib.WIKI_PAGES_DIR).glob("*.md")):
            files.append(f"pages/{p.name}")
        return files

    def view_file(name: str) -> str:
        if not name:
            return ""
        p = root / name
        if not p.exists():
            return f"_File not found: `{name}`_"
        return p.read_text()

    def reset_wiki():
        if root.exists():
            shutil.rmtree(root)
        wiki_lib.init_layout(root)
        wiki_lib.regenerate_index(root)
        wiki_lib.append_log(root, f"## [{wiki_lib.now_utc()}] reset")
        return (
            status_md(),
            gr.update(choices=file_list(), value=wiki_lib.WIKI_INDEX_FILE),
            view_file(wiki_lib.WIKI_INDEX_FILE),
        )

    # ── Ingest ───────────────────────────────────────────────────────────────

    def ingest(source: str, title_override: str):
        src = (source or "").strip()
        if not src:
            yield (
                "_Provide a URL or paste some text first._",
                "",
                status_md(),
                gr.update(),
            )
            return
        try:
            yield ("**Step 1/3**: fetching source…", "", status_md(), gr.update())
            title, markdown, source_url = wiki_lib.fetch_to_markdown(src)
            if title_override and title_override.strip():
                title = title_override.strip()
            slug = wiki_lib.slugify(title)
            log.info(f"Ingest: '{title}' → raw/{slug}.md")

            yield (
                f"**Step 2/3**: summarizing `raw/{slug}.md`…",
                "",
                status_md(),
                gr.update(),
            )
            r1 = llm.chat.completions.create(
                model=model_id,
                messages=wiki_lib.prompt_source_summary(title, source_url, markdown),
                temperature=0.2,
                max_tokens=2048,
            )
            summary_md = r1.choices[0].message.content or ""
            (root / wiki_lib.WIKI_RAW_DIR / f"{slug}.md").write_text(
                summary_md.rstrip() + "\n"
            )

            yield (
                "**Step 3/3**: integrating into concept pages…",
                summary_md,
                status_md(),
                gr.update(choices=file_list()),
            )
            pages = wiki_lib.read_pages(root)
            index_md = (root / wiki_lib.WIKI_INDEX_FILE).read_text()
            pages_dump = wiki_lib.dump_pages_for_prompt(pages)
            r2 = llm.chat.completions.create(
                model=model_id,
                messages=wiki_lib.prompt_integrate(summary_md, index_md, pages_dump),
                temperature=0.2,
                max_tokens=6000,
                response_format={"type": "json_object"},
            )
            raw_ops = r2.choices[0].message.content or ""
            try:
                ops = wiki_lib.parse_json_blob(raw_ops).get("ops", []) or []
            except Exception as e:
                log.warning(f"integration JSON parse failed: {e}; head={raw_ops[:200]!r}")
                ops = []
            touched = wiki_lib.apply_page_ops(root, ops)

            wiki_lib.regenerate_index(root)
            wiki_lib.append_log(
                root,
                f"## [{wiki_lib.now_utc()}] ingest | {title}\n"
                f"- Source: {source_url or '(pasted text)'}\n"
                f"- Raw summary: [[raw/{slug}]]\n"
                f"- Pages touched: "
                + (
                    ", ".join(f"[[{s}]]" for s in touched)
                    if touched
                    else "_none_"
                ),
            )

            touched_list = (
                "\n".join(f"- `pages/{s}.md`" for s in touched) or "_none_"
            )
            done_md = (
                f"**Done.** Ingested `{title}` as `raw/{slug}.md`.\n\n"
                f"**Pages touched:**\n{touched_list}"
            )
            yield (
                done_md,
                summary_md,
                status_md(),
                gr.update(choices=file_list(), value=f"raw/{slug}.md"),
            )
        except Exception as e:
            log.exception("ingest failed")
            yield (
                f"**Error during ingest:** `{type(e).__name__}: {e}`",
                "",
                status_md(),
                gr.update(choices=file_list()),
            )

    # ── Query ────────────────────────────────────────────────────────────────

    def query(message, history):
        if not message or not message.strip():
            yield "", history, status_md()
            return
        question = message.strip()
        history = history + [
            {"role": "user", "content": question},
            {"role": "assistant", "content": ""},
        ]
        yield "", history, status_md()

        index_md = (root / wiki_lib.WIKI_INDEX_FILE).read_text()
        pick = llm.chat.completions.create(
            model=model_id,
            messages=wiki_lib.prompt_pick_pages(question, index_md),
            temperature=0.0,
            max_tokens=512,
            response_format={"type": "json_object"},
        )
        try:
            slugs = wiki_lib.parse_json_blob(
                pick.choices[0].message.content or ""
            ).get("slugs", []) or []
        except Exception:
            slugs = []

        pages = wiki_lib.read_pages(root)
        selected = {s: pages[s] for s in slugs if s in pages} or pages
        pages_dump = wiki_lib.dump_pages_for_prompt(selected, per_page_chars=3000)

        consulted = (
            ", ".join(f"`{s}`" for s in sorted(selected.keys())) or "_none_"
        )
        history[-1]["content"] = f"_Pages consulted: {consulted}_\n\n"
        yield "", history, status_md()

        stream = llm.chat.completions.create(
            model=model_id,
            messages=wiki_lib.prompt_answer(question, pages_dump),
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        buf = ""
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                history[-1]["content"] = (
                    f"_Pages consulted: {consulted}_\n\n{buf}"
                )
                yield "", history, status_md()
        finally:
            stream.close()

        wiki_lib.append_log(
            root,
            f"## [{wiki_lib.now_utc()}] query | {question[:80]}\n"
            f"- Pages consulted: "
            + (
                ", ".join(f"[[{s}]]" for s in sorted(selected.keys()))
                or "_none_"
            ),
        )

    # ── Lint ─────────────────────────────────────────────────────────────────

    def lint():
        det = wiki_lib.deterministic_lint(root)
        header_lines = [
            "## Stats",
            f"- Pages: {det['n_pages']}",
            f"- Raw summaries: {det['n_raw']}",
            f"- Orphans: {len(det['orphans'])}",
            f"- Broken links: {len(det['broken_links'])}",
        ]
        if det["orphans"]:
            header_lines.append("\n## Orphans")
            header_lines.extend(f"- `{s}`" for s in det["orphans"])
        if det["broken_links"]:
            header_lines.append("\n## Broken links")
            header_lines.extend(
                f"- `{src}` → `[[{tgt}]]` (no such page)"
                for src, tgt in det["broken_links"]
            )
        header = "\n".join(header_lines) + "\n\n"
        yield header

        index_md = (root / wiki_lib.WIKI_INDEX_FILE).read_text()
        pages = wiki_lib.read_pages(root)
        pages_dump = wiki_lib.dump_pages_for_prompt(pages, per_page_chars=2000)
        stream = llm.chat.completions.create(
            model=model_id,
            messages=wiki_lib.prompt_lint(index_md, pages_dump),
            temperature=0.3,
            max_tokens=2048,
            stream=True,
        )
        buf = ""
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                yield header + buf
        finally:
            stream.close()

        wiki_lib.append_log(
            root,
            f"## [{wiki_lib.now_utc()}] lint\n"
            f"- Orphans: {len(det['orphans'])}, "
            f"broken links: {len(det['broken_links'])}",
        )

    # ── UI ───────────────────────────────────────────────────────────────────

    with gr.Blocks(title=f"LLM Wiki ({model_id})") as demo:
        gr.Markdown(
            f"# LLM Wiki: Karpathy's pattern on Flyte 2\n"
            f"Model: `{model_id}` · in-cluster vLLM"
        )
        with gr.Row():
            with gr.Column(scale=1):
                status_box = gr.Markdown(status_md())
                reset_btn = gr.Button("Reset wiki", variant="stop")
            with gr.Column(scale=4):
                with gr.Tabs():
                    with gr.Tab("📥 Ingest"):
                        gr.Markdown(
                            "Paste a URL or a chunk of text. The LLM will "
                            "summarize it, then update concept pages across "
                            "the wiki."
                        )
                        src_input = gr.Textbox(
                            label="URL or text",
                            placeholder="https://… or paste any text",
                            lines=4,
                        )
                        title_box = gr.Textbox(
                            label="Title override (optional)",
                            placeholder="Empty → derived from page metadata "
                            "or first line of pasted text",
                        )
                        ingest_btn = gr.Button("Ingest", variant="primary")
                        ingest_status_md = gr.Markdown()
                        gr.Markdown("### Raw summary")
                        raw_summary_md = gr.Markdown()

                    with gr.Tab("💬 Query"):
                        gr.Markdown(
                            "Ask the wiki. The LLM picks relevant pages from "
                            "the index, then answers using only those pages."
                        )
                        chatbot = gr.Chatbot(
                            type="messages", label="Wiki Q&A", height=480
                        )
                        msg = gr.Textbox(
                            label="Your question",
                            placeholder="e.g. What is retrieval-augmented "
                            "generation?",
                        )
                        with gr.Row():
                            send = gr.Button("Send", variant="primary")
                            clear = gr.Button("Clear chat")

                    with gr.Tab("🧹 Lint"):
                        gr.Markdown(
                            "Audit the wiki: orphans, broken links, "
                            "contradictions, missing pages."
                        )
                        lint_btn = gr.Button("Run lint", variant="primary")
                        lint_md = gr.Markdown()

                    with gr.Tab("📂 Browse"):
                        gr.Markdown("Inspect any file in the wiki Dir.")
                        file_dd = gr.Dropdown(
                            choices=file_list(),
                            value=wiki_lib.WIKI_INDEX_FILE,
                            label="File",
                        )
                        refresh_btn = gr.Button("Refresh file list")
                        file_md = gr.Markdown(view_file(wiki_lib.WIKI_INDEX_FILE))

        ingest_btn.click(
            ingest,
            inputs=[src_input, title_box],
            outputs=[ingest_status_md, raw_summary_md, status_box, file_dd],
        )
        send.click(query, inputs=[msg, chatbot], outputs=[msg, chatbot, status_box])
        msg.submit(query, inputs=[msg, chatbot], outputs=[msg, chatbot, status_box])
        clear.click(lambda: [], outputs=chatbot)
        lint_btn.click(lint, inputs=[], outputs=[lint_md])
        reset_btn.click(
            reset_wiki, inputs=[], outputs=[status_box, file_dd, file_md]
        )
        file_dd.change(view_file, inputs=[file_dd], outputs=[file_md])
        refresh_btn.click(
            lambda: gr.update(choices=file_list()), outputs=[file_dd]
        )

    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    import os
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)

    deploy_env = env
    wiki_run = os.environ.get("WIKI_RUN_NAME", "").strip()
    if wiki_run:
        print(f"Pinning chat UI to pipeline run: {wiki_run}")
        deploy_env = env.clone_with(
            wiki_seed_dir=flyte.app.RunOutput(
                type="directory", run_name=wiki_run
            ),
        )
    else:
        print(f"Seeding chat UI from latest run of: {PIPELINE_TASK}")

    app = flyte.with_servecontext(interactive_mode=True).serve(deploy_env)
    print(f"LLM Wiki chat UI deployed: {app.url}")
