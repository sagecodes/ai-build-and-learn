"""Flyte-free core for the Magenta RealTime 2 live app.

Holds the weight download + model load + Gradio UI so it can run two ways:
  - inside the Flyte app pod (mrt_app.py wraps `run_ui` in an @env.server), and
  - directly on the Spark host (run_local.py calls `run_ui` with share=True),
    which uses the host GPU natively and gives a public gradio.live URL.

Nothing here imports flyte, so the local runner needs only jax/magenta/gradio.
"""

from __future__ import annotations

import os

# HuggingFace repo holding the open weights (Apache-2.0 code, CC-BY-4.0 weights).
HF_REPO = "google/magenta-realtime-2"

# Selectable models: display size name -> checkpoint filename in the HF repo.
# small (230M) keeps ahead of real-time playback; base (2.4B) sounds richer but
# generates slower per chunk, so it can buffer.
MODELS = {
    "mrt2_small": "mrt2_small.safetensors",
    "mrt2_base": "mrt2_base.safetensors",
}


def ensure_weights(checkpoint: str) -> str:
    """Download the resources + one checkpoint from HF into MAGENTA_HOME.

    Magenta RT's `paths` module resolves everything under
    `$MAGENTA_HOME/magenta-rt-v2/`:
        resources/musiccoca/      (style text/audio encoder)
        resources/spectrostream/  (audio codec)
        checkpoints/<name>.safetensors  (the transformer)

    We call huggingface_hub directly so this is self-contained and only pulls the
    one checkpoint we load. Returns the local magenta-rt-v2 root. Idempotent:
    hf snapshot_download skips files already present.
    """
    from huggingface_hub import snapshot_download

    base = os.environ.get("MAGENTA_HOME", "/tmp/magenta")
    local_root = os.path.join(base, "magenta-rt-v2")
    os.makedirs(local_root, exist_ok=True)

    print(f"[mrt] downloading weights from {HF_REPO} -> {local_root} ({checkpoint})", flush=True)
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=local_root,
        token=os.environ.get("HF_TOKEN"),
        allow_patterns=[
            "resources/**",            # musiccoca + spectrostream (both needed)
            f"checkpoints/{checkpoint}",  # just the transformer we load
        ],
    )
    print("[mrt] weights ready", flush=True)
    return local_root


def run_ui(model_size: str = "mrt2_small", checkpoint: str | None = None):
    """Download weights, load Magenta RT2, and launch the live Gradio UI. Blocks.

    `model_size` is the model selected at startup; the UI dropdown can switch to
    any other key in MODELS, lazy-loading it on first use. Set GRADIO_SHARE=1 in
    the environment for a public https tunnel.
    """
    import threading
    import time

    import numpy as np

    if model_size not in MODELS:
        raise ValueError(f"Unknown model_size {model_size!r}; choices: {list(MODELS)}")

    # MAGENTA_HOME must be set before importing magenta_rt (paths reads it at
    # import time).
    os.environ.setdefault("MAGENTA_HOME", "/tmp/magenta")

    # gr.Audio(streaming=True) transcodes each chunk to ADTS via pydub, which
    # shells out to ffmpeg/ffprobe. The Flyte image apt-installs ffmpeg; on a
    # bare host that lacks it, the static-ffmpeg wheel vends both binaries and
    # puts them on PATH. Best-effort: if neither is present, streaming fails loud.
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except Exception as e:
        print(f"[mrt] static-ffmpeg unavailable ({type(e).__name__}: {e}); "
              "relying on system ffmpeg for audio streaming", flush=True)

    import gradio as gr
    from magenta_rt import MagentaRT2Jax

    print(f"[mrt] gradio={gr.__version__}", flush=True)

    SR = 48000  # SpectroStream output rate (stereo)
    FRAMES_PER_CHUNK = int(os.environ.get("MRT_FRAMES_PER_CHUNK", "50"))  # 25 frames = 1s
    SECONDS_PER_CHUNK = FRAMES_PER_CHUNK / 25.0
    DEFAULT_PROMPT = "warm lo-fi hip hop, mellow rhodes, dusty drums"

    # Lazily-built, cached model instances keyed by size name. Loading is the
    # expensive part (download miss + JIT compile), so we keep each one warm
    # once built. The GB10's unified memory holds both small and base fine.
    _model_cache: dict = {}
    _model_lock = threading.Lock()

    def get_model(size: str):
        with _model_lock:
            if size in _model_cache:
                return _model_cache[size]
            ensure_weights(MODELS[size])
            t0 = time.time()
            print(f"[mrt] loading {size} (JIT-compiling)…", flush=True)
            m = MagentaRT2Jax(
                size=size,
                checkpoint=MODELS[size],
                temperature=1.3,
                top_k=40,
                cfg_musiccoca=3.0,
                cfg_notes=1.0,
            )
            print(f"[mrt] {size} ready in {time.time() - t0:.1f}s", flush=True)
            _model_cache[size] = m
            return m

    # Pre-load the startup model so Start works immediately.
    ensure_weights(MODELS[model_size])
    initial = get_model(model_size)

    # Shared, mutable control surface. The streaming generator reads it every
    # chunk; the Apply/knob/model events write it. A lock guards swaps so the
    # generator never reads a half-updated style or model.
    lock = threading.Lock()
    state: dict = {
        "running": False,
        "model": initial,         # the MagentaRT2Jax instance the loop generates with
        "model_size": model_size,
        "style_live": initial.embed_style(DEFAULT_PROMPT, use_mapper=True),
        "style_target": None,     # set by apply_prompt; generator glides toward it
        "morph_remaining": 0,     # chunks left in the current glide
        "temperature": 1.3,
        "top_k": 40,
        "cfg": 3.0,
        "prompt": DEFAULT_PROMPT,
    }

    def _to_int16_stereo(wav) -> np.ndarray:
        # wav.samples is float32 [n, 2] in [-1, 1]; gr.Audio streams (sr, ndarray).
        s = np.clip(wav.samples, -1.0, 1.0)
        return (s * 32767.0).astype(np.int16)

    # ── Control handlers ──────────────────────────────────────────────────────

    def switch_model(size: str):
        """Load (if needed) and make `size` the model the live loop generates with.

        Style embeddings live in the shared MusicCoCa space, so the current
        style carries across. The audio `state` is architecture-specific, so the
        loop resets it on a model change (a brief re-seed, not a full restart).
        """
        if size == state["model_size"]:
            return f"model: **{size}**"
        yield f"⏳ loading **{size}**… (first time can take ~40s)"
        m = get_model(size)  # blocks here on first load; the loop keeps the old model meanwhile
        with lock:
            state["model"] = m
            state["model_size"] = size
        yield f"model: **{size}** ✅"

    def apply_prompt(prompt: str, morph_chunks: float):
        """Embed `prompt` (with the current model) and set it as the glide target."""
        prompt = (prompt or "").strip()
        if not prompt:
            return "⚠️ empty prompt"
        with lock:
            model = state["model"]
        target = model.embed_style(prompt, use_mapper=True)
        with lock:
            state["style_target"] = target
            state["morph_remaining"] = max(0, int(morph_chunks))
            state["prompt"] = prompt
            if int(morph_chunks) <= 0:
                state["style_live"] = target  # instant switch, no glide
                state["style_target"] = None
        how = "instantly" if int(morph_chunks) <= 0 else f"over ~{morph_chunks * SECONDS_PER_CHUNK:.0f}s"
        return f"🎚️ shifting to **{prompt}** {how}"

    def set_knobs(temperature: float, top_k: float, cfg: float):
        with lock:
            state["temperature"] = float(temperature)
            state["top_k"] = int(top_k)
            state["cfg"] = float(cfg)
        return f"temp={temperature:g} · top_k={int(top_k)} · cfg={cfg:g}"

    def stop():
        with lock:
            state["running"] = False
        return "⏹️ stopped"

    # ── The streaming generation loop ─────────────────────────────────────────

    def run_stream():
        """Infinite chunk loop. Yields (sr, int16[n,2]) for gr.Audio streaming."""
        with lock:
            if state["running"]:
                return  # already streaming (Start pressed twice): don't start a 2nd loop
            state["running"] = True

        gen_state = None       # audio context for seamless joins
        active_model = None    # detect model switches to reset gen_state
        try:
            while True:
                with lock:
                    if not state["running"]:
                        break
                    model = state["model"]
                    # Glide the live style toward the target, one chunk at a time.
                    if state["style_target"] is not None and state["morph_remaining"] > 0:
                        a, b = state["style_live"], state["style_target"]
                        state["style_live"] = a + (b - a) / state["morph_remaining"]
                        state["morph_remaining"] -= 1
                        if state["morph_remaining"] == 0:
                            state["style_live"] = state["style_target"]
                            state["style_target"] = None
                    style = state["style_live"]
                    temperature = state["temperature"]
                    top_k = state["top_k"]
                    cfg = state["cfg"]

                if model is not active_model:
                    gen_state = None  # model changed: re-seed audio context
                    active_model = model

                wav, gen_state = model.generate(
                    style=style,
                    frames=FRAMES_PER_CHUNK,
                    state=gen_state,
                    temperature=temperature,
                    top_k=top_k,
                    cfg_musiccoca=cfg,
                )
                yield (SR, _to_int16_stereo(wav))
        finally:
            with lock:
                state["running"] = False

    # ── UI ────────────────────────────────────────────────────────────────────

    with gr.Blocks(title="Magenta RealTime 2: Live") as demo:
        gr.Markdown(
            f"# 🎹 Magenta RealTime 2: Live\n"
            f"{SECONDS_PER_CHUNK:g}s chunks @ {SR // 1000}kHz stereo on GPU. "
            "Press **Start**, then type a vibe and **Apply** to morph the track live. "
            "`small` keeps ahead of playback; `base` sounds richer but can buffer."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(
                    value=DEFAULT_PROMPT, label="Vibe / style prompt", lines=2,
                    placeholder="e.g. 'driving techno, acid bassline' or 'ambient drone, deep reverb'",
                )
                morph = gr.Slider(
                    0, 16, value=4, step=1, label="Morph over N chunks",
                    info=f"0 = instant switch · 4 ≈ {4 * SECONDS_PER_CHUNK:.0f}s glide",
                )
                with gr.Row():
                    apply_btn = gr.Button("Apply prompt", variant="primary")
                    start_btn = gr.Button("▶ Start")
                    stop_btn = gr.Button("⏹ Stop")
                status = gr.Markdown("_idle, press Start_")
            with gr.Column(scale=1):
                model_dd = gr.Dropdown(
                    choices=list(MODELS), value=model_size, label="Model",
                    info="base lazy-loads on first pick (~40s)",
                )
                model_status = gr.Markdown(f"model: **{model_size}**")
                temperature = gr.Slider(0.5, 2.0, value=1.3, step=0.05, label="Temperature")
                top_k = gr.Slider(1, 256, value=40, step=1, label="Top-k")
                cfg = gr.Slider(0.0, 7.0, value=3.0, step=0.1, label="Style guidance (CFG)")
                knob_status = gr.Markdown()

        audio = gr.Audio(
            label="Live output", streaming=True, autoplay=True,
            show_download_button=False,
        )

        # Streaming generator drives the audio component; Start kicks it off.
        start_btn.click(run_stream, outputs=audio)
        apply_btn.click(apply_prompt, inputs=[prompt, morph], outputs=status)
        prompt.submit(apply_prompt, inputs=[prompt, morph], outputs=status)
        stop_btn.click(stop, outputs=status)
        model_dd.change(switch_model, inputs=model_dd, outputs=model_status)
        for k in (temperature, top_k, cfg):
            k.release(set_knobs, inputs=[temperature, top_k, cfg], outputs=knob_status)

    demo.queue(default_concurrency_limit=8)
    share = os.environ.get("GRADIO_SHARE", "0") == "1"
    _, local_url, share_url = demo.launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("MRT_PORT", "7870")),
        share=share, prevent_thread_lock=True,
    )
    print(f"[mrt] local URL: {local_url}", flush=True)
    print(f"[mrt] PUBLIC HTTPS URL: {share_url}" if share_url
          else "[mrt] no share URL (set GRADIO_SHARE=1)", flush=True)
    while True:
        time.sleep(3600)
