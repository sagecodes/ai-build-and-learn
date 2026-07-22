"""Voice chat: speak to a local LLM and hear it answer, every layer swappable.

    mic -> Whisper -> ollama (LLM) -> Kokoro/Chatterbox -> your speakers
            ^^^^^^^   ^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^
            all three are dropdowns, changeable mid-conversation

Deploy to the devbox:
    flyte deploy voice_app.py env
    # or run it straight on the host for a fast iteration loop:
    python voice_app.py

This is the payoff for the two pipelines next door: `compare_pipeline` tells you which
TTS model to put in the dropdown, `clone_pipeline` gives it your voice.

── Why this app holds the GPU when the other studios don't ──────────────────────
The image and video studios are deliberately thin LAUNCHERS: they submit runs and hold
no model, so the Spark's single GPU stays free for the pipelines. This app cannot do
that, because holding Whisper and a TTS model resident IS the latency feature. So while
it is up it owns the GPU and the compare/clone runs queue behind it. `scaledown_after`
is the release valve: idle for 15 minutes and the pod exits, freeing the GPU.

── Where the latency actually goes ──────────────────────────────────────────────
The number on screen is time-to-first-audio, because that is what a conversation feels
like. It breaks down as: Whisper on your clip (~0.3s) + the LLM's first sentence
(~1s at the 14 tok/s measured here) + Kokoro synthesizing it (~0.04s). Everything after
that is free, because the LLM generates ~4x faster than the audio plays, so synthesis
stays ahead of your ears for the rest of the reply.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from pathlib import Path

import flyte
import flyte.app

from config import VOICE_APP_NAME, VOICE_APP_PORT, voice_app_pod, voice_image

log = logging.getLogger("voice.app")
logging.basicConfig(level=logging.INFO)

_here = Path(__file__).parent

# The app needs the whole engine room in its image: unlike the launcher studios, this
# one actually loads models, so tts_core/models/voice_core all ride along.
_bundled = voice_image.with_source_file(_here / "voice_core.py") \
                      .with_source_file(_here / "tts_core.py") \
                      .with_source_file(_here / "models.py")

# The backend decides the SHAPE of this pod, which is the whole point of the split:
#
#   vllm    the LLM lives in its own GPU pod (voice_vllm.py), so this one is CPU-only.
#           Kokoro still streams (~5x real-time on CPU) and Whisper runs int8-ish on
#           CPU. Crucially the GPU stays free, so the compare/clone pipelines still run.
#   ollama  the LLM lives HERE, so this pod takes the GPU, and TTS/STT get it too
#           (Kokoro at 78x, Whisper fp16, and Chatterbox voice cloning becomes possible).
#           While it is up, the pipelines queue behind it.
_BACKEND = os.environ.get("LLM_BACKEND", "vllm").strip().lower()
_IS_OLLAMA = _BACKEND == "ollama"

_env_vars = {
    "HF_HOME": "/root/.cache/huggingface",
    "LLM_BACKEND": _BACKEND,
    **({"OLLAMA_MODELS": "/root/.ollama/models"} if _IS_OLLAMA else {}),
    # Passed through from the deploy shell so the app knows where vLLM is.
    **{k: os.environ[k] for k in ("VLLM_URL", "VLLM_MODEL_ID", "GRADIO_SHARE")
       if k in os.environ},
}

env = flyte.app.AppEnvironment(
    name=VOICE_APP_NAME,
    image=_bundled,
    # AppEnvironment drops flyte.Resources(gpu=...), so a GPU has to come from a
    # PodTemplate. The CPU deployment uses plain Resources and no template.
    **({"pod_template": voice_app_pod} if _IS_OLLAMA
       else {"resources": flyte.Resources(cpu="8", memory="24Gi", disk="40Gi")}),
    port=VOICE_APP_PORT,
    requires_auth=False,
    # Scale to zero when idle. This matters far more in the ollama deployment, where
    # idling holds the box's only GPU; on CPU it is just tidiness.
    scaling=flyte.app.Scaling(replicas=(0, 1), scaledown_after=900),
    env_vars=_env_vars,
)


# ── ollama, running inside this pod ──────────────────────────────────────────────

def start_ollama(timeout: int = 120) -> bool:
    """Start `ollama serve` as a subprocess and wait for its API.

    In-pod rather than a separate service because a second GPU pod cannot exist on this
    box (one node, `nvidia.com/gpu: 1`, no time-slicing), so the LLM has to live in the
    same pod as the TTS and STT models. Talking to it over localhost HTTP rather than
    importing it in-process is what gives us the model dropdown for free.
    """
    import urllib.request

    def up() -> bool:
        try:
            urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2)
            return True
        except Exception:
            return False

    if up():
        log.info("[ollama] already running")
        return True

    log.info("[ollama] starting server")
    subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                     stderr=subprocess.STDOUT)
    deadline = time.time() + timeout
    while time.time() < deadline:
        if up():
            log.info("[ollama] up")
            return True
        time.sleep(1.5)
    log.error(f"[ollama] did not come up within {timeout}s")
    return False


# ── The UI ───────────────────────────────────────────────────────────────────────

def build_ui():
    import gradio as gr

    import voice_core as vc

    # Only the ollama deployment runs a server in-pod; the vllm one just needs a URL.
    if vc.LLM_BACKEND == "ollama":
        llm_ok = start_ollama()
        llm_where = "ollama, in this pod"
    else:
        llm_ok = bool(vc.VLLM_URL)
        llm_where = f"vLLM at {vc.VLLM_URL}" if llm_ok else "vLLM (VLLM_URL not set)"

    speaker = vc.Speaker()
    transcriber = vc.Transcriber()
    log.info(f"[app] backend={vc.LLM_BACKEND} llm={llm_where} device={transcriber.device}")

    # A reference clip makes the cloned voice available. Gitignored, so it is normal for
    # this to be absent; the checkbox just stays off.
    ref_wav, ref_txt = _here / "refs" / "sage.wav", _here / "refs" / "sage.txt"
    if ref_wav.exists():
        speaker.ref_wav = str(ref_wav)
        speaker.ref_text = ref_txt.read_text().strip() if ref_txt.exists() else ""
        log.info(f"[clone] reference available: {ref_wav}")

    llm_choices = vc.list_models(fallback=vc.LLM_MODELS) if llm_ok else vc.LLM_MODELS
    llm_default = vc.DEFAULT_LLM if vc.DEFAULT_LLM in llm_choices else llm_choices[0]

    # Cloning needs a clone-capable engine on a GPU. In the CPU deployment Chatterbox
    # runs at a small fraction of real time, so the option is hidden rather than offered
    # and then disappointing.
    tts_choices = vc.available_tts()
    can_clone = bool(speaker.ref_wav) and transcriber.device == "cuda"

    def converse(audio, history, llm, stt, tts, voice, clone, temperature, min_chars):
        """Drive one turn, streaming audio chunks and the transcript as they appear."""
        history = history or []
        if not audio:
            yield history, None, "Record something first.", history
            return

        transcript, last = "", None
        for turn, chunk in vc.converse(
            audio, history, speaker, transcriber,
            llm=llm, stt=stt, tts=tts, voice=voice, clone=bool(clone),
            temperature=float(temperature), min_chars=int(min_chars),
        ):
            last = turn
            if turn.error:
                yield history, None, f"⚠️ {turn.error}", history
                return
            transcript = (
                f"**You:** {turn.user_text}\n\n"
                f"**Assistant:** {turn.reply or '_thinking..._'}"
            )
            if turn.first_audio_seconds:
                transcript += f"\n\n`first audio {turn.first_audio_seconds:.2f}s`"
            # chunk is a .wav path when a sentence finished synthesizing, else None.
            # Gradio appends each one to the streaming player as it arrives.
            yield history, chunk, transcript, history

        if last and last.reply:
            new_history = history + [
                {"role": "user", "content": last.user_text},
                {"role": "assistant", "content": last.reply},
            ]
            stats = (
                f"\n\n`first audio {last.first_audio_seconds:.2f}s · "
                f"stt {last.stt_seconds:.2f}s · total {last.total_seconds:.2f}s · "
                f"{last.audio_seconds:.1f}s of speech in {last.chunks} chunks · "
                f"{last.realtime_factor:.1f}x real-time`"
            )
            yield new_history, None, (
                f"**You:** {last.user_text}\n\n**Assistant:** {last.reply}{stats}"
            ), new_history

    with gr.Blocks(title="Voice chat") as demo:
        gr.Markdown(
            "# Voice chat\n"
            "Speak, and hear the answer as it's generated. Every layer is swappable: "
            "**speech-to-text**, the **LLM**, and the **voice** it answers in.\n\n"
            "Only Kokoro is faster than real time on this box (78x), so it's the one "
            "that streams smoothly. The others are here so you can hear the difference."
        )
        gr.Markdown(
            f"_LLM: **{llm_where}** · this pod is **{transcriber.device.upper()}**_"
            + ("" if llm_ok else "  \n> ⚠️ **the LLM isn't reachable** — replies will fail.")
        )

        with gr.Row():
            llm = gr.Dropdown(
                llm_choices, value=llm_default, label="LLM",
                info=("Swaps on demand, seconds." if vc.LLM_BACKEND == "ollama"
                      else "vLLM serves one model per app; switching means a redeploy."))
            stt = gr.Dropdown(list(vc.STT_MODELS), value=vc.DEFAULT_STT,
                              label="Speech-to-text",
                              info="Its own layer, so the LLM slot stays free.")
            tts = gr.Dropdown(tts_choices, value=tts_choices[0], label="Voice engine",
                              info="Only what this image can actually import.")
            voice = gr.Dropdown(vc.KOKORO_VOICES, value="af_heart", label="Kokoro voice")

        with gr.Row():
            clone = gr.Checkbox(
                value=False, label="Answer in my cloned voice",
                info=("Uses refs/sage.wav via Chatterbox. Slower than real time, so "
                      "expect a lag before each reply." if can_clone
                      else "Needs a GPU pod (LLM_BACKEND=ollama) and refs/sage.wav."),
                interactive=can_clone,
            )
            temperature = gr.Slider(0.0, 1.5, value=0.6, step=0.1, label="Temperature")
            min_chars = gr.Slider(
                20, 200, value=vc.MIN_CHUNK_CHARS, step=10, label="Chunk size (chars)",
                info="Smaller = audio starts sooner but choppier. Under ~1s of audio "
                     "per chunk, playback stutters.")

        history_state = gr.State([])

        with gr.Row():
            with gr.Column():
                mic = gr.Audio(sources=["microphone"], type="filepath", label="Speak")
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear = gr.Button("Clear")
            with gr.Column():
                # streaming=True is what lets the generator push chunk after chunk into
                # one continuous player instead of replacing the clip each time.
                reply = gr.Audio(label="Reply", streaming=True, autoplay=True)
                transcript = gr.Markdown()

        chat = gr.Chatbot(type="messages", label="Conversation", height=260)

        send.click(
            converse,
            inputs=[mic, history_state, llm, stt, tts, voice, clone, temperature, min_chars],
            outputs=[history_state, reply, transcript, chat],
        )
        clear.click(lambda: ([], None, "", []),
                    outputs=[history_state, reply, transcript, chat])

    return demo


@env.server
def voice_server():
    build_ui().launch(
        server_name="0.0.0.0", server_port=VOICE_APP_PORT,
        share=os.environ.get("GRADIO_SHARE") == "1",
    )


if __name__ == "__main__":
    # RUN_MODE=host runs the identical UI on the devbox itself, which is the fast
    # iteration loop: no image build, no pod, and it uses the ollama already installed
    # there with gemma4:26b pulled. Otherwise this deploys the app to the cluster.
    if os.environ.get("RUN_MODE") == "host":
        build_ui().launch(
            server_name="0.0.0.0", server_port=VOICE_APP_PORT,
            share=os.environ.get("GRADIO_SHARE") == "1",
        )
    else:
        flyte.init_from_config(root_dir=_here)
        app = flyte.with_servecontext(interactive_mode=True).serve(env)
        print(f"Voice chat deployed: {app.url}")
