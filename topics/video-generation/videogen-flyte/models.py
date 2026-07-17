"""Registry of open-source video-generation models we compare on Flyte.

Each entry is a `VideoModelSpec`: which HuggingFace repo to pull, which diffusers
pipeline class loads it for text-to-video and (if supported) image-to-video, the
sampler defaults, and the part that matters most here: the *download patterns*.

── Why every spec carries allow/ignore patterns ────────────────────────────────
Video repos are far worse than image repos about shipping several mutually
redundant copies of the same model. A naive `snapshot_download` of
`tencent/HunyuanVideo-1.5` pulls **372GB** because the repo packs eleven separate
33GB transformers (480p/720p × t2v/i2v × distilled/not). You want one. Likewise
`Lightricks/LTX-2.3` is a flat repo where `ltx-2.3-22b-dev.safetensors`,
`-distilled.safetensors` and `-distilled-1.1.safetensors` are each 46GB and you
need exactly one of them.

So `allow_patterns` (a strict allowlist) or `ignore_patterns` (a denylist) is a
required field, not an optimization. `download_gb` is the *measured* size of what
we actually fetch, from the HF API. Check the README table against it.

── Sizes are on-disk, not in-memory ────────────────────────────────────────────
Wan's transformers are stored fp32 (the TI2V-5B transformer is 20GB on disk for a
5B model) and LTX-2.3's Gemma-3 text encoder config says float32 (~44GB on disk
for what becomes ~24GB in bf16). You pay that on the download, not in the GPU
pool: we load bf16. Don't infer "it won't fit" from the download size.

── The defaults are DEMO defaults, not the model card's ────────────────────────
The Spark is bandwidth-bound (~273GB/s vs ~672GB/s on a mid-range discrete card),
and video gen is the most bandwidth-hungry thing you can run on it. Reports on
NVIDIA's forums have Wan 2.2 I2V taking 15–30 minutes for a 5s 720p clip here. A
demo that takes half an hour per cell is a bad demo, so `steps`/`num_frames`/
resolution below are tuned for a ~2–5 minute cell. Each spec's `native` note says
what the model card actually recommends; pass --steps/--num-frames/--width/
--height to get there.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class VideoModelSpec:
    key: str                      # short handle used on the CLI and in reports
    repo: str                     # HuggingFace repo id
    pipeline: str                 # diffusers T2V pipeline class name
    family: str                   # backbone family, for the report
    license: str
    gated: bool                   # needs an HF_TOKEN that accepted the license
    download_gb: float            # measured size of what our patterns actually fetch

    # Estimated RESIDENT footprint once loaded in bf16, weights only: transformer +
    # text encoder + VAE. Activations and the VAE decode sit on top of this, which
    # is why the preflight check in videogen_core adds headroom rather than trusting
    # it exactly. This is the number that decides whether a model fits the GB10's
    # single 119.7GiB unified pool, NOT download_gb, which is inflated by fp32
    # storage. See videogen_core.guard_gpu_memory / preflight_fit.
    est_vram_gb: float = 0.0

    i2v_pipeline: str = ""        # image-to-video class; "" = T2V only
    dtype: str = "bfloat16"       # GB10 (Blackwell) is happiest in bf16
    vae_dtype: str = ""           # override; Wan's VAE must stay fp32 or it artifacts

    # Demo-tuned sampler defaults (see module docstring).
    steps: int = 30
    guidance: float | None = 5.0  # None = don't pass guidance_scale at all
    width: int = 832
    height: int = 480
    num_frames: int = 49
    fps: int = 16

    negative_prompt: str = ""     # model-specific default negative
    has_audio: bool = False       # LTX-2 generates a synced audio track
    distilled: bool = False       # few-step; uses the model's fixed sigma schedule

    # Selective download. Exactly one of these is normally set.
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = ()

    # Extra kwargs passed straight to the pipeline's __call__, as (name, value) pairs
    # (a tuple, not a dict, because this dataclass is frozen/hashable). For params
    # that only one family has and that generate_video shouldn't grow a branch for:
    # SkyReels' diffusion-forcing knobs (ar_step, overlap_history, ...) are the case
    # this exists for. Unknown names will TypeError at call time, which is the point:
    # fail loudly rather than silently ignore a setting you thought was applied.
    extra_call_kwargs: tuple[tuple[str, object], ...] = ()

    # Wan flow-matching schedule shift. The VACE model card is explicit that this is
    # resolution-dependent (3.0 for 480P, 5.0 for 720P) and sets it by rebuilding the
    # scheduler as UniPCMultistepScheduler.from_config(..., flow_shift=...). Left 0 =
    # "don't touch the scheduler", which is deliberate: the Wan/LTX specs above are
    # verified working on their shipped schedulers and this must not silently change
    # them. Only set it where a card asks for it.
    flow_shift: float = 0.0

    native: str = ""              # what the model card actually recommends
    notes: str = ""

    @property
    def supports_i2v(self) -> bool:
        return bool(self.i2v_pipeline)


# Junk that is never load-bearing for `from_pretrained`, in every repo.
_JUNK = ("*.md", "*.gif", "*.mp4", "*.png", "*.jpg", "assets/*", "examples/*")

# Wan's long default negative prompt (from the official model cards). It is doing
# real work: without it Wan drifts toward oversaturated, static, low-motion clips.
# The card ships it in Chinese; this is that prompt verbatim.
_WAN_NEGATIVE = (
    "色调艳丽,过曝,静态,细节模糊不清,字幕,风格,作品,画作,画面,静止,整体发灰,最差质量,"
    "低质量,JPEG压缩残留,丑陋的,残缺的,多余的手指,画得不好的手部,画得不好的脸部,畸形的,"
    "毁容的,形态畸形的肢体,手指融合,静止不动的画面,杂乱的背景,三条腿,背景人很多,倒着走"
)


MODELS: dict[str, VideoModelSpec] = {
    # ── The fast one. Start here; it's the only model that iterates quickly. ────
    # 1.3B, so the transformer is only 5.7GB and the Spark's bandwidth ceiling
    # stops mattering. Quality is clearly a step below the others, which is the
    # point: it's the "what does small buy you" datapoint in the grid.
    "wan21-t2v-1.3b": VideoModelSpec(
        key="wan21-t2v-1.3b",
        repo="Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
        pipeline="WanPipeline",
        family="DiT / flow (1.3B)",
        license="Apache-2.0",
        gated=False,
        download_gb=28.9,
        est_vram_gb=15.0,   # 1.3B transformer + UMT5-xxl text encoder is the bulk
        vae_dtype="float32",
        steps=30,
        guidance=5.0,
        width=832, height=480, num_frames=49, fps=16,
        negative_prompt=_WAN_NEGATIVE,
        ignore_patterns=_JUNK,
        native="832x480, 81 frames @16fps, 50 steps.",
        notes="Smallest credible open T2V. The fast iteration loop.",
    ),

    # ── The default. Apache-2.0, one checkpoint does BOTH t2v and i2v. ──────────
    # 34GB is the cheapest download of any current-generation model here, and
    # being a single checkpoint for text->video and image->video makes it the one
    # model that can demo the whole pipeline (image-gen first frame -> animate).
    "wan22-ti2v-5b": VideoModelSpec(
        key="wan22-ti2v-5b",
        repo="Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        pipeline="WanPipeline",
        i2v_pipeline="WanImageToVideoPipeline",
        family="DiT / flow (5B, T+I)",
        license="Apache-2.0",
        gated=False,
        download_gb=34.2,
        est_vram_gb=26.0,   # 5B transformer + UMT5-xxl + an fp32 VAE
        vae_dtype="float32",
        steps=30,
        guidance=5.0,
        width=832, height=480, num_frames=49, fps=24,
        negative_prompt=_WAN_NEGATIVE,
        ignore_patterns=_JUNK,
        native="1280x704, 121 frames @24fps, 50 steps.",
        notes="Best value: Apache-2.0, T2V+I2V in one 34GB checkpoint. The default.",
    ),

    # ── The frontier one. Only open model that generates SYNCED AUDIO. ──────────
    # 8-step distilled (CFG off), which partly offsets the 95GB download and the
    # 22B size. Lightricks explicitly markets LTX-2 as DGX Spark supported, and
    # ComfyUI users report ~3min for a short 720p clip on this box.
    #
    # NOTE the repo choice: `diffusers/LTX-2.3-Distilled-Diffusers` is the
    # diffusers-layout mirror. The native `Lightricks/LTX-2.3` repo is a 157GB
    # FLAT repo (three mutually-redundant 46GB .safetensors) that WanPipeline-style
    # `from_pretrained` cannot load. Don't point at it.
    "ltx2-distilled": VideoModelSpec(
        key="ltx2-distilled",
        repo="diffusers/LTX-2.3-Distilled-Diffusers",
        pipeline="LTX2Pipeline",
        i2v_pipeline="LTX2ImageToVideoPipeline",
        family="DiT (22B, distilled, +audio)",
        license="LTX-2 Community (free under $10M ARR)",
        gated=False,
        download_gb=101.7,  # measured on a real fetch; the HF API's blob sum said ~95
        est_vram_gb=72.0,   # 22B transformer + Gemma-3 encoder. The tight one on a GB10.
        steps=8,               # distilled: 8 fixed sigmas, see DISTILLED_SIGMA_VALUES
        guidance=1.0,          # distilled: CFG is off
        width=768, height=512, num_frames=57, fps=24,
        has_audio=True,
        distilled=True,
        # Everything else in this repo is load-bearing (transformer/, text_encoder/,
        # connectors/, vae/, audio_vae/, vocoder/). The one real redundancy is that
        # connectors/ ships BOTH a sharded set and a 6.3GB single-file copy, but
        # diffusers picks between them by heuristic, so cutting it risks a load
        # error to save 6GB of 95GB. Not worth it; we only drop the docs/media.
        ignore_patterns=_JUNK,
        native="768x512, 121 frames @24fps, 8 steps. Two-stage upsampler for full quality.",
        notes="22B, 8 steps, and it generates a synced audio track. The showpiece.",
    ),

    # ── The big MoE. Opt-in only: 126GB and slow. ───────────────────────────────
    # Two 14B experts (transformer/ = high-noise, transformer_2/ = low-noise); you
    # need BOTH, so there is nothing to trim. On the Spark expect 15-30 min/clip
    # unless you merge the 4-step lightx2v distill LoRAs. Kept out of DEFAULT.
    "wan22-t2v-a14b": VideoModelSpec(
        key="wan22-t2v-a14b",
        repo="Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        pipeline="WanPipeline",
        family="MoE DiT (2x14B experts)",
        license="Apache-2.0",
        gated=False,
        download_gb=126.2,
        est_vram_gb=68.0,   # BOTH 14B experts stay resident, plus the text encoder
        vae_dtype="float32",
        steps=30,
        guidance=5.0,
        width=832, height=480, num_frames=49, fps=24,
        negative_prompt=_WAN_NEGATIVE,
        ignore_patterns=_JUNK,
        native="1280x720, 81 frames @24fps, 50 steps.",
        notes="Highest-quality Wan. 126GB, slow on the Spark. Opt in deliberately.",
    ),

    # ── The 2024 baseline. Tiny, weak, but proves the harness end to end. ───────
    # 21GB and rock-solid diffusers support since forever. Useful as a smoke test
    # and as the "here's how far this moved in ~18 months" slide.
    "cogvideox-5b": VideoModelSpec(
        key="cogvideox-5b",
        repo="zai-org/CogVideoX-5b",
        pipeline="CogVideoXPipeline",
        i2v_pipeline="CogVideoXImageToVideoPipeline",
        family="DiT (5B, 2024)",
        license="CogVideoX License",
        gated=False,
        download_gb=21.5,
        est_vram_gb=20.0,   # 5B transformer + T5-XXL
        dtype="bfloat16",
        steps=50,
        guidance=6.0,
        width=720, height=480, num_frames=49, fps=8,
        ignore_patterns=_JUNK,
        native="720x480, 49 frames @8fps, 50 steps.",
        notes="The 2024 baseline. Cheap to pull; good smoke test.",
    ),

    # ────────────────────────────────────────────────────────────────────────────
    # Added for the overnight suite. Every repo below was checked against the HF
    # API (exists, ungated, size) and every pipeline class against the installed
    # diffusers 0.39.0. But NONE has been run on this box yet, so the sampler
    # defaults come from each model card rather than from measurement, and
    # est_vram_gb is computed from parameter counts rather than observed. Treat a
    # failure here as expected-ish; the report renders a per-cell error and the
    # other models carry on.
    # ────────────────────────────────────────────────────────────────────────────

    # The third major player, alongside Wan and LTX. 8.3B going toe to toe with
    # models 3x its size is the headline.
    # NOTE the repo: the monolithic `tencent/HunyuanVideo-1.5` is a 372GB trap (it
    # packs ELEVEN 33GB transformers: 480p/720p x t2v/i2v x distilled/not). These
    # pre-split community repos are the fix: one variant, 53GB. Also note diffusers'
    # own docstring points at `hunyuanvideo-community/HunyuanVideo-1.5-480p_t2v`,
    # which 404s; the real id has `-Diffusers-` in the middle.
    "hunyuan-1.5-t2v": VideoModelSpec(
        key="hunyuan-1.5-t2v",
        repo="hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_t2v",
        pipeline="HunyuanVideo15Pipeline",
        family="DiT (8.3B)",
        license="Tencent Hunyuan Community",
        gated=False,
        # 79.1 measured on a real fetch. The HF tree API's blob sum said 53.4, so it
        # under-reports here (as it did for LTX: 95 claimed vs 101.7 actual). Trust a
        # real download over the API when budgeting disk.
        download_gb=79.1,
        est_vram_gb=34.0,   # 8.3B transformer (stored fp32) + Qwen2.5-VL-7B encoder
        steps=30,           # card says 50; trimmed for a demo-sized cell
        # No CFG knob: guidance is embedded in the transformer, so
        # HunyuanVideo15Pipeline.__call__ has no guidance_scale parameter (nor a
        # callback_on_step_end, so it's the one model with no live step counter).
        guidance=None,
        width=832, height=480, num_frames=49, fps=24,
        ignore_patterns=_JUNK,
        native="480p, 121 frames @24fps, 50 steps.",
        notes="Tencent's compact flagship. Strong cinematic quality. Untested here.",
    ),

    # MIT. Not Apache, not 'community', not 'non-commercial': MIT. That makes it the
    # most permissively licensed video model in open source, which is a real result
    # independent of how it scores on quality.
    "kandinsky5-lite": VideoModelSpec(
        key="kandinsky5-lite",
        repo="kandinskylab/Kandinsky-5.0-T2V-Lite-sft-5s-Diffusers",
        pipeline="Kandinsky5T2VPipeline",
        family="DiT (2B)",
        license="MIT",
        gated=False,
        download_gb=23.4,
        est_vram_gb=23.0,   # tiny 2B transformer; the Qwen2.5-VL encoder IS the cost
        steps=30,           # card says 50
        guidance=5.0,
        width=768, height=512, num_frames=49, fps=24,
        # The repo ships flax/tf/pytorch-bin duplicates of the text encoder (~5GB of
        # pure waste that from_pretrained never reads). _JUNK already drops *.bin,
        # *.msgpack and *.h5 via the shared list below.
        ignore_patterns=_JUNK + ("*.msgpack", "*.h5", "*.bin"),
        native="512x768, 121 frames @24fps, 50 steps.",
        notes="MIT licensed. Positioned by its authors as a Wan rival. Untested here.",
    ),

    # Diffusion forcing: generates in autoregressive chunks with an overlapping
    # history, so clip length is effectively unbounded rather than fixed by the
    # latent shape. That is a genuinely different axis from every other model here,
    # which is why it's in despite being another 1.3B.
    "skyreels-v2-df-1.3b": VideoModelSpec(
        key="skyreels-v2-df-1.3b",
        repo="Skywork/SkyReels-V2-DF-1.3B-540P-Diffusers",
        pipeline="SkyReelsV2DiffusionForcingPipeline",
        family="Diffusion forcing (1.3B)",
        license="SkyReels (open)",
        gated=False,
        download_gb=29.0,
        est_vram_gb=15.0,   # Wan-based: 1.3B transformer + UMT5-XXL
        vae_dtype="float32",  # Wan VAE, same fp32 requirement
        steps=30,
        guidance=6.0,
        width=960, height=544, num_frames=49, fps=24,
        negative_prompt=_WAN_NEGATIVE,   # Wan lineage, same failure mode
        ignore_patterns=_JUNK,
        native="544x960, 97 frames @24fps, 30 steps, ar_step=5, causal_block_size=5.",
        notes="Long-video via autoregressive chunks. The odd one out, on purpose.",
    ),

    # ── The video-to-video / control model. See vace.py. ────────────────────────
    # The cheapest real capability in this registry: 19GB, the smallest thing here,
    # and the only NEW AXIS (every other model is text/image -> video; this one takes
    # a video in). VACE unifies restyle, control (depth/pose/scribble),
    # reference-to-video and inpainting behind one pipeline.
    #
    # It is NOT driven like the other specs: WanVACEPipeline.__call__ takes
    # `video` / `mask` / `reference_images` rather than just a prompt, so it gets its
    # own task in vace.py instead of going through generate_for_model. It is in this
    # registry only so it shares fetch_weights' cache and download watchdog.
    #
    # The 14B (Wan-AI/Wan2.1-VACE-14B-diffusers) is the same pipeline at 75.1GB if
    # the 1.3B looks as weak as skyreels-v2-df-1.3b did.
    "wan21-vace-1.3b": VideoModelSpec(
        key="wan21-vace-1.3b",
        repo="Wan-AI/Wan2.1-VACE-1.3B-diffusers",
        pipeline="WanVACEPipeline",
        family="VACE control/v2v (1.3B)",
        license="Apache-2.0",
        gated=False,
        download_gb=19.0,   # HF API blob sum; the smallest model we carry
        est_vram_gb=15.0,   # 1.3B transformer + UMT5-XXL, same shape as wan21-t2v-1.3b
        vae_dtype="float32",   # Wan VAE, same fp32 requirement as the rest of the family
        steps=30,
        guidance=5.0,
        width=832, height=480, num_frames=49, fps=16,
        negative_prompt=_WAN_NEGATIVE,
        ignore_patterns=_JUNK,
        flow_shift=3.0,     # the card: 3.0 for 480P, 5.0 for 720P
        native="832x480, 81 frames @16fps, 50 steps, flow_shift=3.0 (480P).",
        notes="Video-to-video and control. The only model here that takes video IN. Untested.",
    ),

    # The long-video showpiece, and the reason `long_video.py` has a foil.
    #
    # Same architecture and the SAME pipeline class as the 1.3B above, just 10x the
    # parameters: the 1.3B is the model that made long video look bad, and size is the
    # most likely reason. 80.4GB and ungated.
    #
    # Why this is the *principled* long video: diffusion forcing carries the previous
    # chunk's LATENTS forward, not a decoded RGB frame. It's right there in the
    # diffusers source (pipeline_skyreels_v2_diffusion_forcing.py):
    #
    #     prefix_video_latents = video_latents[:, :, -overlap_history_latent_frames:]
    #
    # That is exactly the thing `long_video.py` cannot do (see its docstring): our
    # chain pays a VAE decode->encode round trip per hop and accumulates +32% contrast
    # over 4 hops. This model never leaves latent space, so it shouldn't.
    #
    # NOTE the extra_call_kwargs: without them this is just another 49-frame T2V model.
    # `overlap_history` is REQUIRED once num_frames > base_num_frames (the pipeline
    # raises otherwise and recommends 17 or 37); ar_step/causal_block_size are what
    # make it asynchronous rather than synchronous; addnoise_condition=20 is the
    # card's consistency knob for long generation.
    "skyreels-v2-df-14b": VideoModelSpec(
        key="skyreels-v2-df-14b",
        repo="Skywork/SkyReels-V2-DF-14B-540P-Diffusers",
        pipeline="SkyReelsV2DiffusionForcingPipeline",
        family="Diffusion forcing (14B, latent history)",
        license="SkyReels (open)",
        gated=False,
        download_gb=80.4,   # measured via the HF API blob sum; verify on a real pull
        est_vram_gb=42.0,   # 14B transformer bf16 (~28GB) + UMT5-XXL + VAE
        vae_dtype="float32",   # Wan lineage: the VAE artifacts badly in bf16
        steps=30,
        guidance=6.0,
        width=960, height=544, num_frames=97, fps=24,
        negative_prompt=_WAN_NEGATIVE,   # Wan lineage, same drift-to-static failure
        ignore_patterns=_JUNK,
        extra_call_kwargs=(
            ("ar_step", 5),
            ("causal_block_size", 5),
            ("base_num_frames", 97),
            ("overlap_history", 17),
            ("addnoise_condition", 20),
        ),
        native="544x960, 97+ frames @24fps, 30 steps, ar_step=5, causal_block_size=5.",
        notes="Long video done properly: latent history, not a re-encoded frame. Untested here.",
    ),

    # A second modern Apache-2.0 small model, to check that wan21-1.3b's weaknesses
    # are a size story and not a Wan story.
    "motif-video-2b": VideoModelSpec(
        key="motif-video-2b",
        repo="Motif-Technologies/Motif-Video-2B",
        pipeline="MotifVideoPipeline",
        i2v_pipeline="MotifVideoImage2VideoPipeline",
        family="DiT (2B)",
        license="Apache-2.0",
        gated=False,
        download_gb=17.3,
        est_vram_gb=13.0,
        steps=30,           # card says 50
        # Guidance-distilled: MotifVideoPipeline.__call__ takes no guidance_scale.
        guidance=None,
        width=832, height=480, num_frames=49, fps=24,
        ignore_patterns=_JUNK,
        native="1280x736, 121 frames @24fps, 50 steps.",
        notes="Modern Apache-2.0 2B. The 'is small necessarily bad?' datapoint.",
    ),

    # The thematically perfect model for THIS box. Linear attention with a
    # constant-memory KV cache: our bottleneck is memory bandwidth (~273GB/s), and
    # linear attention is precisely the bet that you can spend less bandwidth per
    # token. If any model punches above its weight on a Spark, it should be this one.
    # Caveat: ~0 downloads on HF, so we are the ones finding the bugs. Cheap enough
    # (14GB) that a failed cell costs almost nothing.
    "sana-video-2b": VideoModelSpec(
        key="sana-video-2b",
        repo="Efficient-Large-Model/SANA-Video_2B_480p_diffusers",
        pipeline="SanaVideoPipeline",
        family="Linear-attention DiT (2B)",
        license="NVIDIA open (non-commercial)",
        gated=False,
        download_gb=14.0,
        est_vram_gb=10.0,
        steps=30,
        guidance=6.0,
        width=832, height=480, num_frames=49, fps=16,
        ignore_patterns=_JUNK,
        native="480p. Linear attention, constant-memory KV cache for long video.",
        notes="Linear attention: the on-theme bet for a bandwidth-bound box. Unproven.",
    ),
}


# The overnight lineup: everything that fits, spread across sizes, licenses and
# architectures. They serialize on the single GPU, so this is a multi-hour job.
# wan22-t2v-a14b is deliberately NOT here: 126GB and ~15-30 min per clip undistilled,
# which would eat the whole night for one model. See the README TODO on the lightx2v
# 4-step distill LoRAs, which is the fix.
OVERNIGHT_MODELS: list[str] = [
    "wan21-t2v-1.3b",       # measured: 210s/clip, peak 16GB
    "wan22-ti2v-5b",        # measured: 139s/clip, peak 26GB
    "ltx2-distilled",       # measured:  33s/clip, peak 73GB, + audio
    "hunyuan-1.5-t2v",
    "kandinsky5-lite",
    "skyreels-v2-df-1.3b",
    "motif-video-2b",
    "sana-video-2b",
]


# The default comparison set: the two Wan models. Together they're a 63GB pull and
# each cell lands in a couple of minutes, so a first run on a cold box finishes
# during the stream rather than after it.
#
# The HEADLINE comparison is Wan-vs-LTX (Apache small vs frontier-with-audio):
#     --models '["wan22-ti2v-5b","ltx2-distilled"]'
# It's not the default only because ltx2-distilled is a 95GB first download.
DEFAULT_MODELS: list[str] = ["wan21-t2v-1.3b", "wan22-ti2v-5b"]


# ── First-frame image models (for the image-to-video path) ──────────────────────
#
# The I2V pipeline needs a starting image. Rather than make you find one, we can
# generate it with a text-to-image model first: prompt -> image -> video, all in
# one run. These are a deliberately small subset of the image-generation demo's
# registry (topics/image-generation), picked to be FAST and UNGATED so the extra
# hop costs a few seconds and no license clicks.
#
# ⚠️ These re-download into THIS project. Flyte's cache is keyed on
# (project, task, version, inputs), so pointing .flyte/config.yaml at
# `video-generation` means the image-generation project's already-cached SDXL does
# NOT carry over. That's why the options here are the cheap ones: sd-turbo is a
# 5GB pull that generates a 512px frame in about a second. It's a one-time cost,
# cached from then on.

# The image repos need an allowlist for exactly the same reason the video repos do,
# and we learned it the same way: a naive pull of `stabilityai/sdxl-turbo` fetches
# **55.5GB**, because the repo ships FOUR redundant copies of the same model:
#
#   sd_xl_turbo_1.0.safetensors        13.9GB  single-file fp32  (from_pretrained can't use it)
#   sd_xl_turbo_1.0_fp16.safetensors    6.9GB  single-file fp16  (same)
#   unet/…model.onnx_data + friends    13.6GB  a full ONNX export (same)
#   unet/…model.safetensors + friends  13.8GB  fp32 diffusers
#   unet/…model.fp16.safetensors …      6.9GB  fp16 diffusers   <- the only set we want
#
# We load in fp16, so we want the last one and nothing else. sd-turbo is the same
# story at smaller scale (13.0GB naive -> 2.6GB). Keeping the fp16 files ONLY works
# if the loader also passes variant="fp16", or diffusers goes looking for the fp32
# filenames and fails: see `variant` below and load_image_pipeline.
_FP16_DIFFUSERS = (
    "model_index.json",     # names the components; from_pretrained reads it first
    "*/*.json",             # per-component config.json + scheduler/tokenizer configs
    "*/*.txt",              # tokenizer merges.txt
    "*/*.fp16.safetensors",  # the weights, fp16 only
)


@dataclass(frozen=True)
class ImageModelSpec:
    key: str
    repo: str
    pipeline: str
    license: str
    gated: bool
    download_gb: float          # size of what the patterns below ACTUALLY fetch
    steps: int = 4
    guidance: float | None = 0.0
    dtype: str = "float16"
    # Backbone family, used to reject a LoRA/base mismatch BEFORE loading (an SDXL
    # adapter on an SD2.1 UNet dies deep inside diffusers on a shape error). Must
    # match LoRASpec.base.
    base: str = "sdxl"
    # "fp16" -> from_pretrained(variant="fp16"), which is what makes it look for
    # `*.fp16.safetensors`. Required whenever allow_patterns keeps only fp16 weights.
    variant: str = ""
    allow_patterns: tuple[str, ...] = ()
    ignore_patterns: tuple[str, ...] = field(default=_JUNK)
    notes: str = ""


IMAGE_MODELS: dict[str, ImageModelSpec] = {
    # 1-4 step distilled SD2.1. ~5GB, ungated, and a first frame in ~1s. The right
    # default: the star of this demo is the video model, not the frame.
    "sd-turbo": ImageModelSpec(
        key="sd-turbo",
        repo="stabilityai/sd-turbo",
        pipeline="AutoPipelineForText2Image",
        license="Stability Non-Commercial",
        gated=False,
        download_gb=2.6,    # measured via the HF API with the patterns below (13.0 without)
        steps=4,
        guidance=0.0,
        base="sd2.1",       # NOT SDXL: the LORAS below will not load onto this
        variant="fp16",
        allow_patterns=_FP16_DIFFUSERS,
        notes="Fastest ungated first frame (~1s). Default. Weak at 512px, and it shows.",
    ),
    # SDXL-Turbo: better frames, still few-step. The upgrade when the first frame is
    # the thing limiting the clip, which on the I2V path it usually is.
    "sdxl-turbo": ImageModelSpec(
        key="sdxl-turbo",
        repo="stabilityai/sdxl-turbo",
        pipeline="AutoPipelineForText2Image",
        license="Stability Non-Commercial",
        gated=False,
        download_gb=6.9,    # measured via the HF API with the patterns below (55.5 without)
        steps=4,
        guidance=0.0,
        variant="fp16",
        allow_patterns=_FP16_DIFFUSERS,
        notes="Better first frame, still few-step. 6.9GB with the fp16 allowlist.",
    ),
}

DEFAULT_IMAGE_MODEL = "sd-turbo"


# ── LoRAs for the first frame ───────────────────────────────────────────────────
#
# The cheap way to get a fine-tuned look MOVING. Training a LoRA on a *video* model
# is an overnight job; these are image LoRAs (a few hundred MB) that style the first
# frame, and the video model then animates whatever it's handed. The LoRA rides on
# the frame, not the motion, so it costs nothing at video time.
#
# `animate --lora <key>` resolves a key below. It also accepts:
#   - any HF repo id      ("TheLastBen/Papercut_SDXL")
#   - an s3:// Dir URI    (what the image-gen demo's `train_lora` outputs, read
#                          cross-project via Dir.from_existing_remote)
#
# ⚠️ TRIGGER WORDS ARE NOT OPTIONAL. A style LoRA is trained to fire on a specific
# token; without it in the prompt the adapter is loaded, fused, and does close to
# nothing, which reads as "the LoRA didn't work". `trigger` is prepended to the frame
# prompt automatically.
#
# ⚠️ These are all SDXL LoRAs, so use them with --image_model sdxl-turbo. sdxl-turbo
# IS SDXL (few-step distilled), so the UNet shapes match and the adapter loads. That
# a base-SDXL-trained LoRA transfers cleanly onto the *turbo* checkpoint is an
# assumption, not a verified fact: expect the style to land, but weaker than on stock
# SDXL. Untested here.

@dataclass(frozen=True)
class LoRASpec:
    key: str
    repo: str
    weight_name: str
    trigger: str          # prepended to the prompt; the LoRA is inert without it
    base: str = "sdxl"
    size_mb: float = 0.0
    notes: str = ""


LORAS: dict[str, LoRASpec] = {
    # On-theme for this repo's boat prompts: a paper boat, rendered as papercut.
    "papercut": LoRASpec(
        key="papercut",
        repo="TheLastBen/Papercut_SDXL",
        weight_name="papercut.safetensors",
        trigger="papercut",
        size_mb=340.8,
        notes="Layered paper-cut style. The demo default: pairs with the boat prompts.",
    ),
    "pixel-art": LoRASpec(
        key="pixel-art",
        repo="nerijs/pixel-art-xl",
        weight_name="pixel-art-xl.safetensors",
        trigger="pixel art",
        size_mb=170.5,
        notes="Flat pixel art. The most obvious style delta, so a good first test.",
    ),
    "3d-render": LoRASpec(
        key="3d-render",
        repo="goofyai/3d_render_style_xl",
        weight_name="3d_render_style_xl.safetensors",
        trigger="3d style, 3d render",
        size_mb=85.5,
        notes="Stylized 3D render look. Smallest of the three.",
    ),
}


def get_lora(key_or_repo: str) -> LoRASpec | None:
    """Resolve a --lora value to a LoRASpec, or None if it's a raw repo/URI.

    Returning None is not a failure: it means "not one of ours", and the caller
    loads it as a plain HF repo id or s3:// Dir with no trigger word of its own.
    """
    return LORAS.get(key_or_repo)


def get_spec(key: str) -> VideoModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(f"Unknown video model {key!r}. Known: {', '.join(MODELS)}") from None


def get_image_spec(key: str) -> ImageModelSpec:
    try:
        return IMAGE_MODELS[key]
    except KeyError:
        raise ValueError(
            f"Unknown image model {key!r}. Known: {', '.join(IMAGE_MODELS)}"
        ) from None


def resolve_models(keys: list[str] | None) -> list[VideoModelSpec]:
    """Map a list of keys (or None -> DEFAULT_MODELS) to VideoModelSpecs, in order."""
    return [get_spec(k) for k in (keys or DEFAULT_MODELS)]
