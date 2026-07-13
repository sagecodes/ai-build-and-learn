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
}


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

@dataclass(frozen=True)
class ImageModelSpec:
    key: str
    repo: str
    pipeline: str
    license: str
    gated: bool
    download_gb: float
    steps: int = 4
    guidance: float | None = 0.0
    dtype: str = "float16"
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
        download_gb=5.2,
        steps=4,
        guidance=0.0,
        notes="Fastest ungated first frame (~1s). Default.",
    ),
    # SDXL-Turbo: better frames, still few-step, ~14GB. Worth it when the first
    # frame is the thing you're judging.
    "sdxl-turbo": ImageModelSpec(
        key="sdxl-turbo",
        repo="stabilityai/sdxl-turbo",
        pipeline="AutoPipelineForText2Image",
        license="Stability Non-Commercial",
        gated=False,
        download_gb=14.4,
        steps=4,
        guidance=0.0,
        notes="Better first frame, still few-step. ~14GB.",
    ),
}

DEFAULT_IMAGE_MODEL = "sd-turbo"


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
