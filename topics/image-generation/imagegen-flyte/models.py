"""Registry of open-source text-to-image models we compare on Flyte.

Each entry is a `ModelSpec`: what HuggingFace repo to pull, which diffusers
pipeline class loads it, and the sampler defaults people usually run it with.
The loader (`imagegen_core.load_pipeline`) resolves `pipeline` by name off the
`diffusers` module and falls back to `AutoPipelineForText2Image` if that class
name is missing (e.g. an older diffusers that predates a brand-new model), so a
wrong/renamed class here degrades to "best effort" instead of crashing.

Families, for the stream's "diffusion vs transformer" framing:
  - "U-Net LDM"  : classic latent diffusion with a U-Net denoiser (SDXL).
  - "MM-DiT"     : multimodal diffusion transformer (SD3.5, Qwen-Image).
  - "DiT / flow" : rectified-flow diffusion transformer (FLUX, Z-Image).
All are "diffusion" in the sampling sense; the backbone is what differs. True
autoregressive image models exist too, but the strongest open weights today are
the DiT/flow transformers below, so that's what we pit against SDXL.

Reality checks (verify on first real run, models move fast):
  - `gated=True` repos need an HF_TOKEN that has accepted the model's license on
    its HF page, or the download 401s.
  - The exact `pipeline` class names for the newest models (FLUX.2, Z-Image,
    Qwen-Image) depend on your installed diffusers version. If one 404s on the
    class lookup, the loader tries AutoPipeline; if that also fails, bump
    diffusers (see config.DIFFUSERS_SPEC) or drop the model from DEFAULT_MODELS.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    key: str                    # short handle used on the CLI and in reports
    repo: str                   # HuggingFace repo id
    pipeline: str               # diffusers pipeline class name (resolved dynamically)
    family: str                 # backbone family, for the report
    license: str
    gated: bool                 # needs an HF_TOKEN that accepted the license
    dtype: str = "bfloat16"     # torch dtype name; Blackwell (GB10) likes bf16
    steps: int = 28             # default num_inference_steps
    guidance: float | None = 4.0  # default guidance_scale; None = don't pass it
    width: int = 1024
    height: int = 1024
    supports_negative: bool = True  # FLUX-style distilled models ignore/lack this
    max_sequence_length: int | None = None  # T5 prompt cap for FLUX/SD3, if set
    quantized: bool = False     # pre-quantized (e.g. bitsandbytes 4-bit) repo
    notes: str = ""


# ── The models ─────────────────────────────────────────────────────────────────
#
# Ordered roughly fast→heavy. Steps/guidance are sane starting points, not
# gospel; the pipeline and app expose them as knobs.

MODELS: dict[str, ModelSpec] = {
    # Distilled FLUX: 4 steps, no CFG. The go-to "fast + genuinely good" default,
    # and Apache-2.0 so it's commercial-friendly.
    "flux1-schnell": ModelSpec(
        key="flux1-schnell",
        repo="black-forest-labs/FLUX.1-schnell",
        pipeline="FluxPipeline",
        family="DiT / rectified flow",
        license="Apache-2.0",
        # Apache-2.0 weights, but BFL still puts a click-through access gate on
        # the repo: accept the license at the model page once, or downloads 403.
        gated=True,
        steps=4,
        guidance=0.0,           # schnell is guidance-distilled; CFG stays off
        supports_negative=False,
        max_sequence_length=256,
        notes="Distilled 4-step FLUX. Fast, no negative prompt. Repo is access-gated.",
    ),
    # Z-Image-Turbo: Apache-2.0, near-FLUX quality in a handful of steps.
    "zimage-turbo": ModelSpec(
        key="zimage-turbo",
        repo="Tongyi-MAI/Z-Image-Turbo",
        pipeline="ZImagePipeline",   # may need a recent diffusers; falls back to Auto
        family="DiT / distilled flow",
        license="Apache-2.0",
        gated=False,
        steps=8,
        guidance=1.0,
        supports_negative=False,
        notes="Few-step, commercial-friendly. Class name may vary by diffusers version.",
    ),
    # SDXL: the deep-ecosystem workhorse. Widest LoRA/ControlNet support, so it's
    # also what lora_finetune.py fine-tunes.
    "sdxl": ModelSpec(
        key="sdxl",
        repo="stabilityai/stable-diffusion-xl-base-1.0",
        pipeline="StableDiffusionXLPipeline",
        family="U-Net LDM",
        license="OpenRAIL++ (open)",
        gated=False,
        dtype="float16",       # SDXL's reference weights are fp16
        steps=30,
        guidance=6.5,
        notes="Classic latent diffusion; the LoRA/ControlNet ecosystem anchor.",
    ),
    # Qwen-Image: MM-DiT, exceptional at rendering text (Latin and Chinese).
    "qwen-image": ModelSpec(
        key="qwen-image",
        repo="Qwen/Qwen-Image",
        pipeline="QwenImagePipeline",
        family="MM-DiT",
        license="Apache-2.0",
        gated=False,
        steps=30,
        guidance=4.0,
        notes="Strong all-rounder; best-in-class text-in-image rendering.",
    ),
    # Qwen-Image-2512: the Dec 2025 update of Qwen-Image (better human realism,
    # detail, text). Same ~20B MM-DiT architecture that already runs here, so it
    # fits the Spark; Apache-2.0 and ungated. Card recommends 50 steps and a
    # recent diffusers (git build) — if QwenImagePipeline doesn't resolve, the
    # loader falls back to AutoPipeline/DiffusionPipeline.
    "qwen-image-2512": ModelSpec(
        key="qwen-image-2512",
        repo="Qwen/Qwen-Image-2512",
        pipeline="QwenImagePipeline",
        family="MM-DiT",
        license="Apache-2.0",
        gated=False,
        steps=50,
        guidance=4.0,
        supports_negative=True,   # Qwen uses true_cfg_scale when a negative is given
        notes="Dec 2025 Qwen-Image update. ~50GB pull; 50 steps so slower. May want git diffusers.",
    ),
    # SD 3.5 Large: Stability's MM-DiT flagship. Gated.
    "sd35-large": ModelSpec(
        key="sd35-large",
        repo="stabilityai/stable-diffusion-3.5-large",
        pipeline="StableDiffusion3Pipeline",
        family="MM-DiT",
        license="Stability Community (gated)",
        gated=True,
        steps=28,
        guidance=4.5,
        max_sequence_length=512,
        notes="MM-DiT with triple text encoders. Accept the license on HF first.",
    ),
    # FLUX.1 [dev]: the non-distilled, higher-fidelity FLUX. Gated, non-commercial.
    "flux1-dev": ModelSpec(
        key="flux1-dev",
        repo="black-forest-labs/FLUX.1-dev",
        pipeline="FluxPipeline",
        family="DiT / rectified flow",
        license="FLUX.1-dev non-commercial (gated)",
        gated=True,
        steps=28,
        guidance=3.5,
        supports_negative=False,
        max_sequence_length=512,
        notes="Full FLUX.1. Slower + better than schnell.",
    ),
    # FLUX.2 [dev]: current open-weight quality benchmark, but a beast. 32B
    # transformer + a separate ~24B Mistral-3 text encoder, so in bf16 it's ~110GB
    # of weights: it will very likely OOM the GB10's 128GB unified pool (BFL's own
    # example loads it with text_encoder=None + a remote encoder to dodge this).
    # For the Spark the realistic option is the 4-bit repo
    # `diffusers/FLUX.2-dev-bnb-4bit` (would need bitsandbytes in the image). Kept
    # out of DEFAULT_MODELS. Class name Flux2Pipeline verified on the HF model card.
    "flux2-dev": ModelSpec(
        key="flux2-dev",
        repo="black-forest-labs/FLUX.2-dev",
        pipeline="Flux2Pipeline",
        family="DiT (next-gen)",
        license="FLUX.2-dev non-commercial (gated)",
        gated=True,
        steps=28,
        guidance=4.0,
        supports_negative=False,
        max_sequence_length=512,
        notes="32B + 24B text encoder; full bf16 likely OOMs the Spark. bnb-4bit variant fits better.",
    ),
    # FLUX.2 [dev], 4-bit: the version that actually fits the Spark. Same 32B+24B
    # model bitsandbytes-quantized to 4-bit -> ~34GB of weights (vs ~110GB bf16),
    # so it loads into the 128GB unified pool with room to spare. Diffusers-format
    # and the mirror repo is ungated. Needs bitsandbytes in the image (added to
    # config.DIFFUSERS_SPEC) and the quantized load path in imagegen_core. UNTESTED
    # on-GPU so far; kept out of DEFAULT_MODELS until a real run confirms it.
    "flux2-dev-4bit": ModelSpec(
        key="flux2-dev-4bit",
        repo="diffusers/FLUX.2-dev-bnb-4bit",
        pipeline="Flux2Pipeline",
        family="DiT (next-gen, 4-bit)",
        license="FLUX.2-dev non-commercial",
        gated=False,             # the 4-bit mirror repo is ungated
        dtype="bfloat16",
        steps=28,
        guidance=4.0,
        supports_negative=False,
        max_sequence_length=512,
        quantized=True,
        notes="4-bit bnb FLUX.2 (~34GB); fits the Spark. Needs bitsandbytes + Flux2Pipeline.",
    ),
    # Sana-Sprint 1.6B: NVIDIA's efficiency play, and the star turn on a Spark:
    # NVIDIA's own model on NVIDIA silicon. Linear-attention DiT + a 32x deep-
    # compression autoencoder + timestep distillation give genuinely good images
    # in ~2 steps, so it's dramatically faster than everything else in the grid.
    "sana-sprint": ModelSpec(
        key="sana-sprint",
        repo="Efficient-Large-Model/Sana_Sprint_1.6B_1024px_diffusers",
        pipeline="SanaSprintPipeline",   # recent diffusers; falls back to Auto
        family="Linear-DiT (few-step)",
        license="NVIDIA open (non-commercial)",
        gated=False,             # freely downloadable, no click-through gate
        steps=2,                 # sprint is distilled to 1–4 steps
        guidance=None,           # timestep-distilled: no CFG, don't pass it
        supports_negative=False,
        notes="NVIDIA's efficient few-step model. The fast one; great on the Spark.",
    ),
    # Chroma: the fully-open (Apache-2.0), un-gated de-distill of FLUX.1-schnell.
    # The direct answer to "you don't need the access-gated FLUX repo." Restores
    # CFG + negative prompts that schnell dropped. ~8.9B; fits the Spark easily.
    # NOTE: the original `lodestones/Chroma` repo is DEPRECATED and ships the whole
    # 1.3TB training-epoch archive (v11..v50, each 17.8GB) — never point at it. The
    # Chroma1-* repos below are proper diffusers-format single releases (~45-63GB).
    # (They also ship a root single-file .safetensors that duplicates transformer/;
    # from_pretrained ignores it, but the fetch still pulls it until we add
    # `Chroma1-*.safetensors` to ignore_patterns.)
    "chroma-hd": ModelSpec(
        key="chroma-hd",
        repo="lodestones/Chroma1-HD",
        pipeline="ChromaPipeline",   # diffusers-format; falls back to Auto/DiffusionPipeline
        family="DiT / rectified flow (FLUX-based)",
        license="Apache-2.0",
        gated=False,
        steps=30,
        guidance=4.0,
        supports_negative=True,
        max_sequence_length=512,
        notes="Flagship high-detail Chroma. Ungated open FLUX-schnell de-distill.",
    ),
    "chroma-base": ModelSpec(
        key="chroma-base",
        repo="lodestones/Chroma1-Base",
        pipeline="ChromaPipeline",
        family="DiT / rectified flow (FLUX-based)",
        license="Apache-2.0",
        gated=False,
        steps=30,
        guidance=4.0,
        supports_negative=True,
        max_sequence_length=512,
        notes="Base Chroma release. Same arch as HD, less detail-tuned.",
    ),
    "chroma-flash": ModelSpec(
        key="chroma-flash",
        repo="lodestones/Chroma1-Flash",
        pipeline="ChromaPipeline",
        family="DiT / rectified flow (FLUX-based, accelerated)",
        license="Apache-2.0",
        gated=False,
        steps=12,                # few-step accelerated variant; tune ~8–16
        guidance=1.0,            # distilled: low/no CFG
        supports_negative=False,
        max_sequence_length=512,
        notes="Speed-distilled Chroma (HD + flash delta). Few steps; empty card, tune settings.",
    ),
    # CogView4-6B: Zhipu's Apache MM-DiT with a GLM-4 text encoder. Another strong
    # text-in-image renderer, so it makes a clean head-to-head against Qwen-Image,
    # both fully open. 6B at 50 steps → not fast; lower --steps for a quick look.
    "cogview4": ModelSpec(
        key="cogview4",
        repo="THUDM/CogView4-6B",
        pipeline="CogView4Pipeline",
        family="MM-DiT (GLM text encoder)",
        license="Apache-2.0",
        gated=False,
        steps=50,
        guidance=3.5,
        supports_negative=True,
        notes="Bilingual, strong text rendering. Pit against qwen-image.",
    ),
    # Lumina-Image 2.0: a compact (2B), current Apache DiT. A good "small but
    # modern" datapoint sitting between tiny Sana and the big MM-DiTs.
    "lumina2": ModelSpec(
        key="lumina2",
        repo="Alpha-VLLM/Lumina-Image-2.0",
        pipeline="Lumina2Pipeline",   # a.k.a. Lumina2Text2ImgPipeline in some versions
        family="DiT / flow (2B)",
        license="Apache-2.0",
        gated=False,
        steps=35,
        guidance=4.0,
        supports_negative=True,
        max_sequence_length=256,
        notes="Compact modern open DiT. Class name may vary; falls back to Auto.",
    ),
    # Krea-2-Turbo: Krea's 12B DiT, post-trained for few-step inference (8 steps,
    # no CFG). Ungated, Krea 2 Community License, ~62GB diffusers-format. We add
    # ONLY Turbo: the sibling Krea-2-Raw is a base checkpoint the card says is
    # "not recommended for inference" (finetuning base only). Krea2Pipeline is very
    # new (Jun 2026) so it may need a git diffusers; the loader falls back to Auto.
    "krea-2-turbo": ModelSpec(
        key="krea-2-turbo",
        repo="krea/Krea-2-Turbo",
        pipeline="Krea2Pipeline",
        family="DiT (12B, turbo)",
        license="Krea 2 Community (open)",
        gated=False,
        steps=8,
        guidance=0.0,            # turbo: guidance-distilled, CFG off
        supports_negative=False,
        notes="Krea's fast 12B model (8 steps). Needs recent diffusers for Krea2Pipeline.",
    ),
}


# The default comparison set: all open (non-gated), no license clicks needed, and
# a spread across the families (fast FLOW, few-step, U-Net, MM-DiT text). Override
# on the CLI with --models '["sdxl","flux1-dev"]' or in the app's multiselect.
DEFAULT_MODELS: list[str] = ["flux1-schnell", "sdxl", "qwen-image"]


def get_spec(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError:
        raise ValueError(
            f"Unknown model {key!r}. Known: {', '.join(MODELS)}"
        ) from None


def resolve_models(keys: list[str] | None) -> list[ModelSpec]:
    """Map a list of keys (or None → DEFAULT_MODELS) to ModelSpecs, in order."""
    return [get_spec(k) for k in (keys or DEFAULT_MODELS)]
