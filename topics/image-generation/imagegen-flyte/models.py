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
    # FLUX.2 [dev]: current open-weight quality benchmark. Newest → highest risk
    # on the pipeline class name; kept out of DEFAULT_MODELS until verified.
    "flux2-dev": ModelSpec(
        key="flux2-dev",
        repo="black-forest-labs/FLUX.2-dev",
        pipeline="Flux2Pipeline",    # verify: needs a diffusers that ships it
        family="DiT (next-gen)",
        license="FLUX.2-dev non-commercial (gated)",
        gated=True,
        steps=28,
        guidance=4.0,
        supports_negative=False,
        max_sequence_length=512,
        notes="Newest BFL open weights. Confirm Flux2Pipeline exists in your diffusers.",
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
