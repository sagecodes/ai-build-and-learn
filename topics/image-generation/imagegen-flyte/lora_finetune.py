"""Fine-tune SDXL with a LoRA on a handful of images, then compare base vs tuned.

The classic DreamBooth-LoRA story: teach the model a new *subject* from ~5 photos
tied to a rare token ("sks dog"), producing a tiny (~a few MB) LoRA adapter you
can load onto stock SDXL. The report shows the same prompt rendered by the base
model (which has never seen "sks dog") next to the LoRA-tuned model.

    train_lora ─> LoRA dir ─> sample_lora ─> base-vs-tuned report

We keep the loop compact and readable rather than feature-complete (no prior-
preservation, EMA, or SNR weighting — see diffusers' train_dreambooth_lora_sdxl.py
for the full production script). It's meant to be read on stream.

Usage (on the devbox):
    # end-to-end: train on the 5-image dog set, then render before/after
    flyte run lora_finetune.py lora_demo --max_steps 400

    # just train, keep the adapter
    flyte run lora_finetune.py train_lora --max_steps 400 --rank 8

Data: `diffusers/dog-example` (5 CC0 photos) by default. Point --dataset_repo at
any HF image dataset (or swap in your own images) to teach a different subject.
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from config import gpu_task_env, orch_env
from imagegen_core import GenResult, pil_to_data_uri, render_before_after

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = gpu_task_env

BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"
# fp16-fixed VAE: SDXL's stock VAE produces NaNs in fp16/bf16, so we load this
# one for both training (encode) and sampling (decode).
VAE_REPO = "madebyollin/sdxl-vae-fp16-fix"
INSTANCE_PROMPT = "a photo of sks dog"
RESOLUTION = 1024


@dataclass
class LoRAResult:
    lora: flyte.io.Dir
    instance_prompt: str
    steps: int
    rank: int


# ──────────────────────────────────────────────────────────────────────────────
# Task — train the LoRA
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def train_lora(
    dataset_repo: str = "diffusers/dog-example",
    instance_prompt: str = INSTANCE_PROMPT,
    max_steps: int = 400,
    rank: int = 8,
    lr: float = 1e-4,
    seed: int = 0,
) -> LoRAResult:
    """Train a DreamBooth-style LoRA on the images in `dataset_repo`."""
    import random

    import torch
    import torch.nn.functional as F
    from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
    from huggingface_hub import snapshot_download
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from PIL import Image
    from torchvision import transforms
    from transformers import (
        CLIPTextModel,
        CLIPTextModelWithProjection,
        CLIPTokenizer,
    )
    from diffusers import StableDiffusionXLPipeline

    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(seed)
    random.seed(seed)

    # ── Load + freeze the base components ─────────────────────────────────────
    log.info("Loading SDXL components…")
    tok1 = CLIPTokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer")
    tok2 = CLIPTokenizer.from_pretrained(BASE_REPO, subfolder="tokenizer_2")
    te1 = CLIPTextModel.from_pretrained(BASE_REPO, subfolder="text_encoder", torch_dtype=dtype).to(device)
    te2 = CLIPTextModelWithProjection.from_pretrained(BASE_REPO, subfolder="text_encoder_2", torch_dtype=dtype).to(device)
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=dtype).to(device)
    unet = UNet2DConditionModel.from_pretrained(BASE_REPO, subfolder="unet", torch_dtype=dtype).to(device)
    noise_scheduler = DDPMScheduler.from_pretrained(BASE_REPO, subfolder="scheduler")

    for m in (te1, te2, vae, unet):
        m.requires_grad_(False)

    # ── Attach LoRA to the U-Net attention projections ────────────────────────
    lora_cfg = LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    unet.add_adapter(lora_cfg)
    lora_params = [p for p in unet.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in lora_params)
    log.info(f"LoRA rank={rank}: {n_trainable/1e6:.2f}M trainable params")

    # ── Encode the (single, shared) instance prompt once ──────────────────────
    # DreamBooth uses one prompt for every image, so we cache its embeddings.
    def encode_prompt(prompt: str):
        embeds_list, pooled = [], None
        for tok, te in ((tok1, te1), (tok2, te2)):
            ids = tok(prompt, padding="max_length", max_length=tok.model_max_length,
                      truncation=True, return_tensors="pt").input_ids.to(device)
            out = te(ids, output_hidden_states=True)
            pooled = out[0]                       # text_encoder_2's projected pooled embed
            embeds_list.append(out.hidden_states[-2])
        prompt_embeds = torch.cat(embeds_list, dim=-1)   # [1, 77, 2048]
        return prompt_embeds, pooled

    with torch.no_grad():
        prompt_embeds, pooled_embeds = encode_prompt(instance_prompt)
    add_time_ids = torch.tensor(
        [[RESOLUTION, RESOLUTION, 0, 0, RESOLUTION, RESOLUTION]],
        device=device, dtype=dtype,
    )

    # ── Load + preprocess the instance images ─────────────────────────────────
    data_dir = Path(snapshot_download(repo_id=dataset_repo, repo_type="dataset"))
    paths = [p for p in data_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    if not paths:
        raise RuntimeError(f"No images found in dataset {dataset_repo}")
    log.info(f"Training on {len(paths)} images from {dataset_repo}")

    tf = transforms.Compose([
        transforms.Resize(RESOLUTION, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(RESOLUTION),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),           # → [-1, 1]
    ])
    pixel_batches = [tf(Image.open(p).convert("RGB")).unsqueeze(0).to(device, dtype) for p in paths]

    # ── Optimize ──────────────────────────────────────────────────────────────
    opt = torch.optim.AdamW(lora_params, lr=lr)
    unet.train()
    vae_scale = vae.config.scaling_factor

    await flyte.report.replace.aio(
        f"<h2>Training SDXL LoRA</h2><p>subject: <b>{instance_prompt}</b> · "
        f"{len(paths)} images · rank {rank} · {max_steps} steps</p>"
    )
    await flyte.report.flush.aio()

    running = 0.0
    for step in range(max_steps):
        pixels = random.choice(pixel_batches)
        with torch.no_grad():
            latents = vae.encode(pixels).latent_dist.sample() * vae_scale
        noise = torch.randn_like(latents)
        t = torch.randint(0, noise_scheduler.config.num_train_timesteps, (1,), device=device).long()
        noisy = noise_scheduler.add_noise(latents, noise, t)

        model_pred = unet(
            noisy, t, prompt_embeds,
            added_cond_kwargs={"text_embeds": pooled_embeds, "time_ids": add_time_ids},
        ).sample
        loss = F.mse_loss(model_pred.float(), noise.float())
        loss.backward()
        opt.step()
        opt.zero_grad()

        running += loss.item()
        if (step + 1) % 25 == 0 or step == 0:
            avg = running / (25 if (step + 1) % 25 == 0 else 1)
            log.info(f"  step {step + 1}/{max_steps}  loss={avg:.4f}")
            await flyte.report.replace.aio(
                f"<h2>Training SDXL LoRA</h2>"
                f"<p>subject: <b>{instance_prompt}</b> · rank {rank}</p>"
                f"<p>step <b>{step + 1}/{max_steps}</b> · loss {avg:.4f}</p>"
            )
            await flyte.report.flush.aio()
            running = 0.0

    # ── Save the adapter in diffusers LoRA format ─────────────────────────────
    # convert_state_dict_to_diffusers rewrites peft's lora_A/lora_B keys into the
    # diffusers layout that save_lora_weights (and later load_lora_weights) want.
    from diffusers.utils import convert_state_dict_to_diffusers

    out_dir = Path(tempfile.mkdtemp(prefix="sdxl_lora_"))
    unet_lora_state = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionXLPipeline.save_lora_weights(str(out_dir), unet_lora_layers=unet_lora_state)
    log.info(f"Saved LoRA to {out_dir}")

    await flyte.report.replace.aio(
        f"<h2>LoRA trained ✅</h2><p>subject: <b>{instance_prompt}</b> · "
        f"rank {rank} · {max_steps} steps · {n_trainable/1e6:.2f}M params</p>"
        f"<p>Now sample it against base SDXL with <code>sample_lora</code>.</p>"
    )
    await flyte.report.flush.aio()

    lora = await flyte.io.Dir.from_local(str(out_dir))
    return LoRAResult(lora=lora, instance_prompt=instance_prompt, steps=max_steps, rank=rank)


# ──────────────────────────────────────────────────────────────────────────────
# Task — sample base vs LoRA-tuned, side by side
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def sample_lora(
    result: LoRAResult,
    prompt: str = "a photo of sks dog wearing a red superhero cape, cinematic",
    steps: int = 30,
    guidance: float = 6.5,
    seed: int = 1234,
) -> flyte.io.Dir:
    """Render the same prompt with base SDXL and with the LoRA loaded."""
    import torch
    from diffusers import AutoencoderKL, StableDiffusionXLPipeline

    device = "cuda"
    vae = AutoencoderKL.from_pretrained(VAE_REPO, torch_dtype=torch.bfloat16)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE_REPO, vae=vae, torch_dtype=torch.bfloat16,
    ).to(device)

    def gen(p):
        g = torch.Generator(device=device).manual_seed(seed)
        return pipe(prompt=p, num_inference_steps=steps, guidance_scale=guidance,
                    generator=g, width=RESOLUTION, height=RESOLUTION).images[0]

    import time
    out_dir = Path(tempfile.mkdtemp(prefix="lora_samples_"))

    log.info("Sampling base SDXL (no LoRA)…")
    t0 = time.time()
    base_img = gen(prompt)
    base_secs = time.time() - t0
    base_img.save(out_dir / "base.png")

    log.info("Loading LoRA + sampling…")
    pipe.load_lora_weights(await result.lora.download())
    t0 = time.time()
    lora_img = gen(prompt)
    lora_secs = time.time() - t0
    lora_img.save(out_dir / "lora.png")

    before = GenResult("base", prompt, base_secs, data_uri=pil_to_data_uri(base_img, max_side=768))
    after = GenResult("lora", prompt, lora_secs, data_uri=pil_to_data_uri(lora_img, max_side=768))
    await flyte.report.replace.aio(render_before_after(
        prompt, before, after,
        meta=f"SDXL · trained {result.steps} steps · rank {result.rank} · "
             f"subject '{result.instance_prompt}'",
    ))
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

@orch_env.task(report=True)
async def lora_demo(
    dataset_repo: str = "diffusers/dog-example",
    instance_prompt: str = INSTANCE_PROMPT,
    eval_prompt: str = "a photo of sks dog wearing a red superhero cape, cinematic",
    max_steps: int = 400,
    rank: int = 8,
    lr: float = 1e-4,
) -> flyte.io.Dir:
    """Train a LoRA then render the base-vs-tuned comparison in one run."""
    await flyte.report.replace.aio("<h2>LoRA demo</h2><p>Step 1/2 — training…</p>")
    await flyte.report.flush.aio()
    result = await train_lora(dataset_repo, instance_prompt, max_steps, rank, lr)

    await flyte.report.replace.aio("<h2>LoRA demo</h2><p>Step 2/2 — sampling base vs tuned…</p>")
    await flyte.report.flush.aio()
    return await sample_lora(result, eval_prompt)


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(lora_demo)
    print(f"LoRA demo run: {run.name}")
    print(f"  {run.url}")
