"""Fine-tune Chroma with a LoRA on a small style dataset, then compare base vs tuned.

The companion to `lora_finetune.py`. Same idea (a few MB of adapter weights teach
a frozen model something new), different backbone, and the differences are the
whole point of reading both:

                      lora_finetune.py (SDXL)      lora_chroma.py (Chroma)
    backbone          U-Net latent diffusion       DiT, rectified flow
    text encoder      two CLIPs (+ pooled embed)   one T5-XXL, no pooled embed
    training target   the noise (epsilon)          the velocity, `noise - latents`
    timesteps         uniform over 1000            logit-normal, weighted to the middle
    latent layout     a 4-channel image grid       16 channels, packed into 2x2 patches

    fetch_weights -> train_chroma_lora -> sample_chroma_lora -> base-vs-tuned report

Why Chroma and not Z-Image: Chroma is an ungated Apache-2.0 de-distill of
FLUX.1-schnell, and critically it is *not* guidance-distilled, so a plain
flow-matching loss trains it the way the textbook says. Z-Image-Turbo is
DMD-distilled; fine-tuning it with this loss walks the weights off the distilled
manifold and degrades its few-step sampling, which is why the trainers that do
support it (ai-toolkit, SimpleTuner) carry a Turbo-specific "assistant adapter".
That's a great topic, and a bad first demo.

Usage (on the devbox):
    # end-to-end: train the yarn-art style LoRA, then render before/after
    flyte run lora_chroma.py chroma_lora_demo

    # a different dataset + longer training
    flyte run lora_chroma.py chroma_lora_demo --dataset tarot --max_steps 1200

    # train only, keep the adapter
    flyte run lora_chroma.py train_only --dataset 3d-icon --rank 16

Datasets live in `lora_data.py`. Chroma variants live in `models.py`; train on
`chroma-hd` or `chroma-base`, never on `chroma-flash` (speed-distilled, same trap
as Z-Image-Turbo).

One measured lesson from the first real run: the size of the before/after tracks
how RARE the trigger is, not how striking the style is. Base Chroma already
renders "yarn art style" well, so that adapter mostly transfers the training
photos' dark backdrop. The rare-token sets (`trtcrd`, `3dicon`) show a far
starker contrast because the base model has no prior for the token at all.
"""

from __future__ import annotations

import logging
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import flyte
import flyte.io
import flyte.report

from compare_pipeline import fetch_weights
from config import cpu_task_env, lora_env, orch_env
from imagegen_core import (
    GenResult,
    free_gpu_memory,
    load_pipeline,
    pil_to_data_uri,
    render_lora_report,
    render_training_report,
    timed_generate,
)
from lora_data import (
    DEFAULT_DATASET,
    get_dataset,
    load_examples,
    load_prepared,
    preprocess,
    save_examples,
)
from models import get_spec

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
log = logging.getLogger(__name__)

env = lora_env

# The VAE downsamples 8x, and Chroma then packs each 2x2 latent block into one
# token, so a side must be divisible by 16 for both stages to land on integers.
VAE_SCALE_FACTOR = 8
PATCH = 2

# Equal thirds of the sigma range, ordered low noise (nearly the image) to high
# (nearly pure noise). The report breaks loss down over these.
SIGMA_BUCKETS = (
    "σ 0.00-0.33 · near image",
    "σ 0.33-0.67 · midway",
    "σ 0.67-1.00 · near noise",
)


# The three tasks a run walks through, for the report's stage indicator.
STAGES = ("fetch", "train", "sample")


def _fmt_secs(s: float) -> str:
    s = int(max(s, 0))
    return f"{s // 3600}h{(s % 3600) // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def _badges(ds, model_key: str, rank: int, resolution: int) -> list[str]:
    return [model_key, "flow matching", f"rank {rank}", f"{resolution}px",
            f"{ds.n_images} images", ds.license]


def _sigma_rows(bucket_sum: list[float], bucket_n: list[int]):
    return [
        (lbl, bucket_sum[i] / bucket_n[i] if bucket_n[i] else 0.0, bucket_n[i])
        for i, lbl in enumerate(SIGMA_BUCKETS)
    ]


@dataclass
class ChromaLoRAResult:
    lora: flyte.io.Dir
    model_key: str
    dataset: str
    trigger: str
    # Training provenance, for the report. All -1 when the adapter was handed to
    # us as a bare URI (see `generate_with_lora`) and we have no idea how it was
    # trained. Better an honest "unknown" than a confident "rank 0".
    steps: int = -1
    rank: int = -1
    resolution: int = -1

    @property
    def provenance(self) -> str:
        if self.rank < 0:
            return "adapter loaded from a URI (training settings unknown)"
        return f"rank {self.rank} · {self.steps} steps · trained at {self.resolution}px"


# ──────────────────────────────────────────────────────────────────────────────
# Chroma's deviations from FLUX, isolated so they're hard to miss.
# Three in total: no `pooled_projections` and no `guidance` kwarg on the
# transformer (both absent from its forward signature), plus the text mask below.
# ──────────────────────────────────────────────────────────────────────────────

def _encode_prompt(tokenizer, text_encoder, prompt: str, max_len: int, device, dtype):
    """T5 embeddings plus the attention mask Chroma's transformer expects.

    Chroma differs from FLUX twice in this function, and both bite silently:

      1. It feeds the tokenizer's padding mask *into* T5. FLUX does not, and runs
         T5 over the padding.
      2. The mask it then hands the transformer keeps one padding token: the
         comparison is `<=` against the sequence length, not `<`. Get this wrong
         and training still runs, it just quietly learns against a prompt one
         token shorter than the one sampling will use.

    Mirrors `ChromaPipeline._get_t5_prompt_embeds`.
    """
    import torch

    tokens = tokenizer(
        prompt, padding="max_length", max_length=max_len,
        truncation=True, return_tensors="pt",
    )
    ids = tokens.input_ids.to(device)
    tok_mask = tokens.attention_mask.to(device)

    embeds = text_encoder(ids, output_hidden_states=False, attention_mask=tok_mask)[0]
    embeds = embeds.to(dtype=dtype, device=device)

    seq_lengths = tok_mask.sum(dim=1)
    positions = torch.arange(tok_mask.size(1), device=device).unsqueeze(0)
    attn_mask = (positions <= seq_lengths.unsqueeze(1)).to(dtype=dtype, device=device)
    return embeds, attn_mask


def _full_attention_mask(text_mask, image_seq_len: int):
    """Extend the text mask over the image tokens, which are never masked."""
    import torch

    ones = torch.ones(
        text_mask.shape[0], image_seq_len,
        device=text_mask.device, dtype=text_mask.dtype,
    )
    return torch.cat([text_mask, ones], dim=1)


# ──────────────────────────────────────────────────────────────────────────────
# Task: fetch the dataset (CPU, cached)
# ──────────────────────────────────────────────────────────────────────────────

@cpu_task_env.task(cache="auto", retries=2)
async def fetch_dataset(dataset: str = DEFAULT_DATASET, limit: int = -1) -> flyte.io.Dir:
    """Materialize a LoRA dataset into a Dir, once, on a CPU pod.

    Exactly the split `fetch_weights` makes, for the same reason: the training
    task used to call `load_examples` itself, which left the GPU idle while
    HuggingFace served a parquet file. `cache="auto"` keys on (dataset, limit),
    so the download happens once and later runs pull the Dir from the blob store.

    These sets are tiny (18 to 78 images), so this is about keeping the GPU busy
    and making the data a real artifact, not about saving bandwidth.
    """
    ds = get_dataset(dataset)
    examples = load_examples(ds, limit=None if limit < 0 else limit)
    dest = Path(tempfile.mkdtemp(prefix=f"data_{dataset}_")) / "images"
    n = save_examples(examples, dest)
    log.info(f"[{dataset}] wrote {n} images from {ds.repo} to {dest}")
    return await flyte.io.Dir.from_local(str(dest))


# ──────────────────────────────────────────────────────────────────────────────
# Task: train the LoRA
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def train_chroma_lora(
    weights: flyte.io.Dir,
    data: flyte.io.Dir,
    dataset: str = DEFAULT_DATASET,
    model_key: str = "chroma-hd",
    resolution: int = 512,
    max_steps: int = 800,
    rank: int = 16,
    lr: float = 1e-4,
    seed: int = 0,
    max_sequence_length: int = 512,
    flip_augment: bool = True,
    gradient_checkpointing: bool = True,
) -> ChromaLoRAResult:
    """Train a style LoRA on Chroma's transformer with a flow-matching loss.

    `weights` and `data` both arrive pre-fetched from cached CPU tasks, so this
    task does nothing but hold the GPU and train. `dataset` is still passed for
    the trigger phrase and the report; the pixels come from `data`.

    `resolution` defaults to 512 rather than Chroma's native 1024 because the
    token count grows with the square: 1024px is 4096 image tokens per step
    against 1024, so it's roughly 4x the time for a demo whose whole point is
    being watchable. Style transfers across resolution well enough to see. Bump
    to 1024 for an adapter you actually intend to keep.
    """
    import torch
    from diffusers import (
        AutoencoderKL,
        ChromaPipeline,
        ChromaTransformer2DModel,
        FlowMatchEulerDiscreteScheduler,
    )
    from diffusers.training_utils import (
        cast_training_params,
        compute_density_for_timestep_sampling,
        compute_loss_weighting_for_sd3,
    )
    from peft import LoraConfig
    from peft.utils import get_peft_model_state_dict
    from transformers import T5EncoderModel, T5TokenizerFast

    if resolution % (VAE_SCALE_FACTOR * PATCH) != 0:
        raise ValueError(
            f"resolution must be divisible by {VAE_SCALE_FACTOR * PATCH}, got {resolution}"
        )

    device, dtype = "cuda", torch.bfloat16
    torch.manual_seed(seed)
    random.seed(seed)

    ds = get_dataset(dataset)
    base = await weights.download()
    examples = load_prepared(await data.download())
    log.info(f"Training a {ds.kind} LoRA on {ds.repo} ({len(examples)} images)")

    thumbs = "".join(
        f'<img src="{pil_to_data_uri(img, max_side=112)}" '
        f'style="height:96px;border-radius:6px;margin:2px" title="{cap[:80]}"/>'
        for img, cap in examples[:12]
    )
    await flyte.report.replace.aio(
        f"<h2>Training a Chroma LoRA</h2>"
        f"<p><b>{ds.repo}</b> · {len(examples)} images · {ds.kind} LoRA · "
        f"trigger <code>{ds.trigger}</code></p><div>{thumbs}</div>"
        f"<p>Step 1/3: encoding captions…</p>"
    )
    await flyte.report.flush.aio()

    # ── 1. Cache text embeddings, then drop T5 ────────────────────────────────
    # The encoders are only needed once. Freeing each before loading the next
    # keeps peak memory at "one big model", not "all three", which is what lets
    # an 8.9B transformer train next to a 4.7B text encoder on one box.
    tokenizer = T5TokenizerFast.from_pretrained(base, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(
        base, subfolder="text_encoder", torch_dtype=dtype,
    ).to(device)
    text_encoder.requires_grad_(False)

    with torch.no_grad():
        encoded = [
            _encode_prompt(tokenizer, text_encoder, cap, max_sequence_length, device, dtype)
            for _, cap in examples
        ]
    del text_encoder
    free_gpu_memory()

    # ── 2. Cache image latents, then drop the VAE ─────────────────────────────
    await flyte.report.replace.aio(
        f"<h2>Training a Chroma LoRA</h2><p>Step 2/3: encoding {len(examples)} images…</p>"
    )
    await flyte.report.flush.aio()

    vae = AutoencoderKL.from_pretrained(base, subfolder="vae", torch_dtype=dtype).to(device)
    vae.requires_grad_(False)

    # Caching latents means no per-step random crop or flip, so we bake the flip
    # in here instead: encode each image twice and pay for it once.
    from PIL import Image as PILImage

    cached: list[tuple] = []
    with torch.no_grad():
        for (img, _), (embeds, text_mask) in zip(examples, encoded):
            views = [img]
            if flip_augment:
                views.append(img.transpose(PILImage.Transpose.FLIP_LEFT_RIGHT))
            for view in views:
                px = preprocess(view, resolution).unsqueeze(0).to(device, dtype)
                lat = vae.encode(px).latent_dist.sample()
                # FLUX-family VAEs shift *and* scale. SDXL's only scales, so
                # copying the SDXL trainer here would bias every latent.
                lat = (lat - vae.config.shift_factor) * vae.config.scaling_factor
                cached.append((lat, embeds, text_mask))

    del vae
    free_gpu_memory()
    log.info(f"Cached {len(cached)} latents at {resolution}px")

    # ── 3. Load the transformer, attach the LoRA, train ───────────────────────
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(base, subfolder="scheduler")
    transformer = ChromaTransformer2DModel.from_pretrained(
        base, subfolder="transformer", torch_dtype=dtype,
    ).to(device)
    transformer.requires_grad_(False)

    # Attention projections only. The double blocks carry the `add_*` context
    # stream (text tokens); the single blocks don't, and PEFT just skips the
    # names it can't find in them.
    transformer.add_adapter(LoraConfig(
        r=rank, lora_alpha=rank, init_lora_weights="gaussian",
        target_modules=[
            "to_q", "to_k", "to_v", "to_out.0",
            "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out",
        ],
    ))
    if gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    # Base weights stay bf16; the LoRA params train in fp32 for stable updates.
    # The forward then runs under autocast so the mixed dtypes meet as bf16.
    cast_training_params(transformer, dtype=torch.float32)
    lora_params = [p for p in transformer.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in lora_params)
    log.info(f"LoRA rank={rank}: {n_trainable / 1e6:.2f}M trainable params")

    opt = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-4)
    transformer.train()

    n_train_ts = scheduler.config.num_train_timesteps
    latent_side = resolution // VAE_SCALE_FACTOR
    n_channels = transformer.config.in_channels // (PATCH * PATCH)
    img_ids = ChromaPipeline._prepare_latent_image_ids(
        1, latent_side // PATCH, latent_side // PATCH, device, dtype,
    )
    image_seq_len = (latent_side // PATCH) ** 2

    t0 = time.time()
    running, history = 0.0, []
    ema = gnorm_ema = None
    bucket_sum, bucket_n = [0.0] * len(SIGMA_BUCKETS), [0] * len(SIGMA_BUCKETS)
    epoch_history: list[tuple[int, float]] = []   # [(epoch, mean loss over it)]
    epoch_loss_sum = epoch_loss_n = 0

    # Real epochs: walk the whole set in a shuffled order, reshuffle each pass.
    # `random.choice` would sample with replacement, so with 156 latents and 800
    # steps some images get shown a dozen times and others never. One epoch is
    # exactly `epoch_len` steps, which is what lets the report mark boundaries.
    epoch_len = len(cached)
    n_epochs = -(-max_steps // epoch_len)   # ceil
    order = list(range(epoch_len))

    for step in range(max_steps):
        pos = step % epoch_len
        if pos == 0:
            random.shuffle(order)
        epoch = step // epoch_len
        latents, embeds, text_mask = cached[order[pos]]

        noise = torch.randn_like(latents)
        # Sample timesteps logit-normally: flow matching learns least from the
        # ends (pure noise, pure image) and most from the middle of the path.
        u = compute_density_for_timestep_sampling(
            "logit_normal", batch_size=1, logit_mean=0.0, logit_std=1.0, mode_scale=1.29,
        )
        idx = (u * n_train_ts).long().clamp(0, n_train_ts - 1)
        timesteps = scheduler.timesteps[idx].to(device)
        sigmas = scheduler.sigmas[idx].to(device=device, dtype=dtype).view(-1, 1, 1, 1)

        # The straight line from image to noise, sampled at sigma.
        noisy = (1.0 - sigmas) * latents + sigmas * noise
        packed = ChromaPipeline._pack_latents(noisy, 1, n_channels, latent_side, latent_side)
        txt_ids = torch.zeros(embeds.shape[1], 3, device=device, dtype=dtype)

        with torch.autocast("cuda", dtype=dtype):
            pred = transformer(
                hidden_states=packed,
                timestep=timesteps.to(dtype) / 1000,
                encoder_hidden_states=embeds,
                txt_ids=txt_ids,
                img_ids=img_ids,
                attention_mask=_full_attention_mask(text_mask, image_seq_len),
                return_dict=False,
            )[0]

        pred = ChromaPipeline._unpack_latents(pred, resolution, resolution, VAE_SCALE_FACTOR)
        # Flow matching regresses the *velocity* along that line, not the noise.
        # This one line is the real difference from the SDXL trainer.
        target = noise - latents
        weighting = compute_loss_weighting_for_sd3("logit_normal", sigmas)
        loss = (weighting.float() * (pred.float() - target.float()) ** 2).mean()

        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        # ── Metrics ───────────────────────────────────────────────────────────
        lv = loss.item()
        running += lv
        # An EMA, because the raw per-step loss is dominated by which sigma the
        # step happened to draw, not by how training is going.
        ema = lv if ema is None else 0.97 * ema + 0.03 * lv
        gn = float(grad_norm)
        gnorm_ema = gn if gnorm_ema is None else 0.97 * gnorm_ema + 0.03 * gn
        # Bucket the loss by noise level. This is the plot that actually shows
        # learning: high-sigma (near-noise) steps are simply harder, forever.
        b = min(int(float(sigmas.reshape(-1)[0]) * len(SIGMA_BUCKETS)), len(SIGMA_BUCKETS) - 1)
        bucket_sum[b] += lv
        bucket_n[b] += 1
        # Per-epoch mean loss: averaging a whole shuffled pass cancels the sigma
        # noise, so this is the one curve that actually trends down as it learns.
        epoch_loss_sum += lv
        epoch_loss_n += 1
        if pos == epoch_len - 1:
            epoch_history.append((epoch + 1, epoch_loss_sum / epoch_loss_n))
            epoch_loss_sum = epoch_loss_n = 0

        if (step + 1) % 25 == 0 or step == 0:
            span = 25 if (step + 1) % 25 == 0 else 1
            avg = running / span
            history.append((step + 1, avg, ema, gnorm_ema))
            log.info(f"  step {step + 1}/{max_steps}  loss={avg:.4f}  ema={ema:.4f}")

            done = step + 1
            elapsed = time.time() - t0
            sps = done / max(elapsed, 1e-6)
            await flyte.report.replace.aio(render_training_report(
                title="Training a Chroma LoRA",
                meta=f"{ds.repo} · {ds.kind} LoRA · trigger '{ds.trigger}'",
                badges=_badges(ds, model_key, rank, resolution),
                stages=STAGES, stage=1,
                step=done, max_steps=max_steps,
                kpis=[
                    ("step", f"{done}/{max_steps}"),
                    ("epoch", f"{epoch + 1}/{n_epochs}"),
                    ("loss (EMA)", f"{ema:.4f}"),
                    ("grad norm", f"{gnorm_ema:.3f}"),
                    ("steps/sec", f"{sps:.2f}"),
                    ("elapsed", _fmt_secs(elapsed)),
                    ("eta", _fmt_secs((max_steps - done) / max(sps, 1e-6))),
                    ("trainable", f"{n_trainable / 1e6:.1f}M"),
                ],
                history=history,
                epoch_history=epoch_history,
                epoch_len=epoch_len,
                sigma_buckets=_sigma_rows(bucket_sum, bucket_n),
                thumbs_html=thumbs,
                footer=f"{len(cached)} cached latents at {resolution}px · lr {lr:g}",
            ))
            await flyte.report.flush.aio()
            running = 0.0

    # ── 4. Save the adapter ───────────────────────────────────────────────────
    # PEFT's own key layout is what the Flux/Chroma loader reads back, so unlike
    # the SDXL trainer there's no convert_state_dict_to_diffusers hop.
    out_dir = Path(tempfile.mkdtemp(prefix="chroma_lora_"))
    ChromaPipeline.save_lora_weights(
        str(out_dir), transformer_lora_layers=get_peft_model_state_dict(transformer),
    )
    log.info(f"Saved LoRA to {out_dir}")

    del transformer
    free_gpu_memory()

    # Keep the charts on the finished task. A completed run whose report is a
    # one-line "done" is a report you can't learn anything from afterwards.
    elapsed = time.time() - t0
    await flyte.report.replace.aio(render_training_report(
        title="Chroma LoRA trained ✅",
        meta=f"{ds.repo} · {ds.kind} LoRA · trigger '{ds.trigger}'",
        badges=_badges(ds, model_key, rank, resolution),
        stages=STAGES, stage=2,
        step=max_steps, max_steps=max_steps,
        kpis=[
            ("steps", str(max_steps)),
            ("epochs", str(len(epoch_history) or n_epochs)),
            ("final loss (EMA)", f"{ema:.4f}" if ema is not None else "n/a"),
            ("grad norm", f"{gnorm_ema:.3f}" if gnorm_ema is not None else "n/a"),
            ("elapsed", _fmt_secs(elapsed)),
            ("steps/sec", f"{max_steps / max(elapsed, 1e-6):.2f}"),
            ("trainable", f"{n_trainable / 1e6:.1f}M"),
            ("resolution", f"{resolution}px"),
        ],
        history=history,
        epoch_history=epoch_history,
        epoch_len=epoch_len,
        sigma_buckets=_sigma_rows(bucket_sum, bucket_n),
        thumbs_html=thumbs,
        footer=(f"Trigger <code>{ds.trigger}</code>. The adapter is this task's Dir "
                f"output; pass its URI to <code>generate_with_lora</code> (or the "
                f"studio's LoRA tab) to sample it without retraining."),
    ))
    await flyte.report.flush.aio()

    return ChromaLoRAResult(
        lora=await flyte.io.Dir.from_local(str(out_dir)),
        model_key=model_key, dataset=dataset, trigger=ds.trigger,
        steps=max_steps, rank=rank, resolution=resolution,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Task: sample base vs LoRA-tuned
# ──────────────────────────────────────────────────────────────────────────────

@env.task(report=True)
async def sample_chroma_lora(
    result: ChromaLoRAResult,
    weights: flyte.io.Dir,
    prompts: list[str] | None = None,
    lora_scale: float = 1.0,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    show_base: bool = True,
) -> flyte.io.Dir:
    """Render every prompt with the LoRA, and (by default) without it too.

    Same pipeline object, same seed, one `load_lora_weights` call between the two
    passes: any difference you see is the adapter and nothing else. That is the
    whole reason the base pass exists, so keep `show_base=True` when you're
    evaluating a fresh adapter.

    `show_base=False` skips it and halves the work, which is what you want once
    the adapter is known good and you just want pictures out of it. That mode is
    the plain generate path the Gradio studio calls.
    """
    spec = get_spec(result.model_key)
    ds = get_dataset(result.dataset)
    prompts = prompts or list(ds.eval_prompts)

    base = await weights.download()
    pipe = load_pipeline(spec, model_path=base)

    kw = {}
    if steps > 0:
        kw["steps"] = steps
    if guidance >= 0:
        kw["guidance"] = guidance

    out_dir = Path(tempfile.mkdtemp(prefix="chroma_lora_samples_"))

    # The base pass must run BEFORE load_lora_weights: the adapter is fused into
    # the live pipeline, so there's no "base" left to sample afterwards.
    before: list[GenResult | None] = [None] * len(prompts)
    if show_base:
        for i, prompt in enumerate(prompts):
            log.info(f"base [{i + 1}/{len(prompts)}] {prompt}")
            img, secs = timed_generate(pipe, spec, prompt, seed=seed, **kw)
            img.save(out_dir / f"base_{i}.png")
            before[i] = GenResult(spec.key, prompt, secs, pil_to_data_uri(img, max_side=640))

    lora_dir = await result.lora.download()
    log.info(f"Loading LoRA from {lora_dir}")
    # Name the adapter rather than relying on the auto-assigned "default_0", so
    # set_adapters has something stable to scale.
    pipe.load_lora_weights(lora_dir, adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[lora_scale])

    after: list[GenResult] = []
    for i, prompt in enumerate(prompts):
        log.info(f"lora [{i + 1}/{len(prompts)}] {prompt}")
        img, secs = timed_generate(pipe, spec, prompt, seed=seed, **kw)
        img.save(out_dir / f"lora_{i}.png")
        after.append(GenResult(spec.key, prompt, secs, pil_to_data_uri(img, max_side=640)))

    # A prompt with no trigger in it exercises the LoRA's *bleed*, not its
    # effect. Worth saying out loud rather than leaving as a puzzling null result.
    missing = [p for p in prompts if ds.trigger.lower() not in p.lower()]
    warn = (
        f" · ⚠️ {len(missing)} prompt(s) omit the trigger '{ds.trigger}', "
        f"so the LoRA may look inert there"
    ) if missing else ""

    # Report the sampler settings actually used, not the -1 sentinels: `generate`
    # substitutes the model's defaults, and "what did it really run at" is the
    # first question a surprising image raises.
    used_steps = steps if steps > 0 else spec.steps
    used_guidance = guidance if guidance >= 0 else spec.guidance
    sampler = (f"seed {seed} · {used_steps} steps · guidance {used_guidance} · "
               f"{spec.width}x{spec.height} · lora_scale {lora_scale}")

    await flyte.report.replace.aio(render_lora_report(
        list(zip(prompts, before, after)),
        title=(f"Chroma LoRA: {ds.repo}" if show_base
               else f"Chroma LoRA generate: {ds.repo}"),
        meta=(f"{spec.key} · {ds.kind} LoRA · {result.provenance} · "
              f"trigger '{result.trigger}'{warn}"),
        trigger=result.trigger,
        sampler=sampler,
    ))
    await flyte.report.flush.aio()
    return await flyte.io.Dir.from_local(str(out_dir))


# ──────────────────────────────────────────────────────────────────────────────
# Orchestrator
# ──────────────────────────────────────────────────────────────────────────────

@orch_env.task(report=True)
async def chroma_lora_demo(
    dataset: str = DEFAULT_DATASET,
    model_key: str = "chroma-hd",
    resolution: int = 512,
    max_steps: int = 800,
    rank: int = 16,
    lr: float = 1e-4,
    prompts: list[str] | None = None,
    show_base: bool = True,
    limit: int = -1,
) -> flyte.io.Dir:
    """Fetch weights + data, train the LoRA, render base-vs-tuned. One run."""
    ds = get_dataset(dataset)
    if model_key == "chroma-flash":
        raise ValueError(
            "chroma-flash is speed-distilled; a plain flow-matching loss degrades it. "
            "Train on chroma-hd or chroma-base."
        )

    await flyte.report.replace.aio(
        f"<h2>Chroma LoRA demo</h2><p>Step 1/3: fetching {model_key} weights and "
        f"{ds.repo} (both cached after the first run)…</p>"
    )
    await flyte.report.flush.aio()
    # Both are cached CPU tasks, so the GPU pod isn't waiting on a download.
    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)
    d = await fetch_dataset.override(short_name=f"fetch {dataset}")(dataset, limit)

    await flyte.report.replace.aio(
        f"<h2>Chroma LoRA demo</h2><p>Step 2/3: training on {ds.repo}…</p>"
    )
    await flyte.report.flush.aio()
    result = await train_chroma_lora(
        weights=w, data=d, dataset=dataset, model_key=model_key,
        resolution=resolution, max_steps=max_steps, rank=rank, lr=lr,
    )

    await flyte.report.replace.aio(
        "<h2>Chroma LoRA demo</h2><p>Step 3/3: sampling base vs tuned…</p>"
    )
    await flyte.report.flush.aio()
    return await sample_chroma_lora(result, w, prompts, show_base=show_base)


@orch_env.task(report=True)
async def generate_with_lora(
    lora_uri: str,
    prompts: list[str] | None = None,
    dataset: str = DEFAULT_DATASET,
    model_key: str = "chroma-hd",
    lora_scale: float = 1.0,
    steps: int = -1,
    guidance: float = -1.0,
    seed: int = 1234,
    show_base: bool = False,
) -> flyte.io.Dir:
    """Generate from an already-trained LoRA. The entry point the studio calls.

    `lora_uri` is the blob-store URI of a `train_only` / `train_chroma_lora`
    output Dir (an `s3://...` path; the Flyte console shows it on the task's
    Outputs tab). Gradio can only hand a task primitives, so the Dir is rebuilt
    from its URI here rather than passed as an artifact.

    `dataset` is only consulted for the trigger phrase and the default prompts,
    so pass the one the adapter was trained on. `show_base` off by default: this
    is the "give me pictures" path, not the "is my adapter any good" path.
    """
    ds = get_dataset(dataset)
    if not lora_uri.strip():
        raise ValueError(
            "lora_uri is empty. Train one first (`flyte run lora_chroma.py train_only`) "
            "and copy the LoRA Dir URI off the run's Outputs tab."
        )

    await flyte.report.replace.aio(
        f"<h2>Chroma LoRA generate</h2><p>Fetching {model_key} weights…</p>"
    )
    await flyte.report.flush.aio()
    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)

    result = ChromaLoRAResult(
        lora=flyte.io.Dir.from_existing_remote(lora_uri.strip()),
        model_key=model_key, dataset=dataset, trigger=ds.trigger,
    )
    return await sample_chroma_lora(
        result, w, prompts, lora_scale=lora_scale, steps=steps,
        guidance=guidance, seed=seed, show_base=show_base,
    )


@orch_env.task
async def train_only(
    dataset: str = DEFAULT_DATASET,
    model_key: str = "chroma-hd",
    resolution: int = 512,
    max_steps: int = 800,
    rank: int = 16,
    lr: float = 1e-4,
    limit: int = -1,
) -> ChromaLoRAResult:
    """Fetch weights + data and train, skipping the sampling pass.

    `train_chroma_lora` takes already-fetched Dirs, which makes it awkward to
    call from the CLI. This is the entry point for "just give me the adapter";
    the LoRA comes back as a Dir artifact on the run.
    """
    w = await fetch_weights.override(short_name=f"fetch {model_key}")(model_key)
    d = await fetch_dataset.override(short_name=f"fetch {dataset}")(dataset, limit)
    return await train_chroma_lora(
        weights=w, data=d, dataset=dataset, model_key=model_key,
        resolution=resolution, max_steps=max_steps, rank=rank, lr=lr,
    )


if __name__ == "__main__":
    import pathlib

    flyte.init_from_config(root_dir=pathlib.Path(__file__).parent)
    run = flyte.run(chroma_lora_demo)
    print(f"Chroma LoRA demo run: {run.name}")
    print(f"  {run.url}")
