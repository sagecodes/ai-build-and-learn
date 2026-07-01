# Open-Source Image Generation: Diffusion Models & LoRA

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event is about generating images with AI. There's no single model we're locked into, the point is to explore what's out there and actually try a few. We'll focus on open-source models, but you're welcome to bring commercial ones (DALL·E, Midjourney, and friends) if you want to compare.

We'll try to look at both major families: **diffusion-based** models (the dominant approach today) and **transformer-based / autoregressive** image models. I'll research and try some of the best open-source options ahead of the stream, and if time allows we'll also try **fine-tuning with LoRA** to teach a model a new style or subject.

Some things to look up to get started:

**Open-source models:**
- FLUX.2 (Black Forest Labs): current open-weight quality benchmark; DiT backbone, up to 4MP, multi-image reference
- Stable Diffusion 3.5 / SDXL (Stability AI): deepest ecosystem — LoRA, ControlNet, inpainting, countless fine-tunes
- Qwen-Image (Alibaba): strong all-rounder, especially good at rendering text (including Chinese)
- Z-Image / Z-Image-Turbo: Apache-2.0, near FLUX.2 quality in just a few steps (fast and commercial-friendly)
- HunyuanImage (Tencent): another high-quality open-weight option

**Tooling:**
- Hugging Face Diffusers: https://github.com/huggingface/diffusers
- ComfyUI: node-based workflows: https://github.com/comfyanonymous/ComfyUI
- LoRA fine-tuning for diffusion models
