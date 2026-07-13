# Open-Source Video Generation: Diffusion & Transformer Models

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event is about generating video with AI, picking up where the image generation event left off. As with images, there's no single model we're committed to — we'll explore several and see how far the open-source options have come. Open source is the focus, but commercial models (Sora, Runway, and others) are fair game if you want to compare.

Most current video models build on the same **diffusion** foundations as image generation, extended across time and increasingly using **transformer/DiT** backbones. We'll try a few open-source text-to-video and image-to-video models and talk through the practical tradeoffs: quality, clip length, speed, and hardware requirements.

Some things to look up to get started:

**Open-source models:**
- Wan 2.2 (Alibaba Tongyi): versatile MoE model: text-to-video, image-to-video, and editing in one
- HunyuanVideo (Tencent): strong cinematic / photorealistic quality
- LTX-Video / LTX-2 (Lightricks): the fast one; ~5s clip in under a minute on a single GPU
- CogVideoX (Zhipu / THUDM): best at following detailed, multi-part prompts
- Mochi 1 (Genmo): flow-matching model known for fluid, coherent motion
- Stable Video Diffusion (Stability AI): earlier image-to-video, still widely used
- Krea text to video

**Tooling:**
- Hugging Face Diffusers: video pipelines: https://github.com/huggingface/diffusers
- ComfyUI: node-based workflows (popular for video too): https://github.com/comfyanonymous/ComfyUI

## The demo

`videogen-flyte/` is what we build on the stream: run a prompt across several open
video models on the DGX Spark and get one side-by-side report **with clips that
play right in the browser**, plus an image-to-video path that generates its own
first frame. See [videogen-flyte/README.md](videogen-flyte/README.md).
