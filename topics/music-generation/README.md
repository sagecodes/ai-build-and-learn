# Open-Source Music Generation: Text-to-Music & Lyrics-to-Song

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event is about generating music and audio with AI. As with the image and video events, there's no single model we're locked into — the point is to explore what's out there and actually try a few. We'll focus on open-source models, but you're welcome to bring commercial ones (Suno, Udio, and friends) if you want to compare — worth noting there isn't a fully open-source Suno equivalent yet, though the gap is closing.

We'll look at the two main flavors: **text-to-music** (instrumental / sound design from a prompt) and **lyrics-to-song** (full tracks with vocals and accompaniment). Under the hood these lean on the same diffusion and transformer/language-model approaches as image and video, applied to audio. I'll research and try some of the best open-source options ahead of the stream, and we'll talk through the practical tradeoffs: quality, track length, controllability, speed, and licensing.

Some things to look up to get started:

**Open-source models:**
- YuE (YuE AI): lyrics-to-song — full tracks up to ~5 min with synchronized vocals and accompaniment
- ACE-Step: fast and controllable — a ~4-min song in seconds; diffusion + linear-transformer design
- MusicGen (Meta / AudioCraft): versatile text-to-music with melody conditioning (note: CC BY-NC — non-commercial output license)
- Stable Audio Open (Stability AI): great for ambient/textural audio, SFX, and samples (short clips, not full songs)

**Tooling:**
- AudioCraft (Meta) — MusicGen / AudioGen: https://github.com/facebookresearch/audiocraft
- Hugging Face — audio models and pipelines: https://huggingface.co/models?pipeline_tag=text-to-audio
