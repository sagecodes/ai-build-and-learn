# World Models with V-JEPA 2: Latent-Space Prediction & Planning

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event looks at world models from a different angle with V-JEPA 2, Meta's self-supervised video model. Where Cosmos generates the future in pixels, JEPA (Joint Embedding Predictive Architecture) predicts in representation space: it learns by predicting the abstract embeddings of masked video rather than reconstructing every pixel. That is the core thesis behind this line of work, Yann LeCun's argument that predicting representations is more efficient and better for reasoning and planning than generating pixels.

Why it makes a good example: it is genuinely different from the generative models we have covered, it is small enough to run and experiment with locally, and it is already integrated into Hugging Face Transformers, so we can get hands-on quickly.

What we could build:

- **Core:** load the frozen V-JEPA 2 encoder, turn short video clips into embeddings, and train a lightweight probe for a task like action recognition or clip retrieval. It shows how strong the self-supervised features are with no generation involved.
- **Stretch:** V-JEPA 2-AC (action-conditioned), a world model post-trained on a small amount of robot data that plans manipulation actions zero-shot in latent space. This ties straight back to the RL and Isaac Sim events.

Hardware: the encoders are ViT-scale (roughly 300M to 1B parameters), so they run comfortably on a DGX Spark (128 GB unified memory) or a single regular GPU.

Some things to look up to get started:

**Model:**
- V-JEPA 2 (Meta): https://github.com/facebookresearch/vjepa2
- Hugging Face Transformers docs and checkpoints: https://huggingface.co/docs/transformers/model_doc/vjepa2
- Paper: "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning": https://huggingface.co/papers/2506.09985

**Background:**
- Meta AI overview: https://ai.meta.com/research/vjepa/
