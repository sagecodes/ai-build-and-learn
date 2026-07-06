# World Models with NVIDIA Cosmos: Physical AI

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event goes bigger on world models with NVIDIA Cosmos, an open family of world foundation models built for physical AI. Where DreamerV3 learns a small world model for a single agent, Cosmos is a large generative model that simulates the physical world itself: predicting future video, running action-conditioned rollouts, and generating synthetic data to train robots and autonomous machines. It ties directly back to the Isaac Sim event.

Cosmos 3 (released 2026) exposes two surfaces: a Reasoner for understanding and planning, and a Generator for world simulation and future prediction. The weights are open (OpenMDW license) and available on Hugging Face.

Good news for local work: the smaller Cosmos models run on a single DGX Spark (128 GB unified memory) at roughly 30 GB for inference in early testing, so we can demo it live rather than only in the cloud.

Some things to look up to get started:

**Model:**
- NVIDIA Cosmos (open platform of world models): https://github.com/nvidia/cosmos
- Cosmos overview: https://www.nvidia.com/en-us/ai/cosmos/
- Cosmos Cookbook (runnable recipes): https://nvidia-cosmos.github.io/cosmos-cookbook/

**Related:**
- V-JEPA 2 (Meta) as an open, prediction-focused world model to compare
