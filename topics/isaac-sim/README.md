# Reinforcement Learning in NVIDIA Isaac Sim: Physical AI & Sim-to-Real

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event picks up where the MuJoCo event left off and steps up to NVIDIA Isaac Sim, a GPU-accelerated, photorealistic robotics simulator built on Omniverse. The goal here is "physical AI": training policies in simulation that can transfer to real robots.

Compared to MuJoCo, Isaac Sim is heavier and more advanced. It runs on NVIDIA RTX GPUs, renders photorealistic scenes with ray tracing, and through Isaac Lab runs thousands of environments in parallel on the GPU for fast, large-scale RL. We'll train a robot policy with Isaac Lab, look at sim-to-real transfer, and talk about when this level of fidelity is worth it versus a lighter simulator like MuJoCo.

Heads up: this one needs an NVIDIA RTX GPU (local or cloud). If you don't have the hardware, it works well as a watch-and-learn-the-workflow session.

Some things to look up to get started:

**Tooling:**
- NVIDIA Isaac Sim (Omniverse robotics simulator): https://github.com/isaac-sim/IsaacSim
- Isaac Lab (GPU-parallel RL framework on Isaac Sim): https://github.com/isaac-sim/IsaacLab
- Isaac Lab docs and getting started: https://developer.nvidia.com/isaac/lab
- RL libraries it integrates: RSL-RL, SKRL, RL-Games
- Physics engines: PhysX 5 and the newer Newton engine
