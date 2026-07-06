# Reinforcement Learning in MuJoCo

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event is about training reinforcement learning agents in MuJoCo, the open-source physics engine widely used for robotics and continuous control. We'll set up simulated environments, train policies to control them, and watch the agents actually learn to move.

MuJoCo (Multi-Joint dynamics with Contact) simulates rigid-body physics fast enough to train on, which is why it's a standard RL benchmark. We'll use it through the Gymnasium environments and a training library, tackle classic control tasks (like teaching a simulated robot to walk), and talk through the practical side: reward design, algorithm choice, and how long training actually takes.

Some things to look up to get started:

**Tooling:**
- MuJoCo: https://github.com/google-deepmind/mujoco
- Gymnasium MuJoCo environments: https://gymnasium.farama.org/environments/mujoco/
- Stable-Baselines3 (RL algorithms): https://github.com/DLR-RM/stable-baselines3
- MuJoCo Playground / MJX (GPU-accelerated, JAX): https://github.com/google-deepmind/mujoco_playground
- DeepMind Control Suite (dm_control): https://github.com/google-deepmind/dm_control
