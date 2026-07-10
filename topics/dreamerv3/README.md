# World Models with DreamerV3

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event kicks off a run on world models: models that learn an internal representation of how an environment works, then use it to predict what happens next and to plan. It's a natural next step after the generation and RL events, tying both threads together.

We're starting with DreamerV3, a model-based RL agent and a great on-ramp to the idea. Dreamer learns a compact world model of its environment from experience, then trains its policy almost entirely inside imagined rollouts of that model rather than the real environment. It's lightweight (trains on a single GPU), works across many tasks with the same settings, and connects directly to the RL and MuJoCo work.

It runs comfortably on modest hardware, including a DGX Spark (128 GB unified memory), so it's easy to train and experiment with live.

Some things to look up to get started:

**Model:**
- DreamerV3 (Danijar Hafner): https://github.com/danijar/dreamerv3
- Paper: "Mastering Diverse Domains through World Models"

**Background:**
- The original "World Models" paper (Ha and Schmidhuber): https://worldmodels.github.io/
