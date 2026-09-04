"""Retrain a policy on a NEW task while the world model stays frozen.

    python transfer.py --configs dmc_vision size12m --task dmc_arena_walk \
        --logdir ~/logdir/transfer --run.from_checkpoint ~/logdir/back/ckpt/<stamp>

This is the experiment the rest of the repo has been setting up. Everything else here
measures how efficiently DreamerV3 learns a task from nothing. This measures what is
left over afterwards: whether the model of the world it built is worth anything once
the goal changes.

── The claim being tested ──────────────────────────────────────────────────────
`walk` and `back` are the same body in the same arena under the same physics, and
their rewards differ by exactly one sign (see arena.py). So an agent trained on one has
learned two separable things:

    the world     how a walker's limbs move, how it falls, how the posts stream past,
                  where the balls go when kicked. NONE of this depends on the goal.
    the goal      which way to go, and what that is worth.

A world model should make the first transferable and leave only the second to relearn.
If that is true, an agent that keeps the world model and rediscovers only the goal
should reach a given return in far fewer environment steps than one starting from
nothing. If it is false, the world model was never more than a feature extractor and
this repo should say so.

The control is the from-scratch curve for the target task, which already exists: the
2026-08-31 flagship run reached 987 on `dmc_arena_walk` in 500k steps.

── What is frozen, and what has to be relearned ────────────────────────────────
Frozen: `enc`, `dyn`, `dec`, `con`. The encoder, the RSSM, the decoder, and the
continuation head. Termination does not change when the reward flips, so `con` transfers
untouched and freezing it keeps the claim clean.

Trained: `rew`, `pol`, `val`. The reward head has to be refit because the reward
genuinely changed; the actor and critic have to be relearned because the old ones are
not merely useless, they are precisely wrong: an expert `back` policy scores the
standing-floor 0.167 under `walk`'s reward. Starting them from the transferred
checkpoint would be starting them in the worst place on the map, so they start fresh.

── How the freeze works ────────────────────────────────────────────────────────
`Agent.__init__` builds one optimiser over every module, and `Optimizer.__call__` takes
gradients with `nj.grad(lossfn, self.modules)`. So the modules list IS the set of
trainable parameters, and passing a subset freezes the rest. They still run forward,
which is not waste: their losses are still computed, so `train/loss/image` during a
transfer run is a live readout of how well the frozen world model explains a task it
was never trained on.

── The upstream feature that was wired up but unreachable ──────────────────────
`embodied/run/train.py` already loads a partial checkpoint:

    if args.from_checkpoint:
      elements.checkpoint.load(args.from_checkpoint, dict(
          agent=bind(agent.load, regex=args.from_checkpoint_regex)))

and `agent.load(data, regex=...)` restores only matching params, skipping the
shape-equality assert that would otherwise reject a checkpoint whose optimiser state no
longer matches. But `from_checkpoint_regex` is in none of the config files, so reading
it raised before it could ever be used. `patches/0002-from-checkpoint-regex.patch` adds
the missing key; the loading logic is upstream's own and is not touched.
"""

from __future__ import annotations

import arena  # noqa: F401  imported for its side effect: registers the domain
import launch

# Restored from the checkpoint AND frozen. Everything not matched here is trained from
# scratch, which is `rew`, `pol` and `val`.
# Checkpoint parameter names are module-rooted with no agent prefix: `enc/...`,
# `dyn/...`, and so on alongside `opt/`, `pol`, `val`, `slowval` and the normalisers.
# Excluding `opt/` is deliberate as well as necessary: the saved optimiser state
# describes moments for modules this run does not train.
FROZEN = ("enc", "dyn", "dec", "con")
FROZEN_REGEX = rf"^({'|'.join(FROZEN)})/"


def freeze_world_model() -> None:
    """Rebind the agent class so its optimiser covers only the trainable modules.

    Rebinding the module attribute rather than editing dreamerv3, for the same reason
    launch.py rebinds `dmc.DMC`: `make_agent` resolves the class with a function-local
    `from .agent import Agent` at call time, so replacing it beforehand is enough.
    """
    import embodied
    from dreamerv3 import agent as dv3agent

    base = dv3agent.Agent

    class FrozenWorldModelAgent(base):

        def __init__(self, obs_space, act_space, config):
            super().__init__(obs_space, act_space, config)
            frozen = [m for m in self.modules if m.name in FROZEN]
            trained = [m for m in self.modules if m.name not in FROZEN]
            assert frozen, f"froze nothing; module names are {[m.name for m in self.modules]}"
            assert trained, "left nothing to train"
            print(f"frozen:  {sorted(m.name for m in frozen)}")
            print(f"trained: {sorted(m.name for m in trained)}")
            # Same optimiser hyperparameters, a smaller parameter set. `train/opt/
            # param_count` in metrics.jsonl will therefore report the TRAINABLE count,
            # which is the number this run should be quoted with.
            self.opt = embodied.jax.Optimizer(
                trained, self._make_opt(**config.opt), summary_depth=1, name="opt")

    dv3agent.Agent = FrozenWorldModelAgent


if __name__ == "__main__":
    from dreamerv3.main import main

    launch.keep_log_observations()
    freeze_world_model()
    print(f"registered dm_control domain 'arena' ({len(arena.SUITE)} tasks)")
    print(f"restoring and freezing: {FROZEN_REGEX}")
    main()
