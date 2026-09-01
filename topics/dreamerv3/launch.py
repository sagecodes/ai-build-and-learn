"""Entry point for training: register the custom domain, then hand over to upstream.

    python launch.py --configs dmc_vision size12m --task dmc_arena_walk --logdir ...

Every flag is upstream's; this file adds none of its own. It exists only because
`arena.py` has to be imported before `dm_control.suite.load` is called, and upstream's
`dreamerv3/main.py` is a script with no hook for that. Importing it here, in the
process that runs the training loop, is the smallest possible intervention: no fork of
dreamerv3, no patch, no sys.meta_path trickery.

── Why this has to be the process entry point, not an import ───────────────────
embodied's Driver puts each environment in its own subprocess when `run.debug=False`.
Those subprocesses are spawned from this one, so they inherit an interpreter that has
already imported `arena`, and `suite.load('arena', 'walk')` resolves inside them too.
If the registration happened somewhere further up (in a Flyte task, say, which then
spawned this as a child) the env workers would be two forks removed and would fail
with `Domain 'arena' does not exist.`
"""

from __future__ import annotations

import functools

import arena  # noqa: F401  imported for its side effect: registers the domain


def keep_log_observations() -> None:
    """Let arena.py's diagnostics survive the pixel path. Two separate losses.

    **One: the pixel path deletes them.** embodied's DMC wrapper takes
    `proprio=False` to mean "the agent learns from pixels", which is what
    `--configs dmc_vision` sets and what this demo wants. It implements that by cutting
    the observation down to four keys:

        basic = ('is_first', 'is_last', 'is_terminal', 'reward')
        if not self._proprio:
            obs = {k: obs[k] for k in basic}

    which also deletes `x_position`, the metres-travelled diagnostic the report plots
    under episode return to tell walking apart from standing still.

    **Two: the `log/` prefix does not survive on the way in.** `FromDM` flattens every
    observation key with `key.replace('/', '_')`, so a dm_control observation named
    `log/x_position` arrives here as `log_x_position`. embodied's own logger keys its
    per-episode aggregation off `key.startswith('log/')`, so the flattened name is
    never recognised as a diagnostic and never reaches metrics.jsonl.

    Both failures are silent. Nothing raises, nothing warns; the key is simply absent
    and the curve is empty. Found ten minutes into a seven hour run, by checking that
    the metric existed rather than assuming it did.

    So this keeps the flattened names through the pixel filter and restores the prefix
    dm_control asked for. Safe by construction: `log/` keys are not observations in any
    sense that reaches the agent, because embodied strips that whole prefix set one
    layer up (`embodied/jax/agent.py:51` asserts none survive). This changes what gets
    logged, never what the policy or the world model can see.

    Rebinding the module attribute rather than editing dreamerv3: `make_env` resolves
    the class with `getattr(module, 'DMC')` at call time, so replacing it before
    training starts is enough and upstream stays untouched.
    """
    import elements
    import numpy as np
    from embodied.envs import dmc

    basic = ("is_first", "is_last", "is_terminal", "reward")

    def restore(mapping):
        """Put back the `log/` prefix that FromDM flattened to `log_`.

        This runs on BOTH the pixel and the proprioceptive path, and it has to. On the
        proprioceptive path nothing filters the observation at all, so a flattened
        `log_ball_distance` would sail through as an ordinary observation and hand the
        agent the distance to the nearest ball in its state vector. That would quietly
        destroy the property arena.py is built around: the balls are in the physics and
        in the pixels, and not in the state. Restoring the prefix is what gets them
        stripped again one layer up.
        """
        return {
            ("log/" + k[len("log_"):] if k.startswith("log_") else k): v
            for k, v in mapping.items()
        }

    def pixels_only(mapping):
        return {k: v for k, v in mapping.items() if k in basic or k.startswith("log/")}

    class DMCKeepingLogs(dmc.DMC):

        @functools.cached_property
        def obs_space(self):
            spaces = restore(self._env.obs_space.copy())
            if not self._proprio:
                spaces = pixels_only(spaces)
            key = "image" if self._image else "log/image"
            spaces[key] = elements.Space(np.uint8, self._size + (3,))
            return spaces

        def step(self, action):
            for key, space in self.act_space.items():
                if not space.discrete:
                    assert np.isfinite(action[key]).all(), (key, action[key])
            obs = restore(self._env.step(action))
            if not self._proprio:
                obs = pixels_only(obs)
            key = "image" if self._image else "log/image"
            obs[key] = self._dmenv.physics.render(*self._size, camera_id=self._camera)
            for key, space in self.obs_space.items():
                if np.issubdtype(space.dtype, np.floating):
                    assert np.isfinite(obs[key]).all(), (key, obs[key])
            return obs

    dmc.DMC = DMCKeepingLogs


if __name__ == "__main__":
    # Imported here rather than at module scope so that `import launch` stays cheap.
    # pipeline.py imports this module at top level purely so Flyte bundles the file
    # into the pod, and that import must not drag in jax on the CPU-only orchestrator.
    from dreamerv3.main import main

    keep_log_observations()
    print(f"registered dm_control domain 'arena' ({len(arena.SUITE)} tasks)")
    main()
