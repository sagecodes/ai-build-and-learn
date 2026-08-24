"""Register our own tasks into NVIDIA's training scripts, without forking them.

Isaac Lab's `scripts/reinforcement_learning/rsl_rl/{train,play}.py` are the entry points
everyone uses, and they resolve `--task` out of the gymnasium registry. The obvious way
to add a task is to fork those scripts, or to vendor a copy of `isaaclab_tasks`. Both age
badly: the scripts are 250 lines of checkpoint handling, hydra plumbing and video wrapping
that we would then own forever.

There is a supported hook instead, and it is barely documented:

    --external_callback spark_envs.register

`train.py:85` defines it, and it is resolved with `string_to_callable(name, separator=".")`,
so the value is `module.attribute` and NOT the `module:attribute` spelling used everywhere
else in Isaac Lab. It is called after `import isaaclab_tasks` and before the hydra
decorator reads the registry, which is exactly the window in which new tasks have to
appear. It may return a list of argv tokens it consumed; returning None means "I consumed
nothing", which is what we want.

The only requirement is that `spark_envs` is importable from the child process, so
whatever runs `train.py` has to put this directory on PYTHONPATH.

── What gets registered ────────────────────────────────────────────────────────
For every robot with a stock rough-terrain task, and every terrain in `terrains.py`:

    Spark-Parkour-G1-v0        Spark-Parkour-G1-Play-v0
    Spark-Stairs-G1-v0         Spark-Stairs-G1-Play-v0
    Spark-Stones-G1-v0         Spark-Stones-G1-Play-v0
    ... x 10 robots

Each one is the stock env cfg with `scene.terrain.terrain_generator` swapped and nothing
else touched. That is the point: the rewards, observations, action scaling and PPO
hyperparameters stay NVIDIA's tuned values, so when a robot does better on stairs than on
stepping stones, the terrain is the only thing that changed.

── Two traps, both found the hard way ──────────────────────────────────────────
1. **Deep-copy the terrain config per task.** `TerrainGeneratorCfg` objects are ordinary
   mutable dataclass instances held as module-level singletons. Isaac Lab's own `_PLAY`
   configs do `self.scene.terrain.terrain_generator.num_rows = 5` in `__post_init__`,
   which mutates the shared object for everyone in the process. Sharing one config object
   across our six variants of a robot would mean instantiating the Play task silently
   shrinks the training task.

2. **Give every task its own `experiment_name`.** rsl_rl derives its log directory from
   it, and `get_checkpoint_path` walks that directory to find the newest checkpoint. Reuse
   the stock `g1_rough` name and `play.py` will happily load a checkpoint trained on a
   completely different terrain and report it as a success.
"""

from __future__ import annotations

import copy
import importlib

import gymnasium as gym
from isaaclab.utils.configclass import configclass

from terrains import TERRAINS

# Robot key -> the stock rough-terrain task we derive from.
#
# Spot is missing on purpose. `Isaac-Velocity-Flat-Spot-v0` is the only registered Spot
# task: its config subclasses the flat env, has no height scanner, and carries a bespoke
# Spot-specific reward set. Deriving a rough Spot means adding the ray-caster back and
# re-tuning rewards, which is a change to the ROBOT, not to the terrain, and would break
# the "terrain is the only variable" property the zoo comparison depends on.
BASE_TASKS: dict[str, str] = {
    "g1": "Isaac-Velocity-Rough-G1-v0",
    "h1": "Isaac-Velocity-Rough-H1-v0",
    "go2": "Isaac-Velocity-Rough-Unitree-Go2-v0",
    "go1": "Isaac-Velocity-Rough-Unitree-Go1-v0",
    "a1": "Isaac-Velocity-Rough-Unitree-A1-v0",
    "anymal_b": "Isaac-Velocity-Rough-Anymal-B-v0",
    "anymal_c": "Isaac-Velocity-Rough-Anymal-C-v0",
    "anymal_d": "Isaac-Velocity-Rough-Anymal-D-v0",
    "cassie": "Isaac-Velocity-Rough-Cassie-v0",
    "digit": "Isaac-Velocity-Rough-Digit-v0",
}

# Display names, used for task ids and for report headings.
ROBOT_LABELS: dict[str, str] = {
    "g1": "G1",
    "h1": "H1",
    "go2": "Go2",
    "go1": "Go1",
    "a1": "A1",
    "anymal_b": "Anymal-B",
    "anymal_c": "Anymal-C",
    "anymal_d": "Anymal-D",
    "cassie": "Cassie",
    "digit": "Digit",
}

# Chase camera for the replay clips. Metres, relative to the robot base once
# origin_type is "asset_root". Behind, beside and slightly above, looking at hip height:
# close enough to read the gait, wide enough to see the next two footholds.
CHASE_EYE = (2.6, 2.6, 1.5)
CHASE_LOOKAT = (0.0, 0.0, 0.4)
CHASE_RESOLUTION = (1280, 720)

# Populated by register(). Kept so callers can ask what exists without re-deriving.
REGISTERED: list[str] = []


def _load(entry_point: str):
    """Resolve a gym `module:Attribute` entry point to the object itself."""
    mod_name, attr = entry_point.split(":")
    return getattr(importlib.import_module(mod_name), attr)


def _derive_env_cfg(base_cls, robot: str, terrain_name: str, play: bool) -> type:
    """Subclass a stock env cfg and swap in our terrain generator.

    The swap happens AFTER `super().__post_init__()` on purpose. The parent's post-init is
    where Isaac Lab decides things that depend on the terrain, and where the `_PLAY`
    variants shrink whatever generator they find. Replacing first would just get
    overwritten; replacing last means we own the final state.
    """
    generator = copy.deepcopy(TERRAINS[terrain_name])

    def __post_init__(self):  # noqa: N807
        super(cls, self).__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(generator)
        if play:
            # NOT the shrink the stock _PLAY configs do (num_rows=5, num_cols=5,
            # curriculum=False), and that is deliberate. Those three lines are what make
            # a replay cheap, and they also destroy the only two coordinates worth
            # filming by:
            #
            #   curriculum=False  switches the generator from "row index IS difficulty"
            #                     to `difficulty = uniform(*difficulty_range)` per patch
            #                     (terrain_generator.py:227). Row 9 stops meaning
            #                     "hardest" and starts meaning nothing at all.
            #   num_cols=5        re-spreads the sub-terrains over five columns instead
            #                     of twenty, so a column index no longer picks out the
            #                     same sub-terrain it picks out during training.
            #
            # Keeping the training grid means `record.py --terrain_level 9
            # --terrain_col 12` puts the robot on the hardest row of the same stepping
            # stones it trained on, and the report can say so honestly. The cost is
            # generating 200 terrain patches for a one-robot replay, which is seconds.

            # ── The viewport chase camera ────────────────────────────────────────────
            #
            # record.py does not use this: it drives its own Camera sensors, because the
            # Kit viewport capture renders everything except the robot on this box. These
            # lines stay for anyone running NVIDIA's `play.py` by hand, where the default
            # viewer is a FIXED camera at world (7.5, 7.5, 7.5) looking at the origin. On
            # a flat plane that happens to frame the robot, which is why nobody notices;
            # on generated terrain the robot spawns metres away and walks off.
            #
            # origin_type="asset_root" re-anchors eye and lookat to the robot's base every
            # frame, so this is a follow cam, not a repositioned static one.
            #
            # The default viewer is a FIXED camera at world (7.5, 7.5, 7.5) looking at the
            # world origin. On a flat plane that happens to frame the robot, which is why
            # nobody notices. On generated terrain the robot spawns on some patch metres
            # away and walks off, so the clip is a lovely wide shot of the terrain with a
            # 12-pixel robot in it. Measured on the first parkour render: a Go2 occupying
            # about 0.1% of a 1280x720 frame.
            #
            # origin_type="asset_root" re-anchors eye and lookat to the robot's base every
            # frame, so this is a follow cam, not a repositioned static one.
            self.viewer.origin_type = "asset_root"
            self.viewer.asset_name = "robot"
            self.viewer.eye = CHASE_EYE
            self.viewer.lookat = CHASE_LOOKAT
            self.viewer.resolution = CHASE_RESOLUTION

    # The robot key is in the name even though the base class already carries it. Two
    # robots CAN share a base config class, and a name collision here would silently
    # point one task's entry point at the other task's class.
    name = f"Spark_{terrain_name}_{robot}_env" + ("_play" if play else "")
    cls = configclass(type(name, (base_cls,), {"__post_init__": __post_init__, "__module__": __name__}))
    globals()[name] = cls  # so `spark_envs:Name` resolves as a gym entry point string
    return cls


def _derive_agent_cfg(base_cls, robot: str, terrain_name: str) -> type:
    """Subclass the stock rsl_rl runner cfg only to give it a private log directory."""
    experiment = f"spark_{terrain_name}_{robot}"

    def __post_init__(self):  # noqa: N807
        post = getattr(super(cls, self), "__post_init__", None)
        if post is not None:
            post()
        self.experiment_name = experiment

    name = f"Spark_{terrain_name}_{robot}_agent"
    cls = configclass(type(name, (base_cls,), {"__post_init__": __post_init__, "__module__": __name__}))
    globals()[name] = cls
    return cls


def register() -> None:
    """The `--external_callback` entry point. Idempotent, so a re-import is harmless."""
    import isaaclab_tasks  # noqa: F401  - populates the registry we derive from

    for robot, base_id in BASE_TASKS.items():
        if base_id not in gym.registry:
            # A robot can vanish between Isaac Lab releases. Skipping loudly beats an
            # AttributeError twenty minutes into a zoo run.
            print(f"[spark_envs] base task missing, skipping {robot}: {base_id}")
            continue

        base_kwargs = gym.spec(base_id).kwargs
        base_env_cls = _load(base_kwargs["env_cfg_entry_point"])
        base_play_cls = _load(gym.spec(base_id.replace("-v0", "-Play-v0")).kwargs["env_cfg_entry_point"])
        base_agent_cls = _load(base_kwargs["rsl_rl_cfg_entry_point"])

        for terrain_name in TERRAINS:
            label = f"Spark-{terrain_name.title()}-{ROBOT_LABELS[robot]}"
            agent_cls = _derive_agent_cfg(base_agent_cls, robot, terrain_name)

            for task_id, base, play in (
                (f"{label}-v0", base_env_cls, False),
                (f"{label}-Play-v0", base_play_cls, True),
            ):
                if task_id in gym.registry:
                    continue
                env_cls = _derive_env_cfg(base, robot, terrain_name, play)
                gym.register(
                    id=task_id,
                    entry_point="isaaclab.envs:ManagerBasedRLEnv",
                    disable_env_checker=True,
                    kwargs={
                        "env_cfg_entry_point": f"{__name__}:{env_cls.__name__}",
                        "rsl_rl_cfg_entry_point": f"{__name__}:{agent_cls.__name__}",
                    },
                )
                REGISTERED.append(task_id)

    print(f"[spark_envs] registered {len(REGISTERED)} tasks over {len(TERRAINS)} terrains")


if __name__ == "__main__":
    register()
    for task_id in REGISTERED:
        print(task_id)
