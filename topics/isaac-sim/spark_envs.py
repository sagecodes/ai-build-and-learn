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
    Spark-Leap-G1-v0           Spark-Leap-G1-Play-v0
    ... x 10 robots

The first three are the stock env cfg with `scene.terrain.terrain_generator` swapped and
nothing else touched. That is the point: the rewards, observations, action scaling and PPO
hyperparameters stay NVIDIA's tuned values, so when a robot does better on stairs than on
stepping stones, the terrain is the only thing that changed.

`Leap` is the exception, and it is the interesting one. Swapping the terrain is not enough
to teach a jump, because the stock reward set actively forbids one; `leap` therefore also
carries a REWARD PROFILE, a named set of edits applied on top of the robot's own tuning.
See REWARD_PROFILES below for what changes and, more usefully, why each edit is there.
Note what that costs: a `Leap` result is a fact about a terrain AND a reward set together,
so it is not comparable with the other three the way they are comparable with each other.

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


# ── Reward profiles ─────────────────────────────────────────────────────────────
#
# Everything above this line keeps NVIDIA's rewards untouched, and the docstring at the
# top of this file explains why: if the terrain is the only variable, then a robot doing
# better on stairs than on stepping stones is a fact about stairs. That property is worth
# a lot and it is not free to give up.
#
# `leap` gives it up on purpose, because the first parkour run ran into the wall it was
# always going to hit. Three hours of Go2 on the 13-sub-terrain course, reward climbing
# the whole way, and the robot walked to the lip of every trench and stopped. It was not
# undertrained. `Isaac-Velocity-Rough-*` asks for one thing, "track a commanded planar
# velocity", and the reward set that comes with it contains this line
# (velocity_env_cfg.py:295):
#
#     lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
#
# which is a squared penalty on vertical base velocity and the largest-weighted penalty
# in the set. Clearing a 0.45 m trench at 1 m/s needs roughly 0.45 s of flight, so a
# takeoff near 2.7 m/s, so an instantaneous -2.0 * 2.7^2 = -14.6 against Go2 task rewards
# that cap at 1.5 + 0.75 = 2.25. PPO solved the problem it was given. The answer to that
# problem is: never leave the ground.
#
# So a profile is a named, auditable set of edits to a stock reward set, applied AFTER the
# robot's own __post_init__ so it lands on top of the per-robot tuning rather than under
# it. Registered by terrain name, so `leap` gets one and the other three get None and stay
# byte-for-byte what NVIDIA ships.
#
# ── What the leap profile is actually betting on ────────────────────────────────
# Not that the air-time bonus teaches jumping. It is far too small for that. The bet is
# that VELOCITY TRACKING ALREADY PAYS FOR THE JUMP and always did: a robot stopped at the
# lip of a trench earns ~0 on a 1.5-weighted term for the remaining fifteen seconds of its
# episode, which dwarfs anything else in the set. Crossing is worth enormous reward. The
# reason it never learned is that the first few exploratory hops cost -14.6 each and were
# extinguished long before any of them landed on the far side.
#
# These edits do not add an incentive. They clear the path to one that was there all
# along.
def fell_into_hole(env, threshold: float = -0.4, asset_cfg=None):
    """Terminate when the robot has dropped below the patch it spawned on.

    `MeshGapTerrainCfg` hardcodes `terrain_height = 1.0` (mesh_terrains.py:584), so a
    missed gap is a ONE METRE FALL, and the stock termination set does not catch it:
    `base_contact` only fires when the base itself takes a contact force, and a robot
    that drops in feet-first and stays upright at the bottom of a trench never triggers
    it. It just stands down there for the remaining fifteen seconds of its episode,
    earning nothing on a 1.5-weighted velocity-tracking term and paying impact and
    torque penalties on the way in.

    Measured cost of not having this, on the first 4000-iteration leap run: mean reward
    spiking to -58 against a typical +10, and `Curriculum/terrain_levels` collapsing from
    4.16 to 1.46 and climbing back three times over the run. That is not a policy learning
    slowly, it is a policy being knocked over by a heavy-tailed reward and relearning.

    ── Why this and not `mdp.root_height_below_minimum` ────────────────────────────
    Isaac Lab ships that one and it is wrong here, as its own docstring says: it compares
    against a WORLD-frame z, which only means anything on flat ground. Our patches sit at
    twelve different heights across the curriculum grid.

    `env.scene.env_origins` is the origin of the patch each env is currently assigned to,
    and the terrain curriculum rewrites it on every promotion or demotion
    (terrain_importer.py:329), so it is always the surface the robot spawned on. Height
    relative to that is the terrain-aware version of the same question.

    `threshold` is absolute (metres relative to the patch surface) and the caller computes
    it, because the right value is ROBOT-RELATIVE and getting that wrong fails silently in
    the direction that looks fine. The trench is a fixed 1.0 m deep for everyone, so a Go2
    standing at the bottom has its base at -0.60 and a G1, which stands 0.74 m tall, has
    its base at -0.26. A threshold of -0.4 catches the Go2 and NEVER FIRES FOR THE G1: the
    humanoid would stand in the trench for the rest of its episode and the termination
    would report 0.0000 forever, looking exactly like "no robot ever fell in".

    `_leap_profile` sets it to `nominal_standing_height - 0.6`, i.e. "the base has dropped
    more than 0.6 m below where it stands". That is more than any feature on this terrain
    (the rails go up, not down) and less than the 1.0 m trench, for every robot in the zoo:

        Go2   0.40 -> -0.20   in-trench -0.60   fires
        G1    0.74 -> +0.14   in-trench -0.26   fires
        H1    1.05 -> +0.45   in-trench +0.05   fires

    It also fires on the way DOWN rather than at the bottom: a Go2 crosses -0.20 while its
    body is only 0.2 m into the trench, well before impact.

    Note this fires DURING the fall rather than on impact, which is the point: the episode
    ends before the landing transient is ever added to the return.
    """
    import torch  # noqa: F401  - imported here to keep this module's top level light

    asset = env.scene["robot" if asset_cfg is None else asset_cfg.name]
    root_z = asset.data.root_pos_w.torch[:, 2]
    return (root_z - env.scene.env_origins[:, 2]) < threshold


def _leap_profile(cfg, play: bool) -> None:
    """Let the robot leave the ground, and give the curriculum a rung it can reach."""

    # ── 1. The one that matters ────────────────────────────────────────────────
    #
    # -2.0 -> -0.05, not -> 0.0. Zero is tempting and it is a trap: with no cost at all on
    # vertical motion and a positive air-time term, bouncing on the spot is free reward,
    # and a pogo-stick policy on flat ground is a real attractor. A whisper of penalty is
    # enough to make bouncing pointless while leaving a deliberate leap affordable: the
    # same 2.7 m/s takeoff now costs -0.36 instead of -14.6.
    #
    # H1 sets this term to None outright (h1/rough_env_cfg.py:28) and Digit rebuilds it,
    # so this is guarded rather than assumed. `leap` registers for all ten robots even
    # though the demo is a quadruped.
    #
    # Written as "relax, never tighten". G1 already sets this term to 0.0 in its own
    # config and H1 deletes it outright, so a blind assignment would ADD a penalty to the
    # bipeds, which is the exact opposite of what this profile is for. The bipeds also do
    # not need the anti-pogo whisper: `feet_air_time_positive_biped` only pays during
    # SINGLE STANCE, so hopping on both feet earns nothing and is not an attractor there.
    z = getattr(cfg.rewards, "lin_vel_z_l2", None)
    if z is not None and z.weight < -0.05:
        z.weight = -0.05

    # ── 2. Pay for hang time, but only above a walking stride ──────────────────
    #
    # `feet_air_time` is (last_air_time - threshold) summed over feet, paid ONLY on the
    # step where a foot touches down (rewards.py:43). The threshold is therefore the line
    # between "this was a stride" and "this was a flight", and it has to sit just above a
    # normal trot or the term punishes ordinary walking: Go2 trots with ~0.25 s of air per
    # foot, so the stock 0.5 s means every single footfall is scored negative.
    #
    # 0.25 s makes a trot break even and a 0.55 s leap pay. The weight goes to 0.5 from the
    # 0.01 the quadruped configs ship, which sounds drastic and is not: at 0.01 the term
    # was numerically switched off.
    #
    # The bipeds (G1, H1, Cassie, Digit) use `feet_air_time_positive_biped` instead, where
    # `threshold` is a CLAMP on the reward rather than an offset subtracted from it.
    # Raising it there means something else entirely, so that branch only touches weight.
    air = getattr(cfg.rewards, "feet_air_time", None)
    if air is not None:
        if getattr(air.func, "__name__", "") == "feet_air_time":
            air.params["threshold"] = 0.25
            air.weight = 0.5
        else:
            air.params["threshold"] = 0.5
            air.weight = 1.0

    # ── 3. Stop charging for the explosion ─────────────────────────────────────
    #
    # A leap is, mechanically, the largest joint acceleration and the fastest action change
    # a locomotion policy ever produces. Both are penalised. Neither penalty is anywhere
    # near -2.0 and neither would block the behaviour on its own, but they are a headwind
    # pointed at exactly the thing we are trying to learn, so both are halved. Not removed:
    # they are what keeps the policy smooth enough to be worth deploying.
    for term, factor in (("dof_acc_l2", 0.5), ("action_rate_l2", 0.5)):
        if (t := getattr(cfg.rewards, term, None)) is not None:
            t.weight *= factor

    # ── 4. Give it a run-up ────────────────────────────────────────────────────
    #
    # The stock forward command tops out at 1.0 m/s. Horizontal speed is half of what a
    # gap crossing is made of, and at 1.0 m/s a 0.45 m trench needs 0.45 s of flight; at
    # 2.0 m/s it needs 0.22 s, which is a bound rather than a stunt. The lateral range
    # comes IN, from +/-1.0 to +/-0.5, because a sideways command on a ring-shaped trench
    # is a crossing attempted from a standstill and mostly teaches the robot to fall in.
    ranges = cfg.commands.base_velocity.ranges
    ranges.lin_vel_x = (-1.0, 2.0)
    ranges.lin_vel_y = (-0.5, 0.5)

    # ── 5. Scale the ladder to the robot ───────────────────────────────────────
    #
    # `SPARK_LEAP_CFG` is tuned for a Go2: 0.05 to 0.38 m of gap, with the interesting
    # rows sitting just past its ~0.30 m step-over reach. Handing that same ladder to a G1
    # is not a harder task, it is a trivial one. A humanoid with a 0.74 m base height
    # strides 0.38 m without noticing, so it would top out the curriculum on the first
    # afternoon, measure zero flight time the whole way, and produce a chart that looks
    # like a triumph and means nothing.
    #
    # Base height is the proxy, read off the robot's own config rather than tabulated here
    # so a new entry in BASE_TASKS needs no bookkeeping. It is a rough proxy (a quadruped
    # crosses with its front-to-rear foot span, a biped with its step length) but both
    # scale with leg length and so does this number.
    #
    # Capped at 2.0 because H1 and Digit stand at 1.05 m, and 2.6x would ask a biped for a
    # metre-wide gap, which is past the point where the reward set can help.
    nominal = 0.40  # the Go2, which the base ladder is tuned for
    height = nominal
    robot = getattr(cfg.scene, "robot", None)
    init = getattr(robot, "init_state", None)
    if init is not None and getattr(init, "pos", None):
        height = float(init.pos[2])
    scale = min(height / nominal, 2.0)

    if scale != 1.0:
        gen = cfg.scene.terrain.terrain_generator
        # Already deep-copied per task by _derive_env_cfg, so this cannot leak into
        # another robot's config. That is trap #1 in this file's header.
        for name, attr in (("gaps", "gap_width_range"), ("rails", "rail_height_range")):
            sub = gen.sub_terrains.get(name)
            if sub is not None:
                lo, hi = getattr(sub, attr)
                setattr(sub, attr, (round(lo * scale, 3), round(hi * scale, 3)))

    # ── 6. End an episode that has fallen in ───────────────────────────────────
    #
    # The most valuable line in this profile after the z-velocity weight, and it is not a
    # reward change at all. Without it a failed crossing is fifteen seconds of dead
    # episode and a landing transient; with it the attempt simply ends. See
    # `fell_into_hole` above for the measurements that made this necessary.
    #
    # Falling is still expensive, because ending early forfeits every remaining second of
    # tracking reward. It is just no longer catastrophic, and that difference is what a
    # policy needs in order to be willing to try a crossing at all.
    #
    # Added rather than replacing anything, so `base_contact` and `time_out` still apply.
    from isaaclab.managers import TerminationTermCfg

    cfg.terminations.fell_into_hole = TerminationTermCfg(
        # Robot-relative, and it has to be: the trench is 1.0 m deep for every robot but
        # they do not all stand at the same height, so a fixed threshold that catches a
        # Go2 never fires for a G1. See the docstring on fell_into_hole.
        func=fell_into_hole, params={"threshold": round(height - 0.6, 3)}
    )

    # ── 7. Start at the bottom of the ladder ───────────────────────────────────
    #
    # `max_init_terrain_level=5` (velocity_env_cfg.py:94) spawns every env on row 5 of the
    # grid and lets the curriculum sort it out. On the stock rough course that is fine,
    # because row 5 is walkable. Here row 5 is a 0.23 m trench, which a policy on iteration
    # 1 cannot cross, so all 4096 envs spend their first few hundred iterations being
    # demoted one row per episode (~42 iterations each) just to reach ground they can learn
    # on. Row 1 is a 3.6 cm crack: something to trip over, which is the point. It is the
    # only rung an untrained policy clears by accident, and that accident is the whole
    # bootstrap.
    #
    # Not applied to the Play variants: those are for filming, record.py places the robot
    # on an explicit row, and pinning the initial level would fight it.
    if not play:
        cfg.scene.terrain.max_init_terrain_level = 1


# Terrain name -> the reward edits that terrain needs, or absent for "none, keep NVIDIA's".
REWARD_PROFILES = {
    "leap": _leap_profile,
}


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
    profile = REWARD_PROFILES.get(terrain_name)

    def __post_init__(self):  # noqa: N807
        super(cls, self).__post_init__()
        self.scene.terrain.terrain_generator = copy.deepcopy(generator)
        # After the terrain swap and after the robot's own post-init, so the profile is
        # editing the FINAL weights rather than base-class ones the robot then overwrites.
        # Go2 is the case that makes this concrete: it sets feet_air_time.weight = 0.01 in
        # its own post-init, so a profile that ran earlier would be silently reverted.
        if profile is not None:
            profile(self, play)
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
