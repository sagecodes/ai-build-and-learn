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


GO2_NOMINAL_HEIGHT = 0.40
"""Base height of the robot every hand-tuned distance in this file is written against."""


def _nominal_height(cfg) -> float:
    """The robot's standing base height, off its own config rather than a table here.

    Read from `init_state.pos[2]` so a new entry in BASE_TASKS needs no bookkeeping, and
    defaulted to the Go2 rather than raising: a robot whose config does not spell this out
    gets Go2-sized numbers, which is wrong but survivable, where a crash twenty minutes
    into a zoo run is not.
    """
    init = getattr(getattr(cfg.scene, "robot", None), "init_state", None)
    if init is not None and getattr(init, "pos", None):
        return float(init.pos[2])
    return GO2_NOMINAL_HEIGHT


def _size_scale(cfg) -> float:
    """How much bigger than a Go2 is this robot? Used to scale every tuned distance.

    A rough proxy (a quadruped crosses a gap with its front-to-rear foot span, a biped
    with its step length) but both scale with leg length and so does this number. Capped
    at 2.0 because H1 and Digit stand at 1.05 m, and 2.6x would ask a biped for a
    metre-wide gap, which is past the point where the reward set can help.
    """
    return min(_nominal_height(cfg) / GO2_NOMINAL_HEIGHT, 2.0)


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
    # Base height is the proxy. See `_size_scale` above for why, and for the cap.
    height = _nominal_height(cfg)
    scale = _size_scale(cfg)

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


# ── Reading the terrain in the reward, not just in the observation ──────────────
#
# The fact `leap` never used: THE ROBOT COULD ALREADY SEE THE GAP.
#
# `Isaac-Velocity-Rough-*` mounts a `RayCasterCfg` on the base (velocity_env_cfg.py:112):
# a 1.6 m x 1.0 m grid at 0.1 m resolution, 187 rays, yaw-aligned, cast straight down onto
# `/World/ground`. `mdp.height_scan` feeds that into the policy every step, so a trench is
# already 187 numbers in the input vector. Over our 1.0 m-deep gap it reads about +0.9
# against -0.1 on flat ground: nine times the +/-0.1 observation noise, and well inside the
# +/-1.0 clip. Not a camera and not pixels, and that is why it runs in a pod with CUDA and
# no Vulkan.
#
# So the dog was never blind. EVERY REWARD TERM IN `_leap_profile` WAS. `lin_vel_z_l2`,
# `feet_air_time`, `dof_acc_l2`, `action_rate_l2` and the two tracking terms are all
# proprioceptive: not one of them can tell the lip of a trench from the middle of the
# platform. `leap` paid for hang time everywhere and for velocity everywhere, and the run
# measured 0.08 s of flight at iteration 0 and 0.08 s at iteration 4750.
#
# ── Why this is the missing gradient, and not just another weight ───────────────
# Section 8 of the README ends on a real wall: velocity tracking pays enormously for
# having crossed a trench and nothing at all for trying, so every failed attempt scores
# identically badly and PPO has nothing to climb. Relaxing `lin_vel_z_l2` from -2.0 to
# -0.05 made an attempt AFFORDABLE. It did not make one WORTH MAKING.
#
# Two terms below fix that, and the ordering they create is the whole point:
#
#     walked to the lip  <  jumped and fell in  <  jumped and nearly made it  <  crossed
#
# Under `leap`, the middle two were indistinguishable from the first. Neither term needs
# the robot to succeed before it pays, which is the property the old objective could not
# provide at any weight.
#
# ── On privilege ────────────────────────────────────────────────────────────────
# These read `ray_hits_w` raw: no noise, no clip. That is deliberate and it is not a
# cheat. A reward function is simulator-side scaffolding that never ships with the policy,
# so it is allowed ground truth the actor is not. THE OBSERVATION IS UNTOUCHED: the policy
# sees exactly the noisy, clipped scan `leap` saw, which is what keeps a `vault` run a
# fair comparison against the `leap` baseline rather than a different task.


def _gap_fraction(env, sensor_cfg, depth: float, x_range, y_half: float):
    """What fraction of a window of the height scan is a hole? Returns 0..1 per env.

    ── The reference height ────────────────────────────────────────────────────────
    `env.scene.env_origins[:, 2]`, the same frame `fell_into_hole` uses and for the same
    reason: our patches sit at twelve different heights across the curriculum grid, so a
    world-frame z means nothing, and the rays under the base are no good either because
    the robot is airborne for exactly the part of the manoeuvre we care about. The patch
    origin is fixed, is the surface the robot spawned on, and the terrain curriculum
    rewrites it on every promotion (terrain_importer.py:329).

    On `MeshGapTerrainCfg` the platform and the far side both sit at patch level and the
    trench floor is a hardcoded 1.0 m below (mesh_terrains.py:584), so `depth` cleanly
    separates the two. Rails go UP, so they read negative and never trip this.

    ── Fraction, not maximum ───────────────────────────────────────────────────────
    A max-depth version saturates the instant any ray finds the trench, which is 0.7 m
    out, and then sits flat for the whole approach: it pays a takeoff just as well from
    too far away as from the lip. The fraction peaks when the gap fills the window, which
    IS the takeoff moment, so the shaping term has a maximum in the right place instead of
    a plateau. For a 0.25 m trench in a 0.6 m window that peak is around 0.4, not 1.0; the
    weights below are set against that, not against a normalised signal.

    ── Guards ──────────────────────────────────────────────────────────────────────
    A ray that hits no mesh comes back `wp.inf` (kernels.py:197), and `ref - inf` is
    -inf, which is harmless. `ref - (-inf)` would be +inf and would read as an infinitely
    deep hole, so both ends are guarded rather than just the one that bites today. The
    5.0 m ceiling is the same guard for any finite-but-absurd hit off the terrain border.
    """
    import torch  # kept out of this module's top level, same as fell_into_hole

    sensor = env.scene.sensors[sensor_cfg.name]

    # Ray offsets in the sensor's LOCAL frame, so +x is robot-forward regardless of yaw:
    # `ray_alignment="yaw"` rotates these at cast time, it does not rewrite them. Row 0 is
    # enough because every env shares one pattern. Recomputed per call on purpose: it is
    # 187 elements of comparison against a 4096 x 187 x 3 tensor read, and a cached mask
    # on a live sensor object is a stale-state bug waiting for the first config change.
    starts = sensor.ray_starts.torch[0]
    window = (
        (starts[:, 0] >= x_range[0]) & (starts[:, 0] <= x_range[1])
        & (starts[:, 1].abs() <= y_half)
    )

    hit_z = sensor.data.ray_hits_w.torch[:, window, 2]
    drop = env.scene.env_origins[:, 2].unsqueeze(1) - hit_z
    deep = torch.isfinite(hit_z) & (drop > depth) & (drop < 5.0)
    return deep.float().mean(dim=1)


def gap_takeoff(env, sensor_cfg, asset_cfg, depth: float, x_range, y_half: float,
                min_speed: float = 0.5, vz_cap: float = 2.5):
    """Pay upward velocity, but only with a trench in front and forward speed already on.

    This is the term that gets the first hop off the ground. It is dense, it fires several
    steps BEFORE the robot commits, and it does not care whether the attempt works.

    Three factors, and each one is load-bearing:

      * `gap`, from the forward window, so pushing off in the middle of the platform pays
        nothing. This is the "when it sees a gap" half of the whole idea.
      * `vz` clamped to positive, so the landing does not pay. Capped at `vz_cap` so a
        catapult off a rail cannot dominate the return.
      * `fwd`, a ramp to full at `min_speed`. WITHOUT THIS THE TERM IS A POGO STICK: a
        robot that stands at the lip of a trench and bounces would farm `gap * vz`
        indefinitely, and hopping on the spot is a real attractor once `lin_vel_z_l2` is
        down at -0.05. Requiring forward speed means the only way to collect is to be
        moving at the gap, which is the behaviour we are buying.

    `lin_vel_z_l2` still charges -0.05 * vz^2 underneath this, so a 2.0 m/s takeoff costs
    -0.20 and earns roughly +0.8 at weight 1.0. Net positive at the lip, net negative
    anywhere else, which is exactly the shape wanted.

    ── Two different frames, on purpose ────────────────────────────────────────────
    `vz` is WORLD frame and `vx` is BASE frame, and mixing them is deliberate.

    Isaac Lab's own `lin_vel_z_l2` reads `root_lin_vel_b[:, 2]`, the base-frame z, and as
    a penalty that is fine. As a REWARD it is exploitable: the base z-axis points out of
    the robot's back, so pitching nose-up tilts it into the direction of travel and a
    robot running flat out at 2 m/s with a 30 degree rear reads +1.0 m/s of "vertical"
    velocity without a single foot leaving the ground. Paying for that buys a dog that
    pops a wheelie at every trench. `root_lin_vel_w[:, 2]` is the height of the base
    actually changing, which is the thing being bought.

    `vx` stays base-frame because the question there is "is it moving along its own
    heading", not "is it moving north".
    """
    gap = _gap_fraction(env, sensor_cfg, depth, x_range, y_half)
    data = env.scene[asset_cfg.name].data
    vz = data.root_lin_vel_w.torch[:, 2].clamp(min=0.0, max=vz_cap)
    fwd = (data.root_lin_vel_b.torch[:, 0] / min_speed).clamp(min=0.0, max=1.0)
    return gap * vz * fwd


def gap_flight(env, sensor_cfg, contact_cfg, asset_cfg, depth: float, x_range,
               y_half: float, vx_cap: float = 3.0):
    """Pay forward progress while every foot is off the ground and a trench is beneath.

    The payoff term, and the one that gives a FAILED crossing a gradient. A robot that
    launches and drops into the trench is airborne over a hole for the ~0.3 s of its fall
    and collects some of this before `fell_into_hole` ends the episode; a robot that gets
    further collects more. That single ordering is what `leap` could not express: under it,
    "jumped and nearly made it" and "never left the lip" scored the same.

    Diving in on purpose is not a hack worth worrying about, and the arithmetic is worth
    writing down rather than hoping. A fall banks roughly 0.3 s of this term and then
    forfeits the remaining ~15 s of a 1.5-weighted tracking term, which is two orders of
    magnitude more. Failing stays very expensive. It is just no longer INDISTINGUISHABLE
    from not trying, and that is the entire fix.

    `x_range` is centred on the base rather than ahead of it: by the time this term should
    pay, the gap is under the robot, not in front of it.

    Airborne means EVERY tracked foot has non-zero `current_air_time`. A trot has one or
    two feet down at all times, so an ordinary stride scores zero here no matter how long
    or fast it is; only a genuine flight phase pays. That is the same definition record.py
    measures the clips with, so the reward curve and the reported flight seconds are
    talking about the same event.
    """
    import torch  # for the zeros_like fallback below

    under = _gap_fraction(env, sensor_cfg, depth, x_range, y_half)

    air = env.scene.sensors[contact_cfg.name].data.current_air_time
    if air is None:  # track_air_time off; the term is meaningless, not fatal
        return torch.zeros_like(under)
    airborne = (air.torch[:, contact_cfg.body_ids] > 0.0).all(dim=1).float()

    vx = env.scene[asset_cfg.name].data.root_lin_vel_b.torch[:, 0].clamp(min=0.0, max=vx_cap)
    return under * airborne * vx


def _vault_profile(cfg, play: bool) -> None:
    """`leap`, plus the two reward terms that read the height scanner.

    Deliberately built ON TOP of `_leap_profile` rather than beside it. Everything that
    profile does is still necessary: a jump the reward set forbids at -14.6 cannot be
    bought back by adding a bonus, a run that has no terminate-on-fall trains on a
    heavy-tailed reward and keeps getting knocked over, and a curriculum that starts on
    row 5 spends hundreds of iterations climbing down to ground it can learn on. `vault`
    adds ONE new idea to that, so a difference between the two runs is attributable to it.
    """
    _leap_profile(cfg, play)

    from isaaclab.managers import RewardTermCfg, SceneEntityCfg

    # Windows scale with the robot for the same reason the gap ladder does: 0.7 m in front
    # of a Go2's base is its next footfall, and in front of an H1's it is under its own
    # knee. Same `_size_scale` the gap widths use, so a robot on a 2x ladder gets a 2x
    # window and the two stay in step.
    scale = _size_scale(cfg)

    # The reference geometry, all in metres and all Go2-sized before scaling:
    #   depth  0.30  well under the 1.0 m trench, well over the 0.35 m rails and the
    #                0.20 m boxes, so only a real hole trips it.
    #   ahead  0.10 to 0.70  the takeoff window. The scanner reaches 0.80 m and a Go2's
    #                front feet are ~0.20 m ahead of its base, so this is "the lip is
    #                between one and four steps away". Starting at 0.10 rather than 0.0
    #                keeps the two windows from overlapping under the body.
    #   under -0.40 to 0.40  the flight window, centred: by now the gap is beneath.
    #   y_half 0.30  narrower than the scanner's 0.50 so a trench off to the side, which
    #                the robot is running PARALLEL to, does not read as one in the way.
    depth = round(0.30 * scale, 3)
    ahead = (round(0.10 * scale, 3), round(0.70 * scale, 3))
    under = (round(-0.40 * scale, 3), round(0.40 * scale, 3))
    y_half = round(0.30 * scale, 3)

    # Reuse whatever foot bodies this robot's own `feet_air_time` term resolved, rather
    # than tabulating a regex per robot: Go2 is `.*_foot`, the base cfg is `.*FOOT`, and
    # the bipeds differ again. Deep-copied because `SceneEntityCfg.resolve` writes
    # `body_ids` back into the object and two terms should not share one.
    air_term = getattr(cfg.rewards, "feet_air_time", None)
    contact_cfg = (
        copy.deepcopy(air_term.params["sensor_cfg"])
        if air_term is not None and "sensor_cfg" in air_term.params
        else SceneEntityCfg("contact_forces", body_names=".*FOOT")
    )

    scan_cfg = SceneEntityCfg("height_scanner")
    robot_cfg = SceneEntityCfg("robot")

    # Weights, against a `track_lin_vel_xy_exp` that is 1.5 on the Go2 and is the biggest
    # thing in the set. Both terms are scaled by `step_dt` by the manager like every other
    # term, so these are comparable with the stock numbers as written.
    #
    # takeoff 1.0: peaks near gap 0.4 * vz 2.0 = 0.8 for the handful of steps a push-off
    #   lasts. Large enough to survive the -0.20 that `lin_vel_z_l2` charges for the same
    #   push, small enough that it cannot out-earn just running when there is no gap.
    # flight 2.0: the payoff, and intentionally the better deal. Around 0.4 * 1.5 = 0.6
    #   per step at weight 2.0 is 1.2, near the tracking ceiling, for the ~15 steps a
    #   crossing takes. A crossing should be the best thing that can happen on this course.
    cfg.rewards.gap_takeoff = RewardTermCfg(
        func=gap_takeoff, weight=1.0,
        params={"sensor_cfg": scan_cfg, "asset_cfg": robot_cfg,
                "depth": depth, "x_range": ahead, "y_half": y_half},
    )
    cfg.rewards.gap_flight = RewardTermCfg(
        func=gap_flight, weight=2.0,
        params={"sensor_cfg": scan_cfg, "contact_cfg": contact_cfg, "asset_cfg": robot_cfg,
                "depth": depth, "x_range": under, "y_half": y_half},
    )


# Terrain name -> the reward edits that terrain needs, or absent for "none, keep NVIDIA's".
REWARD_PROFILES = {
    "leap": _leap_profile,
    "vault": _vault_profile,
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
