"""A custom DeepMind Control domain: the planar walker, in a world with stuff in it.

`suite.load('walker', 'walk')` gives you a walker on an empty checkered strip. With a
camera that tracks the walker's centre of mass, a walker that is walking and a walker
that is shuffling on the spot look nearly identical in pixels. That is a problem for
two different reasons at once:

  * You cannot tell from the video whether the policy is really moving, so the reward
    number is the only evidence you have, and the reward number is exactly the thing
    you want a second opinion on.
  * The world model has almost nothing to predict except the walker's own limbs. A
    model that only ever has to predict the consequences of its own actions is a
    weaker demonstration than one that also has to predict things it does not control.

This domain fixes both with the same change: put objects in the world.

    ┌────────────────────────────────────────────────────────────┐
    │  ▌   ▌   ▌   ▌   █   ▌   ▌   ▌   ▌   █   ▌   ▌   ▌   ▌     │  markers
    │        ╷                                                   │
    │       ╱ ╲            ●          ●              ●           │  walker + balls
    │  ─────────────────────────────────────────────────────────  │  floor
    └────────────────────────────────────────────────────────────┘

Two kinds of prop, and they are deliberately different in kind:

  **Markers** are the row of posts along the far edge of the track, every 0.5 m, with
  a taller one every 5 m. They have `contype=0 conaffinity=0`, so they are invisible to
  the physics: nothing can touch them and they cannot change the reward by even a
  floating point epsilon. They exist to be *seen*. Because the camera tracks the
  walker, the posts stream past the background exactly in proportion to real forward
  progress, which turns "is this thing actually travelling?" into something you can
  answer by looking at the video for two seconds.

  **Balls** are real, free bodies with mass, and the walker kicks them. They are the
  part the world model has to earn: their motion is a consequence of physics rather
  than of the policy's actions, so predicting them correctly is only possible if the
  model has learned something about the world rather than about itself.

── The reward is byte-for-byte the stock walker reward ─────────────────────────
`Walk` subclasses `dm_control.suite.walker.PlanarWalker` and does not override
`get_reward`. That is the whole point. `walker_walk` is a task with published
DreamerV3 numbers (~950 return, reached in a few hundred thousand steps), and keeping
its reward means those numbers still apply here: if this run underperforms, the props
are not the excuse. The balls are light (0.25 kg against a ~20 kg walker) and low
friction, so they scatter on contact rather than acting as obstacles.

── Proprioception is byte-for-byte the stock walker observation ────────────────
Adding a free body adds six degrees of freedom to `qpos`/`qvel`, and adding bodies
shifts the rows of `xmat`. Left alone, that would silently change the proprioceptive
observation and quietly break the comparison to stock `walker_walk`. `ArenaPhysics`
therefore slices both by explicit name, so a proprio agent here sees exactly the 9
velocities and 14 orientations it would see in the stock domain, and has NO channel
through which it could perceive a ball.

Which makes the split clean and worth saying out loud on stream: the balls exist in
the physics, they exist in the pixels, and they do not exist in the state vector. Only
a model that learns from pixels has to explain them.

── Registration ────────────────────────────────────────────────────────────────
Importing this module registers the domain with dm_control's suite, so
`suite.load('arena', 'walk')` works and DreamerV3's `--task dmc_arena_walk` resolves
(embodied's DMC wrapper splits the task id on the first underscore into domain and
task). Nothing in dm_control or dreamerv3 is forked or patched; `suite._DOMAINS` is a
plain dict and this adds a key to it.
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET

import numpy as np
from dm_control.rl import control
from dm_control.suite import common, walker
from dm_control.utils import containers

# Stock walker constants, re-declared rather than imported: the leading underscore
# means they are private and a dm_control bump could rename them, and a silently
# different time limit would make every comparison to published numbers wrong.
_DEFAULT_TIME_LIMIT = 25
_CONTROL_TIMESTEP = 0.025
_WALK_SPEED = 1
_RUN_SPEED = 8

# The walker's own bodies and joints, in the order the stock domain reports them.
# `ArenaPhysics` uses these to slice the props back out of the observation.
_WALKER_BODIES = (
    "torso",
    "right_thigh", "right_leg", "right_foot",
    "left_thigh", "left_leg", "left_foot",
)
_WALKER_JOINTS = (
    "rootz", "rootx", "rooty",
    "right_hip", "right_knee", "right_ankle",
    "left_hip", "left_knee", "left_ankle",
)

# ── Track layout ────────────────────────────────────────────────────────────────
# The floor in walker.xml is a plane at pos="248 0 0" size="250 .8 .2", i.e. a strip
# 1.6 m wide in y whose drawn surface starts at x = -2. Everything below stays inside
# that, or it would hover over the void.
_TRACK_START, _TRACK_END, _MARKER_SPACING = -2.0, 30.0, 0.5
_MARKER_Y = 0.62          # far edge of the strip, so posts never occlude the walker
_TALL_EVERY = 10          # every 10th post (5 m) is a tall distance marker

# Balls, jittered per episode so the model cannot memorise where they are.
#
# Spacing is a compromise between two failure modes. Too far apart and there is no
# ball in shot for most of the episode, so the world model rarely has to predict one:
# the tracking camera sees roughly +/- 1.2 m of track, so at 1 m/s a 2.5 m spacing puts
# a ball on screen about half the time. Too close together and the walker spends the
# episode wading through them.
#
# Small and light for the same reason. The foot capsule has radius 0.05, so a 0.10 m
# ball is punted rather than stood on, and 0.15 kg against a ~20 kg walker is a nudge.
# The first one sits 0.8 m ahead so there is a ball in the very first frame, before the
# policy has learned to go anywhere.
_BALL_XS = tuple(0.8 + 2.5 * i for i in range(11))
_BALL_RADIUS = 0.10
_BALL_MASS = 0.15
_BALL_JITTER = 0.5

SUITE = containers.TaggedTasks()


# ── Model ───────────────────────────────────────────────────────────────────────


def _add_materials(root: ET.Element) -> None:
    """Bright, high-contrast colours. These have to read at 64x64."""
    asset = ET.SubElement(root, "asset")
    for name, rgba in (
        ("prop_ball", "0.95 0.35 0.15 1"),   # orange
        ("prop_post_a", "0.20 0.65 0.95 1"),  # blue
        ("prop_post_b", "0.95 0.90 0.30 1"),  # yellow
        ("prop_post_tall", "0.90 0.25 0.55 1"),  # magenta
    ):
        ET.SubElement(asset, "material", name=name, rgba=rgba)


def _add_markers(world: ET.Element) -> None:
    """The row of posts. Bare worldbody geoms, not bodies, for two reasons.

    A geom with no enclosing body is welded to the world, so it costs no degrees of
    freedom. And because it is not a body, it does not appear in `xmat` at all, which
    is what keeps `ArenaPhysics.orientations()` returning the stock 14 numbers no
    matter how many posts get added here.
    """
    count = int(round((_TRACK_END - _TRACK_START) / _MARKER_SPACING)) + 1
    for i in range(count):
        x = _TRACK_START + i * _MARKER_SPACING
        tall = i % _TALL_EVERY == 0
        half_h = 0.36 if tall else 0.16
        material = "prop_post_tall" if tall else ("prop_post_a", "prop_post_b")[i % 2]
        ET.SubElement(
            world, "geom",
            name=f"post_{i}", type="box",
            pos=f"{x:.3f} {_MARKER_Y} {half_h:.3f}",
            size=f"0.07 0.07 {half_h:.3f}",
            material=material,
            # Invisible to the physics. This is the line that guarantees the posts
            # cannot influence the reward.
            contype="0", conaffinity="0",
        )


def _add_balls(world: ET.Element) -> None:
    """Free bodies the walker can kick.

    `condim=6` turns on rolling friction; without it a sphere on a plane rolls
    forever, which looks wrong and gives the world model a trivially linear thing to
    predict. Explicit contype/conaffinity because the file-level default is
    `contype=1 conaffinity=0`, which would let the balls pass through the floor.
    """
    for i, x in enumerate(_BALL_XS):
        body = ET.SubElement(
            world, "body", name=f"ball_{i}", pos=f"{x:.3f} 0 {_BALL_RADIUS:.3f}"
        )
        ET.SubElement(body, "freejoint", name=f"ball_{i}")
        ET.SubElement(
            body, "geom",
            name=f"ball_{i}", type="sphere", size=f"{_BALL_RADIUS:.3f}",
            material="prop_ball", mass=f"{_BALL_MASS}",
            contype="1", conaffinity="1", condim="6",
            friction="0.6 0.02 0.004",
        )


def get_model_and_assets():
    """Stock walker.xml with the props injected, plus dm_control's shared assets.

    Editing the parsed tree rather than string-splicing a copy of walker.xml: this
    stays correct if dm_control edits the walker, and there is exactly one definition
    of the walker in play rather than a fork that drifts.
    """
    root = ET.fromstring(common.read_model("walker.xml"))
    root.set("model", "planar walker arena")
    world = root.find("worldbody")
    assert world is not None, "walker.xml has no worldbody"
    _add_materials(root)
    _add_markers(world)
    _add_balls(world)
    return ET.tostring(root, encoding="unicode"), common.ASSETS


# ── Physics ─────────────────────────────────────────────────────────────────────


class ArenaPhysics(walker.Physics):
    """Walker physics that reports the walker, and only the walker.

    Everything here exists so that the proprioceptive observation is identical to the
    stock domain's. `torso_height`, `torso_upright` and `horizontal_velocity` are
    inherited unchanged: they index by name or read a named sensor, so the extra
    bodies do not disturb them.
    """

    def orientations(self):
        """Stock returns `xmat[1:]`, which would now include the balls.

        Two steps rather than the stock one-liner: dm_control's named indexer
        broadcasts a name list against a field list instead of taking their outer
        product, so `xmat[bodies, ['xx', 'xz']]` raises. Rows by name, then columns 0
        and 2 of the flattened 3x3, which are xx and xz.
        """
        return self.named.data.xmat[list(_WALKER_BODIES)][:, [0, 2]].ravel()

    def velocity(self):
        """Stock returns all of `qvel`, which now has 6 extra dofs per ball."""
        return self.named.data.qvel[list(_WALKER_JOINTS)]

    def torso_x(self):
        """Distance travelled along the track. Logged, never observed."""
        return self.named.data.xpos["torso", "x"]

    def nearest_ball_distance(self):
        """Horizontal gap to the closest ball, in metres."""
        torso = self.named.data.xpos["torso", "x"]
        balls = [self.named.data.xpos[f"ball_{i}", "x"] for i in range(len(_BALL_XS))]
        return float(min(abs(b - torso) for b in balls))


# ── Task ────────────────────────────────────────────────────────────────────────


class ArenaWalker(walker.PlanarWalker):
    """Stock PlanarWalker. Same reward, same joint randomisation, extra props.

    `get_reward` is inherited and untouched, which is the claim this whole domain
    rests on.
    """

    # Which way along the track this task pays to travel. Only `back` flips it. It
    # exists so that "metres travelled" means the same thing for every task in the
    # domain: progress in the direction the reward asked for, not progress along +x.
    _direction = 1

    def initialize_episode(self, physics):
        # Stock randomisation first: it walks every joint in the model, so it must run
        # before the ball positions are set or it would clobber them. It leaves the
        # linear dofs of free joints alone but does randomise their quaternions, which
        # for a sphere is harmless.
        super().initialize_episode(physics)
        # Raw qpos/qvel slices rather than dm_control's named setter. The named setter
        # reshapes a view to assign into it, which numpy 2.5 deprecates, and the
        # warning fires once per ball per episode: eleven lines of traceback several
        # times a minute, straight into the log tail the report shows.
        for i, x in enumerate(_BALL_XS):
            jid = physics.model.name2id(f"ball_{i}", "joint")
            qadr = physics.model.jnt_qposadr[jid]
            vadr = physics.model.jnt_dofadr[jid]
            jitter = self.random.uniform(-_BALL_JITTER, _BALL_JITTER)
            # A freejoint's qpos is [x, y, z, qw, qx, qy, qz]. Writing all seven resets
            # orientation too, so no episode starts with a pre-spun ball.
            physics.data.qpos[qadr:qadr + 7] = (
                x + jitter, 0.0, _BALL_RADIUS, 1.0, 0.0, 0.0, 0.0
            )
            physics.data.qvel[vadr:vadr + 6] = 0.0

    def get_observation(self, physics):
        """Stock observation, plus two `log/` diagnostics.

        embodied strips every key beginning with `log/` before the agent ever sees it
        (`embodied/jax/agent.py` asserts on it), and its logger aggregates them per
        episode. So these two are recorded in metrics.jsonl and are invisible to the
        policy, the encoder and the world model.

        `log/x_position` is the honesty check. Episode return can climb through
        postures that score well without travelling, and the max of this over an
        episode says in metres whether the walker actually went anywhere.

        It is signed by `_direction`, so for `back` a walker correctly reversing down
        the track logs a positive number. Without that the report aggregates it as
        `epstats/log/x_position/max` and a perfect backwards run would show ~0 metres
        travelled, which reads exactly like a policy that never left the spot.
        """
        obs = super().get_observation(physics)
        obs["log/x_position"] = np.float64(self._direction * physics.torso_x())
        obs["log/ball_distance"] = np.float64(physics.nearest_ball_distance())
        return obs


class _ReversedPhysics:
    """`ArenaPhysics` with the sign of forward travel flipped, and nothing else.

    Delegates every attribute to the real physics except `horizontal_velocity`. That
    is what lets `BackwardWalker.get_reward` call dm_control's own implementation
    rather than restating it: the standing term, the tolerance margins, the linear
    sigmoid and the `(5 * move + 1) / 6` blend are literally the stock code, and the
    entire difference between `walk` and `back` is one minus sign.

    Reimplementing `get_reward` instead would work and would be shorter, but it would
    quietly fork the reward. The stock reward is the reason `walk` can be compared to
    published `walker_walk` numbers, and the transfer experiment needs `back` to be
    the same task with the goal reversed rather than a different task that happens to
    involve walking.
    """

    def __init__(self, physics):
        self._physics = physics

    def __getattr__(self, name):
        # Only reached for attributes this class does not define, so
        # `horizontal_velocity` below wins and everything else passes through.
        return getattr(self._physics, name)

    def horizontal_velocity(self):
        return -self._physics.horizontal_velocity()


class BackwardWalker(ArenaWalker):
    """Same body, same world, same reward function, opposite direction.

    The point of this task is the world model transfer: the physics of walking is
    byte-for-byte what `walk` already learned, so a world model trained on `walk`
    should predict `back` almost perfectly while the policy trained alongside it is
    exactly wrong. That isolates what a world model actually buys you. Changing the
    dynamics as well as the goal would confound the two.
    """

    _direction = -1

    def get_reward(self, physics):
        return super().get_reward(_ReversedPhysics(physics))


def _make(move_speed, time_limit, random, environment_kwargs, task_cls=ArenaWalker):
    physics = ArenaPhysics.from_xml_string(*get_model_and_assets())
    task = task_cls(move_speed=move_speed, random=random)
    return control.Environment(
        physics, task, time_limit=time_limit,
        control_timestep=_CONTROL_TIMESTEP, **(environment_kwargs or {}),
    )


@SUITE.add("benchmarking")
def walk(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walk at 1 m/s down a track with posts and kickable balls."""
    return _make(_WALK_SPEED, time_limit, random, environment_kwargs)


@SUITE.add("benchmarking")
def back(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Walk at 1 m/s the OTHER way. Stock reward with the velocity sign flipped."""
    return _make(
        _WALK_SPEED, time_limit, random, environment_kwargs, task_cls=BackwardWalker,
    )


@SUITE.add("benchmarking")
def run(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Same world, 8 m/s target. Much harder; here for the sweep, not the headline."""
    return _make(_RUN_SPEED, time_limit, random, environment_kwargs)


@SUITE.add("benchmarking")
def stand(time_limit=_DEFAULT_TIME_LIMIT, random=None, environment_kwargs=None):
    """Same world, no movement reward. Useful as a control: the posts never move, so
    the world model has strictly less to predict, and the balls are never kicked."""
    return _make(0, time_limit, random, environment_kwargs)


def register(name: str = "arena") -> None:
    """Make `suite.load(name, ...)` and DreamerV3's `--task dmc_arena_walk` work.

    `suite._DOMAINS` is built once at import by scanning the suite package's module
    globals, and it is an ordinary dict afterwards, so adding to it is enough. Called
    at import time below; exposed as a function so the intent is greppable.
    """
    from dm_control import suite

    suite._DOMAINS[name] = sys.modules[__name__]


register()
