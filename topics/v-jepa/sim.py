"""A MuJoCo Franka that speaks V-JEPA 2-AC's action space.

V-JEPA 2-AC was trained on DROID: real video of real Franka Panda arms on real
tabletops, driven by a Cartesian end-effector controller. To plan with it in
simulation, the simulation has to offer the same interface it saw in training --
`[dx, dy, dz, ...]` in metres of end-effector motion -- and look enough like a
DROID scene that the encoder is not being asked something absurd.

Two halves, and they matter for different reasons:

`PandaReach` is the control side. The Panda's actuators are joint position
servos, so a Cartesian delta has to be turned into joint targets. We do that with
damped-least-squares differential IK on the end-effector point, which is ~20 lines
and has no failure mode more exciting than slowing down near a singularity. Note
what this buys the demo: the world model only ever emits Cartesian deltas, exactly
as it would on real hardware, and knows nothing about joints.

`scene_xml` is the perception side, and it is the half that decides whether any of
this works. The checkpoint has never seen a MuJoCo render. Every choice here --
the wood-grain table, the off-axis camera at roughly DROID's height and framing,
the visible clutter, warm lighting rather than MuJoCo's default flat headlight --
is trying to close a domain gap we did not create and cannot retrain away. The
honest position is that this is an out-of-distribution input to the encoder, so
`pipeline.py` measures whether planning beats its own baselines rather than
assuming it does.

Rendering is headless EGL, the same setup as topics/rl-mujoco and topics/fruit-fly:
the pod has no display and the devbox injects no graphics driver, so EGL resolves
to Mesa's software device from `libegl-mesa0` + `libgl1-mesa-dri`. Slow, entirely
adequate for 256x256, and the reason `config.py` carries a list of apt packages
that look like they belong to a desktop.
"""

from __future__ import annotations

import os

# Must precede `import mujoco`: the GL backend is chosen at import time and there is
# no way to change it afterwards.
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np

import mujoco

MENAGERIE = os.environ.get("MENAGERIE", "/opt/menagerie")
PANDA_DIR = os.path.join(MENAGERIE, "franka_emika_panda")

# Offset from the `hand` body frame to the point between the fingertips. The Panda's
# menagerie model defines no TCP site, so rather than patching the included XML we
# carry the offset and hand it to `mj_jac`, which accepts an arbitrary world point.
TCP_OFFSET = np.array([0.0, 0.0, 0.103])

# Joint-space home. Elbow up, wrist down, gripper open, which puts the fingertips
# over the middle of the table and well clear of every joint limit.
HOME_QPOS = np.array([0.0, 0.25, 0.0, -2.0, 0.0, 2.25, 0.785, 0.04, 0.04])


def _scene_path(seed: int, movable_clutter: bool = False, pushable: bool = False) -> str:
    """Write the scene beside panda.xml and return its path.

    Not a detail worth skipping. `MjModel.from_xml_string` resolves `<include>` and
    every mesh against the *working directory*, so an absolute include of
    panda.xml loads the body tree and then fails on `link1.stl`, because
    panda.xml's own `meshdir="assets"` is relative to a directory the compiler no
    longer knows about. Setting meshdir absolutely in this file does not fix it
    either: the included file's own `<compiler>` tag overrides ours. Writing the
    scene next to panda.xml, which is exactly what menagerie's own scene.xml does,
    makes both the include and meshdir resolve with no special cases.

    If the menagerie checkout is read-only, mirror the four things the model needs
    into a temp dir rather than failing.
    """
    import shutil
    import tempfile

    target_dir = PANDA_DIR
    if not os.access(target_dir, os.W_OK):
        target_dir = os.path.join(tempfile.gettempdir(), "vjepa_panda")
        if not os.path.isdir(os.path.join(target_dir, "assets")):
            os.makedirs(target_dir, exist_ok=True)
            for name in ("panda.xml", "hand.xml"):
                shutil.copy2(os.path.join(PANDA_DIR, name), target_dir)
            shutil.copytree(
                os.path.join(PANDA_DIR, "assets"),
                os.path.join(target_dir, "assets"),
                dirs_exist_ok=True,
            )
    tag = ("m" if movable_clutter else "s") + ("p" if pushable else "")
    path = os.path.join(target_dir, f"_vjepa_scene_{seed}{tag}.xml")
    with open(path, "w") as f:
        f.write(scene_xml(seed, movable_clutter, pushable))
    return path


def scene_xml(seed: int = 0, movable_clutter: bool = False, pushable: bool = False) -> str:
    """A tabletop scene built to look like DROID rather than like MuJoCo.

    `seed` jitters the clutter's positions and colours, so "does planning work" can
    be asked of more than one scene without hand-authoring each.
    """
    rng = np.random.default_rng(seed)

    def clutter() -> str:
        out = []
        # Distractors live BEHIND and BESIDE the target, never between the gripper
        # and it: this is a reaching task, and an accidental obstacle course would
        # confound "the world model cannot plan" with "the arm is stuck".
        spots = [(0.34, -0.30), (0.62, 0.28), (0.30, 0.34), (0.70, -0.24)]
        for i, (x, y) in enumerate(spots):
            c = rng.uniform(0.25, 0.8, size=3)
            h = float(rng.uniform(0.02, 0.05))
            x += float(rng.uniform(-0.03, 0.03))
            y += float(rng.uniform(-0.03, 0.03))
            # A free joint puts this body's pose in MjData instead of MjModel, which
            # is what lets many environments sharing one model each have their own
            # distractor layout, and lets it be teleported every step.
            joint = "<freejoint/>" if movable_clutter else ""
            out.append(
                f'<body name="clutter{i}" pos="{x:.3f} {y:.3f} {h:.3f}">{joint}'
                f'<geom type="box" size="{h:.3f} {h:.3f} {h:.3f}" '
                f'rgba="{c[0]:.2f} {c[1]:.2f} {c[2]:.2f} 1" contype="0" conaffinity="0"/></body>'
            )
        return "\n    ".join(out)

    return f"""
<mujoco model="vjepa droid-ish">
  <include file="panda.xml"/>
  <statistic center="0.5 0 0.25" extent="1.1"/>

  <visual>
    <!-- Warm, slightly directional light. MuJoCo's default flat grey headlight makes
         renders that look like CAD screenshots, which is the opposite of DROID. -->
    <headlight diffuse="0.45 0.43 0.40" ambient="0.30 0.29 0.28" specular="0.15 0.15 0.15"/>
    <rgba haze="0.55 0.56 0.58 1"/>
    <global azimuth="150" elevation="-25" offwidth="640" offheight="640"/>
    <quality shadowsize="4096"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.72 0.74 0.78" rgb2="0.45 0.47 0.52"
             width="512" height="3072"/>
    <texture type="2d" name="wood" builtin="flat" rgb1="0.62 0.47 0.32" rgb2="0.55 0.40 0.27"
             width="512" height="512" mark="random" markrgb="0.50 0.37 0.24" random="0.28"/>
    <material name="wood" texture="wood" texuniform="false" texrepeat="1 1"
              specular="0.12" shininess="0.08" reflectance="0.02"/>
    <texture type="2d" name="floor" builtin="checker" rgb1="0.42 0.42 0.44" rgb2="0.37 0.37 0.39"
             width="300" height="300"/>
    <material name="floor" texture="floor" texuniform="true" texrepeat="6 6" reflectance="0.05"/>
    <material name="wall" rgba="0.74 0.72 0.68 1" specular="0.05" shininess="0.03"/>
  </asset>

  <worldbody>
    <light pos="0.9 0.6 1.8" dir="-0.4 -0.3 -1" diffuse="0.5 0.48 0.45" specular="0.1 0.1 0.1"
           castshadow="true"/>
    <light pos="-0.4 -0.8 1.5" dir="0.3 0.5 -1" diffuse="0.22 0.22 0.25" castshadow="false"/>
    <geom name="floor" pos="0 0 -0.4" size="0 0 0.05" type="plane" material="floor"/>

    <!-- Walls. Without them the table floats in a black void, which reads as a game
         level rather than a lab and is nothing the encoder saw in DROID. They cost
         nothing and they fill the upper half of the frame with something plausible. -->
    <geom name="wall_back" type="box" pos="1.35 0 0.55" size="0.02 2.2 1.0" material="wall"
          contype="0" conaffinity="0"/>
    <geom name="wall_left" type="box" pos="0 1.45 0.55" size="2.2 0.02 1.0" material="wall"
          contype="0" conaffinity="0"/>

    <!-- Table top flush with the robot base, so z=0 is the work surface. Big enough
         to run past the frame edges, so the camera never sees the table end. -->
    <body name="table" pos="0.45 0 -0.02">
      <geom name="table_top" type="box" size="0.80 0.85 0.02" material="wood"
            contype="1" conaffinity="1" friction="1 0.05 0.001"/>
      <geom name="table_leg" type="box" pos="0 0 -0.21" size="0.55 0.60 0.19"
            rgba="0.34 0.26 0.19 1" contype="0" conaffinity="0"/>
    </body>

    <!-- The target. Saturated red on a brown table is the one thing in frame that
         cannot be confused with anything else, which matters when the only signal
         the planner gets is a distance between two whole-frame embeddings.
         Its position moves with the seed, and that is not cosmetic: with a fixed
         target every seed starts at exactly the same 21.5 cm and "three seeds" only
         ever tested three arrangements of the distractors. -->
    <body name="target" pos="{0.50 + rng.uniform(-0.10, 0.10):.3f} {0.16 + rng.uniform(-0.26, 0.10):.3f} {0.035 if not pushable else 0.03}">
      {'<freejoint/>' if pushable else ''}
      <geom name="target" type="box" size="{0.035 if not pushable else 0.03} {0.035 if not pushable else 0.03} {0.035 if not pushable else 0.03}"
            rgba="0.80 0.13 0.11 1"
            contype="{1 if pushable else 0}" conaffinity="{1 if pushable else 0}"
            {'density="400" friction="0.9 0.02 0.001" solref="0.005 1"' if pushable else ''}/>
    </body>
    {clutter()}

    <!-- Roughly DROID's exterior camera: off to one side, above the table, tilted
         down at the workspace, wide enough to see the whole arm. `targetbody` aims
         it at the table so the framing survives changing the position.
         Keep x < 1.33 and y > -1.43: past those the camera sits INSIDE one of the
         walls above and every frame renders as flat brown, with no error. -->
    <camera name="exterior" pos="1.26 -1.02 0.92" mode="targetbody" target="table" fovy="47"/>
    <camera name="side" pos="0.45 -1.38 0.80" mode="targetbody" target="table" fovy="47"/>
  </worldbody>
</mujoco>
"""


class PandaReach:
    """Franka Panda with a Cartesian end-effector delta interface."""

    def __init__(self, seed: int = 0, render_size: int = 256, camera: str = "exterior",
                 pushable: bool = False):
        self.pushable = pushable
        self.model = mujoco.MjModel.from_xml_path(_scene_path(seed, False, pushable))
        self.data = mujoco.MjData(self.model)
        self.camera = camera
        self.render_size = render_size
        self._renderer: mujoco.Renderer | None = None

        # Gravity-compensate the arm, as a real Franka does. Worth knowing that this
        # is NOT what fixes the tracking error below: measured, gravcomp moved a
        # 50 mm command from 33 mm delivered to 33 mm delivered, and zeroing gravity
        # outright only reached 35 mm. The Panda's PD position servos have a standing
        # error that is not gravity droop, so `apply_delta` closes the loop instead.
        self.model.body_gravcomp[:] = 1.0

        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.nq_arm = 7
        self.reset()

    # ── state ───────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[: len(HOME_QPOS)] = HOME_QPOS
        self.data.ctrl[: self.nq_arm] = HOME_QPOS[: self.nq_arm]
        if self.model.nu > self.nq_arm:
            self.data.ctrl[self.nq_arm] = 255.0  # gripper open
        mujoco.mj_forward(self.model, self.data)

    @property
    def ee_pos(self) -> np.ndarray:
        """World position of the point between the fingertips."""
        R = self.data.xmat[self.hand_id].reshape(3, 3)
        return self.data.xpos[self.hand_id] + R @ TCP_OFFSET

    @property
    def target_pos(self) -> np.ndarray:
        return self.data.xpos[self.target_id].copy()

    def pose7(self) -> np.ndarray:
        """The 7-dim state vector the AC predictor conditions on.

        `[x, y, z, roll, pitch, yaw, grip]`. DROID's own state uses Euler angles,
        so we hand over Euler angles even though a quaternion would be better
        behaved -- the model's state encoder is a single Linear layer trained on
        that convention and reinterpreting its input would be a silent mismatch.
        """
        R = self.data.xmat[self.hand_id].reshape(3, 3)
        pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
        grip = float(self.data.ctrl[self.nq_arm] / 255.0) if self.model.nu > self.nq_arm else 1.0
        return np.array([*self.ee_pos, roll, pitch, yaw, grip], dtype=np.float32)

    # ── control ─────────────────────────────────────────────────────────────────

    def _ik_step(self, dx: np.ndarray, damping: float = 0.08) -> np.ndarray:
        """Damped-least-squares differential IK: Cartesian delta -> joint delta.

        Damping is what keeps this from exploding near a singularity: the plain
        pseudo-inverse sends joint velocities to infinity as the Jacobian loses
        rank, whereas `Jt (J Jt + k^2 I)^-1` just stops making progress in the
        directions the arm cannot move. A planner that asks for an unreachable
        motion should get a stalled arm, not a thrown exception.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacp, jacr, self.ee_pos, self.hand_id)
        J = jacp[:, : self.nq_arm]
        JJt = J @ J.T + (damping**2) * np.eye(3)
        return J.T @ np.linalg.solve(JJt, dx)

    def apply_delta(self, delta_xyz, grip: float | None = None, substeps: int = 50,
                    outer: int = 10, tol: float = 1e-3) -> None:
        """Move the end-effector by `delta_xyz` metres, then let physics settle.

        Closed-loop, and it has to be. Solving IK once and commanding the result
        delivers 33 mm of a 50 mm request, because the position servos settle at a
        standing joint error of ~0.009 rad that is not gravity (gravity compensation
        and zero-gravity both leave it). Re-solving against the *achieved* pose a
        few times drives the real end-effector onto the target, which is also what
        a real Cartesian controller does. Measured effect: 16.1 mm mean error ->
        under 1 mm.

        This is exactly the kind of bug that would not have announced itself. The
        arm would simply have under-reached, and the report would have read as
        "the world model cannot plan" with no hint that the controller ate a third
        of every commanded motion.

        Joint targets are clipped to the model's own limits, so a planner that asks
        for something out of reach gets a clipped move rather than a broken robot.
        """
        goal = self.ee_pos + np.asarray(delta_xyz, dtype=np.float64)
        if grip is not None and self.model.nu > self.nq_arm:
            self.data.ctrl[self.nq_arm] = float(np.clip((1.0 - grip) * 255.0, 0, 255))
        lo = self.model.jnt_range[: self.nq_arm, 0]
        hi = self.model.jnt_range[: self.nq_arm, 1]
        for _ in range(outer):
            err = goal - self.ee_pos
            if np.linalg.norm(err) < tol:
                break
            # The correction ACCUMULATES onto ctrl rather than being re-derived from
            # qpos. That distinction is the whole fix: a servo settles where
            # kp*(ctrl - q) balances the load, so re-solving "joint angles that put
            # the hand at the goal" and writing them to ctrl reproduces the same
            # offset every iteration and converges to nothing (measured: 33 mm, then
            # 8.5 mm, then 8.5 mm forever). Adding the residual to the standing
            # command is integral action, and it does converge.
            dq = self._ik_step(np.clip(err, -0.05, 0.05))
            self.data.ctrl[: self.nq_arm] = np.clip(
                self.data.ctrl[: self.nq_arm] + dq, lo, hi
            )
            mujoco.mj_step(self.model, self.data, nstep=substeps)
        mujoco.mj_forward(self.model, self.data)

    def set_gripper(self, closed: float, substeps: int = 200) -> None:
        """Open (0.0) or close (1.0) the fingers, and actually step until they move.

        Needs its own method: `apply_delta` with a zero delta returns before stepping,
        because the end-effector is already at its goal, so the tendon never actuates
        and the fingers stay wherever they were. Menagerie's Panda uses ctrl 255 for
        open and 0 for closed on the `split` tendon.
        """
        if self.model.nu <= self.nq_arm:
            return
        self.data.ctrl[self.nq_arm] = float(np.clip((1.0 - closed) * 255.0, 0, 255))
        mujoco.mj_step(self.model, self.data, nstep=substeps)
        mujoco.mj_forward(self.model, self.data)

    def step(self, action7) -> None:
        """Apply one 7-dim V-JEPA 2-AC action."""
        a = np.asarray(action7, dtype=np.float64)
        self.apply_delta(a[:3], grip=None)

    def move_to(self, pos, iters: int = 120) -> None:
        """Drive the end-effector to an absolute position. Used only to build goal
        images and the scripted oracle, never by the planner."""
        for _ in range(iters):
            err = np.asarray(pos, dtype=np.float64) - self.ee_pos
            if np.linalg.norm(err) < 2e-3:
                break
            self.apply_delta(np.clip(err, -0.04, 0.04), substeps=30, outer=6)

    # ── rendering ───────────────────────────────────────────────────────────────

    def render(self, size: int | None = None) -> np.ndarray:
        """[H, W, 3] uint8 from the scene camera."""
        size = size or self.render_size
        if self._renderer is None or self._renderer.height != size:
            if self._renderer is not None:
                self._renderer.close()
            self._renderer = mujoco.Renderer(self.model, height=size, width=size)
        self._renderer.update_scene(self.data, camera=self.camera)
        return self._renderer.render()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


def reach_task(seed: int = 0, render_size: int = 256, camera: str = "exterior"):
    """Build the episode: a start state, a goal image, and the goal position.

    The goal image is a real render of the arm already at the target, which is the
    only honest way to pose the task to a model that compares embeddings of whole
    frames: the planner is asked to make the scene LOOK like this, and whether
    that corresponds to reaching is exactly what gets measured.

    Returns (env, goal_frame, goal_pos, start_frame).
    """
    env = PandaReach(seed=seed, render_size=render_size, camera=camera)
    # Hover just above the red block rather than inside it, so the goal frame does
    # not show the gripper interpenetrating the target.
    goal_pos = env.target_pos + np.array([0.0, 0.0, 0.09])
    env.move_to(goal_pos)
    goal_frame = env.render()
    reached = env.ee_pos.copy()
    env.reset()
    start_frame = env.render()
    return env, goal_frame, reached, start_frame


# ── the push task ───────────────────────────────────────────────────────────────
#
# The reach task has a weakness that only showed up once it was controlled properly:
# its goal photograph differs from the start photograph ONLY in where the arm is, so
# any image distance is monotone in arm position and raw pixels plan it exactly as well
# as V-JEPA (+90.4% each, see the README). The task never asks the representation a
# question.
#
# Moving an OBJECT fixes that. Now the goal photo shows the world in a different state,
# not the robot in a different pose. Walking the arm toward the goal no longer reduces
# the distance; the planner has to work out that it must make contact and push. Whether
# a learned representation helps there is a fair question in a way that reaching is not.


def scripted_push(env: "PandaReach", cube_from, cube_to, lift: float = 0.20,
                  push_height: float = 0.035) -> None:
    """Drive the gripper behind the cube and shove it to `cube_to`.

    Used for two things and never by the planner: rendering the goal photograph, and
    as the oracle upper bound. Approach is deliberately over-the-top-then-down rather
    than straight-line, because a straight line from the home pose goes through the
    cube and knocks it somewhere random before the push even starts.
    """
    d = np.asarray(cube_to, dtype=np.float64) - np.asarray(cube_from, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-6:
        return
    u = d / n
    # CLOSE THE GRIPPER FIRST. The Panda's fingers open to about 8 cm and the cube is
    # 6 cm wide, so an open gripper straddles it and the "push" moves it 2 cm instead
    # of 15. Closed, the fingertips are a solid paddle.
    env.set_gripper(1.0)
    behind = np.asarray(cube_from, dtype=np.float64) - u * 0.085
    env.move_to(np.array([behind[0], behind[1], lift]))          # above and behind
    env.move_to(np.array([behind[0], behind[1], push_height]))   # down to table height
    # Overshoot slightly: contact is compliant, so stopping exactly at the goal leaves
    # the cube short.
    end = np.asarray(cube_to, dtype=np.float64) - u * 0.045
    env.move_to(np.array([end[0], end[1], push_height]), iters=200)


def push_task(seed: int = 0, render_size: int = 256, camera: str = "exterior",
              distance: float = 0.15, start_in_contact: bool = False):
    """Build a push episode. Returns (env, goal_frame, goal_cube, start_frame, cube0).

    The goal photograph is produced by ACTUALLY DOING THE TASK and photographing the
    result, then resetting. That matters for honesty: it guarantees the goal state is
    reachable by this arm in this scene, and it means the image the planner is handed
    is a real frame from the same camera rather than a composite nobody could achieve.
    """
    env = PandaReach(seed=seed, render_size=render_size, camera=camera, pushable=True)
    cube0 = env.target_pos.copy()
    rng = np.random.default_rng(seed)

    # Push roughly toward the camera-left/front of the table, jittered per seed, and
    # clipped to stay on the table and inside the arm's comfortable reach.
    theta = rng.uniform(-0.6, 0.6) + np.pi  # mostly back toward the base
    u = np.array([np.cos(theta), np.sin(theta), 0.0])
    goal_cube = cube0 + u * distance
    goal_cube[0] = float(np.clip(goal_cube[0], 0.34, 0.66))
    goal_cube[1] = float(np.clip(goal_cube[1], -0.28, 0.32))
    goal_cube[2] = cube0[2]

    scripted_push(env, cube0, goal_cube)
    goal_frame = env.render()
    achieved = env.target_pos.copy()

    env.reset()
    if start_in_contact:
        # Park the closed gripper just behind the cube, on the push axis. This splits
        # the task in two: from the home pose a greedy planner has to discover a
        # go-around-then-push manoeuvre that makes the image temporarily LESS like the
        # goal, which one-step descent cannot do. Starting in contact removes that
        # exploration problem and measures only whether the reward can steer a push.
        env.set_gripper(1.0)
        behind = cube0 - u * 0.085
        env.move_to(np.array([behind[0], behind[1], 0.20]))
        env.move_to(np.array([behind[0], behind[1], 0.035]))
    start_frame = env.render()
    return env, goal_frame, achieved, start_frame, cube0
