"""A fly riding a robot: the connectome steering a body it never evolved for.

Everything else in this repo puts the fly's brain in a fly. This puts it on the back of
a four-legged robot roughly twenty-five times its length and lets it drive.

That is a fair test of the interface rather than a joke about tiredness. The claim the
whole repo rests on is that the connectome's output is a DESCENDING COMMAND, about 1,300
neurons carrying something closer to "turn left" than to "extend the left front femur",
and that a central pattern generator downstream turns that into legs. If that is true,
the same two numbers should drive a body with the wrong number of legs, the wrong mass
and the wrong gait, with nothing about the brain changed. This module is that experiment.

── The robot ───────────────────────────────────────────────────────────────────
Written from scratch in MJCF rather than pulled from a model zoo, for two reasons: the
pod has no network at run time and nothing else in this repo downloads anything, and
everything here has to be in flygym's units, which are MILLIMETRES with gravity at
-9810 mm/s^2. A trunk 60 mm long next to a 2.5 mm fly is about the size ratio of a mouse
carrying a housefly.

Those units are also the first trap. Holding this robot up takes roughly 3.5e5 of joint
torque per leg (8.8 g of mass, 9810 mm/s^2, a 16 mm lever, four legs), so position
servos with gains that look sane in SI are three orders of magnitude too weak and the
robot melts into the floor on the first step. `kp` here is 2.5e6.

── The gait, and the version of it that did not work ───────────────────────────
The first gait put a sine on the hip and a raised-cosine on the knee, which is what the
fly's own controller does. Measured, it walked, and the walk was uncontrollable:

    knee lift    drive 0.15   0.25   0.35   0.45      forward mm in 3 s
      0.50            40.1   36.5  -15.0  -25.4
      0.80            65.9   61.9   -0.1  -26.4
      1.10            52.0   52.7   53.2   53.4

Speed either fell with stride amplitude or ignored it entirely, and at high knee lift the
robot was swimming on its knee pump with the hips contributing nothing. A steering
command that cannot change speed cannot steer.

The fix is an explicit stance/swing split instead of sinusoids on both joints. During
stance the foot is planted and the hip sweeps BACKWARD at constant rate, which is the
push; during swing the knee folds and the hip returns. Amplitude then means stride
length, and a bigger stride on one side turns the robot. Measured at the tuned operating
point:

    drive (0.7, 0.7)   forward  70.4 mm in 3 s, heading drift  -0.1 deg
    drive (1.0, 0.4)   heading  +24.8 deg
    drive (0.4, 1.0)   heading  -31.4 deg

Near-zero drift walking straight, and the turn flips sign with the side. That drift
number matters more than it looks: any bias in the gait would show up in the fixation
result as if the brain had produced it.

── What the fly contributes ────────────────────────────────────────────────────
Nothing about the brain or the eyes changes. The fly is attached rigidly to a site on the
robot's trunk, its compound eyes look out from up there, and `bridge.Bridge` produces the
same two-number descending command it produces when the fly is walking on its own legs.
Measured from the saddle, with the robot's own arena around it:

    pillar dead ahead     facets darkened   L 0.0485   R 0.0527
    pillar 90 mm to left                    L 0.0222   R 0.0000
    pillar 90 mm to right                   L 0.0000   R 0.0208

which is the same clean lateralisation the ground-level fly gets, at about a third of the
amplitude.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

BRAIN_DT = 0.015
PHYSICS_DT = 1e-4
OUTPUT_FPS = 25

# Leg layout: name, x and y of the hip on the trunk, in mm.
LEGS = (("fl", +22.0, +11.0), ("fr", +22.0, -11.0), ("hl", -22.0, +11.0), ("hr", -22.0, -11.0))
# Diagonal pairs move together, which is a trot.
PHASE_OFFSET = {"fl": 0.0, "hr": 0.0, "fr": np.pi, "hl": np.pi}

# The tuned operating point. See the table in the module docstring.
GAIT_FREQ = 5.0          # Hz
STRIDE = 0.35            # radians of hip sweep at drive 1.0
KNEE_STANCE = 1.10       # radians. Puts the trunk at about 20 mm.
KNEE_LIFT = 0.60         # extra radians during swing, to clear the foot
SPAWN_Z = 22.0

# How the fly's command becomes the robot's. The fly's bridge emits drives around a base
# of 0.5 in [0.1, 1.6]; the robot is only well behaved between about 0.3 and 1.2, and
# outside that band it walks backwards. So the DIFFERENCE is what carries over, scaled,
# around a fixed base. This is the one place the two bodies are glued together.
RIDE_BASE = 0.7
# Why 2.5 and not something modest: the fly's smoothed imbalance in this arena runs about
# +/-0.05, which `descending_signal` turns into drives differing by roughly 0.18. The
# robot's usable band is 0.3 to 1.2, so it needs a differential near 0.9 to turn at any
# useful rate, and a gain of 0.6 delivered 0.11 of it. Measured with that gain, the robot
# corrected 3 degrees of bearing in 2.25 s while the target drifted away from it.
RIDE_TURN = 2.5
RIDE_LIMITS = (0.30, 1.20)

# The two bodies steer in OPPOSITE directions from the same command, and this sign is the
# only thing in the module that had to be measured rather than derived.
#
# A fly turns toward the side whose legs push LESS, so in `bridge.descending_signal` a
# left turn means lowering the left drive and raising the right. This robot is the other
# way round: a longer stride on the left sweeps the left side further back, and the body
# rotates toward the left. Measured over 0.9 s from a standing start:
#
#     command (1.18, 0.30)  left stride longer    heading  +8.8 deg   (turns left)
#     command (0.30, 1.18)  right stride longer   heading -14.0 deg   (turns right)
#
# So the fly's command has to be flipped before the robot sees it. Getting this wrong
# does not look like a bug, it looks like a fly that steers away from the target, which
# is exactly what the first closed-loop run did.
RIDE_TURN_SIGN = -1.0

# How far the hips yaw across a stance, at full turn command. Steering by differential
# STRIDE was tried first and abandoned: measured, it turned +53.6 deg in one direction
# and only -7.0 deg in the other from the same magnitude of command, so a right turn was
# very nearly a no-op and the fly could only steer one way. Yawing the planted feet
# instead rotates the body directly and is symmetric by construction.
# Measured, unwrapped heading change over 3 s at speed 0.7:
#
#     yaw sweep    turn -1.0   -0.5    0.0    +0.5   +1.0     degrees per 3 s
#       0.15          -142.0  -78.8   -1.0   +80.2  +136.4
#       0.25          -225.4 -122.0   -1.0  +117.6  +222.3
#       0.40          -362.6 -186.4   -1.0  +179.4  +388.2
#
# Symmetric, monotonic, and within a degree of straight at zero command, which is what
# differential stride never managed. 0.15 gives about 45 deg/s at full command, enough to
# correct a 45 degree bearing error in a second without the loop oscillating.
#
# A warning for anyone re-measuring this: read the heading UNWRAPPED. At these rates the
# robot can turn more than 180 degrees in the measurement window, which wraps and reads
# as a large turn in the opposite direction. That artifact cost an hour and looked
# exactly like a steering bug.
YAW_SWEEP = 0.15


def quadruped_mjcf(arena_half_size: float = 2000.0) -> str:
    """The robot, its arena, and the lighting the fly needs to see anything."""
    legs = ""
    for name, x, y in LEGS:
        legs += f"""
      <body name="{name}_hip" pos="{x} {y} 0">
        <joint name="{name}_yaw" type="hinge" axis="0 0 1" range="-0.6 0.6"/>
        <joint name="{name}_hip" type="hinge" axis="0 1 0" range="-1.2 1.2"/>
        <geom type="capsule" fromto="0 0 0 0 0 -16" size="2.6" rgba=".55 .58 .65 1"
              mass="0.35" group="2"/>
        <body name="{name}_knee" pos="0 0 -16">
          <joint name="{name}_knee" type="hinge" axis="0 1 0" range="-0.1 2.2"/>
          <geom type="capsule" fromto="0 0 0 0 0 -16" size="2.1" rgba=".45 .48 .55 1"
                mass="0.25" group="2"/>
          <geom name="{name}_foot" type="sphere" pos="0 0 -16" size="2.8" rgba=".2 .2 .25 1"
                friction="1.4 0.01 0.001" mass="0.1" group="2"/>
        </body>
      </body>"""
    actuators = "".join(
        f'<position name="{n}_{j}" joint="{n}_{j}" kp="2.5e6" dampratio="1" '
        f'ctrlrange="{lo} {hi}" forcerange="-4e6 4e6"/>'
        for n, _, _ in LEGS
        for j, lo, hi in (("yaw", -0.6, 0.6), ("hip", -1.2, 1.2), ("knee", -0.1, 2.2))
    )
    return f"""
<mujoco model="rider">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{PHYSICS_DT}" gravity="0 0 -9810" cone="elliptic" noslip_iterations="3"/>
  <visual>
    <global offheight="2048" offwidth="2048"/>
    <headlight ambient="0.5 0.5 0.5" diffuse="0.6 0.6 0.6" specular="0 0 0"/>
  </visual>
  <asset>
    <texture name="skybox" type="skybox" builtin="gradient" rgb1="1 1 1" rgb2="1 1 1"
             width="10" height="10"/>
    <texture name="checker" type="2d" builtin="checker" width="300" height="300"
             rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4"/>
    <material name="grid" texture="checker" texrepeat="60 60" reflectance="0"/>
  </asset>
  <worldbody>
    <light pos="0 0 400" dir="0 0 -1" diffuse=".8 .8 .8"/>
    <geom name="floor" type="plane" size="{arena_half_size} {arena_half_size} 1"
          material="grid" friction="1.4 0.01 0.001"/>
    <body name="trunk" pos="0 0 {SPAWN_Z}">
      <freejoint name="root"/>
      <!-- group 2: the eye renderer ignores groups 1 and 2, so the robot is invisible to
           its own passenger while staying visible to the chase camera. Without this the
           fly spends the run watching legs swing past and reports 0.13 of its retina
           darkened by the vehicle it is sitting on. -->
      <geom name="trunk" type="box" size="30 12 5" rgba=".72 .52 .22 1" mass="6.0" group="2"/>
      <camera name="chase" mode="trackcom" pos="-150 -110 90" xyaxes="0.6 -0.8 0 0.45 0.35 0.82"/>
      {legs}
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>"""


@dataclass
class TrotGait:
    """Stance/swing trot. Input is (left_drive, right_drive), the fly's own interface.

    Stance is the first half of each cycle: the foot is planted and the hip sweeps
    backward at a constant rate, which is what pushes the robot forward. Swing is the
    second half: the knee folds to clear the ground and the hip returns. Stride amplitude
    per side is what the descending command scales, so a longer stride on the left turns
    the robot.
    """

    freq: float = GAIT_FREQ
    stride: float = STRIDE
    knee_stance: float = KNEE_STANCE
    knee_lift: float = KNEE_LIFT
    duty: float = 0.5
    phase: float = 0.0

    yaw_sweep: float = YAW_SWEEP

    def step(self, drive: np.ndarray, dt: float = PHYSICS_DT) -> dict[str, float]:
        """Advance the gait by `dt` and return joint targets keyed by actuator name.

        `drive` is (speed, turn): the mean of the fly's two descending drives sets stride
        length, and their difference sets the yaw sweep. Keeping those two decoupled is
        what makes steering symmetric; see `YAW_SWEEP`.
        """
        self.phase = (self.phase + 2 * np.pi * self.freq * dt) % (2 * np.pi)
        speed, turn = float(drive[0]), float(drive[1])
        targets: dict[str, float] = {}
        for name, _, _ in LEGS:
            hip, knee, yaw = self._leg(self.phase + PHASE_OFFSET[name], speed, turn)
            targets[f"{name}_hip"] = hip
            targets[f"{name}_knee"] = knee
            targets[f"{name}_yaw"] = yaw
        return targets

    def _leg(self, phase: float, speed: float, turn: float) -> tuple[float, float, float]:
        phase = phase % (2 * np.pi)
        stance_end = 2 * np.pi * self.duty
        amplitude = self.stride * speed
        sweep = self.yaw_sweep * turn
        if phase < stance_end:
            # Stance: foot planted. The hip sweeps back (that is the push) and the yaw
            # sweeps across, which drags the planted foot sideways and rotates the body.
            fraction = phase / stance_end
            return amplitude * (1 - 2 * fraction), self.knee_stance, sweep * (1 - 2 * fraction)
        fraction = (phase - stance_end) / (2 * np.pi * (1 - self.duty))
        return (
            amplitude * (-1 + 2 * fraction),
            self.knee_stance + self.knee_lift * np.sin(np.pi * fraction),
            sweep * (-1 + 2 * fraction),
        )


def ride_command(descending: np.ndarray) -> np.ndarray:
    """Turn the fly's two-number descending command into the robot's (speed, turn).

    The fly emits a drive per side. The robot wants a speed and a yaw rate, so the mean
    of the two sides becomes speed and their difference becomes turn. That is the entire
    interface between the two animals, and it is four lines because the descending
    command really is that abstract.

    The fly's drives sit around 0.5 in a band of 0.1 to 1.6; the robot walks cleanly
    between 0.3 and 1.2 and goes backwards outside it, so speed is held near a fixed base
    rather than inherited. Turn is scaled hard because the fly's smoothed imbalance is
    only about +/-0.05 and the robot needs close to a full sweep to turn at a useful rate.
    """
    left, right = float(descending[0]), float(descending[1])
    lo, hi = RIDE_LIMITS
    speed = float(np.clip(RIDE_BASE + 0.5 * ((left + right) - 1.0), lo, hi))
    turn = float(np.clip(RIDE_TURN_SIGN * RIDE_TURN * (left - right), -1.0, 1.0))
    return np.array([speed, turn])


class RiderWorld:
    """A four-legged robot with a fly on its back, advanced one brain tick at a time.

    Mirrors `body.FlyWorld`'s interface closely enough that the closed loop reads the
    same: `look()` gives two eye darkenings, `step(command)` advances one tick.
    """

    def __init__(
        self,
        object_xy: tuple[float, float] = (240.0, 0.0),
        object_radius: float = 25.0,
        object_height: float = 80.0,
        spawn_heading_deg: float = 45.0,
        camera_res: tuple[int, int] = (360, 480),
        retina_threshold: float = 0.12,
        with_object: bool = True,
    ) -> None:
        import mujoco as mj

        self.object_xy = object_xy
        self.object_radius = object_radius
        self.spawn_heading_deg = spawn_heading_deg
        self.camera_res = camera_res

        self._model, self._data = self._build(
            object_xy if with_object else None, object_radius, object_height,
            spawn_heading_deg,
        )
        self._gait = TrotGait()
        self._steps_per_tick = int(round(BRAIN_DT / PHYSICS_DT))
        self._act = {
            mj.mj_id2name(self._model, mj.mjtObj.mjOBJ_ACTUATOR, i): i
            for i in range(self._model.nu)
        }
        self._trunk = mj.mj_name2id(self._model, mj.mjtObj.mjOBJ_BODY, "trunk")
        self._retina = None
        self._eye_renderer = None
        self._eye_renderer_model = None
        self._scene_option = None
        self.reference = self._empty_reference(object_radius, object_height, spawn_heading_deg)
        self.threshold = retina_threshold
        self.frames: list[np.ndarray] = []
        self.trajectory: list[tuple[float, float, float]] = []
        self._chase = mj.Renderer(self._model, height=camera_res[0], width=camera_res[1])
        self._record_pose()

    # ── Construction ────────────────────────────────────────────────────────────

    def _build(self, object_xy, object_radius, object_height, heading_deg):
        import mujoco as mj
        from flygym_demo.complex_terrain import make_locomotion_fly

        spec = mj.MjSpec.from_string(quadruped_mjcf())
        if object_xy is not None:
            spec.worldbody.add_geom(
                name="target",
                type=mj.mjtGeom.mjGEOM_CYLINDER,
                pos=[object_xy[0], object_xy[1], object_height / 2],
                size=[object_radius, object_height / 2, 0.0],
                rgba=[0.05, 0.05, 0.05, 1.0],
                contype=0,
                conaffinity=0,
            )
        fly = make_locomotion_fly("nmf", colorize=True)
        fly.add_vision()
        # The fly spec ships a `neutral` keyframe sized for a world containing only a
        # fly. Attaching it next to eight more actuators makes that keyframe the wrong
        # width and the compile fails with "Keyframe neutral has invalid ctrl size".
        # flygym's own add_fly deletes it for the same reason.
        for keyframe in [k for k in fly.mjcf_root.keys if k.name == "neutral"]:
            fly.mjcf_root.delete(keyframe)
        saddle = spec.body("trunk").add_site(name="rider", pos=[6.0, 0.0, 5.0])
        spec.attach(fly.mjcf_root, prefix="fly/", site=saddle)

        model = spec.compile()
        data = mj.MjData(model)
        half = np.deg2rad(heading_deg) / 2
        data.qpos[3:7] = [np.cos(half), 0.0, 0.0, np.sin(half)]
        data.qpos[2] = SPAWN_Z
        mj.mj_forward(model, data)
        return model, data

    def _empty_reference(self, object_radius, object_height, heading_deg) -> np.ndarray:
        """What the rider's eyes see with the pillar removed. Same idea as body.Retina."""
        import mujoco as mj

        model, data = self._build(None, object_radius, object_height, heading_deg)
        reference = self._read_eyes(model, data)
        log.info("rider empty-arena reference: %.4f / %.4f mean intensity", *reference.mean(axis=1))
        return reference

    # ── Vision ──────────────────────────────────────────────────────────────────

    def _read_eyes(self, model, data) -> np.ndarray:
        import mujoco as mj

        if self._retina is None:
            from flygym.vision.retina import Retina

            self._retina = Retina()
            self._scene_option = mj.MjvOption()
            # Groups 1 and 2 are the fly's own head and markers, hidden from its eyes to
            # avoid self-occlusion. Same choice flygym makes.
            self._scene_option.geomgroup[1] = 0
            self._scene_option.geomgroup[2] = 0
        # Cache the renderer per model. Building one costs a GL context setup, and an
        # earlier version built a fresh one on every tick, which dominated the run: a
        # 450-tick ride spent most of its wall clock creating and destroying renderers
        # rather than simulating anything.
        if getattr(self, "_eye_renderer_model", None) is not model:
            if self._eye_renderer is not None:
                try:
                    self._eye_renderer.close()
                except Exception:  # noqa: BLE001
                    pass
            self._eye_renderer = mj.Renderer(
                model, height=self._retina.nrows, width=self._retina.ncols
            )
            self._eye_renderer_model = model
        out = []
        for camera in ("fly/l_eye_cam_camera", "fly/r_eye_cam_camera"):
            self._eye_renderer.update_scene(data, camera, scene_option=self._scene_option)
            image = self._retina.correct_fisheye(self._eye_renderer.render())
            out.append(self._retina.raw_image_to_hex_pxls(image))
        return np.asarray(out).mean(axis=2)

    def look(self) -> np.ndarray:
        """(2,) fraction of each eye darkened relative to the empty arena."""
        now = self._read_eyes(self._model, self._data)
        darker = (self.reference - now) > 0.05
        return darker.mean(axis=1)

    # ── The loop ────────────────────────────────────────────────────────────────

    def step(self, descending: np.ndarray) -> None:
        """One brain tick: 150 physics steps under this command, then one frame."""
        import mujoco as mj

        command = ride_command(descending)
        for _ in range(self._steps_per_tick):
            for joint, target in self._gait.step(command).items():
                self._data.ctrl[self._act[joint]] = target
            mj.mj_step(self._model, self._data)
        self._chase.update_scene(self._data, "chase")
        self.frames.append(self._chase.render())
        self._record_pose()

    def _record_pose(self) -> None:
        x, y, _ = self.position
        self.trajectory.append((x, y, self.heading_deg))

    @property
    def position(self) -> np.ndarray:
        return self._data.xpos[self._trunk].copy()

    @property
    def heading_deg(self) -> float:
        forward = self._data.xmat[self._trunk].reshape(3, 3)[:, 0]
        return float(np.degrees(np.arctan2(forward[1], forward[0])))

    @property
    def distance_to_object(self) -> float:
        x, y, _ = self.position
        return float(np.hypot(self.object_xy[0] - x, self.object_xy[1] - y))

    @property
    def bearing_deg(self) -> float:
        x, y, _ = self.position
        error = (
            np.degrees(np.arctan2(self.object_xy[1] - y, self.object_xy[0] - x))
            - self.heading_deg
        )
        return float((error + 180) % 360 - 180)

    @property
    def upright(self) -> bool:
        """Has the robot fallen over? Useful as a run-ending condition."""
        up = self._data.xmat[self._trunk].reshape(3, 3)[2, 2]
        return bool(up > 0.5)

    def close(self) -> None:
        for renderer in (self._chase, self._eye_renderer):
            try:
                if renderer is not None:
                    renderer.close()
            except Exception:  # noqa: BLE001
                pass
