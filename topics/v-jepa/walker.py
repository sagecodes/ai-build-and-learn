"""The Unitree G1 humanoid as a robot V-JEPA can steer: RL legs, a world-model planner.

Nothing learns to walk without a reward, so the legs here are NOT V-JEPA. They are the
PPO policy from topics/rl-mujoco (MJX + Brax, 300M steps on rough terrain), which
turns a joystick command (forward m/s, sideways m/s, yaw rad/s) into 29 joint targets
at 50 Hz. This module wraps that policy so the world model only ever decides the one
thing a planner should decide: where to go next.

    world model  ->  "walk 40 cm that way"         (one decision every 0.6 s)
    this module  ->  joystick command, heading hold
    PPO policy   ->  29 joint targets at 50 Hz
    MuJoCo       ->  physics

That split is the hierarchical-planning argument for world models in miniature: the
low level is trained with a reward once, and everything above it plans toward a
PHOTOGRAPH with no reward at all.

── Measured on the Spark (CPU physics, jax 0.9.2, mujoco 3.13) ─────────────────
100 control steps (2 s of robot time) run in ~1 s on CPU, after a ~12 s jit compile.
The policy tracks forward/back well (asked 1.0 m/s, gets 0.88), strafes at about half
speed, and turns weakly (~0.35 rad/s when asked for 1.0). So actions are expressed in
WORLD coordinates and converted to body-frame commands using the robot's own heading,
and the heading is held rather than steered.

── What is visual-only ─────────────────────────────────────────────────────────
The coloured pads and pillars are added to the RENDER scene, not the physics model:
the policy was trained on this terrain and nothing the planner does should change
what its feet touch. They exist so a goal photograph means something to a person
("stand on the blue pad") and so the chase camera has landmarks to see move.
"""

from __future__ import annotations

import copy
import functools
import os
import pickle

import numpy as np

# Where the rl-mujoco checkpoint lives in the devbox blob store (run rlswb4sgxg6j2vwfr9gr,
# 300M steps, rough terrain, baseline preset). A remote File, so no staging is needed.
G1_CHECKPOINT = (
    "s3://flyte-data/d6/physical-ai/development/rlswb4sgxg6j2vwfr9gr/"
    "nfggzwfe3koyd2ww5n1qdj97/1/e9e6b700224580aff7c17fef99c9d4d7/g1_checkpoint.pkl"
)

DECISION_STEPS = 30      # control steps per planner decision: 0.6 s at 50 Hz
MAX_STEP_M = 0.45        # longest move the planner may ask for in one decision
POSE_SCALE = 0.1         # metres of floor -> AC-predictor units (DROID moves are ~cm)

# Pads: the places a goal photo can ask for. Spawn is the origin.
PADS = {
    "red": (np.array([2.1, 0.3]), (0.90, 0.20, 0.20)),
    "blue": (np.array([0.2, 2.2]), (0.20, 0.35, 0.95)),
    "yellow": (np.array([-1.9, 0.6]), (0.95, 0.80, 0.15)),
    "green": (np.array([0.4, -2.0]), (0.20, 0.80, 0.30)),
}
PAD_RADIUS = 0.35
ARENA = 3.0              # random walks are herded back inside this radius

# The camera the MODEL sees through: it follows the robot from 13 m up, nearly straight
# down, at a fixed world angle. Chosen by measurement, not taste. Robot on a 13x13 grid
# of positions, energy of each view against a goal photo (task-free, no planner):
#
#   camera, floor                 rank corr with distance   best neighbour is closer
#   4.2 m / -38 deg, stud grid              0.32                     60%
#   10 m / -75 deg, random studs            0.71                     78%
#   13 m / -82 deg, random studs            0.80  (0.85 centred)     84% (89% centred)
#   raw pixels, same views                 -0.09 to 0.37             ~55%
#
# A regular stud grid aliases (every 0.75 m looks the same), and a low camera loses
# the pads once they leave the frame. People get a nicer low chase view (SHOW_CAM) in
# the videos, always labelled as not what the model sees.
MODEL_CAM = (13.0, 225.0, -82.0)
SHOW_CAM = (4.6, 225.0, -30.0)

_PILLAR_COLOURS = [(0.9, 0.3, 0.1), (0.1, 0.6, 0.9), (0.9, 0.9, 0.9), (0.6, 0.2, 0.8),
                   (0.2, 0.8, 0.5), (0.95, 0.6, 0.1), (0.3, 0.3, 0.9), (0.9, 0.2, 0.5)]


def _yaw(quat) -> float:
    w, x, y, z = [float(v) for v in quat]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


class G1Walker:
    """One G1, one arena, a chase camera for the model and a map camera for people."""

    def __init__(self, checkpoint_path: str, seed: int = 0, chase=MODEL_CAM,
                 studs: str = "scatter"):
        self.chase = chase
        self.studs = studs
        self.model_sees_robot = True
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
        import jax
        import mujoco
        from brax.training.acme import running_statistics
        from brax.training.agents.ppo import networks as ppo_networks
        from mujoco_playground import registry

        self.jax = jax
        with open(checkpoint_path, "rb") as f:
            ck = pickle.load(f)
        name = {"rough": "G1JoystickRoughTerrain", "flat": "G1JoystickFlatTerrain"}[ck.get("terrain", "rough")]
        cfg = registry.get_default_config(name)
        cfg.njmax = 128          # see topics/rl-mujoco envs.py: 90 overflows and MJX drops contacts
        cfg.impl = "jax"         # CPU; the GPU belongs to V-JEPA
        # Training shoves the robot at random intervals to make the gait robust. Here
        # that would be unexplained motion the world model is blamed for predicting.
        cfg.push_config.enable = False
        self.env = registry.load(name, config=cfg)
        net = functools.partial(ppo_networks.make_ppo_networks, **ck["network_factory"])(
            self.env.observation_size, self.env.action_size,
            preprocess_observations_fn=running_statistics.normalize)
        self._policy = jax.jit(ppo_networks.make_inference_fn(net)(ck["params"], deterministic=True))
        self._reset = jax.jit(self.env.reset)
        self._step = jax.jit(self.env.step)
        self.dt = float(self.env.dt)

        # Render-only copy of the model: brighter, and nothing we change here can
        # reach the MJX physics, which was built from the original at load time.
        m = copy.deepcopy(self.env.mj_model)
        m.vis.headlight.ambient[:] = 0.6
        m.vis.headlight.diffuse[:] = 0.6
        m.vis.headlight.specular[:] = 0.1
        for i in range(m.nmat):
            m.mat_rgba[i, :3] = np.clip(m.mat_rgba[i, :3] * 1.8 + 0.08, 0, 1)
        self.m = m
        self.d = mujoco.MjData(m)
        self._renderers: dict[int, mujoco.Renderer] = {}
        self.trail: list[np.ndarray] = []
        self.reset(seed)

    # ── state ───────────────────────────────────────────────────────────────────

    def reset(self, seed: int = 0) -> None:
        self.rng = self.jax.random.PRNGKey(seed)
        self.state = self._reset(self.jax.random.PRNGKey(seed))
        self.heading = self.yaw        # held, not steered; see module docstring
        self.fell = False
        self.trail = [self.pos.copy()]
        self._set_command((0.0, 0.0, 0.0))
        self.settle(25)

    @property
    def qpos(self) -> np.ndarray:
        return np.asarray(self.state.data.qpos)

    @property
    def pos(self) -> np.ndarray:
        return self.qpos[:2].copy()

    @property
    def yaw(self) -> float:
        return _yaw(self.qpos[3:7])

    def pose7(self) -> np.ndarray:
        """The state vector the AC predictor conditions on, in its DROID-ish units."""
        x, y = self.pos * POSE_SCALE
        return np.array([x + 0.5, y, 0.2, 0.0, 0.0, self.yaw, 0.0], dtype=np.float32)

    def snapshot(self):
        return (self.state, self.rng, self.fell, list(self.trail))

    def restore(self, snap) -> None:
        self.state, self.rng, self.fell, trail = snap
        self.trail = list(trail)

    # ── control ─────────────────────────────────────────────────────────────────

    def _set_command(self, cmd) -> None:
        import jax.numpy as jp

        self._cmd = jp.array(cmd, dtype=jp.float32)
        self.state.info["command"] = self._cmd

    def _run(self, n: int) -> None:
        for _ in range(n):
            self.rng, k = self.jax.random.split(self.rng)
            act, _ = self._policy(self.state.obs, k)
            self.state = self._step(self.state, act)
            # The env resamples its command mid-episode during training. Pin it.
            self.state.info["command"] = self._cmd
            if float(self.state.done) > 0.5:
                self.fell = True
                return

    def settle(self, n: int = 25) -> None:
        self._set_command((0.0, 0.0, 0.0))
        self._run(n)

    def walk(self, dxy) -> None:
        """Try to move the torso by `dxy` metres in WORLD coordinates in 0.6 s.

        The move is converted to a body-frame joystick command with the robot's own
        heading, which is what an IMU would give a real robot. Yaw is used only to hold
        the heading, because the policy barely turns. Asking for more than the policy
        can deliver simply under-delivers; the world model has to learn that too.
        """
        dxy = np.asarray(dxy, dtype=np.float64)
        n = float(np.linalg.norm(dxy))
        if n > MAX_STEP_M:
            dxy = dxy / n * MAX_STEP_M
        v = dxy / (DECISION_STEPS * self.dt)
        c, s = np.cos(self.yaw), np.sin(self.yaw)
        vx, vy = c * v[0] + s * v[1], -s * v[0] + c * v[1]
        dyaw = (self.heading - self.yaw + np.pi) % (2 * np.pi) - np.pi
        self._set_command((float(np.clip(vx, -1, 1)), float(np.clip(vy, -0.5, 0.5)),
                           float(np.clip(1.5 * dyaw, -1, 1))))
        self._run(DECISION_STEPS)
        self.trail.append(self.pos.copy())

    # ── rendering ───────────────────────────────────────────────────────────────

    def _renderer(self, size: int):
        import mujoco

        if size not in self._renderers:
            self._renderers[size] = mujoco.Renderer(self.m, height=size, width=size)
        return self._renderers[size]

    def _decorate(self, scn, goal: str | None = None, trail: bool = False) -> None:
        import mujoco

        def add(gtype, size, pos, rgba, mat=None):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, gtype, np.asarray(size, np.float64), np.asarray(pos, np.float64),
                                np.eye(3).flatten() if mat is None else mat, np.asarray(rgba, np.float32))
            scn.ngeom += 1

        for name, (xy, rgb) in PADS.items():
            ring = 1.0 if name == goal else 0.0
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, [PAD_RADIUS, 0.02, 0], [xy[0], xy[1], 0.06], [*rgb, 1.0])
            if ring:
                add(mujoco.mjtGeom.mjGEOM_CYLINDER, [PAD_RADIUS + 0.12, 0.012, 0], [xy[0], xy[1], 0.05],
                    [1, 1, 1, 0.9])
        for k, rgb in enumerate(_PILLAR_COLOURS):
            a = 2 * np.pi * k / len(_PILLAR_COLOURS) + 0.3
            # 7 m out: any closer and the chase camera, which trails ~3.3 m behind the
            # robot, ends up with a pillar filling the frame whenever it nears an edge.
            add(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.3, 1.4, 0],
                [7.0 * np.cos(a), 7.0 * np.sin(a), 1.4], [*rgb, 1.0])
        # A faint grid of floor studs, 0.75 m apart. The rocky texture is dark and fine,
        # and these give the encoder (and the viewer) an unmistakable sense of motion.
        for gx, gy in self._stud_xy():
            add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.05, 0, 0], [gx, gy, 0.04], [0.85, 0.85, 0.8, 1.0])
        if trail:
            for p in self.trail[:: max(1, len(self.trail) // 60)]:
                add(mujoco.mjtGeom.mjGEOM_SPHERE, [0.06, 0, 0], [p[0], p[1], 0.12], [1.0, 0.95, 0.3, 1.0])

    def _stud_xy(self):
        if self.studs == "none":
            return []
        if self.studs == "grid":
            g = np.arange(-3.75, 3.76, 0.75)
            return [(x, y) for x in g for y in g]
        r = np.random.default_rng(7)
        return r.uniform(-4.5, 4.5, size=(260, 2)).tolist()

    def _sync(self) -> None:
        import mujoco

        self.d.qpos[:] = self.qpos
        mujoco.mj_forward(self.m, self.d)

    def render_chase_at(self, xy, size: int = 256) -> np.ndarray:
        """The chase view with the robot's current posture teleported to `xy`. Render
        only; for measuring the energy landscape without walking every grid point."""
        saved = self.state
        q = self.state.data.qpos.at[0:2].set(np.asarray(xy, dtype=np.float32))
        self.state = self.state.replace(data=self.state.data.replace(qpos=q))
        try:
            return self.render_chase(size)
        finally:
            self.state = saved

    def _follow(self, size: int, view, robot: bool = True) -> np.ndarray:
        import mujoco

        self._sync()
        r = self._renderer(size)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [*self.pos, 0.45]
        cam.distance, cam.azimuth, cam.elevation = view
        opt = mujoco.MjvOption()
        if not robot:
            opt.geomgroup[2] = 0      # the G1's visual meshes all live in group 2
        r.update_scene(self.d, cam, opt)
        self._decorate(r.scene)
        return r.render()

    def render_chase(self, size: int = 256) -> np.ndarray:
        """What the world model sees: a camera that follows the robot at a FIXED world
        angle. It never rotates with the robot, so when the robot moves the whole floor
        and every landmark slide across the frame. See MODEL_CAM for why it is so high."""
        return self._follow(size, self.chase, robot=self.model_sees_robot)

    def render_show(self, size: int = 384) -> np.ndarray:
        """For people only: a low chase camera where the robot is big enough to watch."""
        return self._follow(size, SHOW_CAM)

    def render_map(self, size: int = 384, goal: str | None = None) -> np.ndarray:
        """For people: the whole arena from above, with the path walked so far."""
        import mujoco

        self._sync()
        r = self._renderer(size)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = [0.0, 0.2, 0.0]
        cam.distance, cam.azimuth, cam.elevation = 8.2, 180.0, -70.0
        r.update_scene(self.d, cam)
        self._decorate(r.scene, goal=goal, trail=True)
        return r.render()

    def close(self) -> None:
        for r in self._renderers.values():
            r.close()
        self._renderers = {}


def pad_distance(pos: np.ndarray, pad: str) -> float:
    return float(np.linalg.norm(np.asarray(pos) - PADS[pad][0]))


def oracle_step(pos: np.ndarray, target: np.ndarray) -> np.ndarray:
    d = np.asarray(target) - np.asarray(pos)
    n = float(np.linalg.norm(d))
    return d if n <= MAX_STEP_M else d / n * MAX_STEP_M
