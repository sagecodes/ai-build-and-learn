"""The body: NeuroMechFly v2 in MuJoCo, with compound eyes wired to the outside world.

flygym 2.1.0 (the March 2026 v2 rewrite) supplies the fly: a micro-CT body, 6 legs,
position-actuated joints, leg adhesion, and two compound eyes of 721 ommatidia each.
What this module adds is everything between that model and a brain:

  * an arena with something in it worth looking at,
  * a retina that turns rendered ommatidia into a number per eye,
  * a walking controller that accepts a two-number descending command,
  * a loop that advances exactly one brain-tick of physics at a time and leaves one
    video frame behind each time.

── Why the walking controller is not part of the brain ─────────────────────────
The connectome in this repo is a BRAIN connectome. The neurons that actually move legs
live in the ventral nerve cord, which is a different dataset (MANC) that this model
does not contain. The real interface between them is the descending neurons, about
1,300 cells that are the only route from head to body, and what they carry is closer to
"turn left" than to "extend the left front femur by 4 degrees".

So the split here is the animal's own: the connectome decides WHERE TO GO, and a
central pattern generator (flygym's `HybridTurningController`, a coupled-oscillator
tripod gait with stumbling and retraction reflexes) decides how to move six legs to get
there. Its input is `descending_signal`, shape (2,), a drive for the left and right
halves of the body. That two-number interface is not a simplification invented for this
demo: it is the standard abstraction in the fly-locomotion literature and it is exactly
what DNa01/DNa02 are believed to carry.

Measured on the DGX Spark: 2,900 physics steps/s at timestep 1e-4 (0.29x realtime),
2 ms per ommatidia readout once the retina and eye renderer are warm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

log = logging.getLogger(__name__)

# One brain tick. Eon Systems used 15 ms for the same brain-body coupling; at the fly's
# ~13 mm/s walking speed that is 0.2 mm of travel, a fifteenth of a body length, per
# decision. Physics runs at 1e-4 s, so a tick is 150 physics steps.
BRAIN_DT = 0.015
PHYSICS_DT = 1e-4

# Video: one frame per brain tick, exactly. flygym renders when
# `playback_speed / output_fps` seconds of sim time have elapsed, so setting
# playback_speed to BRAIN_DT * output_fps makes that interval equal to one tick and the
# body frames line up 1:1 with the brain frames. The resulting video plays at 0.375x
# realtime, which is about right for watching a fly place its feet.
OUTPUT_FPS = 25
PLAYBACK_SPEED = BRAIN_DT * OUTPUT_FPS


@dataclass(frozen=True)
class Scene:
    """The arena. One dark pillar on flat ground, and where the fly starts.

    Distances are millimetres, which is flygym's unit: the fly is about 2.5 mm long,
    so a pillar 10 mm away is four body lengths, far enough that reaching it is a real
    walk and near enough to subtend a visible angle on the compound eye.

    The pillar's SIZE is not cosmetic, it sets how much signal the loop has to work
    with. Measured here, peak left-right darkening difference across a bearing sweep:

        r=1.5 mm at 12 mm   subtends 14 deg    L-R up to 0.019
        r=3.0 mm at 10 mm   subtends 33 deg    L-R up to 0.086      <- default
        r=4.0 mm at 10 mm   subtends 44 deg    L-R up to 0.129

    The 14-degree post is a real fly-vision stimulus but it moves the descending
    populations by less than their per-tick noise, so the fly cannot act on it. 33
    degrees is in the range used in real object-fixation experiments and produces a
    differential the brain can actually resolve.
    """

    object_xy: tuple[float, float] = (12.0, 0.0)
    object_radius: float = 3.0
    object_height: float = 8.0
    object_rgba: tuple[float, float, float, float] = (0.05, 0.05, 0.05, 1.0)
    object_type: str = "cylinder"          # cylinder | sphere | box | capsule
    # What is ON the object, tasted when the fly arrives: "none", "sugar" or "bitter".
    # The fly cannot see the difference -- both look like the same dark shape -- which
    # is the point: the valence has to come from the taste circuit, not the eyes.
    food: str = "none"
    taste_margin: float = 2.5              # mm from the surface at which the tarsi taste
    spawn_xy: tuple[float, float] = (0.0, 0.0)
    spawn_z: float = 0.5
    # Heading in degrees, counter-clockwise from +x. Offsetting the fly's initial
    # heading from the pillar is the whole experiment: a fly spawned pointing straight
    # at the target reaches it by walking forward, which proves nothing.
    #
    # The size of the offset decides whether the CONTROLS can pass by accident. A fly
    # that never turns misses the pillar by `distance * sin(offset)`, so at the first
    # geometry tried here (10 mm, 35 degrees) a straight walker passes within 5.4 mm
    # of the centre, and the arrival threshold is 5.0 mm: the no-brain control came
    # within 5.5 mm having done nothing at all. At 12 mm and 60 degrees a straight
    # walker misses by 10.4 mm, twice the threshold, so arriving requires turning.
    spawn_heading_deg: float = 60.0
    with_object: bool = True

    # ── The object can move, which is what makes a swat a swat ──────────────────
    #
    # "loom" flies the object at the fly on a ballistic line aimed at wherever the
    # fly was when the approach started, the way a swat is aimed. "recede" runs the
    # same path backwards, starting at contact range and retreating. The pair is the
    # control that matters: a looming and a receding disc sweep the SAME sequence of
    # retinal sizes, in opposite time order, so any response that is about approach
    # rather than about size has to differ between them.
    #
    # Measured on this box with a 1.8 mm sphere closing from 18 mm at 22.7 mm/s, total
    # darkening across both eyes:
    #
    #     distance    18.0   15.3   11.2    7.1    4.4    3.0 mm
    #     darkening  0.015  0.025  0.042  0.079  0.168  0.287
    #
    # a 19-fold swing, against the 0.075 the stationary fixation pillar produces. A
    # looming object is not a subtle stimulus.
    motion: str = "none"                   # none | loom | recede
    motion_speed: float = 22.0             # mm/s along the approach line
    motion_start_distance: float = 18.0    # mm from the fly when the approach begins
    motion_stop_distance: float = 2.5      # mm: closer than this the sphere clips
                                           # through the eyes and the signal collapses
    motion_delay_ticks: int = 25           # quiet ticks before the swat, for a baseline
    motion_azimuth_deg: float = 0.0        # where it comes from, relative to the fly's
                                           # heading at the moment the approach starts
    motion_elevation_deg: float = 12.0     # and from how far above

    # ── The tethered drum ───────────────────────────────────────────────────────
    #
    # "drum" swaps the ground arena for flygym's TetheredWorld, which holds the fly's
    # body rigidly in place while its legs keep walking, and rings it with vertical
    # bars that rotate around it. This is the oldest experiment in fly vision: a
    # tethered fly in a rotating striped drum turns with the drum, and the strength of
    # that optomotor response is the classic readout of the motion pathway.
    #
    # `drum_bars` is the whole experimental design in one number:
    #
    #   12 bars   a full drum. Rotating it changes almost nothing about how much of
    #             each eye is dark, so it is a pure MOTION stimulus.
    #    1 bar    a single stripe sweeping around, which changes which eye is dark as
    #             it goes, so it is a POSITION stimulus.
    #
    # This repo's retina reads one number per eye, the fraction of ommatidia darker
    # than an empty arena, which is motion-blind by construction. So the one-bar
    # condition should work and the twelve-bar one should not, and the point of running
    # both is to find out whether the failure is the encoder's or the brain's.
    arena: str = "ground"                  # ground | drum
    drum_bars: int = 12
    drum_radius: float = 12.0              # mm from the fly to the bars
    drum_bar_radius: float = 0.9           # mm, each bar's own thickness
    drum_height: float = 10.0
    drum_speed_deg_s: float = 0.0          # + is counter-clockwise seen from above
    drum_start_deg: float = 0.0

    def spawn_quat(self) -> np.ndarray:
        half = np.deg2rad(self.spawn_heading_deg) / 2
        return np.array([np.cos(half), 0.0, 0.0, np.sin(half)])


@dataclass
class Retina:
    """Turns two rendered eye images into one number per eye: how dark is it.

    The signal this demo needs is "is there a dark object, and on which side", and the
    naive version of that (mean intensity per eye) does not work. Measured on this box
    in an EMPTY arena, the two eyes do not agree: 0.4563 of left ommatidia fall below
    the dark threshold versus 0.4674 of right ones. The ground plane fills the lower
    half of both eyes and the two views of it are not identical, so a raw left-right
    comparison reports an object at all times, in a fixed direction, with nothing there.

    The fix is to measure that empty-world view once and compare against it. Then the
    same azimuth sweep is clean and signed correctly at every angle:

        object azimuth   -90   -60   -30   -15    0   +15   +30   +60   +90
        left  eye       .000  .000  .000 +.003 +.044 +.028 +.025 +.015 +.019
        right eye      +.017 +.022 +.021 +.029 +.046 +.003  .000  .000  .000

    (Fractions of ommatidia darkened relative to the empty arena. +y is the fly's left,
    so positive azimuth means the object is on the left, and the left eye is the one
    that sees it. It is also visible in both eyes straight ahead, as it should be: the
    fly's binocular overlap is real.)
    """

    reference: np.ndarray            # (2, n_ommatidia), the empty-world view
    threshold: float = 0.12          # below this an ommatidium counts as dark
    adapt: float = 0.0               # >0 lets the reference drift toward what is seen
    # "absolute" counts facets that fall below `threshold` when the empty arena did not.
    # "contrast" counts facets that merely got DARKER than the empty arena by
    # `contrast_drop`, whatever their absolute level.
    #
    # The difference decides what can be seen at all. A dark pillar crosses the absolute
    # threshold easily. Another FLY does not: measured, an amber fly against this
    # arena's floor darkens 0 of 721 facets by the absolute test at every distance
    # tried, while by the contrast test it darkens 4 facets at 4 mm, 3 at 6 mm and 2 at
    # 9 mm. So a conspecific is not an invisible object in this arena, it is a very
    # nearly invisible one, and which of those two it is depends on this setting.
    mode: str = "absolute"
    contrast_drop: float = 0.05

    def __call__(self, ommatidia: np.ndarray) -> np.ndarray:
        """(2, n_ommatidia, 2) raw readout -> (2,) darkening per eye, in [0, 1]."""
        intensity = ommatidia.mean(axis=2)
        if self.mode == "contrast":
            darker = (self.reference - intensity) > self.contrast_drop
        else:
            darker = (intensity < self.threshold) & (self.reference >= self.threshold)
        drive = darker.mean(axis=1)
        if self.adapt > 0:
            # Photoreceptor adaptation. Off by default: a perfectly fixating fly stops
            # generating retinal change and the object fades out from under it, which
            # is a real property of real eyes and an annoying one in a 3-second clip.
            self.reference = (1 - self.adapt) * self.reference + self.adapt * intensity
        return drive


class FlyWorld:
    """A fly in an arena, advanced one brain-tick at a time.

    Usage:

        world = FlyWorld(Scene())
        for tick in range(200):
            eyes = world.look()                  # (2,) left/right darkening
            world.step(descending_signal)        # 150 physics steps + 1 video frame
    """

    def __init__(
        self,
        scene: Scene = Scene(),
        camera_res: tuple[int, int] = (320, 448),
        seed: int = 0,
        retina_adapt: float = 0.0,
        retina_mode: str = "absolute",
    ) -> None:
        from flygym.simulation import Simulation
        from flygym_demo.complex_terrain import (
            HybridTurningController,
            get_default_locomotion_dof_order,
        )

        self.scene = scene
        self._sim = self._build(scene, camera_res)
        self._ctrl = HybridTurningController(
            timestep=PHYSICS_DT,
            output_dof_order=get_default_locomotion_dof_order(),
        )
        self._ctrl.reset(seed=seed)
        self._steps_per_tick = int(round(BRAIN_DT / PHYSICS_DT))
        self.retina = Retina(
            reference=self._empty_world_reference(scene, camera_res),
            adapt=retina_adapt,
            mode=retina_mode,
        )
        self._retina_viz = None
        self.trajectory: list[tuple[float, float, float]] = []
        # The object's position is held HERE, not on the frozen Scene, because it can
        # move. Everything that asks "how far away is it" reads this.
        self.object_xyz = np.array(
            [scene.object_xy[0], scene.object_xy[1], self._object_z(scene)], float
        )
        self._tick = 0
        self._approach_from: np.ndarray | None = None
        self._approach_to: np.ndarray | None = None
        if scene.motion in ("loom", "recede"):
            # Fix the approach line NOW, from the spawn pose, and park the object at its
            # launch point. It is therefore visible and motionless through the baseline
            # ticks, worth about 0.015 of darkening, and the ONLY thing that changes
            # when the approach begins is that it starts moving. That is the experiment:
            # any response has to be to the motion, not to the object's existence.
            self._arm_approach()
        self._record_pose()

    # ── Construction ────────────────────────────────────────────────────────────

    @staticmethod
    def _object_z(scene: Scene) -> float:
        """Height of the object's CENTRE above the ground, in mm.

        A sphere sits on the ground at its own radius; a cylinder or box is specified
        by half-height, so its centre is half its height up. Getting this wrong buries
        the food or floats it.
        """
        if scene.object_type == "sphere":
            return scene.object_radius
        return scene.object_height / 2

    @staticmethod
    def _bar_angles(scene: Scene, elapsed_s: float) -> np.ndarray:
        """Where each drum bar is, in radians, after `elapsed_s` of rotation."""
        base = np.deg2rad(scene.drum_start_deg) + np.deg2rad(scene.drum_speed_deg_s) * elapsed_s
        return base + np.arange(scene.drum_bars) * 2 * np.pi / max(scene.drum_bars, 1)

    @staticmethod
    def _make_world(scene: Scene, with_object: bool):
        from flygym.compose import FlatGroundWorld
        from flygym.utils.mjcf import GEOM_TYPES
        from flygym_demo.complex_terrain import make_locomotion_fly

        if scene.arena == "drum":
            from flygym.compose import TetheredWorld

            # TetheredWorld holds the fly's body rigidly (as a mocap body) while its
            # legs keep moving, which is exactly a tethered preparation: the animal
            # walks and goes nowhere, and its intended turn is readable only from the
            # command, not from the trajectory. There is no ground in this world, which
            # is fine and is also why the empty-arena retinal reference is mostly sky.
            world = TetheredWorld()
            if with_object:
                for i, theta in enumerate(FlyWorld._bar_angles(scene, 0.0)):
                    world.mjcf_root.worldbody.add_geom(
                        type=GEOM_TYPES["cylinder"],
                        name=f"bar{i}",
                        pos=(
                            scene.drum_radius * np.cos(theta),
                            scene.drum_radius * np.sin(theta),
                            scene.drum_height / 2,
                        ),
                        size=(scene.drum_bar_radius, scene.drum_height / 2, 0.0),
                        rgba=scene.object_rgba,
                        contype=0,
                        conaffinity=0,
                    )
            fly = make_locomotion_fly("nmf", colorize=True)
            fly.add_vision()
            return world, fly

        world = FlatGroundWorld()
        if with_object:
            # A sphere sits ON the ground at its own radius; a cylinder/box is given as
            # a half-height, so its centre is half its height up. Getting this wrong
            # buries the food or floats it.
            z = FlyWorld._object_z(scene)
            if scene.object_type == "sphere":
                size = (scene.object_radius, 0.0, 0.0)
            elif scene.object_type == "box":
                size = (scene.object_radius, scene.object_radius, scene.object_height / 2)
            else:
                size = (scene.object_radius, scene.object_height / 2, 0.0)
            world.mjcf_root.worldbody.add_geom(
                type=GEOM_TYPES[scene.object_type],
                name="target",
                pos=(scene.object_xy[0], scene.object_xy[1], z),
                size=size,
                rgba=scene.object_rgba,
                # contype/conaffinity 0: the pillar is a landmark, not an obstacle. A
                # fly that walks into a solid post stops dead and the clip ends with a
                # collision instead of an arrival.
                contype=0,
                conaffinity=0,
            )
        # colorize=True applies flygym's own per-segment materials and textures: an
        # amber body, translucent wings, red compound eyes. Purely cosmetic, costs
        # nothing at runtime, and makes the clip read as a fly rather than a mechanism.
        fly = make_locomotion_fly("nmf", colorize=True)
        fly.add_vision()
        return world, fly

    def _build(self, scene: Scene, camera_res: tuple[int, int]):
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D

        world, fly = self._make_world(scene, scene.with_object)
        self.camera = fly.add_tracking_camera()
        world.add_fly(
            fly,
            np.array([scene.spawn_xy[0], scene.spawn_xy[1], scene.spawn_z]),
            Rotation3D("quat", tuple(scene.spawn_quat())),
        )
        sim = Simulation(world, timestep=PHYSICS_DT)
        # The object is a geom on the WORLDBODY, so its pose lives in `mj_model.geom_pos`
        # and MuJoCo recomputes `geom_xpos` from it on every step. Writing that array is
        # therefore all it takes to move a landmark around at runtime, with no mocap
        # body, no free joint and no extra degrees of freedom in the physics.
        self._target_gid = -1
        self._bar_gids: list[int] = []
        if scene.with_object:
            import mujoco as mj

            if scene.arena == "drum":
                self._bar_gids = [
                    mj.mj_name2id(sim.mj_model, mj.mjtObj.mjOBJ_GEOM, f"bar{i}")
                    for i in range(scene.drum_bars)
                ]
            else:
                self._target_gid = mj.mj_name2id(
                    sim.mj_model, mj.mjtObj.mjOBJ_GEOM, "target"
                )
        self.renderer = sim.set_renderer(
            self.camera,
            camera_res=camera_res,
            playback_speed=PLAYBACK_SPEED,
            output_fps=OUTPUT_FPS,
        )
        sim.reset()
        # Settle the legs onto the ground before the first decision. Without it the
        # first few ticks are the fly falling 0.5 mm, and the eyes see the ground
        # swinging past, which the brain faithfully reacts to.
        sim.warmup()
        return sim

    def _empty_world_reference(
        self, scene: Scene, camera_res: tuple[int, int]
    ) -> np.ndarray:
        """What each eye sees with the pillar removed. See `Retina` for why.

        Built as a throwaway second simulation, which costs about two seconds. The
        alternative, hard-coding the numbers, would silently rot the moment anyone
        changes the ground texture, the lighting or the fly's spawn height.
        """
        from flygym.simulation import Simulation
        from flygym.utils.math import Rotation3D

        world, fly = self._make_world(scene, with_object=False)
        world.add_fly(
            fly,
            np.array([scene.spawn_xy[0], scene.spawn_xy[1], scene.spawn_z]),
            Rotation3D("quat", tuple(scene.spawn_quat())),
        )
        sim = Simulation(world, timestep=PHYSICS_DT)
        sim.reset()
        sim.warmup()
        reference = sim.get_ommatidia_readouts("nmf").mean(axis=2)
        log.info(
            "empty-world eye reference: %.4f / %.4f mean intensity (L/R)",
            reference[0].mean(), reference[1].mean(),
        )
        return reference

    # ── The loop ────────────────────────────────────────────────────────────────

    def look(self) -> np.ndarray:
        """Render both compound eyes and return (2,) darkening, left eye first."""
        return self.retina(self._sim.get_ommatidia_readouts("nmf"))

    def raw_eyes(self) -> np.ndarray:
        """The two eye images as rendered, (2, 512, 450, 3) uint8. For the report."""
        return self._sim.get_raw_vision("nmf")

    def eye_views(self) -> tuple[np.ndarray, np.ndarray]:
        """Both compound eyes as hexagonal facet images, left first.

        This is the fly's actual visual world: 721 ommatidia per eye, each one facet,
        at the sampling the animal really has. It is also the honest picture of how
        little the loop has to work with -- the pillar is a handful of dark facets.
        """
        import video as VID

        if self._retina_viz is None:
            from flygym.vision.retina import Retina

            self._retina_viz = Retina()
        reading = self._sim.get_ommatidia_readouts("nmf")
        return (
            VID.eye_view(self._retina_viz.hex_pxls_to_human_readable(
                reading[0], color_8bit=True)),
            VID.eye_view(self._retina_viz.hex_pxls_to_human_readable(
                reading[1], color_8bit=True)),
        )

    def step(self, descending_signal: np.ndarray) -> None:
        """Advance one brain tick: 150 physics steps under this command, then a frame."""
        from flygym_demo.complex_terrain import (
            HybridControllerObservation,
            apply_locomotion_action,
        )

        signal = np.asarray(descending_signal, dtype=float)
        self._tick += 1
        self._advance_object()
        self._advance_drum()
        for _ in range(self._steps_per_tick):
            obs = HybridControllerObservation.from_sim(self._sim, "nmf")
            apply_locomotion_action(self._sim, "nmf", self._ctrl.step(signal, obs))
            self._sim.step()
        self._sim.render_as_needed()
        self._record_pose()

    # ── Where is it ─────────────────────────────────────────────────────────────

    def _record_pose(self) -> None:
        x, y, _ = self.position
        self.trajectory.append((x, y, self.heading_deg))
        self._prev_distance = getattr(self, "_distance_now", self.distance_to_object)
        self._distance_now = self.distance_to_object

    @property
    def position(self) -> np.ndarray:
        """Thorax position in mm."""
        return self._sim.get_body_positions("nmf")[0]

    @property
    def heading_deg(self) -> float:
        """Heading in degrees, counter-clockwise from +x, in [-180, 180]."""
        body_id = self._sim._internal_bodyids_by_fly["nmf"][0]
        forward = self._sim.mj_data.xmat[body_id].reshape(3, 3)[:, 0]
        return float(np.degrees(np.arctan2(forward[1], forward[0])))

    # ── A moving object ─────────────────────────────────────────────────────────

    @property
    def object_xy(self) -> tuple[float, float]:
        """Where the object is NOW, which is not where the scene put it."""
        return (float(self.object_xyz[0]), float(self.object_xyz[1]))

    def place_object(self, xyz) -> None:
        """Move the object. Takes effect on the next physics step or render."""
        self.object_xyz = np.asarray(xyz, float)
        if self._target_gid >= 0:
            self._sim.mj_model.geom_pos[self._target_gid] = self.object_xyz

    def _advance_drum(self) -> None:
        """Rotate the drum for this tick by writing the bars' world positions."""
        if not self._bar_gids:
            return
        elapsed = self._tick * BRAIN_DT
        for gid, theta in zip(self._bar_gids, self._bar_angles(self.scene, elapsed)):
            self._sim.mj_model.geom_pos[gid] = [
                self.scene.drum_radius * np.cos(theta),
                self.scene.drum_radius * np.sin(theta),
                self.scene.drum_height / 2,
            ]

    @property
    def stimulus_azimuth_deg(self) -> float:
        """Where the nearest drum bar is, relative to the fly's nose, in degrees.

        Positive means the bar is to the fly's left, matching `bearing_deg`. This is the
        drum's ground truth: it sweeps steadily when the drum turns, and it is what the
        descending command is being asked to track.
        """
        if not self._bar_gids:
            return float("nan")
        angles = np.degrees(self._bar_angles(self.scene, self._tick * BRAIN_DT))
        relative = (angles - self.heading_deg + 180) % 360 - 180
        return float(relative[np.argmin(np.abs(relative))])

    def _advance_object(self) -> None:
        """Fly the object along its approach line for this tick.

        The line is fixed at the moment the approach begins, aimed at where the fly was
        standing then, so it is ballistic: a swat that has been launched does not steer
        after the fly. That is both realistic and necessary, because an object that
        homed in would be measuring the fly's evasion by making it impossible.
        """
        scene = self.scene
        if scene.motion not in ("loom", "recede") or self._target_gid < 0:
            return
        elapsed = self._tick - scene.motion_delay_ticks
        if elapsed < 0 or self._approach_from is None:
            return
        travelled = min(
            elapsed * BRAIN_DT * scene.motion_speed,
            float(np.linalg.norm(self._approach_to - self._approach_from)),
        )
        if scene.motion == "recede":
            # The same path, walked backwards from contact. Identical sequence of
            # retinal sizes, opposite time order.
            span = float(np.linalg.norm(self._approach_to - self._approach_from))
            travelled = span - travelled
        direction = self._approach_to - self._approach_from
        norm = float(np.linalg.norm(direction))
        if norm > 1e-9:
            self.place_object(self._approach_from + direction / norm * travelled)

    def _arm_approach(self) -> None:
        """Fix the approach line: from `motion_start_distance` out, in to contact."""
        scene = self.scene
        fly = self.position
        heading = np.deg2rad(self.heading_deg + scene.motion_azimuth_deg)
        elevation = np.deg2rad(scene.motion_elevation_deg)
        # Unit vector pointing from the fly out toward where the object comes from.
        away = np.array(
            [np.cos(heading) * np.cos(elevation),
             np.sin(heading) * np.cos(elevation),
             np.sin(elevation)],
            float,
        )
        target_z = self._object_z(scene)
        contact = np.array([fly[0], fly[1], target_z], float) + away * scene.motion_stop_distance
        start = np.array([fly[0], fly[1], target_z], float) + away * scene.motion_start_distance
        self._approach_from, self._approach_to = start, contact
        self.place_object(contact if scene.motion == "recede" else start)

    @property
    def time_to_contact(self) -> float:
        """Seconds until the object reaches the fly, or inf while it is not closing.

        Computed from the ACTUAL closing rate rather than from the object's progress
        along its line, because both ends move: the object flies in at a fixed speed and
        the fly walks. Using the object's path parameter alone reports the wrong time
        whenever the fly is doing anything, which is most of the run.

        Read from the physics, where the brain cannot get at it. This is the x-axis
        every looming result is plotted against.
        """
        if self.scene.motion not in ("loom", "recede") or self._approach_from is None:
            return float("nan")
        closing = (self._prev_distance - self.distance_to_object) / BRAIN_DT
        if closing <= 1e-6:
            return float("inf")
        return self.distance_to_object / closing

    @property
    def tasting(self) -> bool:
        """Is the fly close enough for its tarsi to be on the food?

        Real flies taste with their feet: gustatory receptor neurons sit in the tarsal
        bristles, which is why a fly lands on your drink before extending its proboscis.
        A distance threshold stands in for tarsal contact here, because the food is a
        non-colliding landmark rather than something the legs can stand on.
        """
        return self.distance_to_object <= self.scene.object_radius + self.scene.taste_margin

    @property
    def distance_to_object(self) -> float:
        if self._bar_gids:
            # In the drum there is no single object to be far from; the bars ring the
            # fly at a fixed radius. Reporting that radius keeps every downstream chart
            # and score well defined instead of measuring a phantom at the origin.
            return float(self.scene.drum_radius)
        dx = self.object_xyz[0] - self.position[0]
        dy = self.object_xyz[1] - self.position[1]
        return float(np.hypot(dx, dy))

    @property
    def bearing_deg(self) -> float:
        """Where the object is relative to where the fly is pointing.

        Positive means the object is to the fly's left. This is the error signal the
        whole loop is supposed to drive to zero, and it is computed from the physics,
        where neither the brain nor the controller can reach it.
        """
        if self._bar_gids:
            return self.stimulus_azimuth_deg
        dx = self.object_xyz[0] - self.position[0]
        dy = self.object_xyz[1] - self.position[1]
        error = np.degrees(np.arctan2(dy, dx)) - self.heading_deg
        return float((error + 180) % 360 - 180)

    @property
    def frames(self) -> list[np.ndarray]:
        """Body-camera frames, one per tick."""
        buffer = self.renderer.frames
        return buffer[next(iter(buffer))] if buffer else []

    def close(self) -> None:
        # flygym's Renderer.__del__ calls into EGL during interpreter teardown, which
        # raises a cosmetic but alarming EGLError after the context is already gone.
        # Closing explicitly, and swallowing it, keeps the task logs readable.
        try:
            self._sim.close()
        except Exception:  # noqa: BLE001
            pass
