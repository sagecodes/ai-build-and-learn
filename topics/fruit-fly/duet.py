"""Two flies, two brains, one arena.

Everything else in this repo is one animal alone with an object. This is two, each
driven by its own independent copy of the FlyWire connectome, walking in the same
physics and racing for the same pillar.

── What two flies can and cannot be to each other ──────────────────────────────
The first thing to measure, before building any of it, is whether one fly can SEE the
other. Measured here, fly A's 721 ommatidia per eye looking at fly B across an
otherwise empty arena:

    separation                    4 mm    6 mm    9 mm   12 mm
    facets darkened, absolute        0       0       0       0     of 721
    facets darkened, contrast        4       3       2       1     of 721

By the absolute test this repo's retina uses everywhere else, a fly is INVISIBLE at
every distance: an amber body against a bright floor never crosses the "this facet is
dark" threshold that a black pillar crosses easily. By a contrast test, which asks only
whether a facet got darker than it was in an empty arena, a fly is visible and tiny:
three or four facets, against the roughly 54 that the fixation pillar darkens and the
200-odd a looming sphere does.

So a conspecific is about an order of magnitude weaker than a stimulus this model
already struggles with, and the README's measurement applies: a signal that small moves
the descending populations by less than their own per-tick noise. Two connectome flies
will not court, chase or follow each other here, and this module does not pretend
otherwise. `retina_mode="contrast"` is offered so the claim can be checked rather than
asserted.

── What is real, then ──────────────────────────────────────────────────────────
They share a world. Both can see the pillar, both steer for it with their own brain,
and they are solid to each other: they collide, block each other's path, and occlude
each other's view of the target. That interaction is entirely physical and it is not
scripted anywhere, which makes "do two identical brains reach the same pillar the same
way" a question with a real answer.

Two networks of 138,639 neurons co-exist in one process, which costs about 6.4 GB and
runs at roughly half the speed of a single-fly run.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

import body as BODY

log = logging.getLogger(__name__)

# Where the two flies start, relative to the pillar. Far enough apart that neither
# begins inside the other, and offset in opposite directions so a straight walk from
# either misses and the arrival has to be steered, exactly as in the single-fly demo.
# 12 mm apart, mirrored about the line to the pillar. Measured at 7 mm they start
# already inside each other's personal space and spend 51 of 60 ticks crowded, so the
# run becomes a record of one collision rather than of two approaches; at 12 mm they
# converge on the target and meet part way, which is the interesting case.
DEFAULT_SPAWNS = (
    ((0.0, -6.0), +60.0),
    ((0.0, +6.0), -60.0),
)


@dataclass
class DuetWorld:
    """Two flies in one arena, advanced one brain-tick at a time.

    Mirrors `body.FlyWorld`'s interface, but every method takes which fly it means.
    The two flies are called "a" and "b" and those names are also their MuJoCo prefixes.

    Args:
        scene: The arena and the pillar, shared by both flies.
        spawns: ((x, y), heading_deg) per fly.
        marker_radius: Radius in mm of a dark sphere mounted on each fly's thorax. 0
            leaves the flies their own colour, which is the honest default and means
            they are nearly invisible to one another; a positive value is the
            simulated equivalent of painting a fly, so that "can they see each other"
            can be turned on and measured rather than argued about.
        retina_mode: "absolute" or "contrast". See `body.Retina`.
    """

    scene: BODY.Scene = field(default_factory=BODY.Scene)
    spawns: tuple = DEFAULT_SPAWNS
    camera_res: tuple[int, int] = (320, 448)
    seed: int = 0
    marker_radius: float = 0.0
    retina_mode: str = "contrast"

    names: tuple[str, str] = ("a", "b")
    trajectories: dict = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        from flygym_demo.complex_terrain import (
            HybridTurningController,
            get_default_locomotion_dof_order,
        )

        self._sim = self._build()
        self._controllers = {}
        for name in self.names:
            controller = HybridTurningController(
                timestep=BODY.PHYSICS_DT,
                output_dof_order=get_default_locomotion_dof_order(),
            )
            # Different seeds, or the two central pattern generators start in lockstep
            # and the flies step in perfect unison for the whole run, which looks like
            # a bug and is really just two identical oscillators started identically.
            controller.reset(seed=self.seed + self.names.index(name))
            self._controllers[name] = controller

        self._steps_per_tick = int(round(BODY.BRAIN_DT / BODY.PHYSICS_DT))
        self.retinas = {
            name: BODY.Retina(
                reference=self._reference(name),
                mode=self.retina_mode,
            )
            for name in self.names
        }
        self._retina_viz = None
        self.trajectories = {name: [] for name in self.names}
        self._tick = 0
        self._record_pose()

    # ── Construction ────────────────────────────────────────────────────────────

    def _make_world(self, with_object: bool, only: str | None = None):
        from flygym.compose import FlatGroundWorld
        from flygym.utils.math import Rotation3D
        from flygym.utils.mjcf import GEOM_TYPES
        from flygym_demo.complex_terrain import make_locomotion_fly

        scene = self.scene
        world = FlatGroundWorld()
        if with_object:
            world.mjcf_root.worldbody.add_geom(
                type=GEOM_TYPES[scene.object_type],
                name="target",
                pos=(scene.object_xy[0], scene.object_xy[1],
                     BODY.FlyWorld._object_z(scene)),
                size=(scene.object_radius, scene.object_height / 2, 0.0)
                if scene.object_type != "sphere"
                else (scene.object_radius, 0.0, 0.0),
                rgba=scene.object_rgba,
                contype=0,
                conaffinity=0,
            )
        for name, ((x, y), heading) in zip(self.names, self.spawns):
            if only is not None and name != only:
                continue
            fly = make_locomotion_fly(name, colorize=True)
            fly.add_vision()
            if self.marker_radius > 0:
                fly.bodyseg_to_mjcfbody[fly.root_segment].add_geom(
                    type=GEOM_TYPES["sphere"],
                    name=f"{name}_marker",
                    pos=(0.0, 0.0, self.marker_radius + 0.35),
                    size=(self.marker_radius, 0.0, 0.0),
                    rgba=(0.03, 0.03, 0.03, 1.0),
                    contype=0,
                    conaffinity=0,
                    group=0,
                )
            half = np.deg2rad(heading) / 2
            world.add_fly(
                fly,
                np.array([x, y, scene.spawn_z]),
                Rotation3D("quat", (np.cos(half), 0.0, 0.0, np.sin(half))),
                # flygym 2.1.0 names its ground-contact sensors per LEG and not per
                # fly, so adding a second fly with them on dies with
                # "repeated name 'ground_contact_lf_leg' in sensor". The walking
                # controller does not use those sensors: it reads contact forces
                # straight out of mj_data, so switching them off costs nothing here.
                add_ground_contact_sensors=False,
            )
        return world

    def _build(self):
        from flygym.simulation import Simulation

        world = self._make_world(with_object=self.scene.with_object)
        # One camera per fly, so the report can show either animal's point of view, and
        # the frames stay 1:1 with the ticks the way the single-fly demo guarantees.
        cameras = []
        for name in self.names:
            fly = world.fly_lookup[name]
            cameras.append(fly.add_tracking_camera(name=f"{name}_cam"))
        sim = Simulation(world, timestep=BODY.PHYSICS_DT)
        self.renderer = sim.set_renderer(
            cameras,
            camera_res=self.camera_res,
            playback_speed=BODY.PLAYBACK_SPEED,
            output_fps=BODY.OUTPUT_FPS,
        )
        sim.reset()
        sim.warmup()
        return sim

    def _reference(self, name: str) -> np.ndarray:
        """What this fly's eyes see with the pillar AND the other fly removed.

        Both have to go. Referencing against a world that still contains the other fly
        would define it as part of the background, which is precisely the thing being
        measured.
        """
        from flygym.simulation import Simulation

        world = self._make_world(with_object=False, only=name)
        fly = world.fly_lookup[name]
        camera = fly.add_tracking_camera(name=f"{name}_ref_cam")
        sim = Simulation(world, timestep=BODY.PHYSICS_DT)
        sim.set_renderer(camera, camera_res=self.camera_res,
                         playback_speed=BODY.PLAYBACK_SPEED, output_fps=BODY.OUTPUT_FPS)
        sim.reset()
        sim.warmup()
        reference = sim.get_ommatidia_readouts(name).mean(axis=2)
        try:
            sim.close()
        except Exception:  # noqa: BLE001
            pass
        log.info("%s empty-world reference: %.4f mean intensity", name, reference.mean())
        return reference

    # ── The loop ────────────────────────────────────────────────────────────────

    def look(self, name: str) -> np.ndarray:
        return self.retinas[name](self._sim.get_ommatidia_readouts(name))

    def eye_views(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        import video as VID

        if self._retina_viz is None:
            from flygym.vision.retina import Retina as FlyRetina

            self._retina_viz = FlyRetina()
        reading = self._sim.get_ommatidia_readouts(name)
        return (
            VID.eye_view(self._retina_viz.hex_pxls_to_human_readable(reading[0], color_8bit=True)),
            VID.eye_view(self._retina_viz.hex_pxls_to_human_readable(reading[1], color_8bit=True)),
        )

    def step(self, signals: dict[str, np.ndarray]) -> None:
        """Advance one brain tick with BOTH flies' commands applied together."""
        from flygym_demo.complex_terrain import (
            HybridControllerObservation,
            apply_locomotion_action,
        )

        self._tick += 1
        for _ in range(self._steps_per_tick):
            for name in self.names:
                obs = HybridControllerObservation.from_sim(self._sim, name)
                action = self._controllers[name].step(
                    np.asarray(signals[name], float), obs
                )
                apply_locomotion_action(self._sim, name, action)
            self._sim.step()
        self._sim.render_as_needed()
        self._record_pose()

    def _record_pose(self) -> None:
        for name in self.names:
            x, y, _ = self.position(name)
            self.trajectories[name].append((x, y, self.heading_deg(name)))

    # ── Where is everyone ───────────────────────────────────────────────────────

    def position(self, name: str) -> np.ndarray:
        return self._sim.get_body_positions(name)[0]

    def heading_deg(self, name: str) -> float:
        body_id = self._sim._internal_bodyids_by_fly[name][0]
        forward = self._sim.mj_data.xmat[body_id].reshape(3, 3)[:, 0]
        return float(np.degrees(np.arctan2(forward[1], forward[0])))

    def distance_to_object(self, name: str) -> float:
        x, y, _ = self.position(name)
        return float(np.hypot(self.scene.object_xy[0] - x, self.scene.object_xy[1] - y))

    def bearing_deg(self, name: str) -> float:
        x, y, _ = self.position(name)
        error = (
            np.degrees(np.arctan2(self.scene.object_xy[1] - y, self.scene.object_xy[0] - x))
            - self.heading_deg(name)
        )
        return float((error + 180) % 360 - 180)

    @property
    def separation(self) -> float:
        """How far apart the two flies are, in mm."""
        a, b = (self.position(n) for n in self.names)
        return float(np.hypot(a[0] - b[0], a[1] - b[1]))

    def frames(self, name: str) -> list[np.ndarray]:
        buffer = self.renderer.frames
        return buffer.get(f"{name}_cam", []) if buffer else []

    def close(self) -> None:
        try:
            self._sim.close()
        except Exception:  # noqa: BLE001
            pass
