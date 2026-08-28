"""Terrain generators. The part of Isaac Lab that MuJoCo has no answer for.

Isaac Lab ships 23 sub-terrain generators and its default rough-terrain config
(`isaaclab/terrains/config/rough.py`) uses six of them: two stair pyramids, a grid of
boxes, uniform noise, and two slopes. Every rough-locomotion paper screenshot you have
seen is that one config. The other seventeen are sitting right there.

This file is the wider config, plus two narrow ones for when you want a single idea in
the frame rather than a mosaic.

── How the generator actually works ────────────────────────────────────────────
A TerrainGeneratorCfg lays out a `num_rows` x `num_cols` grid of square patches. With
`curriculum=True` the ROW index is DIFFICULTY: row 0 is difficulty 0.0, the last row is
1.0, and every sub-terrain reads that number to interpolate its own MISSING ranges (a
0.05-0.23 m step height becomes 0.05 m in row 0 and 0.23 m in the last row). Columns are
variety: which sub-terrain you land on, sampled by `proportion`.

So the robot does not "get put on hard terrain". It gets PROMOTED, one row at a time, by
the terrain curriculum in the environment's curriculum manager, and a policy that cannot
do stairs simply never sees the tall ones. That is why a parkour config with a pit in it
does not just kill every robot on iteration 1.

── Importable at module top level, deliberately ────────────────────────────────
Unlike `checks.py`, whose isaacsim imports MUST stay inside functions, `isaaclab.terrains`
imports fine with no SimulationApp anywhere. Verified on this box:

    $ISAACSIM_PYTHON_EXE -c "import isaaclab.terrains; import isaaclab_tasks"   # ok

The rule is narrower than "Isaac needs Kit first": `isaacsim.core.*` needs Kit, isaaclab's
CONFIG classes are ordinary dataclasses and do not. That distinction is what lets this
file, and `spark_envs.py` next to it, be plain top-level imports, which in turn is what
lets Flyte bundle them into the pod. See the note in pipeline.py.
"""

from __future__ import annotations

import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg

# ── The wide one ────────────────────────────────────────────────────────────────
#
# Thirteen sub-terrains. Proportions are relative weights over columns and are
# normalised by the generator, so they are written to sum to 1.0 only for readability.
#
# The six from NVIDIA's default are kept at the front, deliberately: this config is meant
# to be diffed against `ROUGH_TERRAINS_CFG`, and keeping the familiar ones makes the seven
# new ones obvious. Their proportions are dropped from 0.2/0.1 to make room.
#
# A note on the hard ones. Gaps, pits and stepping stones are HOLES: a robot that misses
# falls, terminates, and gets demoted a row. They are here because they are the terrains
# that make a locomotion policy learn to look where it is going (which is what the height
# scanner is for), and the curriculum means they cost you nothing until the policy is good
# enough to reach them. If a run stalls, these are the first proportions to cut.
SPARK_PARKOUR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,      # difficulty levels, 0.0 to 1.0
    num_cols=20,      # variety
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # ── the six NVIDIA ships ──
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.12,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.10, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.06, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.06, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),

        # ── the seven it does not ──
        # Trenches across the patch. 0.1 m at difficulty 0 is a crack in the pavement;
        # 0.6 m at difficulty 1 is a real jump for a Go2 and a stride for a G1.
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.08, gap_width_range=(0.1, 0.6), platform_width=3.0
        ),
        # A square depression with a platform in the middle. Tests stepping DOWN, which
        # is a different skill from stepping up and is where most policies faceplant.
        "pit": terrain_gen.MeshPitTerrainCfg(
            proportion=0.05, pit_depth_range=(0.05, 0.5), platform_width=4.0, double_pit=True
        ),
        # Two low bars to step over. Cheap, and it is the closest thing here to the
        # doorway thresholds and cable trays a real robot actually trips on.
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.05, rail_thickness_range=(0.05, 0.2), rail_height_range=(0.05, 0.3), platform_width=2.0
        ),
        # Discrete footholds over a void. The hardest thing in this config and the one
        # that most obviously needs the height scanner rather than blind proprioception.
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.08,
            stone_height_max=0.0,
            stone_width_range=(0.35, 1.0),
            stone_distance_range=(0.05, 0.35),
            holes_depth=-2.0,
            platform_width=2.0,
            border_width=0.25,
        ),
        # Smooth sinusoidal ground. No sharp edges, so it isolates balance from foothold
        # selection: the robot cannot see anything useful, it just has to stay upright.
        "wave": terrain_gen.HfWaveTerrainCfg(
            proportion=0.06, amplitude_range=(0.05, 0.3), num_waves=4, border_width=0.25
        ),
        # Scattered blocks to walk around or over. This is the "cluttered warehouse"
        # terrain and the one that looks most like somewhere a robot would be deployed.
        "obstacles": terrain_gen.HfDiscreteObstaclesTerrainCfg(
            proportion=0.06,
            obstacle_height_mode="choice",
            obstacle_width_range=(0.4, 1.2),
            obstacle_height_range=(0.05, 0.4),
            num_obstacles=12,
            platform_width=2.0,
            border_width=0.25,
        ),
        # Repeated boxes, laid out by the curriculum rather than sampled randomly, so the
        # spacing tightens as difficulty rises. Reads as a field of crates on camera.
        "crates": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=0.06,
            platform_width=2.0,
            # "box", not the inherited default. MeshRepeatedBoxesTerrainCfg declares
            # object_type = "{DIR}.utils:make_box", which configclass resolves to the
            # ResolvableString "isaaclab.terrains.trimesh.utils:make_box". But the
            # generator (mesh_terrains.py:762) tests `isinstance(cfg.object_type, str)`
            # FIRST, and ResolvableString is a str, so it takes the string branch and
            # looks up `make_isaaclab.terrains.trimesh.utils:make_box`, finds nothing,
            # and dies with "must be a string or a callable. Received: None". The short
            # spelling is the documented one and hits `make_box` in the module globals.
            object_type="box",
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=6, height=0.05, size=(0.6, 0.6), max_yx_angle=0.0, degrees=True
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=18, height=0.3, size=(0.6, 0.6), max_yx_angle=0.0, degrees=True
            ),
        ),
    },
)
"""Thirteen sub-terrains, curriculum on. The wide config: variety over focus."""


# ── The narrow ones ─────────────────────────────────────────────────────────────
#
# A 13-way mosaic is the right thing to TRAIN on and the wrong thing to FILM. At one
# robot per clip you get whichever patch it spawned on, and most of the time that is a
# patch of gentle noise. These put a single, legible idea in every frame.

SPARK_STAIRS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "up": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.5, step_height_range=(0.05, 0.23), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
        "down": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.5, step_height_range=(0.05, 0.23), step_width=0.3,
            platform_width=3.0, border_width=1.0, holes=False,
        ),
    },
)
"""Stairs only, up and down. The classic shot."""


SPARK_STONES_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=10,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        "stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.6, stone_height_max=0.0, stone_width_range=(0.35, 1.0),
            stone_distance_range=(0.05, 0.35), holes_depth=-2.0,
            platform_width=2.0, border_width=0.25,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.4, gap_width_range=(0.1, 0.6), platform_width=3.0
        ),
    },
)
"""Holes only. The terrain that forces the policy to use the height scanner."""


# ── The one built around a skill, not a look ────────────────────────────────────
#
# The three configs above are terrain variations on ONE task: track a commanded planar
# velocity. This one exists because that task cannot express a jump, and the first
# parkour run proved it: three hours of training, and the robot walked to the edge of
# every trench and stopped.
#
# The terrain half of the fix is here; the reward half is REWARD_PROFILES in
# spark_envs.py, and neither works without the other. Read them together.
#
# Three things are deliberately different from the `stones` config, which is the closest
# relative:
#
#   1. **The ladder is finer and it starts lower.** 12 rows instead of 10, and gaps of
#      0.05 m rather than 0.1 m at the bottom. Row 0 has to be something a policy that
#      has never left the ground can walk over by accident, because that accident is the
#      only way it ever discovers there is a gap to clear. The top row is 0.38 m, not the
#      0.6 m the parkour config asks for: 0.6 m is roughly a Go2 body length and is the
#      number the published parkour papers reach with a purpose-built reward, not with
#      velocity tracking plus two edited weights.
#
#      That ceiling was 0.45 m on the first run and came DOWN afterwards, which sounds
#      like giving up and is the opposite. A Go2 crosses a trench by STEPPING for as long
#      as the gap is narrower than its front-to-rear foot span, roughly 0.30 m; past that
#      a flight phase stops being optional. The first run parked at row 4.5, a 0.21 m gap,
#      still stepping, with flight time flat at the untrained value for all 4000
#      iterations. So the rows that decide this demo are the handful between 0.25 and
#      0.35 m, and spreading 12 rungs over 0.05-0.45 m put only three of them in that
#      band. Over 0.05-0.38 m the rung is 3 cm and five rows land exactly where the
#      behaviour has to change. Lowering the ceiling buys resolution where it is needed.
#
#   2. **Gaps dominate, but they do not monopolise.** 55% of columns, against `stones`'
#      40%. The other 45% is rails, boxes and noise: ground the robot can walk on. A
#      terrain that is nothing but holes trains a policy that cannot walk, and a policy
#      that cannot walk never builds up the forward speed a leap is made of.
#
#   3. **Dict order is load-bearing.** With `curriculum=True` the generator assigns
#      sub-terrains to columns by cumulative proportion in ITERATION ORDER
#      (terrain_generator.py:245), not randomly. Gaps first means gaps are columns 0-10,
#      which is why `record.py --terrain_col 5` reliably films a trench. Reorder this
#      dict and every hardcoded column number in the repo points somewhere else.
SPARK_LEAP_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    # 12 rows, so the gap grows ~3.6 cm per promotion instead of ~5.6 cm. The curriculum
    # can only ever be as smooth as its rungs, and a rung the policy cannot reach is
    # where a run stalls.
    num_rows=12,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    curriculum=True,
    sub_terrains={
        # A ring-shaped trench around a central platform (mesh_terrains.py:559), so the
        # robot spawns inside and every direction out is a jump.
        #
        # platform_width=2.5, not the 3.0 the parkour config uses. Promotion needs the
        # robot 4 m from its origin (size[0] / 2, curriculums.py:49) and the patch is
        # only 8 m wide, so that 4 m is the patch edge no matter what: shrinking the
        # platform does not make the target closer, it just moves the trench earlier and
        # leaves more solid ground to land on and walk out over. The cost is a shorter
        # run-up, which is why this is 2.5 and not smaller.
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.55, gap_width_range=(0.05, 0.38), platform_width=2.5
        ),
        # Two bars across the patch. A hop, not a leap, and that matters: this is the
        # rung between "lifts its feet" and "leaves the ground", and a curriculum with
        # nothing in that band tends to sit at the bottom of it.
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.20, rail_thickness_range=(0.05, 0.2),
            rail_height_range=(0.05, 0.35), platform_width=2.0
        ),
        # Solid ground with edges. Somewhere to actually walk, and the terrain that keeps
        # the gait from degenerating into a bounce.
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.10, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
    },
)
"""Gap-dominated, fine curriculum, low first rung. Pair with the `leap` reward profile."""

# Which column to point a camera at, per terrain, when you want the sub-terrain the
# config is NAMED for rather than whichever patch the robot happened to spawn on.
#
# These are indices into `num_cols` and they follow from the proportions above: the
# generator walks the sub_terrains dict in order and hands each one a contiguous block of
# columns sized by its normalised proportion. For `leap`, gaps take 55% of 20 columns, so
# 0-10 are trenches and 5 is comfortably inside that block. Change a proportion and this
# number has to move with it.
FILM_COLS: dict[str, int] = {
    "parkour": 12,
    "stairs": 5,
    "stones": 5,
    "leap": 5,
}


TERRAINS: dict[str, TerrainGeneratorCfg] = {
    "parkour": SPARK_PARKOUR_CFG,
    "stairs": SPARK_STAIRS_CFG,
    "stones": SPARK_STONES_CFG,
    "leap": SPARK_LEAP_CFG,
}
"""Name -> config, so a CLI flag or a Flyte task parameter can pick one."""
