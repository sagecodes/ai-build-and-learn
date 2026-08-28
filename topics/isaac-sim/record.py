"""Film a trained policy with our own cameras, because Isaac Lab's viewport capture drops the robot.

── Why this file exists instead of `play.py --video` ───────────────────────────
Isaac Lab's replay path wraps the env in `gym.wrappers.RecordVideo`, which calls
`env.render()`, which grabs the Kit VIEWPORT camera `/OmniverseKit_Persp`. On this box,
headless, that capture renders the terrain, the sky and the velocity-command arrow
markers, and DOES NOT RENDER THE ROBOT. Four clips, two robots (Go2 and Anymal-C), three
terrains, `--rendering_mode` balanced and quality: every one is a beautiful empty
landscape with a floating green arrow where the robot should be.

It is not the asset, the physics or Fabric:

  * A `Camera` SENSOR renders the same Anymal-C perfectly (`render_probe.py`), so
    articulations do render headless.
  * Teleporting that robot 5 m and moving the camera with it renders it at the new
    position, so live poses do reach the renderer. Fabric is delivering.
  * `play.py --disable_fabric` changes nothing, and grepping shows why: the flag is
    declared at `play.py:66` and then never read anywhere in the file. It is a dead
    argument. Do not use it to draw conclusions.

So the fault is specific to the viewport capture. Rather than keep bisecting Kit
settings, this records from `Camera` sensors, which is better anyway:

  * The framing follows the robot. The stock viewer is a FIXED camera at world
    (7.5, 7.5, 7.5) looking at the origin; on generated terrain the robot walks out of
    frame almost immediately, and even before that it is a handful of pixels.
  * We get the onboard views for free, which is the whole point of Isaac Sim over MJX.
    Same mechanism, different pose: RGB and depth from the robot's head.

Both cameras are positioned in WORLD SPACE from the robot's root pose every frame,
rather than parented to a body prim. That is deliberate: the base link is called `base`
on Anymal and the Unitrees, `torso_link` on G1 and H1, `pelvis` on Cassie, so a
prim-parented camera needs a per-robot lookup table that will rot. A world camera driven
from `root_pos_w` / `root_quat_w` works for every articulation in the zoo.

Run it as a script; it is spawned as a child process, never imported into Flyte's
interpreter. See the shutdown note at the top of pipeline.py.

`--serve` keeps it up and films whatever checkpoint is named on stdin, which is how the
report gets clips DURING a three-hour training run rather than only after it. See serve().
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# This script is always spawned as a child process, and never with this directory as
# CWD: rsl_rl writes its checkpoints relative to CWD, so the caller runs from a
# writable scratch dir instead. That makes `import spark_envs` below a coin flip
# depending on who spawned us. Putting our own directory on the path first removes the
# question, and means the Flyte task does not have to remember to set PYTHONPATH.
sys.path.insert(0, str(Path(__file__).resolve().parent))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Record a trained Isaac Lab policy")
    p.add_argument("--task", required=True, help="gym id, e.g. Spark-Parkour-Anymal-C-Play-v0")
    p.add_argument("--checkpoint", default=None, help="path to model_*.pt (default: newest for the task)")
    p.add_argument("--logs", default="logs/rsl_rl", help="root to search for checkpoints")
    # STEPS ARE 50 Hz, NOT FRAMES OF VIDEO. The locomotion envs run sim.dt=0.005 with
    # decimation=4 (velocity_env_cfg.py:363), so one env step is 20 ms of robot time and
    # the old default of 300 was a SIX SECOND clip. 600 is twelve seconds; the episode
    # limit is episode_length_s=20.0, i.e. 1000 steps, which is the real ceiling here.
    p.add_argument("--steps", type=int, default=600, help="env steps to record (50 per second of robot time)")
    p.add_argument("--out", default="clips", help="directory for the mp4s and summary.json")
    # 50, to match that control rate. At 30 the footage plays at 0.6x and every gait
    # looks more deliberate than it is, which is a flattering lie.
    p.add_argument("--fps", type=int, default=50)
    p.add_argument("--crf", type=int, default=24,
                   help="x264 quality, lower is better. The thumbnail strip uses a higher one")
    # 960x540, not 1280x720. The report renders the chase clip at max-width 760 px, so
    # 720p was already being downscaled in the browser: those extra pixels only ever
    # existed to be base64'd into the page. The onboard cameras are half this again.
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--height", type=int, default=540)
    p.add_argument("--onboard", action="store_true", help="also record the robot's own RGB and depth")
    p.add_argument("--rendering_mode", default="quality", choices=["performance", "balanced", "quality"])
    # Where on the terrain grid to film. See _place() for why these matter so much.
    p.add_argument("--terrain_level", type=int, default=None,
                   help="difficulty ROW to spawn on (default: random, which usually means easy)")
    p.add_argument("--terrain_col", type=int, default=None,
                   help="sub-terrain COLUMN to spawn on (default: 0, the first in the dict)")
    # The learning timeline. Cheap because Kit boots once for the whole strip.
    p.add_argument("--timeline", type=int, default=4,
                   help="how many EARLIER checkpoints to film as a progress strip (0 to skip)")
    p.add_argument("--timeline_steps", type=int, default=250,
                   help="env steps per timeline clip; shorter than the hero shot on purpose")
    # Serve mode. The other end is Snapshotter in train.py; see serve() for the protocol.
    p.add_argument("--serve", action="store_true",
                   help="stay up and film checkpoints named on stdin, one JSON command per line")
    return p


def _iter_of(ckpt: Path) -> int:
    """Iteration number out of `model_1500.pt`. -1 if it is not that shape."""
    stem = ckpt.stem
    return int(stem.split("_")[-1]) if stem.startswith("model_") and stem.split("_")[-1].isdigit() else -1


def earlier_checkpoints(final: Path, count: int) -> list[Path]:
    """`count` checkpoints spread across training, oldest first, excluding the final one.

    rsl_rl saves every 50 iterations, so a 1500-iteration run leaves ~30 files sitting in
    the log directory that nothing ever looks at. They are the only record of what the
    policy looked like WHILE it was learning, and filming a handful of them is what turns
    a clip of a robot into a clip of a robot getting better.

    Evenly spaced by iteration rather than by file index, and the first real checkpoint is
    always included: `model_0.pt` is the untrained policy, which is the most useful frame
    of the whole strip because it is the before picture.
    """
    if count <= 0:
        return []
    pool = sorted(
        (p for p in final.parent.glob("model_*.pt") if _iter_of(p) >= 0 and p != final),
        key=_iter_of,
    )
    if len(pool) <= count:
        return pool
    # Spread over the pool, always keeping the earliest.
    idx = sorted({round(i * (len(pool) - 1) / max(count - 1, 1)) for i in range(count)})
    return [pool[i] for i in idx]


def newest_checkpoint(logs: Path, experiment: str) -> Path:
    """Newest `model_*.pt` under logs/<experiment>/. rsl_rl timestamps every run directory."""
    candidates = sorted((logs / experiment).glob("*/model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f"no checkpoint under {logs / experiment}")
    return candidates[-1]


def _sub_terrain_name(gen_cfg, col: int, num_cols: int) -> str | None:
    """Which sub-terrain lives in a given column.

    NOT `list(sub_terrains)[col]`. The generator does not put one sub-terrain per
    column; it spreads them across the columns by PROPORTION, so a sub-terrain with
    proportion 0.12 of 20 columns occupies two or three of them and most dict entries
    never line up with their own index. This is the same arithmetic as
    `TerrainGenerator._generate_curriculum_terrains` (terrain_generator.py:243), copied
    rather than imported because it is four lines and it is buried in a private method.
    """
    import numpy as np

    subs = getattr(gen_cfg, "sub_terrains", None)
    if not subs:
        return None
    proportions = np.array([c.proportion for c in subs.values()], dtype=float)
    proportions /= proportions.sum()
    idx = int(np.min(np.where(col / num_cols + 0.001 < np.cumsum(proportions))[0]))
    return list(subs.keys())[idx]


def _scan_grid(pattern_cfg) -> list[int] | None:
    """Shape to reshape the flat height-scan into, as [rows, cols], or None.

    The scanner is a `GridPatternCfg`, and `grid_pattern` (patterns.py:45) builds it as
    `arange(-size/2, size/2, resolution)` on each axis, meshgridded with indexing="xy",
    then flattened. That makes the flat array (len(y), len(x)) once reshaped, i.e. rows
    are the LATERAL axis and columns run fore-aft. For the stock rough-locomotion scanner
    that is 1.0 m / 0.1 m = 11 rows by 1.6 m / 0.1 m = 17 columns, which is the 187 rays
    the sensor reports.

    Returns None for the "yx" ordering rather than guessing: a transposed picture of
    what the robot sees is worse than no picture.
    """
    if getattr(pattern_cfg, "ordering", "xy") != "xy":
        return None
    res = getattr(pattern_cfg, "resolution", None)
    size = getattr(pattern_cfg, "size", None)
    if not res or not size:
        return None
    return [int(round(size[1] / res)) + 1, int(round(size[0] / res)) + 1]


def _depth_range(d) -> list[float]:
    """The 2nd and 98th percentile of finite depth, in metres.

    A fixed near/far is what makes depth clips useless. The first attempt here mapped
    0.2 m to 10 m onto the ramp, but a camera aimed 2 m ahead of a walking robot sees
    almost nothing outside 0.5-3 m, so 90% of the ramp went unused and every frame came
    out as one flat tan gradient. Percentiles spend the whole ramp on the range the
    footage actually occupies, and they adapt when a taller robot or a deeper pit
    changes it.
    """
    import numpy as np

    finite = d[np.isfinite(d)]
    if finite.size == 0:
        return [0.2, 5.0]
    lo, hi = np.percentile(finite, [2.0, 98.0])
    # Never let the window collapse: a robot facing a wall has near-constant depth, and
    # a zero-width range would divide by zero and produce a solid colour.
    if hi - lo < 0.25:
        hi = lo + 0.25
    return [round(float(lo), 3), round(float(hi), 3)]


def _colourise_depth(d) -> list:
    """Metres -> a jet-style RGB ramp: near = red, far = blue, no-hit = black.

    Hand-rolled for the same reason as the reward curve SVG in train.py: the image has
    no matplotlib and this is three lines of arithmetic. The piecewise-linear triangles
    below are the standard jet approximation, which is the wrong colormap for
    quantitative work and the right one here, where the job is to make a 30 cm step
    visibly different from the ground it sits on at a glance.
    """
    import numpy as np

    lo, hi = _depth_range(d)
    sky = ~np.isfinite(d)  # a ray that hit nothing; kept out of the ramp entirely
    x = ((np.nan_to_num(d, posinf=hi, neginf=hi) - lo) / (hi - lo)).clip(0.0, 1.0)

    # Flip so near is the hot end, then squeeze into the vivid middle of the ramp: the
    # raw triangles bottom out at half brightness, so a robot's own feet would render a
    # muddy dark red.
    t = 0.12 + 0.76 * (1.0 - x)
    r = (1.5 - np.abs(4.0 * t - 3.0)).clip(0.0, 1.0)
    g = (1.5 - np.abs(4.0 * t - 2.0)).clip(0.0, 1.0)
    b = (1.5 - np.abs(4.0 * t - 1.0)).clip(0.0, 1.0)

    rgb = np.stack([r, g, b], axis=-1)
    rgb[sky] = 0.0
    return list((rgb * 255).astype(np.uint8))


def _place(env, scene, level: int | None, col: int | None) -> dict:
    """Pin which terrain patch the robot spawns on. Returns what it landed on.

    This is the difference between a clip of a robot on stepping stones and a clip of a
    robot on a slightly bumpy car park, and it is entirely non-obvious.

    `TerrainImporter._compute_env_origins_curriculum` (terrain_importer.py:335) runs
    whenever the terrain came from a generator, REGARDLESS of the generator's own
    `curriculum` flag, and it assigns each env a patch like this:

        terrain_levels = randint(0, max_init_level + 1)   # the ROW, i.e. difficulty
        terrain_types  = floor(arange(num_envs) / (num_envs / num_cols))   # the COLUMN

    With `--num_envs 1`, which is what filming uses, both of those degenerate badly:
    the column is always 0, so you always get whichever sub-terrain happens to be first
    in the dict, and the row is a uniform random draw over the whole difficulty range,
    so most takes land on an easy row. The first parkour clip filmed here was an
    Anymal-C on near-flat ground, on a 13-terrain parkour course.

    Overwriting the levels after the importer has run, then resetting, is the honest fix:
    the origins are read at reset to place the robot, so a reset is all it takes.
    """
    terrain = getattr(scene, "terrain", None)
    origins = getattr(terrain, "terrain_origins", None)
    if terrain is None or origins is None:
        # A flat-plane env (no generator) has no grid to pin. Not an error.
        return {}

    rows, cols = origins.shape[0], origins.shape[1]
    if level is not None:
        terrain.terrain_levels[:] = min(level, rows - 1)
    if col is not None:
        terrain.terrain_types[:] = min(col, cols - 1)
    terrain.env_origins[:] = origins[terrain.terrain_levels, terrain.terrain_types]

    # The robot is already standing somewhere else. Reset re-places it on the new origin.
    env.reset()

    lvl = int(terrain.terrain_levels[0])
    typ = int(terrain.terrain_types[0])
    patch = {
        "row": lvl,
        "col": typ,
        "rows": rows,
        "cols": cols,
        "difficulty": round(lvl / max(rows - 1, 1), 2),
        "sub_terrain": _sub_terrain_name(terrain.cfg.terrain_generator, typ, cols),
    }
    print(f"[record] filming patch {patch}", flush=True)
    return patch


def _t(x):
    """Unwrap Isaac's tensor wrapper if there is one.

    Sensor `.data` fields come back as a wrapper in some backends and a bare torch tensor
    in others, and the two spellings are mixed even inside Isaac Lab's own source: the
    locomotion rewards write `contact_sensor.data.last_air_time.torch[...]` while this file
    reads `robot.data.root_pos_w[:, 0]` directly two hundred lines up. Both work on their
    own object. Asking for `.torch` only when it exists is the only spelling that works on
    either.
    """
    return x.torch if hasattr(x, "torch") else x


def _flight_phases(grounded: list[bool], xy: list, dt: float) -> dict:
    """Turn a per-step contact trace into the two numbers that describe a jump.

    This is the measurement that answers the actual question. A reward curve going up says
    a number went up; the terrain-level curve says the curriculum promoted the robot; only
    this says the robot LEFT THE GROUND, and how far it got while it was off it.

    `grounded[i]` is "some part of the robot was touching something at step i". The
    contact sensor tracks `{ENV_REGEX_NS}/Robot/.*`, every body and not just the feet
    (velocity_env_cfg.py:79), so a stumble that lands on the knees reads as grounded and
    does not get scored as a graceful leap. A flight phase is a maximal run of False.

    Both numbers are deliberately about the LONGEST phase rather than the mean. Means over
    a whole clip are dominated by the trot, where all four feet are briefly airborne
    between footfalls for a few hundredths of a second; those show up here as a hundred
    tiny phases and drag any average down to something that describes walking. The
    interesting event in a leap clip is the single biggest one.

    `span_m` is straight-line horizontal distance covered between takeoff and touchdown.
    On a gap patch it is directly comparable with the trench width, which is what makes it
    the number worth putting in a report: "0.34 s of flight, 0.38 m covered" next to "the
    trench at row 7 is 0.28 m" is a claim anyone can check.
    """
    best_len, best_start = 0, -1
    run_len, run_start = 0, 0
    for i, on_ground in enumerate(grounded):
        if on_ground:
            run_len = 0
            continue
        if run_len == 0:
            run_start = i
        run_len += 1
        if run_len > best_len:
            best_len, best_start = run_len, run_start

    phases = 0
    prev = True
    for on_ground in grounded:
        if prev and not on_ground:
            phases += 1
        prev = on_ground

    span = 0.0
    if best_len > 0:
        # Clamped because the last flight of a clip can still be in the air on the final
        # frame, in which case there is no touchdown sample to measure to.
        end = min(best_start + best_len, len(xy) - 1)
        a, b = xy[best_start], xy[end]
        span = float(((b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2) ** 0.5)

    return {
        "longest_s": round(best_len * dt, 3),
        "longest_span_m": round(span, 3),
        "phases": phases,
        # Fraction of the clip spent with nothing touching the ground. A trotting Go2
        # sits around 0.1-0.2; a policy that has learned to bounce instead of walk shows
        # up here as a number that is far too high, which is the failure mode the
        # -0.05 lin_vel_z_l2 whisper in spark_envs.py exists to prevent.
        "airborne_frac": round(1.0 - (sum(grounded) / max(len(grounded), 1)), 3),
        "steps": len(grounded),
    }


def _encode(frames, path: Path, fps: int, crf: int = 20, scale: float = 1.0) -> None:
    """Write RGB uint8 frames to H.264. PyAV, because it is what the image already has.

    `scale` shrinks on the way out, for clips the report shows small anyway: the timeline
    strip renders at 300 px wide, so encoding it at the camera's full 1280x720 spends
    megabytes on pixels no one will ever see.
    """
    import av

    if not frames:
        return
    h, w = frames[0].shape[:2]
    if scale != 1.0:
        # x264 needs even dimensions for yuv420p.
        w, h = int(w * scale) // 2 * 2, int(h * scale) // 2 * 2
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    # ── crf is load-bearing here, far more than it looks ────────────────────────
    # These frames come out of a PATH TRACER, so every pixel carries sampling noise,
    # and noise is the one thing H.264 cannot compress: at crf 26 x264 faithfully
    # preserves the grain and a five-second 640x360 clip lands at 6.3 MB, which is
    # roughly 10 Mbps for a thumbnail. Measured on the same clip, re-encoded:
    #
    #     640x360 crf 26   5.84 MB      480x270 crf 30   0.14 MB
    #     640x360 crf 32   0.34 MB      480x270 crf 34   0.05 MB
    #
    # The cliff between 26 and 32 is x264 deciding the grain is not worth bits. For
    # anything that gets base64'd into a report on every repaint, be on the far side
    # of that cliff.
    stream.options = {"crf": str(crf), "preset": "medium"}
    for frame in frames:
        picture = av.VideoFrame.from_ndarray(frame, format="rgb24")
        if scale != 1.0:
            picture = picture.reformat(width=w, height=h, format="yuv420p")
        container.mux(stream.encode(picture))
    container.mux(stream.encode())
    container.close()


def main() -> None:
    args = build_parser().parse_args()

    # Kit first. Nothing from isaaclab.sim, isaacsim.core or the RL stack can be imported
    # before SimulationApp exists, which is why every import below is inside this function.
    from isaaclab.app import AppLauncher

    launcher_args = argparse.Namespace(
        headless=True,
        enable_cameras=True,  # without it the RTX renderer is never started and every frame is black
        rendering_mode=args.rendering_mode,
        device="cuda:0",
    )
    app_launcher = AppLauncher(launcher_args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import Camera, CameraCfg
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from isaaclab_tasks.utils import load_cfg_from_registry
    from isaaclab_tasks.utils.hydra import resolve_presets

    import spark_envs

    spark_envs.register()

    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")

    # Isaac Lab 6.1 ships configs containing PresetCfg placeholders: a field can hold
    # `preset(default=0.0, newton_mjwarp=0.01)` instead of a number, so the same config
    # serves PhysX and Newton. NVIDIA's scripts fold those away inside hydra, and this
    # script deliberately does not use hydra, so it has to fold them itself. Skipping
    # this fails deep inside the actuator with
    #   TypeError: Invalid type for parameter value: <class '...hydra._Preset'>
    # which reads like a corrupt config and is not.
    env_cfg = resolve_presets(env_cfg)
    agent_cfg = resolve_presets(agent_cfg)

    # The runner config in the repo is written for an older rsl_rl than the one the image
    # ships, and NVIDIA migrates it at runtime rather than editing every task's config.
    # Without this the runner dies with
    #   TypeError: MLPModel.__init__() got an unexpected keyword argument 'stochastic'
    import importlib.metadata as _md

    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, _md.version("rsl-rl-lib"))
    env_cfg.scene.num_envs = 1
    env_cfg.sim.device = "cuda:0"

    # Switch the terrain curriculum OFF for filming, or `--terrain_level` cannot hold.
    # `terrain_levels_vel` (locomotion/velocity/mdp/curriculums.py:27) runs on every
    # reset and promotes or demotes the env by comparing how far the robot is from its
    # env origin. Move the origin under a standing robot, as _place() does, and that
    # distance is suddenly the width of the terrain: the term reads it as a robot that
    # walked brilliantly, promotes it past the last row, and line 323 then bounces it to
    # a RANDOM row. Pinning row 4 and landing on row 0 is the symptom.
    #
    # Setting the term to None is the supported spelling; velocity_env_cfg.py:378 checks
    # for exactly this and turns the generator's own curriculum off to match.
    if getattr(getattr(env_cfg, "curriculum", None), "terrain_levels", None) is not None:
        env_cfg.curriculum.terrain_levels = None

    # ── The two cameras ─────────────────────────────────────────────────────────
    # Added to the scene CONFIG, not to the built scene: InteractiveScene walks
    # `cfg.__dict__` (interactive_scene.py:887) and spawns whatever it finds that is not
    # an InteractiveSceneCfg field, so an attribute assigned here is picked up. It also
    # sorts sensors after assets, so a camera can reference a robot prim that does not
    # exist yet at config time.
    common = dict(
        update_period=0.0,  # every render, not on a timer
        height=args.height,
        width=args.width,
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, clipping_range=(0.05, 1.0e5)),
    )
    env_cfg.scene.chase_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/chase_cam", data_types=["rgb"], **common
    )
    if args.onboard:
        env_cfg.scene.onboard_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/onboard_cam",
            # distance_to_image_plane is the depth the robot would actually get from a
            # stereo rig or a depth camera, which is the point of filming it at all.
            data_types=["rgb", "distance_to_image_plane"],
            **{**common, "width": args.width // 2, "height": args.height // 2},
        )

    env = gym.make(args.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=getattr(agent_cfg, "clip_actions", None))

    experiment = agent_cfg.experiment_name
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    policy = None

    # Serve mode resolves nothing up front: the checkpoints it films do not exist yet
    # when it boots, because the run that writes them is still on iteration 1.
    checkpoint: Path | None = None
    earlier: list[Path] = []
    if not args.serve:
        checkpoint = Path(args.checkpoint) if args.checkpoint else newest_checkpoint(Path(args.logs), experiment)
        earlier = earlier_checkpoints(checkpoint, args.timeline)
        print(f"[record] {args.task} <- {checkpoint}", flush=True)
        if earlier:
            print(f"[record] timeline: {[_iter_of(c) for c in earlier]} then final", flush=True)

    scene = env.unwrapped.scene

    robot = scene["robot"]
    chase: Camera = scene["chase_cam"]
    onboard: Camera | None = scene["onboard_cam"] if args.onboard else None

    def look(cam: Camera, eye: torch.Tensor, target: torch.Tensor) -> None:
        cam.set_world_poses_from_view(eye, target)

    def yaw_of(quat: torch.Tensor) -> torch.Tensor:
        """Heading from a wxyz quaternion. Only z-rotation matters for a follow cam."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def film(
        steps: int, want_onboard: bool,
        level: int | None = args.terrain_level, col: int | None = args.terrain_col,
    ) -> tuple[list, list, list, list, dict]:
        """Roll the CURRENTLY LOADED policy for `steps` and return the frames.

        Everything expensive is already built by the time this is called: Kit is up, the
        terrain mesh is generated, the cameras exist. Filming another checkpoint is then
        just `runner.load()` plus this loop, which is why the whole learning timeline
        costs one Kit boot instead of one per clip.

        `level` and `col` default to the command-line patch and are arguments only so
        that serve() can move the camera to a different patch between clips.
        """
        chase_frames: list = []
        onboard_frames: list = []
        depth_frames: list = []
        scans: list = []
        # Per-step contact trace, for _flight_phases below. Cheap: two scalars a step.
        grounded: list[bool] = []
        track_xy: list = []
        cam2 = onboard if want_onboard else None

        patch = _place(env, scene, level, col)
        # get_observations() returns a single TensorDict in this rsl_rl; older tutorials
        # unpack (obs, extras) and fail with "not enough values to unpack".
        obs = env.get_observations()
        with torch.inference_mode():
            for _ in range(steps):
                pos = robot.data.root_pos_w
                yaw = yaw_of(robot.data.root_quat_w)

                # Behind and to the side, in the robot's own frame, so the camera swings
                # with it instead of ending up nose-on when the robot turns around.
                back, side, up = 2.8, 1.4, 1.2
                eye = torch.stack(
                    [
                        pos[:, 0] - back * torch.cos(yaw) - side * torch.sin(yaw),
                        pos[:, 1] - back * torch.sin(yaw) + side * torch.cos(yaw),
                        pos[:, 2] + up,
                    ],
                    dim=1,
                )
                # Aim a touch above the root so the body sits on the horizon line rather
                # than dead centre. Built by cloning rather than adding a fresh tensor:
                # `root_pos_w.device` is an isaacsim Device wrapper, not a torch.device,
                # so `torch.tensor(..., device=pos.device)` dies with
                #   TypeError: argument 'device' must be torch.device, not Device
                target = pos.clone()
                target[:, 2] += 0.1
                look(chase, eye, target)

                if cam2 is not None:
                    head = torch.stack(
                        [pos[:, 0] + 0.35 * torch.cos(yaw), pos[:, 1] + 0.35 * torch.sin(yaw),
                         pos[:, 2] + 0.05], dim=1
                    )
                    # Aim slightly down: a walking robot cares about the next two
                    # footholds, not the horizon.
                    ahead = torch.stack(
                        [head[:, 0] + 2.0 * torch.cos(yaw), head[:, 1] + 2.0 * torch.sin(yaw),
                         head[:, 2] - 0.8], dim=1
                    )
                    look(cam2, head, ahead)

                actions = policy(obs)
                obs, _, _, _ = env.step(actions)

                chase_frames.append(chase.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
                if cam2 is not None:
                    onboard_frames.append(cam2.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
                    depth_frames.append(cam2.data.output["distance_to_image_plane"][0].cpu().numpy())

                # The height scanner already exists on every rough locomotion env: it is
                # the terrain observation the policy is actually conditioned on. Recording
                # it lets the report show what the policy SEES next to the camera view.
                if "height_scanner" in scene.sensors:
                    hs = scene["height_scanner"]
                    scans.append((hs.data.pos_w[0, 2] - hs.data.ray_hits_w[0, :, 2]).cpu().numpy())

                # Is ANY part of the robot touching anything right now?
                #
                # `current_air_time` is per tracked body and resets to 0 the instant that
                # body makes contact, so the minimum across bodies is 0 if and only if
                # something is down. Bodies that have not touched since the last reset
                # just accumulate, which is why this is a min and not a sum: the base
                # spends the whole clip "airborne" and would swamp any other reduction.
                # `current_air_time` is None unless the sensor was configured with
                # track_air_time=True. Every rough locomotion env sets it (the air-time
                # reward needs it), but record.py also films stock flat tasks, so the
                # attribute is checked rather than assumed.
                if "contact_forces" in scene.sensors:
                    air = _t(scene["contact_forces"].data.current_air_time)
                    if air is not None:
                        grounded.append(bool(air[0].min().item() <= 0.0))
                        track_xy.append(_t(robot.data.root_pos_w)[0, :2].tolist())

        # Carried on the patch dict rather than as a sixth return value: film() is called
        # from three places and none of them would use a new tuple slot, whereas all three
        # already forward `patch` straight into their JSON.
        if grounded:
            patch["flight"] = _flight_phases(grounded, track_xy, float(env.unwrapped.step_dt))

        return chase_frames, onboard_frames, depth_frames, scans, patch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Serve mode: film the policy WHILE it is being trained ───────────────────
    def serve() -> None:
        """Film on demand. One JSON command per stdin line, one JSON result per clip.

        The whole point is that Kit boots ONCE. A snapshot is then `runner.load()` plus
        a short roll, tens of seconds, against the two-and-a-bit minutes a fresh process
        spends booting Kit and generating the 200-patch terrain mesh. That difference is
        what makes filming every few hundred iterations affordable while a training run
        is using the same GPU.

        Commands are `{"checkpoint": path, "iteration": n, "steps": n, "level": n,
        "col": n}`, and `{"stop": true}` ends the loop. Results are
        `{"ok": true, "iteration": n, "clip": path, ...}` or `{"ok": false, "error": ...}`.
        The other end is Snapshotter in train.py.
        """
        nonlocal policy
        print("[record] serve: ready", flush=True)
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
            except ValueError:
                print(f"[record] serve: not JSON, ignoring: {line[:120]}", flush=True)
                continue
            if cmd.get("stop"):
                print("[record] serve: stopping", flush=True)
                break

            it = int(cmd.get("iteration", -1))
            try:
                # A checkpoint half-written by the training process raises here, and
                # that is the common failure: nothing in this loop is allowed to end the
                # daemon, because the next request a minute later will be fine.
                runner.load(str(cmd["checkpoint"]))
                policy = runner.get_inference_policy(device=env.unwrapped.device)
                frames, _, _, _, patch = film(
                    int(cmd.get("steps", args.steps)),
                    want_onboard=False,
                    level=cmd.get("level", args.terrain_level),
                    col=cmd.get("col", args.terrain_col),
                )
                clip = out / f"snap_{max(it, 0):06d}.mp4"
                _encode(frames, clip, args.fps, args.crf)
                result = {"ok": bool(frames), "iteration": it, "clip": str(clip),
                          "frames": len(frames), "row": patch.get("row"),
                          # The jump measurement for THIS checkpoint. It is what turns the
                          # snapshot strip from a set of thumbnails into a curve you can
                          # read: flight time per checkpoint, filmed on a fixed row.
                          "flight": patch.get("flight"),
                          # Reported because it is the number that decides whether the
                          # live report stays loadable. See _encode.
                          "kb": round(clip.stat().st_size / 1024) if clip.exists() else 0}
            except Exception as exc:  # noqa: BLE001 - deliberately everything
                result = {"ok": False, "iteration": it, "error": f"{type(exc).__name__}: {exc}"}
            print(json.dumps(result), flush=True)

    if args.serve:
        serve()
        env.close()
        simulation_app.close()
        return

    # ── The learning timeline ───────────────────────────────────────────────────
    # Earlier checkpoints, filmed oldest-first, so the report can show the gait being
    # learned rather than only its end state. Chase camera only and a shorter roll:
    # this is a thumbnail strip, not the hero shot.
    timeline = []
    for ckpt in earlier:
        it = _iter_of(ckpt)
        runner.load(str(ckpt))
        policy = runner.get_inference_policy(device=env.unwrapped.device)
        frames, _, _, _, _ = film(args.timeline_steps, want_onboard=False)
        clip = out / f"iter_{it:06d}.mp4"
        # Quarter size and a coarser crf: the report shows these at 300 px wide, so the
        # camera's full 1280x720 would be megabytes spent on pixels nobody sees.
        _encode(frames, clip, args.fps, max(args.crf, 30), scale=0.5)
        timeline.append({"iteration": it, "clip": str(clip), "frames": len(frames)})
        print(f"[record] timeline iter {it}: {len(frames)} frames -> {clip.name}", flush=True)

    # The hero shot last, so the env ends on the final policy.
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    chase_frames, onboard_frames, depth_frames, scans, patch = film(args.steps, args.onboard)
    _encode(chase_frames, out / "chase.mp4", args.fps, args.crf)

    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "steps": args.steps,
        "frames": len(chase_frames),
        "patch": patch,
        "timeline": timeline,
        "clips": {"chase": str(out / "chase.mp4")},
    }

    if onboard_frames:
        _encode(onboard_frames, out / "onboard.mp4", args.fps, args.crf)
        summary["clips"]["onboard"] = str(out / "onboard.mp4")
        d = np.stack(depth_frames)
        # Isaac hands depth back as (H, W, 1), so the stack is (N, H, W, 1) and every
        # colourised frame comes out ndim 4. PyAV rejects that with
        #   ValueError: Expected numpy array with ndim `3` but got `4`
        # Drop the trailing axis if it is there; older builds return (N, H, W) already.
        if d.ndim == 4:
            d = d[..., 0]
        _encode(_colourise_depth(d), out / "depth.mp4", args.fps, args.crf)
        summary["clips"]["depth"] = str(out / "depth.mp4")
        summary["depth_range_m"] = _depth_range(d)

    if scans:
        arr = np.nan_to_num(np.stack(scans))
        # Which single frame to put in the report. NOT the last one: the sub-terrain
        # platforms are 2-4 m wide and the scanner only reaches 1.6 m x 1.0 m, so a robot
        # standing anywhere near where it spawned reads perfectly flat, and a policy that
        # falls over early leaves you with 187 identical numbers. Measured on the first
        # run that recorded this: clearance 0.516 m to 0.516 m, a solid block of colour.
        #
        # The frame with the most spread is the one where the scanner is actually looking
        # at an edge, which is the only frame worth drawing. Its index goes in the summary
        # so the report can say when it happened rather than implying it is the end state.
        peak = int(np.argmax(arr.std(axis=1)))
        summary["height_scan"] = {
            "rays": int(arr.shape[1]),
            "grid": _scan_grid(scene["height_scanner"].cfg.pattern_cfg),
            "mean_clearance_m": round(float(arr.mean()), 4),
            "max_clearance_m": round(float(arr.max()), 4),
            "peak_frame": peak,
            "frames": int(arr.shape[0]),
            # 187 floats, so the report can draw the grid without loading the .npy.
            "peak": [round(float(v), 4) for v in arr[peak]],
        }
        np.save(out / "height_scan.npy", arr)

    # Sizes, on one line, because every one of these ends up base64'd into the report
    # and a run that quietly produced a 90 MB page should say so in its own log.
    summary["clip_kb"] = {
        name: round(Path(p).stat().st_size / 1024)
        for name, p in summary["clips"].items() if Path(p).exists()
    }
    print(f"[record] clip sizes KB: {summary['clip_kb']}", flush=True)

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    env.close()
    # Same rule as checks.py: everything must be printed and written BEFORE this call,
    # because Kit's fast shutdown calls os._exit() and never returns.
    simulation_app.close()


if __name__ == "__main__":
    main()
