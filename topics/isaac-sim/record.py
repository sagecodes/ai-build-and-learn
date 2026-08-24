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
    p.add_argument("--steps", type=int, default=300, help="env steps to record")
    p.add_argument("--out", default="clips", help="directory for the mp4s and summary.json")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--onboard", action="store_true", help="also record the robot's own RGB and depth")
    p.add_argument("--rendering_mode", default="quality", choices=["performance", "balanced", "quality"])
    # Where on the terrain grid to film. See _place() for why these matter so much.
    p.add_argument("--terrain_level", type=int, default=None,
                   help="difficulty ROW to spawn on (default: random, which usually means easy)")
    p.add_argument("--terrain_col", type=int, default=None,
                   help="sub-terrain COLUMN to spawn on (default: 0, the first in the dict)")
    return p


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


def _encode(frames, path: Path, fps: int) -> None:
    """Write RGB uint8 frames to H.264. PyAV, because it is what the image already has."""
    import av

    if not frames:
        return
    h, w = frames[0].shape[:2]
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
    # Visually lossless-ish. These clips are watched at full size in a Flyte report and
    # then base64'd into HTML, so the size/quality knob is worth setting explicitly.
    stream.options = {"crf": "20", "preset": "medium"}
    for frame in frames:
        container.mux(stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")))
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
    checkpoint = Path(args.checkpoint) if args.checkpoint else newest_checkpoint(Path(args.logs), experiment)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    print(f"[record] {args.task} <- {checkpoint}", flush=True)

    scene = env.unwrapped.scene
    patch = _place(env, scene, args.terrain_level, args.terrain_col)

    robot = scene["robot"]
    chase: Camera = scene["chase_cam"]
    onboard: Camera | None = scene["onboard_cam"] if args.onboard else None

    def look(cam: Camera, eye: torch.Tensor, target: torch.Tensor) -> None:
        cam.set_world_poses_from_view(eye, target)

    def yaw_of(quat: torch.Tensor) -> torch.Tensor:
        """Heading from a wxyz quaternion. Only z-rotation matters for a follow cam."""
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    chase_frames: list = []
    onboard_frames: list = []
    depth_frames: list = []
    scans: list = []

    # get_observations() returns a single TensorDict in this rsl_rl; older tutorials
    # unpack (obs, extras) and fail with "not enough values to unpack".
    obs = env.get_observations()
    with torch.inference_mode():
        for _ in range(args.steps):
            pos = robot.data.root_pos_w
            yaw = yaw_of(robot.data.root_quat_w)

            # Behind and to the side, in the robot's own frame, so the camera swings with
            # it instead of ending up nose-on when the robot turns around.
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
            # `root_pos_w.device` is an isaacsim Device wrapper, not a torch.device, so
            # `torch.tensor(..., device=pos.device)` dies with
            #   TypeError: argument 'device' must be torch.device, not Device
            target = pos.clone()
            target[:, 2] += 0.1
            look(chase, eye, target)

            if onboard is not None:
                head = torch.stack(
                    [pos[:, 0] + 0.35 * torch.cos(yaw), pos[:, 1] + 0.35 * torch.sin(yaw), pos[:, 2] + 0.05], dim=1
                )
                # Aim slightly down: a walking robot cares about the next two footholds,
                # not the horizon.
                ahead = torch.stack(
                    [head[:, 0] + 2.0 * torch.cos(yaw), head[:, 1] + 2.0 * torch.sin(yaw), head[:, 2] - 0.8], dim=1
                )
                look(onboard, head, ahead)

            actions = policy(obs)
            obs, _, _, _ = env.step(actions)

            chase_frames.append(chase.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
            if onboard is not None:
                onboard_frames.append(onboard.data.output["rgb"][0, ..., :3].cpu().numpy().astype(np.uint8))
                depth_frames.append(onboard.data.output["distance_to_image_plane"][0].cpu().numpy())

            # The height scanner already exists on every rough locomotion env: it is the
            # terrain observation the policy is actually conditioned on. Recording it
            # lets the report show what the policy SEES next to what the camera sees.
            if "height_scanner" in scene.sensors:
                hs = scene["height_scanner"]
                scans.append((hs.data.pos_w[0, 2] - hs.data.ray_hits_w[0, :, 2]).cpu().numpy())

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    _encode(chase_frames, out / "chase.mp4", args.fps)

    summary = {
        "task": args.task,
        "checkpoint": str(checkpoint),
        "steps": args.steps,
        "frames": len(chase_frames),
        "patch": patch,
        "clips": {"chase": str(out / "chase.mp4")},
    }

    if onboard_frames:
        _encode(onboard_frames, out / "onboard.mp4", args.fps)
        summary["clips"]["onboard"] = str(out / "onboard.mp4")
        d = np.stack(depth_frames)
        # Isaac hands depth back as (H, W, 1), so the stack is (N, H, W, 1) and every
        # colourised frame comes out ndim 4. PyAV rejects that with
        #   ValueError: Expected numpy array with ndim `3` but got `4`
        # Drop the trailing axis if it is there; older builds return (N, H, W) already.
        if d.ndim == 4:
            d = d[..., 0]
        _encode(_colourise_depth(d), out / "depth.mp4", args.fps)
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

    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)

    env.close()
    # Same rule as checks.py: everything must be printed and written BEFORE this call,
    # because Kit's fast shutdown calls os._exit() and never returns.
    simulation_app.close()


if __name__ == "__main__":
    main()
