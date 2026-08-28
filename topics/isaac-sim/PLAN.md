# Isaac Sim, episode 2: terrain, sensors, a robot zoo, and manipulation

Working plan. Episode 1 (the current README) proved Isaac Sim runs on the Spark bare metal
and in a Flyte pod, and got Anymal-C walking on flat ground with a clip in the report.
This is what comes next. Written down so it survives a disconnect.

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done and verified.

Where it stands: phases 0-2 are done and verified, phase 3 is built and in the report but
has only been filmed with a 2-iteration throwaway policy. What is left is GPU time (a real
training run), then the zoo and manipulation.

All of it runs in a Flyte pod, not just on the host: `flyte run pipeline.py parkour`
trains one of our registered tasks and puts the chase, onboard and depth clips in the
report. Verified end to end at `--iterations 3 --num_envs 64`, which returned
`filmed_on: gaps, difficulty: 1.0, clips: [chase, depth, onboard]` in 4.7 minutes.

---

## Decisions already made

Three tracks, chosen by Sage on 2026-08-23, plus the render fix folded in rather than
landed separately:

1. **Rough terrain + sensors** (primary). Custom generated terrain, height scanner and
   an onboard camera in the report.
2. **Robot zoo compare**. One terrain, several robots, clips side by side.
3. **Manipulation**. A second flavour: hand or contact-rich assembly.

---

## Verified facts (measured on this box, do not re-derive)

Isaac Lab **6.1.17** at `/home/sage/isaac/IsaacLab`, Isaac Sim 6.0.1 at
`/home/sage/isaac/IsaacSim`. 197 registered `Isaac-*` gym envs, 52 robot asset configs.

- **isaaclab config classes import BEFORE Kit boots.** Verified:
  `$ISAACSIM_PYTHON_EXE -c "import isaaclab.terrains; import isaaclab_tasks"` succeeds
  with no `SimulationApp`. So terrain configs and `gym.register` calls can live at module
  top level. Only `isaacsim.core.*` needs the app booted first (that is the `checks.py`
  rule, and it still holds there).
- **`--external_callback` is the supported hook for registering our own envs** into
  NVIDIA's `train.py` / `play.py` without forking them. Defined at
  `scripts/reinforcement_learning/rsl_rl/train.py:85`, resolved by
  `string_to_callable(name, separator=".")`, so the value is `module.attribute`, e.g.
  `spark_envs.register`. It is called before the hydra decorator and before the app
  launches. It should return a list of unconsumed argv tokens (or None).
- **Both `train.py` and `play.py` under `scripts/reinforcement_learning/rsl_rl/` now emit
  a DeprecationWarning** pointing at `./isaaclab.sh train --rl_library rsl_rl`. They still
  work. `isaaclab.sh` is a bash wrapper, so the scripts stay the better target from a pod.
- **The grain is path-tracer noise.** Isaac Sim 6.0 renders with the RT2 real-time path
  tracer and Isaac Lab defaults to the `balanced` preset. Presets live in
  `/home/sage/isaac/IsaacLab/apps/rendering_modes/{performance,balanced,quality}.kit`
  and `--rendering_mode` is a plain AppLauncher flag (`app_launcher.py:560`, default
  `"balanced"`).

  | setting | balanced (what we shipped) | quality |
  |---|---|---|
  | `rtx.rtpt.maxBounces` | 2 | 3 |
  | `rtx.post.dlss.execMode` | 1 Balanced | 2 Quality |
  | adaptive disocclusion sampling | off | on, spp 4 |
  | ambient occlusion | off | on |
  | `rtx.raytracing.subpixel.mode` | 0 | 1 |

  `RenderCfg` (`isaaclab/sim/simulation_cfg.py:23`) also exposes
  `antialiasing_mode` (`"DLAA"` renders at native res instead of DLSS-upscaling),
  `samples_per_pixel`, `enable_dl_denoiser`, `enable_ambient_occlusion`. The quality
  preset itself comments that `rtx.rtpt.splitGlass` / `splitClearcoat` are the knobs to
  reach for "only if noise is observed".
- **23 sub-terrain generators exist**; the shipped `ROUGH_TERRAINS_CFG`
  (`isaaclab/terrains/config/rough.py`) uses only 6.
  Height field: `HfRandomUniform`, `HfPyramidSloped`, `HfInvertedPyramidSloped`,
  `HfPyramidStairs`, `HfInvertedPyramidStairs`, `HfDiscreteObstacles`, `HfWave`,
  `HfSteppingStones`.
  Trimesh: `MeshPlane`, `MeshPyramidStairs`, `MeshInvertedPyramidStairs`,
  `MeshRandomGrid`, `MeshRails`, `MeshPit`, `MeshBox`, `MeshGap`, `MeshFloatingRing`,
  `MeshStar`, `MeshRepeatedObjects`, `MeshRepeatedPyramids`, `MeshRepeatedBoxes`,
  `MeshRepeatedCylinders`.
- **Locomotion tasks registered**: `Isaac-Velocity-{Flat,Rough}-` for `G1`, `H1`,
  `Unitree-A1`, `Unitree-Go1`, `Unitree-Go2`, `Anymal-B`, `Anymal-C`, `Anymal-D`,
  `Cassie`, `Digit`, plus `Isaac-Velocity-Flat-Spot-v0` (**Spot is flat only**, there is
  no registered rough Spot task) and `Isaac-Navigation-Flat-Anymal-C-v0`.
- **Sensors available**: `TiledCamera` (the fast one), `Camera`, `RayCaster`,
  `RayCasterCamera`, `MultiMeshRayCaster(+Camera)`, `ContactSensor`, `Imu`,
  `FrameTransformer`, `JointWrench`, `pva`, and a TacSL tactile sensor demo.
  `ANYMAL_LIDAR_CFG` exists as a ready-made lidar-equipped robot.
- **Newton is in this build** alongside PhysX (`SimulationCfg.use_newton_actuators`,
  `sim/utils/newton_model_utils.py`), and `scripts/sim2sim_transfer/config/` ships
  `newton_to_physx_{g1,h1,go2,anymal_d}.yaml`. Not in scope for this episode, kept here
  because it is the obvious episode 3.
- **Measured baseline from episode 1**: Anymal-C flat, 4096 envs, 1500 iterations = 27 min
  on the Spark, mean reward -2.45 -> 11.38. rsl_rl collects 24 steps/env/iteration.
- **num_envs sweep on the parkour task** (Anymal-C, 12 iterations each, measured
  2026-08-24 on the host). Rough terrain costs ~4x flat per iteration: 4.08 s/iter here
  against 1.08 s/iter for the flat baseline, which is the height-scanner raycasts, the
  200-patch terrain mesh and a 235-dim observation instead of ~48.

  | num_envs | s/iter | env-steps/s | 1500 iters | peak host RSS |
  |---|---|---|---|---|
  | 2048 | 2.48 | 19,819 | 62 min | 5.9 GB |
  | **4096** | **4.08** | **24,094** | **102 min** | **7.1 GB** |
  | 8192 | 7.80 | 25,206 | 195 min | 10.0 GB |

  **4096 is the knee.** Throughput saturates there: 8192 buys 4.6% more env-steps/sec for
  double the wall clock. It is also the batch size NVIDIA's rough Anymal-C PPO config is
  tuned for, so moving off it means retuning hyperparameters, not just waiting longer.

  Memory is NOT the constraint on this box, compute is. Peak host RSS at 8192 envs is
  10 GB against a 119 GB unified pool, so the `train_env` request of 64Gi has enormous
  headroom and there is no point capping envs for memory reasons.
- **`--steps` is 50 Hz env steps, NOT frames of video.** `velocity_env_cfg.py:363` sets
  `decimation=4` on `sim.dt=0.005`, so one env step is 20 ms of robot time:

  | steps | robot time |
  |---|---|
  | 90 | 1.8 s (the first snapshot default: every clip read as a robot giving up) |
  | 250 | 5 s (snapshots now) |
  | 600 | 12 s (the hero shot now) |
  | 1000 | 20 s, and the hard ceiling: `episode_length_s = 20.0` |

  The encode fps must match that control rate too. It was 30, so every clip also played
  at 0.6x, which flatters the gait. Now 50.
- **`Curriculum/terrain_levels` is printed to stdout** every iteration by rsl_rl, e.g.
  `Curriculum/terrain_levels: 3.4065`. It is the mean difficulty row the envs are on, and
  it is the number that says the PARKOUR worked rather than that the walking worked.
  train.py parses it into a second curve.

---

## Build order

### Phase 0. Render fix `[x]`
Superseded by something worse, then fixed properly. `--rendering_mode quality` works and
is now record.py's default, but grain turned out to be the least of it: the Kit VIEWPORT
capture that `play.py --video` uses does not render the robot at all on this box,
headless. Balanced and quality both produce a beautiful empty landscape.

Diagnosis is written up at the top of `record.py`. In short: a `Camera` SENSOR renders
the same robot perfectly, and teleporting it proves live poses reach the renderer, so the
fault is the viewport capture specifically. `play.py --disable_fabric` is a dead argument
(declared at `play.py:66`, never read); do not draw conclusions from it.

Fix: `record.py` films from its own `Camera` sensors instead, which also buys the onboard
views for free. Verified: robot in frame, 1280x720, chase cam tracking.

### Phase 1. Custom terrain `[x]`
`terrains.py`. Three configs: `SPARK_PARKOUR_CFG` (13 sub-terrains against NVIDIA's six,
curriculum on), plus `SPARK_STAIRS_CFG` and `SPARK_STONES_CFG` for when a clip needs one
legible idea rather than a mosaic.

One trap found: `MeshRepeatedBoxesTerrainCfg` needs `object_type="box"`, not the
inherited default. The default is a `ResolvableString`, which IS a `str`, so
`mesh_terrains.py:762` takes the string branch, looks up a nonsense name, and dies with
`must be a string or a callable. Received: None`.

### Phase 2. Register our own tasks `[x]`
`spark_envs.py`, `register()`, passed as `--external_callback spark_envs.register`.
Thirty tasks: three terrains x ten robots, each with a `-Play-v0`. Verified end to end by
three short training runs (Go2 parkour, Go2 stairs, Anymal-C parkour), each producing
checkpoints and an ONNX export.

Both traps in the file header were real. A third one surfaced later, in the Play config:
the stock `_PLAY` shrink (`num_rows=5, num_cols=5, curriculum=False`) destroys the two
coordinates worth filming by, so our Play configs deliberately keep the training grid.

### Phase 3. Sensors in the report `[~]`
The thing MJX cannot do, so it is the payoff shot. Built in `record.py`; in the report as
of the parkour task, pending a real training run to film.
- Chase `Camera`, onboard RGB and onboard depth, all three as mp4 in the report.
- Depth is colourised against the 2nd/98th percentile of the clip, not a fixed near/far.
  A fixed 0.2-10 m range renders every frame as one flat tan gradient, because a camera
  aimed 2 m ahead of a walking robot only ever sees about 1-4 m.
- Height scanner: recorded, and the last frame is drawn in the report as an 11x17 grid of
  cells. Column-to-sub-terrain is NOT dict order; it is cumulative proportion.
- Still to do: contact forces, and a `TiledCamera` instead of `Camera` if the render cost
  ever matters (it does not at one robot).

### Phase 3b. Mid-training replays in the live report `[~]`
The MuJoCo demo films the current policy at every eval boundary and drops the clip into
the live report; a reward curve says a number went up, footage says it went up for the
right reason. Isaac cannot copy the shape, because training is a child process and its
weights are not ours to reach into. What it does have is rsl_rl writing `model_*.pt`
every 50 iterations, which is a perfectly good stream of live policies.

- `record.py --serve` stays up and films whatever checkpoint is named on stdin, one JSON
  command per line. Kit boots ONCE, so a snapshot is `runner.load()` plus a short roll
  instead of the ~2 min a fresh process spends booting Kit and building the 200-patch
  terrain mesh. That difference is what makes filming every few hundred iterations
  affordable while training holds the same GPU.
- `Snapshotter` in train.py drives it: started lazily on the first request (so its Kit
  boot does not land on top of the training process's own), fed from a checkpoint scan
  on iteration boundaries, drained on a thread so a 30 s render never stalls the loop
  that is consuming rsl_rl's stdout. It skips a boundary if the previous clip is still
  rendering, so the report stays current instead of queueing stale clips.
- Checkpoints are filtered by AGE (5 s), not just mtime: the newest file is regularly a
  `torch.save` still in flight, and loading one raises deep in unpickling.
- Snapshots film a FIXED row (`snapshot_level`, default 4 of 10), which is the opposite
  of what the hero shot wants: the strip is only worth looking at if the ground stays
  the same and the policy is the only thing changing.
- Everything is best-effort. Failed load, dead daemon, empty clip: it logs and the run
  carries on. Nothing here is worth losing three hours of training over.

### Phase 4. Robot zoo `[ ]`
One terrain, N robots, one report, clips side by side.
- Candidates: G1 (continuity with the MuJoCo episode), Go2, Anymal-C, Digit, Cassie,
  H1. Spot only if we add a rough config ourselves, since only flat is registered.
- This is N training runs, so it is the GPU-time sink. Drive it serially: >3 concurrent
  Flyte runs deadlock on this box.

### Phase 5. Manipulation `[ ]`
Pick one, both are registered and need no downloads:
- `Isaac-Repose-Cube-Shadow-OpenAI-LSTM-Direct-v0` (the famous in-hand reorientation), or
- `Isaac-Factory-PegInsert-Direct-v0` / `NutThread` (contact-rich, sim-to-real flavoured).

---

## Constraints to respect

- **Child process, always.** Kit's shutdown cancels every asyncio task in the process,
  including Flyte's. See the long comment at the top of `pipeline.py`.
- **Top-level imports only** for anything that must reach the pod: Flyte bundles the
  modules the task module imports at import time. `terrains.py` and `spark_envs.py` must
  be imported at the top of `pipeline.py` even though the child process is the real user.
- **One GPU.** Orchestrator tasks stay CPU-only or they deadlock their own children.
- **Disk.** The training image is ~25 GB and `train_env` asks for 80Gi. A full devbox disk
  has evicted this cluster before.

## Files

```
existing                          new in this episode
────────                          ───────────────────
config.py    images + task envs   terrains.py    the terrain generator configs
pipeline.py  Flyte tasks          spark_envs.py  gym.register via --external_callback
train.py     rsl_rl driver        record.py      films the policy from Camera sensors
checks.py    the physics checks
```

`flyte run pipeline.py parkour` is the episode-2 entry point, next to `smoke` and `walk`.

## Two things that bit hard, worth not re-deriving

**Flyte bundling vs. the orchestrator image.** `flyte run` defaults to
`copy_style="loaded_modules"`: it bundles the files of modules the CLI actually imported.
The usual fix is a top-level import in `pipeline.py`, and it CANNOT be used here, because
`pipeline.py` is imported by the orchestrator too and the orchestrator runs on the plain
isaac-sim image, which has no Isaac Lab. `terrains.py` and `spark_envs.py` import
`isaaclab.terrains` at module level, so importing them there breaks every orchestrator
pod. They ride along via `train_env.include=(...)` in config.py instead, which is unioned
into whatever the copy style found.

**`flyte.report.log` APPENDS. Use `replace`.** A live-updating report has to call
`flyte.report.replace(...)` then `flyte.report.flush()`, which is what the MuJoCo demo
next door does throughout. `log()` appends to the main tab, so repainting a progress
chart every 25 iterations stacks ~60 complete copies of the report, videos included, and
the reader scrolls through all of them. `Tab.replace` exists too, so the rsl_rl log tail
lives in its own tab instead of pushing the clips off screen. Several different charts in
one tab is fine; the same chart sixty times is not.

**Echo child stdout, or the pod log is one line.** `_run_streaming` consumes rsl_rl's
stdout to parse the curves, which means `kubectl logs` shows NOTHING for the whole run
unless the parser also prints. A 100-minute training run that is only observable through
a report is a bad trade for one `log.info`.

**Pre-run GPU ritual on the Spark**, in order:
1. `nvidia-smi` reports `[N/A]` for memory on GB10. `torch.cuda.mem_get_info()` is the
   only honest read.
2. `free` actively misleads: it claimed 109 GB "available" while CUDA could see 15.7 GiB,
   because 94 GB was page cache that CUDA cannot use. `sync && echo 3 | sudo tee
   /proc/sys/vm/drop_caches` (needs a real TTY) took CUDA from 15.7 to 109.3 GiB.
3. Check rustfs RSS. It leaks, and it steals from the same unified pool.
4. Confirm no pod is still `Terminating` on the GPU, or the new one sits Unschedulable.

**Never leave `--branch main` in a Dockerfile.** The training image's tag is an md5 of
`Dockerfile.train`'s contents, so it rebuilds whenever that file changes, and the first
rebuild since August died 1.5 seconds in: upstream had added `tabs 4` to `isaaclab.sh`,
the container has no `ansi+tabs` terminfo entry, and `set -e` did the rest. Now pinned to
`6a7acb03`, the same commit as the bare-metal tree, so host and pod agree.
