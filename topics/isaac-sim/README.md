# Isaac Sim: teaching a robot dog to cross ground it cannot walk on

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This one is about [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim), the GPU-accelerated robotics simulator built on Omniverse, and [Isaac Lab](https://github.com/isaac-sim/IsaacLab), the RL framework on top of it. It picks up where the [MuJoCo event](../rl-mujoco) left off, where we got a Unitree G1 walking with MJX + Brax PPO at 4,096 parallel environments on one GPU.

The goal here is narrow and it is deliberately a goal that MuJoCo cannot easily reach: **put a Unitree Go2 on terrain that is full of holes, and get it across.** Not on a flat plane, not on gentle noise, but on trenches wide enough that walking into one is a fall.

That turns out to be two different problems wearing the same coat, and separating them is the whole lesson of this episode:

1. **Generating the terrain.** Isaac Lab ships 23 sub-terrain generators and a curriculum system that promotes a robot onto harder ground as it earns it. This part is a config file and it works the first time.
2. **Making the robot want to jump.** This part does not work the first time, and no amount of extra training fixes it, because the stock reward function contains a term that prices a jump at roughly minus fifteen. A robot that walks to the lip of every trench and stops is not undertrained. It is correct.

The second problem is the interesting one, and this repo does not fully solve it. It gets a Go2 across 0.245 m trenches on ground it was never trained for, by removing what was forbidding the behaviour and stabilising what was destroying the training. It also shows, with a measurement rather than an opinion, that the remaining gap is not a tuning problem. [Section 8](#8-why-the-dog-would-not-jump) is the whole autopsy, including the two things that turned out to matter more than the reward weights everyone reaches for first.

The first half of this README is a tutorial: what Isaac Sim actually is, how a physics step works, how it becomes an RL environment, and where the terrain and curriculum machinery lives. The second half is the demo, including the reward autopsy above and every trap that cost real time on this box.

**Contents**

1. [How Isaac Sim works](#1-how-isaac-sim-works)
2. [From a simulator to an RL environment](#2-from-a-simulator-to-an-rl-environment)
3. [Terrain, and the curriculum that goes with it](#3-terrain-and-the-curriculum-that-goes-with-it)
4. [The learning half: PPO and rsl_rl](#4-the-learning-half-ppo-and-rsl_rl)
5. [Getting it onto a DGX Spark](#5-getting-it-onto-a-dgx-spark)
6. [Reading this repo](#6-reading-this-repo)
7. [The demo](#7-the-demo) and how to run it
8. [Why the dog would not jump](#8-why-the-dog-would-not-jump), which is the point of the episode
9. [Things that cost real debugging time](#9-things-that-cost-real-debugging-time)
10. [A guided tour of the code](#10-a-guided-tour-of-the-code), if you are presenting it

---

## 1. How Isaac Sim works

MuJoCo is a physics engine with a renderer bolted on the side. Isaac Sim is the other way round: it is an **application platform** (Omniverse Kit) with physics as one of the things plugged into it. That single structural fact explains most of what feels heavy about it, and most of what it can do that MuJoCo cannot.

### Kit, and why the first line of every script is the same

```python
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})   # <- must be FIRST

# only now can these be imported
from isaacsim.core.api import World
import isaaclab.sim as sim_utils
```

`SimulationApp` boots Kit: an extension registry, a USD stage, a renderer, a physics backend, an asset pipeline. It takes tens of seconds. Nothing from `isaacsim.core.*` can be imported before it, because those modules are Python bindings into plugins that do not exist until Kit has loaded them.

This is not a style preference, it is a hard constraint, and it shapes every file in this repo. `checks.py` keeps its `isaacsim` imports inside functions for exactly this reason. It is also why `record.py` and the training script run as **child processes** rather than library calls: Kit wants to own the process, including its asyncio event loop and its exit (see [section 9](#9-things-that-cost-real-debugging-time)).

The narrower version of the rule, which is worth knowing because it buys you a lot: **`isaacsim.core.*` needs Kit, but Isaac Lab's config classes do not.** `isaaclab.terrains` is ordinary Python dataclasses. That is what lets `terrains.py` and `spark_envs.py` in this repo be plain top-level imports, which in turn is what lets Flyte bundle them into a pod.

### USD is the scene, and it is a real database

MuJoCo has MJCF, an XML file you compile into an `mjModel`. Isaac Sim has [USD](https://openusd.org/), Pixar's scene description, and the difference is bigger than "another file format":

- The scene is a **live, layered, composable stage**. You can reference another USD file into yours, override one attribute of it, and the original is untouched. Robot descriptions, environments and materials compose instead of being copy-pasted.
- **Prims** (primitives) live at paths like `/World/envs/env_0/Robot/base`, and you address things with those paths. The `{ENV_REGEX_NS}` you see all over Isaac Lab configs is a placeholder for `/World/envs/env_.*`, so one config line applies to all 4,096 copies.
- It is the same format the rest of the industry uses for 3D content, which is the actual argument for it: assets come from somewhere.

The cost is that a scene is heavier to build than parsing an XML file. Generating the 240 terrain patches this demo uses takes tens of seconds, every run.

### Physics: PhysX, and now Newton

Isaac Sim 6 ships two physics backends, and the velocity locomotion environments used here declare both through a preset:

- **PhysX**, NVIDIA's long-standing GPU rigid-body engine.
- **Newton**, the newer solver built on [Warp](https://github.com/NVIDIA/warp), with an MJWarp path that is (yes) MuJoCo's solver reimplemented in Warp.

You mostly do not choose. `velocity_env_cfg.py` carries a `RoughPhysicsCfg` preset that picks per backend, and one line in it is worth reading because it is the difference between learning and not learning on rough ground:

```python
default_shape_cfg=NewtonShapeCfg(margin=0.01)   # 1 cm collision margin
```

Without that margin, non-Anymal robots fail to develop stable contact on triangle-mesh terrain. Generated terrain is a mesh, not an analytic plane, and mesh contact is fussy.

### Two clocks, same as MuJoCo

```
sim.dt      = 0.005   physics timestep        200 Hz
decimation  = 4       physics steps per action
step_dt     = 0.02    one env step             50 Hz  <- the policy's clock
episode_length_s = 20.0                              -> 1000 env steps per episode
```

Every "steps" number in this repo is at 50 Hz. `--steps 600` is twelve seconds of robot time, and 1000 is a full episode. Encode replay video at 50 fps and it plays back at real speed.

### Actuators: the policy commands poses, not torques

The Go2's actuators are position-controlled. The policy's 12 outputs are **offsets from the default joint pose**, scaled by `actions.joint_pos.scale` (0.25 for the Go2), and a stiff PD loop chases them. This is the same arrangement as the MuJoCo demo next door and for the same reason: "what pose should I aim for at 50 Hz" is a far easier credit-assignment problem than "how much torque for each of 12 joints", and it is how real quadrupeds are commanded, so the policy stays deployable.

---

## 2. From a simulator to an RL environment

Isaac Lab's `ManagerBasedRLEnv` is where a stage full of prims becomes a decision problem. The design is worth understanding because **the entire jumping problem in this episode is a two-line change inside it**.

An environment config is a set of managers, each a `@configclass` holding named terms:

| Manager | What it declares | Go2 rough locomotion |
|---|---|---|
| `scene` | terrain, robot, sensors, lights | generated terrain, Go2, height scanner, contact sensor |
| `observations` | what the policy sees | 235 numbers, listed below |
| `actions` | how outputs reach the robot | 12 joint-position offsets, scale 0.25 |
| `commands` | what the robot is asked to do | a resampled target velocity |
| `rewards` | the objective | 10 weighted terms |
| `terminations` | when an episode ends | base touched the ground, or 20 s elapsed |
| `events` | randomization | material properties, pushes, reset poses |
| `curriculum` | what changes over training | terrain difficulty |

Every term is `(function, weight, params)`, and every one is addressable by name from a subclass. That is the whole extension mechanism, and it is why this repo never forks Isaac Lab.

### The observation, in full

For the Go2 (`velocity_env_cfg.py:161`):

```
base_lin_vel         3     body-frame linear velocity
base_ang_vel         3     body-frame angular velocity
projected_gravity    3     which way is down, in body frame
velocity_commands    3     what it has been ASKED to do
joint_pos           12     relative to default pose
joint_vel           12
actions             12     what it did last step
height_scan        187     a 17 x 11 grid of ground clearance under the body
                   ---
                   235
```

Two of those deserve a note.

**`velocity_commands`** is the task. The policy is not "walking", it is tracking a commanded planar velocity that gets resampled every 10 seconds. Remove this and the robot has no idea what it is being asked for.

**`height_scan`** is the terrain sensor: a ray-caster 20 m above the base, firing straight down on a 0.1 m grid covering 1.6 m by 1.0 m, reported as clearance and clipped to +/-1 m. It is the only reason a policy can react to a trench it has not yet stepped in. `record.py` records it and the Flyte report draws it, because "what the policy actually sees" is a better picture than the camera view for understanding a failure.

### The reward, in full, because it is the story

```python
# -- task
track_lin_vel_xy_exp    weight  1.5   (Go2 raises it from 1.0)
track_ang_vel_z_exp     weight  0.75  (Go2 raises it from 0.5)
# -- penalties
lin_vel_z_l2            weight -2.0   <- remember this one
ang_vel_xy_l2           weight -0.05
dof_torques_l2          weight -0.0002
dof_acc_l2              weight -2.5e-7
action_rate_l2          weight -0.01
feet_air_time           weight  0.01  (Go2 lowers it from 0.125)
undesired_contacts      None          (Go2 removes it)
flat_orientation_l2     weight  0.0
dof_pos_limits          weight  0.0
```

Two positive terms and eight penalties, and the two positive ones both say the same thing: **match the commanded velocity**. Everything else is a smoothness or safety tax.

`lin_vel_z_l2` is a squared penalty on *vertical* base velocity at the largest weight in the set. Hold on to that for [section 8](#8-why-the-dog-would-not-jump).

Every term is logged per-iteration by name, which almost nobody reads:

```
Episode_Reward/track_lin_vel_xy_exp: 0.8123
Episode_Reward/lin_vel_z_l2:        -0.0412
Episode_Reward/feet_air_time:        0.0006
```

`train.py` in this repo scrapes two of these out of the training log and charts them live, because total reward is an average of ten things and hides the one you care about.

---

## 3. Terrain, and the curriculum that goes with it

This is the part MuJoCo has no answer for, and the reason to reach for Isaac Lab at all.

### The generator

A `TerrainGeneratorCfg` lays out a `num_rows` x `num_cols` grid of square patches and fills each one from a library of sub-terrain generators. Isaac Lab ships 23; NVIDIA's default rough config uses six. Every rough-locomotion screenshot you have seen is that one config.

```python
SPARK_PARKOUR_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0), num_rows=10, num_cols=20, curriculum=True,
    sub_terrains={
        "pyramid_stairs":  MeshPyramidStairsTerrainCfg(proportion=0.12, step_height_range=(0.05, 0.23), ...),
        "gaps":            MeshGapTerrainCfg(proportion=0.08, gap_width_range=(0.1, 0.6), ...),
        "stepping_stones": HfSteppingStonesTerrainCfg(proportion=0.08, holes_depth=-2.0, ...),
        ...
    },
)
```

Two axes, and they mean completely different things:

- **Rows are difficulty.** With `curriculum=True`, row 0 is difficulty 0.0 and the last row is 1.0. Every sub-terrain interpolates its own ranges against that number: a `step_height_range=(0.05, 0.23)` is a 5 cm step on row 0 and a 23 cm step on the last row.
- **Columns are variety.** Which sub-terrain you get. And this is assigned **deterministically by cumulative proportion in dict order**, not randomly (`terrain_generator.py:245`). A sub-terrain with `proportion=0.55` listed first owns the first 55% of columns. That is a real API detail with real consequences: it is why `record.py --terrain_col 5` reliably films a trench in this repo, and why reordering a `sub_terrains` dict silently repoints every hardcoded column number.

### The curriculum

The robot is not "put on hard terrain". It is **promoted**, one row at a time, by a curriculum term (`mdp/curriculums.py:27`):

```python
distance  = ||robot_xy - env_origin_xy||
move_up   = distance > size[0] / 2                              # walked half a patch
move_down = distance < ||command_xy|| * episode_length_s * 0.5   # walked less than half of what it was asked
```

Walk far enough and you go up a row. Fall short and you go down one. That is all it is, and three things fall out of it that matter more than they look:

1. **A parkour course with a pit in it does not kill every robot on iteration 1.** The hard rows are simply unreachable until the policy is good enough, so hard sub-terrains cost nothing early.
2. **The mean terrain row is the honest progress metric on rough ground.** Total reward climbs happily when a policy just gets smoother on ground it already owns. `Curriculum/terrain_levels` only climbs when robots reach ground they used to fall off. This repo charts it directly under the reward curve for that reason.
3. **It never reaches the top row, even when solved.** An env that clears the last level is sent to a *random* row rather than kept there (`terrain_importer.py:325`), so the mean settles below the maximum by design. Do not read that as failure.

One more geometric fact, because it bit this demo: promotion needs `size[0] / 2` = 4 m of travel from the patch origin, and patches are 8 m. So promotion always means reaching the *edge* of your patch, regardless of what is in the middle of it. Shrinking a gap terrain's central platform does not move the goal closer; it just puts the trench earlier and leaves more room to land.

---

## 4. The learning half: PPO and rsl_rl

Isaac Lab does not implement RL. It ships adapters for four libraries (rsl_rl, rl_games, skrl, sb3) and entry-point scripts for each. This repo uses **rsl_rl**, which is the one the published Isaac Lab locomotion results come from.

The Go2 rough config (`agents/rsl_rl_ppo_cfg.py`):

```python
num_steps_per_env = 24            # rollout length per env per iteration
actor  = MLP [512, 256, 128], elu
critic = MLP [512, 256, 128], elu
clip_param = 0.2,  entropy_coef = 0.01,  gamma = 0.99,  lam = 0.95
learning_rate = 1e-3, schedule = "adaptive", desired_kl = 0.01
save_interval = 50                # a checkpoint every 50 iterations
```

Ordinary PPO. Three things about it are load-bearing here:

**One iteration is `num_envs * 24` environment steps.** At 4,096 envs that is 98,304 steps per iteration, which is why wall-clock per iteration is roughly constant and why "iterations" is the natural unit.

**The adaptive LR schedule targets a KL of 0.01.** You do not tune the learning rate; it tunes itself against how far the policy moved. This matters when you change reward weights, because it absorbs a lot of the scale change you would otherwise have to compensate for by hand.

**`save_interval = 50` is a free stream of live policies.** This repo exploits that: a long-lived `record.py --serve` daemon films the newest checkpoint every N iterations and drops the clip into the live Flyte report, so a three-hour run is watchable while it runs. That is the Isaac answer to what Brax's `policy_params_fn` gives you in the MuJoCo demo. It cannot be done the Brax way here because training is a child process and its weights are not ours to reach into.

### Registering your own task without forking anything

Isaac Lab's `train.py` resolves `--task` out of the gymnasium registry, so the obvious ways to add a task are to fork the 250-line entry-point script or vendor a copy of `isaaclab_tasks`. Both age badly. There is a supported hook instead, and it is barely documented:

```bash
python train.py --task=Spark-Leap-Go2-v0 --external_callback spark_envs.register
```

`train.py:85` resolves that with `string_to_callable(name, separator=".")`, so the value is `module.attribute` and **not** the `module:attribute` spelling used everywhere else in Isaac Lab. It is called after `import isaaclab_tasks` and before hydra reads the registry, which is exactly the window new tasks have to appear in.

`spark_envs.py` uses it to register 80 tasks: 4 terrains x 10 robots x (train, play).

---

## 5. Getting it onto a DGX Spark

Before any of the above, Isaac Sim has to exist on the box, and on a Grace-Blackwell Spark that is genuinely non-trivial.

| | x86 workstation | DGX Spark (GB10) |
|---|---|---|
| Install | `pip install isaacsim[all]` | **no wheel exists**, build from source |
| Compiler | whatever | **gcc-11**, not the gcc-13 DGX OS 7 ships |
| OpenMP | just works | must preload the system `libgomp`, or Kit dies |
| Disk | a few GB | ~50 GB of build artifacts |
| Build time | none | 10 to 15 min compile |

Three of those five rows are silent failures. The libgomp one fails about twenty seconds into startup with a symbol lookup error that looks like a broken build and is not: PyTorch's aarch64 wheels bundle their own OpenMP runtime, Isaac Sim's native extensions want the system one, and whichever the loader sees first wins the whole process. `env.sh` fixes it in one line.

```bash
cd topics/isaac-sim

# Stage 1. Needs sudo and a human: installs gcc-11 and shows you the
# NVIDIA Omniverse license, which you accept yourself.
./setup.sh prereqs

# Stages 2 and 3. No prompts.
./setup.sh isaacsim     # clone + build, 10-15 min
./setup.sh isaaclab     # clone + pip install rsl_rl / rl_games / skrl / sb3

# Did it work?
source env.sh
$ISAACSIM_PYTHON_EXE smoke_test.py
```

`smoke_test.py` checks four things, in the order they actually fail:

1. **Kit boots headless.** Catches libgomp, missing X11 headers, a bad build.
2. **PhysX is on the GB10.** The one worth having. PhysX falling back to CPU does not raise; it just runs, quietly, roughly 50x slower, and every parallel-env benchmark you take afterwards is meaningless.
3. **Gravity is real.** A cube dropped from 2 m falls, and no further than `0.5*g*t^2` allows. Separates "Kit started" from "physics is stepping".
4. **Throughput.** Physics steps/sec, so there is a number to compare against when something later feels slow.

Nothing installs into this repo. Both trees go to `$ISAAC_ROOT` (default `~/isaac`). The local `.venv` holds `flyte` only; there is no `isaacsim` in it and there cannot be.

### And the same thing in a pod

```bash
uv venv .venv && uv pip install --python .venv/bin/python -r requirements.txt
./nvgfx.sh                                # stage the graphics driver, see section 9
./.venv/bin/flyte run pipeline.py smoke
```

The container path needs no source build. It starts `FROM nvcr.io/nvidia/isaac-sim:6.0.0`, which publishes a real arm64 manifest and pulls anonymously. The source build is still worth having: it is the fast iteration loop, and it is the control number the containerised run is compared against.

---

## 6. Reading this repo

```
bare metal on the host           in a Flyte pod
─────────────────────            ──────────────
setup.sh    builds from source   Dockerfile        smoke image, FROM nvcr.io/nvidia/isaac-sim
env.sh      source before use    Dockerfile.train  + torch + Isaac Lab, ~25 GB
smoke_test.py  a shim over ↓     config.py         Flyte images and task environments
              checks.py  ←──── the physics, shared by both so the numbers compare
                                 pipeline.py       the Flyte tasks and entry points
                                 train.py          drive rsl_rl, scrape its log, build the report
                                 record.py         replay a policy through the RTX renderer
                                 terrains.py       terrain generator configs
                                 spark_envs.py     register our tasks + reward profiles
                                 nvgfx.sh          stage the NVIDIA graphics driver (section 9)
```

The two things worth knowing before opening any of it:

**`checks.py` is deliberately the only place the physics lives**, so the bare-metal number and the containerised number measure the same thing.

**`terrains.py`, `spark_envs.py` and `record.py` reach the pod through `train_env.include`, not through imports.** They import `isaaclab` at module level, and `pipeline.py` is loaded by the orchestrator too, which runs on an image with no Isaac Lab in it. A top-level import would break every orchestrator pod before it could schedule anything. The long comment in `config.py` explains it.

---

## 7. The demo

Four entry points, in the order they were built. All of them run against the `physical-ai` Flyte project and all of them put their results in a live Flyte report.

```bash
# 1. Does the GPU reach a pod at all?
flyte run pipeline.py smoke

# 2. Flat ground. The "it learns" baseline.
flyte run pipeline.py walk

# 3. Our own 13-sub-terrain course, filmed with our own cameras.
flyte run pipeline.py parkour
flyte run pipeline.py parkour --task_id Spark-Stairs-Go2-v0

# 4. The jump.
flyte run pipeline.py leap
flyte run pipeline.py leap --iterations 5     # is the plumbing alive?
```

Task ids are `Spark-{Parkour,Stairs,Stones,Leap}-<Robot>-v0` over the ten robots in `spark_envs.BASE_TASKS`. Note the robot spelling is the display name, so `Spark-Stairs-Go2-v0`, not `Spark-Stairs-Unitree-Go2-v0`.

Each training task does two things in **one pod**, deliberately: train with rsl_rl, then replay the trained policy through the RTX renderer to an mp4 that is base64'd into the report. Splitting them would mean shipping a checkpoint through blob storage and queueing for the same single GPU twice.

### Measured on this box

DGX Spark, GB10, DGX OS 7.2.3 / Ubuntu 24.04, driver 580.126.09, CUDA 13.0, aarch64.

| | |
|---|---|
| Anymal-C flat, 4096 envs | ~1.0 s/iteration; a walking policy in 1500 iterations / 27 min |
| `num_envs` throughput knee | **4096**. Compute-bound, not memory-bound: 8192 envs still only uses about 10 GB |
| Episode | 20 s = 1000 env steps at 50 Hz |
| Checkpoints | every 50 iterations, ~6 MB tarred with the ONNX export |

### What lands in the report

Ordered so the answer is in the first screen, because a Flyte report opens scrolled to the top and repaints in place while the run is live:

1. **The chase-camera replay**, RTX-rendered in the pod.
2. **"Did it actually jump?"** Longest unbroken flight in the clip and the distance covered during it, measured off the contact sensor rather than inferred from reward. See below.
3. Mean reward, then mean terrain row, then the air-time reward term.
4. The mid-training snapshot strip: the same robot on the same patch with the same camera, at several points in training.
5. What the height scanner saw, drawn as a grid.

---

## 8. Why the dog would not jump

This is the episode.

The first parkour run was a Go2 on the 13-sub-terrain course for three hours. The reward curve climbed the whole way. The robot walked to the lip of every trench and stopped.

The instinct is "train it longer". That is wrong, and you can prove it is wrong without running anything.

### The autopsy

Every `Spark-Parkour-*` task is the stock `Isaac-Velocity-Rough-*` env with the terrain generator swapped and **nothing else touched**. That was a deliberate design property (if terrain is the only variable, a robot doing better on stairs than on stepping stones is a fact about stairs) and it is exactly what caps the result. The task being optimized is *track a commanded planar velocity of at most 1 m/s*. Jumping is never asked for.

Worse, it is priced. From `velocity_env_cfg.py:295`, inherited unchanged by the Go2:

```python
lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=-2.0)
```

A squared penalty on vertical base velocity, at the largest weight in the set. Clearing a 0.45 m trench at 1 m/s needs roughly 0.45 s of flight, so a takeoff near 2.7 m/s, so an instantaneous **-2.0 x 2.7² = -14.6** against Go2 task rewards that cap at 1.5 + 0.75 = 2.25.

And the one term that pays for being airborne, `feet_air_time`, is set to `0.01` for this robot, down from the base class's 0.125. It is numerically switched off.

PPO solved the problem it was given. The answer to that problem is: never leave the ground.

The curriculum then quietly hides the evidence. Gap patches promote only if the robot travels 4 m from a 3 m platform, which means crossing. A policy that cannot cross gets demoted off those columns and settles on ground it can walk, so `Curriculum/terrain_levels` plateaus while mean reward keeps climbing. Two charts, two different stories, and only one of them is about the thing you asked for.

### The fix, in two halves

Neither half works alone.

**Half one: a terrain with a rung it can reach** (`terrains.SPARK_LEAP_CFG`).

- 12 rows instead of 10, and gaps from **0.05 m** rather than 0.1 m. Row 0 has to be something a policy that has never left the ground clears by accident, because that accident is the only bootstrap available.
- Top row **0.38 m**, not 0.6 m. A Go2 crosses a trench by *stepping* for as long as the gap is narrower than its front-to-rear foot span, roughly 0.30 m; past that a flight phase stops being optional. So the rows that decide this demo are the handful between 0.25 and 0.35 m, and the ceiling came *down* from an initial 0.45 m to put five rungs in that band instead of three. Lowering it buys resolution exactly where the behaviour has to change.
- Gaps take 55% of columns, not 100%. A course that is nothing but holes trains a policy that cannot walk, and a policy that cannot walk never builds the forward speed a leap is made of. The other 45% is rails, boxes and noise.
- **The ladder scales to the robot.** 0.05 to 0.38 m is a Go2 ladder. A G1 humanoid strides 0.38 m without noticing, so handing it the same config would top out the curriculum on the first afternoon and measure zero flight the whole way: a chart that looks like a triumph and means nothing. The profile reads the robot's own nominal standing height out of its config and scales the gap and rail ranges by it, capped at 2x. Go2 0.40 m gets 0.05-0.38; G1 0.74 m gets 0.09-0.70; H1 1.05 m gets 0.10-0.76.

**Half two: a reward profile** (`spark_envs.REWARD_PROFILES["leap"]`), applied on top of the robot's own tuning:

| Term | Stock (Go2) | Leap | Why |
|---|---|---|---|
| `lin_vel_z_l2` | -2.0 | **-0.05** | The one that matters. Not 0.0: with no cost at all and a positive air-time term, bouncing on the spot is free reward and a pogo-stick policy is a real attractor. A whisper keeps bouncing pointless and leaves a deliberate leap affordable, at -0.36 instead of -14.6. Applied as *relax, never tighten*, because G1 already sets this to 0.0 and H1 deletes it: a blind assignment would **add** a penalty to the bipeds, which is the opposite of the point. |
| `terminations.fell_into_hole` | absent | **added** | The most valuable line after the one above, and not a reward change at all. See below. |
| `feet_air_time` threshold | 0.5 s | **0.25 s** | The line between "stride" and "flight". A Go2 trots with ~0.25 s of air per foot, so the stock 0.5 s scores *every footfall* negative. |
| `feet_air_time` weight | 0.01 | **0.5** | It was switched off. |
| `dof_acc_l2`, `action_rate_l2` | -2.5e-7, -0.01 | **halved** | A leap is the largest joint acceleration and fastest action change a locomotion policy produces. Neither penalty would block it alone, but both point at the thing being learned. |
| command `lin_vel_x` | (-1, 1) | **(-1, 2)** | A run-up. At 2 m/s a 0.45 m trench needs 0.22 s of flight, which is a bound rather than a stunt. |
| command `lin_vel_y` | (-1, 1) | **(-0.5, 0.5)** | Sideways at a ring-shaped trench is a crossing from a standstill, and mostly teaches falling in. |
| `max_init_terrain_level` | 5 | **1** | Row 5 of the leap grid is a 23 cm trench. Starting there means all 4,096 envs spend hundreds of iterations being demoted one row per episode just to reach ground they can learn on. |

**Note what is not in that table: nothing rewards jumping.** Velocity tracking already paid enormously for crossing a gap, and always did, because a robot stopped at the lip earns nothing on a 1.5-weighted term for the remaining fifteen seconds of its episode. The edits do not add an incentive. They clear the path to one that was there all along, by making the first few exploratory hops survivable instead of instantly extinguished.

The honest cost: a `Leap` result is a fact about a terrain **and** a reward set together, so it is not comparable with the other three the way they are comparable with each other. The other three still ship NVIDIA's rewards byte for byte.

### What the first run proved, and the fix it forced

4000 iterations of the above, and the flight measurement was flat:

```
iteration    0 (untrained) : 0.16 s of flight over 0.08 m
iteration 3750 (final)     : 0.18 s of flight over 0.05 m
```

Sixteen clips, same row, same camera, and not one of them is distinguishable from the untrained policy. The air-time reward term never crossed zero either. The policy reached row 4.45, a 21 cm trench, by **stepping** across it.

The curriculum curve said the rest:

```
row 4.16 (it 3174) -> 1.46 (it 3524) -> 4.57 (it 3924)     collapse and re-learn, three times
mean reward -58.05 at iteration 3949, against a typical +10
```

That is not a policy learning slowly. It is a policy being knocked over by a heavy-tailed reward.

The cause is in the terrain, not the reward. `MeshGapTerrainCfg` hardcodes `terrain_height = 1.0` (`mesh_terrains.py:584`), so a missed gap is a **one metre fall**, and the stock termination set does not catch it: `base_contact` only fires when the base itself takes a contact force, and a robot that drops in feet-first and stays upright at the bottom of a trench never triggers it. It stands down there for the remaining fifteen seconds of its episode, earning nothing on a 1.5-weighted term and paying impact and torque penalties on the way in.

So the leap profile adds a termination, and Isaac Lab's own `root_height_below_minimum` is the wrong one: it compares against a **world-frame** z, which its docstring admits only works on flat ground, and our patches sit at twelve different heights. The terrain-aware version is one line:

```python
(root_pos_w[:, 2] - env.scene.env_origins[:, 2]) < threshold
```

`env_origins` is the patch the env is currently assigned to, and the terrain curriculum rewrites it on every promotion (`terrain_importer.py:329`), so it is always the surface the robot spawned on.

**The threshold has to be robot-relative, and getting that wrong fails silently in the flattering direction.** The trench is 1.0 m deep for everyone, but a Go2 standing at the bottom has its base at -0.60 while a G1 has its base at -0.26. A fixed `-0.4` catches the Go2 and *never fires for the G1*: the humanoid would stand in the trench all episode and the termination would report `0.0000` forever, which reads exactly like "no robot ever fell in". The profile uses `nominal_standing_height - 0.6`, which is more than any feature on this terrain and less than the trench, for every robot in the zoo. Confirmed by inverting it: at a threshold above standing height it fires on 100% of episodes, so the frame and sign are right.

The effect, same task, same everything else:

```
without terminate-on-fall:  row 4.16 -> 1.46 -> 4.57   collapses, reward spikes to -58
with it:                    0.30 -> 1.54 -> 3.35 -> 4.67 -> 5.37 -> 5.97 -> 6.40   monotonic, reward steady 11-15
```

`fell_into_hole` accounts for about 1.5% of terminations: exactly the falls `base_contact` was missing.

### And the answer, after 5000 stable iterations

The fix was worth two full rows of curriculum, and then it stopped dead:

```
row  0.77 (it   24) -> 3.01 -> 4.81 -> 5.63 -> 5.97 -> 6.48 (it 2124)
     6.28   6.39   6.43   6.40   6.53   6.59   6.55   6.50    (it 2474 .. 4924)
```

Flat for the last 2900 iterations. Not oscillating, not creeping: converged. Final mean reward 14.79, `track_lin_vel_xy_exp` at 0.97 of its ceiling, 64% of episodes surviving the full twenty seconds. It is a good policy. It plateaued at row 6.6, which on this ladder is a **0.245 m** trench.

And the flight measurement, over twenty clips from iteration 0 to 4750:

```
0.08  0.04  0.10  0.14  0.12  0.06  0.10  0.18  0.12  0.10
0.12  0.06  0.04  0.10  0.06  0.08  0.08  0.12  0.08  0.08   seconds
```

Untrained is 0.08 s. Final is 0.08 s. There is no trend in there, and the air-time reward term sat at -0.0033 the whole way without ever crossing zero.

**So the two runs together settle the question, and the answer is neither "more time" nor "these weights".** Fixing the instability raised the ceiling from 0.21 m to 0.245 m and made the climb monotonic. It did not produce a jump, and 2900 flat iterations say more time will not either. The policy converged to the best thing velocity tracking can buy: a robot that walks up to the exact limit of its own step-over reach, roughly its front-to-rear foot span, and no further.

That is the honest ceiling of this approach, and it is worth stating plainly rather than dressing up:

> **Velocity tracking cannot express a jump.** Relaxing the penalties lets a robot leave the ground; it does not give it a reason to. Crossing a trench pays, but only for a policy that has already crossed one, and PPO will not find that first crossing by chance when every failed attempt terminates the episode.

Getting past it needs a different *kind* of objective, not different weights on this one:

- **A goal-reaching reward.** Pay for progress toward a waypoint on the far side rather than for matching an instantaneous velocity. This is what [Extreme Parkour](https://extreme-parkour.github.io/) and the robot-parkour-learning line of work do, and it is why they reach 0.6 m gaps.
- **A jump command.** Add a term to the command manager (a target base height, or a discrete "jump now") and a reward that tracks it, so the behaviour can be *asked for* instead of hoped for. Then a jump has a gradient at every point, not only after the first accidental success.

Both are a new task family rather than a profile on this one, which is exactly why `REWARD_PROFILES` is keyed by terrain and cheap to add to.

### Measuring it, rather than believing the curve

Reward going up is an argument that a jump probably happened. This is the observation:

```python
# record.py, during the replay
grounded[i] = min(contact_sensor.current_air_time[i]) <= 0     # is ANY body touching?
```

The contact sensor tracks `{ENV_REGEX_NS}/Robot/.*`, every body and not just the feet, so a stumble that lands on the knees reads as grounded. A **flight phase** is a maximal run of `False`, and the report gives the longest one plus the straight-line distance covered during it. On a gap patch that distance is directly comparable with the trench width, which is what makes it worth printing: "0.34 s of flight, 0.38 m covered" next to "the trench at row 7 is 0.28 m" is a claim anyone can check.

It is also the number that can embarrass the run, which is why it is there. A policy that learned to shuffle to the lip and stop reads 0.00 s no matter how good the reward curve looks. For calibration, an untrained Go2 (two iterations) on a 0.086 m row 1 trench measures 0.12 s and 0.063 m: that is a trot, not a jump.

Alongside it, `train.py` scrapes `Episode_Reward/feet_air_time` out of the training log every iteration and charts it with the zero line forced into view. On the leap profile that term is negative while the policy shuffles and crosses zero when strides get longer than a walk. Watching it turn positive is watching the jump appear, several hundred iterations before it is big enough to see in a clip.

---

## 9. Things that cost real debugging time

### Kit owns the process, so run it as a child

Not an import problem: `import flyte` and `import isaacsim` coexist fine in one interpreter. It fails at **shutdown**, and both modes of `SimulationApp.close()` are wrong in a Flyte pod:

- `fast_shutdown=True` (the default) calls `os._exit()`. The process vanishes, Flyte never records a return value, and the run reads as an unexplained pod exit.
- `fast_shutdown=False` cancels **every asyncio task in the process**, including Flyte's own. Observed in a real pod: `Cancelling <Task ... coro=<load_and_run_task()>>`, then a pod that sat Running for 28 minutes holding the GPU until it was aborted by hand.

A child process gives Kit its own event loop to tear down however it likes. Everything must also be printed and written **before** `simulation_app.close()`, because the fast path never returns.

### The renderer needs a different half of the driver than CUDA does

The nastiest one in this repo, because the symptom points at the wrong thing.

The NVIDIA container stack splits the userspace driver into **capabilities**. `compute` is libcuda and friends. `graphics` is `libGLX_nvidia`, `libnvidia-glcore`, the RT core and OptiX. They are separate sets and you can have one without the other.

Flyte's devbox image ships `NVIDIA_DRIVER_CAPABILITIES=compute,utility`, and k3s runs *inside* that container, so every task pod inherits the choice. Nothing at the pod level can widen it: not the Isaac image's own `NVIDIA_DRIVER_CAPABILITIES=all`, not `runtimeClassName: nvidia`, not a hostPath mount described by a Flyte `pod_template` (flyte 2.2.1 accepts that last one without complaint and the backend silently drops it).

What you get is a pod where `nvidia-smi` is happy, 4,096 envs train at full speed, and only the replay dies:

```
[Error] [omni.rtx] VkResult: ERROR_INCOMPATIBLE_DRIVER
[Error] [omni.rtx] vkCreateInstance failed. Vulkan 1.1 is not supported
[Error] [omni.gpu_foundation_factory.plugin] Failed to create any GPU devices
[Error] [omni.kit.renderer.plugin] GPU Foundation is not initialized!
```

followed by several hundred `cudaErrorIllegalAddress` messages that are all **fallout** from the renderer never starting. Read those first and you will spend the afternoon chasing a GPU memory bug that does not exist. `/etc/vulkan/icd.d/nvidia_icd.json` names `libGLX_nvidia.so.0` and nothing ever mounted it.

`./nvgfx.sh` copies that library and its dependency closure out of the host driver into the build context, and `Dockerfile.train` COPYs them to `/opt/nvgfx`, which is at the front of `LD_LIBRARY_PATH`. Run it once after `flyte start devbox --gpu` and again after any driver update. The proper fix is recreating the devbox with wider capabilities, and it costs the cluster: `/var/lib/rancher/k3s` is an anonymous docker volume, so the in-cluster registry goes with it and both images have to be rebuilt.

### When Kit dies it dies loudly, so a log tail is useless

A single render fault prints three lines per allocation it then fails to free, hundreds in a row. A 25-line tail is 25 copies of the consequence and none of the cause. `record_clips()` in `train.py` keeps two buffers: the last 60 lines verbatim, and the **first** 60 lines that are not fault-fallout. The second one is the useful one.

Related: `record.py` exiting **0** with no `summary.json` is a real failure and the common one, because Kit's fast shutdown calls `os._exit()`. The missing file is the only honest signal.

### The viewport capture films everything except the robot

`play.py --video` captures the Kit viewport, and headless on this box that renders the terrain, the lighting and no robot. `record.py` drives its own `Camera` sensors instead, which fixes it and throws in the onboard RGB and depth views for free.

While you are there: the default viewer is a **fixed** camera at world (7.5, 7.5, 7.5) looking at the origin. On a flat plane that happens to frame the robot, which is why nobody notices. On generated terrain the robot spawns on a patch metres away and walks off: measured on the first parkour render, a Go2 occupying about 0.1% of a 1280x720 frame.

### Deep-copy your terrain configs, and name your experiments

Two traps in `spark_envs.py`, both found the hard way.

`TerrainGeneratorCfg` objects are ordinary mutable dataclasses held as module-level singletons, and Isaac Lab's own `_PLAY` configs do `self.scene.terrain.terrain_generator.num_rows = 5` in `__post_init__`. Share one config object across variants and instantiating the Play task silently shrinks the training task.

rsl_rl derives its log directory from `experiment_name`, and `get_checkpoint_path` walks that directory for the newest checkpoint. Reuse the stock `unitree_go2_rough` name and `play.py` will happily load a checkpoint trained on completely different terrain and report it as a success.

### Everything else

- **`MeshRepeatedBoxesTerrainCfg` needs `object_type="box"`.** Its declared default resolves to a `ResolvableString`, the generator tests `isinstance(cfg.object_type, str)` first (`mesh_terrains.py:762`), takes the string branch, looks up `make_isaaclab.terrains.trimesh.utils:make_box`, and dies with "must be a string or a callable. Received: None".
- **`--external_callback` uses `module.attribute`**, not the `module:attribute` spelling used everywhere else in Isaac Lab.
- **Set your logger level explicitly.** `pipeline.py` calls `logging.basicConfig(level=WARNING)`, so a module logger inherits WARNING from the root and every `log.info()` is dropped. That is how a 100-minute training run came to have exactly one line in `kubectl logs`.
- **Base64 mid-training clips are not free.** These are path-traced frames, so they are full of sampling noise, and x264 spends enormous bitrate preserving noise: the first snapshot filmed at 640x360 crf 26 came out at 6.3 MB for five seconds, and every clip is re-encoded into the page on every repaint. The snapshot path uses a much coarser crf for that reason alone.
- **Give the orchestrator no GPU.** An orchestrator pod holds its resources for as long as its children run, so a GPU-holding orchestrator deadlocks its own GPU child on "Insufficient nvidia.com/gpu".

---

## 10. A guided tour of the code

If you are presenting this, roughly this order:

1. **`checks.py`** (5 min). The smallest complete thing: boot Kit, drop a cube, check gravity is real and PhysX is on the GPU. Establishes the `SimulationApp`-first rule and the CPU-fallback trap.
2. **`terrains.py`** (10 min). Where the demo becomes visual. Read `SPARK_PARKOUR_CFG` against NVIDIA's six-sub-terrain default, then the rows-are-difficulty / columns-are-variety explanation, then the fact that column assignment is deterministic by dict order.
3. **`spark_envs.py`** (15 min, the centrepiece). `--external_callback` as the no-fork extension point, then the derivation of 80 tasks, then `REWARD_PROFILES` and the `lin_vel_z_l2 = -2.0` autopsy. This is the part people remember.
4. **`pipeline.py`** (10 min). Four entry points and the one-pod train-and-film shape. The child-process reasoning lives in its header.
5. **`train.py`** (10 min). Scraping rsl_rl's stdout into live charts, and `Snapshotter`: filming a policy while it trains, from a daemon that boots Kit once.
6. **`record.py`** (10 min). Camera sensors instead of the viewport, the height-scanner recording, and `_flight_phases` as the measurement that can prove the demo wrong.
7. **`nvgfx.sh`** (5 min, optional but a good war story). Vulkan, driver capabilities, and why a hundred CUDA errors were all a red herring.

---

## Reference

- [Isaac Sim](https://github.com/isaac-sim/IsaacSim) and its [docs](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab), and the [manager-based env tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [OpenUSD](https://openusd.org/)
- [Warp](https://github.com/NVIDIA/warp) and Newton
- The MuJoCo episode next door: [`topics/rl-mujoco`](../rl-mujoco)
- Extreme Parkour ([paper](https://extreme-parkour.github.io/)), for what a purpose-built parkour reward looks like when velocity tracking is not the objective
