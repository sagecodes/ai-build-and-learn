# Isaac Sim: teaching a robot dog to cross ground it cannot walk on

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This one is about [NVIDIA Isaac Sim](https://github.com/isaac-sim/IsaacSim), the GPU-accelerated robotics simulator built on Omniverse, and [Isaac Lab](https://github.com/isaac-sim/IsaacLab), the RL framework on top of it. It picks up where the [MuJoCo event](../rl-mujoco) left off, where we got a Unitree G1 walking with MJX + Brax PPO at 4,096 parallel environments on one GPU.

The goal here is narrow and it is deliberately a goal that MuJoCo cannot easily reach: **put a Unitree Go2 on terrain that is full of holes, and get it across.** Not on a flat plane, not on gentle noise, but on trenches wide enough that walking into one is a fall.

That turns out to be two different problems wearing the same coat, and separating them is the whole lesson of this episode:

1. **Generating the terrain.** Isaac Lab ships 23 sub-terrain generators and a curriculum system that promotes a robot onto harder ground as it earns it. This part is a config file and it works the first time.
2. **Making the robot want to jump.** This part does not work the first time, and no amount of extra training fixes it, because the stock reward function contains a term that prices a jump at roughly minus fifteen. A robot that walks to the lip of every trench and stops is not undertrained. It is correct.

The second problem is the interesting one, and this repo does not fully solve it. It gets a Go2 across 0.245 m trenches on ground it was never trained for, by removing what was forbidding the behaviour and stabilising what was destroying the training. It also shows, with a measurement rather than an opinion, that the remaining gap is not a tuning problem. [Section 8](#8-why-the-dog-would-not-jump) is the whole autopsy, including the two things that turned out to matter more than the reward weights everyone reaches for first.

This README is in three parts. **Sections 1 to 4 are a tutorial**: what Isaac Sim actually is, how a physics step works, how it becomes an RL environment, and where the terrain and curriculum machinery lives. **Sections 5 to 10 are the demo**, including the reward autopsy above and every trap that cost real time on this box. **Sections 11 to 14 are for building your own thing**: importing a robot that NVIDIA never shipped, the sensors and the second workflow this demo does not use, what the other six task families in Isaac Lab optimise and how each one's reward is written, and the shortlist of things worth building here next.

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
11. [Bringing your own robot](#11-bringing-your-own-robot): assets, converters, articulations, actuators
12. [Sensors, and the two workflows](#12-sensors-and-the-two-workflows)
13. [The rest of Isaac Lab](#13-the-rest-of-isaac-lab-and-the-reward-that-defines-each-task), and the reward that defines each task
14. [What to build next here](#14-what-to-build-next-here)

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
                                 test_vault_rewards.py  check the gap rewards, no Kit needed
                                 nvgfx.sh          stage the NVIDIA graphics driver (section 9)
```

The two things worth knowing before opening any of it:

**`checks.py` is deliberately the only place the physics lives**, so the bare-metal number and the containerised number measure the same thing.

**`terrains.py`, `spark_envs.py` and `record.py` reach the pod through `train_env.include`, not through imports.** They import `isaaclab` at module level, and `pipeline.py` is loaded by the orchestrator too, which runs on an image with no Isaac Lab in it. A top-level import would break every orchestrator pod before it could schedule anything. The long comment in `config.py` explains it.

---

## 7. The demo

Five entry points, in the order they were built. All of them run against the `physical-ai` Flyte project and all of them put their results in a live Flyte report.

```bash
# 1. Does the GPU reach a pod at all?
flyte run pipeline.py smoke

# 2. Flat ground. The "it learns" baseline.
flyte run pipeline.py walk

# 3. Our own 13-sub-terrain course, filmed with our own cameras.
flyte run pipeline.py parkour
flyte run pipeline.py parkour --task_id Spark-Stairs-Go2-v0

# 4. The jump, with a blind reward set that has been relaxed to allow one.
flyte run pipeline.py leap
flyte run pipeline.py leap --iterations 5     # is the plumbing alive?

# 5. The same ground, with a reward that reads the height scanner. Section 8.
flyte run pipeline.py vault
```

Task ids are `Spark-{Parkour,Stairs,Stones,Leap,Vault}-<Robot>-v0` over the ten robots in `spark_envs.BASE_TASKS`. Note the robot spelling is the display name, so `Spark-Stairs-Go2-v0`, not `Spark-Stairs-Unitree-Go2-v0`.

`leap` and `vault` share a terrain generator object on purpose, so the only difference between those two runs is the reward set and a result is attributable to it. The first three keep NVIDIA's rewards untouched, so among *them* the terrain is the only variable. Those are two different comparisons and it is worth keeping them straight.

The two `vault` reward terms are pure functions of a height-scan tensor, so they can be checked in about two seconds without a GPU:

```bash
source env.sh && "$ISAACSIM_PYTHON_EXE" test_vault_rewards.py
```

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

There is also a third option that is neither, and it is the one this repo tried next. It is below.

### The third option: let the reward see what the policy already sees

The two fixes above are both new task families. There is a cheaper one hiding in plain sight, and finding it means re-reading the autopsy with one question in mind: *did the robot know the gap was there?*

It did. `Isaac-Velocity-Rough-*` mounts a height scanner on the base (`velocity_env_cfg.py:112`) and feeds it straight into the policy observation:

```python
height_scanner = RayCasterCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
    ray_alignment="yaw",
    pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
    mesh_prim_paths=["/World/ground"],
)
```

187 rays on a 1.6 m x 1.0 m grid, yaw-aligned to the base, cast down onto the terrain mesh. `mdp.height_scan` turns that into `base_z - hit_z - 0.5`, so our 1.0 m trench reads about **+0.9 against -0.1 on flat ground**: nine times the ±0.1 observation noise and well inside the ±1.0 clip. A trench is already 187 numbers in the input vector, 0.8 m before the base reaches it.

**So the dog was never blind. Every reward term was.** `lin_vel_z_l2`, `feet_air_time`, `dof_acc_l2`, `action_rate_l2` and the two tracking terms are all proprioceptive; not one can tell the lip of a trench from the middle of the platform. `leap` paid for hang time everywhere and velocity everywhere, and then measured 0.08 s of flight at iteration 0 and 0.08 s at iteration 4750.

That reframes the wall at the end of the autopsy. The problem was never that velocity tracking cannot pay for a jump: it pays enormously, but only *after* one succeeds. The problem is that it cannot pay for an **attempt**, so every failure scored identically and there was nothing for PPO to climb. Relaxing `lin_vel_z_l2` from -2.0 to -0.05 made an attempt affordable. It did not make one worth making.

`Spark-Vault-*` is `leap` plus two reward terms that read the same scanner (`spark_envs.py`):

| term | pays for | why it is shaped that way |
| --- | --- | --- |
| `gap_takeoff`, weight 1.0 | upward velocity **×** trench in the forward window **×** forward speed | Dense, and it fires several steps before the robot commits. The forward-speed gate is not optional: without it, bouncing on the spot at the lip farms the term forever, and pogoing is a real attractor once `lin_vel_z_l2` is down at -0.05. |
| `gap_flight`, weight 2.0 | forward speed **×** all feet airborne **×** trench underneath | The payoff, and the term that gives a *failed* crossing a gradient. A robot that launches and drops in is airborne over a hole for the ~0.3 s of its fall and banks some of it; one that gets further banks more. |

Together they create the ordering `leap` could not express:

```
walked to the lip  <  jumped and fell in  <  jumped and nearly made it  <  crossed
```

Under `leap` the middle two were indistinguishable from the first.

Three details worth stealing:

- **Depth is measured against `env.scene.env_origins[:, 2]`**, the patch the env is currently assigned to, for the same reason `fell_into_hole` uses it: our patches sit at twelve different heights, so a world-frame z means nothing, and the rays under the base are no use because the robot is airborne for exactly the part of the manoeuvre that matters.
- **`gap_takeoff` reads world-frame vertical velocity, not base-frame.** Isaac Lab's own `lin_vel_z_l2` uses `root_lin_vel_b[:, 2]`, which is fine for a penalty and exploitable as a reward: the base z-axis points out of the robot's back, so a robot running at 2 m/s with a 30° nose-up rear reads +1.0 m/s of "vertical" velocity with every foot still on the ground. Pay for that and you buy a dog that pops a wheelie at every trench.
- **The gap signal is a fraction of the window, not a maximum.** A max-depth version saturates the instant any ray finds the trench, 0.7 m out, then sits flat for the whole approach and pays a takeoff from too far away exactly as well as from the lip. The fraction peaks when the gap fills the window, which *is* the takeoff moment.

**On privilege, because this is the part that is easy to oversell on a stream.** The new terms read `ray_hits_w` raw: no noise, no clip. That is legitimate, because a reward function is simulator-side scaffolding that never ships with the policy. The observation is untouched, so the policy still sees exactly the noisy, clipped scan `leap` saw and a `vault` run stays a fair comparison rather than a different task.

But the scanner itself is not a sensor you could bolt to a real Go2. It casts rays from 20 m above the robot, through its own body, onto the ground mesh, with no occlusion, no field of view and no range limit. It is a simulator query wearing a sensor's clothes. This is the **teacher** half of the standard two-stage sim-to-real recipe (Lee et al. 2020, Miki et al. 2022, Extreme Parkour); the student half is the distillation run in [section 14](#take-the-cheat-sensor-away), and it is not built. A `vault` clip is evidence that the behaviour is learnable, not that the robot could do it outdoors.

The number to read is `Episode_Reward/gap_flight`, charted in the report next to the air-time term. It is **structurally pinned at zero** until the robot is airborne over a trench while moving forward, because there is no other way to earn it: not a long stride, not a bounce on flat ground, not a wheelie. So the chart has no interesting shape, only an interesting moment. The iteration it lifts off zero is the iteration the jump was invented, and if it is still flat at the end of the run, the run failed and it failed unambiguously.

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
3. **`spark_envs.py`** (15 min, the centrepiece). `--external_callback` as the no-fork extension point, then the derivation of 100 tasks, then `REWARD_PROFILES` and the `lin_vel_z_l2 = -2.0` autopsy. This is the part people remember.
4. **`_vault_profile`, still in `spark_envs.py`** (10 min, the payoff). The reveal is one sentence: the robot could always see the gap, and every reward term was blind to it. Then `gap_takeoff` and `gap_flight`, and why each factor in them is load-bearing. `test_vault_rewards.py` runs in two seconds and makes a good live demo of "here is the pogo exploit, here is the term that closes it".
5. **`pipeline.py`** (10 min). Five entry points and the one-pod train-and-film shape. The child-process reasoning lives in its header.
6. **`train.py`** (10 min). Scraping rsl_rl's stdout into live charts, and `Snapshotter`: filming a policy while it trains, from a daemon that boots Kit once.
7. **`record.py`** (10 min). Camera sensors instead of the viewport, the height-scanner recording, and `_flight_phases` as the measurement that can prove the demo wrong.
8. **`nvgfx.sh`** (5 min, optional but a good war story). Vulkan, driver capabilities, and why a hundred CUDA errors were all a red herring.

---

## 11. Bringing your own robot

Everything above uses robots that were already in the box. The first thing anyone wants to do next is bring their own, and none of the ten tasks in `BASE_TASKS` shows you how, because they all start from a USD file NVIDIA already published.

### Where the assets actually come from

There is no robot in this repo, and there is no robot in the Isaac Sim container either. `UNITREE_GO2_CFG` (`isaaclab_assets/robots/unitree.py:142`) points at:

```python
usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd"
```

and that constant resolves, through `isaaclab/utils/assets.py:50`, to a setting parsed out of `apps/isaaclab.python.kit`:

```
persistent.isaac.asset_root.cloud = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0"
```

So every training pod streams the robot, and the terrain materials, and the ANYmal policy the navigation task uses, from an S3 bucket in us-west-2 on first touch. That is worth knowing for three reasons: it is why a first run is slower than the second, it is the thing that breaks in an air-gapped cluster, and it is the reason `NUCLEUS_ASSET_ROOT_DIR` is a variable rather than a constant. Point that setting at a local mirror and the whole asset layer moves with it.

Three constants sit on top of the same root, and picking the wrong one is a 404 rather than an error you can read:

| Constant | Resolves to | Holds |
|---|---|---|
| `NUCLEUS_ASSET_ROOT_DIR` | the bucket root | everything below |
| `ISAAC_NUCLEUS_DIR` | `.../Isaac` | props, environments, sensors |
| `ISAACLAB_NUCLEUS_DIR` | `.../Isaac/IsaacLab` | the robots and the pre-trained policies |

### URDF in, USD out

Isaac Sim does not load URDF at runtime. It converts, once, and then loads USD. Three converters ship in `isaaclab/sim/converters/`, and each has a CLI wrapper in `scripts/tools/`:

```bash
# URDF (ROS robots)
./isaaclab.sh -p scripts/tools/convert_urdf.py my_robot.urdf my_robot.usd \
    --joint-stiffness 0.0 --joint-damping 0.0 --merge-joints

# MJCF (anything from the MuJoCo episode next door)
./isaaclab.sh -p scripts/tools/convert_mjcf.py my_robot.xml my_robot.usd

# A single mesh, for props and obstacles rather than articulations
./isaaclab.sh -p scripts/tools/convert_mesh.py crate.obj crate.usd --collision-approximation convexDecomposition
```

The MJCF one is the interesting door: an MJX model from `topics/rl-mujoco` converts into a USD that Isaac Lab can drive, which makes a genuine cross-simulator comparison possible on the same articulation rather than on two people's idea of the same robot.

`--merge-joints` is the flag that bites. URDF authors routinely insert fixed joints as naming scaffolding, and each one that survives becomes an articulation link that PhysX has to solve. Merging them is usually free and occasionally destroys a frame something else refers to by name.

### What an `ArticulationCfg` is made of

This is the Go2, trimmed to its load-bearing parts, and it is the file you write when you bring your own robot:

```python
UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,           # without this, contact sensors read zero, silently
        rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, ...),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,       # cheap, and the reason legs pass through each other
            solver_position_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),                     # the number the leap profile reads to scale the gap ladder
        joint_pos={".*L_hip_joint": 0.1, "F[L,R]_thigh_joint": 0.8, ".*_calf_joint": -1.5, ...},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={"base_legs": DCMotorCfg(joint_names_expr=[".*_hip_joint", ...], effort_limit=23.5, ...)},
)
```

Four things in there are worth calling out because they are silent when wrong:

- **`activate_contact_sensors=True` is a spawn-time flag, not a sensor setting.** Forget it and every `ContactSensor` in the scene reports zero forces forever. Section 8's whole flight measurement, and the `base_contact` termination, are downstream of this one boolean.
- **Joint names are regex.** `.*_calf_joint` matches four joints on a quadruped and zero on a typo, and zero matches is not an error. Print `robot.joint_names` once after import and check the count.
- **`init_state.pos[2]` is a real API.** `spark_envs._leap_profile` reads it to scale the gap ladder per robot, which is only legitimate because every config in `isaaclab_assets` sets it to the nominal standing height.
- **`enabled_self_collisions=False` is the default for a reason.** Self-collision is expensive and, on a legged robot at 4096 envs, mostly buys you a policy that has learned not to cross its own legs. Turn it on when the task is manipulation.

### Actuators are the sim-to-real knob

The `actuators` dict is where a simulated robot stops matching a real one, and Isaac Lab gives you a ladder of fidelity rather than a switch:

| Model | What it does | Cost |
|---|---|---|
| `ImplicitActuatorCfg` | PhysX solves the PD internally. Stiffness and damping are solver terms | free, and the least realistic |
| `IdealPDActuatorCfg` | PD computed explicitly, torque applied as an external force | cheap |
| `DCMotorCfg` | adds a torque-speed curve: `saturation_effort`, `velocity_limit` | cheap, and what Go2/G1/H1 use |
| `DelayedPDActuatorCfg` | adds a randomised action delay, in steps | cheap, and closer to a real control loop |
| `RemotizedPDActuatorCfg` | for linkage-driven joints where the effective gear ratio varies with angle | Cassie and Digit knees |
| `ActuatorNetLSTMCfg` / `ActuatorNetMLPCfg` | a learned network from `(pos error, velocity, history)` to measured torque | one forward pass per step |

The Anymals are the only robots here that use the learned one (`anymal.py:45`), and they load it from Nucleus as TorchScript:

```python
ANYDRIVE_3_LSTM_ACTUATOR_CFG = ActuatorNetLSTMCfg(
    network_file=f"{ISAACLAB_NUCLEUS_DIR}/ActuatorNets/ANYbotics/anydrive_3_lstm_jit.pt",
    ...
)
```

That is the ANYbotics sim-to-real result in a config field: they measured a real ANYdrive under load, fit a network to it, and now every simulated ANYmal in Isaac Lab inherits a drivetrain that lags and saturates the way the hardware does. If you ever wonder why the Anymals feel different to train than the Unitrees on identical terrain, this line is most of the answer.

### Adding a robot to this repo

Once a robot has an `ArticulationCfg` and a registered rough-terrain task, it costs one line here:

```python
# spark_envs.py
BASE_TASKS = {
    ...,
    "my_robot": "Isaac-Velocity-Rough-MyRobot-v0",
}
ROBOT_LABELS = {..., "my_robot": "MyRobot"}
```

and `register()` derives eight tasks from it, the leap profile scales its own gap ladder off the new robot's standing height, and `record.py` films it with the same chase camera. That property is the entire reason `spark_envs.py` derives rather than copies.

---

## 12. Sensors, and the two workflows

### The sensor menu

Section 2 introduces the height scanner because the stock task uses it. The `isaaclab.sensors` package ships eight; these are the six worth knowing for a locomotion task, and this repo already touches three: `record.py` creates its own cameras, and reads the height scanner and the contact sensor that the environment already owns.

| Sensor | What it is | Used here |
|---|---|---|
| `RayCaster` | ray casts against a static mesh. The height scanner is this, pointed down | the `height_scan` observation |
| `RayCasterCamera` | the same machinery arranged as an image plane. Depth without the renderer | not yet, and it should be |
| `ContactSensor` | per-body net contact force and air time | `base_contact`, `feet_air_time`, and the flight measurement |
| `Camera` / `TiledCamera` | real RTX rendering. `TiledCamera` batches all envs into one render pass | `record.py`, for the chase and onboard views |
| `Imu` | linear acceleration and angular velocity at a body frame | no, and it is what a real robot actually has |
| `FrameTransformer` | relative pose between named frames | no |

**The height scanner is a cheat sensor and it is worth being explicit about that.** It ray casts against the terrain mesh, which means it returns exact ground height through the robot's own body, in the dark, with no noise beyond the uniform noise the config adds. No Go2 has one. Everything in section 8 is measured with it, so every number in this repo is an upper bound on what the same policy would do on hardware. Section 14 has the two ways out.

**`TiledCamera` versus `Camera` is not a style choice.** `Camera` renders one viewport per instance. `TiledCamera` renders every environment into a single tiled image in one pass, which is the difference between filming one robot and training 4096 policies from pixels. If a vision task ever lands in this repo, it is `TiledCamera` or nothing.

**`RayCasterCamera` gives you depth without the renderer.** That matters here more than anywhere, because section 9 is the story of a pod that had CUDA but no Vulkan. A ray-cast depth image is computed in Warp against the terrain mesh, so it runs in a training pod that cannot start the RTX renderer at all. `dexsuite` ships both variants side by side (`config/kuka_allegro/camera_cfg.py`: `depth64` on a `TiledCamera`, `raycaster_depth64` on a `RayCasterCamera`) which makes it the reference for choosing between them.

### Manager-based and direct, and why this repo is manager-based

Everything in this README so far is the **manager-based** workflow: the environment is a pile of config classes, and eight managers (`Action`, `Observation`, `Reward`, `Termination`, `Command`, `Curriculum`, `Event`, `Recorder`) assemble the MDP out of named terms at startup. Roughly half of Isaac Lab's tasks are not written that way.

The **direct** workflow is a single class subclassing `DirectRLEnvCfg` and `DirectRLEnv`, where you write `_get_observations`, `_get_rewards`, `_get_dones` and `_reset_idx` as methods. No managers, no terms, no registry of named rewards.

| | Manager-based | Direct |
|---|---|---|
| The reward is | a dict of named terms with weights | one method returning a tensor |
| Changing it means | editing a config field | editing code |
| You get for free | per-term logging (`Episode_Reward/feet_air_time`), curricula, events | nothing |
| It costs you | eight managers of indirection | writing the plumbing yourself |
| Used by | all locomotion, navigation, locomanipulation, dexsuite | AMP, factory, forge, the hands, quadcopter |

This repo is manager-based because it had to be. The entire leap profile is fifty lines that reach into `cfg.rewards.lin_vel_z_l2.weight` and `cfg.terminations`, and the whole live report is built on rsl_rl logging one scalar per reward term. In a direct env, `REWARD_PROFILES` would be a fork of the environment class, and section 8's "the air-time term sat at -0.0033 the whole way" would have been a print statement someone had to remember to add.

Choose direct when the reward is genuinely one computation over intermediate state that no term boundary respects. Factory's keypoint distance is the honest example: three squashing kernels over one distance that is expensive to compute, and splitting it into terms would compute it three times.

---

## 13. The rest of Isaac Lab, and the reward that defines each task

Six task families ship alongside the velocity tasks this repo builds on. Each one is a different answer to "what is the objective", which is exactly the question section 8 ran aground on, so they are worth reading as a menu of ways out rather than as a catalogue.

Everything below is registered in a stock Isaac Lab install. No new download.

### Navigation: pay for arriving, not for velocity

`Isaac-Navigation-Flat-Anymal-C-v0`, manager-based, and the smallest complete environment in Isaac Lab. Four rewards:

```python
# navigation/config/anymal_c/navigation_env_cfg.py:77
termination_penalty          = RewTerm(mdp.is_terminated,                weight=-400.0)
position_tracking            = RewTerm(mdp.position_command_error_tanh,  weight=0.5,  params={"std": 2.0})
position_tracking_fine_grained = RewTerm(mdp.position_command_error_tanh, weight=0.5, params={"std": 0.2})
orientation_tracking         = RewTerm(mdp.heading_command_error_abs,    weight=-0.2)
```

and `position_command_error_tanh` is four lines:

```python
distance = torch.linalg.norm(command[:, :3], dim=1)
return 1 - torch.tanh(distance / std)
```

**The coarse-plus-fine pair is the whole trick.** One term at `std=2.0` is a gradient that reaches from three metres out; one at `std=0.2` is nearly flat until the last half metre and then very steep. Together they say "get closer" everywhere and "get exact" at the end, without either being a sparse reward. Every goal-reaching task in Isaac Lab does this, and section 14 is where it comes back.

The other half of this env is the action space, and it is the part worth stealing:

```python
# navigation/config/anymal_c/navigation_env_cfg.py:52
pre_trained_policy_action = mdp.PreTrainedPolicyActionCfg(
    policy_path=f"{ISAACLAB_NUCLEUS_DIR}/Policies/ANYmal-C/Blind/policy.pt",
    low_level_decimation=4,
    low_level_actions=LOW_LEVEL_ENV_CFG.actions.joint_pos,
    low_level_observations=LOW_LEVEL_ENV_CFG.observations.policy,
)
```

The action term `torch.jit.load`s a walking policy and the high-level policy's action **is the velocity command** fed to it. Rates fall out of the decimation: the low level runs at 50 Hz, `decimation = LOW_LEVEL_ENV_CFG.decimation * 10` puts the high level at 5 Hz. So the observation is three terms (base velocity, gravity, the pose command), the episode is 8 seconds, and the thing trains in minutes because it is a tiny MDP wrapped around a policy that already works.

That is directly buildable here: the leap policy is a checkpoint, `train.py` already exports TorchScript alongside ONNX, and pointing `policy_path` at it turns this repo's walking result into somebody else's action space.

### Humanoid AMP: no reward function at all

`Isaac-Humanoid-AMP-{Walk,Run,Dance}-Direct-v0`, direct workflow, and the reason to read it is this method:

```python
# direct/humanoid_amp/humanoid_amp_env.py:107
def _get_rewards(self) -> torch.Tensor:
    return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
```

A constant. Then in the agent config:

```yaml
# agents/skrl_walk_amp_cfg.yaml
task_reward_scale: 0.0
style_reward_scale: 2.0
```

The constant is multiplied by zero. **One hundred percent of the learning signal comes from a discriminator**, a third network next to the policy and the value function, trained to tell "a window of this policy's motion" from "a window of the reference motion". The policy's reward is how well it fools that discriminator. The reference is a `.npz` of joint positions, velocities and body poses in `motions/`, sampled at random times by `MotionLoader`, and the env publishes a two-frame `amp_obs` window in `self.extras` for the agent to discriminate on.

**This is the direct answer to the wall in section 8.** There, the objective could not express a jump and no weight fixed it. Here there is no weight to fix: you show it a motion and the objective becomes "move like that". A leap from mocap, or from a MuJoCo rollout, or from a hand-animated key sequence, needs no reward engineering at all.

Two costs, and both are real for this repo. It is **skrl-only**: the only agent entry point registered is `skrl_amp_cfg_entry_point`, so `train.py` and every chart in the live report would need a second scraper. And the reward number stops meaning anything, because it is a discriminator score against a moving opponent, so "mean reward went up" is no longer evidence of anything. The measurement would have to be the flight-phase one from section 8, which this repo already has.

### Locomanipulation: extend a walking reward set with a task

`Isaac-Tracking-LocoManip-Digit-v0` inherits the Digit walking rewards and adds end-effector tracking on top:

```python
# locomanipulation/tracking/config/digit/loco_manip_env_cfg.py:24
class DigitLocoManipRewards(DigitRewards):
    joint_deviation_arms = None                                       # delete an inherited term
    left_ee_pos_tracking          = RewTerm(position_command_error,      weight=-2.0)
    left_ee_pos_tracking_fine_grained = RewTerm(position_command_error_tanh, weight=2.0, params={"std": 0.05})
    left_end_effector_orientation_tracking = RewTerm(orientation_command_error, weight=-0.2)
    # and the same three for the right arm
```

Coarse plus fine again, at `std=0.05` this time because a wrist has to land within centimetres. Note `joint_deviation_arms = None`: **setting an inherited term to `None` deletes it**, which is the same idiom H1 uses on `lin_vel_z_l2` and which `_leap_profile` already guards against. The `__post_init__` then flattens the terrain, removes the height scanner and drops the terrain curriculum, because a Digit balancing two wrist targets does not need holes as well.

`Isaac-PickPlace-Locomanipulation-G1-Abs-v0` is the odd one out and worth knowing about precisely because of what it is not: it has **no reward terms at all**. Its only agent entry point is `robomimic_bc_cfg_entry_point` pointing at `bc_rnn_low_dim.json`. It is behaviour cloning from teleoperated demonstrations recorded with `scripts/tools/record_demos.py`, not RL. Isaac Lab is not only an RL framework, and this is the task that proves it.

### Drone navigation: the same reward, a different embodiment

`Isaac-Navigation-3DObstacles-ARL-Robot-1-v0`:

```python
# drone_arl/navigation/config/arl_robot_1/navigation_env_cfg.py:244
goal_dist_exp1    = RewTerm(distance_to_goal_exp_curriculum,   weight=2.0, params={"std": 7.0})
goal_dist_exp2    = RewTerm(distance_to_goal_exp_curriculum,   weight=4.0, params={"std": 0.5})
velocity_reward   = RewTerm(velocity_to_goal_reward_curriculum, weight=0.5)
action_rate_l2    = RewTerm(mdp.action_rate_l2,                weight=-0.05)
action_magnitude_l2 = RewTerm(mdp.action_l2,                   weight=-0.05)
termination_penalty = RewTerm(mdp.is_terminated,               weight=-100.0)
```

Coarse at `std=7.0`, fine at `std=0.5`, a term paying for velocity pointed at the goal, two smoothness penalties, and a large penalty for hitting anything. That is the same skeleton as the ANYmal navigation env on a completely different robot, which is the point of including it: **goal-reaching rewards are a fixed shape, and terrain, gravity and embodiment are the variables.** The `_curriculum` suffix on the first three is a per-term curriculum, distinct from the terrain curriculum this repo uses.

### Factory and Forge: contact-rich assembly, and the nested-kernel reward

`Isaac-Factory-PegInsert-Direct-v0` and friends, direct workflow. The reward is a distance between two sets of keypoints, one rigidly attached to the held part and one to the target pose, pushed through **three** squashing kernels at different scales:

```python
# direct/factory/factory_tasks_cfg.py:79
keypoint_coef_baseline = [5, 4]      # general movement toward the fixed object
keypoint_coef_coarse   = [50, 2]     # aligning the assets
keypoint_coef_fine     = [100, 0]    # the last inch, or threading
```

plus two binary terms, `curr_engaged` and `curr_success`, that fire on thresholds. Same coarse-to-fine idea as the navigation tasks, one rung deeper, because peg insertion has three genuinely different phases and one kernel cannot be steep enough for the third without being flat in the first.

Forge adds force sensing and contact-aware control on the same task family. If the interest is industrial rather than legged, this is where Isaac earns its keep over MuJoCo, and the direct workflow stops looking like extra work.

### Dexsuite: the asymmetric actor-critic, and the template for vision

`Isaac-Dexsuite-Kuka-Allegro-{Lift,Reorient}-v0`, manager-based, and its reward is a clean staircase:

```python
# manipulation/dexsuite/dexsuite_env_cfg.py:361
fingers_to_object    = RewTerm(mdp.object_ee_distance,             weight=1.0,  params={"std": 0.4})
position_tracking    = RewTerm(mdp.position_command_error_tanh,    weight=2.0,  params={"std": 0.2})
orientation_tracking = RewTerm(mdp.orientation_command_error_tanh, weight=4.0,  params={"std": 1.5})
success              = RewTerm(mdp.success_reward,                 weight=10)
action_l2 / action_rate_l2                                         weight=-0.005 each
```

Reach the object, then move it to a pose, then a large bonus for being there. But the reason to open this file is the agent config:

```python
# manipulation/dexsuite/config/kuka_allegro/agents/rsl_rl_ppo_cfg.py:86
obs_groups={"actor": ["policy", "proprio", "base_image"], "critic": ["policy", "proprio", "perception"]},
actor=CNN_POLICY_CFG,
```

**The actor sees a depth image and the critic sees privileged state.** That is asymmetric actor-critic, in two lines of config, using `RslRlCNNModelCfg` for the encoder, and it is the exact shape a vision-based locomotion policy needs: the value function may cheat during training because it is thrown away at deployment; the policy may not, because it has to run on the robot. This is the only worked example of it in Isaac Lab, and section 14 points back at it.

Alongside it, `dexsuite/adr_curriculum.py` is automatic domain randomization: a `DifficultyScheduler` curriculum term that climbs with success rate, and then a stack of `mdp.modify_term_cfg` terms that widen observation noise and physics ranges as it does. `modify_term_cfg` is the general mechanism and it is worth knowing on its own: it takes an `address` string like `"observations.proprio.joint_pos.noise.n_min"` and rewrites that config field mid-run, which means **any** field in the environment config can be put on a curriculum without writing a curriculum function for it.

### The one-line summary of all of it

| Task family | Workflow | Where the objective lives |
|---|---|---|
| Velocity (this repo) | manager | ~12 reward terms, hand-weighted, tracking a commanded velocity |
| Navigation | manager | 4 terms: coarse and fine distance to a goal pose |
| Drone navigation | manager | the same 4, plus per-term curricula |
| Locomanipulation tracking | manager | a walking reward set plus 6 end-effector tracking terms |
| Pick-place (G1) | manager | nowhere. Behaviour cloning from demonstrations |
| Humanoid AMP | direct | a discriminator against mocap. The env's reward is a constant |
| Factory / Forge | direct | three nested kernels over one keypoint distance |
| Dexsuite | manager | a reach-then-place staircase, with an asymmetric actor-critic |

---

## 14. What to build next here

Section 8 ends on a wall: velocity tracking cannot express a jump, and 2900 flat iterations say more training will not change that. Everything below is a way past it or around it, ordered by what it costs against what it settles. All of it is config in this repo plus machinery that already exists in Isaac Lab.

### The cheap experiment: an exploration bonus

rsl_rl ships Random Network Distillation and **nothing in Isaac Lab uses it**. `grep -rn RslRlRndCfg source/isaaclab_tasks` returns nothing, and yet the field is right there on the PPO config (`isaaclab_rl/rsl_rl/rl_cfg.py:217`).

RND keeps two networks over an observation slice, one frozen random target and one predictor trained to match it, and pays an intrinsic reward equal to how badly the predictor misses (`rsl_rl/extensions/rnd.py:163`). States the agent has visited often are predicted well and pay nothing; novel states pay. That is aimed precisely at the sentence section 8 ends on: *PPO will not find that first crossing by chance when every failed attempt terminates the episode.* Being airborne over a trench is the most novel state on the course.

It is an `AGENT_PROFILES` dict next to `REWARD_PROFILES`, keyed the same way:

```python
algorithm.rnd_cfg = RslRlRndCfg(
    weight=0.005,
    weight_schedule=RslRlRndCfg.LinearWeightScheduleCfg(  # curiosity early, task reward late
        final_value=0.0, initial_step=0, final_step=1500,
    ),
    reward_normalization=True,
    state_normalization=True,
)
obs_groups = {"actor": ["policy"], "critic": ["policy"], "rnd_state": ["rnd_state"]}
```

The one non-obvious part is that last line: `get_rnd_state` reads `self.obs_groups["rnd_state"]`, so the key has to exist and the environment has to publish a matching observation group. Give it the base linear velocity and the height scan and it is curious about *where the robot is relative to the ground*, which is the right thing to be curious about here. Give it the whole policy observation and it will be curious about joint noise.

Cost: one 3-hour run against a baseline that already has published numbers. It may not work. That is still a result, and it is a falsifiable one, which is more than "try more weights" ever was.

### The real fix: ask for the jump

Section 8's third option, `Spark-Vault-*`, keeps the velocity objective and gives the reward the height scanner, which buys a gradient for an attempt without a new task family. Both shapes below go further and replace the objective outright. They are still worth building, and the `vault` result is what says how much they are needed.

Two shapes, and the second is the one to build:

**Hierarchical**, the navigation env's shape. Freeze the converged leap policy, load it with `PreTrainedPolicyActionCfg`, and train a 5 Hz policy whose action is the velocity command. Cheap to train and directly reuses a checkpoint this repo already produced. It inherits the ceiling, though: the low level still cannot jump, so the high level learns to route around trenches rather than over them. Worth building as a demo, not as an answer.

**A goal command on the flat task**, which is the answer. Replace `base_velocity` with a pose command on the far side of a trench and pay coarse-plus-fine for arriving, exactly as section 13's four navigation terms do:

```python
cfg.commands.target = mdp.UniformPose2dCommandCfg(...)
cfg.rewards.reach        = RewTerm(position_command_error_tanh, weight=1.0, params={"std": 2.0})
cfg.rewards.reach_fine   = RewTerm(position_command_error_tanh, weight=1.0, params={"std": 0.2})
```

The difference from the current leap profile is not the size of the numbers. It is that **crossing a trench now has a gradient before it succeeds**: a robot that gets halfway across and falls scores better than one that never left the lip, where under velocity tracking it scored worse. That is the property PPO needs and the one the current task cannot provide at any weight.

This is a new task family rather than a profile, which is exactly what `REWARD_PROFILES` being keyed by terrain was designed to make cheap. `Spark-Leapgoal-*` alongside `Spark-Leap-*`, sharing the terrain and nothing else.

### Take the cheat sensor away

Every result in this repo is measured with a height scanner that ray casts exact ground truth through the robot's own body. Two supported ways to earn it back, both listed in section 12:

**Distill it.** rsl_rl has a full distillation runner and Isaac Lab registers `rsl_rl_distillation_cfg_entry_point` on the ANYmal-D. Read that file first and then ignore its shape: it maps `obs_groups = {"student": ["policy"], "teacher": ["policy"]}`, the same group twice, so it demonstrates plumbing and teaches nothing. The version worth building points the teacher at the height scan and the student at proprioception plus a short history, then measures the gap. That number, "how much of the 0.245 m survives without the scanner", is the most interesting single measurement left in this project.

**Or give it eyes.** `RslRlCNNModelCfg` plus the asymmetric `obs_groups` from dexsuite, with a `RayCasterCamera` rather than a `TiledCamera` so it runs in a pod that has CUDA and no Vulkan. Considerably more work, and the more honest demo.

### Three knobs that are simply not wired up

- **Symmetry augmentation.** `RslRlSymmetryCfg` (`rl_cfg.py:220`) mirrors observations and actions to get four samples per rollout step at no simulation cost. `mdp/symmetry/anymal.py` is the only implementation in the tree, so a Go2 version means writing `compute_symmetric_states` against its joint order: left-right, front-back, and the diagonal. It is registered on the Anymals as `rsl_rl_with_symmetry_cfg_entry_point`, and Isaac Lab's `train.py` selects it with `--agent`. **`spark_envs.register()` only ever emits `rsl_rl_cfg_entry_point`**, so nothing derived here can currently be reached by `--agent` at all. That is a five-line fix in the `gym.register` kwargs and it unlocks the recurrent and distillation variants at the same time.
- **A recurrent policy.** `RslRlPpoActorCriticRecurrentCfg`, one entry point away once the above is fixed. The case for it here is concrete: a trench passes out of the scanner's view before the front feet reach it.
- **A command curriculum.** The terrain gets promoted a row at a time; the velocity command range never moves. `_leap_profile` sets `lin_vel_x = (-1.0, 2.0)` at iteration 1 and the policy spends its first few hundred iterations being asked for 2 m/s it cannot produce on ground it cannot cross. A `CurriculumTermCfg` that widens the range with the terrain level is a dozen lines, and with `mdp.modify_term_cfg` (section 13, dexsuite) addressing `"commands.base_velocity.ranges.lin_vel_x"` it is closer to three.

And one measurement, not a feature: **three seeds**. The headline claim, that row 6.6 is a converged ceiling rather than an unlucky run, currently rests on one run per configuration. Three seeds serialise fine overnight on one GPU and would make it unarguable.

### Getting the policy off the box

Worth stating because this repo exports a policy every 50 iterations and never says what for. `train.py` writes TorchScript and ONNX next to each checkpoint, which is enough to load somewhere else and not enough to *run* somewhere else: an exported network is a function from a tensor to a tensor, and everything that gives those tensors meaning (which observation terms, in which order, with which scaling, and how the actions map back onto joint targets) lives in the environment config that did not come with it.

Isaac Lab's answer is **LEAPP** (Lightweight Export Annotations for Policy Pipelines), and it is a good fit here for a narrow reason: it supports manager-based environments trained with rsl_rl, and nothing else. That is exactly this repo.

```bash
./isaaclab.sh -p -m pip install leapp
python scripts/reinforcement_learning/leapp/rsl_rl/export.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 --checkpoint /path/to/model_4950.pt
python scripts/reinforcement_learning/leapp/deploy.py \
    --task Isaac-Velocity-Rough-Unitree-Go2-v0 --leapp_model exported/model.yaml
```

The export writes the policy plus a YAML describing its input and output semantics, and `deploy.py` runs that bundle back through `LeappDeploymentEnv`, which is the cheapest possible check that the export did not silently reorder an observation. The related `--export_io_descriptors` flag on Isaac Lab's `train.py` dumps the same semantics at training time and is also unused here.

**Those commands name a stock task on purpose, and that is the catch.** Neither `export.py` nor `deploy.py` accepts `--external_callback`; only `train.py` and `play.py` do (`grep -c external_callback` over the four scripts gives 4, 4, 0, 0). So `--task Spark-Leap-Go2-v0` does not resolve in either of them, because nothing in that process ever calls `spark_envs.register()`. Section 4's no-fork extension hook buys you training and replay, and stops at the door of the deployment path. The fix is a two-line patch to those scripts or a `sitecustomize` that registers on import, not a redesign, but it is much cheaper to find written down here than at the end of a training run.

---

## Reference

**The platform**

- [Isaac Sim](https://github.com/isaac-sim/IsaacSim) and its [docs](https://docs.isaacsim.omniverse.nvidia.com/)
- [Isaac Lab](https://github.com/isaac-sim/IsaacLab), and the [manager-based env tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_manager_rl_env.html)
- [The direct workflow tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/03_envs/create_direct_rl_env.html), which is section 12's other half
- [Adding your own robot](https://isaac-sim.github.io/IsaacLab/main/source/how-to/write_articulation_cfg.html) and the [URDF importer](https://docs.isaacsim.omniverse.nvidia.com/latest/robot_setup/import_urdf.html), for section 11
- [OpenUSD](https://openusd.org/)
- [Warp](https://github.com/NVIDIA/warp) and Newton

**The learning half**

- [rsl_rl](https://github.com/leggedrobotics/rsl_rl)
- [LEAPP](https://github.com/nvidia-isaac/leapp), the export path in section 14, and its [Isaac Lab guide](https://isaac-sim.github.io/IsaacLab/main/source/policy_deployment/05_leapp/exporting_policies_with_leapp.html)
- Curiosity-driven exploration for legged robots ([Schwarke et al., 2023](https://proceedings.mlr.press/v229/schwarke23a.html)), which is the RND that `RslRlRndCfg` implements
- Symmetry considerations in RL for legged locomotion ([Mittal et al., 2024](https://arxiv.org/abs/2403.04359)), behind `RslRlSymmetryCfg`
- AMP: Adversarial Motion Priors ([Peng et al., 2021](https://xbpeng.github.io/projects/AMP/)), the objective behind the humanoid tasks in section 13
- [robomimic](https://robomimic.github.io/), the imitation-learning path the G1 pick-place task uses instead of a reward

**Next door and further out**

- The MuJoCo episode next door: [`topics/rl-mujoco`](../rl-mujoco)
- Extreme Parkour ([paper](https://extreme-parkour.github.io/)), for what a purpose-built parkour reward looks like when velocity tracking is not the objective
