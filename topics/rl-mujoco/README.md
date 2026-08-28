# Reinforcement Learning in MuJoCo: teaching a humanoid to walk

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This one is about reinforcement learning in [MuJoCo](https://github.com/google-deepmind/mujoco), the physics engine that most robotics RL runs on. The goal is narrow and honest: get a **Unitree G1 humanoid to actually walk**, on one DGX Spark, and be able to watch it happen.

Humanoid locomotion is the standard hard case. A quadruped is stable enough that a mediocre policy still looks like walking; a humanoid falls over, and every shortcut in your reward function shows up immediately as a robot that shuffles, hops, or dives forward to farm the velocity term.

The first half of this README is a tutorial: what MuJoCo is, how a physics step works, how a simulator becomes an RL environment, and why the GPU version (MJX + Brax) is the thing that makes a humanoid reachable at all. The second half is the demo itself: what it runs, what it measured, and every trap that cost real time on this box.

**Contents**

1. [How MuJoCo works](#1-how-mujoco-works)
2. [From a simulator to an RL environment](#2-from-a-simulator-to-an-rl-environment)
3. [Why this needs a GPU: MJX](#3-why-this-needs-a-gpu-mjx)
4. [The learning half: PPO and Brax](#4-the-learning-half-ppo-and-brax)
5. [Reading this repo](#5-reading-this-repo)
6. [The demo](#6-the-demo) and how to run it
7. [Things that cost real debugging time on this box](#7-things-that-cost-real-debugging-time-on-this-box)
8. [A guided tour of the code](#8-a-guided-tour-of-the-code), if you are presenting it

---

## 1. How MuJoCo works

MuJoCo (Multi-Joint dynamics with Contact) is a rigid-body physics engine. It answers one question, very fast, over and over: given where every body is right now, how fast each is moving, and what forces the actuators are applying, where is everything 2 milliseconds from now?

Everything else in this repo is a consequence of that one question being cheap enough to ask a few hundred million times.

### Two objects, and only two

```python
model = mujoco.MjModel.from_xml_path("scene.xml")   # constants
data  = mujoco.MjData(model)                        # state
mujoco.mj_step(model, data)                         # advance one timestep
```

- **`mjModel`** is everything that does not change while the simulation runs: link geometry, masses and inertias, joint definitions and limits, actuator gains, sensor definitions, solver settings. It is compiled once from an MJCF XML file and is read-only afterwards.
- **`mjData`** is everything that does change: `qpos` (positions), `qvel` (velocities), `ctrl` (what you command the actuators to do), `sensordata`, the current contact list, `time`. A step reads `data`, and writes the next `data`.

That split matters more than it looks. Every batching trick in Part 3 works because the model is shared across thousands of simulations and only the data has to be duplicated.

### A whole simulation in 20 lines

This runs against nothing but `pip install mujoco`. A rod on a hinge, one motor, one second of simulated time:

```python
import mujoco

XML = """
<mujoco>
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="floor" type="plane" size="5 5 .1"/>
    <body name="pole" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom name="rod" type="capsule" fromto="0 0 0  0 0 -0.5" size="0.04"/>
    </body>
  </worldbody>
  <actuator>
    <motor joint="hinge" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
data  = mujoco.MjData(model)

data.qpos[0] = 0.3                          # start tilted 0.3 rad
for _ in range(500):                        # 500 * 2ms = 1 simulated second
    data.ctrl[0] = -0.5 * data.qvel[0]      # a one-line "policy": damping
    mujoco.mj_step(model, data)

print(model.nq, model.nv, model.nu)
print(f"angle {data.qpos[0]:+.3f} rad, vel {data.qvel[0]:+.3f} rad/s, t={data.time:.2f}s")
```

```
1 1 1
angle +0.028 rad, vel +0.525 rad/s, t=1.00s
```

That loop is the entire RL setup in miniature. Replace the one-line damping controller with a neural network, replace "did the rod stay up" with a reward function, and you have this project. Everything from here is scale and bookkeeping.

### Generalized coordinates, and why the G1 code slices at `[7:]`

MuJoCo does not track each body's world position. It tracks *joint* coordinates, and derives world positions from them. So `qpos` is one number per positional degree of freedom, `qvel` one per velocity degree of freedom, and those two counts differ whenever a free-floating body is involved: a free body's orientation needs 4 numbers as a quaternion in `qpos` but only 3 as an angular velocity in `qvel`.

The G1 is a floating-base robot, so its layout is:

```
qpos = [ x y z  qw qx qy qz | 29 joint angles      ]   nq = 36
qvel = [ vx vy vz  wx wy wz | 29 joint velocities  ]   nv = 35
```

This is why Playground's env code says `joint_angles = data.qpos[7:]` and `joint_vel = data.qvel[6:]`. It is skipping the floating base to get the actuated joints. When you see a stray 7 or 6 in robotics simulation code, this is nearly always what it is.

### What actually happens inside `mj_step`

1. **Forward kinematics.** Turn joint coordinates into world positions and orientations for every body, geom, and site.
2. **Inertia and bias.** Build the mass matrix, then compute the forces that exist without any actuation: gravity, Coriolis, centrifugal, passive springs and damping.
3. **Collision detection.** Find geom pairs that are touching or nearly touching, and produce a contact list.
4. **Constraint solve.** Contacts, joint limits, friction and equality constraints all become rows in one constraint problem, which MuJoCo solves as a convex optimization. This is the expensive step and the interesting one: MuJoCo's contacts are *soft*, so instead of instantaneous impulses you get a well-behaved, differentiable-ish system that a solver can chew through quickly. That is a large part of why this engine, rather than a game physics engine, is the one robotics RL standardized on.
5. **Integrate.** Apply the resulting accelerations to advance `qvel` then `qpos` by `timestep`.

Step 4 is where the `njmax` trap in this repo comes from. The solver's constraint buffer is a fixed size, `njmax` rows, chosen up front. A humanoid standing on two feet generates a lot of rows: Playground's default for this env is `njmax = 29*2 + 8*4 = 90`, and on this box we measured overflow at 93 within the first rollout. See [the gotchas](#7-things-that-cost-real-debugging-time-on-this-box); the short version is that overflow silently drops constraints instead of raising, so the feet start sinking through the floor and you are quietly training on different physics than you think.

### Actuators: the policy does not output torque

An MJCF `<motor>` applies its `ctrl` value as a raw torque, which is what the toy example above uses. The G1 does not. All 29 of its actuators are `<position>` actuators: `ctrl` is a *target joint angle*, and the actuator applies `kp * (target - q)` to chase it (`kp` is 75 on the hip pitch, for instance), with the damping half of the PD loop coming from each joint's own `damping` attribute rather than the actuator.

This is the single most important thing to understand about the action space, because it means the policy is not learning "how much torque for each of 29 joints at 50 Hz", which is a brutally hard credit-assignment problem. It is learning "what pose should I be aiming for", and a stiff low-level controller handles the rest. Real humanoids are commanded the same way, so this also keeps the policy deployable.

### Sensors, and the renderer

The model declares sensors (gyro, accelerometer, foot-contact flags, foot velocities) and `data.sensordata` holds their readings. That is not decoration: it is the line between what a real robot could measure and what only the simulator knows, which is exactly the distinction the asymmetric actor-critic in Part 4 is built on.

Rendering is a separate subsystem (`mujoco.Renderer`, an OpenGL/EGL context) and completely optional. Nothing in the physics knows or cares whether you are drawing. This repo never renders during training on the GPU; it renders in a separate CPU task, for the same reason.

---

## 2. From a simulator to an RL environment

A physics engine steps state. Reinforcement learning needs a *decision problem*. The gap between them is small and entirely conventional:

| RL concept | What supplies it | For the G1 |
|---|---|---|
| observation | a function of `mjData` | 103 numbers, sensor-shaped |
| action | written into `data.ctrl` | 29 joint-angle offsets |
| reward | a function of `mjData` + the action | weighted sum of ~24 terms |
| termination | a predicate on `mjData` | fell over, or 1000 steps elapsed |
| reset | re-initialize `mjData` | back to the `knees_bent` keyframe |

An "environment" is just those five things bolted onto `mj_step`. The loop the agent sees:

```python
state = env.reset(rng)
while not state.done:
    action = policy(state.obs)          # 29 numbers
    state = env.step(state, action)     # runs 10 physics steps
    total += state.reward
```

We do not hand-roll any of it. `mujoco_playground` ships DeepMind's own `G1JoystickFlatTerrain` / `G1JoystickRoughTerrain`, which is the config their published G1 results come from. Starting from something that demonstrably walks and *then* turning knobs is the whole lesson of the previous attempt at this demo.

### Two clocks: 50 Hz decisions over 500 Hz physics

```
sim_dt  = 0.002   physics timestep       500 Hz
ctrl_dt = 0.02    one env step           50 Hz   ->  10 physics substeps per action
episode_length = 1000 env steps           =  20 simulated seconds
```

The policy acts at 50 Hz; the physics runs at 500 Hz underneath it. Both halves of that are deliberate. The PD actuators and the contact solver need small timesteps to stay stable, so the physics has to be fast. But a policy that re-decided at 500 Hz would need ten times the samples to cover the same behavior, and no real robot's control stack runs a learned policy that fast anyway. 50 Hz is the number the hardware people use, so it is the number to train at.

### The action: 29 numbers, and what they mean

```python
motor_targets = default_pose + action * action_scale   # action_scale = 0.5
```

The network's output is *not* an absolute pose. It is an offset, scaled by 0.5 rad, from a fixed knees-bent standing pose stored as a keyframe in the model. So a policy that outputs all zeros stands still in a reasonable posture. That is a strong prior for free: the network starts at "stand", not at "flail", and exploration is exploration around a sane pose rather than around the origin of a 29-dimensional torque space.

### The observation: 103 numbers, and why exactly those

Measured on this box, `env.observation_size` is a dict, not a vector:

```
state             (103,)   what the POLICY sees      onboard-sensor shaped
privileged_state  (216,)   what the VALUE net sees   ground-truth sim state
```

The 103 breaks down as:

| Slice | Size | What it is |
|---|---|---|
| local linear velocity | 3 | noisy, pelvis frame |
| gyro | 3 | noisy angular velocity |
| gravity vector | 3 | which way is down, in body frame: this is the tilt sense |
| **command** | 3 | the joystick: target `[vx, vy, yaw]` |
| joint angles | 29 | relative to the default pose |
| joint velocities | 29 | |
| last action | 29 | so the policy can be smooth with respect to itself |
| gait phase | 4 | `cos`/`sin` of a clock, so it can learn a rhythm |

Every one of those is something a real G1 could measure, and each is corrupted at training time with the noise level in `noise_config`. `privileged_state` appends the *clean* versions plus things only a simulator knows: true global velocities, actuator forces, root height, per-foot contact flags and velocities, air time. Part 4 explains why the critic gets to cheat.

The **command** entry is what "joystick" means. Each episode samples a random target velocity (x in [-1, 1] m/s, y in [-0.5, 0.5], yaw in [-1, 1] rad/s) and the reward is for *tracking* it. The policy is not learning "walk forward", it is learning "go the speed and direction I am told". That is harder, and the result is steerable, which makes a much better demo than a robot that only walks north.

### The reward: two terms that want walking, twenty-two that decide how it looks

The reward is a weighted sum. The weights are DeepMind's, not ours: they live in Playground's own
`default_config()`, at `mujoco_playground/_src/locomotion/g1/joystick.py:53`, and nothing in this repo
declares them. Abridged and reordered here to put the task terms first (the real list is 24 keys, grouped
by joint / feet / energy / pose):

```python
tracking_lin_vel  =  1.0     # exp(-error^2 / 0.25): match commanded velocity
tracking_ang_vel  =  0.75    # match commanded yaw rate
feet_phase        =  1.0     # follow the gait clock
feet_air_time     =  2.0     # reward a foot spending time off the ground
feet_clearance    =  0.0     # OFF by default: reward the swing foot lifting
orientation       = -2.0     # penalize the torso tipping
termination       = -100.0   # penalize falling, hard
stand_still       = -1.0     # penalize shuffling when commanded zero
feet_slip         = -0.25    # penalize a planted foot sliding
dof_pos_limits    = -1.0     # penalize hitting joint limits
action_rate       =  0.0     # (available) penalize jerky action changes
energy            =  0.0     # (available) penalize power draw
...
```

Read the shape rather than the individual numbers. Of the 24 terms, exactly two describe the *task*: the tracking pair. Two or three more shape the gait into something that looks like walking. Everything else is a penalty that closes off some cheaper way of scoring the first two. Left unpenalized, a velocity-tracking reward is happily maximized by a robot that dives forward, or vibrates, or slides its feet without ever lifting them. Most of reward engineering for locomotion is playing whack-a-mole with those, which is exactly why this project treats the tuned baseline as the control and each [preset](#reward-presets) as a small labelled diff on top of it.

The `exp(-error^2 / sigma)` kernel on tracking is worth noting too: it is bounded and smooth, so it saturates as the policy gets close instead of paying out forever, and the gradient does not explode when the error is huge at the start.

---

## 3. Why this needs a GPU: MJX

Do the arithmetic first, because it is what decides the whole architecture.

DeepMind's recipe for this env is **200 million environment steps**. At 50 Hz that is 4 million simulated seconds, roughly **46 days of robot time**. Published humanoid locomotion policies land in the 10^8 to 10^9 range; this is not an unusually greedy number.

A single CPU core running MuJoCo on a 29-DoF humanoid does low thousands of steps per second. Even at 5,000 steps/s, 200M steps is 11 hours of pure single-core physics, and that is before a single gradient is computed. A previous version of this demo ran Gymnasium + PyTorch PPO across 5 workers at 32 vectorized envs each: **160 simulations**, on CPU. It got the robot to stand, shuffle, and fall over, and its own notes end with a table of six attempts whose last row is marked TBD.

The reward function was not really the problem. Sample count was.

### MJX: the same model, as pure functions

**MJX** is MuJoCo re-expressed so that a step is a *pure function* over JAX arrays: no mutation, no pointers, take a state and return a new one.

```python
mx = mjx.put_model(model)                          # constants to device
dx = mjx.put_data(model, mujoco.MjData(model))     # state to device
dx = mjx.step(mx, dx)                              # returns a NEW dx
```

That purity is the whole trick, because pure functions compose with JAX's transformations. `jax.vmap` turns "step one simulation" into "step N simulations" without you writing a line of batching code, and `jax.jit` compiles the result into one GPU program:

```python
N = 1024
keys = jax.random.split(jax.random.PRNGKey(0), N)
qpos = jax.vmap(lambda k: dx.qpos.at[0].set(jax.random.uniform(k, minval=-.5, maxval=.5)))(keys)
batch = jax.vmap(lambda q: dx.replace(qpos=q))(qpos)   # 1024 different starting states

step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))  # model shared, data batched
for _ in range(100):
    batch = step(mx, batch)

print(batch.qpos.shape)     # (1024, 1)
```

`in_axes=(None, 0)` is the shared-model/batched-data split from Part 1 made literal: one copy of the constants, 1024 copies of the state.

The payoff is not just parallelism, it is that the *entire* training loop, simulation plus reward plus advantage estimation plus the network update, compiles into a single XLA program. Nothing returns to Python between steps. There is no per-step host-to-device copy and no Python interpreter overhead in a loop that executes billions of times.

Measured on this box (GB10, sm_121, arm64) on 2026-08-04 with mujoco 3.11.0:

```
bare env.step
  1024 envs   32,005 env-steps/sec
  2048 envs   59,757 env-steps/sec      scales close to linearly

end-to-end PPO (gradient updates included), 1024 envs
  2,129,920 steps in 66s  ->  ~32,300 steps/sec
  one-off compile         ~26s before the first eval
```

At 1024 envs the gradient updates are essentially free next to the simulation, so the end-to-end rate matches the bare simulation rate. 200M steps lands on the order of an hour at 4096 envs. That is the difference between "a stream segment" and "next week".

### What MJX gives up

Nothing is free, and every one of these shows up somewhere in this repo:

- **Static shapes.** XLA needs to know every array size at compile time, so the constraint buffer cannot grow on demand. Hence `njmax` being a config value you have to set correctly rather than something the engine figures out.
- **No renderer.** MJX has no drawing code at all. Replays here rebuild the same env with `impl="jax"` on CPU and draw through stock MuJoCo's renderer.
- **Compile time.** The first step compiles a long list of kernels, about 40 seconds on this box. The first iteration always looks hung. It is not.
- **Branch-free control flow.** Python `if` on a traced value does not work; everything becomes `jp.where` and masks. This is invisible when you use Playground's envs, and very visible the moment you write your own.

One version note: in MuJoCo 3.11 MJX dispatches to [`mujoco_warp`](https://github.com/google-deepmind/mujoco_warp) kernels (`impl="warp"`) rather than pure XLA. The `impl="jax"` path still exists and is what the CPU replay uses, so in this repo the policy is *trained* on Warp and *replayed* on JAX. Two implementations of one model; they should agree closely, but they are not bit-identical.

---

## 4. The learning half: PPO and Brax

MJX gives you a fast, differentiable-shaped simulator. **Brax** supplies the RL algorithms written in the same JAX style, so the learner fuses into the same compiled program as the physics instead of sitting in Python above it.

### PPO in one paragraph

Proximal Policy Optimization keeps two networks: a **policy** (actor) that maps an observation to a Gaussian distribution over the 29 actions, and a **value** function (critic) that predicts how much total reward to expect from a state. Training goes in rounds. Roll out the current policy across every parallel env for a fixed number of steps; use the critic to turn the observed rewards into an *advantage* per action, meaning "how much better did this turn out than expected"; then take a few gradient steps that raise the probability of positive-advantage actions and lower the rest. The "proximal" part is a clip on how far the action probabilities are allowed to move in one round, which is what stops a single unlucky batch from destroying a policy that took an hour to train. An entropy bonus keeps the Gaussian from collapsing to a point too early, which is how the policy keeps exploring.

PPO is on-policy: the data is thrown away after each round. That is wasteful in sample terms and exactly why the batches are so large. When the simulator is nearly free, throwing away samples is the right trade.

### Brax's actual entry point

The whole learner is one function call. From `train.py`:

```python
make_inference_fn, trained_params, _ = ppo.train(
    environment=env,
    progress_fn=progress,               # called at each eval boundary: (step, metrics)
    policy_params_fn=policy_params_fn,  # also at each eval: (step, make_policy, params)
    network_factory=network_factory,
    randomization_fn=randomizer,
    wrap_env_fn=wrapper.wrap_for_brax_training,
    seed=seed,
    **params,
)
```

Everything interesting is in `params`, and we read those out of `mujoco_playground.config.locomotion_params.brax_ppo_config(...)` rather than copying the numbers, so a Playground release that re-tunes the env re-tunes us too:

| Knob | Value | What it controls |
|---|---|---|
| `num_timesteps` | 200,000,000 | total environment steps in the run |
| `num_envs` | 8192 (we default 4096) | simulations stepping in parallel |
| `unroll_length` | 20 | steps collected per env before an update round |
| `batch_size` x `num_minibatches` | 256 x 32 | 8192 env-trajectories per update round |
| `num_updates_per_batch` | 4 | gradient epochs over each batch |
| `discounting` | 0.97 | how far ahead the critic looks (~33 steps, 0.7 s) |
| `learning_rate` | 3e-4 | |
| `entropy_cost` | 0.005 | exploration pressure |
| `clipping_epsilon` | 0.2 | the PPO trust region |
| `policy/value_hidden_layer_sizes` | (512, 256, 128) | both networks, MLPs |
| `normalize_observations` | True | running mean/std over observations |

Two pieces of arithmetic worth internalizing:

```
env steps per update round = batch_size * num_minibatches * unroll_length
                           = 256 * 32 * 20  =  163,840

constraint: num_envs must divide batch_size * num_minibatches (= 8192)
```

That constraint is why the sensible `num_envs` values here are 8192, 4096, 2048, 1024 and not, say, 6000. At 4096 envs each round is two 20-step unrolls across all envs; at 8192 it is one. Break the divisibility and Brax asserts before training starts.

### Asymmetric actor-critic: let the critic cheat

This is the piece worth stealing for other projects. The policy sees only `state` (103, sensor-shaped, noisy). The critic sees `privileged_state` (216, ground truth: true body velocities, contact forces, actuator forces).

The reasoning: the critic exists only during training. It is a scaffold, discarded at deployment. Estimating "how well is this going" from noisy partial observations is much harder than from ground truth, and a bad value estimate means noisy advantages, which means a noisy gradient for the policy. So give the critic everything, keep the policy honest, and you get faster learning without giving up a policy that could run on hardware.

Which is why `train.py` asserts rather than assumes:

```python
assert net_cfg["policy_obs_key"] == "state"
assert net_cfg["value_obs_key"] == "privileged_state"
```

Swap them and you either cripple the critic or train a policy that cannot exist on a real robot. Nothing crashes; it just quietly trains the wrong thing.

### Domain randomization

`registry.get_domain_randomizer` gives each of the thousands of parallel envs slightly different friction, link masses and motor gains. The policy therefore cannot memorize one exact physics, and has to find a gait that works across a family of them. On CPU this would be a real cost; here the randomization is vmapped along with everything else and is effectively free.

Note the `wrap_env_fn=wrapper.wrap_for_brax_training` line above: Playground's wrapper is what applies the auto-reset and the randomization vmap. Use Brax's default wrapper instead and the randomization is silently dropped.

---

## 5. Reading this repo

Concepts to files, in the order the tutorial introduced them:

| Concept | Where it lives |
|---|---|
| Which env, `njmax`, reward presets, the shape assertions | `envs.py` |
| MJX + Brax PPO on the GPU, the callbacks, the checkpoint | `train.py` |
| The CPU replay: MuJoCo's renderer, mp4 encoding | `render.py` |
| Charts, tables, the video block | `reports.py` |
| Orchestrators: `walk` and `compare_presets` | `pipeline.py` |
| Flyte envs, images, and every Spark-specific pin | `config.py` |
| Gradio launcher (holds no GPU, trains nothing) | `app.py` |

Both interesting callbacks are in `train.py`. `progress_fn(step, metrics)` fires at each eval boundary and streams the reward curve into the live Flyte report. `policy_params_fn(step, make_policy, params)` fires at the same points but is the only hook that hands over the *live weights*, which is what makes the mid-training video below possible.

---

## 6. The demo

```
walk (orchestrator, CPU)
  ├── train_policy  (GPU)   Brax PPO over MJX, 4096 envs, live reward curve
  └── render_replay (CPU) ×2  trained policy AND a random policy, both -> mp4
```

Against the old CPU attempt:

| | Old (CPU PPO) | This (MJX + Brax) |
|---|---|---|
| Physics | CPU MuJoCo | GPU, compiled (MJX on Warp) |
| Parallel envs | 160 (5 workers × 32) | 4,096 on one GPU |
| Inner loop | Python | one XLA program, no Python per step |
| Reward | hand-written, 10 terms | DeepMind's tuned `G1JoystickFlatTerrain` |
| Measured throughput | never recorded | **59,757 env-steps/sec** at 2,048 envs |

**Mid-training video.** Every ~15% of the run (`--snapshot_every_pct`, 0 to disable) the current policy is filmed for 250 steps and dropped into the *training task's* live report, overwriting the previous clip, so you can watch the gait develop instead of waiting an hour to find out it learned to hop. Underneath the clip is a filmstrip of one still per snapshot: the reward curve tells you the number went up, the filmstrip tells you whether it went up for the right reason.

This hangs off `policy_params_fn`, for the reason given in Part 5: `progress_fn` gets only `(step, metrics)`, so it cannot render anything. Cost is roughly 10s per snapshot after a one-off single-env compile, under 2% of a full run. The whole thing is wrapped in a `try/except` that logs and continues, because no rendering problem should ever kill an hour of training.

The reward curve streams into the Flyte report while training runs. Two replay clips land there when it finishes: the trained policy and, next to it, a **random policy on the same command and camera**. The trained clip on its own is unevaluable, because "is that good walking?" has no answer without the before-picture.

Verified end to end on the devbox (project `physical-ai`, run `rfjnchxkvtbv2tztgf69`, 2026-08-04). A deliberately tiny 4M-step run produced:

```
best eval reward   -3.08
trained policy     survived 33 / 300 steps
random policy      survived  8 / 300 steps
```

4x longer upright after 4M steps, which is 2% of the tuned recipe. It is not walking yet, and it is not supposed to be: that run exists to prove the pipeline, not the policy. The report came back at 296KB with both clips embedded and playable.

### Run it

Runs go to the **`physical-ai`** Flyte project (`.flyte/config.yaml`).

```bash
# Confirm the curve is climbing (~10 min)
flyte run pipeline.py walk --num_timesteps 20000000 --preset baseline

# The real run. DeepMind's recipe for this env is 200M steps.
flyte run pipeline.py walk --num_timesteps 200000000 --num_envs 8192 --preset baseline

# Several reward presets, one chart. These SERIALIZE: one GPU.
flyte run pipeline.py compare_presets --presets '["baseline","high-step"]'
```

Or launch from the studio:

```bash
python app.py        # deploys the Gradio launcher, prints its URL
```

| Flag | Default | What it does |
|---|---|---|
| `--num_timesteps` | 20,000,000 | Total environment steps. The tuned recipe is 200M |
| `--num_envs` | 4096 | Parallel simulations on the GPU. Must divide 8192; scales nearly linearly |
| `--preset` | `baseline` | Reward preset, see below |
| `--num_evals` | 10 | Evaluation points, so also how often the chart updates |
| `--domain_randomization` | true | Per-env friction, mass and motor gains |
| `--replay_steps` | 500 | Length of the replay clip |
| `--terrain` | `rough` | `rough` or `flat`, see below |

### Terrain

Two Playground envs, same code path, same tuned PPO config, same observation and action shapes. They differ only in the scene:

| `--terrain` | Scene | Notes |
|---|---|---|
| `rough` (default) | `scene_mjx_feetonly_rough_terrain.xml` | 10x10m heightfield, 0.05m relief, rocky texture |
| `flat` | `scene_mjx_feetonly_flat_terrain.xml` | a bare plane |

`rough` is the better default on both counts: it's a genuinely harder problem, and it doesn't look like the robot is walking on nothing. The terrain is stored in the checkpoint, and `render.py` reads it from there rather than from a default, because replaying a rough-terrain policy on flat ground would silently compare two different problems.

Known cosmetic wart: the rough scene ships no skybox, so its background renders black where flat's is grey. Affects the replay's looks, not the physics.

### Reward presets

Each preset is a small diff on top of DeepMind's scales, not a rewrite. Only the named keys change.

| Preset | Diff | Why |
|---|---|---|
| `baseline` | none | The control. Whatever this scores is the number to beat |
| `high-step` | `feet_clearance: 1.0`, `feet_air_time: 3.0` | Anti-shuffle. Playground ships `feet_clearance` at **0.0**, so it's off by default; turning it on is the most direct answer to a robot that slides its feet |
| `smooth` | `action_rate: -0.75`, `energy: -0.001`, `dof_acc: -2.5e-7` | Penalize jerk. Slower, cleaner gait |
| `turner` | `tracking_ang_vel: 1.5`, `tracking_lin_vel: 1.25` | The 300M baseline tracks forward speed well and barely turns at all; this doubles the yaw term |
| `no-perturb` | pushes off, sensor noise off | Learns fastest, generalizes worst. Good for a first end-to-end check |

---

## 7. Things that cost real debugging time on this box

**`njmax` overflows.** Playground ships `njmax=90` for this env (`29*2 + 8*4`). On the Spark the first rollout printed `nefc overflow - please increase njmax to 93` repeatedly. As Part 1 explained, `njmax` caps the constraint rows the solver can hold, and a humanoid in ground contact makes more than 90 of them. On overflow MJX **silently drops constraints** rather than erroring, so feet sink through the floor and you are no longer training on the physics you think you are. We set `njmax=128`. Fixing it also roughly doubled throughput.

**Page cache starves CUDA, and the error lies about it.** With 86GB in `buff/cache`, `cuMemGetInfo` reported only **18.0 GiB free of 119.7 GiB** and JAX died with:

```
Failed to create stream executor for device CUDA:0: CUDA_ERROR_OUT_OF_MEMORY
RuntimeError: Unable to initialize backend 'cuda': INTERNAL: no supported devices
found for platform CUDA (you may need to uninstall the failing plugin package)
```

That reads like a broken install. It is not, and the suggested fix is exactly wrong. `free` is no help either: it reported 99GB "available" the whole time. The tell is `cuMemGetInfo`, not `free`. Lowering `XLA_PYTHON_CLIENT_MEM_FRACTION` does **not** help, because the failure is at stream-executor creation, before any arena is sized.

This is not hypothetical: it killed a 200M-step run at minute zero, an hour after the failure mode was first written down here.

Two things now guard against it.

**`train.py` waits instead of dying.** `_wait_for_gpu()` probes the CUDA driver through `ctypes` and retries for up to 15 minutes before giving up. It runs **before `import jax`**, and that ordering is load-bearing: jax caches a failed backend init for the life of the process, so a task that imports jax during a starved window stays broken even after the box recovers.

**Freeing the cache needs no root.** `drop_caches` wants sudo, but briefly allocating and releasing a large anonymous block makes the kernel evict reclaimable cache to satisfy it:

```python
N = 45 * 2**30
buf = bytearray(N)
for i in range(0, N, 1 << 22):   # touch every 4MB to actually commit
    buf[i] = 1
del buf
```

Measured: `buff/cache` 98GB → 54GB, free 1GB → 45GB, `cuCtxCreate` 2 → 0.

Practical rule stands anyway: do not start a run while a big blob scan, model fetch, or docker build is hammering the disk.

**`XLA_PYTHON_CLIENT_PREALLOCATE=false` is mandatory.** JAX preallocates 75% of "GPU memory" by default. On a discrete card that reserves VRAM nobody wanted. On GB10 there is one 119.7GiB pool shared with the OS, so the default claims ~90GB of the RAM the OS is living in.

**`jax[cuda13]`, not `cuda12`.** The GB10 driver is CUDA 13.0. `jax-cuda13-plugin` publishes `manylinux_2_27_aarch64` wheels, so the arm64 install is clean.

**jax is pinned to 0.9.2, and it has to be.** brax 0.14.2 (the current release) still calls `jax.device_put_replicated` in `ppo/train.py:756`. jax removed it in 0.10, so an unpinned install resolves to 0.10.2 and every run dies the moment training starts:

```
AttributeError: jax.device_put_replicated is deprecated; use jax.device_put instead.
```

Nothing in the install warns you; it fails after the env has already built and compiled. 0.9.2 still ships the function (as a `DeprecationWarning`) and its cuda13 plugin still enumerates the GB10. Lift the pin when brax moves off the old pmap API.

**MJX runs on Warp now.** MuJoCo 3.11's MJX uses `mujoco_warp` (`impl="warp"`), not pure JAX/XLA. It reports `CUDA Toolkit 12.9, Driver 13.0`, sees the GB10 as `sm_121, 120 GiB, mempool enabled`, and JIT-compiles a long list of kernels on first use (~40s). That is a one-off, but it is why the first iteration looks hung.

**`mujoco_menagerie` must be baked into Playground's own package directory.** `MENAGERIE_PATH` is `<site-packages>/mujoco_playground/external_deps/mujoco_menagerie`, hardcoded relative to the module file, with no env var to redirect it. An earlier version of `config.py` cloned it to `/opt/mujoco_menagerie` and that did nothing whatsoever: every task pod still printed `mujoco_menagerie not found. Downloading...` and re-cloned at runtime, costing minutes per task and putting a network dependency inside a job that should have none. The fix is to call Playground's own installer at image build time:

```python
.with_commands([
    "python -c 'from mujoco_playground._src import mjx_env; mjx_env.ensure_menagerie_exists()'",
])
```

That also inherits its pinned `MENAGERIE_COMMIT_SHA`, so the robot XMLs can't drift from the version the env code expects.

**Rendering is CPU, and a different backend.** MJX's fast path needs the GPU and MJX has no renderer. The replay rebuilds the same env with `impl="jax"` (loads in 0.5s on CPU) and draws through headless EGL. So the policy is *trained* on Warp and *replayed* on JAX. Those are two implementations of one model and should agree closely, but they are not bit-identical: if a replay disagrees badly with the eval reward, suspect this first.

---

## 8. A guided tour of the code

If you are reading this to present it, or just want the shortest path through the repo, this is the order that works. Four stops, arranged so each one answers the question the previous one raises.

### Stop 0: the rod, live in a REPL

Not a file in this repo. Paste the [20-line snippet from Part 1](#a-whole-simulation-in-20-lines) into a Python prompt and run it. `MjModel`, `MjData`, `mj_step`, a one-line controller, printed state. Everything that follows is that same loop with a neural network in the controller slot and several thousand copies of `MjData`. Starting anywhere else means starting with abstraction.

### Stop 1: [`envs.py`](envs.py), the physics lesson and the RL lesson in one file

266 lines, and the highest teaching value per line in the repo. Three places to stop.

**The `njmax` fix** is the best "physics engines have sharp edges" story here, and it follows straight from the constraint solve in [Part 1](#what-actually-happens-inside-mj_step):

```python
# Playground ships `njmax=90` for this env. On this box that overflows almost
# immediately: the very first rollout printed
#
#     nefc overflow - please increase njmax to 93
#
# repeatedly. njmax caps the constraint-Jacobian rows the solver can hold, and a
# humanoid in contact with the ground generates more than 90 of them. On overflow MJX
# silently DROPS constraints rather than erroring, which means feet sink through the
# floor and the physics quietly stops being the physics you think you are training on.
NJMAX = 128
```

The point to land: this is not a crash, it is a *log line*. Training runs to completion, the reward curve climbs, and the number it climbed to describes physics where the feet pass through the floor. Fixing it also roughly doubled throughput.

**The presets** are the reward-shaping argument in a form you can read out loud. Each is a diff on DeepMind's tuned scales, never a rewrite:

```python
# The tutorial's diagnosis was "it shuffles instead of stepping". Playground
# already has the two terms that fix that, and one of them is OFF by default:
#   feet_air_time   2.0  (on)  rewards a foot spending time in the air
#   feet_clearance  0.0  (OFF) rewards the swing foot actually lifting
# Turning clearance on is the single most direct answer to shuffling.
"high-step": RewardPreset(
    key="high-step",
    scales={"feet_clearance": 1.0, "feet_air_time": 3.0},
    notes="Anti-shuffle: turn on foot clearance, push air time harder.",
),
```

The previous attempt at this demo wrote all ten reward terms by hand, watched the robot shuffle, and could not say which term was wrong. Holding the baseline fixed and naming each change is what makes the comparison mean anything.

**`build_env` asserts its own shapes** rather than trusting them:

```python
env = registry.load(name, config=cfg)

if env.action_size != EXPECTED_ACTION_SIZE:            # 29
    raise RuntimeError(...)
obs = env.observation_size
if not isinstance(obs, dict):                          # asymmetric actor-critic
    raise RuntimeError(...)
for key, want in EXPECTED_OBS_SIZES.items():           # state 103, privileged 216
    ...
```

`EXPECTED_ACTION_SIZE = 29` and `{"state": 103, "privileged_state": 216}` were measured on this box, not guessed. A Playground release that changes the observation layout then fails loudly here, at env build, instead of quietly training a mis-shaped network for an hour.

### Stop 2: [`train.py`](train.py), where the payoff is

347 lines, and two windows are enough.

**The learner is one call.** This is the moment to point out that everything in [Part 4](#4-the-learning-half-ppo-and-brax) collapses to this:

```python
make_inference_fn, trained_params, _ = ppo.train(
    environment=env,
    progress_fn=progress,               # (step, metrics) at each eval boundary
    policy_params_fn=policy_params_fn,  # (step, make_policy, params), same boundaries
    network_factory=network_factory,
    randomization_fn=randomizer,
    # Playground's wrapper adds the auto-reset and domain-randomization vmap that
    # Brax's own wrapper does not know about. Using Brax's default here silently
    # drops the randomization.
    wrap_env_fn=wrapper.wrap_for_brax_training,
    seed=seed,
    **{k: v for k, v in params.items() if k != "network_factory"},
)
```

Two things worth saying out loud. `wrap_env_fn` is a silent-failure trap: leave it off and you still get a trained policy, just one that has never seen a friction it did not expect. And the `**params` splat is deliberate, because those numbers come from Playground's tuned config rather than from us.

**The two callbacks** are where the live report comes from, and the difference between them is a genuinely non-obvious API detail:

```python
def policy_params_fn(step, make_policy, params):
    """Never let a rendering problem kill an hour of training."""
    ...
    try:
        _render_snapshot(step, make_policy, params)   # film the CURRENT policy
    except Exception as exc:
        log.warning(f"  snapshot at step {step:,} failed ({exc}); training continues")

def progress(step, metrics):
    history.append({...})
    flyte.report.replace(reports.progress_html(...))  # stream the reward curve
    flyte.report.flush()
```

`progress_fn` gets only `(step, metrics)`, so it can chart but cannot render. `policy_params_fn` fires at the same eval boundaries and is the **only** hook that hands over the live weights, which is the entire reason mid-training video is possible at all. Note the `try/except` around the snapshot: rendering is a nice-to-have and training is an hour of GPU, so a broken frame must never take the run with it.

**Optional, for a war-stories segment:** `_wait_for_gpu()` probes CUDA through `ctypes` and retries for 15 minutes rather than dying on a transient:

```python
_wait_for_gpu()      # must run BEFORE `import jax`

import jax
```

The ordering is the whole trick. JAX caches a failed backend init for the life of the process, so a task that imports jax during a starved window stays broken even after the box recovers. See [the gotchas](#7-things-that-cost-real-debugging-time-on-this-box) for why the box gets starved in the first place.

### Stop 3: [`pipeline.py`](pipeline.py), the shape of a run

Read `walk` top to bottom; it is short. The two lines that matter:

```python
trained_f, random_f = await asyncio.gather(
    render_replay(checkpoint=checkpoint, steps=replay_steps, policy="trained"),
    render_replay(checkpoint=checkpoint, steps=replay_steps, policy="random"),
)
```

That is the evaluation argument in two lines. A clip of the trained policy on its own is unevaluable, because "is that good walking?" has no answer without the before-picture on the same command and the same camera. The random policy costs one extra CPU task and makes the whole report honest.

### Two one-liners, if the traps come up

[`config.py`](config.py) bakes the robot XMLs into the image at build time, and it has to land in Playground's own package directory:

```python
.with_commands([
    "python -c 'from mujoco_playground._src import mjx_env; "
    "mjx_env.ensure_menagerie_exists()'",
])
```

[`render.py`](render.py) rebuilds the same env on the CPU backend, which is the trained-on-Warp / replayed-on-JAX split from [Part 3](#what-mjx-gives-up):

```python
cfg = envs.build_env_config(preset, terrain=terrain)
cfg.impl = "jax"        # MJX has no renderer; this one runs on CPU and draws
env = registry.load(envs.env_name(terrain), config=cfg)
```

### What to skip

[`reports.py`](reports.py) is HTML plumbing (matplotlib to base64, a table helper, a `<video>` block) and [`app.py`](app.py) is Gradio boilerplate around a launcher that holds no GPU. Both are worth having and neither teaches anything about MuJoCo.

**If you only open one file: [`envs.py`](envs.py).** `njmax` plus the presets is the physics lesson and the RL lesson in about 90 lines.

---

## Where to take it next

- **8192 envs.** DeepMind's config uses it and we default to 4096. Worth measuring whether the scaling holds.
- **Reward sweeps as a fan-out.** `compare_presets` already does the shape; with more than one GPU it would actually parallelize.
- **Turning.** The `turner` preset exists because the 300M baseline tracks forward velocity well (commanded +1.0 m/s gave +0.88) and essentially ignores yaw (commanded +1.0 rad/s gave -0.06). Whether reweighting fixes it is an open question.
- **Sim-to-real gap.** The policy only ever sees `state`, so it is deployable in principle. What breaks first on hardware is the interesting question.

## Tooling

- MuJoCo: https://github.com/google-deepmind/mujoco
- MuJoCo docs, including the MJCF reference: https://mujoco.readthedocs.io
- MuJoCo Playground: https://github.com/google-deepmind/mujoco_playground
- MJX (MuJoCo on GPU): https://mujoco.readthedocs.io/en/stable/mjx.html
- mujoco_warp (the kernel backend MJX uses now): https://github.com/google-deepmind/mujoco_warp
- Brax (JAX RL algorithms): https://github.com/google/brax
- mujoco_menagerie (the robot models): https://github.com/google-deepmind/mujoco_menagerie
- PPO, the original paper: https://arxiv.org/abs/1707.06347
