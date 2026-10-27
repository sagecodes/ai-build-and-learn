# Chapter 12 bonus: PPO teaches a humanoid robot to walk

**Builds on:** [chapter 12](../ch12_actor_critic)'s PPO
**Run:** `flyte run pipeline.py ch12_bonus_humanoid` (NVIDIA GPU only; about 50 minutes per reward preset, two presets by default)

Acronyms and symbols are all in the [glossary](../README.md#glossary).

## The idea

Chapter 12 used PPO to swing up a pendulum: one torque, three observations, eight simulators on the CPU. This bonus runs the same algorithm on a **Unitree G1 humanoid** in MuJoCo: 29 joint targets fifty times a second, 103 observations, and **4,096 copies of the robot simulated at once on the GPU**. Nothing tells it what walking looks like. It is paid for following a joystick command ("forward at 1 m/s") without falling, and it finds a gait on its own.

Then it trains a second time with a small change to the reward, to show that the reward, not the learner, decides *how* it walks.

| | chapter 12: Pendulum | bonus: G1 humanoid |
|---|---|---|
| actions per step | 1 (a torque) | 29 (joint targets) |
| what the actor sees | 3 numbers | 103 numbers |
| what the critic sees | the same 3 | 216: the 103 plus true values only a simulator knows |
| simulators at once | 8, CPU | 4,096, GPU |
| moves per batch | 2,048 | 163,840 |
| clip | 20% | 20% |
| training steps | 400,000 | 200,000,000 |

## The two reward presets

| preset | what changes |
|---|---|
| `baseline` | nothing: DeepMind's tuned reward from MuJoCo Playground |
| `high-step` | `feet_clearance` 0 → −0.5 (a moving foot pays for being away from step height, so sliding the feet costs), `feet_air_time` 2 → 3 (more pay when a foot lands after a real swing) |

Any preset from [rl-mujoco](../../rl-mujoco#reward-presets) works: `--presets '["baseline","smooth"]'`.

## How the code does it

The training lives in [rl-mujoco](../../rl-mujoco), which has the GPU image (jax, MJX, Brax) and a long write-up of how it works. This project's task calls rl-mujoco's deployed tasks by name, so the grokking image stays CPU-only and never installs jax:

```
ch12_bonus_humanoid (CPU, this project)
  ├── g1-train.train_policy   (GPU)  once per preset, one after another: one GPU
  └── g1-render.render_replay (CPU)  each trained policy, plus one random policy
```

`humanoid.py` reads what comes back and builds the report. rl-mujoco's checkpoint is a pickle full of Brax objects, so `read_checkpoint` unpickles it with stand-ins for every brax/jax class: the weights become placeholders, and the eval history and run settings come through intact.

## Running it

Needs an NVIDIA GPU devbox (`flyte start devbox --gpu`); MJX does not run on a Mac. Deploy rl-mujoco's tasks once, into the same project:

```bash
cd ../rl-mujoco && flyte deploy --all pipeline.py && cd ../grokking-rl

flyte run pipeline.py ch12_bonus_humanoid                       # baseline + high-step, 200M steps each
flyte run pipeline.py ch12_bonus_humanoid --presets '["high-step"]' --num_timesteps 20000000   # a quick look
flyte run pipeline.py book --bonus true                         # the whole book, the humanoid as the last tab

# reuse a checkpoint from an earlier rl-mujoco run instead of training it again
flyte run pipeline.py ch12_bonus_humanoid --presets '["high-step"]' \
    --checkpoints '["s3://.../g1_checkpoint.pkl"]'
```

The training task's own report (click into `train_policy` in the run) shows the live reward curve and a clip of the robot every ~15% of training, so you can watch the gait form.

## In the report

1. **Before and after**: a random policy next to each trained one, same command, same camera.
2. **The same algorithm, a different size**: chapter 12's PPO next to this one, number by number.
3. **Same PPO, different reward, different gait**: learning curves for each preset, and the feet air-time term with the weights divided back out.
4. **The critic gets to cheat**: why the critic sees 216 numbers and the actor only 103.
