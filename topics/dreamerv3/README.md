# World Models with DreamerV3

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event kicks off a run on world models: models that learn an internal representation of how an environment works, then use it to predict what happens next and to plan. It's a natural next step after the generation and RL events, tying both threads together.

We start with [DreamerV3](https://github.com/danijar/dreamerv3), which is the best on-ramp to the idea. It runs on the same MuJoCo physics as the [MuJoCo event](../rl-mujoco) next door, so the comparison between model-free and model-based RL is direct rather than hand-waved.

## What a world model is

A world model is a neural network that has learned to answer one question:

> Given what has happened so far, and what I do next, what happens next?

That is it. It is a **learned simulator**. Not hand written physics like MuJoCo, but a
network that watched an environment and worked out its rules well enough to continue it
on its own.

The name comes from the idea that an agent carries a compressed model of its world
around inside it, and can consult that model instead of the world:

```
   WITHOUT a world model              WITH a world model
   ─────────────────────              ──────────────────

   "what happens if I                 "what happens if I
    lean forward?"                     lean forward?"
          │                                   │
          ▼                                   ▼
   ┌─────────────┐                     ┌─────────────┐
   │ try it in   │                     │ ask the     │
   │ the world   │                     │ model       │
   └─────────────┘                     └─────────────┘
   slow, risky, and                    fast, safe, and wrong
   the only way to know                in ways you can measure
```

Three capabilities fall out of that one question, and they are what everyone actually
wants a world model for:

- **Imagination.** Roll the model forward past the end of the data. That is how
  DreamerV3 trains a policy without touching the simulator.
- **Planning.** Try several action sequences in the model and pick the one that ends
  best, without any of them happening for real. This is the thing you want before you
  put a policy on a robot that can break.
- **Representation.** To predict a scene you have to encode what is in it. A model
  trained only to predict tends to discover objects, contact and momentum for free,
  because there is no other way to get the prediction right.

### The part people get wrong

A world model is **not** the same as a policy, and it does not want anything. It is
reward-agnostic: a world model of a walker that flails on the ground forever is a
perfectly good world model, as long as the flailing is predicted accurately.

Prediction alone gives you a very good video predictor and an agent that does nothing.
Turning it into behaviour needs two more pieces, which is exactly what DreamerV3 adds
and what [How it works](#how-it-works) is about.

### Two families, and they are diverging

The label "world model" now covers two things that share an idea and almost no code.

| | **models you act with** | **models you watch** |
|---|---|---|
| examples | DreamerV3, PlaNet, TD-MPC | Cosmos, Genie, GAIA, Sora |
| predicts in | a small latent space | pixels, at high fidelity |
| size | 10M to 200M parameters | billions |
| judged by | does the agent get better | does the video look right and obey physics |
| trained on | its own experience, online | enormous offline video corpora |

This repo is squarely in the left column. The right column is where the
[video generation event](../video-generation) already went, and where Cosmos and
V-JEPA 2 will go later in this arc. Worth keeping the distinction in mind, because
"world model" in a paper title could mean either.

## Where DreamerV3 fits, and why it is first

The lineage is short and unusually legible:

```
  2018  World Models (Ha & Schmidhuber)   VAE + RNN + tiny controller.
          │                               Proved you could train a policy
          │                               entirely inside a dream.
          ▼
  2019  PlaNet                            Dropped the pixel decoder from the
          │                               planning loop. Introduced the RSSM,
          │                               the deterministic + stochastic latent.
          ▼
  2020  Dreamer / DreamerV2               Replaced planning with a learned actor
          │                               and critic trained on imagined rollouts.
          │                               Strong, but needed per-domain tuning.
          ▼
  2023  DreamerV3            ◄── we are here
          │                               Same architecture, plus the robustness
          │                               tricks that let ONE hyperparameter set
          ▼                               work from control to Atari to Minecraft.
  2024+ Cosmos, Genie, V-JEPA 2, GAIA     Huge models trained on internet video.
                                          Different column of the table above.
```

It is first in this arc for four reasons, and the fourth is the real one:

1. **It is complete.** Perception, dynamics, reward, planning and control are all
   present, and small enough to hold in your head at once. The newer models are usually
   only the prediction half.
2. **It fits on one box.** 10.5M parameters, one GPU, one afternoon. Cosmos is a
   different order of expense.
3. **Every part is observable.** Each component has its own loss curve, and its failure
   has a distinct visible signature. That is unusual and it is what makes this a good
   teaching model.
4. **It makes the claim falsifiable.** You can point the decoder at the imagined latents
   and *watch the model's prediction next to what actually happened*, frame by frame.
   Almost nothing else in ML lets you look directly at what the network believes.

That last one is what the rest of this README is built around.

## What DreamerV3 actually is

Most RL you have seen is **model-free**. PPO, the algorithm behind the walking G1 in
[rl-mujoco](../rl-mujoco), never learns what the environment *is*. It tries things,
sees what scored well, and shifts its policy toward whatever worked. The only way for
it to find out what an action does is to do it, in the real simulator, millions of
times.

```
  MODEL-FREE          the simulator is in the inner loop

    ┌──────────┐   action    ┌───────────────┐
    │  policy  │ ──────────► │   SIMULATOR   │
    │          │ ◄────────── │    (MuJoCo)   │
    └──────────┘  obs, reward└───────────────┘
         ▲                           │
         └───────────────────────────┘
          every gradient step needs fresh simulator steps
          rl-mujoco: 4,096 parallel envs, 20,000,000 steps
```

DreamerV3 is **model-based**. It spends its experience learning a compact model of the
environment's dynamics, then trains its policy almost entirely *inside that model*, on
imagined rollouts. The simulator is used to collect data and to check the model, not to
run the millions of trials the policy needs.

```
  MODEL-BASED         the simulator is only a data source

    ┌──────────┐   action    ┌───────────────┐
    │  policy  │ ──────────► │   SIMULATOR   │
    │          │ ◄────────── │    (MuJoCo)   │
    └──────────┘  obs, reward└───────┬───────┘
         ▲                           │ ~20 steps/s, the expensive part
         │                           ▼
         │                   ┌───────────────┐
         │                   │ replay buffer │
         │                   └───────┬───────┘
         │                           │
         │                           ▼
         │  trains inside    ┌───────────────┐
         └────────────────── │  WORLD MODEL  │  learns to predict
            imagined rollouts│    (RSSM)     │  what happens next
                             └───────────────┘
```

That buys two things:

- **Sample efficiency.** Real environment steps are the expensive resource, whether
  that is a slow simulator or an actual robot. Dreamer replaces most of them with
  imagined ones, which are a forward pass through a small network rather than a physics
  engine.
- **Generality.** DreamerV3's headline claim is that one fixed set of hyperparameters
  works across very different domains, from continuous control to Atari to Minecraft,
  which is unusual in RL and is what the paper "Mastering Diverse Domains through World
  Models" is named for.

## How it works

Two halves, trained together, doing different jobs. The world model learns what the
world does. The actor and critic learn what to do about it, without ever touching the
world.

### What we wrote, and what it worked out

Worth drawing this line before anything else, because every piece below sits on one
side of it.

**We wrote the goal, by hand, in full.** `ArenaWalker` inherits `get_reward` from
dm_control's `PlanarWalker` and never overrides it, so the objective is the stock
walker reward: three quantities read straight out of the physics engine, hand-picked
and hand-weighted.

```python
standing    = tolerance(torso_height,        bounds=(1.2, inf))   # be tall
upright     = (1 + torso_upright) / 2                             # point upwards
move_reward = tolerance(horizontal_velocity, bounds=(1.0, inf))   # travel at 1 m/s

reward = stand_reward * (5 * move_reward + 1) / 6
```

That is the complete specification of "walking". Note what it does not mention: legs,
feet, stepping, gait, balance, the posts, the balls. Note too that it reads `physics`
rather than the image, so it is computed from privileged simulator state the agent is
never shown. Even the shaping is a human decision: `stand_reward` *multiplies*, which
encodes the curriculum "you cannot walk before you can stand".

Everything else is the model's problem.

| we defined | the agent worked out |
|---|---|
| the reward function, over simulator state | what reward *looks like*, from 64x64 pixels |
| a body with 6 torques | what a torque does to a limb |
| the world, as an XML file | the world's dynamics, well enough to run it forward |
| that upright and moving pays | a gait that is upright and moving |
| 15 steps of imagination per latent | which action sequences are worth imagining |

The third row is what the demo is built around. Nothing tells the model that balls
exist or that balls roll; having to redraw the frame is the only reason it ever finds
out.

Two consequences of that split, both of which catch people out:

**The agent never sees the reward function.** It receives one scalar per step and has
to learn to predict that scalar from pixels. The actor then maximises *the learned
prediction*, never the real reward, because inside an imagined rollout there is no
MuJoCo left to ask. An agent with a bad `rew` curve is chasing a hallucinated goal.

**The world model does not know the goal at all.** It is reward-agnostic, and left to
itself would model a walker flailing on the floor with great accuracy and be perfectly
satisfied. That is precisely what makes the `back` task interesting: flip one sign in
`horizontal_velocity` and the world model is still exactly right while the actor is
not merely useless but maximally wrong. `transfer.py` freezes the first and retrains
the second to find out how much of the original cost was which.

### What the agent actually sees

Before any of the mechanism, the input. The flagship run is
`--configs dmc_vision size12m --task dmc_arena_walk`, and `dmc_vision` sets
`env.dmc.proprio: False`. Together with `pixels_only` in launch.py, the observation
dict handed to the agent on every step is exactly this:

```
  image            (64, 64, 3) uint8     the camera, tracking the walker
  reward           scalar
  is_first / is_last / is_terminal       episode flags
```

Then upstream narrows it once more. From `dreamerv3/agent.py`:

```python
exclude = ('is_first', 'is_last', 'is_terminal', 'reward')
enc_space = {k: v for k, v in obs_space.items() if k not in exclude}
```

**The encoder's only input is the image.** Reward is not perceived, it is predicted: it
enters the graph solely as the target of the reward head's loss. The 9 velocities and
14 orientations that `ArenaPhysics` goes to such lengths to preserve do not exist on
this path at all, and neither do `log/x_position` or `log/ball_distance`, which
embodied strips one layer above the agent. Out the other side come 6 continuous joint
torques. That is the whole interface.

So nothing in the training signal names a leg, a ball or a post. The only pressure on
the encoder is *compress this frame well enough to redraw it and predict its reward*,
and everything the model ends up knowing about the world fell out of that one demand.
It has to spend latent capacity on the balls because reconstruction is scored on them,
which is why they are in the domain at all: see
[The arena](#the-arena-a-world-with-things-in-it).

One consequence is worth stating before the diagrams rather than after:

```
   pixels ──► encoder ──► z(t) ──┐
                                 ├──► ACTOR ──► torques
                         h(t) ──►┘

   the policy is a function of the latent, never of the image
```

The actor maps `(h, z)` to actions and has never been shown a pixel. Pixels reach it
only through the encoder that builds `z`, and its gradient never flows back through a
real frame; it flows through 15 steps of *imagined* latent dynamics. DreamerV3 is
model-free RL on a compact learned state, run inside a simulator it wrote itself.

### Half one: the world model, an RSSM

The world model is a **Recurrent State-Space Model**. It never works in pixels. It
compresses each observation into a small latent state in two parts, and does all its
predicting there:

- `h`, a **deterministic** recurrent state carried by a GRU. The model's memory of
  everything that has happened so far. 2,048 numbers at the size preset used here.
- `z`, a **stochastic** state: 32 categorical variables of 16 classes each. What the
  model is uncertain about.

One timestep of the cell, with the pieces named as they appear in the loss curves:

```
   h(t-1) ─┐
   z(t-1) ─┼─►┌─────────────────────┐
   a(t-1) ─┘  │  GRU  (deterministic│──► h(t)  "everything I remember"
              │       memory)       │
              └──────────┬──────────┘
                         │
            ┌────────────┴─────────────┐
            ▼                          ▼
    ┌───────────────┐          ┌────────────────┐
    │   DYNAMICS    │          │    ENCODER     │ ◄── obs(t)
    │  the PRIOR:   │          │ the POSTERIOR: │     64x64 pixels
    │ what I expect │          │ what I can see │
    │ before seeing │          │                │
    └───────┬───────┘          └────────┬───────┘
            │                           │
          ẑ(t)  ◄── KL divergence ───► z(t)
            │      logged as dyn, rep   │
            └─────────────┬─────────────┘
                          ▼
              ┌───────────────────────┐
              │ decoder  -> the image │   loss: image
              │ reward   -> r(t)      │   loss: rew
              │ continue -> keep going│   loss: con
              └───────────────────────┘
```

The interesting piece is **dynamics**, because it has to guess what the world will look
like next *before seeing it*. The encoder gets to look. Training pulls the two together,
and that gap is the KL that shows up in the logs as `dyn` and `rep`. They are the same
divergence with the gradient flowing to opposite sides: `dyn` drags the predictor toward
the encoder, `rep` drags the encoder toward something the predictor can anticipate. Both
are clipped by "free bits" so the model does not burn capacity driving an already small
KL to zero.

### Observing versus imagining, which is the whole trick

The same cell runs in two modes, and the difference between them is the entire idea:

```
  OBSERVING                          IMAGINING
  (green border in the video)        (red border)

   obs ──► encoder ──► z(t)           obs   ✗  not available
                        │                          │
   the latent is CORRECTED            the latent is INVENTED
   by what is really there            by the dynamics predictor
                        │                          │
              h(t) ─────┘             h(t) ────────┘
                                      errors compound with
   ground truth every step            nothing to correct them
```

Once dynamics is good enough, you can cut the observation off and let the model run
forward on its own. That is the dream, and it is what the report's video shows: the
green half is the model watching, the red half is the model alone.

### Half two: the actor and critic, trained inside the dream

The policy never trains on real trajectories. Every latent in a training batch becomes
the root of a short imagined rollout:

```
  from replay:  o1 → o2 → o3 → ...  → o64      REAL, 64 frames from MuJoCo
                 │    │    │           │       x 16 sequences = 1,024 latents
                 ▼    ▼    ▼           ▼
                ┌┴┐  ┌┴┐  ┌┴┐         ┌┴┐      each is a starting point
                │ │  │ │  │ │   ...   │ │
                │ │  │ │  │ │         │ │      15 steps forward, in latent
                ▼ ▼  ▼ ▼  ▼ ▼         ▼ ▼      space, no pixels decoded

   the actor picks the actions, the reward head supplies the rewards,
   the continue head decides when an imagined episode ends
```

The critic is trained on the returns of those imagined trajectories, and the actor is
trained to maximise them. Nothing is decoded to pixels for any of it, which is what
makes imagination cheap.

Put the numbers together and the bargain becomes obvious. This run does one gradient
step per four environment steps, and each gradient step imagines
`1,024 x 15 = 15,360` transitions. **For every single step it takes in MuJoCo, the
agent takes roughly 3,800 steps inside its own head.**

### What actually makes it walk

Worth stating plainly, because the dream video only shows the predicting half and it is
easy to come away thinking that is all there is.

The world model does not want the walker to walk. It is reward-agnostic. Left alone it
would learn to predict a walker flailing on the ground with great accuracy and be
entirely satisfied. Two extra pieces turn prediction into behaviour:

```
   the world model learns          latent(t), action ──► latent(t+1)
                                          "what happens next"

   the REWARD HEAD learns          latent ──► reward
                                          "how good is this state"
                                              │
                                              ▼
   the ACTOR searches inside the model for action sequences that score
   well under that PREDICTED reward, and the CRITIC scores the ones that
   pay off beyond the 15 step horizon.

                         walking is simply what scores.
```

So the answer to "is it learning to walk, or just to predict?" is both, in one gradient
step, and the two are not separable: the actor can only search action sequences the
dynamics model can simulate, and the reward head can only score states the encoder can
represent. A better world model directly makes a better policy possible.

The two halves also show up as different curves. From this run at 3% of training:

```
  world model, learning to predict     actor + critic, learning to act
  ───────────────────────────────      ─────────────────────────────────
  image   628.6 ──► 61.4               train/ret       0.001 ──► 0.758
  rew       5.43 ──► 0.50              value            3.75 ──► 1.16
  con       0.26 ──► 0.02              ent/action       7.87 ──► 0.91
                                       real return        42 ──► 75
```

`ent/action` is the policy's entropy, and it collapsing from 7.9 to 0.9 is the actor
going from near-random flailing to committed choices. That is control, not prediction.

**One caveat on reading the return early.** Walker's reward is
`stand_reward * (5 * move_reward + 1) / 6`. The standing term *multiplies*, so moving
pays almost nothing until the walker is upright: the agent has to learn to stand before
walking is even worth attempting. An early climb from 42 to 75 against a ceiling near
950 is it getting off the floor, not walking. This is exactly why metres travelled is
plotted next to the return, and at that point in the run it was still hovering around
2 m of the ~25 m a real walk covers in a 25 second episode.

### The training loop, all of it

```
  every env step          │  every 4th env step (train_ratio 256)
  ──────────────────────  │  ─────────────────────────────────────────────
  act in MuJoCo           │  1. sample 16 x 64 real frames from replay
  store the transition    │  2. train the world model to predict them
  in the replay buffer    │  3. imagine 15 steps from each of the 1,024
                          │  4. train actor and critic on the imagined returns
                          │
  ~20 per second          │  ~5 per second, and the bottleneck on this box
```

### The tricks that made v3 work

DreamerV2 needed per-domain tuning. V3's contribution is largely a set of robustness
tricks that let one configuration work everywhere: **symlog** squashing of rewards and
observations so wildly different magnitudes stop mattering, **two-hot** encoding that
turns value regression into classification over a fixed set of bins, **free bits** on
the KL, and normalising returns by a percentile range so the actor's gradient scale is
stable whether rewards are sparse or dense.

### Reading it back off the report

Every piece above has a curve, and they fail in ways you can tell apart:

| curve | the piece | what it means when it falls |
|---|---|---|
| `image` | decoder | the model can reproduce what it saw |
| `dyn`, `rep` | the KL | the model can anticipate what it has not seen yet |
| `rew`, `con` | heads | it knows what earns reward, and when an episode ends |
| `policy`, `value` | actor, critic | the agent is exploiting the model |

`image` falling while `dyn` stays flat is a model that can describe the present and
cannot predict the future, and it is exactly what the dream video shows as a red half
that dissolves into fog. Those two curves and that video are the same fact told twice.

## Showing the dream, which is the only demo that matters

Everything above is a description. The demonstration is different, and it is the thing
this repo is built around.

DreamerV3 already knows how to make it. `Agent.report()` takes six real sequences out
of the replay buffer, lets the world model watch the first half, then **hides the
images** and makes it predict the rest from actions alone. It stacks the result into a
grid: the true frames on top, the model's reconstruction in the middle, the difference
at the bottom, with a green border while the model can still see and a red border once
it has gone blind.

```
  ┌── green: the model is being shown the real frames ──┐┌── red: imagination ──┐
  │ true    ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓ ││ ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  │
  │ pred    ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  ▓▓▓▓ ││ ▓▓▓▓  ▓▓▓▓  ▓▓▓▓  │  <- has to invent these
  │ error   ░░░░  ░░░░  ░░░░  ░░░░ ││ ░░░░  ▒▒▒▒  ▓▓▓▓  │  <- and error compounds
  └─────────────────────────────────┘└──────────────────┘
```

Two things follow from that, and both shaped the design here.

**It only exists with pixel observations.** With `--configs dmc_proprio` the agent's
observation is a state vector. embodied renames the rendered image to `log/image` and
strips every `log/` key before the agent sees it, which
`embodied/jax/agent.py:51` asserts on. The decoder therefore has no image head,
`self.dec.imgkeys` is empty, and `report()` produces no video at all. `dmc_vision` is
the default in `pipeline.py` for exactly this reason.

**Getting it out costs nothing.** Dreamer's logger has a `scope` output on by default,
and `scope` writes every 4-D uint8 array it is handed as an h264 mp4 on disk:

```
<logdir>/scope/report-openloop-image.mp4/<step>-<id>.mp4    the dream
<logdir>/scope/epstats-policy_image.mp4/<step>-<id>.mp4     what really happened
```

So "render a video periodically through a seven hour run" is a directory listing, not
a second process and not a checkpoint reload. `scopevid.py` is that directory listing.
The report shows the newest clip of each, plus a filmstrip of one still per report
across the whole run, which is where the world model visibly sharpens.

## The arena: a world with things in it

The stock `walker_walk` is a walker on an empty checkered strip, and it is a bad
demo for two separate reasons that have the same fix.

The camera tracks the walker's centre of mass, so **a walker that is walking and a
walker that is shuffling on the spot look nearly identical**. That leaves the reward
number as the only evidence, and the reward number is precisely the thing you want a
second opinion on. Separately, an empty world gives the model almost nothing to
predict except the consequences of its own actions, which is the weaker half of what a
world model does.

`arena.py` puts objects in the world. It is a real `dm_control` domain, registered into
`suite._DOMAINS` on import, so `--task dmc_arena_walk` just works with no fork of
dm_control and no patch to dreamerv3.

```
    ┌────────────────────────────────────────────────────────────┐
    │  ▌   ▌   ▌   █   ▌   ▌   ▌   ▌   █   ▌   ▌   ▌   ▌   ▌     │  markers
    │        ╷                                                   │
    │       ╱ ╲            ●          ●              ●           │  walker + balls
    │  ─────────────────────────────────────────────────────────  │  floor
    └────────────────────────────────────────────────────────────┘
```

**Markers** are 65 posts along the far edge of the track, every 0.5 m with a taller one
every 5 m. They carry `contype=0 conaffinity=0`, so they are invisible to the physics:
nothing can touch them and they cannot move the reward by an epsilon. They exist to be
seen. Because the camera tracks the walker, they stream past in exact proportion to
real forward travel, which turns "is this thing actually going anywhere" into something
you answer by looking at the video for two seconds.

**Balls** are eleven real free bodies with mass, and the walker kicks them. They are
the part the world model has to earn: their motion follows from physics rather than
from the policy's actions, so predicting them is only possible if the model has learned
something about the world and not just about itself.

### Two invariants, and why they are worth the code

**The reward is the stock reward.** `ArenaWalker` subclasses
`dm_control.suite.walker.PlanarWalker` and does not override `get_reward`. That keeps
DreamerV3's published `walker_walk` numbers applicable, so if a run underperforms the
props are not an excuse. Verified: same seed, same zero actions, arena and stock both
return 0.0138; over twelve random-action episodes, stock scores 30.6 ± 1.9 and arena
31.3 ± 2.4.

**The proprioceptive observation is the stock observation.** Adding a free body adds
six dofs to `qvel` and shifts the rows of `xmat`, which would silently change the state
vector. `ArenaPhysics` slices both by explicit name, so a proprio agent sees exactly the
9 velocities and 14 orientations it would see in the stock domain.

Which makes the split clean, and it is the sentence worth saying out loud on stream:
**the balls exist in the physics, they exist in the pixels, and they do not exist in the
state vector.** Only a model that learns from pixels has to explain them.

### The reward-hacking check

Walker's reward is `stand_reward * (5 * move_reward + 1) / 6`. A policy that learns the
standing half and never moves still banks a sixth of the movement term and plateaus
near a return of 300 looking perfectly respectable. So `arena.py` also logs

```python
obs['log/x_position'] = physics.torso_x()
```

which embodied aggregates per episode and strips before the agent sees it. The report
plots **metres actually travelled** directly beneath episode return. Score climbing
while distance stays flat is that failure mode, and it shows up in the curve long
before it is obvious in the video.

## Why bother, when PPO already walks

The short version. **PPO learns to drive by only ever driving.** Every lesson costs a
real trip, and the only feedback is whether you arrived. **Dreamer drives a little,
builds a mental model of how the car responds, then practises in its head thousands of
times**, going out occasionally to check the model still matches reality.

That is not a metaphor here, it is the actual arithmetic. This run takes one gradient
step per four environment steps, and each gradient step imagines `1,024 x 15 = 15,360`
transitions:

```
   per real environment step   ─────►   ~3,840 imagined steps

   per second of wall clock:
     20 real steps  ──────────────────►  76,800 imagined steps
     0.5 s of real experience lived      32 MINUTES of practice imagined
```

Every second, the agent lives half a second of real life and daydreams half an hour.

### The benefit that is easy to miss

PPO learns **only from the score**. One number per step, and most of it is noise. A
step that scores nothing teaches it almost nothing.

Dreamer learns from **the whole picture**. Every single frame, it has to reconstruct
64x64x3 pixels, predict the reward, and predict whether the episode ends. That is
thousands of training signals per step instead of one, and none of them depend on the
agent doing anything good. It is why `image` loss falls from 628 to 25 long before the
return moves at all: the model is learning what the world *is* while the policy is
still flailing on the floor.

### The catch, which is the important half

Sample efficiency is not the same as speed. Both numbers below are measured on this
same box:

| | steps needed | steps per second |
|---|---|---|
| PPO ([rl-mujoco](../rl-mujoco), MJX, 2,048 envs) | ~20,000,000 | **59,757** |
| DreamerV3 (here, 4 envs) | ~350,000 | **20** |

Dreamer needs roughly 50x fewer steps and runs roughly 3,000x slower per step. For
*this* task on *this* box, PPO very likely finishes sooner in wall clock, because MJX
parallelises thousands of simulators on the GPU and makes environment steps nearly
free.

So the rule is not "model-based is better". It is:

> **Model-based wins when a step is expensive.**

A real robot that can break. A simulator that will not parallelise. A system where
exploring is unsafe or slow or costs money. When steps are cheap, you are not buying
speed.

### What you are actually buying

PPO's output is a policy. It knows what to do and can tell you nothing about the world.
Dreamer's output includes **a simulator you can roll forward, plan with, reuse under a
different reward, and look directly at**. That last one is the red-bordered dream strip
this whole repo is built around, and there is no equivalent artifact anywhere in a PPO
run.

### The same budget, measured

`baseline.py` runs PPO at DreamerV3's exact step budget so this stops being a citation.
Both rows measured on this box:

| at 500,000 environment steps | observation | best return | wall clock |
|---|---|---|---|
| PPO (stable-baselines3, MlpPolicy) | state, 24 numbers | **223** | 43 min |
| DreamerV3 (`dmc_vision size12m`) | 64x64 pixels | **987** | 6.98 h |

Note which way the handicap runs. PPO got the **easier** observation, a clean state
vector, while Dreamer had to learn from pixels, and Dreamer still finished 4.4x higher.
Walker's return is capped at 1,000, so 987 is a solved task and 223 is a walker that
has learned to stand up and not much else. Meanwhile PPO burned through those steps in
43 minutes against Dreamer's seven hours, which is the same tradeoff as before: far
fewer steps, far more compute per step.

Two caveats, because this comparison is easy to overclaim:

**These are stable-baselines3 defaults, not a tuned PPO.** dm_control's action scaling
and reward magnitudes are not what those defaults were chosen for, and a PPO tuned for
this task would do better than 223. The number to read here is the order of magnitude,
not the decimal.

**PPO's step count is entangled with its parallelism**, and this one used 8
environments. That is not a footnote, it is worth its own measurement, which is what
`braxppo.py` does on the MJX stack from the [rl-mujoco](../rl-mujoco) event.

### How much of the gap is parallelism

Brax PPO on `WalkerWalk`, state observations, same task and budget at three environment
counts, asking how many environment steps each needs to reach a return of 900. All
three measured on this box:

| parallel envs | steps per update | best return | steps to reach 900 | wall clock |
|---|---|---|---|---|
| 64 | 61,440 | 941.0 | **7,372,800** | 35.5 min |
| 512 | 491,520 | 960.0 | 19,660,800 | 14.6 min |
| 2,048 (Playground's default) | 983,040 | 969.7 | 29,491,200 | 22.9 min |

**The curves do not collapse**, so parallelism was not a rounding error in the
comparison. Between 64 and 2,048 environments the same algorithm on the same task needs
four times as many environment steps to reach the same score. Read as gradient updates
instead, the ordering reverses: 120, 40 and 30. The low-parallelism run does four times
the optimisation work on a quarter of the data, which is the tradeoff in one line.

That means the honest headline is the **smallest**-parallelism row. DreamerV3 reached
987 in 500,000 steps, so against the most sample-efficient PPO here the ratio is about
**15x**, and against Playground's own 2,048-environment default it would be about 59x.
Quoting the second number without saying how many environments produced it is exactly
the overclaim this file exists to prevent.

Two things keep this honest in the other direction. PPO does **solve** the task, at
941 to 970, so this is a sample-efficiency story and not a capability one. And it gets
there in about thirteen minutes of wall clock against Dreamer's seven hours: Dreamer
wins on samples by roughly 15x and loses on wall clock by roughly 32x.

One caveat on two of the three rows. Brax overshoots its step budget through epoch
rounding, so the 2,048-environment run actually ran about 29.5M steps against a 20M
request and crossed 900 on its final evaluation. Its number is a lower bound rather
than a measured threshold, and 19.7M is marginal for the same reason. Only the 64-row
crossed comfortably inside its budget, which is the other reason it is the row to quote.

## Why MuJoCo, and why next to rl-mujoco

DreamerV3's standard continuous-control benchmark is the **DeepMind Control Suite**
(`dm_control`), which runs on MuJoCo. That is the same physics engine as the
[rl-mujoco](../rl-mujoco) event, so this is a genuine like-for-like: same simulator,
same class of locomotion problem, two opposite approaches.

The shape of the two workloads is completely different, and it shows up in the numbers.
MJX steps thousands of environments in parallel on the GPU. Dreamer steps a handful on
the CPU and trains a small network on the GPU. Measured here, that makes the **gradient
step** the bottleneck, not the simulator: ~5 gradient steps/s for the 12M parameter
model on pixels, which at `train_ratio 256` pins the environment at ~20 steps/s. The
simulator itself will do 694 steps/s in the arena and 891 in the stock domain, so it
spends most of its time waiting.

That is why `--envs` defaults to 4 here rather than upstream's 16. Environment count
buys no throughput when training is the limit, and fewer environments means the first
episode finishes at step 4,000 instead of 16,000, so the live report gets a score and a
rollout video four times sooner.

## Run it

```bash
cd topics/dreamerv3
./setup.sh                      # venv, upstream checkout, the Blackwell patch
```

On the host:

```bash
export PYTHONPATH=~/dreamerv3:.
./.venv/bin/python launch.py \
  --configs dmc_vision size12m --task dmc_arena_walk \
  --logdir ~/logdir/arena --run.log_every 10
```

In a Flyte pod, against the `world-models` project:

```bash
./.venv/bin/flyte run pipeline.py dream                       # the flagship, ~7 h
./.venv/bin/flyte run pipeline.py dream --steps 20000         # is the plumbing alive?
./.venv/bin/flyte run pipeline.py dream --task_id dmc_walker_walk   # stock domain
./.venv/bin/flyte run pipeline.py dream --config dmc_proprio  # state vector, no dream
```

`launch.py` rather than upstream's `dreamerv3/main.py` because the `arena` domain has
to be registered inside the process that loads the environments, and upstream's entry
point has no hook for that. It adds no flags of its own.

## Status: it dreams, and it walks

Verified on this box (DGX Spark, GB10, aarch64, driver 580.126.09, CUDA 13.0), Flyte
run `rdmmb2c4p5g9rnsdgxkf` in the `world-models` project, 2026-08-31.

```
task             dmc_arena_walk, dmc_vision size12m, 64x64 pixels
agent            10,494,158 parameters
run              500,000 env steps in 418.9 min (6.98 h), 496 episodes
throughput       19.9 env steps/s, ~5 gradient steps/s (the gradient step is the limit)

episode return   42.2  ->  934.8 final, 987.3 best
metres travelled  1.9  ->   53.2 best
world model      image 628.6 -> 24.9 | rew 5.43 -> 0.57 | con 0.26 -> 0.02
actor / critic   ent/action 7.87 -> 1.07 | value 3.75 -> 1.24

dream clips      139, one every ~3 min, all non-black
final replay     600 frames at 480x480, luminance mean 71.9, 0 black
```

Read the two headline numbers together. Walker's per-step reward is capped at 1.0 and
the episode is 1,000 steps, so 987 is within a couple of percent of the arithmetic
ceiling and comfortably at the published DreamerV3 asymptote for this task. And 53 m in
a 25 second episode is about 2.1 m/s, which is worth a second look: `move_reward`
saturates at 1 m/s, so **there was no reward whatsoever for going faster than half that
speed**. It did anyway, presumably because a committed gait is easier to keep upright
than a careful one.

Three things worth knowing about the shape of the curve:

**The world model learns long before the policy does.** `image` fell from 628 to 52 in
the first 4% of the run, while the return was still 85 and the walker was flailing on
the floor. That gap is the argument for model-based RL in one line: every frame teaches
the model something, whether or not the agent did anything good.

**`dyn` and `rep` go UP, and that is not a regression.** They ended at 11.3 having
started at 10.8, while every other loss fell. The KL is measured against a moving
target: as the policy improves it visits faster and more novel states, so the world
genuinely becomes harder to predict even as the model gets better at predicting it.

**Return climbing early is standing up, not walking.** The reward is
`stand_reward * (5 * move_reward + 1) / 6`, and the standing term multiplies, so moving
pays nothing until the walker is upright. At 4% the return had doubled to 85 while
distance sat at 2.8 m. The distance curve is what separates those two stories, and it
is the reason it is in the report.

## Things that cost time, so you do not pay twice

**Upstream's pinned JAX cannot use this GPU.** `requirements.txt` upstream says
`jax[cuda12]==0.4.33`. The GB10 is Blackwell, compute capability sm_121, and CUDA 12
builds of that vintage ship no kernels for it. This uses `jax[cuda13]==0.9.2`, the
version [rl-mujoco](../rl-mujoco) already proved on this box.

**Moving JAX forward breaks `jax.jit`, in exactly six places.** Newer JAX made every
argument after `fun` keyword-only, and upstream still calls it positionally:

```
TypeError: jit() takes from 0 to 1 positional arguments but 5 were given
```

`patches/0001-jax-jit-keyword-only.patch` converts those six call sites to keyword
arguments. No behaviour change. It is applied at image build time against a pinned
upstream commit, with `git apply --check` first so a future bump fails the build loudly
instead of producing an image that dies at agent init.

**`jax.prealloc` reserves 75% of "GPU memory", and on a GB10 that is system memory.**
Measured on the host: 90 GB reserved for a 10M parameter model, 107 GB of 119.7 GB in
use. Harmless on the host, an instant kill inside a pod with a cgroup limit.
`pipeline.py` passes `--jax.prealloc False`.

**A pixel replay buffer costs ~151 KB per transition, not 12 KB.** The frame is only
12 KB; the rest is the `replay_context` carry states stored alongside it. Upstream's
`replay.size` default of 5e6 would want 750 GB, and a 500k-step run would want 75 GB.
`pipeline.py` caps it at 200k transitions, about 30 GB.

**`dmc_vision` does not inherit a size preset, and `dmc_proprio` does.** `dmc_proprio`
merges `size1m`; `dmc_vision` merges nothing, so it silently means `size200m`, which is
twenty times the compute per gradient step for a task this small. Pass the preset
explicitly: `--configs dmc_vision size12m`.

**`run.log_every` is a wall clock timer, not a step count, and it defaults to
minutes.** A short smoke run finishes before it ever fires, writes an empty logdir, and
looks exactly like a run that did nothing. It does not help that training also only
*starts* once the replay buffer holds `batch_size * batch_length` transitions, and that
the first episode only ends after `run.envs * 1000` steps. Pass `--run.log_every 10`
when smoke testing, and do not conclude anything from an empty logdir.

**dm_control's named indexer broadcasts, it does not take an outer product.** The stock
walker gets away with `xmat[1:, ['xx', 'xz']]` because a slice and a list compose. Swap
the slice for a list of body names and it raises `shape mismatch: indexing arrays could
not be broadcast together with shapes (7,) (2,)`. Index rows first, then columns.

**Setting a free joint through the named accessor floods stdout.** `named.data.qpos['ball_0'] = [...]`
reshapes a view to assign into it, which numpy 2.5 deprecates. Eleven balls times
several episodes a minute is a wall of tracebacks in the log tail the report shows.
Write the raw `qpos[adr:adr+7]` slice instead.

**torch segfaults on import in an aarch64 pod, and the guard cannot catch it.**
Only relevant to `baseline.py`, which is the one thing here that uses torch. Building
a plain Adam optimizer calls `torch.utils._triton.has_triton_package()`, which does
`import triton`, and triton's aarch64 native extension crashes on import:

```
Fatal Python error: Segmentation fault
  torch/utils/_triton.py:39   has_triton_package
  triton/__init__.py:8 -> runtime/autotuner.py -> knobs.py:15   <- SIGSEGV
```

`has_triton_package` wraps that import in `except ImportError`, which is useless,
because a segfault is not an exception. The pod exits 139 three seconds in with no
Python traceback at all. `sys.modules["triton"] = None` before importing torch turns
the crash into the ImportError torch already knows how to handle, and costs nothing,
since stable-baselines3 never calls `torch.compile`.

Two things made this findable, and neither was guessing. `faulthandler.enable()`, which
converts a silent native crash into a stack; and filtering dm_control's numpy 2.5
DeprecationWarning, which otherwise repeats so often that the last forty lines of
`kubectl logs` contain nothing else. Arm both before debugging anything native.

**A CPU task still has to fit next to the GPU task.** This box has 20 cores and a
Dreamer training pod requests 12 of them. A companion CPU task asking for 6 sits in
`Pending` with `0/1 nodes are available: 1 Insufficient cpu` for as long as the seven
hour run lasts, which quietly defeats the point of making it CPU-only. `kubectl
describe node` under "Allocated resources" is the check; `baseline.py` asks for 3.

**A run's code is snapshotted at submit, not at commit.** `flyte run` bundles the
source tree as it is the moment you press enter, so editing a file five minutes after
launching a seven hour job means the code that produced the result is neither what is
on disk nor what is in git. That is recoverable only if you committed it, and the pod's
logs age out, so the numbers survive and the thing that produced them does not. This
run has one such artifact: a flagship reporting 11,347,664 parameters where current
code builds 10,494,158 agents on the identical task, config and size, with no way left
to explain the difference. Commit before you launch anything long, or accept that the
result is a measurement you cannot repeat.

**Nothing in the logdir survives the pod.** Dreamer writes everything, checkpoints
included, under `/tmp/dreamer/<task>`, and Flyte takes that with it when the task
exits. Two seven hour runs finished here before anyone noticed that the only thing left
behind was the HTML report: the trained agents were gone, which makes every follow-up
demo a retrain. `_persist` in pipeline.py now uploads `config.yaml`, `ckpt`,
`metrics.jsonl` and `scope` as a `Dir` output. It stages a subset rather than uploading
the logdir because the replay buffer is about 30 GB at `_REPLAY_SIZE` and none of it is
needed to load an agent.

**`run.save_every` defaults to 900 seconds, which makes short runs silently agent-less.**
Any run under fifteen minutes finishes having written no checkpoint at all, so a
persistence step uploads a directory with no agent in it and the run still reports
success. Nothing raises. This is the third silent-absence bug in this demo, after the
two `log/` failures above, and the lesson repeats: check that the artifact exists
rather than assuming the code that writes it ran. It is a `save_every` parameter now.

**Playground now defaults MuJoCo to the warp backend, which is not installed.** Only
relevant to `braxppo.py`. `mujoco_playground` 0.2.0 returns `impl: warp` from
`get_default_config`, which sends `mjx.put_model` down a path that reads
`mujoco_warp.types.GraphMode.WARP`. There is no aarch64 wheel for mujoco-warp, so
mujoco-mjx 3.12 falls back to a stub where `GraphMode` is plain `int` and the pod dies
five seconds in with

```
AttributeError: type object 'int' has no attribute 'WARP'
```

which names neither warp nor the missing package. `config_overrides={"impl": "jax"}`
fixes it. Worth knowing that `topics/rl-mujoco` pins the same unpinned `mujoco`,
`mujoco-mjx` and `playground`, so it will hit this the next time its image rebuilds.

**Do not name a Flyte task parameter `task`.** It collides with the runner's own
argument and the run dies before it starts, with `_Runner.run() got multiple values for
argument 'task'`. This uses `task_id`.

**`PLATFORM` must be a tuple.** flyte hands it to `docker buildx build --platform`, so
a bare `"linux/arm64"` string is iterated character by character into
`l,i,n,u,x,/,a,r,m,6,4` and buildx rejects `"l"` as an unknown operating system.

**Filming needs the envs in process.** embodied's `Driver` defaults to one subprocess
per environment. You can step a subprocess env perfectly well, but you cannot reach its
`physics` object from the parent, so there is nothing to render. `replay.py` builds its
Driver with `parallel=False`.

**The checkpoint directory contains a pointer file, not just checkpoints.** Alongside
the timestamped directories sits a 22-byte `latest` file. Taking the last entry by name
grabs that pointer and dies on `assert exists(path)`. Hand `elements.Checkpoint` the
directory and let it resolve which checkpoint to load.

**A black clip is a valid mp4.** It embeds in a report as a black rectangle and nothing
complains, which is how the Isaac Sim demo lost an afternoon. Every clip here goes
through a mean-luminance probe before it reaches the report.

## Reference

The lineage, in order, if you want to read it as a sequence rather than a pile:

- [World Models](https://worldmodels.github.io/) (Ha and Schmidhuber, 2018). Still the
  best first read: an interactive paper, and the one that established that a policy can
  be trained entirely inside a dream.
- "Learning Latent Dynamics for Planning from Pixels" (PlaNet, 2019), where the RSSM
  and its deterministic-plus-stochastic latent come from.
- "Dream to Control" (Dreamer, 2020) and "Mastering Atari with Discrete World Models"
  (DreamerV2, 2021), which replace planning with an actor and critic trained on
  imagined rollouts.
- [DreamerV3](https://github.com/danijar/dreamerv3) (Danijar Hafner), and its paper
  "Mastering Diverse Domains through World Models". The code this repo runs.

And the tooling:

- [DeepMind Control Suite](https://github.com/google-deepmind/dm_control), the MuJoCo
  task set, and the base `walker` domain that `arena.py` extends
- [rl-mujoco](../rl-mujoco), the model-free counterpart on the same physics engine
