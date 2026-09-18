# World Models with V-JEPA 2: prediction in representation space

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This one looks at world models from a different angle. Where [Cosmos](../cosmos) generates the future in pixels, [V-JEPA 2](https://github.com/facebookresearch/vjepa2) predicts in representation space: it learns by predicting the abstract embeddings of masked video rather than reconstructing every pixel. That is the core thesis of the JEPA line of work, Yann LeCun's argument that predicting representations is more efficient, and better for reasoning and planning, than generating pixels.

It runs on Flyte, on one DGX Spark, and the reports have video in them.

## The thing that makes this demo different

V-JEPA 2 **has no decoder**. The predictor emits 1024-dimensional vectors and there is no head in any released checkpoint that turns one back into a pixel. So "show me what it predicted" is not a screenshot anyone can take, and any image claiming to be one is a projection of a vector, not a prediction.

That constraint shapes everything here. Every frame in these reports is one of three things:

- the model's **literal input**, with the masked patches blacked out
- a **per-patch number we computed**, painted onto those same pixels
- a **rendering of a token**, shown next to the identical rendering of the *encoder's own token* at the same position, which is that rendering's ceiling

Nothing is a vector dressed up as an image. The third case is the newest and the one that needs the most care: the `occlude` task does show you what the predictor thought was behind the occluder, by pasting in the real pixels of the nearest patch in a bank of other clips, and it is only interpretable because the ceiling panel is in the same frame. The corollary is that the demo has to carry its weight in measurements rather than in pretty pictures, so every score is reported next to its own chance floor.

## What it does

```
vjepa (orchestrator, CPU)
  ├── inpaint (GPU)   hide part of a clip, predict it in latent space, score it twice
  ├── probe   (GPU)   freeze the encoder, train one linear layer, 5-way action recognition
  └── scale   (GPU)   ViT-L vs ViT-g, same clips, same masks, same probe

V-JEPA 2-AC, the action-conditioned world model (separate, needs the 11.7 GB checkpoint)
  ├── plan    (GPU)   a Franka in MuJoCo reaches a goal PHOTO, planned in latent space
  ├── dream   (GPU)   open-loop latent rollout, made watchable by retrieval
  └── adapt   (GPU)   fine-tune the predictor on 480 sim transitions, then re-plan

readout (GPU)   is the picture inside the token? Fit a linear map and find out
push    (GPU)   show it a photo of the finished job and let it work out the rest

The mechanism tasks: how it works, rather than how well
  ├── occlude  (GPU)  slide an occluder across a clip and RENDER the prediction
  ├── energy   (GPU)  the E in energy-based model, mapped over candidate completions
  ├── ladder   (GPU)  where in the 24 layers appearance becomes meaning
  └── collapse (GPU)  train a tiny JEPA from scratch four ways; watch one collapse
```

```bash
./setup.sh                                  # local venv (the pods build their own image)
.venv/bin/python smoke_test.py --full       # ~4 min, checks the claims before spending a pod

.venv/bin/flyte run pipeline.py inpaint     # ~30s after the image is built
.venv/bin/flyte run pipeline.py inpaint --clip archery --context 0.75
.venv/bin/flyte run pipeline.py probe       # ~3 min
.venv/bin/flyte run pipeline.py scale       # ~6 min, two encoders
.venv/bin/flyte run pipeline.py vjepa       # all three in sequence, ~10 min

.venv/bin/flyte run pipeline.py occlude     # ~70s, the prediction rendered
.venv/bin/flyte run pipeline.py energy      # ~60s
.venv/bin/flyte run pipeline.py ladder      # ~3 min, all 24 layers
.venv/bin/flyte run pipeline.py collapse    # ~25 min, six models trained from scratch
.venv/bin/flyte run pipeline.py mechanism   # all four in sequence
```

Runs to the `world-models` Flyte project, alongside [topics/cosmos](../cosmos) and [topics/dreamerv3](../dreamerv3). The three are the same question asked three ways: Cosmos predicts the future in pixels, Dreamer learns a latent world model of one environment from its own experience, V-JEPA 2 predicts representations learned self-supervised from internet video.

Data is [`nateraw/kinetics-mini`](https://huggingface.co/datasets/nateraw/kinetics-mini): 100 clips, 5 classes, ungated, under 200 MB, and already the dataset the transformers V-JEPA 2 docs use.

Verified end to end on the devbox on 2026-08-10: all four tasks green, including the full `vjepa` orchestrator (run `rrzl29rg57rpt2l85h6x`). The four mechanism tasks were verified on 2026-09-17 (`rt548mj42wdmn6fvrdgq`, `rcvt55cswlmt68f5429g`, `r89wqwp8znhg9cjdfkvv`, `rzmbthndjdc9t2q7cqdx`). Every number in this README comes from one of those runs, and `smoke_test.py --mech` re-checks the claims on the host before a pod is spent.

> If a report renders blank in the Flyte console, that is not a bug in the demo: forward port **30002** (rustfs) in VSCode so the browser can reach the object store the report assets live in.

## The finding: this checkpoint inpaints, it does not forecast

This is the part worth the stream time, and it was not what I expected going in.

The obvious experiment for a "world model" is: show it the first half of a clip, ask it to predict the second half, measure. Do that and the numbers are bad. The interesting question is *why*, and the answer is that **it is the wrong question to ask this checkpoint**.

V-JEPA 2's pretraining masks are **tubes**: a spatial block removed across the *entire* temporal extent of the clip. Every masked token always had visible tokens from its own timestep to attend to. No token was ever predicted from a strictly earlier one. So temporal extrapolation is not something the predictor does badly, it is something it was never asked to do.

`inpaint` runs both masks against the same clip, same predictor, same scoring, with **the same fraction of tokens hidden** (two 8x8 tube blocks is ~48%; a half-clip future mask is 50%) so that the only difference is the *shape* of the hole.

Scoring is deliberately paranoid. Cosine similarity between two ViT tokens is a number that always looks encouraging, so the headline metric is a retrieval one: for each predicted token, find its nearest neighbour **among the masked tokens only** (restricting candidates to the targets is what stops a model scoring well by copying a visible neighbour), then measure *how far off in the raster* that neighbour is. `median dt` is the gap in tubelets. Each mask is also scored with its predictions shuffled, which gives the chance floor for that exact mask.

Measured on the Spark, `facebook/vjepa2-vitl-fpc64-256`, 64 frames of `val/bowling/--dVV4_CSvw`, run `rscs9t9hhlft7hsk5ddz`:

| mask | hidden | cosine | top-1 | median dt | chance dt | time localised |
|---|---|---|---|---|---|---|
| **tube** (pretraining mask) | 48% | 0.317 | 3.9% | **0.0** | 10.0 | **1.00** |
| **future** (never trained on) | 50% | 0.276 | 1.2% | **3.0** | 5.0 | **0.40** |

Under the mask it was trained on, the predictor's retrieved token is at **exactly the right moment**, median 0 tubelets off where chance is 10. Asked to extrapolate forward instead, it lands 3 tubelets away against a chance level of 5. It is returning a plausible representation of roughly the right scene at the *wrong time*.

Note the cosine column barely moves (0.317 vs 0.276) while the localisation collapses. That is the whole reason the report leads with `dt` and not with cosine: **cosine is the metric that would have let this pass unnoticed.**

The same pattern holds across every clip and both encoder sizes tested. It is also, reassuringly, exactly what the architecture predicts, and it is the gap that **V-JEPA 2-AC**, the action-conditioned post-train, exists to close.

> **Scope note.** The pretrained predictor is the honest scope of everything above, and `inpaint` measures exactly where it stops being a world model. V-JEPA 2-AC, the action-conditioned post-train that closes this gap, now has its own section below: it is not on the Hub, but the weights are a direct download and it turns out to be very much runnable.

## The other finding: the features are strong, and you have to centre them

`probe` freezes the encoder, mean-pools each clip into one vector, and trains a single 1024x5 linear layer. Nothing else is trained anywhere.

Run `r8r29vr8db5gc25cgm6n`, 100 clips, the dataset's own 50/50 train/val split:

| | accuracy |
|---|---|
| **V-JEPA 2 features + linear probe** | **78%** |
| raw pixels + the same linear probe | 40% |
| V-JEPA 2 1-NN retrieval, no training at all | 82% |
| chance | 20% |

The pixel baseline is the one that matters. "78% on 5-way action recognition" is unreadable on its own; 78% where a downsampled space-time thumbnail of the same clips gets 40% is a claim about the *representation*.

The second result is a trap worth knowing about. **V-JEPA 2's token cloud sits in a narrow cone.** The mean cosine between two random patch tokens of the same clip is 0.28 at the last layer and 0.92 at layer 8: raw cosine is dominated by a component every token shares and tells you almost nothing. Subtract the per-clip mean token first and random pairs drop to ~0.00, while adjacent patches sit at 0.33 and distant patches at 0.08.

That is not cosmetic, and the demo puts a number on it rather than asserting it: the *same* retrieval scores **82% centred and 76% raw**. Every similarity in this repo is computed on centred features for that reason.

## Does a bigger encoder fix the forecasting? No

`scale` runs both measurements on two encoder sizes, same clips, same masks, same probe. Run `rzcfj6hrb2z9b65fsxnj`, 32 frames, masks matched at 48% / 50%:

| encoder | params | linear probe | 1-NN retrieval | tube, time localised | future, time localised |
|---|---|---|---|---|---|
| ViT-L/16 | 326M | 78% | **82%** | 1.00 (dt 0, chance 5) | 0.00 (dt 3, chance 2) |
| ViT-g/16 | 1035M | **82%** | 79% | 1.00 (dt 0, chance 5) | 0.00 (dt 2, chance 2) |

Three separate things worth noticing. Tripling the parameters buys **4 points of probe accuracy** and, on this small a benchmark, slightly *worse* retrieval, so the two ways of asking "are the features good?" do not agree. Both models already saturate the in-distribution mask. And neither one can extrapolate forward in time at all: **scaling the encoder does not turn an inpainter into a forecaster**, which is what you would expect if the limit is the pretraining objective rather than capacity.

## What did not work, and is therefore not in the demo

Worth writing down, because the first version of this demo had all of it and it was all quietly wrong.

- **PCA-of-patch-embeddings feature videos.** The DINOv2 trick of projecting patch tokens to 3 components and rendering them as RGB. On V-JEPA 2 it is noise: the top 3 components explain ~20% of variance and the result is a rainbow with no visible object structure. Tried layers 8/14/20/24 and four normalisations; none were legible.
- **Query-patch correspondence tracking.** Cosine from one patch to all others, over time. Diffuse and uninformative once the anisotropy above is accounted for.
- **A "what is moving" salience map.** Hot on the *blank white wall*, because in flat regions the centred embedding has small norm and its direction is noise. An artifact of normalising, not a signal.
- **Overlays composited on the source clip.** `AutoVideoProcessor` resizes the shortest edge to 292 and centre-crops to 256, so a 480x270 clip loses most of its width. Painting a 16x16 patch grid onto a naive resize of the *original* is misaligned by tens of pixels, and the heatmaps looked plausible the whole time. `clips.shown_pixels()` inverts the normalisation and hands back exactly the tensor the patch embedding consumed; every overlay composites on that.

The general lesson: V-JEPA 2 is not DINOv2. It has no register tokens, its spatial grid is coarse (16x16), and its patch tokens do not give clean semantic maps out of the box. A demo whose centrepiece is a picture that does not hold up is worse than one that leads with a measurement.

## V-JEPA 2-AC: planning a real robot's world model in a simulator it has never seen

The section above ends on a scope note: V-JEPA 2-AC, the action-conditioned post-train
that closes the forecasting gap, was not on the Hub and had no `transformers` class,
so it was out of scope. That has changed enough to be worth a second demo.

`vjepa2-ac-vitg.pt` is 11.7 GB at a direct URL out of the vjepa2 README. It is a ViT-g
encoder plus a **frame-causal, action-conditioned predictor**:

```
predictor(tokens of frame t, action_t, pose_t)  ->  tokens of frame t+1
```

7-DoF actions `[dx, dy, dz, droll, dpitch, dyaw, grip]` in metres of end-effector
motion, the interface DROID's Cartesian controller exposes. That is a real forward
model, and it makes planning possible without a reward function, without training and
still without a decoder: photograph the goal, encode it once, and search for the action
sequence whose **imagined** latent lands closest to the goal's embedding.

```bash
flyte run pipeline.py plan      # Franka reaches a goal photo, planned in latent space
flyte run pipeline.py dream     # open-loop rollout, decoded by retrieval
```

### Getting it to run at all

Four things cost real time, and none of them are in any documentation.

- **`torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")` is broken on
  main.** `src/hub/backbones.py` ships `VJEPA_BASE_URL = "http://localhost:8300"` under
  a `# for testing` comment, so the hub entrypoint fetches the checkpoint from your own
  machine and dies on connection refused. `ac.py` builds the two modules directly and
  loads the `.pt` itself.
- **`pip install git+https://github.com/facebookresearch/vjepa2` installs nothing
  importable.** Its `setup.py` declares no `packages=` and no `py_modules=`, so pip
  reports success and `import src` still fails. The image clones the repo at a pinned
  commit instead.
- **bfloat16 does not work.** The AC predictor's RoPE upcasts q and k to float32 and
  leaves v alone, so SDPA raises `Expected query, key, and value to have the same
  dtype`. fp32 only, which costs about 20 ms per sampled action.
- **`app.vjepa_droid.transforms` imports cv2 at module scope** for what, at the settings
  the AC model uses, is a bilinear resize and an ImageNet normalise. `ac.transform` is
  six lines; `smoke_test.py --ac` asserts it is **bit-exact** against upstream's
  (measured max abs diff 0.0) rather than assuming it.

### The simulator has to be right before the world model can be wrong

`sim.py` is a Franka Panda on a wooden table with clutter and a red target block, lit
and framed to look like a DROID scene rather than like MuJoCo. The arm takes Cartesian
deltas, which is the only interface the world model speaks.

The part worth stealing is the controller. The Panda's actuators are PD position servos
with no integral term, and solving IK once and commanding the result **delivers 33 mm of
a 50 mm request**. Gravity compensation does not fix it and neither does settling time
(9.2 mm residual at 60 substeps, still 9.2 mm at 1000). Re-deriving joint targets from
the drooped pose reproduces the same offset every iteration and converges to nothing.
Accumulating the correction onto `ctrl` instead is integral action, and it converges:
**16.1 mm mean error becomes 0.66 mm.**

That is a bug that would never have announced itself. The arm would simply have
under-reached, and the report would have read "the world model cannot plan" with no hint
that the controller was eating a third of every commanded motion.

Two smaller traps, both silent: a bare `<compiler angle="radian"/>` anywhere in the
scene resets `meshdir` globally, including the one `panda.xml` sets for itself, so every
Panda mesh fails to open with a path that looks almost right. And a camera placed past
the back wall renders **flat brown with no error at all**, which is why `smoke_test.py`
checks the render's standard deviation.

### The finding: the representation transfers, the dynamics do not

Five policies, four scenes, one controller, 10 steps, 5 cm per step. Only the policy
differs. Run `rvfcl6l4d54bl9ljnt7z`, mean over 4 seeds (the target moves with the seed,
so start distances range 12.7 to 28.1 cm):

| policy | next latent comes from | gap closed | final | reached (of 4) |
|---|---|---|---|---|
| `jepa` (CEM, horizon 2) | **the world model** | **-103%** | 40.7 cm | 0 |
| `greedy` (1-step grid) | **the world model** | -131% | 50.3 cm | 0 |
| `random` | - | -2% | 19.9 cm | 0 |
| **`lookahead`** (1-step grid) | **the simulator** | **+90%** | **1.8 cm** | **4** |
| `oracle` (told the coordinates) | - | +100% | 0.1 cm | 4 |

Per seed, `lookahead` closes +83%, +95%, +94%, +88% and reaches every time; `jepa`
closes -115%, -124%, -95%, -79% and never does. The pattern is not one lucky scene.

Read the middle column. `greedy` and `lookahead` are **the same search with the same
reward over the same action grid**. The only difference is whether the next latent is
*imagined by the predictor* or *observed by taking the action*. One drives the arm into
its joint limits; the other puts the gripper 1.8 cm from a block it was never told the
position of, on every scene.

So the two halves of this world model transfer very differently to a simulator it has
never seen:

- **The encoder half works, but does not prove anything about V-JEPA.** Its embedding
  distance to a goal photograph correlates **r = +0.77** with true end-effector
  distance, and searching on it alone closes **90%** of the gap with nothing trained
  and no coordinate ever shown to the planner. The catch is in the next section: raw
  pixel difference closes **exactly the same 90%**, so on this task the reward is not
  where V-JEPA earns its keep.
- **The action-conditioned predictor does not.** It is not merely noisy, it is
  *inverted*: correlation between the energy it imagines for an action and the energy
  that action really produces is **r = -0.82**.

### Why, measured three ways

Because "the domain gap" explains everything and therefore nothing, here is what was
actually ruled in and out.

**1. The action is being ignored.** For a grid of candidate actions from the same frame:

| quantity | value |
|---|---|
| \|dream(a) - true(a)\| | 0.40220 |
| \|dream(a) - true(**some other** a)\| | 0.40234 |
| \|z0 - true(a)\|, i.e. predict no change | **0.31212** |

The first two are the same to four decimal places: which action you condition on barely
moves the prediction. And **standing still is a better predictor of the next latent than
the world model is.**

**2. It is the image, not the state vector.** The obvious suspect is that our 7-dim pose
is out of DROID's distribution. It is not. Crossing frame source against pose source and
measuring how far the prediction moves when the action changes:

| input | action sensitivity |
|---|---|
| real DROID frame + real DROID pose | **0.1199** |
| real DROID frame + our MuJoCo pose | 0.1170 |
| MuJoCo frame + real DROID pose | 0.0674 |
| MuJoCo frame + our MuJoCo pose | 0.0674 |

Swapping the pose changes nothing. Swapping the *frame* halves it. The gripper
convention makes no difference either (0.0654 vs 0.0674).

**3. The positive control.** On upstream's own real Franka trajectory the same code
works: the energy landscape over an action grid correlates **r = +0.80** with distance
to the ground-truth action, and playing the trajectory backwards flips all three
components of the recovered action. The predictor is fine. It just does not accept
synthetic renders.

Worth being clear about the limits: this is four scenes of one reaching task, on
renders that are our own attempt at looking like DROID. It says the pretrained AC
checkpoint does not transfer to *these* images, not that latent planning cannot work in
simulation. The obvious next experiments are making the renders photoreal (Cosmos
Transfer, next door in [topics/cosmos](../cosmos)) and fine-tuning the predictor on a few
hundred sim rollouts, which is cheap given the encoder already works.

One tempting dead end, recorded so nobody repeats it: searching all 48 axis
permutations and sign flips does find a remapping that scores +0.85 instead of -0.82.
It is not a frame convention. It is 48 candidates fitted to 27 points, and the positive
control above already shows our convention is the right one on real data.

### The control that deflates half of the above

The table above was first written with `random` and `oracle` as the only controls, and
that was not enough. Swapping the reward for the two cheapest imaginable alternatives,
holding the search, the scenes and the controller fixed:

| reward used by `lookahead` | seed 0 | seed 1 | seed 2 | seed 3 | mean |
|---|---|---|---|---|---|
| V-JEPA 2 embedding distance | +84.2% | +95.9% | +93.8% | +87.9% | **+90.4%** |
| raw pixel L1 to the goal photo | +84.2% | +95.6% | +93.8% | +88.1% | **+90.4%** |
| the SAME ViT, randomly initialised | +84.2% | +95.6% | +93.8% | +87.9% | **+90.4%** |

They agree to a tenth of a percent because they pick the same action at almost every
step. On a fixed camera watching a static scene where the only thing that moves is the
arm, *any* image distance is monotone in arm position, so the task never asks the
representation a question. "V-JEPA's features make a good reward" is not supported here;
"this task is easy" is.

**What this does and does not touch.** It deflates the reward claim only. The dynamics
result is a comparison *within* a fixed reward -- `greedy` and `lookahead` use the same
embedding distance and differ solely in whether the next latent is imagined or observed
-- so swapping the reward changes both arms equally and the -103% versus +90% gap
stands. The same goes for `adapt`, where the reward is held constant and only the
predictor changes.

The lesson is the one the top of this README already makes about cosine similarity,
one level up: a number that moves in the right direction is not evidence until
something that should *not* work has been tried and failed.

### The fix: the broken half is the cheap half

If the encoder already works and only the dynamics fail, the obvious move is to adapt
the dynamics. Freeze the encoder, drive the arm around with random actions from random
workspace positions, encode every transition once, and fine-tune **only** the 305M
predictor on exactly the L1 objective it was originally trained with. No new losses.

```bash
flyte run pipeline.py adapt --episodes 40 --train_steps 400
```

Run `r2gg2rfj6fkv5j5pdx92`, 480 transitions, 400 steps, **4.3 minutes of training**:

| | validation L1 | greedy planning |
|---|---|---|
| predict no change (the bar) | 0.2958 | - |
| pretrained AC | 0.3906 (**above** the bar) | **-120%** |
| **adapted** | **0.2696** (below it) | **+82%** |
| `lookahead`, i.e. a *perfect* dynamics model | - | +91% |

The pretrained checkpoint sits *above* the standstill line, which is the precise sense
in which it is not a world model for these images: on a MuJoCo render, predicting that
nothing happens is a better forecast than asking it. Four minutes of adaptation puts it
below the line, and takes planning from -120% to +82% against a ceiling of +91%.

So after six minutes end to end, a learned world model is within about nine points of
having the simulator itself as its dynamics.

### How little data? 24 seconds of it

480 transitions was an arbitrary choice, so here is the curve. Collect once, then
fine-tune from the pretrained weights on nested subsets, same held-out set and same 400
steps every time, so the only thing varying is how much the model saw.

| transitions | collect time | val L1 | vs the standstill bar (0.2775) | greedy planning |
|---|---|---|---|---|
| 0 (pretrained) | - | 0.3841 | above | **-123%** |
| 30 | 6 s | 0.2954 | above | +2% |
| 60 | 12 s | 0.2792 | above | +45% |
| **120** | **24 s** | **0.2640** | **below** | **+83%** |
| 240 | 48 s | 0.2572 | below | +82% |
| 510 | 102 s | 0.2545 | below | +85% |

The knee is at about 120 transitions, which is **24 seconds of an arm moving randomly**.
Everything past that buys three points.

The more useful observation is the alignment between the two right-hand columns. The
moment the validation L1 drops below the standstill baseline, between 60 and 120
transitions, is the same moment planning goes from mediocre to working (+45% to +83%).
That baseline is not just a rhetorical device for saying the pretrained checkpoint is
bad: **it predicts whether the world model will actually be usable, from an offline
number, without running the robot at all.** Cheap enough to compute that there is no
reason not to, and it is the first thing worth checking on any new domain.

### What did not fix it, and is worth knowing

Before training anything, three cheaper hypotheses were tested on the theory that the
predictor is upset by how *clean* CG renders are. All three failed:

| render style | corr(imagined energy, true energy) | greedy planning |
|---|---|---|
| raw MuJoCo | +0.089 | -123% |
| + gaussian sensor noise | **+0.613** | **-181%** |
| + blur | +0.001 | - |
| + JPEG artifacts | +0.192 | - |

Sensor noise looks like a triumph in the middle column and is a disaster in the right
one. It moved the action-ranking correlation from +0.09 to +0.61 and made closed-loop
planning *worse*. A correlation over 27 candidate actions at a single state is cheap to
move by accident; the quantity anyone actually cares about is whether the arm reaches
the block. Measure that one.

### Watching it dream: imagination vs reality

```bash
flyte run pipeline.py imagine     # ~20 min: bank, dream, adapt, dream again, plan
```

The numbers above are about what the arm does. `imagine` shows what the model *thinks*.
The robot gets three choreographies as lists of hand movements (reach the red block
from above, sweep up and over, trace a square), and the world model dreams each one
open loop from the first camera frame alone: encode it, then repeatedly predict "after
this move the scene looks like...", feeding its own prediction back in. It never looks
at the simulator again. Then the simulator does the same moves, and the report plays
**pretrained dream | adapted dream | reality** side by side. There is no reward
anywhere, and the adaptation between the two dreams is reward-free as well: random
flailing plus "predict your own next frame".

Each imagined embedding is shown as the nearest of 343 real photos of the arm spread
across the workspace (none from the move being dreamed), and that photo's gripper
position gives "where the model thinks the hand went" in centimetres.

Run `r22lpglp94r59m4snz6j`, 53 imagined moves across the three choreographies:

| | imagined hand vs real hand |
|---|---|
| pretrained | **30.3 cm** |
| imagine nothing moves (ignore the actions) | 16.7 cm |
| **adapted** (240 random transitions, 4.2 min) | **10.3 cm** |
| decode floor (the TRUE future through the same photo lookup) | 8.3 cm |

The pretrained dream is not frozen, it is actively *wrong*: worse than imagining that
nothing moves. The adapted dream sits two centimetres above the best any dream could
score with a lookup this coarse, and holds there for 24 moves open loop. Greedy
planning in the same run: -120% before, +76% after.

The table in the report also has the full-history rollout (predictor fed its whole
imagined past, upstream's `dream`): 24.4 cm pretrained, 14.4 cm adapted. Worse for the
adapted model because adaptation only ever trained one frame of context, which is why
the video uses the one-frame rollout.

## A humanoid walks to a photograph

```bash
flyte run pipeline.py walk        # ~50 min
```

The Franka moves a few centimetres. `walk` puts the same world model in charge of a
Unitree G1 humanoid. The **planner** is V-JEPA 2-AC, pretrained on real videos of a robot
arm, which has never seen legs. Its action slot is reused for "walk this far this way":
`adapt` adds no parameters, it only teaches the existing action embedding what a move
means on this body.

**The legs are a callback.** They are the exact G1 we taught to walk earlier in this
repo, in [`topics/rl-mujoco`](../rl-mujoco/README.md): MJX + Brax PPO, 300M steps on rough
terrain, Flyte run `rlswb4sgxg6j2vwfr9gr` in the `physical-ai` project. Nothing is
retrained or copied. `walker.G1_CHECKPOINT` points at that run's `g1_checkpoint.pkl` in
the devbox blob store and the task downloads it as a Flyte `File`. That policy is the
only part of this demo ever trained with a reward, and it is why the robot never falls:
V-JEPA does not balance anything, it only picks where to go.

```
V-JEPA 2-AC      ->  "walk 40 cm that way"      one decision every 0.6 s, from a photo
walker.py        ->  joystick command, heading held
PPO (rl-mujoco)  ->  29 joint targets at 50 Hz  trained once, with a reward
MuJoCo           ->  physics
```

The low level learns to walk once, with a reward. Everything above it plans toward a
photograph with no reward at all. That split is the hierarchical-planning argument for
world models in miniature.

The report is live while it runs: the latest random-play episode and a coverage map
during collection, the current frame at every decision, a clip of each finished
episode, and the distance chart filling in policy by policy.

1. The G1 wanders at random: 720 moves, no goal, no reward, zero falls.
2. The predictor is fine-tuned to predict the next view after each move (4.4 min).
3. It is shown a photo of the robot standing on a coloured pad. Each decision it imagines
   walking each of 8 directions for 4 moves, takes the first step of the path whose
   imagined ending looks most like the photo, and looks again.

Run `rrxvpj7glkqbqpqkn7sp`, 4 pads x 2 starts:

| policy | reached the pad | final distance |
|---|---|---|
| random walking | 0/8 | 2.57 m |
| V-JEPA pretrained | 0/8 | 2.24 m |
| **V-JEPA adapted** | **8/8** | **0.19 m** |
| lookahead (same reward, simulator as dynamics) | 3/4 | 0.30 m |
| oracle (told the coordinates) | 8/8 | 0.08 m |

Then a tour: four pads in a row, each given only as a photo, the planner deciding for
itself when it has arrived. 4/4.

**Three design decisions that were measured, not guessed.** Robot teleported over a
13x13 grid, each view scored against a goal photo:

- The model's camera follows the robot from **13 m up**. A low chase camera loses the pads
  as soon as they leave the frame (rank correlation with distance 0.32 vs 0.80).
- Floor studs are **scattered at random**. A regular grid looks identical every 0.75 m.
- Energy is **centred cosine**, not raw L1 (0.85 vs 0.80), as `energy` predicted.

**Here V-JEPA beats raw pixels, which it never did on the Franka.** Pixel L1 over the same
views correlates -0.09 to 0.37 with distance. A camera that moves with the robot shifts
the whole image, which destroys pixel matching and not the learned features.

**One-move lookahead fails even with perfect dynamics.** From 13 m up, a 0.45 m move
shifts the view by less than one 16 px patch; the lowest-energy single move got closer
only 28% of the time. Looking 4 moves ahead gets it right 83% of the time.

**What does not work:** dreaming a whole 10-move walk open loop. The adapted dream is
0.92 m off on average, barely better than imagining the robot never moves (1.00 m); the
lookup floor is 0.41 m. Four moves ahead is enough to plan with; ten is not. That is
the argument for re-planning every step.

## What is actually inside a token? The picture is not

```bash
flyte run pipeline.py readout
```

Every other task here works around V-JEPA 2 having no decoder. `readout` asks whether
that is a packaging problem or a fact about the representation, by fitting the best
possible **linear** map from one patch token back to the 2x16x16 patch of pixels it was
embedded from. Closed form ridge regression, so there is no learning rate and no step
count to blame for a bad result.

The controls are the experiment. Run `rvpkvfxgqhz5zwgx8l2b`, 196,608 tokens from 48
clips, scored on held-out tokens:

| readout from | PSNR | R2 | what it is |
|---|---|---|---|
| a random 1024-d projection of the true pixels | **38.76 dB** | **+0.997** | the ceiling: the method works |
| each patch's own mean colour | 17.73 dB | +0.603 | knowing only average brightness |
| **V-JEPA 2 patch tokens** | **11.97 dB** | **-0.494** | what the model keeps |
| V-JEPA tokens, wrong patches | 11.10 dB | -0.008 | the floor |

A random projection of 1536 pixel values into 1024 dimensions is very nearly lossless,
and the same linear map inverts it almost perfectly. Handed a V-JEPA token instead, it
does **worse than painting each patch its own average colour**, and barely better than
being given the wrong patch entirely. The negative R2 means it is worse than predicting
the dataset mean.

The report shows this as three panels side by side: the original frame, the control
readout (sharp), and the V-JEPA readout (coloured mush). The middle panel is the
load-bearing one, because without it a blurry third panel is just as easily a weak
decoder as a representation that discarded the pixels.

**This is the JEPA thesis working, not failing.** The argument for predicting in
representation space is that pixels are mostly unpredictable detail and a model that
refuses to spend capacity on them has it free for structure. The same tokens that
cannot reproduce a 16x16 square support **78% five-way action recognition** from a
single linear layer, where the same probe on raw pixels gets 40% and chance is 20%.
Appearance discarded, meaning kept.

It also explains something the top of this README only asserts. There is no released
V-JEPA 2 decoder because there is nothing for one to read.

> Worth recording as a near miss: the first version of this trained a 2048-wide MLP
> decoder instead, got 13.9 dB on V-JEPA tokens, and looked like a finding. It was not.
> The same MLP managed only 13.9 dB on the random-projection control that ridge
> regression inverts at 38.8 dB, so the MLP was capacity-limited and its V-JEPA number
> measured nothing. A failed readout is only evidence once something that *should*
> succeed has been run through the identical pipeline.

## Harder task: a photograph of the finished job

```bash
flyte run pipeline.py push --seeds 16
```

The reach task has a flaw that only its controls revealed: its goal photo differed from
the start photo **only in where the arm was**, so any image distance was monotone in arm
position and raw pixels planned it exactly as well as V-JEPA. The task never asked the
representation a question.

`push` asks a harder one. The robot is shown one photograph of the same scene with the
**block already moved**, and that photograph is the entire specification. It is never
told where the block is, where it should end up, or that a block is involved. Each step
it tries all 27 candidate moves in a copy of the simulator, photographs each result,
keeps whichever looks most like the reference, executes that for real, and asks again.
Twenty-two times.

The score is the **block's** progress, never the arm's, so a policy that poses the arm
to match the picture without pushing anything earns nothing.

Sixteen scenes, the only difference between rows being the space the two images are
compared in. Run `rdbfzszn7twmxz96k4lw`:

| comparing images as | block progress | block moved | solved | vs random |
|---|---|---|---|---|
| scripted oracle (told the answer) | +79.8% +- 10.7 | +10.8 cm | 14/16 | - |
| V-JEPA 2 embeddings | +45.6% +- 31.4 | +6.3 cm | 5/16 | **3.4 sigma** |
| raw pixel subtraction | +33.7% +- 30.8 | +4.2 cm | 2/16 | 2.6 sigma |
| an untrained net, same architecture | +28.8% +- 29.7 | +3.7 cm | 2/16 | 2.2 sigma |
| random actions, photo ignored | +7.6% +- 32.0 | +0.8 cm | 1/16 | - |

**The first claim holds.** A photograph of the finished job is a sufficient
instruction: every policy that looks at it moves the block several centimetres toward
where it belongs, and one that ignores it does not. No reward function, no
demonstration, no coordinates, and the robot is never told a block is involved.

**The second claim does not.** V-JEPA's advantage over raw pixel subtraction is +11.9
points with a standard error of 11.0, which is 1.1 sigma, which is nothing. An
untrained network of the same architecture is another 5 points behind that, also inside
the noise. The learned representation is not what is doing the work.

That second finding is worth dwelling on, because it is a trap this demo walked into in
real time and nearly published. At 3 scenes the V-JEPA-over-pixels gap was +47 points
and looked decisive. At 6 it was +27. At 16 it is +11.9 against a standard error of
11.0. Nothing changed except the number of scenes.

The one column that still favours V-JEPA is `solved`, the count of scenes where the
block finished within 5 cm of its photographed position: 5 of 16 against 2 of 16. That
is a more forgiving reading of the same runs, and at those counts it is not significant
either (Fisher exact p ~ 0.39). It is recorded here because it is the honest reason to
run this task again with more scenes rather than to declare it settled.

### The cold-start version, which nothing solves

The numbers above start with the gripper already parked behind the block. Run it from
the arm's home pose instead (`--cold_start`) and every learned policy fails: the block
moves under 1 cm while the scripted oracle still moves it 10 cm. The reason is
structural rather than perceptual. To push, the arm must first travel *around* to the
far side of the block, and every step of that detour makes the camera image **less**
like the reference photograph. One-step descent on image similarity cannot climb out of
that, no matter what space the images are compared in. It is a local minimum in the
objective, not a failure of the features, and a longer planning horizon is the thing
that would address it.

## The mechanism tasks: how it works, rather than how well

Everything above measures how *well* V-JEPA 2 does things. These four are about how it
works, and they exist because the most important facts about JEPA are invisible in a
report that only scores a finished checkpoint: what a prediction looks like when you
render it honestly, what shape its energy function has, where in the network appearance
turns into meaning, and what the training objective does when you take away the one
architectural choice that keeps it alive.

```bash
flyte run pipeline.py occlude     # slide an occluder across a clip and RENDER the prediction
flyte run pipeline.py energy      # the E in energy-based model
flyte run pipeline.py ladder      # where in the 24 layers appearance becomes meaning
flyte run pipeline.py collapse    # train a tiny JEPA four ways plus seed sweep; watch one collapse
flyte run pipeline.py mechanism   # all four, one run
```

## Rendering the prediction, with its own ceiling in the frame

The top of this README insists that "show me what V-JEPA predicted" is not a screenshot
anyone can take. That is true and it is also an excuse, because a hole and a heatmap do
not tell you what the model thought was behind the hole. `occlude` renders it, and the
rule that makes the rendering honest is that **the same rendering of the encoder's own
token is in the panel next to it**.

Two renderings, both lossy before the predictor is involved at all:

- a **patch-retrieval mosaic**: for each predicted token, find the nearest token in a
  bank built from *other* clips and paste the 16x16 pixels that token was embedded
  from. Every pixel is a real pixel of a real video; nothing is invented.
- the **linear readout** from `readout`, applied to the predicted token instead of the
  true one.

The bank and the ridge map are fitted on train clips, the clip shown is a val clip
(asserted, not assumed), and the ceiling panel is the identical rendering of the
encoder's own tokens at the identical positions. So the prediction can never look better
than the method allows, and a reader can see how much of the gap is the method.

The headline video is an `8x8` occluder **sliding across the clip** while the predictor
fills it in: the model's input with a moving hole, the prediction, the ceiling, and the
original, four aligned panels in one mp4.

### The hypothesis was half wrong

Three masks, all hiding exactly **25%** of the tokens (an 8x8 block of a 16x16 grid is
25%, and so is 4 of 16 tubelets), differing only in the shape of the hole. Run
`rt548mj42wdmn6fvrdgq`, 32 frames of `val/bowling/--dVV4_CSvw`:

| mask | cosine | chance | lift | time localised | same-action retrieval |
|---|---|---|---|---|---|
| **sweep** (hole moves) | 0.262 | 0.001 | +0.261 | **1.00** | 98.8% |
| **static** (hole fixed) | 0.261 | -0.008 | +0.269 | **1.00** | 96.3% |
| **future** (end of clip) | 0.266 | 0.013 | +0.253 | **0.00** | 99.8% |

The prediction was that a *moving* hole would be easiest, because it leaves every hidden
patch position visible at some other moment. **It makes no difference at all**: sweep
and static are inside noise of each other on every metric. Whatever the predictor is
doing, it is not looking the patch up at another time.

What does separate is the future mask, and it separates completely: 1.00 against 0.00 on
time localisation, at matched hidden fraction, which is `inpaint`'s finding reproduced
with a third mask shape in the comparison.

Also note the cosine column: **0.262, 0.261, 0.266**. It is flat to three decimal places
across a mask that works perfectly and a mask that fails completely. This README already
says cosine is the metric that would let the forecasting failure pass unnoticed; here it
is, letting it pass unnoticed again.

### The finding: right about what, wrong about when

The last column is the one worth the stream time. It is the fraction of patches pasted
into the hole that came from a clip of the **same action**, where that action is only
**20%** of the bank.

Under every mask, including the one where the prediction cannot place the content in time
at all, that number is **96 to 99%**. The predicted vector knows what kind of event it is
filling in, with no action label anywhere in the task, while simultaneously being unable
to say when it happens. Those are two different failures and this measurement separates
them.

It also explains why the mosaic is legible to a human and scores like noise under PSNR.
Hole-only PSNR is 6.7 dB for the prediction against a 7.8 dB ceiling and a 6.6 dB
shuffled floor: essentially nothing. A retrieved patch with the right content and the
wrong colour is a large pixel error, which is `readout`'s finding arriving from a
different direction. **The token does not carry appearance, so a pixel metric cannot see
whether the prediction was right.** The report prints PSNR and says so rather than
leaving it out.

### It does not matter what is under the hole either

One occluder size moved over a 3x3 grid of positions, with how much motion each position
covered as the covariate:

| hole (h, w) | motion under it | cosine | lift over chance |
|---|---|---|---|
| (4, 0) | **11.01** | 0.242 | +0.250 |
| (4, 4) | 9.93 | 0.261 | +0.269 |
| (8, 0) | 9.04 | 0.210 | +0.220 |
| (8, 4) | 7.99 | 0.241 | +0.254 |
| (0, 0) | 6.74 | 0.214 | +0.222 |
| (4, 8) | 6.64 | 0.275 | +0.288 |
| (8, 8) | 5.48 | 0.262 | +0.274 |
| (0, 4) | 5.30 | 0.245 | +0.253 |
| (0, 8) | **3.70** | 0.238 | +0.249 |

Motion under the hole varies by **3x** and the score does not follow it: **r = -0.13 with
n = 9.** The first version of this sampled the diagonal instead, where the covariate only
spanned 5.5 to 9.8, and I assumed the null was a coverage problem. It is not: with three
times the range it is still a null. Prediction quality on this clip depends on which
patch you hide much less than it looks like it should, which is mildly reassuring about
every aggregate score in this demo.

## The E in energy-based model, and the distance it was trained with is the wrong one

JEPA is introduced as an energy-based model. The energy of a pair is

```
E(x, y) = D( Predictor(Encoder(x)), Encoder(y) )
```

with `x` the visible context, `y` a candidate for the hidden part, and `D` the L1
distance V-JEPA 2 was actually trained on. Training pushes the energy down on real
pairs. **Nothing pushes any energy up**: JEPA has no negatives and no partition
function, so whether the result is a well-shaped energy function is an empirical
question. `energy` maps it three ways, with one encoder pass per candidate and no
training at all: a well, a graded ladder of eleven candidates, and the degenerate
controls that decide whether it is a preference over completions at all.

Eleven candidate completions, graded from the truth to noise, with the intended ordering
written down in `energy.EXPECTED_ORDER` **before** anything was measured so that the rank
correlation is a prediction and not a fit. Run `rcvt55cswlmt68f5429g`, 32 frames of
`val/bowling/--dVV4_CSvw`, two 8x8 tube blocks (48% hidden):

| candidate | E, raw L1 (the objective) | E, centred cosine |
|---|---|---|
| **true completion** | 2.0753 (rank 6) | **0.7041 (rank 1)** |
| same clip, blurred | 2.0714 | 0.7229 |
| same clip, 1 tubelet late | 2.0778 | 0.7387 |
| first frame frozen | 2.1064 | 0.7475 |
| same clip, played backwards | 2.0733 | 0.7521 |
| same clip, 4 tubelets late | 2.0856 | 0.7619 |
| same clip, frames shuffled | 2.0894 | 0.7765 |
| uniform noise | 2.2413 | 0.7877 |
| different action | 2.0049 | 0.8489 |
| same action, different clip | 1.9820 | 0.8525 |
| **flat grey** | **0.9468 (rank 1)** | 1.0000 (rank 11) |
| *shuffled pairing (chance)* | *2.1143* | *1.0019* |
| *context mean (no model)* | *1.9947* | *0.9992* |

Read the two columns against each other.

**Under the distance the model was trained with, the energy is not a preference over
completions.** Flat grey scores **2.2x lower** than the true completion. So does another
clip entirely, and so does the no-model "answer with the average of what you can see"
baseline. Five of the eleven candidates score below the truth, so it ranks 6th. Rank
correlation with the pre-registered ordering is **rho = -0.09**: nothing.

**Subtract the component every token shares first and all of it snaps into place.** The
truth becomes the unique minimum, on 6 of 6 clips tested; flat grey becomes the worst
candidate of the eleven; and rho is **+0.89**. That is the same `jepa.center` correction
the rest of this demo applies without comment, and here it is the difference between a
broken energy and a working one.

The same split shows up in the wells. Roll the candidate clip forward and backward in
time and plot the energy:

| | range as a fraction of its floor | minimum at |
|---|---|---|
| raw L1 | **0.9%** (flat) | -1 tubelet |
| centred cosine | **10.4%** | **0 tubelets** |

Reading only the objective's own curve would say the predictor cannot locate the hidden
content in time, which is the opposite of what `inpaint` measures. The spatial well
bottoms out at -1 patch, which is 0.2% away from 0 and effectively tied.

### Why this is the same finding as `collapse`

A degenerate embedding with a small norm is close to *everything* in raw L1. That is
exactly the solution the JEPA objective admits and exactly what the EMA target exists to
keep training away from. **Nothing keeps a candidate at inference time away from it**, so
flat grey wins. The two tasks are the same defect seen from opposite ends, and `collapse`
below is what happens when you let training find it.

### The one ordering anomaly, recorded rather than smoothed over

Uniform noise (0.788) scores *lower* than another clip of the same action (0.853). Both
cross-clip candidates are the worst real videos on the list, worse than the same clip
with its frames shuffled. So this energy is much more "is this the same scene" than "is
this the same kind of event", which is worth knowing before anyone reaches for it as a
semantic similarity.

## Where the pixels go: the same two measurements at all 24 layers

`readout` establishes the headline fact of this demo at the last layer only: a linear map
recovers almost nothing of a 16x16 patch from a V-JEPA token, while the same tokens
support 78% five-way action recognition. Appearance discarded, meaning kept. `ladder`
runs both measurements at **every** layer of the same forward pass, so that sentence gets
a location in the network as well as a number.

Two details are load-bearing. Intermediate layers are taken **before** the encoder's
final LayerNorm, whose learned affine is calibrated for the last layer, so the final
layer appears twice, raw and normed. And the tokens come from an explicit loop over
`model.encoder.layer` rather than `output_hidden_states=True`: in transformers 5 that
flag is served by output-capture hooks and `VJEPA2Encoder.forward` itself returns only
`last_hidden_state`, so a release that changed the capture machinery would hand back
`None` and this task would silently have nothing to plot.

### The pixel half, which is as clean as this repo gets

Run `r89wqwp8znhg9cjdfkvv`, 12 clips x 1024 tokens per layer, scored on held-out tokens:

| layer | linear readout to pixels | R2 |
|---|---|---|
| ceiling: random projection of the true pixels | **36.95 dB** | +0.998 |
| 1 | **36.29 dB** | +0.998 |
| 4 | 29.21 dB | +0.987 |
| 8 | 22.83 dB | +0.945 |
| 12 | 18.94 dB | +0.865 |
| 16 | 16.35 dB | +0.755 |
| 20 | 16.00 dB | +0.735 |
| 24 | 15.76 dB | +0.720 |
| **24 + the final LayerNorm** | **12.80 dB** | **+0.445** |
| baseline: each patch's own mean colour | 15.87 dB | +0.727 |
| floor: tokens paired with the wrong patches | 9.72 dB | -0.136 |

At layer 1 the token is **essentially a lossless linear recoding of its own patch**:
36.29 dB against a 36.95 dB ceiling that is itself a near-lossless random projection of
the true pixels. From there it decays monotonically for 24 layers, flattens out around
layer 16, and settles onto the flat-grey baseline by layer 24 (15.76 dB against 15.87),
i.e. by the end a linear map can get no more out of a token than the patch's average
colour.

Two things fall out of that table that were not the reason for building it.

**The final LayerNorm costs 3 dB on its own**, taking the output from 15.76 dB to 12.80
and from just above the flat-grey baseline to well below it. That single op discards more
linearly-decodable pixel information than layers 20 through 24 put together. It is also
why the `readout` task measures 11.97 dB: `get_vision_features` returns the post-LayerNorm
output, which is the right thing for it to use and worth knowing is not the same as
"layer 24".

**Nothing here contradicts `readout`, and the mechanism is now visible.** The information
does not vanish somewhere in the middle of the network; it is given up smoothly, layer by
layer, which is what "spends its capacity elsewhere" looks like when you measure it.

### The semantic half, which is noisier and still points the right way

The same 25 depths, mean-pooled, with one linear layer trained per depth on the dataset's
own 50/50 split, plus leave-one-out retrieval which trains nothing:

| layer | pixel R2 | action probe | 1-NN retrieval | raw random-pair cosine |
|---|---|---|---|---|
| 1 | **+0.998** | 64% | 47% | 0.660 |
| 7 | +0.961 | 60% | 50% | **0.914** |
| 12 | +0.865 | 64% | 62% | 0.892 |
| 16 | +0.756 | 60% | 70% | 0.858 |
| 20 | +0.735 | 66% | 69% | 0.713 |
| 23 | +0.724 | 74% | **84%** | 0.544 |
| 24 | +0.720 | 74% | 78% | 0.261 |
| **final (LN)** | **+0.445** | **78%** | 82% | 0.270 |

The two curves go opposite ways, which is the point, but be honest about the sizes. The
pixel curve falls by **0.55 R2** smoothly over 24 layers. The probe curve rises by **14
points**, and almost all of it arrives in the last two layers; the val split is 50 clips,
so one clip is 2% and a 14-point move is 7 clips. The retrieval curve is the better-
behaved of the two and tells the same story more clearly, from 47% at layer 1 to 84% at
layer 23.

Worth noticing that layer 1 already probes at **64%**, above the 40% raw-pixel baseline
from the `probe` task. A near-lossless linear recoding of the pixels is *already* a better
clip descriptor than a downsampled space-time thumbnail, so the probe's absolute level is
not all "semantics" and only the *change* along the curve is about abstraction.

**The question this was built to answer has a reassuring answer.** The predictor is
trained to match the final encoder layer, and the final layer is also where the probe
peaks (78%, `gap_to_final` 0.0). The world model is predicting the network's most
linearly informative representation, not an earlier or later one. Retrieval peaks one
layer earlier at 84% against 82%, which is one clip and not a finding.

### The anisotropy correction is not a constant

The last column is the mean cosine between two random patch tokens, uncentred. It is
**0.66** at layer 1, climbs to **0.914** at layer 7, and falls to **0.26** at layer 24.

So the token cone tightens through the first third of the network and opens up again at
the end, and nearly all of that opening happens in the last layer. Everywhere else in
this demo `jepa.center()` is applied without comment and the README quotes "0.92 at layer
8 and 0.28 at the last layer" as if those were two facts about V-JEPA; they are two points
on this curve. Anyone reading a raw cosine from the middle of this network is reading
almost entirely the component every token shares.

## Why JEPA needs an EMA teacher: train a tiny one four ways

This is the only task here that trains a representation from scratch, and the only one
that does not load V-JEPA 2 at all. That is deliberate: **a released checkpoint cannot
show you the failure mode its authors already designed around.**

The JEPA objective is trivially satisfiable. "Predict the representation of the hidden
part" is solved perfectly by a representation that is *constant*: emit the same vector
for every input, predict that vector, loss zero, forever. Nothing in the loss forbids it.
That is why the targets in a real JEPA come from an exponential moving average of the
encoder with no gradient flowing into them, and it is why "we predict in representation
space" is only half the idea.

So: a **0.9M parameter** JEPA, on synthetic video, four ways, with everything but the
target held fixed. Same data, same masks, same steps, same optimiser, same encoder and
predictor shapes.

| arm | the prediction target comes from |
|---|---|
| `ema` | an EMA copy of the encoder, no gradient. V-JEPA's own recipe. |
| `stopgrad` | the encoder itself, detached. Tests whether the EMA is doing the work. |
| `none` | the encoder itself, **with** gradients. The degenerate solution is reachable. |
| `pixels` | the true pixels. A masked autoencoder, i.e. the thing JEPA argues against. |
| `random` | nothing. Never trained. The control. |

The world is one shape (square, disc or triangle) in one of three colours moving in one
of eight directions across a cluttered static background, 64x64, 8 frames. Three
generative factors, drawn independently (asserted in `smoke_test.py`), and the trajectory
is **centred** rather than started uniformly: if the start position were uniform, "moving
right" would have to start on the left to stay in frame and the direction probe could be
solved from a single frame.


### The trap: the arm that learned nothing wins the loss curve

Run `rzmbthndjdc9t2q7cqdx`, 2000 steps per arm, 92 to 161 seconds each:

| target comes from | loss | pair cos | eff. rank | feature std | colour | shape | direction |
|---|---|---|---|---|---|---|---|
| an EMA of the encoder (V-JEPA) | 0.3088 | **0.195** | **22.1** | **0.722** | 98.7% | 55.7% | **49.3%** |
| the encoder, detached | 0.2448 | 0.779 | 7.9 | 0.401 | **99.3%** | 48.0% | 40.2% |
| **the encoder, with gradients** | **0.0013** | **1.000** | **1.02** | **0.0003** | 89.5% | 41.8% | 37.9% |
| the true pixels (MAE) | 0.0682 | 0.693 | 8.2 | 0.435 | 97.2% | 57.1% | 48.9% |
| nothing, never trained | - | - | - | - | 97.3% | **62.8%** | 48.1% |

Chance is 33% for colour, 33% for shape, 12.5% for direction. Pair cosine is the mean
cosine between two *different* clips' features, uncentred, so 1.00 means the encoder
returns the same direction whatever it is shown.

**The arm that collapsed has the lowest loss by a factor of 243.** Anyone comparing
training curves would pick it. It got there by emitting the same vector for every input:
pair cosine **1.000**, effective rank **1.02**, and a per-dimension standard deviation
**2490x smaller** than V-JEPA's own recipe. And it cost every probe: colour 89% against
99%, shape 42% against 56%, direction 38% against 49%.

That is the entire argument for the asymmetric target, in one table. The EMA is not a
trick that helps convergence. It is the thing that stops the objective from being
satisfied by a representation containing nothing, and **the loss cannot tell you that it
happened.**

### Where the collapse actually happens, which is not where I assumed

The obvious mechanism is the encoder's final LayerNorm gain being driven to zero so
every input maps to the bias vector. That is not what happens. Measured after 600 steps:

| | LayerNorm gain, mean magnitude | pooled std **before** the LayerNorm | pooled std after |
|---|---|---|---|
| ema | 0.992 | 1.788 | 0.633 |
| none (collapsed) | 0.970 | **0.062** | 0.0009 |

The gain is essentially untouched (1.0 at initialisation). The collapse is in the
**body**: the transformer blocks already map every clip to nearly the same activation, a
29x loss of between-clip spread before the LayerNorm is applied. The LayerNorm then
removes the scale that is left, taking 0.062 to 0.0009, which is why the pooled
descriptor looks even flatter than the body alone would explain.

### Is stopping the gradient enough on its own?

SimSiam's claim for images is that a stop-gradient plus a predictor head avoids collapse
with no EMA at all. The `stopgrad` arm is that claim tested on video, and it came out
both ways on identical settings, so the task runs it at extra seeds:

| seed | final pair cosine | final loss | |
|---|---|---|---|
| 0 | 0.779 | 0.2448 | collapsing |
| 1 | 0.412 | 0.2362 | drifting |
| 2 | **0.193** | 0.2755 | healthy |

A spread of **0.59** on the one number that matters, against 0.195 for the EMA arm and
1.000 for the arm with no stop-grad at all, and the three seeds land at three different
points on the way down. An earlier run of the identical configuration gave 0.812 / 0.239
/ 0.780, so which seed collapses is not even stable between runs. Watching a single log
shows why: the pair cosine oscillates between 0.47 and 0.91 within a few hundred steps.
**Stop-grad alone sits near a bifurcation, and which side it lands on is not predictable
from the settings.** That is a different claim from "stop-grad does not work", and it is
the practical case for the EMA target: the EMA arm has never collapsed in any run here.

### Two measurement traps that give the opposite answer

Both are the first thing a reader would reach for. Both are in the report so that it can
say so.

**Centred effective rank ranks the collapsed arm as the richest representation in the
experiment: 66.1 against 22.4 for the healthy one.** Subtracting the per-dimension mean
removes exactly the constant vector that collapse produced and leaves unstructured
numerical residue, which is close to full rank. Every collapse detector here is therefore
uncentred, and the centred number is printed beside them as a warning.

**Mean pooling makes the motion probe impossible in principle**, and it does not fail
quietly. Under plain mean pooling instead of the 2x2-per-tubelet pooling the table above
uses:

| direction probe | ema | stopgrad | none | pixels | random |
|---|---|---|---|---|---|
| 2x2 per tubelet | **49.3%** | 40.2% | 37.9% | 48.9% | 48.1% |
| mean over all tokens | 12.3% | 16.7% | **30.7%** | 11.8% | 25.0% |

Chance is 12.5%. Under mean pooling the EMA arm is *at chance* and the **collapsed** arm
scores highest of the five. Direction of motion is a statement about how position changes
with time, and a mean over every token has destroyed position and time before the probe
sees anything; what survives is an artifact that happens to favour the degenerate model.
This is the same class of mistake as reading a cosine similarity without its chance
floor, which is what the top of this README is about.

### What the untrained control allows you to claim, and at which label budget

The probes separate collapse from health cleanly. Against a randomly initialised encoder
of the same shape, the answer depends entirely on how many labels the probe gets, which
is why the task sweeps it:

| colour probe | 25 labels | 1000 labels |
|---|---|---|
| the two healthy latent arms | **74.8% / 88.8%** | 98.7% / 99.3% |
| collapsed (`none`) | 49.4% | 89.5% |
| **untrained** | 51.4% | 97.3% |

| other factors, 25 labels | ema | untrained |
|---|---|---|
| shape | 35.5% | 37.9% |
| direction | 11.3% | 12.4% (chance is 12.5%) |

**On colour, both healthy latent arms beat an untrained encoder by 23 to 37 points when
labels are scarce, and the collapsed arm falls below it.** By 1000 labels the gap is 1 to
2 points, because a random convolutional basis plus a linear layer fitted on a thousand
examples has caught up. That is exactly the shape a representation advantage is supposed
to have, and it is invisible in the full-budget column everyone would read first.

Be careful about the size of it, though: an earlier run of the identical configuration
put the EMA arm at 91.7% rather than 74.8% at 25 labels, so the gap was +40 points there
and +23 here. The *sign* is stable across runs and the *magnitude* is not, at this scale.

On shape the untrained encoder is ahead at every budget, so this objective on this world
did not learn shape. On direction everything is at chance at 25 labels, so that end of
the curve says nothing about any arm.

None of this is a statement about V-JEPA 2: it is 0.9M parameters trained for two
minutes on synthetic shapes. `probe` on Kinetics is where frozen-feature quality is
demonstrated at scale.

### The one arm you can look at is not the best one

The pixel arm's predictor outputs pixels, so it has a decoder and the report shows its
reconstruction beside the original and the masked input. The three latent arms have
nothing to show there at all, which is the constraint the whole of the rest of this demo
is built around, arriving at the end as a picture.


## How it is put together

| file | what is in it |
|---|---|
| `config.py` | Flyte image and task environments, checkpoint ids, Spark tuning |
| `clips.py` | fetch/decode Kinetics clips, and recover the exact pixels the model saw |
| `jepa.py` | model load, the token raster, the two masks, and the scoring with its floors |
| `probing.py` | linear probe, retrieval, dataset encoding |
| `viz.py` | mp4 encode/probe, masked and heatmap videos, matplotlib charts |
| `reports.py` | the report HTML, shared palette with the other world-model demos |
| `ac.py` | V-JEPA 2-AC: loader, transform, dreaming, CEM planning, the GB10 memory guard |
| `sim.py` | the MuJoCo Franka, its Cartesian controller, and the DROID-ish scene |
| `plan.py` | the closed-loop episode, the five policies, and the retrieval decode |
| `adapt.py` | transition collection and the predictor-only fine-tune |
| `decode.py` | the linear readout from tokens back to pixels, and its controls |
| `rlreward.py` | the batched reach env, the five reward functions, and a compact SAC |
| `occlude.py` | the moving occluder, the patch-retrieval mosaic, and its ceiling |
| `energy.py` | candidate completions, the energy wells, and the degenerate controls |
| `layers.py` | per-layer tokens, and the readout and probe curves over depth |
| `collapse.py` | the synthetic world, a tiny JEPA, its four training arms, and the probes |
| `pipeline.py` | the Flyte tasks |
| `smoke_test.py` | runs the real code paths on the host, and asserts the findings above |

A few things that are load-bearing and non-obvious:

- **Token index is `t*G*G + h*G + w`**, temporal-major, where `G = 16` and `t` indexes *tubelets* of 2 frames. `VJEPA2RopeAttention.get_position_ids` decodes exactly that arithmetic back out of whatever index you hand it, so masks are plain flat indices. `jepa.check_layout()` is the tripwire if a transformers release ever changes it: bad masks would still be valid indices and would still produce plausible numbers for the wrong tokens.
- **Mean pooling, not the attentive pooler.** `VJEPA2AttentivePooler` exists in the architecture but its weights are only trained in the *classifier* checkpoints. In a pretrained-only checkpoint it is randomly initialised, so using it would be measuring noise.
- **Probe features are standardised on train statistics only.** A small effect at n=50, and exactly the kind of leak that makes a benchmark number quietly wrong.
- **The orchestrator is CPU-only.** One GPU on this box, and an orchestrator pod holds its resources while its children run, so a GPU-holding orchestrator deadlocks its own GPU child on "Insufficient nvidia.com/gpu". Same trap as the cosmos, videogen, mujoco and Isaac Sim demos.
- **`output_hidden_states=True` is not how you get per-layer tokens here.** In transformers 5 that flag is served by output-capture hooks on the decorated forward, and `VJEPA2Encoder.forward` itself returns only `last_hidden_state`. A release that changed the capture machinery would hand back `None` and `ladder` would silently have nothing to plot, so `layers.per_layer` loops `model.encoder.layer` explicitly and `smoke_test.py` asserts its last entry matches the model's own forward.
- **The clip catalogue is sorted by path.** `[p for sp, _, p in catalog if sp == "train"][:12]` is twelve clips of whichever action sorts first, which makes every cross-class measurement in `occlude` either NaN or meaningless. `occlude.spread_over_classes` interleaves, and `occlude.same_class` returns NaN rather than a plausible number when the bank turns out to hold one class.
- **`decode.ridge_fit` builds its bias column on the tensor's own device.** It looks like a detail because everything else in this repo solves on the CPU. `ladder` solves on the GPU 25 times, and a `torch.ones(...)` defaulting to the CPU fails with "expected all tensors to be on the same device" from two calls down. The smoke test now runs that solve on whichever device the task will use.
- **Mean pooling makes a motion probe impossible in principle.** Direction of motion is a statement about how position changes with time, and a mean over all tokens has destroyed position and time before a linear probe sees anything. In `collapse` it puts the direction probe at chance for every arm, *including* arms that demonstrably encode direction, with the untrained control scoring highest. Pooling to a 2x2 spatial grid per tubelet fixes it. This is the same class of mistake as reading a cosine without its floor.
- **Centred effective rank ranks a collapsed encoder as the richest representation in the experiment.** Subtracting the per-dimension mean removes exactly the constant vector that collapse produces, leaving unstructured numerical residue which is close to full rank: measured, 66.2 for the collapsed arm against 23.4 for the healthy one. Every collapse detector in `collapse.spread` is therefore uncentred, and the centred number is printed next to them as a warning.

## Hardware notes (DGX Spark, GB10, arm64)

This is by far the cheapest demo in the repo to run, which is itself the point: no generation means no diffusion loop.

| | |
|---|---|
| ViT-L encode, 64 frames | **0.58 s**, 1.2 GiB peak |
| ViT-g encode, 32 frames | 0.77 s, 2.1 GiB peak |
| `inpaint` end to end in a pod | **32 s** |
| `probe`, 100 clips | ~3 min, mostly download and decode |
| Largest checkpoint on disk | 4.4 GB (ViT-g, 1035M params) |
| V-JEPA 2-AC checkpoint | **11.7 GB**, 1317M params, 7.7 GiB resident |
| AC load with `mmap=True` | **0.4 s** (5.1 s and ~12 GiB resident without it) |
| One CEM plan (32 samples x 5 iters, horizon 2) | ~11 s, fp32 only |
| `plan`, 5 policies x 10 steps | ~3.5 min in a pod |
| `adapt`, collect 480 + train 400 steps | 1.5 min + **4.3 min** |
| `occlude` end to end in a pod | **72 s**, including a 49k-token bank |
| `energy`, 11 candidates + two wells + 6 ranking clips | **61 s** |
| `ladder`, 25 ridge solves + 25 probes over 100 clips | **3.3 min** |
| One ridge solve, 12k tokens x 1024 dims, float64 | 0.4 s on the GPU |
| `collapse`, the tiny JEPA, 2000 steps | **92 to 161 s per arm**, 6 arms + probes |

Two GB10 traps bit hard enough to be worth repeating. **CUDA can only use memory that is
genuinely free, and `free -g` will lie to you about how much that is.** A task pod died
with `CUDA error: out of memory` raised by `cuDevicePrimaryCtxRetain` -- before
allocating anything -- while the OS reported 97 GiB available; `cuMemGetInfo` said 20.1.
The missing 78 GiB was page cache left by copying the 11.7 GB checkpoint, and the kernel
does not reclaim it in time. `ac.guard_memory()` checks the driver's number and refuses
with a readable message instead. The second trap is the cure for the first:
`torch.load(..., mmap=True)` takes that checkpoint from 5.1 s and ~12 GiB resident to
**0.4 s and 0.5 GiB**, and leaves no second copy in cache.

`torch` is installed on its own layer from the cu130 index; the plain-PyPI aarch64 wheel is CPU-only and the only symptom is `torch.cuda.is_available() == False` at encode time. PyAV rather than imageio-ffmpeg, because aarch64 wheels reliably exist for one and not the other. Do not `torch.compile` the encoder: Triton does not emit working SASS for sm_121a yet.

## Some things to look up to get started

**Model**
- V-JEPA 2 (Meta): https://github.com/facebookresearch/vjepa2
- Transformers docs and checkpoints: https://huggingface.co/docs/transformers/model_doc/vjepa2
- Paper, "V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning": https://huggingface.co/papers/2506.09985

**V-JEPA 2-AC and planning**
- The AC checkpoint (direct download, not on the Hub): https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt
- Upstream's energy-landscape notebook and CEM planner: https://github.com/facebookresearch/vjepa2/tree/main/notebooks
- DROID, the dataset the AC model's actions and poses come from: https://droid-dataset.github.io/
- MuJoCo Menagerie's Franka Panda: https://github.com/google-deepmind/mujoco_menagerie

**The mechanism tasks**
- SimSiam, "Exploring Simple Siamese Representation Learning", which is the claim the `stopgrad` arm tests: https://arxiv.org/abs/2011.10566
- BYOL, where the EMA target encoder comes from: https://arxiv.org/abs/2006.07733
- LeCun's energy-based-model framing of JEPA, "A Path Towards Autonomous Machine Intelligence": https://openreview.net/forum?id=BZ5a1r-kVsf
- "Understanding Dimensional Collapse in Contrastive Self-supervised Learning", on why rank is the thing to measure: https://arxiv.org/abs/2110.09348

**Background**
- Meta AI overview: https://ai.meta.com/research/vjepa/
- The anisotropy problem in contextual embeddings, which is what `centre()` is fixing: https://arxiv.org/abs/1909.00512
