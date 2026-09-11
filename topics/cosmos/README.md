# World Models with NVIDIA Cosmos: Physical AI

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event goes bigger on world models with NVIDIA Cosmos, an open family of world foundation models built for physical AI. Where DreamerV3 learns a small world model for a single agent, Cosmos is a large generative model that simulates the physical world itself: predicting future video, running action-conditioned rollouts, and generating synthetic data to train robots and autonomous machines. It ties directly back to the Isaac Sim event.

Cosmos 3 (released 2026) exposes two surfaces: a Reasoner for understanding and planning, and a Generator for world simulation and future prediction. The weights are open (OpenMDW license) and available on Hugging Face.

Good news for local work: the smaller Cosmos models run on a single DGX Spark (128 GB unified memory) at roughly 30 GB for inference, so we can demo it live rather than only in the cloud.

## What Cosmos is actually for

NVIDIA's production use cases for Cosmos, and how much of each this repo demonstrates.
The honest status matters more than the list: four are covered in depth, one is not
touched at all, and two are blocked behind a licence click.

The mental model underneath all of them:

| piece | job |
| --- | --- |
| Omniverse / Isaac Sim | physics and ground truth |
| **Cosmos Transfer** | make it look like the real world |
| **Cosmos Predict** | simulate what might happen next |
| **Cosmos Reason** | understand and reason about what is happening |
| **Cosmos 3** | Predict and Reason in **one checkpoint** |

That last row is not marketing. `Cosmos3-Nano`'s two manifests point at the same shards
and both experts fit in memory at once (46.0 GiB, measured), which is
[how the understanding surface is reached](#the-other-surface-and-how-to-reach-it) for
free. Transfer is still a separate model.

### 1. Future-state prediction and world modeling: **covered in depth**

Given current observations, predict plausible future states. This is the part that
rhymes with Dreamer: instead of only "what am I looking at", a robot can ask "what
happens next".

`imagine`, `extend`, `horizon`, `emerge`, `counterfact`, `judge`. Measured: a rollout's
[content drifts at segment 7 while its physics holds until 17](#what-it-measured-the-agent-never-gives-up-the-world-dissolves-around-it),
and [the action channel demonstrably drives the prediction](#why-counterfact-is-the-one-to-run-first).

### 2. Physical reasoning and planning: **covered**

Spatial and temporal understanding: "if this person walks forward, will they enter the
road", "what sequence of actions should the robot perform". NVIDIA's Cosmos Reason.

`plan` (decompose a goal, then imagine each step), `judge` (grade your own rollout),
`blind` (rule on clips with no access to the conditions), `choose` (score imagined
futures against a goal). Measured honestly: this surface is a
[marginal instrument at short clip lengths](#the-understanding-surface-and-what-it-is-honestly-worth),
and it wants **categories, not numeric scores**.

### 3. Autonomous vehicle training and validation: **covered**

Generate driving scenarios that are rare, dangerous, or expensive to capture: unusual
weather, sudden pedestrians, odd traffic behaviour, plus variations of existing
sequences.

`dream` does exactly this, from one real dashcam frame: lane changes, stopping at a red
light, rain on the windscreen, a truck braking ahead.
[Half of them were the wrong behaviour](#what-it-measured-half-the-dreams-were-the-wrong-behaviour)
and the critic caught all three, which is the number anyone building an AV data pipeline
needs.

### 4. Robot learning and humanoid robotics: **mostly covered**

Post-train Cosmos on a robot's embodiment and environment, then use it for world
modeling, action prediction, synthetic training data, and closed-loop simulation.

`rollout` (actions to video), `policy` (goal to actions and video), `invert` (video to
actions), [`odyssey`](#odyssey-an-agent-inside-its-own-dream) (a closed-loop agent
acting inside its own dream), `train` (a policy trained on dreams, tested on reality).

**What is missing is stage 2**: post-training Cosmos on a specific robot. Without it you
can only work in embodiments the checkpoint already knows, and
[which those are is not what the table says](#every-experiment-what-it-found-and-what-it-is-for)
, `embodiments` verifies `droid_lerobot` works and `pusht` does not.

### 5. Synthetic data generation for robotics: **half covered, half gated**

Simulate scenes, then convert structured outputs (depth, segmentation, LiDAR, bounding
boxes) into photorealistic video while preserving geometry, so you can vary lighting,
textures, environments and object placement cheaply.

The **Predict** half is built: `dream` generates, `judge` critiques, `invert` labels.
`cycle` prices it at [about 3x the label error](#what-it-measured-dreaming-triples-the-label-error)
and `detail` shows that cost is
[a distribution gap, not lost detail](#why-the-tax-exists-not-blur-a-distribution-gap).

The **Transfer** half needs `nvidia/Cosmos-Transfer2.5-2B`, which is gated.

### 6. Vision and video AI agents: **not built**

Agents that continuously understand video: factories, warehouses, traffic cameras, smart
spaces. "Alert me if a forklift enters the pedestrian area", "summarise what happened on
camera 8", "find every occurrence of someone entering this zone".

Nothing here does this yet, and it is the cheapest gap to close: the understanding
surface already answers questions about arbitrary video in 1.3 to 4.1 seconds. It needs
a task, not a capability.

### 7. Sim-to-real and scenario variation: **gated**

The subtle, practical one. You may already have a physically correct Isaac Sim
environment that looks synthetic. Cosmos preserves the motion and geometry while pushing
the rendering toward realism:

```
Isaac Sim / Omniverse  ->  structured simulation  ->  Cosmos Transfer
                       ->  huge photorealistic dataset  ->  train robot policy
```

This is the path where Cosmos makes the most sense, and it sidesteps every problem this
repo hit: Isaac supplies the actions, so there is **no label tax at all**; you never
leave Isaac's robot, so **no embodiment mismatch**; and Transfer restyles rather than
generates, so **no off-distribution conditioning frame**.

Blocked on accepting the licence for `nvidia/Cosmos-Transfer2.5-2B` (and
`nvidia/Cosmos-Reason2-*`, both verified gated against this box's token).

### Why `pusht` failing matters for all of this

Cosmos is a model **of realistic environments**, and that is not a caveat, it is the
thing itself. Handed a PushT frame, a white background with flat coloured shapes, it did
not degrade gracefully: it **discarded the scene and rendered a photorealistic robot arm
on a wooden desk**, under both action conventions. Its prior asserted itself.

The consequence runs through every use case above. A lightweight gym-style simulator with
abstract rendering is off-distribution, so the sim has to look real, or something has to
make it look real. That something is Transfer, which is why row 7 is not a nice-to-have.

## What is actually in here

Twenty-one experiments and four orchestrators against one checkpoint (`nvidia/Cosmos3-Nano`, 16B, ~33 GB). Every
one of them puts its video in the Flyte report, because a world model you cannot watch
is a number you have to take on faith.

Ten of them drive the generation surface, and they are four questions rather than ten
demos. Three drive the model's OTHER surface, which is the same weights loaded as a
vision-language model, and use it to measure the first ten. The last three put both
surfaces to work on the question the whole "world model as synthetic data" pitch turns
on: can you train on what it dreams?

**What can it generate?**

| task | what it does |
| --- | --- |
| `imagine` | text (and optionally sound) to a predicted world; `--sound` verified to mux a real AAC stereo 48 kHz track, not silence |
| `compare` | one sentence versus the structured JSON caption the model was trained on |
| `emerge` | the model's prediction of the finished clip, decoded at eight points along the schedule (measured: motion is at 95% of final by step 20 of 35, detail at 90% by step 25) |
| `gallery` | all four physical-AI scenes, from a single model load |

**What does it know about actions?** Three modes, one per direction, and they are the
reason this is a world model rather than a video model.

| task | given | predicts |
| --- | --- | --- |
| `rollout` | a frame and a sequence of robot actions | the video that follows |
| `policy` | a frame and a goal | the actions AND the video |
| `invert` | a video | the actions that produced it |

**Is the action channel doing anything at all?**

| task | what it does |
| --- | --- |
| `counterfact` | same frame, same seed, four different action sequences, diffed |

**What happens if you refuse to stop?**

| task | what it does |
| --- | --- |
| `extend` | a generated clip, then continuations conditioned on itself (measured: 165 frames, sharpness -13%) |
| `horizon` | the long one: hours of autoregressive rollout, rendering as it goes |

**Can it measure itself?** The same shards, loaded as a vision-language model instead.
No extra download, and the full walkthrough is [below](#the-other-surface-and-how-to-reach-it).

| task | the loop |
| --- | --- |
| `plan` | it decomposes a goal into subtasks, then imagines each subtask it wrote |
| `judge` | it rolls a world forward, then watches its own rollout and scores it |
| `blind` | it rules on the four counterfactuals with no access to the actions |

**Can you train on what it dreams?** The question the whole "world model as synthetic
data" pitch turns on, and the one everything above only sets up.

| task | what it does |
| --- | --- |
| `cycle` | actions to video to actions, scored against the actions we started with |
| `detail` | the control on `cycle`: is that tax lost detail, or a distribution gap? |
| `robust` | the control on `counterfact`: does the ordering survive a change of seed? |
| `dream` | generate behaviours nobody demonstrated, critique them, label the survivors |
| `choose` | imagine four futures, pick the one that serves the goal, twice, with opposite goals |
| `odyssey` | an agent acting inside its own dream for hours, narrating itself as it goes |

Every one of these has its own block in
[Every experiment, what it found, and what it is for](#every-experiment-what-it-found-and-what-it-is-for),
with the measured result and what the capability is used for outside a demo. Start there
if you want the index rather than the argument.

### Why `counterfact` is the one to run first

Everything else on the list takes it on trust that the model is conditioning on the
actions it is handed, and a video model that quietly ignored its action channel and
just continued the scene plausibly would look exactly as convincing in a demo.

So `counterfact` holds the conditioning frame, the prompt, the seed and the schedule
fixed, and changes only the actions. Four sequences, all anchored at the recorded
starting pose so that none of them ask the model for something outside its training
distribution:

- `recorded` -- what the real robot did. The baseline.
- `held` -- the first action, repeated. The commanded pose never changes, so a model
  that is reading the actions has to predict a robot that **stops**.
- `reversed` -- the same motion backwards.
- `amplified` -- the same motion, doubled about its starting pose. Tests whether
  action *magnitude* registers or only direction.

The output is a bar chart of mean absolute pixel difference from `recorded`. Bars near
zero for every variant would mean the phrase "world model" is not earned here. They are
not near zero.

### Why `horizon` runs for hours

A world model that holds together for two seconds is a video model with good manners.
The claim that it has learned physics is a claim about what happens when you keep going,
so `horizon` keeps going: the four action chunks that ship in the checkpoint first (the
only part of the rollout with real actions behind it), then video-to-video continuations
for as long as you asked, each one conditioned on the model's own previous output and
never re-anchored to anything real.

It repaints the report after every single segment, the same way `topics/dreamerv3` paints
its training run: the growing stitched clip, the newest segment on its own, and three
drift charts. The full clip is re-encoded every fifth segment rather than every one,
because each repaint uploads the report and doing that ninety four times with a growing
video pushes most of a gigabyte through an object store that already leaks heap.

**What a 94-segment run actually did** (4 action chunks + 90 continuations, 3200 frames,
2.8 hours):

| | action chunks | segments 0-14 | 15-29 | 30-44 | 45-59 | 60-74 | 75-89 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sharpness | 734 to 560 | 272 | 82 | 116 | 136 | 136 | 86 |
| motion | 2.5 to 5.6 | 4.7 | 4.1 | 5.9 | 7.3 | 5.7 | 5.9 |
| luminance | 107 | 107 | 115 | 113 | 115 | 109 | 113 |

The result is not the one "error accumulation" leads you to expect. Sharpness collapses
**fast** (734 to a mean of 82 within about fifteen self-conditioned segments) and then
**stops falling**, wandering between roughly 50 and 250 for the remaining seventy five
without ever recovering the action-driven level. Motion goes the other way, rising from
2.5 to a mean near 6 and peaking at 10.8, so the rollout never freezes. Luminance drifts
up about 7%.

So it does not blur to nothing and it does not seize up. It loses most of its detail
quickly, then settles into a soft, more agitated steady state that stays a plausible
video indefinitely while no longer being the video it started as. Watch the frame strip
rather than the charts for the part that matters, which is the scene quietly ceasing to
be a supermarket. That is the honest form of the central unsolved problem for using
generative world models as simulators.

## The other surface, and how to reach it

Cosmos 3 is an omnimodal model. The marketing splits it into a Reasoner for
understanding and planning and a Generator for world simulation, which makes them sound
like two products. On disk they are one set of weights with two front doors, and the
proof is the checkpoint's own file layout:

```
nvidia/Cosmos3-Nano/
├── model_index.json                 <- diffusers reads this
├── model.safetensors.index.json     <- transformers reads this
├── transformer/
│   └── diffusion_pytorch_model-0000{1..7}-of-00007.safetensors
└── vision_encoder/
    └── model.safetensors
```

Both manifests point at the same shards. Check it yourself without downloading anything:

```python
import json
from huggingface_hub import hf_hub_download

index = json.load(open(hf_hub_download("nvidia/Cosmos3-Nano", "model.safetensors.index.json")))
print(sorted(set(index["weight_map"].values())))
# ['transformer/diffusion_pytorch_model-00001-of-00007.safetensors', ...,
#  'vision_encoder/model.safetensors']
```

So on a box where the shared cache is already staged for the generation tasks, the
understanding surface costs **zero extra download**. That is the whole reason it is
worth having here: this box has one disk and the last 33 GB fetch onto it evicted the
cluster.

### Loading it

`Cosmos3OmniForConditionalGeneration` ships in the same transformers line that
`config.py` already required for the Qwen3-VL vision tower (pinned to 5.16.1; see the
version-drift note below for why it is a pin and not a floor). No new dependency, no
`trust_remote_code`, no second repo, no second download.

```python
from transformers import AutoProcessor, Cosmos3OmniForConditionalGeneration

processor = AutoProcessor.from_pretrained(path)          # resolves to Qwen3VLProcessor
model = Cosmos3OmniForConditionalGeneration.from_pretrained(
    path,
    dtype=torch.bfloat16,   # the only precision NVIDIA tests
    device_map="cuda",      # NEVER .to("cuda"); see the double-copy trap below
)
```

That is `world.load_reasoner`. Measured on the Spark: **1.4 minutes, 16.3 GB resident**,
against 2.6 minutes and ~30 GB for the diffusion expert. It is the cheaper half of the
checkpoint by some distance.

### Asking it things

Ordinary chat-template messages, with an image or a list of frames in the content:

```python
messages = [{"role": "user", "content": [
    {"type": "image", "image": pil_image},
    {"type": "text", "text": "The task is to put flower into the red bottle. "
                             "Generate a plan consisting of subtasks."},
]}]
inputs = processor.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True,
    return_dict=True, return_tensors="pt",
).to(model.device)

out = model.generate(**inputs, max_new_tokens=512, do_sample=False)
text = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:],
                              skip_special_tokens=True)[0]
```

Two things in there are load-bearing. Slicing `out[:, inputs["input_ids"].shape[1]:]`
drops the prompt, which for a video question is thousands of vision tokens rather than a
cosmetic prefix. And `do_sample=False` is not timidity: these answers get plotted as
measurements next to deterministic pixel statistics, and a sampled score would put
temperature noise into a chart.

Swap `{"type": "image"}` for `{"type": "video", "video": frames}` and it watches a clip.
Pass a list of PIL frames you sampled yourself rather than a path, or transformers warns
that it is guessing 24 fps because no video metadata came with them. `world.ask` handles
both, and `world.sample_frames` takes 8 evenly spaced frames because the answers stop
changing well before the frame budget does.

Real answers from this checkpoint, at 1.3 to 4.4s each:

| shown | asked | answered |
| --- | --- | --- |
| the bundled planning photo | "generate a plan of subtasks" | *"Move the arm to the flower. Grasp the flower. Move the arm to the red bottle. Place the flower in the red bottle."* |
| `example_t2v_output.mp4` | "does the motion obey real-world physics, 1-10" | *"8. The sponge moves smoothly and naturally as it is lowered onto the plate, maintaining consistent contact and pressure without any unrealistic floating."* |
| a bundled AV clip | "describe what is happening" | *"A vehicle is driving on a road behind a truck."* |

It is not infallible, which is worth knowing before you build a measurement on it. Shown
the 4-chunk supermarket rollout it answered *"picking up a peach from the fruit shelf"*
and ignored the "list the subtasks in order" half of the question entirely.

### The handover, which is the part that bites

They co-exist fine, and I claimed otherwise here for most of a day on the strength of an
estimate I never checked. Measured: generation expert alone **29.6 GiB**, both resident
**46.0 GiB**, **50.5 GiB** peak with a VAE decode running while the language expert sits
there, and both still work. Against a 96 GiB pod limit that is comfortable, and it means
a task that alternates between the surfaces does not have to pay 4 minutes of reloading
per round.

What releasing buys is **headroom, not feasibility**: one expert at a time peaks at 30
GiB instead of 50, which matters on a box where a leaky object store can quietly take 40
GiB and where an oversized load hangs the machine rather than raising. So sequential
stays the default for tasks that generate and then judge, and it is the right call for
them, but it was the right call for the wrong stated reason. The handover looks like:

```python
pipe = None          # the caller's own reference, first
world.release()      # gc.collect() then torch.cuda.empty_cache()
model, processor = world.load_reasoner(repo)
```

Both steps matter and neither is optional. `gc.collect()` because diffusers pipelines
hold reference cycles and CPython will not free those on refcount alone, and
`empty_cache()` because until it runs the caching allocator holds every freed block as
reserved-but-unused, so the next model sees a pool that is still full. `release()`
returns what is still allocated and every task prints it into its report as a
`Handover` row, because "did the first model actually go away" is the one number worth
having when the second load fails.

Prove the whole path on the host before sending anything to the cluster:

```bash
./preflight.sh
./.venv/bin/python smoke_test.py --reason
```

That loads the understanding expert, plans, watches a clip, releases, loads the
generation expert, and renders one clip from the plan it just wrote. If it passes and
the pod does not, the problem is the pod.

### What the three tasks do with it

The point is not three more demos. It is that the model becomes an instrument for
measuring itself, which is a thing you cannot do with a video generator and a separate
VLM without introducing a second vendor's opinion into your evaluation.

| task | the loop |
| --- | --- |
| `plan` | understanding decomposes a goal, then generation imagines each subtask it wrote |
| `judge` | generation rolls a world forward, then understanding watches it and scores it |
| `blind` | generation produces the four counterfactuals, then understanding rules on them with no access to the actions |

**`judge` is the one that changes an existing result.** The 94-segment `horizon` run
measures sharpness, inter-frame motion and luminance, and all three share a blind spot:
they can say whether the rollout is still a well-formed *video*, never whether it is
still a video *of the same thing*. That is why the honest summary of it had to be a
hedged human sentence, "stays a plausible video while no longer being the video it
started as." `judge` measures that instead, with two series pixel statistics cannot
produce:

- **plausibility**, the model grading its own generation 1 to 10
- **description overlap**, how many content words each segment's description still
  shares with segment 0

Both are plotted against sharpness, so the report shows where they agree and where they
part company. Over 21 segments they part company by ten of them: content drift starts at
segment 7 and the plausibility score holds until 17. See the results section below.

`world.description_overlap` is a deliberately crude Jaccard set overlap rather than an
embedding distance: it needs no second model, and every point on the chart can be checked
by reading the two sentences printed beside it, which is not true of a cosine similarity.

**`blind` closes a gap in the best result in this repo.** `counterfact` proves that
changing the actions changes the pixels, measured as a mean absolute difference. That
proves the clips are not identical; it cannot prove the difference is *the difference
the actions describe*, because a model with a decorative action channel that merely
reseeded on it would also produce four different clips. So `blind` shows each clip to
the understanding expert unlabelled, out of order, with no action tensor anywhere, and
asks one question: is this robot moving, or holding still? The `held` variant commands
the starting pose at every step. A judge that cannot see the actions has to be able to
see that it stopped.

## Can you train on what it dreams? The GR00T-Dreams question

This is the use case the whole "world foundation model" pitch rests on, and NVIDIA has a
name and a shipped pipeline for it: **GR00T-Dreams** (the research is called DreamGen).
It is the same loop as `topics/dreamerv3` pointed in the opposite direction.

> **DreamerV3 dreams to train itself. Cosmos dreams to train something else.**

Dreamer learns a small world model from its own experience and runs its actor-critic
inside it, so the dream is private scaffolding and the payoff is the robot walking.
Cosmos is pretrained on the world and you never train it at all: you harvest it. The
dream leaves the model as a labeled dataset that a completely different policy network
trains on.

NVIDIA's pipeline has six stages, and this repo already has four of them:

| # | stage | in this repo |
| --- | --- | --- |
| 1 | collect a few real teleop trajectories | the checkpoint ships 4 chunks and 2 answer-keyed clips |
| 2 | post-train Cosmos on them | no, and not needed to demonstrate the rest |
| 3 | prompt with an image and a new instruction, generate many **dreams** | `imagine`, `rollout`, `extend` |
| 4 | **filter the bad dreams with a video critic** | `judge` |
| 5 | **label the survivors with an inverse dynamics model** | `invert` |
| 6 | train a visuomotor policy on the synthetic trajectories | not yet |

`dream` runs stages 3, 4 and 5 in one task. `cycle` measures whether their output is
worth anything.

Stage 4 is Cosmos Reason's documented job in NVIDIA's own diagram, and here it is the
understanding surface of the same checkpoint. Stage 5 is not a separate model either:
inverse dynamics is a mode flag on the same weights.

### `cycle`: the number that decides whether any of it works

Every stage above rests on one assumption that nobody states out loud. **Actions
recovered from generated video have to be accurate enough to train on.** If they are
not, the pipeline is an elaborate way to manufacture label noise, and no amount of
generating more dreams fixes it.

`cycle` measures it, on the only assets in the checkpoint where it can be scored
honestly. The inverse-dynamics examples are the one place a real video is paired with
the real actions that produced it, so the round trip has an answer key instead of being
graded against itself:

1. inverse dynamics on the **real** clip, which is the floor
2. forward dynamics from **those same real actions**, which produces a dreamed clip
3. inverse dynamics on the **dreamed** clip, which is the round trip

Steps 1 and 3 are the identical operation on two videos of the same event, one filmed and
one imagined. So the difference between their errors isolates what the generation step
cost, and nothing else. Quoting step 3 on its own would blame the dream engine for the
inverse model's own inaccuracy, which is why the report always charts both bars.

```python
example = world.load_inverse_example(path, 0)      # 61 real frames + [60, 9] real actions
a_real, _ = world.invert(pipe, example)            # 1. the floor

fwd = world.forward_meta_from_inverse(example)     # reshape it for a forward rollout
dreamed, _ = world.rollout_chunk(pipe, fwd, example["truth"])   # 2. dream the same actions

dream = dict(example); dream["frames"] = dreamed
a_dream, _ = world.invert(pipe, dream)             # 3. read the dream back

world.action_error(example["truth"], a_real)       # against the same answer key
world.action_error(example["truth"], a_dream)
```

### What it measured: dreaming triples the label error

Both bundled examples, 35 denoising steps, one model load, 21 minutes:

| | inverse on the **real** clip | inverse on the **dreamed** clip | tax |
| --- | --- | --- | --- |
| example 0 | 0.0160 | 0.0534 | **3.3x** |
| example 1 | 0.0101 | 0.0364 | **3.6x** |

MAE on the moving channel, against the actions the real robot executed. A separate probe
at 20 steps rather than 35 gave 0.0166 against 0.0497, so **3.0x to 3.6x across three
independent measurements at two step counts.** Normalised to the channel's own range,
example 0 goes from 4.0% error to 12.0%.

So one round trip through the generator costs roughly a factor of three in label
accuracy, and the forward pass is where it goes: the inverse model reads a real video
about three times better than it reads Cosmos's rendering of the same actions.

### Why the tax exists: not blur, a distribution gap

The obvious reading of that 3x is that generated video is softer, so inverse dynamics has
less to read. `detail` tests it by degrading the **real** clip to exactly the dreamed
clip's sharpness and handing that to the same model:

| | example 0 | example 1 |
| --- | --- | --- |
| inverse on the real clip | 0.0160 | 0.0101 |
| inverse on the real clip, **blurred to the dreamed clip's sharpness** | **0.0161** | **0.0105** |
| inverse on the dreamed clip | 0.0534 | 0.0364 |
| share of the gap blur reproduces | **0%** | **2%** |

**Blur explains none of it.** Real footage carrying exactly as little spatial detail as
the generated clip is read essentially as well as the pristine original: 0.0160 becomes
0.0161. Whatever makes the dreamed clip hard to read, it is not the missing detail.

The sharpness gap was small to begin with, which is its own correction to the obvious
story: the dreamed clips scored 173 and 66 against the real clips' 190 and 93, so 9% and
29% softer, not the mush the word "generated" invites you to picture. The blur radii
needed to match them were 0.28 and 0.41 pixels.

So the tax is a **distribution gap, not an information one.** The generated video is
legible; it just is not camera footage, and the inverse model was trained on camera
footage. That points somewhere specific and cheap: **fine-tune the labeller on generated
video**, which you can do today with data this pipeline already produces, rather than
waiting for a better generator, which you cannot. It also means the 3x is probably not a
floor. Nothing here says a labeller that had seen Cosmos output could not close most of
it.

The honest limit, since it cuts against the conclusion: blur matches one axis. Generated
video also differs in temporal coherence and in its artifacts, and this control holds
neither of those fixed. Landing on the dreamed error would have been strong evidence that
detail was sufficient; missing it by this much is strong evidence that detail is *not*
sufficient, and good but not conclusive evidence for the distribution story specifically.
The next control would hold sharpness fixed and perturb temporal consistency instead.

**What that means for the data engine.** It is not a verdict either way, and it would be
easy to spin it as one. Three times an already-small error is still a small error: 0.05
on a channel whose range is 0.41 is roughly 12% noise, which for behaviour cloning is
survivable and for anything needing precise contact is probably not. The useful framing
is that this is a **budget**, and every other stage of the pipeline spends against it.
Filtering with `judge` should raise the floor by discarding the dreams whose physics went
wrong; longer autoregressive rollouts should lower it, since `judge` already shows
plausibility collapsing after segment 17 and nothing about those late segments deserves
to be trusted as training data.

Also worth saying plainly: this is n=2, one embodiment (`av`, 9-D), one seed, and the
only two clips in the checkpoint that ship an answer key. It is a real measurement of a
real quantity on a very small sample.

Read the **moving-channel** error, not the overall mean. These action vectors are mixed:
in a given clip most channels barely leave their start value, so an average across all
nine is dominated by channels where there was nothing to get right. `action_error`
already identifies which channels moved and reports them separately, and the report
charts only those.

### `dream`: behaviours nobody demonstrated

The conditioning clip shows a car following a truck down a road. The instructions ask
for a lane change, a stop at a red light, a right turn and rain on the windscreen, none
of which that clip contains. That is the whole DreamGen proposition: a world model
supplying demonstrations of things nobody demonstrated.

```python
# stage 3: one real frame, six language instructions, six novel behaviours
result, _ = world.generate(pipe, "The ego vehicle changes lane to the left...",
                           image=first_frame, num_frames=61)

# stage 4: the SAME checkpoint, loaded as a VLM, acting as the critic
behaviour, _ = world.ask(model, processor,
                         "Does the vehicle move sideways into a different lane? "
                         "Answer yes or no...", video=dream_video)
plausible, _ = world.ask(model, processor, Q_PLAUSIBLE_SHORT, video=dream_video)

# stage 5: inverse dynamics turns the survivors into trajectories
actions, _ = world.invert(pipe, {**example, "frames": dream_video})
```

**A dream has to clear two bars, not one.** Physically plausible, and actually showing
the behaviour that was requested. Those come apart in the direction that matters: a clip
can be beautifully physical and show the wrong thing, and as training data that is
*worse* than an obviously broken one, because it is a correctly-labelled demonstration of
something you did not ask for. A broken clip gets thrown out by anyone looking; a
plausible clip of the wrong behaviour gets trained on.

The threshold is a lenient 5 of 10 on purpose. Over-filtering a synthetic dataset
discards exactly the unusual behaviours it was generated to supply, which is the failure
mode that quietly turns a data engine into an expensive copy of your existing dataset.

**Three model loads, in NVIDIA's order rather than the convenient one.** Generating and
labelling in a single pass would save a 2.6 minute reload, at the cost of running ~170s
of inverse dynamics on dreams the critic was about to discard. At six scenarios that is
close to a wash. At sixty it is not, and the order that scales is the one worth writing
down. Each handover prints what it got back, so a leak between stages shows up as a
number rather than as an OOM three stages later.

**The control scenario is the first one**, and it is the behaviour the real clip does
show. If the critic rejects that, the critic is the problem and none of the other
verdicts mean anything. That is the kind of check that costs one extra generation and
saves you from publishing a rejection rate that is really a broken prompt.

### What it measured: half the dreams were the wrong behaviour

Six scenarios, 35 minutes, 6 generated / 3 kept / 3 labelled with `(60, 9)` action
tensors:

| scenario | plausibility | behaviour confirmed | |
| --- | --- | --- | --- |
| drives forward *(control)* | 10 | *"Yes, the vehicle is driving forward along the road."* | **kept** |
| slows as the truck brakes | 10 | *"Yes. The vehicle ahead is braking, as indicated by its illuminated brake lights."* | **kept** |
| changes lane to the left | 10 | *"No, the vehicle drives straight."* | rejected |
| stops at a red light | 10 | *"No, the vehicle does not come to a stop."* | rejected |
| turns right at the intersection | 10 | *"Yes. The vehicle turns right onto a two-lane road."* | **kept** |
| rain begins to fall | 10 | *"No, the sky is overcast but it is not raining."* | rejected |

**The control survived, so the rejection rate means something.** And the rejections are
checkable rather than a shrug: the critic did not say "bad clip", it said the car drove
straight, did not stop, and that the sky was overcast but dry. Watch those three and you
can confirm it yourself in seconds.

So the generator ignored half the instructions and continued the behaviour it was
conditioned on. That is a **50% rejection rate on instruction following**, and it is the
single most important number for anyone planning to use this as a data engine, because
those three clips are exactly the trap described above: physically flawless video of the
wrong thing. Ship them unfiltered and you have taught a policy that "change lane" means
drive straight.

**Every plausibility score was 10 out of 10, so that bar filtered nothing.** All the work
was done by the behaviour check. This is now the third task to find the same thing: the
1-10 scale saturates unless quality has grossly collapsed. `judge` does get 4s out of it,
but only from segment 17 of a self-conditioned rollout, by which point the video has
visibly fallen apart. Treat the scalar as a **detector of gross failure**, not as a
graded quality signal, and put the real filtering load on questions with categorical
answers. Consistent guidance across `blind`, `judge` and `dream`: ask this model for a
category, not a number.

**What this deliberately does not do is stage 6.** Training a policy on the output would
need more data than the bundled assets provide, and a task that trained something on six
trajectories and reported an accuracy would be theatre. What `dream` and `cycle` together
give you is the two numbers you would want before attempting it: how many dreams survive
critique, and how much label accuracy the generation step costs.

## `choose`: planning in imagination

The oldest argument for owning a world model, and the closest thing here to what
`topics/dreamerv3` does with its: **you do not have to try an action in the world if you
can predict what it would do.**

Four action sequences roll forward from one identical conditioning frame. The
understanding surface of the same checkpoint then scores each imagined future against a
goal written in plain words, and the winner is the action a planner would execute.

Dreamer does this scoring with a learned value function that costs millions of
environment steps to train, and it is sharp, and it works only in the one environment it
was trained for. The scorer here is a pretrained model that has never seen this robot,
cost nothing, and works on anything. It is correspondingly blunter. That trade is the
interesting part of putting the two demos next to each other.

### The control is the whole design

Scoring four clips against one goal and announcing a winner proves nothing. The
top-scoring clip might just be the one the model finds most appealing, and from a single
ranking you cannot tell that apart from planning.

So the same four futures are scored **twice, against goals that want opposite things**:

| goal | the question asked of every clip | wants |
| --- | --- | --- |
| get the item moved | *"Does the robot reach for and move an item in this video?"* | large movement |
| hold the arm still and disturb nothing | *"Does the robot hold still without disturbing anything?"* | stationary |

The claim is only earned if the winner **moves**. `held`, the sequence that commands the
arm to its starting pose at every step, should lose the first goal and win the second. If
one candidate wins both, the scorer is ranking on something other than the goal and the
report says so in those words rather than quietly presenting a winner.

Two details that keep the control honest. The motion bucket is measured **once per clip**
and reused as the tiebreak for both goals, because re-asking per goal would let the same
clip get two different motion readings, and then a winner could change because the
measurement wandered rather than because the goal did. And the ranking uses that
three-way bucket rather than a 1-10 score, because the bucket is the instrument `blind`
actually validated (4 concordant pairs, 0 discordant) and the scale is the one that
saturates.

**Nothing is executed.** There is no robot, so the output is the choice and not its
consequences. This is the selection step of a planner, not a closed control loop, and the
report says so.

### What it measured: the choice moved

15 minutes, four rollouts and twelve questions. Every clip's two answers, and they are
complementary on every one:

| candidate | judged motion | *"does it move an item?"* | *"does it hold still?"* |
| --- | --- | --- | --- |
| `recorded` | small movement | No | Yes |
| `held` | stationary | No | Yes |
| `reversed` | stationary | No | Yes |
| `amplified` | small movement | **Yes** *"The robot arm moves a red apple."* | **No** *"the robot is moving its arms."* |

**Winner for "get the item moved": `amplified`. Winner for "hold the arm still":
`held`.** The choice moved with the goal, which is the thing a single ranking could not
have shown.

Two caveats worth saying out loud rather than burying.

**The "still" goal was won on a tiebreak, not on a measurement.** Three candidates
answered yes, and of those `held` and `reversed` both scored `stationary`, so they tied
exactly and the winner fell out of list order. `held` is the right answer and it was not
really chosen; a planner with two genuinely indistinguishable options is a planner with
a coin, and the report shows the raw answers so you can see that is what happened.

**The scorer is deaf to moderate motion.** It called `recorded` (measured inter-frame
motion 1.94, a real robot really picking up fruit) stationary, and only `amplified` at
2.68 registered as movement. That is the same saturation `blind` found from the other
direction, and it means this planner would happily discard a perfectly good demonstration
for not looking busy enough. It gets the extremes right and it is blunt in the middle,
which for a scorer that cost nothing and has never seen this robot is about what you
would expect, and is exactly the gap that Dreamer's expensive per-environment value
function is buying you.

## `odyssey`: an agent inside its own dream

The closest thing here to what `topics/dreamerv3` does, and the thing a video generator
cannot do at all.

At every step the model is handed one frame and a task, and **policy mode denoises the
action channel and the pixel channel together**: it decides what the robot should do and
renders the consequence of having done it, in a single pass. Take the last frame, hand it
back, repeat. Nobody supplies actions, nobody supplies a physics engine, nobody supplies
an environment. **The agent and the universe are the same 16B network.**

Both experts stay resident for this one (46.0 GiB, measured), which is what makes the
narration live rather than a second pass. The understanding surface describes each step
as it happens and the description is burned into the film, so the artifact is one
continuous video with the dreamer's own subtitles.

```python
frame = meta["first_frame"]
for i in range(steps_count):
    step_meta = dict(meta); step_meta["first_frame"] = frame
    seg, actions, _ = world.policy(pipe, step_meta, seed=seed + i)   # decides AND renders
    caption, _ = world.ask(model, processor, Q_DESCRIBE, video=seg)  # narrates, live
    pieces.append(media.label_frames(seg, f"{i}: {caption[:54]}"))
    frame = seg[-1]        # the RAW frame, never the captioned one
```

That last line matters. Feeding back a frame with a subtitle painted on it would ask the
model to treat its own text overlay as part of the world, and it would faithfully start
rendering it. The narration is also generated from the raw frames for the same reason:
showing the model a clip with its own previous caption in it would let that leak into the
next description and fake exactly the continuity this task is trying to measure.

### What it measured: the agent never gives up, the world dissolves around it

40 steps, 33 minutes, 641 frames of continuous film:

| | step 0 | step 39 | |
| --- | --- | --- | --- |
| **action magnitude the agent chose** | 0.3226 | 0.3235 | **steady** |
| sharpness (variance of Laplacian) | 658 | 190 | -71% |
| inter-frame motion | 8.8 | 3.9 | -56% |
| description overlap with step 0 | 1.00 | 0.62 | wanders, low of 0.20 |

**This is the dissociation the effort metric exists for, and it lands on the interesting
side.** Action magnitude is what the agent *decided* to do; inter-frame motion is what
the video *did*. In `horizon` only the second is observable, so a rollout going quiet is
uninterpretable: you cannot tell a policy that stopped trying from a renderer that
stopped rendering. Here they separate cleanly. The agent commits exactly as hard at step
39 as at step 0, oscillating between 0.276 and 0.361 the whole way with no trend, while
the world it is acting in loses 71% of its detail and more than half its motion.

**The policy is fine. The world is dissolving underneath it.**

### The scene survives far better than `horizon`'s, and that is a mechanism not luck

`judge` watched a video-to-video rollout degenerate into floating plastic bags by segment
17. This one is still recognisably a supermarket at step 39. The difference is structural:
each policy step is a **fresh generation anchored on one frame**, not a continuation of a
chained video, so the blur that compounds in `horizon` never gets a second pass to
compound through. If you want a long rollout that stays coherent, condition on a frame and
re-decide, rather than conditioning on your own last clip.

What churns instead is **object identity**, constantly, while the setting holds:

> red apple → orange → apple → tomato → red bell pepper → bag of cucumbers → bag of
> vegetables → green bell pepper → banana → carrot → bag of carrots → bag of onions →
> tomato → red apple

It wanders the produce aisle for forty steps and arrives back at a red apple. The overlap
series is not monotone at all: it drops to 0.30 by step 3, recovers to 1.00 for steps 7
through 11, bottoms out at 0.20 at step 33 and ends at 0.62. Long-horizon drift here is a
random walk through a category, not a slide out of it.

### The thing nobody asked for, which is the most interesting part

**The agent never finishes anything.** Read the captions again: almost every one is
*"picking up"*. In forty steps it initiates forty grasps and completes essentially none,
never putting anything down, never progressing past the first beat of the task it was
given. It is not confused and it is not idle: the effort number says it is working just
as hard at the end as at the start.

There is no goal state in this loop, nothing that tells the model a task is complete, and
no reward. So it does the only thing the setup asks for, forever. That is worth putting
next to `topics/dreamerv3`, where the entire apparatus of actor, critic and return exists
precisely to answer "am I making progress", and where the robot walks because something
told it that walking was worth more than standing. Cosmos has the world model and no
notion of better, and this film is what that looks like for forty steps.

## Every experiment, what it found, and what it is for

The deep dives above and the measured results in
[Status](#status-the-generation-numbers-say-what-they-should-the-judged-ones-say-less)
below are the narrative. **This is the index**: one block per task, what it does, what it
measured here, and where the capability is actually used. Applications are
listed because a demo that only ever says "look, video" is hard to argue budget from, and
because several of these modes map onto things people are paying for today.

The four domains everything below serves: **robot learning** (synthetic demonstrations,
policy evaluation without hardware), **autonomous vehicles** (rare-event scenario
generation), **QA and acceptance testing** of generative pipelines, and **simulation**
where a hand-written physics engine is too expensive or too wrong.

### What it can generate

**`imagine`** - text, and optionally an image, into a predicted world. `--sound` muxes a
real AAC stereo track, verified not silence.
*Found:* 45 frames at 832x480 in 129s, 3.7 s/step.
*Used for:* seeding a synthetic dataset from a written scenario; producing footage of
events too rare or too dangerous to film, which is the whole AV rare-event case; concept
previsualisation before anyone builds a rig.

**`compare`** - the same scene and seed prompted twice, once as a sentence and once as
the structured JSON caption the model was trained on.
*Found:* the structured form is materially better, which is why NVIDIA tells you to
upsample prompts with an LLM first.
*Used for:* deciding whether your generation pipeline needs an LLM prompt-upsampling
stage. That is a real cost and quality decision before a bulk run, not a curiosity.

**`emerge`** - one generation decoded at eight points along the denoising schedule.
*Found:* motion is at 95% of final by **step 20 of 35**, detail at 90% by step 25. The
model commits to what happens before it commits to what it looks like.
*Used for:* **choosing your step count.** If motion settles at step 20, a bulk data run
that only needs correct dynamics can cut denoising by ~40%. On a job generating thousands
of clips that is the difference between a week and four days.

**`gallery`** - all four physical-AI scenes from a single model load.
*Found:* no black frames, inter-frame motion 3.2 to 6.7 across scenes.
*Used for:* a regression suite. Run it after a version bump or a driver change and diff
the numbers before trusting anything else.

### What it knows about actions

**`rollout`** - one observed frame plus a sequence of robot actions, into the video that
follows. Autoregressive across chunks.
*Found:* one 17-frame chunk at 560x640 in 49 to 58s.
*Used for:* replaying a recorded trajectory to see what it would look like; sanity
checking a controller's commands before they touch hardware.

**`policy`** - a frame and a goal, and it returns the actions **and** the video, having
been given neither.
*Found:* predicts a 16x29 action tensor; disagrees with the human demonstration at MAE
0.35 on moving channels, which is not a failure since the task admits many executions.
*Used for:* a zero-training baseline policy; bootstrapping first demonstrations for a task
nobody has demonstrated yet. Also the engine behind [`odyssey`](#odyssey-an-agent-inside-its-own-dream).

**`invert`** - a video in, the actions that produced it out. Inverse dynamics is a mode
flag on the same weights, not a second model.
*Found:* MAE 0.016 on the moving channel against the answer key that ships in the
checkpoint.
*Used for:* **action-labelling unlabelled video.** This is the commercially load-bearing
mode: it turns human demonstration footage, or any video of a task being done, into
trajectories a policy can train on. It is stage 5 of the GR00T-Dreams pipeline.

### Is the action channel real?

**`counterfact`** - same frame, same seed, four different action sequences, diffed.
*Found:* motion `held` 0.41 < `reversed` 1.54 < `recorded` 1.94 < `amplified` 2.68.
Commanding a hold nearly stops the predicted robot.
*Used for:* an **acceptance test on any action-conditioned checkpoint** before you trust
it as a simulator. A model that quietly ignores its action channel looks identical in a
demo and is worthless as an environment. Run this first on anything new.

**`robust`** ([detail](#the-counterfactual-ordering-survives-a-change-of-seed)) - the same
four variants at three seeds.
*Found:* **3 of 3 seeds reproduce the ordering.** The extremes are solid; `reversed` and
`recorded` come within 0.05 at seed 1 and should not be leaned on.
*Used for:* the check you run before stating a result publicly. Cheap insurance against
publishing a property of seed 0.

**`blind`** - the counterfactuals judged by the understanding surface with no access to
the actions, unlabelled and shuffled.
*Found:* the three-way motion bucket recovers the ordering, **4 concordant pairs, 0
discordant, 2 tied**. The 1-10 scale gave inverted noise.
*Used for:* an eval harness where the grader must not see the condition. Also the source
of this repo's most reusable lesson: **ask this model for a category, not a number.**

### What happens if you refuse to stop

**`extend`** - a generated clip, then continuations conditioned on itself.
*Found:* 165 frames across 4 segments, sharpness -13%.
*Used for:* getting past the fixed clip length of one forward pass.

**`horizon`** ([detail](#why-horizon-runs-for-hours)) - hours of autoregressive rollout,
repainting the report as it goes.
*Found:* 94 segments, 3200 frames, 2.8 hours. Sharpness collapses fast (734 to a mean of
82 within ~15 self-conditioned segments) then **stops falling**; motion rises rather than
freezing.
*Used for:* establishing the **usable episode length** of a generative simulator. If you
intend to train or evaluate inside one, this is the number that bounds your episode.

### Can it measure itself?

**`plan`** - it decomposes a goal into subtasks, then generates a clip for each subtask it
wrote.
*Found:* answered in one sentence first, then produced a four-step plan when asked for a
numbered list. Both are shown in the report.
*Used for:* task decomposition for hierarchical control; auto-generating the prompt set
for a data-generation run instead of writing scenario prompts by hand.

**`judge`** - the model watches its own long rollout and scores each segment.
*Found:* over 21 segments, **description overlap breaks at segment 7 while plausibility
holds until 17.** The rollout stops being *about* the same thing long before it stops
being believable.
*Used for:* an **automatic QA gate** on generated data, replacing human review; drift
detection that pixel statistics cannot do. This is Cosmos Reason's documented role in
NVIDIA's own pipeline.

### Can you train on what it dreams?

**`cycle`** ([detail](#cycle-the-number-that-decides-whether-any-of-it-works)) - actions to
video to actions, scored against the actions we started with.
*Found:* **dreaming triples the label error** (0.016 to 0.053, 3.3x and 3.6x on two
examples).
*Used for:* the **go/no-go on a synthetic data programme.** It tells you the noise floor
on your labels before you spend GPU-months generating them. Roughly 12% noise is
survivable for behaviour cloning and probably not for precise contact work.

**`detail`** ([detail](#why-the-tax-exists-not-blur-a-distribution-gap)) - the control on
`cycle`: blur the real clip to the dreamed clip's sharpness and re-read it.
*Found:* **blur explains 0% and 2% of the gap.** It is a distribution gap, not an
information one.
*Used for:* deciding where the money goes. Detail loss would mean waiting for a better
generator; a distribution gap means **fine-tuning the labeller on generated video**, which
is cheap and possible today.

**`dream`** ([detail](#dream-behaviours-nobody-demonstrated)) - generate behaviours nobody
demonstrated, critique them, label the survivors. Stages 3 to 5 of GR00T-Dreams.
*Found:* **3 of 6 kept.** The generator ignored half the instructions and kept driving
straight; the critic caught all three. Every plausibility score was 10/10, so the
behaviour check did all the filtering.
*Used for:* the synthetic data pipeline itself. The rejection rate is your **generation
yield**, and the failures are the dangerous kind: physically flawless video of the wrong
behaviour, which as training data is worse than an obviously broken clip.

**`choose`** ([detail](#choose-planning-in-imagination)) - imagine four futures, score them
against a goal in words, pick one. Then do it again with the opposite goal.
*Found:* **the winner flipped** (`amplified` for "move the item", `held` for "hold still"),
which is the control that makes a single ranking falsifiable. The scorer is deaf to
moderate motion: it called a real pick-up stationary.
*Used for:* model-predictive control and action selection **without a trained value
function**. Dreamer needs millions of environment steps to learn its critic; this one cost
nothing and works on any embodiment, and is correspondingly blunt.

**`odyssey`** ([detail](#odyssey-an-agent-inside-its-own-dream)) - a closed-loop agent
acting inside its own dream, narrating itself live.
*Found:* over 40 steps the **agent never gives up** (action magnitude 0.3226 to 0.3235,
flat) while **the world dissolves** (sharpness -71%, motion -56%). Frame-anchored loops
stay coherent far longer than video-to-video chaining.
*Used for:* **long-horizon policy evaluation without hardware** - stress-testing a
controller in imagination for hundreds of steps at no risk and no rig time. The
action-magnitude series is the part worth stealing: it separates "the policy gave up" from
"the renderer froze", which are indistinguishable in the video alone.

### Orchestrators

| task | runs | roughly |
| --- | --- | --- |
| `world_models` | the eight short generation ones, in sequence | under an hour |
| `reasoning` | `plan`, `blind`, `judge` | ~1.5 hours |
| `data_engine` | `cycle`, `choose`, `dream` | ~1 hour |
| `overnight` | the remaining short ones, then `horizon` | all night |

All four are CPU-only pods on purpose. A GPU-holding orchestrator deadlocks its own GPU
child on "Insufficient nvidia.com/gpu", and a shell loop on the host gets killed for
memory long before an overnight run finishes.

## Run it

```bash
./setup.sh                       # venv, cu130 torch, diffusers
./fetch.sh                       # the 33 GB checkpoint, resumable
./preflight.sh                   # free the unified pool, and say whether it worked
./.venv/bin/python smoke_test.py # prove it works without Flyte in the way
./.venv/bin/python smoke_test.py --reason   # the other surface, and the handover

# then, on the cluster
./.venv/bin/flyte run pipeline.py gallery
./.venv/bin/flyte run pipeline.py counterfact
./.venv/bin/flyte run pipeline.py policy
./.venv/bin/flyte run pipeline.py emerge
./.venv/bin/flyte run pipeline.py extend --segments 3
./.venv/bin/flyte run pipeline.py world_models   # the eight short ones, in sequence
./.venv/bin/flyte run pipeline.py horizon        # hours; watch the report fill in
./.venv/bin/flyte run pipeline.py overnight      # the short ones, then horizon

# the data engine
./.venv/bin/flyte run pipeline.py cycle          # actions -> video -> actions, vs truth
./.venv/bin/flyte run pipeline.py dream          # generate, critique, label (~35 min)
./.venv/bin/flyte run pipeline.py choose         # planning in imagination (~15 min)
./.venv/bin/flyte run pipeline.py odyssey        # closed-loop agent, narrated (~35 min)
./.venv/bin/flyte run pipeline.py odyssey --steps_count 120   # the overnight film
./.venv/bin/flyte run pipeline.py data_engine    # all three, in sequence (~1 hr)

# controls
./.venv/bin/flyte run pipeline.py robust         # is the counterfact ordering seed-proof?
./.venv/bin/flyte run pipeline.py detail         # is the round-trip tax blur or domain?

# the understanding surface
./.venv/bin/flyte run pipeline.py plan           # plan a goal, then imagine the steps
./.venv/bin/flyte run pipeline.py plan --clips 0 # the plan alone, ~3 min
./.venv/bin/flyte run pipeline.py blind          # counterfactuals, judged unlabelled
./.venv/bin/flyte run pipeline.py judge          # a rollout, scored by its own author (~50 min)
./.venv/bin/flyte run pipeline.py judge --segments 5   # plumbing smoke test; shows no drift
./.venv/bin/flyte run pipeline.py reasoning      # all three, in sequence
```

Re-run `./preflight.sh` after any `flyte run` that rebuilt the image. The build fills the
page cache, and the pod that starts next dies on its first CUDA call. See the page-cache
note below.

Prefer `overnight` over a shell loop for anything unattended. A GPU task here holds
most of the 119 GiB unified pool for as long as it runs, so a driving loop sitting on
the host is a candidate to be killed for memory long before the run finishes. That is
not hypothetical: the first attempt at this sequence died exactly that way, with the
shell loop killed while the pod it was waiting on carried on perfectly happily. A
CPU-only orchestrator pod holds 4 GiB, the cluster does the sequencing, and nothing on
your machine has to stay alive. Each child records a failure instead of raising, so one
broken task costs only the tasks after it rather than the whole night.

### The shared model cache, which is worth setting up first

Every task pod used to pull its own copy of the 33 GB checkpoint into `/tmp/hf`. That
cost more wall clock than the generation did, and it is the same shape as the failure in
`reference_flyte_devbox_disk_eviction`: a big model fetch onto an already-full disk
evicted the whole cluster, control plane included.

Stage it once instead, and every pod mounts it:

```bash
docker exec flyte-devbox mkdir -p /var/lib/kubelet/hf-cache/hub
docker cp ~/.cache/huggingface/hub/models--nvidia--Cosmos3-Nano \
    flyte-devbox:/var/lib/kubelet/hf-cache/hub/
docker exec flyte-devbox chmod -R a+rwX /var/lib/kubelet/hf-cache
```

The path is a hostPath **inside the devbox**, which is the part that is easy to get
wrong. k3s runs inside the `flyte-devbox` container, so a task pod's hostPath resolves
against that container's filesystem and not the real host: a hostPath of
`/home/sage/.cache/huggingface` mounts an empty directory and every task silently
re-downloads. `/var/lib/kubelet` is a docker volume, so what is staged there survives
`flyte stop devbox` / `flyte start devbox --gpu`.

Set `COSMOS_SHARED_CACHE=0` on any cluster that is not this devbox.

## What it costs on this box

Measured on the DGX Spark (GB10, 119.7 GiB unified), Cosmos3-Nano in BF16, 35 denoising
steps, with the shared model cache mounted.

| step | cost |
| --- | --- |
| resolve the checkpoint | **0.0 min** (was a 33 GB download per pod) |
| load the pipeline onto the device | **2.5 to 2.6 min**, 7 shards at ~22s each |
| text-to-video, 45 frames at 832x480 | **129s**, 3.7 s/step |
| one action chunk, 17 frames at 560x640 | **49 to 58s**, 1.4 s/step |
| one VAE decode, 45 frames at 480p | **10 to 14s** |
| load the *understanding* expert | **1.4 min**, 16.3 GB resident |
| release it and hand the pool back | **seconds**, to 0.0 GiB allocated |
| one question about one image | **3.6 to 4.4s** |
| one question about a clip, 8 frames | **1.3 to 4.1s** |

The load is now the fixed cost of every task, which is why tasks that need several
clips (`compare`, `counterfact`, `gallery`) generate them all from one load instead of
fanning out. There is one GPU, so fanning out would not help even if the load were free.

The understanding surface is the cheap half of the checkpoint by some distance: it loads
in a bit over half the time, holds about half the memory, and answers in seconds rather
than minutes because nothing is being denoised. A task that generates and then judges
pays both loads, and the judging is a rounding error on top of the generation.

Whole tasks, wall clock including the load: `policy` 4.6 min, `counterfact` 6.3 min,
`emerge` 7 min.

### Checking a report instead of trusting it

A green task and a report that shows a video are different claims, and this repo has
lost afternoons to the gap: a black clip is a valid mp4, a silent track is a valid
audio stream, and an over-budget clip is replaced by an apologetic paragraph. All three
look like success from the outside.

Reports are stored at
`<shard>/<project>/<domain>/<run>/<action>/<attempt>/report.html` in rustfs, where
`<shard>` is a **random two characters chosen per run** (`ki`, `o0`, `ih`, `l9`, `5c`,
`yk` were all seen in one session), so read it off the pod rather than hardcoding it:

```bash
kubectl get pod <run>-a0-0 -n flyte -o jsonpath='{.spec.containers[0].args}' \
  | tr ',' '\n' | grep s3://
```

Then assert on the bytes: count `<video src="data:video/mp4;base64,` matches,
base64-decode each and check bytes 4:8 are `ftyp`, and for a `--sound` run decode the
audio stream and check its peak is not zero. `obstore` is already in the flyte venv;
`boto3` is not. Do not `list('')` the whole bucket, it does not finish.

## Status: the generation numbers say what they should, the judged ones say less

Every task below ran on the Spark and its report was checked by decoding the bytes, not
by trusting a green run. The generation results held up and reproduced; the
understanding-surface results are more qualified, and the qualification is written out
rather than rounded off.

**`counterfact` is the result to lead with.** Same conditioning frame, same seed, only
the actions differ:

| variant | inter-frame motion | divergence from `recorded` |
| --- | --- | --- |
| `held` (first action repeated) | **0.41** | 5.66 |
| `reversed` | 1.54 | **9.65** |
| `recorded` | 1.94 | 0 (baseline) |
| `amplified` (2x about the start) | **2.68** | 4.97 |

Commanding the robot to hold its pose makes the predicted robot nearly stop; doubling
the motion makes it move more than the real demonstration did. That ordering, and not
any single clip, is the evidence the action channel is doing work.

### The counterfactual ordering survives a change of seed

`counterfact` is the result this repo leads with, and it had been measured at exactly one
seed. It reproduced to two decimal places across separate days, which is a fact about
determinism and says nothing about whether the ordering belongs to the actions or to seed
0. `robust` re-runs all four variants at three seeds:

| variant | seed 0 | seed 1 | seed 2 | spread |
| --- | --- | --- | --- | --- |
| `held` | 0.99 | 1.11 | 1.17 | 0.18 |
| `reversed` | 2.10 | 2.25 | 2.10 | 0.15 |
| `recorded` | 2.48 | 2.30 | 2.23 | 0.25 |
| `amplified` | 3.30 | 3.27 | 2.90 | 0.40 |

**3 of 3 seeds reproduce `held < reversed < recorded < amplified`.** The ordering is a
property of the actions rather than of a lucky seed, which is what makes it safe to state
as a result rather than as an anecdote.

Two things to carry with it. **The extremes are solid and the middle is nearly tied**:
`held` and `amplified` are separated by more than two full units at every seed, while
`reversed` and `recorded` come within 0.05 of each other at seed 1. So "commanding a hold
nearly stops the robot, and doubling the motion overshoots the demonstration" is the
robust claim; "reversed moves slightly less than recorded" is riding on a gap smaller
than the seed-to-seed spread and should not be leaned on.

**These numbers are measured before mp4 encoding and the table further up is measured
after it.** `clip_stats` reads the model's raw output frames; `media.probe` decodes the
report's compressed clip, and lossy encoding smooths inter-frame differences, so the same
four rollouts read 0.99 / 2.10 / 2.48 / 3.30 raw and 0.41 / 1.54 / 1.94 / 2.68 through
the codec. Both are honest measurements of different things, and the ordering is
identical either way, which is a small piece of good news about the result. But the
README quoted one and the tasks compute the other with the same label on both, and a
number that changes depending on where you stand deserves to say where it was standing.

**`emerge`** decodes the model's prediction of the finished clip at eight points.
Sharpness climbs 60 to 259 and inter-frame motion 1.3 to 3.6, and both flatten early:
motion is at 95% of its final value by **step 20 of 35** and detail at 90% by **step
25**. The model commits to what happens before it commits to what it looks like.

**`policy`** predicts a 16x29 action tensor and its consequences from one frame and a
goal, with no actions supplied. It disagrees with the human demonstration (MAE 0.35 on
the moving channels), which is not by itself a failure: the task admits many valid
executions and the model was never told which one was recorded.

**`extend`** chains 4 segments into 165 frames, dropping the 5 reproduced conditioning
frames per join. Sharpness falls 13% first to last, but the middle segment scores worse
than the one after it, so three continuations show the mechanism without establishing a
trend. That is what `horizon` is for.

**`gallery`** generates all four scenes at 832x480 with no black frames and inter-frame
motion between 3.2 and 6.7. **`imagine --sound`** muxes a genuine AAC stereo 48 kHz
track (peak 0.587, RMS 0.073), not silence.

**`horizon`** ran all 94 segments in 168 minutes, producing 3200 frames (5.3 minutes of
continuous predicted video) with a 9.2 MB stitched clip embedded in the report. Sharpness
-75% overall. The shape of the decline is the finding and is written up above.

**`invert`, `rollout`, `compare`** predate this round and are unchanged.

### The understanding surface, and what it is honestly worth

**The generation half reproduced exactly.** `blind` regenerates the same four
counterfactual rollouts, and the measured inter-frame motion came back at 1.94 / 0.41 /
1.54 / 2.68 for `recorded` / `held` / `reversed` / `amplified`, matching the numbers
above to two decimal places across separate runs on separate days. That is worth stating
on its own: the pixel measurement in `counterfact` is reproducible.

**`plan` works, and needed one nudge to get there.** Asked the checkpoint's own planning
prompt, the pod answered in a single sentence:

> *"Move the flower to the left of the red bottle."*

which is a restatement of the goal, not a decomposition. Asked once more with
`Answer as a numbered list, with one short step on each line.` appended, it produced:

> *1. Move the arm to the flower. 2. Grasp the flower. 3. Move the arm to the red bottle.
> 4. Place the flower in the red bottle.*

That is the same four-step plan the older transformers on the host gives unprompted, so
the newer one has not lost the ability, only the habit of volunteering it. The task asks
the open question first and the formatted one only on failure, on purpose: leading with
"give me a numbered list" would be us supplying the shape of the decomposition, and
whether the model decomposes at all is the claim. Both answers go in the report so a
reader can see which one the clips came from.

The four subtasks then drove four image-conditioned clips off the real photograph, 112s
each at 720x480, all with real motion (inter-frame 2.19, 2.73, 2.29, 0.61) and no black
frames. The last one, *"Place the flower in the red bottle"*, is the one to watch: it
scores a quarter of the motion of the other three, which is roughly what you would expect
of the step that ends with the arm holding still over the target, and is also exactly the
kind of thing this pipeline cannot distinguish from the generator failing to render the
step at all. The clip is in the report; judge it yourself.

**`judge` is the result of this round.** 21 segments (1 action chunk + 20 self-conditioned
continuations, 50 minutes), then the diffusion expert dropped and the understanding
expert loaded from the same shards to watch each segment with no index, no ordering and
none of the numbers. Three series, one point per segment:

| | segment 0 | end | where it breaks |
| --- | --- | --- | --- |
| sharpness (variance of Laplacian) | 734 | 140 | early and gradually, as in `horizon` |
| description overlap with segment 0 | 1.0 | 0.17 | **segment 7** |
| physical plausibility, judged 1-10 | 8 | 4 | **segment 17** |

**The two drifts are ten segments apart, and that is the finding.** What the rollout is
*about* comes apart long before what it *does* stops being believable. The described
subject wanders the whole time, each one a plausible supermarket object and none of them
the last one:

> apple → apple → apple → orange → banana → papaya → dragon fruit → pomegranate → apple
> → "a fruit" → onion → plastic bag of vegetables → plastic bag of fruits → yellow pear
> → red apple → bag of apples

and through all of that the plausibility score sits at 7 or 8. Only at segment 17 does it
collapse to 4 and stay there, and the justifications stop being generic and start naming
a specific physical violation:

> *"The plastic bag is stretched and distorted in a way that doesn't align with how it
> would behave."*

> *"The plastic bag is seen floating in the air without any visible support, which
> defies physics."*

That is the thing `horizon`'s pixel statistics could not say. Sharpness told us the
rollout got softer; it could not tell us the apple had become a bag, nor that the bag was
floating. The model can, about its own output, from the same weights.

Two honest caveats. The plausibility series is not monotone: isolated 4s appear at
segments 6 and 8 and then recover to 8, so the signal is the sustained collapse at the
end and not any single point. And the description overlap is a Jaccard set, so a segment
scoring 0.857 rather than 1.0 can just be the word "arm" appearing or not. It separates
"apple in a supermarket" from "plastic bag floating"; do not read three decimal places
into it.

The default is 20 continuations for this reason. Five was the original default and it was
measured: scores 8,8,8,8,8,7 and overlap 0.86 to 1.0, which is a rollout that has not
drifted yet, so the instrument comes back untested rather than validated. Use
`--segments 5` as a plumbing smoke test and nothing else.

**The judge recovers the ordering, coarsely, and it took two attempts to find a question
it could answer.** The first version asked for a 1 to 10 motion rating on each unlabelled
clip and got `held` 10, `recorded` 1, `reversed` 1, `amplified` 2, which is close to the
reverse of the truth. Worse, it contradicted its own moving/still answer about the same
clip in two cases out of four, once pairing *"still, the robot's arms are stationary"*
with *"10. The robot's arms are actively moving, picking up and placing fruits."*

A 1 to 10 scale on 1.7 seconds of subtle manipulation is asking for a calibration the
model does not have. Replacing it with a three-way bucket it can answer, and showing all
17 frames instead of 8, produced this:

| variant | judged bucket | measured motion |
| --- | --- | --- |
| `held` | stationary | 0.41 |
| `reversed` | stationary | 1.54 |
| `recorded` | small movement | 1.94 |
| `amplified` | small movement | 2.68 |

Ranked against the pixel measurement that is **4 concordant pairs, 0 discordant, 2
tied**. Wherever the judge commits to an order it is the right order, and the two ties
are the bucket declining to separate clips rather than getting them wrong. `world.rank_agreement`
computes this, and a tie is scored as neither right nor wrong on purpose: punishing a
coarse instrument for being coarse would make the number meaningless.

**The binary question got weaker as the bucket got stronger, which is the finding.** In
the same run the moving/still verdict called three of four clips "still", including
`recorded`. In the earlier run it called `held` still and `recorded` moving, exactly
right. Two runs, opposite conclusions about which of the two questions works. Both the
frame count and the wording changed between them, so neither is cleanly attributable,
and the honest summary is that **this judge is a marginal instrument at this clip
length**: it recovers a coarse ordering, and its answers move when you change how you
ask. The report shows both questions per clip and flags the disagreements rather than
reconciling them, because an unreliable instrument is a result and hiding it would
devalue every other judged number on the page.

So: the pixel divergence in `counterfact` remains the load-bearing measurement, and the
blind judge is corroboration rather than a replacement. That is a smaller claim than
"the model can grade itself", and it is the one the data supports.

## Things that cost time, so you do not pay twice

**Twenty autoplaying videos is a blank tab.** `media.video_html` emitted
`autoplay muted loop` on every clip, which is right for a report with one or two of them
and wrong for `judge`, whose default embeds one per segment. A 21-segment report asked
the browser to decode 21 looping video streams simultaneously and forever. The report was
completely intact on the object store (verified by decoding the bytes: 21 videos, 0
invalid, 6.8 MB) and could still render as an empty panel in front of you, which is the
worst kind of failure because every server-side check passes.

`video_html` now takes `autoplay=False`, which also sets `preload="none"` so the decoder
is not handed the bytes until someone presses play. One or two clips: leave it on. More
than about four: turn it off. Worth checking before blaming the network, because the two
look identical:

```bash
# is the report actually empty, or just unrenderable?
kubectl get pod <run>-a0-0 -n flyte -o jsonpath='{.spec.containers[0].args}' | tr ',' '\n' | grep s3://
# then count <video> tags in the stored bytes and check each decodes to an ftyp box
```

**The presigned report URL hardcodes `localhost:30002`.** If your browser is not on the
box, that resolves to your own machine and the panel is blank however healthy the object
store is. Forward 30002, or re-sign against the Tailscale IP, which serves the same
bytes: signing with `endpoint=f"http://{tailscale_ip}:30002"` produces a URL that works
from anywhere on the tailnet, because SigV4 covers the Host header and rustfs is already
listening on `0.0.0.0`.

**The pod and the host were running different software, and `smoke_test.py` did not
know.** `config.py` asked for `diffusers>=0.39.0` and `transformers>=5.11`, floors rather
than pins, which sounds like the careful choice and is not. The venv `setup.sh` built had
torch 2.13.0 / transformers 5.14.1 / diffusers 0.39.0; an image rebuilt from the same
file resolved to torch 2.14.0 / transformers 5.16.1 / diffusers 0.40.0. That guts the
one claim `smoke_test.py` exists to make, which is "if this passes and the pod does not,
the problem is the pod".

The symptom that exposed it is small and would have been easy to explain away. The same
planning question, greedy (`do_sample=False`), same weights, same image, answered:

- on the host: *"Move the arm to the flower. Grasp the flower. Move the arm to the red
  bottle. Place the flower in the red bottle."* (stable across four repeat calls)
- in the pod: *"Move the flower to the left of the red bottle."*

Both are pinned exactly now, to the versions the numbers below were measured on. Check
the two agree before trusting a host result as a control:

```bash
./.venv/bin/python -c "import torch,transformers,diffusers; print(torch.__version__, transformers.__version__, diffusers.__version__)"
docker run --rm --entrypoint python localhost:30000/cosmos3:<tag> -c "import torch,transformers,diffusers; print(torch.__version__, transformers.__version__, diffusers.__version__)"
```

The host venv is still on the older three; `setup.sh` does not pin them, and matching it
up means a fresh torch download. Until that happens, treat a host smoke test as evidence
that the *model* works and not as a control on the pod.

**`preflight.sh` was silently doing nothing on every box without passwordless sudo.**
The fallback that forces page-cache reclaim by hand referenced `$HERE` without ever
setting it, so under `set -euo pipefail` the whole path aborted with
`HERE: unbound variable` after printing "falling back to forcing reclaim by hand". The
line above it made it look like the fallback had run. Now that it is fixed it does real
work: 44.7 GiB free became 84.2 GiB in one pass. If a script tells you it is falling back
to something, check that the something happened.

**The two experts do not co-exist, and on this box that is a hung machine.** 30 GB of
diffusion expert plus 16 GB of language expert plus a VAE decode does not fit in one
119.7 GiB pool shared with the OS. `pipe = None` alone is not enough to get the first one
back: diffusers pipelines hold reference cycles, so it takes `gc.collect()`, and the
caching allocator keeps every freed block reserved until `torch.cuda.empty_cache()`. That
is `world.release()`, and each task prints what it got back as a `Handover` row.

**Sample the frames yourself before handing a clip to the reasoner.** Passing a list of
PIL frames avoids `Asked to sample fps frames per second but no video metadata was
provided ... defaulting to fps=24`, which is transformers guessing at the frame rate of a
10 fps robot clip. Eight evenly spaced frames is enough; the answers stop changing well
before the frame budget does.

**Page cache makes CUDA fail to start, and `free` will tell you everything is fine.**
After staging 33 GB into the devbox, 107 GiB of the unified pool was page cache and 4 GiB
was genuinely free. Every task died on its first CUDA call, before loading a single
weight:

```
torch.AcceleratorError: CUDA error: out of memory
Returning 2 (CUDA_ERROR_OUT_OF_MEMORY) from cuDevicePrimaryCtxRetain
```

while `MemAvailable` cheerfully reported 110 GiB. Page cache is reclaimable for ordinary
allocations and **not** for creating a CUDA context: the driver does not push the kernel
hard enough to evict it. Without passwordless sudo for `drop_caches`, the fix is to take
the pages from the cache by allocating and touching anonymous memory until `MemFree` is
healthy, then giving it straight back. `preflight.sh` does this automatically now.

**A `flyte run` that rebuilds the image is itself a trigger**, and that is the one that
will catch you, because it happens inside the command you thought was the run. Editing
`config.py` invalidates the pip layer, the rebuild writes several GB of layers through
the page cache, and the pod that starts seconds later dies on `cuDevicePrimaryCtxRetain`
before loading a weight. Measured immediately after one such build: **4.5 GiB free and
102.9 GiB of cache**, which `preflight.sh` turned back into 99.0 GiB free in one pass.
So the rule is not "run preflight before a session", it is **run it after anything that
moves gigabytes**, image builds included.

**`diffusers`, not `cosmos-framework`.** NVIDIA's cookbook sets Cosmos 3 up with a single
`uv sync`, which resolves `natten` and friends: no aarch64 wheels, builds from source
against CUDA 13. The same shape of problem that made the Isaac Sim demo give up on
`pip install isaacsim`. `diffusers>=0.39` ships the whole model natively and routes
attention through `dispatch_attention_fn`, so torch SDPA is enough and no CUDA extension
is involved.

**The safety checker is a gated download.** `Cosmos3OmniPipeline.from_pretrained` defaults
to `enable_safety_checker=True`, which constructs a `CosmosSafetyChecker` from the separate
`cosmos_guardrail` package and pulls a **gated** Llama Guard checkpoint. On a box whose
token has not accepted that licence the pipeline fails at *construction*. Every task here
passes `enable_safety_checker=False`, which means these runs have no content guardrail:
fine for a demo you drive yourself, not fine for anything taking prompts from someone else.

**`device_map="cuda"`, never `.to("cuda")`.** `from_pretrained(...).to("cuda")` materialises
the model on the host and then copies it, so it needs twice the model on one unified pool.

**No `torch.compile`.** Triton does not emit working SASS for sm_121a, so it fails or
silently falls back to something slower than eager.

**Action runs reject `height` / `width` / `num_frames`.** Resolution comes from
`action.resolution_tier` and the frame count from `action.chunk_size + 1`. Passing the
usual arguments raises.

**Video-to-video needs `condition_video_keep="last"`.** The default is `"first"`, which
conditions on the *start* of the clip you hand it and regenerates what you already have
instead of continuing it. The returned segment also reproduces its 5 conditioning frames,
so a stitch has to drop them or the seam is a five-frame stutter.

## What else is on the shelf

Surveyed against the HF API 2026-09-10, so the gating is checked rather than assumed
(`model_info` reports a repo as ungated even when its files 401; test an actual file).

Ungated, and none of them tried here yet:

| repo | size | why it would be worth a task |
| --- | --- | --- |
| `nvidia/Cosmos3-Edge` | 9.2 GB | 4B against this 16B on identical prompts. The size-versus-quality axis this demo has never run, and it is already named in `config.py` as unused. Same diffusers layout, same action assets. |
| `nvidia/Cosmos3-Edge-Policy-DROID` | 9.2 GB | An actual VLA policy post-trained on DROID, which is a different thing from the generic `policy` mode here. Base against robot-tuned on the same frame and goal. |
| `nvidia/Cosmos3-Nano-Policy-DROID` | 32.9 GB | The same question at this checkpoint's size. |
| `nvidia/Cosmos-Embed1-448p` | 2.4 GB | A video embedding model. Would give `horizon` and `counterfact` a semantic distance instead of a mean pixel difference, and would give `judge`'s description overlap something better than a Jaccard set. Needs `trust_remote_code`. |

Gated, so they need a licence accepted against `HF_TOKEN` first (both verified to 401 on
a real file, not just on metadata):

- **`nvidia/Cosmos-Transfer2.5-2B`** (55 GB) is the one to want. It is the sim2real
  ControlNet with edge, depth, seg and blur variants, diffusers 0.40 ships
  `Cosmos2_5_TransferPipeline`, and it loads off `revision="diffusers/general"` branches
  of that repo rather than a converted checkpoint. Point it at a depth or segmentation
  render from `../isaac-sim` and it is the cross-topic demo this repo has been walking
  toward.
- `nvidia/Cosmos-Reason2-2B` / `-8B` / `-32B` (4.9 / 17.5 / 64 GB). Largely redundant now
  that the built-in understanding surface works; worth it only to ask whether a dedicated
  reasoner beats the omni model's own expert at the judging above, which given how
  marginal that instrument turned out to be is a real question.

`nvidia/Cosmos3-Super-Image2Video-4Step` is 129 GB and does not fit the 119.7 GiB pool in
BF16. Skip it.

## Reference

- NVIDIA Cosmos (open platform of world models): https://github.com/nvidia/cosmos
- Cosmos overview: https://www.nvidia.com/en-us/ai/cosmos/
- Cosmos Cookbook (runnable recipes): https://nvidia-cosmos.github.io/cosmos-cookbook/
- V-JEPA 2 (Meta) as an open, prediction-focused world model to compare: `../v-jepa`
- `../dreamerv3` -- the same idea learned from scratch for one small environment
- `../isaac-sim`, `../rl-mujoco` -- where the physics is a hand-written engine instead
