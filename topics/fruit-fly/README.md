# A simulated brain, based on a real fly's wiring

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This one is about the fruit fly connectome: the complete wiring diagram of an adult *Drosophila* brain, 138,639 neurons and 54 million synapses, reconstructed from electron microscopy and released for anyone to download. The goal is narrow and checkable: **take that wiring diagram, turn it into a running brain, bolt it onto a physics-simulated fly body, and see whether the animal does something.**

It does, four times. On one DGX Spark, with no GPU and no training of any kind:

- a fly whose only controller is the measured wiring of a real brain **turns 60° toward a pillar it can see and walks to it**, while three controls, including one where that same wiring is shuffled, walk straight past;
- when it arrives at a drop it cannot see the contents of, **sugar makes it stop and extend its proboscis while bitter does not**, because sugar drives 30 of the 110 feeding motor neurons in this brain and bitter drives none;
- the same brain, bolted to the back of a **four-legged robot** twenty-five times its length, **drives it to the pillar** in 7.67 s, while the shuffled control gets barely further than no brain at all;
- and the fly's **head-direction compass is recoverable from the wiring alone**. The ring's order is not in the annotation, so it comes out of the connectivity; put a bump of activity on it and the downstream response rotates with it to within **5.2 degrees**, while the shuffled control is completely silent.

Then the uncomfortable parts, which are the most interesting results here. Turning synaptic strength up so that *more* of the brain participates makes the fly **worse**. Cutting 1,697 neurons out of the visual pathway changes nothing while cutting **two** costs it a third of its speed.

And this brain **cannot see motion at all**. Swat at it with a looming sphere and the giant fibre never fires; put it in a rotating striped drum and the two directions of rotation differ by sixty times *less* than the noise in a single run. Both failures have the same cause and it is one row of a table: L1 is glutamatergic in this dataset and dominates Mi1's input at **-122,574 signed synapses**, which clamps the medulla off and makes T4, T5 and every looming detector downstream unreachable. Tracking that down is the most useful thing in this repo.

The first half of this README is a tutorial: what a connectome is, how a wiring diagram becomes a spiking network, what it has and does not have in common with an artificial network, how you attach one to a body, and why the obvious way to do that fails. The second half is the demo: what it runs, what it measured, and every trap that cost real time on this box.

**Contents**

1. [Three things people mash together](#1-three-things-people-mash-together)
2. [From a wiring diagram to a brain](#2-from-a-wiring-diagram-to-a-brain)
2b. [Is this a brain, or just a neural network with extra steps?](#2b-is-this-a-brain-or-just-a-neural-network-with-extra-steps)
3. [The body](#3-the-body)
4. [The bridge, and why the obvious version fails](#4-the-bridge-and-why-the-obvious-version-fails)
5. [Reading this repo](#5-reading-this-repo)
6. [The demo](#6-the-demo) and how to run it
6b. [Four more experiments](#6b-four-more-experiments-and-the-one-measurement-that-explains-three-of-them): the swat, the drum, the compass, and two flies at once
6c. [A reaction you can see, and training it](#6c-a-reaction-you-can-actually-see-and-an-attempt-to-train-it)
6d. [What it is releasing](#6d-what-it-is-releasing-not-just-which-cells-fired)
6e. [Putting the fly on a robot](#6e-putting-the-fly-on-a-robot)
7. [Things that cost real debugging time on this box](#7-things-that-cost-real-debugging-time-on-this-box)
8. [A guided tour of the code](#8-a-guided-tour-of-the-code), if you are presenting it

---

## 1. Three things people mash together

"Google mapped a fly brain and people are training it to do funny things" is three separate projects, and keeping them apart is most of understanding the field.

**The map.** [FlyWire](https://flywire.ai) is the complete connectome of an adult female *Drosophila melanogaster*: every neuron, every synapse between them, reconstructed from a 7,000-slice electron-microscopy volume. Google's role was the machine vision, flood-filling networks that trace neurons through the stack; the proofreading took a decade and 33 years of cumulative human effort. Published in *Nature*, October 2024. It is a **wiring diagram, not a running brain**: a list of which cell connects to which and how strongly. You can download the whole thing right now, and this repo does: the connectivity table is a 100 MB parquet file.

**The body.** Two open MuJoCo models of the animal itself:

- **flybody** (Google DeepMind + HHMI Janelia, *Nature* 2025), 67 body parts and 102 degrees of freedom, with MuJoCo extended for fluid forces and foot adhesion. They trained reinforcement-learning policies for walking and vision-guided flight.
- **NeuroMechFly v2**, the `flygym` package (EPFL), a micro-CT body with compound-eye vision, antennal olfaction, leg adhesion and a brain/ventral-nerve-cord control split. Version 2.x landed in March 2026: a full rewrite, roughly 10x faster on CPU and 300x on GPU via MJWarp, and it now carries the flybody model too, which makes it the superset. **This demo uses flygym 2.1.0.**

**The funny things.** In March 2026 [Eon Systems](https://eon.systems) connected the two: a connectome-derived brain driving NeuroMechFly v2 in real time, syncing every 15 ms. Their emulated fly groomed when dusted, walked toward sugar and fed on contact, with no training and no scripts. Their code is not released.

Everything Eon used is open, which is what makes this weekend project possible:

| piece | what it is | package |
|---|---|---|
| the wiring | FlyWire 783, 138,639 neurons, 15.1M connected pairs | a parquet file |
| the brain | Shiu et al. 2024's leaky integrate-and-fire model over that wiring | `brian2` |
| the body | NeuroMechFly v2, MuJoCo, compound eyes | `flygym` |
| the eyes-to-brain map | this repo | `bridge.py` |

All of it runs on Python 3.12 with aarch64 wheels, so the whole stack installs clean on the Spark.

---

## 2. From a wiring diagram to a brain

A connectome gives you two tables. One says which neurons exist. The other has one row per *ordered pair* of connected neurons, carrying how many synapses that pair shares and the presynaptic neuron's predicted neurotransmitter:

```
Presynaptic_Index  Postsynaptic_Index  Connectivity  Excitatory  Excitatory x Connectivity
      70171               118401             10          +1                 +10
      70171                92016              3          -1                  -3
```

That last column is the whole model. **It is the synaptic weight**, and nobody chose it: it is a count of synapses in a real animal, signed by that cell's neurotransmitter.

To make it run you need a neuron model. This repo uses the one from Shiu et al. 2024 (*Nature*), the reference LIF model built on exactly this dataset, so the numbers here can be checked against a published paper. Every neuron is a leaky integrator with a threshold:

```
dv/dt = (v_0 - v + g) / t_mbr      membrane, leaks back to rest with a 20 ms constant
dg/dt = -g / tau                   synaptic input, decays with a 5 ms constant
if v > v_th:  spike, v = v_rst     threshold at -45 mV, 7 mV above rest
```

with eight constants, all from the literature and listed in `brain.py`. When a neuron spikes, every downstream partner's `g` jumps by that pair's weight times one global scale `w_syn` = 0.275 mV. **`w_syn` is the only free parameter in the entire brain.** There is no training, no fitting, no reward.

In Brian2 the whole thing is about fifteen lines:

```python
neurons = NeuronGroup(138_639, EQS, threshold="v > v_th", reset="v = v_rst; g = 0*mV", ...)
synapses = Synapses(neurons, neurons, "w : volt", on_pre="g += w", delay=1.8*ms)
synapses.connect(i=connectome.pre, j=connectome.post)     # 15,091,983 connections
synapses.w = connectome.weight * w_syn                    # the EM synapse counts
```

Measured on this box: **1.9 s to build, 3.2 GB resident, and about 0.3 s per 100 ms of brain time** once Brian2's Cython codegen has compiled (a one-off 12 s). A whole fly brain runs at roughly a third of real time on a CPU, which is the single most surprising number in this project.

### Two things to know before you expect behaviour

**It is completely silent at rest.** Zero input, zero spikes, forever. There is no spontaneous activity in this model, so nothing happens until you drive something. That sounds obvious and it has a sharp consequence for the bridge, in section 4.

**It has no map of itself.** The connectome is 138,639 anonymous numbers until you join it against the [cell typing](https://github.com/flyconnectome/flywire_annotations) (Schlegel et al. 2024, *Nature*), which names them. That join is what lets this repo say "the sugar-sensing gustatory neurons" or "DNa02, left" instead of a 19-digit ID, and 138,625 of the 138,639 neurons carry a name. The composition is worth seeing:

| class | neurons | |
|---|---|---|
| optic | 77,541 | more than half the brain is vision |
| central | 32,383 | |
| sensory | 16,907 | of which 10,855 are photoreceptors |
| visual_projection | 8,038 | optic lobe out to central brain |
| ascending | 1,750 | body to brain |
| **descending** | **1,303** | **brain to body. the only way out** |
| motor | 110 | |

That last row but one is the interface this whole demo turns on.

---

## 2b. Is this a brain, or just a neural network with extra steps?

This is the question the demo gets asked most, so it is worth answering with numbers. The short version: the *architecture* is measured rather than designed, the *chemistry* is present but flattened almost to nothing, and the things that make a brain a brain other than wiring are mostly missing.

### What makes it unlike an artificial network

**The weights were not fitted, they were counted.** Every number in the 15,091,983-row edge list is how many synapses an electron microscope found between that specific pair of cells in one real animal. There is no loss function anywhere in this repo. The single global scale `w_syn` is the only free parameter, and section 6's `gain` task exists precisely to show how much that one knob changes the answer.

**Dale's law is obeyed, exactly.** A neuron's sign comes from the transmitter *it* releases, so a cell is excitatory to everything it touches or inhibitory to everything it touches, never both. Checked directly against this dataset: **0 of 138,005 presynaptic neurons have a mixed outgoing sign**. An artificial layer has no such constraint; its rows are freely mixed in sign, and that freedom is a large part of what makes backprop work. A connectome cannot use it.

**It is sparse, recurrent, and delayed.** 15.1M connections out of a possible 19.2 billion is a density of **0.0785%**. There are no layers and no forward direction: **26.6% of connected pairs have a return edge**, and every synapse carries a fixed 1.8 ms delay, so the network is a dynamical system running in continuous time rather than a function evaluated once. There are no autapses (no neuron synapses onto itself).

**The units are physical.** Millivolts, milliseconds, hertz. `w_syn` is 0.275 mV of depolarisation per synapse against a threshold sitting 7 mV above rest, which is why it takes roughly 25 simultaneous synapses to fire a cell and why so much of the brain stays silent. Half of all connected pairs (**49.7%**) share exactly one synapse and contribute 0.275 mV each; the median pair shares 2 and the largest shares 2,405.

### Where the chemistry actually is

Yes, the neurotransmitters are in the data. Every neuron carries a predicted transmitter from a classifier trained on EM appearance, and 87,837 of them also carry a literature-derived ground truth:

| transmitter | neurons | what this model does with it |
|---|---|---|
| acetylcholine | 86,193 | `+1`, the main excitatory transmitter |
| glutamate | 24,875 | `-1`, inhibitory in the fly |
| GABA | 19,171 | `-1` |
| dopamine | 5,909 | `+1` or `-1`, as if it were a fast synapse |
| serotonin | 2,282 | same |
| octopamine | 216 | same |

And that last column is the important one. **The model collapses six transmitters into a single `+1/-1` sign.** So the chemistry is mapped, and then almost all of it is thrown away:

**Neuromodulation is gone.** Dopamine, serotonin and octopamine are not fast point-to-point signals in a real brain; they set *state*, over seconds to minutes, across whole regions. Hunger, arousal, and the valence signal that makes learning possible all live there. Here, **8,407 modulatory neurons are simulated as if they were ordinary fast synapses**, which is the single largest thing this model is not.

**There is no plasticity at all.** Weights are constants. The mushroom body, the fly's learning centre, is fully present in the wiring (**5,177 Kenyon cells, 96 MBONs, 331 dopaminergic neurons**) and in this model it cannot learn anything, ever. A fly that can be trained is one of the most robust facts in the field, and none of it is reachable from a connectome alone.

**Receptors are not in the connectome, and that bites.** The sign above is the *presynaptic* cell's transmitter, but what a synapse actually does depends on the receptor on the *postsynaptic* side, which electron microscopy cannot see. The demo runs straight into this: photoreceptors are histaminergic, histamine is not one of the six transmitters the classifier predicts, so R1-6 gets labelled acetylcholine and the very first synapse of the visual system has the wrong sign. Section 6b traces what that costs, all the way to a dead motion pathway.

**Gap junctions are missing.** Electrical synapses are largely invisible to this reconstruction, and the fly's escape circuit is built on them. The giant fibre's most important inputs are simply not in the dataset.

**Spikes are assumed.** Most of the fly's optic lobe does not spike at all; photoreceptors and lamina cells signal with graded potentials. An integrate-and-fire model has no way to represent that, which is why `flyvis` exists as a separate, trained, graded model of exactly this circuit.

### So what is it

It is a **wiring diagram of one real animal, run forward in time under a deliberately simple neuron model**. That is enough to produce the fixation behaviour in section 6 and the compass in the swat-and-compass experiments, and it is provably not enough for motion vision or learning. The honest framing is that the connectome constrains the architecture completely and constrains the dynamics barely at all, and every result here is a statement about how far the architecture alone gets you.

---

## 3. The body

`flygym` 2.1.0 supplies a fly with 6 legs, position-actuated joints, leg adhesion, and two compound eyes of 721 ommatidia each. Stepping it is MuJoCo:

```python
world = FlatGroundWorld()
fly = make_locomotion_fly("nmf", colorize=True)
fly.add_vision()
world.add_fly(fly, spawn_position, spawn_rotation)
sim = Simulation(world, timestep=1e-4)
```

Measured here: **2,900 physics steps per second**, so 0.29x real time, and **2 ms** per readout of both compound eyes.

### The brain does not move legs

This is the design decision everything else follows from, and the animal made it, not me.

The connectome in this repo is a **brain** connectome. The neurons that actually move legs live in the ventral nerve cord, a different dataset (MANC) that this model does not contain. The real interface between them is the descending neurons, ~1,300 cells that are the only route from head to body, and what they carry is much closer to *"turn left"* than to *"extend the left front femur by 4 degrees"*.

So the split here is the fly's own. The connectome decides **where to go**. A central pattern generator decides **how to move six legs to get there**: `flygym`'s `HybridTurningController`, a coupled-oscillator tripod gait with stumbling and retraction reflexes. Its entire input is:

```python
descending_signal   # shape (2,): drive for the left and right halves of the body
```

Two numbers. That is not a simplification invented for this demo; it is the standard abstraction in the fly locomotion literature and it is roughly what DNa01 and DNa02 are believed to carry.

Measured, so the sign convention is not a guess:

| `descending_signal` | heading change over 1.2 s |
|---|---|
| `[1.3, 0.7]` | **-123°**, turns right |
| `[0.7, 1.3]` | **+133°**, turns left |
| `[1.0, 1.0]` | +6.5°, straight |

---

## 4. The bridge, and why the obvious version fails

Now connect them. Every 15 ms:

```
eyes   = body.look()                  render 2 x 721 ommatidia
rates  = bridge.apply_vision(...)     darkening -> Hz on real photoreceptor neurons
tick   = brain.run(0.015)             138,639 LIF neurons step forward
signal = bridge.descending_signal()   1,291 descending neurons -> 2 numbers
body.step(signal)                     150 physics steps, 1 video frame
```

15 ms is the interval Eon used for the same coupling. At the fly's ~13 mm/s walking speed that is 0.2 mm of travel per decision.

Four things had to be measured rather than assumed, and three of them broke the obvious design.

### 4.1 Which senses can steer at all

Drive one side's sensory neurons, and read the left and right descending populations. If the left and right rows don't flip sign, that channel cannot tell the body which way to turn. Measured here, 300 ms at 150 Hz:

| stimulus | DN pop left | DN pop right | R − L |
|---|---|---|---|
| photoreceptors, left | 217 Hz | 377 Hz | **−160** |
| photoreceptors, right | 327 Hz | 173 Hz | **+153** |
| mechanosensory, left | 8,670 Hz | 4,673 Hz | **+3,997** |
| mechanosensory, right | 2,913 Hz | 7,207 Hz | **−4,293** |
| olfactory, left | 2,390 Hz | 2,510 Hz | −120 |
| olfactory, right | 2,410 Hz | 2,387 Hz | +23 |

Vision and touch flip cleanly. Vision is **contralateral** (left eye drives right descending neurons) and touch is **ipsilateral**, both as the anatomy predicts. **Smell does not lateralise at all in this model**, which is why this demo is a visual one and not the odour-tracking demo I originally planned. Real flies track odours mostly by sampling over time rather than comparing two antennae, so this is arguably the model being right.

### 4.2 Not DNa02. The population.

The obvious readout is DNa02: one neuron per side, the best-known steering cells in the fly, described in a dozen papers. Measured here, it does not work:

| stimulus | DNa02 left | DNa02 right |
|---|---|---|
| ORN_DM1 **left** only | 50 Hz | 2 Hz |
| ORN_DM1 **right** only | 54 Hz | 0 Hz |
| ORN_DM1 both | 50 Hz | 0 Hz |
| ORN_DM4 left / right | 50 / 46 Hz | 0 / 0 Hz |

DNa02-left fires at ~50 Hz and DNa02-right is silent **no matter which side you stimulate**. A fly steered by that pair turns the same way forever. This is not a bug: it is what a single cell does in a model where nothing is tuned, where one cell of a pair happens to sit closer to threshold and therefore wins every time. The 1,291-neuron descending population does not have the problem, because the bias averages out across cells. The report plots DNa02 next to the population for exactly this reason.

### 4.3 The brain has a floor, and a silent brain looks like a working one

Drive the eye's photoreceptors and watch the descending populations:

| photoreceptor drive | 5 | 10 | 25 | 50 | 75 | 100 | 150 | 200 Hz |
|---|---|---|---|---|---|---|---|---|
| DN left | 0 | 0 | 23 | 87 | 127 | 140 | 210 | 257 Hz |
| DN right | 0 | 0 | 60 | 213 | 240 | 297 | 367 | 393 Hz |

**Below 25 Hz nothing propagates at all.** The first version of this bridge mapped "fraction of the eye that darkened" (0 to about 0.075 for a pillar) straight onto 0–150 Hz, delivered 7–11 Hz, and produced a perfectly functioning fly with a completely silent brain: and the run *completed successfully*, with plausible video, and only the traces showed the descending neurons flat at zero for its entire life.

The fix is the one the animal uses. A photoreceptor in daylight is not silent waiting for an object; it fires tonically and an object **modulates** that rate. So both eyes get 50 Hz always, which parks the visual system in its responsive band, and darkening adds up to 150 Hz on top.

### 4.4 An empty world does not look empty

"Which eye is darker" fails before it starts. In an **empty** arena the two eyes disagree: 0.4563 of left ommatidia fall below the dark threshold versus 0.4674 of right ones, because the ground plane fills the lower half of both eyes and the two views of it are not identical. A raw left-right comparison therefore reports an object at all times, in a fixed direction, with nothing there.

So the retina measures the empty world once at startup and compares against it. Then the azimuth sweep is clean and correctly signed at every angle (fractions of ommatidia darkened *relative to empty*; +azimuth is the fly's left):

| object azimuth | −90 | −60 | −30 | −15 | 0 | +15 | +30 | +60 | +90 |
|---|---|---|---|---|---|---|---|---|---|
| left eye | .000 | .000 | .000 | +.003 | **+.044** | +.028 | +.025 | +.015 | +.019 |
| right eye | +.017 | +.022 | +.021 | +.029 | **+.046** | +.003 | .000 | .000 | .000 |

Straight ahead it is in both eyes, which is the fly's real binocular overlap.

### 4.5 What is left over, and what is chosen

After all that, the bridge is:

```python
rate_eye  = 50 Hz + 1300 * darkening        # capped at 200 Hz
imbalance = (DN_right - DN_left) / (DN_right + DN_left) - bias
steer     = turn_gain * smooth(imbalance) * turn_sign * toward
signal    = [base_drive - steer, base_drive + steer]
```

`bias` and `turn_sign` are **measured** per wiring by `bridge.calibrate()`, not chosen: the brain is asked what it does with each eye lit, and the sign falls out. Three things are chosen by hand and they are the honest weak points of the demo: `turn_gain` (how hard to steer), `base_drive` (how fast to walk), and `behaviour`: whether "toward" or "away". **The connectome supplies the lateralisation. It does not supply the valence of a plain dark post**, and pretending otherwise would be the lie in this project.

Which is exactly why the shuffled control exists.

---

## 5. Reading this repo

| file | what it is |
|---|---|
| `connectome.py` | the wiring. Loads the three files, joins them, resolves `select(cell_type="DNa02")` into indices. Also `shuffled()`, the null model. |
| `brain.py` | 138,639 LIF neurons in Brian2, steppable 15 ms at a time. |
| `body.py` | the arena, the fly, the retina, the walking controller. |
| `bridge.py` | eyes in, descending neurons out. Every measurement in section 4 is in its docstrings. |
| `brainviz.py` | the brain lighting up, at the neurons' real anatomical coordinates. |
| `compass.py` | recovering the head-direction ring from connectivity, and driving it. |
| `learn.py` | the mushroom body, and the one place in this repo where a synapse changes. |
| `chemistry.py` | what the brain releases per tick, split by transmitter. |
| `rider.py` | a four-legged robot, its gait, and the fly bolted to its back. |
| `duet.py` | two flies in one arena, and the measurement of whether they can see each other. |
| `video.py` | the four-panel cockpit and the H.264 encode. |
| `run.py` | the closed loop and the four modes. |
| `pipeline.py` | the Flyte tasks. |
| `reports.py` | the report HTML. |

---

## 6. The demo

```
fixate (orchestrator, CPU)
  ├── fly_once connectome   the real FlyWire wiring, eyes connected
  ├── fly_once shuffled     same neurons, same out-degrees, same weights, partners permuted
  ├── fly_once blind        real wiring running, photoreceptors held at tonic
  └── fly_once nobrain      no brain at all, constant forward drive
```

A dark pillar stands 12 mm away, 60° off the fly's nose. **A fly that never turns misses it by 10.4 mm**, twice the arrival threshold, so arriving requires steering. Each run ends when the fly gets within 2 mm of the pillar's surface, or after the tick budget.

Everything is **CPU-only**, and that is the point rather than a limitation: nothing here contends for the Spark's single GPU, and the orchestrator-deadlock trap from the videogen and rl-mujoco demos cannot fire.

```bash
# The main event: the connectome fly plus three controls, one report.
flyte run pipeline.py fixate --ticks 260

# Just the fly, fastest path to the clip.
flyte run pipeline.py fixate --modes '["connectome"]' --ticks 260

# Walk to the drop, taste it, decide. Sugar vs bitter vs tasteless.
flyte run pipeline.py feed

# Cut named cell types out of the brain and see what breaks.
flyte run pipeline.py lesion --targets '["L1","L2","Lai","DNp28"]'

# Sweep the one free parameter and watch the brain wake up (and get worse).
flyte run pipeline.py gain --w_syn_values '[0.275,0.55,1.1]'

# How far off the nose can the pillar start and still be found?
flyte run pipeline.py sweep --headings '[20,40,60,75]'

# Brain only, no body: which sensory channels lateralise?
flyte run pipeline.py probe

# Swat at it. A sphere flies in; does anything know something is coming?
flyte run pipeline.py swat --ticks 90

# Find the head-direction ring in the wiring, then put a bump on it. 40 seconds.
flyte run pipeline.py compass

# A tethered fly in a rotating striped drum. The oldest experiment in fly vision.
flyte run pipeline.py drum

# Two flies, two independent brains, one arena and one pillar.
flyte run pipeline.py duet --ticks 260

# Poke one antenna and watch it swerve. The biggest reaction in the model.
flyte run pipeline.py poke

# Train it: punish an odour and watch the mushroom body stop responding.
flyte run pipeline.py teach

# Put the fly on a four-legged robot and let it drive.
flyte run pipeline.py ride --ticks 750
```

| task | what it answers |
|---|---|
| `fixate` | Does the connectome steer the body, and do the controls fail? |
| `feed` | Does the taste circuit produce the right behaviour for sugar and not bitter? |
| `lesion` | Which named cells is the behaviour actually running through? |
| `gain` | How much of the brain is switched on, and does more of it help? |
| `sweep` | How large a bearing error can it correct? |
| `probe` | Which sensory channels lateralise at all? (no body, fastest) |
| `swat` | A sphere flies at the fly. Does the escape circuit fire? |
| `compass` | Is the head-direction ring recoverable from wiring, and does it rotate? |
| `drum` | Optomotor: does a rotating drum make a tethered fly turn? |
| `duet` | Two flies, two brains, one pillar, in the same physics |
| `poke` | Touch one antenna. Does the fly swerve, and does the side matter? |
| `teach` | Change a synapse. Can this brain learn about a specific odour? |
| `ride` | Can the connectome steer a body it never evolved for? |

| Flag | Default | What it does |
|---|---|---|
| `--ticks` | 200 | Brain ticks of 15 ms. 260 is ~3.9 s of fly time |
| `--modes` | all four | Which runs to do |
| `--spawn_heading_deg` | 60 | How far off the nose the pillar starts. Smaller makes the controls look better by accident |
| `--object_distance` | 12 | mm. Further away means a smaller visual angle and a weaker signal |
| `--turn_gain` | 0.6 | Steering strength per unit of neural imbalance |
| `--behaviour` | attract | `attract` or `avoid`. The one behavioural choice made by hand |

Runs land in the **`physical-ai`** Flyte project (`.flyte/config.yaml`), the same one as the rl-mujoco G1 and the Isaac Sim demos.

### The report

Each run's report carries a four-panel cockpit clip, every panel the same 15 ms instant:

```
+---------------------------+---------------------------+
|  the fly (tracking cam)   |  its brain, 138,639 cells |
|                           +---------------------------+
|                           |  descending command       |
+-------------+-------------+---------------------------+
| left eye    | right eye   |  where it has been        |
| 721 facets  | 721 facets  |  (overhead, live)         |
+-------------+-------------+---------------------------+
```

The brain panel is not a diagram. Every neuron is drawn at its real position in the FAFB volume, so when the pillar slides across the left eye's facets you watch the fly's left optic lobe light up in the panel beside it, the descending bars tip, and the track on the map bend. Underneath are the traces: what each eye saw, both descending populations, the steering signal, the bearing error read from the physics where the brain cannot reach it, and DNa02 failing to lateralise next to the population that succeeds. Then a leaderboard of which named cell types did the most spiking.

### What it measured

Measured on the devbox, 2026-09-11/12, Flyte project `physical-ai`. Pillar 12 mm away, 60° off the nose, so a fly that never turns misses by 10.4 mm.

**Run `r5d4w4l5x88pbxh2f8xd`: the fixation experiment and its controls:**

| | connectome | shuffled | blind | nobrain |
|---|---|---|---|---|
| reached the pillar | **yes, 1.35 s** | no | no | no |
| closest approach | **4.9 mm** | 11.1 mm | 10.6 mm | 11.1 mm |
| path walked | **11.0 mm** | 34.0 mm | 33.1 mm | 33.0 mm |
| final bearing error | **13°** | 57° | 173° | 158° |
| spikes | 1,845,984 | 2,993,010 | 2,151,331 | 0 |

Only the intact wiring with its eyes connected turns 60° and arrives. The other three walk a straight 33 mm line past the pillar and out of the arena. The shuffled brain spikes *more* than the intact one and gets nowhere, which is the cleanest way to say that the behaviour is in the wiring and not in the activity.

**Run `rmd7r9cd74nqnxsp2b6w`: feeding. Same drop, same approach, different taste:**

| | sugar | bitter | tasteless |
|---|---|---|---|
| peak proboscis drive | **1.00** | 0.06 | 0.06 |
| ticks spent feeding | **274** | 0 | 0 |
| closest approach | 3.9 mm | 0.5 mm | 6.0 mm |

Both drops are the same size, the same colour and the same shape; the fly cannot see the difference. Tarsal contact happens at the same tick in both runs. Then sugar drives 30 of the 110 feeding motor neurons to 2,467 Hz, the walking command collapses from 1.30/0.70 to 0.15/0.00, and the fly stops and feeds for four seconds. Bitter drives **zero** motor neurons and the fly walks straight over it.

Nothing in this repo encodes that sugar is food. The only difference between those two runs is which 129 or 65 gustatory receptor neurons the contact drove, and where their axons go.

**Run `rqj92428qrnzfp84mmjr`: the synaptic gain sweep, and the most uncomfortable result here:**

| `w_syn` | spikes | reached | closest |
|---|---|---|---|
| **0.275 mV** (published) | 1,845,984 | **yes, 1.35 s** | **4.9 mm** |
| 0.550 mV | 7,628,885 | no | 10.8 mm |
| 1.100 mV | 11,814,001 | no | 11.0 mm |

Turning synapses up wakes the brain: at 0.275 mV only 13 central-brain neurons fire, at 0.55 mV nearly 12,000 do. **And the fly gets worse.** Six times the spikes, and it can no longer find a pillar it was finding easily. The extra activity is not thought, it is noise swamping a small visual asymmetry. The published value is not just a fitted constant; it is close to the only setting where this model does anything coherent, and that is worth sitting with before anyone says a connectome simulation is "the animal".

**Run `rm48n24sn82dhv848g9b`: lesions:**

| cut | neurons | reached | time | path | bearing |
|---|---|---|---|---|---|
| nothing (intact) |: | yes | 1.35 s | 11.0 mm | 13° |
| L1 | 1,591 | yes | 1.35 s | 11.0 mm | 12° |
| L2 | 1,697 | yes | 1.35 s | 11.0 mm | 12° |
| Lai | 311 | yes | 1.35 s | 11.0 mm | 14° |
| **DNp28** | **2** | yes | **1.78 s** | **14.8 mm** | **26°** |

Removing 1,697 lamina monopolar cells, the entire L2 output of the retina, changes nothing measurable. Removing **two** neurons costs the fly a third of its speed and doubles its final bearing error. DNp28 is simply the descending neuron that fires hardest in this model (106.7 Hz), and the whole steering signal leans on it.

An earlier version of this experiment cut LC4, LPLC2 and DNa02: the looming detectors and the textbook steering pair: and got **bit-identical** results to intact, because at the published `w_syn` **those cells never fire at all**. A lesion of a silent neuron is a no-op. That failure is what sent me to measure who actually fires, in the next section.

### Who is actually doing this

Driving one eye at 150 Hz for 300 ms, and counting who responds:

| population | neurons | fired | total rate |
|---|---|---|---|
| sensory (driven) | 16,352 | 10,779 | 1,017,964 Hz |
| optic | 77,530 | 2,515 | 41,710 Hz |
| visual_projection | 8,038 | **17** | 597 Hz |
| central | 32,379 | **7** | 72 Hz |
| descending | 1,299 | **16** | 450 Hz |

899,000 of the 954,000 spikes in a run are photoreceptors. The signal reaches the lamina (L1, L2, L3, Lai), trickles into the medulla (Tm1), and then **essentially stops**. The entire behaviour in this demo rides on about 18 descending neurons:

```
DNp28(l) 106.7 Hz   DNp40(r) 63.3   DNp22(r) 63.3   DNpe017(r) 50.0
DNb06(r)  43.3      DNa16(l) 36.7   DNp102(r) 30.0  DNp19(r) 30.0
DNp18(r)  26.7      DNp20(r) 23.3   DNge107(r) 20.0 DNp19(l) 13.3  ...
```

Drive the **left** eye and the **right**-side cells dominate: the contralateral visual pathway, found by measurement rather than assumed. These are real named cells from the literature, and they are, as far as this model is concerned, the fly's steering wheel.

---

## 6b. Four more experiments, and the one measurement that explains three of them

The fixation demo above uses the **difference** between the two descending populations. These four ask what else this brain can be made to do, and between them they find the place where the fly's visual system stops working in this model, precisely enough to name the synapse.

### `swat`: a sphere flies at the fly

A 1.8 mm sphere closes from 18 mm at 22 mm/s and arrives about 0.7 s later. That is a far bigger stimulus than the fixation pillar: total darkening across the two eyes goes from 0.015 to 0.287, a **19-fold swing**, against the pillar's 0.075. Three embodied conditions run, and the middle one is the control that matters:

| | what it is |
|---|---|
| `loom` | the sphere flies in |
| `recede` | the same path walked backwards. Identical retinal sizes, opposite time order |
| `static` | the sphere sits at its launch point and never moves |

A looming object is symmetric, so it is nearly invisible in the left-right difference and shows up only in the **sum**, a channel this repo did not previously read. Measured over 90 ticks:

| condition | summed descending rate | ratio | peak startle |
|---|---|---|---|
| loom | 427 → 1,067 Hz | x2.50 | 0.68 |
| recede | 744 → 1,067 Hz | x1.43 | 0.21 |
| static | 408 → 800 Hz | x1.96 | 0.40 |

Read that table honestly and it is already bad news for the looming story. The **static** run, with nothing moving at all, produces a x1.96 swing on its own. The looming run's x2.50 is barely above a control where the sphere just sits there, so most of what the common-mode detector is picking up is the brain's own fluctuation. The embodied numbers cannot settle this, which is exactly why the next task exists.

Then `looming_probe` runs the same stimulus with no body at all, and that is where the honest part is. Two results:

**The escape circuit never fires.** LPLC2 is the fly's canonical looming detector and DNp01 is the giant fibre, the largest axon in the animal. Both are silent at every synaptic gain tried. The stage-by-stage autopsy says exactly where the signal stops:

```
photoreceptors R1-6/R7/R8      10,050 / 10,582     fires
lamina monopolar L1/L2/L3         887 / 4,730      fires
lamina L5 + amacrine Lai          137 / 1,887      fires
medulla Mi1/Tm1/Tm3                73 / 4,895      barely
T4, the ON motion detector          0 / 6,241      DEAD
T5, the OFF motion detector         0 / 6,005      DEAD
lobula plate tangential HS/VS       0 / 24         DEAD
looming LC4 + LPLC2                 0 / 314        DEAD
giant fibre DNp01                   0 / 2          DEAD
all descending neurons             13 / 1,299
```

**And the brain cannot tell an approach from a retreat.** Replay the captured stimulus forwards and backwards into a fresh brain and correlate the two. That number means nothing on its own, because with 13 active descending cells the trace is mostly Poisson noise: measured without smoothing it reads r = +0.37, which looks like selectivity and is not. So the forward stimulus is also run **twice with different seeds**, to measure how well the brain agrees with *itself*. At the published gain:

```
w_syn 0.275   reversal r=+0.929   ceiling r=+0.887   tracks its own stimulus +0.924 / +0.941
w_syn 0.550   reversal r=-0.424   ceiling r=+0.955   tracks its own stimulus +0.958 / -0.371
```

At the published gain the reversal is *above* the ceiling: a receding sphere matches the looming response as well as a repeat of the looming stimulus does. This is a light meter, not an escape circuit.

The second row is the trap, and the report refuses to read it. At w_syn 0.55 the reversal is r = -0.424, which looks like strong direction selectivity. It is not: the receding run tracks **its own** stimulus at r = -0.371 while the looming one manages +0.958. The network has ignited and flat-lined, and a flat trace correlated against a ramp gives whatever it gives. So the verdict is gated on both directions tracking their own stimulus before the reversal number is allowed to mean anything, and at this gain the report refuses to call it either way.

### The reason, in one row of a table

Why is T4 dead? Its dominant excitatory driver is Mi1, and Mi1's input budget in this dataset is:

```
into Mi1        signed synapses
  L1              -122,574        <- glutamatergic, so inhibitory in this model
  Pm08             -45,585
  L5               +44,127
  Dm1              -22,360
  L3               +19,323
```

L1 is the first cell of the ON pathway and here it **clamps the medulla off**. Downstream of the lamina, nothing can be driven, so T4, T5, the HS/VS tangential cells and LPLC2 are all unreachable no matter what the eye does.

The root cause is one synapse earlier, and it is a data problem rather than a modelling choice. Real photoreceptors are **histaminergic** and *inhibit* L1; light hyperpolarises the downstream cell. Histamine is not one of the six transmitters the FlyWire classifier predicts, so R1-6 comes back labelled acetylcholine (5,343 of them), and the very first synapse of the visual system has the wrong sign. Flipping all 56,386 photoreceptor output synapses to inhibitory and adding tonic lamina drive was tried, and it does not rescue the pathway either: the optic lobe is largely **non-spiking** in the real animal, and an integrate-and-fire model has no way to represent a graded signal. This is exactly the gap `flyvis` exists to fill.

### `drum`: the oldest experiment in fly vision

A tethered fly in a rotating striped drum turns with the drum. flygym's `TetheredWorld` holds the body rigidly while the legs keep walking, so the fly's intention is readable only in the descending command, which is the point. Five runs in two groups:

| condition | mean steering command | its SD |
|---|---|---|
| 12 bars, counter-clockwise | +0.0016 | 0.038 |
| 12 bars, clockwise | +0.0010 | 0.038 |
| 12 bars, still | -0.0119 | 0.029 |
| **1 bar, counter-clockwise** | **+0.0980** | 0.051 |
| 1 bar, clockwise | +0.0059 | 0.040 |

The full drum is a **motion** stimulus: rotating it barely changes how much of each eye is dark. Its two directions differ by 0.0006, about sixty times *smaller* than the noise within a single run. There is no optomotor response.

A single bar is a **position** stimulus, and the same rig with the same brain separates its two sweep directions by 0.092, **150 times** the rotating drum's separation. So the rig works and the fly is steerable; motion specifically is what this model cannot see. Two independent failures stack here and the demo measures both: this repo's retina reports one number per eye and is motion-blind by construction, and the connectome's motion detectors are dead anyway.

### `compass`: a ring attractor, recovered and then driven

The one unambiguous success of the four, and the only one that needs no body at all. It runs in **40 seconds**.

The central complex holds the insect head-direction system: EPG neurons tile the ellipsoid body into wedges and a bump of activity sits on that ring tracking which way the animal faces. The problem is that FlyWire labels all 47 EPG cells `EPG` with no wedge number, so **the order around the ring is simply not in the metadata**. It has to be recovered from connectivity.

Delta7 cells tile the protocerebral bridge, so two EPG cells that neighbour each other on the ring share Delta7 partners. Build the cosine similarity between EPG cells over their Delta7 connections, take the two leading non-trivial eigenvectors of the normalised Laplacian, and the cells land on a circle. That is a claim with a check attached: in the recovered order, similarity must fall off with circular distance.

```
EPG x all partners       eigenvalues 0.885/0.905/0.912   similarity vs distance -0.47
EPG x central complex    eigenvalues 0.888/0.904/0.914   similarity vs distance -0.50
EPG x Delta7             eigenvalues 0.588/0.683/0.875   similarity vs distance -0.78   <- used
```

Then drive six adjacent cells on the recovered ring at eight positions around it, and see where the response lands:

```
                       the real wiring          shuffled wiring
Delta7    slope -1, residual  5.2 deg RMS, r -0.976      silent, 0 of 42 cells
PFL3      slope +1, residual 13.0 deg RMS, r +0.989      silent, 0 of 24 cells
```

**Five degrees.** Move the bump, and the inhibitory ring around it moves by the same amount. PFL3 is the population that turns the compass into a steering command, the last stage before the descending neurons. The shuffled control does not merely degrade, it goes completely silent.

The slope sign is arbitrary and is not part of the claim: the two rings are embedded independently and each has its own handedness. The residual is the claim.

### `duet`: two flies, two brains, one arena

Two independent copies of the connectome in one physics simulation, about 2.6 GB, roughly half the speed of a single fly. It buys two things.

A **simultaneous control**: a real connectome and a shuffled one in the same arena at the same instant, so they cannot differ by lighting, spawn or luck. And **interference**: the flies are solid, so they collide, block each other and occlude each other's view of the pillar, none of it scripted.

Measured, 260 ticks each:

| condition | fly | closest (mm) | net approach (mm) | ticks crowded |
|---|---|---|---|---|
| connectome vs connectome | a | 5.0 | +7.9 | 45 |
| connectome vs connectome | b | 4.4 | +8.6 | 45 |
| **connectome vs shuffled** | **a, connectome** | **0.2** | **+12.8** | 0 |
| connectome vs shuffled | b, shuffled | 6.9 | +6.0 | 0 |

Two real brains both solve it and spend 45 ticks in each other's way, passing within 2.4 mm. Against a shuffled brain in the same arena at the same instant, the real one closes 12.8 mm and ends essentially on the pillar while the shuffle closes 6.0 mm and stops 6.9 mm short, and they never touch, so that gap is not contact and it is not luck.

What they are not is social, and that was measured before any of it was built. One fly looking at another across an empty arena:

```
separation                 4 mm   6 mm   9 mm   12 mm
facets darkened, absolute     0      0      0      0    of 721
facets darkened, contrast     4      3      2      1    of 721
```

By the absolute test this repo's retina uses everywhere else, **a fly is invisible at every distance**: an amber body against a bright floor never crosses the darkness threshold a black pillar crosses easily. By a contrast test it is visible and tiny, three or four facets against the pillar's fifty-four. That is an order of magnitude below a stimulus already near this model's noise floor, so these two will not court or chase. `--marker_radius` gives each fly a dark dorsal marker, the way a real experimenter paints one, so the claim can be checked rather than argued.

---

## 6c. A reaction you can actually see, and an attempt to train it

Two more, prompted by the honest complaint that the visual experiments produce very
little visible reaction. They do, and there is a reason: vision is the weakest channel
in this model. The other two channels are not weak at all.

### `poke`: touch is 25 times louder than sight

Every visual experiment here fights for signal. A pillar darkens 54 of 721 facets, the
descending populations move by less than their own per-tick noise, and the fixation loop
only works because 150 ms of smoothing is applied. Touch is nothing like that. Measured
open loop, 300 ms of one-sided mechanosensory drive:

```
poke left  @ 25 Hz -> descending  left 1,383  right   750    imbalance -0.297
poke right @ 25 Hz -> descending  left   373  right 1,147    imbalance +0.509
poke left  @ 75 Hz -> descending  left 4,893  right 2,570    imbalance -0.311
poke right @ 75 Hz -> descending  left 1,403  right 3,703    imbalance +0.450
```

The sign flips cleanly with the side at every drive level, and it does so in a single
tick with no smoothing. So `poke` puts that in a walking fly: it strolls in an empty
arena with nothing to look at, and every 35 ticks an experimenter deflects the bristles
on one side of its head, alternating left and right. The population is 1,363 cells on
the left and 1,293 on the right, mostly `BM_InOm`, the bristles between the ommatidia,
plus Johnston's organ in the antenna.

| wiring | poked left | poked right | not poked | apart |
|---|---|---|---|---|
| **connectome** | **-0.154** | **+0.198** | -0.006 | **4.28 SD** |
| shuffled | -0.061 | -0.020 | -0.125 | 0.39 SD |

The two poke directions land on opposite sides of a quiet baseline that is itself
essentially zero, and they are four standard deviations apart. The shuffled control does
not separate at all, at 0.39 SD, despite having the same 2,656 mechanosensory neurons
driven just as hard.
Heading is reported too and is deliberately the weaker number: a walking fly wanders
about 10 degrees per 180 ms on its own, so the behaviour is noisier than the command
driving it. Measuring the swerve and not the command is how the first version of this
experiment produced a shuffled control that looked identical to the real one.

### `teach`: the one place a synapse changes

Every other task runs a fixed brain. This one modifies it, using the rule the animal
uses. Dopamine arriving at a mushroom body compartment while a Kenyon cell is active
**depresses** that cell's synapse onto the output neuron there, so the odour stops
driving that channel. It is not Hebbian and there is no gradient anywhere. Two of the
three factors are read from the connectome rather than chosen: which Kenyon cells the
odour activates, and which of the 96 MBONs the driven dopaminergic cells actually reach.

The whole apparatus is in FlyWire: 5,177 Kenyon cells, 96 MBONs across 62,261
connections, 2 APL neurons for feedback inhibition, and the teaching signal split the
way the literature describes it, 16 PPL1 cells for punishment onto 85 MBONs and 307 PAM
cells for reward onto 68.

**The mechanism works.** Depression drops the trained odour's MBON output by about half,
and the identical procedure on shuffled wiring changes nothing at all (1 Kenyon cell
active, 0 edges depressed, 0.0%).

**The memory is not odour-specific**, and walking down the pathway says exactly where
that goes wrong. A channel counts as firing only if it drives at least 2% of the stage,
which matters: counting the near-silent ones as firing reports an antennal lobe overlap
of 0.171 and makes the problem disappear.

```
stage                        active per odour        odours firing   overlap
ORN (olfactory receptors)    463 /  97 /  78 / 411       4 of 4       0.107   <- identity is here
ALLN (lateral, inhibitory)   360 /   0 /   0 / 359       2 of 4       0.997
ALPN (projection neurons)    543 /   5 /   4 / 545       2 of 4       0.993   <- and gone here
Kenyon cells               1,503 /   0 /   0 / 1,479     2 of 4       0.950
MBON                          43 /   0 /   0 /    39     2 of 4       0.864
```

Odour identity is crisp at the receptors: all four channels fire and they overlap by
0.107. One synapse later it is gone. Two of the four odours fail to cross into the
antennal lobe at all, and the two that do each drive **543 and 545 of the 685 projection
neurons**, overlapping at 0.993. In a living fly one glomerulus drives its own handful of
projection neurons and lateral inhibition keeps the rest quiet. Here every odour that
gets through becomes the same generic "an odour is present" signal, recruits the same
~1,500 Kenyon cells, and punishing one punishes them all.

Measured end to end: the trained odour's MBON output fell **48.7%** and the odour the fly
was never punished for fell **46.8%**. A specificity of **1.9 points out of 48**. The
shuffled control depressed 0 edges and changed nothing, so the mechanism is real; it is
the odour code that is not.

That is the same failure as the dead motion pathway, from the same cause: nothing in this
model does gain control, so any sufficient input ignites a whole region.

Four fixes were tried and are recorded in `learn.py` so nobody spends the afternoon
again: raising the Kenyon cell threshold (code goes from 65% of cells to 0.4%, overlap
only falls from 0.99 to 0.88), deleting all 293,762 Kenyon-to-Kenyon edges (0.956),
lowering the drive (below the 20-to-25 Hz ignition cliff odours do not stop overlapping,
they stop firing), and both together.

There is also a control this model cannot run. An unpaired trial, dopamine with no odour,
should produce no learning; here it produces 46.6%, because PPL1 reaches the Kenyon cells
over **10,720 excitatory edges** and dopamine in this model is a `+1` sign like any other
transmitter. The teaching signal is itself a sensory drive. That is the flattening of
neuromodulation from section 2b, showing up as a broken experiment.

---

## 6d. What it is releasing, not just which cells fired

Every other readout here counts spikes. A raster treats a GABA spike and a dopamine spike
as the same event, and in a head they are not remotely the same event. So every embodied
run now also carries a **chemistry** trace, and it costs one dot product per tick.

Each neuron in the annotation carries a predicted transmitter, and each has an outgoing
synapse budget: the total number of synapses it makes onto everything downstream. One
spike therefore delivers that many synaptic events of that one transmitter.

| transmitter | neurons | outgoing synapses | per spike |
|---|---|---|---|
| acetylcholine | 86,025 | 30,460,673 | 354 |
| GABA | 19,147 | 12,600,845 | **658** |
| glutamate | 24,858 | 9,471,876 | 381 |
| dopamine | 5,905 | 1,384,531 | 234 |
| serotonin | 2,201 | 436,206 | 198 |
| octopamine | **210** | 137,839 | **656** |

Two things in that table are worth stopping on. A GABAergic spike delivers nearly twice
the synaptic events of a cholinergic one, so **a raster that looks balanced is not**. And
the 210 octopaminergic neurons are individually the second-biggest broadcasters in the
brain, which is exactly what a modulatory system should look like.

Measured over a walking fly being poked, in thousands of synaptic events per tick:

```
acetylcholine   mean 560.4k   peak 1,202.3k
GABA            mean 393.4k   peak 1,022.2k
glutamate       mean 162.7k   peak   293.4k
serotonin       mean  11.4k   peak    29.8k
dopamine        mean   8.4k   peak    29.1k
octopamine      mean   6.8k   peak    43.2k

excitation / inhibition ratio   mean 1.41   peak 2.63
```

Because Dale's law holds exactly in this dataset, the excitation/inhibition split is
exact rather than estimated: every neuron is purely one or the other, so the E/I ratio is
a real quantity and not a fit.

The `teach` task gets a second version of this, and it is the one case where training
changes the chemistry directly: Kenyon cells are cholinergic, so depressing their
synapses onto the output neurons by 90% means the same cells firing now deliver fewer
cholinergic events. Every other transmitter should barely move, and a large swing in the
GABA or dopamine row would mean the edit reached further than intended.

**The caveat is the point, though.** This is release, not effect. The model collapses all
six transmitters onto a `+1/-1` sign, so dopamine in the simulation is a fast excitatory
synapse and nothing more. The dopamine trace is a real count of what a real fly would be
releasing, drawn next to a simulation that does not implement what that release does.
That gap is section 2b, and `teach` walks straight into it: the unpaired control fails
because dopamine, being just a `+1`, drives Kenyon cells directly over 10,720 edges.

---

## 6e. Putting the fly on a robot

The sharpest test of the claim the whole repo rests on. `bridge.py` argues that what a
connectome sends to a body is a **descending command**, roughly "turn left", and that a
central pattern generator downstream works out the legs. If that is actually true, the
same 1,300 neurons should drive a body with the wrong number of legs, the wrong mass and
a completely different gait, with nothing about the brain changed.

So the fly is bolted to the back of a four-legged robot about twenty-five times its
length, its compound eyes look out from up there, and the same pillar-fixation task runs.

```bash
flyte run pipeline.py ride --ticks 750
```

Nothing about the brain, the retina, the bridge or the calibration differs from the
walking demo. The entire interface between the two animals is four lines:

```python
speed = RIDE_BASE + 0.5 * ((left + right) - 1.0)     # the mean is how fast
turn  = RIDE_TURN_SIGN * RIDE_TURN * (left - right)  # the difference is which way
```

### The robot, and three things that had to be measured

It is written from scratch in MJCF rather than pulled from a model zoo, because the pod
has no network at run time and because everything must be in flygym's units, which are
**millimetres with gravity at -9810 mm/s²**. A 60 mm trunk next to a 2.5 mm fly is about
a mouse carrying a housefly.

**The servo gains are three orders of magnitude larger than they look.** Holding this
robot up takes roughly 3.5e5 of joint torque per leg (8.8 g, 9810 mm/s², a 16 mm lever,
four legs). Gains that would be sane in SI collapse it into the floor on the first step.

**Differential stride does not steer it.** The obvious gait, a sine on the hip and a
raised cosine on the knee, walked but could not be controlled: speed either fell with
stride amplitude or ignored it entirely, and at high knee lift the robot was swimming on
its knee pump with the hips contributing nothing.

```
knee lift   drive 0.15   0.25    0.35    0.45     forward mm in 3 s
  0.50           40.1   36.5   -15.0   -25.4
  0.80           65.9   61.9    -0.1   -26.4
  1.10           52.0   52.7    53.2    53.4
```

Replacing it with an explicit stance/swing split fixed the walking but not the steering,
which came out grossly asymmetric: **+53.6 degrees one way and -7.0 the other** from the
same size of command, so the fly could only turn left. The fix is a dedicated yaw joint
per leg that drags the planted feet sideways during stance, which is symmetric by
construction:

```
yaw sweep   turn -1.0    -0.5     0.0    +0.5    +1.0     degrees per 3 s
  0.15         -142.0   -78.8    -1.0   +80.2  +136.4
  0.25         -225.4  -122.0    -1.0  +117.6  +222.3
  0.40         -362.6  -186.4    -1.0  +179.4  +388.2
```

Symmetric, monotonic, and within a degree of straight at zero command. That last column
matters as much as the turning: any drift in the gait would show up in the result as if
the brain had produced it.

**The two bodies steer in opposite directions from the same command.** A fly turns toward
the side whose legs push *less*; this robot turns toward the side that strides *further*.
So `RIDE_TURN_SIGN` is -1, and it was measured rather than assumed, exactly like the
visual turn sign in section 4. Getting it wrong does not look like a bug, it looks like a
fly steering away from the thing it is looking at.

### Does it work

Measured, a pillar 150 mm away and 40 degrees off the nose, 750 ticks (11.25 s of robot
time) per mode:

| mode | closest approach | net approach | final bearing | reached |
|---|---|---|---|---|
| **connectome** | **45 mm** | **+103 mm** | 47° | **YES, in 7.67 s** |
| shuffled | 91 mm | +57 mm | 72° | no |
| nobrain | 86 mm | +62 mm | 138° | no |

**The fly drove the robot to the pillar.** The shuffled brain, with the same neurons, the
same out-degrees and the same weight distribution, got barely further than having no
brain at all. So the descending command really does carry across to a body with the wrong
number of legs, the wrong mass and a gait the animal never had.

It is slower than the fly on its own legs, and that is worth being precise about rather
than hiding: at 450 ticks the same run ended 11 mm short of arrival (56 mm, against 95 mm
for the shuffle and 86 mm for no brain). The ordering was already right at that length;
the robot simply needed another four seconds to finish closing, because its turn rate at
a typical descending imbalance is a few degrees per second and it keeps walking while it
corrects.

### What the rider can see

The robot is put in MuJoCo geom group 2, which the eye renderer ignores, so the fly
cannot see the vehicle it is sitting on. Without that it spends the run watching legs
swing past and reports 13% of its retina darkened by the robot. With it, the view from
the saddle lateralises the same way the ground-level fly's does, at about a third of the
amplitude:

```
pillar dead ahead      L 0.0319   R 0.0305    L-R +0.0014
pillar 120 mm left     L 0.0153   R 0.0000    L-R +0.0153
pillar 120 mm right    L 0.0000   R 0.0139    L-R -0.0139
```

---

## 7. Things that cost real debugging time on this box

**A correlation without a noise ceiling is not a result.** The time-reversal test compares the brain's response to a looming stimulus against its response to the same stimulus reversed. Unsmoothed, it read r = +0.37, which looks like the two are different and therefore like motion sensitivity. They are not different; with only 13 descending neurons firing, both traces are mostly Poisson noise. Running the *same* stimulus twice with different seeds gives a ceiling of r = +0.887, and the reversal scores +0.929 against it. Any comparison of two noisy responses needs the same-stimulus repeat before it means anything.

**A saturated network looks selective.** At w_syn 0.55 the same test reads r = -0.42, which looks like a strong difference. It is the network igniting: the receding run tracks *its own* stimulus at r = -0.37 while the looming one tracks its own at +0.96, so a flat trace is being correlated against a ramp.

The gate that catches it has to be **"does each direction track its own stimulus"**, and not a saturation fraction. Saturation was the first thing tried and it is too blunt to gate on: measured as the fraction of the run within 20% of the peak, an ordinary ramping response at the published gain scores 47% against the maximum and 53% against the 95th percentile, which is right next to the 79% that the genuinely flat-lined run scores. Both forms of the statistic put a healthy response and a dead one within a factor of 1.5 of each other. It is reported in the panel as context for *why* a run failed the gate, never as the gate itself.

**Spectral embeddings are only defined up to sign, and the reference population matters more.** Recovering the compass ring gave a 13-degree tracking residual on the devbox and 70 degrees in a pod from identical inputs. Two separate causes: LAPACK does not promise a sign for an eigenvector (fixed by pinning each vector's largest-magnitude entry positive), and PFL3 was being embedded against EPG when it should have been Delta7. The second was worth 13 degrees against 70. Both are why every ring in the report prints its own quality score.

**flygym 2.1.0 cannot have two flies with ground-contact sensors on.** The sensors are named per leg and not per fly, so the second `add_fly` dies with `repeated name 'ground_contact_lf_leg' in sensor`. Pass `add_ground_contact_sensors=False`; the walking controller reads contact forces straight out of `mj_data` and never touches those sensors.

**A fly is not a visual stimulus.** Two flies in an arena see nothing of each other: 0 of 721 ommatidia darkened at every distance, because the retina asks whether a facet is *dark* and an amber fly on a bright floor never is. A contrast test finds 3 facets at 6 mm. Worth measuring before building a demo that depends on it.

**Moving a landmark at runtime needs no mocap body.** A geom on the worldbody has its pose in `mj_model.geom_pos`, and MuJoCo recomputes `geom_xpos` from it every step, so writing that array is the whole of it. No free joint, no extra degrees of freedom, no constraint solver involvement.

**Time-to-contact has two moving ends.** Computing it from the approaching object's progress along its own path reports the wrong number whenever the fly is also walking, which is most of the run. Use the actual closing rate.

**`signal only works in main thread of the main interpreter`.** `brian2/__init__.py` installs a SIGINT handler at import, and CPython only allows that from the main thread. A Flyte task that offloads its blocking work with `asyncio.to_thread` imports brian2 *in the worker*, and the import itself dies. Fix is `brain.preimport()` on the main thread first; Python caches the module and the later import inside the thread is a no-op.

**A silent brain looks exactly like a working one.** Covered in 4.3. The run completes, the video plays, the fly walks, and the descending traces are flat zero for the whole run. If the loop does something implausible, plot the middle of it before touching a gain.

**A single named neuron is not a signal.** Covered in 4.2. Everyone's first design is DNa02.

**The eyes cost 2 ms, not 126 ms.** An early benchmark timed 20 `get_ommatidia_readouts` calls at 126 ms each. The first call lazily constructs the Retina and the eye renderer, which takes 2.5 s, and 2.5/20 = 125. Warm it before timing anything.

**Flyte bundles only top-level imports.** A sibling module imported lazily inside a function body is not seen by `flyte run`, the pod comes up without it, and the run dies with `ModuleNotFoundError` minutes in. Every sibling here is imported at the top of `pipeline.py` on purpose.

**g++ is not optional in the image.** Brian2's default codegen target is Cython, which compiles generated C++ *at runtime inside the pod*. Without a compiler it silently falls back to its numpy target, roughly 30x slower on a network this sparse, turning a 30-second run into a quarter of an hour.

**MuJoCo's EGL destructor screams on the way out.** `Renderer.__del__` calls into EGL during interpreter teardown, after the context is gone, and floods the task log with `EGLError` tracebacks that look fatal and are cosmetic. `FlyWorld.close()` swallows them.

**Scoring the wrong thing.** The first metric set had `final_distance` and "bearing error over the last third". A fly walking 16 mm/s crosses a 12 mm arena in under a second and keeps going, so those measure the *departure*. In one run the fly closed from 9.5 mm to 3.1 mm: touching a 3 mm post: and scored as a failure. Runs now end on arrival, like real fixation experiments.

**The control can pass by accident.** Covered at the end of section 6. If a straight line nearly solves the task, a null model will nearly solve it too, and your result is geometry.

---

## 8. A guided tour of the code

If you are presenting this, go in this order.

1. **`connectome.py:load()`**: three files, one join, 138,639 neurons. Show `describe()`: 15,091,983 connected pairs, 54,492,922 synapses, 9.06M excitatory pairs against 6.03M inhibitory. Then `select(cell_type="DNa02")` returning exactly two rows, one per side. The wiring diagram is a dataframe.

2. **`brain.py:_build()`**: fifteen lines from that dataframe to a spiking network. Stop on `synapses.w = conn.weight * w_syn` and say out loud that this is a count of synapses in a real animal and the only free parameter is the scale.

3. **`bridge.py`**, top of file: the two measured tables. This is where the demo earns its keep: the famous steering cell doesn't work, the population does, and both facts are measurements printed by code in this repo.

4. **`run.py:closed_loop()`**: the five-line loop. Then the four modes, and why `shuffled` is the one that matters.

5. **The report**: the cockpit clip, then scroll to the DNa02 trace pinned flat next to the population that works.

6. **`connectome.py:shuffled()`**: the null model in four lines, and what it does and does not preserve (out-degree and the weight distribution yes; in-degree no).

### What the four new experiments add to the tour

7. **`compass.embed_ring`**: the ring order is not in the metadata and comes out of the connectivity, with a pass/fail number attached. Show the eigenvalues and the -0.78. Then `bump_response` at eight positions and the 5.2-degree residual next to a shuffled control that is completely silent.

8. **`looming_probe`'s autopsy table**: read it top to bottom and stop at the two lines where the count goes to zero. Then `into Mi1` in section 6b, and the -122,574. That is the whole explanation for why this fly cannot see motion, and it is one row of a table.

9. **The noise ceiling**: the time-reversal test is the one place in this repo where a plausible-looking number (r = +0.37) was wrong and running the same stimulus twice is what caught it.

### Where this goes next

**BANC**, the adult brain-and-nerve-cord connectome, is the upgrade that would change the claim rather than extend it. Every motor neuron in a BRAIN connectome is a feeding motor neuron, which is why `bridge.py` hands a two-number descending command to a hand-written walking controller instead of driving legs. BANC contains the leg motor neurons, so the descending signal could drive the body directly and the central-pattern-generator stand-in could go.

The mechanosensory channel is the strongest lateralised input in the whole model, by a factor of 25 over vision, and it is already wired up in `SENSORY_GROUPS`: touch one antenna and the fly should veer. The sugar pathway (129 gustatory receptor neurons, `cell_sub_class="sugar/water"`) is the one Shiu et al. built their paper on and the one Eon used for feeding. And `flyvis` (Lappalainen et al. 2024) is a connectome-constrained, *trained* model of the visual system that would replace this repo's crude "count the dark facets" retina with the real optic-lobe computation.

### Sources

- FlyWire connectome: [Dorkenwald et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07558-y)
- Whole-brain annotation and cell typing: [Schlegel et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07686-5) · [data](https://github.com/flyconnectome/flywire_annotations)
- The LIF brain model: [Shiu et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07763-9) · [code](https://github.com/philshiu/Drosophila_brain_model)
- NeuroMechFly v2 / flygym: [Wang-Chen et al., *Nature Methods* 2024](https://www.nature.com/articles/s41592-024-02497-y) · [docs](https://neuromechfly.org)
- flybody: [Vaxenburg et al., *Nature* 2025](https://www.nature.com/articles/s41586-025-09029-4) · [code](https://github.com/TuragaLab/flybody)
- Connectome-constrained visual models: [Lappalainen et al., *Nature* 2024](https://www.nature.com/articles/s41586-024-07939-3) · [flyvis](https://github.com/TuragaLab/flyvis)
- The embodied emulation this demo chases: [Eon Systems, March 2026](https://eon.systems/updates/embodied-brain-emulation)
