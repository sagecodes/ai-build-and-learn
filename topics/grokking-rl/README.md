# Grokking Deep Reinforcement Learning: one learning project per chapter

These are my own learning projects, inspired by the concepts in each chapter of *Grokking Deep Reinforcement Learning* by Miguel Morales (Manning). Building something I can run, watch and poke at is how I make an idea stick, so each chapter's core idea gets turned into a small project here.

**This is not a replacement for the book, not even close.** The book is where the real explanations live: the intuition, the math worked through step by step, and the why behind every algorithm. These projects only make sense next to it. If any of this looks interesting, go buy it:

👉 **[Grokking Deep Reinforcement Learning on Manning](https://www.manning.com/books/grokking-deep-reinforcement-learning)**, and use code **AIBookClub** for 45% off.

Reinforcement learning (RL) is learning to act from trial, error and reward, rather than from labelled examples. Each chapter that has an algorithm in it gets a folder with a small, self-contained project and a README covering the idea I took from the chapter and how the code implements it.

Every acronym and symbol used here is spelled out in the [glossary](#glossary) below.

The early chapters run on the same little worlds the book uses (bandit walks, slippery walks, frozen lake), written from scratch as plain transition tables so nothing is hidden behind a library. Every example runs as a Flyte task on the devbox, and its report draws the worlds as grids, replays episodes step by step next to what the agent observes, and animates the learning (value iteration spreading out from the goal, a greedy policy forming over training).

## Chapter map

| Ch | Book chapter | Example | Status |
|---|---|---|---|
| 1 | Introduction to deep RL | concepts only | n/a |
| 2 | Mathematical foundations of RL | [ch02_mdps](ch02_mdps): Markov decision processes (MDPs) as tables, slip, discounting | ✅ |
| 3 | Balancing immediate and long-term goals | [ch03_planning](ch03_planning): policy evaluation, policy iteration, value iteration | ✅ |
| 4 | Balancing gathering and using information | [ch04_bandits](ch04_bandits): 8 exploration strategies (ε-greedy, UCB, Thompson sampling, ...) on the 10-armed bandit | ✅ |
| 5 | Evaluating agents' behaviours | [ch05_prediction](ch05_prediction): Monte Carlo (MC), temporal difference (TD(0), n-step TD, TD(λ)) | ✅ |
| 6 | Improving agents' behaviours | [ch06_control](ch06_control): MC control, SARSA, Q-learning, Double Q | ✅ |
| 7 | Achieving goals more effectively and efficiently | [ch07_traces_dyna](ch07_traces_dyna): SARSA(λ), Q(λ), Dyna-Q, trajectory sampling | ✅ |
| 8 | Introduction to value-based deep RL | [ch08_nfq](ch08_nfq): NFQ (neural fitted Q iteration) vs a binned table on **CartPole**, with the network lighting up as it plays | ✅ |
| 9 | More stable value-based methods | [ch09_dqn](ch09_dqn): DQN (deep Q-network), Double DQN on **LunarLander**, one idea added at a time | ✅ |
| 10 | Sample-efficient value-based methods | [ch10_dueling_per](ch10_dueling_per): Dueling DDQN (double DQN), PER (prioritized experience replay) on LunarLander, 16 seeds each | ✅ |
| 11 | Policy-gradient and actor-critic methods | [ch11_policy_gradient](ch11_policy_gradient): REINFORCE, VPG (vanilla policy gradient), A2C (advantage actor-critic), GAE (generalized advantage estimation) on LunarLander, 16 seeds, plus a λ dial | ✅ |
| 12 | Advanced actor-critic methods | [ch12_actor_critic](ch12_actor_critic): DDPG, TD3, SAC, PPO on **Pendulum** (torque as a live dial), then SAC teaching a **MuJoCo HalfCheetah** to run, on video | ✅ |
| 12+ | Bonus, beyond the book | [ch12_bonus_humanoid](ch12_bonus_humanoid): chapter 12's PPO scaled to 4,096 GPU simulators teaches a **Unitree G1 humanoid** to walk, then a reward tweak changes its gait (NVIDIA GPU only) | ✅ |
| 13 | Toward artificial general intelligence | concepts only | n/a |

From chapter 8 on, the book's concepts move into worlds worth watching, well beyond the book's own examples: gymnasium's CartPole, LunarLander and Pendulum, then a MuJoCo HalfCheetah. The tables become neural networks (torch), and the reports show the networks themselves: activations lighting up as the agent plays, and what the network has learned drawn as a picture. This project ends on a bonus: chapter 12's PPO at robot scale, teaching a humanoid to walk with [rl-mujoco](../rl-mujoco)'s GPU training. For more RL in large sims on this box, see [isaac-sim](../isaac-sim) and [dreamerv3](../dreamerv3).

## Glossary

**Acronyms**

| Term | Stands for | Meaning |
|---|---|---|
| RL | reinforcement learning | learning to act by trial and error, guided by reward |
| MDP | Markov decision process | states, actions, transition probabilities, rewards, discount. "Markov": what happens next depends only on the current state and action, not on the history ([ch 2](ch02_mdps)) |
| DP | dynamic programming | solving an MDP exactly using its transition table: policy/value iteration ([ch 3](ch03_planning)) |
| GPI | generalized policy iteration | the evaluate-then-improve loop behind every control method ([ch 6](ch06_control)) |
| BW, BSW | bandit walk, bandit slippery walk | 3-cell corridors from chapter 2 |
| SWF, SWS | slippery walk five, slippery walk seven | corridors with 5 or 7 inner cells where moves can slip |
| RW | random walk | chapter 5's corridor where a coin flip decides every move |
| FL4, FL8 | frozen lake 4x4, frozen lake 8x8 | slippery grid worlds with holes and one goal |
| UCB | upper confidence bound | bandit strategy: pick the arm whose optimistic estimate is highest ([ch 4](ch04_bandits)) |
| MC | Monte Carlo | learn from complete episodes, using the actual return ([ch 5](ch05_prediction)) |
| TD | temporal difference | learn after every step, from the difference between consecutive estimates ([ch 5](ch05_prediction)) |
| TD(0), TD(λ) | one-step TD, TD with eligibility traces | the 0 and λ are the trace-decay parameter; 0 means "look one step ahead" |
| RMS | root mean square | the error measure in chapter 5: `sqrt(mean((V - V_true)²))` |
| SARSA | State, Action, Reward, State, Action | the five things its update uses: (s, a, r, s', a') ([ch 6](ch06_control)) |
| Q-learning | Q = "quality" of an action | learns Q(s, a), the value of taking action a in state s |
| Dyna-Q | Dyna (Sutton's architecture name) + Q-learning | Q-learning plus planning on a learned model ([ch 7](ch07_traces_dyna)) |
| NFQ | neural fitted Q iteration | Q-learning with a neural network refit on batches of experience (ch 8) |
| DQN, DDQN | deep Q-network, double deep Q-network | NFQ plus a replay buffer and a target network; double = Double Q's fix applied to DQN (ch 9) |
| PER | prioritized experience replay | replay surprising transitions more often (ch 10) |
| REINFORCE | (a name, not really an acronym) | the original Monte Carlo policy-gradient algorithm (ch 11) |
| VPG | vanilla policy gradient | REINFORCE plus a learned value baseline (ch 11) |
| A2C, A3C | advantage actor-critic, asynchronous advantage actor-critic | a policy (actor) trained with a value function (critic) (ch 11) |
| GAE | generalized advantage estimation | TD(λ)'s idea applied to the advantage (ch 11) |
| DDPG | deep deterministic policy gradient | actor-critic for continuous actions (ch 12) |
| TD3 | twin delayed DDPG | DDPG with two critics and delayed actor updates (ch 12) |
| SAC | soft actor-critic | actor-critic that also rewards keeping the policy random (entropy) (ch 12) |
| PPO | proximal policy optimization | policy gradient with a clipped step so updates stay small (ch 12) |
| MJX | MuJoCo XLA | the MuJoCo physics engine compiled to run thousands of simulations at once on a GPU ([bonus](ch12_bonus_humanoid)) |
| AGI | artificial general intelligence | chapter 13's topic |

Tooling: **CPU/GPU** central/graphics processing unit; **HTML** the format of the Flyte report pages; **PNG** image files; **ASCII** plain-text characters (the terminal charts).

**Symbols**

| Symbol | Meaning |
|---|---|
| s, a, r, s' | state, action, reward, next state |
| π (pi) | a policy: which action to take in each state |
| V(s) | state value: expected return from s when following π |
| Q(s, a) | action value: expected return from taking a in s, then following π |
| V\*, Q\*, π\* | the optimal values and the optimal policy |
| G | the return: total (discounted) reward from a point onwards |
| P(s' \| s, a) | transition probability: the chance of landing in s' after taking a in s |
| γ (gamma) | discount factor: how much a reward one step later is worth (0 to 1) |
| α (alpha) | step size / learning rate: how far each update moves an estimate |
| ε (epsilon) | exploration rate: chance of a random action in ε-greedy |
| λ (lambda) | trace decay: how far back along the path a TD error is credited |
| τ (tau) | temperature in softmax exploration: high = more random |
| δ (delta) | TD error: target minus current estimate |
| E(s, a) | eligibility trace: how recently and often (s, a) was visited |

## The one data structure

Every world in chapters 2 to 7 is the same kind of dict, the shape Gym uses for `env.P`:

```python
P[state][action] = [(probability, next_state, reward, done), ...]
```

```
slippery walk five (SWF): 7 cells, start in the middle, +1 for leaving on the right

   H  [1]  [2]  [3]  [4]  [5]  G        choose → :  50% move →
   0   1    2    3    4    5   6                    33% stay put
            start ^                                 17% move ←  (the ice pushes back)

frozen lake 4x4 (FL4): choose a direction, get it 1/3 of the time,
slide to either side the other 2/3

   S . . .
   . H . H          H = hole (episode ends, 0)
   . . . H          G = goal (episode ends, +1)
   H . . G
```

Chapters 2 and 3 read `P` directly: that is planning, with a model. From chapter 4 on, the agent only gets to *sample* from it through `TabularEnv.step()`, which is exactly the model-free setting, and chapter 3's exact answers become the ground truth every learner is scored against.

## Setup

Two ways to run it; pick one. Every chapter runs on the CPU (the networks are tiny), so no GPU is needed on either path. The one exception is the optional [humanoid bonus](ch12_bonus_humanoid), which needs an NVIDIA GPU devbox.

- **A Flyte devbox on your machine (recommended).** A Linux box or an **Apple Silicon Mac**, with Docker running. Every chapter runs as a Flyte task, and you get the full reports in the Flyte UI: the worlds drawn as grids, the replays, the learning animations, the videos.
- **No devbox: `flyte run --local`, or the plain scripts.** Nothing but Python. `--local` writes the same report as an HTML file; the plain scripts print tables and ASCII charts and save a PNG.

What you need, by path:

| | Devbox | No devbox (`--local` / scripts) |
|---|---|---|
| Python | 3.10 or newer (3.11 used here) | same |
| [`uv`](https://docs.astral.sh/uv/) | yes, for the venv | yes, for the venv |
| Docker | yes: Docker Desktop on a Mac, Docker Engine on Linux, running | no |
| `swig` + a C++ compiler | yes: `box2d-py` (LunarLander) builds from source | same |
| `flyte` | 2.2.1, from `requirements.txt` | same |
| CPU / memory | chapters 9 to 12 ask for 8 CPUs and 16 GB per pod | whatever you have; the big chapters just take longer |
| GPU | not used | not used |

In every case, start here:

```bash
git clone https://github.com/sagecodes/ai-build-and-learn
cd ai-build-and-learn/topics/grokking-rl

brew install swig                  # Mac (plus `xcode-select --install` if you have no compiler)
# sudo apt install swig build-essential    # Linux

uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install -r requirements.txt
```

<details id="setup-devbox">
<summary><b>On your own machine: a Flyte devbox (recommended)</b></summary>

A devbox is a Flyte control plane in one container on your machine. The task image is built for `linux/arm64`, which is native on both an Apple Silicon Mac and an NVIDIA DGX Spark (where this was built).

```bash
flyte start devbox                 # Apple Silicon Mac or Linux
flyte create config --endpoint dns:///localhost:30080 --insecure --builder local \
    -p flytesnacks -d development -o .flyte/config.yaml
```

`flytesnacks` is the project every devbox starts with; I use my own (`physical-ai`), and any project works.

**On an Apple Silicon Mac:**

- **Give Docker Desktop room.** Chapters 9 to 12 each ask for a pod with 8 CPUs and 16 GB (`cpu_env` in `config.py`), and a `book` run fits two at a time. Set Docker Desktop's resources to at least 8 CPUs and 20 GB of memory, or those chapters sit in `Pending` forever. Chapters 2 to 8 need only 2 CPUs and 4 GB each.
- **`--apple-silicon` is not needed.** Newer Flyte releases (2.12+) add `flyte start devbox --apple-silicon`, which runs tasks that ask for a GPU natively on the Mac's GPU. No task here asks for one, so the plain devbox is all you need, and it works with the `flyte` 2.2.1 pinned in `requirements.txt`.
- **First run builds the image** (torch, MuJoCo, and `box2d-py` from source), so expect several minutes before the first chapter starts; after that it is cached.

On an x86 Linux machine, change `PLATFORM` in `config.py` to `("linux/amd64",)`. With an NVIDIA GPU you can start the devbox with `--gpu`, but nothing here will use it.

Then run any chapter, or the whole book:

```bash
flyte run pipeline.py ch03_planning                  # one chapter: the run is named after it
flyte run pipeline.py ch06_control --args "--env sws --episodes 3000"   # with script flags
flyte run pipeline.py book                                     # all of them, a tab per chapter
```

The [humanoid bonus](ch12_bonus_humanoid#running-it) needs `flyte start devbox --gpu` on an NVIDIA machine and one extra deploy; its README has the steps.

</details>

<details id="setup-local">
<summary><b>No devbox: <code>flyte run --local</code>, or the plain scripts</b></summary>

`--local` runs the same task in your own process and writes the same report, with no Docker and no cluster:

```bash
flyte run --local pipeline.py ch03_planning
```

Each chapter is also a plain script. Run it from this folder and you get the console version: tables, ASCII charts, and a PNG next to the script.

```bash
python -m ch04_bandits.bandits
python -m ch06_control.control --env fl4 --episodes 10000
```

Chapter 12's HalfCheetah video renders MuJoCo headless with EGL (`MUJOCO_GL=egl`), which macOS does not have. On a Mac, use the devbox for chapter 12: the pod renders with OSMesa instead.

</details>

## Layout

```
envs.py          the MDPs (walks, frozen lakes) + TabularEnv, the sampler
viz.py           report HTML: value/policy grids, episode replays, sliders
asciichart.py    terminal line and bar charts
config.py        Flyte image + task environments (CPU for tabular, GPU later)
pipeline.py      Flyte tasks: one per chapter (`ch03_planning`, ...), `chapter --number N`, `ch12_bonus_humanoid`, and `book`
chNN_topic/      one folder per chapter: README.md + the example
```

The chapter scripts build on each other the same way the book does: chapter 5 reuses chapter 3's policy evaluation as the answer key, chapter 6 reuses chapter 5's step-size schedule, and chapter 7 races its methods against chapter 6's Q-learning.
