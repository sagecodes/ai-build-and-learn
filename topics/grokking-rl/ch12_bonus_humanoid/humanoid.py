"""Chapter 12 bonus: chapter 12's PPO, scaled up until it teaches a humanoid robot to walk.

The training is not here. It lives in ../rl-mujoco (MJX + Brax PPO on the GPU, a Unitree
G1 humanoid), and pipeline.py's `ch12_bonus_humanoid` task calls its deployed tasks:
`g1-train.train_policy` once per reward preset, then `g1-render.render_replay` to film
each trained policy and one random policy. This module reads what comes back and builds
the report section, so this project's image never needs jax, brax or a GPU.

    # once, so the tasks exist to be called
    cd ../rl-mujoco && flyte deploy --all pipeline.py
    # then
    flyte run pipeline.py ch12_bonus_humanoid
"""

import base64
import io
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import viz

# The preset diffs, as rl-mujoco's envs.PRESETS applies them on top of Playground's G1
# scales. Only what the report explains; anything else falls back to the default.
PRESET_NOTES = {
    "baseline": "DeepMind's tuned reward, untouched",
    "high-step": "feet_clearance 0 → −0.5 (a moving foot pays for being off step height), "
                 "feet_air_time 2 → 3 (more pay for time a foot spends in the air)",
    "smooth": "penalties for jerky, power-hungry motion",
    "turner": "twice the pay for turning when told to",
    "no-perturb": "no random shoves and no sensor noise",
}
# The air-time term's weight per preset, to undo it: the metric arrives already scaled.
AIR_TIME_SCALE = {"high-step": 3.0}
COLORS = ["#74b9ff", "#fdcb6e", "#55efc4", "#ff7675", "#a29bfe"]


class _Stub:
    """Stands in for any brax/jax object while unpickling: the weights are not needed."""

    def __init__(self, *a, **k):
        pass

    def __setstate__(self, state):
        pass

    def __call__(self, *a, **k):
        return _Stub()


class _NoBrax(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("brax", "jax", "jaxlib", "flax"):
            return _Stub
        return super().find_class(module, name)


def read_checkpoint(path):
    """rl-mujoco's checkpoint without its weights: the eval history and run settings.
    The pickle holds Brax objects, so a plain pickle.load needs brax installed."""
    with open(path, "rb") as f:
        ckpt = _NoBrax(f).load()
    ckpt.pop("params", None)
    return ckpt


def _png(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90, facecolor="#0f0f23")
    plt.close(fig)
    return f'<img src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}" style="max-width:100%">'


def _axes(ax, ylabel):
    ax.set_facecolor("#1a1a2e")
    ax.set_xlabel("environment steps (millions)", color="#ccc")
    ax.set_ylabel(ylabel, color="#ccc")
    ax.tick_params(colors="#888")
    ax.grid(alpha=0.15)


def _curves(runs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6), facecolor="#0f0f23")
    for (ckpt, _), color in zip(runs, COLORS):
        h = ckpt["history"]
        x = [r["step"] / 1e6 for r in h]
        a1.plot(x, [r["reward"] for r in h], color=color, label=ckpt["preset"])
        scale = AIR_TIME_SCALE.get(ckpt["preset"], 2.0)
        if all("feet_air_time" in r for r in h):
            a2.plot(x, [r["feet_air_time"] / scale for r in h], color=color, label=ckpt["preset"])
    _axes(a1, "evaluation score")
    _axes(a2, "feet air-time reward (unweighted)")
    for ax in (a1, a2):
        ax.legend(facecolor="#1a1a2e", labelcolor="#ccc", edgecolor="#333")
    fig.tight_layout()
    return _png(fig)


def _video(replay, caption):
    if not replay.get("mp4_b64"):
        return f'<div class="card">{replay.get("html", "")}<br>{caption}</div>'
    return (f'<div style="display:inline-block;margin:4px;text-align:center;vertical-align:top">'
            f'<video src="data:video/mp4;base64,{replay["mp4_b64"]}" controls loop muted autoplay '
            f'playsinline style="width:360px"></video><div style="font-size:12px;max-width:360px">'
            f"{caption}</div></div>")


def _stay_up(replay, steps):
    secs = replay["steps"] / replay.get("fps", 50)
    if replay["fell"]:
        return f"fell after {secs:.1f} s"
    return f"walked the whole {steps / replay.get('fps', 50):.0f} s without falling"


def _air_time_summary(runs):
    """Measured, not asserted: each preset's last unweighted air-time reading."""
    ends = [(c["preset"], c["history"][-1]["feet_air_time"] / AIR_TIME_SCALE.get(c["preset"], 2.0))
            for c, _ in runs if c["history"] and "feet_air_time" in c["history"][-1]]
    if len(ends) < 2:
        return "Train a second preset next to this one to see the reward change the gait."
    ends.sort(key=lambda e: -e[1])
    (top, a), (low, b) = ends[0], ends[-1]
    return (f"At the last evaluation, {top} earns {a:.2f} of unweighted air-time reward per "
            f"episode against {low}'s {b:.2f}, {a / b:.1f}x as much (the term pays each time a foot "
            "lands after 0.2 to 0.5 s in the air). Watch the clips above for how that looks."
            if b > 0 else
            f"At the last evaluation, {top} earns {a:.2f} of unweighted air-time reward per "
            f"episode against {low}'s {b:.2f}.")


def report(runs, random_replay, replay_steps):
    """runs: [(checkpoint dict, replay dict)], one per preset. Returns the HTML parts."""
    trained = []
    for ckpt, rep in runs:
        h = ckpt["history"]
        mins = ckpt.get("elapsed_s", 0) / 60
        trained.append(_video(rep, f"<b>{ckpt['preset']}</b> after {ckpt['num_timesteps'] / 1e6:.0f}M "
                                   f"steps ({mins:.0f} min on one GPU): {_stay_up(rep, replay_steps)}, "
                                   f"best eval score {max(r['reward'] for r in h):.1f}"))
    before = _video(random_replay, f"<b>random actions</b> (no training): {_stay_up(random_replay, replay_steps)}")

    first = runs[0][0]
    n_envs = first.get("num_envs", 4096)
    total = first.get("num_timesteps", 200_000_000)
    sps = total / max(first.get("elapsed_s", 1), 1)
    presets = "".join(f"<li><b>{c['preset']}</b>: {PRESET_NOTES.get(c['preset'], 'a reward preset from rl-mujoco')}</li>"
                      for c, _ in runs)

    parts = [
        "<p style='font-size:16px'><b>In one sentence:</b> the PPO that swung up a pendulum in "
        "chapter 12, scaled from 8 simulators to thousands on a GPU, teaches a 29-joint humanoid "
        "robot to walk, and a two-number change to its reward changes how it walks.</p>",
        '<div class="card"><b>The robot.</b> A Unitree G1 humanoid in MuJoCo physics. Fifty times a '
        "second the actor outputs <b>29 numbers</b>, a target angle for every joint (hips, knees, "
        "ankles, waist, arms), and a stiff motor controller under each joint chases its target. It "
        "sees <b>103 numbers</b>: joint angles and speeds, which way is down, how it is spinning, its "
        "last action, and a <b>joystick command</b> (\"go forward at 1 m/s\", \"turn left\"). The "
        "reward pays for matching the joystick, and twenty-odd smaller terms close off cheap tricks "
        "(diving forward, vibrating, sliding the feet). Falling costs −100 and ends the episode. "
        "Nothing tells it what walking looks like.</div>",

        "<h3>1. Before and after</h3>",
        viz.row(before, *trained),
        viz.explain(
            how="Every clip gets the same joystick command, forward at 1 m/s, and the same "
                f"{replay_steps / random_replay.get('fps', 50):.0f} seconds. The trained clips show the "
                "policy's mean action, the centre of its bell curve, not a random draw from it.",
            notice=f"With random joint targets the robot {_stay_up(random_replay, replay_steps)}. "
                   "The trained policy has no idea what a leg is: whatever gait it has, it found "
                   f"because that is what scored, over {max(c['num_timesteps'] for c, _ in runs):,} "
                   "steps of trial and error."),

        "<h3>2. The same algorithm as chapter 12, at a different size</h3>",
        "<table class='dc'><tr><th></th><th>chapter 12: Pendulum</th><th>bonus: G1 humanoid</th></tr>"
        "<tr><td>actions per step</td><td>1 (a torque)</td><td>29 (joint targets)</td></tr>"
        "<tr><td>what the actor sees</td><td>3 numbers</td><td>103 numbers</td></tr>"
        "<tr><td>what the critic sees</td><td>the same 3</td><td>216: the 103 plus true values only a simulator knows</td></tr>"
        f"<tr><td>simulators running at once</td><td>8, on the CPU</td><td>{n_envs:,}, on the GPU</td></tr>"
        "<tr><td>moves per batch</td><td>2,048</td><td>163,840</td></tr>"
        "<tr><td>passes over each batch</td><td>10</td><td>4</td></tr>"
        "<tr><td>clip (how far one update may move the policy)</td><td>20%</td><td>20%</td></tr>"
        "<tr><td>γ (patience)</td><td>0.99</td><td>0.97 (about 0.7 s of lookahead)</td></tr>"
        "<tr><td>network</td><td>64 × 64</td><td>512 × 256 × 128</td></tr>"
        f"<tr><td>training steps</td><td>400,000</td><td>{total:,} (~{sps:,.0f} per second)</td></tr></table>",
        viz.explain(
            how="Same loop as chapter 12's PPO: collect a batch with the current policy, improve it "
                "for a few passes, but never let one update move any action's odds by more than 20%. "
                "The physics runs in MJX, MuJoCo compiled for the GPU, so stepping the robots, scoring "
                "them and updating the networks all happen without leaving the GPU.",
            notice="Chapter 12 found PPO the most sample-hungry learner: it throws each batch away "
                   "after using it. With thousands of simulators that stops mattering: fresh "
                   f"experience costs almost nothing ({sps:,.0f} steps a second here), and PPO's "
                   "stability is worth more than sample efficiency. That trade is why PPO, not SAC, "
                   "is the default for training robots in simulation."),

        "<h3>3. Same PPO, different reward, different gait</h3>",
        f"<div class='card'>Each run starts from the same tuned reward and changes a few weights:<ul style='margin:4px 0'>{presets}</ul></div>",
        _curves(runs),
        viz.explain(
            how="Left: the evaluation score (each preset scores against its own reward, so compare "
                "shapes, not heights). Right: the feet air-time term with each preset's weight "
                "divided back out, so the lines measure the same thing: how often a foot lands "
                "after a real swing through the air.",
            notice=("The algorithm, the network, the robot and the training budget are identical; "
                    "only what gets paid for differs. " if len({c["num_timesteps"] for c, _ in runs}) == 1
                    else "These runs trained for different budgets, so part of any gap is just "
                         "training time; train both for the same steps for a fair comparison. ")
                   + _air_time_summary(runs)
                   + " Most of the craft of robot RL is in the reward, not the learner."),

        "<h3>4. The critic gets to cheat</h3>",
        viz.explain(
            how="This is chapter 11 and 12's actor-critic split, used on purpose. The actor only "
                "sees the 103 numbers a real robot's sensors could give it, with noise added. The "
                "critic sees 216, including the robot's true velocity, the force on each foot, and "
                "how long each foot has been in the air.",
            notice="The critic only grades; it never acts, and it is thrown away after training. So "
                   "it can use information a real robot would never have. Better grades mean "
                   "cleaner advantages, which mean a cleaner signal for the actor, while the actor "
                   "stays something that could run on hardware."),
    ]
    parts.append(viz.takeaways([
        "Chapter 12's PPO, unchanged in idea, walks a humanoid: what changed is scale.",
        "Thousands of GPU simulators make PPO's on-policy appetite cheap, and its stability valuable.",
        "The reward is the design: changing what gets paid for changes how it walks.",
        "An actor-critic critic can see more than the actor, because it is only needed during training.",
    ]))
    return parts
