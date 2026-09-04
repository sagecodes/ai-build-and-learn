"""Drive a trained world model open-loop under action sequences you choose.

Named for the open-loop imagination it measures, and NOT `dream.py`: pipeline.py
defines a Flyte task called `dream`, which would shadow the module import and make
`dream.run(...)` resolve to the task object. That failure surfaces only in a pod.

    python openloop.py --logdir ~/logdir/smoke-back       # both artifacts

Dreamer already writes an open-loop grid during training (see scopevid.py), and that
grid is the demo the rest of this repo is built around. It answers "can the model
imagine?" qualitatively. This file answers the two questions it cannot:

  **How far can it imagine before it is wrong?** `report-openloop-image.mp4` shows one
  fixed horizon with a red border and leaves the viewer to judge the drift by eye. The
  same forward pass, scored per imagination step against the truth, is a curve: error
  against horizon, in pixels. That curve is what justifies `imag_horizon`, which this
  config sets to 15, and it is the honest version of "the model is good".

  **Is it a simulator or a video predictor?** A model that merely extrapolates the
  clip it just saw would produce the same future no matter what you asked of it. So
  ask it for several futures from ONE latent state under different action sequences.
  If they differ, and differ in the direction the actions imply, the model has learned
  that actions cause outcomes rather than that frames follow frames.

── How it gets inside the agent ────────────────────────────────────────────────
`embodied.jax.Agent` compiles exactly three entry points, `train`, `report` and
`policy` (see `nj.pure(self.model.X)` in embodied/jax/agent.py). There is no hook for
adding a fourth. So `DreamAgent` subclasses the inner model and overrides `report`,
which gets compiled with the right mesh, params and sharding for free. No fork, no new
plumbing: the cost of the whole approach is one subclass.

`RSSM.imagine(carry, policy, length, ...)` takes EITHER a callable policy or a concrete
array of actions, which it scans over (rssm.py). Upstream's own open-loop report uses
the array form to replay the true actions. Feeding it a hand-built array instead is
what makes a counterfactual, and it is why this needs no new RSSM code either.

── The one config change, and why it is safe ───────────────────────────────────
The saved config has `replay_context: 1`, which puts encoder/dynamics/decoder carry
entries into `ext_space` and therefore makes them required keys of every batch. Those
come out of the replay buffer, which is exactly the thing a finished run does not keep.
Setting it to 0 here drops that requirement. It is safe because `replay_context` is
read in only three places (agent.py 94, 132, 144) and none of them build a module: the
parameter shapes are identical, so the checkpoint loads unchanged.
"""

from __future__ import annotations

import argparse
import logging
from functools import partial as bind
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# How many context frames the model watches before it goes blind. Upstream's report
# uses half the sequence; keeping that convention makes the numbers here comparable to
# the mp4 the training run already produced.
_CONTEXT = 16
_HORIZON = 32

# Row order in the counterfactual video. `real` last so it reads as the reference.
_ROWS = ("zero", "pos", "neg", "real")


def _load_config(logdir: Path, batch: int, length: int):
    """The run's own config, with the replay-context requirement dropped."""
    import elements
    import ruamel.yaml as yaml

    raw = yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text())
    config = elements.Config(raw)
    return config.update({
        "replay_context": 0,
        "batch_size": batch,
        "batch_length": length,
        "report_length": length,
        "jax.prealloc": False,
    })


def _dream_agent_cls(context: int, horizon: int):
    """Build the DreamAgent class. Deferred so importing this module needs no jax.

    Same reason replay.py imports inside its function: `flyte run` bundles this file
    next to pipeline.py, and a module-level jax import would run at bundle time.
    """
    import jax
    import jax.numpy as jnp
    from dreamerv3 import agent as dv3agent

    class DreamAgent(dv3agent.Agent):
        """Upstream's agent with `report` replaced by the two dream measurements.

        Overriding `report` rather than adding a method is deliberate: the outer
        `embodied.jax.Agent` compiles `train`, `report` and `policy` and nothing else,
        so this is the only entry point that arrives with params, mesh and sharding
        already wired. The training-metric and gradnorm halves of upstream's report
        are dropped; nothing here is training.
        """

        def report(self, carry, data):
            carry, obs, prevact, _ = self._apply_replay_context(carry, data)
            enc_carry, dyn_carry, dec_carry = carry
            B, T = obs["is_first"].shape
            K, H = context, min(horizon, T - context)
            assert K + H <= T, (K, H, T)

            # `loss` is the cheapest way to get encoder tokens for the whole window,
            # and it returns the carries already advanced the way the model expects.
            _, (new_carry, _, outs, _) = self.loss(
                carry, obs, prevact, training=False)

            head = lambda xs, n=K: jax.tree.map(lambda x: x[:, :n], xs)
            mid = lambda xs: jax.tree.map(lambda x: x[:, K:K + H], xs)

            # Watch the context. This is the green-bordered half of the training mp4.
            obs_carry, _, obsfeat = self.dyn.observe(
                dyn_carry, head(outs["tokens"]), head(prevact),
                head(obs["is_first"]), training=False)

            metrics = {}
            blind = jnp.zeros_like(mid(obs["is_first"]))

            # The decoder's own noise floor. Reconstructing a frame the model is
            # LOOKING AT still costs error, so imagination can never beat this number
            # and comparing raw imagination error to a perfect frozen frame conflates
            # "cannot predict motion" with "cannot draw sharply". Everything below is
            # read relative to this.
            _, _, obsrecon = self.dec(
                dec_carry, obsfeat, head(obs["is_first"]), training=False)
            for key in self.dec.imgkeys:
                truth = obs[key][:, :K].astype(jnp.float32) / 255.0
                pred = jnp.clip(obsrecon[key].pred(), 0.0, 1.0)
                metrics[f"fidelity/floor/{key}"] = jnp.abs(pred - truth).mean()

            # ── 1. Fidelity: imagine the true actions, score every horizon step ──
            _, truefeat, _ = self.dyn.imagine(
                obs_carry, mid(prevact), length=H, training=False)
            true_dec, _, truerecon = self.dec(
                dec_carry, truefeat, blind, training=False)
            for key in self.dec.imgkeys:
                truth = obs[key][:, K:K + H].astype(jnp.float32) / 255.0
                pred = jnp.clip(truerecon[key].pred(), 0.0, 1.0)
                # Mean over batch and pixels, kept per horizon step. This is the
                # curve: how wrong is step 1 of imagination, step 2, ... step H.
                err = jnp.abs(pred - truth).mean(axis=(0, 2, 3, 4))
                metrics[f"fidelity/mae/{key}"] = err
                # A scale-free reading of the same thing: the error a model would get
                # by freezing the last frame it actually saw. Imagination is only
                # worth anything while it beats this.
                frozen = obs[key][:, K - 1: K].astype(jnp.float32) / 255.0
                metrics[f"fidelity/frozen_mae/{key}"] = jnp.abs(
                    frozen - truth).mean(axis=(0, 2, 3, 4))
            # Reward is the other thing the actor consumes from the dream, so its
            # drift matters as much as the pixels do.
            rew = self.rew(self.feat2tensor(truefeat), 2).pred()
            metrics["fidelity/reward_mae"] = jnp.abs(
                rew - obs["reward"][:, K:K + H]).mean(axis=0)
            metrics["fidelity/reward_pred"] = rew.mean(axis=0)
            metrics["fidelity/reward_true"] = obs["reward"][:, K:K + H].mean(axis=0)

            # ── 2. Counterfactuals: one latent, several futures ──────────────────
            RB = min(4, B)
            act_key = list(self.act_space.keys())[0]
            shape = (RB, H, *self.act_space[act_key].shape)
            plans = {
                "zero": jnp.zeros(shape, jnp.float32),
                "pos": jnp.full(shape, 0.8, jnp.float32),
                "neg": jnp.full(shape, -0.8, jnp.float32),
                "real": mid(prevact)[act_key][:RB],
            }
            start = jax.tree.map(lambda x: x[:RB], obs_carry)
            for name, act in plans.items():
                _, feat, _ = self.dyn.imagine(
                    start, {act_key: act}, length=H, training=False)
                _, _, recon = self.dec(
                    jax.tree.map(lambda x: x[:RB], dec_carry), feat,
                    blind[:RB], training=False)
                for key in self.dec.imgkeys:
                    frames = jnp.clip(recon[key].pred() * 255, 0, 255).astype(
                        jnp.uint8)
                    metrics[f"dream/{name}/{key}"] = frames
                # If these are all equal the model ignored the actions, so the
                # summary number is the spread between plans, not any one of them.
                metrics[f"dream/{name}/reward"] = self.rew(
                    self.feat2tensor(feat), 2).pred().mean()

            carry = (*new_carry, {k: data[k][:, -1] for k in self.act_space})
            return carry, metrics

    return DreamAgent


def _build(logdir: Path, config, cls):
    """`main.make_agent`, but instantiating our subclass instead of upstream's.

    Copied rather than called because make_agent hardcodes `from .agent import Agent`.
    The keyword list has to track upstream; it is short and it fails loudly if it
    drifts, because Config rejects unknown keys.
    """
    import arena  # noqa: F401  registers the domain before make_env resolves it
    import elements
    from dreamerv3 import main as dv3main

    env = dv3main.make_env(config, 0)
    notlog = lambda k: not k.startswith("log/")
    obs_space = {k: v for k, v in env.obs_space.items() if notlog(k)}
    act_space = {k: v for k, v in env.act_space.items() if k != "reset"}
    env.close()

    agent = cls(obs_space, act_space, elements.Config(
        **config.agent,
        logdir=str(logdir),
        seed=config.seed,
        jax=config.jax,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        replay_context=config.replay_context,
        report_length=config.report_length,
        replica=config.replica,
        replicas=config.replicas,
    ))
    cp = elements.Checkpoint(logdir / "ckpt")
    cp.agent = agent
    cp.load(keys=["agent"])
    return agent, obs_space, act_space


def collect(config, agent, obs_space, act_space, batch: int, length: int) -> dict:
    """Run the trained policy and stack real transitions into one [B, T] batch.

    The model is only asked to imagine futures of states it might actually reach, so
    the context frames come from the policy acting in the environment rather than from
    random actions. `parallel=False` for the same reason replay.py needs it: the
    transitions have to come back into this process.
    """
    import embodied
    import numpy as np
    from dreamerv3 import main as dv3main

    need = batch * length
    buf: list[dict] = []
    driver = embodied.Driver([bind(dv3main.make_env, config, 0)], parallel=False)
    driver.on_step(lambda tran, _: buf.append(tran))
    driver.reset(agent.init_policy)
    while len(buf) < need:
        driver(lambda *a: agent.policy(*a, mode="eval"), steps=need - len(buf) + 1)
    log.info("collected %d transitions", len(buf))

    keys = list(obs_space) + list(act_space)
    out = {}
    for key in keys:
        stacked = np.stack([t[key] for t in buf[:need]])
        out[key] = stacked.reshape((batch, length, *stacked.shape[1:]))
    # `consec` and `stepid` are required by ext_space. With replay_context disabled
    # (see the module docstring) nothing reads their values, only their shapes.
    out["consec"] = np.zeros((batch, length), np.int32)
    out["stepid"] = np.zeros((batch, length, 20), np.uint8)
    return out


def run(logdir: Path, batch: int = 8, context: int = _CONTEXT,
        horizon: int = _HORIZON) -> dict:
    """Load the agent at `logdir` and return the dream measurements as numpy."""
    length = context + horizon
    config = _load_config(logdir, batch, length)
    cls = _dream_agent_cls(context, horizon)
    agent, obs_space, act_space = _build(logdir, config, cls)
    data = collect(config, agent, obs_space, act_space, batch, length)
    # Same placement the training stream gives its batches. Without it the arrays are
    # plain numpy and jax refuses the implicit host-to-device transfer under the
    # report's sharding.
    from embodied.jax import internal
    data = internal.device_put(data, agent.train_sharded)
    data["seed"] = agent._seeds(0, agent.train_mirrored)
    carry = agent.init_report(batch)
    _, mets = agent.report(carry, data)
    return {k: v for k, v in mets.items() if not k.startswith("params/")}


def _summarise(mets: dict, context: int) -> str:
    """The one-paragraph version, for a terminal and for the report caption."""
    import numpy as np

    lines = []
    for key in sorted(k for k in mets if k.startswith("fidelity/mae/")):
        name = key.split("/")[-1]
        err = np.asarray(mets[key])
        frozen = np.asarray(mets[key.replace("/mae/", "/frozen_mae/")])
        floor = float(np.asarray(mets[f"fidelity/floor/{name}"]))
        # Two readings, because neither alone is honest.
        #
        # `useful` is the contiguous prefix over which imagining beats "assume nothing
        # moves". It is the harsh one: the frozen frame is the real image and carries
        # no decoder error, so a model can predict the motion correctly and still lose.
        # Deliberately the prefix and not the last winning step: a model that wins at
        # 3, loses at 4 and wins again at 27 is not usable to 27.
        worse = np.where(err >= frozen)[0]
        useful = int(worse[0]) if len(worse) else len(err)
        # `sharp` is how long imagination stays within twice the decoder's own floor,
        # which isolates the dynamics from the drawing.
        over = np.where(err > 2 * floor)[0]
        sharp = int(over[0]) if len(over) else len(err)
        lines.append(
            f"{name}: step 1 mae {err[0]:.4f} -> step {len(err)} {err[-1]:.4f} "
            f"(decoder floor {floor:.4f}); within 2x floor for {sharp}/{len(err)} "
            f"steps; beats a frozen frame for {useful}/{len(err)}"
        )
    if "fidelity/reward_mae" in mets:
        r = np.asarray(mets["fidelity/reward_mae"])
        lines.append(f"reward mae {r[0]:.3f} at step 1 -> {r[-1]:.3f} at step {len(r)}")
    spread = {k: float(np.asarray(v)) for k, v in mets.items()
              if k.startswith("dream/") and k.endswith("/reward")}
    if spread:
        lo, hi = min(spread.values()), max(spread.values())
        lines.append(
            "imagined reward by action plan: "
            + ", ".join(f"{k.split('/')[1]} {v:.2f}" for k, v in sorted(spread.items()))
            + f"  (spread {hi - lo:.2f}; near zero means the model ignored the actions)"
        )
    return "\n".join(lines)


def save_videos(mets: dict, out: Path, fps: int = 10) -> list[Path]:
    """Write the counterfactual dreams as one mp4 per image key.

    All plans are stacked into a single frame, one row per action sequence, so the
    comparison is a single video rather than four the viewer has to align by eye. Row
    order is fixed by `_ROWS` so `real` is always the bottom row and the eye has a
    reference to read the others against.
    """
    import numpy as np

    import replay

    out.mkdir(parents=True, exist_ok=True)
    # `dream/<plan>/reward` is a scalar and lives alongside the frame stacks, so
    # select by rank rather than by name.
    keys = sorted({k.split("/")[2] for k in mets
                   if k.startswith("dream/") and np.asarray(mets[k]).ndim == 5})
    written = []
    for key in keys:
        rows = []
        for name in _ROWS:
            arr = mets.get(f"dream/{name}/{key}")
            if arr is None:
                continue
            arr = np.asarray(arr)          # [B, T, H, W, C]
            # Lay the batch out along x, the plans down y.
            b, t, h, w, c = arr.shape
            rows.append(arr.transpose(1, 2, 0, 3, 4).reshape(t, h, b * w, c))
        if not rows:
            continue
        video = np.concatenate(rows, axis=1)
        # Nearest-neighbour upscale for the same reason scopevid.py does it: 64x64
        # blurs into mush once a browser scales it.
        video = np.repeat(np.repeat(video, 3, axis=1), 3, axis=2)
        path = out / f"counterfactual-{key}.mp4"
        path.write_bytes(replay.encode(list(video), fps=fps))
        written.append(path)
        log.info("wrote %s (%d frames, %dx%d)", path, len(video), *video.shape[1:3])
    return written


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logdir", type=Path, required=True)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--context", type=int, default=_CONTEXT)
    parser.add_argument("--horizon", type=int, default=_HORIZON)
    parser.add_argument("--out", type=Path, default=None,
                        help="directory for the counterfactual mp4s")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    mets = run(args.logdir, args.batch, args.context, args.horizon)
    if args.out:
        save_videos(mets, args.out)
    print()
    print(_summarise(mets, args.context))


if __name__ == "__main__":
    main()
