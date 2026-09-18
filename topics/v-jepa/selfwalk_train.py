"""The PPO half of `selfwalk`: train a G1 from scratch against the distilled V-JEPA reward.

Runs in the MJX image (jax on the GPU, no torch). It never sees V-JEPA itself, only the
small reward net `selfwalk_label` fitted to V-JEPA's scores, baked into the env as
jax arrays so the whole PPO loop stays inside one XLA program.

The live report is modelled on topics/rl-mujoco's: at every eval the CURRENT policy is
filmed from the side and the clip replaces the previous one, next to the reference
walker V-JEPA is comparing it to, with a filmstrip of every snapshot so far. That is
the only honest way to watch a learned reward: a reward curve that climbs while the
robot visibly does something silly is reward hacking, and the number alone will not
tell you.
"""

from __future__ import annotations

import base64
import functools
import io
import logging
import pickle
import time

import flyte
import flyte.report
import numpy as np
from flyte.io import File

import reports
import selfwalk as sw
import viz
from config import mjx_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

SNAPSHOT_STEPS = 300          # 6 s of robot time per filmed snapshot


def _ppo_params(num_timesteps: int, num_envs: int, num_evals: int):
    from mujoco_playground.config import locomotion_params

    p = locomotion_params.brax_ppo_config(sw.ENV_NAME)
    p.num_timesteps, p.num_envs, p.num_evals = num_timesteps, num_envs, num_evals
    return p


def _thumb(frame) -> str:
    from PIL import Image

    img = Image.fromarray(np.asarray(frame, np.uint8))
    img.thumbnail((220, 124))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


@mjx_env.task(report=True)
async def train_student(
    label: File,
    round_i: int = 0,
    num_timesteps: int = 50_000_000,
    restore: File | None = None,
    num_envs: int = 4096,
    num_evals: int = 10,
    relabel_envs: int = 64,
    relabel_steps: int = 400,
    seed: int = 0,
) -> File:
    """One round of PPO. Returns a pickle: params, eval history, the last snapshot clip,
    and stochastic rollouts of the final policy for the next round's V-JEPA relabel."""
    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo
    from mujoco_playground import wrapper

    with open(await label.download(), "rb") as f:
        lab = pickle.load(f)
    prev = None
    if restore is not None:
        with open(await restore.download(), "rb") as f:
            prev = pickle.load(f)

    env = sw.make_student_env(lab["rewnet"], impl="warp")
    params = _ppo_params(num_timesteps, num_envs, num_evals)
    net_cfg = dict(params.network_factory)
    network_factory = functools.partial(ppo_networks.make_ppo_networks, **net_cfg)
    log.info("round %d: %s steps, %d envs, restore=%s, device %s", round_i, f"{num_timesteps:,}",
             num_envs, prev is not None, jax.devices())

    cam = sw.ClipCamera(env.mj_model)
    ref_html = viz.video_html(lab["reference_mp4"], "the target: the rl-mujoco G1, as a video only",
                              max_width=480)
    start_steps = int(prev["total_steps"]) if prev else 0
    history: list[dict] = list(prev["history"]) if prev else []
    snaps: list[dict] = list(prev["snaps"]) if prev else []
    latest: dict = {}
    started = time.time()
    jit = {}

    def paint(stage: str) -> None:
        try:
            h = history[-1] if history else {}
            rows = [("round", str(round_i)), ("stage", stage),
                    ("env steps (all rounds)", f"{h.get('step', start_steps):,}"),
                    ("distilled V-JEPA reward / episode", f"{h.get('reward', 0):.2f}"),
                    ("episode length (of 1000)", f"{h.get('ep_len', 0):.0f}"),
                    ("walked forward / episode", f"{h.get('fwd_m', 0):.2f} m"),
                    ("elapsed this round", f"{(time.time() - started) / 60:.1f} min")]
            body = reports.progress_html(f"selfwalk round {round_i}: a G1 learns to walk from V-JEPA's opinion",
                                         "No gait reward. The only reward is a small net fitted to V-JEPA's "
                                         "score of how much a clip looks like the reference walker.", rows)
            if latest:
                cap = (f"student at {latest['step']:,} steps: survived {latest['alive']}/{SNAPSHOT_STEPS}, "
                       f"walked {latest['fwd']:.2f} m forward, reward {latest['r']:.2f}/step")
                body += reports.side_by_side([("student now", viz.video_html(latest["mp4"], cap, max_width=480)),
                                              ("reference", ref_html)])
            else:
                body += ref_html
            if history:
                xs = [h["step"] / 1e6 for h in history]
                body += viz.curve_chart("Distilled V-JEPA reward per episode", "env steps (M)", "reward",
                                        {"reward": list(zip(xs, [h["reward"] for h in history]))})
                body += viz.curve_chart("What the robot actually does", "env steps (M)", "per episode",
                                        {"metres walked forward": list(zip(xs, [h["fwd_m"] for h in history])),
                                         "episode length / 100": list(zip(xs, [h["ep_len"] / 100 for h in history]))})
            body += reports.filmstrip("Every snapshot so far (M steps, metres forward)",
                                      [(f"{s['step'] / 1e6:.0f}M {s['fwd']:.1f}m", s["thumb"]) for s in snaps])
            flyte.report.replace(body, do_flush=True)
        except Exception as exc:  # noqa: BLE001 - never kill a training run over a paint
            log.warning("paint failed: %s", exc)

    paint("compiling")

    def film(make_policy, p, steps: int = SNAPSHOT_STEPS):
        if not jit:
            jit["reset"], jit["step"] = jax.jit(env.reset), jax.jit(env.step)
        pol = jax.jit(make_policy(p, deterministic=True))
        rng = jax.random.PRNGKey(0)
        s = jit["reset"](rng)
        qs, rs, fwd = [np.asarray(s.data.qpos)], [], 0.0
        for _ in range(steps):
            rng, k = jax.random.split(rng)
            a, _ = pol(s.obs, k)
            s = jit["step"](s, a)
            qs.append(np.asarray(s.data.qpos)); rs.append(float(s.metrics["jepa_reward"]))
            fwd += float(s.metrics["fwd_m"])
            if float(s.done) > 0.5:
                break
        frames = cam.follow(np.stack(qs))
        return frames, len(qs) - 1, fwd, float(np.mean(rs)) if rs else 0.0

    def policy_params_fn(step, make_policy, p):
        try:
            frames, alive, fwd, r = film(make_policy, p)
            latest.update(step=start_steps + int(step), mp4=viz.encode_mp4(frames, fps=50), alive=alive,
                          fwd=fwd, r=r)
            snaps.append({"step": start_steps + int(step), "fwd": fwd,
                          "thumb": _thumb(frames[len(frames) // 2])})
            log.info("  snapshot @ %s: survived %d/%d, %.2f m forward", f"{start_steps + int(step):,}",
                     alive, SNAPSHOT_STEPS, fwd)
            paint("training")
        except Exception as exc:  # noqa: BLE001
            log.warning("snapshot failed at %s: %s", step, exc)

    def progress(step, metrics):
        row = {"step": start_steps + int(step),
               "reward": float(metrics.get("eval/episode_reward", 0.0)),
               "fwd_m": float(metrics.get("eval/episode_fwd_m", 0.0)),
               "ep_len": float(metrics.get("eval/avg_episode_length", 0.0)),
               "round": round_i}
        history.append(row)
        log.info("  step %s | reward %.2f | %.2f m forward | ep len %.0f", f"{row['step']:,}",
                 row["reward"], row["fwd_m"], row["ep_len"])
        paint("training")

    make_inference_fn, trained, _ = ppo.train(
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        restore_params=prev["params"] if prev else None,
        seed=seed + round_i,
        **{k: v for k, v in params.items() if k != "network_factory"},
    )
    elapsed = time.time() - started
    paint("rolling out the final policy for V-JEPA to judge")

    # Stochastic rollouts of the final policy: the next round's relabel data. Sampled,
    # not deterministic, so the relabel covers what PPO will actually explore next.
    net = network_factory(env.observation_size, env.action_size,
                          preprocess_observations_fn=running_statistics.normalize)
    stoch = jax.jit(ppo_networks.make_inference_fn(net)(trained, deterministic=False))
    traj = sw.rollout(env, stoch, relabel_envs, relabel_steps, (0.0, 0.0, 0.0), 0.0, "policy",
                      seed=1000 + round_i)
    cam.close()

    out = {"params": jax.device_get(trained), "history": history, "network_factory": net_cfg,
           "total_steps": start_steps + num_timesteps, "round": round_i, "elapsed_s": round(elapsed, 1),
           "final_mp4": latest.get("mp4", b""), "final": {k: latest.get(k) for k in ("step", "alive", "fwd", "r")},
           "snaps": snaps, "rollouts": traj}
    path = f"/tmp/student_r{round_i}.pkl"
    with open(path, "wb") as f:
        pickle.dump(out, f)
    log.info("round %d done in %.1f min", round_i, elapsed / 60)
    return await File.from_local(path)
