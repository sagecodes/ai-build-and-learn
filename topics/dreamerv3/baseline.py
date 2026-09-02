"""The model-free control: PPO on the same task, at the same environment-step budget.

    flyte run baseline.py compare                    # PPO from state and from pixels
    flyte run baseline.py ppo --observation pixels   # just one

The README claims DreamerV3 is far more sample efficient than model-free RL. Until this
file existed, that claim rested on the published literature and on Dreamer's own curve,
with nothing measured on this box. This measures it.

── The comparison is a fixed step budget, not a fixed wall clock ───────────────
Both agents get 500,000 environment steps on `walker_walk`, and the question is only
"how far did each get". That is the axis model-based methods are supposed to win on,
and it is the honest one: PPO will finish those steps far sooner in wall clock, because
it does perhaps a thousandth of the computation per step. Wall clock is reported too,
precisely because it points the other way and the README says so.

The two observation modes matter separately:

  * `state`  : 24 numbers, the same proprioception `dmc_proprio` would give Dreamer.
    This is PPO on easy mode and the fairer test of the algorithm.
  * `pixels` : the same 64x64 frames the flagship run learned from, which is the true
    like-for-like against the headline result and is brutally hard for PPO. A CNN
    trained only by policy gradient gets one scalar of signal per step to learn vision
    from, where Dreamer's decoder gets 12,288.

── Why this does not reuse rl-mujoco's PPO ─────────────────────────────────────
That one is Brax PPO on MJX, which is a GPU-batched physics engine with a state-only
observation and a different robot. None of it transfers to dm_control pixels. This uses
stable-baselines3 through `shimmy`, which is the boring, well-trodden path.

── Why a separate image ────────────────────────────────────────────────────────
stable-baselines3 needs torch, and putting torch next to jax in the Dreamer image would
add gigabytes to every training pod that will never import it. Same reasoning as the
per-adapter images in the text-to-speech demo. Note that the default aarch64 wheel is
the CUDA build (torch 2.13.0+cu130), not a CPU one, so the image is large either way.

── Why it asks for the GPU ─────────────────────────────────────────────────────
Measured here before this was written, which is the only reason the numbers below are
not a guess:

    state   MlpPolicy, 4 envs      500 steps/s    500k in ~17 min
    pixels  CnnPolicy, 4 envs       31 steps/s    500k in ~4.4 HOURS on CPU

The pixel run is bottlenecked on CNN forward and backward passes, not on the simulator
(which does 388 steps/s on its own). Running that on CPU would take longer than the
DreamerV3 run it is a baseline for, while also competing for the same cores Dreamer
needs for MuJoCo. So this takes the GPU and runs serially with the Dreamer runs.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import flyte
import flyte.report

import arena  # noqa: F401  bundled so the arena domain is available here too
import reports
import scopevid
from config import GL_APT, PLATFORM, REGISTRY

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Matches the flagship run so the comparison is against a real measured curve.
_DEFAULT_STEPS = 500_000

image = (
    flyte.Image.from_debian_base(name="dv3-baseline", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("ffmpeg", *GL_APT)
    .with_pip_packages(
        # shimmy is the maintained dm_control -> gymnasium bridge; without it every
        # dm_control env has to be hand-wrapped and the wrapping is where the bugs are.
        "stable-baselines3", "shimmy[dm-control]", "gymnasium", "dm_control", "mujoco",
        "torch", "opencv-python-headless",
        # Not pulled in transitively; gymnasium.envs.mujoco imports it and dies without.
        "packaging",
        "av", "imageio", "pillow", "numpy",
        "flyte==2.2.1", "connectrpc==0.10.*",
    )
    .with_env_vars({"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
)

# Two environments, because the two baselines have genuinely different shapes and
# putting both on the GPU would make the cheap one queue behind a Dreamer run for hours
# to do seventeen minutes of work.
#
# `state` is MlpPolicy on a 24-number observation: 500 steps/s measured, CPU is fine,
# and it is small enough to run alongside a Dreamer training pod on this 20 core box.
# Sized to fit ALONGSIDE a running Dreamer job, which is the whole reason this one is
# CPU-only. That job's pod requests 12 of this box's 20 cores, so anything much larger
# than this sits Pending with `Insufficient cpu` until the Dreamer run finishes, which
# defeats the point. DummyVecEnv steps its environments sequentially and the MlpPolicy
# is tiny, so 3 cores is not the bottleneck: measured at 500 steps/s either way.
cpu_env = flyte.TaskEnvironment(
    name="dv3-baseline-cpu",
    image=image,
    resources=flyte.Resources(cpu="3", memory="12Gi", disk="30Gi"),
)

# `pixels` is CnnPolicy, and measured at 31 steps/s on CPU it would take 4.4 hours,
# longer than the DreamerV3 run it exists as a baseline for. It takes the GPU and
# therefore has to wait its turn behind the Dreamer runs.
gpu_env = flyte.TaskEnvironment(
    name="dv3-baseline-gpu",
    image=image,
    resources=flyte.Resources(cpu="8", memory="32Gi", gpu=1, disk="40Gi"),
)

orch_env = flyte.TaskEnvironment(
    name="dv3-baseline-orch",
    image=image,
    resources=flyte.Resources(cpu="1", memory="4Gi"),
    depends_on=[cpu_env, gpu_env],
)


def _make_env(task_id: str, observation: str, seed: int):
    """One dm_control env, wrapped for stable-baselines3.

    Pixels go through the same 64x64 render the Dreamer run used, so "PPO from pixels"
    means the same pixels rather than a more forgiving resolution.

    Note what is NOT here: frame stacking. Stacking inside the env produces a 4-D
    observation of shape (3, 64, 64, 3), which SB3's CnnPolicy cannot consume because
    NatureCNN wants (C, H, W). Stacking belongs at the vector-env level, where
    VecFrameStack concatenates along the channel axis and gives the (9, 64, 64) that
    SB3 expects. Verified before this ran anywhere.
    """
    import gymnasium as gym
    from shimmy.registration import DM_CONTROL_SUITE_ENVS  # noqa: F401  registers ids

    domain, task = task_id.split("_", 1)
    kwargs = dict(render_mode="rgb_array") if observation == "pixels" else {}
    env = gym.make(f"dm_control/{domain}-{task}-v0", **kwargs)
    if observation == "pixels":
        from gymnasium.wrappers import AddRenderObservation, ResizeObservation

        # Replace the state observation with the rendered frame. A single frame carries
        # no velocity, and unlike Dreamer's RSSM a PPO policy has no recurrent state to
        # recover it from, hence the stacking one layer up.
        env = AddRenderObservation(env, render_only=True)
        env = ResizeObservation(env, (64, 64))
    else:
        from gymnasium.wrappers import FlattenObservation

        env = FlattenObservation(env)
    env.reset(seed=seed)
    return env


async def _ppo(
    task_id: str = "walker_walk",
    observation: str = "state",
    steps: int = _DEFAULT_STEPS,
    n_envs: int = 8,
    seed: int = 0,
    device: str = "auto",
    snapshots: bool = True,
) -> dict:
    """Train PPO for `steps` env steps, filming the policy as it goes."""
    # A native crash here exits 139 with no Python traceback at all, and dm_control's
    # numpy 2.5 DeprecationWarning floods the log so hard that the last 40 lines of
    # `kubectl logs` contain nothing else. faulthandler turns the segfault into a stack;
    # the filter makes that stack findable. Both exist because the first run of this
    # task died in three seconds and said nothing about why.
    import faulthandler
    import os
    import sys
    import warnings

    faulthandler.enable()
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="dm_control.*")

    # MUST happen before torch is imported, which the stable_baselines3 import below
    # does. The aarch64 PyPI torch wheel is the CUDA build (2.13.0+cu130), and in a pod
    # with no `nvidia.com/gpu` there is no device for it to find. Constructing an Adam
    # optimizer triggers a lazy `torch._dynamo` import, that import probes the CUDA
    # driver, and the probe SEGFAULTS: exit 139, no Python traceback, three seconds in.
    # Hiding the devices makes torch take its CPU path and never probe at all.
    #
    # Only for the CPU baseline. `ppo_pixels` runs in gpu_env, has a real device, and
    # wants it: the CNN is far slower on CPU (31 steps/s measured).
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # And block triton outright, which is the actual cause of that segfault. Building
    # an Adam optimizer calls `torch.utils._triton.has_triton_package()`, which does
    # `import triton`, and triton's aarch64 native extension crashes on import here
    # (triton/knobs.py line 15). has_triton_package guards with `except ImportError`,
    # but a SIGSEGV is not an exception, so the guard cannot help.
    #
    # Poisoning sys.modules turns the crash into the ImportError that torch already
    # knows how to handle. Unconditional, both devices: stable-baselines3 never calls
    # torch.compile, so nothing in this workload has any use for triton.
    sys.modules["triton"] = None

    import numpy as np
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    def thunk(i):
        return lambda: Monitor(_make_env(task_id, observation, seed + i))

    venv = DummyVecEnv([thunk(i) for i in range(n_envs)])
    if observation == "pixels":
        # (64, 64, 3) x 3 frames -> (9, 64, 64) once SB3 transposes to channel-first.
        venv = VecFrameStack(venv, n_stack=3)
    policy = "CnnPolicy" if observation == "pixels" else "MlpPolicy"
    model = PPO(policy, venv, verbose=0, seed=seed, device=device)
    log.info("PPO %s on %s, %s steps", policy, task_id, f"{steps:,}")

    curve: list[tuple[float, float]] = []
    t0 = time.monotonic()
    # Mid-training footage, same idea as the snapshots in topics/rl-mujoco: the latest
    # clip answers "what is it doing now", the strip of stills answers "is it getting
    # better". A learning curve alone cannot tell a walker from a shuffler.
    film = {"mp4": b"", "step": 0, "probe": ""}
    strip: list[tuple[int, bytes]] = []
    snap_every = max(1, steps // 6) if snapshots else 0

    class Progress(BaseCallback):
        """Collect finished-episode returns, film the policy, and repaint the report."""

        def __init__(self):
            super().__init__()
            self.last_paint = 0.0
            self.next_snap = snap_every

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                ep = info.get("episode")
                if ep is not None:
                    curve.append((float(self.num_timesteps), float(ep["r"])))
            if snap_every and self.num_timesteps >= self.next_snap:
                self.next_snap += snap_every
                shot = _film(model, task_id, observation, seed, frames_wanted=150)
                if shot:
                    film.update(
                        mp4=scopevid.encode(np.stack(shot), fps=40),
                        step=self.num_timesteps,
                        probe=scopevid.luminance(np.stack(shot)),
                    )
                    strip.append((self.num_timesteps, scopevid.png(shot[-1])))
            now = time.monotonic()
            if now - self.last_paint >= 30:
                self.last_paint = now
                flyte.report.replace(
                    _html(task_id, observation, self.num_timesteps, steps,
                          curve, now - t0, film, strip),
                    do_flush=True,
                )
            return True

    model.learn(total_timesteps=steps, callback=Progress())
    secs = time.monotonic() - t0

    # The final clip is longer and larger: this is the one people actually watch.
    final = _film(model, task_id, observation, seed, frames_wanted=400, size=(480, 480))
    if final:
        film.update(
            mp4=scopevid.encode(np.stack(final), fps=40),
            step=steps,
            probe=scopevid.luminance(np.stack(final)),
        )
        strip.append((steps, scopevid.png(final[-1])))
        log.info("final clip: %s", film["probe"])
    flyte.report.replace(
        _html(task_id, observation, steps, steps, curve, secs, film, strip),
        do_flush=True,
    )

    best = max((y for _, y in curve), default=0.0)
    tail = [y for _, y in curve[-20:]]
    result = {
        "algorithm": f"PPO ({policy})",
        "task": task_id,
        "observation": observation,
        "steps": steps,
        "minutes": round(secs / 60, 1),
        "episodes": len(curve),
        "return_best": round(best, 1),
        "return_final_20": round(float(np.mean(tail)), 1) if tail else 0.0,
        "steps_per_sec": round(steps / secs, 1),
        "clip": film["probe"] or "no clip",
    }
    log.info("result: %s", result)
    return result


def _physics(env):
    """Walk gymnasium's wrapper chain down to dm_control's physics object.

    Same approach as replay.py, and needed for the same reason: shimmy only builds a
    MujocoRenderer for `render_mode="human"`, so asking the env to render is not an
    option. Going to the physics directly also means the clip is filmed at whatever
    resolution reads well in a report, independent of the 64x64 the policy observes.
    """
    while True:
        inner = getattr(env, "env", None)
        if inner is None:
            break
        env = inner
    dmenv = getattr(env, "_env", None)
    return getattr(dmenv, "physics", None)


def _film(model, task_id, observation, seed, frames_wanted=200, size=(240, 240)):
    """Roll the current policy out on a fresh env and render it.

    Returns [] rather than raising if anything about rendering fails. This is called
    mid-training, and a camera problem must never take down a run that is otherwise
    fine, exactly as in pipeline.py.
    """
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    try:
        inner = DummyVecEnv([lambda: _make_env(task_id, observation, seed + 999)])
        venv = VecFrameStack(inner, n_stack=3) if observation == "pixels" else inner
        physics = _physics(inner.envs[0])
        if physics is None:
            log.warning("could not reach dm_control physics; no video")
            return []
        frames, obs = [], venv.reset()
        for _ in range(frames_wanted):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, _, _ = venv.step(action)
            frames.append(physics.render(*size, camera_id=0))
        venv.close()
        return frames
    except Exception as exc:  # noqa: BLE001
        log.warning("filming failed, continuing without video: %s", exc)
        return []


def _html(task, observation, step, total, curve, secs, film=None, strip=None) -> str:
    pct = 100.0 * step / total if total else 0.0
    best = max((y for _, y in curve), default=0.0)
    rows = [
        ("Algorithm", "PPO (stable-baselines3), model-free"),
        ("Task", f"{task}, {observation}"),
        ("Progress", f"{step:,} / {total:,} env steps ({pct:.0f}%)"),
        ("Wall clock", f"{secs / 60:.1f} min"),
        ("Throughput", f"{step / max(secs, 1e-9):,.0f} env steps/s"),
        ("Episodes", f"{len(curve)}"),
        ("Return, best", f"{best:.1f}"),
    ]
    body = (
        "<h2>Model-free baseline: PPO on the same budget</h2>"
        + reports.panel("Run", reports.table(rows))
        + "<br/>"
    )
    if film and film.get("mp4"):
        body += (
            reports.heading("The policy, filmed during training")
            + reports.note(
                "Rendered from the dm_control physics alongside training, so it is a "
                "recording of the policy rather than a separate run. Compare it "
                "against the DreamerV3 clip at the same step count: the return curve "
                "below cannot tell a walker from a shuffler, and this can."
            )
            + reports.video_html(
                film["mp4"], f"step {film['step']:,} &middot; {film['probe']}", 480
            )
        )
        if strip and len(strip) >= 2:
            body += "<br/>" + reports.note(
                "The last frame of each snapshot, oldest on the left."
            ) + reports.filmstrip(strip, height=150)
        body += "<br/>"
    return (
        body
        + reports.note(
            "The comparison that matters is against DreamerV3 at the same number of "
            "environment steps. Walker's return is capped at 1,000 and DreamerV3 "
            "reached 987 in 500,000 steps from pixels. Wall clock is reported here "
            "because it points the other way: PPO gets through these steps far faster, "
            "and that is exactly the tradeoff the README describes."
        )
        + reports.curve(curve, f"PPO episode return ({observation})")
    )


@cpu_env.task(report=True)
async def ppo_state(
    task_id: str = "walker_walk",
    steps: int = _DEFAULT_STEPS,
    n_envs: int = 8,
    seed: int = 0,
) -> dict:
    """PPO from proprioception. CPU, ~17 min for 500k steps."""
    return await _ppo(task_id, "state", steps, n_envs, seed, "cpu")


@gpu_env.task(report=True)
async def ppo_pixels(
    task_id: str = "walker_walk",
    steps: int = _DEFAULT_STEPS,
    n_envs: int = 8,
    seed: int = 0,
) -> dict:
    """PPO from the same 64x64 frames DreamerV3 learned from. Needs the GPU."""
    return await _ppo(task_id, "pixels", steps, n_envs, seed, "auto")


@orch_env.task(report=True)
async def compare(
    task_id: str = "walker_walk",
    steps: int = _DEFAULT_STEPS,
    seed: int = 0,
) -> dict:
    """PPO from state and from pixels, same budget, one report.

    Runs sequentially rather than concurrently. Both are CPU tasks asking for 12 cores
    on a 20 core box, so overlapping them would make each slower and the throughput
    numbers meaningless.
    """
    out = {}
    out["state"] = await ppo_state(task_id=task_id, steps=steps, seed=seed)
    log.info("state: %s", out["state"])
    out["pixels"] = await ppo_pixels(task_id=task_id, steps=steps, seed=seed)
    log.info("pixels: %s", out["pixels"])
    flyte.report.replace(
        "<h2>PPO baselines</h2><pre>" + json.dumps(out, indent=2) + "</pre>",
        do_flush=True,
    )
    return out


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(compare))
