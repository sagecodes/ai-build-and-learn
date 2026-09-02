"""Brax PPO on MJX, as the model-free control, with the parallelism confound measured.

    flyte run braxppo.py sweep                    # 64 / 512 / 2048 envs, one report
    flyte run braxppo.py ppo --num_envs 512       # one point of it

── Why this exists next to baseline.py ─────────────────────────────────────────
`baseline.py` runs stable-baselines3 PPO, which is the vanilla reference implementation
and the right thing for the PIXEL comparison, because MJX has no renderer in its
batched GPU path and therefore cannot do pixels at all.

This file is the STATE comparison, and it uses the same stack as the event next door
(`topics/rl-mujoco`): mujoco_playground's MJX port of the DeepMind Control Suite, with
Brax PPO on top. Two reasons that is the better tool here. It is thousands of times
faster, so a sweep is affordable. And it is what the sibling event actually used, so
the comparison connects two demos in this repo rather than introducing a third stack.

── The thing this is really measuring ──────────────────────────────────────────
It is tempting to put "DreamerV3 needed 500k steps, PPO needed 60M" on a slide and stop.
That number is real, and it comes from the two projects' own default configs for
walker_walk. But it quietly conflates two different things, and the sweep is here to
separate them.

PPO with `num_envs` parallel environments consumes `num_envs * unroll_length` steps per
gradient update. Playground's default for WalkerWalk is 2,048 envs and an unroll of 30,
so **every single update costs 61,440 environment steps**. Massive parallelism buys
wall clock and gradient stability, and it costs sample efficiency per step more or less
by construction. Comparing that against DreamerV3 running 4 environments and calling the
difference "model-based sample efficiency" would be measuring the batch size and
crediting the algorithm.

So this runs the same task and budget at several `num_envs` and reports return against
environment steps for each. If the curves collapse onto each other, parallelism was not
the story and the algorithmic gap is real. If they separate, the honest headline number
is the one from the smallest `num_envs`, and this file is what stops the README
overclaiming.

── What it cannot do ───────────────────────────────────────────────────────────
Pixels. MJX steps thousands of environments on the GPU precisely by not rendering any
of them. The pixel baseline stays in baseline.py on stable-baselines3, and it is the
one that is a true like-for-like against the flagship run.
"""

from __future__ import annotations

import logging
import time

import flyte
import flyte.report

import reports
from config import GL_APT, PLATFORM, REGISTRY

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# The MJX + Brax stack, copied deliberately rather than imported: topics/rl-mujoco is a
# sibling directory, not an installable package, and a cross-topic import would make
# either demo unable to run without the other. Keep this list in step with
# `topics/rl-mujoco/config.py::MJX_SPEC`, which carries the reasoning for each pin.
MJX_SPEC = (
    "jax[cuda13]==0.9.2",
    "mujoco", "mujoco-mjx", "brax",
    "playground",  # imports as `mujoco_playground`
    "numpy", "Pillow",
    "av", "imageio",
    "flyte==2.2.1",
    "connectrpc==0.10.*",
)

# WalkerWalk is the MJX port of the same dm_control task the Dreamer run trained on.
# A port, not the identical code, which is worth saying out loud: same reward
# structure and same task, re-expressed for a batched GPU solver.
ENV_NAME = "WalkerWalk"

# The points of the sweep. 2,048 is Playground's own default for this env, 64 is small
# enough to be in the same regime as Dreamer's 4, and 512 sits between them.
DEFAULT_ENV_COUNTS = (64, 512, 2048)

# Enough to reach the plateau at every point of the sweep without spending an hour on
# it. Playground's default budget for WalkerWalk is 60M, which is the number worth
# quoting, but the curve has long since flattened by 20M.
DEFAULT_STEPS = 20_000_000

image = (
    flyte.Image.from_debian_base(name="dv3-brax", registry=REGISTRY, platform=PLATFORM)
    .with_apt_packages("git", "ffmpeg", *GL_APT)
    .with_pip_packages(*MJX_SPEC)
    # Playground fetches the robot XMLs on first use. Baking them in keeps a git clone
    # out of every task pod; see topics/rl-mujoco/config.py for the full story about
    # why this must land in Playground's own package directory.
    .with_commands([
        "python -c 'from mujoco_playground._src import mjx_env; "
        "mjx_env.ensure_menagerie_exists()'",
    ])
)

gpu_env = flyte.TaskEnvironment(
    name="dv3-brax",
    image=image,
    resources=flyte.Resources(cpu="8", memory="64Gi", gpu=1, disk="50Gi"),
    # MuJoCo's renderer needs a GL context and a pod has no display, so EGL is the only
    # option that works headless. `mujoco.Renderer` raises on construction without it
    # rather than falling back to something that works. The matching runtime libs are
    # the GL_APT packages on the image: they are apt, not pip, because the renderer
    # dlopens libEGL and fails with a bare ImportError if they are absent.
    env_vars={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"},
)

orch_env = flyte.TaskEnvironment(
    name="dv3-brax-orch",
    image=image,
    resources=flyte.Resources(cpu="1", memory="4Gi"),
    depends_on=[gpu_env],
)


def _encode_mp4(frames, fps: int) -> bytes:
    """Frames (H, W, 3) uint8 -> H.264 mp4 bytes, via PyAV.

    Copied from `topics/rl-mujoco/render.py` for the same reason MJX_SPEC is: that
    directory is a sibling, not an installable package. PyAV rather than
    imageio-ffmpeg because PyAV publishes manylinux aarch64 wheels. yuv420p because
    anything else fails to play in a browser.
    """
    import io

    import av
    import numpy as np

    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        h, w = np.asarray(frames[0]).shape[:2]
        # H.264 requires even dimensions; odd ones fail at encoder open with a message
        # that never mentions the size.
        stream.width = w - (w % 2)
        stream.height = h - (h % 2)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "23"}
        for frame in frames:
            arr = np.asarray(frame, dtype=np.uint8)[: stream.height, : stream.width]
            container.mux(stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")))
        container.mux(stream.encode())  # flush
    return buf.getvalue()


def _pick_camera(mj_model) -> str | int:
    """Prefer a tracking camera so the walker does not stroll out of frame."""
    import mujoco

    names = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, i)
        for i in range(mj_model.ncam)
    ]
    for preferred in ("track", "tracking", "side", "front"):
        if preferred in names:
            return preferred
    return names[0] if names else -1


def _film(env, make_policy, params, steps: int = 400):
    """Roll the trained policy out once and return (mp4 bytes, frames rendered).

    This is the answer to an obvious question about this file: MJX cannot render during
    training, because stepping thousands of environments on the GPU is exactly the
    trick of not rendering any of them. That says nothing about afterwards. One
    environment, on the CPU MuJoCo model, rendered once the policy is trained, costs
    seconds and is the only way to see what the model-free baseline actually looks like
    next to Dreamer's walker.

    Never raises. Training is the expensive part and a GL problem must not throw away a
    finished sweep point, which is the same rule pipeline.py's replay video follows.
    """
    import jax

    try:
        reset, step = jax.jit(env.reset), jax.jit(env.step)
        inference = jax.jit(make_policy(params, deterministic=True))
        # Same seed every time so clips from different sweep points are comparable.
        rng = jax.random.PRNGKey(0)
        state = reset(rng)
        traj = [state]
        for _ in range(steps):
            rng, act_rng = jax.random.split(rng)
            action, _ = inference(state.obs, act_rng)
            state = step(state, action)
            traj.append(state)
            if float(state.done) > 0.5:
                break
        frames = env.render(traj, height=270, width=360, camera=_pick_camera(env.mj_model))
        return _encode_mp4(frames, int(round(1.0 / env.dt))), len(frames)
    except Exception as exc:  # noqa: BLE001
        log.warning("could not film the policy, sweep point is still good: %s", exc)
        return b"", 0


def _wait_for_gpu(timeout_s: int = 900, poll_s: int = 20) -> None:
    """Block until CUDA can actually initialize, instead of dying on a transient.

    Lifted from topics/rl-mujoco/train.py, and load-bearing for the same reason: GB10
    shares one 119.7 GiB pool between the OS page cache and the GPU, so heavy disk I/O
    elsewhere can leave CUDA unable to create a context at all, reported as

        Unable to initialize backend 'cuda': INTERNAL: no supported devices found

    which reads like a broken install and is nothing of the sort. It matters more here
    than usual: this task is meant to run right after a seven hour Dreamer job, which
    is exactly when the page cache is at its dirtiest. Probing with ctypes rather than
    importing jax is deliberate, because jax caches a failed backend init process-wide.
    """
    import ctypes

    cuda = ctypes.CDLL("libcuda.so.1")
    deadline = time.time() + timeout_s
    attempt = 0
    while True:
        attempt += 1
        cuda.cuInit(0)
        dev, ctx = ctypes.c_int(), ctypes.c_void_p()
        if cuda.cuDeviceGet(ctypes.byref(dev), 0) == 0 and \
                cuda.cuCtxCreate_v2(ctypes.byref(ctx), 0, dev) == 0:
            free, total = ctypes.c_size_t(), ctypes.c_size_t()
            cuda.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
            cuda.cuCtxDestroy_v2(ctx)
            log.info("GPU ready after %d probe(s): %.1f GiB free of %.1f",
                     attempt, free.value / 2**30, total.value / 2**30)
            return
        if time.time() >= deadline:
            raise RuntimeError(
                f"CUDA could not initialize within {timeout_s}s. On GB10 this is "
                "almost always page-cache pressure, not a broken driver."
            )
        log.warning("  CUDA unavailable (probe %d); retrying in %ds", attempt, poll_s)
        time.sleep(poll_s)


@gpu_env.task(report=True)
async def ppo(
    num_envs: int = 2048,
    steps: int = DEFAULT_STEPS,
    seed: int = 0,
) -> dict:
    """Brax PPO on WalkerWalk at one `num_envs`, reporting return against env steps."""
    import functools

    _wait_for_gpu()

    import jax
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.agents.ppo import train as ppo_train
    from mujoco_playground import registry, wrapper
    from mujoco_playground.config import dm_control_suite_params

    log.info("device: %s", jax.devices())
    # `impl="jax"` is load-bearing, not a preference. Playground 0.2.0 defaults this
    # env's config to `impl: warp`, which sends `mjx.put_model` down a path that reads
    # `mujoco_warp.types.GraphMode.WARP`. mujoco-warp is not installed here and has no
    # aarch64 wheel, so mujoco-mjx 3.12 falls back to a stub where `GraphMode` is
    # plain `int`, and the pod dies five seconds in with
    #
    #     AttributeError: type object 'int' has no attribute 'WARP'
    #
    # which names neither warp nor the missing package. The JAX backend is what this
    # sweep wants anyway: it is the one brax PPO trains through.
    env = registry.load(ENV_NAME, config_overrides={"impl": "jax"})
    params = dm_control_suite_params.brax_ppo_config(ENV_NAME)
    params.num_timesteps = steps
    params.num_envs = num_envs
    # batch_size cannot exceed num_envs: Brax reshapes a rollout of `num_envs`
    # trajectories into minibatches, and asking for 1,024 out of 64 fails deep inside
    # the jitted update with a shape error that does not name the cause.
    params.batch_size = min(int(params.batch_size), num_envs)

    net_cfg = dict(params.get("network_factory", {}))
    network_factory = (
        functools.partial(ppo_networks.make_ppo_networks, **net_cfg)
        if net_cfg else ppo_networks.make_ppo_networks
    )

    curve: list[tuple[float, float]] = []
    started = time.time()
    last_paint = [0.0]

    def progress(step: int, metrics: dict):
        ret = float(metrics.get("eval/episode_reward", 0.0))
        curve.append((float(step), ret))
        log.info("  %10d steps  return %7.1f", step, ret)
        now = time.time()
        if now - last_paint[0] >= 20:
            last_paint[0] = now
            flyte.report.replace(
                _html(num_envs, step, steps, curve, now - started, params),
                do_flush=True,
            )

    log.info("Brax PPO on %s: %s steps across %s envs", ENV_NAME, f"{steps:,}",
             f"{num_envs:,}")
    make_policy, ppo_params, _ = ppo_train.train(
        environment=env,
        progress_fn=progress,
        network_factory=network_factory,
        wrap_env_fn=wrapper.wrap_for_brax_training,
        seed=seed,
        **{k: v for k, v in params.items() if k != "network_factory"},
    )
    secs = time.time() - started
    mp4, n_frames = _film(env, make_policy, ppo_params)
    log.info("filmed %s frames (%.0f KB)", n_frames, len(mp4) / 1024)
    flyte.report.replace(
        _html(num_envs, steps, steps, curve, secs, params, mp4, n_frames), do_flush=True
    )

    best = max((y for _, y in curve), default=0.0)
    # The number the whole file exists to produce: how many environment steps this
    # configuration needed to get where DreamerV3 got in 500,000.
    target = 900.0
    reached = next((int(s) for s, y in curve if y >= target), None)
    result = {
        "algorithm": "Brax PPO on MJX",
        "env": ENV_NAME,
        "num_envs": num_envs,
        "steps": steps,
        "minutes": round(secs / 60, 1),
        "steps_per_sec": round(steps / secs),
        "return_best": round(best, 1),
        "return_final": round(curve[-1][1], 1) if curve else 0.0,
        # NOT `num_envs * unroll_length`, which is what this reported at first and is
        # wrong by a factor of num_minibatches. Brax collects `batch_size *
        # num_minibatches` unrolls per gradient update, and batch_size is clamped to
        # num_envs above, so the real per-update cost at Playground's 2,048-env default
        # is 983,040 environment steps, not 61,440. 61,440 is the 64-env figure.
        "steps_per_update": (
            min(int(params.get("batch_size", 0)), num_envs)
            * int(params.get("unroll_length", 0))
            * int(params.get("num_minibatches", 1))
        ),
        f"steps_to_{int(target)}": reached,
        "clip": f"{n_frames} frames, {len(mp4) / 1024:.0f} KB" if mp4 else "no clip",
    }
    log.info("result: %s", result)
    return result


def _html(num_envs, step, total, curve, secs, params, mp4=b"", n_frames=0) -> str:
    best = max((y for _, y in curve), default=0.0)
    per_update = (
        min(int(params.get("batch_size", 0)), num_envs)
        * int(params.get("unroll_length", 0))
        * int(params.get("num_minibatches", 1))
    )
    rows = [
        ("Algorithm", "Brax PPO on MJX, model-free"),
        ("Environment", f"{ENV_NAME} (mujoco_playground)"),
        ("Parallel envs", f"{num_envs:,}"),
        ("Steps per update", f"{per_update:,}"),
        ("Progress", f"{step:,} / {total:,} env steps"),
        ("Wall clock", f"{secs / 60:.1f} min"),
        ("Throughput", f"{step / max(secs, 1e-9):,.0f} env steps/s"),
        ("Return, best", f"{best:.1f}"),
    ]
    return (
        f"<h2>Model-free baseline: Brax PPO, {num_envs:,} parallel envs</h2>"
        + reports.panel("Run", reports.table(rows))
        + "<br/>"
        + reports.note(
            "DreamerV3 reached 987 on this task in 500,000 environment steps with 4 "
            "environments. Read the x axis, not the wall clock: PPO gets through these "
            "steps thousands of times faster, and needs far more of them. The point of "
            "sweeping the environment count is that PPO's step total is partly a "
            "function of its own parallelism, so quoting a single ratio without saying "
            "how many environments produced it is not a fair comparison."
        )
        + reports.curve(curve, f"Brax PPO episode return, {num_envs:,} envs")
        + (
            "<br/>" + reports.heading("The trained policy")
            + reports.note(
                "One environment, rendered on the CPU MuJoCo model after training. MJX "
                "cannot render during training because stepping thousands of "
                "environments on the GPU is precisely the trick of not rendering any of "
                "them, but that only ever applied to training. Same seed at every sweep "
                "point, so the clips are comparable to each other."
            )
            + reports.video_html(mp4, f"{n_frames} frames &middot; {num_envs:,} envs", 420)
            if mp4
            else ""
        )
    )


@orch_env.task(report=True)
async def sweep(
    env_counts: list[int] | None = None,
    steps: int = DEFAULT_STEPS,
    seed: int = 0,
) -> dict:
    """The same task and budget at several `num_envs`, run one at a time.

    Sequential on purpose. There is one GPU, so concurrent submissions would queue
    anyway, and worse, any that did overlap would contend and make the throughput
    numbers meaningless.
    """
    counts = env_counts or list(DEFAULT_ENV_COUNTS)
    out = {}
    for n in counts:
        out[str(n)] = await ppo(num_envs=n, steps=steps, seed=seed)
        log.info("%s envs: %s", n, out[str(n)])

    rows = [("DreamerV3, 4 envs, pixels", "987 in 500,000 steps (measured)")]
    for n in counts:
        r = out[str(n)]
        reached = r.get("steps_to_900")
        rows.append((
            f"Brax PPO, {n:,} envs, state",
            f"best {r['return_best']:.0f}, "
            + (f"reached 900 at {reached:,} steps" if reached else "never reached 900")
            + f", {r['minutes']:.0f} min",
        ))
    flyte.report.replace(
        "<h2>Sample efficiency against parallelism</h2>"
        + reports.panel("Steps needed to reach a return of 900", reports.table(rows))
        + reports.note(
            "If the PPO rows move a lot with environment count, then the headline "
            "'model-based needs N times fewer steps' depends on which PPO you compare "
            "against, and the honest number is the smallest-parallelism row."
        ),
        do_flush=True,
    )
    return out


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(sweep))
