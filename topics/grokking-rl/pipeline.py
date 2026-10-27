"""Flyte entry points: run any chapter's example on the devbox, or the whole book.

    # One chapter, by its own task (the run is named after it). Extra script flags
    # go through `args`.
    flyte run pipeline.py ch03_planning
    flyte run pipeline.py ch06_control --args "--env sws --episodes 3000"

    # Every chapter at once, one report with a tab per chapter.
    flyte run pipeline.py book

    # Bonus: chapter 12's PPO teaching a humanoid robot to walk. Needs an NVIDIA GPU and
    # ../rl-mujoco deployed once (`flyte deploy --all pipeline.py` there).
    flyte run pipeline.py ch12_bonus_humanoid
    flyte run pipeline.py book --bonus true

Each chapter stays a plain `python -m chNN_x.module` program; this file runs its `main()`,
which returns the report's HTML parts (grids, replays, figures), and adds what it printed.
No devbox? `flyte run --local pipeline.py ch03_planning` still writes the report.
"""

from __future__ import annotations

import asyncio
import contextlib
import html
import io
import json
import logging
import os
import shlex
import tempfile
from pathlib import Path

import flyte
import flyte.remote
import flyte.report
from flyte.io import File

# Every chapter module is imported at top level so `flyte run` bundles it into the
# task's code (a lazy import inside the task dies with ModuleNotFoundError in the pod).
from ch02_mdps import mdps
from ch03_planning import planning
from ch04_bandits import bandits
from ch05_prediction import prediction
from ch06_control import control
from ch07_traces_dyna import traces_dyna
from ch08_nfq import nfq
from ch09_dqn import dqn
from ch10_dueling_per import dueling_per
from ch11_policy_gradient import pg
from ch12_actor_critic import actor_critic
from ch12_bonus_humanoid import humanoid
import viz
from config import cpu_env, light_env, orch_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

CHAPTERS = {
    2: ("Mathematical foundations: Markov decision processes (MDPs)", mdps),
    3: ("Planning: policy and value iteration", planning),
    4: ("Exploration: multi-armed bandits", bandits),
    5: ("Prediction: Monte Carlo (MC) vs temporal difference (TD)", prediction),
    6: ("Control: Monte Carlo, SARSA (state-action-reward-state-action), Q-learning, Double Q", control),
    7: ("Traces and planning: SARSA(lambda), Q(lambda), Dyna-Q", traces_dyna),
    8: ("Deep RL begins: a neural network instead of a table (NFQ on CartPole)", nfq),
    9: ("Stable deep Q-learning: DQN and Double DQN on LunarLander", dqn),
    10: ("Sample efficiency: dueling networks and prioritized replay", dueling_per),
    11: ("Learning the policy directly: REINFORCE to actor-critic", pg),
    12: ("Continuous control: DDPG, TD3, SAC, PPO, and a running HalfCheetah", actor_critic),
}

def _chapter_html(number: int, text: str, parts: list[str]) -> str:
    title, _ = CHAPTERS[number]
    return (
        f"<h2>Chapter {number}: {html.escape(title)}</h2>"
        + "".join(parts)
        + viz.details("Raw console output (what `python -m ...` prints)", viz.text(text))
    )


async def _run_chapter(number: int, args: str) -> File:
    """Run one chapter's example inside the current task. Returns its report section as
    an HTML file: a File, not a str, because chapter 7's section (sliders and replays on
    the 8x8 lake) is several MB, and a literal that size fails with UnableToConvertError.
    """
    if number not in CHAPTERS:
        raise ValueError(f"no example for chapter {number}; have {sorted(CHAPTERS)}")
    _, module = CHAPTERS[number]

    out = tempfile.mkdtemp(prefix=f"ch{number:02d}-")
    os.environ["GROKKING_OUT"] = out
    for m in (bandits, prediction, control, traces_dyna, nfq, dqn, dueling_per, pg, actor_critic):
        m.OUT = Path(out)  # modules read the env var at import, which already happened

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        parts = await asyncio.to_thread(module.main, shlex.split(args))
    text = buf.getvalue()
    log.info(text)

    section = _chapter_html(number, text, parts)
    await flyte.report.replace.aio(viz.STYLE + section)
    await flyte.report.flush.aio()
    path = Path(out) / "section.html"
    path.write_text(section)
    return await File.from_local(str(path))


# One task per chapter, so a run is named after what it is ("ch10_dueling_per") instead
# of "chapter". Each takes the same extra CLI flags as the script, as one string:
#     flyte run pipeline.py ch06_control --args "--env sws --episodes 3000"

@light_env.task(report=True)
async def ch02_mdps(args: str = "") -> File:
    return await _run_chapter(2, args)


@light_env.task(report=True)
async def ch03_planning(args: str = "") -> File:
    return await _run_chapter(3, args)


@light_env.task(report=True)
async def ch04_bandits(args: str = "") -> File:
    return await _run_chapter(4, args)


@light_env.task(report=True)
async def ch05_prediction(args: str = "") -> File:
    return await _run_chapter(5, args)


@light_env.task(report=True)
async def ch06_control(args: str = "") -> File:
    return await _run_chapter(6, args)


@light_env.task(report=True)
async def ch07_traces_dyna(args: str = "") -> File:
    return await _run_chapter(7, args)


@light_env.task(report=True)
async def ch08_nfq(args: str = "") -> File:
    return await _run_chapter(8, args)


@cpu_env.task(report=True)
async def ch09_dqn(args: str = "") -> File:
    return await _run_chapter(9, args)


@cpu_env.task(report=True)
async def ch10_dueling_per(args: str = "") -> File:
    return await _run_chapter(10, args)


@cpu_env.task(report=True)
async def ch11_policy_gradient(args: str = "") -> File:
    return await _run_chapter(11, args)


@cpu_env.task(report=True)
async def ch12_actor_critic(args: str = "") -> File:
    return await _run_chapter(12, args)


# The bonus trains with rl-mujoco's tasks (MJX + Brax PPO on the GPU), called by name so
# this image never needs jax or brax. Resolved on first call, so importing this file
# works without them deployed.
g1_train = flyte.remote.Task.get("g1-train.train_policy", auto_version="latest")
g1_render = flyte.remote.Task.get("g1-render.render_replay", auto_version="latest")


@orch_env.task(report=True)
async def ch12_bonus_humanoid(
    presets: list[str] | None = None,
    num_timesteps: int = 200_000_000,
    num_envs: int = 4096,
    replay_steps: int = 500,
    checkpoints: list[str] | None = None,
) -> File:
    """Chapter 12's PPO at robot scale: train a Unitree G1 humanoid to walk with each
    reward preset (default baseline and high-step, ~50 min each on one GPU, run one
    after the other), then film every policy next to a random one.

    `checkpoints` reuses rl-mujoco `train_policy` outputs (their s3:// URIs) instead of
    training; pass `--presets '[]'` with it to skip training entirely.
    """
    keys = ["baseline", "high-step"] if presets is None else presets
    title = "<h2>Chapter 12 bonus: PPO teaches a humanoid robot to walk</h2>"

    async def status(msg):
        await flyte.report.replace.aio(viz.STYLE + title + f"<p>{html.escape(msg)}</p>")
        await flyte.report.flush.aio()

    ckpt_files = [File.from_existing_remote(uri) for uri in checkpoints or []]
    # One GPU: train the presets one after another rather than queueing them all at once.
    for i, key in enumerate(keys):
        await status(f"Training preset {key} ({i + 1}/{len(keys)}): {num_timesteps:,} steps "
                     f"across {num_envs:,} simulated robots. The training task's own report "
                     "shows the live reward curve and mid-training clips.")
        ckpt_files.append(await g1_train(num_timesteps=num_timesteps, num_envs=num_envs, preset=key))
    if not ckpt_files:
        raise ValueError("nothing to show: pass presets to train or checkpoints to reuse")

    await status(f"Filming {len(ckpt_files)} trained policies and a random one...")
    replays = await asyncio.gather(
        g1_render(checkpoint=ckpt_files[0], steps=replay_steps, policy="random"),
        *[g1_render(checkpoint=c, steps=replay_steps, policy="trained") for c in ckpt_files],
    )

    def load(path):
        with open(path) as f:
            return json.load(f)

    random_replay = load(await replays[0].download())
    runs = [(humanoid.read_checkpoint(await c.download()), load(await r.download()))
            for c, r in zip(ckpt_files, replays[1:])]

    section = title + "".join(humanoid.report(runs, random_replay, replay_steps))
    await flyte.report.replace.aio(viz.STYLE + section)
    await flyte.report.flush.aio()
    path = Path(tempfile.mkdtemp(prefix="ch12-bonus-")) / "section.html"
    path.write_text(section)
    return await File.from_local(str(path))


TASKS = {2: ch02_mdps, 3: ch03_planning, 4: ch04_bandits, 5: ch05_prediction, 6: ch06_control,
         7: ch07_traces_dyna, 8: ch08_nfq, 9: ch09_dqn, 10: ch10_dueling_per,
         11: ch11_policy_gradient, 12: ch12_actor_critic}


@cpu_env.task(report=True)
async def chapter(number: int, args: str = "") -> File:
    """Any chapter by number. Handy in scripts; the run is just named "chapter", so for
    a named run use the chapter's own task (e.g. `flyte run pipeline.py ch09_dqn`)."""
    return await _run_chapter(number, args)


@orch_env.task(report=True)
async def book(chapters: list[int] | None = None, bonus: bool = False) -> str:
    """Run every chapter's example in parallel; one report, one tab per chapter.
    `bonus` adds the chapter 12 humanoid as a last tab (GPU, roughly two more hours)."""
    numbers = chapters or sorted(CHAPTERS)
    await flyte.report.replace.aio(
        viz.STYLE + f"<h2>Grokking Deep RL</h2><p>Running chapters {numbers}...</p>"
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(*[TASKS[n]() for n in numbers],
                                 *([ch12_bonus_humanoid()] if bonus else []))
    sections = [Path(await f.download()).read_text() for f in files]
    if bonus:
        bonus_section = sections.pop()

    index = "".join(
        f"<li>Chapter {n}: {html.escape(CHAPTERS[n][0])}</li>" for n in numbers
    )
    for n, section in zip(numbers, sections):
        flyte.report.get_tab(f"Chapter {n}").replace(viz.STYLE + section)
    if bonus:
        flyte.report.get_tab("Bonus: humanoid").replace(viz.STYLE + bonus_section)
        index += "<li>Bonus: chapter 12's PPO teaches a humanoid robot to walk</li>"
    await flyte.report.replace.aio(
        viz.STYLE + "<h2>Grokking Deep RL: chapter examples</h2>"
        f"<p>Each chapter has its own tab.</p><ul>{index}</ul>"
    )
    await flyte.report.flush.aio()
    return f"ran chapters {numbers}"
