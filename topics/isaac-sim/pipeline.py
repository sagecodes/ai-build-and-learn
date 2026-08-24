"""Isaac Sim on Flyte: prove the container simulates, in a pod, with a report.

    flyte run pipeline.py smoke

Runs to the `physical-ai` project (.flyte/config.yaml), same as the MuJoCo demo.

This is the containerised twin of running `smoke_test.py` by hand on the host. It runs
the SAME script, so the two numbers measure the same thing and the comparison is
honest: bare metal on the Spark versus the NGC container under Flyte's device plugin.
That comparison answers "does the GPU actually reach the pod", which is the question
that decides whether any RL work is possible here.

── Why the simulator runs in a CHILD PROCESS ───────────────────────────────────
Not because of imports. `import flyte` and `import isaacsim` coexist fine in one
interpreter once the Dockerfile bakes python.sh's environment in, and the first
version of this file did exactly that, in-process.

It fails at SHUTDOWN, because Kit and Flyte both want to own the asyncio event loop.
SimulationApp.close() has two modes and in a Flyte pod both are wrong:

  fast_shutdown=True   (the default) calls os._exit() internally. The process is gone
                       instantly, Flyte never records a return value, and the run
                       reads as an unexplained pod exit rather than a result.

  fast_shutdown=False  shuts down gracefully by CANCELLING EVERY ASYNCIO TASK in the
                       process. Observed in a real pod, that includes Flyte's own:

                         Cancelling <Task ... coro=<load_and_run_task() ...>>
                         Cancelling <Task ... coro=<Controller.watch_for_errors() ...>>

                       followed by "Cannot enter into task ... while another task is
                       being executed" and a pod that sat Running for 28 minutes
                       holding the GPU until it was aborted by hand.

A child process gives Kit its own event loop to tear down however it likes. Flyte's
loop is never touched, and the task is an ordinary task that returns an ordinary dict.
This is also the shape Isaac Lab wants: its RL entry points are scripts you invoke,
not libraries you drive, so the same pattern extends to training.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import flyte
import flyte.report

# Top-level import, and it has to be. Flyte bundles the modules the task module
# imports at import time; anything it only references by filename never reaches the
# pod. checks.py is safe to import (its isaacsim imports are inside functions),
# smoke_test.py is not (it boots a SimulationApp on import), which is exactly why the
# runner lives in checks.py and this spawns checks.__file__ below.
#
# terrains.py, spark_envs.py and record.py are the exception that proves the rule: they
# cannot be imported here, because this module is loaded by the ORCHESTRATOR too and
# that image has no Isaac Lab. They reach the pod through `train_env.include` instead.
# See the long comment in config.py.
import checks
import train as trainer
from config import gpu_env, orch_env, train_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@gpu_env.task(report=True)
async def isaac_smoke(steps: int = 240, drop_height: float = 2.0) -> dict:
    """Boot Kit headless in the pod, drop a cube, report what the GPU actually did."""
    out_path = Path(tempfile.gettempdir()) / "smoke.json"
    script = Path(checks.__file__)

    # sys.executable, not /isaac-sim/python.sh. They are equivalent here precisely
    # because the Dockerfile baked python.sh's environment into the image, so the
    # interpreter Flyte is running under is already fully set up for Isaac Sim. Going
    # through python.sh would also work; this way the child inherits the pod's env
    # unchanged, which is one less thing that can differ between host and container.
    proc = subprocess.run(
        [sys.executable, str(script), "--steps", str(steps),
         "--drop_height", str(drop_height), "--json", str(out_path)],
        capture_output=True,
        text=True,
        timeout=900,
    )

    # Exit code alone is not enough to explain a failure, and Kit's startup noise means
    # the interesting lines are never near the top. Keep the tail for the report.
    tail = "\n".join(proc.stdout.strip().splitlines()[-25:])
    if not out_path.exists():
        err = "\n".join((proc.stderr or "").strip().splitlines()[-25:])
        _write_report([], "unknown", tail, err, proc.returncode)
        raise RuntimeError(f"smoke_test.py produced no JSON (exit {proc.returncode}). stderr tail:\n{err}")

    data = json.loads(out_path.read_text())
    _write_report(data["checks"], data["gpu"], tail, proc.stderr[-2000:], proc.returncode)
    log.info("isaac_smoke: %s/%s passed on %s", data["passed"], data["total"], data["gpu"])
    return data


def _write_report(checks: list[dict], gpu: str, stdout_tail: str, stderr_tail: str, rc: int) -> None:
    rows = "".join(
        f"<tr><td style='padding:6px 14px'>{'PASS' if c['ok'] else 'FAIL'}</td>"
        f"<td style='padding:6px 14px'><b>{c['name']}</b></td>"
        f"<td style='padding:6px 14px;font-family:monospace'>{c['detail']}</td></tr>"
        for c in checks
    )
    failed = [c["name"] for c in checks if not c["ok"]]
    verdict = "all checks passed" if checks and not failed else f"FAILED: {', '.join(failed) or 'no results'}"
    flyte.report.log(
        f"<h2>Isaac Sim in a Flyte pod</h2>"
        f"<p><b>{len(checks) - len(failed)}/{len(checks)}</b> &mdash; {verdict} "
        f"(smoke_test.py exit {rc})</p>"
        f"<p>GPU as Warp sees it: <code>{gpu}</code></p>"
        f"<table style='border-collapse:collapse'>{rows}</table>"
        f"<h3>stdout (tail)</h3><pre style='font-size:11px'>{stdout_tail}</pre>"
        f"<h3>stderr (tail)</h3><pre style='font-size:11px'>{stderr_tail}</pre>",
        do_flush=True,
    )


@train_env.task(report=True)
async def walk_task(
    task_id: str = "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    num_envs: int = 4096,
    iterations: int = 1500,
) -> dict:
    """Train the quadruped, then film it. Both in this pod, on this GPU.

    Defaults are the measured recipe, not guesses: 4096 envs / 1500 iterations is what
    produced a walking Anymal-C on this box in 27 minutes, mean reward -2.45 -> 11.38.
    """
    t0 = time.monotonic()
    rewards, tail = trainer.train(task_id, num_envs, iterations)
    log.info("trained %s iterations, reward %.2f -> %.2f", len(rewards), rewards[0], rewards[-1])

    # Filmed AFTER training, and never allowed to fail the run: half an hour of
    # training must not be thrown away because a renderer hiccuped.
    clip = trainer.record(task_id)
    secs = time.monotonic() - t0
    trainer.report_final(task_id, num_envs, iterations, rewards, clip, secs, tail)

    return {
        "task": task_id,
        "num_envs": num_envs,
        "iterations": iterations,
        "reward_start": rewards[0],
        "reward_final": rewards[-1],
        "reward_max": max(rewards),
        "minutes": round(secs / 60, 1),
        "clip_kb": round(clip.stat().st_size / 1024) if clip and clip.exists() else 0,
    }


@train_env.task(report=True)
async def parkour_task(
    task_id: str = "Spark-Parkour-Anymal-C-v0",
    num_envs: int = 4096,
    iterations: int = 1500,
    steps: int = 300,
    terrain_level: int = 9,
    terrain_col: int = 12,
) -> dict:
    """Train on our own terrain, then film it with our own cameras.

    Same one-pod shape as `walk_task`, with the two differences that make this episode 2:

      * the task is one of ours (`Spark-*`), registered into NVIDIA's train.py through
        `--external_callback spark_envs.register`, so the terrain is a 13-sub-terrain
        parkour course instead of NVIDIA's six;
      * the replay is record.py, not `play.py --video`, because the Kit viewport capture
        renders everything on this box except the robot.

    `terrain_level` and `terrain_col` choose which patch to film on. They default to the
    hardest row of the parkour course, on gaps, because a random patch is usually a
    boring one. See _place() in record.py.
    """
    t0 = time.monotonic()
    rewards, tail = trainer.train(task_id, num_envs, iterations)
    log.info("trained %s iterations, reward %.2f -> %.2f", len(rewards), rewards[0], rewards[-1])

    clips = trainer.record_clips(task_id, steps=steps, terrain_level=terrain_level, terrain_col=terrain_col)
    secs = time.monotonic() - t0
    trainer.report_parkour(task_id, num_envs, iterations, rewards, clips, secs, tail)

    patch = clips.get("patch") or {}
    return {
        "task": task_id,
        "num_envs": num_envs,
        "iterations": iterations,
        "reward_start": rewards[0],
        "reward_final": rewards[-1],
        "reward_max": max(rewards),
        "minutes": round(secs / 60, 1),
        "filmed_on": patch.get("sub_terrain"),
        "difficulty": patch.get("difficulty"),
        "clips": sorted((clips.get("clips") or {}).keys()),
    }


@orch_env.task(report=True)
async def smoke(steps: int = 240, drop_height: float = 2.0) -> dict:
    """Entry point. CPU-only orchestrator so it cannot deadlock its own GPU child."""
    result = await isaac_smoke(steps=steps, drop_height=drop_height)
    log.info("result: %s", result)
    return result


@orch_env.task(report=True)
async def walk(
    task_id: str = "Isaac-Velocity-Flat-Anymal-C-Direct-v0",
    num_envs: int = 4096,
    iterations: int = 1500,
) -> dict:
    """Teach a robot dog to walk, in a pod, and put the gait in the report.

        flyte run pipeline.py walk
        flyte run pipeline.py walk --iterations 50          # is the plumbing alive?

    CPU-only orchestrator on purpose: it holds its resources for as long as its child
    runs, so asking for the GPU here would deadlock its own GPU child forever.
    """
    result = await walk_task(task_id=task_id, num_envs=num_envs, iterations=iterations)
    log.info("result: %s", result)
    return result


@orch_env.task(report=True)
async def parkour(
    task_id: str = "Spark-Parkour-Anymal-C-v0",
    num_envs: int = 4096,
    iterations: int = 1500,
    steps: int = 300,
    terrain_level: int = 9,
    terrain_col: int = 12,
) -> dict:
    """Teach a robot to cross rough ground, and put what it sees in the report.

        flyte run pipeline.py parkour
        flyte run pipeline.py parkour --iterations 5      # is the plumbing alive?
        flyte run pipeline.py parkour --task_id Spark-Stairs-Unitree-Go2-v0

    Valid task ids are `Spark-{Parkour,Stairs,Stones}-<Robot>-v0` for the ten robots in
    spark_envs.BASE_TASKS. CPU-only orchestrator, same as `walk`: it holds its resources
    while its child runs, so asking for the GPU here deadlocks its own GPU child.
    """
    result = await parkour_task(
        task_id=task_id, num_envs=num_envs, iterations=iterations,
        steps=steps, terrain_level=terrain_level, terrain_col=terrain_col,
    )
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(smoke))
