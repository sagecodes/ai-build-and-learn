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
    flyte.report.replace(
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
    # The terrain-level curve is discarded here on purpose: this task is the episode-1
    # FLAT demo, which has no terrain curriculum, so the list is always empty.
    rewards, _levels, _air, tail = trainer.train(task_id, num_envs, iterations)
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
    # Steps are 50 Hz env steps, not frames of video: 600 is twelve seconds of robot
    # time, and 1000 is the episode limit (episode_length_s = 20.0).
    steps: int = 600,
    terrain_level: int = -1,
    terrain_col: int = 12,
    timeline: int = 4,
    snapshot_every: int = 250,
    snapshot_steps: int = 250,
    snapshot_level: int = 4,
) -> dict:
    """Train on our own terrain, then film it with our own cameras.

    Same one-pod shape as `walk_task`, with the two differences that make this episode 2:

      * the task is one of ours (`Spark-*`), registered into NVIDIA's train.py through
        `--external_callback spark_envs.register`, so the terrain is a 13-sub-terrain
        parkour course instead of NVIDIA's six;
      * the replay is record.py, not `play.py --video`, because the Kit viewport capture
        renders everything on this box except the robot.

    `terrain_level` and `terrain_col` choose which patch to film on, because a random
    patch is usually a boring one. `terrain_level=-1`, the default, means "wherever the
    curriculum got to"; pass a row number to override. See _place() in record.py.

    `snapshot_every` films the CURRENT policy every N iterations and puts the clip in
    the live report, so a three-hour run is watchable while it runs instead of only
    afterwards. Set 0 to turn it off. Those clips are filmed on a fixed row
    (`snapshot_level`) so they are comparable to each other, which is the opposite of
    what the hero shot wants. See Snapshotter in train.py.
    """
    t0 = time.monotonic()
    snap = (
        trainer.Snapshotter(task_id, every=snapshot_every, steps=snapshot_steps,
                            level=snapshot_level, col=terrain_col)
        if snapshot_every > 0
        else None
    )
    try:
        rewards, levels, airtime, tail = trainer.train(task_id, num_envs, iterations, snap=snap)
    finally:
        # Before anything else touches the GPU: the daemon is holding a booted Kit, and
        # the final render is about to want the box to itself.
        if snap is not None:
            snap.close()
    log.info("trained %s iterations, reward %.2f -> %.2f, terrain row %.2f -> %.2f",
             len(rewards), rewards[0], rewards[-1],
             levels[0] if levels else -1, levels[-1] if levels else -1)

    # Film where the policy actually GOT TO, not where we hoped it would.
    #
    # terrain_level=-1 means "ask the curriculum". Hardcoding row 9 is right when the
    # policy can do row 9 and embarrassing when it cannot: the clip is then a trained
    # robot falling into a 0.6 m trench it never learned to cross, which reads as a
    # broken demo rather than an honest difficulty ceiling. The mean terrain row at the
    # end of training IS the answer to "how hard can this policy go", so use it.
    level = terrain_level
    if level < 0:
        level = round(levels[-1]) if levels else 0
        log.info("filming on terrain row %s (curriculum reached %.2f)", level, levels[-1] if levels else 0.0)

    clips = trainer.record_clips(task_id, steps=steps, terrain_level=level,
                                 terrain_col=terrain_col, timeline=timeline)
    secs = time.monotonic() - t0
    snaps = snap.clips if snap else []
    trainer.report_parkour(task_id, num_envs, iterations, rewards, levels, clips, secs, tail,
                           snaps=snaps, snap_level=snapshot_level if snap else None,
                           airtime=airtime)

    # Get the trained policy OUT of the pod before it evaporates.
    #
    # rsl_rl writes checkpoints to /tmp/isaac-run inside the container, and when the task
    # finishes that filesystem goes with it. An 87-minute run whose only surviving output
    # is an embedded mp4 cannot be re-filmed, compared against, or deployed: the actual
    # product of training is the weights. Learned the expensive way, once.
    # The blob URI travels in the dict rather than the File object itself: this task is
    # annotated `-> dict`, and an untyped dict is not a place Flyte can serialise a File.
    # The upload has already happened either way, so the path is all anyone needs to
    # fetch it later.
    policy = await trainer.export_policy(task_id)

    patch = clips.get("patch") or {}
    # The jump measurement, taken off the contact sensor during the replay rather than
    # inferred from the reward. On a `Spark-Leap-*` run this is the result; everything
    # else in this dict is a training signal that only argues for it. See _flight_phases
    # in record.py.
    flight = patch.get("flight") or {}
    return {
        "policy": policy.path if policy else None,
        "task": task_id,
        "num_envs": num_envs,
        "iterations": iterations,
        "reward_start": rewards[0],
        "reward_final": rewards[-1],
        "reward_max": max(rewards),
        "minutes": round(secs / 60, 1),
        # The curriculum result. On rough terrain this matters more than the reward:
        # it is the mean difficulty ROW the envs ended up on, out of num_rows.
        "terrain_row_start": round(levels[0], 2) if levels else None,
        "terrain_row_final": round(levels[-1], 2) if levels else None,
        "filmed_on": patch.get("sub_terrain"),
        "difficulty": patch.get("difficulty"),
        "clips": sorted((clips.get("clips") or {}).keys()),
        "snapshots": len(snaps),
        # Seconds of unbroken flight, and metres covered during it.
        "flight_s": flight.get("longest_s"),
        "flight_span_m": flight.get("longest_span_m"),
        "airborne_frac": flight.get("airborne_frac"),
        # Where the air-time reward term ended up. Positive means the policy is taking
        # flights rather than strides; see _AIR_RE in train.py.
        "airtime_term_final": round(airtime[-1], 4) if airtime else None,
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
    # Steps are 50 Hz env steps, not frames of video: 600 is twelve seconds of robot
    # time, and 1000 is the episode limit (episode_length_s = 20.0).
    steps: int = 600,
    terrain_level: int = -1,
    terrain_col: int = 12,
    timeline: int = 4,
    snapshot_every: int = 250,
    snapshot_steps: int = 250,
    snapshot_level: int = 4,
) -> dict:
    """Teach a robot to cross rough ground, and put what it sees in the report.

        flyte run pipeline.py parkour
        flyte run pipeline.py parkour --iterations 5      # is the plumbing alive?
        flyte run pipeline.py parkour --iterations 3000 --snapshot_every 200
        flyte run pipeline.py parkour --task_id Spark-Stairs-Go2-v0

    Valid task ids are `Spark-{Parkour,Stairs,Stones,Leap}-<Robot>-v0`, where <Robot> is a
    value from spark_envs.ROBOT_LABELS and NOT the stock task's spelling: it is
    `Spark-Stairs-Go2-v0`, not `Spark-Stairs-Unitree-Go2-v0`. `leap` also carries a reward
    profile, so prefer the `leap` entry point below, which sets the defaults that go with
    it. CPU-only orchestrator, same as `walk`: it holds its resources
    while its child runs, so asking for the GPU here deadlocks its own GPU child.
    """
    result = await parkour_task(
        task_id=task_id, num_envs=num_envs, iterations=iterations,
        steps=steps, terrain_level=terrain_level, terrain_col=terrain_col, timeline=timeline,
        snapshot_every=snapshot_every, snapshot_steps=snapshot_steps,
        snapshot_level=snapshot_level,
    )
    log.info("result: %s", result)
    return result


@orch_env.task(report=True)
async def leap(
    task_id: str = "Spark-Leap-Go2-v0",
    num_envs: int = 4096,
    # Longer than the parkour default, and the reason is the curriculum rather than the
    # policy. `leap` starts every env on row 1 of 12 (see _leap_profile) instead of row 5
    # of 10, so it has more rungs to climb and starts further down them. The gait is
    # usually there by ~1000; everything after that is buying trench width.
    iterations: int = 4000,
    steps: int = 600,
    terrain_level: int = -1,
    # Column 5 of 20. The leap terrain gives its first 55% of columns to `gaps`, which is
    # columns 0-10, so this reliably films a trench rather than whichever sub-terrain the
    # robot happened to spawn on. terrains.FILM_COLS is the same number, and cannot be
    # imported here: this module is loaded by the orchestrator too, and that image has no
    # Isaac Lab in it. See the note in config.py.
    terrain_col: int = 5,
    timeline: int = 4,
    snapshot_every: int = 250,
    snapshot_steps: int = 250,
    # Row 4 of 12 is a ~0.20 m trench: wide enough that clearing it is unambiguous on
    # camera, narrow enough that a half-trained policy has a chance. Fixed for every
    # snapshot so the strip compares policies rather than terrain.
    snapshot_level: int = 4,
) -> dict:
    """Teach the dog to jump, which is a different task from teaching it to walk.

        flyte run pipeline.py leap
        flyte run pipeline.py leap --iterations 5              # is the plumbing alive?
        flyte run pipeline.py leap --task_id Spark-Leap-A1-v0

    The `parkour` task above trains velocity tracking on rough ground, and it will never
    produce a jump no matter how long it runs. The reward set it inherits contains
    `lin_vel_z_l2` at weight -2.0, a squared penalty on vertical velocity, which prices a
    trench crossing at about -14.6 against task rewards that cap at 2.25. A three-hour run
    of it walks to the lip of every gap and stops, and that is the correct answer to the
    question it was asked.

    This entry point changes the question. Two things move together and neither works
    alone:

      * `terrains.SPARK_LEAP_CFG`, a gap-dominated course with a finer curriculum and a
        first rung (a 5 cm crack) that an untrained policy clears by accident;
      * `spark_envs.REWARD_PROFILES["leap"]`, which drops the vertical-velocity penalty to
        -0.05, gives the air-time term a threshold just above a walking stride and a weight
        that is not effectively zero, halves the two smoothness penalties that a leap
        maximises, and widens the forward command to 2 m/s so there is a run-up.

    Note what is NOT here: nothing rewards jumping directly. Velocity tracking already
    paid enormously for crossing a gap, because a robot stopped at the edge earns nothing
    on a 1.5-weighted term for the rest of its twenty-second episode. The profile removes
    the thing that was extinguishing the attempts.

    The result to read is `flight_s` in the returned dict and the "Did it actually jump?"
    section of the report: seconds of unbroken flight measured off the contact sensor
    during the replay. Reward and terrain row both climb whether or not the robot ever
    leaves the ground; that number does not.
    """
    result = await parkour_task(
        task_id=task_id, num_envs=num_envs, iterations=iterations,
        steps=steps, terrain_level=terrain_level, terrain_col=terrain_col, timeline=timeline,
        snapshot_every=snapshot_every, snapshot_steps=snapshot_steps,
        snapshot_level=snapshot_level,
    )
    log.info("result: %s", result)
    return result


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(smoke))
