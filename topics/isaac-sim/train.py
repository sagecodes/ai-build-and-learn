"""Train a quadruped to walk in a Flyte pod, and put the gait in the report.

The GPU task is `walk_task` below. It does two things in ONE pod, deliberately:

    train   Isaac Lab + rsl_rl PPO, thousands of envs on the GB10
    record  replay the trained policy through the RTX renderer -> mp4 -> report

The MuJoCo demo next door splits these into two tasks because MJX has no renderer, so
the replay has to happen somewhere else. Isaac Sim renders perfectly well, and this
box has exactly one GPU, so splitting would mean shipping a checkpoint through blob
storage and then queueing for the same GPU again. One task is both simpler and faster.

── Everything here runs Isaac Lab as a CHILD PROCESS ───────────────────────────
Same reason as the smoke test, and it is not about imports: Kit's shutdown cancels
every asyncio task in the process, including Flyte's own runtime. See pipeline.py.
It also happens to be how Isaac Lab is designed to be driven; its RL entry points are
scripts, not libraries.

The upside is that the reward curve can be streamed: we read rsl_rl's stdout line by
line and push the curve into the live Flyte report as it trains, so you can watch the
number climb instead of waiting half an hour to find out it learned to sit down.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

import flyte.report
from flyte.io import File

log = logging.getLogger(__name__)
# Explicit, and load-bearing. pipeline.py calls `logging.basicConfig(level=WARNING)` and
# then raises the level on ITS OWN module logger only, so this one inherits WARNING from
# the root and every log.info() below is silently dropped. That is how a 100-minute
# training run came to have exactly one line in `kubectl logs`.
log.setLevel(logging.INFO)

# Where Isaac Lab lives in the training image (see Dockerfile.train).
ISAACLAB = Path("/isaac-lab")
RSL_RL = ISAACLAB / "scripts" / "reinforcement_learning" / "rsl_rl"

# Our own modules, wherever Flyte unpacked them in the pod. Resolved from __file__
# rather than hardcoded: the task code does not live at a fixed path.
HERE = Path(__file__).resolve().parent

# rsl_rl writes checkpoints to ./logs/rsl_rl/<experiment>/<timestamp>/ RELATIVE TO CWD.
# /isaac-lab is chmod a+rX in the image (readable, not writable), so running from
# there would fail on the first checkpoint save. Everything runs from here instead.
WORKDIR = Path("/tmp/isaac-run")

_REWARD_RE = re.compile(r"Mean reward:\s*(-?[\d.]+)")
_ITER_RE = re.compile(r"Learning iteration\s+(\d+)/(\d+)")

# The number that says whether the PARKOUR worked, as opposed to whether the walking
# worked. rsl_rl prints the curriculum manager's extras every iteration:
#     Curriculum/terrain_levels: 3.4065
# It is the mean difficulty row the envs are currently on, out of `num_rows`. Reward can
# climb simply because the policy got good at the easy rows it started on; this rising is
# the robot being PROMOTED onto ground it previously fell off.
_LEVEL_RE = re.compile(r"Curriculum/terrain_levels:\s*([\d.]+)")

# Lines that are pure fallout, not the fault. When Kit's renderer dies it prints three of
# these per allocation it then fails to free, hundreds in a row, and they all carry the
# same message: something already went wrong. Keeping them in a bounded tail buffer means
# the buffer holds nothing but copies of the consequence, and the CAUSE, which is on the
# handful of lines just before the first one, is pushed out. Filtered out of the tail and
# counted instead. See _clean_tail.
_NOISE_RE = re.compile(
    r"carb\.cudainterop|omni\.rtx\] CUDA error in freeAsync|Failed to free CUDA device memory"
    r"|IHydraTexture refResource had no GPU foundation"
)

# The other number that says whether the JUMP worked. Isaac Lab's RewardManager logs every
# term separately as `Episode_Reward/<name>` (reward_manager.py:120), rsl_rl prints any
# extras key containing a slash verbatim (rsl_rl/utils/logger.py:176), so the training log
# already carries one line per reward term every iteration and nobody reads them:
#
#     Episode_Reward/feet_air_time: 0.0641
#
# On the `leap` profile that term is (air_time - 0.25 s) summed over feet at each touchdown,
# so it is negative while the policy shuffles, crosses zero when strides lengthen past a
# walk, and climbs once the robot is actually leaving the ground. Watching it turn positive
# is watching the jump appear, three hundred iterations before it is big enough to see in a
# clip. Absolute scale is not meaningful (the manager multiplies every term by step_dt);
# the sign and the shape are.
_AIR_RE = re.compile(r"Episode_Reward/feet_air_time:\s*(-?[\d.]+)")

# Every reward and termination term, not just the air-time one. rsl_rl already prints all
# of them once per iteration and this function already reads every line, so capturing them
# costs a regex and turns the pod log from "the number went down" into "the number went
# down because dof_acc_l2 was -1250 on the iteration a quarter of the robots fell into a
# trench". The first leap run was diagnosed by inference from four scraped numbers; there
# was no reason for that to be hard.
# The headline number for a `vault` run, and the reason that profile exists. `gap_flight`
# pays forward speed while every foot is off the ground AND the height scanner sees a
# trench underneath (spark_envs.gap_flight), so unlike `feet_air_time` it CANNOT be earned
# by a long stride, a bounce on flat ground, or anything else that is not a crossing.
#
# It starts at exactly 0.0 and stays there for as long as the policy shuffles to the lip,
# because a robot that never leaves the ground over a hole cannot collect a single step of
# it. The iteration it becomes non-zero is the iteration the jump was invented. That is a
# sharper signal than the air-time term, which on `leap` sat at -0.0033 for 4750
# iterations and left it genuinely ambiguous whether anything was happening.
#
# Absent from a `leap` or `parkour` log, which is fine: the series stays empty and the
# chart it feeds does not render.
_GAP_RE = re.compile(r"Episode_Reward/gap_flight:\s*(-?[\d.eE+]+)")

_TERM_RE = re.compile(r"(Episode_(?:Reward|Termination)/[\w.]+):\s*(-?[\d.eE+]+)")


def _child_env() -> dict[str, str]:
    """Environment for the Isaac child processes, with our own modules importable.

    Isaac Lab's `train.py --external_callback spark_envs.register` does a plain
    `importlib` on that string, and the child runs from WORKDIR, not from here, so
    without this it dies with ModuleNotFoundError before Kit even starts.

    APPENDED, never assigned: the training image sets a 15-entry PYTHONPATH in the
    Dockerfile that every isaacsim import depends on. Replacing it breaks the container.
    """
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{HERE}:{existing}" if existing else str(HERE)
    return env


def _run_streaming(argv: list[str], on_line, timeout: float | None = None) -> int:
    """Run a child process, calling on_line for each stdout line. Returns exit code.

    `timeout` is a WATCHDOG, not a `proc.wait(timeout=...)`, and it has to be: this
    function blocks on the stdout iterator, so a child that wedges with the pipe open
    never reaches wait() and no amount of waiting there would help. A timer thread that
    kills the process is the only thing that unblocks the read.

    Left as None for training, deliberately. Training legitimately runs for half an hour
    and a wrong guess there would kill a good run. It is set for the replay, where a
    wedged Kit would otherwise hold this box's single GPU until someone aborts the pod
    by hand.
    """
    proc = subprocess.Popen(
        argv,
        cwd=str(WORKDIR),
        env=_child_env(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None

    watchdog = None
    if timeout is not None:
        def _kill() -> None:
            log.warning("child exceeded %.0fs, killing: %s", timeout, argv[1])
            proc.kill()

        watchdog = threading.Timer(timeout, _kill)
        watchdog.daemon = True
        watchdog.start()

    try:
        for line in proc.stdout:
            on_line(line.rstrip())
        return proc.wait()
    finally:
        if watchdog is not None:
            watchdog.cancel()


def _ckpt_iter(ckpt: Path) -> int:
    """Iteration number out of `model_1500.pt`. -1 if it is not that shape."""
    stem = ckpt.stem
    return int(stem.split("_")[-1]) if stem.startswith("model_") and stem.split("_")[-1].isdigit() else -1


def _newest_checkpoint(min_age: float = 5.0) -> Path | None:
    """The newest `model_*.pt` rsl_rl has FINISHED writing, or None if there is none yet.

    The age filter is the whole reason this is not a one-line glob. rsl_rl saves every 50
    iterations and we go looking on iteration boundaries, so the newest file is sometimes
    a torch.save that is still in flight; loading it raises somewhere deep in unpickling.
    Five seconds is generous for a 5 MB MLP and costs nothing: if the newest file is too
    fresh, the caller simply asks again on the next iteration.
    """
    root = WORKDIR / "logs" / "rsl_rl"
    now = time.time()
    done = [p for p in root.glob("*/*/model_*.pt") if now - p.stat().st_mtime > min_age]
    return max(done, key=lambda p: p.stat().st_mtime, default=None)


class Snapshotter:
    """Films the policy WHILE it is training, and drops the clips into the live report.

    This is the Isaac answer to what the MuJoCo demo next door does with brax's
    `policy_params_fn`: a reward curve tells you a number went up, and only footage tells
    you it went up for the right reason. Watching a three-hour run climb from -3 to 12
    with no picture until the very end is how you discover at minute 180 that the robot
    learned to shuffle on its knees.

    Isaac cannot use brax's shape, because training here is a CHILD PROCESS (see the
    header of pipeline.py) and its weights are not ours to reach into. What it does have
    is rsl_rl writing `model_*.pt` every 50 iterations, which is a perfectly good stream
    of live policies as long as something is willing to render them.

    ── Why a daemon and not one process per clip ───────────────────────────────────
    A fresh `record.py` costs ~2 min before the first frame: Kit boots, extensions load,
    the 200-patch terrain mesh is generated. Doing that eight times over a run, on the
    SAME GPU the training is using, is minutes of contention bought for nothing. So
    `record.py --serve` boots that once and then sits blocked on stdin; each snapshot is
    a `runner.load()` and a short roll. Idle between clips it costs GPU memory (single
    figures of GB, against a 119 GB unified pool) and no compute.

    Everything here is best-effort by construction. A snapshot that fails, a daemon that
    dies, a checkpoint caught mid-write: all of it logs and moves on, because none of it
    is worth losing a training run over.
    """

    def __init__(
        self, task_id: str, every: int, steps: int = 250, level: int = 4, col: int = 12,
        width: int = 480, height: int = 270, fps: int = 50, crf: int = 32,
    ) -> None:
        # The Play variant, same derivation as record_clips(): a caller cannot ask for a
        # snapshot of a different terrain than the one being trained on.
        self.play_id = task_id.replace("-v0", "-Play-v0")
        self.every = every
        # Steps are 50 Hz env steps, not frames: 250 is five seconds of robot time. The
        # first version of this filmed 90, which is 1.8 s, and every clip read as a robot
        # that took two steps and gave up. The episode limit is 1000 steps (20 s).
        self.steps = steps
        # A FIXED patch for every snapshot, unlike the hero shot which follows the
        # curriculum. The strip is only worth looking at if the ground does not change
        # underneath it: same row, same column, same camera, only the policy differs.
        self.level = level
        self.col = col
        self.size = (width, height)
        self.fps = fps
        # A MUCH coarser crf than the hero shot, and the number is measured rather than
        # taste. These are path-traced frames, so they are full of sampling noise, and
        # x264 spends enormous bitrate preserving noise: the first snapshot filmed at
        # 640x360 crf 26 came out at 6.3 MB for five seconds. Each of these is base64'd
        # into the live report on EVERY repaint, so at five clips that is a 42 MB page
        # rewritten every ninety seconds for three hours. See the note in record.py's
        # _encode for the measurements.
        self.crf = crf
        self.out = WORKDIR / "snapshots"
        self.clips: list[dict] = []
        self._proc: subprocess.Popen | None = None
        self._pending = 0
        self._dead = False
        self._stopping = False
        self._last: str | None = None
        self._tail: list[str] = []

    def _start(self) -> None:
        """Boot the daemon. Lazily, on the first request, and never twice.

        Lazily because booting a second Kit at t=0 lands on top of the training process
        doing its own boot: same shader caches, same extension registry, same 8 CPUs.
        By the time the first checkpoint is worth filming, that is all long done.
        """
        width, height = self.size
        argv = [
            sys.executable, str(HERE / "record.py"), "--task", self.play_id, "--serve",
            "--logs", str(WORKDIR / "logs" / "rsl_rl"), "--out", str(self.out),
            "--steps", str(self.steps), "--fps", str(self.fps), "--crf", str(self.crf),
            "--width", str(width), "--height", str(height),
            # `performance`, not the `quality` the hero shot uses. These are 640x360
            # thumbnails rendered next to a live training run; a third RT bounce buys
            # nothing at this size and the GPU has better things to do.
            "--rendering_mode", "performance",
            "--terrain_level", str(self.level), "--terrain_col", str(self.col),
        ]
        log.info("snapshot daemon starting: %s on row %s", self.play_id, self.level)
        self._proc = subprocess.Popen(
            argv, cwd=str(WORKDIR), env=_child_env(),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self) -> None:
        """Read the daemon's stdout on a thread, collecting clips as they land.

        A thread and not the main loop, because the main loop is busy consuming rsl_rl's
        stdout: blocking there for the thirty seconds a render takes would fill the
        training process's stdout pipe and stall the training itself.
        """
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            line = line.rstrip()
            self._tail.append(line)
            del self._tail[:-40]
            if not line.startswith("{"):
                continue
            try:
                res = json.loads(line)
            except ValueError:
                continue
            self._pending = max(0, self._pending - 1)
            if res.get("ok"):
                self.clips.append({"iteration": res["iteration"], "clip": res["clip"],
                                   "frames": res.get("frames", 0),
                                   # record.py measures the longest flight phase in every
                                   # clip it films. Carried through here so the snapshot
                                   # strip can be captioned with a number instead of
                                   # leaving "is it jumping yet" to the eye.
                                   "flight": res.get("flight")})
                fl = res.get("flight") or {}
                log.info("snapshot iter %s: %s frames (%.1fs) on row %s -> %s, %s KB"
                         " | longest flight %.2fs over %.2fm",
                         res["iteration"], res.get("frames"),
                         res.get("frames", 0) / 50, res.get("row"),
                         Path(res["clip"]).name, res.get("kb"),
                         fl.get("longest_s", 0.0), fl.get("longest_span_m", 0.0))
            else:
                log.warning("snapshot iter %s failed: %s", res.get("iteration"), res.get("error"))
        self._dead = True
        if self._stopping:
            log.info("snapshot daemon stopped after %s clips", len(self.clips))
        else:
            log.warning("snapshot daemon exited; no more mid-training clips. tail:\n%s",
                        "\n".join(self._tail[-15:]))

    def request(self, ckpt: Path, iteration: int) -> None:
        """Ask for a clip of `ckpt`. Returns immediately; the clip lands on the thread."""
        if self._dead:
            return
        # rsl_rl saves every 50 iterations, so with a short interval (or while the
        # daemon is still booting) the newest checkpoint can be one already filmed.
        # Two identical clips in the strip is worse than none.
        if str(ckpt) == self._last:
            return
        if self._proc is None:
            self._start()
        if self._pending:
            # Rendering is slower than the interval. Skipping keeps the report current
            # instead of queueing clips that arrive an hour after they were asked for.
            log.info("snapshot still rendering, skipping iteration %s", iteration)
            return
        cmd = {"checkpoint": str(ckpt), "iteration": iteration,
               "steps": self.steps, "level": self.level, "col": self.col}
        try:
            assert self._proc is not None and self._proc.stdin is not None
            self._proc.stdin.write(json.dumps(cmd) + "\n")
            self._proc.stdin.flush()
            self._pending += 1
            self._last = str(ckpt)
        except (BrokenPipeError, ValueError, OSError) as exc:
            log.warning("snapshot daemon unreachable (%s); continuing without clips", exc)
            self._dead = True

    def close(self, timeout: float = 420.0) -> None:
        """Shut the daemon down before the final render wants the GPU to itself."""
        if self._proc is None:
            return
        self._stopping = True
        try:
            if self._proc.stdin is not None:
                self._proc.stdin.write('{"stop": true}\n')
                self._proc.stdin.flush()
                self._proc.stdin.close()
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            # It finishes the clip it is on first, which is why this waits minutes
            # rather than seconds.
            self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            log.warning("snapshot daemon did not stop in %.0fs; killing it", timeout)
            self._proc.kill()
        self._dead = True


def train(
    task_id: str, num_envs: int, iterations: int, snap: "Snapshotter | None" = None
) -> tuple[list[float], list[float], list[float], list[str]]:
    """Run rsl_rl PPO. Returns (reward, terrain-level, air-time curves, tail of the log).

    Our own `Spark-*` tasks are registered through `--external_callback`, which is Isaac
    Lab's supported hook for exactly this (see the header of spark_envs.py). Stock
    `Isaac-*` tasks are left alone, so episode 1's flat Anymal-C run is byte-for-byte
    the command it always was.

    The terrain-level curve is empty for flat tasks, which have no curriculum to report.
    The air-time curve is populated for every task that keeps a `feet_air_time` reward
    term, which is all of them, but it only means much on `Spark-Leap-*` where the term
    has been given a weight that is not effectively zero. See _AIR_RE.
    """
    WORKDIR.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []
    levels: list[float] = []
    airtime: list[float] = []
    gapflight: list[float] = []
    tail: list[str] = []
    # Term name -> its value as of the most recent iteration. Overwritten rather than
    # accumulated: this is a snapshot for the periodic log line, not a curve.
    terms: dict[str, float] = {}
    # `next_snap` starts at 1 on purpose when snapshots are on: the first checkpoint
    # rsl_rl writes is model_0.pt, the UNTRAINED policy, and that is the most useful
    # frame in the whole strip because it is the before picture.
    state = {"iter": 0, "total": iterations, "flushed": 0, "next_snap": 1}

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-40]

        if m := _ITER_RE.search(line):
            state["iter"], state["total"] = int(m.group(1)), int(m.group(2))
            if snap is not None and state["iter"] >= state["next_snap"]:
                # The boundary only moves once there is something to film, so a run
                # whose first checkpoint is still being written just asks again next
                # iteration instead of skipping a whole interval.
                if (ckpt := _newest_checkpoint()) is not None:
                    state["next_snap"] = state["iter"] + snap.every
                    snap.request(ckpt, _ckpt_iter(ckpt))
        if m := _LEVEL_RE.search(line):
            levels.append(float(m.group(1)))
        if m := _AIR_RE.search(line):
            airtime.append(float(m.group(1)))
        if m := _GAP_RE.search(line):
            gapflight.append(float(m.group(1)))
        if m := _TERM_RE.search(line):
            terms[m.group(1)] = float(m.group(2))
        if m := _REWARD_RE.search(line):
            rewards.append(float(m.group(1)))
            # Repaint every 25 points. Flushing every iteration would spend more time
            # writing HTML than training; never flushing means an empty report for
            # half an hour, which is the failure mode this is here to avoid.
            if len(rewards) - state["flushed"] >= 25:
                state["flushed"] = len(rewards)
                report_progress(task_id, num_envs, rewards, levels, state["iter"], state["total"],
                                snap.clips if snap else [], airtime, gapflight)
                # Echo to stdout as well. Without this the pod log is ONE line for the
                # whole run: this function consumes rsl_rl's stdout and would otherwise
                # swallow it, so `kubectl logs` shows nothing and a 100-minute run is
                # only observable through the report.
                log.info("iter %s/%s | reward %.2f | terrain row %s | air-time term %s"
                         "%s",
                         state["iter"], state["total"], rewards[-1],
                         f"{levels[-1]:.2f}" if levels else "n/a",
                         f"{airtime[-1]:+.4f}" if airtime else "n/a",
                         f" | gap-flight term {gapflight[-1]:+.4f}" if gapflight else "")
                # The breakdown, sorted by magnitude so whatever is dominating the return
                # is first. Ten terms is enough to see a penalty run away and short enough
                # to stay one line per repaint rather than a wall.
                if terms:
                    top = sorted(terms.items(), key=lambda kv: -abs(kv[1]))[:10]
                    log.info("        %s", "  ".join(
                        f"{k.split('/')[-1]}={v:+.4f}" for k, v in top))

    argv = [sys.executable, str(RSL_RL / "train.py"), f"--task={task_id}", "--headless",
            "--num_envs", str(num_envs), "--max_iterations", str(iterations)]
    if task_id.startswith("Spark-"):
        argv += ["--external_callback", "spark_envs.register"]

    rc = _run_streaming(argv, on_line)
    if rc != 0:
        raise RuntimeError(f"training exited {rc}. log tail:\n" + "\n".join(tail))
    return rewards, levels, airtime, gapflight, tail


def record(task_id: str, video_length: int = 300) -> Path | None:
    """Replay the trained policy through the RTX renderer. Returns the mp4, if any.

    --num_envs 1 on purpose: the default viewport camera frames the whole grid, and at
    4096 envs the robots are ant-sized specks. With one robot you can actually see the
    gait, which is the entire point of putting a video in the report.
    """
    tail: list[str] = []

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-25]

    rc = _run_streaming(
        [sys.executable, str(RSL_RL / "play.py"), f"--task={task_id}", "--headless",
         "--num_envs", "1", "--video", "--video_length", str(video_length)],
        on_line,
    )
    if rc != 0:
        log.warning("play.py exited %s; continuing without video. tail:\n%s", rc, "\n".join(tail))
        return None

    # Newest mp4 under the run's log tree. play.py names it rl-video-step-0.mp4 and
    # buries it a few levels down, so glob rather than guessing the path.
    clips = sorted(WORKDIR.glob("logs/**/*.mp4"), key=lambda p: p.stat().st_mtime)
    return clips[-1] if clips else None


def record_clips(
    task_id: str,
    steps: int = 300,
    terrain_level: int | None = None,
    terrain_col: int | None = None,
    timeline: int = 4,
    timeout: float = 2700.0,
) -> dict:
    """Film a trained policy with record.py. Returns its summary dict, or {} on failure.

    This is the replacement for `record()` above, and the reason it exists is in the
    header of record.py: `play.py --video` captures the Kit viewport, and headless on
    this box that capture renders the terrain and everything else EXCEPT the robot.
    record.py drives its own Camera sensors instead, which fixes that and throws in the
    onboard RGB and depth views for free.

    `task_id` is the TRAINING id. The Play variant is derived here rather than asked for,
    so a caller cannot accidentally film a different terrain than it trained on.

    Never allowed to raise: a rendering problem must not throw away a finished training
    run. The report has a branch for "no clips" and says so plainly.
    """
    play_id = task_id.replace("-v0", "-Play-v0")
    out = WORKDIR / "clips" / play_id
    tail: list[str] = []

    # Two buffers, and the second one is the useful one.
    #
    # `tail` is the last 60 lines verbatim. `head` is the FIRST 60 lines that are not pure
    # fault-fallout, which is where the actual cause lives: by the time a Kit render fault
    # has finished printing, the line that explains it is a thousand lines up. A run that
    # dies gets both, and a run that succeeds is never asked for either.
    noise = {"n": 0}
    head: list[str] = []

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-60]
        if _NOISE_RE.search(line):
            noise["n"] += 1
        elif len(head) < 60:
            head.append(line)

    argv = [sys.executable, str(HERE / "record.py"), "--task", play_id,
            "--logs", str(WORKDIR / "logs" / "rsl_rl"), "--steps", str(steps),
            "--out", str(out), "--onboard", "--timeline", str(timeline)]
    if terrain_level is not None:
        argv += ["--terrain_level", str(terrain_level)]
    if terrain_col is not None:
        argv += ["--terrain_col", str(terrain_col)]

    # 30 minutes is roughly 6x the measured 4.7 min for a full train-and-film cycle at
    # small settings, so it only ever fires on a genuine wedge.
    rc = _run_streaming(argv, on_line, timeout=timeout)
    summary_path = out / "summary.json"
    if rc != 0 or not summary_path.exists():
        # rc == 0 with no summary.json is a REAL failure and the most common one here:
        # Kit's fast shutdown path calls os._exit(), so a renderer that dies still exits
        # clean. The missing file is the only honest signal.
        log.warning(
            "record.py exited %s and wrote no summary; continuing without clips.\n"
            "-- first %s meaningful lines (%s fault-fallout lines suppressed) --\n%s\n"
            "-- last %s lines verbatim --\n%s",
            rc, len(head), noise["n"], "\n".join(head), len(tail), "\n".join(tail),
        )
        return {}
    summary = json.loads(summary_path.read_text())
    # record.py prints this too, but _run_streaming eats its stdout, so without this the
    # only place the report's own weight is visible is inside a pod that no longer exists.
    log.info("clip sizes KB: %s", summary.get("clip_kb"))
    return summary


async def export_policy(task_id: str) -> File | None:
    """Upload the trained policy out of the pod, as a tarball. None if there is nothing.

    Everything rsl_rl leaves behind for the run goes in: the final `model_*.pt`, and the
    `exported/` directory holding `policy.pt` and `policy.onnx`. The ONNX is the one that
    matters for anything outside Isaac Lab, and it is the smallest artefact that still
    represents 87 minutes of GPU time.

    Tarred rather than returned as several Files because the interesting unit is "the
    policy", and because the exported directory's filenames are fixed, so three separate
    outputs would collide the moment a second robot is trained.
    """
    # Newest run directory under logs/rsl_rl/<experiment>/<timestamp>/. Not filtered by
    # experiment name: one pod runs exactly one training, so the newest IS this one, and
    # matching on the name would just be another thing to keep in sync with spark_envs.
    runs = sorted(
        (p for p in (WORKDIR / "logs" / "rsl_rl").glob("*/*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
    )
    if not runs:
        log.warning("no rsl_rl run directory to export")
        return None
    run_dir = runs[-1]

    models = sorted(run_dir.glob("model_*.pt"), key=lambda p: p.stat().st_mtime)
    if not models:
        log.warning("no checkpoint in %s to export", run_dir)
        return None

    bundle = WORKDIR / f"policy_{task_id}.tar.gz"
    with tarfile.open(bundle, "w:gz") as tar:
        tar.add(models[-1], arcname=models[-1].name)
        exported = run_dir / "exported"
        if exported.is_dir():
            tar.add(exported, arcname="exported")
    log.info("exported %s (%.1f MB) from %s", bundle.name, bundle.stat().st_size / 1e6, run_dir.name)
    return await File.from_local(bundle)


# ── Report ──────────────────────────────────────────────────────────────────────

def _curve_svg(
    rewards: list[float], w: int = 760, h: int = 240,
    stroke: str = "#5cf", title: str = "", fmt: str = ".2f", zero: bool = False,
) -> str:
    """A curve as a hand-rolled SVG polyline.

    Hand-rolled because the training image has no matplotlib and adding it to a 25 GB
    image for one line chart is a poor trade. An SVG polyline needs no dependency and
    scales in the browser.

    `zero=True` forces y=0 into the drawn range. By default the axis is autoscaled to the
    data and the dashed zero line therefore only appears when the curve happens to
    straddle it, which is right for reward (where zero means nothing) and wrong for the
    air-time term (where the sign is the entire message, and "still below zero" is a
    result the chart has to be able to show).
    """
    if len(rewards) < 2:
        return "<p><i>not enough points yet</i></p>"

    lo, hi = min(rewards), max(rewards)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    span = (hi - lo) or 1.0
    pad = 34
    pts = " ".join(
        f"{pad + i * (w - 2 * pad) / (len(rewards) - 1):.1f},"
        f"{h - pad - (r - lo) / span * (h - 2 * pad):.1f}"
        for i, r in enumerate(rewards)
    )
    # `lo <= 0 <= hi`, not `lo < 0 < hi`: with zero=True the axis has just been widened to
    # touch zero, so an all-negative curve has hi == 0 exactly and the strict form would
    # drop the one line the chart was widened to show.
    zero_line = ""
    if lo <= 0 <= hi:
        y = h - pad - (0 - lo) / span * (h - 2 * pad)
        zero_line = (
            f"<line x1='{pad}' y1='{y:.1f}' x2='{w - pad}' y2='{y:.1f}' "
            f"stroke='#888' stroke-dasharray='4 4'/>"
        )
    label = (
        f"<text x='{w - pad}' y='16' fill='{stroke}' font-size='12' text-anchor='end'>{title}</text>"
        if title else ""
    )
    return (
        f"<svg viewBox='0 0 {w} {h}' style='width:100%;max-width:{w}px;background:#111'>"
        f"{zero_line}"
        f"<polyline points='{pts}' fill='none' stroke='{stroke}' stroke-width='2'/>"
        f"<text x='{pad}' y='16' fill='#aaa' font-size='12'>max {hi:{fmt}}</text>"
        f"<text x='{pad}' y='{h - 10}' fill='#aaa' font-size='12'>min {lo:{fmt}}</text>"
        f"{label}"
        f"</svg>"
    )


def _video_tag(path: Path, caption: str, max_width: int = 760) -> str:
    """A clip, base64'd inline. No JS, no external file, survives the pod."""
    if not path.exists():
        return ""
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (
        f"<figure style='margin:0 0 14px 0'>"
        f"<video controls autoplay loop muted playsinline "
        f"style='width:100%;max-width:{max_width}px;background:#000'>"
        f"<source src='data:video/mp4;base64,{b64}' type='video/mp4'></video>"
        f"<figcaption style='color:#888;font-size:12px'>{caption} "
        f"&middot; {path.stat().st_size / 1024:.0f} KB</figcaption></figure>"
    )


def _scan_svg(values: list[float], grid: list[int] | None, when: str = "", cell: int = 18) -> str:
    """The height scanner's last frame, as a grid of coloured cells.

    This is the picture that justifies the whole episode. The reward curve says the
    policy improved and the clip says it walks; this says what it was LOOKING at while
    it did, which is the sensor MJX has no answer for. Each cell is one downward ray:
    the value is how far below the sensor the ground came back, so a dark cell is a
    ledge or a hole and a bright one is ground close under the belly.

    Rows are lateral, columns run fore-aft. See _scan_grid() in record.py.
    """
    if not values or not grid or grid[0] * grid[1] != len(values):
        return ""
    rows, cols = grid
    lo, hi = min(values), max(values)
    span = hi - lo
    if span < 1e-4:
        # Every ray came back the same. Honest and not worth 187 identical rectangles.
        return (
            f"<p style='color:#888'>The scanner read flat ground for the whole clip "
            f"({lo:.2f} m under the sensor, {len(values)} rays). The sub-terrain platforms "
            f"are wider than the 1.6&times;1.0 m scan, so a robot that stays near where it "
            f"spawned never sees an edge.</p>"
        )

    cells = []
    for i, v in enumerate(values):
        r, c = divmod(i, cols)
        # Near ground = bright, deep = dark. Same reading as the depth clip.
        t = 1.0 - (v - lo) / span
        red = int(40 + 215 * t)
        green = int(30 + 120 * t)
        blue = int(60 + 40 * (1 - t))
        cells.append(
            f"<rect x='{c * cell}' y='{r * cell}' width='{cell - 1}' height='{cell - 1}' "
            f"fill='rgb({red},{green},{blue})'/>"
        )
    return (
        f"<svg viewBox='0 0 {cols * cell} {rows * cell}' "
        f"style='width:100%;max-width:{cols * cell * 2}px;background:#111'>"
        f"{''.join(cells)}</svg>"
        f"<p style='color:#888;font-size:12px'>{rows}&times;{cols} = {len(values)} rays "
        f"&middot; clearance {lo:.2f} m to {hi:.2f} m &middot; bright is close, dark is a drop"
        f"{when}</p>"
    )


def _timeline_html(timeline: list[dict]) -> str:
    """The learning strip: the same robot, same terrain, at several points in training.

    This is the thing a reward curve cannot tell you and the thing everyone actually
    wants to see. All the clips autoplay together, so iteration 0 flailing next to
    iteration 1450 walking is one glance rather than two runs and a memory.

    Laid out in a horizontally scrolling row: the page must never scroll sideways, so the
    overflow lives in this container.
    """
    if not timeline:
        return ""
    cells = []
    for entry in timeline:
        clip = Path(entry["clip"])
        if not clip.exists():
            continue
        b64 = base64.b64encode(clip.read_bytes()).decode()
        cells.append(
            f"<figure style='margin:0;flex:0 0 auto;width:300px'>"
            f"<video autoplay loop muted playsinline style='width:300px;background:#000'>"
            f"<source src='data:video/mp4;base64,{b64}' type='video/mp4'></video>"
            f"<figcaption style='color:#aaa;font-size:12px;text-align:center'>"
            f"iteration <b>{entry['iteration']}</b></figcaption></figure>"
        )
    if not cells:
        return ""
    return (
        f"<h3>Learning to do it</h3>"
        f"<p style='color:#888;font-size:12px'>Same robot, same terrain, same camera. "
        f"Only the checkpoint changes.</p>"
        f"<div style='display:flex;gap:12px;overflow-x:auto;padding-bottom:8px'>{''.join(cells)}</div>"
    )


def _snapshots_html(
    snaps: list[dict], level: int | None = None, max_clips: int = 5,
    title: str = "What it looks like right now",
) -> str:
    """Mid-training clips: the current policy, filmed while it is still learning.

    The newest one gets the big frame, because "what does it look like RIGHT NOW" is the
    question this answers, and the earlier ones sit next to it as a strip so the answer
    has something to be compared against.

    Capped at `max_clips`, keeping the first and the most recent: every clip in here is
    base64'd into the HTML on every repaint, and a three-hour run produces enough
    snapshots to turn a live report into a 40 MB page.
    """
    if not snaps:
        return ""
    ordered = sorted(snaps, key=lambda s: s["iteration"])
    picked = ordered if len(ordered) <= max_clips else [ordered[0], *ordered[-(max_clips - 1):]]
    latest, earlier = picked[-1], picked[:-1]

    strip = "".join(
        f"<figure style='margin:0;flex:0 0 auto;width:200px'>"
        f"<video autoplay loop muted playsinline style='width:200px;background:#000'>"
        f"<source src='data:video/mp4;base64,"
        f"{base64.b64encode(Path(s['clip']).read_bytes()).decode()}' type='video/mp4'></video>"
        f"<figcaption style='color:#aaa;font-size:12px;text-align:center'>"
        f"iter <b>{s['iteration']}</b></figcaption></figure>"
        for s in earlier if Path(s["clip"]).exists()
    )
    where = f" on difficulty row <b>{level}</b>" if level is not None else ""
    return (
        f"<h3>{title}</h3>"
        f"<p style='color:#888;font-size:12px'>The live policy, pulled from the newest "
        f"checkpoint and filmed{where} while training carries on. Same patch and same "
        f"camera every time, so the only thing changing across the strip is the policy. "
        f"{len(ordered)} snapshot{'s' if len(ordered) != 1 else ''} so far.</p>"
        + _video_tag(
            Path(latest["clip"]),
            f"iteration {latest['iteration']}, filmed mid-run &middot; "
            f"{latest.get('frames', 0) / 50:.1f} s of robot time",
            480,
        )
        + (f"<div style='display:flex;gap:10px;overflow-x:auto;padding-bottom:8px'>{strip}</div>"
           if strip else "")
    )


def _levels_svg(levels: list[float]) -> str:
    """The curriculum curve, and the one worth reading first on rough terrain.

    Reward alone cannot distinguish "the policy got better" from "the policy got better
    at the easy rows it happened to start on". This is the mean difficulty row the envs
    are standing on, so it going up means the terrain curriculum kept promoting them onto
    ground they used to fall off. Flat on this chart with a rising reward is a policy
    that has plateaued and is farming the easy end of the course.
    """
    if len(levels) < 2:
        return ""
    return (
        f"<h3>Terrain difficulty the robots earned</h3>"
        f"{_curve_svg(levels, stroke='#fc6', title='mean terrain row', fmt='.2f')}"
        f"<p style='color:#888;font-size:12px'>mean row <b>{levels[0]:.2f}</b> &rarr; "
        f"<b>{levels[-1]:.2f}</b>. Rows are difficulty: row 0 is the gentle end of every "
        f"sub-terrain, the last row is the hardest the config asks for. Envs are promoted "
        f"when they walk further than half a patch and demoted when they do not "
        f"(curriculums.py:49). This never reaches the top row even on a solved course: an "
        f"env that clears the last one is sent to a RANDOM row rather than kept there "
        f"(terrain_importer.py:325), so the mean settles below the maximum by design.</p>"
    )


def _airtime_svg(airtime: list[float]) -> str:
    """The air-time reward term over training. On `leap`, this is the jump appearing.

    Deliberately drawn separately from mean reward rather than folded into it. Total
    reward on this task is dominated by velocity tracking, which climbs steadily whether
    or not the robot ever leaves the ground, so a jump is invisible in it. This one term
    is not: it is negative while the policy shuffles, and it crosses zero at the moment
    strides get longer than the 0.25 s threshold the leap profile sets.

    The zero line is drawn because the sign is the whole point of the chart.
    """
    if len(airtime) < 2:
        return ""
    crossed = next((i for i, v in enumerate(airtime) if v > 0), None)
    when = (
        f" Crossed zero at point <b>{crossed}</b> of {len(airtime)}."
        if crossed is not None else
        " Never crossed zero: the policy is still taking walking strides, not flights."
    )
    return (
        f"<h3>Time off the ground, as a reward term</h3>"
        f"{_curve_svg(airtime, stroke='#6cf', title='Episode_Reward/feet_air_time', fmt='+.4f', zero=True)}"
        f"<p style='color:#888;font-size:12px'>"
        f"<code>Episode_Reward/feet_air_time</code>, one point per logged iteration: "
        f"(air time &minus; 0.25 s) summed over the feet at each touchdown. Below zero is "
        f"a walk, above zero is a flight phase.{when} The absolute scale is arbitrary "
        f"(every reward term is scaled by <code>step_dt</code>); the sign is not.</p>"
    )


def _flight_html(flight: dict, snaps: list[dict] | None = None) -> str:
    """The measurement, as opposed to the reward. Did it leave the ground, and how far.

    Everything else in this report is a training signal: reward went up, the curriculum
    promoted, a term turned positive. Those are all arguments that a jump probably
    happened. This is the observation itself, taken off the contact sensor during the
    replay: the longest interval in the clip where NO body of the robot was touching
    anything, and the straight-line distance covered during it.

    It is also the number that can embarrass the run, which is why it is here. A policy
    that learned to shuffle to the lip and stop reads 0.00 s no matter how good the reward
    curve looks.
    """
    if not flight:
        return ""
    longest = flight.get("longest_s", 0.0)
    span = flight.get("longest_span_m", 0.0)
    verdict = (
        "That is a flight phase, not a stride: all four feet left the ground together."
        if longest >= 0.20 else
        "Short enough to be an ordinary trot rather than a jump. A trotting quadruped is "
        "briefly airborne between footfalls; a leap is not."
    )

    # The same measurement across the mid-training snapshots, if there are any. Filmed on
    # a fixed row with a fixed camera, so the numbers are comparable down the column and
    # the reader can see the flight time grow rather than take the final one on trust.
    history = ""
    rows = [
        (s["iteration"], s["flight"])
        for s in sorted(snaps or [], key=lambda s: s["iteration"])
        if s.get("flight")
    ]
    if len(rows) > 1:
        cells = "".join(
            f"<tr><td style='padding:4px 14px'>{it}</td>"
            f"<td style='padding:4px 14px'>{f.get('longest_s', 0):.2f} s</td>"
            f"<td style='padding:4px 14px'>{f.get('longest_span_m', 0):.2f} m</td>"
            f"<td style='padding:4px 14px'>{f.get('phases', 0)}</td></tr>"
            for it, f in rows
        )
        history = (
            f"<p style='color:#888;font-size:12px;margin-top:18px'>The same measurement on "
            f"every mid-training clip, all filmed on the same row:</p>"
            f"<table style='border-collapse:collapse;font-size:13px'>"
            f"<tr style='color:#888'><th style='padding:4px 14px;text-align:left'>iteration</th>"
            f"<th style='padding:4px 14px;text-align:left'>longest flight</th>"
            f"<th style='padding:4px 14px;text-align:left'>distance covered</th>"
            f"<th style='padding:4px 14px;text-align:left'>flight phases</th></tr>"
            f"{cells}</table>"
        )

    return (
        f"<h3>Did it actually jump?</h3>"
        f"<p style='font-size:15px'>Longest unbroken flight in the clip: "
        f"<b>{longest:.2f} s</b>, covering <b>{span:.2f} m</b> of ground. "
        f"{flight.get('phases', 0)} flight phases over "
        f"{flight.get('steps', 0) / 50:.1f} s of robot time; "
        f"<b>{flight.get('airborne_frac', 0) * 100:.0f}%</b> of the clip airborne.</p>"
        f"<p style='color:#888;font-size:12px'>{verdict} Measured off the contact sensor, "
        f"which tracks every body on the robot and not just the feet, so a landing on the "
        f"knees counts as touching down. See <code>_flight_phases</code> in record.py.</p>"
        f"{history}"
    )


def _scan_when(scan: dict) -> str:
    """Say which frame the scan grid is from, so it is not read as the end state."""
    if "peak_frame" not in scan:
        return ""
    return f" &middot; frame {scan['peak_frame']} of {scan.get('frames', '?')}, the busiest one"


def _gapflight_svg(gapflight: list[float]) -> str:
    """The gap-flight reward term over training. On `vault`, this is the answer.

    Drawn separately from the air-time chart above it because the two say different
    things, and the difference is the whole reason `vault` exists.

    `feet_air_time` is blind: it pays a long stride and a bounce on flat ground exactly as
    well as it pays a trench crossing, so its sign is suggestive and no more.
    `gap_flight` is not: it is forward speed multiplied by "every foot is off the ground"
    multiplied by "the height scanner sees a hole underneath", so there is no way to earn
    it except by being airborne over a gap while moving. It is structurally pinned at 0.0
    until the robot leaves the ground over a trench for the first time.

    So this chart has no interesting shape, only an interesting MOMENT: the point it
    lifts off zero is the iteration the jump was invented, and everything after that is
    the policy getting better at something it can already do. If the line is still flat at
    the end of the run, the run failed, and it failed unambiguously.
    """
    if len(gapflight) < 2:
        return ""
    lift = next((i for i, v in enumerate(gapflight) if v > 0), None)
    when = (
        f" First non-zero at point <b>{lift}</b> of {len(gapflight)}: "
        f"the first airborne crossing."
        if lift is not None else
        " <b>Never left zero.</b> No env has yet been airborne over a gap while moving "
        "forward, so the policy is still stepping across rather than jumping."
    )
    return (
        f"<h3>Airborne over a hole, as a reward term</h3>"
        f"{_curve_svg(gapflight, stroke='#fc6', title='Episode_Reward/gap_flight', fmt='+.4f', zero=True)}"
        f"<p style='color:#888;font-size:12px'>"
        f"<code>Episode_Reward/gap_flight</code>, one point per logged iteration: forward "
        f"speed while every foot is in the air and the height scanner reads a trench "
        f"beneath the base. Unlike the air-time term above, this one cannot be earned by "
        f"walking.{when}</p>"
    )


def report_parkour(
    task_id: str, num_envs: int, iterations: int, rewards: list[float],
    levels: list[float], clips: dict, secs: float, tail: list[str],
    snaps: list[dict] | None = None, snap_level: int | None = None,
    airtime: list[float] | None = None, gapflight: list[float] | None = None,
) -> None:
    """Final report for a run on our own terrain: curve, clips, and the sensor view."""
    patch = clips.get("patch") or {}
    scan = clips.get("height_scan") or {}
    paths = {k: Path(v) for k, v in (clips.get("clips") or {}).items()}

    where = ""
    if patch:
        where = (
            f"<p>Filmed on row <b>{patch['row']}</b> of {patch['rows']} "
            f"(difficulty <b>{patch['difficulty']}</b>), sub-terrain "
            f"<b>{patch.get('sub_terrain') or 'unknown'}</b>, column {patch['col']} "
            f"of {patch['cols']}.</p>"
        )

    if paths:
        side = "".join(
            _video_tag(paths[key], caption, 380)
            for key, caption in (("onboard", "what the robot sees (RGB)"),
                                 ("depth", "and in depth: red near, blue far"))
            if key in paths
        )
        footage = (
            (_video_tag(paths["chase"], "chase camera, RTX, rendered in the pod") if "chase" in paths else "")
            + f"<div style='display:flex;gap:14px;flex-wrap:wrap'>{side}</div>"
        )
    else:
        footage = "<p><b>No clips.</b> Training succeeded; the replay render did not. See the log tail.</p>"

    steps = num_envs * 24 * max(len(rewards), 1)
    rate = steps / secs if secs else 0

    flyte.report.replace(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; {iterations} iterations &middot; {secs / 60:.1f} min</p>"
        f"<p>mean reward <b>{rewards[0]:.2f}</b> &rarr; <b>{rewards[-1]:.2f}</b> "
        f"&middot; ~<b>{rate:,.0f}</b> env-steps/sec</p>"
        # ── Order matters, and it is footage first ─────────────────────────────
        #
        # The charts used to lead. They are the wrong thing to open with: a Flyte report
        # opens scrolled to the top and repaints in place while the run is live, so
        # whatever is in the first screen is the only thing most people ever see. Two
        # SVG curves in that slot means the answer to "did it work" is below the fold,
        # every time, on every repaint.
        #
        # So: the clip, then the one measurement that says whether what is in the clip
        # was a jump, then the training signals that argue for it. Curves are evidence.
        # The video is the result.
        f"<h3>The trained policy</h3>{where}{footage}"
        f"{_flight_html(patch.get('flight') or {}, snaps)}"
        f"{_curve_svg(rewards, title='mean reward')}"
        f"{_levels_svg(levels)}"
        f"{_gapflight_svg(gapflight or [])}"
        f"{_airtime_svg(airtime or [])}"
        f"{_timeline_html(clips.get('timeline') or [])}"
        f"{_snapshots_html(snaps or [], snap_level, max_clips=6, title='Filmed while it trained')}"
        f"<h3>What the policy actually sees</h3>"
        f"{_scan_svg(scan.get('peak', []), scan.get('grid'), _scan_when(scan))}",
    )
    # rsl_rl's log tail is 40 lines of monospace and pushes the clips off the screen.
    # Its own tab keeps the main view to charts and footage.
    flyte.report.get_tab("rsl_rl log").replace(
        f"<pre style='font-size:11px'>{chr(10).join(tail[-40:])}</pre>"
    )
    flyte.report.flush()


def report_progress(
    task_id: str, num_envs: int, rewards: list[float], levels: list[float], it: int, total: int,
    snaps: list[dict] | None = None, airtime: list[float] | None = None,
    gapflight: list[float] | None = None,
) -> None:
    """Live repaint while training. Watching the terrain row climb is the good bit."""
    now = f" &middot; mean terrain row <b>{levels[-1]:.2f}</b>" if levels else ""
    # The live headline for a leap run. Reward and terrain row both move for reasons that
    # have nothing to do with jumping, so the sign of this term is the one number on the
    # page that says whether the thing being trained for is happening yet.
    air = f" &middot; air-time term <b>{airtime[-1]:+.4f}</b>" if airtime else ""
    # On a `vault` run this outranks everything else on the line: it is zero until the
    # robot is airborne over a trench, so a non-zero value IS the result.
    gap = f" &middot; gap-flight term <b>{gapflight[-1]:+.4f}</b>" if gapflight else ""
    latest = next((s["flight"] for s in sorted(snaps or [], key=lambda s: -s["iteration"])
                   if s.get("flight")), None)
    flew = (
        f"<p>Longest flight in the newest clip: <b>{latest['longest_s']:.2f} s</b> "
        f"over <b>{latest['longest_span_m']:.2f} m</b>.</p>"
        if latest else ""
    )
    # Same rule as report_final below: the newest clip goes above the curves. This one
    # repaints every 25 logged iterations for hours, and the whole reason the snapshot
    # daemon exists is so there is something to look at while it does. Putting the charts
    # first would bury it under two SVGs on every single repaint.
    flyte.report.replace(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; iteration <b>{it}</b>/{total} "
        f"&middot; mean reward <b>{rewards[-1]:.2f}</b> (from {rewards[0]:.2f}){now}{air}{gap}</p>"
        f"{flew}"
        f"{_snapshots_html(snaps or [])}"
        f"{_curve_svg(rewards, title='mean reward')}"
        f"{_levels_svg(levels)}"
        f"{_gapflight_svg(gapflight or [])}"
        f"{_airtime_svg(airtime or [])}",
        do_flush=True,
    )


def report_final(
    task_id: str, num_envs: int, iterations: int, rewards: list[float],
    clip: Path | None, secs: float, tail: list[str],
) -> None:
    """Final report: the curve says the number went up, the clip says it went up for the right reason."""
    if clip and clip.exists():
        b64 = base64.b64encode(clip.read_bytes()).decode()
        # Base64 straight into a <video> tag: no JS, no external file, and the clip
        # survives in the report after the pod is gone.
        video = (
            f"<video controls autoplay loop muted style='width:100%;max-width:760px'>"
            f"<source src='data:video/mp4;base64,{b64}' type='video/mp4'></video>"
            f"<p style='color:#888;font-size:12px'>{clip.stat().st_size / 1024:.0f} KB, "
            f"rendered headless through the RTX renderer in the pod</p>"
        )
    else:
        video = "<p><b>No clip.</b> Training succeeded; the replay render did not. See the log tail.</p>"

    # env-steps/sec is the number that compares to the MuJoCo demo. rsl_rl collects
    # num_steps_per_env (24 for this config) steps per env per iteration.
    steps = num_envs * 24 * max(len(rewards), 1)
    rate = steps / secs if secs else 0

    flyte.report.replace(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; {iterations} iterations &middot; {secs / 60:.1f} min</p>"
        f"<p>mean reward <b>{rewards[0]:.2f}</b> &rarr; <b>{rewards[-1]:.2f}</b> "
        f"&middot; ~<b>{rate:,.0f}</b> env-steps/sec</p>"
        # Footage above the curve, for the reason spelled out in report_parkour.
        f"<h3>The trained policy</h3>{video}"
        f"{_curve_svg(rewards, title='mean reward')}"
        f"<h3>log tail</h3><pre style='font-size:11px'>{chr(10).join(tail[-20:])}</pre>",
        do_flush=True,
    )
