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
import threading
from pathlib import Path

import flyte.report

log = logging.getLogger(__name__)

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


def train(task_id: str, num_envs: int, iterations: int) -> tuple[list[float], list[str]]:
    """Run rsl_rl PPO. Returns (reward curve, tail of the log).

    Our own `Spark-*` tasks are registered through `--external_callback`, which is Isaac
    Lab's supported hook for exactly this (see the header of spark_envs.py). Stock
    `Isaac-*` tasks are left alone, so episode 1's flat Anymal-C run is byte-for-byte
    the command it always was.
    """
    WORKDIR.mkdir(parents=True, exist_ok=True)
    rewards: list[float] = []
    tail: list[str] = []
    state = {"iter": 0, "total": iterations, "flushed": 0}

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-40]

        if m := _ITER_RE.search(line):
            state["iter"], state["total"] = int(m.group(1)), int(m.group(2))
        if m := _REWARD_RE.search(line):
            rewards.append(float(m.group(1)))
            # Repaint every 25 points. Flushing every iteration would spend more time
            # writing HTML than training; never flushing means an empty report for
            # half an hour, which is the failure mode this is here to avoid.
            if len(rewards) - state["flushed"] >= 25:
                state["flushed"] = len(rewards)
                report_progress(task_id, num_envs, rewards, state["iter"], state["total"])

    argv = [sys.executable, str(RSL_RL / "train.py"), f"--task={task_id}", "--headless",
            "--num_envs", str(num_envs), "--max_iterations", str(iterations)]
    if task_id.startswith("Spark-"):
        argv += ["--external_callback", "spark_envs.register"]

    rc = _run_streaming(argv, on_line)
    if rc != 0:
        raise RuntimeError(f"training exited {rc}. log tail:\n" + "\n".join(tail))
    return rewards, tail


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
    timeout: float = 1800.0,
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

    def on_line(line: str) -> None:
        tail.append(line)
        del tail[:-25]

    argv = [sys.executable, str(HERE / "record.py"), "--task", play_id,
            "--logs", str(WORKDIR / "logs" / "rsl_rl"), "--steps", str(steps),
            "--out", str(out), "--onboard"]
    if terrain_level is not None:
        argv += ["--terrain_level", str(terrain_level)]
    if terrain_col is not None:
        argv += ["--terrain_col", str(terrain_col)]

    # 30 minutes is roughly 6x the measured 4.7 min for a full train-and-film cycle at
    # small settings, so it only ever fires on a genuine wedge.
    rc = _run_streaming(argv, on_line, timeout=timeout)
    summary_path = out / "summary.json"
    if rc != 0 or not summary_path.exists():
        log.warning("record.py exited %s; continuing without clips. tail:\n%s", rc, "\n".join(tail))
        return {}
    return json.loads(summary_path.read_text())


# ── Report ──────────────────────────────────────────────────────────────────────

def _curve_svg(rewards: list[float], w: int = 760, h: int = 240) -> str:
    """Reward curve as a hand-rolled SVG polyline.

    Hand-rolled because the training image has no matplotlib and adding it to a 25 GB
    image for one line chart is a poor trade. An SVG polyline needs no dependency and
    scales in the browser.
    """
    if len(rewards) < 2:
        return "<p><i>not enough points yet</i></p>"

    lo, hi = min(rewards), max(rewards)
    span = (hi - lo) or 1.0
    pad = 34
    pts = " ".join(
        f"{pad + i * (w - 2 * pad) / (len(rewards) - 1):.1f},"
        f"{h - pad - (r - lo) / span * (h - 2 * pad):.1f}"
        for i, r in enumerate(rewards)
    )
    zero = ""
    if lo < 0 < hi:
        y = h - pad - (0 - lo) / span * (h - 2 * pad)
        zero = f"<line x1='{pad}' y1='{y:.1f}' x2='{w - pad}' y2='{y:.1f}' stroke='#888' stroke-dasharray='4 4'/>"
    return (
        f"<svg viewBox='0 0 {w} {h}' style='width:100%;max-width:{w}px;background:#111'>"
        f"{zero}"
        f"<polyline points='{pts}' fill='none' stroke='#5cf' stroke-width='2'/>"
        f"<text x='{pad}' y='16' fill='#aaa' font-size='12'>max {hi:.2f}</text>"
        f"<text x='{pad}' y='{h - 10}' fill='#aaa' font-size='12'>min {lo:.2f}</text>"
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


def _scan_when(scan: dict) -> str:
    """Say which frame the scan grid is from, so it is not read as the end state."""
    if "peak_frame" not in scan:
        return ""
    return f" &middot; frame {scan['peak_frame']} of {scan.get('frames', '?')}, the busiest one"


def report_parkour(
    task_id: str, num_envs: int, iterations: int, rewards: list[float],
    clips: dict, secs: float, tail: list[str],
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

    flyte.report.log(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; {iterations} iterations &middot; {secs / 60:.1f} min</p>"
        f"<p>mean reward <b>{rewards[0]:.2f}</b> &rarr; <b>{rewards[-1]:.2f}</b> "
        f"&middot; ~<b>{rate:,.0f}</b> env-steps/sec</p>"
        f"{_curve_svg(rewards)}"
        f"<h3>The trained policy</h3>{where}{footage}"
        f"<h3>What the policy actually sees</h3>"
        f"{_scan_svg(scan.get('peak', []), scan.get('grid'), _scan_when(scan))}"
        f"<h3>log tail</h3><pre style='font-size:11px'>{chr(10).join(tail[-20:])}</pre>",
        do_flush=True,
    )


def report_progress(task_id: str, num_envs: int, rewards: list[float], it: int, total: int) -> None:
    flyte.report.log(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; iteration <b>{it}</b>/{total} "
        f"&middot; mean reward <b>{rewards[-1]:.2f}</b> (from {rewards[0]:.2f})</p>"
        f"{_curve_svg(rewards)}",
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

    flyte.report.log(
        f"<h2>{task_id}</h2>"
        f"<p>{num_envs:,} parallel envs &middot; {iterations} iterations &middot; {secs / 60:.1f} min</p>"
        f"<p>mean reward <b>{rewards[0]:.2f}</b> &rarr; <b>{rewards[-1]:.2f}</b> "
        f"&middot; ~<b>{rate:,.0f}</b> env-steps/sec</p>"
        f"{_curve_svg(rewards)}"
        f"<h3>The trained policy</h3>{video}"
        f"<h3>log tail</h3><pre style='font-size:11px'>{chr(10).join(tail[-20:])}</pre>",
        do_flush=True,
    )
