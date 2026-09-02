"""DreamerV3 on Flyte: learn a world model of MuJoCo, in a pod, and watch it dream.

    flyte run pipeline.py dream                      # the flagship, ~7 h for 500k steps
    flyte run pipeline.py dream --steps 20000        # is the plumbing alive?
    flyte run pipeline.py dream --task_id dmc_walker_walk   # the stock domain

Runs to the `world-models` project (.flyte/config.yaml), the one that will also hold
the Cosmos and V-JEPA 2 events. The robotics demos stay in `physical-ai`.

── What this run is trying to show ─────────────────────────────────────────────
A world model is a model that predicts what happens next. The way to demonstrate one
is not a reward curve, it is to make it predict and then show the prediction next to
what actually happened. DreamerV3 already builds exactly that picture every few
minutes of training, and `scopevid.py` lifts it out of the logdir into the report, so
the report is a live feed of the model's imagination sharpening.

That requires **pixel** observations. With `--config dmc_proprio` the agent's
observation is a state vector, embodied strips the rendered image out before the agent
ever sees it, the decoder has no image head, and `Agent.report` produces no video at
all. `dmc_vision` is the default here for that reason.

── Why the agent runs in a CHILD PROCESS ───────────────────────────────────────
Same reason as the Isaac Sim demo, and again not about imports. DreamerV3's entry
point is a script that owns its own process: it parses flags, builds a logger, spawns
environment workers through `portal`, and runs a blocking training loop. Importing it
and calling `main()` inside a Flyte task would put those env worker subprocesses and
Flyte's asyncio runtime in the same process, which is the shape that cost the Isaac
Sim demo a 28-minute hung pod.

Spawning it means the task can also stream stdout and repaint the report while
training runs, instead of producing nothing for seven hours.

`launch.py` rather than upstream's `dreamerv3/main.py`, because the custom `arena`
domain has to be registered with dm_control inside the process that loads the
environments. See the note at the top of launch.py.

── Reading results ─────────────────────────────────────────────────────────────
Dreamer writes `metrics.jsonl` in its logdir, one JSON object per log flush, and that
is the honest source for the curves: they come from the agent's own logger rather than
from anything scraped out of stdout.

Two things to know, both measured the hard way on the host:

  * `run.log_every` is a WALL CLOCK timer, not a step count, and it defaults to
    minutes. A short run finishes before it ever fires and writes an empty logdir,
    which looks exactly like a run that did nothing. This task sets it explicitly.

  * Training only starts once the replay buffer holds batch_size * batch_length
    transitions, and the first episode only ends after `run.envs * 1000` steps. Below
    that the loop is pure data collection, so a very short run can legitimately show
    no loss curves and no score at all.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import flyte
import flyte.io
import flyte.report

# Top level so Flyte bundles these into the pod. `arena` and `launch` are not called
# from here at all: launch.py is executed as a subprocess and arena.py is imported by
# launch.py, but neither file would reach the pod if nothing imported it.
import arena  # noqa: F401
import launch  # noqa: F401
import replay
import reports
import scopevid
from config import DREAMER_ROOT, gpu_env, orch_env

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# Repaint the report at most this often. Videos are only re-encoded when Dreamer has
# actually written a new clip, so a repaint with nothing new is just HTML.
_REPAINT_SECS = 30

# Measured on this box: the training step is the bottleneck at ~5 gradient steps/s for
# size12m on pixels, which with train_ratio 256 pins the environment at ~20 steps/s.
# 500k steps is therefore about 7 hours, and it is roughly where DreamerV3's published
# walker_walk vision curve reaches its plateau.
_DEFAULT_STEPS = 500_000

# A pixel transition costs ~151 KB in the replay buffer (measured), so upstream's 5e6
# default would want 750 GB. This keeps the most recent 200k transitions, ~30 GB.
_REPLAY_SIZE = 200_000

# What has to outlive the pod for a trained agent to be reusable. `config.yaml` and
# `ckpt` are the hard requirement, and they are exactly what replay.py already loads:
# it rebuilds the agent from the config, then points `elements.Checkpoint` at the
# DIRECTORY and lets it resolve which checkpoint to read. `metrics.jsonl` and `scope`
# ride along because they are small and they are the run's evidence; regenerating them
# costs another seven hours.
#
# Everything else is deliberately left behind, above all the replay buffer. A pixel
# transition is ~151 KB, so `_REPLAY_SIZE` on disk is about 30 GB, and none of it is
# needed to load a trained agent. That is the whole reason this stages a subset instead
# of uploading the logdir.
_KEEP = ("config.yaml", "ckpt", "metrics.jsonl", "scope")


async def _persist(logdir: Path) -> flyte.io.Dir | None:
    """Copy the reusable part of the logdir somewhere durable and upload it.

    Without this a seven hour run leaves nothing behind but its report: Dreamer writes
    into `/tmp/dreamer/<task>` and the pod takes that with it when it exits, so the
    agent that earned the numbers is gone and every later demo has to retrain it.

    Never raises. Training is the expensive part, and a blob store hiccup must not
    throw away a finished run. Same rule the replay video follows.
    """
    # Not cleaned up on purpose. `Dir.from_local` can return a lazily-uploaded handle
    # whose upload happens after this returns, so deleting the staging directory here
    # would be deleting the thing being uploaded. The pod is about to be destroyed
    # anyway, and the staged copy is only the checkpoint, not the 30 GB buffer.
    staged = Path(tempfile.mkdtemp(prefix="dreamer-model-"))
    try:
        for name in _KEEP:
            src = logdir / name
            if not src.exists():
                log.warning("not persisting %s: not in the logdir", name)
                continue
            if src.is_dir():
                shutil.copytree(src, staged / name)
            else:
                shutil.copy2(src, staged / name)
        size = sum(f.stat().st_size for f in staged.rglob("*") if f.is_file())
        log.info("persisting %.0f MB: %s", size / 1e6,
                 ", ".join(sorted(p.name for p in staged.iterdir())))
        return await flyte.io.Dir.from_local(str(staged))
    except Exception as exc:  # noqa: BLE001
        log.warning("could not persist the model, run is still good: %s", exc)
        return None


def _read_metrics(path: Path) -> dict:
    """Parse Dreamer's metrics.jsonl into the curves the report draws.

    `distance` is the interesting one and it is why arena.py logs it. Episode return
    can rise through postures that score well without going anywhere, so the report
    plots metres actually travelled beside the score. If score climbs and distance does
    not, the policy found a way to be paid for standing still.
    """
    out = {"score": [], "distance": [], "losses": {}, "fps": {}, "ram": [], "params": ""}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        step = row.get("step")
        if step is None:
            continue
        step = float(step)
        if "episode/score" in row:
            out["score"].append((step, float(row["episode/score"])))
        if "epstats/log/x_position/max" in row:
            out["distance"].append((step, float(row["epstats/log/x_position/max"])))
        if "replay/ram_gb" in row:
            out["ram"].append((step, float(row["replay/ram_gb"])))
        if "train/opt/param_count" in row:
            # The trainer's own count of what it optimises. Preferred over summing the
            # parameter table, which also lists optimiser moment slots and the critic's
            # slow target copy and so reports about 8% too many.
            out["params"] = f"{int(row['train/opt/param_count']):,} parameters"
        for key, value in row.items():
            if key.startswith("train/loss/") and isinstance(value, (int, float)):
                out["losses"].setdefault(key.rsplit("/", 1)[-1], []).append(
                    (step, float(value))
                )
            elif key.startswith("fps/") and isinstance(value, (int, float)):
                out["fps"][key.rsplit("/", 1)[-1]] = float(value)
    return out


class _Film:
    """Keeps the newest clip of each kind, and a filmstrip of stills over training.

    The newest clip answers "what is it doing now". The filmstrip answers "is it
    getting better", which for a world model is the whole question and is invisible in
    any single frame. Stills are only appended when Dreamer writes a genuinely new
    clip, so a repaint that finds nothing new costs a directory listing.
    """

    def __init__(self, keep: int = 12):
        self.keep = keep
        self.latest: dict[str, dict] = {}
        self.strip: dict[str, dict[int, bytes]] = {"dream": {}, "rollout": {}}

    def refresh(self, logdir: Path) -> bool:
        changed = False
        for kind, fn in (("dream", scopevid.dream), ("rollout", scopevid.rollout)):
            try:
                found = fn(logdir)
            except Exception as exc:  # noqa: BLE001
                # A clip half-written when we listed the directory is normal and must
                # never take down a seven hour training run.
                log.warning("harvest %s failed: %s", kind, exc)
                continue
            if not found:
                continue
            if self.latest.get(kind, {}).get("step") == found["step"]:
                continue
            self.latest[kind] = found
            self.strip[kind][found["step"]] = found["still"]
            changed = True
        return changed

    def thinned(self, kind: str) -> list[tuple[int, bytes]]:
        """Evenly spaced stills, always including the first and the last.

        The first one matters most: it is the model before it knew anything, and it is
        what makes the last one legible as progress.
        """
        items = sorted(self.strip[kind].items())
        if len(items) <= self.keep:
            return items
        idx = {0, len(items) - 1}
        step = (len(items) - 1) / (self.keep - 1)
        idx |= {int(round(i * step)) for i in range(self.keep)}
        return [items[i] for i in sorted(idx)]


@gpu_env.task(report=True)
async def train(
    task_id: str = "dmc_arena_walk",
    config: str = "dmc_vision",
    size: str = "size12m",
    steps: int = _DEFAULT_STEPS,
    envs: int = 4,
    replay_steps: int = 600,
    save_every: int = 900,
) -> tuple[dict, flyte.io.Dir | None]:
    """Train DreamerV3 on a DMC task and report what the world model learned.

    Returns the run summary and the trained agent. The second half is what makes a run
    reusable: `config.yaml` plus `ckpt`, which is everything replay.py needs to load
    the policy again without retraining it.

    Defaults are the flagship: the custom `arena` domain (see arena.py) learned from
    pixels, at the 12M parameter preset.

      task_id  `dmc_arena_walk` is walker_walk's reward in a world with posts and
               kickable balls. `dmc_walker_walk` is the stock domain, for comparison.
      config   `dmc_vision` learns from 64x64 pixels and is the only setting that
               produces a dream video. `dmc_proprio` learns from a state vector.
      size     upstream's presets. `dmc_vision` alone means size200m, which is 20x the
               compute per gradient step for a task this small.
      envs     upstream defaults to 16, which delays the first finished episode to
               step 16,000. Training is the bottleneck here, not the simulator, so
               fewer environments costs no throughput and gives the live report a
               score and a rollout video four times sooner.
      save_every  seconds between checkpoint writes, upstream's `run.save_every`.
               Worth passing explicitly because the default is 900, so any run
               shorter than fifteen minutes finishes having written no checkpoint at
               all, and `_persist` then uploads a directory with no agent in it. That
               failure is silent: the run succeeds and the Dir exists.
    """
    here = Path(__file__).parent.resolve()
    logdir = Path("/tmp/dreamer") / task_id
    logdir.mkdir(parents=True, exist_ok=True)
    metrics = logdir / "metrics.jsonl"

    argv = [
        sys.executable, str(here / "launch.py"),
        "--configs", config, size,
        "--task", task_id,
        "--logdir", str(logdir),
        "--run.steps", str(steps),
        "--run.envs", str(envs),
        # See the module docstring: log_every is a wall-clock timer whose default is
        # minutes, which is how a working run produces an empty logdir. report_every
        # is the one that governs how often a new dream clip appears.
        "--run.log_every", "60",
        "--run.report_every", "180",
        "--run.save_every", str(save_every),
        "--replay.size", str(_REPLAY_SIZE),
        # jax preallocates 75% of device memory at startup, and on a GB10 device memory
        # IS system memory: measured at 90 GB reserved on the host for a 10M parameter
        # model. In a pod with a cgroup limit that is an instant kill.
        "--jax.prealloc", "False",
    ]
    log.info("launching: %s", " ".join(argv))

    t0 = time.monotonic()
    tail: list[str] = []
    last_paint = 0.0
    film = _Film()
    env = dict(
        os.environ,
        # launch.py has to find both arena.py (next to it) and dreamerv3 (in the image).
        PYTHONPATH=f"{here}:{DREAMER_ROOT}",
    )
    proc = subprocess.Popen(
        argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        bufsize=1, cwd=str(logdir), env=env,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.rstrip()
        tail.append(line)
        del tail[:-80]
        # Echo the interesting lines to the task's own stdout. Consuming the child's
        # output without re-emitting it leaves `kubectl logs` completely silent for
        # the whole run, so the only way to see whether a seven hour job is alive is
        # to open the report. Dreamer's terminal output is one banner block per log
        # flush, so this is a handful of lines a minute, not a firehose.
        if line.startswith(("---", "Start", "Logdir", "Error", "Traceback")) or (
            " / " in line and "score" in line
        ):
            print(line, flush=True)
        now = time.monotonic()
        if now - last_paint >= _REPAINT_SECS:
            last_paint = now
            film.refresh(logdir)
            data = _read_metrics(metrics)
            done = int(data["score"][-1][0]) if data["score"] else 0
            flyte.report.replace(
                reports.progress_html(
                    task_id, config, size, done, steps, data, film,
                    time.monotonic() - t0,
                ),
                do_flush=True,
            )
    rc = proc.wait()
    secs = time.monotonic() - t0

    film.refresh(logdir)
    data = _read_metrics(metrics)

    # rc alone is not trusted, for the same reason the Isaac Sim demo stopped trusting
    # it: a trainer that dies early can still exit 0, and an empty metrics file with a
    # clean exit is the signature. Fail here, while the log tail still explains it.
    if rc != 0:
        raise RuntimeError(f"dreamer exited {rc}. log tail:\n" + "\n".join(tail[-25:]))
    if not data["losses"] and not data["score"]:
        raise RuntimeError(
            "dreamer exited 0 but logged no metrics at all, which means it did not "
            "train. log tail:\n" + "\n".join(tail[-25:])
        )

    # Film the trained policy at a resolution a human can watch. The live rollout in
    # the report is the agent's own 64x64 observation; this one re-renders the same
    # scene at 480x480 so the arena is actually legible. Never allowed to fail the
    # run: training is the expensive part and a camera problem must not throw it away.
    video_html, clip_probe = "", ""
    try:
        frames, ep_score, travelled = replay.record(logdir, steps=replay_steps)
        mp4 = replay.encode(frames)
        clip_probe = replay.probe(mp4)
        if mp4:
            video_html = replay.video_html(
                mp4,
                f"{len(mp4) / 1024:.0f} KB &middot; trained policy, episode return "
                f"{ep_score:.1f}, travelled {travelled:.1f} m &middot; {clip_probe}",
            )
        log.info("replay: %s", clip_probe)
    except Exception as exc:  # noqa: BLE001
        log.warning("replay failed, reporting without video: %s", exc)
        clip_probe = f"replay failed: {exc}"

    flyte.report.replace(
        reports.final_html(
            task_id, config, size, steps, secs, data, film, data["params"], tail,
            video_html, clip_probe,
        ),
        do_flush=True,
    )
    score = data["score"]
    best = max((y for _, y in score), default=0.0)
    far = max((y for _, y in data["distance"]), default=0.0)
    log.info("trained %s for %s steps, best score %.1f", task_id, steps, best)
    model = await _persist(logdir)
    return {
        "task": task_id,
        "config": f"{config} {size}",
        "steps": steps,
        "minutes": round(secs / 60, 1),
        "episodes": len(score),
        "score_first": round(score[0][1], 2) if score else 0.0,
        "score_final": round(score[-1][1], 2) if score else 0.0,
        "score_best": round(best, 2),
        "metres_best": round(far, 2),
        "params": data["params"],
        "dream_clips": film.latest.get("dream", {}).get("count", 0),
        "loss_keys": sorted(data["losses"]),
        "clip": clip_probe,
    }, model


@orch_env.task(report=True)
async def dream(
    task_id: str = "dmc_arena_walk",
    config: str = "dmc_vision",
    size: str = "size12m",
    steps: int = _DEFAULT_STEPS,
    envs: int = 4,
    replay_steps: int = 600,
    save_every: int = 900,
) -> tuple[dict, flyte.io.Dir | None]:
    """Entry point. CPU-only orchestrator so it cannot deadlock its own GPU child."""
    result, model = await train(
        task_id=task_id, config=config, size=size, steps=steps,
        envs=envs, replay_steps=replay_steps, save_every=save_every,
    )
    log.info("result: %s", result)
    log.info("model: %s", model.path if model else "not persisted")
    return result, model


if __name__ == "__main__":
    flyte.init_from_config()
    print(flyte.run(dream))
