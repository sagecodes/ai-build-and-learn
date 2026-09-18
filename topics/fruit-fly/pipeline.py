"""The orchestrators: run a brain built from a real fly's wiring, in a simulated body.

Three entry points:

    # The main event. One connectome fly plus the three controls, one report.
    flyte run pipeline.py fixate

    # Just the connectome fly, longer, no controls. Fastest way to see the clip.
    flyte run pipeline.py fixate --modes '["connectome"]' --ticks 300

    # Brain only, no body: which sensory channels lateralise at all?
    flyte run pipeline.py probe

Everything runs in the `physical-ai` project, the same one as the rl-mujoco G1 and the
Isaac Sim demos.

── A note on "parallel" ────────────────────────────────────────────────────────
`fixate` submits its four runs concurrently and they really do run concurrently: this
demo never asks for the GPU, so nothing here contends for the Spark's single one. Each
run asks for 8 CPUs of the box's 20, so expect two at a time and the rest Pending.
Budget roughly 2x the single-run time for the full set of four.

The orchestrator is CPU-only and tiny, like every other orchestrator in this repo, for
the reason learned the hard way in the videogen demo: an orchestrator pod holds its
resources for as long as its children run, so a fat orchestrator starves its own
children.
"""

from __future__ import annotations

import asyncio
import json
import logging
import pathlib
import tempfile

import flyte
import flyte.report
import numpy as np
from flyte.io import File

from config import brain_env, orch_env
import body as BODY
import brain as BRAIN
import brainviz as VIZ
import bridge as BR          # noqa: F401  (imported by run; named here so flyte bundles it)
import chemistry as CHEM
import compass as COMPASS
import connectome as C
import duet as DUET
import learn as LEARN
import rider as RIDE       # noqa: F401  (used by run.ride_loop; named so flyte bundles it)      # noqa: F401  (used by run.duet_loop; named here so flyte bundles it)
import reports
import run as RUN
import video as VID

# Every sibling module is imported AT TOP LEVEL, and that is load-bearing rather than
# tidy: `flyte run` walks the entrypoint's top-level imports to decide what to bundle
# into the task pod. A module imported lazily inside a function body is not seen, the
# pod comes up without it, and the run dies with ModuleNotFoundError several minutes
# in. None of these pull brian2 or flygym at import time (both are imported inside the
# functions that need them), so the orchestrator stays cheap to start.

logging.basicConfig(level=logging.WARNING, format="%(message)s", force=True)
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
# The sibling modules that do their own measuring and reporting. Without this their
# log.info lines are swallowed by the WARNING baseline and a task that worked perfectly
# prints nothing at all, which is how the first `teach` run came back empty.
for _module in ("compass", "learn", "run", "bridge", "duet"):
    logging.getLogger(_module).setLevel(logging.INFO)

DEFAULT_MODES = ["connectome", "shuffled", "blind", "nobrain"]


# ── Brain only ──────────────────────────────────────────────────────────────────


@brain_env.task(report=True)
async def probe(duration: float = 0.3, rate_hz: float = 150.0) -> str:
    """Drive each sensory channel one side at a time; report descending lateralisation.

    No body, no physics, no video. This is the experiment that decides what the
    embodied demo can be built on, and it is worth running on its own because the
    answer is not obvious: the fly's most famous steering cells fail this test and its
    olfactory system does too.
    """
    BRAIN.preimport()
    conn = C.load()
    dn_left, dn_right = conn.sided(super_class="descending")

    await flyte.report.replace.aio(
        reports.header("brain only: which sensory channels can steer?")
        + reports.note(f"138,639 neurons. Each condition is a fresh brain driven for "
                       f"{duration * 1000:.0f} ms at {rate_hz:.0f} Hz.")
    )
    await flyte.report.flush.aio()

    conditions = []
    for channel, criteria in C.SENSORY_GROUPS.items():
        left, right = conn.sided(**criteria)
        for side_label, idx in (("left", left), ("right", right)):
            if len(idx) == 0:
                continue
            rates = BRAIN.probe(conn, {"stim": idx}, duration=duration, rate_hz=rate_hz)
            conditions.append(
                {
                    "label": f"{channel} ({len(idx):,} neurons), {side_label}",
                    "dn_left": float(rates[dn_left].sum()),
                    "dn_right": float(rates[dn_right].sum()),
                    "active": int((rates > 1).sum()),
                }
            )
            log.info("%s -> DN L=%.0f R=%.0f", conditions[-1]["label"],
                     conditions[-1]["dn_left"], conditions[-1]["dn_right"])
            await flyte.report.replace.aio(
                reports.header("brain only: which sensory channels can steer?")
                + reports.probe_panel(conditions)
            )
            await flyte.report.flush.aio()

    return json.dumps(conditions)


@brain_env.task(report=True)
async def looming_probe(
    ticks: int = 80,
    object_radius: float = 1.8,
    start_distance: float = 18.0,
    speed: float = 22.0,
    w_syn_values: list[float] | None = None,
) -> str:
    """Swat at the fly with no brain in the loop, then replay the stimulus two ways.

    Three questions, none of which need a body in the loop:

    1. **Where does a looming stimulus die?** `connectome.VISUAL_PATHWAY` walks the eye
       to the legs, stage by named stage, counting how many cells of each fire at the
       closest point of the approach. This is the single most informative measurement in
       the repo, because it says exactly where the signal stops rather than only that
       the fly did not react.

    2. **Do the escape cells fire?** LPLC2 is the canonical looming detector and DNp01
       is the giant fibre. Measured here at every synaptic gain tried: LPLC2 0 of 210.

    3. **Is this brain sensitive to APPROACH, or only to size?** The stimulus is
       captured once from the physics (a real sphere really flying at a real fly, with
       the fly standing still so the trace is clean) and then replayed into a fresh
       brain forwards and backwards. A looming and a receding disc sweep exactly the
       same retinal sizes in opposite time order, so a brain with any motion sensitivity
       must respond differently to the two.

       That comparison needs a NOISE CEILING to mean anything, and the first version of
       this task did not have one. At the published synaptic gain only about 13
       descending neurons fire at all, so the summed rate is a handful of spikes per
       15 ms window and two runs of the SAME stimulus already disagree. Measured that
       way, looming against reversed-receding correlated at only r = +0.37, which looks
       like a difference and is really just noise. So this runs the forward stimulus
       twice with different seeds to measure how well the brain agrees with ITSELF, and
       the reversal is scored against that ceiling rather than against 1.0. If the two
       correlations match, the brain cannot tell an approach from a retreat.
    """
    BRAIN.preimport()
    conn = C.load()
    w_syn_values = w_syn_values or [0.275, 0.55]

    await flyte.report.replace.aio(
        reports.header("the swat, brain only: where does a looming stimulus die?")
        + reports.note("capturing the stimulus from the physics...")
    )
    await flyte.report.flush.aio()

    # ── The stimulus, measured rather than modelled ─────────────────────────────
    # A sphere really flown at a real fly whose legs are still, so the darkening trace
    # is the object's and not the walk's. Two numbers per tick, exactly reversible.
    def capture() -> list[tuple[float, float]]:
        scene = BODY.Scene(
            object_radius=object_radius, object_type="sphere", spawn_heading_deg=0.0,
            motion="loom", motion_delay_ticks=0, motion_speed=speed,
            motion_start_distance=start_distance,
        )
        world = BODY.FlyWorld(scene=scene)
        trace = []
        for _ in range(ticks):
            eyes = world.look()
            trace.append((float(eyes[0]), float(eyes[1])))
            world.step(np.zeros(2))
        world.close()
        return trace

    stimulus = await asyncio.to_thread(capture)
    peak = max(l + r for l, r in stimulus)
    log.info("captured %d ticks, peak darkening %.3f", len(stimulus), peak)

    def replay(wiring, trace, w_syn: float, seed: int = 0) -> dict:
        """Drive the photoreceptors with a recorded darkening trace; read everything."""
        link_eyes = wiring.sided(**C.SENSORY_GROUPS["photoreceptors"])
        dn_left, dn_right = wiring.sided(super_class="descending")
        params = dict(BRAIN.PARAMS)
        params["w_syn"] = w_syn * 1e-3
        brain = BRAIN.Brain(
            conn=wiring,
            stim_groups={"eye_left": link_eyes[0], "eye_right": link_eyes[1]},
            params=params,
            seed=seed,
        )
        totals, per_stage, escape_peak = [], {}, {}
        stage_idx = {label: wiring.indices(**crit) for label, crit in C.VISUAL_PATHWAY}
        stage_active = {label: 0 for label in stage_idx}
        escape_idx = {
            label: wiring.indices(cell_type=types) for label, types in C.ESCAPE_TYPES.items()
        }
        for left, right in trace:
            brain.set_rate("eye_left", min(BR.TONIC_HZ + left * BR.DEFAULT_EYE_GAIN, BR.MAX_EYE_HZ))
            brain.set_rate("eye_right", min(BR.TONIC_HZ + right * BR.DEFAULT_EYE_GAIN, BR.MAX_EYE_HZ))
            tick = brain.run(BODY.BRAIN_DT)
            totals.append(tick.rate(dn_left) + tick.rate(dn_right))
            for label, idx in stage_idx.items():
                stage_active[label] = max(stage_active[label], int((tick.counts[idx] > 0).sum()))
            for label, idx in escape_idx.items():
                escape_peak[label] = max(escape_peak.get(label, 0.0), tick.rate(idx))
        per_stage = {
            label: {"active": stage_active[label], "cells": int(len(stage_idx[label]))}
            for label, _ in C.VISUAL_PATHWAY
        }
        return {"dn_total": totals, "stages": per_stage, "escape": escape_peak}

    def smooth(values, window: int = 5) -> np.ndarray:
        """Boxcar. The raw summed rate is a few spikes per tick and correlating it
        unsmoothed measures the Poisson process, not the brain."""
        v = np.asarray(values, float)
        if len(v) < window:
            return v
        kernel = np.ones(window) / window
        return np.convolve(v, kernel, mode="same")

    def corr(a, b) -> float:
        a, b = smooth(a), smooth(b)
        if a.std() <= 0 or b.std() <= 0:
            return float("nan")
        return float(np.corrcoef(a, b)[0, 1])

    out: dict = {"stimulus": stimulus, "gains": {}}
    for w_syn in w_syn_values:
        forward = await asyncio.to_thread(replay, conn, stimulus, w_syn, 0)
        repeat = await asyncio.to_thread(replay, conn, stimulus, w_syn, 1)
        backward = await asyncio.to_thread(replay, conn, stimulus[::-1], w_syn, 0)
        f = np.asarray(forward["dn_total"], float)
        b = np.asarray(backward["dn_total"], float)[::-1]
        ceiling = corr(f, repeat["dn_total"])
        reversal = corr(f, b)
        stim = np.asarray([l + r for l, r in stimulus], float)

        def saturation(values) -> float:
            """Fraction of the run spent within 20% of a ROBUST peak rate.

            Context for the ignited regime: above the published synaptic gain this
            network can switch itself on and stay on, and a response pinned near its
            ceiling is not tracking anything. Measured against the 95th percentile
            rather than the maximum, because at these rates the maximum is a single
            Poisson outlier and 0.8 times it can land near the mean, which made an
            ordinary ramping response read as 47% saturated.
            """
            v = smooth(values)
            ceiling = float(np.percentile(v, 95))
            return float((v > 0.8 * ceiling).mean()) if ceiling > 0 else 0.0

        out["gains"][f"{w_syn}"] = {
            "forward": forward["dn_total"],
            "repeat": repeat["dn_total"],
            "backward_reversed": b.tolist(),
            "time_reversal_corr": reversal,
            "noise_ceiling_corr": ceiling,
            "stimulus_corr": corr(f, stim),
            # Does the receding run track its OWN stimulus? If it does not, the reversal
            # test has nothing to say and the saturation number explains why.
            "stimulus_corr_backward": corr(backward["dn_total"], stim[::-1]),
            "saturation_forward": saturation(f),
            "saturation_backward": saturation(backward["dn_total"]),
            "stages": forward["stages"],
            "escape": forward["escape"],
        }
        blob = out["gains"][f"{w_syn}"]
        log.info(
            "w_syn %.3f: reversal r=%+.3f, ceiling r=%+.3f, stimulus r=%+.3f/%+.3f "
            "(fwd/bwd), saturation %.0f%%/%.0f%%",
            w_syn, reversal, ceiling, blob["stimulus_corr"], blob["stimulus_corr_backward"],
            100 * blob["saturation_forward"], 100 * blob["saturation_backward"],
        )
        await flyte.report.replace.aio(
            reports.header("the swat, brain only: where does a looming stimulus die?")
            + reports.looming_panel(out)
        )
        await flyte.report.flush.aio()

    return json.dumps(out)


@brain_env.task(report=True)
async def compass(
    positions: int = 8,
    duration: float = 0.3,
    w_syn_mv: float = 0.55,
    readouts: list[str] | None = None,
    seed: int = 0,
) -> str:
    """Find the fly's head-direction ring in the wiring, then put a bump on it.

    The central complex holds the insect compass: EPG neurons tile the ellipsoid body
    into wedges, and in a living fly one bump of activity sits on that ring and tracks
    which way the animal is facing. This task asks whether that ring is recoverable from
    a connectome with no anatomy and no labels, and then whether it behaves like one.

    Two halves, and the first is a measurement in its own right. The FlyWire annotation
    calls all 47 EPG cells "EPG" with no wedge number, so their order around the ring is
    simply not in the metadata. `compass.embed_ring` recovers it by spectral embedding of
    which Delta7 cells each EPG shares, and reports a pass/fail number with it: in the
    recovered order, connectivity similarity has to fall off with circular distance.
    Measured, r = -0.78.

    The second half drives six adjacent cells on that recovered ring at eight positions
    around it and asks where the response lands downstream. Measured, w_syn 0.55:

        Delta7   slope -1, residual 5.2 deg RMS, circular correlation -0.976
        PFL3     slope -1, residual 13.0 deg RMS, circular correlation -0.989

    Five degrees. Move the bump, and the inhibitory ring that surrounds it moves by the
    same amount. PFL3 is the population that turns the compass into a steering command,
    so this is the last stage before the descending neurons.

    Nothing here has a body, which makes it the cheapest task in the repo: no physics,
    no rendering, and the answer in about a minute.
    """
    BRAIN.preimport()
    conn = C.load()
    readouts = readouts or ["Delta7", "PFL3"]

    await flyte.report.replace.aio(
        reports.header("the compass: a ring attractor, recovered and then driven")
        + reports.note("recovering the ring order from connectivity...")
    )
    await flyte.report.flush.aio()

    def build_rings(wiring):
        driven = COMPASS.embed_ring(wiring, "EPG", COMPASS.partner_for("EPG"))
        reads = {
            name: COMPASS.embed_ring(wiring, name, COMPASS.partner_for(name))
            for name in readouts
        }
        return driven, reads

    driven, reads = await asyncio.to_thread(build_rings, conn)

    def as_dict(ring) -> dict:
        return {
            "name": ring.name,
            "x": ring.x.tolist(),
            "y": ring.y.tolist(),
            "quality": ring.quality,
            "radius_cv": ring.radius_cv,
            "eigengap": ring.eigengap,
            "cells": int(len(ring)),
        }

    out: dict = {
        "rings": {r.name: as_dict(r) for r in [driven, *reads.values()]},
        "conditions": {},
    }
    await flyte.report.replace.aio(
        reports.header("the compass: a ring attractor, recovered and then driven")
        + reports.compass_panel(out)
    )
    await flyte.report.flush.aio()

    # The shuffled control keeps the REAL rings, because the question is whether the
    # real wiring is what carries the bump around them. Re-embedding the shuffle would
    # ask a different and much weaker question.
    for label, wiring in (("the real wiring", conn), ("shuffled wiring", conn.shuffled(seed=seed))):
        records = await asyncio.to_thread(
            COMPASS.bump_response, wiring, driven, reads,
            positions, duration, w_syn_mv, seed,
        )
        out["conditions"][label] = {
            "records": records,
            "fits": {name: COMPASS.tracking_fit(records, name) for name in reads},
        }
        for name, fit in out["conditions"][label]["fits"].items():
            log.info(
                "%-16s %-7s slope %+.0f, residual %.1f deg, circular r %+.3f",
                label, name, fit["slope"], fit["residual_deg"], fit["circular_corr"],
            )
        await flyte.report.replace.aio(
            reports.header("the compass: a ring attractor, recovered and then driven")
            + reports.compass_panel(out)
        )
        await flyte.report.flush.aio()

    return json.dumps(
        {
            "rings": {k: {"quality": v["quality"], "cells": v["cells"]}
                      for k, v in out["rings"].items()},
            "fits": {label: block["fits"] for label, block in out["conditions"].items()},
        }
    )


# ── One embodied run ────────────────────────────────────────────────────────────


@brain_env.task(report=True, retries=1)
async def fly_once(
    mode: str = "connectome",
    ticks: int = 200,
    spawn_heading_deg: float = 60.0,
    object_distance: float = 12.0,
    object_radius: float = 3.0,
    behaviour: str = "attract",
    turn_gain: float = 1.8,
    base_drive: float = 0.5,
    silence: str = "",
    w_syn_mv: float = 0.0,
    food: str = "none",
    object_type: str = "cylinder",
    motion: str = "none",
    motion_speed: float = 22.0,
    motion_delay_ticks: int = 25,
    startle_gain: float = 0.0,
    arena: str = "ground",
    drum_bars: int = 12,
    drum_speed_deg_s: float = 0.0,
    n_pokes: int = 0,
    poke_every: int = 35,
    poke_len: int = 14,
    poke_first: str = "left",
    touch_hz: float = BR.TOUCH_HZ,
    seed: int = 0,
) -> File:
    """One fly, one arena, `ticks` brain ticks. Returns its clip; reports as it goes.

    The task's own report carries the video and the inside-the-loop traces. The
    orchestrator's report carries the comparison across modes, because a single run in
    isolation cannot tell you whether anything was learned.
    """
    # Must happen on the main thread, before the to_thread below. See brain.preimport.
    BRAIN.preimport()
    conn = C.load()
    scene = BODY.Scene(
        object_xy=(object_distance, 0.0),
        object_radius=object_radius,
        spawn_heading_deg=spawn_heading_deg,
        food=food,
        object_type=object_type,
        motion=motion,
        motion_speed=motion_speed,
        motion_delay_ticks=motion_delay_ticks,
        motion_start_distance=object_distance,
        arena=arena,
        drum_bars=drum_bars,
        drum_speed_deg_s=drum_speed_deg_s,
        drum_radius=object_distance if arena == "drum" else 12.0,
        # A tethered fly is held clear of the ground by a mocap body, so it spawns a
        # little higher than a walking one and its legs hang.
        spawn_z=2.0 if arena == "drum" else 0.5,
        # A drop of syrup, not a monolith. The fly cannot tell sugar from bitter by
        # looking, so both are the same shape and the same colour: only the taste
        # circuit can separate them.
        object_rgba=(0.05, 0.05, 0.05, 1.0) if food == "none" else (0.09, 0.06, 0.03, 1.0),
    )

    # Alternating pokes, left then right, so a single run contains both directions and
    # the comparison is within-animal rather than between two runs.
    sides = ("left", "right") if poke_first == "left" else ("right", "left")
    pokes = [
        (poke_every * (k + 1), poke_len, sides[k % 2]) for k in range(max(n_pokes, 0))
    ]

    lesion = f" &nbsp;|&nbsp; silenced: <b>{silence}</b>" if silence else ""

    def head(extra: str = "") -> str:
        return reports.header(
            f"mode <b>{mode}</b> - {reports.MODE_BLURB.get(mode, '')}{lesion}{extra}"
        )

    await flyte.report.replace.aio(
        head()
        + reports.connectome_panel(conn.describe(), VIZ.legend_rows(conn))
        + reports.note("building the network and calibrating...")
    )
    await flyte.report.flush.aio()

    progress = {"html": ""}

    def on_tick(done: int, total: int, traces: dict) -> None:
        bearing = traces["bearing"][-1] if traces["bearing"] else 0.0
        distance = traces["distance"][-1] if traces["distance"] else 0.0
        progress["html"] = reports.note(
            f"tick {done}/{total} - {distance:.1f} mm from the pillar, "
            f"bearing {bearing:+.0f} deg"
        )

    result = await asyncio.to_thread(
        RUN.closed_loop,
        conn,
        mode=mode,
        ticks=ticks,
        scene=scene,
        behaviour=behaviour,
        turn_gain=turn_gain,
        base_drive=base_drive,
        seed=seed,
        silence=silence,
        w_syn_mv=w_syn_mv,
        startle_gain=startle_gain,
        pokes=pokes,
        touch_hz=touch_hz,
        on_tick=on_tick,
    )

    mp4 = VID.encode(result.frames)
    cal_rows = result.calibration.table() if result.calibration else None
    still = result.frames[len(result.frames) // 2] if result.frames else None
    await flyte.report.replace.aio(
        head()
        + reports.run_panel(result, mp4, cal_rows)
        + reports.traces_panel(result.traces)
        + reports.chemistry_panel(result.traces)
        + (
            reports.leaderboard_panel(conn, result.total_counts, result.duration_s)
            if result.total_counts is not None and result.total_counts.sum() > 0
            else ""
        )
        + reports.connectome_panel(conn.describe(), VIZ.legend_rows(conn))
    )
    await flyte.report.flush.aio()
    log.info(result.summary())

    # The clip goes out as a File and the numbers ride along in a sidecar json, so the
    # orchestrator can build the comparison without re-running anything.
    out = pathlib.Path(tempfile.mkdtemp()) / f"{mode}.mp4"
    out.write_bytes(mp4)
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "mode": result.mode if not silence else f"-{silence}",
                "food": food,
                "motion": motion,
                "arena": arena,
                "n_pokes": n_pokes,
                "w_syn_mv": result.w_syn_mv,
                "ticks": result.ticks,
                "duration_s": result.duration_s,
                "wall_s": result.wall_s,
                "total_spikes": result.total_spikes,
                "metrics": result.metrics,
                "trajectory": result.trajectory,
                "traces": {k: v for k, v in result.traces.items()},
            }
        )
    )
    return await File.from_local(str(meta))


@brain_env.task(report=True, retries=1)
async def duet_once(
    mode_a: str = "connectome",
    mode_b: str = "connectome",
    ticks: int = 240,
    object_distance: float = 12.0,
    object_radius: float = 3.0,
    turn_gain: float = 1.8,
    base_drive: float = 0.5,
    marker_radius: float = 0.0,
    retina_mode: str = "contrast",
    seed: int = 0,
) -> File:
    """One arena, two flies, two independent brains. Returns the clip and the numbers."""
    BRAIN.preimport()
    conn = C.load()
    scene = BODY.Scene(
        object_xy=(object_distance, 0.0), object_radius=object_radius,
    )
    label = f"{mode_a} vs {mode_b}"

    await flyte.report.replace.aio(
        reports.header(f"two flies, two brains: {label}")
        + reports.note("building two whole-brain networks...")
    )
    await flyte.report.flush.aio()

    progress = {"html": ""}

    def on_tick(done: int, total: int, traces: dict) -> None:
        progress["html"] = reports.note(
            f"tick {done}/{total} - A is {traces['a']['distance'][-1]:.1f} mm out, "
            f"B is {traces['b']['distance'][-1]:.1f} mm out"
        )

    result = await asyncio.to_thread(
        RUN.duet_loop, conn,
        modes=(mode_a, mode_b), ticks=ticks, scene=scene, turn_gain=turn_gain,
        base_drive=base_drive, marker_radius=marker_radius, retina_mode=retina_mode,
        seed=seed, on_tick=on_tick,
    )

    mp4 = VID.encode(result.frames)
    await flyte.report.replace.aio(
        reports.header(f"two flies, two brains: {label}")
        + reports.duet_panel(result, mp4)
    )
    await flyte.report.flush.aio()
    log.info(result.summary())

    out = pathlib.Path(tempfile.mkdtemp()) / f"duet-{mode_a}-{mode_b}.mp4"
    out.write_bytes(mp4)
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps(
            {
                "label": label,
                "modes": result.modes,
                "ticks": result.ticks,
                "duration_s": result.duration_s,
                "wall_s": result.wall_s,
                "metrics": result.metrics,
                "trajectories": result.trajectories,
                "traces": result.traces,
                "separation": result.separation,
                "total_spikes": result.total_spikes,
            }
        )
    )
    return await File.from_local(str(meta))


@orch_env.task(report=True)
async def duet(
    ticks: int = 240,
    object_distance: float = 12.0,
    marker_radius: float = 0.0,
    seed: int = 0,
) -> str:
    """Two flies, two independent connectome brains, one arena and one pillar.

    Everything else in this repo is one animal alone. This is the same brain twice, in
    the same physics, at the same moment, and it buys two things the single-fly demo
    cannot have.

    First, a **simultaneous control**. The second condition puts a real connectome
    against a shuffled one side by side in one arena: same lighting, same floor, same
    pillar, same instant. The two animals cannot differ by anything except their wiring,
    which is a stronger statement than running them one after the other.

    Second, **interference**. The flies are solid. They collide, block each other, and
    occlude each other's view of the target, and none of that is scripted.

    What they are NOT is social. Measured before any of this was built, one fly looking
    at another across an empty arena darkens 0 of its 721 ommatidia by the absolute test
    this repo's retina uses, and 4, 3, 2 and 1 facets at 4, 6, 9 and 12 mm by a contrast
    test. An amber fly against a bright floor is about an order of magnitude weaker than
    the fixation pillar, which is itself close to this model's noise floor. So these two
    will not court or chase, and `marker_radius` exists to test that claim rather than
    argue it: give each fly a dark dorsal marker, the way a real experimenter paints
    one, and see whether anything changes.
    """
    conditions = [("connectome", "connectome"), ("connectome", "shuffled")]
    await flyte.report.replace.aio(
        reports.header("two flies, two brains, one pillar")
        + reports.note(
            f"{ticks} ticks ({ticks * BODY.BRAIN_DT:.1f} s of fly time) each. Two whole "
            f"brains run in one process, which costs about 2.6 GB and half the speed of "
            f"a single fly."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            duet_once(
                mode_a=a, mode_b=b, ticks=ticks, object_distance=object_distance,
                marker_radius=marker_radius, seed=seed,
            )
            for a, b in conditions
        ]
    )
    blobs = []
    for handle in files:
        with open(await handle.download()) as f:
            blobs.append(json.load(f))

    await flyte.report.replace.aio(
        reports.header("two flies, two brains, one pillar")
        + reports.duet_comparison_panel(blobs, (object_distance, 0.0), 3.0)
        + reports.note(
            "Each condition's clip and inside-the-loop traces live in its own task "
            "report; this page is the comparison."
        )
    )
    await flyte.report.flush.aio()

    summary = {b["label"]: b["metrics"] for b in blobs}
    for label, metrics in summary.items():
        for name, m in metrics.items():
            log.info(
                "%-26s fly %s closest %.1f mm, approach %+.1f mm, crowded %d ticks",
                label, name, m["closest_distance"], m["approach"],
                int(m["ticks_crowded"]),
            )
    return json.dumps(summary)


# ── The comparison ──────────────────────────────────────────────────────────────


@orch_env.task(report=True)
async def fixate(
    modes: list[str] | None = None,
    ticks: int = 200,
    spawn_heading_deg: float = 60.0,
    object_distance: float = 12.0,
    object_radius: float = 3.0,
    behaviour: str = "attract",
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """Run the connectome fly and its controls, then put them in one report.

    `modes` defaults to all four. Trimming it to ["connectome"] is the fast path for
    looking at the clip; keeping "shuffled" in is what makes the result a claim.
    """
    modes = modes or DEFAULT_MODES
    unknown = [m for m in modes if m not in ("connectome", "shuffled", "blind", "nobrain")]
    if unknown:
        raise ValueError(f"unknown modes {unknown}")

    await flyte.report.replace.aio(
        reports.header(
            f"fixation experiment: {', '.join(modes)} - "
            f"{ticks} ticks ({ticks * BODY.BRAIN_DT:.1f} s of fly time) each"
        )
        + reports.note(
            "Each mode is a separate task; open them from the run's task list to see "
            "the video and the inside-the-loop traces for that one. This page is the "
            "comparison."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            fly_once(
                mode=mode,
                ticks=ticks,
                spawn_heading_deg=spawn_heading_deg,
                object_distance=object_distance,
                object_radius=object_radius,
                behaviour=behaviour,
                turn_gain=turn_gain,
                seed=seed,
            )
            for mode in modes
        ]
    )

    results = []
    for handle in files:
        with open(await handle.download()) as f:
            results.append(_Loaded(json.load(f)))

    await flyte.report.replace.aio(
        reports.header(
            f"fixation experiment: {', '.join(modes)} - "
            f"{ticks * BODY.BRAIN_DT:.1f} s of fly time each"
        )
        + reports.comparison_panel(results, (object_distance, 0.0), object_radius)
        + reports.note(
            "Videos and per-run traces live in each mode's own task report."
        )
    )
    await flyte.report.flush.aio()

    summary = {r.mode: r.metrics for r in results}
    for mode, metrics in summary.items():
        log.info(
            "%-11s closest %.1f mm, net approach %+.1f mm, |bearing| %.0f deg",
            mode, metrics["closest_distance"], metrics["approach"],
            metrics["mean_abs_bearing_late"],
        )
    return json.dumps(summary)


@orch_env.task(report=True)
async def swat(
    ticks: int = 90,
    object_radius: float = 1.8,
    start_distance: float = 18.0,
    speed: float = 22.0,
    startle_gain: float = 1.0,
    seed: int = 0,
) -> str:
    """Swat at the fly. Does anything in this brain know that something is coming?

    A 1.8 mm sphere flies in at 22 mm/s from 18 mm out, aimed at where the fly is
    standing, and arrives about 0.7 s later. That is a big stimulus: measured, total
    darkening across the two eyes goes from 0.015 to 0.287, a 19-fold swing, against
    the 0.075 the stationary fixation pillar manages.

    Three embodied conditions, and the middle one is the control that matters:

        loom     the sphere flies in
        recede   the same path walked backwards, from contact outward. Identical
                 retinal sizes, opposite time order.
        static   the sphere sits at its launch point and never moves.

    Fixation reads the DIFFERENCE between the two descending populations. A looming
    object is symmetric, so it is close to invisible in that difference and shows up
    only in the SUM, which is a channel this repo did not previously read. `startle_gain`
    is the one hand-made number: it scales the walking drive down in proportion to how
    far the summed descending rate sits above its own baseline, the same kind of explicit
    choice as `behaviour="attract"`. Set it to 0 to watch the measurement with the legs
    left out of it.

    `looming_probe` runs alongside with no body at all and carries the honest part: the
    stage-by-stage autopsy of where the signal stops, and the time-reversal test.
    """
    await flyte.report.replace.aio(
        reports.header("the swat: a sphere flies at the fly")
        + reports.note(
            f"A {object_radius * 2:.1f} mm sphere closing from {start_distance:.0f} mm "
            f"at {speed:.0f} mm/s. The receding run is the same path backwards, which "
            f"sweeps the same retinal sizes in the opposite time order."
        )
    )
    await flyte.report.flush.aio()

    conditions = [("loom", "loom"), ("recede", "recede"), ("static", "none")]
    gathered = await asyncio.gather(
        *[
            fly_once(
                mode="connectome", ticks=ticks, object_distance=start_distance,
                object_radius=object_radius, object_type="sphere",
                spawn_heading_deg=0.0,
                # turn_gain 0 keeps the fly walking straight. With steering left on it
                # chases the sphere, the approach geometry stops being the experiment,
                # and the retinal trace becomes a record of the walk instead of the swat.
                turn_gain=0.0, base_drive=0.35,
                motion=motion, motion_speed=speed, startle_gain=startle_gain, seed=seed,
            )
            for _, motion in conditions
        ],
        looming_probe(ticks=ticks, object_radius=object_radius,
                      start_distance=start_distance, speed=speed),
    )
    files, probe_json = gathered[:-1], gathered[-1]
    probe_data = json.loads(probe_json)

    results = []
    for handle, (label, _) in zip(files, conditions):
        with open(await handle.download()) as f:
            loaded = _Loaded(json.load(f))
        loaded.mode = label
        results.append(loaded)

    await flyte.report.replace.aio(
        reports.header("the swat: a sphere flies at the fly")
        + reports.swat_panel(results)
        + reports.chemistry_panel(results[0].traces, f"the {results[0].mode} run")
        + reports.looming_panel(probe_data)
    )
    await flyte.report.flush.aio()

    summary = {r.mode: r.metrics for r in results}
    for label, metrics in summary.items():
        log.info(
            "%-7s summed descending %.0f -> %.0f Hz (x%.2f), peak startle %.2f",
            label, metrics.get("baseline_dn_total", 0.0), metrics.get("peak_dn_total", 0.0),
            metrics.get("startle_ratio", 0.0), metrics.get("peak_startle", 0.0),
        )
    summary["time_reversal"] = {
        g: v["time_reversal_corr"] for g, v in probe_data["gains"].items()
    }
    return json.dumps(summary)


@orch_env.task(report=True)
async def drum(
    ticks: int = 200,
    speed: float = 60.0,
    radius: float = 12.0,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """The oldest experiment in fly vision, on a tethered fly, run twice on purpose.

    A tethered fly inside a rotating striped drum turns with the drum. That optomotor
    response is the standard assay for the motion pathway, and the fly's body being
    held still is what makes it clean: the animal cannot actually turn, so its intention
    is readable only in the descending command.

    Five runs, in two groups that ask different questions of the same rig:

        12 bars, clockwise / counter-clockwise / still
            A full drum. Rotating it barely changes how much of each eye is dark, so
            this is a pure MOTION stimulus. The optomotor test is whether the mean
            steering command FLIPS SIGN between the two directions.

        1 bar, clockwise / counter-clockwise
            A single stripe sweeping around. Which eye is dark changes as it goes, so
            this is a POSITION stimulus, and the test is whether the steering command
            tracks where the bar is.

    Running both is the point. This repo's retina reads one number per eye, the fraction
    of ommatidia darker than an empty arena, which is motion-blind by construction: the
    12-bar condition is expected to fail on the encoder alone, before the brain gets a
    say. Measured in an empty drum, total darkening across both eyes holds at 0.15 to
    0.19 whichever way the drum spins, while a single bar sweeping left drives the
    left-right difference to +0.031.

    So a null result here is not evidence that the connectome cannot do optomotor. It is
    evidence about where the bottleneck is, and the `looming_probe` autopsy says the
    same thing from the other end: T4 and T5, the fly's elementary motion detectors,
    do not fire in this model at any synaptic gain tried.
    """
    conditions = [
        ("12 bars, counter-clockwise", 12, +speed),
        ("12 bars, clockwise", 12, -speed),
        ("12 bars, still", 12, 0.0),
        ("1 bar, counter-clockwise", 1, +speed),
        ("1 bar, clockwise", 1, -speed),
    ]
    await flyte.report.replace.aio(
        reports.header("the drum: a tethered fly in a rotating striped arena")
        + reports.note(
            f"{ticks} ticks ({ticks * BODY.BRAIN_DT:.1f} s of fly time) each, drum at "
            f"{speed:.0f} deg/s and {radius:.0f} mm. The fly is held in place, so every "
            f"result is in the command and none of it is in the trajectory."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            fly_once(
                mode="connectome", ticks=ticks, arena="drum", drum_bars=bars,
                drum_speed_deg_s=spin, object_distance=radius, turn_gain=turn_gain,
                base_drive=0.5, spawn_heading_deg=0.0, seed=seed,
            )
            for _, bars, spin in conditions
        ]
    )
    results = []
    for handle, (label, _, _) in zip(files, conditions):
        with open(await handle.download()) as f:
            loaded = _Loaded(json.load(f))
        loaded.mode = label
        results.append(loaded)

    await flyte.report.replace.aio(
        reports.header("the drum: a tethered fly in a rotating striped arena")
        + reports.drum_panel(results)
    )
    await flyte.report.flush.aio()

    summary = {r.mode: r.metrics for r in results}
    for label, metrics in summary.items():
        log.info(
            "%-26s mean imbalance %+.4f (sd %.4f), imbalance vs bar azimuth r=%+.3f",
            label, metrics.get("mean_imbalance", 0.0), metrics.get("imbalance_sd", 0.0),
            metrics.get("azimuth_corr", float("nan")),
        )
    return json.dumps(summary)


@orch_env.task(report=True)
async def poke(
    ticks: int = 225,
    n_pokes: int = 6,
    poke_every: int = 35,
    poke_len: int = 14,
    touch_hz: float = BR.TOUCH_HZ,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """Poke the fly on one antenna and watch it swerve. The biggest reaction in the model.

    Every visual experiment in this repo fights for signal: a pillar darkens 54 of 721
    facets, the descending populations move by less than their own per-tick noise, and
    the loop only works because 150 ms of smoothing is applied to the imbalance. Touch is
    not like that. Measured open loop, 300 ms of one-sided mechanosensory drive:

        poke left  @ 75 Hz -> descending left 4,893  right 2,570   imbalance -0.311
        poke right @ 75 Hz -> descending left 1,403  right 3,703   imbalance +0.450

    The sign flips cleanly with the side and the magnitude is an order of magnitude above
    anything vision produces. This task puts that in a walking fly: it strolls in an empty
    arena with nothing to look at, and every `poke_every` ticks an experimenter deflects
    the bristles on one side of its head, alternating left and right.

    The measurement is the steering command during the poked ticks, against the quiet
    ticks in the same run. Heading is reported too and is the weaker number on purpose: a
    walking fly wanders about 10 degrees per 180 ms on its own, so the behaviour is
    noisier than the command that drives it.

    Measured, 6 pokes at 150 Hz over 225 ticks:

        connectome   poke left -0.171   poke right +0.170   quiet +0.001   3.94 SD apart
        shuffled     poke left -0.113   poke right -0.015   quiet -0.018   0.89 SD apart
    """
    modes = ["connectome", "shuffled"]
    await flyte.report.replace.aio(
        reports.header("the poke: touch one antenna, watch it swerve")
        + reports.note(
            f"{n_pokes} pokes of {poke_len} ticks at {touch_hz:.0f} Hz, alternating "
            f"left and right, in an empty arena with nothing to see. The mechanosensory "
            f"population is 1,363 cells on the left and 1,293 on the right, mostly the "
            f"bristles between the ommatidia plus Johnston's organ in the antenna."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            fly_once(
                mode=mode, ticks=ticks, spawn_heading_deg=0.0,
                # Nothing to look at: the object is parked far outside the arena so the
                # visual channel contributes only its tonic baseline and the run is a
                # clean test of touch.
                object_distance=200.0, object_radius=1.0,
                turn_gain=turn_gain, base_drive=0.5,
                n_pokes=n_pokes, poke_every=poke_every, poke_len=poke_len,
                touch_hz=touch_hz, seed=seed,
            )
            for mode in modes
        ]
    )
    results = []
    for handle, mode in zip(files, modes):
        with open(await handle.download()) as f:
            loaded = _Loaded(json.load(f))
        loaded.mode = mode
        results.append(loaded)

    await flyte.report.replace.aio(
        reports.header("the poke: touch one antenna, watch it swerve")
        + reports.poke_panel(results)
        + reports.chemistry_panel(results[0].traces, "the connectome run")
    )
    await flyte.report.flush.aio()

    summary = {r.mode: r.metrics for r in results}
    for mode, m in summary.items():
        log.info(
            "%-11s imbalance left %+.3f right %+.3f quiet %+.3f, separation %.2f SD",
            mode, m.get("imbalance_left", 0.0), m.get("imbalance_right", 0.0),
            m.get("imbalance_quiet", 0.0), m.get("imbalance_separation_sd", 0.0),
        )
    return json.dumps(summary)


@brain_env.task(report=True)
async def teach(
    trained: str = "ORN_DA1",
    control: str = "ORN_VM4",
    valence: str = "punish",
    learning_rate: float = 0.9,
    odor_hz: float = LEARN.ODOR_HZ,
    clip_ticks: int = 60,
    seed: int = 0,
) -> str:
    """Train the fly. Punish an odour and watch the mushroom body stop responding to it.

    This is the only task in the repo that CHANGES a synapse. The rule is the animal's:
    dopamine arriving while a Kenyon cell is active depresses that cell's synapse onto
    the mushroom body output neuron in the same compartment, so the odour stops driving
    that channel. Two of the three factors are read from the connectome rather than
    chosen: which Kenyon cells the odour activates, and which of the 96 MBONs the driven
    dopaminergic cells actually reach (PPL1 reaches 85 of them, PAM 68).

    It works, and then it fails in an interesting place. The depression drops the trained
    odour's MBON output by about half and does nothing at all on shuffled wiring. But the
    memory is not odour-SPECIFIC, and `learn.specificity` says exactly why: odour identity
    is present at the receptors and gone one synapse later, because a single glomerulus
    drives 543 of the 685 projection neurons in this model. Everything downstream sees the
    same generic "an odour is present" signal, so every odour recruits the same ~1,500
    Kenyon cells and punishing one punishes them all.

    That is the same failure as the dead motion pathway in `swat`, from the same cause:
    nothing in this model does gain control, so any sufficient input ignites a whole
    region. See `learn.py` for the four fixes that were tried and did not work.
    """
    BRAIN.preimport()
    conn = C.load()
    mb = LEARN.MushroomBody.build(conn)

    await flyte.report.replace.aio(
        reports.header("teaching the fly: associative learning in the mushroom body")
        + reports.note("measuring the odour code before training anything...")
    )
    await flyte.report.flush.aio()

    # First the question the whole demo depends on: are odours distinguishable at all?
    autopsy = await asyncio.to_thread(LEARN.specificity, conn, mb, odor_hz, seed=seed)
    await flyte.report.replace.aio(
        reports.header("teaching the fly: associative learning in the mushroom body")
        + reports.specificity_panel(autopsy)
    )
    await flyte.report.flush.aio()

    conditions = {}
    for label, wiring in (("the real wiring", conn), ("shuffled wiring", conn.shuffled(seed=seed))):
        conditions[label] = await asyncio.to_thread(
            LEARN.conditioning, wiring, mb, trained, control, valence, learning_rate,
            True, seed,
        )
        await flyte.report.replace.aio(
            reports.header("teaching the fly: associative learning in the mushroom body")
            + reports.learning_panel(conditions)
            + reports.specificity_panel(autopsy)
        )
        await flyte.report.flush.aio()

    # The clip: the same odour presented to the same brain before and after training,
    # with the mushroom body drawn on its own so the change is visible rather than
    # inferred from a table.
    real = conditions["the real wiring"]

    chem_traces: dict[str, dict[str, list[float]]] = {}

    def make_clip() -> bytes:
        naive = conn
        trained_conn, _ = LEARN.train(
            conn, mb, trained, valence=valence, learning_rate=learning_rate, seed=seed
        )
        before_counts, before_hz = LEARN.present_frames(
            naive, mb, trained, ticks=clip_ticks, odor_hz=odor_hz, seed=seed
        )
        after_counts, after_hz = LEARN.present_frames(
            trained_conn, mb, trained, ticks=clip_ticks, odor_hz=odor_hz, seed=seed
        )
        # The bottom row draws the 96 OUTPUT neurons only, and that choice is the whole
        # point of the picture. A panel containing the Kenyon cells is dominated by them
        # and barely changes (measured: 0.98 of its former brightness), because training
        # does not touch the Kenyon cells at all. It weakens their synapses onto the
        # outputs. So the input stays lit and the output goes dark, and showing both is
        # the clearest statement of what was learned.
        canvases = {
            "brain_before": VIZ.BrainCanvas(conn, width=440),
            "brain_after": VIZ.BrainCanvas(conn, width=440),
            "mb_before": VIZ.BrainCanvas(conn, width=440, subset=mb.mbon, spread=6),
            "mb_after": VIZ.BrainCanvas(conn, width=440, subset=mb.mbon, spread=6),
        }
        # One shared colour scale per row, set from the BEFORE run, so the after clip
        # going dark is a real difference and not an autoscale artefact.
        for key, history in (
            ("brain_before", before_counts), ("brain_after", before_counts),
            ("mb_before", before_counts), ("mb_after", before_counts),
        ):
            canvases[key].autoscale(history)
            canvases[key].reset()

        # Chemistry, sampled with each run's OWN wiring. That matters here: training
        # depresses Kenyon cell synapses onto the output neurons, and Kenyon cells are
        # cholinergic, so the trained brain literally has a smaller acetylcholine budget
        # to release. The after trace is lower for two reasons at once, and this is the
        # only task in the repo where that is true.
        for label, wiring, counts in (
            ("before training", naive, before_counts),
            ("after training", trained_conn, after_counts),
        ):
            chem = CHEM.Neurochemistry(wiring)
            collected: dict[str, list[float]] = {k: [] for k in chem.trace_keys()}
            for frame in counts:
                for key, value in chem.sample(frame).items():
                    collected[key].append(value)
            chem_traces[label] = collected

        peak = max(max(before_hz, default=1.0), max(after_hz, default=1.0))
        frames = []
        for i in range(min(len(before_counts), len(after_counts))):
            canvases["brain_before"].update(before_counts[i])
            canvases["brain_after"].update(after_counts[i])
            canvases["mb_before"].update(before_counts[i])
            canvases["mb_after"].update(after_counts[i])
            frames.append(
                VID.compose_learning(
                    canvases["brain_before"].frame(), canvases["brain_after"].frame(),
                    canvases["mb_before"].frame(), canvases["mb_after"].frame(),
                    tick=i, t=(i + 1) * BODY.BRAIN_DT,
                    hz_before=before_hz[i], hz_after=after_hz[i], peak_hz=peak,
                    odor=trained,
                )
            )
        return VID.encode(frames)

    mp4 = await asyncio.to_thread(make_clip)
    await flyte.report.replace.aio(
        reports.header("teaching the fly: associative learning in the mushroom body")
        + reports.learning_video_panel(mp4, trained, real)
        + reports.learning_chemistry_panel(chem_traces)
        + reports.learning_panel(conditions)
        + reports.specificity_panel(autopsy)
    )
    await flyte.report.flush.aio()

    for stage in autopsy:
        log.info(
            "%-28s cells %6d  active %-22s  odours firing %d/%d  overlap %.3f",
            stage["stage"], stage["cells"],
            "/".join(str(v) for v in stage["active"].values()),
            stage["odors_firing"], len(stage["active"]), stage["jaccard"],
        )
    for label, result in conditions.items():
        trained, control = result["trained_odor"], result["control_odor"]
        log.info(
            "%-16s %s %.0f -> %.0f Hz (%+.1f%%) | %s %.0f -> %.0f Hz (%+.1f%%) | "
            "specificity %+.1f pts | %d edges depressed",
            label,
            trained, result["before"][trained]["mbon_hz"],
            result["after"][trained]["mbon_hz"], result["trained_drop_pct"],
            control, result["before"][control]["mbon_hz"],
            result["after"][control]["mbon_hz"], result["control_drop_pct"],
            result["specificity_pct"], result["training"]["edges_depressed"],
        )
    return json.dumps({"specificity": autopsy, "conditions": conditions})


@brain_env.task(report=True, retries=1)
async def ride_once(
    mode: str = "connectome",
    ticks: int = 450,
    object_distance: float = 150.0,
    object_radius: float = 25.0,
    spawn_heading_deg: float = 40.0,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> File:
    """One robot-riding run. Returns the clip and the numbers."""
    BRAIN.preimport()
    conn = C.load()

    def head() -> str:
        return reports.header(
            f"a fly riding a robot: <b>{mode}</b> - "
            f"{reports.MODE_BLURB.get(mode, '')}"
        )

    await flyte.report.replace.aio(head() + reports.note("building the robot..."))
    await flyte.report.flush.aio()

    progress = {"html": ""}

    def on_tick(done: int, total: int, traces: dict) -> None:
        progress["html"] = reports.note(
            f"tick {done}/{total} - {traces['distance'][-1]:.0f} mm from the pillar, "
            f"bearing {traces['bearing'][-1]:+.0f} deg"
        )

    result = await asyncio.to_thread(
        RUN.ride_loop, conn, mode=mode, ticks=ticks, object_distance=object_distance,
        object_radius=object_radius, spawn_heading_deg=spawn_heading_deg,
        turn_gain=turn_gain, seed=seed, on_tick=on_tick,
    )

    mp4 = VID.encode(result.frames)
    await flyte.report.replace.aio(
        head()
        + reports.ride_panel(result, mp4)
        + reports.chemistry_panel(result.traces, f"the {mode} ride")
    )
    await flyte.report.flush.aio()
    log.info(result.summary())

    out = pathlib.Path(tempfile.mkdtemp()) / f"ride-{mode}.mp4"
    out.write_bytes(mp4)
    meta = out.with_suffix(".json")
    meta.write_text(
        json.dumps({
            "mode": result.mode, "ticks": result.ticks, "duration_s": result.duration_s,
            "wall_s": result.wall_s, "total_spikes": result.total_spikes,
            "metrics": result.metrics, "trajectory": result.trajectory,
            "traces": result.traces,
        })
    )
    return await File.from_local(str(meta))


@orch_env.task(report=True)
async def ride(
    modes: list[str] | None = None,
    ticks: int = 450,
    object_distance: float = 150.0,
    spawn_heading_deg: float = 40.0,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """Put the fly on a robot and let it drive. Does the connectome steer a body it never had?

    This is the sharpest test of the claim the whole repo rests on. `bridge.py` argues
    that what a connectome sends to a body is a DESCENDING COMMAND, roughly "turn left",
    and that a central pattern generator downstream works out the legs. If that is really
    true then the same 1,300 neurons should drive a body with the wrong number of legs,
    the wrong mass and a completely different gait, with nothing about the brain changed.

    So the fly is attached to the back of a four-legged robot about twenty-five times its
    length, its compound eyes look out from up there, and `rider.ride_command` converts
    its per-side drives into the robot's speed and yaw. Four lines. Nothing else differs
    from the walking demo: same connectome, same retina, same bridge, same calibration.

    The robot is written from scratch in MJCF because the pod has no network at run time
    and everything must be in flygym's millimetres. Two things in it were measured the
    hard way and are documented in `rider.py`: the servo gains (three orders of magnitude
    larger than they would be in SI, or the robot melts), and the steering mechanism
    (differential stride turned +53.6 deg one way and -7.0 deg the other, so it was
    replaced with yaw joints that are symmetric by construction).
    """
    modes = modes or ["connectome", "shuffled", "nobrain"]
    await flyte.report.replace.aio(
        reports.header("a fly riding a robot")
        + reports.note(
            f"A pillar {object_distance:.0f} mm away, {spawn_heading_deg:.0f} degrees off "
            f"the nose. A robot that never turns misses it by "
            f"{object_distance * abs(np.sin(np.deg2rad(spawn_heading_deg))):.0f} mm, so "
            f"arriving means the fly steered it there."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            ride_once(
                mode=mode, ticks=ticks, object_distance=object_distance,
                spawn_heading_deg=spawn_heading_deg, turn_gain=turn_gain, seed=seed,
            )
            for mode in modes
        ]
    )
    results = []
    for handle in files:
        with open(await handle.download()) as f:
            results.append(_Loaded(json.load(f)))

    await flyte.report.replace.aio(
        reports.header("a fly riding a robot")
        + reports.ride_comparison_panel(results, (object_distance, 0.0), 25.0)
    )
    await flyte.report.flush.aio()

    summary = {r.mode: r.metrics for r in results}
    for mode, m in summary.items():
        log.info(
            "%-11s closest %.0f mm, approach %+.0f mm, |bearing| %.0f deg, reached %.0f",
            mode, m["closest_distance"], m["approach"], m["mean_abs_bearing_late"],
            m["reached"],
        )
    return json.dumps(summary)


@orch_env.task(report=True)
async def lesion(
    targets: list[str] | None = None,
    ticks: int = 260,
    spawn_heading_deg: float = 60.0,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """Cut named cell types out of the brain and see whether the fly still gets there.

    This is the experiment a connectome makes possible and a trained network does not.
    Every neuron here has a NAME from the literature, so "remove LC4 and LPLC2" is a
    sentence about specific, studied cells rather than about weights. `Connectome.silence`
    zeroes their outgoing edges: they still integrate and still spike, they just stop
    being heard by anything downstream.

    The default targets are the lobula columnar types, the neurons that carry visual
    features out of the optic lobe toward the descending neurons, plus the textbook
    steering pair as a contrast. If the fly survives losing DNa02 but not LC neurons,
    that is a statement about which part of the pathway this behaviour actually runs on.
    """
    # These targets are chosen from MEASUREMENT, not from the literature, and the first
    # version of this task got that wrong. Cutting LC4, LPLC2 or DNa02 changed the run
    # by exactly nothing -- bit-identical spike counts -- because at the published
    # w_syn those cells never fire at all. A lesion of a silent neuron is a no-op.
    #
    # What does fire, driving one eye at 150 Hz for 300 ms: the photoreceptors
    # themselves (899k of 954k spikes), the lamina monopolar cells L1/L2/L3 immediately
    # downstream, Lai, Tm1 in the medulla, and then 18 descending neurons. That is the
    # whole active pathway, so that is what there is to cut.
    targets = targets or ["L1", "L2", "Lai", "DNp28"]
    await flyte.report.replace.aio(
        reports.header(f"lesion experiment: intact vs {', '.join(targets)}")
        + reports.note(
            "Each lesion zeroes the outgoing synapses of every neuron of that type, "
            "both sides. The intact run is the same wiring with nothing cut."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        fly_once(mode="connectome", ticks=ticks, spawn_heading_deg=spawn_heading_deg,
                 turn_gain=turn_gain, seed=seed),
        *[
            fly_once(mode="connectome", ticks=ticks, spawn_heading_deg=spawn_heading_deg,
                     turn_gain=turn_gain, silence=target, seed=seed)
            for target in targets
        ],
    )
    results = []
    for handle in files:
        with open(await handle.download()) as f:
            results.append(_Loaded(json.load(f)))
    results[0].mode = "intact"

    await flyte.report.replace.aio(
        reports.header(f"lesion experiment: intact vs {', '.join(targets)}")
        + reports.comparison_panel(results, (12.0, 0.0), 3.0)
        + reports.note(
            "A lesion that changes nothing is as informative as one that breaks the "
            "behaviour: it says the pathway does not run through those cells."
        )
    )
    await flyte.report.flush.aio()
    summary = {r.mode: r.metrics for r in results}
    for mode, metrics in summary.items():
        log.info("%-18s reached=%.0f closest %.1f mm", mode, metrics["reached"],
                 metrics["closest_distance"])
    return json.dumps(summary)


@orch_env.task(report=True)
async def sweep(
    headings: list[float] | None = None,
    ticks: int = 300,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """How far off the nose can the pillar start and still be found?

    A tuning curve for the whole animal. The straight-line miss distance grows as
    `distance * sin(heading)`, so this is also a sweep of how much steering the task
    demands: at 20 degrees a fly that never turns passes within 4.1 mm and nearly
    succeeds by accident; at 75 degrees it misses by 11.6 mm.
    """
    headings = headings or [20.0, 40.0, 60.0, 75.0]
    await flyte.report.replace.aio(
        reports.header(f"bearing sweep: pillar starting {headings} degrees off the nose")
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            fly_once(mode="connectome", ticks=ticks, spawn_heading_deg=h,
                     turn_gain=turn_gain, seed=seed)
            for h in headings
        ]
    )
    results = []
    for handle, heading in zip(files, headings):
        with open(await handle.download()) as f:
            loaded = _Loaded(json.load(f))
        loaded.mode = f"{heading:.0f} deg"
        results.append(loaded)

    await flyte.report.replace.aio(
        reports.header(f"bearing sweep: {headings} degrees off the nose")
        + reports.comparison_panel(results, (12.0, 0.0), 3.0)
    )
    await flyte.report.flush.aio()
    return json.dumps({r.mode: r.metrics for r in results})


@orch_env.task(report=True)
async def gain(
    w_syn_values: list[float] | None = None,
    ticks: int = 260,
    spawn_heading_deg: float = 60.0,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """Sweep the one free parameter in the brain and watch how much of it wakes up.

    `w_syn` is millivolts of postsynaptic depolarisation per synapse, and it is the
    only number in `brain.PARAMS` that was fitted rather than measured. Shiu et al. set
    it to 0.275 mV to reproduce specific stimulation experiments. Measured here,
    driving one eye at 150 Hz for 300 ms:

        w_syn (mV)                    0.275     0.55      1.1      2.2
        central-brain neurons firing     13   11,813    8,007   19,830
        descending neurons firing        18      391      517      766

    At the published value the signal dies about two synapses past the retina and the
    entire steering behaviour rides on 18 descending cells. Doubling it wakes up a
    third of the central brain. This task asks the question that follows: does a fly
    with more of its brain switched on behave BETTER, or does it just seize?
    """
    w_syn_values = w_syn_values or [0.275, 0.55, 1.1]
    await flyte.report.replace.aio(
        reports.header(f"synaptic gain sweep: w_syn = {w_syn_values} mV")
        + reports.note(
            "One parameter, the volts a single synapse delivers. Everything else, "
            "including which neuron connects to which and how many synapses they "
            "share, is measured from the animal and identical across these runs."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            fly_once(mode="connectome", ticks=ticks, spawn_heading_deg=spawn_heading_deg,
                     turn_gain=turn_gain, w_syn_mv=w, seed=seed)
            for w in w_syn_values
        ]
    )
    results = []
    for handle, w in zip(files, w_syn_values):
        with open(await handle.download()) as f:
            loaded = _Loaded(json.load(f))
        loaded.mode = f"w_syn {w} mV"
        results.append(loaded)

    await flyte.report.replace.aio(
        reports.header(f"synaptic gain sweep: w_syn = {w_syn_values} mV")
        + reports.comparison_panel(results, (12.0, 0.0), 3.0)
    )
    await flyte.report.flush.aio()
    return json.dumps({r.mode: r.metrics for r in results})


@orch_env.task(report=True)
async def feed(
    ticks: int = 340,
    spawn_heading_deg: float = 45.0,
    object_distance: float = 12.0,
    turn_gain: float = 1.8,
    seed: int = 0,
) -> str:
    """Walk to the food, taste it, and decide whether to eat. Sugar against bitter.

    Two modalities chained, and the second one is the pathway Shiu et al.'s paper was
    built on. Vision steers the fly to a drop it cannot taste from a distance. When its
    tarsi arrive (real flies taste with their feet), the gustatory receptor neurons for
    whatever is actually on the drop start firing, and the feeding circuit either runs
    or does not.

    Measured before any of this was wired up, 500 ms of drive at 150 Hz:

        sugar GRNs  (129 cells) -> 30 of 110 feeding motor neurons fire, 1,610 Hz,
                                   including MN10, a named proboscis motor neuron
        bitter GRNs  (65 cells) ->  0 of 110 fire, 0 Hz

    Nothing in this repo encodes "sugar is good". Both drops are the same shape and the
    same colour, the approach is identical, and the only thing that differs is which
    129 or 65 sensory neurons the contact drives. The proboscis extends for one and not
    the other because of how the animal is wired.

    A third run with `food="none"` is the control for the stopping itself: a fly that
    arrives at a tasteless drop should walk straight over it.
    """
    conditions = [("sugar", "sugar"), ("bitter", "bitter"), ("tasteless", "none")]
    await flyte.report.replace.aio(
        reports.header("feeding: same drop, same approach, different taste")
        + reports.note(
            "The fly cannot see the difference. Watch the proboscis bar in the command "
            "panel at the moment of arrival."
        )
    )
    await flyte.report.flush.aio()

    files = await asyncio.gather(
        *[
            fly_once(
                mode="connectome", ticks=ticks, spawn_heading_deg=spawn_heading_deg,
                object_distance=object_distance, object_radius=4.0,
                turn_gain=turn_gain, food=food, object_type="sphere", seed=seed,
            )
            for _, food in conditions
        ]
    )
    results = []
    for handle, (label, _) in zip(files, conditions):
        with open(await handle.download()) as f:
            loaded = _Loaded(json.load(f))
        loaded.mode = label
        results.append(loaded)

    await flyte.report.replace.aio(
        reports.header("feeding: same drop, same approach, different taste")
        + reports.feeding_panel(results)
        + reports.comparison_panel(results, (object_distance, 0.0), 4.0)
    )
    await flyte.report.flush.aio()
    summary = {r.mode: r.metrics for r in results}
    for label, metrics in summary.items():
        log.info("%-10s peak proboscis drive %.2f, %d ticks feeding, closest %.1f mm",
                 label, metrics.get("peak_feeding", 0.0),
                 int(metrics.get("ticks_feeding", 0)), metrics["closest_distance"])
    return json.dumps(summary)


class _Loaded:
    """A run's numbers, read back from its sidecar json.

    Quacks like `run.RunResult` for the report helpers, minus the frames: the
    orchestrator never needs 200 decoded video frames in memory to draw a trajectory.
    """

    def __init__(self, blob: dict) -> None:
        self.mode = blob["mode"]
        self.ticks = blob["ticks"]
        self.duration_s = blob["duration_s"]
        self.wall_s = blob["wall_s"]
        self.total_spikes = blob["total_spikes"]
        self.metrics = blob["metrics"]
        self.trajectory = [tuple(p) for p in blob["trajectory"]]
        self.traces = blob["traces"]
        self.food = blob.get("food", "none")
        self.motion = blob.get("motion", "none")
        self.arena = blob.get("arena", "ground")
        self.w_syn_mv = blob.get("w_syn_mv", 0.275)


if __name__ == "__main__":
    flyte.init_from_config()
