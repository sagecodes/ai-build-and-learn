"""The closed loop, and the three controls that keep it honest.

    for each 15 ms tick:
        eyes   = body.look()                  render 2 x 721 ommatidia
        rates  = bridge.apply_vision(...)     darkening -> Hz on real photoreceptors
        tick   = brain.run(0.015)             138,639 LIF neurons, one step
        signal = bridge.descending_signal()   1,291 descending neurons -> 2 numbers
        body.step(signal)                     150 physics steps, 1 video frame

Four modes run through that same loop, and the difference between them is the
experiment:

    connectome  the real FlyWire wiring, eyes connected. The claim.
    shuffled    every edge's postsynaptic partner permuted. Same neurons, same
                out-degrees, same weight distribution, wiring destroyed. If the fly
                still finds the pillar, the behaviour was never in the connectome.
    blind       the real wiring, running, but the photoreceptors are held at 0 Hz.
                Separates "the brain is steering" from "the body drifts that way".
    nobrain     no brain at all, constant forward drive. The shape of not steering.

A run is scored on numbers the brain cannot reach, read straight out of the physics:
distance to the pillar, closest approach, and absolute bearing error over the last
third of the run.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

import body as BODY
import bridge as BR
import chemistry as CHEM
import brainviz as VIZ
import connectome as C
import video as VID
from connectome import Connectome

log = logging.getLogger(__name__)

MODES = ("connectome", "shuffled", "blind", "nobrain")


@dataclass
class RunResult:
    """Everything one closed-loop run produced."""

    mode: str
    ticks: int
    duration_s: float
    wall_s: float
    trajectory: list[tuple[float, float, float]]
    traces: dict[str, list[float]]
    metrics: dict[str, float]
    calibration: BR.Calibration | None
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    total_spikes: int = 0
    # Spikes per neuron over the whole run, for the "which cells were busiest" panel.
    total_counts: np.ndarray | None = field(default=None, repr=False)
    silenced: str = ""
    silenced_n: int = 0
    w_syn_mv: float = 0.275

    def summary(self) -> str:
        m = self.metrics
        lesion = f" [-{self.silenced} x{self.silenced_n}]" if self.silenced else ""
        return (
            f"{self.mode:11s}{lesion} start {m['start_distance']:5.1f} mm -> "
            f"closest {m['closest_distance']:5.1f} mm  "
            f"(final {m['final_distance']:5.1f})  "
            f"|bearing| {m['mean_abs_bearing_late']:5.1f} deg  "
            f"path {m['path_length']:5.1f} mm  "
            f"{'REACHED in ' + format(m['time_to_reach_s'], '.2f') + 's' if m['reached'] else 'did not reach'}  "
            f"{self.total_spikes:,} spikes in {self.wall_s:.0f}s"
        )


def closed_loop(
    conn: Connectome,
    mode: str = "connectome",
    ticks: int = 200,
    scene: BODY.Scene | None = None,
    calibration: BR.Calibration | None = None,
    behaviour: str = "attract",
    turn_gain: float = 0.8,
    base_drive: float = 1.0,
    eye_gain: float = BR.DEFAULT_EYE_GAIN,
    seed: int = 0,
    brain_width: int = 480,
    stop_on_reach: bool = True,
    reach_margin: float = 2.0,
    silence: str = "",
    w_syn_mv: float = 0.0,
    startle_gain: float = 0.0,
    pokes: list | None = None,
    touch_hz: float = BR.TOUCH_HZ,
    on_tick=None,
) -> RunResult:
    """Run one fly for up to `ticks` brain ticks (200 = 3 s of fly time).

    `stop_on_reach` ends the run when the fly gets within `reach_margin` mm of the
    pillar's surface, and it is on by default because leaving it off measures the wrong
    thing. A fly walking at 16 mm/s covers 24 mm in 1.5 s, so after it arrives it keeps
    going and walks out the other side: in a measured 180-tick run the fly closed from
    9.5 mm to 3.1 mm (touching a 3 mm post) and then ended 12 mm away facing backwards,
    which scores as a failure on final distance and a success on every honest metric.
    Real object-fixation experiments end at arrival. So does this.

    `on_tick(i, ticks, traces)` is called every 20 ticks, for live reporting.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    scene = scene or BODY.Scene()
    # A poke schedule is a list of (start_tick, duration_ticks, side). Expanded here
    # into one entry per tick so the loop stays a lookup rather than a search.
    poke_at: dict[int, str] = {}
    for start, length, side in (pokes or []):
        for k in range(int(length)):
            poke_at[int(start) + k] = side
    wiring = conn.shuffled(seed=seed) if mode == "shuffled" else conn
    lesion_size = 0
    if silence:
        cut = wiring.resolve(silence)
        lesion_size = len(cut)
        wiring = wiring.silence(cut)
        log.info("silenced %s: %d neurons cut out of the conversation", silence, lesion_size)
    started = time.time()

    world = BODY.FlyWorld(scene=scene, seed=seed)
    canvas = VIZ.BrainCanvas(conn, width=brain_width)

    brain = None
    link = None
    if mode != "nobrain":
        if calibration is None:
            import brain as _B
            _p = dict(_B.PARAMS)
            if w_syn_mv > 0:
                _p["w_syn"] = w_syn_mv * 1e-3
            calibration = BR.calibrate(wiring, seed=seed, params=_p)
            log.info(
                "calibration: sign %+.0f, separation %.3f",
                calibration.turn_sign, calibration.separation,
            )
        link = BR.Bridge(
            conn=wiring,
            calibration=calibration,
            base_drive=base_drive,
            turn_gain=turn_gain,
            eye_gain=eye_gain,
            behaviour=behaviour,
            startle_gain=startle_gain,
        )
        import brain as BRAIN

        params = dict(BRAIN.PARAMS)
        if w_syn_mv > 0:
            # The model's ONE free parameter, and it decides what kind of brain this is.
            # Measured, driving one eye at 150 Hz for 300 ms:
            #
            #   w_syn (mV)   0.275     0.55      1.1      2.2
            #   central-brain neurons firing     13   11,813    8,007   19,830
            #   descending neurons firing        18      391      517      766
            #
            # At the published 0.275 the signal dies about two synapses past the
            # retina and the whole behaviour rides on 18 descending cells. Doubling it
            # wakes up a third of the central brain. Neither value is "correct": the
            # paper fitted 0.275 to reproduce specific stimulation experiments, and
            # this is the knob that says how much of the connectome is participating.
            params["w_syn"] = w_syn_mv * 1e-3
        brain = BRAIN.Brain(conn=wiring, stim_groups=link.stim_groups(),
                            params=params, seed=seed)

    # What the brain is RELEASING, not just which cells fired. Costs one dot product per
    # tick against a per-neuron synapse budget precomputed here. Built before the traces
    # dict because it names the keys that dict has to carry.
    chem = CHEM.Neurochemistry(conn)
    traces: dict[str, list[float]] = {
        k: [] for k in (
            "eye_left", "eye_right", "rate_left", "rate_right",
            "dn_left", "dn_right", "drive_left", "drive_right",
            "imbalance", "distance", "bearing", "motor", "feeding", "taste_hz",
            # The swat's channels: the descending SUM (fixation reads the difference),
            # how far above its own baseline that sum is, and the physics ground truth
            # for when the object actually arrives.
            "dn_total", "startle", "time_to_contact",
            # The poke experiment: which antenna is being touched (-1 left, +1 right,
            # 0 neither) and the fly's absolute heading, which is what a touch changes.
            "poke", "heading",
        )
    }
    for key in chem.trace_keys():
        traces[key] = []
    for label in C.ESCAPE_TYPES:
        traces[f"escape:{label}"] = []
    for cell_type in C.NAMED_DESCENDING:
        traces[f"{cell_type}_left"] = []
        traces[f"{cell_type}_right"] = []

    object_track: list[tuple[float, float]] = []
    counts_history: list[np.ndarray] = []
    eye_history: list[tuple[np.ndarray, np.ndarray]] = []
    traj_history: list[list[tuple[float, float, float]]] = []
    statuses: list[VID.Status] = []
    total_counts = np.zeros(conn.n, np.float64)
    total_spikes = 0

    for i in range(ticks):
        eyes = world.look()

        if brain is None:
            signal = BR.constant_signal(base_drive)
            readout = {"dn_left": 0.0, "dn_right": 0.0, "motor": 0.0}
            rates = (0.0, 0.0)
            counts = np.zeros(conn.n, np.float32)
            imbalance = 0.0
            feeding = 0.0
            taste_hz = 0.0
            poke_side = poke_at.get(i, "none")
            startle = 0.0
            escape = {f"escape:{label}": 0.0 for label in C.ESCAPE_TYPES}
        else:
            rates = link.apply_vision(brain, eyes, blind=(mode == "blind"))
            taste_hz = link.apply_taste(brain, scene.food, world.tasting)
            poke_side = poke_at.get(i, "none")
            link.apply_touch(brain, poke_side, touch_hz)
            tick = brain.run(BODY.BRAIN_DT)
            counts = tick.counts
            readout = link.read_descending(tick)
            escape = link.read_escape(tick)
            startle = link.update_startle(readout)
            signal = link.descending_signal(readout)
            imbalance = link.imbalance
            feeding = link.feeding(readout)
            total_spikes += int(counts.sum())

        counts_history.append(counts)
        total_counts += counts
        # Grab the facet view BEFORE stepping, so it matches the eyes this decision was
        # actually made from rather than the world one tick later.
        eye_history.append(world.eye_views())
        world.step(signal)
        traj_history.append(list(world.trajectory))

        traces["eye_left"].append(float(eyes[0]))
        traces["eye_right"].append(float(eyes[1]))
        traces["rate_left"].append(float(rates[0]))
        traces["rate_right"].append(float(rates[1]))
        traces["drive_left"].append(float(signal[0]))
        traces["drive_right"].append(float(signal[1]))
        traces["imbalance"].append(float(imbalance))
        traces["feeding"].append(float(feeding))
        traces["taste_hz"].append(float(taste_hz))
        traces["distance"].append(world.distance_to_object)
        traces["bearing"].append(world.bearing_deg)
        traces["poke"].append(
            -1.0 if poke_side == "left" else (1.0 if poke_side == "right" else 0.0)
        )
        traces["heading"].append(world.heading_deg)
        for key, value in chem.sample(counts).items():
            traces[key].append(value)
        traces["dn_total"].append(float(readout["dn_left"] + readout["dn_right"]))
        traces["startle"].append(float(startle))
        traces["time_to_contact"].append(float(world.time_to_contact))
        for key, value in escape.items():
            traces[key].append(float(value))
        object_track.append(world.object_xy)
        for key, value in readout.items():
            traces.setdefault(key, []).append(float(value))

        statuses.append(
            VID.Status(
                tick=i,
                t=(i + 1) * BODY.BRAIN_DT,
                drive=(float(signal[0]), float(signal[1])),
                dn=(readout["dn_left"], readout["dn_right"]),
                imbalance=float(imbalance),
                feeding=float(feeding),
                food=scene.food,
                distance=world.distance_to_object,
                bearing=world.bearing_deg,
                spikes=total_spikes,
                startle=float(startle),
                time_to_contact=world.time_to_contact,
            )
        )
        if on_tick is not None and (i % 20 == 0 or i == ticks - 1):
            on_tick(i + 1, ticks, traces)

        if (
            stop_on_reach
            and scene.motion == "none"
            and scene.food == "none"
            and world.distance_to_object <= scene.object_radius + reach_margin
        ):
            log.info(
                "reached the pillar at tick %d (%.2f s of fly time), %.1f mm from centre",
                i + 1, (i + 1) * BODY.BRAIN_DT, world.distance_to_object,
            )
            break

    # Compose only after the run, so the colour scale can be set from the activity the
    # run actually produced instead of a guess made before it started.
    canvas.autoscale(counts_history)
    canvas.reset()
    body_frames = world.frames
    frames = []
    for i, counts in enumerate(counts_history):
        canvas.update(counts)
        if i < len(body_frames):
            frames.append(
                VID.compose(
                    body_frames[i],
                    canvas.frame(),
                    eye_history[i],
                    statuses[i],
                    traj_history[i],
                    object_track[i] if i < len(object_track) else scene.object_xy,
                    scene.object_radius,
                )
            )

    result = RunResult(
        mode=mode,
        ticks=len(counts_history),
        duration_s=len(counts_history) * BODY.BRAIN_DT,
        wall_s=time.time() - started,
        trajectory=list(world.trajectory),
        traces=traces,
        metrics=_score(traces, world, scene, ticks, reach_margin),
        silenced=silence,
        silenced_n=lesion_size,
        w_syn_mv=w_syn_mv or 0.275,
        calibration=calibration,
        frames=frames,
        total_spikes=total_spikes,
        total_counts=total_counts,
    )
    world.close()
    log.info(result.summary())
    return result


def _score(
    traces: dict[str, list[float]],
    world,
    scene: BODY.Scene,
    ticks_allowed: int,
    reach_margin: float,
) -> dict[str, float]:
    distance = np.asarray(traces["distance"], float)
    bearing = np.abs(np.asarray(traces["bearing"], float))
    path = np.asarray([(x, y) for x, y, _ in world.trajectory], float)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.zeros(1)
    late = max(1, len(bearing) // 3)
    return {
        "start_distance": float(distance[0]),
        "final_distance": float(distance[-1]),
        "closest_distance": float(distance.min()),
        "approach": float(distance[0] - distance.min()),
        "mean_abs_bearing_late": float(bearing[-late:].mean()),
        "path_length": float(steps.sum()),
        # Straightness: 1.0 is a beeline, 0 is a fly that walked in a circle. Useful
        # for telling "steered toward it" apart from "wandered into it".
        "straightness": float(
            np.linalg.norm(path[-1] - path[0]) / max(steps.sum(), 1e-6)
        ),
        "reached": float(distance.min() <= scene.object_radius + reach_margin),
        # Seconds of fly time to arrival, or the full budget if it never arrived. This
        # is the headline number: it separates "walked there" from "wandered there"
        # far better than distance does, because a random walker gets to the pillar
        # eventually and takes forever doing it.
        "time_to_reach_s": float(
            len(distance) * BODY.BRAIN_DT
            if distance.min() <= scene.object_radius + reach_margin
            else ticks_allowed * BODY.BRAIN_DT
        ),
        "ticks_used": float(len(distance)),
        # Feeding: peak proboscis drive, and how much of the run was spent halted on
        # the food. Zero for a fly that never arrived and for one that arrived on
        # something bitter.
        "peak_feeding": float(max(traces.get("feeding", [0.0]))),
        "ticks_feeding": float(sum(1 for f in traces.get("feeding", []) if f > 0.15)),
        **_swat_score(traces, scene),
        **_drum_score(traces, scene),
        **_poke_score(traces),
    }


def _poke_score(traces: dict[str, list[float]]) -> dict[str, float]:
    """Did poking one antenna turn the fly, and does the direction flip with the side?

    A poke is scored on the fly's ACTUAL HEADING, read from the physics, over the ticks
    from the start of the poke to a fixed window after it. Heading is unwrapped before
    differencing, because a fly crossing the +/-180 boundary would otherwise register a
    360 degree turn.

    The result is the pair `turn_after_left_poke` and `turn_after_right_poke`. A touch
    response means they have OPPOSITE signs and both exceed the turning the fly does
    between pokes on its own, which is `turn_unpoked` and is the noise floor.
    """
    poke = np.asarray(traces.get("poke", []), float)
    heading = np.asarray(traces.get("heading", []), float)
    if len(poke) == 0 or len(heading) != len(poke) or not np.any(poke != 0):
        return {}
    unwrapped = np.degrees(np.unwrap(np.radians(heading)))
    window = 12                                   # ticks, 180 ms: a turn takes time

    def turns(side: float) -> list[float]:
        out = []
        starts = [
            i for i in range(len(poke))
            if poke[i] == side and (i == 0 or poke[i - 1] != side)
        ]
        for i in starts:
            end = min(i + window, len(unwrapped) - 1)
            if end > i:
                out.append(float(unwrapped[end] - unwrapped[i]))
        return out

    # The steering command during each kind of tick. This is the direct measurement and
    # it is much cleaner than heading: a walking fly wanders about 10 degrees per 180 ms
    # on its own, so a heading change of that size says nothing, while the descending
    # imbalance is read every tick and has no inertia.
    imbalance = np.asarray(traces.get("imbalance", []), float)
    during = {}
    if len(imbalance) == len(poke):
        for label, mask in (
            ("left", poke == -1.0), ("right", poke == 1.0), ("quiet", poke == 0.0)
        ):
            during[label] = (
                float(imbalance[mask].mean()) if mask.any() else float("nan"),
                float(imbalance[mask].std()) if mask.any() else float("nan"),
            )

    left, right = turns(-1.0), turns(+1.0)
    quiet = [
        float(unwrapped[i + window] - unwrapped[i])
        for i in range(0, len(unwrapped) - window)
        if not np.any(poke[i:i + window] != 0)
    ]
    out = {
        "turn_after_left_poke": float(np.mean(left)) if left else float("nan"),
        "turn_after_right_poke": float(np.mean(right)) if right else float("nan"),
        "turn_unpoked": float(np.mean(np.abs(quiet))) if quiet else float("nan"),
        "n_left_pokes": float(len(left)),
        "n_right_pokes": float(len(right)),
    }
    if left and right:
        out["poke_separation"] = abs(out["turn_after_left_poke"] - out["turn_after_right_poke"])
        out["poke_flips_sign"] = float(
            out["turn_after_left_poke"] * out["turn_after_right_poke"] < 0
        )
    for label, (mean, sd) in during.items():
        out[f"imbalance_{label}"] = mean
        out[f"imbalance_{label}_sd"] = sd
    if "left" in during and "right" in during:
        gap = abs(during["left"][0] - during["right"][0])
        noise = max(during["left"][1], during["right"][1], 1e-9)
        out["imbalance_separation"] = gap
        # The headline: how far apart the two poke directions drive the steering
        # command, measured in units of the command's own within-condition scatter.
        out["imbalance_separation_sd"] = gap / noise
    return out


def _drum_score(traces: dict[str, list[float]], scene: BODY.Scene) -> dict[str, float]:
    """What the tethered drum measured. Empty unless the fly is in one.

    A tethered fly cannot turn, so the whole result lives in the descending command.
    Two numbers:

    `mean_imbalance` is the average steering signal over the second half of the run,
    after the smoothing has settled. THE OPTOMOTOR TEST IS ITS SIGN: a fly that turns
    with a rotating drum must produce opposite signs for a drum spun one way and the
    other. If clockwise and counter-clockwise give the same sign, or both give zero,
    there is no optomotor response.

    `azimuth_corr` is the correlation between that steering signal and where the nearest
    bar actually is, which is the position-tracking test. It is the one the single-bar
    condition is designed to pass.
    """
    if scene.arena != "drum":
        return {}
    imbalance = np.asarray(traces.get("imbalance", []), float)
    azimuth = np.asarray(traces.get("bearing", []), float)
    if len(imbalance) < 4:
        return {"mean_imbalance": 0.0, "azimuth_corr": float("nan")}
    half = len(imbalance) // 2
    late_i, late_a = imbalance[half:], azimuth[half:]
    corr = float("nan")
    if late_i.std() > 1e-9 and late_a.std() > 1e-9:
        corr = float(np.corrcoef(late_i, late_a)[0, 1])
    drive = np.asarray(traces.get("drive_left", []), float) - np.asarray(
        traces.get("drive_right", []), float
    )
    return {
        "mean_imbalance": float(late_i.mean()),
        "abs_mean_imbalance": float(abs(late_i.mean())),
        "imbalance_sd": float(late_i.std()),
        "azimuth_corr": corr,
        "mean_drive_asymmetry": float(drive[half:].mean()) if len(drive) > half else 0.0,
        "drum_speed_deg_s": float(scene.drum_speed_deg_s),
        "drum_bars": float(scene.drum_bars),
    }


def _swat_score(traces: dict[str, list[float]], scene: BODY.Scene) -> dict[str, float]:
    """What the looming experiment measured.

    Computed for every run, stationary object included, because the static condition is
    the swat's baseline control and it needs the same numbers to be comparable. An
    earlier version skipped it when nothing was moving and the static column of the swat
    report read as 0 Hz, which looks like a dead brain rather than a quiet one.

    The headline numbers are the common-mode ones. `startle_ratio` is the peak summed
    descending rate divided by its baseline over the quiet ticks before the approach
    began, so 1.0 means an approaching object did not move the descending population at
    all. `ttc_at_peak` is where in the approach that peak landed, in seconds before
    contact, which is the number a real escape study reports.
    """
    total = np.asarray(traces.get("dn_total", []), float)
    if len(total) == 0:
        return {"startle_ratio": 0.0, "peak_dn_total": 0.0, "baseline_dn_total": 0.0}
    quiet = max(1, min(scene.motion_delay_ticks, len(total)))
    baseline = float(total[:quiet].mean())
    # The peak is searched over the APPROACH only. Searching the whole run finds the
    # noise spike that the quiet ticks always contain and reports a latency from before
    # the object moved.
    window = total[quiet:]
    peak_i = quiet + int(np.argmax(window)) if len(window) else int(np.argmax(total))
    ttc = np.asarray(traces.get("time_to_contact", []), float)
    out = {
        "baseline_dn_total": baseline,
        "peak_dn_total": float(total[peak_i]),
        "startle_ratio": float(total[peak_i] / baseline) if baseline > 0 else 0.0,
        "peak_startle": float(max(traces.get("startle", [0.0])[quiet:] or [0.0])),
        "baseline_startle": float(max(traces.get("startle", [0.0])[:quiet] or [0.0])),
        "ttc_at_peak": float(ttc[peak_i]) if peak_i < len(ttc) else float("nan"),
    }
    for key, values in traces.items():
        if key.startswith("escape:") and values:
            out[f"peak {key}"] = float(max(values))
    return out


# ── Two flies ───────────────────────────────────────────────────────────────────


@dataclass
class DuetResult:
    """What one two-fly run produced. One entry per fly in each dict."""

    ticks: int
    duration_s: float
    wall_s: float
    modes: dict[str, str]
    trajectories: dict[str, list]
    traces: dict[str, dict[str, list[float]]]
    metrics: dict[str, dict[str, float]]
    separation: list[float]
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    total_spikes: dict[str, int] = field(default_factory=dict)

    def summary(self) -> str:
        parts = []
        for name, m in self.metrics.items():
            parts.append(
                f"{name}({self.modes[name]}) closest {m['closest_distance']:.1f} mm"
                + (f", reached in {m['time_to_reach_s']:.2f}s" if m["reached"] else ", did not reach")
            )
        return " | ".join(parts) + f" | closest approach between them {min(self.separation):.1f} mm"


def duet_loop(
    conn: Connectome,
    modes: tuple[str, str] = ("connectome", "connectome"),
    ticks: int = 240,
    scene: BODY.Scene | None = None,
    turn_gain: float = 1.8,
    base_drive: float = 0.5,
    marker_radius: float = 0.0,
    retina_mode: str = "contrast",
    seed: int = 0,
    brain_width: int = 360,
    on_tick=None,
) -> DuetResult:
    """Two flies, two independent brains, one arena and one pillar.

    `modes` is per fly, so ("connectome", "shuffled") races a real brain against the
    null model in the same physics, under identical lighting, at the same moment. That
    is a stronger version of the control the single-fly demo runs sequentially: the two
    animals cannot differ by anything except their wiring.

    Both brains are built up front and stepped in the same loop, which costs about
    6.4 GB and runs at roughly half a single fly's speed.
    """
    import duet as DUET

    scene = scene or BODY.Scene(object_xy=(12.0, 0.0), object_radius=3.0)
    started = time.time()
    names = ("a", "b")

    world = DUET.DuetWorld(
        scene=scene, seed=seed, marker_radius=marker_radius, retina_mode=retina_mode
    )
    canvases = {n: VIZ.BrainCanvas(conn, width=brain_width) for n in names}

    import brain as BRAIN

    brains, links = {}, {}
    # Cached per MODE, not per fly. A calibration is a property of the wiring, not of
    # the animal, so two connectome flies share one and measuring it twice would cost
    # three extra whole-network builds for an identical answer. Keying it by mode is
    # what makes that safe: an earlier version kept a single `calibration` variable and
    # reused whichever one happened to be computed first, so ("shuffled", "connectome")
    # quietly ran the real fly on the shuffled brain's turn sign.
    calibrations: dict[str, BR.Calibration] = {}
    for name, mode in zip(names, modes):
        wiring = conn.shuffled(seed=seed) if mode == "shuffled" else conn
        if mode == "nobrain":
            brains[name], links[name] = None, None
            continue
        if mode not in calibrations:
            calibrations[mode] = BR.calibrate(wiring, seed=seed)
        calibration = calibrations[mode]
        links[name] = BR.Bridge(
            conn=wiring, calibration=calibration, base_drive=base_drive,
            turn_gain=turn_gain, behaviour="attract",
        )
        brains[name] = BRAIN.Brain(
            conn=wiring, stim_groups=links[name].stim_groups(), seed=seed
        )

    keys = ("eye_left", "eye_right", "dn_left", "dn_right", "drive_left", "drive_right",
            "imbalance", "distance", "bearing")
    traces = {n: {k: [] for k in keys} for n in names}
    counts_history = {n: [] for n in names}
    statuses = {n: [] for n in names}
    traj_history = {n: [] for n in names}
    total_spikes = {n: 0 for n in names}
    separation: list[float] = []
    reached_at: dict[str, int] = {}

    for i in range(ticks):
        signals, counts = {}, {}
        for name in names:
            eyes = world.look(name)
            if brains[name] is None:
                signals[name] = BR.constant_signal(base_drive)
                readout = {"dn_left": 0.0, "dn_right": 0.0, "motor": 0.0}
                counts[name] = np.zeros(conn.n, np.float32)
                imbalance = 0.0
            else:
                links[name].apply_vision(brains[name], eyes)
                tick = brains[name].run(BODY.BRAIN_DT)
                counts[name] = tick.counts
                readout = links[name].read_descending(tick)
                signals[name] = links[name].descending_signal(readout)
                imbalance = links[name].imbalance
                total_spikes[name] += int(counts[name].sum())

            traces[name]["eye_left"].append(float(eyes[0]))
            traces[name]["eye_right"].append(float(eyes[1]))
            traces[name]["dn_left"].append(readout["dn_left"])
            traces[name]["dn_right"].append(readout["dn_right"])
            traces[name]["drive_left"].append(float(signals[name][0]))
            traces[name]["drive_right"].append(float(signals[name][1]))
            traces[name]["imbalance"].append(float(imbalance))
            traces[name]["distance"].append(world.distance_to_object(name))
            traces[name]["bearing"].append(world.bearing_deg(name))
            counts_history[name].append(counts[name])
            statuses[name].append(
                VID.Status(
                    tick=i, t=(i + 1) * BODY.BRAIN_DT,
                    drive=(float(signals[name][0]), float(signals[name][1])),
                    dn=(readout["dn_left"], readout["dn_right"]),
                    imbalance=float(imbalance),
                    distance=world.distance_to_object(name),
                    bearing=world.bearing_deg(name),
                    spikes=total_spikes[name],
                )
            )

        separation.append(world.separation)
        world.step(signals)
        for name in names:
            traj_history[name].append(list(world.trajectories[name]))
            if (
                name not in reached_at
                and world.distance_to_object(name) <= scene.object_radius + 2.0
            ):
                reached_at[name] = i + 1
                log.info("fly %s reached the pillar at tick %d", name, i + 1)

        if on_tick is not None and (i % 20 == 0 or i == ticks - 1):
            on_tick(i + 1, ticks, traces)
        # Both flies arriving ends it; one arriving does not, because the interesting
        # part is whether the second one still gets there with the first in the way.
        if len(reached_at) == len(names):
            break

    used = len(separation)
    for name in names:
        canvases[name].autoscale(counts_history[name])
        canvases[name].reset()

    frames = []
    body_frames = {n: world.frames(n) for n in names}
    for i in range(used):
        for name in names:
            canvases[name].update(counts_history[name][i])
        if all(i < len(body_frames[n]) for n in names):
            frames.append(
                VID.compose_duet(
                    body_frames["a"][i], body_frames["b"][i],
                    canvases["a"].frame(), canvases["b"].frame(),
                    statuses["a"][i], statuses["b"][i],
                    traj_history["a"][i], traj_history["b"][i],
                    scene.object_xy, scene.object_radius, separation[i],
                )
            )

    metrics = {}
    for name in names:
        distance = np.asarray(traces[name]["distance"], float)
        bearing = np.abs(np.asarray(traces[name]["bearing"], float))
        path = np.asarray([(x, y) for x, y, _ in world.trajectories[name]], float)
        steps = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.zeros(1)
        late = max(1, len(bearing) // 3)
        metrics[name] = {
            "start_distance": float(distance[0]),
            "final_distance": float(distance[-1]),
            "closest_distance": float(distance.min()),
            "approach": float(distance[0] - distance.min()),
            "mean_abs_bearing_late": float(bearing[-late:].mean()),
            "path_length": float(steps.sum()),
            "straightness": float(
                np.linalg.norm(path[-1] - path[0]) / max(steps.sum(), 1e-6)
            ),
            "reached": float(name in reached_at),
            "time_to_reach_s": float(
                reached_at.get(name, used) * BODY.BRAIN_DT
            ),
            "ticks_used": float(used),
            # How much of the run this fly spent close enough to the other one to be
            # in its way. The pillar is 3 mm and a fly is 2.5 mm long, so under 5 mm
            # they are genuinely crowding each other.
            "ticks_crowded": float(sum(1 for s in separation if s < 5.0)),
            "closest_separation": float(min(separation)),
        }

    result = DuetResult(
        ticks=used,
        duration_s=used * BODY.BRAIN_DT,
        wall_s=time.time() - started,
        modes=dict(zip(names, modes)),
        trajectories={n: list(world.trajectories[n]) for n in names},
        traces=traces,
        metrics=metrics,
        separation=separation,
        frames=frames,
        total_spikes=total_spikes,
    )
    world.close()
    log.info(result.summary())
    return result


# ── The fly riding a robot ──────────────────────────────────────────────────────


@dataclass
class RideResult:
    """What one robot-riding run produced."""

    mode: str
    ticks: int
    duration_s: float
    wall_s: float
    trajectory: list[tuple[float, float, float]]
    traces: dict[str, list[float]]
    metrics: dict[str, float]
    frames: list[np.ndarray] = field(default_factory=list, repr=False)
    total_spikes: int = 0

    def summary(self) -> str:
        m = self.metrics
        return (
            f"{self.mode:11s} start {m['start_distance']:5.0f} mm -> closest "
            f"{m['closest_distance']:5.0f} mm  |bearing| {m['mean_abs_bearing_late']:5.1f} deg  "
            f"{'REACHED in ' + format(m['time_to_reach_s'], '.2f') + 's' if m['reached'] else 'did not reach'}  "
            f"{self.total_spikes:,} spikes in {self.wall_s:.0f}s"
        )


def ride_loop(
    conn: Connectome,
    mode: str = "connectome",
    ticks: int = 400,
    object_distance: float = 200.0,
    object_radius: float = 25.0,
    spawn_heading_deg: float = 45.0,
    turn_gain: float = 1.8,
    base_drive: float = 0.5,
    reach_margin: float = 20.0,
    seed: int = 0,
    brain_width: int = 420,
    on_tick=None,
) -> RideResult:
    """The connectome steering a four-legged robot it is sitting on top of.

    Same brain, same eyes, same bridge as every other experiment. The only new thing is
    `rider.ride_command`, four lines that turn the fly's per-side drives into the robot's
    speed and yaw. If the fly reaches the pillar from up there, the descending command
    really is body-independent, which is the claim the whole repo is built on.
    """
    import rider as RIDE

    started = time.time()
    wiring = conn.shuffled(seed=seed) if mode == "shuffled" else conn

    world = RIDE.RiderWorld(
        object_xy=(object_distance, 0.0),
        object_radius=object_radius,
        spawn_heading_deg=spawn_heading_deg,
    )
    canvas = VIZ.BrainCanvas(conn, width=brain_width)
    chem = CHEM.Neurochemistry(conn)

    brain = link = None
    if mode != "nobrain":
        import brain as BRAIN

        calibration = BR.calibrate(wiring, seed=seed)
        link = BR.Bridge(
            conn=wiring, calibration=calibration, base_drive=base_drive,
            turn_gain=turn_gain, behaviour="attract",
        )
        brain = BRAIN.Brain(conn=wiring, stim_groups=link.stim_groups(), seed=seed)

    keys = ("eye_left", "eye_right", "dn_left", "dn_right", "imbalance",
            "distance", "bearing", "heading", "speed_cmd", "turn_cmd")
    traces: dict[str, list[float]] = {k: [] for k in keys}
    for key in chem.trace_keys():
        traces[key] = []

    counts_history, statuses, traj_history = [], [], []
    total_spikes = 0
    reached_at = None

    for i in range(ticks):
        eyes = world.look()
        if brain is None:
            signal = BR.constant_signal(base_drive)
            readout = {"dn_left": 0.0, "dn_right": 0.0, "motor": 0.0}
            counts = np.zeros(conn.n, np.float32)
            imbalance = 0.0
        else:
            link.apply_vision(brain, eyes)
            tick = brain.run(RIDE.BRAIN_DT)
            counts = tick.counts
            readout = link.read_descending(tick)
            signal = link.descending_signal(readout)
            imbalance = link.imbalance
            total_spikes += int(counts.sum())

        command = RIDE.ride_command(signal)
        counts_history.append(counts)
        statuses.append(
            VID.Status(
                tick=i, t=(i + 1) * RIDE.BRAIN_DT,
                drive=(float(command[0]), float(command[1])),
                dn=(readout["dn_left"], readout["dn_right"]),
                imbalance=float(imbalance),
                distance=world.distance_to_object,
                bearing=world.bearing_deg,
                spikes=total_spikes,
            )
        )
        world.step(signal)
        traj_history.append(list(world.trajectory))

        traces["eye_left"].append(float(eyes[0]))
        traces["eye_right"].append(float(eyes[1]))
        traces["dn_left"].append(readout["dn_left"])
        traces["dn_right"].append(readout["dn_right"])
        traces["imbalance"].append(float(imbalance))
        traces["distance"].append(world.distance_to_object)
        traces["bearing"].append(world.bearing_deg)
        traces["heading"].append(world.heading_deg)
        traces["speed_cmd"].append(float(command[0]))
        traces["turn_cmd"].append(float(command[1]))
        for key, value in chem.sample(counts).items():
            traces[key].append(value)

        if on_tick is not None and (i % 20 == 0 or i == ticks - 1):
            on_tick(i + 1, ticks, traces)
        if reached_at is None and world.distance_to_object <= object_radius + reach_margin:
            reached_at = i + 1
            log.info("the robot reached the pillar at tick %d", reached_at)
            break
        if not world.upright:
            log.info("the robot fell over at tick %d", i + 1)
            break

    used = len(counts_history)
    canvas.autoscale(counts_history)
    canvas.reset()
    frames = []
    for i in range(min(used, len(world.frames))):
        canvas.update(counts_history[i])
        frames.append(
            VID.compose_ride(
                world.frames[i], canvas.frame(), statuses[i], traj_history[i],
                (object_distance, 0.0), object_radius,
            )
        )

    distance = np.asarray(traces["distance"], float)
    bearing = np.abs(np.asarray(traces["bearing"], float))
    path = np.asarray([(x, y) for x, y, _ in world.trajectory], float)
    steps = np.linalg.norm(np.diff(path, axis=0), axis=1) if len(path) > 1 else np.zeros(1)
    late = max(1, len(bearing) // 3)
    metrics = {
        "start_distance": float(distance[0]),
        "final_distance": float(distance[-1]),
        "closest_distance": float(distance.min()),
        "approach": float(distance[0] - distance.min()),
        "mean_abs_bearing_late": float(bearing[-late:].mean()),
        "path_length": float(steps.sum()),
        "straightness": float(np.linalg.norm(path[-1] - path[0]) / max(steps.sum(), 1e-6)),
        "reached": float(reached_at is not None),
        "time_to_reach_s": float((reached_at or ticks) * RIDE.BRAIN_DT),
        "ticks_used": float(used),
        "upright": float(world.upright),
        "mean_turn_cmd": float(np.mean(traces["turn_cmd"])),
    }
    result = RideResult(
        mode=mode, ticks=used, duration_s=used * RIDE.BRAIN_DT,
        wall_s=time.time() - started, trajectory=list(world.trajectory),
        traces=traces, metrics=metrics, frames=frames, total_spikes=total_spikes,
    )
    world.close()
    log.info(result.summary())
    return result
