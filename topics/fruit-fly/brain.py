"""The brain: 138,639 leaky integrate-and-fire neurons wired by the FlyWire connectome.

There is no training here and no fitted parameters. Every synapse's strength is the
number of synapses the electron microscope actually found between that pair of neurons,
signed by the presynaptic cell's predicted neurotransmitter, times one global scale
`w_syn`. That is the whole model. If the fly does something, the wiring did it.

The neuron model and all eight constants are Shiu et al. 2024 (Nature), "A leaky
integrate-and-fire computational model based on the connectome of the entire adult
Drosophila brain reveals insights into sensorimotor processing", reproduced from their
reference implementation so that the brain-only numbers here can be checked against a
published one:

    v_0   -52 mV   resting potential        (Kakaria & de Bivort 2017)
    v_th  -45 mV   spike threshold              7 mV of headroom, total
    t_mbr  20 ms   membrane time constant
    tau     5 ms   synaptic (alpha) time constant  (Jurgensen et al. 2022)
    t_rfc 2.2 ms   refractory period            (Lazar et al. 2021)
    t_dly 1.8 ms   synaptic delay               (Paul et al. 2015)
    w_syn .275 mV  volts per synapse        the ONE free parameter in the model
    f_poi    250   external drive scale

── What this file adds to that reference implementation ────────────────────────
Shiu et al. run open loop: pick neurons, inject Poisson spikes at a fixed rate, run for
one second, count spikes. A body needs something else, so `Brain` differs in two ways:

1. **It can be stepped.** `run(0.015)` advances 15 ms and returns the spikes from those
   15 ms alone, leaving every membrane potential intact for the next call. The body
   moves in between. 15 ms is the brain-body sync interval Eon Systems used for the
   same coupling, and at the fly's ~13 mm/s walking speed it is well under a body
   length of travel per tick.

2. **Its inputs are variable.** The reference uses `PoissonInput`, whose rate is fixed
   at construction time. A fly whose eyes are attached to the world needs the rate to
   change every tick, so this uses a `PoissonGroup` plus explicit synapses instead,
   which exposes a writable `rates` array. Same drive, same weight (`w_syn * f_poi` =
   68.75 mV, far above the 7 mV threshold, so one external event reliably means one
   spike), but steerable.

Measured on the DGX Spark (GB10, arm64), full network, Cython codegen:

    build                1.9 s
    first run (compile) 12.3 s     one-off, cached in CYTHON_CACHE_DIR
    thereafter          0.3 s per 100 ms of brain at low activity, ~1 s at high
    peak RSS            3.2 GB
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from connectome import Connectome

log = logging.getLogger(__name__)

# Shiu et al. 2024, table above. Held as plain floats in SI units and converted to
# Brian2 quantities inside `_build`, so that this module can be imported (and the
# constants inspected) without paying Brian2's ~2 s import.
PARAMS: dict[str, float] = {
    "v_0": -52e-3,
    "v_rst": -52e-3,
    "v_th": -45e-3,
    "t_mbr": 20e-3,
    "tau": 5e-3,
    "t_rfc": 2.2e-3,
    "t_dly": 1.8e-3,
    "w_syn": 0.275e-3,
    "f_poi": 250.0,
}

# The membrane equation. `g` is the alpha-synapse conductance every incoming spike adds
# to; `rfc` is per-neuron so that externally driven cells can have their refractory
# period zeroed without touching anyone else's.
EQS = """
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
vth                            : volt
"""


def preimport() -> None:
    """Import brian2 from the MAIN THREAD, before any worker thread needs it.

    `brian2/__init__.py` installs a SIGINT handler at import time (so that Ctrl-C can
    interrupt a long `run()`), and CPython only permits installing a signal handler
    from the main thread. A Flyte task that offloads its blocking work with
    `asyncio.to_thread` therefore imports brian2 in a worker, and the import itself
    dies with:

        ValueError: signal only works in main thread of the main interpreter

    Calling this once on the main thread first fixes it permanently: Python caches the
    module, so the later import inside the worker is a no-op that never reaches the
    signal call. Costs about two seconds.
    """
    import brian2  # noqa: F401


@dataclass
class Tick:
    """What one `Brain.run()` returns: the spikes from that interval and nothing else.

    Attributes:
        counts: Spike count per neuron during this tick, shape (n_neurons,). This is
            the array brainviz.py lights the anatomy with.
        duration: Length of the tick in seconds, so callers can divide into Hz.
        t_end: Brain clock time at the end of the tick, in seconds.
    """

    counts: np.ndarray
    duration: float
    t_end: float

    def rate(self, idx: np.ndarray) -> float:
        """Summed population firing rate over `idx`, in Hz."""
        if len(idx) == 0:
            return 0.0
        return float(self.counts[idx].sum() / self.duration)


@dataclass
class Brain:
    """A runnable whole-brain LIF network.

    Args:
        conn: The wiring. Pass `conn.shuffled()` for the null model.
        stim_groups: Named sets of Brian indices that can be driven externally, e.g.
            ``{"eye_left": [...], "eye_right": [...]}``. Only neurons named here can
            ever receive input from outside; everything else is driven purely by the
            connectome.
        params: Override any of `PARAMS`.
        record: Brian indices to record spikes from. `None` records everything, which
            is what the brain visualiser needs and costs ~12 bytes per spike.
        seed: Seeds Brian2's Poisson streams.
        thresholds: Optional per-neuron spike threshold overrides, as
            ``{name: (indices, volts)}``. Everything not named keeps `params["v_th"]`.

            This exists for the Kenyon cells and the reason is measured. At the uniform
            threshold, ANY odour drives 65% of the 5,177 Kenyon cells and two completely
            different odour channels overlap with a Jaccard index of 0.99. A mushroom
            body whose code is that dense cannot support odour-specific learning,
            because depressing "the cells active for odour A" depresses nearly every
            cell. Real Kenyon cells are famously hard to fire and respond to about 5 to
            10% of odours; this is the knob that says so.
    """

    conn: Connectome
    stim_groups: dict[str, np.ndarray] = field(default_factory=dict)
    params: dict[str, float] = field(default_factory=lambda: dict(PARAMS))
    record: np.ndarray | None = None
    seed: int = 0
    thresholds: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)

    _net: object = field(default=None, init=False, repr=False)
    _mon: object = field(default=None, init=False, repr=False)
    _poisson: object = field(default=None, init=False, repr=False)
    _slices: dict[str, slice] = field(default_factory=dict, init=False, repr=False)
    _cursor: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        self._build()

    # ── Construction ────────────────────────────────────────────────────────────

    def _build(self) -> None:
        import brian2 as b2

        # Cython, not numpy. Brian2's numpy target is roughly 30x slower on a network
        # this sparse; see config._BUILD_APT for why g++ is in the image.
        b2.prefs.codegen.target = "cython"
        b2.seed(self.seed)
        b2.BrianLogger.log_level_warn()

        p = self.params
        ns = {
            "v_0": p["v_0"] * b2.volt,
            "v_rst": p["v_rst"] * b2.volt,
            "v_th": p["v_th"] * b2.volt,
            "t_mbr": p["t_mbr"] * b2.second,
            "tau": p["tau"] * b2.second,
        }

        neurons = b2.NeuronGroup(
            N=self.conn.n,
            model=EQS,
            method="linear",
            # Per-neuron, not a namespace constant, so a population can be made harder
            # to fire than the rest of the brain. See `thresholds`.
            threshold="v > vth",
            # `g = 0*mV` on reset is Shiu et al.'s: a spike discharges the synaptic
            # conductance as well as the membrane, so input arriving during the
            # refractory period is discarded rather than accumulated.
            reset="v = v_rst; g = 0*mV",
            refractory="rfc",
            namespace=ns,
            name="brain",
        )
        neurons.v = ns["v_0"]
        neurons.g = 0
        neurons.rfc = p["t_rfc"] * b2.second
        neurons.vth = p["v_th"] * b2.volt
        for name, (idx, volts) in self.thresholds.items():
            idx = np.asarray(idx, dtype=np.int64)
            if len(idx):
                neurons.vth[idx] = volts * b2.volt
                log.info("threshold override %s: %d neurons at %.1f mV",
                         name, len(idx), volts * 1e3)

        synapses = b2.Synapses(
            neurons,
            neurons,
            model="w : volt",
            on_pre="g += w",
            delay=p["t_dly"] * b2.second,
            name="connectome",
        )
        synapses.connect(i=self.conn.pre, j=self.conn.post)
        synapses.w = self.conn.weight * p["w_syn"] * b2.volt

        objects = [neurons, synapses]

        # One Poisson source per stimulable neuron, all groups concatenated into a
        # single PoissonGroup so there is exactly one object to update per tick.
        if self.stim_groups:
            targets, offset = [], 0
            for name, idx in self.stim_groups.items():
                idx = np.asarray(idx, dtype=np.int64)
                self._slices[name] = slice(offset, offset + len(idx))
                targets.append(idx)
                offset += len(idx)
            targets = np.concatenate(targets) if targets else np.array([], np.int64)

            # Externally driven cells get no refractory period, matching the reference
            # implementation: the Poisson train IS their spike train, and a 2.2 ms
            # refractory floor would silently cap every sensory neuron at 454 Hz.
            neurons.rfc[targets] = 0 * b2.second

            self._poisson = b2.PoissonGroup(len(targets), rates=0 * b2.Hz, name="drive")
            drive = b2.Synapses(
                self._poisson,
                neurons,
                on_pre="v_post += w_ext",
                namespace={"w_ext": p["w_syn"] * p["f_poi"] * b2.volt},
                name="drive_syn",
            )
            drive.connect(i=np.arange(len(targets)), j=targets)
            objects += [self._poisson, drive]

        record = True if self.record is None else np.asarray(self.record, dtype=np.int64)
        self._mon = b2.SpikeMonitor(neurons, record=record, name="spikes")
        objects.append(self._mon)

        self._net = b2.Network(*objects)
        log.info(
            "brain built: %d neurons, %d edges, %d stimulable",
            self.conn.n,
            len(self.conn.pre),
            sum(len(v) for v in self.stim_groups.values()),
        )

    # ── Driving it ──────────────────────────────────────────────────────────────

    def set_rate(self, group: str, hz: float | np.ndarray) -> None:
        """Set the external drive on a named stimulus group, in Hz.

        A scalar drives every neuron in the group at the same rate; an array must match
        the group's length. Rates persist until changed, so a group set once and never
        touched again keeps firing.
        """
        import brian2 as b2

        if self._poisson is None:
            raise RuntimeError("this Brain was built with no stim_groups")
        if group not in self._slices:
            raise KeyError(f"no stim group {group!r}; have {sorted(self._slices)}")
        sl = self._slices[group]
        width = sl.stop - sl.start
        values = np.full(width, float(hz)) if np.isscalar(hz) else np.asarray(hz, float)
        if values.shape != (width,):
            raise ValueError(f"group {group!r} has {width} neurons, got {values.shape}")
        # Negative rates are silently clipped rather than raising: they arise naturally
        # from a bridge gain applied to a signed sensory difference, and clipping at
        # zero is the biologically correct response (a neuron cannot fire negatively).
        self._poisson.rates[sl] = np.clip(values, 0, None) * b2.Hz

    def run(self, duration: float) -> Tick:
        """Advance the brain by `duration` seconds and return that interval's spikes."""
        import brian2 as b2

        self._net.run(duration * b2.second)
        # The monitor accumulates for the whole run; take only what arrived since the
        # last call. Slicing the dynamic array is a view-copy of the new tail, not of
        # the whole history, so this stays O(spikes this tick).
        fired = np.asarray(self._mon.i[self._cursor:], dtype=np.int64)
        self._cursor += len(fired)
        counts = np.bincount(fired, minlength=self.conn.n).astype(np.float32)
        return Tick(counts=counts, duration=duration, t_end=float(self._net.t))

    # ── Whole-history readout, for the brain-only probe and the raster ───────────

    @property
    def spike_times(self) -> tuple[np.ndarray, np.ndarray]:
        """Every spike so far as (neuron_index, time_seconds)."""
        return (
            np.asarray(self._mon.i[:], dtype=np.int64),
            np.asarray(self._mon.t[:], dtype=np.float64),
        )

    @property
    def num_spikes(self) -> int:
        return int(self._mon.num_spikes)

    def rates(self, elapsed: float) -> np.ndarray:
        """Mean firing rate per neuron over `elapsed` seconds, in Hz."""
        i, _ = self.spike_times
        return np.bincount(i, minlength=self.conn.n) / elapsed


def probe(
    conn: Connectome,
    stimulus: dict[str, np.ndarray],
    duration: float = 0.3,
    rate_hz: float = 150.0,
    seed: int = 0,
) -> np.ndarray:
    """Open-loop experiment: drive these neurons, run, return per-neuron rates in Hz.

    This is the Shiu et al. protocol, and it is what `pipeline.probe_brain` sweeps to
    find out which sensory channels lateralise before any body is attached. Builds a
    fresh network each call precisely so that no membrane potential leaks between
    conditions.
    """
    brain = Brain(conn=conn, stim_groups=stimulus, seed=seed)
    for name in stimulus:
        brain.set_rate(name, rate_hz)
    brain.run(duration)
    return brain.rates(duration)
