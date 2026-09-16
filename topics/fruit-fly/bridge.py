"""The bridge: eyes into the connectome, descending neurons back out to the legs.

This is the only file in the repo where a number is chosen rather than measured, so it
is the one to read sceptically. Three things happen here:

  1. Eye darkening (0..1 per eye) becomes a Poisson firing rate on that eye's
     photoreceptor neurons. One free parameter: `eye_gain`, in Hz per unit darkening.

  2. The brain runs, and the left and right DESCENDING POPULATIONS are read out.

  3. Their imbalance becomes the two-number descending command the walking controller
     wants. Two free parameters: `base_drive` (how fast to walk) and `turn_gain` (how
     hard to steer per unit of neural imbalance).

Everything else, including the SIGN of the turn, is measured by `calibrate()` against
the brain itself rather than asserted here.

── Why the population, and not DNa02 ───────────────────────────────────────────
The obvious design is to read the two DNa02 neurons, the best-known steering cells in
the fly, one per side, and turn toward whichever fires harder. Measured on this box,
with 500 ms of drive at 150 Hz, that design does not work:

    stimulus                     DNa02 left   DNa02 right
    ORN_DM1 left only               50 Hz         2 Hz
    ORN_DM1 right only              54 Hz         0 Hz
    ORN_DM1 both                    50 Hz         0 Hz
    ORN_DM4 left / right         50 / 46 Hz     0 / 0 Hz

DNa02-left fires at about 50 Hz and DNa02-right is silent NO MATTER WHICH SIDE IS
STIMULATED. A fly steered by that pair turns the same way forever. This is not a bug in
the loop, it is what a single cell does in a model where every synapse has the same
strength per synapse and nothing is tuned: one cell of the pair happens to sit closer
to threshold, and it wins every time.

The 1,291-neuron descending population does not have that problem, because the bias
averages out across cells. Same protocol, summed population rate:

    stimulus                     DN pop left   DN pop right   L-R
    photoreceptors left             217 Hz        377 Hz      -160
    photoreceptors right            327 Hz        173 Hz      +153
    mechanosensory left           8,670 Hz      4,673 Hz    +3,997
    mechanosensory right          2,913 Hz      7,207 Hz    -4,293
    olfactory left                2,390 Hz      2,510 Hz      -120
    olfactory right               2,410 Hz      2,387 Hz       +23

Vision and touch flip sign cleanly with the side of the stimulus; vision is
contralateral (left eye drives right descending neurons) and touch is ipsilateral,
which is what the anatomy predicts for both. Smell does not lateralise at all in this
model, which is why this demo is visual and not an odour-tracking one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

import connectome as C
from connectome import Connectome

log = logging.getLogger(__name__)

# ── The operating point, measured ───────────────────────────────────────────────
#
# This model has NO spontaneous activity: with zero input, zero neurons fire, forever.
# It also has a hard floor below which nothing propagates. Measured here, driving the
# left eye's photoreceptors for 300 ms and summing the descending populations:
#
#     photoreceptor drive     5     10     25     50     75    100    150    200 Hz
#     DN left                 0      0     23     87    127    140    210    257 Hz
#     DN right                0      0     60    213    240    297    367    393 Hz
#     descending signal      none  none  weak   good   good   good   good   saturating
#
# Below 25 Hz the brain is silent no matter what the eye sees. So the first version of
# this bridge, which mapped "fraction of the eye that darkened" (0 to about 0.075 for a
# pillar at this distance) directly onto 0-150 Hz, delivered 7-11 Hz and produced a
# perfectly functioning fly with a completely silent brain.
#
# The fix is the one the animal uses. A photoreceptor in daylight is not silent waiting
# for an object; it fires tonically and an object MODULATES that rate. So both eyes get
# TONIC_HZ all the time, which parks the visual system in the responsive band, and the
# darkening adds EYE_GAIN on top. With the numbers below, an eye seeing nothing sits at
# 50 Hz and an eye with the pillar square in it reaches about 150 Hz.
TONIC_HZ = 50.0
DEFAULT_EYE_GAIN = 1300.0
MAX_EYE_HZ = 200.0

# Drive on the gustatory receptor neurons while the fly's tarsi are on the food. Same
# 150 Hz the reference implementation uses for every stimulation experiment.
TASTE_HZ = 150.0

# Drive on one side's mechanosensory neurons while that antenna is being poked.
#
# This is by far the strongest lateralised channel in the model and it is not close.
# Measured, 300 ms of one-sided drive, summed descending population rates:
#
#     poke left  @  25 Hz -> DN left  1,383  right    750   imbalance -0.297
#     poke right @  25 Hz -> DN left    373  right  1,147   imbalance +0.509
#     poke left  @  75 Hz -> DN left  4,893  right  2,570   imbalance -0.311
#     poke right @  75 Hz -> DN left  1,403  right  3,703   imbalance +0.450
#     poke left  @ 150 Hz -> DN left  8,513  right  4,610   imbalance -0.297
#     poke right @ 150 Hz -> DN left  2,843  right  7,107   imbalance +0.428
#
# The sign flips cleanly with the side at every drive level, and the magnitude is an
# order of magnitude above what the visual pathway manages. A fly that cannot see a
# pillar without 150 ms of smoothing reacts to a touch in a single tick.
#
# The population is 1,363 left and 1,293 right, and the biggest component is BM_InOm
# (1,113 cells), the bristle mechanoreceptors between the ommatidia, plus Johnston's
# organ (JO-*) in the antenna. So "poke" here means deflecting the bristles on one side
# of the head, which is a real and very old fly experiment.
TOUCH_HZ = 75.0
# Feeding motor-neuron rate that counts as a full proboscis extension, from the
# measurement above (sugar drives the population to ~1,610 Hz).
FEEDING_FULL_HZ = 1200.0


@dataclass
class Calibration:
    """What the brain does when you shine a light in one eye. Measured, not assumed.

    Attributes:
        dn_rates: Condition -> (left population Hz, right population Hz).
        bias: The imbalance with BOTH eyes at tonic drive and nothing to see. This is
            not zero, because the two descending populations are not identically
            excitable, and every steering decision subtracts it.
        asym_left_eye: Imbalance, bias removed, when the left eye is the brighter-
            driven one.
        asym_right_eye: Same for the right eye.
        turn_sign: +1 if a positive imbalance should make the fly turn LEFT. Derived
            from `asym_left_eye` so that "attract" always means "walk toward it",
            whichever way the connectome happens to route the signal.
        separation: |asym_left_eye - asym_right_eye|. How much signal there is to
            steer with at all. Near zero means this sensory channel cannot steer this
            body, and the run should be reported as such rather than tuned until it
            moves. The shuffled control is expected to land here.
    """

    dn_rates: dict[str, tuple[float, float]]
    bias: float
    asym_left_eye: float
    asym_right_eye: float
    turn_sign: float
    separation: float

    def table(self) -> list[tuple[str, str]]:
        rows = [
            (f"DN population, {k}", f"L {l:,.0f} Hz / R {r:,.0f} Hz")
            for k, (l, r) in self.dn_rates.items()
        ]
        rows += [
            ("intrinsic bias (both eyes equal)", f"{self.bias:+.3f}"),
            ("imbalance, left eye brighter", f"{self.asym_left_eye:+.3f}"),
            ("imbalance, right eye brighter", f"{self.asym_right_eye:+.3f}"),
            ("separation", f"{self.separation:.3f}"),
            ("turn sign", f"{self.turn_sign:+.0f}"),
        ]
        return rows


def calibrate(
    conn: Connectome,
    duration: float = 0.3,
    tonic_hz: float = TONIC_HZ,
    modulation_hz: float = 100.0,
    seed: int = 0,
    params: dict | None = None,
) -> Calibration:
    """Measure what an object on each side does to the descending populations.

    Three conditions at the loop's real operating point, not at some convenient
    extreme: both eyes tonic (nothing to see), left eye brighter, right eye brighter.
    The first condition is the control that gives us `bias`, and subtracting it is what
    stops a merely lopsided brain from being read as a permanent turn command.

    Costs three fresh brain builds, about 30 seconds. Run once per wiring: the shuffled
    control needs its own calibration precisely because the question being asked is
    whether it still HAS a sign to measure.
    """
    import brain as B

    eye_left, eye_right = conn.sided(**C.SENSORY_GROUPS["photoreceptors"])
    dn_left, dn_right = conn.sided(super_class="descending")
    groups = {"eye_left": eye_left, "eye_right": eye_right}
    bright = tonic_hz + modulation_hz

    rates: dict[str, tuple[float, float]] = {}
    for label, (hz_l, hz_r) in (
        ("both eyes equal", (tonic_hz, tonic_hz)),
        ("left eye brighter", (bright, tonic_hz)),
        ("right eye brighter", (tonic_hz, bright)),
    ):
        brain = B.Brain(conn=conn, stim_groups=groups, seed=seed,
                        params=dict(params) if params else dict(B.PARAMS))
        brain.set_rate("eye_left", hz_l)
        brain.set_rate("eye_right", hz_r)
        brain.run(duration)
        per_neuron = brain.rates(duration)
        rates[label] = (float(per_neuron[dn_left].sum()), float(per_neuron[dn_right].sum()))
        log.info("calibrate %-18s DN L=%.0f R=%.0f", label, *rates[label])

    def asym(key: str) -> float:
        left, right = rates[key]
        total = left + right
        return 0.0 if total <= 0 else (right - left) / total

    bias = asym("both eyes equal")
    asym_left = asym("left eye brighter") - bias
    asym_right = asym("right eye brighter") - bias
    # If the LEFT eye seeing more produces a positive imbalance, then positive
    # imbalance means "something is on my left", so turning toward it means left.
    turn_sign = 1.0 if asym_left >= asym_right else -1.0
    return Calibration(
        dn_rates=rates,
        bias=bias,
        asym_left_eye=asym_left,
        asym_right_eye=asym_right,
        turn_sign=turn_sign,
        separation=abs(asym_left - asym_right),
    )


@dataclass
class Bridge:
    """Carries signals between `body.FlyWorld` and `brain.Brain`.

    Args:
        conn: The wiring, for the photoreceptor and descending index sets.
        calibration: From `calibrate()`. Supplies the turn sign.
        base_drive: Forward walking drive when the two sides balance. 1.0 is the
            controller's nominal amplitude and gives roughly 16 mm/s; 0.5 halves that
            and buys the fly time to turn before it walks past the pillar.
        turn_gain: Steering strength, in drive units per unit of neural imbalance.
            Measured: at 0.6 the fly turns about 25 deg/s, enough to correct a 35 deg
            error before it passes the pillar and NOT enough for 60 deg (a run at 60
            deg closed only 11.7 mm -> 10.8 mm). At 1.8 it corrects 60 deg in 1.3 s and
            arrives with a 6 deg bearing error.
        eye_gain: Hz of photoreceptor drive added per unit of eye darkening.
        tonic_hz: Baseline photoreceptor rate, always on. See the dose-response
            table at the top of this file for why it is not zero.
        behaviour: "attract" walks toward the object, "avoid" away from it. This flips
            one sign and nothing else; it is the one behavioural choice made by hand,
            and it is made explicit rather than buried because the connectome supplies
            the lateralisation but not the valence of a plain dark post.
        smoothing: Exponential smoothing on the imbalance, per tick. This one is
            load-bearing. Measured under a scripted stimulus at the loop's operating
            point, the per-tick imbalance separates the conditions cleanly in the mean
            (+0.209 left brighter, -0.115 right brighter, +0.027 neither) but carries a
            per-tick standard deviation of 0.22 to 0.41: a single 15 ms window holds
            less signal than noise. 0.9 keeps about a 150 ms memory, cuts the noise
            roughly fourfold, and is in the range of real visual-turning latencies in
            walking flies.
    """

    conn: Connectome
    calibration: Calibration
    base_drive: float = 0.5
    turn_gain: float = 1.8
    eye_gain: float = DEFAULT_EYE_GAIN
    tonic_hz: float = TONIC_HZ
    behaviour: str = "attract"
    smoothing: float = 0.9
    drive_limits: tuple[float, float] = (0.1, 1.6)
    startle_gain: float = 0.0
    startle_fast: float = 0.4
    startle_tau: float = 0.97
    startle_warmup: int = 12

    eye_left: np.ndarray = field(init=False, repr=False)
    eye_right: np.ndarray = field(init=False, repr=False)
    dn_left: np.ndarray = field(init=False, repr=False)
    dn_right: np.ndarray = field(init=False, repr=False)
    motor: np.ndarray = field(init=False, repr=False)
    mech_left: np.ndarray = field(init=False, repr=False)
    mech_right: np.ndarray = field(init=False, repr=False)
    sugar: np.ndarray = field(init=False, repr=False)
    bitter: np.ndarray = field(init=False, repr=False)
    named: dict[str, dict[str, np.ndarray]] = field(init=False, repr=False)
    escape: dict[str, np.ndarray] = field(init=False, repr=False)
    _asym: float = field(default=0.0, init=False, repr=False)
    _dn_baseline: float = field(default=0.0, init=False, repr=False)
    _dn_smooth: float = field(default=0.0, init=False, repr=False)
    _startle: float = field(default=0.0, init=False, repr=False)
    _startle_ticks: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.behaviour not in ("attract", "avoid"):
            raise ValueError("behaviour must be 'attract' or 'avoid'")
        self.eye_left, self.eye_right = self.conn.sided(
            **C.SENSORY_GROUPS["photoreceptors"]
        )
        self.dn_left, self.dn_right = self.conn.sided(super_class="descending")
        # In a BRAIN connectome every motor neuron is a feeding motor neuron: the ones
        # that move legs live in the ventral nerve cord, which is a different dataset.
        # So this population is the proboscis musculature, and its firing IS the
        # feeding response. Measured, 500 ms at 150 Hz:
        #
        #   sugar GRNs  -> 30 of 110 motor neurons fire, 1,610 Hz, including MN10
        #   bitter GRNs ->  0 of 110 motor neurons fire, 0 Hz
        #
        # That dissociation is not built in anywhere. It is what the wiring does.
        self.mech_left, self.mech_right = self.conn.sided(
            **C.SENSORY_GROUPS["mechanosensory"]
        )
        self.motor = self.conn.indices(super_class="motor")
        self.sugar = self.conn.indices(cell_sub_class="sugar/water")
        self.bitter = self.conn.indices(cell_sub_class="bitter")
        # The classic named cells, tracked alongside the population purely so the
        # report can show them failing to lateralise while the population succeeds.
        self.named = {}
        for cell_type in C.NAMED_DESCENDING:
            left, right = self.conn.sided(cell_type=cell_type)
            if len(left) and len(right):
                self.named[cell_type] = {"left": left, "right": right}
        # The looming-escape ensemble, read alongside the population for the swat.
        self.escape = {
            label: self.conn.indices(cell_type=types)
            for label, types in C.ESCAPE_TYPES.items()
        }

    # ── Outbound: world -> brain ────────────────────────────────────────────────

    def stim_groups(self) -> dict[str, np.ndarray]:
        """The neuron sets `brain.Brain` should make stimulable."""
        groups = {"eye_left": self.eye_left, "eye_right": self.eye_right}
        if len(self.mech_left):
            groups["mech_left"] = self.mech_left
        if len(self.mech_right):
            groups["mech_right"] = self.mech_right
        if len(self.sugar):
            groups["sugar"] = self.sugar
        if len(self.bitter):
            groups["bitter"] = self.bitter
        return groups

    def apply_taste(self, brain, food: str, in_contact: bool) -> float:
        """Drive the gustatory receptor neurons when the fly's tarsi are on the food.

        Returns the rate applied, in Hz. Both groups are always driven (at zero when
        not in contact) so that a run never depends on which groups happened to exist.
        """
        rate = TASTE_HZ if in_contact else 0.0
        brain.set_rate("sugar", rate if food == "sugar" else 0.0)
        brain.set_rate("bitter", rate if food == "bitter" else 0.0)
        return rate

    def apply_touch(self, brain, side: str, hz: float = TOUCH_HZ) -> tuple[float, float]:
        """Poke one antenna, or neither. Returns the two drive rates in Hz.

        Both groups are always written, at zero when not being poked, so a run never
        depends on which groups happened to be set last tick.
        """
        left = hz if side == "left" else 0.0
        right = hz if side == "right" else 0.0
        brain.set_rate("mech_left", left)
        brain.set_rate("mech_right", right)
        return left, right

    def apply_vision(self, brain, eyes: np.ndarray, blind: bool = False) -> tuple[float, float]:
        """Push one tick of eye darkening into the brain. Returns the two rates in Hz.

        `blind` holds both eyes at tonic, so the brain is still running and still
        descending: the control isolates "the eyes are steering it" from "the brain
        being on at all is steering it".
        """
        if blind:
            rates = (self.tonic_hz, self.tonic_hz)
        else:
            rates = (
                min(self.tonic_hz + float(eyes[0]) * self.eye_gain, MAX_EYE_HZ),
                min(self.tonic_hz + float(eyes[1]) * self.eye_gain, MAX_EYE_HZ),
            )
        brain.set_rate("eye_left", rates[0])
        brain.set_rate("eye_right", rates[1])
        return rates

    # ── Inbound: brain -> legs ──────────────────────────────────────────────────

    def read_descending(self, tick) -> dict[str, float]:
        """Population and named-cell rates for one tick, in Hz."""
        out = {
            "dn_left": tick.rate(self.dn_left),
            "dn_right": tick.rate(self.dn_right),
            "motor": tick.rate(self.motor),
        }
        for cell_type, sides in self.named.items():
            out[f"{cell_type}_left"] = tick.rate(sides["left"])
            out[f"{cell_type}_right"] = tick.rate(sides["right"])
        return out

    def read_escape(self, tick) -> dict[str, float]:
        """Population rates of the named looming-escape cells, in Hz."""
        return {f"escape:{label}": tick.rate(idx) for label, idx in self.escape.items()}

    def update_startle(self, readout: dict[str, float]) -> float:
        """Track the COMMON-MODE descending rate and return how far above baseline it is.

        Fixation reads the DIFFERENCE between the two descending populations; a looming
        object is symmetric and moves both together, so it is invisible to that
        difference and shows up only in the sum. This is that second channel.

        TWO time constants, and the demo does not work with one. Measured on a 60-tick
        approach, the raw summed rate jumps between 133 and 600 Hz from tick to tick
        with nothing in the scene changing: a 15 ms window at these rates holds less
        signal than noise, exactly as the steering imbalance does. A single-EMA detector
        built on that fires at 0.91 during the quiet baseline, before the object has
        moved at all.

        So the sum is smoothed fast (`startle_fast` 0.55, about a 2-tick memory) to make
        a usable signal, and compared against a slow baseline (`startle_tau` 0.97, about
        a 500 ms memory) built from that smoothed value. The fast one tracks the
        approach; the slow one does not have time to absorb it.

        The first `startle_warmup` ticks are a warm-up and report zero. Without it the
        baseline is seeded from tick 1, which at these rates is as likely to be the
        lowest value of the run as any other, and every later tick then looks like a
        doubling: measured, a detector with no warm-up reported a full startle of 1.00
        during the quiet ticks before the object had moved at all.

        Returns the relative excess, clipped to [0, 1]: 0 means the sum is at or below
        its own recent average, 1 means it has doubled.
        """
        total = readout["dn_left"] + readout["dn_right"]
        self._startle_ticks += 1
        if self._startle_ticks <= self.startle_warmup:
            # Running mean while warming up, so the baseline starts from the quiet
            # period's average rather than from one noisy sample.
            k = self._startle_ticks
            self._dn_baseline += (total - self._dn_baseline) / k
            self._dn_smooth = self._dn_baseline
            self._startle = 0.0
            return 0.0
        self._dn_smooth = (
            (1 - self.startle_fast) * self._dn_smooth + self.startle_fast * total
        )
        excess = (self._dn_smooth - self._dn_baseline) / max(self._dn_baseline, 1e-6)
        self._startle = float(np.clip(excess, 0.0, 1.0))
        self._dn_baseline = (
            self.startle_tau * self._dn_baseline + (1 - self.startle_tau) * self._dn_smooth
        )
        return self._startle

    @property
    def startle(self) -> float:
        """The current common-mode excess, 0..1. See `update_startle`."""
        return self._startle

    def descending_signal(self, readout: dict[str, float]) -> np.ndarray:
        """Turn a descending readout into the controller's (2,) command.

        Returns [left_drive, right_drive]. A fly turns toward the side whose legs push
        LESS, so steering left means lowering the left drive and raising the right.
        """
        left, right = readout["dn_left"], readout["dn_right"]
        total = left + right
        # Subtract the intrinsic bias measured with both eyes equal. Without this the
        # fly turns constantly in one direction with nothing in front of it, which is
        # exactly what the single-cell DNa02 readout does.
        instantaneous = 0.0 if total <= 0 else (right - left) / total - self.calibration.bias
        self._asym = self.smoothing * self._asym + (1 - self.smoothing) * instantaneous

        toward = 1.0 if self.behaviour == "attract" else -1.0
        # turn_sign converts "positive imbalance" into "the object is on my left",
        # as measured by calibrate(); toward decides whether we approach or flee.
        steer = self.turn_gain * self._asym * self.calibration.turn_sign * toward

        # A feeding fly stops walking. The DECISION to feed is the connectome's (the
        # proboscis motor neurons fire, or they do not), and this line is the one place
        # that turns it into locomotion: walking drive is scaled down in proportion to
        # how hard those motor neurons are firing. Sugar therefore halts the fly and
        # bitter does not, without either outcome being written anywhere.
        walk = self.base_drive * (1.0 - self.feeding(readout))
        # A startled fly stops. `startle_gain` is the one hand-made number in the swat,
        # in the same spirit as `behaviour`: the connectome supplies the common-mode
        # signal and its timing, and this line decides what the legs do about it. At 0,
        # the default, the swat is a pure measurement and the fly keeps walking.
        if self.startle_gain > 0:
            walk *= max(0.0, 1.0 - self.startle_gain * self._startle)
        lo, hi = self.drive_limits
        return np.clip(np.array([walk - steer, walk + steer]), 0.0, hi)

    def feeding(self, readout: dict[str, float]) -> float:
        """How hard the feeding motor neurons are firing, on a 0..1 scale."""
        return float(np.clip(readout.get("motor", 0.0) / FEEDING_FULL_HZ, 0.0, 1.0))

    @property
    def imbalance(self) -> float:
        """The smoothed descending imbalance currently steering the fly."""
        return self._asym


def constant_signal(base_drive: float = 1.0) -> np.ndarray:
    """The no-brain control: walk forward, steer never."""
    return np.array([base_drive, base_drive])
