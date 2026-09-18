"""Teaching the fly: associative learning in the mushroom body.

Every other experiment in this repo runs a fixed brain. This one changes it, and it is
the only place where a synapse is not the number an electron microscope measured.

The mushroom body is the fly's learning centre and the whole apparatus is present in
FlyWire: 5,177 Kenyon cells carrying a sparse odour code, 96 mushroom body output
neurons (MBONs) reading it out over 62,261 connections, 2 APL neurons providing feedback
inhibition, and the dopaminergic teaching signal split the way the literature describes
it, 16 PPL1 cells for punishment onto 85 MBONs and 307 PAM cells for reward onto 68.

The learning rule is the one the animal uses and it is not Hebbian. Dopamine arriving at
a mushroom body compartment while a Kenyon cell is active DEPRESSES that cell's synapse
onto the MBON in that compartment (Hige et al. 2015, Owald & Waddell 2015). The odour
stops driving that output channel, and behaviour follows the MBONs that are left. So
`train()` is a three-factor rule, and two of the three factors are read from the
connectome rather than chosen here: which Kenyon cells the odour activates, and which
MBONs the driven dopaminergic neurons actually reach.

── What works, what does not, and exactly where it breaks ──────────────────────
The machinery works. Dopamine-gated depression drops the trained odour's MBON output by
48.7%, and the same procedure on shuffled wiring changes nothing (1 Kenyon cell active,
0 edges depressed, 0.0%). So the circuit is real and the rule bites.

What fails is SPECIFICITY, and it fails for a reason that can be pointed at. Learning in
a mushroom body is odour-specific because the Kenyon cell code is sparse and different
odours activate different cells. Measured here, three odour channels at 20 Hz, counting
cells active at each stage and the overlap between the odours that fire at all:

    stage                 active per odour        overlap (Jaccard)
    ORN (receptors)       463 / 411 /  97              0.21
    ALPN (projection)     543 / 545 /   5              0.34
    Kenyon cells        1,503 / 1,479 /  0              0.95
    MBON                   43 /   39 /  0              0.86

Odour identity is present at the receptor layer and essentially gone one synapse later:
a single glomerulus drives 543 of the 685 projection neurons, which is most of the
antennal lobe. In a living fly, one glomerulus drives its own handful of projection
neurons and lateral inhibition keeps the rest quiet. Here there is no such gain control,
so every odour becomes the same generic "an odour is present" signal, every odour
recruits the same ~1,500 Kenyon cells, and depressing "the cells active for odour A"
depresses the cells for odour B as well.

Four fixes were tried and none of them work, which is worth recording so nobody spends
the afternoon again:

  * Raising the Kenyon cell threshold. Takes the code from 65% of cells to 0.4% and the
    overlap only falls from 0.99 to 0.88. The same cells keep winning, because they are
    the ones with the most input regardless of the odour.
  * Deleting all 293,762 Kenyon-to-Kenyon edges. Overlap 0.956, no better than intact.
  * Lowering the drive. There is a cliff between 20 and 25 Hz where the antennal lobe
    ignites, and below it odours do not stop overlapping, they stop firing at all: at
    20 Hz, ORN_VA1d and ORN_DL3 produce zero Kenyon cells while ORN_DA1 and ORN_VM4
    produce 1,503 and 1,479 that overlap at 0.95. The network is bistable, not graded.
  * Both together. Same answer.

A word of warning about the statistic, because it produced a wrong result here first: a
mean Jaccard over all odour pairs reads 0.16 at 20 Hz and looks like beautiful
decorrelation. It is an artifact of the two silent channels, whose empty sets score 0
against everything. Only compare odours that actually fire.

── The classic control cannot be run in this model ─────────────────────────────
An unpaired control, dopamine delivered with no odour, ought to produce no learning.
Here it produces 46.6%, almost as much as the paired case, and the reason is measured:
PPL1 reaches the Kenyon cells over 10,720 edges and dopamine in this model is a
`+1` sign like any other transmitter, so the teaching signal is itself an excitatory
drive that activates 1,474 Kenyon cells on its own. This is the flattening of
neuromodulation described in the README, showing up as a broken experiment.

So the control used here is DIFFERENTIAL conditioning instead: train on one odour, test
both, and compare. That one works, and it is the standard assay in the literature.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from connectome import Connectome

log = logging.getLogger(__name__)

# Measured above: the antennal lobe ignites between 20 and 25 Hz and every odour
# collapses onto the same Kenyon cell pattern. This is the one number the whole demo
# stands on.
ODOR_HZ = 20.0
# Dopaminergic teaching drive. Well above the odour, because a punishment should not be
# ambiguous, and the DANs are few.
TEACH_HZ = 150.0
# Kenyon cells are famously hard to fire. -38 mV against a -52 mV rest is 14 mV of
# headroom instead of the brain-wide 7, and takes the code from 32% of cells to 14%.
KC_THRESHOLD = -38e-3
PRESENT_S = 0.3

# Four odour channels, each a named olfactory receptor neuron type. These are real
# glomeruli: DA1 is the pheromone channel, VA1d and VA1v are courtship-related, DL3 and
# VM4 are general odorants.
ODOR_CHANNELS = ("ORN_DA1", "ORN_VA1d", "ORN_DL3", "ORN_VM4")

# A stage counts an odour channel as "firing" only if it activates at least this
# fraction of the stage's cells. See `specificity` for why non-emptiness is not enough.
MIN_ACTIVE_FRACTION = 0.02


@dataclass
class MushroomBody:
    """Every index the learning experiment needs, resolved once."""

    kc: np.ndarray
    mbon: np.ndarray
    apl: np.ndarray
    ppl1: np.ndarray
    pam: np.ndarray
    descending: np.ndarray
    odors: dict[str, np.ndarray] = field(default_factory=dict)

    @classmethod
    def build(cls, conn: Connectome, channels=ODOR_CHANNELS) -> "MushroomBody":
        ann = conn.ann
        types = ann["cell_type"].fillna("")

        def by_pattern(pattern: str) -> np.ndarray:
            return np.unique(ann[types.str.match(pattern, na=False)]["bi"].to_numpy())

        mb = cls(
            kc=conn.indices(cell_class="Kenyon_Cell"),
            mbon=by_pattern(r"^MBON"),
            apl=np.unique(ann[types.str.fullmatch("APL", na=False)]["bi"].to_numpy()),
            ppl1=by_pattern(r"^PPL1"),
            pam=by_pattern(r"^PAM"),
            descending=conn.indices(super_class="descending"),
            odors={c: conn.indices(cell_type=c) for c in channels},
        )
        log.info(
            "mushroom body: %d Kenyon cells, %d MBONs, %d APL, %d PPL1, %d PAM, "
            "odour channels %s",
            len(mb.kc), len(mb.mbon), len(mb.apl), len(mb.ppl1), len(mb.pam),
            {k: len(v) for k, v in mb.odors.items()},
        )
        return mb

    def teacher(self, valence: str) -> np.ndarray:
        """The dopaminergic population for a punishment or a reward."""
        if valence == "punish":
            return self.ppl1
        if valence == "reward":
            return self.pam
        raise ValueError("valence must be 'punish' or 'reward'")


def present(
    conn: Connectome,
    mb: MushroomBody,
    odor: str,
    teach: np.ndarray | None = None,
    odor_hz: float = ODOR_HZ,
    teach_hz: float = TEACH_HZ,
    duration: float = PRESENT_S,
    kc_threshold: float = KC_THRESHOLD,
    seed: int = 0,
) -> np.ndarray:
    """Present one odour, optionally paired with dopamine. Returns per-neuron Hz.

    A fresh network every call, deliberately: no membrane potential may leak from the
    training trial into the test trial, or the "memory" being measured is partly just
    the brain still being warm.
    """
    import brain as BRAIN

    groups = {"odor": mb.odors[odor]}
    if teach is not None and len(teach):
        groups["teach"] = teach
    network = BRAIN.Brain(
        conn=conn,
        stim_groups=groups,
        seed=seed,
        thresholds={"kenyon": (mb.kc, kc_threshold)},
    )
    network.set_rate("odor", odor_hz)
    if "teach" in groups:
        network.set_rate("teach", teach_hz)
    network.run(duration)
    return network.rates(duration)


def train(
    conn: Connectome,
    mb: MushroomBody,
    odor: str,
    valence: str = "punish",
    learning_rate: float = 0.9,
    paired: bool = True,
    **kwargs,
) -> tuple[Connectome, dict]:
    """One conditioning trial. Returns the modified wiring and what it changed.

    The three-factor rule, with two factors measured:

      1. WHICH KENYON CELLS. Read from the brain: present the odour and record which of
         the 5,177 cells actually spiked. Not chosen, not anatomical, measured per trial.
      2. WHICH MBONS. Read from the connectome: the output neurons that the driven
         dopaminergic population actually synapses onto. PPL1 reaches 85 of the 96 MBONs,
         PAM reaches 68.
      3. COINCIDENCE. Only synapses that satisfy both are changed.

    `paired=False` is the control every conditioning experiment needs: the dopamine is
    delivered with NO odour, so the Kenyon cells that were active during the teaching
    signal are whatever fires without an odour, which in this silent model is almost
    nothing. An animal that learns from that is not learning, it is drifting.
    """
    teach = mb.teacher(valence)
    rates = present(conn, mb, odor, teach=teach if paired else None, **kwargs) if paired \
        else _teach_only(conn, mb, teach, **kwargs)
    active = mb.kc[rates[mb.kc] > 1]

    # Which MBONs the dopamine actually reaches, straight out of the wiring.
    taught = np.unique(conn.post[np.isin(conn.pre, teach) & np.isin(conn.post, mb.mbon)])

    edges = np.isin(conn.pre, active) & np.isin(conn.post, taught)
    weight = conn.weight.copy()
    weight[edges] *= (1.0 - learning_rate)
    report = {
        "odor": odor,
        "valence": valence,
        "paired": bool(paired),
        "active_kc": int(len(active)),
        "kc_fraction": float(len(active) / max(len(mb.kc), 1)),
        "taught_mbons": int(len(taught)),
        "edges_depressed": int(edges.sum()),
        "synapses_depressed": float(np.abs(conn.weight[edges]).sum()),
        "learning_rate": learning_rate,
    }
    log.info(
        "train %s (%s, %s): %d Kenyon cells active, %d edges onto %d MBONs depressed by %.0f%%",
        odor, valence, "paired" if paired else "UNPAIRED", report["active_kc"],
        report["edges_depressed"], report["taught_mbons"], 100 * learning_rate,
    )
    return (
        Connectome(n=conn.n, pre=conn.pre, post=conn.post, weight=weight, ann=conn.ann),
        report,
    )


def _teach_only(conn, mb, teach, odor_hz=ODOR_HZ, teach_hz=TEACH_HZ,
                duration=PRESENT_S, kc_threshold=KC_THRESHOLD, seed=0) -> np.ndarray:
    """Dopamine with no odour, for the unpaired control."""
    import brain as BRAIN

    network = BRAIN.Brain(
        conn=conn, stim_groups={"teach": teach}, seed=seed,
        thresholds={"kenyon": (mb.kc, kc_threshold)},
    )
    network.set_rate("teach", teach_hz)
    network.run(duration)
    return network.rates(duration)


def present_frames(
    conn: Connectome,
    mb: MushroomBody,
    odor: str,
    ticks: int = 60,
    tick_s: float = 0.015,
    odor_hz: float = ODOR_HZ,
    kc_threshold: float = KC_THRESHOLD,
    seed: int = 0,
) -> tuple[list[np.ndarray], list[float]]:
    """Present an odour one tick at a time, so the result can be watched rather than read.

    Same stimulus as `present`, stepped instead of run in one go, returning the per-neuron
    spike counts for every tick and the mushroom body output rate alongside. That is what
    the video is made of: run this before training and again after, and the second clip is
    the first one with the mushroom body gone dark.
    """
    import brain as BRAIN

    network = BRAIN.Brain(
        conn=conn, stim_groups={"odor": mb.odors[odor]}, seed=seed,
        thresholds={"kenyon": (mb.kc, kc_threshold)},
    )
    network.set_rate("odor", odor_hz)
    counts, mbon_hz = [], []
    for _ in range(ticks):
        tick = network.run(tick_s)
        counts.append(tick.counts)
        mbon_hz.append(tick.rate(mb.mbon))
    return counts, mbon_hz


def read_out(rates: np.ndarray, mb: MushroomBody) -> dict[str, float]:
    """What the mushroom body is telling the rest of the brain."""
    return {
        "mbon_hz": float(rates[mb.mbon].sum()),
        "mbon_active": int((rates[mb.mbon] > 1).sum()),
        "kc_hz": float(rates[mb.kc].sum()),
        "kc_active": int((rates[mb.kc] > 1).sum()),
        "descending_hz": float(rates[mb.descending].sum()),
        "descending_active": int((rates[mb.descending] > 1).sum()),
    }


def conditioning(
    conn: Connectome,
    mb: MushroomBody,
    trained: str,
    control: str,
    valence: str = "punish",
    learning_rate: float = 0.9,
    paired: bool = True,
    seed: int = 0,
) -> dict:
    """The whole experiment: test both odours, train on one, test both again.

    The comparison that matters is not "did the trained odour's response drop". A drop
    that happens to every odour is not a memory, it is fatigue or a broken network. The
    result is the DIFFERENCE between what happened to the trained odour and what
    happened to a second odour the fly was never punished for.
    """
    before = {o: read_out(present(conn, mb, o, seed=seed), mb) for o in (trained, control)}
    after_conn, report = train(
        conn, mb, trained, valence=valence, learning_rate=learning_rate,
        paired=paired, seed=seed,
    )
    after = {o: read_out(present(after_conn, mb, o, seed=seed), mb) for o in (trained, control)}

    def drop(odor: str) -> float:
        was, now = before[odor]["mbon_hz"], after[odor]["mbon_hz"]
        return 0.0 if was <= 0 else 100.0 * (was - now) / was

    result = {
        "trained_odor": trained,
        "control_odor": control,
        "training": report,
        "before": before,
        "after": after,
        "trained_drop_pct": drop(trained),
        "control_drop_pct": drop(control),
        "specificity_pct": drop(trained) - drop(control),
    }
    log.info(
        "conditioning %s vs %s: MBON output fell %.1f%% for the trained odour and "
        "%.1f%% for the control, specificity %+.1f points",
        trained, control, result["trained_drop_pct"], result["control_drop_pct"],
        result["specificity_pct"],
    )
    return result


# The olfactory pathway, in order, for the specificity autopsy. Same idea as
# `connectome.VISUAL_PATHWAY`: walking down it and counting says WHERE a property is
# lost rather than only that it is missing at the end.
OLFACTORY_PATHWAY: list[tuple[str, dict[str, str]]] = [
    ("ORN (olfactory receptors)", {"super_class": "sensory", "cell_class": "olfactory"}),
    ("ALLN (lateral, inhibitory)", {"cell_class": "ALLN"}),
    ("ALPN (projection neurons)", {"cell_class": "ALPN"}),
    ("Kenyon cells", {"cell_class": "Kenyon_Cell"}),
    ("MBON (mushroom body output)", {}),        # resolved by name, see specificity()
]


def specificity(
    conn: Connectome,
    mb: MushroomBody,
    odor_hz: float = ODOR_HZ,
    kc_threshold: float = KC_THRESHOLD,
    seed: int = 0,
) -> list[dict]:
    """Walk the olfactory pathway and ask, at each stage, whether odours are distinct.

    The whole learning demo turns on this. Returns one record per stage with how many
    cells each odour activates and the mean overlap between the odours that fire at all.

    Overlap is computed ONLY over odour channels that meaningfully drive the stage. A
    mean that includes a silent channel is flattering nonsense: an empty set scores a
    Jaccard of 0 against everything, which reads as perfect decorrelation and is really
    a dead channel.

    "Meaningfully" needs a threshold and not just non-emptiness, which is the subtler
    version of the same trap and it bit this function first. At the projection neurons,
    two of the four odour channels drive 543 and 545 cells and the other two drive 5 and
    4. Counting all four as firing reports an overlap of 0.171, which looks like the
    antennal lobe preserving odour identity beautifully. Counting only the two that are
    actually driving the stage reports what is really happening. A channel therefore has
    to activate at least `MIN_ACTIVE_FRACTION` of the stage, or 5 cells, whichever is
    larger.
    """
    stages = []
    for label, criteria in OLFACTORY_PATHWAY:
        idx = mb.mbon if not criteria else conn.indices(**criteria)
        stages.append((label, idx))

    active: dict[str, dict[str, set]] = {label: {} for label, _ in stages}
    for odor in mb.odors:
        rates = present(conn, mb, odor, odor_hz=odor_hz,
                        kc_threshold=kc_threshold, seed=seed)
        for label, idx in stages:
            active[label][odor] = set(idx[rates[idx] > 1].tolist())

    out = []
    for label, idx in stages:
        per_odor = active[label]
        floor = max(5, int(MIN_ACTIVE_FRACTION * len(idx)))
        firing = {k: v for k, v in per_odor.items() if len(v) >= floor}
        names = list(firing)
        overlaps = [
            len(firing[names[i]] & firing[names[j]]) / len(firing[names[i]] | firing[names[j]])
            for i in range(len(names))
            for j in range(i + 1, len(names))
        ]
        out.append({
            "stage": label,
            "cells": int(len(idx)),
            "floor": int(floor),
            "active": {k: len(v) for k, v in per_odor.items()},
            "odors_firing": len(firing),
            "jaccard": float(np.mean(overlaps)) if overlaps else float("nan"),
        })
        log.info(
            "%-28s active %s  |  %d/%d odours fire, overlap %.3f",
            label, list(out[-1]["active"].values()), len(firing), len(per_odor),
            out[-1]["jaccard"],
        )
    return out


def code_sparseness(conn: Connectome, mb: MushroomBody, odor_hz: float,
                    kc_threshold: float = KC_THRESHOLD, seed: int = 0) -> dict:
    """How sparse and how distinct the Kenyon cell code is at a given odour drive.

    This is the measurement the whole demo rests on, so it ships as a task rather than
    as a comment. Returns mean fraction of cells active and mean pairwise overlap.
    """
    sets = {}
    for odor in mb.odors:
        rates = present(conn, mb, odor, odor_hz=odor_hz, kc_threshold=kc_threshold, seed=seed)
        sets[odor] = set(mb.kc[rates[mb.kc] > 1].tolist())
    names = list(sets)
    overlaps = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = sets[names[i]], sets[names[j]]
            overlaps.append(len(a & b) / max(len(a | b), 1))
    active = [len(s) for s in sets.values()]
    return {
        "odor_hz": odor_hz,
        "kc_active_mean": float(np.mean(active)),
        "kc_fraction": float(np.mean(active) / max(len(mb.kc), 1)),
        "jaccard_mean": float(np.mean(overlaps)) if overlaps else 0.0,
        "per_odor": {k: len(v) for k, v in sets.items()},
    }
