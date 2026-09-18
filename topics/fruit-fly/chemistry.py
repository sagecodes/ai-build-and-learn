"""What the brain is releasing, tick by tick.

Every other readout in this repo counts spikes. This one asks what those spikes actually
deliver, which is a different question and a more chemical one: a spike from a GABA
neuron and a spike from a dopaminergic neuron are the same event in a raster and very
different events in a head.

Two things are tracked, and both come straight out of the connectome with no extra
assumptions:

  1. **Release per transmitter.** Every neuron in the FlyWire annotation carries a
     predicted transmitter, and every neuron has an outgoing synapse budget: the total
     number of synapses it makes onto everything downstream. A spike therefore delivers
     that many synaptic events of that one transmitter. Summing over the cells that
     spiked in a tick gives release per transmitter per tick.

  2. **Excitation against inhibition.** Dale's law holds exactly in this dataset (checked:
     0 of 138,005 presynaptic neurons have a mixed outgoing sign), so every neuron is
     purely excitatory or purely inhibitory and the E/I balance is exact rather than
     estimated.

Measured budgets, the whole brain:

    transmitter      neurons   total outgoing synapses   synapses per spike
    acetylcholine     86,025            30,460,673              354
    GABA              19,147            12,600,845              658
    glutamate         24,858             9,471,876              381
    dopamine           5,905             1,384,531              234
    serotonin          2,201               436,206              198
    octopamine           210               137,839              656

Two things worth noticing before reading any chart made from this. GABA cells deliver
nearly twice as many synapses per spike as cholinergic ones, so an inhibitory spike is
worth about two excitatory spikes and a raster that looks balanced is not. And the 210
octopaminergic neurons are individually the second-biggest broadcasters in the brain,
which is what a modulatory system is supposed to look like.

── The honest caveat ───────────────────────────────────────────────────────────
This is release, not effect. The model collapses all six transmitters onto a `+1/-1`
sign, so dopamine in the simulation is a fast excitatory synapse and nothing more: the
dopamine trace in a chart is a real count of what a real fly would be releasing, next to
a simulation that does not implement what that release DOES. Section 2b of the README is
about exactly this gap, and the `teach` task runs into it head on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from connectome import Connectome

log = logging.getLogger(__name__)

# The six transmitters the FlyWire classifier predicts, in descending order of how much
# of the brain releases them. "unknown" catches the 279 neurons with no prediction.
TRANSMITTERS = (
    "acetylcholine", "glutamate", "gaba", "dopamine", "serotonin", "octopamine",
)

# Which are inhibitory in this model. Glutamate is inhibitory in the fly, which is the
# single most consequential line in this file: it is why L1 shuts down the medulla and
# why the motion pathway is dead. See README section 6b.
INHIBITORY = ("gaba", "glutamate")


@dataclass
class Neurochemistry:
    """Per-tick transmitter release, precomputed once from the wiring.

    Args:
        conn: The connectome. Outgoing synapse budgets are summed from its edge list.
        normalise: Divide every trace by 1,000 so charts are in thousands of synaptic
            events rather than raw counts, which run to millions.
    """

    conn: Connectome
    normalise: float = 1e3

    budget: np.ndarray = field(init=False, repr=False)
    masks: dict[str, np.ndarray] = field(init=False, repr=False)
    excitatory: np.ndarray = field(init=False, repr=False)
    inhibitory: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        n = self.conn.n
        # Outgoing synapses per neuron, split by sign. Dale's law means exactly one of
        # these is non-zero for any given cell.
        self.excitatory = np.zeros(n)
        self.inhibitory = np.zeros(n)
        np.add.at(self.excitatory, self.conn.pre, np.clip(self.conn.weight, 0, None))
        np.add.at(self.inhibitory, self.conn.pre, -np.clip(self.conn.weight, None, 0))
        self.budget = self.excitatory + self.inhibitory

        ann = self.conn.ann
        labels = ann["top_nt"].fillna("unknown").to_numpy()
        bi = ann["bi"].to_numpy(np.int64)
        self.masks = {}
        for name in TRANSMITTERS:
            weights = np.zeros(n)
            hits = bi[labels == name]
            weights[hits] = self.budget[hits]
            self.masks[name] = weights
            log.info(
                "%-14s %6d neurons, %12.0f outgoing synapses",
                name, len(hits), weights.sum(),
            )

    def release(self, counts: np.ndarray) -> dict[str, float]:
        """Synaptic events released this tick, per transmitter, in thousands."""
        counts = np.asarray(counts, float)
        return {
            name: float(counts @ weights / self.normalise)
            for name, weights in self.masks.items()
        }

    def balance(self, counts: np.ndarray) -> dict[str, float]:
        """Excitatory and inhibitory events this tick, and the ratio between them.

        `ei_ratio` is excitation over inhibition. Above 1 the brain is net driving
        itself, below 1 it is net damping itself. It is not a stable quantity in this
        model: watch it during the ignition the `gain` task produces.
        """
        counts = np.asarray(counts, float)
        exc = float(counts @ self.excitatory / self.normalise)
        inh = float(counts @ self.inhibitory / self.normalise)
        return {
            "excitatory": exc,
            "inhibitory": inh,
            "ei_ratio": exc / inh if inh > 0 else float("nan"),
        }

    def trace_keys(self) -> list[str]:
        """The trace names this adds to a run, for pre-allocating the dict."""
        return [f"nt:{name}" for name in TRANSMITTERS] + [
            "nt:excitatory", "nt:inhibitory", "nt:ei_ratio"
        ]

    def sample(self, counts: np.ndarray) -> dict[str, float]:
        """Everything, keyed to match `trace_keys`."""
        out = {f"nt:{k}": v for k, v in self.release(counts).items()}
        out.update({f"nt:{k}": v for k, v in self.balance(counts).items()})
        return out
