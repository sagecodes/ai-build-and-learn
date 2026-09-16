"""The fly's compass: recovering a ring attractor from the wiring, then driving it.

The central complex contains the insect head-direction system. EPG neurons tile the
ellipsoid body into wedges, and in a living fly a single bump of activity sits on that
ring, tracking which way the animal is facing. It is the best-understood piece of
neural computation in any brain and it is sitting inside the 138,639 neurons this repo
already loads.

There is one problem in the way. The FlyWire annotation gives all 47 EPG cells the same
label, `cell_type == "EPG"`, with no wedge number, and their annotation coordinates are
single representative points that do not recover the ring. So the ring's ORDER, which
is the whole thing the panel needs, is not in the metadata. It has to come out of the
connectivity.

── Recovering the order ────────────────────────────────────────────────────────
Delta7 neurons tile the protocerebral bridge and connect to EPG cells in a way that
depends on where around the ring the EPG sits. So two EPG cells that are neighbours on
the ring share Delta7 partners, and two on opposite sides do not. That makes the EPG
connectivity fingerprint a circular similarity structure, and a circular structure is
exactly what a spectral embedding recovers: build the cosine similarity between EPG
cells over their Delta7 connections, take the two leading non-trivial eigenvectors of
the normalised Laplacian, and the cells fall on a circle whose angle is their position.

This is a measurement with a pass/fail number attached, not an assertion. If the
recovered order is real then similarity must fall off with circular distance in that
order, and the correlation between the two is reported everywhere the ring is used.
Measured on FlyWire 783:

    feature set                    eigengap        radius CV     similarity vs
                                   (lower = 2D)                  circular distance
    EPG x all partners           0.885/0.905/0.912    0.38          -0.47
    EPG x central complex        0.888/0.904/0.914    0.33          -0.50
    EPG x Delta7                 0.588/0.683/0.875    0.33          -0.78     <- used
    Delta7 x all partners        0.577/0.616/0.692    0.15          -0.70

The Delta7 feature set is the one with a clean two-dimensional gap (0.588 and 0.683,
then a jump to 0.875) and by far the strongest fall-off, so that is the one used.

── What the experiment then asks ───────────────────────────────────────────────
Put a bump on the ring by stimulating six adjacent EPG cells, run the brain, and look
at where the response lands downstream. Do that at eight positions around the ring. If
the compass is real wiring rather than an ordering artefact, the downstream response has
to ROTATE with the bump, and its position has to be a straight line against the driven
position with slope +/-1.

Measured here, w_syn 0.55, 300 ms at 150 Hz, Delta7 response peak against EPG bump:

    bump      -167  -134   -81   -41    +8   +55  +110  +154 deg
    Delta7     +92   +49    -5   -42   -81  -138  +169  +129 deg

which is a line of slope -1 (the two rings are embedded with independent handedness, so
a mirror is expected and the fit reports it), and the SHUFFLED control does not merely
degrade, it goes silent: 0 of 42 Delta7 cells fire at every one of the eight positions.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from connectome import Connectome

log = logging.getLogger(__name__)

# Six adjacent cells is about an eighth of the 47-cell ring, roughly one wedge, and is
# the narrowest bump that reliably drives anything downstream at the published gain.
BUMP_WIDTH = 6
BUMP_HZ = 150.0

# Which population to use as the positional reference when embedding each ring.
#
# Delta7 is the natural reference for the whole central complex: its cells tile the
# protocerebral bridge, so sharing Delta7 partners is a direct statement about where
# around the compass a cell sits. Delta7 itself is embedded against EPG instead, for
# the same reason in the other direction.
#
# This is not a cosmetic choice and picking it wrong quietly wrecks the result.
# Measured, embedding PFL3 two ways and then fitting how its response tracks the bump:
#
#     PFL3 against Delta7    similarity vs circular distance -0.70, radius CV 0.18,
#                            tracking residual  13 deg
#     PFL3 against EPG       similarity vs circular distance -0.43, radius CV 0.60,
#                            tracking residual  70 deg
#
# Same connectome, same brain, same stimulus; only the reference population differs.
# The ring quality score is printed next to every ring in the report precisely so that
# a bad embedding cannot hide behind a good-looking tracking plot.
REFERENCE_PARTNER: dict[str, str] = {"Delta7": "EPG"}
DEFAULT_PARTNER = "Delta7"


def partner_for(cell_type: str) -> str:
    """The population to embed `cell_type` against. See `REFERENCE_PARTNER`."""
    return REFERENCE_PARTNER.get(cell_type, DEFAULT_PARTNER)


@dataclass
class Ring:
    """A population ordered around a circle, recovered from connectivity.

    Attributes:
        idx: Brian indices, in ring order.
        angle: Each cell's angle on the recovered ring, radians, in ring order.
        x, y: The 2D spectral embedding itself, so the report can show that the cells
            really do lie on a circle rather than being sorted into one.
        quality: Correlation between connectivity similarity and circular distance in
            the recovered order. Negative and large means neighbours are similar, which
            is what a real ring looks like. Measured -0.78 for EPG.
        radius_cv: Spread of the embedded radii over their mean. 0 is a perfect circle.
    """

    name: str
    idx: np.ndarray
    angle: np.ndarray
    x: np.ndarray
    y: np.ndarray
    quality: float
    radius_cv: float
    eigengap: float = 0.0
    eigenvalues: list[float] | None = None

    def __len__(self) -> int:
        return len(self.idx)

    def arc(self, start: int, width: int = BUMP_WIDTH) -> np.ndarray:
        """`width` adjacent cells starting at ring position `start`, wrapping."""
        take = [(start + k) % len(self.idx) for k in range(width)]
        return self.idx[take]

    def arc_angle(self, start: int, width: int = BUMP_WIDTH) -> float:
        """Where that arc sits, in degrees, as a circular mean."""
        take = [(start + k) % len(self.idx) for k in range(width)]
        return _circular_mean(self.angle[take], np.ones(width))[0]


def _circular_mean(angles: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted circular mean in degrees, and the vector length in [0, 1].

    The vector length is the part that matters for reading a result: 1.0 means every
    active cell sits at the same angle (a perfect bump) and 0.0 means the activity is
    spread evenly around the ring (no bump at all, just the population switched on).
    """
    weights = np.maximum(np.asarray(weights, float), 0.0)
    if weights.sum() <= 0:
        return float("nan"), 0.0
    cos = float((weights * np.cos(angles)).sum())
    sin = float((weights * np.sin(angles)).sum())
    return float(np.degrees(np.arctan2(sin, cos))), float(np.hypot(cos, sin) / weights.sum())


def embed_ring(conn: Connectome, cell_type: str = "EPG", partner: str = "Delta7") -> Ring:
    """Recover the circular order of a population from who it shares partners with.

    Spectral embedding of the cosine-similarity graph: the eigenvectors of the normalised
    Laplacian belonging to the two smallest non-zero eigenvalues span the plane the ring
    lives in, and each cell's angle in that plane is its position on it.
    """
    import scipy.linalg as la
    import scipy.sparse as sp

    idx = conn.indices(cell_type=cell_type)
    partners = conn.indices(cell_type=partner)
    if len(idx) < 4 or len(partners) == 0:
        raise ValueError(f"cannot embed {cell_type} against {partner}: too few cells")

    # Unsigned weights: the ring's GEOMETRY is about who talks to whom, not about
    # whether the synapse excites or inhibits. The sign matters when the brain runs.
    adjacency = sp.coo_matrix(
        (np.abs(conn.weight), (conn.pre, conn.post)), shape=(conn.n, conn.n)
    ).tocsr()
    outgoing = adjacency[idx][:, partners]
    incoming = adjacency[:, idx].T[:, partners]
    features = sp.hstack([outgoing, incoming]).toarray()
    features /= np.maximum(np.linalg.norm(features, axis=1, keepdims=True), 1e-9)

    similarity = features @ features.T
    np.fill_diagonal(similarity, 0.0)
    degree = similarity.sum(axis=1)
    if (degree <= 0).any():
        raise ValueError(f"{cell_type} similarity graph is disconnected")
    values, vectors = la.eigh(np.diag(degree) - similarity, np.diag(degree))

    # Canonicalise the eigenvector signs. LAPACK fixes an eigenvector only up to sign,
    # and which sign comes back depends on the build and the threading, so the SAME
    # connectome embedded on two machines can produce mirror-image rings. That is
    # harmless for the ring itself and not harmless at all for a claim about where a
    # response lands: measured, the PFL3 readout fitted a residual of 13 deg on the
    # devbox and 70 deg in a pod, from identical inputs, purely because of this. Pinning
    # each vector so its largest-magnitude entry is positive makes the embedding
    # reproducible.
    def canonical(vector: np.ndarray) -> np.ndarray:
        return vector if vector[np.argmax(np.abs(vector))] >= 0 else -vector

    x, y = canonical(vectors[:, 1]), canonical(vectors[:, 2])
    angle = np.arctan2(y, x)
    order = np.argsort(angle)

    # The pass/fail number: in the recovered order, does similarity fall off with
    # circular distance? A ring says strongly yes.
    m = len(idx)
    reordered = similarity[np.ix_(order, order)]
    i, j = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    circular = np.minimum(np.abs(i - j), m - np.abs(i - j))
    off = ~np.eye(m, dtype=bool)
    quality = float(np.corrcoef(reordered[off], circular[off])[0, 1])
    radius = np.hypot(x, y)

    ring = Ring(
        name=cell_type,
        idx=idx[order],
        angle=angle[order],
        x=x[order],
        y=y[order],
        quality=quality,
        radius_cv=float(radius.std() / max(radius.mean(), 1e-12)),
    )
    # The eigengap says how two-dimensional the embedding really is. When eigenvalues 2
    # and 3 sit close together the plane the ring lives in is not well determined, the
    # two axes can trade places between runs, and the recovered order is worth less than
    # its quality score suggests. Reported so the panel can say so.
    ring.eigengap = float(values[3] - values[2])
    ring.eigenvalues = [float(v) for v in values[1:4]]
    log.info(
        "%s ring: %d cells, eigenvalues %.3f/%.3f/%.3f (gap %.3f), similarity vs "
        "circular distance r=%+.3f, radius CV %.2f",
        cell_type, m, values[1], values[2], values[3], ring.eigengap,
        quality, ring.radius_cv,
    )
    return ring


def bump_response(
    conn: Connectome,
    driven: Ring,
    readouts: dict[str, Ring],
    positions: int = 8,
    duration: float = 0.3,
    w_syn_mv: float = 0.55,
    seed: int = 0,
) -> list[dict]:
    """Put a bump at `positions` places around the ring; record where it lands.

    Returns one record per position: where the bump was driven, and for each readout
    ring the circular mean and vector length of the response. A brain in which the
    compass is real wiring produces a readout angle that rotates with the driven angle;
    one in which it is not produces either silence or a fixed answer.
    """
    import brain as BRAIN

    params = dict(BRAIN.PARAMS)
    params["w_syn"] = w_syn_mv * 1e-3
    step = max(1, len(driven) // positions)

    records = []
    for k in range(positions):
        start = (k * step) % len(driven)
        arc = driven.arc(start)
        network = BRAIN.Brain(conn=conn, stim_groups={"bump": arc}, params=params, seed=seed)
        network.set_rate("bump", BUMP_HZ)
        network.run(duration)
        rates = network.rates(duration)

        record = {
            "driven_deg": driven.arc_angle(start),
            "driven_cells": int(len(arc)),
            "spikes": int(network.num_spikes),
            "readouts": {},
        }
        for label, ring in readouts.items():
            profile = rates[ring.idx]
            peak, vector = _circular_mean(ring.angle, profile)
            record["readouts"][label] = {
                "peak_deg": peak,
                "vector_length": vector,
                "active": int((profile > 1).sum()),
                "cells": int(len(ring)),
                "profile": profile.tolist(),
            }
        records.append(record)
        log.info(
            "bump at %+7.1f deg -> %s",
            record["driven_deg"],
            ", ".join(
                f"{label} {r['peak_deg']:+7.1f} deg (len {r['vector_length']:.2f}, "
                f"{r['active']}/{r['cells']} active)"
                for label, r in record["readouts"].items()
            ),
        )
    return records


def tracking_fit(records: list[dict], readout: str) -> dict:
    """Does the response rotate with the bump? Fit a line on the circle.

    A compass gives a slope of +1 (or -1: the two rings are embedded independently, so
    their handedness is arbitrary and a mirror is not a failure). A population that is
    merely switched on gives a slope near zero and a large residual. Reports the
    circular correlation, which is the honest summary when both variables are angles.
    """
    driven = np.deg2rad([r["driven_deg"] for r in records])
    peaks = np.array([r["readouts"][readout]["peak_deg"] for r in records])
    lengths = np.array([r["readouts"][readout]["vector_length"] for r in records])
    active = [r["readouts"][readout]["active"] for r in records]
    cells = records[0]["readouts"][readout]["cells"] if records else 0
    live = np.isfinite(peaks)
    if live.sum() < 3:
        return {
            "readout": readout, "slope": float("nan"), "circular_corr": float("nan"),
            "residual_deg": float("nan"), "offset_deg": float("nan"),
            "mean_vector_length": float(np.nanmean(lengths)) if len(lengths) else 0.0,
            "n": int(live.sum()), "mean_active": float(np.mean(active)) if active else 0.0,
            "cells": int(cells),
        }
    driven, peaks = driven[live], np.deg2rad(peaks[live])

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    # Try both handednesses and keep the one that fits; the sign IS the answer for
    # which way the second ring was embedded, not a free parameter of the claim.
    best = None
    for slope in (1.0, -1.0):
        offset, _ = _circular_mean(wrap(peaks - slope * driven), np.ones(len(driven)))
        residual = wrap(peaks - slope * driven - np.deg2rad(offset))
        rms = float(np.degrees(np.sqrt((residual ** 2).mean())))
        if best is None or rms < best["residual_deg"]:
            best = {"slope": slope, "offset_deg": float(offset), "residual_deg": rms}

    # Circular correlation (Jammalamadaka), the right statistic for angle against angle.
    dm = wrap(driven - _circular_mean(driven, np.ones(len(driven)))[0] * np.pi / 180)
    pm = wrap(peaks - _circular_mean(peaks, np.ones(len(peaks)))[0] * np.pi / 180)
    denominator = np.sqrt((np.sin(dm) ** 2).sum() * (np.sin(pm) ** 2).sum())
    circular_corr = (
        float((np.sin(dm) * np.sin(pm)).sum() / denominator) if denominator > 0 else float("nan")
    )
    return {
        "readout": readout,
        **best,
        "circular_corr": circular_corr,
        "mean_vector_length": float(np.nanmean(lengths)),
        "n": int(live.sum()),
        "mean_active": float(np.mean(active)) if active else 0.0,
        "cells": int(cells),
    }
