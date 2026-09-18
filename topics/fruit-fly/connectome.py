"""The wiring diagram: FlyWire 783, and how to ask it for a named group of neurons.

This module does exactly two things.

1. Loads the three files that make up the connectome (see config._CONNECTOME_FETCH for
   where they come from and why they are pinned):

       completeness.csv    138,639 rows. Which neurons exist, in the order that fixes
                           every index used downstream. Row number = Brian index.
       connectivity.parquet 15,091,983 rows. The edge list. One row per ORDERED PAIR of
                           connected neurons, NOT per synapse: `Connectivity` is how
                           many synapses that pair shares, and `Excitatory` is +1/-1
                           from the presynaptic neuron's predicted neurotransmitter.
                           Their product is the signed weight, and it is the only thing
                           that makes this a brain rather than a graph.
       annotations.tsv     138,625 of those neurons, cell-typed by hand (Schlegel et
                           al. 2024). This is what turns a 19-digit root ID into
                           "DNa02, left".

2. Resolves selections like `select(cell_sub_class="sugar/water")` into the integer
   indices that brain.py stimulates and reads.

── Why the index bookkeeping matters more than it looks ────────────────────────
Three ID spaces are in play and mixing them up produces a brain that runs perfectly
and means nothing:

    root_id       720575940624319124   FlyWire's identifier. 19 digits, survives across
                                       datasets, appears in every paper and in Codex.
    brian index   0 .. 138,638         Position in completeness.csv. This is what
                                       Brian2's NeuronGroup, the edge list's
                                       Presynaptic_Index, and every array here use.
    ommatidium    0 .. 720             A facet of the simulated eye. Has NOTHING to do
                                       with either of the above; see bridge.py.

The edge list ships with its indices already resolved against completeness.csv, so the
one rule is: never reorder the completeness table. `load()` reads it with the row order
untouched, and everything else keys off that.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Baked into the image at /opt/connectome. On a devbox checkout, point
# CONNECTOME_DIR at wherever you fetched the three files (setup.sh does this).
DATA_DIR = Path(os.environ.get("CONNECTOME_DIR", "/opt/connectome"))

_FILES = {
    "completeness": "completeness.csv",
    "connectivity": "connectivity.parquet",
    "annotations": "annotations.tsv",
}


@dataclass(frozen=True)
class Connectome:
    """The loaded wiring diagram.

    Attributes:
        n: Number of neurons (138,639).
        pre: Presynaptic Brian index per edge, shape (15,091,983,).
        post: Postsynaptic Brian index per edge, same shape.
        weight: Signed synapse count per edge, same shape. Positive = excitatory,
            negative = inhibitory, magnitude = number of synapses.
        ann: Annotation table, already joined to the Brian index and filtered to the
            neurons that are actually in the model. Carries a `bi` column.
    """

    n: int
    pre: np.ndarray
    post: np.ndarray
    weight: np.ndarray
    ann: pd.DataFrame

    # ── Selection ───────────────────────────────────────────────────────────────

    def select(self, **criteria: str | list[str]) -> pd.DataFrame:
        """Rows of the annotation table matching every criterion.

        Each keyword is an annotation column; a string matches exactly, a list matches
        any member. Empty results are returned, not raised on: a caller asking for a
        cell type that does not exist in this dataset wants to see zero rows and the
        reason, which `describe()` prints.

            select(cell_type="DNa02")                    both DNa02 neurons
            select(cell_type="DNa02", side="left")       one of them
            select(cell_sub_class="sugar/water")         129 sugar-sensing GRNs
            select(super_class="descending")             1,291 descending neurons
        """
        mask = pd.Series(True, index=self.ann.index)
        for column, value in criteria.items():
            if column not in self.ann.columns:
                raise KeyError(
                    f"no annotation column {column!r}; available: "
                    f"{sorted(self.ann.columns)}"
                )
            if isinstance(value, (list, tuple, set)):
                mask &= self.ann[column].isin(list(value))
            else:
                mask &= self.ann[column] == value
        return self.ann[mask]

    def indices(self, **criteria: str | list[str]) -> np.ndarray:
        """`select`, reduced to a sorted array of unique Brian indices."""
        return np.unique(self.select(**criteria)["bi"].to_numpy())

    def sided(self, **criteria: str | list[str]) -> tuple[np.ndarray, np.ndarray]:
        """`indices`, split into (left, right).

        Neurons annotated `center` belong to neither and are dropped. That is 76 of the
        10,855 photoreceptors and a handful of descending neurons: midline cells cannot
        contribute to a left-versus-right comparison by construction, and including
        them in both halves would add the same number to each side of every difference.
        """
        left = np.unique(self.select(**criteria, side="left")["bi"].to_numpy())
        right = np.unique(self.select(**criteria, side="right")["bi"].to_numpy())
        return left, right

    def names(self, idx: np.ndarray) -> list[str]:
        """Human-readable `celltype(side)` labels for Brian indices, for plots."""
        lookup = self.ann.set_index("bi")
        out = []
        for i in np.asarray(idx):
            if i in lookup.index:
                row = lookup.loc[i]
                row = row.iloc[0] if isinstance(row, pd.DataFrame) else row
                ct = row["cell_type"] or row["cell_class"] or "?"
                out.append(f"{ct}({str(row['side'])[0]})")
            else:
                out.append(f"#{i}")
        return out

    # ── Controls ────────────────────────────────────────────────────────────────

    def shuffled(self, seed: int = 0) -> "Connectome":
        """The same brain with its wiring destroyed, for the null model.

        Permutes the postsynaptic index of every edge. This preserves, exactly:
        the number of neurons, the number of edges, every neuron's OUT-degree, and the
        entire distribution of signed weights. It destroys only which neuron talks to
        which. So a behaviour that survives this shuffle was never coming from the
        connectome, it was coming from the wrapper around it, and that is the single
        most useful thing this repo can tell you.

        In-degree is NOT preserved (a uniform permutation gives every neuron a Poisson
        in-degree instead of the real heavy-tailed one). A degree-preserving
        double-edge swap would fix that and costs minutes on 15M edges; the cheap
        version is enough to kill any structured response, which is what it is for.
        """
        rng = np.random.default_rng(seed)
        return Connectome(
            n=self.n,
            pre=self.pre,
            post=rng.permutation(self.post),
            weight=self.weight,
            ann=self.ann,
        )

    def silence(self, indices: np.ndarray) -> "Connectome":
        """The same brain with a set of neurons cut out of the conversation.

        Every OUTGOING edge from `indices` has its weight set to zero. The cells still
        exist, still integrate, still spike; they just stop being heard. This is the
        reference implementation's definition of silencing, and it is the closest
        in-silico analogue of the optogenetic silencing experiments the connectome
        literature is built on.

        Zeroing outgoing weights rather than deleting the rows keeps every index stable,
        so the same neuron is the same number across an intact run and a lesioned one
        and the brain visualiser keeps drawing them in the same place.
        """
        indices = np.asarray(indices, dtype=np.int64)
        weight = self.weight.copy()
        weight[np.isin(self.pre, indices)] = 0.0
        return Connectome(n=self.n, pre=self.pre, post=self.post, weight=weight,
                          ann=self.ann)

    def resolve(self, name: str) -> np.ndarray:
        """Indices for a group named by cell_type, then cell_class, then super_class.

        Lets a command line say `--silence LC4` or `--silence visual_projection`
        without the caller needing to know which column the name lives in.
        """
        for column in ("cell_type", "cell_class", "super_class", "cell_sub_class"):
            idx = self.indices(**{column: name})
            if len(idx):
                return idx
        raise KeyError(f"no neurons match {name!r} in any annotation column")

    # ── Reporting ───────────────────────────────────────────────────────────────

    def describe(self) -> dict[str, int | float]:
        excit = int((self.weight > 0).sum())
        return {
            "neurons": self.n,
            "edges": len(self.pre),
            "synapses": int(np.abs(self.weight).sum()),
            "excitatory_edges": excit,
            "inhibitory_edges": len(self.weight) - excit,
            "annotated": len(self.ann),
            "mean_out_degree": round(len(self.pre) / self.n, 1),
        }


@functools.lru_cache(maxsize=1)
def load(data_dir: str | None = None) -> Connectome:
    """Read the three files and join them. Cached: this costs ~2 s and ~1.5 GB.

    Raises a pointed error rather than a pandas one if the files are missing, because
    the fix differs by context: in a pod the image build should have fetched them, on
    a devbox `./setup.sh` should have.
    """
    root = Path(data_dir) if data_dir else DATA_DIR
    missing = [f for f in _FILES.values() if not (root / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"connectome data missing from {root}: {missing}. "
            "In a Flyte pod this is baked in at image build time (config._CONNECTOME_FETCH); "
            "on the devbox run ./setup.sh, or set CONNECTOME_DIR."
        )

    completeness = pd.read_csv(root / _FILES["completeness"], index_col=0)
    edges = pd.read_parquet(
        root / _FILES["connectivity"],
        columns=["Presynaptic_Index", "Postsynaptic_Index", "Excitatory x Connectivity"],
    )

    annotations = pd.read_csv(root / _FILES["annotations"], sep="\t", low_memory=False)
    # root_id -> row number in completeness.csv, which IS the Brian index.
    root_to_index = {int(r): i for i, r in enumerate(completeness.index)}
    annotations["bi"] = annotations["root_id"].map(root_to_index)
    annotations = annotations.dropna(subset=["bi"]).copy()
    annotations["bi"] = annotations["bi"].astype(int)
    # Downstream code does string comparisons on these; NaN would make `== "left"`
    # quietly False in some columns and raise in others.
    for column in ("cell_type", "cell_class", "cell_sub_class", "super_class", "side"):
        annotations[column] = annotations[column].fillna("").astype(str)

    return Connectome(
        n=len(completeness),
        pre=edges["Presynaptic_Index"].to_numpy(np.int32),
        post=edges["Postsynaptic_Index"].to_numpy(np.int32),
        weight=edges["Excitatory x Connectivity"].to_numpy(np.float64),
        ann=annotations,
    )


# ── The named groups this demo actually uses ────────────────────────────────────
#
# Every one of these is a query against the annotation table, not a hard-coded ID list,
# so they stay meaningful if the dataset is bumped to a later FlyWire release.
#
# SENSORY (things we drive):
#   photoreceptors  10,855 R1-6/R7/R8 cells, 5,474 left + 5,305 right + 76 midline.
#                   The retina is only partially reconstructed in FAFB, so this is not
#                   the full 6,000-per-eye a real fly has; it is enough to carry a
#                   left-right difference, which is all the loop needs.
#   mechanosensory   2,668 bristle and chordotonal neurons. Measured here to be by far
#                   the most strongly lateralised input in the whole model.
#   sugar GRNs         129 gustatory receptor neurons, `sugar/water`. Shiu et al.'s
#                   feeding pathway starts here.
#   bitter GRNs         65 the aversive counterpart.
#
# DESCENDING (things we read):
#   descending       1,291 neurons, 645 left + 646 right. The ONLY route from brain to
#                   ventral nerve cord: whatever this brain tells this body, it says
#                   through these cells. Reading the population, not a named pair, is
#                   deliberate and the reason is measured in the README.
#   DNa01/DNa02     the classic steering pair, kept as named readouts for the report
#                   because the literature is about them and because watching DNa02
#                   fail to lateralise is the most instructive plot in the demo.
#   MDN             backward walking. DNp09 freezing.

SENSORY_GROUPS: dict[str, dict[str, str]] = {
    "photoreceptors": {"super_class": "sensory", "cell_class": "visual"},
    "mechanosensory": {"super_class": "sensory", "cell_class": "mechanosensory"},
    "sugar": {"cell_sub_class": "sugar/water"},
    "bitter": {"cell_sub_class": "bitter"},
    "olfactory": {"super_class": "sensory", "cell_class": "olfactory"},
}

NAMED_DESCENDING = ("DNa01", "DNa02", "DNp09", "MDN")

# The looming-escape cells, by name, for the swat experiment. Every one of these is a
# studied neuron with a literature behind it:
#
#   LPLC2  210 cells, the canonical looming detector, tuned to outward motion in all
#          directions (an expanding edge) and the main driver of the giant fibre.
#   LC4    104 cells, tuned to angular velocity rather than size, the fast half of the
#          two-channel escape system.
#   LC6    125 cells, another looming-responsive lobula columnar type.
#   DNp01    2 cells, the GIANT FIBRE. One per side, the largest axon in the fly, and
#          the trigger for the short-mode escape jump.
#   DNp02/03/04/06/11  the rest of the descending escape ensemble, 2 cells each.
ESCAPE_TYPES: dict[str, list[str]] = {
    "LPLC2": ["LPLC2"],
    "LC4": ["LC4"],
    "LC6": ["LC6"],
    "giant fibre (DNp01)": ["DNp01"],
    "escape DNs": ["DNp02", "DNp03", "DNp04", "DNp06", "DNp11"],
}

# The visual pathway, in order, from the eye to the legs. Counting how many cells of
# each stage fire under a stimulus is the most informative single measurement in this
# repo, because it says exactly WHERE a signal stops. Measured here, driving every
# photoreceptor at 200 Hz: the lamina fires, and nothing past it does.
VISUAL_PATHWAY: list[tuple[str, dict[str, list[str] | str]]] = [
    ("photoreceptors R1-6/R7/R8", {"cell_type": ["R1-6", "R7", "R8"]}),
    ("lamina monopolar L1/L2/L3", {"cell_type": ["L1", "L2", "L3"]}),
    ("lamina L5 + amacrine Lai", {"cell_type": ["L5", "Lai"]}),
    ("medulla Mi1/Tm1/Tm3", {"cell_type": ["Mi1", "Tm1", "Tm3"]}),
    ("T4, the ON motion detector", {"cell_type": ["T4a", "T4b", "T4c", "T4d"]}),
    ("T5, the OFF motion detector", {"cell_type": ["T5a", "T5b", "T5c", "T5d"]}),
    ("lobula plate tangential HS/VS", {"cell_type": [
        "HSE", "HSN", "HSS", "H2", "VS1", "VS2", "VS3", "VS4",
        "VS5", "VS6", "VS7", "VS8"]}),
    ("looming LC4 + LPLC2", {"cell_type": ["LC4", "LPLC2"]}),
    ("giant fibre DNp01", {"cell_type": ["DNp01"]}),
    ("all descending neurons", {"super_class": "descending"}),
]
