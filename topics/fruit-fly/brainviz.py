"""The brain lighting up: 138,639 neurons at their real anatomical positions, glowing.

Every neuron in the FlyWire annotation table carries `pos_x, pos_y, pos_z`, its actual
location in the FAFB electron-microscopy volume. So the brain in these videos is not a
diagram or a force-directed graph: it is where the cells are in a real fly's head, and
a flash in the lower middle of the frame really is the subesophageal zone.

Rendering is pure numpy, no matplotlib figure, for one reason: there are 375 frames in
a 3-second run and 138,639 points in each, and a scatter plot of that takes about a
second per frame. Instead every neuron is binned to a pixel ONCE at construction, and
each frame is then an `np.add.at` into a float buffer plus a colormap lookup, which
costs about 3 ms. The glow (activity persisting and fading over a few frames) is a
first-order decay on that buffer, which is also what makes 15 ms ticks legible at 25
frames per second instead of strobing.

── Reading the picture ─────────────────────────────────────────────────────────
Frontal view, looking at the fly's face, so the fly's LEFT is on the RIGHT of frame,
the way you would see it standing in front of it. The two large lobes filling most of
the width are the optic lobes: 77,541 of the 138,639 neurons are visual, which is the
single most striking fact about a fly brain and is obvious the moment you plot it. The
central brain is the bridge between them, and the stalk descending below is where the
descending neurons leave for the body.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from connectome import Connectome

# Which anatomical axes map to (horizontal, vertical) in the image.
#   frontal: looking at the face. x is left-right, y is dorsal-ventral.
#   dorsal:  looking down on the head. x is left-right, z is anterior-posterior.
VIEWS = {"frontal": ("pos_x", "pos_y"), "dorsal": ("pos_x", "pos_z")}


def _colormap(name: str, n: int = 256) -> np.ndarray:
    """A (n, 3) uint8 lookup table. Falls back to a hand-rolled ramp without mpl."""
    try:
        from matplotlib import colormaps

        table = colormaps[name](np.linspace(0, 1, n))[:, :3]
        return (table * 255).astype(np.uint8)
    except Exception:
        ramp = np.linspace(0, 1, n)[:, None]
        base = np.array([[0.0, 0.0, 0.0], [1.0, 0.55, 0.1], [1.0, 1.0, 0.85]])
        idx = (ramp * (len(base) - 1)).astype(int).clip(0, len(base) - 2)
        frac = ramp * (len(base) - 1) - idx
        return ((base[idx] * (1 - frac) + base[idx + 1] * frac) * 255).astype(np.uint8)


@dataclass
class BrainCanvas:
    """Renders per-tick spike counts as frames of a glowing anatomical brain.

    Args:
        conn: The connectome, for positions and annotations.
        width: Frame width in pixels. Height follows from the brain's aspect ratio.
        view: "frontal" or "dorsal".
        decay: Per-frame multiplier on the activity buffer. 0.6 gives a ~4-frame
            afterglow, long enough to see a wave move and short enough to still read
            as movement.
        gain: Spikes-per-pixel that saturates the colormap. Set from a calibration
            run rather than guessed; `autoscale` does this.
        cmap: Any matplotlib colormap name. "inferno" reads well on dark backgrounds.
    """

    conn: Connectome
    width: int = 560
    view: str = "frontal"
    decay: float = 0.6
    gain: float = 3.0
    cmap: str = "inferno"
    # Draw only these neurons, and frame the picture on them. Used by the learning demo
    # to zoom the camera onto the mushroom body, which is 5,177 Kenyon cells and 96
    # output neurons out of 138,639 and is otherwise a few dozen pixels of a whole brain.
    subset: np.ndarray | None = None
    # Draw each neuron as a disc of this pixel radius instead of a single pixel. A
    # whole brain has 138,639 points and needs none of this; a 96-cell population needs
    # it badly, or the panel is 96 lit pixels in a 440-wide frame and reads as noise.
    spread: int = 0

    height: int = field(init=False, default=0)
    _px: np.ndarray = field(init=False, repr=False, default=None)
    _py: np.ndarray = field(init=False, repr=False, default=None)
    _heat: np.ndarray = field(init=False, repr=False, default=None)
    _under: np.ndarray = field(init=False, repr=False, default=None)
    _offsets: list = field(init=False, repr=False, default_factory=list)
    _lut: np.ndarray = field(init=False, repr=False, default=None)

    def __post_init__(self) -> None:
        if self.view not in VIEWS:
            raise ValueError(f"view must be one of {sorted(VIEWS)}, got {self.view!r}")
        hcol, vcol = VIEWS[self.view]
        ann = self.conn.ann

        h = pd.to_numeric(ann[hcol], errors="coerce").to_numpy(float)
        v = pd.to_numeric(ann[vcol], errors="coerce").to_numpy(float)
        bi = ann["bi"].to_numpy(np.int64)
        ok = np.isfinite(h) & np.isfinite(v)
        if self.subset is not None:
            ok &= np.isin(bi, np.asarray(self.subset, dtype=np.int64))
        h, v, bi = h[ok], v[ok], bi[ok]
        if len(h) == 0:
            raise ValueError("nothing to draw: the subset has no positioned neurons")

        # Percentile bounds, not min/max: a handful of neurons have positions at the
        # edge of the imaged volume and would otherwise shrink the brain into the
        # middle third of the frame. On a SUBSET those percentiles would clip real cells
        # instead of outliers, so a zoomed canvas takes the full extent plus a margin.
        if self.subset is None:
            h0, h1 = np.percentile(h, [0.1, 99.9])
            v0, v1 = np.percentile(v, [0.1, 99.9])
        else:
            # np.ptp(x), not x.ptp(): the method was removed from ndarray in numpy 2.
            pad_h = 0.06 * max(float(np.ptp(h)), 1.0)
            pad_v = 0.06 * max(float(np.ptp(v)), 1.0)
            h0, h1 = h.min() - pad_h, h.max() + pad_h
            v0, v1 = v.min() - pad_v, v.max() + pad_v
        aspect = (v1 - v0) / max(h1 - h0, 1e-9)
        self.height = int(round(self.width * aspect))

        # Frontal view: mirror horizontally so the fly's left is on the right of frame,
        # i.e. the view you get standing in front of the animal.
        hn = (h - h0) / (h1 - h0)
        if self.view == "frontal":
            hn = 1.0 - hn
        self._px = np.clip((hn * (self.width - 1)).astype(np.int32), 0, self.width - 1)
        self._py = np.clip(
            (((v - v0) / (v1 - v0)) * (self.height - 1)).astype(np.int32),
            0, self.height - 1,
        )

        # Map from Brian index -> pixel, for the subset that has coordinates. Neurons
        # without a position simply never light up; there are 14 of them.
        self._bi = bi
        self._heat = np.zeros((self.height, self.width), np.float32)
        self._lut = _colormap(self.cmap)
        # Offsets making up the disc each neuron is stamped with. Radius 0 is one pixel,
        # which is the whole-brain default and costs nothing.
        r = max(int(self.spread), 0)
        offsets = [
            (dy, dx)
            for dy in range(-r, r + 1)
            for dx in range(-r, r + 1)
            if dy * dy + dx * dx <= r * r
        ]
        self._offsets = offsets or [(0, 0)]

        # The static anatomy underlay: cell density, log-compressed, dim and cool. It
        # is what makes the frame legible as a brain while the network is silent.
        density = np.zeros((self.height, self.width), np.float32)
        for dy, dx in self._offsets:
            ys = np.clip(self._py + dy, 0, self.height - 1)
            xs = np.clip(self._px + dx, 0, self.width - 1)
            np.add.at(density, (ys, xs), 1.0)
        density = np.log1p(density)
        density /= max(density.max(), 1e-6)
        # A zoomed canvas has far fewer cells, so the silhouette needs lifting or the
        # frame reads as empty black between flashes.
        if self.subset is not None:
            density = np.clip(density * 2.2, 0, 1)
        self._under = np.zeros((self.height, self.width, 3), np.float32)
        self._under[..., 0] = density * 52    # a cold blue-grey silhouette
        self._under[..., 1] = density * 72
        self._under[..., 2] = density * 115

    # ── Per-frame ───────────────────────────────────────────────────────────────

    def update(self, counts: np.ndarray) -> None:
        """Add one tick of spikes (per-neuron counts, from `brain.Tick.counts`)."""
        self._heat *= self.decay
        spiked = counts[self._bi]
        nz = np.nonzero(spiked)[0]
        if len(nz):
            for dy, dx in self._offsets:
                ys = np.clip(self._py[nz] + dy, 0, self.height - 1)
                xs = np.clip(self._px[nz] + dx, 0, self.width - 1)
                np.add.at(self._heat, (ys, xs), spiked[nz])

    def frame(self) -> np.ndarray:
        """The current frame as (height, width, 3) uint8."""
        # sqrt, not linear: spike counts per pixel are heavy-tailed, and a linear
        # ramp puts almost every active pixel in the bottom eighth of the colormap
        # where it is indistinguishable from the anatomy underneath.
        norm = np.sqrt(np.clip(self._heat / max(self.gain, 1e-6), 0, 1))
        rgb = self._lut[(norm * (len(self._lut) - 1)).astype(np.uint8)].astype(np.float32)
        # Screen-blend the glow over the anatomy so quiet regions still show the
        # silhouette instead of going pure black.
        out = self._under + rgb * norm[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    def autoscale(self, counts_history: list[np.ndarray], percentile: float = 99.5) -> float:
        """Pick `gain` from a pilot run so the busiest pixels just saturate.

        Guessing this wrong is the difference between a brain that looks dead and one
        that looks like a white blob, and the right value moves by an order of
        magnitude between a resting brain and a visually driven one.
        """
        peaks = []
        saved_heat = self._heat.copy()
        self._heat[:] = 0
        for counts in counts_history:
            self.update(counts)
            peaks.append(np.percentile(self._heat, percentile))
        self._heat = saved_heat
        self.gain = float(max(np.max(peaks) if peaks else 1.0, 1e-3))
        return self.gain

    def reset(self) -> None:
        self._heat[:] = 0


def legend_rows(conn: Connectome) -> list[tuple[str, str]]:
    """Composition of the brain being drawn, for the report caption."""
    counts = conn.ann["super_class"].value_counts()
    return [(str(k), f"{int(v):,}") for k, v in counts.head(6).items()]
