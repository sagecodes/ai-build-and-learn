"""Terminal charts, so every example shows its curves without opening a PNG.

    print(line({"TD(0)": rms_td, "MC": rms_mc}, title="RMS error", logy=True))
    print(bars({"UCB": 148.7, "Thompson": 68.3}, title="total regret"))

The same text lands in the Flyte report's <pre> block.
"""

import numpy as np

MARKERS = "*o+x#@%&"


def _resample(y, width):
    y = np.asarray(y, dtype=float)
    edges = np.linspace(0, len(y), width + 1).astype(int)
    return np.array([y[a:max(b, a + 1)].mean() for a, b in zip(edges[:-1], edges[1:])])


def line(series, title="", height=12, width=64, logy=False, ymin=None, ymax=None,
         xlabel="", xmax=None):
    """Overlay several curves on one character grid. Later series draw on top."""
    data = {k: _resample(v, width) for k, v in series.items()}
    allv = np.concatenate(list(data.values()))
    if logy:
        allv = allv[allv > 0]
    lo = ymin if ymin is not None else allv.min()
    hi = ymax if ymax is not None else allv.max()
    f = np.log10 if logy else (lambda v: v)
    flo, fhi = f(lo), f(hi)
    span = (fhi - flo) or 1.0

    grid = [[" "] * width for _ in range(height)]
    for marker, ys in zip(MARKERS, data.values()):
        for x, yv in enumerate(ys):
            if logy and yv <= 0:
                continue
            row = int(round((f(yv) - flo) / span * (height - 1)))
            grid[height - 1 - min(max(row, 0), height - 1)][x] = marker

    def label(r):
        v = fhi - span * r / (height - 1)
        return f"{10 ** v if logy else v:9.3g}"

    out = [f"  {title}"] if title else []
    for r, row in enumerate(grid):
        tick = label(r) if r in (0, height // 2, height - 1) else " " * 9
        out.append(f"{tick} |{''.join(row)}")
    out.append(" " * 10 + "+" + "-" * width)
    if xmax is not None:
        out.append(" " * 11 + "0" + f"{xmax:>{width - 1}}" + (f"  {xlabel}" if xlabel else ""))
    out.append("  " + "   ".join(f"{m} {k}" for m, k in zip(MARKERS, data)))
    return "\n".join(out)


def bars(values, title="", width=40, fmt="{:.1f}"):
    """Horizontal bar chart of name -> number (NaN entries are skipped)."""
    values = {k: v for k, v in values.items() if v == v}
    if not values:
        return f"  {title}: no data" if title else ""
    vmax = max(abs(v) for v in values.values()) or 1.0
    pad = max(len(k) for k in values)
    out = [f"  {title}"] if title else []
    for k, v in values.items():
        n = int(round(abs(v) / vmax * width))
        out.append(f"  {k:>{pad}} |{'#' * n}{' ' * (width - n)} {fmt.format(v)}")
    return "\n".join(out)
