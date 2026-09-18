"""Turning V-JEPA 2's inputs and measurements into something a Flyte report can show.

── Why there is no "generated video" here ──────────────────────────────────────
Cosmos and the video-generation demos base64 a clip the model produced. V-JEPA 2 has
no decoder: the predictor emits 1024-dimensional vectors and there is no head anywhere
in the released checkpoints that turns one back into pixels. Anything claiming to show
"what V-JEPA 2 predicted" as an image is showing you a projection of a vector, not a
prediction.

So the video in these reports is honest about what it is:

  masked_video()   The model's ACTUAL input with the masked tokens blacked out. Not a
                   visualisation, the literal thing the encoder was given. This is what
                   makes the inpainting task legible: you can see the hole.
  heat_video()     A per-patch MEASUREMENT painted onto those same pixels. Always a
                   number we computed (prediction quality), never a claim about
                   semantics.
  clip_video()     The source clip, for the retrieval results, where the point is
                   whether two clips are the same kind of event.

Everything composites onto `clips.shown_pixels()` output, i.e. the post-crop tensor the
patch embedding actually consumed, so patch (h, w) is pixels [16h:16h+16, 16w:16w+16]
and the overlay cannot drift out of alignment with the grid.

Same three-step encode/probe/embed contract as topics/cosmos/media.py, for the same
reasons: PyAV because aarch64 wheels exist, a luminance probe because a black clip is a
perfectly valid mp4 that a report will happily embed as a black rectangle, and base64
into a <video> tag so the report is one self-contained document with no object-store
round trip that can point at the wrong rustfs.
"""

from __future__ import annotations

import base64
import io
import logging

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# base64 inflates by 4/3 and a Flyte report is HTML held in memory to render.
_EMBED_LIMIT_MB = 24


# ── mp4 ─────────────────────────────────────────────────────────────────────────


def encode_mp4(frames, fps: int = 12, crf: int = 24) -> bytes:
    """[T, H, W, 3] uint8 -> H.264 mp4 bytes."""
    frames = np.asarray(frames)
    if frames.size == 0:
        return b""
    import av

    buf = io.BytesIO()
    h, w = frames.shape[1:3]
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        # libx264 needs even dimensions for yuv420p; 256 already is, but overlays get
        # built at other sizes and a stray odd height is an unhelpful way to fail.
        stream.width, stream.height = w - (w % 2), h - (h % 2)
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "veryfast"}
        for frame in frames:
            cropped = np.ascontiguousarray(frame[: stream.height, : stream.width])
            for pkt in stream.encode(av.VideoFrame.from_ndarray(cropped, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return buf.getvalue()


def probe(mp4: bytes) -> str:
    """One line saying whether the clip shows anything, and whether it moves.

    Two failure modes that both decode without error and have both bitten this repo:
    every frame black (a renderer that produced nothing), and every frame identical (a
    "video" that is one still repeated).
    """
    if not mp4:
        return "no clip"
    try:
        import av

        with av.open(io.BytesIO(mp4)) as c:
            grays = [f.to_ndarray(format="gray").astype("float32") for f in c.decode(video=0)]
        if not grays:
            return "clip decodes to ZERO frames"
        means = [float(g.mean()) for g in grays]
        motion = (
            float(np.mean([np.abs(b - a).mean() for a, b in zip(grays, grays[1:])]))
            if len(grays) > 1
            else 0.0
        )
        return (
            f"{len(means)} frames, {grays[0].shape[1]}x{grays[0].shape[0]}, "
            f"luminance min {min(means):.1f} / mean {sum(means) / len(means):.1f} / "
            f"max {max(means):.1f}, {sum(1 for m in means if m < 1.0)} black, "
            f"inter-frame motion {motion:.2f}"
        )
    except Exception as exc:  # noqa: BLE001
        return f"probe failed: {exc}"


def video_html(mp4: bytes, caption: str = "", max_width: int = 420) -> str:
    """base64 an mp4 into a self-contained <video> tag. No JS, no external assets."""
    if not mp4:
        return '<p style="color:#888;font-family:monospace;">no clip</p>'
    mb = len(mp4) / 2**20
    if mb > _EMBED_LIMIT_MB:
        return (
            f'<p style="color:#888;font-family:monospace;">clip is {mb:.1f} MB, over the '
            f"{_EMBED_LIMIT_MB} MB embed limit</p>"
        )
    b64 = base64.b64encode(mp4).decode()
    cap = (
        f'<p style="color:#888;font-family:monospace;font-size:12px;margin:6px 0 0;">{caption}</p>'
        if caption
        else ""
    )
    return (
        f'<div style="background:#0f0f23;padding:12px;border-radius:8px;">'
        f'<video src="data:video/mp4;base64,{b64}" controls autoplay loop muted playsinline '
        f'style="max-width:{max_width}px;width:100%;border:2px solid #333;border-radius:4px;'
        f'display:block;"></video>{cap}</div>'
    )


def strip(frames, count: int = 6, width: int = 130) -> str:
    """A row of evenly spaced stills as inline PNGs.

    Always alongside the clip, never instead of it: a strip survives a browser that
    will not autoplay a data: URI video, and laying frames out in time is how you see
    whether something is changing, which a short loop hides.
    """
    frames = np.asarray(frames)
    if frames.size == 0:
        return ""
    from PIL import Image

    n = len(frames)
    picks = [round(i * (n - 1) / max(count - 1, 1)) for i in range(min(count, n))]
    cells = ""
    for idx in picks:
        img = Image.fromarray(frames[idx])
        img.thumbnail((width, width * 2))
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
        cells += (
            f'<figure style="margin:0;"><img src="data:image/png;base64,{b64}" '
            f'style="width:{width}px;border-radius:3px;display:block;"/>'
            f'<figcaption style="color:#888;font-family:monospace;font-size:10px;'
            f'text-align:center;padding-top:3px;">frame {idx}</figcaption></figure>'
        )
    return (
        f'<div style="display:flex;gap:6px;flex-wrap:wrap;background:#0f0f23;padding:12px;'
        f'border-radius:8px;">{cells}</div>'
    )


# ── overlays ────────────────────────────────────────────────────────────────────


def _patch_size(shown: np.ndarray, grid: int) -> int:
    return shown.shape[1] // grid


def masked_video(shown: np.ndarray, mask3d, tubelet: int = 2) -> np.ndarray:
    """The model's input with the masked tokens blacked out.

    Frame f belongs to tubelet f // tubelet_size, and patch (h, w) of that tubelet is
    the 16x16 pixel block at [16h:16h+16, 16w:16w+16]. Blanking exactly those blocks is
    what the encoder's context actually was.
    """
    mask = np.asarray(mask3d)
    out = shown.copy()
    grid = mask.shape[1]
    ps = _patch_size(shown, grid)
    for f in range(len(out)):
        t = min(f // tubelet, mask.shape[0] - 1)
        hs, ws = np.nonzero(mask[t])
        for h, w in zip(hs, ws):
            out[f, h * ps : (h + 1) * ps, w * ps : (w + 1) * ps] = 20
    return out


def _hot(x: np.ndarray) -> np.ndarray:
    """Blue (low) -> orange (high). NaN renders as flat grey, i.e. 'not measured'."""
    nan = np.isnan(x)
    s = np.nan_to_num(x, nan=0.0)
    rgb = np.stack(
        [
            np.clip(s * 1.7, 0, 1),
            np.clip(s * 1.2 - 0.25, 0, 1) * 0.8,
            np.clip(1 - s * 1.9, 0, 1) * 0.95,
        ],
        -1,
    )
    rgb[nan] = 0.28
    return rgb


def heat_video(
    shown: np.ndarray, field, tubelet: int = 2, alpha: float = 0.62,
    lo: float | None = None, hi: float | None = None,
) -> np.ndarray:
    """Paint a per-patch [T, G, G] measurement onto the pixels the model saw.

    Scaled between the 5th and 95th percentile of the FINITE values so the colours use
    their range; NaN (an unmasked token, nothing to score) stays grey.
    """
    from PIL import Image

    field = np.asarray(field, dtype=np.float32)
    finite = field[np.isfinite(field)]
    if finite.size == 0:
        return shown.copy()
    lo = float(np.percentile(finite, 5)) if lo is None else lo
    hi = float(np.percentile(finite, 95)) if hi is None else hi
    norm = (field - lo) / (hi - lo + 1e-6)
    norm = np.where(np.isfinite(field), np.clip(norm, 0, 1), np.nan)

    h, w = shown.shape[1:3]
    out = np.empty_like(shown)
    for f in range(len(shown)):
        t = min(f // tubelet, field.shape[0] - 1)
        heat = (_hot(norm[t]) * 255).astype(np.uint8)
        heat = np.asarray(Image.fromarray(heat).resize((w, h), Image.NEAREST))
        out[f] = ((1 - alpha) * shown[f] + alpha * heat).astype(np.uint8)
    return out


def side_by_side_video(left: np.ndarray, right: np.ndarray, gap: int = 6) -> np.ndarray:
    """Two aligned clips in one mp4, so a viewer cannot compare the wrong frames."""
    n = min(len(left), len(right))
    sep = np.full((n, left.shape[1], gap, 3), 15, dtype=np.uint8)
    return np.concatenate([left[:n], sep, right[:n]], axis=2)


# ── charts ──────────────────────────────────────────────────────────────────────


def _fig_html(fig, width: int = 620) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor="#0f0f23")
    import matplotlib.pyplot as plt

    plt.close(fig)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" style="max-width:{width}px;width:100%;'
        f'border-radius:6px;display:block;margin:8px 0;"/>'
    )


def _axes(title: str, xlabel: str, ylabel: str, size=(6.4, 3.4)):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=size, facecolor="#0f0f23")
    ax.set_facecolor("#1a1a2e")
    ax.set_title(title, color="#fdcb6e", fontsize=11)
    ax.set_xlabel(xlabel, color="#ccc", fontsize=9)
    ax.set_ylabel(ylabel, color="#ccc", fontsize=9)
    ax.tick_params(colors="#888", labelsize=8)
    for s in ax.spines.values():
        s.set_color("#333")
    ax.grid(alpha=0.15, color="#888")
    return fig, ax


def horizon_chart(
    series: dict[str, list[tuple[int, float]]],
    hlines: dict[str, float] | None = None,
) -> str:
    """Prediction quality against how far ahead it was asked to predict.

    `hlines` carries the two references that make the curve readable: the in-
    distribution tube-mask score (the ceiling this predictor reaches when asked the
    question it was trained on) and the shuffled chance floor.
    """
    fig, ax = _axes(
        "Latent prediction quality vs horizon",
        "tubelets ahead of the last visible frame (1 tubelet = 2 frames)",
        "mean centered cosine to truth",
    )
    colors = ["#00b894", "#fdcb6e", "#74b9ff", "#e17055"]
    for i, (label, pts) in enumerate(series.items()):
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=3.5, lw=1.8, color=colors[i % len(colors)], label=label)
    for i, (label, y) in enumerate((hlines or {}).items()):
        ax.axhline(y, ls="--", lw=1.4, color=["#888", "#74b9ff", "#e17055"][i % 3], label=label)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def bar_chart(title: str, groups: list[str], series: dict[str, list[float]],
              ylabel: str, floor: float | None = None, floor_label: str = "chance") -> str:
    """Grouped bars: the metric for each model, with the floor drawn across it."""
    fig, ax = _axes(title, "", ylabel, size=(6.4, 3.2))
    colors = ["#00b894", "#fdcb6e", "#74b9ff", "#e17055"]
    n = max(len(series), 1)
    width = 0.8 / n
    xs = np.arange(len(groups))
    for i, (label, vals) in enumerate(series.items()):
        pos = xs - 0.4 + width * (i + 0.5)
        ax.bar(pos, vals, width * 0.9, label=label, color=colors[i % len(colors)])
        for x, v in zip(pos, vals):
            ax.text(x, v, f"{v:.2f}", ha="center", va="bottom", color="#ccc", fontsize=7)
    if floor is not None:
        ax.axhline(floor, ls="--", lw=1.4, color="#888", label=floor_label)
    ax.set_xticks(xs)
    ax.set_xticklabels(groups, fontsize=8)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def confusion_chart(cm: np.ndarray, labels: list[str], title: str) -> str:
    """Confusion matrix. Which classes it confuses is more informative than the total."""
    fig, ax = _axes(title, "predicted", "true", size=(4.6, 4.0))
    ax.grid(False)
    ax.imshow(cm, cmap="magma", interpolation="nearest")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    hi = cm.max() if cm.size else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                    color="#fff" if cm[i, j] < hi * 0.6 else "#000")
    return _fig_html(fig, width=380)


# ── V-JEPA 2-AC: planning and dreaming ──────────────────────────────────────────
#
# Same rule as everything above: nothing here is a vector dressed up as an image.
# The planning video is the simulator's own camera. The dream video is a strip of
# REAL frames retrieved by nearest neighbour to each dreamed latent, captioned as
# such, with the retrieval distance plotted next to it so a viewer can see the
# moment the dream stops corresponding to anything real.


def annotate(frame: np.ndarray, lines: list[tuple[str, tuple[int, int, int]]],
             origin: tuple[int, int] = (6, 6), scale: int = 2) -> np.ndarray:
    """Burn small text into a frame with a 5x7 bitmap font.

    Written out rather than pulled from PIL because the caption has to survive
    mp4 encoding and be readable in a report at 420px wide, and because a missing
    font file in a pod is a silly way to lose a run.
    """
    out = frame.copy()
    x0, y0 = origin
    # Dark backing band first. A MuJoCo tabletop is bright and beige, and pale text on
    # it is unreadable at the width a report actually renders; blending rather than
    # filling keeps the frame visible underneath.
    if lines:
        w = max(len(t) for t, _ in lines) * 6 * scale + 2 * x0
        h = len(lines) * (8 * scale + 3) + y0
        band = out[: min(h, out.shape[0]), : min(w, out.shape[1])]
        band[:] = (band.astype(np.uint16) * 35 // 100).astype(np.uint8)
    for row, (text, colour) in enumerate(lines):
        y = y0 + row * (8 * scale + 3)
        x = x0
        for ch in text.upper():
            glyph = _FONT.get(ch, _FONT[" "])
            for c, col in enumerate(glyph):
                for r in range(7):
                    if col & (1 << r):
                        yy, xx = y + r * scale, x + c * scale
                        if 0 <= yy < out.shape[0] - scale and 0 <= xx < out.shape[1] - scale:
                            out[yy : yy + scale, xx : xx + scale] = colour
            x += 6 * scale
    return out


# 5x7 bitmap font, one byte per column, LSB = top row.
_FONT = {
    " ": (0, 0, 0, 0, 0), "0": (62, 65, 65, 65, 62), "1": (0, 66, 127, 64, 0),
    "2": (98, 81, 73, 73, 70), "3": (34, 65, 73, 73, 54), "4": (24, 20, 18, 127, 16),
    "5": (39, 69, 69, 69, 57), "6": (60, 74, 73, 73, 48), "7": (1, 113, 9, 5, 3),
    "8": (54, 73, 73, 73, 54), "9": (6, 73, 73, 41, 30), ".": (0, 96, 96, 0, 0),
    "-": (8, 8, 8, 8, 8), ":": (0, 54, 54, 0, 0), "/": (32, 16, 8, 4, 2),
    "%": (35, 19, 8, 100, 98), "+": (8, 8, 62, 8, 8), "=": (20, 20, 20, 20, 20),
    "(": (0, 28, 34, 65, 0), ")": (0, 65, 34, 28, 0), "|": (0, 0, 119, 0, 0),
    "A": (126, 9, 9, 9, 126), "B": (127, 73, 73, 73, 54), "C": (62, 65, 65, 65, 34),
    "D": (127, 65, 65, 34, 28), "E": (127, 73, 73, 73, 65), "F": (127, 9, 9, 9, 1),
    "G": (62, 65, 73, 73, 122), "H": (127, 8, 8, 8, 127), "I": (0, 65, 127, 65, 0),
    "J": (32, 64, 65, 63, 1), "K": (127, 8, 20, 34, 65), "L": (127, 64, 64, 64, 64),
    "M": (127, 2, 12, 2, 127), "N": (127, 4, 8, 16, 127), "O": (62, 65, 65, 65, 62),
    "P": (127, 9, 9, 9, 6), "Q": (62, 65, 81, 33, 94), "R": (127, 9, 25, 41, 70),
    "S": (38, 73, 73, 73, 50), "T": (1, 1, 127, 1, 1), "U": (63, 64, 64, 64, 63),
    "V": (31, 32, 64, 32, 31), "W": (63, 64, 56, 64, 63), "X": (99, 20, 8, 20, 99),
    "Y": (3, 4, 120, 4, 3), "Z": (97, 81, 73, 69, 67),
}

_GREEN = (0, 184, 148)
_AMBER = (253, 203, 110)
_GREY = (205, 205, 205)
_RED = (225, 112, 85)


def episode_video(ep, goal_frame: np.ndarray, label: str = "") -> np.ndarray:
    """The closed-loop run, with the goal photo pinned beside it.

    The goal is in every frame on purpose. This is a goal-conditioned task and the
    only thing the planner is optimising is "make the left image look like the
    right one"; a viewer who cannot see the target cannot judge whether it worked.
    """
    h = ep.frames[0].shape[0]
    goal = _resize_nn(goal_frame, h)
    out = []
    for t, f in enumerate(ep.frames):
        left = annotate(
            f,
            [
                (f"{label or ep.policy}  step {t}/{len(ep.frames) - 1}", _AMBER),
                (f"dist to goal {ep.dist[t] * 100:5.1f} cm", _GREEN if ep.dist[t] < ep.dist[0] else _GREY),
                (f"latent energy {ep.energy[t]:.4f}", _GREY),
            ],
        )
        right = annotate(goal.copy(), [("goal photo", _AMBER)])
        sep = np.full((h, 6, 3), 40, np.uint8)
        out.append(np.concatenate([left, sep, right], axis=1))
    return np.stack(out)


def compare_video(eps: dict, goal_frame: np.ndarray) -> np.ndarray:
    """Every policy side by side on one timeline, so nobody compares wrong frames."""
    names = list(eps)
    h = eps[names[0]].frames[0].shape[0]
    n = max(len(e.frames) for e in eps.values())
    goal = _resize_nn(goal_frame, h)
    out = []
    for t in range(n):
        tiles = []
        for name in names:
            ep = eps[name]
            i = min(t, len(ep.frames) - 1)
            col = _GREEN if ep.dist[i] < ep.dist[0] * 0.6 else (_AMBER if ep.dist[i] < ep.dist[0] else _RED)
            tiles.append(
                annotate(
                    ep.frames[i],
                    [(name, _AMBER), (f"{ep.dist[i] * 100:4.1f} cm", col)],
                )
            )
        tiles.append(annotate(goal.copy(), [("goal", _AMBER)]))
        sep = np.full((h, 6, 3), 40, np.uint8)
        row = [tiles[0]]
        for tile in tiles[1:]:
            row += [sep, tile]
        out.append(np.concatenate(row, axis=1))
    return np.stack(out)


def dream_video(bank_big: np.ndarray, idx: np.ndarray, retr_dist: np.ndarray,
                true_frames: np.ndarray) -> np.ndarray:
    """The dream, decoded by retrieval, next to what really happened.

    LEFT is not a reconstruction and the caption says so on every frame. It is the
    nearest real photograph, out of a bank of a few hundred, to the latent the
    world model imagined -- the closest thing to "showing the dream" that a model
    with no decoder permits. RIGHT is the same action sequence executed in the
    simulator. The number under the left panel is the retrieval distance: when it
    climbs, the dream has drifted somewhere no real frame lives, and the left panel
    should be read as the model failing rather than as the model predicting.
    """
    h = true_frames[0].shape[0]
    out = []
    for t in range(len(idx)):
        left = _resize_nn(bank_big[idx[t]], h)
        left = annotate(
            left,
            [
                (f"dreamed t+{t + 1}", _AMBER),
                ("nearest real frame", _GREY),
                (f"retrieval {retr_dist[t]:.4f}", _GREY),
            ],
        )
        right = annotate(true_frames[t], [(f"actual t+{t + 1}", _AMBER), ("simulator", _GREY)])
        sep = np.full((h, 6, 3), 40, np.uint8)
        out.append(np.concatenate([left, sep, right], axis=1))
    return np.stack(out)


def _resize_nn(img: np.ndarray, h: int) -> np.ndarray:
    """Nearest-neighbour resize to height h. No scipy/PIL round trip for one box."""
    if img.shape[0] == h:
        return img
    w = int(round(img.shape[1] * h / img.shape[0]))
    yi = (np.arange(h) * img.shape[0] / h).astype(int).clip(0, img.shape[0] - 1)
    xi = (np.arange(w) * img.shape[1] / w).astype(int).clip(0, img.shape[1] - 1)
    return img[yi][:, xi]


def progress_chart(eps: dict, threshold: float | None = None) -> str:
    """Distance to goal over time, one line per policy. The headline chart."""
    fig, ax = _axes("Distance to goal, closed loop", "environment step", "end-effector to goal (cm)")
    colors = {"jepa": "#00b894", "oracle": "#74b9ff", "greedy": "#fdcb6e",
              "lookahead": "#a29bfe", "random": "#e17055"}
    for name, ep in eps.items():
        ax.plot(
            np.arange(len(ep.dist)), np.array(ep.dist) * 100,
            marker="o", ms=3, lw=2.0 if name == "jepa" else 1.4,
            color=colors.get(name, "#aaa"), label=name,
            ls="-" if name in ("jepa", "oracle") else "--",
        )
    if threshold is not None:
        ax.axhline(threshold * 100, ls=":", lw=1.3, color="#888", label="reached")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def energy_vs_distance_chart(eps: dict) -> str:
    """The question the whole demo turns on: is latent energy a usable reward?

    Every (step, policy) pair is one point: the world model's energy on the x axis,
    the true end-effector distance to the goal on the y. If the encoder can see
    this scene at all, the cloud slopes upward and the correlation in the title is
    positive. A flat cloud means the energy is measuring something other than
    progress, and no amount of planning on top of it would help.
    """
    fig, ax = _axes("Latent energy vs true distance", "|z - z_goal| (L1)", "true distance to goal (cm)")
    colors = {"jepa": "#00b894", "oracle": "#74b9ff", "greedy": "#fdcb6e",
              "lookahead": "#a29bfe", "random": "#e17055"}
    xs, ys = [], []
    for name, ep in eps.items():
        ax.scatter(ep.energy, np.array(ep.dist) * 100, s=22, alpha=0.85,
                   color=colors.get(name, "#aaa"), label=name, edgecolors="none")
        xs += list(ep.energy)
        ys += list(np.array(ep.dist) * 100)
    if len(xs) > 2:
        r = float(np.corrcoef(xs, ys)[0, 1])
        b, a = np.polyfit(xs, ys, 1)
        gx = np.linspace(min(xs), max(xs), 2)
        ax.plot(gx, a + b * gx, lw=1.3, ls="--", color="#888")
        ax.set_title(f"Latent energy vs true distance   (r = {r:+.3f})", color="#fdcb6e", fontsize=11)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def dream_chart(dream_err: np.ndarray, true_err: np.ndarray, floor: float) -> str:
    """How fast the dream goes wrong, against the only floor that means anything.

    `floor` is the mean L1 between two UNRELATED real frames' latents. A dream
    whose error reaches the floor is no better than a randomly chosen frame, which
    is the point at which "the world model imagined the future" stops being true.
    """
    fig, ax = _axes("Dream divergence", "steps dreamed ahead", "L1 to the true future latent")
    h = np.arange(1, len(dream_err) + 1)
    ax.plot(h, dream_err, marker="o", ms=4, lw=2.0, color="#00b894", label="dreamed vs actual")
    ax.plot(h, true_err, marker="s", ms=3.5, lw=1.4, ls="--", color="#74b9ff",
            label="stand still vs actual")
    ax.axhline(floor, ls=":", lw=1.4, color="#e17055", label="two unrelated frames (floor)")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def cem_chart(traces: list) -> str:
    """CEM convergence, every planning call overlaid.

    Each faint line is one call to the planner: energy of the best sampled action
    sequence against iteration. They should fall. Flat lines mean the search is
    not finding anything the initial distribution did not already contain.
    """
    fig, ax = _axes("CEM convergence, every planning call", "CEM iteration", "best sampled energy")
    for i, tr in enumerate(traces):
        if not tr.energy_best:
            continue
        ax.plot(np.arange(1, len(tr.energy_best) + 1), tr.energy_best,
                lw=1.2, alpha=0.55, color="#00b894")
    if traces and traces[0].energy_best:
        m = np.mean([t.energy_best for t in traces if len(t.energy_best) == len(traces[0].energy_best)], axis=0)
        ax.plot(np.arange(1, len(m) + 1), m, lw=2.6, color="#fdcb6e", label="mean")
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def energy_landscape_chart(grid: np.ndarray, energy: np.ndarray, best: np.ndarray,
                           truth: np.ndarray | None, title: str) -> str:
    """The energy surface over a 2D slice of the action space.

    Marginalised over the third axis by taking the minimum, not the mean: what a
    planner can reach in the (x, y) plane is bounded by the best z available, and
    averaging would smear a real basin into a gradient.
    """
    fig, ax = _axes(title, "action dx (m)", "action dy (m)", size=(5.0, 4.2))
    ax.grid(False)
    xs = np.unique(grid[:, 0])
    ys = np.unique(grid[:, 1])
    surf = np.full((len(ys), len(xs)), np.nan)
    for (dx, dy, _), e in zip(grid, energy):
        i, j = int(np.argmin(abs(ys - dy))), int(np.argmin(abs(xs - dx)))
        surf[i, j] = e if np.isnan(surf[i, j]) else min(surf[i, j], e)
    im = ax.imshow(surf, origin="lower", cmap="viridis", aspect="auto",
                   extent=[xs[0], xs[-1], ys[0], ys[-1]])
    fig.colorbar(im, ax=ax).ax.tick_params(colors="#888", labelsize=7)
    ax.scatter([best[0]], [best[1]], s=150, marker="*", color="#fdcb6e",
               edgecolors="#0f0f23", linewidths=0.8, label="energy minimum", zorder=5)
    if truth is not None:
        ax.scatter([truth[0]], [truth[1]], s=110, marker="X", color="#e17055",
                   edgecolors="#0f0f23", linewidths=0.8, label="direction to goal", zorder=5)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc", loc="upper left")
    return _fig_html(fig)


def adapt_chart(history, standstill: float, before: float) -> str:
    """Validation L1 during fine-tuning, against the two lines that give it meaning.

    `standstill` is the L1 you get by predicting the scene does not change; a dynamics
    model below it has earned the word prediction and one above it has not. `before` is
    where the pretrained checkpoint starts, which on MuJoCo renders is ABOVE standstill.
    """
    fig, ax = _axes("Adapting the predictor to the simulator", "fine-tuning step",
                    "validation L1 to the true next latent")
    if history:
        xs = [h[0] for h in history]
        ax.plot(xs, [h[2] for h in history], marker="o", ms=4, lw=2.2, color="#00b894",
                label="adapted, held-out transitions")
        ax.plot(xs, [h[1] for h in history], lw=1.0, alpha=0.5, color="#fdcb6e",
                label="training batch")
    ax.axhline(standstill, ls="--", lw=1.5, color="#e17055",
               label="predict no change (the bar to clear)")
    ax.axhline(before, ls=":", lw=1.5, color="#888", label="pretrained checkpoint")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def triptych_video(truth: np.ndarray, control: np.ndarray, vjepa: np.ndarray) -> np.ndarray:
    """Truth, the decodability ceiling, and what V-JEPA's tokens give back.

    The middle panel is the load-bearing one. Without it a blurry right-hand panel is
    just as easily a weak decoder as a representation that threw the pixels away, and
    the reader has no way to tell which.
    """
    h = truth.shape[1]
    sep = np.full((h, 6, 3), 40, np.uint8)
    out = []
    n = min(len(truth), len(control), len(vjepa))
    for t in range(n):
        a = annotate(truth[t].copy(), [("ORIGINAL FRAME", _AMBER)])
        # Keep every caption under 21 characters: the font is 6*scale px per glyph,
        # so at scale 2 a 256px panel fits 21 before the text runs off the edge.
        b = annotate(control[t].copy(), [("RANDOM PROJECTION", _GREY),
                                         ("OF TRUE PIXELS", _GREY),
                                         ("METHOD WORKS", _GREEN)])
        c = annotate(vjepa[t].copy(), [("V-JEPA TOKENS", _GREY),
                                       ("SAME METHOD", _GREY),
                                       ("APPEARANCE GONE", _RED)])
        out.append(np.concatenate([a, sep, b, sep, c], axis=1))
    return np.stack(out)


def readout_chart(rows: dict) -> str:
    """Linear decodability, every condition against the two that give it meaning."""
    fig, ax = _axes("How much of the picture survives in one token?", "",
                    "linear readout to pixels (PSNR, dB)", size=(6.4, 3.4))
    order = ["rand", "grey", "vjepa", "shuf"]
    labels = {"rand": "random projection\nof true pixels", "grey": "each patch's\nmean colour",
              "vjepa": "V-JEPA 2\ntokens", "shuf": "shuffled\n(floor)"}
    colors = {"rand": "#74b9ff", "grey": "#fdcb6e", "vjepa": "#00b894", "shuf": "#e17055"}
    xs = np.arange(len(order))
    vals = [rows[k]["psnr"] for k in order]
    ax.bar(xs, vals, 0.62, color=[colors[k] for k in order])
    for x, k in zip(xs, order):
        ax.text(x, rows[k]["psnr"], f"{rows[k]['psnr']:.1f} dB\nR2 {rows[k]['r2']:+.2f}",
                ha="center", va="bottom", color="#ccc", fontsize=7.5)
    ax.axhline(rows["shuf"]["psnr"], ls="--", lw=1.3, color="#888", label="floor")
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[k] for k in order], fontsize=8)
    ax.set_ylim(0, max(vals) * 1.25)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def keeps_chart(pixel_r2: float, probe_acc: float, pixel_acc: float, chance: float) -> str:
    """The two halves of the sentence: appearance discarded, meaning kept."""
    fig, ax = _axes("The same tokens, asked two different questions", "",
                    "fraction of the way from chance to perfect", size=(6.4, 3.2))
    # Appearance is scored as R2 clipped at 0 (negative means worse than the mean),
    # semantics as the gap from chance closed. Both then live on a 0..1 axis.
    appearance = max(0.0, pixel_r2)
    semantics = (probe_acc - chance) / (1.0 - chance)
    pixels_sem = (pixel_acc - chance) / (1.0 - chance)
    ax.bar([0], [appearance], 0.5, color="#e17055", label="can you redraw the patch?")
    ax.bar([1], [semantics], 0.5, color="#00b894", label="can you name the action?")
    ax.bar([1.55], [pixels_sem], 0.5, color="#fdcb6e", label="same question, raw pixels")
    for x, v, t in ((0, appearance, f"R2 {pixel_r2:+.2f}"),
                    (1, semantics, f"{probe_acc:.0%} acc"),
                    (1.55, pixels_sem, f"{pixel_acc:.0%} acc")):
        ax.text(x, v, t, ha="center", va="bottom", color="#ccc", fontsize=8)
    ax.set_xticks([0, 1.27])
    ax.set_xticklabels(["APPEARANCE\ndiscarded", "MEANING\nkept"], fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


def still(frame: np.ndarray, width: int = 300, caption: str = "") -> str:
    """One labelled still as an inline PNG."""
    from PIL import Image

    img = Image.fromarray(frame)
    img.thumbnail((width, width * 2))
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    b64 = base64.b64encode(buf.getvalue()).decode()
    cap = (
        f'<figcaption style="color:#888;font-family:monospace;font-size:11px;'
        f'padding-top:4px;">{caption}</figcaption>' if caption else ""
    )
    return (
        f'<figure style="margin:0;background:#0f0f23;padding:12px;border-radius:8px;">'
        f'<img src="data:image/png;base64,{b64}" style="width:100%;border-radius:4px;'
        f'display:block;"/>{cap}</figure>'
    )


def push_video(eps: dict, goal_frame: np.ndarray) -> np.ndarray:
    """Every comparison space on one timeline, with the reference photo pinned right.

    Captions carry the BLOCK's distance, not the arm's, because that is what is being
    scored and a viewer watching the arm alone would misread a policy that poses
    convincingly without pushing anything.
    """
    names = list(eps)
    h = eps[names[0]].frames[0].shape[0]
    n = max(len(e.frames) for e in eps.values())
    goal = _resize_nn(goal_frame, h)
    sep = np.full((h, 6, 3), 40, np.uint8)
    out = []
    for t in range(n):
        tiles = []
        for nm in names:
            ep = eps[nm]
            i = min(t, len(ep.frames) - 1)
            d0, d = ep.cube_dist[0], ep.cube_dist[i]
            col = _GREEN if d < d0 * 0.6 else (_AMBER if d < d0 else _RED)
            tiles.append(annotate(ep.frames[i].copy(),
                                  [(nm[:21], _AMBER), (f"BLOCK {d * 100:4.1f} CM", col)]))
        tiles.append(annotate(goal.copy(), [("REFERENCE PHOTO", _AMBER), ("TASK DONE", _GREEN)]))
        row = [tiles[0]]
        for tile in tiles[1:]:
            row += [sep, tile]
        out.append(np.concatenate(row, axis=1))
    return np.stack(out)


def push_progress_chart(eps: dict) -> str:
    """Block distance to its photographed position, over the episode."""
    fig, ax = _axes("Distance from the block to where the photograph shows it",
                    "planning step", "block to goal (cm)")
    colors = {"V-JEPA 2 embeddings": "#00b894", "raw pixels": "#fdcb6e",
              "random net": "#a29bfe", "random actions": "#e17055",
              "scripted oracle": "#74b9ff"}
    for nm, ep in eps.items():
        ax.plot(np.arange(len(ep.cube_dist)), np.array(ep.cube_dist) * 100,
                marker="o", ms=3, lw=2.2 if "V-JEPA" in nm else 1.4,
                color=colors.get(nm, "#aaa"), label=nm,
                ls="-" if ("V-JEPA" in nm or "oracle" in nm) else "--")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig)


# ── The mechanism tasks: occlude, energy, ladder, collapse ─────────────────────
#
# Same rule as everything above. `labelled_row` is the only new way of showing
# pixels, and every panel it draws is either the model's literal input, real pixels
# retrieved from a bank, or a rendering whose CEILING is drawn in the panel beside it.
# Nothing gains a caption it did not earn.


def labelled_row(panels: list[tuple[np.ndarray, list[tuple[str, tuple[int, int, int]]]]],
                 gap: int = 6) -> np.ndarray:
    """N aligned clips in one mp4, each with its own burnt-in caption lines.

    Captions stay under 21 characters: the bitmap font is 6*scale pixels per glyph, so
    at scale 2 a 256px panel fits 21 before the text runs off the edge.
    """
    n = min(len(frames) for frames, _ in panels)
    h = panels[0][0].shape[1]
    sep = np.full((h, gap, 3), 40, np.uint8)
    out = []
    for t in range(n):
        tiles = [annotate(frames[t].copy(), lines) for frames, lines in panels]
        row = [tiles[0]]
        for tile in tiles[1:]:
            row += [sep, tile]
        out.append(np.concatenate(row, axis=1))
    return np.stack(out)


def curve_chart(title: str, xlabel: str, ylabel: str,
                series: dict[str, list[tuple[float, float]]],
                hlines: dict[str, float] | None = None,
                vlines: dict[str, float] | None = None,
                size=(6.4, 3.4), width: int = 620, logx: bool = False) -> str:
    """One or more (x, y) curves with optional reference lines. The generic plot."""
    fig, ax = _axes(title, xlabel, ylabel, size=size)
    colors = ["#00b894", "#fdcb6e", "#74b9ff", "#e17055", "#a29bfe", "#55efc4"]
    for i, (label, pts) in enumerate(series.items()):
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker="o", ms=3.2, lw=1.9, color=colors[i % len(colors)],
                label=label)
    for i, (label, y) in enumerate((hlines or {}).items()):
        ax.axhline(y, ls="--", lw=1.3, color=["#888", "#74b9ff", "#e17055"][i % 3],
                   label=label)
    for label, x in (vlines or {}).items():
        ax.axvline(x, ls=":", lw=1.3, color="#888", label=label)
    if logx:
        ax.set_xscale("log")
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc")
    return _fig_html(fig, width=width)


def energy_chart(scores: dict[str, dict], floors: dict[str, dict], key: str = "l1",
                 true_name: str = "true completion",
                 degenerate: tuple[str, ...] = ()) -> str:
    """Candidate completions ranked by energy, lowest at the top.

    The true completion is green and the degenerate candidates are red, because the
    finding that matters is not where the truth lands in absolute terms but whether
    anything cheap landed below it.
    """
    order = sorted(scores, key=lambda k: scores[k][key])
    vals = [scores[k][key] for k in order]
    fig, ax = _axes(f"Energy of each candidate completion ({key})", f"E = {key} distance",
                    "", size=(6.6, 0.42 * len(order) + 1.3))
    ys = np.arange(len(order))[::-1]
    cols = ["#00b894" if k == true_name else ("#e17055" if k in degenerate else "#74b9ff")
            for k in order]
    ax.barh(ys, vals, 0.62, color=cols)
    for y, v in zip(ys, vals):
        ax.text(v, y, f" {v:.4f}", va="center", color="#ccc", fontsize=7.5)
    for i, (label, s) in enumerate(floors.items()):
        ax.axvline(s[key], ls="--", lw=1.3, color=["#888", "#fdcb6e"][i % 2], label=label)
    ax.set_yticks(ys)
    ax.set_yticklabels(order, fontsize=8)
    ax.set_xlim(0, max(vals) * 1.16)
    ax.grid(axis="y", alpha=0)
    if floors:
        ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc",
                  loc="lower right")
    return _fig_html(fig, width=660)


def twin_chart(title: str, xlabel: str, x: list[float],
               left: tuple[str, list[float]], right: tuple[str, list[float]],
               refs: dict[str, float] | None = None,
               marks: dict[str, float] | None = None) -> str:
    """Two quantities on two y-axes against depth. Appearance left, meaning right."""
    fig, ax = _axes(title, xlabel, left[0], size=(7.0, 3.6))
    ax.plot(x, left[1], marker="o", ms=3.2, lw=2.0, color="#e17055", label=left[0])
    for i, (label, y) in enumerate((refs or {}).items()):
        ax.axhline(y, ls="--", lw=1.2, color=["#888", "#74b9ff"][i % 2], label=label)
    ax2 = ax.twinx()
    ax2.plot(x, right[1], marker="s", ms=3.2, lw=2.0, color="#00b894", label=right[0])
    ax2.set_ylabel(right[0], color="#ccc", fontsize=9)
    ax2.tick_params(colors="#888", labelsize=8)
    for s in ax2.spines.values():
        s.set_color("#333")
    for label, xv in (marks or {}).items():
        ax.axvline(xv, ls=":", lw=1.4, color="#fdcb6e")
        ax.text(xv, ax.get_ylim()[1], f" {label}", color="#fdcb6e", fontsize=7.5,
                va="top", rotation=90)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, facecolor="#1a1a2e", edgecolor="#333",
              labelcolor="#ccc", loc="center right")
    return _fig_html(fig, width=680)


def factor_chart(rows: dict[str, dict], factors: list[str], chances: dict[str, float],
                 title: str = "Frozen-feature probes on the known factors") -> str:
    """Grouped bars, one group per generative factor, with each group's own chance line.

    `bar_chart` takes a single floor, and these factors have different numbers of
    classes (3 colours, 3 shapes, 8 directions), so a single dashed line across the
    whole plot would be wrong for two of the three groups.
    """
    fig, ax = _axes(title, "", "linear probe accuracy", size=(7.0, 3.4))
    colors = ["#00b894", "#fdcb6e", "#74b9ff", "#e17055", "#a29bfe"]
    arms = list(rows)
    width = 0.8 / max(len(arms), 1)
    xs = np.arange(len(factors))
    for i, arm in enumerate(arms):
        pos = xs - 0.4 + width * (i + 0.5)
        vals = [rows[arm][f]["acc"] for f in factors]
        ax.bar(pos, vals, width * 0.88, label=arm, color=colors[i % len(colors)])
        for xv, v in zip(pos, vals):
            ax.text(xv, v, f"{v:.2f}", ha="center", va="bottom", color="#ccc", fontsize=6.5)
    for j, f in enumerate(factors):
        ax.plot([j - 0.42, j + 0.42], [chances[f]] * 2, ls="--", lw=1.4, color="#888",
                label="chance" if j == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(factors, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#ccc", ncol=2)
    return _fig_html(fig, width=700)


def scatter_chart(title: str, xlabel: str, ylabel: str, xs: list[float], ys: list[float],
                  labels: list[str] | None = None) -> str:
    """A scatter with a least-squares line and Pearson r in the title."""
    fig, ax = _axes(title, xlabel, ylabel, size=(6.0, 3.4))
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    ax.scatter(x, y, s=34, color="#00b894", zorder=3)
    if len(x) > 1 and x.std() > 0:
        m, c = np.polyfit(x, y, 1)
        xr = np.linspace(x.min(), x.max(), 8)
        ax.plot(xr, m * xr + c, ls="--", lw=1.4, color="#fdcb6e")
    for i, lab in enumerate(labels or []):
        ax.annotate(lab, (x[i], y[i]), fontsize=6.5, color="#888",
                    textcoords="offset points", xytext=(4, 4))
    return _fig_html(fig, width=560)


def pearson(x: list[float], y: list[float]) -> float:
    a, b = np.asarray(x, float), np.asarray(y, float)
    if len(a) < 2 or a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])
