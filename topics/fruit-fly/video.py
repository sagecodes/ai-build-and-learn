"""The cockpit: everything the fly is doing, in one frame, four panels.

    +---------------------------+---------------------------+
    |  the fly                  |  its brain                |
    |  (tracking camera)        |  138,639 neurons, glowing |
    |                           +---------------------------+
    |                           |  descending command       |
    +-------------+-------------+---------------------------+
    | left eye    | right eye   |  where it has been        |
    | 721 facets  | 721 facets  |  (overhead, live)         |
    +-------------+-------------+---------------------------+

Every panel is the same instant: body.py renders exactly one camera frame per 15 ms
brain tick, and the brain canvas, eye views and map are all built from that same tick.
So when the pillar slides across the left eye's facets, the left optic lobe lights up
in the panel next to it, the descending bars tip, and the track on the map bends. That
is the entire thesis of the demo and it is one image.

Encoding is PyAV rather than imageio-ffmpeg, matching the videogen and rl-mujoco demos:
PyAV ships manylinux aarch64 wheels and needs no separate ffmpeg binary. The output is
H.264 in an mp4 that a `<video>` tag plays inline in a Flyte report with no javascript.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np

_BG = (14, 14, 30)
_PANEL = (22, 22, 44)
_TEXT = (205, 205, 220)
_DIM = (130, 130, 150)
_ACCENT = (0, 184, 148)
_HILITE = (253, 203, 110)
_LEFT_C = (108, 176, 255)
_RIGHT_C = (255, 140, 110)
_TRACK = (0, 184, 148)

# Panel geometry. Fixed, because a video whose layout moves between frames is
# unwatchable and because every panel then has a known place in the README.
W, H = 960, 636
_BODY = (8, 22, 464, 332)
_EYE_L = (8, 380, 228, 248)
_EYE_R = (244, 380, 228, 248)
_BRAIN = (480, 22, 472, 196)
_CMD = (480, 240, 472, 114)
_MAP = (480, 380, 472, 248)


@dataclass
class Status:
    """The numbers drawn in the command panel, once per tick."""

    tick: int
    t: float                        # brain clock, seconds
    drive: tuple[float, float]      # left, right descending command
    dn: tuple[float, float]         # left, right descending population, Hz
    imbalance: float
    distance: float                 # mm to the object
    bearing: float                  # degrees, + is object to the fly's left
    spikes: int
    feeding: float = 0.0            # proboscis motor drive, 0..1
    food: str = "none"
    startle: float = 0.0            # common-mode descending excess, 0..1
    time_to_contact: float = float("nan")   # seconds until a looming object arrives


# ── numpy drawing primitives ────────────────────────────────────────────────────


def _resize(frame: np.ndarray, width: int, height: int | None = None) -> np.ndarray:
    """Nearest-neighbour resize. No scipy, no PIL, and fast enough at 25 fps."""
    h, w = frame.shape[:2]
    height = height or max(1, int(round(h * width / w)))
    yi = (np.arange(height) * h // height).clip(0, h - 1)
    xi = (np.arange(width) * w // width).clip(0, w - 1)
    return frame[yi][:, xi]


def _fit(frame: np.ndarray, box: tuple[int, int, int, int]) -> tuple[np.ndarray, int, int]:
    """Scale to fit inside (x, y, w, h) preserving aspect; return it and its offset."""
    _, _, bw, bh = box
    h, w = frame.shape[:2]
    scale = min(bw / w, bh / h)
    out = _resize(frame, max(1, int(w * scale)), max(1, int(h * scale)))
    return out, (bw - out.shape[1]) // 2, (bh - out.shape[0]) // 2


def _paste(canvas: np.ndarray, frame: np.ndarray, box: tuple[int, int, int, int]) -> None:
    x, y, bw, bh = box
    canvas[y:y + bh, x:x + bw] = _PANEL
    fitted, dx, dy = _fit(frame, box)
    fh, fw = fitted.shape[:2]
    canvas[y + dy:y + dy + fh, x + dx:x + dx + fw] = fitted


def _text(canvas: np.ndarray, xy: tuple[int, int], text: str, color, size: int = 12) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont

        pil = Image.fromarray(canvas)
        draw = ImageDraw.Draw(pil)
        try:
            font = ImageFont.load_default(size=size)
        except TypeError:      # Pillow < 10 takes no size argument
            font = ImageFont.load_default()
        draw.text(xy, text, fill=tuple(color), font=font)
        canvas[:] = np.asarray(pil)
    except Exception:  # noqa: BLE001
        pass


def _bar(canvas: np.ndarray, x: int, y: int, w: int, h: int, frac: float, color) -> None:
    canvas[y:y + h, x:x + w] = (44, 44, 68)
    filled = int(w * float(np.clip(frac, 0, 1)))
    if filled > 0:
        canvas[y:y + h, x:x + filled] = color


def _line(canvas: np.ndarray, p0, p1, color, width: int = 1) -> None:
    x0, y0 = p0
    x1, y1 = p1
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.linspace(x0, x1, steps).astype(int)
    ys = np.linspace(y0, y1, steps).astype(int)
    h, w = canvas.shape[:2]
    for dx in range(width):
        for dy in range(width):
            xi = np.clip(xs + dx, 0, w - 1)
            yi = np.clip(ys + dy, 0, h - 1)
            canvas[yi, xi] = color


def _disc(canvas: np.ndarray, cx: int, cy: int, r: int, color) -> None:
    h, w = canvas.shape[:2]
    y0, y1 = max(0, cy - r), min(h, cy + r + 1)
    x0, x1 = max(0, cx - r), min(w, cx + r + 1)
    if y1 <= y0 or x1 <= x0:
        return
    yy, xx = np.ogrid[y0:y1, x0:x1]
    canvas[y0:y1, x0:x1][(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = color


# ── panels ──────────────────────────────────────────────────────────────────────


def eye_view(hex_reading: np.ndarray) -> np.ndarray:
    """One compound eye's 721 facets as an amber hexagonal image.

    `flygym.vision.Retina.hex_pxls_to_human_readable` returns two channels, the pale
    and yellow ommatidia types (the fly has two spectral classes of photoreceptor).
    They are combined with a max here: the demo's retina reads luminance, not colour,
    and a two-channel image cannot be shown as one anyway.
    """
    mono = np.asarray(hex_reading)
    if mono.ndim == 3:
        mono = mono.max(axis=2)
    mono = mono.astype(np.float32) / 255.0
    return np.stack([mono * 255, mono * 190, mono * 70], axis=-1).astype(np.uint8)


def _command_panel(canvas: np.ndarray, status: Status) -> None:
    x, y, w, h = _CMD
    canvas[y:y + h, x:x + w] = _PANEL
    _text(canvas, (x + 8, y + 6), "descending command", _HILITE, 11)
    _text(canvas, (x + 8, y + 26), f"L {status.dn[0]:6.0f} Hz", _LEFT_C, 12)
    _bar(canvas, x + 108, y + 29, 200, 9, status.drive[0] / 1.6, _LEFT_C)
    _text(canvas, (x + 316, y + 26), f"{status.drive[0]:.2f}", _LEFT_C, 12)
    _text(canvas, (x + 8, y + 46), f"R {status.dn[1]:6.0f} Hz", _RIGHT_C, 12)
    _bar(canvas, x + 108, y + 49, 200, 9, status.drive[1] / 1.6, _RIGHT_C)
    _text(canvas, (x + 316, y + 46), f"{status.drive[1]:.2f}", _RIGHT_C, 12)

    # The imbalance, centre-zero, because its SIGN is the decision.
    mid = x + 208
    _text(canvas, (x + 8, y + 70), "imbalance", _DIM, 11)
    canvas[y + 76:y + 86, x + 108:x + 308] = (44, 44, 68)
    canvas[y + 74:y + 88, mid:mid + 1] = (90, 90, 120)
    span = int(np.clip(status.imbalance, -1, 1) * 100)
    if span >= 0:
        canvas[y + 76:y + 86, mid:mid + max(span, 1)] = _LEFT_C
    else:
        canvas[y + 76:y + 86, mid + span:mid] = _RIGHT_C
    _text(canvas, (x + 316, y + 70), f"{status.imbalance:+.3f}", _ACCENT, 12)
    _text(
        canvas, (x + 8, y + 92),
        f"t {status.t:5.2f}s   {status.spikes:,} spikes", _DIM, 11,
    )
    if np.isfinite(status.time_to_contact):
        # The swat. Time-to-contact is physics ground truth and the startle bar is what
        # the brain did about it, drawn side by side so the latency is readable off the
        # frame rather than only off a chart.
        hot = status.startle > 0.15
        ttc = min(status.time_to_contact, 9.99)
        label = "INCOMING" if status.distance > 3.0 else "CONTACT"
        _text(canvas, (x + 240, y + 6), f"{label}  ttc {ttc:+.2f}s",
              _HILITE if hot else _DIM, 11)
        _text(canvas, (x + 240, y + 92), f"startle {status.startle:4.2f}",
              _HILITE if hot else _DIM, 11)
        _bar(canvas, x + 380, y + 95, 80, 8, status.startle,
             _HILITE if hot else (90, 90, 120))
    elif status.food != "none":
        colour = _HILITE if status.feeding > 0.15 else _DIM
        label = "FEEDING" if status.feeding > 0.15 else "proboscis"
        _text(canvas, (x + 240, y + 92),
              f"{label} {status.feeding:4.2f}  ({status.food})", colour, 11)
        _bar(canvas, x + 380, y + 95, 80, 8, status.feeding, colour)


def _map_panel(
    canvas: np.ndarray,
    trajectory: list[tuple[float, float, float]],
    target_xy: tuple[float, float],
    target_r: float,
    box: tuple[int, int, int, int] | None = None,
) -> None:
    x, y, w, h = box or _MAP
    canvas[y:y + h, x:x + w] = _PANEL
    _text(canvas, (x + 8, y + 6), "where it has been (mm, overhead)", _HILITE, 11)

    pts = [(px, py) for px, py, _ in trajectory] + [target_xy, (0.0, 0.0)]
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    margin = target_r + 4
    lo = min(min(xs), min(ys)) - margin
    hi = max(max(xs), max(ys)) + margin
    span = max(hi - lo, 1e-6)
    pad = 26

    def sx(v):
        return int(x + pad + (w - 2 * pad) * (v - lo) / span)

    def sy(v):  # +y is the fly's left, and points UP, as on a map
        return int(y + h - pad - (h - 2 * pad - 12) * (v - lo) / span)

    _disc(canvas, sx(target_xy[0]), sy(target_xy[1]),
          max(int((w - 2 * pad) * target_r / span), 3), (120, 120, 130))
    _disc(canvas, sx(0.0), sy(0.0), 3, _DIM)
    for i in range(1, len(trajectory)):
        _line(canvas,
              (sx(trajectory[i - 1][0]), sy(trajectory[i - 1][1])),
              (sx(trajectory[i][0]), sy(trajectory[i][1])), _TRACK, 2)
    if trajectory:
        fx, fy, heading = trajectory[-1]
        cx, cy = sx(fx), sy(fy)
        _disc(canvas, cx, cy, 4, _HILITE)
        # A stub in the direction the fly is actually pointing, which is not the same
        # as the direction it has been travelling and is the more interesting one.
        rad = np.deg2rad(heading)
        _line(canvas, (cx, cy),
              (int(cx + 16 * np.cos(rad)), int(cy - 16 * np.sin(rad))), _HILITE, 2)


def compose(
    body: np.ndarray,
    brain: np.ndarray,
    eyes: tuple[np.ndarray, np.ndarray] | None,
    status: Status,
    trajectory: list[tuple[float, float, float]],
    target_xy: tuple[float, float],
    target_r: float,
) -> np.ndarray:
    """One cockpit frame."""
    canvas = np.zeros((H, W, 3), np.uint8)
    canvas[:] = _BG

    _text(canvas, (8, 4), "the fly", _HILITE, 12)
    _paste(canvas, body, _BODY)

    _text(canvas, (480, 4), "its brain  -  138,639 neurons, FlyWire 783", _HILITE, 12)
    _paste(canvas, brain, _BRAIN)

    _text(canvas, (8, 362), "left eye  -  721 facets", _LEFT_C, 11)
    _text(canvas, (244, 362), "right eye", _RIGHT_C, 11)
    if eyes is not None:
        _paste(canvas, eyes[0], _EYE_L)
        _paste(canvas, eyes[1], _EYE_R)
    else:
        canvas[_EYE_L[1]:_EYE_L[1] + _EYE_L[3], _EYE_L[0]:_EYE_L[0] + _EYE_L[2]] = _PANEL
        canvas[_EYE_R[1]:_EYE_R[1] + _EYE_R[3], _EYE_R[0]:_EYE_R[0] + _EYE_R[2]] = _PANEL

    _command_panel(canvas, status)
    _map_panel(canvas, trajectory, target_xy, target_r)

    _text(
        canvas, (W - 250, 6),
        f"{status.distance:5.1f} mm   bearing {status.bearing:+6.1f} deg",
        _HILITE if abs(status.bearing) < 25 else _TEXT, 12,
    )
    return canvas


# ── output ──────────────────────────────────────────────────────────────────────


def encode(frames: list[np.ndarray], fps: int = 25, crf: int = 23) -> bytes:
    """H.264 mp4 bytes from a list of RGB frames.

    Dimensions are forced even: libx264 with yuv420p rejects odd width or height.
    """
    import av

    if not frames:
        raise ValueError("no frames to encode")
    height, width = frames[0].shape[:2]
    width -= width % 2
    height -= height % 2

    buffer = io.BytesIO()
    with av.open(buffer, mode="w", format="mp4") as container:
        stream = container.add_stream("libx264", rate=fps)
        stream.width, stream.height = width, height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf), "preset": "medium"}
        for frame in frames:
            image = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frame[:height, :width]), format="rgb24"
            )
            for packet in stream.encode(image):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def filmstrip(frames: list[np.ndarray], count: int = 5, width: int = 300) -> np.ndarray:
    """Evenly spaced stills in a row, for when a video will not play."""
    if not frames:
        raise ValueError("no frames")
    picks = np.linspace(0, len(frames) - 1, min(count, len(frames))).astype(int)
    tiles = [_resize(frames[i], width) for i in picks]
    gap = np.full((tiles[0].shape[0], 4, 3), 30, np.uint8)
    out = []
    for i, tile in enumerate(tiles):
        out.append(tile)
        if i < len(tiles) - 1:
            out.append(gap)
    return np.concatenate(out, axis=1)


# ── Two flies ───────────────────────────────────────────────────────────────────

DW, DH = 960, 660
_D_CAM_A = (8, 22, 464, 300)
_D_CAM_B = (488, 22, 464, 300)
_D_BRAIN_A = (8, 344, 464, 150)
_D_BRAIN_B = (488, 344, 464, 150)
_D_MAP = (8, 516, 464, 136)
_D_CMD = (488, 516, 464, 136)


def compose_duet(
    body_a: np.ndarray,
    body_b: np.ndarray,
    brain_a: np.ndarray,
    brain_b: np.ndarray,
    status_a: Status,
    status_b: Status,
    traj_a: list,
    traj_b: list,
    target_xy: tuple[float, float],
    target_r: float,
    separation: float,
) -> np.ndarray:
    """One frame of two flies: both points of view, both brains, one map."""
    canvas = np.zeros((DH, DW, 3), np.uint8)
    canvas[:] = _BG

    _text(canvas, (8, 4), "fly A  -  its own brain", _LEFT_C, 12)
    _text(canvas, (488, 4), "fly B  -  a second, independent brain", _RIGHT_C, 12)
    _paste(canvas, body_a, _D_CAM_A)
    _paste(canvas, body_b, _D_CAM_B)

    _text(canvas, (8, 330), "brain A  -  138,639 neurons", _LEFT_C, 11)
    _text(canvas, (488, 330), "brain B", _RIGHT_C, 11)
    _paste(canvas, brain_a, _D_BRAIN_A)
    _paste(canvas, brain_b, _D_BRAIN_B)

    # One map, both tracks, so "who got there first" is a single picture.
    x, y, w, h = _D_MAP
    canvas[y:y + h, x:x + w] = _PANEL
    _text(canvas, (x + 8, y + 6), "both flies, overhead (mm)", _HILITE, 11)
    pts = [(px, py) for px, py, _ in list(traj_a) + list(traj_b)] + [target_xy]
    xs = [p[0] for p in pts] or [0.0]
    ys = [p[1] for p in pts] or [0.0]
    margin = target_r + 4
    lo = min(min(xs), min(ys)) - margin
    hi = max(max(xs), max(ys)) + margin
    span = max(hi - lo, 1e-6)
    pad = 22

    def sx(v):
        return int(x + pad + (w - 2 * pad) * (v - lo) / span)

    def sy(v):
        return int(y + h - pad - (h - 2 * pad - 10) * (v - lo) / span)

    _disc(canvas, sx(target_xy[0]), sy(target_xy[1]),
          max(int((w - 2 * pad) * target_r / span), 3), (120, 120, 130))
    for traj, colour in ((traj_a, _LEFT_C), (traj_b, _RIGHT_C)):
        for i in range(1, len(traj)):
            _line(canvas, (sx(traj[i - 1][0]), sy(traj[i - 1][1])),
                  (sx(traj[i][0]), sy(traj[i][1])), colour, 2)
        if traj:
            _disc(canvas, sx(traj[-1][0]), sy(traj[-1][1]), 4, colour)

    x, y, w, h = _D_CMD
    canvas[y:y + h, x:x + w] = _PANEL
    _text(canvas, (x + 8, y + 6), "descending commands", _HILITE, 11)
    for row, (status, colour, label) in enumerate(
        ((status_a, _LEFT_C, "A"), (status_b, _RIGHT_C, "B"))
    ):
        top = y + 26 + row * 46
        _text(canvas, (x + 8, top), f"{label}  {status.dn[0]:5.0f}/{status.dn[1]:5.0f} Hz",
              colour, 11)
        _bar(canvas, x + 150, top + 3, 150, 8, status.drive[0] / 1.6, colour)
        _bar(canvas, x + 150, top + 15, 150, 8, status.drive[1] / 1.6, colour)
        _text(canvas, (x + 312, top), f"{status.distance:5.1f} mm", _TEXT, 11)
        _text(canvas, (x + 312, top + 16), f"bearing {status.bearing:+5.0f}", _DIM, 11)
    _text(canvas, (x + 8, y + h - 22),
          f"t {status_a.t:5.2f}s    the two flies are {separation:5.1f} mm apart",
          _DIM, 11)
    return canvas


# ── Learning ────────────────────────────────────────────────────────────────────

LW, LH = 960, 620
_L_BRAIN_A = (8, 44, 464, 210)
_L_BRAIN_B = (488, 44, 464, 210)
_L_MB_A = (8, 286, 464, 250)
_L_MB_B = (488, 286, 464, 250)


def compose_learning(
    brain_before: np.ndarray,
    brain_after: np.ndarray,
    mb_before: np.ndarray,
    mb_after: np.ndarray,
    tick: int,
    t: float,
    hz_before: float,
    hz_after: float,
    peak_hz: float,
    odor: str,
    label_before: str = "before training",
    label_after: str = "after training",
) -> np.ndarray:
    """One frame of the same odour presented to the same brain, before and after.

    Left column is the naive fly, right column is the same animal after the odour was
    paired with dopamine. Top row is the whole brain for context, bottom row is the
    mushroom body on its own, which is where the change actually is: 5,177 Kenyon cells
    and 96 output neurons that are a few dozen pixels of a whole-brain view.
    """
    canvas = np.zeros((LH, LW, 3), np.uint8)
    canvas[:] = _BG

    _text(canvas, (8, 6), f"odour {odor}  -  the same stimulus, the same brain", _HILITE, 13)
    _text(canvas, (8, 26), label_before, _LEFT_C, 12)
    _text(canvas, (488, 26), label_after, _RIGHT_C, 12)

    _paste(canvas, brain_before, _L_BRAIN_A)
    _paste(canvas, brain_after, _L_BRAIN_B)
    _text(canvas, (8, 266),
          "the 96 mushroom body OUTPUT neurons  -  the only thing training changed",
          _DIM, 11)
    _text(canvas, (488, 266), "same cells, same odour, after being punished", _DIM, 11)
    _paste(canvas, mb_before, _L_MB_A)
    _paste(canvas, mb_after, _L_MB_B)

    # The readout bars, on a shared scale so the two sides are comparable by eye.
    top = LH - 70
    scale = max(peak_hz, 1.0)
    for x, hz, colour, label in (
        (8, hz_before, _LEFT_C, label_before),
        (488, hz_after, _RIGHT_C, label_after),
    ):
        _text(canvas, (x, top), f"mushroom body output  {hz:6.0f} Hz", colour, 12)
        _bar(canvas, x, top + 20, 464, 14, hz / scale, colour)
    _text(canvas, (8, LH - 22), f"t {t:5.2f}s   tick {tick}", _DIM, 11)
    if hz_before > 0:
        # Spell out the direction. A signed percentage here reads as a gain to half the
        # people who see it, and the whole point is that the output got quieter.
        drop = 100.0 * (hz_before - hz_after) / hz_before
        word = "quieter" if drop >= 0 else "louder"
        _text(canvas, (488, LH - 22), f"{abs(drop):.0f}% {word} on this tick", _HILITE, 11)
    return canvas


# ── The fly riding a robot ──────────────────────────────────────────────────────

RW, RH = 960, 600
_R_BODY = (8, 22, 560, 400)
_R_BRAIN = (576, 22, 376, 190)
_R_CMD = (576, 226, 376, 120)
_R_MAP = (576, 358, 376, 234)


def compose_ride(
    scene: np.ndarray,
    brain: np.ndarray,
    status: Status,
    trajectory: list,
    target_xy: tuple[float, float],
    target_r: float,
) -> np.ndarray:
    """One frame of a fly driving a four-legged robot."""
    canvas = np.zeros((RH, RW, 3), np.uint8)
    canvas[:] = _BG

    _text(canvas, (8, 4), "a fly riding a robot it is steering", _HILITE, 13)
    _paste(canvas, scene, _R_BODY)

    _text(canvas, (576, 4), "the fly's brain  -  138,639 neurons", _HILITE, 12)
    _paste(canvas, brain, _R_BRAIN)

    x, y, w, h = _R_CMD
    canvas[y:y + h, x:x + w] = _PANEL
    _text(canvas, (x + 8, y + 6), "what the fly is telling the robot", _HILITE, 11)
    # status.drive carries (speed, turn) for a ride, not (left, right).
    speed, turn = status.drive
    _text(canvas, (x + 8, y + 28), f"speed {speed:5.2f}", _ACCENT, 12)
    _bar(canvas, x + 110, y + 31, 150, 9, speed / 1.2, _ACCENT)
    _text(canvas, (x + 8, y + 50), "turn", _TEXT, 12)
    mid = x + 185
    canvas[y + 53:y + 63, x + 110:x + 260] = (44, 44, 68)
    canvas[y + 51:y + 65, mid:mid + 1] = (90, 90, 120)
    span = int(np.clip(turn, -1, 1) * 74)
    colour = _LEFT_C if span >= 0 else _RIGHT_C
    if span >= 0:
        canvas[y + 53:y + 63, mid:mid + max(span, 1)] = colour
    else:
        canvas[y + 53:y + 63, mid + span:mid] = colour
    _text(canvas, (x + 268, y + 50), f"{turn:+.2f}", colour, 12)
    _text(canvas, (x + 8, y + 74),
          f"descending  L {status.dn[0]:5.0f}  R {status.dn[1]:5.0f} Hz", _DIM, 11)
    _text(canvas, (x + 8, y + 94),
          f"t {status.t:5.2f}s   {status.distance:5.0f} mm to go   "
          f"bearing {status.bearing:+5.0f}", _DIM, 11)

    _map_panel(canvas, trajectory, target_xy, target_r, box=_R_MAP)
    return canvas
