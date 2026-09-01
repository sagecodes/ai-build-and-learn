"""Read the videos DreamerV3 already writes, so the report can show them as it trains.

There is no extra rendering, no second process and no checkpoint reloading anywhere in
this file. DreamerV3's logger has a `scope` output enabled by default, and `scope`
writes every 4-D uint8 array it is handed as an h264 mp4 on disk. Two of the things the
agent logs are exactly the two videos this demo wants:

    <logdir>/scope/report-openloop-image.mp4/<step>-<id>.mp4    the dream
    <logdir>/scope/epstats-policy_image.mp4/<step>-<id>.mp4     what really happened

So "render video periodically during a long run" costs one directory listing per
repaint. The clips are already there; the training loop wrote them.

── What the two columns are ────────────────────────────────────────────────────
**The dream** comes from `Agent.report()`. It takes a batch of six real sequences,
lets the world model watch the first half, and then makes it predict the second half
from actions alone with the images hidden. The grid is three rows: the true frames on
top, the model's reconstruction in the middle, the difference at the bottom. The
border is green while the model is still being shown observations and turns red the
moment it goes blind. Everything after the red line is imagination.

**The rollout** is the actual policy in the actual environment, stacked frame by frame
by the training loop's own episode logger. This is the honest one. Episode return can
climb for reasons that have nothing to do with the task; this shows what the body is
really doing, and because the arena has a row of posts in the background you can tell
walking from shuffling at a glance.

── Why the frames get scaled up with np.repeat ─────────────────────────────────
The agent sees 64x64. Embedded at native size in a report that is a postage stamp, and
scaled up by the browser it turns into a blur. `np.repeat` is nearest-neighbour, so a
pixel becomes a hard square block and the picture stays exactly as sharp as the data
really is. Anything smoother would be inventing detail the model did not predict, in a
report whose entire purpose is showing how good the predictions are.
"""

from __future__ import annotations

import base64
import io
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# scope names a column after the metric key with '/' replaced by '-', plus the format
# extension. `report/openloop/image` and `epstats/policy_image` are the keys logged by
# dreamerv3/agent.py and embodied/run/train.py respectively.
DREAM = "report-openloop-image.mp4"
ROLLOUT = "epstats-policy_image.mp4"
# With `--configs dmc_proprio` the image is not an agent observation, so embodied
# renames it to `log/image` and there is no dream at all, only this rollout.
ROLLOUT_PROPRIO = "epstats-policy_log-image.mp4"


def clips(logdir: Path, column: str) -> list[tuple[int, Path]]:
    """Every clip written for one column, oldest first, as (step, path).

    Filenames are `{step:020}-{id}.mp4`, so lexical order is step order. The directory
    also holds a small `index` file that scope maintains, hence the suffix filter.
    """
    folder = Path(logdir) / "scope" / column
    if not folder.is_dir():
        return []
    out = []
    for path in sorted(folder.iterdir()):
        if path.suffix != ".mp4" or "-" not in path.name:
            continue
        try:
            out.append((int(path.name.split("-", 1)[0]), path))
        except ValueError:
            continue
    return out


def decode(path: Path) -> np.ndarray:
    """mp4 file -> (T, H, W, 3) uint8."""
    import av

    with av.open(str(path)) as container:
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(video=0)]
    return np.stack(frames) if frames else np.zeros((0, 1, 1, 3), np.uint8)


def trim_black(frames: np.ndarray) -> np.ndarray:
    """Drop the black frames the agent appends as a separator.

    `Agent.report` ends its grid with `0 * video[:, :10]`, ten black frames that mark
    where the clip loops. Useful in a video player, useless as the last still in a
    filmstrip, and misleading in a luminance probe.
    """
    if not len(frames):
        return frames
    bright = [i for i, f in enumerate(frames) if f.max() > 8]
    return frames[: bright[-1] + 1] if bright else frames


def encode(frames: np.ndarray, fps: int = 10, scale: int = 1) -> bytes:
    """Frames -> h264 mp4 bytes, optionally block-scaled up first.

    h264 requires even dimensions, and the failure when they are odd happens at
    encoder open with a message that never mentions the size, so the frame is cropped
    rather than trusted.
    """
    import av

    if not len(frames):
        return b""
    if scale > 1:
        frames = np.repeat(np.repeat(frames, scale, axis=1), scale, axis=2)
    h, w = frames.shape[1:3]
    h, w = h - h % 2, w - w % 2
    buf = io.BytesIO()
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "veryfast"}
        for frame in frames:
            arr = np.ascontiguousarray(frame[:h, :w], dtype=np.uint8)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return buf.getvalue()


def png(frame: np.ndarray, scale: int = 1) -> bytes:
    if scale > 1:
        frame = np.repeat(np.repeat(frame, scale, axis=0), scale, axis=1)
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(frame, dtype=np.uint8)).save(buf, "PNG")
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def luminance(frames: np.ndarray) -> str:
    """Say whether a clip actually shows anything.

    A black clip is a valid mp4 and a report embeds it as a black rectangle without
    complaint, which is how the Isaac Sim demo lost an afternoon to a renderer that
    had silently failed. Cheap to check, so always checked.
    """
    if not len(frames):
        return "no frames"
    means = frames.reshape(len(frames), -1).mean(axis=1)
    return (
        f"{len(frames)} frames, luminance min {means.min():.1f} / "
        f"mean {means.mean():.1f} / max {means.max():.1f}, "
        f"{int((means < 1.0).sum())} black"
    )


# ── The two harvests ────────────────────────────────────────────────────────────


def dream(logdir: Path) -> dict | None:
    """The newest open-loop prediction grid, plus the single most telling still.

    The still is the LAST frame of the imagined half of the first sequence. Last,
    because prediction error compounds and the end of the horizon is where a weak
    model has visibly fallen apart. First sequence only, because six side by side is
    unreadable once eight of these are lined up as a filmstrip.
    """
    found = clips(logdir, DREAM)
    if not found:
        return None
    step, path = found[-1]
    frames = trim_black(decode(path))
    if not len(frames):
        return None
    # The grid is (T, 3 * 68, B * 68): three rows true/pred/error, one column per
    # sequence. One column is 68 px wide including scope's 2 px border on each side.
    column = frames.shape[2] // 6 if frames.shape[2] >= 6 * 60 else frames.shape[2]
    return {
        "step": step,
        "count": len(found),
        "mp4": encode(frames, fps=8, scale=2),
        "still": png(frames[-1, :, :column], scale=2),
        "probe": luminance(frames),
        "frames": len(frames),
    }


def rollout(logdir: Path, tail: int = 400) -> dict | None:
    """The newest real-environment episode footage from the training loop itself.

    `tail` trims to the last N frames. The episode logger concatenates every episode
    finished since the previous log flush, which on a long run is thousands of frames
    of mostly the same thing; the end is the most recent behaviour and the only part
    worth embedding.
    """
    found = clips(logdir, ROLLOUT) or clips(logdir, ROLLOUT_PROPRIO)
    if not found:
        return None
    step, path = found[-1]
    frames = decode(path)
    if not len(frames):
        return None
    frames = frames[-tail:]
    return {
        "step": step,
        "count": len(found),
        "mp4": encode(frames, fps=40, scale=4),  # 40 Hz is the DMC control rate
        "still": png(frames[-1], scale=3),
        "probe": luminance(frames),
        "frames": len(frames),
    }
