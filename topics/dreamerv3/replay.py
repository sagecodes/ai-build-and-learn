"""Film a trained DreamerV3 policy acting in the real environment.

Dreamer trains on proprioception here, so there is no image anywhere in the agent's
observation space and nothing to save as a video by accident. The clip therefore comes
from rendering the MuJoCo scene *alongside* the policy: the agent sees the same state
vector it was trained on, and we ask the underlying dm_control physics for a camera
frame at every step purely for the report. Rendering does not change what the agent
observes, so the video is an honest recording of the policy rather than a different
run.

Two consequences worth knowing:

  * The environments run IN PROCESS (`parallel=False`). embodied's Driver defaults to
    running each env in its own subprocess, and a subprocess's `physics` object is not
    reachable from here, so a parallel driver can be stepped but not filmed.

  * Rendering needs a working GL stack in the pod. `MUJOCO_GL=egl` plus the NVIDIA
    graphics driver libraries, which the flyte devbox does not inject by default. See
    the note in config.py and the full diagnosis in topics/isaac-sim/README.md.
"""

from __future__ import annotations

import base64
import logging
from functools import partial as bind
from pathlib import Path

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def _find_physics(env):
    """Walk embodied's wrapper chain down to dm_control's physics object.

    make_env returns the DMC env wrapped in several embodied wrappers, and the
    attribute holding the wrapped env is not consistently named, so this looks for the
    dm_control env rather than assuming a path.
    """
    seen = set()
    stack = [env]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        dmenv = getattr(obj, "_dmenv", None)
        if dmenv is not None and hasattr(dmenv, "physics"):
            return dmenv.physics
        for attr in ("_env", "env", "_envs"):
            inner = getattr(obj, attr, None)
            if inner is not None:
                stack.append(inner)
    return None


def record(logdir: Path, steps: int = 500, size: tuple[int, int] = (480, 480)):
    """Run the trained policy for `steps` and return (frames, score, metres).

    Returns ([], 0.0, 0.0) rather than raising if anything about rendering fails: a run
    that trained successfully must not be failed by the camera.
    """
    import arena  # noqa: F401  registers the custom domain before make_env resolves it
    import elements
    import embodied
    import ruamel.yaml as yaml
    from dreamerv3 import main as dv3main

    config = elements.Config(
        yaml.YAML(typ="safe").load((logdir / "config.yaml").read_text())
    )

    agent = dv3main.make_agent(config)
    # Point Checkpoint at the DIRECTORY and let it resolve which one to load. The dir
    # holds timestamped checkpoints plus a 22-byte `latest` pointer file; picking the
    # last entry by name grabs that pointer and dies on `assert exists(path)`, because
    # it is a file describing a checkpoint rather than one.
    cp = elements.Checkpoint(logdir / "ckpt")
    cp.agent = agent
    cp.load(keys=["agent"])

    driver = embodied.Driver([bind(dv3main.make_env, config, 0)], parallel=False)
    physics = _find_physics(driver.envs[0])
    if physics is None:
        log.warning("could not reach dm_control physics; no video")
        return [], 0.0, 0.0

    frames: list = []
    score = [0.0]
    # Where the torso started and how far ahead of that it ever got. Reported next to
    # the return for the same reason arena.py logs it during training: a score is a
    # claim, metres travelled is a fact. `xpos` is read straight from the physics, not
    # from the agent, so a policy cannot influence it except by actually moving.
    start = [None]
    far = [0.0]

    def on_step(tran, _worker):
        score[0] += float(tran["reward"])
        x = float(physics.named.data.xpos["torso", "x"])
        if start[0] is None or tran["is_first"]:
            start[0] = x
        far[0] = max(far[0], x - start[0])
        if len(frames) < steps:
            frames.append(physics.render(*size, camera_id=0))

    driver.on_step(on_step)
    driver.reset(agent.init_policy)
    driver(lambda *a: agent.policy(*a, mode="eval"), steps=steps)
    log.info(
        "recorded %s frames, episode return %.1f, travelled %.2f m",
        len(frames), score[0], far[0],
    )
    return frames, score[0], far[0]


def encode(frames: list, fps: int = 30) -> bytes:
    """Frames to H.264 mp4 bytes, small enough to base64 into a report.

    PyAV rather than imageio-ffmpeg, for the same reason as the video-generation and
    Isaac Sim demos: aarch64 wheels exist for it and reliably do not for the other.
    """
    if not frames:
        return b""
    import io

    import av
    import numpy as np

    buf = io.BytesIO()
    h, w = frames[0].shape[:2]
    with av.open(buf, "w", format="mp4") as out:
        stream = out.add_stream("libx264", rate=fps)
        stream.width, stream.height = w, h
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "28", "preset": "veryfast"}
        for frame in frames:
            arr = np.ascontiguousarray(frame, dtype=np.uint8)
            for pkt in stream.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                out.mux(pkt)
        for pkt in stream.encode():
            out.mux(pkt)
    return buf.getvalue()


def probe(mp4: bytes) -> str:
    """Say whether the clip actually shows anything.

    A black clip is a valid mp4 that a report will happily embed as a black rectangle,
    which is exactly how the Isaac Sim demo lost an afternoon. Mean luminance separates
    a real render from a renderer that silently produced nothing.
    """
    if not mp4:
        return "no clip"
    try:
        import io

        import av
        with av.open(io.BytesIO(mp4)) as c:
            means = [float(f.to_ndarray(format="gray").mean()) for f in c.decode(video=0)]
        if not means:
            return "clip decodes to ZERO frames"
        dark = sum(1 for m in means if m < 1.0)
        return (
            f"{len(means)} frames, luminance min {min(means):.1f} / mean "
            f"{sum(means) / len(means):.1f} / max {max(means):.1f}, {dark} black"
        )
    except Exception as exc:
        return f"probe failed: {exc}"


def video_html(mp4: bytes, caption: str = "") -> str:
    b64 = base64.b64encode(mp4).decode()
    cap = f'<p style="color:#888;margin:6px 0 0;">{caption}</p>' if caption else ""
    return (
        f'<div style="font-family:monospace;background:#0f0f23;padding:16px;border-radius:8px;">'
        f'<video src="data:video/mp4;base64,{b64}" controls autoplay loop muted playsinline '
        f'style="max-width:640px;width:100%;border:2px solid #333;border-radius:4px;display:block;">'
        f"</video>{cap}</div>"
    )
