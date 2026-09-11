"""A small MuJoCo scene, to give Cosmos Transfer the input it is actually designed for.

`restyle` first pointed at PushT, on the reasoning that it was the clip Cosmos Predict
failed on and therefore a nice narrative for what Transfer can do instead. That was
symmetry rather than sense. PushT is a flat 2D diagram: a white field with coloured
polygons, and there is no photorealistic counterpart for Transfer to move it toward.

Transfer exists for a specific shape of input, and it is this one: a scene that is
**three-dimensional and physically correct but looks synthetic**, which is what Isaac Sim
and Omniverse produce. Flat shading, hard edges, untextured primitives, a camera with no
lens character. That is what the sim2real pipeline actually has to convert.

Self-contained MJCF rather than a dependency on topics/rl-mujoco, which would drag in
mujoco_playground, brax and jax for a scene that needs none of them. Headless rendering
uses the EGL recipe that topic already worked out (MUJOCO_GL=egl plus a handful of mesa
apt packages); config.py carries both.

The other half of why this is the right input: **the simulator supplies the actions.**
That is the whole reason the Transfer path has no label tax. `cycle` measured a 3x cost
for actions recovered from generated video by inverse dynamics; here the actions are
ground truth by construction because we commanded them.
"""

from __future__ import annotations

import logging
import os

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# A pusher on a slide joint nudging a free block across a table. Deliberately plain:
# primitive geoms, flat colours, one light. It should look like a simulator, because
# looking like a simulator is the problem Transfer is being asked to solve.
SCENE = """
<mujoco model="push">
  <option timestep="0.004" gravity="0 0 -9.81"/>
  <visual>
    <headlight ambient="0.4 0.4 0.4" diffuse="0.7 0.7 0.7"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.9 0.9 0.9" rgb2="0.75 0.75 0.78"
             width="300" height="300"/>
    <material name="gridmat" texture="grid" texrepeat="6 6" reflectance="0.05"/>
    <material name="tablemat" rgba="0.85 0.85 0.88 1" reflectance="0.02"/>
    <material name="blockmat" rgba="0.15 0.35 0.85 1"/>
    <material name="pushermat" rgba="0.85 0.25 0.2 1"/>
    <material name="goalmat" rgba="0.2 0.75 0.35 0.4"/>
  </asset>
  <worldbody>
    <light pos="0.4 -0.4 1.2" dir="-0.3 0.3 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="gridmat"/>
    <geom name="table" type="box" pos="0 0 0.1" size="0.40 0.30 0.1" material="tablemat"/>
    <geom name="goal" type="box" pos="0.20 0.0 0.204" size="0.09 0.09 0.002" material="goalmat"/>
    <body name="block" pos="-0.10 0.0 0.27">
      <freejoint/>
      <geom name="block" type="box" size="0.065 0.065 0.065" material="blockmat" mass="0.2"/>
    </body>
    <body name="pusher" pos="-0.32 0.0 0.32">
      <joint name="px" type="slide" axis="1 0 0" range="-0.42 0.42"/>
      <joint name="py" type="slide" axis="0 1 0" range="-0.3 0.3"/>
      <geom name="pusher" type="capsule" fromto="0 0 -0.065 0 0 0.07" size="0.04"
            material="pushermat" mass="1"/>
    </body>
    <camera name="rig" pos="0.05 -0.70 0.62" mode="targetbody" target="block" fovy="42"/>
  </worldbody>
  <actuator>
    <position joint="px" kp="220" ctrlrange="-0.42 0.42"/>
    <position joint="py" kp="220" ctrlrange="-0.3 0.3"/>
  </actuator>
</mujoco>
"""


def rollout(frames: int = 29, width: int = 640, height: int = 384, substeps: int = 12):
    """Render a scripted push IN A CHILD PROCESS. Returns (rgb frames, commanded actions).

    The subprocess is not defensive style, it is required. Rendering here and then
    importing diffusers in the same process SEGFAULTS on this box: MuJoCo brings up an
    EGL context (with `libEGL warning: egl: failed to create dri2 screen` on the way),
    and the CUDA initialisation inside the torch import then dies on it. `faulthandler`
    being armed in world.py is the only reason that showed up as a traceback pointing at
    `import diffusers` rather than as a bare exit 139.

    topics/isaac-sim reached the identical conclusion for the identical reason and runs
    its simulator in a child process too. An EGL context that dies with the child cannot
    poison the parent's CUDA context, and the frames come back over a file.

    The actions returned are the position targets actually commanded, so they are ground
    truth rather than a recovered estimate. That is the point of sourcing the input from
    a simulator: whatever Transfer does to the pixels, these labels stay correct, because
    they were never inferred from pixels in the first place.
    """
    import subprocess
    import sys
    import tempfile

    import numpy as np
    from PIL import Image

    with tempfile.TemporaryDirectory() as tmp:
        out = f"{tmp}/scene.npz"
        proc = subprocess.run(
            [sys.executable, __file__, out, str(frames), str(width), str(height),
             str(substeps)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0 or not os.path.exists(out):
            raise RuntimeError(
                f"mujoco render subprocess failed ({proc.returncode}):\n"
                f"{proc.stderr[-2000:]}"
            )
        blob = np.load(out)
        rgb, depth, actions = blob["rgb"], blob["depth"], blob["actions"]

    out_frames = [Image.fromarray(f).convert("RGB") for f in rgb]
    depth_frames = [Image.fromarray(f).convert("RGB") for f in depth]
    log.info("mujoco: %s frames at %sx%s, actions %s, real depth buffer (child process)",
             len(out_frames), width, height, actions.shape)
    return out_frames, actions, depth_frames


def _render(path: str, frames: int, width: int, height: int, substeps: int) -> None:
    """The actual rendering. Only ever runs as __main__ in a child process."""
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

    import mujoco
    import numpy as np

    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    cam = mujoco.MjvCamera()
    cam.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "rig")
    cam.type = mujoco.mjtCamera.mjCAMERA_FIXED

    # A REAL depth buffer, which is the whole reason to source from a simulator. Canny
    # edges are a derived, lossy stand-in: measured across three sources they were sparse
    # enough to be ambiguous (PushT), dominated by furniture (a badly framed table), or so
    # dense on a cluttered real scene that Transfer simply reproduced the edge map. Depth
    # is what Isaac and Omniverse actually hand to Transfer, and a simulator can give it
    # exactly rather than infer it.
    depth_renderer = mujoco.Renderer(model, height=height, width=width)
    depth_renderer.enable_depth_rendering()

    rgb, depth, actions = [], [], []
    for i in range(frames):
        # Drive the pusher straight through the block, with a slight lateral drift so the
        # block rotates as well as translates. A pure translation is a weaker test: it
        # looks the same whether or not the geometry was preserved.
        t = i / max(frames - 1, 1)
        ctrl = np.array([-0.32 + 0.62 * t, 0.05 * np.sin(3.0 * t)], dtype=np.float32)
        data.ctrl[:] = ctrl
        for _ in range(substeps):
            mujoco.mj_step(model, data)
        renderer.update_scene(data, camera=cam)
        rgb.append(renderer.render().copy())
        depth_renderer.update_scene(data, camera=cam)
        depth.append(depth_renderer.render().copy())
        actions.append(ctrl)
    renderer.close()
    depth_renderer.close()

    # Normalise depth to 8-bit over the scene's own near/far, ignoring the infinite
    # background the renderer returns for rays that hit nothing. Left as raw metres it is
    # mostly a single saturated value and carries no usable structure.
    d = np.stack(depth)
    # Percentiles at BOTH ends, not min-to-99th. Using the raw minimum as the near plane
    # lets one stray pixel set the scale, and the first version of this came out almost
    # uniformly white (mean 225 of 255) and carried no usable structure at all. A depth
    # control signal with no contrast is worth less than no control signal, because the
    # model still has to honour it.
    # 10 metres, not 100. The renderer returns ~51 m for rays that hit nothing, and in
    # this scene that is 10% of the frame (the sky above the horizon). A threshold of 100
    # let those through, so the 98th percentile WAS the far plane and every real surface
    # got compressed into the top 2% of the range: a uniformly white image with a mean of
    # 225. The scene's actual geometry lives between 0.53 and 2.3 m.
    finite = d[np.isfinite(d) & (d < 10.0)]
    near, far = ((float(np.percentile(finite, 2)), float(np.percentile(finite, 98)))
                 if finite.size else (0.0, 1.0))
    d = np.clip((d - near) / max(far - near, 1e-6), 0.0, 1.0)
    d = ((1.0 - d) * 255).astype("uint8")      # near = bright, the usual convention
    np.savez_compressed(path, rgb=np.stack(rgb), depth=d, actions=np.stack(actions))


if __name__ == "__main__":
    import sys

    _render(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]),
            int(sys.argv[5]))
