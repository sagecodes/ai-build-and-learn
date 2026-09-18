"""Can V-JEPA teach a humanoid to walk? No PPO legs underneath this time.

`walk` put V-JEPA in charge of WHERE to go and borrowed the legs from topics/rl-mujoco.
This asks for the legs themselves. A fresh G1 policy starts from nothing and its ONLY
reward is V-JEPA's opinion of how much its movement looks like walking. No
hand-written gait reward, no velocity tracking, no foot-clearance terms.

── What "looks like walking" means here ────────────────────────────────────────
The rl-mujoco G1 is filmed walking forward. That is a callback with a twist: the old
policy is now only a VIDEO. The student never sees its joint angles or actions, only
V-JEPA's embedding of clips of it.

    score(clip) = max over reference clips of the mean, over the 2048 spatio-temporal
                  tokens, of the centred cosine between the two clips' tokens

Token-wise rather than mean-pooled because mean pooling destroys motion (measured in
`collapse`), and max over a bank of reference clips because a gait has a phase: a
clip of the left foot swinging should not be penalised for not being the right.

Each clip is 16 frames, every 2nd control step (0.62 s), filmed from the side by a
camera that is STATIC for the length of the clip: it is placed at the robot's heading
and position at the clip's first frame, so walking forward shows up as the robot
moving across the frame, and standing still does not.

── Why there is a small network between V-JEPA and PPO ─────────────────────────
PPO for a humanoid needs ~10^8 steps at ~10^5 steps/s. Rendering and encoding a clip
takes milliseconds. So V-JEPA scores a few thousand clips, a small MLP learns to
predict that score from the robot's state over the same window (`window_features`),
and PPO trains against the MLP inside MJX at full speed. The obvious failure is the
policy finding states the MLP over-rates. The fix is rounds: after each PPO round, the
student's own clips are rendered and scored by the REAL V-JEPA, added to the data,
and the MLP is refitted. The gap between the MLP's opinion and V-JEPA's on the
student's clips is reported every round; it is how you see reward hacking.

── What is NOT learned from V-JEPA (said out loud) ─────────────────────────────
- Episode termination on a fall (torso past horizontal, or feet tangled) is the env's
  own rule. It is a physics fact, not a reward term, but it does mean "falling ends
  the fun" is known without V-JEPA.
- The policy observation includes Playground's gait clock (a sin/cos phase), as it did
  for the teacher. It is an input, not a reward, but it is a periodic prior.
"""

from __future__ import annotations

import pickle
import time

import numpy as np

ENV_NAME = "G1JoystickFlatTerrain"

CLIP_FRAMES = 16
FRAME_STRIDE = 2
WINDOW = (CLIP_FRAMES - 1) * FRAME_STRIDE + 1      # 31 control steps, 0.62 s at 50 Hz
OFFSETS = (0, 10, 20, 30)                          # which window steps the reward net reads
NQ, NV = 36, 35
RAW = NQ + NV
FEAT = 4 * 68 + 6 + 3                              # see window_features
CLIP_SIZE = 256
CLIP_CAM = (2.6, -8.0)                             # distance (m), elevation (deg); side-on

# The same checkpoint `walk` uses for its legs (topics/rl-mujoco run rlswb4sgxg6j2vwfr9gr).
TEACHER_CHECKPOINT = (
    "s3://flyte-data/d6/physical-ai/development/rlswb4sgxg6j2vwfr9gr/"
    "nfggzwfe3koyd2ww5n1qdj97/1/e9e6b700224580aff7c17fef99c9d4d7/g1_checkpoint.pkl"
)

# What V-JEPA is asked to score in round 0. Only "walk" is the target; the rest exist
# so the score has something to rank, and so the MLP sees what not-walking looks like.
#   name            (command vx, vy, yaw)   action noise   actor
SOURCES = {
    "walk":         ((1.0, 0.0, 0.0), 0.0, "policy"),
    "walk-slow":    ((0.5, 0.0, 0.0), 0.0, "policy"),
    "march":        ((0.0, 0.0, 0.0), 0.0, "policy"),   # zero command: steps in place
    "backward":     ((-0.6, 0.0, 0.0), 0.0, "policy"),
    "strafe":       ((0.0, 0.5, 0.0), 0.0, "policy"),
    "turn":         ((0.0, 0.0, 1.0), 0.0, "policy"),
    "stumble":      ((1.0, 0.0, 0.0), 0.6, "policy"),
    "freeze":       ((0.0, 0.0, 0.0), 0.0, "zero"),      # hold the default pose
    "flail":        ((0.0, 0.0, 0.0), 0.0, "random"),    # uniform random actions
}
TARGET = "walk"


# ── env ─────────────────────────────────────────────────────────────────────────


def env_config(impl: str):
    from mujoco_playground import registry

    cfg = registry.get_default_config(ENV_NAME)
    cfg.njmax = 128                    # 90 overflows and MJX silently drops contacts
    cfg.impl = impl
    cfg.push_config.enable = False     # unexplained shoves would be scored as gait
    cfg.noise_config.level = 0.0
    return cfg


def load_env(impl: str = "jax"):
    from mujoco_playground import registry

    return registry.load(ENV_NAME, config=env_config(impl))


def make_student_env(rewnet: dict, impl: str = "warp"):
    """The G1 env with its reward REPLACED by the distilled V-JEPA reward.

    Everything else is Playground's G1 (physics, termination, observation). The
    command is pinned to zero: the reward never looks at it, so it is a constant input.
    """
    import jax.numpy as jp
    from mujoco_playground._src.locomotion.g1 import joystick as g1j

    cfg = env_config(impl)
    cfg.lin_vel_x = [0.0, 0.0]
    cfg.lin_vel_y = [0.0, 0.0]
    cfg.ang_vel_yaw = [0.0, 0.0]
    offs = jp.asarray(OFFSETS)

    class Student(g1j.Joystick):
        def __init__(self, params, **kw):
            super().__init__(**kw)
            self._rp = {k: jp.asarray(v) for k, v in params.items()}

        def reset(self, rng):
            s = super().reset(rng)
            raw = jp.concatenate([s.data.qpos, s.data.qvel])
            s.info["jhist"] = jp.tile(raw, (WINDOW, 1))
            s.info["jstep"] = jp.zeros((), jp.int32)
            s.metrics["jepa_reward"] = jp.zeros(())
            s.metrics["fwd_m"] = jp.zeros(())
            return s

        def step(self, state, action):
            hist0, jstep = state.info["jhist"], state.info["jstep"]
            s = super().step(state, action)
            raw = jp.concatenate([s.data.qpos, s.data.qvel])
            # The auto-reset wrapper restores data and obs but NOT info, so after a
            # fall the buffer still holds the dead episode. jstep == 0 marks a fresh
            # episode: start the window over, as the labelled windows do.
            hist = jp.where(jstep == 0, jp.tile(raw, (WINDOW, 1)),
                            jp.concatenate([hist0[1:], raw[None]]))
            r = rewnet_apply(self._rp, window_features(hist[offs], jp), jp)
            yaw = _yaw(s.data.qpos[3:7], jp)
            fwd = jp.cos(yaw) * s.data.qvel[0] + jp.sin(yaw) * s.data.qvel[1]
            s.info["jhist"] = hist
            s.info["jstep"] = jp.where(s.done > 0.5, 0, jstep + 1).astype(jp.int32)
            s.metrics["jepa_reward"] = r
            s.metrics["fwd_m"] = fwd * self.dt
            return s.replace(reward=r * self.dt)

    env = Student(rewnet, task="flat_terrain", config=cfg)
    if impl == "warp":
        # MuJoCo-Warp prints "solver iterations limit reached" from INSIDE the CUDA
        # kernels, per env, per step: Playground runs the G1 at 3 solver / 5 line-search
        # iterations on purpose, for speed. Thousands of lines a second of expected
        # behaviour, so silence it (it is a compile-time constant, set before any jit).
        try:
            mm = env._mjx_model
            env._mjx_model = mm.replace(opt=mm.opt.replace(_impl=mm.opt._impl.replace(warn_overflow=0)))
        except Exception as exc:  # noqa: BLE001 - noisy logs are not worth failing a run over
            print(f"[selfwalk] could not silence solver warnings: {exc}", flush=True)
    return env


def load_policy(env, checkpoint_path: str):
    """The rl-mujoco teacher as a batched jitted inference fn."""
    import functools

    import jax
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    with open(checkpoint_path, "rb") as f:
        ck = pickle.load(f)
    net = functools.partial(ppo_networks.make_ppo_networks, **ck["network_factory"])(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    return jax.jit(ppo_networks.make_inference_fn(net)(ck["params"], deterministic=True))


_JIT: dict = {}


def rollout(env, policy, n_envs: int, steps: int, command, noise: float, actor: str,
            seed: int = 0) -> dict:
    """Batched CPU rollout. Returns qpos/qvel [B, T+1, .] and valid [B, T+1].

    A frame is valid up to and including the step the robot fell; after that the
    physics keeps running on a fallen robot, which nobody should score.
    """
    import jax
    import jax.numpy as jp

    if id(env) not in _JIT:                 # compile once per env, not per call
        _JIT[id(env)] = (jax.jit(jax.vmap(env.reset)), jax.jit(jax.vmap(env.step)))
    reset, step = _JIT[id(env)]
    rng = jax.random.PRNGKey(seed)
    rng, k = jax.random.split(rng)
    s = reset(jax.random.split(k, n_envs))
    cmd = jp.tile(jp.asarray(command, jp.float32), (n_envs, 1))
    s.info["command"] = cmd
    qpos, qvel, valid = [np.asarray(s.data.qpos)], [np.asarray(s.data.qvel)], [np.ones(n_envs, bool)]
    alive = np.ones(n_envs, bool)
    for _ in range(steps):
        rng, k1, k2 = jax.random.split(rng, 3)
        if actor == "policy":
            a, _ = policy(s.obs, k1)
            if noise > 0:
                a = jp.clip(a + noise * jax.random.normal(k2, a.shape), -1, 1)
        elif actor == "random":
            a = jax.random.uniform(k1, (n_envs, env.action_size), minval=-1, maxval=1)
        else:
            a = jp.zeros((n_envs, env.action_size))
        s = step(s, a)
        s.info["command"] = cmd
        valid.append(alive.copy())
        alive &= ~(np.asarray(s.done) > 0.5)
        qpos.append(np.asarray(s.data.qpos)); qvel.append(np.asarray(s.data.qvel))
    return {"qpos": np.stack(qpos, 1).astype(np.float32), "qvel": np.stack(qvel, 1).astype(np.float32),
            "valid": np.stack(valid, 1)}


# ── windows and the features the reward net reads ───────────────────────────────


def windows(traj: dict, stride: int, first: int = 0, limit: int | None = None,
            seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Pick (env, end-step) windows from a rollout.

    Returns clip_idx [N, 16] and raw4 [N, 4, RAW]. Windows that start before step 0
    repeat frame 0, which is exactly what the MJX env's history buffer does after a
    reset, so the two agree on what an episode's first half-second looks like.
    """
    qpos, qvel, valid = traj["qpos"], traj["qvel"], traj["valid"]
    B, T = valid.shape
    picks = [(b, t) for b in range(B) for t in range(first, T, stride) if valid[b, t]]
    if limit is not None and len(picks) > limit:
        rng = np.random.default_rng(seed)
        picks = [picks[i] for i in sorted(rng.choice(len(picks), limit, replace=False))]
    raw = np.concatenate([qpos, qvel], -1)
    clip_q, raw4 = [], []
    for b, t in picks:
        ts = np.clip(t - (WINDOW - 1) + FRAME_STRIDE * np.arange(CLIP_FRAMES), 0, None)
        clip_q.append(qpos[b, ts])
        raw4.append(raw[b, np.clip(t - (WINDOW - 1) + np.array(OFFSETS), 0, None)])
    if not picks:
        return np.zeros((0, CLIP_FRAMES, NQ), np.float32), np.zeros((0, 4, RAW), np.float32)
    return np.stack(clip_q).astype(np.float32), np.stack(raw4).astype(np.float32)


def _yaw(q, xp):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return xp.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def window_features(raw4, xp=np):
    """[..., 4, 71] raw qpos|qvel at window steps (0, 10, 20, 30) -> [..., 281].

    Heading-invariant on purpose, like the clip camera: every quantity is either in the
    robot's body frame or relative to its pose at the window's first step. Works with
    numpy (labelling) and jax.numpy (inside the MJX env) alike.
    """
    q = raw4[..., 3:7]
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = [[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
         [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
         [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]]
    v = raw4[..., NQ:NQ + 3]
    grav = xp.stack([-R[2][j] for j in range(3)], -1)
    lin = xp.stack([sum(R[i][j] * v[..., i] for i in range(3)) for j in range(3)], -1)
    per = xp.concatenate([
        raw4[..., 2:3],                       # pelvis height
        grav,                                 # which way is down, in the body frame
        lin,                                  # body-frame velocity
        raw4[..., NQ + 3:NQ + 6],             # body-frame angular velocity
        raw4[..., 7:NQ],                      # 29 joint angles
        0.1 * raw4[..., NQ + 6:],             # 29 joint velocities
    ], -1)                                    # [..., 4, 68]
    yaw = _yaw(q, xp)
    c, s = xp.cos(yaw[..., 0:1]), xp.sin(yaw[..., 0:1])
    dx = raw4[..., 1:, 0] - raw4[..., 0:1, 0]
    dy = raw4[..., 1:, 1] - raw4[..., 0:1, 1]
    dyaw = yaw[..., 1:] - yaw[..., 0:1]
    disp = xp.concatenate([c * dx + s * dy, -s * dx + c * dy], -1)      # forward, left (m)
    turn = xp.arctan2(xp.sin(dyaw), xp.cos(dyaw))
    return xp.concatenate([per.reshape(per.shape[:-2] + (4 * 68,)), disp, turn], -1)


def rewnet_apply(p: dict, x, xp=np):
    """The distilled reward: [..., FEAT] -> [...] in [0, 1]."""
    h = (x - p["mu"]) / p["sd"]
    for i in range(3):
        h = h @ p[f"W{i}"] + p[f"b{i}"]
        if i < 2:
            h = xp.maximum(h, 0.0)
    return 1.0 / (1.0 + xp.exp(-h[..., 0]))


def fit_rewnet(X: np.ndarray, y: np.ndarray, w: np.ndarray | None = None, steps: int = 4000,
               hidden: int = 256, seed: int = 0, on_step=None) -> tuple[dict, dict]:
    """Fit the MLP on (features, V-JEPA reward). Returns (numpy params, held-out stats)."""
    import torch

    torch.manual_seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    n = len(X)
    perm = np.random.default_rng(seed).permutation(n)
    va, tr = perm[: max(1, n // 7)], perm[max(1, n // 7):]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-3
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32, device=dev)
    yt = torch.tensor(y, dtype=torch.float32, device=dev)
    wt = torch.tensor(np.ones(n) if w is None else w, dtype=torch.float32, device=dev)
    net = torch.nn.Sequential(torch.nn.Linear(FEAT, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                              torch.nn.Linear(hidden, 1)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    tri = torch.tensor(tr, device=dev)
    for s in range(steps):
        idx = tri[torch.randint(len(tri), (512,), device=dev)]
        pred = torch.sigmoid(net(Xt[idx])[:, 0])
        loss = (wt[idx] * (pred - yt[idx]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if on_step is not None and (s + 1) % 500 == 0:
            on_step(s + 1, steps, float(loss))
    with torch.no_grad():
        pv = torch.sigmoid(net(Xt[torch.tensor(va, device=dev)])[:, 0]).cpu().numpy()
    lin = [m for m in net if isinstance(m, torch.nn.Linear)]
    p = {"mu": mu.astype(np.float32), "sd": sd.astype(np.float32)}
    for i, m in enumerate(lin):
        p[f"W{i}"] = m.weight.detach().cpu().numpy().T.astype(np.float32)
        p[f"b{i}"] = m.bias.detach().cpu().numpy().astype(np.float32)
    stats = {"val_r": float(np.corrcoef(pv, y[va])[0, 1]) if len(va) > 2 else float("nan"),
             "val_mae": float(np.abs(pv - y[va]).mean()), "n_train": int(len(tr)), "n_val": int(len(va))}
    return p, stats


# ── rendering ───────────────────────────────────────────────────────────────────


class ClipCamera:
    """Renders what V-JEPA is shown, plus a friendlier tracking view for people."""

    def __init__(self, mj_model, size: int = CLIP_SIZE):
        import copy

        import mujoco

        m = copy.deepcopy(mj_model)
        m.vis.global_.offwidth = max(m.vis.global_.offwidth, 640)
        m.vis.global_.offheight = max(m.vis.global_.offheight, 640)
        self.m = m
        self.d = mujoco.MjData(m)
        self.size = size
        self._r: dict[tuple[int, int], object] = {}

    def _renderer(self, h: int, w: int):
        import mujoco

        if (h, w) not in self._r:
            self._r[(h, w)] = mujoco.Renderer(self.m, height=h, width=w)
            if len(self._r) == 1:
                try:
                    from OpenGL import GL
                    print(f"[selfwalk] MuJoCo renders on: {GL.glGetString(GL.GL_VENDOR).decode()} / "
                          f"{GL.glGetString(GL.GL_RENDERER).decode()}", flush=True)
                except Exception as exc:  # noqa: BLE001
                    print(f"[selfwalk] GL vendor unknown: {exc}", flush=True)
        return self._r[(h, w)]

    def _frame(self, qpos, lookat, azimuth, h, w, distance, elevation):
        import mujoco

        self.d.qpos[:] = qpos
        mujoco.mj_forward(self.m, self.d)
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.lookat[:] = lookat
        cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
        r = self._renderer(h, w)
        r.update_scene(self.d, cam)
        return r.render()

    def clip(self, clip_q: np.ndarray) -> np.ndarray:
        """[16, 36] -> [16, S, S, 3]. Camera fixed at frame 0's position and heading."""
        q0 = clip_q[0]
        az = float(np.degrees(_yaw(q0[3:7], np)) + 90.0)
        look = [float(q0[0]), float(q0[1]), 0.62]
        return np.stack([self._frame(q, look, az, self.size, self.size, *CLIP_CAM) for q in clip_q])

    def follow(self, qpos_seq: np.ndarray, h: int = 270, w: int = 480) -> np.ndarray:
        """People's view: side-on, tracking the pelvis, heading fixed from frame 0."""
        az = float(np.degrees(_yaw(qpos_seq[0][3:7], np)) + 90.0)
        return np.stack([self._frame(q, [float(q[0]), float(q[1]), 0.6], az, h, w, 2.9, -10.0)
                         for q in qpos_seq])

    def close(self) -> None:
        for r in self._r.values():
            r.close()
        self._r = {}


# ── V-JEPA scoring ──────────────────────────────────────────────────────────────


class Scorer:
    """V-JEPA 2 encoder + a bank of reference clips. Scores a clip against the bank."""

    def __init__(self, repo: str):
        import torch

        import jepa

        self.torch = torch
        self.model, proc, _ = jepa.load(repo)
        self.device = next(self.model.parameters()).device
        self.dtype = next(self.model.parameters()).dtype
        self.mean = torch.tensor(proc.image_mean, device=self.device).view(1, 1, 3, 1, 1)
        self.std = torch.tensor(proc.image_std, device=self.device).view(1, 1, 3, 1, 1)
        self.bank = None

    def tokens(self, clips: np.ndarray) -> "torch.Tensor":
        """[B, 16, S, S, 3] uint8 -> [B, N, D] centred, L2-normalised tokens."""
        torch = self.torch
        x = torch.as_tensor(clips, device=self.device).permute(0, 1, 4, 2, 3).float() / 255.0
        x = ((x - self.mean) / self.std).to(self.dtype)
        with torch.inference_mode():
            z = self.model.get_vision_features(x).float()
        z = z - z.mean(1, keepdim=True)
        return torch.nn.functional.normalize(z, dim=-1).to(torch.bfloat16)

    def set_bank(self, clips: np.ndarray, batch: int = 8) -> None:
        self.bank = self.torch.cat([self.tokens(clips[i:i + batch]) for i in range(0, len(clips), batch)])

    def score(self, clips: np.ndarray, batch: int = 8) -> np.ndarray:
        """Max over the bank of the mean token-wise cosine. [B] floats in [-1, 1]."""
        out = []
        for i in range(0, len(clips), batch):
            z = self.tokens(clips[i:i + batch])
            sim = self.torch.einsum("bnd,knd->bk", z.float(), self.bank.float()) / z.shape[1]
            out.append(sim.max(1).values.cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


def normalise(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """V-JEPA score -> reward in [0, 1]: 0 at the flailing median, 1 at the walking one."""
    return np.clip((raw - lo) / max(hi - lo, 1e-6), 0.0, 1.0)

