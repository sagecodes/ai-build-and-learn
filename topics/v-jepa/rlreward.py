"""Reinforcement learning where the reward is a photograph.

`plan` showed that V-JEPA 2's embedding distance to a goal image tracks real progress
(r=+0.77) well enough to plan on. This asks the harder question: is it a good enough
signal to *train* on?

    reward_t = -|| encode(camera_t) - encode(goal_photo) ||

No hand-written reward. No shaping terms, no distance-to-target, no success bonus
schedule. You photograph what success looks like and a normal RL loop does the rest.
If that works it removes the part of RL that consumes the most human time, and it is
the natural payoff of the planning result.

── How this differs from topics/dreamerv3, which it superficially resembles ────
DreamerV3 learns a world model from its own experience and trains the policy INSIDE
it, replacing the ENVIRONMENT. It still needs environment rewards; its reward head is
trained on them, so it does nothing for reward design.

This replaces the REWARD FUNCTION and leaves the environment alone: the policy takes
real steps in MuJoCo. The two are complementary, near-opposite substitutions, and
nothing stops you doing both.

── The design decision that makes it an experiment rather than a demo ──────────
The policy observes PROPRIOCEPTION ONLY: end-effector position, arm joint angles, and
its last action. It cannot see where the target is. If the observation contained the
target position, or a goal-relative vector, the policy could solve the task from almost
any reward that vaguely correlated with progress, and every condition below would score
about the same. Withholding it means the reward is the only channel through which the
task can be communicated, which is exactly what is being measured.

The goal is fixed across episodes for the same reason: a per-episode random goal the
policy cannot observe is not learnable by any reward, and would measure nothing.

── The controls ────────────────────────────────────────────────────────────────
  `dense`    -||ee - goal||, the hand-designed reward. The ceiling, and what we are
             trying to replace.
  `jepa`     the thing being tested.
  `pixel`    mean |image - goal image|. The cheap alternative nobody bothers to run.
             If raw pixel difference works as well, V-JEPA is not earning its keep.
  `random`   the SAME ViT architecture with randomly initialised weights. The control
             that decides whether this result is about V-JEPA or merely about
             "distance between two embeddings of an image". Most write-ups skip it.
  `sparse`   +1 inside the success radius and 0 outside. The floor: what you get when
             you refuse to do any reward engineering at all.

Every condition shares the environment, the policy architecture, the seed schedule and
the number of environment steps. Rewards are standardised against statistics gathered
from the same short random rollout, so a condition cannot win by being on a larger
numeric scale.
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import mujoco
import sim

MAXNORM = 0.05
EPISODE = 30
SUCCESS_RADIUS = 0.06
REWARDS = ("dense", "jepa", "jepa_pool", "jepa_cos", "pixel", "random", "sparse")


# ── the batched environment ─────────────────────────────────────────────────────


class VecReach:
    """N Franka arms sharing one MjModel, one renderer and one camera.

    Sharing the model is what makes a learned reward affordable: `mujoco.Renderer` is
    bound to a model, but `update_scene` takes a *data*, so one renderer serves every
    environment at about 1.6 ms a frame. Giving each environment its own model would
    mean N EGL contexts and N renderers for no benefit, since the arms differ only in
    their state.
    """

    def __init__(self, n: int = 16, seed: int = 0, render_size: int = 256,
                 distractors: bool = False):
        self.n = n
        self.distractors = distractors
        self.model = mujoco.MjModel.from_xml_path(sim._scene_path(seed, distractors))
        # Same gravity compensation and the same integral-IK controller the planning
        # demo uses, so a policy's actions are executed as faithfully there as here
        # (0.66 mm mean error on a 50 mm command).
        self.model.body_gravcomp[:] = 1.0
        self.datas = [mujoco.MjData(self.model) for _ in range(n)]
        self.render_size = render_size
        self._renderer = mujoco.Renderer(self.model, height=render_size, width=render_size)
        self.hand_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target")
        self.nq_arm = 7
        self.goal_pos = self.model.body_pos[self.target_id] + np.array([0.0, 0.0, 0.09])
        self.rng = np.random.default_rng(seed)
        self.t = np.zeros(n, dtype=int)
        self.last_action = np.zeros((n, 3), dtype=np.float32)
        # Free-joint qpos starts after the arm's 9 (7 joints + 2 fingers); each free
        # joint owns 7 numbers, 3 of position and 4 of quaternion.
        self.n_free = (self.model.nq - 9) // 7 if distractors else 0
        for i in range(n):
            self._reset_one(i)

    # -- single-environment helpers ------------------------------------------------

    def _reset_one(self, i: int) -> None:
        d = self.datas[i]
        mujoco.mj_resetData(self.model, d)
        q = sim.HOME_QPOS.copy()
        # Jitter the start pose. Without it every episode begins from the same state and
        # the replay buffer never sees the space around it.
        q[: self.nq_arm] += self.rng.uniform(-0.08, 0.08, self.nq_arm)
        d.qpos[: len(q)] = q
        d.ctrl[: self.nq_arm] = q[: self.nq_arm]
        if self.model.nu > self.nq_arm:
            d.ctrl[self.nq_arm] = 255.0
        self._scatter(i)
        mujoco.mj_forward(self.model, d)
        self.t[i] = 0
        self.last_action[i] = 0.0

    def _scatter(self, i: int) -> None:
        """Teleport the distractors somewhere new.

        This is the nuisance the reward has to be invariant to. Without it the arm is
        the only thing in the scene that ever moves, which is the best possible case
        for comparing raw pixels -- and measurably, a randomly initialised network
        scores as well as V-JEPA under those conditions. The distractors are visual
        only (no collision) and are kept out of the corridor between gripper and
        target, so they change what the camera sees without changing the task.
        """
        if not self.n_free:
            return
        d = self.datas[i]
        for k in range(self.n_free):
            base = 9 + 7 * k
            d.qpos[base : base + 3] = [
                self.rng.uniform(0.28, 0.74),
                self.rng.uniform(-0.34, 0.38),
                self.rng.uniform(0.02, 0.24),
            ]
            d.qpos[base + 3 : base + 7] = [1.0, 0.0, 0.0, 0.0]
            d.qvel[6 * k + (self.model.nv - 6 * self.n_free) : ] = 0.0
        d.qvel[self.model.nv - 6 * self.n_free :] = 0.0

    def _ee(self, i: int) -> np.ndarray:
        d = self.datas[i]
        R = d.xmat[self.hand_id].reshape(3, 3)
        return d.xpos[self.hand_id] + R @ sim.TCP_OFFSET

    def _step_one(self, i: int, delta: np.ndarray) -> None:
        """Integral-IK Cartesian step, the same controller as sim.PandaReach."""
        d = self.datas[i]
        goal = self._ee(i) + delta
        lo = self.model.jnt_range[: self.nq_arm, 0]
        hi = self.model.jnt_range[: self.nq_arm, 1]
        for _ in range(6):
            err = goal - self._ee(i)
            if np.linalg.norm(err) < 1e-3:
                break
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            mujoco.mj_jac(self.model, d, jacp, jacr, self._ee(i), self.hand_id)
            J = jacp[:, : self.nq_arm]
            dq = J.T @ np.linalg.solve(J @ J.T + 0.0064 * np.eye(3), np.clip(err, -0.05, 0.05))
            d.ctrl[: self.nq_arm] = np.clip(d.ctrl[: self.nq_arm] + dq, lo, hi)
            mujoco.mj_step(self.model, d, nstep=30)
        mujoco.mj_forward(self.model, d)

    # -- vector interface ----------------------------------------------------------

    def observe(self) -> np.ndarray:
        """Proprioception only: end-effector position, arm joints, last action.

        Deliberately no target position and no goal-relative vector. See the module
        docstring: leaking the goal here would make every reward condition look alike.
        """
        out = np.zeros((self.n, 3 + self.nq_arm + 3), dtype=np.float32)
        for i in range(self.n):
            out[i, :3] = self._ee(i)
            out[i, 3 : 3 + self.nq_arm] = self.datas[i].qpos[: self.nq_arm]
            out[i, 3 + self.nq_arm :] = self.last_action[i]
        return out

    def render(self) -> np.ndarray:
        frames = np.empty((self.n, self.render_size, self.render_size, 3), dtype=np.uint8)
        for i, d in enumerate(self.datas):
            self._renderer.update_scene(d, camera="exterior")
            frames[i] = self._renderer.render()
        return frames

    def distances(self) -> np.ndarray:
        return np.array([np.linalg.norm(self._ee(i) - self.goal_pos) for i in range(self.n)])

    def step(self, actions: np.ndarray):
        """actions in [-1, 1]^3, scaled to MAXNORM metres. Returns (obs, done)."""
        a = np.clip(actions, -1.0, 1.0).astype(np.float32)
        for i in range(self.n):
            self._step_one(i, a[i] * MAXNORM)
            if self.n_free:
                self._scatter(i)
                mujoco.mj_forward(self.model, self.datas[i])
        self.last_action = a
        self.t += 1
        done = self.t >= EPISODE
        return self.observe(), done

    def reset_done(self, done: np.ndarray) -> None:
        for i in np.nonzero(done)[0]:
            self._reset_one(int(i))

    def goal_frame(self) -> np.ndarray:
        """A photograph of success: the arm already at the goal.

        Rendered by driving environment 0 there and then restoring it, so the goal
        image comes from the same camera and the same scene as every observation.
        """
        saved = (self.datas[0].qpos.copy(), self.datas[0].qvel.copy(), self.datas[0].ctrl.copy())
        if self.n_free:
            # The goal photo must not encode one particular distractor layout, or the
            # reward would be partly "put the clutter back where it was in the photo".
            self._scatter(0)
        for _ in range(60):
            err = self.goal_pos - self._ee(0)
            if np.linalg.norm(err) < 2e-3:
                break
            self._step_one(0, np.clip(err, -0.04, 0.04))
        self._renderer.update_scene(self.datas[0], camera="exterior")
        frame = self._renderer.render().copy()
        self.datas[0].qpos[:], self.datas[0].qvel[:], self.datas[0].ctrl[:] = saved
        mujoco.mj_forward(self.model, self.datas[0])
        return frame

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None


# ── the five reward functions ───────────────────────────────────────────────────


class Rewarder:
    """Turns a batch of camera frames into a batch of rewards.

    Every learned variant runs the encoder in bfloat16 at batch 16, which is 5.2 ms a
    frame for ViT-L against 74 ms for the AC model's ViT-g in fp32. bf16 is safe here:
    the dtype bug that breaks V-JEPA 2-AC lives in the AC predictor's RoPE, and the
    encoder alone is unaffected.
    """

    def __init__(self, kind: str, goal_frame: np.ndarray, device: str = "cuda",
                 repo: str = None):
        if kind not in REWARDS:
            raise ValueError(f"unknown reward {kind!r}, expected one of {REWARDS}")
        self.kind = kind
        self.device = device
        self.goal_frame = goal_frame
        self.model = None
        self.scale = (0.0, 1.0)  # (mean, std), fitted on a random rollout

        if kind in ("jepa", "jepa_pool", "jepa_cos", "random"):
            import jepa
            from config import VITL

            repo = repo or VITL
            model, _, _ = jepa.load(repo)
            if kind == "random":
                # Same architecture, same config, fresh weights. If this scores as well
                # as `jepa`, the result is about embedding-distance-as-reward in
                # general and not about what V-JEPA learned from video.
                from transformers import VJEPA2Model

                model = VJEPA2Model(model.config)
            self.model = model.to(device=device, dtype=torch.bfloat16).eval()
            self.z_goal = self._encode(goal_frame[None])

    @torch.no_grad()
    def _encode(self, frames: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        x = x[:, None].repeat(1, 2, 1, 1, 1)  # B, T=2, C, H, W (one tubelet)
        x = x.to(self.device, torch.bfloat16)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=torch.bfloat16) * 255
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=torch.bfloat16) * 255
        x = (x - mean[None, None, :, None, None]) / std[None, None, :, None, None]
        h = self.model.get_vision_features(x)
        return F.layer_norm(h.float(), (h.size(-1),))

    def raw(self, frames: np.ndarray, dists: np.ndarray) -> np.ndarray:
        """Unstandardised reward. Higher is better in every case."""
        if self.kind == "dense":
            return -dists
        if self.kind == "sparse":
            return (dists < SUCCESS_RADIUS).astype(np.float32)
        if self.kind == "pixel":
            d = np.abs(frames.astype(np.float32) - self.goal_frame.astype(np.float32))
            return -d.mean(axis=(1, 2, 3)) / 255.0
        z = self._encode(frames)
        if self.kind in ("jepa", "random"):
            # Per-token L1: every patch counts the same, so four small distractor
            # cubes weigh as much as the whole arm. Measured SNR 0.53 with
            # distractors, i.e. mostly reporting the nuisance.
            return -(z - self.z_goal).abs().mean(dim=[1, 2]).cpu().numpy()
        # Pooled readouts. Mean-pooling over tokens first is what the `probe` task
        # already does for classification, and it spreads any local change across the
        # whole vector instead of letting it dominate a handful of tokens.
        zp, gp = z.mean(1), self.z_goal.mean(1)
        if self.kind == "jepa_pool":
            return -(zp - gp).abs().mean(dim=-1).cpu().numpy()
        # jepa_cos: centre before comparing. V-JEPA's tokens sit in a narrow cone (see
        # the README: random token pairs sit at cosine 0.28), so raw cosine is
        # dominated by a component every embedding shares.
        zc = zp - zp.mean(dim=-1, keepdim=True)
        gc = gp - gp.mean(dim=-1, keepdim=True)
        return F.cosine_similarity(zc, gc, dim=-1).cpu().numpy()

    def fit_scale(self, samples: np.ndarray) -> None:
        """Standardise against a random rollout, identically for every condition.

        Without this the conditions differ by orders of magnitude in numeric scale
        (pixel L1 and negative metres are not comparable), and SAC's entropy temperature
        would be doing different jobs in each. Fitting on a RANDOM policy keeps the
        statistics independent of how well any condition eventually does.
        """
        self.scale = (float(samples.mean()), float(samples.std()) + 1e-8)

    def __call__(self, frames: np.ndarray, dists: np.ndarray) -> np.ndarray:
        m, s = self.scale
        return (self.raw(frames, dists) - m) / s


# ── a compact SAC ───────────────────────────────────────────────────────────────
#
# SAC rather than PPO because every environment step costs a render and an encode, so
# sample efficiency is the binding constraint: PPO would need several hundred thousand
# steps where SAC needs tens of thousands.


def mlp(sizes, out_act=None):
    layers = []
    for i in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[i], sizes[i + 1])]
        if i < len(sizes) - 2:
            layers += [nn.ReLU()]
    if out_act is not None:
        layers += [out_act]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = mlp([obs_dim, hidden, hidden, 2 * act_dim])
        self.act_dim = act_dim

    def forward(self, obs, deterministic=False):
        mu, log_std = self.net(obs).chunk(2, dim=-1)
        log_std = log_std.clamp(-5, 2)
        std = log_std.exp()
        if deterministic:
            return torch.tanh(mu), None
        dist = torch.distributions.Normal(mu, std)
        raw = dist.rsample()
        act = torch.tanh(raw)
        # tanh change-of-variables correction, the usual SAC detail
        logp = (dist.log_prob(raw) - torch.log(1 - act.pow(2) + 1e-6)).sum(-1)
        return act, logp


@dataclass
class Replay:
    capacity: int
    obs_dim: int
    act_dim: int
    obs: np.ndarray = field(init=False)
    act: np.ndarray = field(init=False)
    rew: np.ndarray = field(init=False)
    nobs: np.ndarray = field(init=False)
    done: np.ndarray = field(init=False)
    n: int = 0
    i: int = 0

    def __post_init__(self):
        self.obs = np.zeros((self.capacity, self.obs_dim), np.float32)
        self.act = np.zeros((self.capacity, self.act_dim), np.float32)
        self.rew = np.zeros((self.capacity,), np.float32)
        self.nobs = np.zeros((self.capacity, self.obs_dim), np.float32)
        self.done = np.zeros((self.capacity,), np.float32)

    def add(self, o, a, r, no, d):
        k = len(o)
        idx = (self.i + np.arange(k)) % self.capacity
        self.obs[idx], self.act[idx], self.rew[idx] = o, a, r
        self.nobs[idx], self.done[idx] = no, d
        self.i = (self.i + k) % self.capacity
        self.n = min(self.n + k, self.capacity)

    def sample(self, batch, device):
        j = np.random.randint(0, self.n, batch)
        t = lambda x: torch.as_tensor(x[j], device=device)
        return t(self.obs), t(self.act), t(self.rew), t(self.nobs), t(self.done)


def train_sac(
    env: VecReach,
    rewarder: Rewarder,
    steps: int = 20000,
    warmup: int = 1500,
    batch: int = 256,
    gamma: float = 0.97,
    tau: float = 0.005,
    lr: float = 3e-4,
    updates_per_step: int | None = None,
    device: str = "cuda",
    seed: int = 0,
    on_progress=None,
):
    """Soft actor-critic. Returns (history, final_eval).

    `gamma` is 0.97 rather than the usual 0.99 because episodes are only 30 steps; a
    horizon longer than the episode makes the value target depend mostly on
    bootstrapping past the reset.
    """
    # One gradient update per ENVIRONMENT step, not per vector step. With 16 parallel
    # envs the difference is 16x: the first version of this did 375 updates for 6000
    # collected steps and learned nothing at all, which looks exactly like a bad reward.
    if updates_per_step is None:
        updates_per_step = env.n
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs_dim, act_dim = env.observe().shape[1], 3

    actor = Actor(obs_dim, act_dim).to(device)
    q1 = mlp([obs_dim + act_dim, 256, 256, 1]).to(device)
    q2 = mlp([obs_dim + act_dim, 256, 256, 1]).to(device)
    q1t = mlp([obs_dim + act_dim, 256, 256, 1]).to(device)
    q2t = mlp([obs_dim + act_dim, 256, 256, 1]).to(device)
    q1t.load_state_dict(q1.state_dict())
    q2t.load_state_dict(q2.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=lr)
    log_alpha = torch.zeros(1, device=device, requires_grad=True)
    opt_alpha = torch.optim.Adam([log_alpha], lr=lr)
    target_entropy = -float(act_dim)

    buf = Replay(capacity=max(steps, 20000), obs_dim=obs_dim, act_dim=act_dim)
    history = []
    obs = env.observe()
    collected = 0

    # Fit the reward scale on a short random rollout, before any learning, so every
    # condition is standardised by statistics from the same behaviour.
    cal = []
    for _ in range(max(1, warmup // (4 * env.n))):
        a = np.random.uniform(-1, 1, (env.n, act_dim)).astype(np.float32)
        nobs, done = env.step(a)
        cal.append(rewarder.raw(env.render(), env.distances()))
        env.reset_done(done)
        obs = env.observe()
    rewarder.fit_scale(np.concatenate(cal))

    while collected < steps:
        if collected < warmup:
            act = np.random.uniform(-1, 1, (env.n, act_dim)).astype(np.float32)
        else:
            with torch.no_grad():
                a, _ = actor(torch.as_tensor(obs, device=device))
            act = a.cpu().numpy()
        nobs, done = env.step(act)
        rew = rewarder(env.render(), env.distances())
        # `done` here is a time limit, not a terminal state, so it must NOT cut the
        # bootstrap: doing so teaches the agent the world ends at step 30.
        buf.add(obs, act, rew, nobs, np.zeros_like(done, dtype=np.float32))
        dist_now = env.distances()
        env.reset_done(done)
        obs = env.observe()
        collected += env.n

        if buf.n > batch and collected >= warmup:
            for _ in range(updates_per_step):
                o, a_, r_, no, d_ = buf.sample(batch, device)
                with torch.no_grad():
                    na, nlogp = actor(no)
                    qt = torch.min(q1t(torch.cat([no, na], -1)), q2t(torch.cat([no, na], -1))).squeeze(-1)
                    target = r_ + gamma * (1 - d_) * (qt - log_alpha.exp() * nlogp)
                qa = torch.cat([o, a_], -1)
                lq = F.mse_loss(q1(qa).squeeze(-1), target) + F.mse_loss(q2(qa).squeeze(-1), target)
                opt_q.zero_grad(set_to_none=True); lq.backward(); opt_q.step()

                pa, logp = actor(o)
                qpi = torch.min(q1(torch.cat([o, pa], -1)), q2(torch.cat([o, pa], -1))).squeeze(-1)
                la = (log_alpha.exp().detach() * logp - qpi).mean()
                opt_a.zero_grad(set_to_none=True); la.backward(); opt_a.step()

                lalpha = -(log_alpha.exp() * (logp.detach() + target_entropy)).mean()
                opt_alpha.zero_grad(set_to_none=True); lalpha.backward(); opt_alpha.step()

                with torch.no_grad():
                    for p, pt in zip(list(q1.parameters()) + list(q2.parameters()),
                                     list(q1t.parameters()) + list(q2t.parameters())):
                        pt.mul_(1 - tau).add_(tau * p)

        if collected % (env.n * 25) == 0:
            history.append((collected, float(np.mean(dist_now)), float(np.mean(rew))))
            if on_progress is not None:
                on_progress(collected, steps, float(np.mean(dist_now)))

    final = evaluate_policy(env, actor, device=device)
    return history, final, actor


@torch.no_grad()
def evaluate_policy(env: VecReach, actor, episodes: int = 1, device: str = "cuda"):
    """Deterministic rollout. Reports TRUE distance to goal, never the learned reward.

    The reward is what is being tested, so it cannot also be the score. Every condition
    is judged on the same physical quantity: how close the gripper actually got.
    """
    starts, finals, frames = [], [], []
    for _ in range(episodes):
        for i in range(env.n):
            env._reset_one(i)
        starts.append(env.distances().mean())
        for t in range(EPISODE):
            a, _ = actor(torch.as_tensor(env.observe(), device=device), deterministic=True)
            env.step(a.cpu().numpy())
            if len(frames) < EPISODE:
                frames.append(env.render()[0])
        finals.append(env.distances().mean())
    s, f = float(np.mean(starts)), float(np.mean(finals))
    return {
        "start_cm": s * 100,
        "final_cm": f * 100,
        "closed": (s - f) / max(s, 1e-6),
        "success": float((env.distances() < SUCCESS_RADIUS).mean()),
        "frames": np.stack(frames) if frames else None,
    }
