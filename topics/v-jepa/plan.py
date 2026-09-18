"""Closed-loop visual goal-reaching, planned entirely in V-JEPA 2's latent space.

The loop is four lines of idea and the rest is bookkeeping:

    encode the current camera frame        ->  z
    encode the goal photo (once)           ->  z_goal
    CEM over action sequences, scoring      |   all of this happens in latent space;
      each by |dream(z, a) - z_goal|        |   no pixel is ever predicted
    execute the first action, re-plan

Nothing here is trained. The encoder and predictor are frozen weights that have
never seen this simulator, the reward is a distance between two embeddings, and
the arm has no privileged access to where the target is.

── Why the baselines are the point ─────────────────────────────────────────────
A closed-loop video of an arm approaching a block is extremely easy to produce by
accident. Differential IK plus any vaguely downhill signal will do it, and so will
a bug. Three controls run on the identical task, seed, scene and controller, and
only the policy changes:

  `oracle`  straight line to the goal, clipped to the same per-step budget. The
            ceiling: what this controller achieves when it is simply told where to
            go. Planning cannot beat it, and if it does, something is wrong.
  `random`  isotropic actions at the same magnitude. The floor, and the one that
            matters, because a 5 cm random walk toward a goal 30 cm away still
            closes some distance by chance and would otherwise read as competence.
  `greedy`  one-step energy argmin over a fixed action grid, no CEM, no horizon.
            Separates "the energy landscape is informative" from "the planner is
            doing something clever with it".
  `lookahead` the ablation that turned out to matter most. Identical to `greedy`
            except the next latent comes from ACTUALLY STEPPING THE SIMULATOR and
            re-encoding, instead of from the world model's imagination. Same
            V-JEPA reward, same search, perfect dynamics. It splits the system
            cleanly in two: if `lookahead` works and `greedy` does not, the
            representation is a good reward and the learned dynamics are what
            failed.

`jepa` beating `random` is the claim. `jepa` approaching `oracle` is the strong
version. `jepa` matching `greedy` would say the search is wasted effort, which is
worth knowing and is why greedy is in the list rather than left out.

Note that `lookahead` is not a policy anyone could deploy -- it needs a simulator it
can rewind, which is precisely what a world model is supposed to replace. It is here
as an instrument, not as a proposal.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

import ac


@dataclass
class Episode:
    """One rollout, everything the report needs to show and to score."""

    policy: str
    frames: list[np.ndarray] = field(default_factory=list)  # render-res, for video
    dist: list[float] = field(default_factory=list)  # EE -> goal, metres
    energy: list[float] = field(default_factory=list)  # latent |z - z_goal|
    actions: list[np.ndarray] = field(default_factory=list)
    traces: list = field(default_factory=list)  # CEM instrumentation, jepa only
    seconds: float = 0.0

    @property
    def start_dist(self) -> float:
        return self.dist[0]

    @property
    def final_dist(self) -> float:
        return self.dist[-1]

    @property
    def best_dist(self) -> float:
        return float(min(self.dist))

    @property
    def closed(self) -> float:
        """Fraction of the start distance removed. 1.0 is a perfect reach, 0.0 is no
        progress, negative means it ran away. Reported instead of raw metres because
        the start distance differs between seeds."""
        return float((self.start_dist - self.final_dist) / max(self.start_dist, 1e-6))


def _true_energies(wm, env, z_goal, grid: np.ndarray) -> np.ndarray:
    """Energy of the REAL next frame for each candidate action.

    The world model's `energy_grid` imagines the next latent; this one produces it
    by executing the action, photographing the result and encoding it, then
    rewinding. Same reward, no imagination. Rewinding is a straight restore of
    qpos/qvel/ctrl, which is exact for this model -- there is no contact state or
    solver warm start that would survive the restore and bias the next candidate.
    """
    import mujoco

    qpos = env.data.qpos.copy()
    qvel = env.data.qvel.copy()
    ctrl = env.data.ctrl.copy()
    out = []
    for a3 in grid:
        env.data.qpos[:] = qpos
        env.data.qvel[:] = qvel
        env.data.ctrl[:] = ctrl
        mujoco.mj_forward(env.model, env.data)
        act = np.zeros(7, dtype=np.float32)
        act[:3] = a3
        env.step(act)
        z = wm.encode(env.render(ac.CROP)[None])[:, -ac.TOKENS_PER_FRAME :]
        out.append(float(wm.energy(z, z_goal[:, -ac.TOKENS_PER_FRAME :])))
    env.data.qpos[:] = qpos
    env.data.qvel[:] = qvel
    env.data.ctrl[:] = ctrl
    mujoco.mj_forward(env.model, env.data)
    return np.array(out)


def _clip(a: np.ndarray, maxnorm: float) -> np.ndarray:
    out = np.zeros(7, dtype=np.float32)
    out[:3] = np.clip(a[:3], -maxnorm, maxnorm)
    return out


def run_episode(
    wm: ac.ActionWorldModel,
    env,
    goal_frame: np.ndarray,
    goal_pos: np.ndarray,
    policy: str = "jepa",
    steps: int = 12,
    cem: ac.CEMConfig | None = None,
    render_size: int = 384,
    seed: int = 0,
    grid_n: int = 3,
    on_step=None,
) -> Episode:
    """Run one closed-loop episode under `policy` and return everything measured."""
    cem = cem or ac.CEMConfig()
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device=wm.device).manual_seed(seed)
    ep = Episode(policy=policy)
    t0 = time.time()

    z_goal = wm.encode(goal_frame[None])

    # The one-step action grid the `greedy` control searches. Deliberately the same
    # per-step budget as everything else, so the comparison is about the policy.
    g = np.linspace(-cem.maxnorm, cem.maxnorm, grid_n)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)

    for t in range(steps):
        frame = env.render(render_size)
        obs = frame if render_size == ac.CROP else env.render(ac.CROP)
        z = wm.encode(obs[None])
        pose = torch.tensor(env.pose7(), device=wm.device, dtype=wm.dtype)[None, None]

        ep.frames.append(frame)
        ep.dist.append(float(np.linalg.norm(env.ee_pos - goal_pos)))
        ep.energy.append(float(wm.energy(z[:, -ac.TOKENS_PER_FRAME :], z_goal[:, -ac.TOKENS_PER_FRAME :])))

        if policy == "jepa":
            actions, trace = wm.plan(z, pose, z_goal, cfg=cem, generator=gen)
            ep.traces.append(trace)
            # Model-predictive: the planner returns a whole sequence, we execute only
            # its first action and re-plan from what actually happened. That is what
            # keeps a 2-step horizon useful over a 12-step episode, and what lets the
            # loop recover from the world model being wrong about step 2.
            action = _clip(actions[0], cem.maxnorm)
        elif policy == "greedy":
            e = wm.energy_grid(z, pose, z_goal, grid)
            action = _clip(np.concatenate([grid[int(e.argmin())], np.zeros(4)]), cem.maxnorm)
        elif policy == "lookahead":
            e = _true_energies(wm, env, z_goal, grid)
            action = _clip(np.concatenate([grid[int(e.argmin())], np.zeros(4)]), cem.maxnorm)
        elif policy == "random":
            d = rng.normal(size=3)
            d = d / max(np.linalg.norm(d), 1e-9) * cem.maxnorm
            action = _clip(np.concatenate([d, np.zeros(4)]), cem.maxnorm)
        elif policy == "oracle":
            d = goal_pos - env.ee_pos
            n = np.linalg.norm(d)
            d = d / n * min(n, cem.maxnorm) if n > 1e-9 else d
            action = _clip(np.concatenate([d, np.zeros(4)]), cem.maxnorm)
        else:
            raise ValueError(f"unknown policy {policy!r}")

        ep.actions.append(action.copy())
        env.step(action)
        if on_step is not None:
            on_step(ep, t, steps)

    # Final observation, so `dist` has one entry per frame and the last action's
    # effect is actually scored rather than silently dropped.
    frame = env.render(render_size)
    obs = frame if render_size == ac.CROP else env.render(ac.CROP)
    z = wm.encode(obs[None])
    ep.frames.append(frame)
    ep.dist.append(float(np.linalg.norm(env.ee_pos - goal_pos)))
    ep.energy.append(float(wm.energy(z[:, -ac.TOKENS_PER_FRAME :], z_goal[:, -ac.TOKENS_PER_FRAME :])))
    ep.seconds = time.time() - t0
    return ep


# ── the dream, and how to look at it ────────────────────────────────────────────


def frame_bank(env, n: int = 160, seed: int = 0, size: int = None, render_size: int = 384):
    """Render the arm at many positions across the workspace.

    This is the dictionary that `retrieval_decode` looks the dream up in. It is
    built by actually moving the arm and photographing it, so every entry is a real
    image of a real reachable configuration -- which is what makes the decode
    honest. Poses are sampled on a jittered grid rather than uniformly at random so
    the workspace is covered evenly and the nearest neighbour is never nearest only
    because nothing else was close.
    """
    size = size or ac.CROP
    rng = np.random.default_rng(seed)
    lo = np.array([0.33, -0.28, 0.04])
    hi = np.array([0.72, 0.34, 0.42])
    side = int(round(n ** (1 / 3)))
    pts = []
    for i in range(side):
        for j in range(side):
            for k in range(side):
                f = (np.array([i, j, k]) + rng.uniform(0.15, 0.85, 3)) / side
                pts.append(lo + f * (hi - lo))
    rng.shuffle(pts)
    pts = pts[:n]

    obs, big, poses = [], [], []
    for p in pts:
        env.move_to(p)
        obs.append(env.render(size))
        big.append(env.render(render_size))
        poses.append(env.ee_pos.copy())
    env.reset()
    return np.stack(obs), np.stack(big), np.stack(poses)


def retrieval_decode(wm, dreamed: torch.Tensor, bank_z: torch.Tensor):
    """Nearest real frame to each dreamed latent. Indices and distances.

    V-JEPA 2 has no decoder, so there is no way to render what the predictor
    imagined. The alternative most demos reach for -- projecting the latent to RGB
    somehow -- produces a picture that is not a prediction, and the README of this
    repo is largely a list of those failures. Retrieval sidesteps it: we never
    claim to show the dream, we show *the closest real photograph to the dream*,
    and we report how close that was. When the dream leaves the manifold of real
    frames, the retrieval distance climbs and the video visibly stops tracking,
    which is exactly the failure you want to be able to see.
    """
    # One dreamed step at a time. The broadcast form allocates
    # horizon x bank x 256 x 1408 floats at once -- 1.4 GB for a horizon of 8 against
    # a bank of 125, and it grows with both.
    flat_bank = bank_z.flatten(1)
    idx, dist = [], []
    for t in range(dreamed.shape[0]):
        d = (dreamed[t].flatten()[None, :] - flat_bank).abs().mean(-1)
        j = int(d.argmin())
        idx.append(j)
        dist.append(float(d[j]))
    return np.array(idx), np.array(dist)


def dream_rollout(wm, env, start_frame, actions: np.ndarray, goal_pos=None):
    """Dream an action sequence open-loop, and record the true future beside it.

    Returns (dreamed [H, 256, D], true [H, 256, D], true_frames, poses).

    The comparison is the measurement: both sequences start from the same frame and
    receive the same actions, one rolled forward by the world model in latent space
    and one by actually moving the robot and re-encoding the camera. Where they
    diverge is where the world model is wrong, and `viz.dream_chart` plots that
    against the only floor that means anything here -- the distance between two
    unrelated real frames.
    """
    z0 = wm.encode(start_frame[None])[:, -ac.TOKENS_PER_FRAME :]
    pose0 = torch.tensor(env.pose7(), device=wm.device, dtype=wm.dtype)[None, None]
    a = torch.as_tensor(actions, device=wm.device, dtype=wm.dtype)[None]
    dreamed, _ = wm.dream(z0, pose0, a)

    true_z, true_frames = [], []
    for t in range(len(actions)):
        env.step(actions[t])
        f = env.render(ac.CROP)
        true_frames.append(f)
        true_z.append(wm.encode(f[None])[:, -ac.TOKENS_PER_FRAME :])
    return dreamed[0], torch.cat(true_z, 0), np.stack(true_frames)


# ── the push task: a reference photograph of the finished job ───────────────────
#
# Reaching has a flaw that only surfaced once it was controlled properly: its goal
# photograph differs from the start photograph only in where the ARM is, so any image
# distance is monotone in arm position and raw pixels plan it exactly as well as
# V-JEPA. Pushing an object changes the question. The goal photo now shows the WORLD in
# a different state, and walking the arm toward it is no longer sufficient.
#
# It also creates a way to cheat, which is why the metric below is the cube's
# displacement and never the arm's: a policy that merely poses the arm to match the
# photograph scores zero, however good the image similarity gets.


class PixelReward:
    """Mean |image - goal image|. The cheap alternative that tied with V-JEPA on reach."""

    def __init__(self, goal_frame):
        self.g = goal_frame.astype(np.float32)

    def __call__(self, frames):
        return np.abs(frames.astype(np.float32) - self.g).mean(axis=(1, 2, 3)) / 255.0


class EncoderReward:
    """L1 between layer-normed patch tokens of the frame and of the goal photograph."""

    def __init__(self, model, goal_frame, device="cuda", dtype=torch.bfloat16):
        self.model, self.device, self.dtype = model, device, dtype
        self.zg = self.encode(goal_frame[None])

    @torch.no_grad()
    def encode(self, frames):
        x = torch.as_tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2)
        x = x[:, None].repeat(1, 2, 1, 1, 1).to(self.device, self.dtype)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device, dtype=self.dtype) * 255
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device, dtype=self.dtype) * 255
        x = (x - mean[None, None, :, None, None]) / std[None, None, :, None, None]
        h = self.model.get_vision_features(x)
        return torch.nn.functional.layer_norm(h.float(), (h.size(-1),))

    def __call__(self, frames):
        z = self.encode(frames)
        return (z - self.zg).abs().mean(dim=[1, 2]).cpu().numpy()


@dataclass
class PushEpisode:
    policy: str
    frames: list = field(default_factory=list)
    cube_dist: list = field(default_factory=list)   # cube -> goal, metres
    arm_dist: list = field(default_factory=list)    # gripper -> cube, metres
    energy: list = field(default_factory=list)
    seconds: float = 0.0

    @property
    def closed(self) -> float:
        d0, d1 = self.cube_dist[0], self.cube_dist[-1]
        return float((d0 - d1) / max(d0, 1e-6))

    @property
    def cube_moved(self) -> float:
        return float(self.cube_dist[0] - self.cube_dist[-1])


def run_push_episode(env, goal_frame, goal_cube, reward, policy: str = "search",
                     steps: int = 22, maxnorm: float = 0.05, grid_n: int = 3,
                     render_size: int = 384, seed: int = 0, cube0=None,
                     on_step=None) -> PushEpisode:
    """One push episode.

    `reward` is any callable mapping a batch of frames to a batch of costs, so the
    identical search runs on V-JEPA tokens, raw pixels or a random network and nothing
    else differs. Candidate actions are evaluated by actually stepping the simulator
    and re-rendering, i.e. the `lookahead` setting from the reach task, because the
    pretrained AC predictor's imagined dynamics were shown not to transfer.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    g = np.linspace(-maxnorm, maxnorm, grid_n)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3).astype(np.float32)

    ep = PushEpisode(policy=policy)
    env.set_gripper(1.0)  # closed fingers are the pusher; see sim.scripted_push
    t0 = time.time()

    def record():
        ep.frames.append(env.render(render_size))
        ep.cube_dist.append(float(np.linalg.norm(env.target_pos - goal_cube)))
        ep.arm_dist.append(float(np.linalg.norm(env.ee_pos - env.target_pos)))

    record()
    ep.energy.append(float(reward(env.render(ac.CROP)[None])[0]))

    if policy == "oracle":
        import sim as _sim

        _sim.scripted_push(env, cube0 if cube0 is not None else env.target_pos, goal_cube)
        record()
        ep.energy.append(float(reward(env.render(ac.CROP)[None])[0]))
        ep.seconds = time.time() - t0
        return ep

    for t in range(steps):
        if policy == "random":
            d = rng.normal(size=3)
            a3 = d / max(np.linalg.norm(d), 1e-9) * maxnorm
        else:
            q, v, c = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
            cands = []
            for a in grid:
                env.data.qpos[:] = q; env.data.qvel[:] = v; env.data.ctrl[:] = c
                mujoco.mj_forward(env.model, env.data)
                act = np.zeros(7, dtype=np.float32); act[:3] = a
                env.step(act)
                cands.append(env.render(ac.CROP))
            env.data.qpos[:] = q; env.data.qvel[:] = v; env.data.ctrl[:] = c
            mujoco.mj_forward(env.model, env.data)
            costs = reward(np.stack(cands))
            a3 = grid[int(np.argmin(costs))]

        act = np.zeros(7, dtype=np.float32); act[:3] = a3
        env.step(act)
        record()
        ep.energy.append(float(reward(env.render(ac.CROP)[None])[0]))
        if on_step is not None:
            on_step(ep, t, steps)

    ep.seconds = time.time() - t0
    return ep


def run_push_cem(env, goal_frame, goal_cube, reward, steps: int = 22,
                 horizon: int = 4, samples: int = 24, topk: int = 6, iters: int = 2,
                 maxnorm: float = 0.05, render_size: int = 384, seed: int = 0,
                 on_step=None) -> PushEpisode:
    """Multi-step CEM for the push task, to test the local-minimum hypothesis.

    One-step descent on image similarity fails the cold-start push for a structural
    reason rather than a perceptual one: to push, the gripper must travel AROUND to the
    far side of the block, and every step of that detour makes the camera image LESS
    like the reference photograph. A greedy planner cannot pay that cost, whatever
    space the images are compared in, so the cold-start failure says nothing about the
    features.

    A horizon of several steps can pay it, because only the END of the sequence is
    scored. That makes this a clean test of the explanation: if the failure really is a
    local minimum, extending the horizon should fix it without changing the reward. If
    the horizon does not help either, the diagnosis was wrong.

    Costs `samples * horizon` simulator steps and `samples` encodes per CEM iteration,
    all of it rewound afterwards, which is affordable only because the reward is an
    encoder and not a diffusion model.
    """
    import mujoco

    rng = np.random.default_rng(seed)
    ep = PushEpisode(policy=f"cem-h{horizon}")
    env.set_gripper(1.0)
    t0 = time.time()

    def record():
        ep.frames.append(env.render(render_size))
        ep.cube_dist.append(float(np.linalg.norm(env.target_pos - goal_cube)))
        ep.arm_dist.append(float(np.linalg.norm(env.ee_pos - env.target_pos)))

    record()
    ep.energy.append(float(reward(env.render(ac.CROP)[None])[0]))

    for t in range(steps):
        q, v, c = env.data.qpos.copy(), env.data.qvel.copy(), env.data.ctrl.copy()
        mean = np.zeros((horizon, 3), dtype=np.float32)
        std = np.full((horizon, 3), maxnorm, dtype=np.float32)

        for _ in range(iters):
            seqs = np.clip(
                rng.normal(size=(samples, horizon, 3)).astype(np.float32) * std + mean,
                -maxnorm, maxnorm,
            )
            finals = []
            for s in range(samples):
                env.data.qpos[:] = q; env.data.qvel[:] = v; env.data.ctrl[:] = c
                mujoco.mj_forward(env.model, env.data)
                for h in range(horizon):
                    act = np.zeros(7, dtype=np.float32)
                    act[:3] = seqs[s, h]
                    env.step(act)
                finals.append(env.render(ac.CROP))
            env.data.qpos[:] = q; env.data.qvel[:] = v; env.data.ctrl[:] = c
            mujoco.mj_forward(env.model, env.data)

            costs = reward(np.stack(finals))
            elite = seqs[np.argsort(costs)[:topk]]
            # Momentum keeps the distribution from collapsing onto one sequence after a
            # single lucky sample, which at these sample counts happens easily.
            mean = 0.5 * mean + 0.5 * elite.mean(0)
            std = 0.5 * std + 0.5 * (elite.std(0) + 1e-4)

        act = np.zeros(7, dtype=np.float32)
        act[:3] = np.clip(mean[0], -maxnorm, maxnorm)
        env.step(act)
        record()
        ep.energy.append(float(reward(env.render(ac.CROP)[None])[0]))
        if on_step is not None:
            on_step(ep, t, steps)

    ep.seconds = time.time() - t0
    return ep
