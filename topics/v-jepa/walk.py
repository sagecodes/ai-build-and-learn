"""V-JEPA 2-AC steers a walking humanoid to a photograph.

The Franka tasks (`plan`, `adapt`, `imagine`) showed a pretrained world model's encoder
transfers to simulation and its dynamics do not, and that a few minutes of reward-free
adaptation fixes the dynamics. This asks the same question of a robot the checkpoint
has never seen at all: a Unitree G1 humanoid, whose legs are an RL policy
(walker.py) and whose only high-level action is "walk this far, this way".

    collect   the G1 wanders at random for a few minutes. No goal, no reward. Every
              0.6 s decision is one (frame, move, pose) -> next frame transition.
    adapt     fine-tune ONLY the 305M predictor on those transitions, with its own
              original L1 objective. The encoder stays frozen.
    plan      show it a photo of the robot standing on a coloured pad. Each decision,
              imagine every candidate move, take the one whose imagined future looks
              most like the photo, execute it, look again.
    dream     hand it a whole walk and let it imagine the full thing from the first
              frame, then let the simulator do it.

The action slot of a Franka world model is being reused for footsteps, which sounds
absurd and is exactly the point: `adapt` does not add a single parameter, it only
teaches the existing action embedding what a move means on this body.

── The same four policies as the Franka `plan` task ────────────────────────────
  oracle     walk straight at the pad's coordinates. The ceiling.
  random     random moves of the same size. The floor.
  lookahead  the SAME V-JEPA reward and the same candidate moves, but each move is
             tried for real in the simulator (MJX state is an immutable pytree, so a
             rewind is free) and re-encoded. Perfect dynamics, learned reward.
  jepa       the same search, every candidate IMAGINED by the world model.
             Run once with the pretrained predictor and once adapted.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch

import ac
import plan as planning
import walker as wk

REACHED_M = 0.5


def encode(wm, frames: np.ndarray) -> torch.Tensor:
    """[N, H, W, 3] -> [N, 256, D], chunked."""
    return torch.cat([wm.encode(frames[i : i + 8]).view(-1, ac.TOKENS_PER_FRAME, wm.dim)
                      for i in range(0, len(frames), 8)])


def action7(dxy) -> np.ndarray:
    a = np.zeros(7, dtype=np.float32)
    a[:2] = np.asarray(dxy, dtype=np.float32) * wk.POSE_SCALE
    return a


def _pose(wm, w) -> torch.Tensor:
    return torch.tensor(w.pose7(), device=wm.device, dtype=wm.dtype)[None, None]


_ANG = np.linspace(0, 2 * np.pi, 8, endpoint=False)


# ── collect ─────────────────────────────────────────────────────────────────────


@dataclass
class Play:
    """Random wandering, as latent transitions plus the frames for the lookup bank."""

    Z: torch.Tensor
    A: torch.Tensor
    P: torch.Tensor
    Zn: torch.Tensor
    bank_frames: np.ndarray   # [N, 256, 256, 3] model-view frames after each move
    bank_show: np.ndarray     # [N, R, R, 3] the same moments from the people camera
    bank_pos: np.ndarray      # [N, 2] where the robot was in each
    falls: int = 0
    seconds: float = 0.0


def collect(wm, w, episodes: int = 30, steps: int = 24, seed: int = 0,
            render: int = 384, on_progress=None) -> Play:
    """Wander at random and record every transition under the frozen encoder.

    Moves are uniform in direction and length, with one in eight a deliberate
    stand-still (the planner needs to know what "stay here" looks like), and the robot
    is herded back toward the middle once it strays past the arena radius so the data
    covers the space the planner will actually be asked about.
    """
    rng = np.random.default_rng(seed)
    Z, A, P, Zn, frames, shows, poss = [], [], [], [], [], [], []
    falls, t0 = 0, time.time()
    for ep in range(episodes):
        w.reset(seed=10_000 + seed * 1000 + ep)
        first = len(shows)
        f = w.render_chase(ac.CROP)
        z = encode(wm, f[None])
        for _ in range(steps):
            pose = w.pose7()
            if rng.random() < 0.125:
                d = np.zeros(2)
            else:
                ang = rng.uniform(0, 2 * np.pi)
                d = np.array([np.cos(ang), np.sin(ang)]) * rng.uniform(0.1, wk.MAX_STEP_M)
                if np.linalg.norm(w.pos) > wk.ARENA:
                    home = -w.pos / np.linalg.norm(w.pos)
                    d = (home + 0.4 * rng.normal(size=2)) * rng.uniform(0.25, wk.MAX_STEP_M)
            w.walk(d)
            if w.fell:
                falls += 1
                break
            f2 = w.render_chase(ac.CROP)
            z2 = encode(wm, f2[None])
            Z.append(z.cpu()); Zn.append(z2.cpu())
            A.append(action7(d)); P.append(pose)
            frames.append(f2); shows.append(w.render_show(render)); poss.append(w.pos.copy())
            z = z2
        if on_progress is not None:
            on_progress(ep + 1, episodes, len(A), falls, time.time() - t0,
                        shows[first:], np.array(poss))
    return Play(torch.cat(Z), torch.tensor(np.array(A)), torch.tensor(np.array(P)),
                torch.cat(Zn), np.stack(frames), np.stack(shows), np.stack(poss), falls,
                time.time() - t0)


# ── plan ────────────────────────────────────────────────────────────────────────


@dataclass
class WalkEpisode:
    policy: str
    pad: str
    chase: list = field(default_factory=list)   # big low chase frames (people), for the video
    maps: list = field(default_factory=list)
    seen: list = field(default_factory=list)    # what the model actually saw (256, high cam)
    dist: list = field(default_factory=list)    # metres to the pad centre
    energy: list = field(default_factory=list)
    stopped: int | None = None                  # decision at which the planner stood still twice
    seconds: float = 0.0

    @property
    def closed(self) -> float:
        return float((self.dist[0] - self.dist[-1]) / max(self.dist[0], 1e-6))

    @property
    def reached(self) -> bool:
        return self.dist[-1] < REACHED_M


def goal_photo(w, pad: str, seed: int = 999) -> np.ndarray:
    """A real chase-camera photo of the robot standing on the pad.

    Taken by walking there with the oracle and letting the gait settle, from a
    different spawn than any evaluation episode, so the planner is never shown a
    frame from its own run.
    """
    w.reset(seed=seed)
    for _ in range(30):
        if wk.pad_distance(w.pos, pad) < 0.08:
            break
        w.walk(wk.oracle_step(w.pos, wk.PADS[pad][0]))
    w.settle(40)
    return w.render_chase(ac.CROP)


def energy(z: torch.Tensor, zg: torch.Tensor) -> torch.Tensor:
    """Centred cosine distance, per token, averaged. [N, 256, D] vs [1, 256, D] -> [N].

    NOT the raw L1 the Franka tasks use. Raw feature-space L1 is gamed by low-norm
    embeddings (the `energy` task measured a flat grey video beating the truth), and on
    this camera it is measurably worse: rank correlation with distance 0.80 raw vs 0.85
    centred, best-neighbour-is-closer 84% vs 89%. Centring removes the component every
    token of a frame shares, which is mostly "this is a brown floor from above".
    """
    c = z - z.mean(1, keepdim=True)
    cg = zg - zg.mean(1, keepdim=True)
    return 1 - torch.nn.functional.cosine_similarity(c, cg, dim=-1).mean(1)


# Rays, not steps. Each candidate is "keep walking this way for H moves" (plus
# standing still), scored by how much the END of the ray looks like the goal photo, and
# only its first move is executed before re-planning. Measured reason: from 13 m up a
# single 0.45 m move shifts the image by less than one 16 px V-JEPA patch, and per-patch
# features barely notice, so one-move lookahead has almost no slope far from the goal
# (the lowest-energy single move got closer only 28% of the time, with PERFECT dynamics).
# A ray reaches into the basin around the goal from much further out.
_DIRS = np.stack([np.cos(_ANG), np.sin(_ANG)], 1).astype(np.float32) * wk.MAX_STEP_M
RAYS = np.concatenate([np.zeros((1, 2), np.float32), _DIRS])      # [9, 2]


@torch.no_grad()
def _energies_imagined(wm, z, pose, zg, horizon: int) -> np.ndarray:
    """Dream each ray `horizon` moves ahead, one frame of context per step."""
    n = len(RAYS)
    a = torch.zeros(n, 1, 7, device=wm.device, dtype=wm.dtype)
    a[:, 0, :2] = torch.as_tensor(RAYS * wk.POSE_SCALE, device=wm.device, dtype=wm.dtype)
    zz, pp = z[:, -ac.TOKENS_PER_FRAME:].repeat(n, 1, 1), pose.repeat(n, 1, 1)
    for _ in range(horizon):
        zz, pp = wm.step(zz, a, pp)
    return energy(zz, zg).float().cpu().numpy()


def _energies_real(wm, w, zg, horizon: int) -> np.ndarray:
    """Walk each ray for real, photograph the end, rewind. Perfect dynamics."""
    snap = w.snapshot()
    frames = []
    for d in RAYS:
        w.restore(snap)
        for _ in range(horizon):
            w.walk(d)
        frames.append(w.render_chase(ac.CROP))
    w.restore(snap)
    return energy(encode(wm, np.stack(frames)), zg).float().cpu().numpy()


def run_walk(wm, w, pad: str, goal_frame: np.ndarray, policy: str, steps: int = 14,
             horizon: int = 4, seed: int = 0, render: int = 384, keep_going: bool = False,
             ep: WalkEpisode | None = None, on_step=None) -> WalkEpisode:
    """One closed-loop walk toward a pad under `policy`, from the walker's current state.

    Unless `keep_going`, the episode ends early once the planner chooses to stand still
    on two consecutive decisions: that is the planner's own "I am there", and using the
    ground-truth distance to stop would smuggle the answer in.
    """
    rng = np.random.default_rng(seed)
    ep = ep or WalkEpisode(policy=policy, pad=pad)
    zg = encode(wm, goal_frame[None])
    t0 = time.time()
    still = 0

    def record():
        f = w.render_chase(ac.CROP)
        z = encode(wm, f[None])
        ep.seen.append(f)
        ep.chase.append(w.render_show(render))
        ep.maps.append(w.render_map(render, goal=pad))
        ep.dist.append(wk.pad_distance(w.pos, pad))
        ep.energy.append(float(energy(z, zg)))
        return z

    z = record()
    for t in range(steps):
        if policy == "oracle":
            d = wk.oracle_step(w.pos, wk.PADS[pad][0])
        elif policy == "random":
            ang = rng.uniform(0, 2 * np.pi)
            d = np.array([np.cos(ang), np.sin(ang)]) * wk.MAX_STEP_M
        elif policy == "lookahead":
            d = RAYS[int(np.argmin(_energies_real(wm, w, zg, horizon)))]
        elif policy.startswith("jepa"):
            d = RAYS[int(np.argmin(_energies_imagined(wm, z, _pose(wm, w), zg, horizon)))]
        else:
            raise ValueError(policy)
        w.walk(d)
        z = record()
        still = still + 1 if float(np.linalg.norm(d)) == 0.0 else 0
        if on_step is not None:
            on_step(ep, t, steps)
        if w.fell:
            break
        if still >= 2 and not keep_going and policy != "random":
            ep.stopped = t
            break
    ep.seconds += time.time() - t0
    return ep


# ── dream ───────────────────────────────────────────────────────────────────────


@dataclass
class WalkTruth:
    moves: np.ndarray       # [H, 2] commanded world-frame moves
    chase: np.ndarray       # [H+1] big low chase frames (people)
    maps: np.ndarray        # [H+1] map frames
    pos: np.ndarray         # [H+1, 2]
    z: torch.Tensor         # [H+1, 256, D]
    pose0: np.ndarray


def tour_moves(start: np.ndarray, pads: list[str], max_moves: int = 26) -> np.ndarray:
    """The oracle's moves for walking a tour of pads, planned on paper (no sim)."""
    pos, moves = np.asarray(start, dtype=np.float64).copy(), []
    for pad in pads:
        while len(moves) < max_moves:
            d = wk.oracle_step(pos, wk.PADS[pad][0])
            if np.linalg.norm(d) < 0.05:
                break
            moves.append(d)
            pos = pos + d
    return np.array(moves, dtype=np.float32)


def act_out(wm, w, moves: np.ndarray, render: int = 384) -> WalkTruth:
    pose0 = w.pose7()
    chase, maps, crops, pos = [w.render_show(render)], [w.render_map(render)], [w.render_chase(ac.CROP)], [w.pos.copy()]
    for d in moves:
        w.walk(d)
        chase.append(w.render_show(render)); maps.append(w.render_map(render))
        crops.append(w.render_chase(ac.CROP)); pos.append(w.pos.copy())
    return WalkTruth(moves, np.stack(chase), np.stack(maps), np.stack(pos),
                     encode(wm, np.stack(crops)), pose0)


@dataclass
class WalkDream:
    idx: np.ndarray       # nearest bank frame per imagined step
    pos: np.ndarray       # [H, 2] where the model thinks the robot is
    err: np.ndarray       # [H] metres from the real robot


@torch.no_grad()
def dream(wm, truth: WalkTruth, bank_z: torch.Tensor, bank_pos: np.ndarray) -> WalkDream:
    """Imagine the whole walk from the first frame, one frame of context per step."""
    z = truth.z[:1]
    p = torch.tensor(truth.pose0, device=wm.device, dtype=wm.dtype)[None, None]
    out = []
    for d in truth.moves:
        a = torch.as_tensor(action7(d), device=wm.device, dtype=wm.dtype)[None, None]
        z, p = wm.step(z, a, p)
        out.append(z[0])
    idx, _ = planning.retrieval_decode(wm, torch.stack(out), bank_z)
    return WalkDream(idx, bank_pos[idx], np.linalg.norm(bank_pos[idx] - truth.pos[1:], axis=1))


def dream_references(wm, truth: WalkTruth, bank_z, bank_pos) -> dict:
    tidx, _ = planning.retrieval_decode(wm, truth.z[1:], bank_z)
    return {
        "standstill": np.linalg.norm(truth.pos[:1] - truth.pos[1:], axis=1),
        "floor": np.linalg.norm(bank_pos[tidx] - truth.pos[1:], axis=1),
    }
