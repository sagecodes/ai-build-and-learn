"""Imagination vs reality: watch the world model dream a whole movie, then run it.

`plan` and `adapt` score the world model by what the arm does. This module makes the
thing being scored visible. The robot is handed a short choreography (reach the red
block, sweep up and over, trace a square) as a list of end-effector deltas, and the
world model imagines the entire move open loop, from the first frame only:

    z_0 = encode(first frame)
    z_t+1 = predictor(z_t, action_t, pose_t)        # never looks at the sim again

The simulator then executes the same actions, and the two movies are laid side by
side. Twice: once with the pretrained DROID checkpoint, once after the few minutes of
reward-free adaptation that `adapt` showed fixes planning.

── Showing a dream from a model with no decoder ────────────────────────────────
Each imagined latent is looked up in a bank of real renders of the arm at a few hundred
positions across the workspace, and the panel shows the nearest one. None of the bank
frames come from the choreography being dreamed. That lookup also gives the headline
number for free: the bank frame's gripper position is where the model *thinks* the
hand went, so "imagined gripper vs real gripper" is a distance in centimetres.

Two references keep that number honest:
  standstill    imagine that nothing moves (the hand stays at its start). A dream that
                cannot beat this is not using the actions at all.
  decode floor  run the TRUE future latent through the same lookup. The bank is a grid,
                so even a perfect dream lands a few cm off; this is that resolution.

── Markov rollout ──────────────────────────────────────────────────────────────
The dream feeds back only the last imagined frame, not the whole imagined history.
That is the regime `adapt` fine-tunes (one frame of context in, next frame out) and the
one `greedy` planning queries, so it is the fair setting for both checkpoints. The
full-history rollout the predictor also supports is measured alongside and reported in
the table, not the video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

import ac
import plan as planning

MAXNORM = 0.05
MAX_STEPS = 26

# Workspace the retrieval bank covers. The choreographies stay a few cm inside it, or
# the nearest bank frame to a correct dream would be a wall of the box.
LO = np.array([0.33, -0.28, 0.04])
HI = np.array([0.72, 0.34, 0.42])


def waypoints(play: str, goal_pos: np.ndarray) -> list[np.ndarray]:
    """Absolute end-effector waypoints for each choreography. Home is (0.60, 0, 0.23)."""
    if play == "reach":
        # come in from above, the way a person would, rather than skimming the clutter
        pts = [goal_pos + np.array([0.0, 0.0, 0.14]), goal_pos]
    elif play == "sweep":
        # up, across to the far side, down to the table, back across low
        pts = [(0.60, 0.00, 0.36), (0.55, 0.22, 0.36), (0.55, 0.22, 0.12), (0.52, -0.12, 0.12)]
    elif play == "square":
        pts = [(0.47, -0.12, 0.20), (0.47, 0.18, 0.20), (0.65, 0.18, 0.20), (0.65, -0.12, 0.20),
               (0.47, -0.12, 0.20)]
    else:
        raise ValueError(f"unknown play {play!r}")
    return [np.clip(np.asarray(p, dtype=np.float64), LO + 0.02, HI - 0.02) for p in pts]


def waypoint_actions(start: np.ndarray, pts: list[np.ndarray], maxnorm: float = MAXNORM,
                     max_steps: int = MAX_STEPS) -> np.ndarray:
    """Straight-line legs between waypoints, cut into steps of at most `maxnorm`."""
    pos = np.asarray(start, dtype=np.float64).copy()
    acts = []
    for w in pts:
        while len(acts) < max_steps:
            d = w - pos
            n = float(np.linalg.norm(d))
            if n < 5e-3:
                break
            d = d / n * min(n, maxnorm)
            pos = pos + d
            a = np.zeros(7, dtype=np.float32)
            a[:3] = d
            acts.append(a)
    return np.stack(acts)


@dataclass
class Truth:
    """What the simulator did with the choreography."""

    play: str
    actions: np.ndarray       # [H, 7]
    big: np.ndarray           # [H+1, R, R, 3] frames for the video
    pos: np.ndarray           # [H+1, 3] gripper positions
    z: torch.Tensor           # [H+1, 256, D] encoded frames
    pose0: np.ndarray         # [7] the start state the predictor conditions on


def act_out(wm, env, play: str, actions: np.ndarray, render: int) -> Truth:
    """Execute the choreography for real, from the env's current (home) state."""
    pose0 = env.pose7()
    big, crops, pos = [env.render(render)], [env.render(ac.CROP)], [env.ee_pos.copy()]
    for a in actions:
        env.step(a)
        big.append(env.render(render))
        crops.append(env.render(ac.CROP))
        pos.append(env.ee_pos.copy())
    crops = np.stack(crops)
    z = torch.cat([wm.encode(crops[i : i + 8]).view(-1, ac.TOKENS_PER_FRAME, wm.dim)
                   for i in range(0, len(crops), 8)])
    return Truth(play, actions, np.stack(big), np.stack(pos), z, pose0)


@torch.no_grad()
def dream_markov(wm, z0: torch.Tensor, pose0: np.ndarray, actions: np.ndarray) -> torch.Tensor:
    """Open-loop dream, one frame of context per step. [256, D] -> [H, 256, D]."""
    z = z0[None]
    p = torch.tensor(pose0, device=wm.device, dtype=wm.dtype)[None, None]
    out = []
    for a in actions:
        at = torch.as_tensor(a, device=wm.device, dtype=wm.dtype)[None, None]
        z, p = wm.step(z, at, p)
        out.append(z[0])
    return torch.stack(out)


@torch.no_grad()
def dream_history(wm, z0: torch.Tensor, pose0: np.ndarray, actions: np.ndarray) -> torch.Tensor:
    """Open-loop dream with the full imagined history as context (upstream's `dream`)."""
    p = torch.tensor(pose0, device=wm.device, dtype=wm.dtype)[None, None]
    a = torch.as_tensor(actions, device=wm.device, dtype=wm.dtype)[None]
    d, _ = wm.dream(z0[None], p, a)
    return d[0]


@dataclass
class Imagined:
    """One checkpoint's dream of one choreography, decoded and scored."""

    idx: np.ndarray        # [H] bank index of the nearest real frame
    retr: np.ndarray       # [H] retrieval distance
    pos_err: np.ndarray    # [H] cm, imagined gripper vs real gripper
    lat_err: np.ndarray    # [H] L1, dreamed latent vs true latent
    hist_pos_err: np.ndarray  # [H] cm, same for the full-history rollout


def imagine(wm, truth: Truth, bank_z: torch.Tensor, bank_pos: np.ndarray) -> Imagined:
    dreamed = dream_markov(wm, truth.z[0], truth.pose0, truth.actions)
    idx, retr = planning.retrieval_decode(wm, dreamed, bank_z)
    hist = dream_history(wm, truth.z[0], truth.pose0, truth.actions)
    hidx, _ = planning.retrieval_decode(wm, hist, bank_z)
    real = truth.pos[1:]
    return Imagined(
        idx=idx,
        retr=retr,
        pos_err=np.linalg.norm(bank_pos[idx] - real, axis=1) * 100,
        lat_err=(dreamed - truth.z[1:]).abs().mean(dim=[1, 2]).float().cpu().numpy(),
        hist_pos_err=np.linalg.norm(bank_pos[hidx] - real, axis=1) * 100,
    )


def references(wm, truth: Truth, bank_z: torch.Tensor, bank_pos: np.ndarray) -> dict:
    """The two lines every imagined-position curve is read against."""
    real = truth.pos[1:]
    tidx, _ = planning.retrieval_decode(wm, truth.z[1:], bank_z)
    return {
        "standstill": np.linalg.norm(truth.pos[0][None] - real, axis=1) * 100,
        "floor": np.linalg.norm(bank_pos[tidx] - real, axis=1) * 100,
        "lat_standstill": (truth.z[:1] - truth.z[1:]).abs().mean(dim=[1, 2]).float().cpu().numpy(),
    }
