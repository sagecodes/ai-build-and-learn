"""Adapting V-JEPA 2-AC's dynamics to a simulator it has never seen.

`plan` establishes that the two halves of this world model transfer very differently.
The encoder is fine on MuJoCo renders: its embedding distance to a goal photograph
correlates r=+0.77 with true end-effector distance, and searching on that alone
(`lookahead`) closes 90% of the gap. The action-conditioned predictor is not fine: its
imagined futures rank actions backwards, and planning on them is worse than a random
walk.

Two cheap explanations were tested and rejected before reaching for training. Neither
blur nor JPEG artifacts nor sensor noise fixes it: adding Gaussian noise moved the
action-ranking correlation from +0.09 to +0.61, which looked promising, and then made
closed-loop planning *worse* (-123% to -181%). That is a useful reminder that a
correlation over 27 candidate actions at one state is easy to move by accident, and
the only number worth trusting is the one you actually care about.

What does work is the obvious thing, and it is remarkably cheap:

    freeze the encoder (it already works)
    drive the arm around, collecting (z_t, action, pose) -> z_t+1
    fine-tune ONLY the predictor, on its own original L1 objective

480 transitions and 400 steps -- under seven minutes end to end on one GB10 -- take the
predictor from losing to "predict no change" to beating it, and take greedy planning
from -123% to +88%, which is within noise of the +91% that a *perfect* simulator-backed
dynamics model achieves. The remaining gap between a learned world model and ground
truth, after six minutes of adaptation, is about three percentage points.

The honest framing of the whole demo is therefore neither "it works" nor "it doesn't":
a pretrained world model's *representation* is worth a great deal off the shelf, its
*dynamics* are worth very little off the shelf, and the dynamics are the half that is
cheap to repair.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn.functional as F

import ac
import sim

MAXNORM = 0.05


def collect_transitions(
    wm: ac.ActionWorldModel,
    episodes: int = 40,
    steps: int = 12,
    seed: int = 0,
    on_progress=None,
):
    """Drive the arm around and record latent transitions under the FROZEN encoder.

    Returns (Z, A, P, Zn) on the CPU: context latents, 7-dim actions, 7-dim poses, and
    the latent that actually followed.

    Two choices that matter. Each episode starts from a *random* workspace position
    rather than the home pose, or the whole dataset would describe one corner of the
    space and the adapted predictor would only be right there. And the actions are
    uniform random within the same per-step budget the planner uses, so the model is
    trained on the distribution it will be queried on -- collecting with a goal-seeking
    policy instead would only ever show it good actions, and the planner's entire job
    is telling good ones from bad.
    """
    rng = np.random.default_rng(seed)
    Z, A, P, Zn = [], [], [], []
    t0 = time.time()
    for ep in range(episodes):
        env = sim.PandaReach(seed=int(rng.integers(0, 1000)))
        env.move_to(
            np.array([rng.uniform(0.36, 0.68), rng.uniform(-0.25, 0.30), rng.uniform(0.07, 0.38)])
        )
        for _ in range(steps):
            frame = env.render(ac.CROP)
            pose = env.pose7()
            a3 = rng.uniform(-MAXNORM, MAXNORM, 3).astype(np.float32)
            action = np.concatenate([a3, np.zeros(4, dtype=np.float32)])
            env.step(action)
            nxt = env.render(ac.CROP)
            with torch.no_grad():
                Z.append(wm.encode(frame[None])[:, -ac.TOKENS_PER_FRAME :].cpu())
                Zn.append(wm.encode(nxt[None])[:, -ac.TOKENS_PER_FRAME :].cpu())
            A.append(action)
            P.append(pose)
        env.close()
        if on_progress is not None:
            on_progress(ep + 1, episodes, (ep + 1) * steps, time.time() - t0)
    return (
        torch.cat(Z),
        torch.tensor(np.array(A), dtype=torch.float32),
        torch.tensor(np.array(P), dtype=torch.float32),
        torch.cat(Zn),
    )


def _predict(wm, Z, A, P, idx):
    z = Z[idx].to(wm.device)
    a = A[idx].to(wm.device)[:, None]
    p = P[idx].to(wm.device)[:, None]
    out = wm.predictor(z, a, p)[:, -ac.TOKENS_PER_FRAME :]
    return F.layer_norm(out, (out.size(-1),))


def evaluate(wm, Z, A, P, Zn, idx, chunk: int = 16) -> float:
    """Mean L1 between the predicted next latent and the true one."""
    wm.predictor.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for i in range(0, len(idx), chunk):
            sl = idx[i : i + chunk]
            tot += float((_predict(wm, Z, A, P, sl) - Zn[sl].to(wm.device)).abs().mean()) * len(sl)
            n += len(sl)
    return tot / max(n, 1)


def standstill_baseline(Z, Zn, idx, device) -> float:
    """The only baseline that matters: predict that nothing happens.

    A dynamics model that cannot beat this has not earned the word prediction, and the
    pretrained checkpoint does not beat it on MuJoCo renders (0.391 vs 0.298).
    """
    return float((Z[idx].to(device) - Zn[idx].to(device)).abs().mean())


def finetune(
    wm: ac.ActionWorldModel,
    Z,
    A,
    P,
    Zn,
    train_idx,
    val_idx,
    steps: int = 400,
    batch: int = 8,
    lr: float = 1e-5,
    seed: int = 0,
    on_step=None,
):
    """Fine-tune ONLY the predictor. Returns the val-loss history.

    The encoder is explicitly frozen rather than merely left out of the optimiser: it
    is the half that already works, and a stray gradient into it would quietly destroy
    the one thing this demo has shown to transfer. lr=1e-5 because the objective is
    the model's own and we are adapting, not retraining -- higher rates reach a lower
    training loss and a worse validation loss.
    """
    for q in wm.encoder.parameters():
        q.requires_grad_(False)
    opt = torch.optim.AdamW(wm.predictor.parameters(), lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    history = []
    wm.predictor.train()
    for s in range(steps):
        idx = torch.tensor(rng.choice(train_idx, batch, replace=False))
        loss = (_predict(wm, Z, A, P, idx) - Zn[idx].to(wm.device)).abs().mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(wm.predictor.parameters(), 1.0)
        opt.step()
        if (s + 1) % 50 == 0 or s == 0:
            v = evaluate(wm, Z, A, P, Zn, val_idx)
            wm.predictor.train()
            history.append((s + 1, float(loss.detach()), v))
            if on_step is not None:
                on_step(s + 1, steps, float(loss.detach()), v)
    wm.predictor.eval()
    return history
