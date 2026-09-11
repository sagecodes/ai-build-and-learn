"""A small visuomotor policy, and the two datasets that decide whether dreams are usable.

The question every other task in this repo only sets up: **can you train on what the
world model dreams?** `cycle` measured what a dreamed action LABEL costs (about 3x) and
`detail` showed that cost is a distribution gap rather than lost detail. Neither of them
trained anything, so neither could say whether the resulting data is usable.

This closes it, and the experimental design is the point:

    real set    (real DROID frame_t,    action_t)
    dream set   (COSMOS dreamed frame_t, action_t)   <- the SAME actions
    test set    held-out REAL frames and actions

Both training sets carry identical labels, because the dreams were generated FROM those
actions. So the label noise `cycle` measured is deliberately held at zero here and the
only thing that differs between the two policies is the pixels they learned from. Train
two identical networks, test both on real held-out data, and the gap between them is the
pixel domain gap on its own, with nothing else mixed in.

Small on purpose. The claim being tested is "are these pixels trainable", not "how good
a Franka policy can I build", and a big network would take longer to say the same thing.
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def make_policy(action_dim: int = 10, in_channels: int = 6):
    """A small conv net: a FRAME PAIR in, one action vector out. ~400k parameters.

    Six input channels, not three, and that is a correction rather than a detail. The
    first version of this took a single frame and could not beat predicting the training
    mean -- on either dataset -- because DROID's `action` is a VELOCITY command and
    velocity is not observable in a static image. No amount of data fixes an ill-posed
    input: a policy shown one frame and asked how fast the arm is moving is being asked
    to guess, and guessing the mean is the correct answer.

    Stacking frame t and frame t+1 makes the motion visible, which is the same reason
    inverse dynamics is given a video and not a photograph.

    Five stride-2 convolutions take a 192x320 frame down to a coarse feature map, a
    global average pool removes what is left of the spatial layout, and a two-layer head
    regresses the action. GroupNorm rather than BatchNorm because the batches here are
    small and the two training runs must be comparable: BatchNorm's statistics depend on
    batch composition, which would let the two datasets differ for a reason that has
    nothing to do with the pixels.
    """
    import torch.nn as nn

    def block(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, stride=2, padding=1),
            nn.GroupNorm(min(8, cout), cout),
            nn.SiLU(),
        )

    return nn.Sequential(
        block(in_channels, 32), block(32, 64), block(64, 96), block(96, 128), block(128, 128),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, action_dim),
    )


def frames_to_tensor(frames: list):
    """PIL frames to a normalised NCHW float tensor."""
    import numpy as np
    import torch

    arr = np.stack([np.asarray(f.convert("RGB"), dtype="float32") / 255.0 for f in frames])
    return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()


def pairs_to_tensor(frames: list, chunk: int):
    """Consecutive frame PAIRS as 6-channel inputs, one per action.

    `frames` is a flat list of `chunk`-length windows. Pairs are formed strictly inside a
    window, never across the boundary between two, because consecutive windows are from
    different moments of the episode and a pair spanning them describes a cut rather than
    a motion. That costs one sample per window and buys labels that mean something.

    Returns (tensor, keep) where `keep` indexes the actions that survived.
    """
    import torch

    single = frames_to_tensor(frames)
    xs, keep = [], []
    for w in range(len(frames) // chunk):
        base = w * chunk
        for i in range(chunk - 1):
            xs.append(torch.cat([single[base + i], single[base + i + 1]], dim=0))
            keep.append(base + i)
    return torch.stack(xs), keep


def mean_baseline(y_train, y_test) -> dict:
    """What you get for free by ignoring the images entirely.

    Reported alongside every trained policy, and not optional. The first run of this
    experiment produced two policies at MAE 0.239 and 0.261 and a tidy-looking 1.10x
    ratio between them, and both were WORSE than this baseline's 0.152. Without the
    constant predictor sitting next to them, that ratio reads as a finding about
    synthetic data when it is a comparison of two models that learned nothing.
    """
    import torch

    const = y_train.mean(0, keepdim=True)
    err = (y_test - const).abs()
    span = y_test.max(0).values - y_test.min(0).values
    moving = (span > 0.1 * float(span.max())).nonzero().flatten().tolist() or [0]
    return {"mae": float(err.mean()), "mae_moving": float(err[:, moving].mean())}


def train_policy(
    images,
    actions,
    *,
    epochs: int = 60,
    batch: int = 32,
    lr: float = 1e-3,
    device: str = "cuda",
    seed: int = 0,
    on_epoch=None,
    val=None,
):
    """Behaviour cloning by regression. Returns (model, history).

    `seed` seeds the weights AND the batch order, and both policies are trained with the
    same one. Without that a difference of a few percent between two runs says nothing,
    because two runs of the same data would differ by that much anyway.

    `on_epoch(i, train_loss, val_loss)` is what lets the Flyte report paint a live
    training curve instead of sitting blank for the whole run.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    model = make_policy(actions.shape[-1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.SmoothL1Loss()

    images, actions = images.to(device), actions.to(device)
    n = len(images)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    history = {"train": [], "val": []}

    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=gen)
        total = 0.0
        for i in range(0, n, batch):
            idx = order[i:i + batch]
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(images[idx]), actions[idx])
            loss.backward()
            opt.step()
            total += float(loss) * len(idx)
        sched.step()
        train_loss = total / n

        val_loss = float("nan")
        if val is not None:
            model.eval()
            with torch.inference_mode():
                val_loss = float(evaluate(model, *val, device=device)["mae"])
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        if on_epoch is not None:
            on_epoch(epoch, train_loss, val_loss)
    return model, history


def evaluate(model, images, actions, *, device: str = "cuda", batch: int = 64) -> dict:
    """Mean absolute error on held-out data, overall and on the channels that move.

    Same `moving_dims` logic as `world.action_error`, and for the same reason: a DROID
    action is a mixed vector, most of whose channels barely leave their start value in
    any given window. An average over all ten is dominated by channels where there was
    nothing to get right, and it makes a useless policy look respectable.
    """
    import torch

    model.eval()
    preds = []
    with torch.inference_mode():
        for i in range(0, len(images), batch):
            preds.append(model(images[i:i + batch].to(device)).cpu())
    pred = torch.cat(preds)
    truth = actions.cpu()

    err = (pred - truth).abs()
    span = truth.max(0).values - truth.min(0).values
    moving = (span > 0.1 * float(span.max())).nonzero().flatten().tolist() or [0]
    return {
        "mae": float(err.mean()),
        "mae_moving": float(err[:, moving].mean()),
        "moving_dims": moving,
        "per_dim": [round(float(v), 5) for v in err.mean(0)],
        "pred": pred,
        "truth": truth,
    }
