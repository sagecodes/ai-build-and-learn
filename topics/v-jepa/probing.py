"""Are the frozen features any good? A linear probe and nearest-neighbour retrieval.

This is the standard way to ask what self-supervision bought, and it is deliberately
the least clever code in the demo. The encoder is frozen, nothing is fine-tuned, and
the only trained parameters anywhere are one 1024x5 matrix and its bias. If that
matrix can separate the classes, the information was already in the representation.

Two floors keep the headline number honest:
  * chance, which for 5 balanced classes is 20%
  * the same probe trained on `clips.pixel_baseline()` features, a 576-dimensional
    space-time thumbnail. This is the one that matters: it says how much of the score
    is the representation and how much is that these five classes look different at a
    glance.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def pool(seq: torch.Tensor) -> torch.Tensor:
    """Patch tokens -> one clip vector, by mean pooling.

    Mean pooling, not the attentive pooler: `VJEPA2AttentivePooler` exists in the
    architecture but its weights are only trained in the CLASSIFIER checkpoints
    (…-ssv2, …-diving48). In a pretrained-only checkpoint it is randomly initialised,
    so using it here would be measuring noise.
    """
    return seq.mean(0)


def linear_probe(
    X: torch.Tensor, y: torch.Tensor, train: torch.Tensor,
    steps: int = 600, lr: float = 1e-2, weight_decay: float = 1e-3, seed: int = 0,
    test: torch.Tensor | None = None,
) -> tuple[float, torch.Tensor]:
    """Train a linear classifier on the training split, score the rest.

    Features are standardised using TRAIN statistics only. Using all of them would leak
    the val split into the preprocessing, which is a small effect at this size but
    exactly the kind of thing that makes a benchmark number quietly wrong.

    `test` overrides "everything not in `train`", which a low-shot sweep needs: fitting
    on 25 of 1000 available labels must not silently move the other 975 into the test
    set, or accuracies at different fit sizes are measured on different problems.
    """
    torch.manual_seed(seed)
    test = ~train if test is None else test
    mu, sd = X[train].mean(0), X[train].std(0) + 1e-6
    Z = ((X - mu) / sd).float()

    head = torch.nn.Linear(Z.shape[1], int(y.max()) + 1)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(head(Z[train]), y[train]).backward()
        opt.step()

    with torch.no_grad():
        pred = head(Z[test]).argmax(1)
    return float((pred == y[test]).float().mean()), pred


def confusion(y_true: torch.Tensor, y_pred: torch.Tensor, n: int) -> np.ndarray:
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[t, p] += 1
    return cm


def retrieval(X: torch.Tensor, y: torch.Tensor, centered: bool = True, k: int = 3):
    """Leave-one-out nearest-neighbour accuracy, plus the neighbours themselves.

    No training at all: if clips of the same action are each other's nearest
    neighbours in a frozen embedding space, that is the representation talking.

    `centered=False` exists to put a number on the anisotropy correction rather than
    asserting it. Measured on this box: 82% centered, 76% raw, on the same features.
    """
    Z = X - X.mean(0, keepdim=True) if centered else X
    Z = F.normalize(Z.float(), dim=-1)
    sims = Z @ Z.T
    sims.fill_diagonal_(-2.0)
    top = sims.topk(min(k, len(Z) - 1), dim=1).indices
    acc = float((y[top[:, 0]] == y).float().mean())
    return acc, top


def encode_dataset(model, processor, repo, clips, num_frames, on_progress=None):
    """Encode every clip once: V-JEPA 2 features, pixel-baseline features, labels.

    Clips that fail to decode are skipped rather than fatal, and counted, because one
    bad mp4 in a 100-clip dataset should not lose the run. The count goes in the report
    so a silently shrinking dataset cannot masquerade as a clean one.
    """
    import clips as clip_io

    labels = clip_io.labels_of(clips)
    feats, pixels, ys, splits, kept, failed = [], [], [], [], [], []
    for i, (split, label, path) in enumerate(clips):
        try:
            frames = clip_io.decode(clip_io.fetch(repo, path), num_frames)
            pv = clip_io.preprocess(processor, frames)
            with torch.inference_mode():
                seq = model.get_vision_features(pv.to(next(model.parameters()).device))[0].float()
            feats.append(pool(seq).cpu())
            pixels.append(torch.tensor(clip_io.pixel_baseline(frames)))
            ys.append(labels.index(label))
            splits.append(split)
            kept.append((split, label, path))
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", path, exc)
            failed.append(path)
        if on_progress and (i % 10 == 0 or i == len(clips) - 1):
            on_progress(i + 1, len(clips))

    return {
        "X": torch.stack(feats),
        "P": torch.stack(pixels),
        "y": torch.tensor(ys),
        "train": torch.tensor([s == "train" for s in splits]),
        "clips": kept,
        "labels": labels,
        "failed": failed,
    }
