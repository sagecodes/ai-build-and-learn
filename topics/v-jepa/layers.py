"""Where along the 24 layers does appearance turn into meaning?

`readout` establishes the headline fact of this demo: a linear map recovers almost
nothing of a 16x16 patch from the final-layer token, while the same tokens support 78%
five-way action recognition. Appearance discarded, meaning kept. What it cannot say is
WHERE that happened, because it only ever looks at the last layer.

This module runs both measurements at every layer of the same forward pass:

  pixels    the closed-form ridge readout from decode.py, token -> its 2x16x16 patch,
            scored on held-out tokens. How much of the picture is still linearly
            present at depth l.
  meaning   a linear probe for the action class on mean-pooled layer-l features, plus
            leave-one-out nearest-neighbour retrieval, which trains nothing at all.
  geometry  the random-pair cosine, raw and centred, per layer. The anisotropy that
            `jepa.center` corrects is not constant with depth -- it is measured at 0.92
            around layer 8 and 0.28 at the last layer -- so a reader who takes the
            final-layer number as universal will misread the middle of the network.

Two things make the curves honest rather than decorative.

The readout ceiling and floor are computed once per layer from the same token subset:
a random projection of the true pixels (which inverts almost perfectly, so it says
what "linearly decodable" means at all) and V-JEPA tokens paired with the WRONG
patches. A decaying curve between two flat references is a result; a decaying curve on
its own could be a solver running out of conditioning.

Intermediate layers are taken BEFORE the encoder's final LayerNorm, because that norm's
learned affine is calibrated for the last layer and applying it to layer 8 would be
measuring a transform nobody trained. The final layer is therefore reported twice, raw
and normed, so the size of that choice is visible instead of assumed.

── Why this is the load-bearing curve for the whole JEPA argument ─────────────
The predictor is trained to match the FINAL encoder layer. If the semantic peak turns
out to be earlier than that, then the representation the world model predicts is not
the representation that is best at describing the scene, which is a concrete design
question rather than a philosophical one.
"""

from __future__ import annotations

import logging

import numpy as np
import torch

import clips as clip_io
import decode
import jepa
import probing

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


@torch.inference_mode()
def per_layer(model, pixel_values) -> list[torch.Tensor]:
    """Every layer's tokens for one clip: [L+1][N, D], last entry post-LayerNorm.

    Written as an explicit loop over `model.encoder.layer` rather than asking for
    `output_hidden_states`. In transformers 5 that flag is served by output-capture
    hooks on the decorated forward, and `VJEPA2Encoder.forward` itself returns only
    `last_hidden_state`, so a release that changes the capture machinery would hand
    back None and this module would silently have nothing to plot. The loop is ten
    lines and cannot drift.
    """
    enc = model.encoder
    device = next(model.parameters()).device
    h = enc.embeddings(pixel_values.to(device))
    out = []
    for layer in enc.layer:
        h = layer(h, None)[0]
        out.append(h[0].float())
    out.append(enc.layernorm(h)[0].float())
    return out


def names(n_layers: int) -> list[str]:
    return [f"{i + 1}" for i in range(n_layers)] + ["final (LN)"]


def collect_tokens(model, processor, repo: str, clip_names: list[str], num_frames: int,
                   keep: int = 1024, seed: int = 0, on_progress=None):
    """Tokens at every layer, paired with the pixels they were embedded from.

    Subsamples `keep` tokens per clip, and crucially samples the SAME token indices at
    every layer, so the pixel targets line up across the whole stack. Holding all
    4096 tokens of every clip at 25 layers is several GB for no extra signal.
    """
    tubes, grid = jepa.grid_of(model, num_frames)
    per_layer_tokens: list[list[torch.Tensor]] = []
    pix: list[torch.Tensor] = []
    g = torch.Generator().manual_seed(seed)

    for i, name in enumerate(clip_names):
        try:
            pv, shown = clip_io.load_clip(processor, repo, name, num_frames)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", name, exc)
            continue
        stack = per_layer(model, pv)
        jepa.check_layout(stack[-1], tubes, grid)
        n = stack[-1].shape[0]
        idx = torch.randperm(n, generator=g)[: min(keep, n)]
        if not per_layer_tokens:
            per_layer_tokens = [[] for _ in stack]
        for li, tok in enumerate(stack):
            per_layer_tokens[li].append(tok[idx.to(tok.device)].cpu())
        patches = decode.patches_from_video(shown, grid).reshape(-1, decode.PATCH_DIM)
        pix.append(patches[idx])
        if on_progress is not None:
            on_progress(i + 1, len(clip_names))

    X = [torch.cat(t).double() for t in per_layer_tokens]
    Y = torch.cat(pix).double()
    return X, Y, grid, tubes


def readout_curve(X: list[torch.Tensor], Y: torch.Tensor, val_frac: float = 0.2,
                  seed: int = 0, device: str | None = None) -> tuple[list[dict], dict]:
    """Ridge readout at every layer, plus the ceiling and floor that frame it.

    The split is random over tokens rather than a tail slice, because `collect_tokens`
    concatenates clip by clip and a tail slice would hold out whole clips at the end of
    the list. Held-out CLIPS would be the harder test, but it is a different test, and
    mixing the two silently is how a curve stops meaning one thing.

    `device` moves ONE layer's tokens at a time, not the whole stack. 25 layers of
    float64 tokens is a couple of GB on a box where CUDA can only use what is genuinely
    free, and the solve is the same arithmetic at every depth so there is nothing to
    gain from holding them all there. The references go through the identical code path,
    which is what would catch this precision or a bad solve rather than leaving a
    decaying curve to be read as a finding.
    """
    n = len(Y)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_val = max(int(round(n * val_frac)), 1)
    val, train = perm[:n_val], perm[n_val:]
    Yd = Y if device is None else Y.to(device)

    def fit_score(Xl: torch.Tensor, target: torch.Tensor) -> dict:
        Xd = Xl if device is None else Xl.to(device)
        W = decode.ridge_fit(Xd[train], target[train])
        out = decode.score(decode.ridge_apply(W, Xd[val]), target[val])
        del Xd, W
        return out

    rows = []
    for li, Xl in enumerate(X):
        rows.append(fit_score(Xl, Yd))
        log.info("layer %2d readout: %.2f dB r2 %+.3f", li + 1, rows[-1]["psnr"],
                 rows[-1]["r2"])

    dim = X[-1].shape[1]
    Xr, _ = decode.random_projection(Y, dim, seed=seed)
    shuf = torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1))
    refs = {
        "rand": fit_score(Xr, Yd),
        "grey": decode.score(Yd[val].mean(1, keepdim=True).repeat(1, decode.PATCH_DIM),
                             Yd[val]),
        "shuf": fit_score(X[-1], Yd[shuf]),
    }
    return rows, refs


@torch.inference_mode()
def pooled_dataset(model, processor, repo: str, catalog, num_frames: int,
                   on_progress=None):
    """Mean-pooled features at EVERY layer for every clip, plus labels and the split.

    One forward pass per clip for all 25 layers, which is the whole reason this is a
    separate function from `probing.encode_dataset`: doing it per layer would be 25
    passes over the dataset for exactly the same arithmetic.
    """
    labels = clip_io.labels_of(catalog)
    feats: list[list[torch.Tensor]] = []
    ys, splits, failed = [], [], []
    for i, (split, label, path) in enumerate(catalog):
        try:
            pv, _ = clip_io.load_clip(processor, repo, path, num_frames)
            stack = per_layer(model, pv)
        except Exception as exc:  # noqa: BLE001
            log.warning("skipping %s: %s", path, exc)
            failed.append(path)
            continue
        if not feats:
            feats = [[] for _ in stack]
        for li, tok in enumerate(stack):
            feats[li].append(probing.pool(tok).cpu())
        ys.append(labels.index(label))
        splits.append(split)
        if on_progress is not None and (i % 10 == 0 or i == len(catalog) - 1):
            on_progress(i + 1, len(catalog))

    return {
        "X": [torch.stack(f) for f in feats],
        "y": torch.tensor(ys),
        "train": torch.tensor([s == "train" for s in splits]),
        "labels": labels,
        "failed": failed,
    }


def probe_curve(data: dict, seed: int = 0) -> list[dict]:
    """Linear probe and 1-NN retrieval at every layer, on identical splits."""
    rows = []
    for li, Xl in enumerate(data["X"]):
        acc, _ = probing.linear_probe(Xl, data["y"], data["train"], seed=seed)
        ret, _ = probing.retrieval(Xl, data["y"])
        raw, _ = probing.retrieval(Xl, data["y"], centered=False)
        rows.append({"probe": acc, "retrieval": ret, "retrieval_raw": raw})
        log.info("layer %2d probe %.3f retrieval %.3f (raw %.3f)", li + 1, acc, ret, raw)
    return rows


def geometry_curve(model, processor, repo: str, name: str, num_frames: int) -> list[dict]:
    """Random-pair cosine per layer, raw and centred, on one clip.

    One clip is enough: this is a property of the layer's output geometry, not of the
    dataset, and it is stable enough across clips that averaging would only hide that.
    """
    pv, _ = clip_io.load_clip(processor, repo, name, num_frames)
    return [jepa.anisotropy(tok) for tok in per_layer(model, pv)]


def peaks(readout: list[dict], probe: list[dict]) -> dict:
    """Where each curve is highest, and how far apart those two layers are."""
    best_pix = int(np.argmax([r["r2"] for r in readout]))
    best_sem = int(np.argmax([r["probe"] for r in probe]))
    best_ret = int(np.argmax([r["retrieval"] for r in probe]))
    return {
        "pixel_layer": best_pix + 1,
        "pixel_r2": readout[best_pix]["r2"],
        "probe_layer": best_sem + 1,
        "probe_acc": probe[best_sem]["probe"],
        "probe_final": probe[-1]["probe"],
        "retrieval_layer": best_ret + 1,
        "retrieval_acc": probe[best_ret]["retrieval"],
        # The predictor's target is the final encoder output, so "is the best layer the
        # one the world model predicts?" is the question this number answers.
        "gap_to_final": probe[best_sem]["probe"] - probe[-1]["probe"],
    }
