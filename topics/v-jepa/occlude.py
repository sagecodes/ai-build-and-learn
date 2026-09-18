"""Show the prediction. Honestly, with its own ceiling painted next to it.

Every other task in this demo works around V-JEPA 2 having no decoder by refusing to
render the prediction at all. That is the safe choice and it is also the reason the
reports are harder to read than they should be: a hole and a heatmap do not tell you
what the model thought was behind the hole.

There are exactly two honest ways to turn a predicted token into pixels, and both are
only meaningful when the SAME rendering is applied to the TRUE token beside it:

  mosaic()    For each predicted token, find the nearest token in a bank built from
              other clips and paste the 16x16 pixels that token was embedded from.
              Every pixel shown is a real pixel of a real video; nothing is invented.
              What it shows is "what the model considers equivalent to the hidden
              patch", which is the closest thing to a prediction that exists here.

  readout     The closed-form linear map from decode.py, applied to the predicted
              token instead of the true one. It will be mush, because decode.py
              already measured that appearance is not in the token -- which is
              precisely why the truth panel matters. If the truth panel is mush too,
              then mush is the ceiling and the comparison is still valid.

The rule that makes both legitimate: the bank and the ridge map are fitted on TRAIN
clips only, the query clip is a VAL clip, and the ceiling panel is the identical
rendering of the encoder's own tokens at the identical positions. The prediction can
therefore never look better than the rendering method allows, and the reader can see
how much of the gap is the method and how much is the predictor.

── The three masks, matched on hidden fraction ─────────────────────────────────
`inpaint` compares a static tube against a future mask and finds that this checkpoint
inpaints rather than forecasts. This task adds the case in between, with the SAME
number of tokens hidden in all three (an 8x8 block of a 16x16 grid is 25%, and so is
4 of 16 tubelets), so that only the SHAPE of the hole varies:

  sweep    a block that MOVES as time advances, so each hidden patch position is
           visible at other timesteps. Interpolation with help.
  static   the same block, never moving, so those patch positions are hidden for the
           whole clip. Interpolation without help.
  future   everything after a moment. Extrapolation, which it was never trained on.

The hypothesis was an ordering, sweep > static > future, and HALF OF IT IS WRONG.
Measured: sweep and static are indistinguishable on every metric here (cosine lift
+0.261 vs +0.269, time localisation 1.00 vs 1.00, same-action retrieval 99% vs 96%).
Making the hidden patch position visible at other moments buys the predictor nothing,
so whatever it is doing is not "look up this patch at another time". Only the future
mask separates, and it separates completely: time localisation 0.00.

What the three masks DID show is a dissociation worth more than the ordering was.
Under every mask, including the one where the prediction cannot place the content in
time at all, the patch the mosaic retrieves comes from a clip of the same action about
97% of the time against a bank share of 20%. The prediction is right about WHAT and
wrong about WHEN, and those two failures are separable with the same measurement.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

import decode
import jepa

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

TUBELET = decode.TUBELET
PATCH = decode.PATCH


# ── Masks ───────────────────────────────────────────────────────────────────────


def sweep(tubelets: int, grid: int, size: int = 6) -> torch.Tensor:
    """A `size`x`size` occluder sliding left to right as time advances.

    Vertically centred, and it starts flush left and ends flush right so the whole
    width is covered exactly once. Note what this hides relative to `static` below:
    the same number of tokens, but no patch POSITION is hidden for more than a couple
    of tubelets, so the predictor can see that position at other moments.
    """
    m = torch.zeros(tubelets, grid, grid, dtype=torch.bool)
    h0 = (grid - size) // 2
    span = max(grid - size, 0)
    for t in range(tubelets):
        w0 = int(round(span * t / max(tubelets - 1, 1)))
        m[t, h0 : h0 + size, w0 : w0 + size] = True
    return m


def static(tubelets: int, grid: int, size: int = 6, h0: int | None = None,
           w0: int | None = None) -> torch.Tensor:
    """The same block, not moving. `jepa.tube` with one centred block of a given size."""
    m = torch.zeros(tubelets, grid, grid, dtype=torch.bool)
    h0 = (grid - size) // 2 if h0 is None else h0
    w0 = (grid - size) // 2 if w0 is None else w0
    m[:, h0 : h0 + size, w0 : w0 + size] = True
    return m


def matched_future(tubelets: int, grid: int, share: float) -> torch.Tensor:
    """A future mask hiding as close to `share` of the tokens as the grid allows.

    Rounded to whole tubelets, which is the only resolution a future mask has, so the
    report prints the realised fraction rather than the requested one.
    """
    hide = max(1, int(round(tubelets * share)))
    m = torch.zeros(tubelets, grid, grid, dtype=torch.bool)
    m[tubelets - hide :] = True
    return m


def positions(grid: int, size: int, count: int = 9) -> list[tuple[int, int]]:
    """Top-left corners for the hole-position sweep, as a square grid over the frame.

    A grid rather than the diagonal, which is what this did first. The covariate the
    sweep is against is how much MOTION was under the hole, and on a diagonal through a
    centre-cropped clip the covariate barely varied: 5.5 to 9.8 grey levels over six
    positions, r = -0.15 with n = 6. The grid spans 3.7 to 11.0 over nine positions,
    three times the range, and the answer does not change: r = -0.13. So the null is a
    real null rather than a coverage problem, which is the only reason worth having
    bothered to widen it.

    Returns `round(sqrt(count))**2` positions, so `count` is a target rather than a
    promise and the report prints how many it actually used.
    """
    per_side = max(2, int(round(count**0.5)))
    span = max(grid - size, 0)
    steps = [int(round(span * i / (per_side - 1))) for i in range(per_side)]
    return [(h, w) for h in steps for w in steps]


# ── The bank: tokens whose pixels we still have ────────────────────────────────


def build_bank(model, processor, repo: str, names: list[str], num_frames: int,
               labels: list[str] | None = None, on_progress=None):
    """Encode train clips into (tokens, their pixels, their class id).

    This is `decode.collect` plus a label per token, because "which clip did the
    retrieved patch come from" is the measurement that makes the mosaic more than a
    picture: if the nearest neighbour of a hidden bowling patch is usually a bowling
    patch, the token carries the class of the thing it is looking at.
    """
    X, Y, grid, tubes = decode.collect(model, processor, repo, names, num_frames,
                                       on_progress=on_progress)
    per_clip = tubes * grid * grid
    if labels is None:
        lab = torch.zeros(len(X), dtype=torch.long)
    else:
        # decode.collect skips clips that will not decode, so labels are assigned by
        # position in the surviving stack rather than by position in `names`.
        n_clips = len(X) // per_clip
        lab = torch.tensor(
            np.repeat(np.asarray(labels[:n_clips], dtype=np.int64), per_clip)
        )
    return {
        "X": X.float(),
        "Xd": X,  # float64, for the ridge solve
        "Y": Y,
        "label": lab,
        "grid": grid,
        "tubes": tubes,
        "mu": X.float().mean(0, keepdim=True),
    }


def mosaic(query: torch.Tensor, bank: dict, chunk: int = 4096):
    """Nearest bank token for every query token. Returns (patches, index, cosine).

    Centred on the BANK's mean rather than per-clip, because the comparison is across
    clips and a per-clip mean would put every clip in its own frame of reference. The
    same-class fraction reported next to it is what catches this being wrong: a broken
    geometry shows up as chance-level retrieval, not as a plausible picture.
    """
    mu = bank["mu"].to(query.device)
    B = F.normalize(bank["X"].to(query.device) - mu, dim=-1)
    q = F.normalize(query.float() - mu, dim=-1)
    idx, cos = [], []
    for i in range(0, len(q), chunk):
        sims = q[i : i + chunk] @ B.T
        best = sims.max(1)
        idx.append(best.indices)
        cos.append(best.values)
    idx = torch.cat(idx).cpu()
    return bank["Y"][idx].float(), idx, torch.cat(cos).cpu()


def same_class(idx: torch.Tensor, bank: dict, class_id: int) -> tuple[float, float]:
    """(fraction of retrievals from the query's class, that class's share of the bank).

    NaN when the bank holds only one class, which is not a hypothetical: the clip
    catalogue is sorted by path, so taking the first N train clips gives N clips of
    whichever action sorts first. The caller has to spread the bank across classes for
    this number to mean anything, and NaN is how it finds out that it did not.
    """
    lab = bank["label"]
    if len(torch.unique(lab)) < 2:
        return float("nan"), float("nan")
    hit = float((lab[idx] == class_id).float().mean())
    base = float((lab == class_id).float().mean())
    return hit, base


# ── Rendering into the hole ─────────────────────────────────────────────────────


def fill(shown: np.ndarray, target_ids: torch.Tensor, patches: torch.Tensor,
         grid: int) -> np.ndarray:
    """Paste rendered patches into the hole, leaving the visible context untouched.

    `patches` is [n, 1536] or [n, 2, 16, 16, 3] in [0, 1], in the same order as
    `target_ids`. The result is the model's literal input everywhere it could see, and
    a rendering everywhere it could not, which is the only composite that does not
    quietly overwrite real pixels with a reconstruction.
    """
    p = patches.reshape(-1, TUBELET, PATCH, PATCH, 3).clamp(0, 1)
    p = (p.detach().cpu().numpy() * 255).astype(np.uint8)
    out = shown.copy()
    ppf = grid * grid
    for k, tok in enumerate(target_ids.tolist()):
        t, rest = tok // ppf, tok % ppf
        h, w = rest // grid, rest % grid
        for j in range(TUBELET):
            f = t * TUBELET + j
            if f < len(out):
                out[f, h * PATCH : (h + 1) * PATCH, w * PATCH : (w + 1) * PATCH] = p[k, j]
    return out


def hole_pixels(mask3d, grid: int, shape: tuple[int, int, int]) -> np.ndarray:
    """A [T, H, W] bool mask of exactly the pixels the tokens covered."""
    m = np.asarray(mask3d)
    T, H, W = shape
    out = np.zeros((T, H, W), dtype=bool)
    for t in range(m.shape[0]):
        hs, ws = np.nonzero(m[t])
        for h, w in zip(hs, ws):
            for j in range(TUBELET):
                f = t * TUBELET + j
                if f < T:
                    out[f, h * PATCH : (h + 1) * PATCH, w * PATCH : (w + 1) * PATCH] = True
    return out


def hole_psnr(render: np.ndarray, truth: np.ndarray, hole: np.ndarray) -> float:
    """PSNR over the occluded pixels only.

    Over the whole frame it would be dominated by the context, which both renderings
    copy verbatim from the input, so every method would score 30-odd dB and the
    numbers would say nothing about the prediction.
    """
    a = render[: len(truth)][hole[: len(truth)]].astype(np.float32)
    b = truth[hole[: len(truth)]].astype(np.float32)
    mse = float(np.mean((a - b) ** 2))
    return 99.0 if mse <= 1e-9 else float(10.0 * np.log10(255.0**2 / mse))


def hole_motion(truth: np.ndarray, hole: np.ndarray) -> float:
    """Mean absolute frame-to-frame change inside the hole, in grey levels.

    The covariate for the hole-position sweep. A hole over a blank wall and a hole over
    a swinging arm are not the same task, and an aggregate score over patches that are
    mostly wall will look good for the wrong reason.
    """
    grey = truth.mean(-1)
    d = np.abs(np.diff(grey, axis=0))
    h = hole[1:] if len(hole) > 1 else hole
    sel = d[h[: len(d)]]
    return float(sel.mean()) if sel.size else 0.0


# ── One mask, measured and rendered every way ──────────────────────────────────


def evaluate(model, pixel_values, shown, mask3d, bank, W_ridge, grid: int,
             class_id: int = 0, seed: int = 0) -> dict:
    """Run the predictor under one mask and produce every number and frame set.

    Returns the scores (jepa.score plus its shuffled floor), the mosaic and readout
    renderings of BOTH the prediction and the truth, and the hole-only PSNR of each.
    The truth rendering is not decoration: it is the ceiling for its own method, and
    a prediction PSNR is only readable as a fraction of it.
    """
    tubelets = mask3d.shape[0]
    ctx_ids, tgt_ids = jepa.ids_of(mask3d)
    pred, true = jepa.predict(model, pixel_values, ctx_ids, tgt_ids)

    scored = jepa.score(pred, true, tgt_ids, grid)
    floor = jepa.shuffled_floor(pred, true, tgt_ids, grid, seed=seed)
    scored["loc"] = jepa.localization(scored, floor)
    floor["loc"] = jepa.localization(floor, floor)

    truth = shown[: tubelets * TUBELET]
    hole = hole_pixels(mask3d, grid, truth.shape[:3])

    # Mosaic, for the prediction and for the encoder's own tokens.
    p_patch, p_idx, p_cos = mosaic(pred, bank)
    t_patch, t_idx, _ = mosaic(true, bank)
    perm = torch.randperm(len(pred), generator=torch.Generator().manual_seed(seed + 7))
    s_patch = p_patch[perm]

    # No "input" entry: the caller adds the blacked-out input, because building it is
    # viz's job and this module does not import viz.
    frames = {
        "mosaic_pred": fill(truth, tgt_ids, p_patch, grid),
        "mosaic_true": fill(truth, tgt_ids, t_patch, grid),
        "mosaic_shuf": fill(truth, tgt_ids, s_patch, grid),
        "readout_pred": fill(truth, tgt_ids,
                             decode.ridge_apply(W_ridge, pred.double().cpu()), grid),
        "readout_true": fill(truth, tgt_ids,
                             decode.ridge_apply(W_ridge, true.double().cpu()), grid),
    }
    psnr = {k: hole_psnr(v, truth, hole) for k, v in frames.items()}

    hit, base = same_class(p_idx, bank, class_id)
    hit_t, _ = same_class(t_idx, bank, class_id)
    return {
        "hidden": float(mask3d.float().mean()),
        "n_target": int(len(tgt_ids)),
        "score": scored,
        "floor": floor,
        "psnr": psnr,
        "retrieval": {
            "cos": float(p_cos.mean()),
            # The metric that actually works on a mosaic. PSNR between the rendering
            # and the truth is near its floor whatever the predictor does, because the
            # token does not carry appearance (that is `readout`'s whole finding), so a
            # patch with the right content and the wrong colour scores like noise.
            # Asking instead whether the prediction retrieves the SAME bank patch the
            # encoder's own token retrieves compares the rendering with its own ceiling,
            # which is the only thing it can be compared with. Chance is one over the
            # bank size.
            "agree": float((p_idx == t_idx).float().mean()),
            "agree_chance": 1.0 / len(bank["X"]),
            "same_class_pred": hit,
            "same_class_true": hit_t,
            "same_class_chance": base,
        },
        "motion": hole_motion(truth, hole),
        "frames": frames,
        "truth": truth,
        "mask": mask3d,
        "target_ids": tgt_ids,
    }



def spread_over_classes(catalog, split: str, labels: list[str], total: int):
    """Pick `total` clips of `split`, round-robin over classes. Returns (paths, ids).

    `clips.list_clips` sorts by path, so `[p for sp, _, p in catalog if sp == split][:N]`
    is N clips of whichever action sorts first. Every cross-class measurement in this
    module would then be NaN or meaningless, so the interleave is not a nicety.
    """
    by_class: dict[str, list[str]] = {lb: [] for lb in labels}
    for sp, lb, path in catalog:
        if sp == split:
            by_class[lb].append(path)
    picked: list[tuple[str, str]] = []
    depth = 0
    while len(picked) < total and any(len(v) > depth for v in by_class.values()):
        for lb in labels:
            if len(by_class[lb]) > depth and len(picked) < total:
                picked.append((lb, by_class[lb][depth]))
        depth += 1
    return [p for _, p in picked], [labels.index(lb) for lb, _ in picked]
