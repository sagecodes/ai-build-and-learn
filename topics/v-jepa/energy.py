"""The E in energy-based model, measured on candidate completions.

JEPA is usually introduced as an energy-based model. The energy of a pair (x, y) is

    E(x, y) = D( Predictor(Encoder(x)), Encoder(y) )

with D the L1 distance V-JEPA 2 was actually trained on, x the visible context and y a
candidate for the hidden part. Training pushes the energy down on real pairs. Nothing
pushes it up anywhere: JEPA has no negatives and no partition function, and the only
thing stopping the collapse of all energies to zero is the architecture (see
collapse.py). So "is this a well-shaped energy function?" is an empirical question,
and it is the one this module asks.

Three things get measured, each with the control that makes it readable.

── 1. The energy well ─────────────────────────────────────────────────────────
Roll the candidate clip forward and backward in time and plot the energy. If the
predictor located the hidden content in time, this is a V with its minimum at zero
offset. The same sweep in space says whether it located it spatially. These are the
continuous versions of `inpaint`'s median-dt, and they cannot be fudged by a metric
that always looks encouraging: a flat line is a flat line.

── 2. The candidate ladder ────────────────────────────────────────────────────
Score a graded family: the true completion, the same clip at the wrong moment, the
same clip played backwards, its frames shuffled, a different clip of the same action,
a different action, a blur of the truth, a frozen first frame, flat grey, and noise.
An energy function worth the name puts these in roughly that order. Read across the
whole ladder rather than at the top of it, because the failure mode that matters is
not "the truth is not lowest", it is "something degenerate is lower than the truth".

── 3. The degenerate-candidate control, which is the point ────────────────────
Flat grey and a frozen frame are the cheapest possible answers to "what is behind the
occluder". If either has lower energy than the truth, then the energy is not a
preference over completions and could not be used to generate one, no matter how good
its retrieval numbers look. That is exactly the criticism levelled at energy-based
models with no negative term, so it deserves a row in the table rather than a footnote.

── 4. And the answer depends on which distance you use ────────────────────────
The energy V-JEPA 2 was trained on is an L1 in raw feature space, and measured on this
box that energy is NOT a usable preference over completions: a flat grey video scores
less than half the energy of the true completion, the no-model "average of what you can
see" baseline also beats the truth, and the temporal well is flat to within 1%.

Subtract the component every token shares first -- the same centring correction
`jepa.center` applies everywhere else in this demo, for the same measured reason -- and
all of it snaps into place: the truth becomes the unique minimum, flat grey becomes the
WORST candidate of the eleven, and the temporal well becomes a clean V bottoming exactly
at zero offset. So both metrics are reported side by side everywhere in this task.

That failure is not a curiosity, it is the same one `collapse.py` is about. A low-norm
degenerate embedding has low raw L1 to anything, which is exactly the solution the JEPA
objective admits and exactly what the EMA target exists to keep training away from.
Nothing keeps a CANDIDATE at inference time away from it.

Everything here is one encoder pass per candidate and no training at all.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn.functional as F

import jepa

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)


def to_pixel_values(processor, frames: np.ndarray) -> torch.Tensor:
    """uint8 [T, H, W, 3] -> the normalised [1, T, 3, H, W] tensor, no resize.

    The inverse of `clips.shown_pixels`, which matters: every candidate here is a
    transform of the frames the model already consumed, so re-running the video
    processor would re-crop and re-resize and the candidate would differ from the true
    completion by a geometry change nobody asked for.

    The round trip is exact to within one grey level rather than bit-exact, because
    `shown_pixels` quantises to uint8 on the way out (measured: max absolute difference
    of 1 on the frames, 0.018 on the normalised tensor). That is acceptable for the
    reason it is worth stating: EVERY candidate goes through the same quantisation, the
    true completion included, so no candidate is advantaged. `smoke_test.py` scores the
    re-quantised truth against the encoder's own target tokens so the size of that
    difference is on record instead of assumed.
    """
    mean = torch.tensor(processor.image_mean).view(1, 3, 1, 1)
    std = torch.tensor(processor.image_std).view(1, 3, 1, 1)
    x = torch.as_tensor(frames, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    return ((x - mean) / std).unsqueeze(0)


# ── Candidate completions, all built from the frames the model saw ─────────────


def blur(frames: np.ndarray, radius: int = 5) -> np.ndarray:
    """Box blur, as a 'clean but degraded' candidate. No new dependency."""
    x = frames.astype(np.float32)
    k = np.ones(radius) / radius
    for ax in (1, 2):
        x = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), ax, x)
    return x.clip(0, 255).astype(np.uint8)


def freeze(frames: np.ndarray) -> np.ndarray:
    """The first frame repeated: 'nothing happens'. The most important distractor."""
    return np.repeat(frames[:1], len(frames), axis=0)


def shuffle_frames(frames: np.ndarray, seed: int = 0) -> np.ndarray:
    """Same frames, wrong order. A pooled pixel statistic cannot tell this apart."""
    rng = np.random.default_rng(seed)
    return frames[rng.permutation(len(frames))]


def grey(frames: np.ndarray) -> np.ndarray:
    return np.full_like(frames, 128)


def noise(frames: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=frames.shape, dtype=np.uint8)


def roll_time(frames: np.ndarray, tubelets: int, tubelet: int = 2) -> np.ndarray:
    """Roll by whole tubelets, so the token raster shifts by whole token rows."""
    return np.roll(frames, tubelets * tubelet, axis=0)


def roll_space(frames: np.ndarray, patches: int, patch: int = 16) -> np.ndarray:
    return np.roll(frames, patches * patch, axis=2)


# ── The energy itself ───────────────────────────────────────────────────────────


def energy(pred: torch.Tensor, cand: torch.Tensor) -> dict:
    """E(pred, candidate). L1 is the training objective; the rest is diagnostics.

    `l1` is the quantity V-JEPA 2 minimised, in the units it minimised it in, which is
    why it leads. `cosdist` is on centred features and is scale-free, so it is the one
    to trust when comparing across candidates whose token norms differ wildly (flat
    grey has a much smaller spread than a real clip, and an L1 on raw features is not
    blind to that). Both are reported precisely because they can disagree.
    """
    return {
        "l1": float((pred - cand).abs().mean()),
        "cosdist": float(
            1.0 - F.cosine_similarity(jepa.center(pred), jepa.center(cand), dim=-1).mean()
        ),
    }


@torch.inference_mode()
def candidate_energy(model, processor, pred: torch.Tensor, frames: np.ndarray,
                     target_ids: torch.Tensor) -> dict:
    """Encode one candidate clip and score the prediction against it."""
    pv = to_pixel_values(processor, frames)
    seq = jepa.encode(model, pv)
    cand = seq[target_ids.to(seq.device)]
    return energy(pred, cand)


def build_candidates(truth_frames: np.ndarray, others: dict[str, np.ndarray],
                     seed: int = 0) -> dict[str, np.ndarray]:
    """The graded family, as frame stacks. Separate from scoring so it can be checked.

    `others` carries the cross-clip candidates (same action, different action) as
    already-cropped frame stacks, because fetching them needs the catalogue and this
    module deliberately does not know about the dataset.
    """
    return {
        "true completion": truth_frames,
        "same clip, 1 tubelet late": roll_time(truth_frames, 1),
        "same clip, 4 tubelets late": roll_time(truth_frames, 4),
        "same clip, played backwards": truth_frames[::-1].copy(),
        "same clip, frames shuffled": shuffle_frames(truth_frames, seed=seed),
        "same clip, blurred": blur(truth_frames),
        "first frame frozen": freeze(truth_frames),
        **others,
        "flat grey": grey(truth_frames),
        "uniform noise": noise(truth_frames, seed=seed),
    }


def candidate_ladder(model, processor, pred, truth_frames: np.ndarray, target_ids,
                     others: dict[str, np.ndarray], seed: int = 0,
                     on_progress=None) -> dict[str, dict]:
    """Every candidate in the graded family, scored the same way."""
    cands = build_candidates(truth_frames, others, seed=seed)
    out = {}
    for i, (name, frames) in enumerate(cands.items()):
        out[name] = candidate_energy(model, processor, pred, frames, target_ids)
        if on_progress is not None:
            on_progress(i + 1, len(cands), name)
    return out


def well(model, processor, pred, frames: np.ndarray, target_ids,
         offsets: list[int], axis: str = "time") -> list[tuple[int, dict]]:
    """Energy against the candidate rolled by each offset, in BOTH metrics.

    Both, because they do not agree and the disagreement is the finding. Measured on
    this box, the raw L1 well is flat to within 0.8% and its minimum lands one tubelet
    off; the centred cosine well is a clean V bottoming exactly at zero. Returning only
    the training objective's own number would have reported that the predictor cannot
    locate anything in time, which is the opposite of what `inpaint` measures.
    """
    out = []
    for k in offsets:
        rolled = roll_time(frames, k) if axis == "time" else roll_space(frames, k)
        out.append((k, candidate_energy(model, processor, pred, rolled, target_ids)))
    return out


def argmin_of(curve: list[tuple[int, dict]], key: str = "l1") -> int:
    return min(curve, key=lambda p: p[1][key])[0]


def depth_of(curve: list[tuple[int, dict]], key: str = "l1") -> float:
    """How much of the curve's range the well actually is, as a fraction of its floor.

    A number for "is this well flat?", so the report does not have to leave that to
    the eye. 0.008 is flat; 0.09 is a well.
    """
    vals = [v[key] for _, v in curve]
    lo = min(vals)
    return (max(vals) - lo) / max(lo, 1e-9)


def interpolate(model, processor, pred, a: np.ndarray, b: np.ndarray, target_ids,
                steps: int = 7) -> list[tuple[float, dict]]:
    """Energy along a straight line in PIXEL space from the truth to a distractor.

    The shape of this curve is the difference between an energy function and a score.
    A well-behaved energy has a basin: leaving the truth in any direction costs
    something, monotonically at first. A curve that dips in the middle means there are
    blends of two clips the model likes better than either, which is the classic
    spurious-minimum problem and the reason nobody generates video by descending a
    JEPA energy.
    """
    out = []
    for i in range(steps):
        alpha = i / (steps - 1)
        mix = ((1 - alpha) * a.astype(np.float32) + alpha * b.astype(np.float32))
        mix = mix.clip(0, 255).astype(np.uint8)
        out.append((alpha, candidate_energy(model, processor, pred, mix, target_ids)))
    return out


def latent_interpolate(pred, z_true: torch.Tensor, z_other: torch.Tensor,
                       steps: int = 7) -> list[tuple[float, dict]]:
    """The same line, drawn in representation space instead. No encoder pass needed."""
    out = []
    for i in range(steps):
        alpha = i / (steps - 1)
        out.append((alpha, energy(pred, (1 - alpha) * z_true + alpha * z_other)))
    return out


def floors(pred: torch.Tensor, true: torch.Tensor, seq: torch.Tensor,
           context_ids: torch.Tensor, seed: int = 0) -> dict[str, dict]:
    """The two no-model answers, scored on the true completion like everything else.

    `shuffled prediction` destroys only the pairing, so it says what the energy is
    worth when the correspondence between prediction and target is removed while every
    marginal stays the same. `context mean` answers with the average of what it can
    see, which is the cheapest thing a model could do instead of predicting. The
    predictor has to beat both to have predicted anything.
    """
    perm = torch.randperm(len(pred), device=pred.device,
                          generator=torch.Generator(pred.device).manual_seed(seed))
    mean = seq[context_ids.to(seq.device)].mean(0, keepdim=True).expand_as(true)
    return {
        "shuffled prediction (chance)": energy(pred[perm], true),
        "context mean (no model)": energy(mean, true),
    }


def ranking(scores: dict[str, dict], key: str = "cosdist",
            true_name: str = "true completion"):
    """(is the truth the lowest-energy candidate, its rank, how many candidates)."""
    order = sorted(scores, key=lambda k: scores[k][key])
    rank = order.index(true_name)
    return rank == 0, rank + 1, len(order)


def spearman(x: list[float], y: list[float]) -> float:
    """Rank correlation, written out so scipy is not needed for four numbers."""
    def ranks(v):
        idx = np.argsort(np.asarray(v, dtype=float))
        r = np.empty(len(v))
        r[idx] = np.arange(len(v))
        return r

    a, b = ranks(x), ranks(y)
    a, b = a - a.mean(), b - b.mean()
    denom = float(np.sqrt((a**2).sum() * (b**2).sum()))
    return float((a * b).sum() / denom) if denom > 0 else float("nan")


# The order an energy function SHOULD put the candidates in, written down before the
# measurement so the rank correlation below is a prediction and not a fit. Lower is
# nearer the truth; ties share a number.
EXPECTED_ORDER = {
    "true completion": 0,
    "same clip, 1 tubelet late": 1,
    "same clip, blurred": 2,
    "same clip, 4 tubelets late": 3,
    "first frame frozen": 4,
    "same clip, played backwards": 4,
    "same clip, frames shuffled": 5,
    "same action, different clip": 6,
    "different action": 7,
    "flat grey": 8,
    "uniform noise": 9,
}


def agreement(scores: dict[str, dict], key: str = "cosdist") -> tuple[float, int]:
    """Rank correlation between the measured energies and EXPECTED_ORDER."""
    names = [n for n in scores if n in EXPECTED_ORDER]
    return (
        spearman([scores[n][key] for n in names], [EXPECTED_ORDER[n] for n in names]),
        len(names),
    )
