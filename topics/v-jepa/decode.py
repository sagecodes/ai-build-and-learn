"""What is actually inside a V-JEPA 2 token? Ask by trying to read things out of it.

The README of this demo opens by insisting that "show me what it predicted" is not a
screenshot anyone can take, because V-JEPA 2 has no decoder. That is usually stated as
a packaging problem, as though Meta simply never shipped the head. This module tests
whether it is something stronger: that the information needed to draw the picture is
not in the representation at all.

The test is a readout. Fit the best possible LINEAR map from one patch token to the
2x16x16 patch of pixels it was embedded from, in closed form, and see how well it does:

    W = (X'X + lambda I)^-1 X'Y

Closed form matters. A trained decoder that fails is ambiguous -- too small, too few
steps, wrong learning rate -- and the first version of this module fell into exactly
that trap: a 2048-wide MLP reached 13.9 dB on V-JEPA tokens, which looked like a
finding until the same MLP also managed only 13.9 dB on a control it should have
inverted almost perfectly. Ridge regression has no hyperparameter that can be blamed.

── The controls, which are the entire experiment ───────────────────────────────
  `rand`   a fixed random 1024-dim projection of the true patch pixels. 1536 numbers
           squeezed into 1024 by a random matrix is very nearly lossless, so this is
           the CEILING: whatever it scores is what "linearly decodable" means here.
  `grey`   paint every patch its own mean colour. The amount of credit you get for
           knowing nothing but the average brightness of a 16x16 square.
  `shuf`   V-JEPA tokens paired with the WRONG patches. The floor.

Measured on kinetics-mini, 48 training clips, held-out tokens:

    rand-proj   38.76 dB   R2 +0.997      <- the pipeline works
    flat grey   17.73 dB
    V-JEPA      11.97 dB   R2 -0.494      <- barely above the floor
    shuffled    11.10 dB   R2 -0.008

V-JEPA's tokens are worse than knowing the patch's average colour, and a hair above
being paired with the wrong patch entirely. The appearance is not in there.

── Why that is the right answer rather than a disappointment ───────────────────
It is the JEPA thesis, made measurable. The whole argument for predicting in
representation space is that pixels are mostly unpredictable detail, and a model that
does not spend capacity on them is free to spend it on structure. The same tokens that
cannot reproduce a 16x16 square support 78% five-way action recognition from a single
linear layer, against 40% for raw pixels and 20% chance (see the `probe` task).

So: V-JEPA discards appearance and keeps meaning, and `readout` shows both halves of
that sentence with the same tool, a linear map, on the same tokens.
"""

from __future__ import annotations

import numpy as np
import torch

import clips as clip_io
import jepa

PATCH = 16
TUBELET = 2
PATCH_DIM = TUBELET * PATCH * PATCH * 3  # 1536


def patches_from_video(frames: np.ndarray, grid: int) -> torch.Tensor:
    """[T, H, W, 3] uint8 -> [T/2 * grid * grid, 2, 16, 16, 3] float in [0, 1].

    The ordering must match the token raster exactly: index = t*G*G + h*G + w,
    temporal-major, with t indexing tubelets of 2 frames. Getting this wrong produces
    a scrambled image rather than an error, so `smoke_test.py` asserts the round trip
    through `video_from_patches` is bit-exact.
    """
    T = frames.shape[0]
    tubes = T // TUBELET
    x = torch.as_tensor(frames[: tubes * TUBELET], dtype=torch.float32) / 255.0
    x = x.view(tubes, TUBELET, grid, PATCH, grid, PATCH, 3)
    x = x.permute(0, 2, 4, 1, 3, 5, 6)
    return x.reshape(tubes * grid * grid, TUBELET, PATCH, PATCH, 3)


def video_from_patches(patches: torch.Tensor, grid: int, tubes: int) -> np.ndarray:
    """Inverse of `patches_from_video`. -> [T, H, W, 3] uint8."""
    x = patches.reshape(tubes, grid, grid, TUBELET, PATCH, PATCH, 3)
    x = x.permute(0, 3, 1, 4, 2, 5, 6)
    x = x.reshape(tubes * TUBELET, grid * PATCH, grid * PATCH, 3)
    return (x.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)


def collect(model, processor, repo: str, names: list[str], num_frames: int,
            on_progress=None):
    """Encode clips into (tokens, true patches). Returns float64 for the solve."""
    tubes, grid = jepa.grid_of(model, num_frames)
    toks, pix = [], []
    for i, name in enumerate(names):
        try:
            pv, shown = clip_io.load_clip(processor, repo, name, num_frames)
        except Exception:  # a clip that will not decode should not fail the task
            continue
        seq = jepa.encode(model, pv)
        jepa.check_layout(seq, tubes, grid)
        toks.append(seq.cpu())
        pix.append(patches_from_video(shown, grid))
        if on_progress is not None:
            on_progress(i + 1, len(names))
    X = torch.cat(toks).double()
    Y = torch.cat(pix).reshape(-1, PATCH_DIM).double()
    return X, Y, grid, tubes


def random_projection(Y: torch.Tensor, dim: int, seed: int = 0) -> tuple:
    """A near-lossless 1024-dim view of the true pixels. The decodability ceiling."""
    g = torch.Generator().manual_seed(seed)
    P = torch.randn(PATCH_DIM, dim, generator=g).double() / np.sqrt(PATCH_DIM)
    return Y @ P, P


def _with_bias(X: torch.Tensor) -> torch.Tensor:
    """[N, D] -> [N, D+1]. The ones column is built on X's own device.

    Worth its own function because getting it wrong is invisible until something calls
    this with CUDA tensors: `torch.ones(...)` defaults to the CPU, and the resulting
    `torch.cat` raises "expected all tensors to be on the same device" from inside a
    helper two calls down. `layers.readout_curve` does exactly that, 25 times.
    """
    return torch.cat([X, torch.ones(len(X), 1, dtype=X.dtype, device=X.device)], 1)


def ridge_fit(X: torch.Tensor, Y: torch.Tensor, lam: float = 1e-3):
    """Closed-form ridge with a bias column. Returns W."""
    Xb = _with_bias(X)
    A = Xb.T @ Xb + lam * len(Xb) * torch.eye(Xb.shape[1], dtype=X.dtype, device=X.device)
    return torch.linalg.solve(A, Xb.T @ Y)


def ridge_apply(W: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    return (_with_bias(X) @ W).clamp(0, 1)


def score(pred: torch.Tensor, true: torch.Tensor) -> dict:
    mse = float(((pred - true) ** 2).mean())
    var = float(((true - true.mean(0)) ** 2).mean())
    return {
        "psnr": 99.0 if mse <= 1e-12 else float(10 * np.log10(1.0 / mse)),
        "r2": float(1 - mse / max(var, 1e-12)),
    }


def readout_table(X: torch.Tensor, Y: torch.Tensor, n_val: int = 8192, seed: int = 0):
    """Every condition, fitted and scored the same way. Returns (rows, W_vjepa, P)."""
    Xr, P = random_projection(Y, X.shape[1], seed=seed)
    perm = torch.randperm(len(X), generator=torch.Generator().manual_seed(seed + 1))

    rows = {}
    W_v = ridge_fit(X[:-n_val], Y[:-n_val])
    rows["vjepa"] = score(ridge_apply(W_v, X[-n_val:]), Y[-n_val:])

    W_r = ridge_fit(Xr[:-n_val], Y[:-n_val])
    rows["rand"] = score(ridge_apply(W_r, Xr[-n_val:]), Y[-n_val:])

    Ys = Y[perm]
    W_s = ridge_fit(X[:-n_val], Ys[:-n_val])
    rows["shuf"] = score(ridge_apply(W_s, X[-n_val:]), Ys[-n_val:])

    grey = Y[-n_val:].mean(1, keepdim=True).repeat(1, PATCH_DIM)
    rows["grey"] = score(grey, Y[-n_val:])
    return rows, W_v, W_r, P


def decode_clip(W: torch.Tensor, X: torch.Tensor, grid: int, tubes: int) -> np.ndarray:
    return video_from_patches(
        ridge_apply(W, X).reshape(-1, TUBELET, PATCH, PATCH, 3).float(), grid, tubes
    )


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))
    return 99.0 if mse <= 1e-9 else float(10.0 * np.log10(255.0**2 / mse))
