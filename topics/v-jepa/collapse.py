"""Why JEPA needs an EMA teacher: train a tiny one four ways and watch three of them.

Every other task in this repo treats V-JEPA 2 as a finished checkpoint. That hides the
single most important thing about the method, which is that its objective is trivially
satisfiable. "Predict the representation of the hidden part" is solved perfectly by a
representation that is CONSTANT: emit the same vector for every input, predict that
vector, and the loss is zero forever. Nothing in the loss prevents this. It is why
JEPA's architecture is asymmetric -- the targets come from an exponential moving
average of the encoder and no gradient flows into them -- and why "we predict in
representation space" is only half of the idea.

So this module trains a ~1M parameter JEPA from scratch on synthetic video, four ways,
with everything else held fixed: same data, same masks, same steps, same optimiser,
same encoder and predictor shapes.

  ema        target encoder is an EMA of the online encoder, stop-grad. V-JEPA's recipe.
  stopgrad   target encoder IS the online encoder, detached. Tests whether the EMA is
             doing the work or whether stopping the gradient is already enough (which
             is what SimSiam claims for images, and is worth checking rather than
             asserting).
  none       target encoder is the online encoder WITH gradients. The degenerate
             solution is reachable by gradient descent, so this is the collapse arm.
  pixels     identical encoder and predictor, but the predictor outputs PIXELS and the
             loss is L1 to the true patch. A masked autoencoder, i.e. the thing JEPA
             is arguing against. It cannot collapse, because the targets are fixed.

plus `random`, the same encoder at initialisation, never trained, which is the control
that says whether any of the numbers below are about learning at all.

── What is measured, and why the loss is not the answer ───────────────────────
The headline is that `none` has the LOWEST loss of the three JEPA arms, by a factor of
about 240. A reader comparing training curves would pick it. So the loss goes in the
report next to two things it cannot fake:

  pair cosine      the mean cosine between the pooled features of two different clips,
                   UNCENTRED. 1.00 means the encoder ignores its input. `spread` below
                   explains at length why the centred version of this, and the centred
                   effective rank, both rank the collapsed arm as the best one.
  linear probes    three frozen-feature probes on known generative factors: COLOUR and
                   SHAPE, which are visible in a single frame, and DIRECTION OF MOTION,
                   which is not in any single frame and can only come from the video.

That last split is the reason the synthetic data exists rather than using Kinetics. The
factors are known exactly, they are independent by construction (asserted in
`smoke_test.py`), and one of them is purely dynamic.

── What the untrained control does and does not allow you to claim ────────────
The probes separate collapse from health cleanly: V-JEPA's recipe beats the collapse arm
on all three factors (measured: colour 99% vs 89%, shape 56% vs 42%, direction 49% vs
38%), and it is the only comparison here that has come out the same way on every run.

Against a RANDOMLY INITIALISED encoder of the same shape it is more interesting, and the
answer depends entirely on the label budget, which is why `probe_shots` exists:

  colour       at 1000 labels the untrained encoder is level (97% vs 99%); at 25 labels
               it is 23 to 37 POINTS behind the two healthy latent arms (51% against 75%
               and 89%), and AHEAD of the collapsed one (51% against 49%). That shape,
               a large advantage that disappears as labels arrive, is what a
               representation is for. The magnitude is not stable between runs: the same
               configuration put the EMA arm at 92% rather than 75% on an earlier run,
               so treat the sign as the result and not the size.
  shape        the untrained encoder is ahead at every budget (38% vs 36% at 25 labels,
               63% vs 53% at 1000). Training this objective on this world did not help.
  direction    everything sits at chance at 25 labels, so that end of the curve says
               nothing, and by 1000 labels the arms are level.

A random convolutional basis plus a linear layer fitted on a thousand examples is a
strong model of a world with three shapes and three colours, so the full-budget column
is the wrong place to look. Nothing here is a statement about V-JEPA 2: this is 0.9M
parameters trained for two minutes. `probe` on Kinetics is where frozen-feature quality
is demonstrated at scale (78% five-way action recognition against 40% for raw pixels).
"""

from __future__ import annotations

import copy
import logging
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

# ── The synthetic world ─────────────────────────────────────────────────────────
#
# One moving shape on a cluttered static background. Three generative factors, drawn
# independently: shape, colour, and direction of travel.
#
# The trajectory is centred rather than started uniformly, which is not cosmetic. If
# the start position were uniform, "moving right" would have to start on the left to
# stay in frame, and the direction probe could be solved from the first frame alone.
# Sampling the MIDPOINT and stepping half the trajectory backwards makes start
# position independent of direction, so a direction probe has to use the motion.

SHAPES = ("square", "disc", "triangle")
COLOURS = ((235, 90, 80), (85, 205, 130), (110, 145, 245))
DIRECTIONS = ((1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1))

SIZE = 64
FRAMES = 8
PATCH = 8
TUBELET = 2
RADIUS = 7
SPEED = 2.5


def _shape_mask(kind: str, cx: float, cy: float, r: float, size: int) -> np.ndarray:
    ys, xs = np.mgrid[0:size, 0:size]
    dx, dy = xs - cx, ys - cy
    if kind == "square":
        return (np.abs(dx) <= r) & (np.abs(dy) <= r)
    if kind == "disc":
        return dx**2 + dy**2 <= r**2
    return (dy >= -r) & (dy <= r) & (np.abs(dx) <= (dy + r) / 2.0)


def _background(rng: np.random.Generator, size: int) -> np.ndarray:
    """A random gradient plus static clutter, so appearance carries no label info."""
    ys, xs = np.mgrid[0:size, 0:size].astype(np.float32) / size
    a, b, c = rng.uniform(20, 70, 3)
    base = a * xs + b * ys + c
    img = np.stack([base + rng.uniform(-12, 12) for _ in range(3)], -1)
    for _ in range(rng.integers(2, 5)):
        cx, cy = rng.uniform(0, size, 2)
        r = rng.uniform(2.5, 5.0)
        m = _shape_mask("disc", cx, cy, r, size)
        img[m] = np.clip(img[m] + rng.uniform(-45, 45), 0, 255)
    return img.clip(0, 255)


def make_dataset(n: int, seed: int = 0, size: int = SIZE, frames: int = FRAMES,
                 radius: float = RADIUS, speed: float = SPEED) -> dict:
    """n clips of one shape moving in a straight line. Returns video plus the factors."""
    rng = np.random.default_rng(seed)
    travel = speed * (frames - 1) / 2.0
    lo, hi = radius + travel + 1, size - radius - travel - 1
    if lo >= hi:
        raise ValueError(f"trajectory does not fit: {lo:.1f} >= {hi:.1f}")

    vids = np.empty((n, frames, size, size, 3), dtype=np.uint8)
    shape_id = rng.integers(0, len(SHAPES), n)
    colour_id = rng.integers(0, len(COLOURS), n)
    dir_id = rng.integers(0, len(DIRECTIONS), n)

    for i in range(n):
        bg = _background(rng, size)
        dx, dy = DIRECTIONS[dir_id[i]]
        norm = math.hypot(dx, dy)
        vx, vy = speed * dx / norm, speed * dy / norm
        mx, my = rng.uniform(lo, hi), rng.uniform(lo, hi)
        cx = mx - vx * (frames - 1) / 2.0
        cy = my - vy * (frames - 1) / 2.0
        colour = np.array(COLOURS[colour_id[i]], dtype=np.float32)
        for t in range(frames):
            frame = bg.copy()
            m = _shape_mask(SHAPES[shape_id[i]], cx + vx * t, cy + vy * t, radius, size)
            frame[m] = colour
            vids[i, t] = frame.astype(np.uint8)

    return {
        "video": vids,
        "shape": torch.tensor(shape_id, dtype=torch.long),
        "colour": torch.tensor(colour_id, dtype=torch.long),
        "direction": torch.tensor(dir_id, dtype=torch.long),
    }


FACTORS = {"colour": len(COLOURS), "shape": len(SHAPES), "direction": len(DIRECTIONS)}
# Which factors a single frame can possibly answer, and which need the motion. The
# split is the whole point of the probe table.
STATIC_FACTORS = ("colour", "shape")
DYNAMIC_FACTORS = ("direction",)


def to_tensor(video: np.ndarray) -> torch.Tensor:
    """uint8 [B, T, H, W, 3] -> float [B, 3, T, H, W] in [-1, 1]."""
    x = torch.as_tensor(video, dtype=torch.float32).permute(0, 4, 1, 2, 3) / 127.5 - 1.0
    return x


def patchify(video: torch.Tensor, patch: int = PATCH, tubelet: int = TUBELET):
    """[B, 3, T, H, W] -> [B, N, tubelet*patch*patch*3], same raster as the tokens.

    Index is t*G*G + h*G + w with t over tubelets, matching `jepa.py`'s note about
    V-JEPA 2's own layout. The pixel arm of this experiment is only a fair comparison
    if its targets are in the same order as the latent arm's, so this is asserted in
    `smoke_test.py` rather than eyeballed.
    """
    B, C, T, H, W = video.shape
    g = H // patch
    x = video.permute(0, 2, 3, 4, 1)  # B T H W C
    x = x.reshape(B, T // tubelet, tubelet, g, patch, g, patch, C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7)
    return x.reshape(B, (T // tubelet) * g * g, tubelet * patch * patch * C)


# ── A tiny ViT, and a tiny predictor ───────────────────────────────────────────


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp)), nn.GELU(), nn.Linear(int(dim * mlp), dim)
        )

    def forward(self, x):
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.mlp(self.norm2(x))


class Encoder(nn.Module):
    """Conv3d patch embedding, learned positions, a few blocks. No class token."""

    def __init__(self, dim: int = 128, depth: int = 4, heads: int = 4,
                 size: int = SIZE, frames: int = FRAMES, patch: int = PATCH,
                 tubelet: int = TUBELET):
        super().__init__()
        self.grid = size // patch
        self.tubes = frames // tubelet
        self.n_tokens = self.tubes * self.grid * self.grid
        self.proj = nn.Conv3d(3, dim, (tubelet, patch, patch), (tubelet, patch, patch))
        self.pos = nn.Parameter(torch.zeros(1, self.n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, video: torch.Tensor, keep: torch.Tensor | None = None):
        """`keep` drops the masked tokens BEFORE the blocks, as V-JEPA does.

        This is not an optimisation. If the context encoder saw the whole clip and the
        mask were applied only at the predictor's input, the context tokens would have
        attended to the hidden patches and the prediction task would be partly a copy.
        Note that `transformers`' VJEPA2Model does exactly that (it encodes the full
        video, then splits), which is fine for inference and wrong for training.
        """
        x = self.proj(video).flatten(2).transpose(1, 2) + self.pos
        if keep is not None:
            x = x[:, keep]
        for b in self.blocks:
            x = b(x)
        return self.norm(x)

    def pooled(self, video: torch.Tensor) -> torch.Tensor:
        return self.forward(video).mean(1)


class Predictor(nn.Module):
    """Context tokens plus positioned mask tokens -> one output per target position."""

    def __init__(self, dim: int, out_dim: int, n_tokens: int, depth: int = 2,
                 heads: int = 4):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, out_dim)

    def forward(self, ctx: torch.Tensor, ctx_ids: torch.Tensor, tgt_ids: torch.Tensor):
        B = ctx.shape[0]
        x = ctx + self.pos[:, ctx_ids]
        m = self.mask_token.expand(B, len(tgt_ids), -1) + self.pos[:, tgt_ids]
        h = torch.cat([x, m], 1)
        for b in self.blocks:
            h = b(h)
        return self.head(self.norm(h[:, len(ctx_ids):]))


# ── Masks in the tiny raster ────────────────────────────────────────────────────


def tube_mask(tubes: int, grid: int, size: int = 4, rng: np.random.Generator | None = None):
    """A `size`x`size` spatial block removed across all tubelets. V-JEPA's shape."""
    m = torch.zeros(tubes, grid, grid, dtype=torch.bool)
    if rng is None:
        h = w = (grid - size) // 2
    else:
        h = int(rng.integers(0, grid - size + 1))
        w = int(rng.integers(0, grid - size + 1))
    m[:, h : h + size, w : w + size] = True
    flat = m.reshape(-1)
    return torch.nonzero(~flat).squeeze(1), torch.nonzero(flat).squeeze(1)


# ── What collapse looks like as a number ───────────────────────────────────────


def effective_rank(Z: torch.Tensor, center: bool = False) -> float:
    """exp(entropy of the normalised singular value spectrum). 1.0 means collapse.

    No threshold to tune, unlike "how many singular values exceed epsilon", and a
    single number, so it can be plotted against training step.

    `center=False` is the default and it is the whole subtlety of this module. Read
    the note on `spread` below before changing it.
    """
    Z = Z.detach().double()
    if center:
        Z = Z - Z.mean(0)
    s = torch.linalg.svdvals(Z)
    s = s / (s.sum() + 1e-12)
    s = s[s > 0]
    return float(torch.exp(-(s * s.log()).sum()))


def spread(Z: torch.Tensor) -> dict:
    """The collapse detectors, and the one that gets it backwards.

    `pair_cos` is the headline: the mean cosine between the pooled features of two
    DIFFERENT clips, computed WITHOUT centring. 1.00 means the encoder returns the
    same direction whatever it is shown, which is collapse stated in one number.

    `erank` is the same story spectrally, on row-normalised features, again uncentred.

    `erank_centred` is the version a reader would reach for first, and it is a trap
    this experiment walked into: subtracting the per-dimension mean removes exactly
    the constant vector that collapse produces, leaving the numerical residue, which
    is unstructured and therefore close to FULL rank. Measured here, the collapsed arm
    scores a centred effective rank of 66 against 22 for the healthy one, i.e. the
    metric ranks collapse as the richest representation in the experiment. It is
    reported so that the report can say so, not because it is informative.

    `std` is the shape of it: the collapse is a scale collapse, and three orders of
    magnitude of per-dimension standard deviation separate the arms.

    WHERE it happens is not where it looks like it should. The obvious suspect is the
    encoder's final LayerNorm learned gain being driven to zero, and that is not what
    happens: measured after 600 steps, the gain's mean magnitude is 0.970 in the
    collapsed arm against 0.992 in the healthy one and 1.0 at initialisation, i.e.
    essentially untouched. The collapse is in the BODY. The pooled activation std
    measured BEFORE the final LayerNorm is 1.788 healthy and 0.062 collapsed, so the
    blocks are already mapping every clip to nearly the same activation. The LayerNorm
    then removes the between-clip scale that is left (0.062 -> 0.0009), which makes the
    pooled descriptor look even flatter than the body alone would.
    """
    Z = Z.detach()
    U = F.normalize(Z, dim=-1)
    sims = U @ U.T
    off = sims[~torch.eye(len(Z), dtype=torch.bool, device=Z.device)]
    return {
        "std": float(Z.std(0).mean()),
        "pair_cos": float(off.mean()),
        "erank": effective_rank(U),
        "erank_centred": effective_rank(Z, center=True),
    }


# ── Training ────────────────────────────────────────────────────────────────────

KINDS = ("ema", "stopgrad", "none", "pixels")


def train(kind: str, train_video: np.ndarray, eval_video: np.ndarray,
          steps: int = 2000, batch: int = 64, dim: int = 128, depth: int = 4,
          lr: float = 1e-3, wd: float = 0.04, momentum: float = 0.996,
          mask_size: int = 4, seed: int = 0, log_every: int = 50,
          device: str | None = None, on_progress=None) -> dict:
    """One arm of the experiment. Everything but `kind` is shared across arms.

    Returns the history, the final encoder, and (for `pixels`) the predictor, so the
    report can show the one arm that has a decoder to show.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    enc = Encoder(dim=dim, depth=depth).to(device)
    pix_dim = TUBELET * PATCH * PATCH * 3
    out_dim = pix_dim if kind == "pixels" else dim
    pred = Predictor(dim, out_dim, enc.n_tokens).to(device)
    target = copy.deepcopy(enc).requires_grad_(False) if kind == "ema" else None

    params = list(enc.parameters()) + list(pred.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    X = to_tensor(train_video)
    Xe = to_tensor(eval_video).to(device)
    n_params = sum(p.numel() for p in params)
    history: list[dict] = []
    t0 = time.time()

    for step in range(steps):
        idx = torch.as_tensor(rng.integers(0, len(X), batch))
        vid = X[idx].to(device, non_blocking=True)
        ctx_ids, tgt_ids = tube_mask(enc.tubes, enc.grid, mask_size, rng)

        ctx = enc(vid, keep=ctx_ids)
        out = pred(ctx, ctx_ids, tgt_ids)

        if kind == "pixels":
            tgt = patchify(vid)[:, tgt_ids]
        elif kind == "ema":
            with torch.no_grad():
                tgt = target(vid)[:, tgt_ids]
        elif kind == "stopgrad":
            with torch.no_grad():
                tgt = enc(vid)[:, tgt_ids]
        else:  # "none": the targets are differentiable, so collapse is reachable
            tgt = enc(vid)[:, tgt_ids]

        loss = F.l1_loss(out, tgt)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()

        if kind == "ema":
            with torch.no_grad():
                for p, q in zip(target.parameters(), enc.parameters()):
                    p.mul_(momentum).add_(q.detach(), alpha=1 - momentum)
                for p, q in zip(target.buffers(), enc.buffers()):
                    p.copy_(q)

        if step % log_every == 0 or step == steps - 1:
            enc.eval()
            with torch.no_grad():
                Z = enc.pooled(Xe)
            enc.train()
            row = {"step": step, "loss": float(loss.detach()), **spread(Z)}
            history.append(row)
            log.info("%-9s step %4d loss %.4f pair_cos %.3f erank %5.2f std %.4f "
                     "(centred erank %5.1f)", kind, step, row["loss"], row["pair_cos"],
                     row["erank"], row["std"], row["erank_centred"])
            if on_progress is not None:
                on_progress(kind, step, steps, row)

    enc.eval()
    return {
        "kind": kind,
        "history": history,
        "encoder": enc,
        "predictor": pred,
        "params": n_params,
        "seconds": time.time() - t0,
        "final": history[-1],
    }


def untrained(dim: int = 128, depth: int = 4, seed: int = 0,
              device: str | None = None) -> dict:
    """The control: the same encoder, never trained. Nothing here is fitted."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed + 1234)
    enc = Encoder(dim=dim, depth=depth).to(device).eval()
    return {"kind": "random", "history": [], "encoder": enc, "predictor": None,
            "params": sum(p.numel() for p in enc.parameters()), "seconds": 0.0,
            "final": {}}


# ── Frozen-feature probes on the known factors ─────────────────────────────────


@torch.no_grad()
def features(enc: Encoder, video: np.ndarray, mode: str = "spacetime",
             cells: int = 2, chunk: int = 128) -> torch.Tensor:
    """Frozen features for the probe, pooled one of two ways.

    `pooled` is the clip descriptor the rest of this repo uses: the mean over every
    token, one vector per clip.

    `spacetime` keeps a coarse layout: the mean over a `cells`x`cells` spatial grid
    within each tubelet, concatenated over tubelets.

    The difference is not a tuning knob, it is a fact about what a linear probe can
    possibly do. DIRECTION OF MOTION is a statement about how position changes with
    time, and a mean over all tokens has destroyed both position and time before the
    probe sees anything. Measured here, the direction probe sits at chance for EVERY
    arm under `pooled`, including the arms that demonstrably encode direction, and the
    untrained control scores highest of all. That is the probe failing, not the model,
    and a report that used only `pooled` would have concluded that no objective learns
    about motion. The task prints both.
    """
    device = next(enc.parameters()).device
    X = to_tensor(video)
    out = []
    for i in range(0, len(X), chunk):
        tok = enc(X[i : i + chunk].to(device))
        if mode == "pooled":
            out.append(tok.mean(1).cpu())
            continue
        B, N, D = tok.shape
        grid = enc.grid
        z = tok.reshape(B, enc.tubes, grid, grid, D).permute(0, 1, 4, 2, 3)
        z = F.adaptive_avg_pool2d(z.reshape(B * enc.tubes, D, grid, grid), cells)
        out.append(z.reshape(B, -1).cpu())
    return torch.cat(out)


def probe_all(enc: Encoder, data: dict, train_mask: torch.Tensor, seed: int = 0,
              mode: str = "spacetime") -> dict:
    """Linear probe for every generative factor, on identical splits and features.

    Uses `probing.linear_probe`, the same code the Kinetics probe uses, so the
    standardise-on-train-statistics discipline is the same one the rest of the demo is
    held to rather than a second implementation with its own bugs.
    """
    import probing

    X = features(enc, data["video"], mode=mode)
    rows = {}
    for name, n_class in FACTORS.items():
        acc, _ = probing.linear_probe(X, data[name], train_mask, seed=seed)
        rows[name] = {"acc": acc, "chance": 1.0 / n_class}
    rows["erank"] = effective_rank(F.normalize(X[~train_mask], dim=-1))
    rows["dim"] = int(X.shape[1])
    return rows


SHOTS = (25, 50, 100, 250, 500, 1000)


def probe_shots(enc: Encoder, data: dict, train_mask: torch.Tensor,
                sizes: tuple[int, ...] = SHOTS, seed: int = 0,
                mode: str = "spacetime") -> dict[str, list[tuple[int, float]]]:
    """Probe accuracy against the number of labelled examples, per factor.

    This exists because the full-size probe could not tell the arms apart. At 1000
    labels a RANDOM 2048-dimensional feature scores as well as anything trained: a
    random convolutional basis is a rich enough kernel that a linear layer with that
    many examples can fit these factors through it. That is not a flaw in the control,
    it is the control doing its job, and the answer is not to drop it but to ask the
    question where representation quality actually shows up.

    Low-shot probing is the standard instrument for that. The features are computed
    once and only the number of rows the probe may fit on varies, so nothing else
    differs between the points on the curve. If the arms do not separate at 25 labels
    either, then this task measures the collapse mechanism and says nothing about
    representation quality, and the report has to say so.
    """
    import probing

    X = features(enc, data["video"], mode=mode)
    pool = torch.nonzero(train_mask).squeeze(1)
    out: dict[str, list[tuple[int, float]]] = {f: [] for f in FACTORS}
    for n in sizes:
        if n > len(pool):
            continue
        sub = torch.zeros_like(train_mask)
        sub[pool[:n]] = True
        for factor in FACTORS:
            # The held-out set is every clip outside the TRAIN HALF, identical at every
            # n, so the accuracies are comparable across the curve. Shrinking the fit
            # set must not quietly grow the test set.
            acc, _ = probing.linear_probe(X, data[factor], sub, seed=seed,
                                          test=~train_mask)
            out[factor].append((n, acc))
    return out


@torch.no_grad()
def reconstruct(res: dict, video: np.ndarray, mask_size: int = 4) -> np.ndarray | None:
    """The pixel arm's own reconstruction, which is the only arm that HAS one.

    Included as a visual because it makes the trade concrete: the model whose output
    you can look at is the one whose representation does worse on motion. Returns the
    clip with the predicted patches pasted into the hole, or None for a latent arm.
    """
    if res["kind"] != "pixels":
        return None
    enc, pred = res["encoder"], res["predictor"]
    device = next(enc.parameters()).device
    x = to_tensor(video[None]).to(device)
    ctx_ids, tgt_ids = tube_mask(enc.tubes, enc.grid, mask_size, None)
    out = pred(enc(x, keep=ctx_ids), ctx_ids, tgt_ids)[0].float().cpu()

    patches = out.reshape(len(tgt_ids), TUBELET, PATCH, PATCH, 3)
    patches = ((patches.clamp(-1, 1) + 1.0) * 127.5).numpy().astype(np.uint8)
    frames = video.copy()
    ppf = enc.grid * enc.grid
    for k, tok in enumerate(tgt_ids.tolist()):
        t, rest = tok // ppf, tok % ppf
        h, w = rest // enc.grid, rest % enc.grid
        for j in range(TUBELET):
            f = t * TUBELET + j
            if f < len(frames):
                frames[f, h * PATCH : (h + 1) * PATCH, w * PATCH : (w + 1) * PATCH] = \
                    patches[k, j]
    return frames


def masked_clip(video: np.ndarray, grid: int, tubes: int, mask_size: int = 4) -> np.ndarray:
    """The tiny model's literal input, hole blacked out, for the report."""
    _, tgt_ids = tube_mask(tubes, grid, mask_size, None)
    out = video.copy()
    ppf = grid * grid
    for tok in tgt_ids.tolist():
        t, rest = tok // ppf, tok % ppf
        h, w = rest // grid, rest % grid
        for j in range(TUBELET):
            f = t * TUBELET + j
            if f < len(out):
                out[f, h * PATCH : (h + 1) * PATCH, w * PATCH : (w + 1) * PATCH] = 15
    return out
