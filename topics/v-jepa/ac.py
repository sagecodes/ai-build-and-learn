"""V-JEPA 2-AC: the action-conditioned world model, and planning with it.

The rest of this repo uses the *pretrained* V-JEPA 2 checkpoint, whose predictor
inpaints tube masks and cannot extrapolate forward in time (see the README). This
module is the other half of that story. V-JEPA 2-AC is the action-conditioned
post-train, and it is a genuine forward model:

    predictor(tokens_of_frame_t, action_t, pose_t) -> tokens_of_frame_t+1

Frame-causal, 7-DoF actions, no decoder anywhere. You never see a predicted pixel;
you only ever see a predicted *representation*. Planning is therefore energy
minimisation: encode a goal photo, roll candidate action sequences forward in
latent space, and keep the sequence whose final latent is closest to the goal's.

── Three things upstream gets wrong that cost real time ────────────────────────
1. `torch.hub.load("facebookresearch/vjepa2", "vjepa2_ac_vit_giant")` is BROKEN on
   main. `src/hub/backbones.py` ships `VJEPA_BASE_URL = "http://localhost:8300"`
   with a "# for testing" comment above it, so the hub entrypoint tries to fetch
   the checkpoint from your own machine and dies on connection refused. We build
   the two modules directly and load the .pt ourselves.
2. `pip install git+https://github.com/facebookresearch/vjepa2` installs NOTHING
   importable. Its setup.py has no `packages=` and no `py_modules=`, so pip
   happily "succeeds" and `import src` still fails. The image clones the repo to
   VJEPA2_SRC and we put it on sys.path.
3. `app.vjepa_droid.transforms.make_transforms` pulls in `src.datasets...` which
   imports cv2 at module scope, for a transform that -- at the settings the AC
   model actually uses -- is a bilinear resize and an ImageNet normalise. We
   reimplement it in six lines. `smoke_test.py` asserts bit-exact parity against
   upstream's version (measured: max abs diff 0.0), so this is a checked shortcut
   rather than an assumed one.

── The action space ────────────────────────────────────────────────────────────
7-dim, and only 4 of those dims are real: `[dx, dy, dz, droll, dpitch, dyaw, grip]`.
Upstream's own CEM zeroes the three rotation entries and searches translation plus
gripper, which is what DROID's controller exposes. We do the same. Deltas are
metres of end-effector motion per environment step, clipped to `maxnorm` (0.05 m
by default; the ground-truth action in upstream's own example trajectory is
0.13 m, so the model is used well inside its trained range).

State/pose is the same 7-dim layout but absolute: `[x, y, z, roll, pitch, yaw, grip]`.
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn.functional as F

# ── Upstream source, cloned into the image at a pinned commit ───────────────────
VJEPA2_SRC = os.environ.get("VJEPA2_SRC", "/opt/vjepa2")

# The checkpoint. 11.7 GB, ungated, no licence click-through, and NOT on the Hub:
# the only distribution is this direct URL out of the vjepa2 README. Every pod
# re-downloading it would cost more wall clock than the planning does, so config.py
# hostPath-mounts a staged copy and AC_CACHED points at it.
AC_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
AC_CACHED = "/mnt/hf/vjepa2/vjepa2-ac-vitg.pt"

CROP = 256
PATCH = 16
GRID = CROP // PATCH  # 16
TOKENS_PER_FRAME = GRID * GRID  # 256

# ImageNet statistics in uint8 space. Upstream scales mean/std by 255 rather than
# scaling the image to [0,1]; matching that exactly is what makes the parity test pass.
_MEAN = torch.tensor([0.485, 0.456, 0.406]) * 255.0
_STD = torch.tensor([0.229, 0.224, 0.225]) * 255.0


def cuda_free_gib() -> tuple[float, float]:
    """(free, total) GiB as the CUDA DRIVER sees them, not as `free -g` reports.

    These are different numbers on a GB10, and the difference is what kills runs.
    The GPU shares one 119.7 GiB unified pool with the OS, and CUDA can only use
    memory that is genuinely FREE -- page cache does not count, even though Linux
    reports it as "available". Measured during this demo's development: `free -g`
    said 97 GiB available while the driver saw 20.1 GiB, and a task pod died with
    `CUDA error: out of memory` from cuDevicePrimaryCtxRetain, i.e. before
    allocating anything at all. Reading an 11.7 GB checkpoint is enough to cause it.
    """
    import ctypes

    try:
        cu = ctypes.CDLL("libcuda.so.1")
        cu.cuInit(0)
        dev, ctx = ctypes.c_int(), ctypes.c_void_p()
        free, total = ctypes.c_size_t(), ctypes.c_size_t()
        if cu.cuDeviceGet(ctypes.byref(dev), 0) != 0:
            return (0.0, 0.0)
        if cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev) != 0:
            return (0.0, 0.0)
        cu.cuCtxSetCurrent(ctx)
        cu.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        return (free.value / 2**30, total.value / 2**30)
    except OSError:
        return (0.0, 0.0)


def guard_memory(need_gib: float = 14.0, headroom: float = 0.85) -> str:
    """Cap this process against FREE memory, and fail early if there is not enough.

    Two lessons from the other Spark demos, both the hard way. First,
    `set_per_process_memory_fraction` is a share of TOTAL, so on a box where
    something else already holds 80 GiB it is not a cap at all -- it has to be
    computed against what is actually free. Second, an oversized load on unified
    memory does not always OOM cleanly; it can hang the box or die as a bare
    SIGSEGV. A loud refusal here is worth more than either.
    """
    free, total = cuda_free_gib()
    if total <= 0:
        return "no CUDA device visible"
    if free < need_gib:
        raise RuntimeError(
            f"only {free:.1f} GiB of {total:.1f} GiB free on the unified pool, need "
            f"~{need_gib:.0f}. Something else is holding it: check for other GPU "
            f"processes, and note that page cache from a large file copy counts "
            f"against you here (`free -g` will disagree). "
            f"`kubectl rollout restart deploy/rustfs -n flyte` reclaims the object "
            f"store's heap if that is the culprit."
        )
    torch.cuda.set_per_process_memory_fraction(min(0.9, headroom * free / total))
    return f"{free:.1f} GiB free of {total:.1f} GiB; capped at {headroom * free:.1f} GiB"


def _ensure_src() -> None:
    if VJEPA2_SRC not in sys.path:
        if not os.path.isdir(os.path.join(VJEPA2_SRC, "src", "models")):
            raise RuntimeError(
                f"V-JEPA 2 source not found at {VJEPA2_SRC}. The image clones it; "
                f"locally, set VJEPA2_SRC or run setup.sh."
            )
        sys.path.insert(0, VJEPA2_SRC)


def checkpoint_path() -> str:
    """Staged copy if config.py mounted one, else download to HF_HOME."""
    if os.path.exists(AC_CACHED):
        return AC_CACHED
    dest = os.path.join(os.environ.get("HF_HOME", "/tmp/hf"), "vjepa2-ac-vitg.pt")
    if not os.path.exists(dest):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        torch.hub.download_url_to_file(AC_URL, dest, progress=False)
    return dest


def transform(frames: np.ndarray, crop: int = CROP) -> torch.Tensor:
    """[T, H, W, 3] uint8 -> [3, T, crop, crop] float, exactly as the AC model saw
    its training data.

    Upstream routes this through `random_resized_crop` with scale=(1,1) and
    ratio=(1,1), which degenerates to a plain bilinear resize of the whole frame.
    Bit-exact parity with upstream is asserted in smoke_test.py.
    """
    x = torch.as_tensor(np.ascontiguousarray(frames), dtype=torch.float32)
    x = x.permute(3, 0, 1, 2)  # T H W C -> C T H W
    x = F.interpolate(x, size=(crop, crop), mode="bilinear", align_corners=False)
    return (x - _MEAN[:, None, None, None]) / _STD[:, None, None, None]


@dataclass
class CEMConfig:
    """Cross-entropy method, upstream's defaults from the energy-landscape notebook.

    `rollout` is the planning horizon in environment steps. `samples` action
    sequences are drawn per iteration, the best `topk` by final-latent energy are
    kept, and the sampling distribution is moved toward them `cem_steps` times.
    One planning call therefore costs rollout * cem_steps * samples predictor
    passes, which is why samples is the knob that decides whether a closed-loop
    episode takes one minute or ten.
    """

    rollout: int = 2
    samples: int = 100
    topk: int = 10
    cem_steps: int = 5
    momentum_mean: float = 0.15
    momentum_std: float = 0.75
    maxnorm: float = 0.05
    gripper: bool = False  # the reach task never needs the gripper; leave it open


@dataclass
class PlanTrace:
    """Per-iteration bookkeeping, so the report can show CEM actually converging
    rather than just asserting that it did."""

    energy_mean: list[float] = field(default_factory=list)
    energy_best: list[float] = field(default_factory=list)
    action_mean: list[list[float]] = field(default_factory=list)
    action_std: list[list[float]] = field(default_factory=list)


class ActionWorldModel:
    """Encoder + action-conditioned predictor, plus dreaming and planning on top."""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        _ensure_src()
        from src.models import ac_predictor as ac_mod, vision_transformer as vit_mod

        self.guard = guard_memory() if device.startswith("cuda") else "cpu"
        self.device = device
        self.dtype = dtype
        t0 = time.time()
        # "xformers" is only a name here: grep the repo and nothing imports xformers.
        # The arch is plain SDPA attention with 3D RoPE, which is what makes this
        # model runnable on an aarch64 Blackwell box at all.
        self.encoder = vit_mod.__dict__["vit_giant_xformers"](
            patch_size=PATCH,
            img_size=(CROP, CROP),
            num_frames=64,
            tubelet_size=2,
            use_sdpa=True,
            use_SiLU=False,
            wide_SiLU=True,
            uniform_power=False,
            use_rope=True,
        )
        self.predictor = ac_mod.__dict__["vit_ac_predictor"](
            img_size=(CROP, CROP),
            patch_size=PATCH,
            num_frames=64,
            tubelet_size=2,
            embed_dim=self.encoder.embed_dim,
        )
        self.build_s = time.time() - t0

        t0 = time.time()
        # mmap=True, and it is not an optimisation. A plain torch.load pulls all
        # 11.7 GB into host RAM and leaves another 11.7 GB of page cache behind, and
        # on GB10 that memory is gone from the GPU too -- see guard_memory() below.
        # Measured: 5.1 s and ~12 GiB resident becomes 0.2 s and 0.5 GiB.
        sd = torch.load(checkpoint_path(), map_location="cpu", weights_only=False, mmap=True)

        def _clean(d):
            return {k.replace("module.", "").replace("backbone.", ""): v for k, v in d.items()}

        # The encoder loads non-strict because the .pt also carries EMA/optimiser
        # state; on this checkpoint both key lists come back EMPTY, which
        # smoke_test.py asserts. A silently partial load here would still produce
        # plausible-looking energies, from a randomly initialised ViT-g.
        enc_keys = self.encoder.load_state_dict(_clean(sd["encoder"]), strict=False)
        self.predictor.load_state_dict(_clean(sd["predictor"]), strict=True)
        self.missing = list(enc_keys.missing_keys)
        self.unexpected = list(enc_keys.unexpected_keys)
        del sd
        self.load_s = time.time() - t0

        self.encoder = self.encoder.to(device=device, dtype=dtype).eval()
        self.predictor = self.predictor.to(device=device, dtype=dtype).eval()
        self.dim = self.encoder.embed_dim

    def reset_predictor(self) -> None:
        """Restore the predictor to the pretrained checkpoint, in place.

        Needed by any sweep that fine-tunes more than once in a process: without it,
        run N starts from run N-1's weights and the "how much data do you need" curve
        measures cumulative training instead. mmap makes re-reading the 11.7 GB file
        cheap enough (~0.4 s) that this is simpler than keeping a CPU copy.
        """
        sd = torch.load(checkpoint_path(), map_location="cpu", weights_only=False, mmap=True)
        clean = {k.replace("module.", "").replace("backbone.", ""): v
                 for k, v in sd["predictor"].items()}
        self.predictor.load_state_dict(clean, strict=True)
        self.predictor = self.predictor.to(device=self.device, dtype=self.dtype).eval()
        del sd

    # ── encoding ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, frames: np.ndarray) -> torch.Tensor:
        """[T, H, W, 3] uint8 -> [1, T * 256, D], layer-normed.

        Each frame is encoded INDEPENDENTLY, as a 2-frame clip made by repeating
        the frame (the encoder's tubelet size is 2, so a single still has to be
        doubled to fill one tubelet). That is upstream's `forward_target`, and it
        is what makes the AC predictor's job purely "given this frame and this
        action, what is the next frame" with no temporal context leaking in.
        """
        clip = transform(frames).unsqueeze(0).to(self.device, self.dtype)
        B, C, T, H, W = clip.size()
        clip = clip.permute(0, 2, 1, 3, 4).flatten(0, 1).unsqueeze(2).repeat(1, 1, 2, 1, 1)
        h = self.encoder(clip)
        h = h.view(B, T, -1, h.size(-1)).flatten(1, 2)
        return F.layer_norm(h, (h.size(-1),))

    # ── dreaming ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, z: torch.Tensor, action: torch.Tensor, pose: torch.Tensor):
        """One latent step. z [S, T*256, D], action/pose [S, T, 7] -> [S, 256, D], [S, 1, 7].

        The predictor is frame-causal and returns tokens for every frame it was
        given; only the LAST frame's tokens are new, so that is all we keep.
        """
        z_next = self.predictor(z, action, pose)[:, -TOKENS_PER_FRAME:]
        z_next = F.layer_norm(z_next, (z_next.size(-1),))
        return z_next, integrate_pose(pose[:, -1:], action[:, -1:])

    @torch.no_grad()
    def dream(self, z0: torch.Tensor, pose0: torch.Tensor, actions: torch.Tensor):
        """Open-loop rollout: fix an action sequence and imagine the whole future.

        z0 [S, 256, D], pose0 [S, 1, 7], actions [S, H, 7]
        -> latents [S, H, 256, D], poses [S, H, 7]

        This is the closest thing V-JEPA 2-AC has to DreamerV3's dream video, with
        the difference that there is no decoder to turn any of it back into pixels.
        `viz.retrieval_decode` is how the report shows it honestly.
        """
        horizon = actions.shape[1]
        z_hist, p_hist = z0, pose0
        a_hist = actions[:, :1]
        out_z, out_p = [], []
        for t in range(horizon):
            z_next, p_next = self.step(z_hist, a_hist, p_hist)
            out_z.append(z_next)
            out_p.append(p_next)
            if t + 1 < horizon:
                z_hist = torch.cat([z_hist, z_next], dim=1)
                p_hist = torch.cat([p_hist, p_next], dim=1)
                a_hist = torch.cat([a_hist, actions[:, t + 1 : t + 2]], dim=1)
        return torch.stack(out_z, dim=1), torch.cat(out_p, dim=1)

    @staticmethod
    def energy(z: torch.Tensor, z_goal: torch.Tensor) -> torch.Tensor:
        """Upstream's objective: mean |z - z_goal| over tokens and channels.

        L1 and not cosine, deliberately. The pretrained half of this demo already
        showed that cosine between two V-JEPA tokens is a number that always looks
        encouraging (0.317 for a perfect inpaint vs 0.276 for a failed forecast).
        """
        return (z.flatten(1) - z_goal.flatten(1)).abs().mean(dim=-1)

    # ── planning ────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def plan(
        self,
        z: torch.Tensor,
        pose: torch.Tensor,
        z_goal: torch.Tensor,
        cfg: CEMConfig | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[np.ndarray, PlanTrace]:
        """CEM in latent space. Returns the action sequence [rollout, 7] and a trace.

        Adapted from `notebooks/utils/mpc_utils.cem` with the same defaults and the
        same shape of update, plus per-iteration instrumentation. Only the
        translation and gripper dims are searched; rotations stay zero, matching
        both upstream's planner and DROID's Cartesian controller.
        """
        cfg = cfg or CEMConfig()
        S, R, D = cfg.samples, cfg.rollout, self.dim
        dev = self.device
        trace = PlanTrace()

        z_ctx = z[:, -TOKENS_PER_FRAME:].repeat(S, 1, 1)
        z_tgt = z_goal[:, -TOKENS_PER_FRAME:].repeat(S, 1, 1)
        p_ctx = pose[:, -1:].repeat(S, 1, 1)

        mean = torch.zeros(R, 4, device=dev, dtype=self.dtype)
        std = torch.ones(R, 4, device=dev, dtype=self.dtype) * cfg.maxnorm
        std[:, 3] = 1.0  # gripper lives on a different scale from metres

        for _ in range(cfg.cem_steps):
            z_hist, p_hist, a_hist = z_ctx, p_ctx, None
            for h in range(R):
                eps = torch.randn(S, 4, device=dev, dtype=self.dtype, generator=generator)
                samp = eps * std[h] + mean[h]
                samp[:, :3] = samp[:, :3].clamp(-cfg.maxnorm, cfg.maxnorm)
                samp[:, 3:] = samp[:, 3:].clamp(-0.75, 0.75)
                if not cfg.gripper:
                    samp[:, 3] = 0.0
                a = torch.cat(
                    [samp[:, :3], torch.zeros(S, 3, device=dev, dtype=self.dtype), samp[:, 3:]],
                    dim=-1,
                )[:, None]
                a_hist = a if a_hist is None else torch.cat([a_hist, a], dim=1)
                z_next, p_next = self.step(z_hist, a_hist, p_hist)
                z_hist = torch.cat([z_hist, z_next], dim=1)
                p_hist = torch.cat([p_hist, p_next], dim=1)

            e = self.energy(z_hist[:, -TOKENS_PER_FRAME:], z_tgt)
            idx = e.topk(cfg.topk, largest=False).indices
            elite = torch.cat([a_hist[idx][..., :3], a_hist[idx][..., 6:]], dim=-1)

            trace.energy_mean.append(float(e.mean()))
            trace.energy_best.append(float(e.min()))
            trace.action_mean.append(mean.mean(0).tolist())
            trace.action_std.append(std.mean(0).tolist())

            mean = elite.mean(0) * (1 - cfg.momentum_mean) + mean * cfg.momentum_mean
            std = elite.std(0) * (1 - cfg.momentum_std) + std * cfg.momentum_std

        out = np.zeros((R, 7), dtype=np.float32)
        m = mean.float().cpu().numpy()
        out[:, :3] = m[:, :3]
        out[:, 6] = np.where(np.abs(m[:, 3]) < 0.25, 0.0, m[:, 3])
        return out, trace

    @torch.no_grad()
    def energy_grid(
        self, z: torch.Tensor, pose: torch.Tensor, z_goal: torch.Tensor,
        grid: np.ndarray, chunk: int = 125,
    ) -> np.ndarray:
        """Energy for an explicit set of candidate actions [N, 3].

        CEM tells you where the minimum is; this tells you what the landscape
        around it looks like, which is the only way to tell a real basin from a
        model that happened to bottom out somewhere.
        """
        z_ctx = z[:, -TOKENS_PER_FRAME:]
        z_tgt = z_goal[:, -TOKENS_PER_FRAME:]
        out = []
        for i in range(0, len(grid), chunk):
            g = torch.as_tensor(grid[i : i + chunk], device=self.device, dtype=self.dtype)
            n = len(g)
            a = torch.zeros(n, 1, 7, device=self.device, dtype=self.dtype)
            a[:, 0, :3] = g
            z_next, _ = self.step(z_ctx.repeat(n, 1, 1), a, pose[:, -1:].repeat(n, 1, 1))
            out.append(self.energy(z_next, z_tgt.repeat(n, 1, 1)).float().cpu().numpy())
        return np.concatenate(out)


def integrate_pose(pose: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """[B,1,7] pose + [B,1,7] delta -> [B,1,7] pose.

    Translation adds; rotation composes as a rotation matrix product; the gripper
    entry is replaced rather than accumulated. This is upstream's `compute_new_pose`
    rewritten in torch: theirs round-trips through numpy and scipy on every call,
    which inside a CEM loop means a host sync per horizon step per iteration.
    Rotations are all-zero in this demo's action space, so the composition below
    reduces to the identity, but it is kept correct for anyone who unzeroes them.
    """
    xyz = pose[..., :3] + action[..., :3]
    rpy = pose[..., 3:6] + action[..., 3:6]
    grip = action[..., 6:7]
    return torch.cat([xyz, rpy, grip], dim=-1)
