"""
train.py — GPT training script for AutoResearch (T4 / TinyStories adaptation).

Adapted from Karpathy's AutoResearch train.py.
Key changes from the H100 original:
  - Flash Attention 3 kernel replaced with F.scaled_dot_product_attention (T4 compatible)
  - DEVICE_BATCH_SIZE reduced from 128 to 16 (fits T4 16GB VRAM)
  - TOTAL_BATCH_SIZE reduced from 2**19 to 2**17 (faster gradient accumulation)
  - WINDOW_PATTERN changed from "SSSL" to "LLLL" (no sliding window — no custom kernel)
  - Dataset path points to TinyStories data/ directory

This is the file the AutoResearch agent modifies.
Only hyperparameters and architecture settings should be changed.
Do not modify the data loading or logging sections.

Outputs per run (written to stdout, parsed by metrics.py):
  val_bpb=<float>       — validation bits per byte (lower is better)
  train_loss=<float>    — final training loss
  steps=<int>           — total optimizer steps completed
"""

import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Hyperparameters — agent modifies this section ─────────────────────────────

DEPTH = 8                       # number of transformer layers
ASPECT_RATIO = 64               # model_dim = DEPTH * ASPECT_RATIO
HEAD_DIM = 128                  # attention head dimension
WINDOW_PATTERN = "LLLL"         # L=full attention (SSSL requires Flash Attn 3)

TOTAL_BATCH_SIZE = 2 ** 17      # ~131K tokens per gradient update
DEVICE_BATCH_SIZE = 16          # sequences per forward pass (fits T4 16GB)
MAX_SEQ_LEN = 1024              # context window

LEARNING_RATE = 3e-4
WARMUP_STEPS = 100
WEIGHT_DECAY = 0.1
GRAD_CLIP = 1.0

TRAIN_MINUTES = 5               # fixed training budget per experiment

# ── Derived config (do not modify) ────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
META_PATH = os.path.join(DATA_DIR, "meta.json")

with open(META_PATH) as f:
    _meta = json.load(f)

VOCAB_SIZE = _meta["vocab_size"]
MODEL_DIM = DEPTH * ASPECT_RATIO
NUM_HEADS = MODEL_DIM // HEAD_DIM
GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE // (DEVICE_BATCH_SIZE * MAX_SEQ_LEN)

assert NUM_HEADS > 0, f"MODEL_DIM ({MODEL_DIM}) must be >= HEAD_DIM ({HEAD_DIM})"
assert GRAD_ACCUM_STEPS >= 1, "TOTAL_BATCH_SIZE too small for DEVICE_BATCH_SIZE * MAX_SEQ_LEN"


# ── Model ─────────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch scaled_dot_product_attention."""

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.proj = nn.Linear(model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)  # (B, num_heads, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # T4-compatible: no custom kernel required
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, C)
        return self.proj(out)


class MLP(nn.Module):
    """Feed-forward block with GELU activation."""

    def __init__(self, model_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(model_dim, 4 * model_dim, bias=False)
        self.fc2 = nn.Linear(4 * model_dim, model_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class Block(nn.Module):
    """Single transformer block: LayerNorm → Attention → LayerNorm → MLP."""

    def __init__(self, model_dim: int, num_heads: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(model_dim)
        self.attn = CausalSelfAttention(model_dim, num_heads)
        self.ln2 = nn.LayerNorm(model_dim)
        self.mlp = MLP(model_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """GPT language model."""

    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, MODEL_DIM)
        self.pos_emb = nn.Embedding(MAX_SEQ_LEN, MODEL_DIM)
        self.blocks = nn.ModuleList([Block(MODEL_DIM, NUM_HEADS) for _ in range(DEPTH)])
        self.ln_f = nn.LayerNorm(MODEL_DIM)
        self.lm_head = nn.Linear(MODEL_DIM, VOCAB_SIZE, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))
        x = tok + pos
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_split(split: str) -> np.ndarray:
    path = os.path.join(DATA_DIR, f"{split}.bin")
    return np.fromfile(path, dtype=np.uint16)


def _get_batch(
    data: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    ix = torch.randint(len(data) - MAX_SEQ_LEN, (DEVICE_BATCH_SIZE,))
    x = torch.stack([torch.from_numpy(data[i: i + MAX_SEQ_LEN].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1: i + MAX_SEQ_LEN + 1].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


# ── Learning rate schedule ────────────────────────────────────────────────────

def _get_lr(step: int, total_steps: int) -> float:
    """Linear warmup then cosine decay."""
    if step < WARMUP_STEPS:
        return LEARNING_RATE * (step + 1) / WARMUP_STEPS
    progress = (step - WARMUP_STEPS) / max(total_steps - WARMUP_STEPS, 1)
    return LEARNING_RATE * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── Validation ────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_val_bpb(model: GPT, val_data: np.ndarray, device: torch.device, num_batches: int = 20) -> float:
    """
    Estimate validation bits per byte over num_batches random batches.

    bits per byte = cross_entropy_loss / log(2)
    """
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = _get_batch(val_data, device)
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
        losses.append(loss.item())
    model.train()
    avg_loss = sum(losses) / len(losses)
    return avg_loss / math.log(2)


# ── Training loop ─────────────────────────────────────────────────────────────

def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    train_data = _load_split("train")
    val_data = _load_split("val")

    model = GPT().to(device)
    print(f"Model parameters: {model.num_parameters():,}")
    print(f"MODEL_DIM={MODEL_DIM}, DEPTH={DEPTH}, NUM_HEADS={NUM_HEADS}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # Estimate total steps from a short warmup to calibrate the LR schedule.
    # Run 3 steps, measure throughput, project to the full time budget.
    _warmup_start = time.time()
    _warmup_model = GPT().to(device)
    _warmup_opt = torch.optim.AdamW(_warmup_model.parameters(), lr=LEARNING_RATE)
    for _ in range(3):
        _warmup_opt.zero_grad()
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = _get_batch(train_data, device)
            logits = _warmup_model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            (loss / GRAD_ACCUM_STEPS).backward()
        _warmup_opt.step()
    _secs_per_step = (time.time() - _warmup_start) / 3
    del _warmup_model, _warmup_opt
    estimated_total_steps = max(int(TRAIN_MINUTES * 60 / _secs_per_step), 1)
    print(f"Estimated steps in {TRAIN_MINUTES}min: {estimated_total_steps} ({_secs_per_step:.2f}s/step)")

    deadline = time.time() + TRAIN_MINUTES * 60
    step = 0
    train_loss = 0.0

    model.train()
    while time.time() < deadline:
        lr = _get_lr(step, total_steps=estimated_total_steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation — average loss across micro-batches
        optimizer.zero_grad()
        accum_loss = 0.0
        for _ in range(GRAD_ACCUM_STEPS):
            x, y = _get_batch(train_data, device)
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, VOCAB_SIZE), y.view(-1))
            (loss / GRAD_ACCUM_STEPS).backward()
            accum_loss += loss.item() / GRAD_ACCUM_STEPS

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        train_loss = accum_loss
        step += 1

        if step % 50 == 0:
            elapsed = time.time() - (deadline - TRAIN_MINUTES * 60)
            print(f"step={step} train_loss={train_loss:.4f} lr={lr:.2e} elapsed={elapsed:.0f}s")

    # Final evaluation
    val_bpb = evaluate_val_bpb(model, val_data, device)
    print(f"val_bpb={val_bpb:.6f}")
    print(f"train_loss={train_loss:.6f}")
    print(f"steps={step}")


if __name__ == "__main__":
    train()
