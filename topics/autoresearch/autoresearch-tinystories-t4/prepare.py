"""
prepare.py — Data preparation for AutoResearch (T4 / TinyStories adaptation).

Adapted from Karpathy's AutoResearch prepare.py.
Original used climbmix-400b (400B tokens, H100-scale).
This version uses roneneldan/TinyStories (~2GB, suitable for T4 5-minute runs).

Responsibilities:
  - Download TinyStories from HuggingFace
  - Tokenize using a simple character-level or BPE tokenizer
  - Write train.bin and val.bin to the data/ directory
  - Write tokenizer metadata (vocab_size) to data/meta.json

Run once before starting the agent loop:
    python prepare.py

Static file — AutoResearch never modifies this.
"""

import json
import os
import numpy as np
from datasets import load_dataset

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DATASET_NAME = "roneneldan/TinyStories"
VAL_SPLIT_RATIO = 0.1        # 10% of data for validation
VOCAB_SIZE = 8192             # matches train.py expectation

# ── Tokenizer ─────────────────────────────────────────────────────────────────

def _build_tokenizer(texts: list[str]) -> tuple[dict, dict]:
    """
    Build a simple character-level tokenizer from a list of texts.

    Returns:
        stoi : str -> int mapping
        itos : int -> str mapping
    """
    chars = sorted(set("".join(texts)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def _encode(text: str, stoi: dict) -> list[int]:
    """Encode a string to a list of token IDs, skipping unknown characters."""
    return [stoi[ch] for ch in text if ch in stoi]


# ── Main ──────────────────────────────────────────────────────────────────────

def prepare() -> None:
    """
    Download TinyStories, tokenize, and write train.bin / val.bin.

    Output files:
        data/train.bin  — uint16 array of training token IDs
        data/val.bin    — uint16 array of validation token IDs
        data/meta.json  — tokenizer metadata (vocab_size, stoi, itos)
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Downloading TinyStories from HuggingFace...")
    dataset = load_dataset(DATASET_NAME, split="train")

    texts = dataset["text"]
    total = len(texts)
    split_idx = int(total * (1 - VAL_SPLIT_RATIO))

    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]

    print(f"Building tokenizer on {len(train_texts):,} training stories...")
    stoi, itos = _build_tokenizer(train_texts)
    actual_vocab_size = len(stoi)
    print(f"Vocabulary size: {actual_vocab_size}")

    print("Encoding training split...")
    train_ids = []
    for text in train_texts:
        train_ids.extend(_encode(text, stoi))

    print("Encoding validation split...")
    val_ids = []
    for text in val_texts:
        val_ids.extend(_encode(text, stoi))

    train_arr = np.array(train_ids, dtype=np.uint16)
    val_arr = np.array(val_ids, dtype=np.uint16)

    train_path = os.path.join(DATA_DIR, "train.bin")
    val_path = os.path.join(DATA_DIR, "val.bin")
    meta_path = os.path.join(DATA_DIR, "meta.json")

    train_arr.tofile(train_path)
    val_arr.tofile(val_path)

    meta = {
        "vocab_size": actual_vocab_size,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},  # JSON requires string keys
        "dataset": DATASET_NAME,
        "train_tokens": len(train_ids),
        "val_tokens": len(val_ids),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Done.")
    print(f"  train.bin : {len(train_ids):,} tokens → {train_path}")
    print(f"  val.bin   : {len(val_ids):,} tokens → {val_path}")
    print(f"  meta.json : vocab_size={actual_vocab_size} → {meta_path}")


if __name__ == "__main__":
    prepare()
