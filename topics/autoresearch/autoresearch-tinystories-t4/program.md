# AutoResearch Agent Strategy — TinyStories / T4

## Goal
Minimize `val_bpb` (validation bits per byte) on TinyStories.
Lower is better. A change is kept only if it improves val_bpb by more than 0.001.

## Context
- Training budget: 5 minutes per experiment
- GPU: NVIDIA T4 (16GB VRAM)
- Model: GPT transformer, currently DEPTH=8, MODEL_DIM=512
- Dataset: TinyStories (short children's stories, character-level tokenizer)
- Attention: standard scaled_dot_product_attention (no custom kernels)

## What you can change
Only modify hyperparameters and architecture settings in train.py.
The section clearly marked "agent modifies this section" is your playground.

Safe to experiment with:
- LEARNING_RATE — try values between 1e-4 and 1e-3
- WARMUP_STEPS — try 50 to 500
- WEIGHT_DECAY — try 0.01 to 0.3
- GRAD_CLIP — try 0.5 to 2.0
- DEPTH — try 4 to 12 (watch VRAM — deeper = more memory)
- ASPECT_RATIO — try 32 to 96 (MODEL_DIM = DEPTH * ASPECT_RATIO)
- HEAD_DIM — must divide MODEL_DIM evenly, try 64 or 128
- DEVICE_BATCH_SIZE — keep at 16 or lower (T4 VRAM limit)
- TOTAL_BATCH_SIZE — try 2**16 to 2**18

## What you must not change
- WINDOW_PATTERN — must stay "LLLL" (T4 does not support sliding window kernels)
- DATA_DIR, META_PATH, VOCAB_SIZE — data loading, do not touch
- The training loop structure, data loading functions, or logging print statements
- TRAIN_MINUTES — fixed 5-minute budget per experiment

## Strategy hints
1. Start with learning rate — it has the highest impact on 5-minute runs
2. Warmup steps matter more on short runs than long ones
3. Wider models (higher ASPECT_RATIO) often beat deeper ones on simple datasets
4. Weight decay between 0.05 and 0.15 tends to work well for character-level tasks
5. If a change improves val_bpb by more than 0.01, it is a strong signal — explore nearby values
6. If three experiments in a row are reverted, try a completely different category of change

## What a good experiment looks like
- One focused change with a clear hypothesis
- Example: "Increase WARMUP_STEPS from 100 to 200 — short runs may benefit from slower warmup"
- Example: "Reduce WEIGHT_DECAY from 0.1 to 0.05 — TinyStories is clean text, less regularization may help"
- Not: changing five things at once (makes it impossible to know what helped)

## Reporting
After each experiment, clearly state:
- What you changed and why
- val_bpb before and after
- Whether you kept or reverted the change
- What you plan to try next
