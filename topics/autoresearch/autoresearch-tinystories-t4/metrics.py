"""
metrics.py — val_bpb parsing and experiment decision logic.

Responsibilities:
  - Parse val_bpb from raw training script output
  - Calculate delta between before/after val_bpb
  - Determine whether a change should be kept or reverted
  - Summarize a completed run's performance statistics

All functions are pure (no I/O, no Firestore) so they can be tested independently.
"""

import re
from dataclasses import dataclass
from typing import Optional

# ── Constants ─────────────────────────────────────────────────────────────────

# val_bpb must improve by at least this much to be considered a real gain.
# Filters out noise from 5-minute training runs.
MIN_IMPROVEMENT_THRESHOLD = 0.001

# Regex to extract val_bpb from training output lines such as:
#   "val_bpb=1.8423" or "val bpb: 1.8423" or "val/bpb 1.8423"
_VAL_BPB_PATTERN = re.compile(
    r"val[_\s/]bpb[=:\s]+([0-9]+\.[0-9]+)",
    re.IGNORECASE,
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ExperimentResult:
    """Structured result of a single AutoResearch experiment."""
    val_bpb_before: float
    val_bpb_after: float
    delta: float          # val_bpb_after - val_bpb_before (negative = improvement)
    kept: bool
    train_loss: Optional[float]
    step_count: Optional[int]


@dataclass
class RunSummary:
    """Aggregate statistics for a completed overnight run."""
    total_experiments: int
    kept_count: int
    reverted_count: int
    starting_val_bpb: float
    final_val_bpb: float
    total_improvement: float   # starting - final (positive = improvement)
    best_delta: float          # most negative delta seen (largest single improvement)
    best_experiment_number: int
    success_rate: float        # kept_count / total_experiments


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_val_bpb(training_output: str) -> Optional[float]:
    """
    Extract the final val_bpb value from raw training script stdout.

    Scans all matches and returns the last one found, since training
    logs multiple checkpoints and the final value is what matters.

    Args:
        training_output: Full stdout string from running train.py.

    Returns:
        val_bpb as float, or None if no match found.
    """
    matches = _VAL_BPB_PATTERN.findall(training_output)
    if not matches:
        return None
    return float(matches[-1])


def parse_train_loss(training_output: str) -> Optional[float]:
    """
    Extract the final training loss from raw training script stdout.

    Looks for patterns like "train_loss=1.234" or "loss: 1.234".

    Args:
        training_output: Full stdout string from running train.py.

    Returns:
        train_loss as float, or None if no match found.
    """
    pattern = re.compile(
        r"(?:train[_\s]loss|loss)[=:\s]+([0-9]+\.[0-9]+)",
        re.IGNORECASE,
    )
    matches = pattern.findall(training_output)
    if not matches:
        return None
    return float(matches[-1])


def parse_step_count(training_output: str) -> Optional[int]:
    """
    Extract the total number of optimizer steps from training output.

    Looks for patterns like "step=1024" or "steps: 1024".

    Args:
        training_output: Full stdout string from running train.py.

    Returns:
        step count as int, or None if no match found.
    """
    pattern = re.compile(
        r"step[s]?[=:\s]+([0-9]+)",
        re.IGNORECASE,
    )
    matches = pattern.findall(training_output)
    if not matches:
        return None
    return int(matches[-1])


# ── Decision logic ────────────────────────────────────────────────────────────

def should_keep(val_bpb_before: float, val_bpb_after: float) -> bool:
    """
    Decide whether to keep or revert a train.py change.

    Keeps the change only if val_bpb improved by more than
    MIN_IMPROVEMENT_THRESHOLD (filters out training noise).

    Args:
        val_bpb_before: val_bpb before the change was applied.
        val_bpb_after : val_bpb after the change was applied.

    Returns:
        True if the change should be kept.
    """
    delta = val_bpb_after - val_bpb_before
    return delta < -MIN_IMPROVEMENT_THRESHOLD


def build_experiment_result(
    val_bpb_before: float,
    val_bpb_after: float,
    training_output: str,
) -> ExperimentResult:
    """
    Build a complete ExperimentResult from before/after bpb and raw output.

    Args:
        val_bpb_before  : val_bpb measured before this experiment.
        val_bpb_after   : val_bpb measured after this experiment.
        training_output : Raw stdout from the training run.

    Returns:
        ExperimentResult with all fields populated.
    """
    delta = round(val_bpb_after - val_bpb_before, 6)
    return ExperimentResult(
        val_bpb_before=val_bpb_before,
        val_bpb_after=val_bpb_after,
        delta=delta,
        kept=should_keep(val_bpb_before, val_bpb_after),
        train_loss=parse_train_loss(training_output),
        step_count=parse_step_count(training_output),
    )


# ── Run summary ───────────────────────────────────────────────────────────────

def summarize_run(experiments: list[dict]) -> Optional[RunSummary]:
    """
    Compute aggregate statistics from a list of experiment dicts.

    Accepts the format returned by firestore_logger.get_experiments().

    Args:
        experiments: List of experiment dicts ordered by experiment_number.

    Returns:
        RunSummary, or None if the list is empty.
    """
    if not experiments:
        return None

    total = len(experiments)
    kept = [e for e in experiments if e.get("kept")]
    kept_count = len(kept)

    starting_val_bpb = experiments[0]["val_bpb_before"]
    final_val_bpb = experiments[-1]["val_bpb_after"]
    total_improvement = round(starting_val_bpb - final_val_bpb, 6)

    # Best single experiment — most negative delta
    best = min(experiments, key=lambda e: e.get("delta", 0))

    return RunSummary(
        total_experiments=total,
        kept_count=kept_count,
        reverted_count=total - kept_count,
        starting_val_bpb=starting_val_bpb,
        final_val_bpb=final_val_bpb,
        total_improvement=total_improvement,
        best_delta=best.get("delta", 0.0),
        best_experiment_number=best.get("experiment_number", 0),
        success_rate=round(kept_count / total, 3) if total > 0 else 0.0,
    )
