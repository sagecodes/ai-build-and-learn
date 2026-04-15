"""
agent.py — AutoResearch agent loop.

Responsibilities:
  - Read program.md (strategy guide) and current train.py
  - Call Claude to propose one focused change to train.py
  - Apply the change, run training, measure val_bpb
  - Keep or revert the change based on metrics.should_keep()
  - Log every experiment to Firestore via firestore_logger
  - Repeat until the time budget is exhausted

Usage:
    python agent.py

Environment variables:
    ANTHROPIC_API_KEY   — required, Claude API key
    GCP_PROJECT         — required, GCP project ID for Firestore
    RUN_HOURS           — optional, overnight budget in hours (default: 8)
"""

import difflib
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import anthropic

import firestore_logger
import metrics

# ── Config ────────────────────────────────────────────────────────────────────

TRAIN_SCRIPT = Path(__file__).parent / "train.py"
PROGRAM_MD = Path(__file__).parent / "program.md"
TRAIN_BACKUP = Path(__file__).parent / "train.py.bak"

RUN_HOURS = float(os.getenv("RUN_HOURS", "8"))
RUN_SECONDS = RUN_HOURS * 3600

CLAUDE_MODEL = "claude-sonnet-4-6"

# ── Prompts ───────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are an AI research assistant running automated ML experiments.
Your job is to propose exactly ONE focused change to train.py to improve val_bpb.
You must follow the strategy in program.md exactly.
Respond with two sections:
1. REASONING: one short paragraph explaining what you are changing and why.
2. NEW_TRAIN_PY: the complete updated train.py file with your change applied.
No other text outside these two sections."""

def _build_user_prompt(
    program: str,
    current_train: str,
    experiment_history: list[dict],
) -> str:
    """Build the prompt sent to Claude for each experiment."""
    history_lines = []
    for exp in experiment_history[-10:]:  # last 10 experiments for context
        status = "KEPT" if exp.get("kept") else "REVERTED"
        history_lines.append(
            f"  Exp {exp['experiment_number']}: {exp['change_description']} "
            f"→ delta={exp['delta']:+.4f} [{status}]"
        )
    history_text = "\n".join(history_lines) if history_lines else "  (no experiments yet)"

    return f"""## Strategy Guide (program.md)
{program}

## Current train.py
```python
{current_train}
```

## Experiment History (most recent 10)
{history_text}

Propose ONE change to improve val_bpb. Follow the strategy guide."""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_file(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _compute_diff(before: str, after: str) -> str:
    """Return a unified diff string between two versions of train.py."""
    lines_before = before.splitlines(keepends=True)
    lines_after = after.splitlines(keepends=True)
    diff = difflib.unified_diff(lines_before, lines_after, fromfile="train.py (before)", tofile="train.py (after)")
    return "".join(diff)


def _parse_llm_response(response: str) -> tuple[str, str]:
    """
    Extract REASONING and NEW_TRAIN_PY from Claude's response.

    Returns:
        (reasoning, new_train_py) — both as strings.
    Raises:
        ValueError if the expected sections are not found.
    """
    reasoning = ""
    new_train_py = ""

    if "REASONING:" in response:
        reasoning_start = response.index("REASONING:") + len("REASONING:")
        reasoning_end = response.index("NEW_TRAIN_PY:") if "NEW_TRAIN_PY:" in response else len(response)
        reasoning = response[reasoning_start:reasoning_end].strip()

    if "NEW_TRAIN_PY:" in response:
        code_start = response.index("NEW_TRAIN_PY:") + len("NEW_TRAIN_PY:")
        code_block = response[code_start:].strip()
        # Strip markdown code fences if present
        if code_block.startswith("```"):
            code_block = code_block.split("\n", 1)[1]
        if code_block.endswith("```"):
            code_block = code_block.rsplit("```", 1)[0]
        new_train_py = code_block.strip()

    if not new_train_py:
        raise ValueError("Claude response did not contain NEW_TRAIN_PY section")

    return reasoning, new_train_py


def _run_training() -> tuple[str, int]:
    """
    Run train.py as a subprocess and return (stdout, returncode).

    Captures both stdout and stderr merged into one string.
    """
    result = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT)],
        capture_output=True,
        text=True,
        timeout=600,  # 10-minute hard timeout (5-min run + overhead)
    )
    output = result.stdout + "\n" + result.stderr
    return output, result.returncode


# ── Main loop ─────────────────────────────────────────────────────────────────

def run() -> None:
    """Run the AutoResearch agent loop for RUN_HOURS hours."""
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    program = _read_file(PROGRAM_MD)
    gcp_project = os.getenv("GCP_PROJECT")

    # Snapshot initial config for the run document
    initial_config = {
        "model": CLAUDE_MODEL,
        "run_hours": RUN_HOURS,
        "train_script": TRAIN_SCRIPT.name,
        "dataset": "TinyStories",
        "gpu": "T4",
    }
    run_id = firestore_logger.create_run(config=initial_config, project_id=gcp_project)
    print(f"Run started: {run_id}")

    deadline = time.time() + RUN_SECONDS
    experiment_number = 0
    experiment_history: list[dict] = []

    # Measure baseline val_bpb before any changes
    print("Measuring baseline val_bpb...")
    baseline_output, _ = _run_training()
    current_val_bpb = metrics.parse_val_bpb(baseline_output)
    if current_val_bpb is None:
        print("ERROR: Could not parse baseline val_bpb. Check that train.py runs correctly.")
        return
    print(f"Baseline val_bpb={current_val_bpb:.6f}")

    while time.time() < deadline:
        experiment_number += 1
        exp_start = datetime.now(timezone.utc).isoformat()
        exp_start_time = time.time()

        print(f"\n{'='*60}")
        print(f"Experiment {experiment_number} | val_bpb={current_val_bpb:.6f}")
        print(f"{'='*60}")

        # Back up current train.py
        shutil.copy(TRAIN_SCRIPT, TRAIN_BACKUP)
        current_train = _read_file(TRAIN_SCRIPT)

        # Ask Claude to propose a change
        try:
            message = client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=4096,
                system=_SYSTEM_PROMPT,
                messages=[{
                    "role": "user",
                    "content": _build_user_prompt(program, current_train, experiment_history),
                }],
            )
            response_text = message.content[0].text
        except Exception as e:
            print(f"Claude API error: {e}. Skipping experiment.")
            continue

        # Parse Claude's response
        try:
            reasoning, new_train_py = _parse_llm_response(response_text)
        except ValueError as e:
            print(f"Parse error: {e}. Skipping experiment.")
            continue

        print(f"Change proposed: {reasoning[:200]}")

        # Apply the change
        _write_file(TRAIN_SCRIPT, new_train_py)
        change_diff = _compute_diff(current_train, new_train_py)

        # Run training
        print("Training...")
        try:
            training_output, returncode = _run_training()
        except subprocess.TimeoutExpired:
            print("Training timed out — reverting.")
            shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)
            continue

        if returncode != 0:
            print(f"Training failed (exit {returncode}) — reverting.")
            shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)
            continue

        # Parse results
        new_val_bpb = metrics.parse_val_bpb(training_output)
        if new_val_bpb is None:
            print("Could not parse val_bpb from output — reverting.")
            shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)
            continue

        result = metrics.build_experiment_result(current_val_bpb, new_val_bpb, training_output)
        duration = round(time.time() - exp_start_time, 1)

        # Keep or revert
        if result.kept:
            print(f"KEPT   val_bpb {current_val_bpb:.6f} → {new_val_bpb:.6f} (delta={result.delta:+.6f})")
            current_val_bpb = new_val_bpb
        else:
            print(f"REVERT val_bpb {current_val_bpb:.6f} → {new_val_bpb:.6f} (delta={result.delta:+.6f})")
            shutil.copy(TRAIN_BACKUP, TRAIN_SCRIPT)

        # Log to Firestore
        exp_record = {
            "experiment_number": experiment_number,
            "started_at": exp_start,
            "duration_seconds": duration,
            "change_description": reasoning[:500],
            "change_diff": change_diff,
            "val_bpb_before": result.val_bpb_before,
            "val_bpb_after": result.val_bpb_after,
            "delta": result.delta,
            "kept": result.kept,
            "train_loss": result.train_loss,
            "step_count": result.step_count,
        }
        firestore_logger.log_experiment(run_id=run_id, project_id=gcp_project, **exp_record)
        experiment_history.append(exp_record)

    # Close the run
    firestore_logger.close_run(run_id, experiment_number, project_id=gcp_project)
    print(f"\nRun complete. {experiment_number} experiments. Final val_bpb={current_val_bpb:.6f}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    run()
