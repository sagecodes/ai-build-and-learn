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
    python agent.py               # production run
    python agent.py --dry-run     # smoke test — no Firestore writes

Environment variables:
    ANTHROPIC_API_KEY   — required, Claude API key
    GCP_PROJECT         — required, GCP project ID for Firestore
    RUN_HOURS           — optional, overnight budget in hours (default: 8)
"""

import argparse
import os
import time

import anthropic
from dotenv import load_dotenv

load_dotenv()

import checkpoint
import firestore_logger
import metrics
from core import (
    CLAUDE_MODEL,
    PROGRAM_MD,
    read_file,
    run_single_experiment,
)

# ── Config ────────────────────────────────────────────────────────────────────

RUN_HOURS   = float(os.getenv("RUN_HOURS", "8"))
RUN_SECONDS = RUN_HOURS * 3600


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """Run the AutoResearch agent loop for RUN_HOURS hours."""
    if dry_run:
        print("DRY RUN — Firestore writes disabled. No data will be persisted.")

    client      = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    program     = read_file(PROGRAM_MD)
    gcp_project = os.getenv("GCP_PROJECT")

    initial_config = {
        "model":        CLAUDE_MODEL,
        "run_hours":    RUN_HOURS,
        "train_script": "train.py",
        "dataset":      "TinyStories",
        "gpu":          "T4",
        "mode":         "agent",
    }

    if dry_run:
        run_id = None
        print("Run started: (dry run — no Firestore)")
    else:
        try:
            run_id = firestore_logger.create_run(config=initial_config, project_id=gcp_project)
        except Exception as e:
            print(f"WARNING: Firestore create_run failed: {e}. Continuing without Firestore logging.")
            run_id = None
        print(f"Run started: {run_id}")

    deadline = time.time() + RUN_SECONDS

    # Resume from checkpoint if one exists (e.g. after a crash)
    prior = checkpoint.load()
    if prior is not None:
        print(f"Resuming from checkpoint saved at {prior['saved_at']}")
        print(f"  experiment_number={prior['experiment_number']}  val_bpb={prior['current_val_bpb']:.6f}")
        current_val_bpb    = prior["current_val_bpb"]
        experiment_number  = prior["experiment_number"]
        experiment_history = prior["experiment_history"]
        if run_id is None and prior.get("run_id"):
            run_id = prior["run_id"]
    else:
        experiment_number  = 0
        experiment_history = []
        print("Measuring baseline val_bpb...")
        from core import run_training
        baseline_output, _ = run_training()
        current_val_bpb = metrics.parse_val_bpb(baseline_output)
        if current_val_bpb is None:
            print("ERROR: Could not parse baseline val_bpb. Check that train.py runs correctly.")
            return
        print(f"Baseline val_bpb={current_val_bpb:.6f}")

    while time.time() < deadline:
        experiment_number += 1

        outcome = run_single_experiment(
            client=client,
            program=program,
            run_id=run_id,
            experiment_number=experiment_number,
            current_val_bpb=current_val_bpb,
            experiment_history=experiment_history,
            gcp_project=gcp_project,
            dry_run=dry_run,
        )

        if outcome.skipped:
            continue

        current_val_bpb = outcome.new_val_bpb
        experiment_history.append(outcome.exp_record)
        checkpoint.save(run_id, current_val_bpb, experiment_number, experiment_history)

    # Close the run
    if run_id is not None and not dry_run:
        try:
            firestore_logger.close_run(run_id, experiment_number, project_id=gcp_project)
        except Exception as e:
            print(f"WARNING: Firestore close_run failed: {e}.")

    checkpoint.clear()
    print(f"\nRun complete. {experiment_number} experiments. Final val_bpb={current_val_bpb:.6f}")
    print(f"Run ID: {run_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip all Firestore writes")
    parser.add_argument("--reset", action="store_true", help="Reset train.py to git baseline before starting")
    args = parser.parse_args()

    if args.reset:
        import subprocess as _sp
        print("Resetting train.py to git baseline...")
        _sp.run(["git", "checkout", "HEAD", "--", "train.py"], check=True)
        print("train.py reset.")

    run(dry_run=args.dry_run)
