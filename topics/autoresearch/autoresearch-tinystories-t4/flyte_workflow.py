"""
flyte_workflow.py — Flyte 2 orchestration for AutoResearch.

Runs the same experiment logic as agent.py but orchestrated as Flyte tasks,
giving per-experiment visibility in the Flyte TUI.

Usage (on T4 with venv activated):
    python flyte_workflow.py               # production run
    python flyte_workflow.py --dry-run     # smoke test — no Firestore writes

Environment variables (same as agent.py):
    ANTHROPIC_API_KEY   — required
    GCP_PROJECT         — required
    RUN_HOURS           — optional (default: 8)
"""

import argparse
import os
import time

import anthropic
import flyte
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
    run_training,
)

# ── Config ────────────────────────────────────────────────────────────────────

RUN_HOURS   = float(os.getenv("RUN_HOURS", "8"))
RUN_SECONDS = RUN_HOURS * 3600

# ── Flyte task environment (local execution on T4, no container needed) ───────

env = flyte.TaskEnvironment(name="autoresearch")

# ── Flyte tasks ───────────────────────────────────────────────────────────────

@env.task
async def measure_baseline() -> float:
    """Measure the baseline val_bpb before any changes."""
    print("Measuring baseline val_bpb...")
    baseline_output, _ = run_training()
    val_bpb = metrics.parse_val_bpb(baseline_output)
    if val_bpb is None:
        raise RuntimeError("Could not parse baseline val_bpb. Check that train.py runs correctly.")
    print(f"Baseline val_bpb={val_bpb:.6f}")
    return val_bpb


@env.task
async def run_experiment_task(
    experiment_number: int,
    current_val_bpb: float,
    experiment_history: list[dict],
    run_id: str,
    dry_run: bool,
) -> dict:
    """
    Run one experiment cycle as a Flyte task.

    Returns a dict with keys: skipped, new_val_bpb, exp_record.
    """
    client      = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    program     = read_file(PROGRAM_MD)
    gcp_project = os.getenv("GCP_PROJECT")

    outcome = run_single_experiment(
        client=client,
        program=program,
        run_id=run_id if run_id else None,
        experiment_number=experiment_number,
        current_val_bpb=current_val_bpb,
        experiment_history=experiment_history,
        gcp_project=gcp_project,
        dry_run=dry_run,
    )

    return {
        "skipped":     outcome.skipped,
        "new_val_bpb": outcome.new_val_bpb,
        "exp_record":  outcome.exp_record,
    }


# ── Main workflow ─────────────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    """Run the AutoResearch Flyte workflow for RUN_HOURS hours."""
    if dry_run:
        print("DRY RUN — Firestore writes disabled.")

    gcp_project = os.getenv("GCP_PROJECT")

    initial_config = {
        "model":        CLAUDE_MODEL,
        "run_hours":    RUN_HOURS,
        "train_script": "train.py",
        "dataset":      "TinyStories",
        "gpu":          "T4",
        "mode":         "flyte",
    }

    if dry_run:
        run_id = ""
        print("Run started: (dry run — no Firestore)")
    else:
        try:
            run_id = firestore_logger.create_run(config=initial_config, project_id=gcp_project)
        except Exception as e:
            print(f"WARNING: Firestore create_run failed: {e}. Continuing without logging.")
            run_id = ""
        print(f"Run started: {run_id}")

    deadline = time.time() + RUN_SECONDS

    # Baseline measurement as a Flyte task
    flyte.init()
    baseline_run = flyte.run(measure_baseline)
    current_val_bpb = baseline_run.outputs().o0

    experiment_number  = 0
    experiment_history = []

    while time.time() < deadline:
        experiment_number += 1

        exp_run = flyte.run(
            run_experiment_task,
            experiment_number=experiment_number,
            current_val_bpb=current_val_bpb,
            experiment_history=experiment_history,
            run_id=run_id,
            dry_run=dry_run,
        )
        result = exp_run.outputs().o0  # returns the dict

        if result["skipped"]:
            continue

        current_val_bpb = result["new_val_bpb"]
        experiment_history.append(result["exp_record"])
        checkpoint.save(run_id or None, current_val_bpb, experiment_number, experiment_history)

    # Close run
    if run_id and not dry_run:
        try:
            firestore_logger.close_run(run_id, experiment_number, project_id=gcp_project)
        except Exception as e:
            print(f"WARNING: Firestore close_run failed: {e}.")

    checkpoint.clear()
    print(f"\nFlyte run complete. {experiment_number} experiments. Final val_bpb={current_val_bpb:.6f}")
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
