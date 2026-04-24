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
from datetime import datetime, timezone

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
    run_training,
    apply_and_train,
    evaluate_and_log,
    propose_change,
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
async def propose_change_task(
    experiment_number: int,
    current_val_bpb: float,
    experiment_history: list[dict],
) -> dict:
    """Call Claude and parse the proposed train.py change."""
    client  = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    program = read_file(PROGRAM_MD)

    print(f"\n{'='*60}")
    print(f"Experiment {experiment_number} | val_bpb={current_val_bpb:.6f}")
    print(f"{'='*60}")

    reasoning, new_train_py, current_train = propose_change(client, program, experiment_history)
    print(f"Change proposed: {reasoning[:200]}")

    return {
        "reasoning":      reasoning,
        "new_train_py":   new_train_py,
        "current_train":  current_train,
        "exp_start":      datetime.now(timezone.utc).isoformat(),
        "exp_start_time": time.time(),
    }


@env.task
async def run_training_task(proposal: dict) -> dict:
    """Apply the proposed change to train.py and run training."""
    training_output, returncode, change_diff = apply_and_train(
        proposal["new_train_py"], proposal["current_train"]
    )
    return {**proposal, "training_output": training_output, "returncode": returncode, "change_diff": change_diff}


@env.task
async def evaluate_task(
    training_data: dict,
    current_val_bpb: float,
    experiment_number: int,
    run_id: str,
    dry_run: bool,
) -> dict:
    """Keep/revert decision and Firestore logging."""
    gcp_project = os.getenv("GCP_PROJECT")
    outcome = evaluate_and_log(
        current_val_bpb=current_val_bpb,
        training_output=training_data["training_output"],
        returncode=training_data["returncode"],
        change_diff=training_data["change_diff"],
        reasoning=training_data["reasoning"],
        run_id=run_id if run_id else None,
        experiment_number=experiment_number,
        exp_start=training_data["exp_start"],
        exp_start_time=training_data["exp_start_time"],
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

    # Check for checkpoint before creating a new Firestore run
    prior = checkpoint.load()

    if dry_run:
        run_id = ""
        print("Run started: (dry run — no Firestore)")
    elif prior is not None and prior.get("run_id"):
        # Resume existing run — do not create a new Firestore run
        run_id = prior["run_id"]
    else:
        try:
            run_id = firestore_logger.create_run(config=initial_config, project_id=gcp_project)
        except Exception as e:
            print(f"WARNING: Firestore create_run failed: {e}. Continuing without logging.")
            run_id = ""
    print(f"Run started: {run_id}")

    deadline = time.time() + RUN_SECONDS

    flyte.init(local_persistence=True)

    # Resume from checkpoint if one exists (e.g. after a crash or credit pause)
    if prior is not None:
        print(f"Resuming from checkpoint saved at {prior['saved_at']}")
        print(f"  experiment_number={prior['experiment_number']}  val_bpb={prior['current_val_bpb']:.6f}")
        current_val_bpb    = prior["current_val_bpb"]
        experiment_number  = prior["experiment_number"]
        experiment_history = prior["experiment_history"]
    else:
        # Baseline measurement as a Flyte task
        # local_persistence=True enables TUI visibility via `flyte start tui`
        baseline_run = flyte.run(measure_baseline)
        current_val_bpb = baseline_run.outputs().o0
        experiment_number  = 0
        experiment_history = []

    while time.time() < deadline:
        experiment_number += 1

        try:
            propose_run = flyte.run(
                propose_change_task,
                experiment_number=experiment_number,
                current_val_bpb=current_val_bpb,
                experiment_history=experiment_history,
            )
            proposal = propose_run.outputs().o0
        except Exception as e:
            print(f"Experiment {experiment_number}: proposal failed — {e}. Skipping.")
            continue

        try:
            train_run     = flyte.run(run_training_task, proposal=proposal)
            training_data = train_run.outputs().o0
        except Exception as e:
            print(f"Experiment {experiment_number}: training task failed — {e}. Skipping.")
            continue

        try:
            eval_run = flyte.run(
                evaluate_task,
                training_data=training_data,
                current_val_bpb=current_val_bpb,
                experiment_number=experiment_number,
                run_id=run_id,
                dry_run=dry_run,
            )
            result = eval_run.outputs().o0
        except Exception as e:
            print(f"Experiment {experiment_number}: evaluate task failed — {e}. Skipping.")
            continue

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
