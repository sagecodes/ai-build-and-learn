"""
firestore_logger.py — Firestore read/write for AutoResearch experiment tracking.

Responsibilities:
  - Create and manage run documents (one per overnight session)
  - Log individual experiment results as subcollection documents
  - Provide query functions for the Gradio dashboard

Schema:
  runs/{run_id}/
    started_at        : ISO timestamp
    config            : dict of training hyperparameters
    ended_at          : ISO timestamp (written on run completion)
    total_experiments : int (written on run completion)

  runs/{run_id}/experiments/{exp_id}/
    experiment_number : int
    started_at        : ISO timestamp
    duration_seconds  : float
    change_description: str  — human-readable summary of the change
    change_diff       : str  — unified diff of train.py before/after
    val_bpb_before    : float
    val_bpb_after     : float
    delta             : float — val_bpb_after - val_bpb_before (negative = improvement)
    kept              : bool
    train_loss        : float | None
    step_count        : int | None

Authentication:
  - On GCP instance: uses VM built-in service account automatically
  - Locally: set GOOGLE_APPLICATION_CREDENTIALS to path of reader key JSON
"""

import os
from datetime import datetime, timezone
from typing import Optional

from google.cloud import firestore

# ── Constants ─────────────────────────────────────────────────────────────────

_COLLECTION_RUNS = "runs"
_COLLECTION_EXPERIMENTS = "experiments"


# ── Client ────────────────────────────────────────────────────────────────────

def _get_client(project_id: Optional[str] = None) -> firestore.Client:
    """
    Return an authenticated Firestore client.

    project_id defaults to the GCP_PROJECT env var, then to ADC project.
    database defaults to the FIRESTORE_DATABASE env var, then '(default)'.
    On a GCP VM the service account is picked up automatically via ADC.
    Locally, use gcloud auth application-default login.
    """
    project = project_id or os.getenv("GCP_PROJECT")
    database = os.getenv("FIRESTORE_DATABASE", "(default)")
    return firestore.Client(project=project, database=database)


# ── Run lifecycle ─────────────────────────────────────────────────────────────

def create_run(config: dict, project_id: Optional[str] = None) -> str:
    """
    Create a new run document and return its run_id.

    Args:
        config: Training hyperparameters snapshot (DEPTH, ASPECT_RATIO, etc.)
        project_id: GCP project ID. Defaults to GCP_PROJECT env var.

    Returns:
        run_id: str — Firestore document ID for this run.
    """
    db = _get_client(project_id)
    run_ref = db.collection(_COLLECTION_RUNS).document()
    run_ref.set({
        "started_at": _now(),
        "config": config,
        "ended_at": None,
        "total_experiments": 0,
    })
    return run_ref.id


def close_run(run_id: str, total_experiments: int, project_id: Optional[str] = None) -> None:
    """
    Mark a run as complete with end timestamp and total experiment count.

    Args:
        run_id: The run document ID returned by create_run().
        total_experiments: Total number of experiments run in this session.
        project_id: GCP project ID. Defaults to GCP_PROJECT env var.
    """
    db = _get_client(project_id)
    db.collection(_COLLECTION_RUNS).document(run_id).update({
        "ended_at": _now(),
        "total_experiments": total_experiments,
    })


# ── Experiment logging ────────────────────────────────────────────────────────

def log_experiment(
    run_id: str,
    experiment_number: int,
    started_at: str,
    duration_seconds: float,
    change_description: str,
    change_diff: str,
    val_bpb_before: float,
    val_bpb_after: float,
    kept: bool,
    train_loss: Optional[float] = None,
    step_count: Optional[int] = None,
    project_id: Optional[str] = None,
) -> str:
    """
    Log one experiment result under runs/{run_id}/experiments/.

    Args:
        run_id            : Run document ID from create_run().
        experiment_number : Sequential experiment index (1-based).
        started_at        : ISO timestamp when training started.
        duration_seconds  : Wall-clock seconds the training ran.
        change_description: Human-readable summary of what changed.
        change_diff       : Unified diff string of train.py modifications.
        val_bpb_before    : Validation bpb before this experiment.
        val_bpb_after     : Validation bpb after this experiment.
        kept              : True if change improved val_bpb and was kept.
        train_loss        : Final training loss (optional).
        step_count        : Number of optimizer steps completed (optional).
        project_id        : GCP project ID. Defaults to GCP_PROJECT env var.

    Returns:
        exp_id: str — Firestore document ID for this experiment.
    """
    db = _get_client(project_id)
    exp_ref = (
        db.collection(_COLLECTION_RUNS)
        .document(run_id)
        .collection(_COLLECTION_EXPERIMENTS)
        .document()
    )
    exp_ref.set({
        "experiment_number": experiment_number,
        "started_at": started_at,
        "duration_seconds": duration_seconds,
        "change_description": change_description,
        "change_diff": change_diff,
        "val_bpb_before": val_bpb_before,
        "val_bpb_after": val_bpb_after,
        "delta": round(val_bpb_after - val_bpb_before, 6),
        "kept": kept,
        "train_loss": train_loss,
        "step_count": step_count,
    })
    return exp_ref.id


# ── Dashboard queries ─────────────────────────────────────────────────────────

def get_latest_run_id(project_id: Optional[str] = None) -> Optional[str]:
    """
    Return the document ID of the most recently started run.

    Returns None if no runs exist yet.
    """
    db = _get_client(project_id)
    runs = (
        db.collection(_COLLECTION_RUNS)
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for run in runs:
        return run.id
    return None


def get_run(run_id: str, project_id: Optional[str] = None) -> Optional[dict]:
    """
    Return the run document as a dict, or None if not found.
    """
    db = _get_client(project_id)
    doc = db.collection(_COLLECTION_RUNS).document(run_id).get()
    if doc.exists:
        return {"id": doc.id, **doc.to_dict()}
    return None


def get_experiments(run_id: str, project_id: Optional[str] = None) -> list[dict]:
    """
    Return all experiments for a run, ordered by experiment_number ascending.

    Each dict includes the Firestore document ID as 'id'.
    """
    db = _get_client(project_id)
    docs = (
        db.collection(_COLLECTION_RUNS)
        .document(run_id)
        .collection(_COLLECTION_EXPERIMENTS)
        .order_by("experiment_number")
        .stream()
    )
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


def get_kept_experiments(run_id: str, project_id: Optional[str] = None) -> list[dict]:
    """
    Return only the experiments where the change was kept (kept == True).
    """
    db = _get_client(project_id)
    docs = (
        db.collection(_COLLECTION_RUNS)
        .document(run_id)
        .collection(_COLLECTION_EXPERIMENTS)
        .where("kept", "==", True)
        .order_by("experiment_number")
        .stream()
    )
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


def get_best_experiment(run_id: str, project_id: Optional[str] = None) -> Optional[dict]:
    """
    Return the single experiment with the largest val_bpb improvement (most negative delta).

    Returns None if no experiments exist.
    """
    db = _get_client(project_id)
    docs = (
        db.collection(_COLLECTION_RUNS)
        .document(run_id)
        .collection(_COLLECTION_EXPERIMENTS)
        .where("kept", "==", True)
        .order_by("delta")
        .limit(1)
        .stream()
    )
    for doc in docs:
        return {"id": doc.id, **doc.to_dict()}
    return None


def list_runs(limit: int = 10, project_id: Optional[str] = None) -> list[dict]:
    """
    Return the most recent runs, newest first.

    Useful for the dashboard run selector dropdown.
    """
    db = _get_client(project_id)
    docs = (
        db.collection(_COLLECTION_RUNS)
        .order_by("started_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    return [{"id": doc.id, **doc.to_dict()} for doc in docs]


# ── Internal helpers ──────────────────────────────────────────────────────────

def _now() -> str:
    """Return current UTC time as an ISO 8601 string."""
    return datetime.now(timezone.utc).isoformat()
