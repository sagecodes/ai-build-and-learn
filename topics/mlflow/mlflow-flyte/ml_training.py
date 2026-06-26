"""Classic ML as a proper Flyte workflow, tracked in MLflow.

Instead of one monolithic task, the pipeline is split into the steps you'd
normally model in Flyte — each a separate task, visible in the Flyte UI with
its own compute, logs, and report:

    prepare_data ──┬─> train_model(rf)  ─> evaluate_model(rf) ──┐
                   ├─> train_model(gb)  ─> evaluate_model(gb) ──┤─> ml_pipeline
                   └─> train_model(lr)  ─> evaluate_model(lr) ──┘  (compare)

The split is also the demo's point — a side-by-side of what each tool records:
  - Flyte  tracks the DAG: data → train → eval, per-task compute/logs/reports.
  - MLflow tracks the experiment: params, metrics, model, and the full
    mlflow.evaluate() report. train + eval share ONE MLflow run (eval resumes
    the run by id), so 2 Flyte tasks map to 1 MLflow run.
  - prepare_data touches no MLflow at all — Flyte sees it, MLflow never does.

Run remote:
    flyte run ml_training.py ml_pipeline

Run local:
    flyte run --local ml_training.py ml_pipeline
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import flyte
import pandas as pd

from config import ml_env, MLFLOW_TRACKING_URI

EXPERIMENT = "classic-ml-iris"
LABEL = "label"
SPLIT = "split"


@dataclass
class TrainResult:
    model_name: str
    params: dict
    accuracy: float
    f1: float
    run_id: str
    model_uri: str


@dataclass
class EvalResult:
    model_name: str
    run_id: str
    metrics: dict  # curated subset of mlflow.evaluate() metrics


def _feature_cols(data: pd.DataFrame) -> list[str]:
    return [c for c in data.columns if c not in (LABEL, SPLIT)]


# ── Task 1: data ─────────────────────────────────────────────────────────────
@ml_env.task(report=True)
def prepare_data(test_size: float = 0.2, seed: int = 42) -> pd.DataFrame:
    """Load Iris and split it. Returns one DataFrame with a `split` column.

    Pure data prep — no MLflow. This is the Flyte-vs-MLflow contrast: Flyte
    records this task (and hands the dataset to downstream tasks); MLflow has
    no idea it ran.
    """
    import pandas as pd
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df[LABEL] = iris.target

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[LABEL]
    )
    train_df = train_df.assign(**{SPLIT: "train"})
    test_df = test_df.assign(**{SPLIT: "test"})
    data = pd.concat([train_df, test_df], ignore_index=True)

    counts = data[SPLIT].value_counts().to_dict()
    flyte.report.log(
        f"<h2>Dataset: Iris</h2>"
        f"<p>{len(data)} rows, {len(_feature_cols(data))} features, "
        f"{data[LABEL].nunique()} classes</p>"
        f"<table><tr><th>Split</th><th>Rows</th></tr>"
        f"<tr><td>train</td><td>{counts.get('train', 0)}</td></tr>"
        f"<tr><td>test</td><td>{counts.get('test', 0)}</td></tr></table>"
    )
    print(f"[data] prepared {len(data)} rows ({counts})")
    return data


# ── Task 2: train ────────────────────────────────────────────────────────────
@ml_env.task(report=True)
def train_model(
    data: pd.DataFrame,
    model_type: str = "random_forest",
    n_estimators: int = 100,
    max_depth: int = 5,
) -> TrainResult:
    """Train one sklearn model on the train split; log params/metrics/model to MLflow."""
    import mlflow
    from sklearn.metrics import accuracy_score, f1_score

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    feats = _feature_cols(data)
    train = data[data[SPLIT] == "train"]
    test = data[data[SPLIT] == "test"]
    X_train, y_train = train[feats].values, train[LABEL].values
    X_test, y_test = test[feats].values, test[LABEL].values

    if model_type == "random_forest":
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        params = {"model_type": model_type, "n_estimators": n_estimators, "max_depth": max_depth}
    elif model_type == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        params = {"model_type": model_type, "n_estimators": n_estimators, "max_depth": max_depth}
    elif model_type == "logistic_regression":
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=200, random_state=42)
        params = {"model_type": model_type, "max_iter": 200}
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    with mlflow.start_run(run_name=model_type) as run:
        mlflow.log_params(params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        model_info = mlflow.sklearn.log_model(model, name="model")

        print(f"[train] {model_type}: accuracy={acc:.4f}, f1={f1:.4f} run={run.info.run_id}")
        flyte.report.log(
            f"<h2>Train: {model_type}</h2>"
            f"<table>"
            f"<tr><td>Accuracy</td><td>{acc:.4f}</td></tr>"
            f"<tr><td>F1 Score</td><td>{f1:.4f}</td></tr>"
            f"<tr><td>MLflow Run ID</td><td>{run.info.run_id}</td></tr>"
            f"</table>"
        )
        return TrainResult(
            model_name=model_type,
            params=params,
            accuracy=acc,
            f1=f1,
            run_id=run.info.run_id,
            model_uri=model_info.model_uri,
        )


# ── Task 3: eval ─────────────────────────────────────────────────────────────
@ml_env.task(report=True)
def evaluate_model(data: pd.DataFrame, trained: TrainResult) -> EvalResult:
    """Evaluate the trained model on the test split with mlflow.evaluate().

    Resumes the SAME MLflow run the train task created (by run_id), so the
    confusion matrix / ROC / SHAP report attaches to that run — two Flyte
    tasks, one MLflow run.
    """
    import mlflow

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    feats = _feature_cols(data)
    test = data[data[SPLIT] == "test"]
    eval_df = test[feats + [LABEL]].copy()

    with mlflow.start_run(run_id=trained.run_id):
        result = mlflow.evaluate(
            trained.model_uri,
            data=eval_df,
            targets=LABEL,
            model_type="classifier",
            evaluators=["default"],
        )

    keep = ["accuracy_score", "f1_score", "precision_score", "recall_score", "roc_auc", "log_loss"]
    metrics = {k: float(result.metrics[k]) for k in keep if k in result.metrics}

    print(f"[eval] {trained.model_name}: {metrics}")
    rows = "".join(f"<tr><td>{k}</td><td>{v:.4f}</td></tr>" for k, v in metrics.items())
    flyte.report.log(
        f"<h2>Eval: {trained.model_name}</h2>"
        f"<p>Full report (confusion matrix, ROC/PR, SHAP) logged to MLflow run "
        f"{trained.run_id}.</p>"
        f"<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>"
    )
    return EvalResult(model_name=trained.model_name, run_id=trained.run_id, metrics=metrics)


# ── Orchestrator ─────────────────────────────────────────────────────────────
@ml_env.task(report=True)
async def ml_pipeline(test_size: float = 0.2) -> str:
    """data → (train → eval) per model → compare. Each step is its own Flyte task."""
    import asyncio

    data = await prepare_data.aio(test_size=test_size)

    async def train_then_eval(model_type: str, **kw) -> tuple[TrainResult, EvalResult]:
        trained = await train_model.aio(data=data, model_type=model_type, **kw)
        evaluated = await evaluate_model.aio(data=data, trained=trained)
        return trained, evaluated

    pairs = await asyncio.gather(
        train_then_eval("random_forest", n_estimators=100, max_depth=5),
        train_then_eval("gradient_boosting", n_estimators=100, max_depth=3),
        train_then_eval("logistic_regression"),
    )

    best = max(pairs, key=lambda p: p[0].f1)
    rows = "".join(
        f"<tr><td>{tr.model_name}</td><td>{tr.accuracy:.4f}</td><td>{tr.f1:.4f}</td>"
        f"<td>{ev.metrics.get('roc_auc', float('nan')):.4f}</td>"
        f"<td>{ev.metrics.get('log_loss', float('nan')):.4f}</td><td>{tr.run_id}</td></tr>"
        for tr, ev in sorted(pairs, key=lambda p: p[0].f1, reverse=True)
    )
    await flyte.report.log.aio(
        f"<h2>Model Comparison (train + eval)</h2>"
        f"<table>"
        f"<tr><th>Model</th><th>Accuracy</th><th>F1</th><th>ROC AUC</th><th>Log Loss</th><th>MLflow Run</th></tr>"
        f"{rows}</table>"
        f"<p>Best model: <b>{best[0].model_name}</b> (F1={best[0].f1:.4f})</p>"
        f"<p><small>Flyte ran {1 + 2 * len(pairs)} tasks; MLflow recorded {len(pairs)} runs "
        f"(train + eval share one run each).</small></p>"
    )

    summary = json.dumps({
        "best_model": best[0].model_name,
        "best_f1": best[0].f1,
        "results": [
            {"model": tr.model_name, "accuracy": tr.accuracy, "f1": tr.f1, **ev.metrics}
            for tr, ev in pairs
        ],
    }, indent=2)
    print(f"[pipeline] comparison:\n{summary}")
    return summary
