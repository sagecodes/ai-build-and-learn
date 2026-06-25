"""Classic ML experiment tracking with MLflow + Flyte.

Trains multiple sklearn classifiers on the Iris dataset, logs
hyperparameters, metrics, and the trained model to MLflow. Compare
runs in the MLflow UI.

Run remote:
    flyte run ml_training.py train_and_compare

Run local:
    flyte run --local ml_training.py train_and_compare
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import flyte

from config import ml_env, MLFLOW_TRACKING_URI


@dataclass
class TrainResult:
    model_name: str
    params: dict
    accuracy: float
    f1: float
    run_id: str


@ml_env.task(report=True)
def train_model(model_type: str = "random_forest", n_estimators: int = 100, max_depth: int = 5) -> TrainResult:
    """Train a single sklearn model and log everything to MLflow."""
    import mlflow
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("classic-ml-iris")

    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build the model
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

    with mlflow.start_run(run_name=f"{model_type}") as run:
        mlflow.log_params(params)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Log the model
        mlflow.sklearn.log_model(model, "model")

        print(f"[ml] {model_type}: accuracy={acc:.4f}, f1={f1:.4f}")

        # Flyte report
        report = f"""
        <h2>{model_type}</h2>
        <table>
            <tr><td>Accuracy</td><td>{acc:.4f}</td></tr>
            <tr><td>F1 Score</td><td>{f1:.4f}</td></tr>
            <tr><td>MLflow Run ID</td><td>{run.info.run_id}</td></tr>
        </table>
        """
        flyte.report.log(report)

        return TrainResult(
            model_name=model_type,
            params=params,
            accuracy=acc,
            f1=f1,
            run_id=run.info.run_id,
        )


@ml_env.task(report=True)
async def train_and_compare() -> str:
    """Train multiple models and compare them. Each train_model call dispatches as a separate Flyte task."""
    # Dispatch all training tasks (they show up as child tasks in the Flyte DAG)
    rf = await train_model(model_type="random_forest", n_estimators=100, max_depth=5)
    gb = await train_model(model_type="gradient_boosting", n_estimators=100, max_depth=3)
    lr = await train_model(model_type="logistic_regression")
    results = [rf, gb, lr]

    best = max(results, key=lambda r: r.f1)

    rows = "".join(
        f"<tr><td>{r.model_name}</td><td>{r.accuracy:.4f}</td><td>{r.f1:.4f}</td><td>{r.run_id}</td></tr>"
        for r in sorted(results, key=lambda r: r.f1, reverse=True)
    )
    report = f"""
    <h2>Model Comparison</h2>
    <table>
        <tr><th>Model</th><th>Accuracy</th><th>F1</th><th>Run ID</th></tr>
        {rows}
    </table>
    <p>Best model: <b>{best.model_name}</b> (F1={best.f1:.4f})</p>
    """
    await flyte.report.log.aio(report)

    summary = json.dumps({
        "best_model": best.model_name,
        "best_f1": best.f1,
        "best_accuracy": best.accuracy,
        "all_results": [{"model": r.model_name, "f1": r.f1, "accuracy": r.accuracy} for r in results],
    }, indent=2)
    print(f"[ml] comparison:\n{summary}")
    return summary
