# MLflow on Flyte: ML Experiment Tracking & LLM Agent Tracing

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event covers MLflow end-to-end: classic ML experiment tracking and LLM agent tracing, all running on Flyte with a self-hosted MLflow server.

## What is MLflow?

[MLflow](https://mlflow.org/) is an open-source platform for managing the full ML lifecycle. It's been around since 2018 and has grown from experiment tracking into a comprehensive platform covering:

- **Experiment tracking** — log parameters, metrics, and artifacts across training runs
- **Model registry** — version, stage, and serve models
- **LLM tracing** — auto-capture prompts, completions, tool calls, and latencies from LangChain, OpenAI, and more
- **Evaluation** — built-in metrics for both classic ML and LLM-as-a-judge scoring

MLflow uses a familiar `log_param` / `log_metric` API for classic ML, and OpenTelemetry-based autologging for LLM frameworks.

## What we're building

Two demos, one MLflow server:

**Part 1: Classic ML** — Train three sklearn classifiers (Random Forest, Gradient Boosting, Logistic Regression) on Iris, log params/metrics/models to MLflow, and compare runs in the UI.

**Part 2: LLM Agent Tracing** — Trace a LangGraph research agent with `mlflow.langchain.autolog()`. Every LLM call, tool use, and graph step is captured automatically.

## Files

| File | Purpose |
| --- | --- |
| `mlflow_app.py` | Self-hosted MLflow server (Flyte app) |
| `ml_training.py` | Classic ML: train sklearn models, log to MLflow |
| `agent_tracing.py` | LLM tracing: LangGraph agent with MLflow autolog |
| `config.py` | Flyte environments, endpoints, constants |

## Prerequisites

- A running Flyte 2 devbox (`flyte get project` to confirm)
- [uv](https://docs.astral.sh/uv/)
- API keys: [OpenAI](https://platform.openai.com/api-keys), [Tavily](https://app.tavily.com/) (for Part 2 only)

```bash
flyte create secret OPENAI_API_KEY
flyte create secret TAVILY_API_KEY
```

## Step 1: Set up the local environment

```bash
cd topics/mlflow/mlflow-flyte
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

For local runs, fill in `.env` with your API keys and the MLflow tracking URI.

## Step 2: Deploy the MLflow server

```bash
python mlflow_app.py
```

Open the MLflow UI at: `http://mlflow-server-flytesnacks-development.localhost:30081`

## Step 3: Run classic ML experiment tracking

```bash
# Remote (on the cluster)
flyte run ml_training.py train_and_compare

# Local
flyte run --local ml_training.py train_and_compare
```

Open the MLflow UI and check the `classic-ml-iris` experiment. You'll see three runs (one per model) with logged parameters, metrics, and artifacts. Compare them side-by-side.

## Step 4: Run LLM agent tracing

```bash
# Remote
flyte run agent_tracing.py traced_research --query "What is MLflow and how does it compare to other ML tools?"

# Local
flyte run --local agent_tracing.py traced_research --query "What is MLflow?"
```

Check the `llm-agent-tracing` experiment in the MLflow UI. Click into the run to see the full trace: every LLM call, tool invocation, and graph step with inputs/outputs.

## Teardown

```bash
flyte delete app mlflow-server
```

## Links

- MLflow GitHub: https://github.com/mlflow/mlflow
- MLflow docs: https://mlflow.org/docs/latest/index.html
- MLflow LLM tracing: https://mlflow.org/docs/latest/llms/tracing/index.html
