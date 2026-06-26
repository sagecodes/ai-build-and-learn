# MLflow on Flyte: ML Experiment Tracking, LLM Tracing & LLM-as-a-Judge

Welcome to AI Build & Learn, a weekly AI engineering stream where we pick a new topic and learn by building together.

This event covers MLflow end-to-end: classic ML experiment tracking, LLM agent tracing, and LLM-as-a-judge evaluation — all running on Flyte with a self-hosted MLflow server.

## What is MLflow?

[MLflow](https://mlflow.org/) is an open-source platform for managing the full ML lifecycle. It's been around since 2018 and has grown from experiment tracking into a comprehensive platform covering:

- **Experiment tracking** — log parameters, metrics, and artifacts across training runs
- **Model registry** — version, stage, and serve models
- **Model evaluation** — `mlflow.evaluate()` auto-generates rich reports (confusion matrix, ROC, SHAP) for classic ML
- **LLM tracing** — auto-capture prompts, completions, tool calls, and latencies from LangChain, OpenAI, and more
- **Prompt registry** — version and manage prompts as first-class, tracked entities (MLflow 3)
- **LLM-as-a-judge** — `mlflow.genai.evaluate()` scores GenAI outputs with built-in and custom LLM judges (MLflow 3)

MLflow uses a familiar `log_param` / `log_metric` API for classic ML, and OpenTelemetry-based autologging for LLM frameworks.

## What we're building

Three demos, one MLflow server:

**Part 1: Classic ML** — Train three sklearn classifiers (Random Forest, Gradient Boosting, Logistic Regression) on Iris, log params/metrics/models, and run `mlflow.evaluate()` for a full evaluation report (confusion matrix, ROC/PR curves, SHAP feature importances). Compare runs in the UI.

**Part 2: LLM Agent Tracing** — Trace a LangGraph research agent with `mlflow.langchain.autolog()`. Every LLM call, tool use, and graph step is captured automatically. The agent's system prompt is pulled from the **MLflow Prompt Registry**, so each run links to a versioned prompt.

**Part 3: LLM-as-a-Judge** — Score answers with `mlflow.genai.evaluate()` using all three judge types: **built-in LLM judges** (`Correctness`, `RelevanceToQuery`, `Guidelines`, `Safety`), a **custom LLM judge** built with `make_judge`, and a **custom code judge** (`@scorer` returning a `Feedback` — deterministic Python, no LLM call). Per-row scores and rationales show up in the MLflow Evaluations UI.

## Feature breakdown — what we tested and why it matters

Each piece below is something we ran and verified against the self-hosted server.

**Experiment tracking** (`log_param` / `log_metric` / `log_text`)
What it is: a record of every run's inputs and outputs in one searchable place.
Why it matters: when you train three models with different hyperparameters, the UI lets you sort and compare them side-by-side instead of scraping numbers out of notebook cells or terminal logs. Reproducibility and "which run was best?" become trivial.

**Model storage as artifacts** (`log_model` + server artifact proxying)
What it is: the trained model is uploaded to the server, not just referenced.
Why it matters: we proved this end-to-end — downloaded a logged model and reloaded it with `load_model` to make predictions. Without artifact proxying the model files silently land in the ephemeral task pod and are lost; with it, any teammate can pull `models:/<id>` and run the exact model. This is the difference between "we tracked that a model existed" and "we can actually serve it."

**`mlflow.evaluate()` for classic ML**
What it is: one call that produces a full evaluation report instead of hand-logging metrics.
Why it matters: from a single line we got a confusion matrix, ROC and precision-recall curves, a calibration curve, per-class metrics, and **SHAP** feature-importance plots — all stored on the run. That's model *understanding* (where does it fail? which features drive it?), not just a top-line accuracy number, with zero extra plotting code.

**LLM tracing** (`mlflow.langchain.autolog()`)
What it is: automatic capture of every LLM call, tool invocation, and graph step as a structured trace.
Why it matters: agents are black boxes when they misbehave. The trace shows the exact prompts, the Tavily search the agent chose to run, the tokens, and the latency at each step — so you can debug *why* an answer was wrong, not just see that it was. One line of setup, no manual instrumentation.

**Prompt Registry** (`register_prompt` / `load_prompt`)
What it is: prompts versioned as first-class entities, with runs linked to the version they used.
Why it matters: prompts are as important as code but usually live as loose strings. Here the agent's system prompt is `research-agent-prompt v1`, and every run records which version produced its output. When you tweak the prompt you get v2 and can compare quality across versions — real prompt change management instead of "I think I changed something last week."

**LLM-as-a-judge evaluation runs** (`mlflow.genai.evaluate()`)
What it is: an evaluation run that scores GenAI outputs with judges and logs per-row results plus aggregate metrics — the GenAI analogue of `mlflow.evaluate()`.
Why it matters: LLM quality can't be measured with accuracy/F1. We scored answers with **all three judge types** and got an aggregate score per judge on the run, plus each judge's written rationale per answer:

- **Built-in LLM judges** — `Correctness`, `RelevanceToQuery`, `Guidelines`, `Safety`. Cover the common quality axes out of the box; the `Guidelines` judge takes a plain-English rule (we used "must be factual, avoid speculation").
- **Custom LLM judge** (`make_judge`) — encode any domain rubric in natural language; we built a `conciseness` judge that returns a boolean so MLflow rolls up a pass-rate.
- **Custom code judge** (`@scorer` → `Feedback`) — deterministic Python, **no LLM call** (fast, free, perfectly repeatable). We built `substantive_answer` (answer must be ≥ 20 words); it scored `0.67` — caught a too-thin answer an LLM judge might wave through.

This turns "the demo looked good" into a repeatable, scored regression test you can run on every prompt or model change. Reach for code judges when a rule is objective (length, format, required keywords, valid JSON) and LLM judges when it needs understanding (correctness, tone, relevance).

Judges passed inline to `evaluate()` score answers but don't appear in the experiment's **Judges** UI — that section lists *registered* scorers. `register_judges` (Step 7) registers them so they're visible and reusable.

## Files

| File | Purpose |
| --- | --- |
| `mlflow_app.py` | Self-hosted MLflow server (Flyte app), with artifact proxying so models are stored on the server |
| `ml_training.py` | Classic ML: train sklearn models, log to MLflow, run `mlflow.evaluate()` |
| `agent_tracing.py` | LLM tracing + evaluation: `traced_research` (agent + prompt registry) and `evaluate_agent` (LLM-as-a-judge) |
| `config.py` | Flyte environments, endpoints, constants |

## The MLflow API by example

Everything below is the actual API used in these demos. Point the client at the server once, then log to it from anywhere.

**Tracking: experiments, runs, params, metrics, artifacts**

```python
import mlflow

mlflow.set_tracking_uri("http://mlflow-server-flytesnacks-development.localhost:30081")
mlflow.set_experiment("classic-ml-iris")

with mlflow.start_run(run_name="random_forest") as run:
    mlflow.log_param("n_estimators", 100)          # one value...
    mlflow.log_params({"max_depth": 5, "seed": 42}) # ...or many
    mlflow.log_metric("accuracy", 0.97)
    mlflow.log_text(answer, "answer.md")            # arbitrary artifact
    mlflow.sklearn.log_model(model, name="model")   # versioned model
    print(run.info.run_id)
```

**Autologging: capture an LLM framework with one line**

```python
import mlflow.langchain
mlflow.langchain.autolog()   # every LLM call, tool use, graph step → a trace
# also: mlflow.sklearn.autolog(), mlflow.openai.autolog(), ...
```

**Model evaluation (classic ML): one call, a full report**

```python
import pandas as pd
eval_df = pd.DataFrame(X_test, columns=feature_names)
eval_df["label"] = y_test
mlflow.evaluate(
    model_info.model_uri, data=eval_df, targets="label",
    model_type="classifier",                 # → confusion matrix, ROC/PR, SHAP
)
```

**Prompt Registry: version prompts as first-class entities**

```python
prompt = mlflow.register_prompt(
    name="research-agent-prompt",
    template="Research this topic and answer.\n\nTopic: {{query}}",
    commit_message="v1",
)
prompt = mlflow.load_prompt("research-agent-prompt")   # latest version
text = prompt.format(query="What is MLflow?")          # render it
```

**LLM-as-a-judge: score GenAI outputs**

Three kinds of judges, all passed to the same `evaluate()` call:

```python
import mlflow.genai
from mlflow.genai.scorers import Correctness, RelevanceToQuery, Guidelines, Safety, scorer
from mlflow.genai.judges import make_judge
from mlflow.entities import Feedback

judge = "openai:/gpt-4.1-nano"   # OSS MLflow needs an explicit judge model

# (2) custom LLM judge — a rubric in natural language
conciseness = make_judge(
    name="conciseness",
    instructions="Question: {{ inputs }}\nAnswer: {{ outputs }}\nReturn true if concise.",
    model=judge, feedback_value_type=bool,
)

# (3) custom CODE judge — deterministic Python, no LLM call
@scorer
def substantive_answer(outputs) -> Feedback:
    ok = len(str(outputs).split()) >= 20
    return Feedback(value=ok, rationale="long enough" if ok else "too thin")

mlflow.genai.evaluate(
    data=[{"inputs": {"query": "What is MLflow?"},
           "expectations": {"expected_facts": ["open-source", "ML lifecycle"]}}],
    scorers=[
        Correctness(model=judge), RelevanceToQuery(model=judge),   # (1) built-in LLM judges
        Guidelines(name="tone", guidelines="Be factual.", model=judge),
        Safety(model=judge),
        conciseness,         # (2) custom LLM judge
        substantive_answer,  # (3) custom code judge
    ],
    predict_fn=lambda query: llm.invoke(query).content,   # system under test
)
```

**Querying back: read runs, models, traces**

```python
runs   = mlflow.search_runs(experiment_ids=[exp_id])      # → DataFrame
models = mlflow.search_logged_models(experiment_ids=[exp_id], output_format="list")
traces = mlflow.search_traces(experiment_ids=[exp_id])
model  = mlflow.sklearn.load_model(f"models:/{models[0].model_id}")  # reload + predict
mlflow.artifacts.download_artifacts(artifact_uri=models[0].artifact_location)
```

## Prerequisites

- A running Flyte 2 devbox (`flyte get project` to confirm)
- [uv](https://docs.astral.sh/uv/)
- API keys: [OpenAI](https://platform.openai.com/api-keys), [Tavily](https://app.tavily.com/) (for Parts 2 & 3)

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

## Step 2: Configure the Flyte CLI

Point the CLI at your devbox. This writes `.flyte/config.yaml` (gitignored, so each clone creates its own):

```bash
flyte create config \
  --endpoint dns:///localhost:30080 \
  --insecure \
  --image-builder local \
  --project flytesnacks \
  --domain development
```

Confirm it works:

```bash
flyte get project
```

## Step 3: Deploy the MLflow server

```bash
python mlflow_app.py
```

Open the MLflow UI at: `http://mlflow-server-flytesnacks-development.localhost:30081`

> The server proxies artifacts (`--serve-artifacts`), so models logged from tasks are actually stored on the server and are downloadable/reloadable from the UI. The store is SQLite and session-scoped — redeploying resets experiments.

## Step 4: Run classic ML experiment tracking

```bash
# Remote (on the cluster)
flyte run ml_training.py train_and_compare

# Local
flyte run --local ml_training.py train_and_compare
```

Open the MLflow UI and check the `classic-ml-iris` experiment. You'll see three runs (one per model) with logged parameters, metrics, the trained model, and a full `mlflow.evaluate()` report — confusion matrix, ROC/PR curves, calibration curve, and SHAP plots. Compare them side-by-side.

## Step 5: Run LLM agent tracing

```bash
# Remote
flyte run agent_tracing.py traced_research --query "What is MLflow and how does it compare to other ML tools?"

# Local
flyte run --local agent_tracing.py traced_research --query "What is MLflow?"
```

Check the `llm-agent-tracing` experiment in the MLflow UI. Click into the run to see the full trace: every LLM call, tool invocation, and graph step with inputs/outputs. The run links to the `research-agent-prompt` version it used (Prompts tab).

## Step 6: Run LLM-as-a-judge evaluation

```bash
# Remote
flyte run agent_tracing.py evaluate_agent

# Local
flyte run --local agent_tracing.py evaluate_agent
```

In the `llm-agent-tracing` experiment, open the **Evaluations** tab. Each answer is scored by six judges spanning all three types — four built-in LLM judges (Correctness, RelevanceToQuery, Guidelines, Safety), a custom LLM judge (conciseness via `make_judge`), and a custom code judge (substantive_answer via `@scorer`) — with each judge's rationale per row, plus aggregate metrics on the run (e.g. `substantive_answer/mean`).

## Step 7: Register judges (populate the Judges UI)

The judges in Step 6 are passed *inline* to `evaluate()` — they score answers but don't appear in the experiment's **Judges** section, which lists *registered* scorers. Register them so they show up and are reusable:

```bash
# Remote
flyte run agent_tracing.py register_judges

# Local
flyte run --local agent_tracing.py register_judges
```

Open the `llm-agent-tracing` experiment → **Judges** tab. You'll see the registered judges (`relevance_to_query`, `safety`, `factual_tone`, `conciseness`).

> **OSS MLflow caveats:**
> - Built-in scorers and `make_judge` (LLM) judges can be registered. Custom `@scorer` **code** judges are **Databricks-only** — self-hosted servers block them (they'd run arbitrary code), so the code judge from Step 6 stays inline.
> - **Automatic/scheduled scoring** (`scorer.start()`, which runs judges on live traces) requires an **MLflow AI Gateway** model, not a raw `openai:/` model. We register the judges but don't start scheduled scoring in this setup.

## Teardown

```bash
flyte delete app mlflow-server
```

## Links

- MLflow GitHub: https://github.com/mlflow/mlflow
- MLflow docs: https://mlflow.org/docs/latest/index.html
- MLflow LLM tracing: https://mlflow.org/docs/latest/llms/tracing/index.html
- MLflow LLM-as-a-judge: https://mlflow.org/docs/latest/genai/eval-monitor/
- MLflow Prompt Registry: https://mlflow.org/docs/latest/genai/prompt-registry/
