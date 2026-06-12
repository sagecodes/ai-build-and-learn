"""Phoenix tracing setup, shared by every pipeline task.

Each pipeline step runs in its own Flyte task pod (its own process), so each pod
must register the exporter and instrument LangChain itself. `setup_tracing()` is
idempotent: the first call in a pod wires it up, later calls are no-ops (Flyte
container reuse can run several tasks in one pod).

Because each task is a separate process, Phoenix shows one trace per task
invocation (all under the same project) rather than a single pipeline-wide trace.
That is the interesting contrast with Flyte's own view: Flyte stitches the whole
DAG together across tasks; Phoenix gives the deep LLM/tool detail within each.
"""

import logging

from config import PHOENIX_COLLECTOR_ENDPOINT, PHOENIX_PROJECT_NAME

log = logging.getLogger(__name__)

_tracer_provider = None
_disabled = False


def setup_tracing(project_name: str = PHOENIX_PROJECT_NAME):
    """Register the OTLP exporter and instrument LangChain/LangGraph (once per pod).

    Degrades gracefully: if the tracing libs are missing (e.g. a `--local` run in
    a venv without them) it logs a warning and disables tracing rather than
    crashing the run. The work still runs; only the spans are skipped.
    """
    global _tracer_provider, _disabled
    if _tracer_provider is not None or _disabled:
        return _tracer_provider

    try:
        from phoenix.otel import register
        from openinference.instrumentation.langchain import LangChainInstrumentor
    except Exception as e:
        log.warning(f"[tracing] disabled: tracing libs not importable ({type(e).__name__}: {e})")
        _disabled = True
        return None

    # Do NOT take over the global OTel provider: Flyte itself uses OpenTelemetry
    # for `@flyte.trace` and its action-enqueue calls, and if Phoenix owned the
    # global provider those internal HTTP spans would leak into Phoenix as noise.
    # We pass our provider explicitly to the LangChain instrumentor instead, so
    # Phoenix gets only the LLM/tool/graph spans and Flyte's stay in the Flyte UI.
    try:
        _tracer_provider = register(
            endpoint=PHOENIX_COLLECTOR_ENDPOINT + "/v1/traces",
            project_name=project_name,
            batch=True,
            set_global_tracer_provider=False,
        )
        LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
    except Exception as e:
        log.warning(f"[tracing] disabled: exporter setup failed ({type(e).__name__}: {e})")
        _disabled = True
        _tracer_provider = None
        return None

    log.info(f"[tracing] exporting to {PHOENIX_COLLECTOR_ENDPOINT} (project={project_name})")
    return _tracer_provider


def get_tracer(name: str = "research-pipeline"):
    """A tracer bound to the Phoenix provider, for emitting custom spans.

    Used to mark the final synthesized report as its own span (query in, report
    out) so Phoenix can evaluate it as a unit. Bound to our provider (not the
    global one) so it exports to Phoenix like the instrumented spans. If tracing
    is disabled, falls back to a no-op tracer so callers still work.
    """
    tp = setup_tracing()
    if tp is not None:
        return tp.get_tracer(name)
    from opentelemetry import trace
    return trace.get_tracer(name)


def run_session_id() -> str:
    """The Flyte run name of the current task, used as the Phoenix session id.

    Every task in one pipeline run shares the same run name, so tagging spans
    with it makes Phoenix group the run's traces into a single Session (the
    flat-list problem otherwise: each task is its own process => its own trace).
    """
    try:
        import flyte

        c = flyte.ctx()
        if c is not None and getattr(c, "action", None) is not None and c.action.run_name:
            return c.action.run_name
    except Exception:
        pass
    return "local"


def session_scope():
    """Context manager that tags spans created within with the run's session id.

    Wrap a task's instrumented work in `with session_scope():` so its LLM/tool
    spans carry `session.id`, and Phoenix clumps the whole run together. A no-op
    if the instrumentation lib is unavailable.
    """
    try:
        from openinference.instrumentation import using_session

        return using_session(session_id=run_session_id())
    except Exception:
        from contextlib import nullcontext

        return nullcontext()


def flush():
    """Drain the batch exporter before a short-lived task pod exits."""
    if _tracer_provider is not None:
        _tracer_provider.force_flush()
