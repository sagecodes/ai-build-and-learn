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


def setup_tracing(project_name: str = PHOENIX_PROJECT_NAME):
    """Register the OTLP exporter and instrument LangChain/LangGraph (once per pod)."""
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider

    from phoenix.otel import register
    from openinference.instrumentation.langchain import LangChainInstrumentor

    # Do NOT take over the global OTel provider: Flyte itself uses OpenTelemetry
    # for `@flyte.trace` and its action-enqueue calls, and if Phoenix owned the
    # global provider those internal HTTP spans would leak into Phoenix as noise.
    # We pass our provider explicitly to the LangChain instrumentor instead, so
    # Phoenix gets only the LLM/tool/graph spans and Flyte's stay in the Flyte UI.
    _tracer_provider = register(
        endpoint=PHOENIX_COLLECTOR_ENDPOINT + "/v1/traces",
        project_name=project_name,
        batch=True,
        set_global_tracer_provider=False,
    )
    LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)
    log.info(f"[tracing] exporting to {PHOENIX_COLLECTOR_ENDPOINT} (project={project_name})")
    return _tracer_provider


def get_tracer(name: str = "research-pipeline"):
    """A tracer bound to the Phoenix provider, for emitting custom spans.

    Used to mark the final synthesized report as its own span (query in, report
    out) so Phoenix can evaluate it as a unit. Bound to our provider (not the
    global one) so it exports to Phoenix like the instrumented spans.
    """
    return setup_tracing().get_tracer(name)


def flush():
    """Drain the batch exporter before a short-lived task pod exits."""
    if _tracer_provider is not None:
        _tracer_provider.force_flush()
