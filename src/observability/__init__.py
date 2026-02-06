"""Observability module for OCI AI Agent Coordinator.

Provides tracing (OCI APM), logging (OCI Logging), and custom metrics
(OCI Monitoring) with automatic trace-log correlation for end-to-end observability.

Features:
- OTEL tracing exported to OCI APM
- OCI Logging with trace_id/span_id correlation
- LLM token metrics published to OCI Monitoring

Quick Start:
    from src.observability import init_observability, get_tracer, get_trace_id

    # Initialize on startup
    init_observability(agent_name="coordinator")

    # Get tracer for creating spans
    tracer = get_tracer()

    # Get trace ID for log correlation
    trace_id = get_trace_id()

    # LLM metrics are automatically published via LLMInstrumentor
"""

from src.observability.llm_tracing import (
    AgentInstrumentor,
    GenAIAttributes,
    LLMInstrumentor,
    LLMMetrics,
    LLMSpanContext,
    OracleCodeAssistInstrumentor,
    create_llm_span,
    get_llm_metrics,
    llm_span,
    reset_llm_metrics,
)
from src.observability.metrics import (
    LLMMetricsPublisher,
    LLMUsageRecord,
    get_llm_metrics_publisher,
    init_llm_metrics,
    record_llm_usage,
    shutdown_llm_metrics,
)
from src.observability.oci_logging import (
    AGENT_LOG_IDS,
    build_apm_link,
    get_trace_context,
    get_trace_id,
    init_oci_logging,
    shutdown_oci_logging,
)
from src.observability.tracing import (
    SERVICE_NAMES,
    force_flush_traces,
    get_tracer,
    init_otel_tracing,
    init_otlp_log_export,
    init_tracing,
    is_otel_enabled,
    is_otlp_logs_enabled,
    shutdown_otlp_log_export,
    shutdown_tracing,
    truncate,
)
from src.observability.visualizer import (
    EXAMPLE_QUERIES,
    GRAPH_EDGES,
    GRAPH_NODES,
    EdgeType,
    ExecutionStep,
    NodeStatus,
    WorkflowEdge,
    WorkflowNode,
    WorkflowVisualization,
    WorkflowVisualizer,
    generate_execution_trace_diagram,
    generate_mermaid_diagram,
    get_visualization_data,
)


def init_observability(
    agent_name: str = "coordinator",
    profile: str = "DEFAULT",
) -> None:
    """Initialize all observability components.

    Initializes:
    1. OTEL tracing (exported to OCI APM)
    2. OTLP log export (logs appear in APM span "Logs" tab)
    3. OCI Logging (with trace_id correlation for Log Analytics)
    4. LLM Metrics publisher (to OCI Monitoring)

    Args:
        agent_name: Agent name for service identification
        profile: OCI config profile for logging
    """
    # Initialize tracing first
    init_tracing(component=agent_name)

    # Initialize OTLP log export - sends logs to APM via OTLP
    # so they appear in the span details "Logs" tab
    init_otlp_log_export(component=agent_name)

    # Initialize OCI Logging with trace correlation (for Log Analytics persistence)
    init_oci_logging(agent_name=agent_name, profile=profile)

    # Initialize LLM metrics publisher (for token tracking dashboards)
    init_llm_metrics()


def shutdown_observability(agent_name: str | None = None) -> None:
    """Shutdown all observability components.

    Args:
        agent_name: Specific agent to shutdown, or None for all
    """
    shutdown_llm_metrics()
    shutdown_oci_logging(agent_name)
    shutdown_tracing()


__all__ = [
    # Main initialization
    "init_observability",
    "shutdown_observability",
    # Tracing
    "init_tracing",
    "init_otel_tracing",
    "get_tracer",
    "is_otel_enabled",
    "shutdown_tracing",
    "force_flush_traces",
    "truncate",
    "SERVICE_NAMES",
    # OTLP Log Export (APM span-level log correlation)
    "init_otlp_log_export",
    "is_otlp_logs_enabled",
    "shutdown_otlp_log_export",
    # LLM Observability (GenAI Semantic Conventions)
    "LLMInstrumentor",
    "LLMSpanContext",
    "LLMMetrics",
    "AgentInstrumentor",
    "GenAIAttributes",
    "OracleCodeAssistInstrumentor",
    "llm_span",
    "create_llm_span",
    "get_llm_metrics",
    "reset_llm_metrics",
    # OCI Logging
    "init_oci_logging",
    "shutdown_oci_logging",
    "get_trace_id",
    "get_trace_context",
    "build_apm_link",
    "AGENT_LOG_IDS",
    # Workflow Visualizer
    "WorkflowVisualizer",
    "WorkflowVisualization",
    "WorkflowNode",
    "WorkflowEdge",
    "ExecutionStep",
    "NodeStatus",
    "EdgeType",
    "generate_mermaid_diagram",
    "generate_execution_trace_diagram",
    "get_visualization_data",
    "GRAPH_NODES",
    "GRAPH_EDGES",
    "EXAMPLE_QUERIES",
]
