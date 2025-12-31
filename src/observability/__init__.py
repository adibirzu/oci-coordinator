"""Observability module for OCI AI Agent Coordinator.

Provides tracing (OCI APM) and logging (OCI Logging) with automatic
trace-log correlation for end-to-end observability.

Quick Start:
    from src.observability import init_observability, get_tracer, get_trace_id

    # Initialize on startup
    init_observability(agent_name="coordinator")

    # Get tracer for creating spans
    tracer = get_tracer()

    # Get trace ID for log correlation
    trace_id = get_trace_id()
"""

from src.observability.tracing import (
    SERVICE_NAMES,
    force_flush_traces,
    get_tracer,
    init_otel_tracing,
    init_tracing,
    is_otel_enabled,
    shutdown_tracing,
    truncate,
)
from src.observability.oci_logging import (
    AGENT_LOG_IDS,
    build_apm_link,
    get_trace_context,
    get_trace_id,
    init_oci_logging,
    shutdown_oci_logging,
)


def init_observability(
    agent_name: str = "coordinator",
    profile: str = "DEFAULT",
) -> None:
    """Initialize all observability components.

    Args:
        agent_name: Agent name for service identification
        profile: OCI config profile for logging
    """
    # Initialize tracing first
    init_tracing(component=agent_name)

    # Initialize OCI Logging with trace correlation
    init_oci_logging(agent_name=agent_name, profile=profile)


def shutdown_observability(agent_name: str | None = None) -> None:
    """Shutdown all observability components.

    Args:
        agent_name: Specific agent to shutdown, or None for all
    """
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
    # OCI Logging
    "init_oci_logging",
    "shutdown_oci_logging",
    "get_trace_id",
    "get_trace_context",
    "build_apm_link",
    "AGENT_LOG_IDS",
]
