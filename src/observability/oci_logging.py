"""OCI Logging integration with APM trace correlation.

Sends structured logs to OCI Logging service with automatic
trace ID injection from OpenTelemetry context for correlation
with APM traces.

Each agent has its own dedicated OCI Log for filtering and analysis.

Usage:
    from src.observability.oci_logging import init_oci_logging, get_trace_id

    # Initialize on startup (auto-selects log based on agent name)
    handler = init_oci_logging(agent_name="db-troubleshoot-agent")

    # Get trace ID for manual correlation
    trace_id = get_trace_id()
"""

import json
import logging
import os
import queue
import threading
from datetime import datetime, timezone
from typing import Any

from opentelemetry import trace

import structlog

logger = structlog.get_logger(__name__)

# Batch settings for efficient log submission
BATCH_SIZE = 50
FLUSH_INTERVAL_SECONDS = 5.0
MAX_MESSAGE_SIZE = 32768  # 32KB per log entry

# Agent to Log OCID mapping (environment variable names)
# Uses naming convention: OCI_LOG_ID_{AGENT_NAME_UPPERCASE}
AGENT_LOG_ENV_VARS = {
    "coordinator": "OCI_LOG_ID_COORDINATOR",
    "db-troubleshoot-agent": "OCI_LOG_ID_DB_TROUBLESHOOT",
    "log-analytics-agent": "OCI_LOG_ID_LOG_ANALYTICS",
    "security-threat-agent": "OCI_LOG_ID_SECURITY_THREAT",
    "finops-agent": "OCI_LOG_ID_FINOPS",
    "infrastructure-agent": "OCI_LOG_ID_INFRASTRUCTURE",
    "slack-handler": "OCI_LOG_ID_COORDINATOR",  # Use coordinator log
    "mcp-executor": "OCI_LOG_ID_COORDINATOR",  # Use coordinator log
}

# Agent service names for log identification
AGENT_LOG_IDS = {
    "coordinator": "oci-coordinator-agent",
    "db-troubleshoot-agent": "oci-db-troubleshoot-agent",
    "log-analytics-agent": "oci-log-analytics-agent",
    "security-threat-agent": "oci-security-threat-agent",
    "finops-agent": "oci-finops-agent",
    "infrastructure-agent": "oci-infrastructure-agent",
    "slack-handler": "oci-slack-handler",
    "mcp-executor": "oci-mcp-executor",
}


def get_agent_log_id(agent_name: str) -> str | None:
    """Get the OCI Log OCID for a specific agent.

    Args:
        agent_name: Agent name (e.g., 'db-troubleshoot-agent')

    Returns:
        OCI Log OCID or None
    """
    # Look up the environment variable name for this agent
    env_var = AGENT_LOG_ENV_VARS.get(agent_name)

    if env_var:
        log_id = os.getenv(env_var)
        if log_id:
            return log_id

    # Fall back to default log ID
    return os.getenv("OCI_LOG_ID") or os.getenv("OCI_LOG_ID_COORDINATOR")


class OCILoggingHandler(logging.Handler):
    """Python logging handler that sends logs to OCI Logging service.

    Features:
    - Automatic APM trace ID correlation
    - Per-agent dedicated log streams
    - Batched log submission for efficiency
    - Background thread for async sending
    - Structured JSON log format
    """

    def __init__(
        self,
        log_group_id: str | None = None,
        log_id: str | None = None,
        agent_name: str = "coordinator",
        profile: str = "DEFAULT",
    ):
        """Initialize OCI Logging handler.

        Args:
            log_group_id: OCI Log Group OCID (from env if not provided)
            log_id: OCI Log OCID (auto-selected based on agent if not provided)
            agent_name: Agent name for log stream identification
            profile: OCI config profile
        """
        super().__init__()
        self.log_group_id = log_group_id or os.getenv("OCI_LOG_GROUP_ID")
        self.agent_name = agent_name
        self.profile = profile

        # Auto-select log ID based on agent name
        self.log_id = log_id or get_agent_log_id(agent_name)

        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=10000)
        self._shutdown = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._loggingingestion_client = None

        # Initialize OCI client if configured
        if self.log_group_id and self.log_id:
            self._init_oci_client()
            self._start_flush_thread()
            logger.info(
                "OCI Logging handler initialized",
                agent=agent_name,
                log_id=self.log_id[:50] + "..." if self.log_id else None,
            )
        else:
            logger.warning(
                "OCI Logging not configured",
                log_group_id=bool(self.log_group_id),
                log_id=bool(self.log_id),
                agent=agent_name,
                help="Set OCI_LOG_GROUP_ID and agent-specific OCI_LOG_ID_* variables",
            )

    def _init_oci_client(self) -> None:
        """Initialize OCI Logging Ingestion client."""
        try:
            import oci
            from oci.loggingingestion import LoggingClient

            config = oci.config.from_file(profile_name=self.profile)

            # Get region from environment or config
            region = os.getenv("OCI_LOGGING_REGION") or config.get("region", "eu-frankfurt-1")
            service_endpoint = f"https://ingestion.logging.{region}.oci.oraclecloud.com"

            self._loggingingestion_client = LoggingClient(
                config,
                service_endpoint=service_endpoint,
            )

            logger.info(
                "OCI Logging client initialized",
                region=region,
                agent=self.agent_name,
            )
        except ImportError:
            logger.warning("OCI SDK not installed - OCI Logging disabled")
            self._loggingingestion_client = None
        except Exception as e:
            logger.error("Failed to initialize OCI Logging client", error=str(e))
            self._loggingingestion_client = None

    def _start_flush_thread(self) -> None:
        """Start background thread for batch flushing."""
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name=f"oci-logging-{self.agent_name}",
        )
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background loop that flushes logs periodically."""
        while not self._shutdown.is_set():
            try:
                self._flush_batch()
            except Exception as e:
                logger.error("Error in flush loop", error=str(e))

            self._shutdown.wait(FLUSH_INTERVAL_SECONDS)

        # Final flush on shutdown
        self._flush_batch()

    def _flush_batch(self) -> None:
        """Flush queued log entries to OCI Logging."""
        if not self._loggingingestion_client:
            # Drain queue if no client
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            return

        entries = []
        while len(entries) < BATCH_SIZE and not self._queue.empty():
            try:
                entries.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not entries:
            return

        try:
            from oci.loggingingestion.models import LogEntry, LogEntryBatch, PutLogsDetails

            log_entries = [
                LogEntry(
                    data=json.dumps(entry["data"])[:MAX_MESSAGE_SIZE],
                    id=entry["id"],
                    time=entry["time"],
                )
                for entry in entries
            ]

            # Use agent-specific source and subject for filtering
            batch = LogEntryBatch(
                entries=log_entries,
                source=f"oci-ai-coordinator/{self.agent_name}",
                type="agent-logs",
                subject=AGENT_LOG_IDS.get(self.agent_name, self.agent_name),
                defaultlogentrytime=datetime.now(timezone.utc).isoformat(),
            )

            put_logs_details = PutLogsDetails(
                specversion="1.0",
                log_entry_batches=[batch],
            )

            self._loggingingestion_client.put_logs(
                log_id=self.log_id,
                put_logs_details=put_logs_details,
            )

            logger.debug("Flushed logs to OCI", count=len(entries), agent=self.agent_name)

        except Exception as e:
            logger.error(
                "Failed to send logs to OCI",
                error=str(e),
                count=len(entries),
            )

    def emit(self, record: logging.LogRecord) -> None:
        """Emit a log record to the OCI queue.

        Injects trace context from OpenTelemetry for APM correlation.
        """
        try:
            # Get trace context for APM correlation
            span = trace.get_current_span()
            span_context = span.get_span_context() if span else None

            # Build structured log entry
            log_data: dict[str, Any] = {
                "level": record.levelname,
                "message": record.getMessage(),
                "logger": record.name,
                "agent": self.agent_name,
                "component": getattr(record, "component", None),
                "timestamp": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
            }

            # Add APM trace correlation - CRITICAL for log-trace linking
            if span_context and span_context.is_valid:
                log_data["trace_id"] = format(span_context.trace_id, "032x")
                log_data["span_id"] = format(span_context.span_id, "016x")
                log_data["trace_flags"] = span_context.trace_flags

            # Add exception info if present
            if record.exc_info:
                log_data["exception"] = self.formatException(record.exc_info)

            # Add extra fields from structlog and agent context
            extra_fields = [
                "event", "tool", "server", "duration_ms", "success",
                "skill", "workflow", "step", "database", "query",
                "metric", "severity", "compartment_id",
            ]
            for key in extra_fields:
                if hasattr(record, key):
                    log_data[key] = getattr(record, key)

            entry = {
                "id": f"{record.created}-{id(record)}",
                "time": datetime.fromtimestamp(
                    record.created, tz=timezone.utc
                ).isoformat(),
                "data": log_data,
            }

            # Non-blocking queue put
            try:
                self._queue.put_nowait(entry)
            except queue.Full:
                # Drop oldest entries if queue is full
                try:
                    self._queue.get_nowait()
                    self._queue.put_nowait(entry)
                except queue.Empty:
                    pass

        except Exception:
            self.handleError(record)

    def close(self) -> None:
        """Shutdown the handler and flush remaining logs."""
        self._shutdown.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)
        super().close()


# Global handler instance per agent
_oci_handlers: dict[str, OCILoggingHandler] = {}


def init_oci_logging(
    agent_name: str = "coordinator",
    profile: str = "DEFAULT",
) -> OCILoggingHandler | None:
    """Initialize OCI Logging handler for an agent.

    Automatically selects the correct OCI Log based on agent name
    using the OCI_LOG_ID_{AGENT} environment variables.

    Args:
        agent_name: Agent name for log identification
        profile: OCI config profile

    Returns:
        Handler instance or None if disabled
    """
    global _oci_handlers

    # Check if already initialized for this agent
    if agent_name in _oci_handlers:
        return _oci_handlers[agent_name]

    # Check if logging is enabled
    if os.getenv("OCI_LOGGING_ENABLED", "true").lower() == "false":
        logger.info("OCI Logging disabled by configuration")
        return None

    log_group_id = os.getenv("OCI_LOG_GROUP_ID")
    log_id = get_agent_log_id(agent_name)

    if not log_group_id:
        logger.warning(
            "OCI Log Group not configured",
            help="Set OCI_LOG_GROUP_ID environment variable",
        )
        return None

    if not log_id:
        logger.warning(
            "OCI Log ID not configured for agent",
            agent=agent_name,
            help=f"Set OCI_LOG_ID_{agent_name.upper().replace('-', '_')} environment variable",
        )
        return None

    handler = OCILoggingHandler(
        log_group_id=log_group_id,
        log_id=log_id,
        agent_name=agent_name,
        profile=profile,
    )
    _oci_handlers[agent_name] = handler

    # Attach to root logger
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)

    logger.info(
        "OCI Logging initialized for agent",
        agent=agent_name,
        log_id=log_id[:40] + "..." if log_id else None,
    )

    return handler


def shutdown_oci_logging(agent_name: str | None = None) -> None:
    """Shutdown OCI Logging handler(s).

    Args:
        agent_name: Specific agent to shutdown, or None for all
    """
    global _oci_handlers

    if agent_name:
        handler = _oci_handlers.pop(agent_name, None)
        if handler:
            handler.close()
            logging.getLogger().removeHandler(handler)
    else:
        for name, handler in list(_oci_handlers.items()):
            handler.close()
            logging.getLogger().removeHandler(handler)
        _oci_handlers.clear()

    logger.info("OCI Logging shutdown", agent=agent_name or "all")


def get_trace_id() -> str | None:
    """Get current trace ID for log correlation.

    Returns:
        Hex-encoded trace ID or None if no active span
    """
    span = trace.get_current_span()
    if span:
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return format(ctx.trace_id, "032x")
    return None


def get_trace_context() -> dict[str, str]:
    """Get current trace context for log correlation.

    Returns:
        Dictionary with trace_id, span_id, and trace_flags
    """
    span = trace.get_current_span()
    if span:
        ctx = span.get_span_context()
        if ctx and ctx.is_valid:
            return {
                "trace_id": format(ctx.trace_id, "032x"),
                "span_id": format(ctx.span_id, "016x"),
                "trace_flags": str(ctx.trace_flags),
            }
    return {}


def build_apm_link(trace_id: str) -> str:
    """Build a link to OCI APM trace explorer.

    Args:
        trace_id: Trace ID

    Returns:
        URL to trace in APM console
    """
    apm_domain_id = os.getenv("OCI_APM_DOMAIN_ID")
    region = os.getenv("OCI_LOGGING_REGION", "eu-frankfurt-1")

    if not apm_domain_id:
        return ""

    return (
        f"https://cloud.oracle.com/apm/apm-traces?"
        f"region={region}&apmDomainId={apm_domain_id}&traceId={trace_id}"
    )


def build_log_analytics_link(trace_id: str) -> str:
    """Build a link to Log Analytics filtered by trace ID.

    Args:
        trace_id: Trace ID for correlation

    Returns:
        URL to Log Analytics with trace filter
    """
    region = os.getenv("OCI_LOGGING_REGION", "eu-frankfurt-1")
    namespace = os.getenv("OCI_DEFAULT_TENANCY_NAME", "")

    # Build Log Analytics explore URL with trace_id filter
    return (
        f"https://cloud.oracle.com/loganalytics/explore?"
        f"region={region}&query=trace_id%3D%27{trace_id}%27"
    )
