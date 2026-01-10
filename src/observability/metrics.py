"""OCI Custom Metrics for LLM Token Usage.

Publishes LLM token metrics to OCI Monitoring service for:
- Dashboard visualization in OCI Console
- Alarm configuration on token usage thresholds
- Integration with OCI APM custom metrics

Two approaches are supported:
1. **APM Span Attributes** (automatic) - Token counts in spans, aggregated by APM
2. **OCI Monitoring API** (this module) - Direct metric publishing for custom dashboards

Usage:
    from src.observability.metrics import LLMMetricsPublisher

    # Initialize publisher (typically on startup)
    publisher = LLMMetricsPublisher()

    # Record LLM usage (called after each LLM call)
    publisher.record_llm_usage(
        model="oca/gpt-4.1",
        operation="chat",
        input_tokens=150,
        output_tokens=250,
        latency_ms=1200,
        success=True,
    )

    # Force flush metrics (called on shutdown)
    publisher.flush()

Metrics Published:
    Namespace: oracle_apm_custom (configurable)
    Metrics:
    - llm_input_tokens: Count of input tokens
    - llm_output_tokens: Count of output tokens
    - llm_total_tokens: Total tokens (input + output)
    - llm_latency_ms: Request latency in milliseconds
    - llm_request_count: Number of requests
    - llm_error_count: Number of failed requests
    - llm_cost_estimate_usd: Estimated cost in USD

Dimensions (for filtering/grouping):
    - model: LLM model identifier (e.g., "oca/gpt-4.1")
    - operation: Operation type (chat, completion, embedding)
    - agent: Agent name (coordinator, db-troubleshoot, etc.)
    - system: LLM provider (oracle_code_assist, anthropic, openai)
"""

import os
import queue
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Configuration
NAMESPACE = os.getenv("OCI_METRICS_NAMESPACE", "oracle_apm_custom")
COMPARTMENT_ID = os.getenv("OCI_METRICS_COMPARTMENT_ID") or os.getenv("OCI_COMPARTMENT_ID")
BATCH_SIZE = int(os.getenv("OCI_METRICS_BATCH_SIZE", "20"))
FLUSH_INTERVAL_SECONDS = float(os.getenv("OCI_METRICS_FLUSH_INTERVAL", "60"))
RESOURCE_GROUP = os.getenv("OCI_METRICS_RESOURCE_GROUP", "llm-observability")


@dataclass
class LLMUsageRecord:
    """Record of a single LLM usage event."""

    model: str
    operation: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    success: bool
    agent: str = "coordinator"
    system: str = "oracle_code_assist"
    cost_estimate_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class LLMMetricsPublisher:
    """Publishes LLM token metrics to OCI Monitoring service.

    Features:
    - Batched metric submission for efficiency
    - Background thread for async publishing
    - Automatic dimension extraction from usage records
    - Compatible with OCI APM custom metric dashboards
    """

    # Token cost estimates per 1K tokens (approximate)
    TOKEN_COSTS = {
        "oca/gpt-4.1": {"input": 0.001, "output": 0.002},
        "oca/gpt-4.1": {"input": 0.001, "output": 0.002},
        "claude-sonnet-4-20250514": {"input": 0.003, "output": 0.015},
        "claude-3-opus": {"input": 0.015, "output": 0.075},
        "gpt-4-turbo": {"input": 0.01, "output": 0.03},
        "gpt-4o": {"input": 0.005, "output": 0.015},
    }

    def __init__(
        self,
        namespace: str = NAMESPACE,
        compartment_id: str | None = None,
        resource_group: str = RESOURCE_GROUP,
        profile: str = "DEFAULT",
    ):
        """Initialize LLM metrics publisher.

        Args:
            namespace: OCI Monitoring namespace (default: oracle_apm_custom)
            compartment_id: OCI compartment for metrics storage
            resource_group: Resource group for metric organization
            profile: OCI config profile name
        """
        self.namespace = namespace
        self.compartment_id = compartment_id or COMPARTMENT_ID
        self.resource_group = resource_group
        self.profile = profile

        self._queue: queue.Queue[LLMUsageRecord] = queue.Queue(maxsize=10000)
        self._shutdown = threading.Event()
        self._flush_thread: threading.Thread | None = None
        self._monitoring_client = None
        self._enabled = False

        # Initialize OCI client if configured
        if self.compartment_id:
            self._init_oci_client()
            if self._monitoring_client:
                self._start_flush_thread()
                self._enabled = True
                logger.info(
                    "LLM metrics publisher initialized",
                    namespace=namespace,
                    compartment_id=self.compartment_id[:40] + "...",
                )
        else:
            logger.warning(
                "LLM metrics publisher disabled - no compartment configured",
                help="Set OCI_METRICS_COMPARTMENT_ID or OCI_COMPARTMENT_ID",
            )

    def _init_oci_client(self) -> None:
        """Initialize OCI Monitoring client."""
        try:
            import oci
            from oci.monitoring import MonitoringClient

            config = oci.config.from_file(profile_name=self.profile)
            self._monitoring_client = MonitoringClient(config)

            logger.debug("OCI Monitoring client initialized", profile=self.profile)

        except ImportError:
            logger.warning("OCI SDK not installed - metrics disabled")
        except Exception as e:
            logger.error("Failed to initialize OCI Monitoring client", error=str(e))

    def _start_flush_thread(self) -> None:
        """Start background thread for batch flushing."""
        self._flush_thread = threading.Thread(
            target=self._flush_loop,
            daemon=True,
            name="llm-metrics-publisher",
        )
        self._flush_thread.start()

    def _flush_loop(self) -> None:
        """Background loop that flushes metrics periodically."""
        while not self._shutdown.is_set():
            try:
                self._flush_batch()
            except Exception as e:
                logger.error("Error in metrics flush loop", error=str(e))

            self._shutdown.wait(FLUSH_INTERVAL_SECONDS)

        # Final flush on shutdown
        self._flush_batch()

    def _estimate_cost(self, model: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost in USD for token usage."""
        costs = self.TOKEN_COSTS.get(model, {"input": 0.001, "output": 0.002})
        input_cost = (input_tokens / 1000) * costs["input"]
        output_cost = (output_tokens / 1000) * costs["output"]
        return round(input_cost + output_cost, 6)

    def record_llm_usage(
        self,
        model: str,
        operation: str,
        input_tokens: int,
        output_tokens: int,
        latency_ms: float,
        success: bool = True,
        agent: str = "coordinator",
        system: str = "oracle_code_assist",
    ) -> None:
        """Record LLM usage for metric publishing.

        Args:
            model: LLM model identifier (e.g., "oca/gpt-4.1")
            operation: Operation type (chat, completion, embedding, classification)
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            latency_ms: Request latency in milliseconds
            success: Whether the request succeeded
            agent: Agent name
            system: LLM provider system
        """
        if not self._enabled:
            return

        record = LLMUsageRecord(
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            agent=agent,
            system=system,
            cost_estimate_usd=self._estimate_cost(model, input_tokens, output_tokens),
        )

        try:
            self._queue.put_nowait(record)
        except queue.Full:
            # Drop oldest entry if queue is full
            try:
                self._queue.get_nowait()
                self._queue.put_nowait(record)
            except queue.Empty:
                pass

    def _flush_batch(self) -> None:
        """Flush queued usage records to OCI Monitoring."""
        if not self._monitoring_client:
            # Drain queue if no client
            while not self._queue.empty():
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    break
            return

        records: list[LLMUsageRecord] = []
        while len(records) < BATCH_SIZE and not self._queue.empty():
            try:
                records.append(self._queue.get_nowait())
            except queue.Empty:
                break

        if not records:
            return

        try:
            self._publish_metrics(records)
        except Exception as e:
            logger.error("Failed to publish LLM metrics", error=str(e), count=len(records))

    def _publish_metrics(self, records: list[LLMUsageRecord]) -> None:
        """Publish metric data points to OCI Monitoring."""
        from oci.monitoring.models import MetricDataDetails, PostMetricDataDetails

        metric_data_list = []

        for record in records:
            # Common dimensions for all metrics
            dimensions = {
                "model": record.model,
                "operation": record.operation,
                "agent": record.agent,
                "system": record.system,
            }

            timestamp = record.timestamp

            # Input tokens metric
            metric_data_list.append(
                MetricDataDetails(
                    namespace=self.namespace,
                    compartment_id=self.compartment_id,
                    name="llm_input_tokens",
                    dimensions=dimensions,
                    datapoints=[
                        {
                            "timestamp": timestamp,
                            "value": float(record.input_tokens),
                        }
                    ],
                    resource_group=self.resource_group,
                    metadata={"unit": "tokens"},
                )
            )

            # Output tokens metric
            metric_data_list.append(
                MetricDataDetails(
                    namespace=self.namespace,
                    compartment_id=self.compartment_id,
                    name="llm_output_tokens",
                    dimensions=dimensions,
                    datapoints=[
                        {
                            "timestamp": timestamp,
                            "value": float(record.output_tokens),
                        }
                    ],
                    resource_group=self.resource_group,
                    metadata={"unit": "tokens"},
                )
            )

            # Total tokens metric
            metric_data_list.append(
                MetricDataDetails(
                    namespace=self.namespace,
                    compartment_id=self.compartment_id,
                    name="llm_total_tokens",
                    dimensions=dimensions,
                    datapoints=[
                        {
                            "timestamp": timestamp,
                            "value": float(record.input_tokens + record.output_tokens),
                        }
                    ],
                    resource_group=self.resource_group,
                    metadata={"unit": "tokens"},
                )
            )

            # Latency metric
            metric_data_list.append(
                MetricDataDetails(
                    namespace=self.namespace,
                    compartment_id=self.compartment_id,
                    name="llm_latency_ms",
                    dimensions=dimensions,
                    datapoints=[
                        {
                            "timestamp": timestamp,
                            "value": record.latency_ms,
                        }
                    ],
                    resource_group=self.resource_group,
                    metadata={"unit": "ms"},
                )
            )

            # Request count metric (always 1 per record)
            metric_data_list.append(
                MetricDataDetails(
                    namespace=self.namespace,
                    compartment_id=self.compartment_id,
                    name="llm_request_count",
                    dimensions=dimensions,
                    datapoints=[
                        {
                            "timestamp": timestamp,
                            "value": 1.0,
                        }
                    ],
                    resource_group=self.resource_group,
                    metadata={"unit": "count"},
                )
            )

            # Error count metric (1 if failed, 0 if success)
            if not record.success:
                metric_data_list.append(
                    MetricDataDetails(
                        namespace=self.namespace,
                        compartment_id=self.compartment_id,
                        name="llm_error_count",
                        dimensions=dimensions,
                        datapoints=[
                            {
                                "timestamp": timestamp,
                                "value": 1.0,
                            }
                        ],
                        resource_group=self.resource_group,
                        metadata={"unit": "count"},
                    )
                )

            # Cost estimate metric
            if record.cost_estimate_usd > 0:
                metric_data_list.append(
                    MetricDataDetails(
                        namespace=self.namespace,
                        compartment_id=self.compartment_id,
                        name="llm_cost_estimate_usd",
                        dimensions=dimensions,
                        datapoints=[
                            {
                                "timestamp": timestamp,
                                "value": record.cost_estimate_usd,
                            }
                        ],
                        resource_group=self.resource_group,
                        metadata={"unit": "USD"},
                    )
                )

        # Submit to OCI Monitoring
        post_data = PostMetricDataDetails(
            metric_data=metric_data_list,
            batch_atomicity="NON_ATOMIC",  # Allow partial success
        )

        self._monitoring_client.post_metric_data(post_data)

        logger.debug(
            "Published LLM metrics",
            records=len(records),
            metrics=len(metric_data_list),
        )

    def flush(self) -> None:
        """Force flush all pending metrics."""
        if self._enabled:
            self._flush_batch()

    def shutdown(self) -> None:
        """Shutdown the publisher and flush remaining metrics."""
        self._shutdown.set()
        if self._flush_thread and self._flush_thread.is_alive():
            self._flush_thread.join(timeout=10.0)
        logger.info("LLM metrics publisher shutdown")

    @property
    def enabled(self) -> bool:
        """Check if metrics publishing is enabled."""
        return self._enabled


# Global publisher instance
_metrics_publisher: LLMMetricsPublisher | None = None


def init_llm_metrics(
    namespace: str = NAMESPACE,
    compartment_id: str | None = None,
    resource_group: str = RESOURCE_GROUP,
) -> LLMMetricsPublisher | None:
    """Initialize global LLM metrics publisher.

    Args:
        namespace: OCI Monitoring namespace
        compartment_id: OCI compartment for metrics
        resource_group: Resource group for organization

    Returns:
        Publisher instance or None if disabled
    """
    global _metrics_publisher

    if _metrics_publisher is not None:
        return _metrics_publisher

    _metrics_publisher = LLMMetricsPublisher(
        namespace=namespace,
        compartment_id=compartment_id,
        resource_group=resource_group,
    )

    return _metrics_publisher


def get_llm_metrics_publisher() -> LLMMetricsPublisher | None:
    """Get the global LLM metrics publisher."""
    return _metrics_publisher


def record_llm_usage(
    model: str,
    operation: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    success: bool = True,
    agent: str = "coordinator",
    system: str = "oracle_code_assist",
) -> None:
    """Record LLM usage to global publisher.

    Convenience function that uses the global publisher.

    Args:
        model: LLM model identifier
        operation: Operation type
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens
        latency_ms: Request latency in milliseconds
        success: Whether the request succeeded
        agent: Agent name
        system: LLM provider system
    """
    if _metrics_publisher and _metrics_publisher.enabled:
        _metrics_publisher.record_llm_usage(
            model=model,
            operation=operation,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            success=success,
            agent=agent,
            system=system,
        )


def shutdown_llm_metrics() -> None:
    """Shutdown global LLM metrics publisher."""
    global _metrics_publisher

    if _metrics_publisher:
        _metrics_publisher.shutdown()
        _metrics_publisher = None
