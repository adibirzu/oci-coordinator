"""OCI APM / OpenTelemetry tracing configuration.

Each agent has a unique service name but shares the same APM project.
Traces are exported to OCI APM via OTLP or Zipkin format.

Usage:
    from src.observability.tracing import init_tracing, get_tracer

    # Initialize on startup
    tracer = init_tracing(component="coordinator")

    # Get tracer for a specific component
    tracer = get_tracer("db-troubleshoot-agent")

    # Create spans
    with tracer.start_as_current_span("operation_name") as span:
        span.set_attribute("key", "value")
"""

import json
import os
from collections.abc import Sequence
from typing import Any

import requests
import structlog
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.semconv.resource import ResourceAttributes

logger = structlog.get_logger(__name__)

# Max attribute value length for OCI APM
MAX_ATTR_LENGTH = 4096

# Service name constants for different components
SERVICE_NAMES = {
    "coordinator": "oci-coordinator-agent",
    "db-troubleshoot-agent": "oci-db-troubleshoot-agent",
    "log-analytics-agent": "oci-log-analytics-agent",
    "security-threat-agent": "oci-security-threat-agent",
    "finops-agent": "oci-finops-agent",
    "infrastructure-agent": "oci-infrastructure-agent",
    "mcp-executor": "oci-mcp-executor",
    "slack-handler": "oci-slack-handler",
}

# Global state
_tracer_provider: TracerProvider | None = None
_otel_enabled: bool = False
_current_tracer: trace.Tracer | None = None


class OCIAPMZipkinExporter(SpanExporter):
    """Custom span exporter for OCI APM using Zipkin format.

    OCI APM requires Zipkin v2 JSON format for older endpoints.
    Newer endpoints support OTLP but Zipkin is more reliable.
    """

    def __init__(self, endpoint: str, data_key: str, service_name: str = "unknown"):
        """Initialize OCI APM exporter.

        Args:
            endpoint: Full OCI APM endpoint URL
            data_key: OCI APM data key (public or private)
            service_name: Default service name for spans
        """
        self.endpoint = endpoint
        self.data_key = data_key
        self.service_name = service_name
        self._session = requests.Session()
        self._session.headers.update({
            "Content-Type": "application/json",
            "Authorization": f"dataKey {data_key}",
        })

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        """Export spans to OCI APM."""
        if not spans:
            return SpanExportResult.SUCCESS

        try:
            zipkin_spans = self._convert_to_zipkin(spans)

            response = self._session.post(
                self.endpoint,
                data=json.dumps(zipkin_spans),
                timeout=30,
            )

            if response.status_code in (200, 202):
                return SpanExportResult.SUCCESS
            else:
                logger.warning(
                    "OCI APM export failed",
                    status=response.status_code,
                    response=response.text[:200],
                )
                return SpanExportResult.FAILURE

        except Exception as e:
            logger.error("OCI APM export error", error=str(e))
            return SpanExportResult.FAILURE

    def _convert_to_zipkin(self, spans: Sequence[ReadableSpan]) -> list[dict]:
        """Convert OpenTelemetry spans to Zipkin v2 format."""
        zipkin_spans = []

        for span in spans:
            ctx = span.get_span_context()

            trace_id = format(ctx.trace_id, "032x")
            span_id = format(ctx.span_id, "016x")

            parent_id = None
            if span.parent and span.parent.span_id:
                parent_id = format(span.parent.span_id, "016x")

            zipkin_span = {
                "traceId": trace_id,
                "id": span_id,
                "name": span.name,
                "timestamp": span.start_time // 1000,
                "duration": (span.end_time - span.start_time) // 1000,
                "localEndpoint": {
                    "serviceName": self._get_service_name(span),
                },
            }

            if parent_id:
                zipkin_span["parentId"] = parent_id

            if span.attributes:
                zipkin_span["tags"] = {
                    str(k): str(v)[:MAX_ATTR_LENGTH] for k, v in span.attributes.items()
                }

            if span.kind:
                zipkin_span["kind"] = span.kind.name

            zipkin_spans.append(zipkin_span)

        return zipkin_spans

    def _get_service_name(self, span: ReadableSpan) -> str:
        """Extract service name from span resource."""
        if span.resource and span.resource.attributes:
            svc = span.resource.attributes.get("service.name")
            if svc:
                return str(svc)
        return self.service_name

    def shutdown(self) -> None:
        """Shutdown the exporter."""
        self._session.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        """Force flush is a no-op for this exporter."""
        return True


def _get_apm_config() -> dict[str, str | None]:
    """Get OCI APM configuration from environment."""
    return {
        "endpoint": os.getenv("OCI_APM_ENDPOINT"),
        "private_data_key": os.getenv("OCI_APM_PRIVATE_DATA_KEY"),
        "public_data_key": os.getenv("OCI_APM_PUBLIC_DATA_KEY"),
        "data_upload_endpoint": os.getenv("OCI_APM_DATA_UPLOAD_ENDPOINT"),
    }


def _should_enable_otel() -> bool:
    """Check if OTEL should be enabled."""
    config = _get_apm_config()
    enabled_env = os.getenv("OTEL_TRACING_ENABLED", "true").lower() != "false"
    has_endpoint = bool(config["endpoint"] or config["data_upload_endpoint"])
    has_key = bool(config["private_data_key"] or config["public_data_key"])
    return enabled_env and has_endpoint and has_key


def truncate(val: Any, max_len: int = MAX_ATTR_LENGTH) -> str:
    """Truncate string to max length."""
    str_val = str(val)
    if len(str_val) <= max_len:
        return str_val
    return str_val[: max_len - 3] + "..."


def _build_zipkin_endpoint(base_url: str, use_public: bool = False) -> str:
    """Build OCI APM Zipkin endpoint URL."""
    base_url = base_url.rstrip("/")
    endpoint_type = "public-span" if use_public else "private-span"

    # Handle different URL formats
    if "/20200101" in base_url:
        # Already has API version
        return f"{base_url.split('/20200101')[0]}/20200101/observations/{endpoint_type}?dataFormat=zipkin&dataFormatVersion=2"
    else:
        return f"{base_url}/20200101/observations/{endpoint_type}?dataFormat=zipkin&dataFormatVersion=2"


def init_tracing(
    service_name: str | None = None,
    component: str = "coordinator",
) -> trace.Tracer:
    """Initialize OpenTelemetry tracing with OCI APM export.

    Args:
        service_name: Override service name (defaults to component lookup)
        component: Component type for service name lookup

    Returns:
        Configured tracer instance
    """
    global _tracer_provider, _otel_enabled, _current_tracer

    if not _should_enable_otel():
        logger.info(
            "Tracing disabled",
            reason="missing endpoint or data key",
        )
        _otel_enabled = False
        _current_tracer = trace.get_tracer(__name__)
        return _current_tracer

    # Determine service name
    svc_name = service_name or SERVICE_NAMES.get(component, f"oci-{component}")
    deployment_env = os.getenv("DEPLOYMENT_ENV", "development")

    # Create resource with service metadata
    resource = Resource.create({
        ResourceAttributes.SERVICE_NAME: svc_name,
        ResourceAttributes.SERVICE_VERSION: "1.0.0",
        ResourceAttributes.DEPLOYMENT_ENVIRONMENT: deployment_env,
        "oci.apm.project": "oci-ai-coordinator",
        "cloud.provider": "oci",
    })

    # Create tracer provider
    _tracer_provider = TracerProvider(resource=resource)

    # Get APM configuration
    config = _get_apm_config()
    endpoint = config["endpoint"] or config["data_upload_endpoint"]
    private_key = config["private_data_key"]
    public_key = config["public_data_key"]

    # Try OTLP first, fall back to Zipkin
    exporter_type = "none"

    if endpoint:
        try:
            # Try OTLP exporter first (preferred)
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            # Construct OTLP endpoint
            otlp_endpoint = endpoint.rstrip("/")
            if "/opentelemetry" in otlp_endpoint:
                trace_url = f"{otlp_endpoint}/private/v1/traces"
            elif "/20200101" in otlp_endpoint:
                trace_url = f"{otlp_endpoint}/opentelemetry/private/v1/traces"
            else:
                trace_url = f"{otlp_endpoint}/20200101/opentelemetry/private/v1/traces"

            headers = {"Authorization": f"dataKey {private_key}"}
            exporter = OTLPSpanExporter(endpoint=trace_url, headers=headers)
            _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
            exporter_type = "otlp"

            logger.info(
                "OTLP tracing enabled",
                service=svc_name,
                endpoint=trace_url[:60] + "...",
            )

        except Exception as otlp_error:
            # Fall back to Zipkin exporter
            logger.warning(
                "OTLP export failed, trying Zipkin",
                error=str(otlp_error)[:100],
            )

            try:
                use_public = bool(public_key) and not private_key
                data_key = public_key if use_public else private_key

                if data_key:
                    # Get base URL for Zipkin
                    base_url = config["data_upload_endpoint"] or endpoint.split("/opentelemetry")[0]
                    zipkin_endpoint = _build_zipkin_endpoint(base_url, use_public)

                    exporter = OCIAPMZipkinExporter(
                        endpoint=zipkin_endpoint,
                        data_key=data_key,
                        service_name=svc_name,
                    )
                    _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
                    exporter_type = "zipkin"

                    logger.info(
                        "Zipkin tracing enabled",
                        service=svc_name,
                        key_type="public" if use_public else "private",
                    )

            except Exception as zipkin_error:
                logger.error(
                    "All exporters failed",
                    otlp_error=str(otlp_error)[:100],
                    zipkin_error=str(zipkin_error)[:100],
                )

    # Set as global provider
    trace.set_tracer_provider(_tracer_provider)

    _otel_enabled = exporter_type != "none"
    _current_tracer = trace.get_tracer(svc_name, "1.0.0")

    logger.info(
        "Tracing initialized",
        service=svc_name,
        exporter=exporter_type,
        enabled=_otel_enabled,
    )

    return _current_tracer


def get_tracer(component: str | None = None) -> trace.Tracer:
    """Get a tracer for a specific component.

    Args:
        component: Component name for service identification

    Returns:
        Tracer instance
    """
    if component:
        svc_name = SERVICE_NAMES.get(component, f"oci-{component}")
        return trace.get_tracer(svc_name, "1.0.0")
    return _current_tracer or trace.get_tracer(__name__)


def init_otel_tracing() -> bool:
    """Initialize OpenTelemetry SDK for OCI APM (compatibility function).

    Returns:
        True if tracing was enabled
    """
    tracer = init_tracing()
    return is_otel_enabled()


def is_otel_enabled() -> bool:
    """Check if OTEL is enabled."""
    return _otel_enabled


def shutdown_tracing() -> None:
    """Shutdown tracing and flush pending spans."""
    global _tracer_provider

    if _tracer_provider:
        try:
            _tracer_provider.force_flush(timeout_millis=5000)
            logger.info("Tracing force flush complete")
        except Exception as e:
            logger.warning("Tracing force flush failed", error=str(e))

        _tracer_provider.shutdown()
        logger.info("Tracing shutdown complete")


def force_flush_traces(timeout_ms: int = 5000) -> bool:
    """Force flush pending traces without shutdown.

    Args:
        timeout_ms: Maximum time to wait for flush

    Returns:
        True if flush succeeded
    """
    global _tracer_provider

    if _tracer_provider:
        try:
            return _tracer_provider.force_flush(timeout_millis=timeout_ms)
        except Exception:
            return False
    return True
