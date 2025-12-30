import logging
import os
from typing import Any

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

# Setup logging
logger = logging.getLogger(__name__)

# Max attribute value length for OCI APM
MAX_ATTR_LENGTH = 4096

_tracer_provider: TracerProvider | None = None
_otel_enabled: bool = False


def _get_service_config() -> dict[str, str]:
    """Get service configuration from environment."""
    return {
        "service_name": os.getenv("OTEL_SERVICE_NAME", "oci-ai-coordinator"),
        "service_version": "1.0.0",
        "deployment_env": os.getenv("NODE_ENV", "development"),
    }


def _get_apm_config() -> dict[str, str | None]:
    """Get OCI APM configuration from environment."""
    return {
        "endpoint": os.getenv("OCI_APM_ENDPOINT"),
        "private_data_key": os.getenv("OCI_APM_PRIVATE_DATA_KEY"),
    }


def _should_enable_otel() -> bool:
    """Check if OTEL should be enabled."""
    config = _get_apm_config()
    enabled_env = os.getenv("OTEL_TRACING_ENABLED", "true").lower() != "false"
    return enabled_env and bool(config["endpoint"]) and bool(config["private_data_key"])


def truncate(val: Any, max_len: int = MAX_ATTR_LENGTH) -> str:
    """Truncate string to max length."""
    str_val = str(val)
    if len(str_val) <= max_len:
        return str_val
    return str_val[: max_len - 3] + "..."


def init_otel_tracing() -> bool:
    """Initialize OpenTelemetry SDK for OCI APM."""
    global _tracer_provider, _otel_enabled  # noqa: PLW0603

    if not _should_enable_otel():
        logger.info(
            "[OtelTracing] OTEL tracing disabled - missing endpoint or data key"
        )
        _otel_enabled = False
        return False

    if _tracer_provider:
        logger.info("[OtelTracing] Tracer provider already initialized")
        return True

    config = _get_apm_config()
    svc_config = _get_service_config()

    try:
        # Construct endpoint URL
        endpoint = config["endpoint"].rstrip("/")
        if "/opentelemetry" in endpoint:
            trace_url = f"{endpoint}/private/v1/traces"
        elif "/20200101" in endpoint:
            trace_url = f"{endpoint}/opentelemetry/private/v1/traces"
        else:
            trace_url = f"{endpoint}/20200101/opentelemetry/private/v1/traces"

        logger.info(f"[OtelTracing] Trace endpoint: {trace_url}")

        # Configure exporter
        headers = {"Authorization": f"dataKey {config['private_data_key']}"}
        exporter = OTLPSpanExporter(endpoint=trace_url, headers=headers)

        # Configure resource
        resource = Resource.create(
            {
                ResourceAttributes.SERVICE_NAME: svc_config["service_name"],
                ResourceAttributes.SERVICE_VERSION: svc_config["service_version"],
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: svc_config["deployment_env"],
                "service.namespace": "oci-ai-coordinator",
                "cloud.provider": "oci",
            }
        )

        # Initialize provider
        _tracer_provider = TracerProvider(resource=resource)
        _tracer_provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(_tracer_provider)

        _otel_enabled = True
        logger.info(
            f"[OtelTracing] Initialized for OCI APM: {svc_config['service_name']}"
        )
        return True

    except Exception as e:
        logger.error(f"[OtelTracing] Failed to initialize: {e}")
        return False


def get_tracer():
    """Get the tracer instance."""
    svc_config = _get_service_config()
    return trace.get_tracer(svc_config["service_name"], svc_config["service_version"])


def is_otel_enabled() -> bool:
    """Check if OTEL is enabled."""
    return _otel_enabled
