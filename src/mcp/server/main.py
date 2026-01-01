import logging
import os

import structlog
from fastmcp import FastMCP
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_mcp_observability():
    """Initialize observability for the MCP server."""
    # Check if OTEL is disabled
    if os.getenv("MCP_OTEL_SDK_DISABLED", "false").lower() == "true":
        logger.info("MCP OTEL SDK disabled by configuration")
        return

    # Check for OCI APM endpoint
    endpoint = os.getenv("MCP_OTEL_ENDPOINT") or os.getenv("OCI_APM_ENDPOINT")
    if not endpoint:
        logger.warning("No OTEL endpoint configured for MCP server")
        return

    # Get data key from headers or environment
    # Public key is sufficient for trace ingestion (write-only)
    headers_str = os.getenv("MCP_OTEL_HEADERS", "")
    data_key = None
    if "authorization=" in headers_str:
        data_key = headers_str.split("authorization=")[1].split(",")[0].strip()
    if not data_key:
        data_key = os.getenv("OCI_APM_PUBLIC_DATA_KEY") or os.getenv("OCI_APM_PRIVATE_DATA_KEY")

    if not data_key:
        logger.warning("No OTEL data key configured for MCP server")
        return

    service_prefix = os.getenv("MCP_OTEL_SERVICE_PREFIX", "mcp-oci")
    service_name = f"{service_prefix}-unified"

    try:
        # Create resource with service metadata
        resource = Resource.create({
            ResourceAttributes.SERVICE_NAME: service_name,
            ResourceAttributes.SERVICE_VERSION: "1.0.0",
            "oci.apm.project": "oci-ai-coordinator",
            "cloud.provider": "oci",
            "mcp.server": "oci-unified",
        })

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Try OTLP exporter
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )

            # Construct OTLP endpoint
            otlp_endpoint = endpoint.rstrip("/")
            if "/opentelemetry" not in otlp_endpoint:
                otlp_endpoint = f"{otlp_endpoint}/opentelemetry"
            trace_url = f"{otlp_endpoint}/private/v1/traces"

            headers = {"Authorization": f"dataKey {data_key}"}
            exporter = OTLPSpanExporter(endpoint=trace_url, headers=headers)
            provider.add_span_processor(BatchSpanProcessor(exporter))

            logger.info(f"MCP OTEL tracing enabled: {service_name}")

        except Exception as e:
            logger.warning(f"MCP OTLP exporter failed: {e}")
            return

        # Set as global provider
        trace.set_tracer_provider(provider)

    except Exception as e:
        logger.error(f"MCP observability init failed: {e}")


# Initialize observability
_init_mcp_observability()

# Get tracer for MCP server
_tracer = trace.get_tracer("mcp-oci-unified")

# Initialize FastMCP server
mcp = FastMCP(
    name="oci-unified-server",
    instructions="""Oracle Cloud Infrastructure MCP Server providing comprehensive 
cloud management capabilities through the Model Context Protocol.

Use `search_capabilities` to discover available tools.
"""
)

DOMAINS = {
    "identity": {
        "description": "Compartment listing, tenancy info, and IAM operations",
        "tools": ["oci_list_compartments", "oci_search_compartments", "oci_get_compartment", "oci_get_tenancy", "oci_list_regions"],
    },
    "compute": {
        "description": "Instance management, shapes, and performance metrics",
        "tools": ["oci_compute_list_instances", "oci_compute_get_instance", "oci_compute_start_instance", "oci_compute_stop_instance", "oci_compute_restart_instance"],
    },
    "cost": {
        "description": "Cost analysis, budgeting, and FinOps optimization",
        "tools": ["oci_cost_get_summary"],
    },
    "db": {
        "description": "Autonomous Database and DB Systems management",
        "tools": ["oci_db_list_autonomous", "oci_db_get_metrics"],
    },
    "network": {
        "description": "VCN, Subnet, and Security List management",
        "tools": ["oci_network_list_vcns", "oci_network_list_subnets", "oci_network_list_security_lists"],
    },
    "security": {
        "description": "IAM and Cloud Guard management",
        "tools": ["oci_security_list_users"],
    },
    "observability": {
        "description": "Logs, metrics, and alarms",
        "tools": ["oci_observability_get_metrics"],
    },
    "discovery": {
        "description": "ShowOCI-style resource discovery, caching, and search",
        "tools": ["oci_discovery_run", "oci_discovery_get_cached", "oci_discovery_refresh", "oci_discovery_summary", "oci_discovery_search", "oci_discovery_cache_status"],
    },
    "feedback": {
        "description": "Runtime feedback directives for prompt steering",
        "tools": ["set_feedback", "append_feedback", "get_feedback"],
    },
}

async def _search_capabilities_logic(query: str, domain: str | None = None) -> str:
    """Internal logic for search_capabilities."""
    with _tracer.start_as_current_span("mcp.search_capabilities") as span:
        span.set_attribute("query", query)
        span.set_attribute("domain", domain or "all")

        q = query.lower()
        d = domain.lower() if domain else None

        results = []

        for dom_name, info in DOMAINS.items():
            if d and d != dom_name:
                continue

            if q in dom_name or q in info["description"].lower() or any(q in t for t in info["tools"]):
                results.append(f"## Domain: {dom_name}")
                results.append(f"Description: {info['description']}")
                results.append(f"Tools: {', '.join(info['tools'])}\n")

        span.set_attribute("matches", len(results))

        if not results:
            return f"No domains or tools found matching '{query}'. Available domains: {', '.join(DOMAINS.keys())}"

        return "# Matching OCI Capabilities\n\n" + "\n".join(results)

@mcp.tool()
async def search_capabilities(query: str, domain: str | None = None) -> str:
    """Search for OCI tool domains and capabilities.
    
    Use this to discover available tools matching your intent.
    
    Args:
        query: Search keywords (e.g. 'how to troubleshoot', 'cost', 'instances')
        domain: Optional filter by domain (compute, cost, db, network, security, observability)
    """
    return await _search_capabilities_logic(query, domain)


_feedback_manager = None


def _get_feedback_manager():
    global _feedback_manager
    if _feedback_manager is None:
        from src.memory.manager import SharedMemoryManager

        redis_url = os.getenv("REDIS_URL") or os.getenv("MCP_REDIS_URL")
        _feedback_manager = SharedMemoryManager(redis_url=redis_url)
    return _feedback_manager


@mcp.tool()
async def set_feedback(scope: str, text: str, source: str = "operator") -> dict:
    """Replace runtime feedback directives for a scope."""
    manager = _get_feedback_manager()
    await manager.set_feedback(scope, text, source)
    return {"status": "ok", "scope": scope}


@mcp.tool()
async def append_feedback(scope: str, text: str, source: str = "operator") -> dict:
    """Append a feedback directive for a scope."""
    manager = _get_feedback_manager()
    await manager.append_feedback(scope, text, source)
    return {"status": "ok", "scope": scope}


@mcp.tool()
async def get_feedback(scope: str = "global") -> dict:
    """Get current feedback directives for a scope."""
    manager = _get_feedback_manager()
    entries = await manager.get_feedback_entries(scope)
    return {"scope": scope, "entries": entries}

from src.mcp.server.skills.troubleshoot import register_troubleshoot_skills
from src.mcp.server.tools.compute import register_compute_tools
from src.mcp.server.tools.cost import register_cost_tools
from src.mcp.server.tools.discovery import register_discovery_tools
from src.mcp.server.tools.identity import register_identity_tools
from src.mcp.server.tools.network import register_network_tools
from src.mcp.server.tools.observability import register_observability_tools
from src.mcp.server.tools.security import register_security_tools

# Register tools
register_identity_tools(mcp)  # Register identity tools first for compartment discovery
register_compute_tools(mcp)
register_network_tools(mcp)
register_cost_tools(mcp)
register_security_tools(mcp)
register_observability_tools(mcp)
register_discovery_tools(mcp)  # ShowOCI-style discovery tools

# Register skills
register_troubleshoot_skills(mcp)

if __name__ == "__main__":
    mcp.run()
