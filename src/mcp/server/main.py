import logging
import os
from datetime import datetime

from fastmcp import FastMCP
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.semconv.resource import ResourceAttributes

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _init_mcp_oci_logging():
    """Initialize OCI Logging for MCP server with trace correlation.

    Uses separate log group for MCP server:
    - OCI_MCP_LOG_GROUP_ID: MCP-specific log group
    - OCI_MCP_LOG_ID: Log within the group for MCP server logs

    Falls back to agent log group if MCP-specific not configured.
    """
    # Check if OCI Logging is enabled
    if os.getenv("OCI_LOGGING_ENABLED", "true").lower() == "false":
        logger.info("OCI Logging disabled by configuration")
        return

    # MCP server uses its own log group (different from agent logs)
    log_group_id = os.getenv("OCI_MCP_LOG_GROUP_ID") or os.getenv("OCI_LOG_GROUP_ID")
    log_id = os.getenv("OCI_MCP_LOG_ID") or os.getenv("OCI_LOG_ID_COORDINATOR")

    if not log_group_id or not log_id:
        logger.warning(
            "MCP OCI Logging not configured - set OCI_MCP_LOG_GROUP_ID and OCI_MCP_LOG_ID"
        )
        return

    try:
        # Import OCI Logging handler from observability module
        from src.observability.oci_logging import OCILoggingHandler

        handler = OCILoggingHandler(
            log_group_id=log_group_id,
            log_id=log_id,
            agent_name="mcp-server",
            profile="DEFAULT",
        )

        # Attach to root logger for trace correlation
        root_logger = logging.getLogger()
        root_logger.addHandler(handler)

        logger.info(
            f"MCP OCI Logging initialized - log_id: {log_id[:40]}..."
        )

    except ImportError:
        logger.warning("OCI Logging handler not available - using console only")
    except Exception as e:
        logger.error(f"MCP OCI Logging init failed: {e}")


def _init_mcp_observability():
    """Initialize observability for the MCP server.

    Includes:
    - OTEL tracing to OCI APM (for distributed tracing)
    - OCI Logging with trace_id/span_id correlation (for log-trace linking)
    """
    # Initialize OCI Logging first (so trace IDs are captured in logs)
    _init_mcp_oci_logging()

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
    "dbmgmt": {
        "description": "Database Management service - AWR reports, SQL performance, fleet health",
        "tools": [
            "oci_dbmgmt_list_databases", "oci_dbmgmt_search_databases", "oci_dbmgmt_get_database",
            "oci_dbmgmt_get_awr_report", "oci_dbmgmt_get_metrics",
            "oci_dbmgmt_get_top_sql", "oci_dbmgmt_get_wait_events",
            "oci_dbmgmt_list_sql_plan_baselines", "oci_dbmgmt_get_fleet_health", "oci_dbmgmt_get_sql_report"
        ],
    },
    "opsi": {
        "description": "Operations Insights - database insights, SQL analytics, ADDM, capacity planning",
        "tools": [
            "oci_opsi_list_database_insights", "oci_opsi_get_database_insight",
            "oci_opsi_summarize_resource_stats", "oci_opsi_summarize_sql_insights",
            "oci_opsi_summarize_sql_statistics", "oci_opsi_get_addm_findings",
            "oci_opsi_get_addm_recommendations", "oci_opsi_get_capacity_trend",
            "oci_opsi_get_capacity_forecast", "oci_opsi_list_awr_hubs"
        ],
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
    "system": {
        "description": "Health checks and low-risk server metadata",
        "tools": ["oci_ping"],
    },
    "selectai": {
        "description": "Natural language to SQL, database chat, and AI agent orchestration",
        "tools": [
            "oci_selectai_generate", "oci_selectai_list_profiles",
            "oci_selectai_get_profile_tables", "oci_selectai_run_agent",
            "oci_selectai_ping"
        ],
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


@mcp.tool()
async def oci_ping() -> dict:
    """Lightweight health check for the unified MCP server."""
    return {
        "status": "ok",
        "server": "oci-unified",
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }


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
from src.mcp.server.tools.database import register_database_tools
from src.mcp.server.tools.discovery import register_discovery_tools
from src.mcp.server.tools.identity import register_identity_tools
from src.mcp.server.tools.logan import register_logan_tools
from src.mcp.server.tools.network import register_network_tools
from src.mcp.server.tools.observability import register_observability_tools
from src.mcp.server.tools.opsi import register_opsi_tools
from src.mcp.server.tools.security import register_security_tools
from src.mcp.server.tools.selectai import register_selectai_tools

# Register tools
register_identity_tools(mcp)  # Register identity tools first for compartment discovery
register_compute_tools(mcp)
register_network_tools(mcp)
register_cost_tools(mcp)
register_security_tools(mcp)
register_observability_tools(mcp)
register_logan_tools(mcp)  # Log Analytics tools
register_database_tools(mcp)  # DB Management tools
register_opsi_tools(mcp)  # Operations Insights tools
register_discovery_tools(mcp)  # ShowOCI-style discovery tools
register_selectai_tools(mcp)  # SelectAI NL2SQL tools

# Register skills
register_troubleshoot_skills(mcp)

if __name__ == "__main__":
    transport = os.getenv("TRANSPORT", "stdio")
    mcp.run(transport=transport)
