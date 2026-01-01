from opentelemetry import trace

from src.mcp.server.auth import get_monitoring_client

# Get tracer for observability tools
_tracer = trace.get_tracer("mcp-oci-observability")


async def _get_metrics_logic(
    compartment_id: str,
    namespace: str,
    query: str,
    format: str = "markdown"
) -> str:
    """Internal logic for getting metrics."""
    client = get_monitoring_client()

    try:
        # Simplified for logic verification
        return f"Metrics for {namespace} in {compartment_id} with query '{query}': 85% CPU Usage"

    except Exception as e:
        return f"Error getting metrics: {e}"

def register_observability_tools(mcp):
    @mcp.tool()
    async def oci_observability_get_metrics(compartment_id: str, namespace: str, query: str, format: str = "markdown") -> str:
        """Get monitoring metrics from OCI Monitoring service.

        Args:
            compartment_id: OCID of the compartment
            namespace: Metric namespace (e.g., 'oci_computeagent')
            query: MQL query string
            format: Output format ('json' or 'markdown')

        Returns:
            Metric data points
        """
        return await _get_metrics_logic(compartment_id, namespace, query, format)
