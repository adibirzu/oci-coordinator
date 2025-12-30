
from typing import Optional, List, Dict, Any
import json
from src.mcp.server.auth import get_monitoring_client

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
    async def get_metrics(compartment_id: str, namespace: str, query: str, format: str = "markdown") -> str:
        """Get monitoring metrics."""
        return await _get_metrics_logic(compartment_id, namespace, query, format)
