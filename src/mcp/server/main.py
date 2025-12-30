from fastmcp import FastMCP
import logging
from typing import Optional

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP(
    name="oci-unified-server",
    instructions="""Oracle Cloud Infrastructure MCP Server providing comprehensive 
cloud management capabilities through the Model Context Protocol.

Use `search_capabilities` to discover available tools.
"""
)

DOMAINS = {
    "compute": {
        "description": "Instance management, shapes, and performance metrics",
        "tools": ["list_instances", "start_instance", "stop_instance", "restart_instance"],
    },
    "cost": {
        "description": "Cost analysis, budgeting, and FinOps optimization",
        "tools": ["get_cost_summary", "get_cost_by_service"],
    },
    "db": {
        "description": "Autonomous Database and DB Systems management",
        "tools": ["list_autonomous_db", "get_db_metrics"],
    },
    "network": {
        "description": "VCN, Subnet, and Security List management",
        "tools": ["list_vcns", "list_subnets"],
    },
    "security": {
        "description": "IAM and Cloud Guard management",
        "tools": ["list_users", "list_policies"],
    },
    "observability": {
        "description": "Logs, metrics, and alarms",
        "tools": ["get_metrics", "query_logs"],
    },
}

async def _search_capabilities_logic(query: str, domain: Optional[str] = None) -> str:
    """Internal logic for search_capabilities."""
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
            
    if not results:
        return f"No domains or tools found matching '{query}'. Available domains: {', '.join(DOMAINS.keys())}"
        
    return "# Matching OCI Capabilities\n\n" + "\n".join(results)

@mcp.tool()
async def search_capabilities(query: str, domain: Optional[str] = None) -> str:
    """Search for OCI tool domains and capabilities.
    
    Use this to discover available tools matching your intent.
    
    Args:
        query: Search keywords (e.g. 'how to troubleshoot', 'cost', 'instances')
        domain: Optional filter by domain (compute, cost, db, network, security, observability)
    """
    return await _search_capabilities_logic(query, domain)

from src.mcp.server.tools.compute import register_compute_tools
from src.mcp.server.tools.network import register_network_tools

# Register tools
register_compute_tools(mcp)
register_network_tools(mcp)

if __name__ == "__main__":
    mcp.run()
