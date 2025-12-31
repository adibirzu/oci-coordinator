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

from src.mcp.server.tools.identity import register_identity_tools
from src.mcp.server.tools.compute import register_compute_tools
from src.mcp.server.tools.network import register_network_tools
from src.mcp.server.tools.cost import register_cost_tools
from src.mcp.server.tools.security import register_security_tools
from src.mcp.server.tools.observability import register_observability_tools
from src.mcp.server.tools.discovery import register_discovery_tools
from src.mcp.server.skills.troubleshoot import register_troubleshoot_skills

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
