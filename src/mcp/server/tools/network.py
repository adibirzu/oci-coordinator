from typing import Optional, List, Dict, Any
import json
from src.mcp.server.auth import get_network_client

async def _list_vcns_logic(
    compartment_id: str, 
    limit: int = 20, 
    format: str = "markdown"
) -> str:
    """Internal logic for listing VCNs."""
    client = get_network_client()
    
    try:
        response = client.list_vcns(compartment_id=compartment_id, limit=limit)
        vcns = response.data
        
        if format == "json":
            return json.dumps([
                {
                    "name": v.display_name,
                    "id": v.id,
                    "cidr": v.cidr_block,
                    "state": v.lifecycle_state
                } for v in vcns
            ], indent=2)
            
        lines = ["| Name | State | CIDR | OCID |", "| --- | --- | --- | --- |"]
        for v in vcns:
            lines.append(f"| {v.display_name} | {v.lifecycle_state} | {v.cidr_block} | `{v.id}` |")
            
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error listing VCNs: {e}"

def register_network_tools(mcp):
    """Register network tools with the MCP server."""
    
    @mcp.tool()
    async def list_vcns(
        compartment_id: str, 
        limit: int = 20, 
        format: str = "markdown"
    ) -> str:
        """List virtual cloud networks (VCNs) in a compartment. 
        
        Args:
            compartment_id: OCID of the compartment
            limit: Maximum number of items to return (default 20)
            format: Output format ('json' or 'markdown')
        """
        return await _list_vcns_logic(compartment_id, limit, format)
