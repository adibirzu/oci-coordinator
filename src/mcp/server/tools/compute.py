from typing import Optional, List, Dict, Any
import json
from src.mcp.server.auth import get_compute_client

async def _list_instances_logic(
    compartment_id: str, 
    limit: int = 20, 
    format: str = "markdown"
) -> str:
    """Internal logic for listing instances."""
    client = get_compute_client()
    
    try:
        response = client.list_instances(compartment_id=compartment_id, limit=limit)
        instances = response.data
        
        if format == "json":
            # Simple list of dicts for JSON
            return json.dumps([
                {
                    "name": i.display_name,
                    "id": i.id,
                    "state": i.lifecycle_state,
                    "shape": i.shape
                } for i in instances
            ], indent=2)
            
        # Markdown table for LLM efficiency
        lines = ["| Name | State | Shape | OCID |\n", "| --- | --- | --- | --- |"]
        for i in instances:
            lines.append(f"| {i.display_name} | {i.lifecycle_state} | {i.shape} | `{i.id}` |")
            
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error listing instances: {e}"

def register_compute_tools(mcp):
    """Register compute tools with the MCP server."""
    
    @mcp.tool()
    async def list_instances(
        compartment_id: str, 
        limit: int = 20, 
        format: str = "markdown"
    ) -> str:
        """List compute instances in a compartment. 
        
        Args:
            compartment_id: OCID of the compartment
            limit: Maximum number of items to return (default 20)
            format: Output format ('json' or 'markdown')
        """
        return await _list_instances_logic(compartment_id, limit, format)
