from typing import Optional, List, Dict, Any
import json
from src.mcp.server.auth import get_identity_client

async def _list_users_logic(
    compartment_id: str,
    limit: int = 20,
    format: str = "markdown"
) -> str:
    """Internal logic for listing users."""
    client = get_identity_client()
    
    try:
        response = client.list_users(compartment_id=compartment_id, limit=limit)
        users = response.data
        
        if format == "json":
            return json.dumps([{"name": u.name, "id": u.id, "state": u.lifecycle_state} for u in users], indent=2)
            
        lines = ["| Name | State | OCID |", "| --- | --- | --- |"]
        for u in users:
            lines.append(f"| {u.name} | {u.lifecycle_state} | `{u.id}` |")
            
        return "\n".join(lines)
        
    except Exception as e:
        return f"Error listing users: {e}"

def register_security_tools(mcp):
    @mcp.tool()
    async def list_users(compartment_id: str, limit: int = 20, format: str = "markdown") -> str:
        """List IAM users."""
        return await _list_users_logic(compartment_id, limit, format)
