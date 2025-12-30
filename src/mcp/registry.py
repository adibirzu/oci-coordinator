
from typing import Dict, Any, Optional
import asyncio

class ServerRegistry:
    """Registry for MCP servers."""
    
    def __init__(self):
        self.servers: Dict[str, Any] = {}
        
    def register_server(self, server_id: str, client: Any):
        """Register a server client."""
        self.servers[server_id] = client
        
    def get_client(self, server_id: str) -> Optional[Any]:
        """Get a server client."""
        return self.servers.get(server_id)
