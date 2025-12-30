
from fastmcp import FastMCP
import logging

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

@mcp.tool()
async def search_capabilities(query: str) -> str:
    """Search for available tools and skills matching the intent.
    
    Args:
        query: Natural language intent (e.g. 'how to troubleshoot an instance')
    """
    # This will be implemented in the next task
    return "Available domains: Compute, Network, DB, Cost, Security, Observability. Use specialized tools for each."

if __name__ == "__main__":
    mcp.run()
