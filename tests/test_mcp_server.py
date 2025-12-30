import pytest
from src.mcp.server.main import mcp

def test_mcp_server_instantiation():
    """Verify that the FastMCP server can be instantiated."""
    assert mcp is not None
    assert mcp.name == "oci-unified-server"