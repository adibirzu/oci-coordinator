
import pytest
from src.mcp.server.main import mcp, _search_capabilities_logic

def test_mcp_server_instantiation():
    """Verify that the FastMCP server can be instantiated."""
    assert mcp is not None
    assert mcp.name == "oci-unified-server"

@pytest.mark.asyncio
async def test_search_capabilities_found():
    """Verify that search_capabilities returns tool information."""
    result = await _search_capabilities_logic("compute")
    assert "Domain: compute" in result
    assert "list_instances" in result

@pytest.mark.asyncio
async def test_search_capabilities_not_found():
    """Verify that search_capabilities handles no results."""
    result = await _search_capabilities_logic("nonexistent")
    assert "No domains or tools found" in result
