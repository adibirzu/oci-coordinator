
import pytest
from unittest.mock import MagicMock, patch
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
async def test_list_instances_logic():
    """Verify list_instances tool logic."""
    from src.mcp.server.tools.compute import _list_instances_logic
    
    mock_compute = MagicMock()
    mock_instance = MagicMock()
    mock_instance.display_name = "test-instance"
    mock_instance.id = "ocid1.instance.123"
    mock_instance.lifecycle_state = "RUNNING"
    mock_instance.shape = "VM.Standard.E4.Flex"
    
    mock_response = MagicMock()
    mock_response.data = [mock_instance]
    mock_compute.list_instances.return_value = mock_response
    
    with patch("src.mcp.server.tools.compute.get_compute_client", return_value=mock_compute):
        result = await _list_instances_logic(compartment_id="test-comp", format="markdown")
        assert "test-instance" in result
        assert "RUNNING" in result
        assert "| Name |" in result

@pytest.mark.asyncio
async def test_list_vcns_logic():
    """Verify list_vcns tool logic."""
    from src.mcp.server.tools.network import _list_vcns_logic
    
    mock_network = MagicMock()
    mock_vcn = MagicMock()
    mock_vcn.display_name = "test-vcn"
    mock_vcn.id = "ocid1.vcn.123"
    mock_vcn.lifecycle_state = "AVAILABLE"
    mock_vcn.cidr_block = "10.0.0.0/16"
    
    mock_response = MagicMock()
    mock_response.data = [mock_vcn]
    mock_network.list_vcns.return_value = mock_response
    
    with patch("src.mcp.server.tools.network.get_network_client", return_value=mock_network):
        result = await _list_vcns_logic(compartment_id="test-comp", format="markdown")
        assert "test-vcn" in result
        assert "AVAILABLE" in result
        assert "10.0.0.0/16" in result

@pytest.mark.asyncio
async def test_get_cost_summary_logic():
    """Verify get_cost_summary tool logic."""
    from src.mcp.server.tools.cost import _get_cost_summary_logic
    
    mock_usage = MagicMock()
    # Mock the return value to have .data.items
    mock_item = MagicMock()
    mock_item.computed_amount = 100.5
    mock_item.currency = "USD"
    
    mock_collection = MagicMock()
    mock_collection.items = [mock_item]
    
    mock_response = MagicMock()
    mock_response.data = mock_collection
    mock_usage.request_summarized_usages.return_value = mock_response
    
    with patch("src.mcp.server.tools.cost.get_usage_client", return_value=mock_usage):
        result = await _get_cost_summary_logic(compartment_id="test-comp", format="markdown")
        assert "100.5" in result
        assert "USD" in result
