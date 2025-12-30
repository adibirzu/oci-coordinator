
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

@pytest.mark.asyncio
async def test_troubleshoot_instance_logic():
    """Verify troubleshoot_instance skill logic."""
    from src.mcp.server.skills.troubleshoot import _troubleshoot_instance_logic
    
    mock_compute = MagicMock()
    mock_instance = MagicMock()
    mock_instance.display_name = "broken-instance"
    mock_instance.lifecycle_state = "STOPPED"
    mock_compute.get_instance.return_value = MagicMock(data=mock_instance)
    
    with patch("src.mcp.server.skills.troubleshoot.get_compute_client", return_value=mock_compute):
        result = await _troubleshoot_instance_logic(instance_id="ocid1.instance.123")
        assert "broken-instance" in result
        assert "STOPPED" in result
        assert "Root Cause Analysis" in result

@pytest.mark.asyncio
async def test_list_users_logic():
    """Verify list_users tool logic."""
    from src.mcp.server.tools.security import _list_users_logic
    
    mock_identity = MagicMock()
    mock_user = MagicMock()
    mock_user.name = "test-user"
    mock_user.id = "ocid1.user.123"
    mock_user.lifecycle_state = "ACTIVE"
    
    mock_identity.list_users.return_value = MagicMock(data=[mock_user])
    
    with patch("src.mcp.server.tools.security.get_identity_client", return_value=mock_identity):
        result = await _list_users_logic(compartment_id="test-comp", format="markdown")
        assert "test-user" in result
        assert "ACTIVE" in result

@pytest.mark.asyncio
async def test_get_metrics_logic():
    """Verify get_metrics tool logic."""
    from src.mcp.server.tools.observability import _get_metrics_logic
    
    mock_monitoring = MagicMock()
    
    with patch("src.mcp.server.tools.observability.get_monitoring_client", return_value=mock_monitoring):
        result = await _get_metrics_logic(compartment_id="test-comp", namespace="oci_computeagent", query="CpuUtilization[1m].mean()")
        assert "85% CPU Usage" in result
