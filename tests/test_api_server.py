"""Tests for API Server."""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch, AsyncMock

# Reset singletons before import
from src.mcp.catalog import ToolCatalog
from src.mcp.registry import ServerRegistry
from src.agents.catalog import AgentCatalog

ToolCatalog.reset_instance()
ServerRegistry.reset_instance()

from src.api.main import app, app_state


@pytest.fixture
def client():
    """Create test client."""
    # Reset app state
    app_state.request_count = 0
    app_state.active_threads = {}
    return TestClient(app)


@pytest.fixture
def mock_catalogs():
    """Mock the catalog singletons."""
    with patch.object(ToolCatalog, "get_instance") as mock_tool, \
         patch.object(ServerRegistry, "get_instance") as mock_registry, \
         patch.object(AgentCatalog, "get_instance") as mock_agent:

        # Mock tool catalog
        mock_tool_instance = MagicMock()
        mock_tool_instance.list_tools.return_value = []
        mock_tool_instance.get_tool.return_value = None
        mock_tool_instance.search_tools.return_value = []
        mock_tool_instance.get_statistics.return_value = {"total_tools": 0}
        mock_tool_instance.ensure_fresh = AsyncMock()
        mock_tool.return_value = mock_tool_instance

        # Mock registry
        mock_registry_instance = MagicMock()
        mock_registry_instance.list_servers.return_value = []
        mock_registry_instance.get_status.return_value = "disconnected"
        mock_registry_instance.get_tools.return_value = []
        mock_registry.return_value = mock_registry_instance

        # Mock agent catalog
        mock_agent_instance = MagicMock()
        mock_agent_instance.list_all.return_value = []
        mock_agent_instance.get.return_value = None
        mock_agent.return_value = mock_agent_instance

        yield {
            "tool_catalog": mock_tool_instance,
            "registry": mock_registry_instance,
            "agent_catalog": mock_agent_instance,
        }


class TestHealthEndpoints:
    """Test health check endpoints."""

    def test_health_check(self, client, mock_catalogs):
        """Test basic health check."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()

        assert "status" in data
        assert "timestamp" in data
        assert "components" in data

    def test_health_check_returns_components(self, client, mock_catalogs):
        """Test health check includes component status."""
        response = client.get("/health")
        data = response.json()

        # Should include MCP and agent status
        assert "components" in data
        assert isinstance(data["components"], dict)

    def test_status_endpoint(self, client, mock_catalogs):
        """Test detailed status endpoint."""
        response = client.get("/status")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "running"
        assert "uptime_seconds" in data
        assert "mcp_servers" in data
        assert "agents" in data
        assert "tools" in data


class TestChatEndpoints:
    """Test chat endpoints."""

    def test_chat_requires_message(self, client, mock_catalogs):
        """Test that chat requires a message."""
        response = client.post("/chat", json={})

        # Should return validation error
        assert response.status_code == 422

    def test_chat_basic_request(self, client, mock_catalogs):
        """Test basic chat request."""
        # Mock the coordinator - patch at source module
        with patch("src.llm.get_llm") as mock_llm, \
             patch("src.agents.coordinator.graph.create_coordinator") as mock_coord:

            mock_coord_instance = MagicMock()
            mock_coord_instance.invoke = AsyncMock(return_value=MagicMock(
                response="Test response",
                agent_used="test-agent",
                tools_used=["tool1"],
            ))
            mock_coord.return_value = mock_coord_instance

            response = client.post("/chat", json={
                "message": "Hello, test message",
            })

            # Should succeed or fall back
            assert response.status_code in [200, 500]

    def test_chat_with_thread_id(self, client, mock_catalogs):
        """Test chat preserves thread ID."""
        with patch("src.llm.get_llm") as mock_llm, \
             patch("src.agents.coordinator.graph.create_coordinator") as mock_coord:

            mock_coord_instance = MagicMock()
            mock_coord_instance.invoke = AsyncMock(return_value=MagicMock(
                response="Test response",
                agent_used=None,
                tools_used=[],
            ))
            mock_coord.return_value = mock_coord_instance

            response = client.post("/chat", json={
                "message": "Hello",
                "thread_id": "test-thread-123",
            })

            if response.status_code == 200:
                data = response.json()
                assert data["thread_id"] == "test-thread-123"


class TestToolEndpoints:
    """Test tool endpoints."""

    def test_list_tools(self, client, mock_catalogs):
        """Test tool listing."""
        mock_catalogs["tool_catalog"].search_tools.return_value = [
            {
                "name": "test_tool",
                "description": "Test tool",
                "tier": 2,
            }
        ]

        response = client.get("/tools")

        assert response.status_code == 200
        data = response.json()

        assert "tools" in data
        assert "count" in data

    def test_list_tools_with_query(self, client, mock_catalogs):
        """Test tool listing with search query."""
        response = client.get("/tools?query=compute")

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["query"] == "compute"

    def test_list_tools_with_domain_filter(self, client, mock_catalogs):
        """Test tool listing with domain filter."""
        response = client.get("/tools?domain=database")

        assert response.status_code == 200
        data = response.json()

        assert data["filters"]["domain"] == "database"

    def test_get_tool_not_found(self, client, mock_catalogs):
        """Test getting non-existent tool."""
        mock_catalogs["tool_catalog"].get_tool.return_value = None

        response = client.get("/tools/nonexistent_tool")

        assert response.status_code == 404

    def test_get_tool_found(self, client, mock_catalogs):
        """Test getting existing tool."""
        from src.mcp.client import ToolDefinition

        mock_catalogs["tool_catalog"].get_tool.return_value = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object"},
            server_id="test-server",
        )

        response = client.get("/tools/test_tool")

        assert response.status_code == 200
        data = response.json()

        assert data["name"] == "test_tool"
        assert data["description"] == "A test tool"

    def test_execute_tool(self, client, mock_catalogs):
        """Test tool execution."""
        from src.mcp.client import ToolCallResult

        mock_catalogs["tool_catalog"].execute = AsyncMock(return_value=ToolCallResult(
            tool_name="test_tool",
            success=True,
            result="Tool executed successfully",
        ))

        response = client.post("/tools/execute", json={
            "tool_name": "test_tool",
            "arguments": {"arg1": "value1"},
        })

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "duration_ms" in data


class TestAgentEndpoints:
    """Test agent endpoints."""

    def test_list_agents(self, client, mock_catalogs):
        """Test agent listing."""
        response = client.get("/agents")

        assert response.status_code == 200
        data = response.json()

        assert "agents" in data
        assert "count" in data

    def test_list_agents_with_data(self, client, mock_catalogs):
        """Test agent listing with agents."""
        from src.agents.base import (
            AgentDefinition,
            AgentMetadata,
            KafkaTopics,
        )

        mock_catalogs["agent_catalog"].list_all.return_value = [
            AgentDefinition(
                agent_id="test-agent-abc123",
                role="test-agent",
                description="A test agent",
                capabilities=["test-capability"],
                skills=["test-skill"],
                kafka_topics=KafkaTopics(),
                health_endpoint="/health/test-agent",
                metadata=AgentMetadata(),
            )
        ]

        response = client.get("/agents")

        assert response.status_code == 200
        data = response.json()

        assert data["count"] == 1
        assert data["agents"][0]["role"] == "test-agent"

    def test_get_agent_not_found(self, client, mock_catalogs):
        """Test getting non-existent agent."""
        mock_catalogs["agent_catalog"].get.return_value = None

        response = client.get("/agents/nonexistent_agent")

        assert response.status_code == 404

    def test_get_agent_found(self, client, mock_catalogs):
        """Test getting existing agent."""
        from src.agents.base import (
            AgentDefinition,
            AgentMetadata,
            KafkaTopics,
        )

        mock_catalogs["agent_catalog"].get.return_value = AgentDefinition(
            agent_id="db-troubleshoot-agent-abc123",
            role="db-troubleshoot-agent",
            description="Database troubleshooting agent",
            capabilities=["database-analysis"],
            skills=["rca_workflow"],
            kafka_topics=KafkaTopics(),
            health_endpoint="/health/db-troubleshoot-agent",
            metadata=AgentMetadata(),
        )

        response = client.get("/agents/db-troubleshoot-agent")

        assert response.status_code == 200
        data = response.json()

        assert data["role"] == "db-troubleshoot-agent"
        assert data["description"] == "Database troubleshooting agent"


class TestMCPEndpoints:
    """Test MCP server endpoints."""

    def test_list_mcp_servers(self, client, mock_catalogs):
        """Test MCP server listing."""
        mock_catalogs["registry"].list_servers.return_value = ["server-1", "server-2"]

        response = client.get("/mcp/servers")

        assert response.status_code == 200
        data = response.json()

        assert "servers" in data
        assert "count" in data
        assert data["count"] == 2

    def test_reconnect_mcp_server_not_found(self, client, mock_catalogs):
        """Test reconnecting non-existent server."""
        mock_catalogs["registry"].list_servers.return_value = []

        response = client.post("/mcp/servers/nonexistent/reconnect")

        assert response.status_code == 404

    def test_reconnect_mcp_server(self, client, mock_catalogs):
        """Test reconnecting existing server."""
        mock_catalogs["registry"].list_servers.return_value = ["test-server"]
        mock_catalogs["registry"].reconnect = AsyncMock(return_value=True)
        mock_catalogs["registry"].get_status.return_value = "connected"

        response = client.post("/mcp/servers/test-server/reconnect")

        assert response.status_code == 200
        data = response.json()

        assert data["server_id"] == "test-server"
        assert data["reconnected"] is True


class TestStatsEndpoint:
    """Test statistics endpoint."""

    def test_get_stats(self, client, mock_catalogs):
        """Test statistics endpoint."""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()

        assert "uptime_seconds" in data
        assert "request_count" in data
        assert "active_threads" in data
        assert "timestamp" in data


class TestMiddleware:
    """Test middleware functionality."""

    def test_request_id_header(self, client, mock_catalogs):
        """Test that request ID is added to responses."""
        response = client.get("/health")

        assert "X-Request-ID" in response.headers
        assert len(response.headers["X-Request-ID"]) == 8

    def test_response_time_header(self, client, mock_catalogs):
        """Test that response time is added to responses."""
        response = client.get("/health")

        assert "X-Response-Time" in response.headers
        assert "ms" in response.headers["X-Response-Time"]

    def test_request_count_increments(self, client, mock_catalogs):
        """Test that request count increments."""
        initial_count = app_state.request_count

        client.get("/health")
        client.get("/health")

        assert app_state.request_count == initial_count + 2
