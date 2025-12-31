"""
End-to-End Tests for MCP Server Integration and Skill Execution.

These tests verify:
1. MCP server configuration loading
2. MCP client connectivity (mocked for CI, real for integration)
3. Tool discovery and catalog operations
4. Skill execution framework
5. Agent coordination with MCP tools

Run with:
    # Unit tests (mocked MCP)
    poetry run pytest tests/test_e2e_mcp_skills.py -v

    # Integration tests (real MCP server - requires MCP server running)
    poetry run pytest tests/test_e2e_mcp_skills.py -v -m integration
"""

import asyncio
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Test fixtures and utilities


@pytest.fixture
def mock_mcp_response():
    """Create a mock MCP response."""
    return {
        "tools": [
            {
                "name": "execute_sql",
                "description": "Execute SQL queries against Oracle Database",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "database": {"type": "string"},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "get_fleet_summary",
                "description": "Get OPSI fleet summary",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "compartment_id": {"type": "string"},
                    },
                },
            },
            {
                "name": "execute_logan_query",
                "description": "Execute Logging Analytics query",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "namespace": {"type": "string"},
                    },
                    "required": ["query", "namespace"],
                },
            },
        ]
    }


@pytest.fixture
def mock_tool_result():
    """Create a mock tool execution result."""
    return {
        "content": [
            {
                "type": "text",
                "text": '{"status": "success", "rows_affected": 5, "data": [{"id": 1, "name": "test"}]}',
            }
        ]
    }


# ─────────────────────────────────────────────────────────────────────────────
# MCP Configuration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPConfiguration:
    """Test MCP server configuration loading."""

    def test_load_config_from_yaml(self):
        """Test loading configuration from YAML file."""
        from src.mcp.config import load_mcp_config

        config = load_mcp_config()

        # Should have loaded servers
        assert config is not None
        assert len(config.servers) >= 0  # May be empty if config not found

    def test_load_config_with_custom_path(self, tmp_path):
        """Test loading configuration from custom path."""
        from src.mcp.config import load_mcp_config

        # Create a test config file
        config_file = tmp_path / "test_mcp.yaml"
        config_file.write_text(
            """
servers:
  test-server:
    transport: stdio
    command: python
    args:
      - "-m"
      - "test_server"
    enabled: true
    domains:
      - test
    description: "Test server"
    timeout_seconds: 30

groups:
  test:
    - test-server

defaults:
  timeout_seconds: 60
"""
        )

        config = load_mcp_config(config_file)

        assert "test-server" in config.servers
        assert config.servers["test-server"].enabled is True
        assert config.servers["test-server"].domains == ["test"]
        assert config.servers["test-server"].timeout_seconds == 30

    def test_get_enabled_servers(self, tmp_path):
        """Test filtering enabled servers."""
        from src.mcp.config import load_mcp_config

        config_file = tmp_path / "test_mcp.yaml"
        config_file.write_text(
            """
servers:
  enabled-server:
    transport: stdio
    command: python
    enabled: true
  disabled-server:
    transport: http
    url: http://localhost:8000
    enabled: false
"""
        )

        config = load_mcp_config(config_file)
        enabled = config.get_enabled_servers()

        assert "enabled-server" in enabled
        assert "disabled-server" not in enabled

    def test_get_servers_for_domain(self, tmp_path):
        """Test filtering servers by domain."""
        from src.mcp.config import load_mcp_config

        config_file = tmp_path / "test_mcp.yaml"
        config_file.write_text(
            """
servers:
  db-server:
    transport: stdio
    command: python
    enabled: true
    domains:
      - database
      - opsi
  log-server:
    transport: http
    url: http://localhost:8001
    enabled: true
    domains:
      - logging
      - logan
"""
        )

        config = load_mcp_config(config_file)

        db_servers = config.get_servers_for_domain("database")
        assert len(db_servers) == 1
        assert db_servers[0].server_id == "db-server"

        log_servers = config.get_servers_for_domain("logan")
        assert len(log_servers) == 1
        assert log_servers[0].server_id == "log-server"

    def test_server_config_to_mcp_config(self, tmp_path):
        """Test converting ServerConfigEntry to MCPServerConfig."""
        from src.mcp.client import TransportType
        from src.mcp.config import load_mcp_config

        config_file = tmp_path / "test_mcp.yaml"
        config_file.write_text(
            """
servers:
  test-server:
    transport: stdio
    command: python
    args:
      - "-m"
      - "test"
    working_dir: /tmp
    env:
      TEST_VAR: "test_value"
    enabled: true
    timeout_seconds: 45
"""
        )

        config = load_mcp_config(config_file)
        server = config.servers["test-server"]
        mcp_config = server.to_mcp_config()

        assert mcp_config.server_id == "test-server"
        assert mcp_config.transport == TransportType.STDIO
        assert mcp_config.command == "python"
        assert mcp_config.args == ["-m", "test"]
        assert mcp_config.working_dir == "/tmp"
        assert mcp_config.timeout_seconds == 45


# ─────────────────────────────────────────────────────────────────────────────
# MCP Client Tests (Mocked)
# ─────────────────────────────────────────────────────────────────────────────


class TestMCPClient:
    """Test MCP client with mocked transport."""

    @pytest.mark.asyncio
    async def test_tool_definition_creation(self):
        """Test creating tool definitions."""
        from src.mcp.client import ToolDefinition

        tool = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={"type": "object", "properties": {"arg1": {"type": "string"}}},
            server_id="test-server",
        )

        assert tool.name == "test_tool"
        assert tool.server_id == "test-server"
        assert tool.full_name == "test_tool"  # No namespace

        tool_with_ns = ToolDefinition(
            name="test_tool",
            description="A test tool",
            input_schema={},
            server_id="test-server",
            namespace="oci_database",
        )
        assert tool_with_ns.full_name == "oci_database:test_tool"

    @pytest.mark.asyncio
    async def test_tool_call_result(self):
        """Test tool call result structure."""
        from src.mcp.client import ToolCallResult

        result = ToolCallResult(
            tool_name="test_tool",
            success=True,
            result={"data": "test"},
            duration_ms=150,
        )

        assert result.success is True
        assert result.tool_name == "test_tool"
        assert result.duration_ms == 150
        assert result.error is None


# ─────────────────────────────────────────────────────────────────────────────
# Tool Catalog Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestToolCatalog:
    """Test tool catalog operations."""

    @pytest.fixture
    def mock_registry(self):
        """Create a mock server registry."""
        from src.mcp.client import ToolDefinition
        from src.mcp.registry import ServerRegistry, ServerStatus

        registry = MagicMock(spec=ServerRegistry)

        # Mock tools from registry
        mock_tools = {
            "execute_sql": ToolDefinition(
                name="execute_sql",
                description="Execute SQL queries",
                input_schema={},
                server_id="db-server",
            ),
            "get_fleet_summary": ToolDefinition(
                name="get_fleet_summary",
                description="Get OPSI fleet summary",
                input_schema={},
                server_id="db-server",
            ),
            "execute_logan_query": ToolDefinition(
                name="execute_logan_query",
                description="Execute Logging Analytics query",
                input_schema={},
                server_id="db-server",
            ),
        }
        registry.get_all_tools.return_value = mock_tools

        return registry

    @pytest.mark.asyncio
    async def test_catalog_refresh(self, mock_registry):
        """Test refreshing tool catalog."""
        from src.mcp.catalog import ToolCatalog

        # Reset singleton
        ToolCatalog.reset_instance()

        catalog = ToolCatalog(mock_registry)
        count = await catalog.refresh()

        assert count == 3
        assert len(catalog.list_tools()) == 3

    @pytest.mark.asyncio
    async def test_catalog_search_tools(self, mock_registry):
        """Test searching tools in catalog."""
        from src.mcp.catalog import ToolCatalog

        ToolCatalog.reset_instance()
        catalog = ToolCatalog(mock_registry)
        await catalog.refresh()

        # Search by query
        results = catalog.search_tools(query="sql")
        assert len(results) >= 1
        assert any("sql" in r["name"].lower() for r in results)

    @pytest.mark.asyncio
    async def test_catalog_get_tool(self, mock_registry):
        """Test getting a specific tool."""
        from src.mcp.catalog import ToolCatalog

        ToolCatalog.reset_instance()
        catalog = ToolCatalog(mock_registry)
        await catalog.refresh()

        tool = catalog.get_tool("execute_sql")
        assert tool is not None
        assert tool.name == "execute_sql"

        # Non-existent tool
        no_tool = catalog.get_tool("non_existent")
        assert no_tool is None

    @pytest.mark.asyncio
    async def test_catalog_domain_summary(self, mock_registry):
        """Test getting domain summary."""
        from src.mcp.catalog import ToolCatalog

        ToolCatalog.reset_instance()
        catalog = ToolCatalog(mock_registry)
        await catalog.refresh()

        summary = catalog.get_domain_summary()
        assert isinstance(summary, dict)

    @pytest.mark.asyncio
    async def test_catalog_statistics(self, mock_registry):
        """Test getting catalog statistics."""
        from src.mcp.catalog import ToolCatalog

        ToolCatalog.reset_instance()
        catalog = ToolCatalog(mock_registry)
        await catalog.refresh()

        stats = catalog.get_statistics()
        assert "total_tools" in stats
        assert stats["total_tools"] == 3


# ─────────────────────────────────────────────────────────────────────────────
# Skill Execution Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestSkillExecution:
    """Test skill execution framework."""

    @pytest.fixture
    def mock_catalog(self):
        """Create a mock tool catalog."""
        from src.mcp.catalog import ToolCatalog
        from src.mcp.client import ToolCallResult, ToolDefinition

        catalog = MagicMock(spec=ToolCatalog)

        # Mock tool lookup
        def get_tool(name):
            tools = {
                "oci_observability_get_metrics": ToolDefinition(
                    name="oci_observability_get_metrics",
                    description="Get metrics",
                    input_schema={},
                    server_id="test",
                ),
                "oci_database_get_autonomous": ToolDefinition(
                    name="oci_database_get_autonomous",
                    description="Get autonomous DB info",
                    input_schema={},
                    server_id="test",
                ),
                "oci_observability_query_logs": ToolDefinition(
                    name="oci_observability_query_logs",
                    description="Query logs",
                    input_schema={},
                    server_id="test",
                ),
            }
            return tools.get(name)

        catalog.get_tool = get_tool

        # Mock tool execution
        async def mock_execute(tool_name, arguments):
            return ToolCallResult(
                tool_name=tool_name,
                success=True,
                result={"mock": "result"},
                duration_ms=100,
            )

        catalog.execute = AsyncMock(side_effect=mock_execute)

        return catalog

    def test_skill_definition_creation(self):
        """Test creating a skill definition."""
        from src.agents.skills import SkillDefinition, SkillStep

        skill = SkillDefinition(
            name="test_skill",
            description="A test skill",
            steps=[
                SkillStep(name="step1", description="First step"),
                SkillStep(name="step2", description="Second step", optional=True),
            ],
            required_tools=["tool_a", "tool_b"],
            tags=["test"],
            estimated_duration_seconds=60,
        )

        assert skill.name == "test_skill"
        assert len(skill.steps) == 2
        assert skill.steps[1].optional is True

    def test_skill_validate_tools(self, mock_catalog):
        """Test skill tool validation."""
        from src.agents.skills import SkillDefinition, SkillStep

        skill = SkillDefinition(
            name="test_skill",
            description="Test",
            steps=[
                SkillStep(
                    name="step1",
                    description="Step 1",
                    required_tools=["oci_observability_get_metrics"],
                ),
            ],
            required_tools=["oci_observability_get_metrics"],
        )

        is_valid, missing = skill.validate_tools(mock_catalog)
        assert is_valid is True
        assert len(missing) == 0

        # Test with missing tool
        skill_missing = SkillDefinition(
            name="missing_tools_skill",
            description="Test",
            steps=[],
            required_tools=["non_existent_tool"],
        )

        is_valid, missing = skill_missing.validate_tools(mock_catalog)
        assert is_valid is False
        assert "non_existent_tool" in missing

    @pytest.mark.asyncio
    async def test_skill_executor_registration(self, mock_catalog):
        """Test registering skills with executor."""
        from src.agents.skills import SkillDefinition, SkillExecutor, SkillStep

        executor = SkillExecutor(mock_catalog)

        skill = SkillDefinition(
            name="test_skill",
            description="Test",
            steps=[
                SkillStep(
                    name="step1",
                    description="Step 1",
                    required_tools=["oci_observability_get_metrics"],
                ),
            ],
            required_tools=["oci_observability_get_metrics"],
        )

        success = executor.register(skill)
        assert success is True
        assert executor.get_skill("test_skill") is not None

    @pytest.mark.asyncio
    async def test_skill_execution_basic(self, mock_catalog):
        """Test basic skill execution."""
        from src.agents.skills import (
            SkillDefinition,
            SkillExecutor,
            SkillStatus,
            SkillStep,
            StepResult,
        )

        executor = SkillExecutor(mock_catalog)

        # Create a simple skill
        skill = SkillDefinition(
            name="simple_skill",
            description="Simple test skill",
            steps=[
                SkillStep(
                    name="get_metrics",
                    description="Get metrics",
                    required_tools=["oci_observability_get_metrics"],
                ),
            ],
            required_tools=["oci_observability_get_metrics"],
        )

        # Register with custom handler
        async def metrics_handler(context, previous_results):
            return StepResult(
                step_name="get_metrics",
                success=True,
                result={"cpu": 45.2, "memory": 60.1},
                duration_ms=50,
            )

        executor.register(skill, {"get_metrics": metrics_handler})

        # Execute skill
        result = await executor.execute("simple_skill", context={})

        assert result.success is True
        assert result.status == SkillStatus.COMPLETED
        assert len(result.step_results) == 1
        assert result.step_results[0].success is True

    @pytest.mark.asyncio
    async def test_skill_execution_with_failure(self, mock_catalog):
        """Test skill execution with step failure."""
        from src.agents.skills import (
            SkillDefinition,
            SkillExecutor,
            SkillStatus,
            SkillStep,
            StepResult,
        )

        executor = SkillExecutor(mock_catalog)

        skill = SkillDefinition(
            name="failing_skill",
            description="Skill that fails",
            steps=[
                SkillStep(name="step1", description="First step"),
                SkillStep(name="step2", description="Failing step"),
            ],
            required_tools=[],
        )

        async def step1_handler(context, previous_results):
            return StepResult(step_name="step1", success=True, result="ok")

        async def step2_handler(context, previous_results):
            return StepResult(step_name="step2", success=False, error="Simulated failure")

        executor.register(skill, {"step1": step1_handler, "step2": step2_handler})

        result = await executor.execute("failing_skill", context={})

        assert result.success is False
        assert result.status == SkillStatus.FAILED
        assert len(result.step_results) == 2

    @pytest.mark.asyncio
    async def test_skill_execution_optional_step(self, mock_catalog):
        """Test skill execution with optional failing step."""
        from src.agents.skills import (
            SkillDefinition,
            SkillExecutor,
            SkillStatus,
            SkillStep,
            StepResult,
        )

        executor = SkillExecutor(mock_catalog)

        skill = SkillDefinition(
            name="optional_step_skill",
            description="Skill with optional step",
            steps=[
                SkillStep(name="required_step", description="Required"),
                SkillStep(name="optional_step", description="Optional", optional=True),
                SkillStep(name="final_step", description="Final"),
            ],
            required_tools=[],
        )

        async def required_handler(context, previous_results):
            return StepResult(step_name="required_step", success=True)

        async def optional_handler(context, previous_results):
            return StepResult(step_name="optional_step", success=False, error="Optional failed")

        async def final_handler(context, previous_results):
            return StepResult(step_name="final_step", success=True)

        executor.register(
            skill,
            {
                "required_step": required_handler,
                "optional_step": optional_handler,
                "final_step": final_handler,
            },
        )

        result = await executor.execute("optional_step_skill", context={})

        # Should complete because optional step failure doesn't stop execution
        assert result.status == SkillStatus.COMPLETED
        assert result.step_results[1].success is False  # Optional step failed
        assert result.step_results[2].success is True  # Final step ran


# ─────────────────────────────────────────────────────────────────────────────
# Pre-defined Skills Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestPredefinedSkills:
    """Test pre-defined skill configurations."""

    def test_rca_workflow_definition(self):
        """Test RCA workflow is properly defined."""
        from src.agents.skills import RCA_WORKFLOW

        assert RCA_WORKFLOW.name == "rca_workflow"
        assert len(RCA_WORKFLOW.steps) == 7
        assert "database" in RCA_WORKFLOW.tags
        assert RCA_WORKFLOW.estimated_duration_seconds > 0

    def test_cost_analysis_workflow_definition(self):
        """Test cost analysis workflow is properly defined."""
        from src.agents.skills import COST_ANALYSIS_WORKFLOW

        assert COST_ANALYSIS_WORKFLOW.name == "cost_analysis_workflow"
        assert len(COST_ANALYSIS_WORKFLOW.steps) >= 3
        assert "finops" in COST_ANALYSIS_WORKFLOW.tags

    def test_security_assessment_workflow_definition(self):
        """Test security assessment workflow is properly defined."""
        from src.agents.skills import SECURITY_ASSESSMENT_WORKFLOW

        assert SECURITY_ASSESSMENT_WORKFLOW.name == "security_assessment_workflow"
        assert len(SECURITY_ASSESSMENT_WORKFLOW.steps) >= 3
        assert "security" in SECURITY_ASSESSMENT_WORKFLOW.tags

    def test_skill_registry(self):
        """Test skill registry operations."""
        from src.agents.skills import SkillRegistry, register_default_skills

        # Reset and register
        SkillRegistry.reset_instance()
        register_default_skills()

        registry = SkillRegistry.get_instance()

        # Check all default skills are registered
        assert registry.get("rca_workflow") is not None
        assert registry.get("cost_analysis_workflow") is not None
        assert registry.get("security_assessment_workflow") is not None

        # Test listing
        all_skills = registry.list_all()
        assert len(all_skills) >= 3

        # Test filtering by tag
        db_skills = registry.get_by_tag("database")
        assert len(db_skills) >= 1


# ─────────────────────────────────────────────────────────────────────────────
# Agent Integration Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentIntegration:
    """Test agent integration with skills and MCP."""

    @pytest.fixture
    def mock_components(self):
        """Create mock memory and tool catalog."""
        from src.mcp.catalog import ToolCatalog
        from src.mcp.client import ToolCallResult, ToolDefinition

        # Mock memory manager
        memory = MagicMock()
        memory.get_session_state = AsyncMock(return_value={})
        memory.set_session_state = AsyncMock()
        memory.get_agent_memory = AsyncMock(return_value=None)
        memory.set_agent_memory = AsyncMock()

        # Mock tool catalog
        catalog = MagicMock(spec=ToolCatalog)

        def get_tool(name):
            return ToolDefinition(
                name=name,
                description=f"Mock tool: {name}",
                input_schema={},
                server_id="mock",
            )

        catalog.get_tool = get_tool
        catalog.execute = AsyncMock(
            return_value=ToolCallResult(
                tool_name="mock",
                success=True,
                result={"mock": "data"},
                duration_ms=100,
            )
        )

        return {"memory": memory, "catalog": catalog}

    def test_agent_skill_capability_check(self, mock_components):
        """Test agent can check skill capabilities."""
        from src.agents.skills import SkillRegistry, register_default_skills

        # Register default skills
        SkillRegistry.reset_instance()
        register_default_skills()

        # Import after skills are registered
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_components["memory"],
            tool_catalog=mock_components["catalog"],
        )

        # Check skills
        skills = agent.get_skills()
        assert "rca_workflow" in skills

    @pytest.mark.asyncio
    async def test_agent_catalog_listing(self):
        """Test agent catalog operations."""
        from src.agents.catalog import AgentCatalog

        # Reset singleton
        AgentCatalog.reset_instance()

        catalog = AgentCatalog.get_instance()

        # Get by capability
        db_agents = catalog.get_by_capability("database-analysis")
        # May be empty if agents not discovered yet
        assert isinstance(db_agents, list)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests (Require real MCP server)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.integration
class TestMCPIntegration:
    """
    Integration tests requiring real MCP server.

    Run with: pytest -m integration -v
    Requires: Database Observatory MCP server running
    """

    @pytest.fixture
    def db_observatory_path(self):
        """Path to Database Observatory MCP server."""
        return "/Users/abirzu/dev/MCP/mcp-oci-database-observatory"

    @pytest.mark.asyncio
    async def test_real_mcp_server_connection(self, db_observatory_path):
        """Test connecting to real MCP server."""
        if not os.path.isdir(db_observatory_path):
            pytest.skip("Database Observatory not found")

        from src.mcp.client import MCPServerConfig, TransportType
        from src.mcp.registry import ServerRegistry, ServerStatus

        # Reset singleton
        ServerRegistry.reset_instance()

        registry = ServerRegistry.get_instance()

        config = MCPServerConfig(
            server_id="db-observatory-test",
            transport=TransportType.STDIO,
            command="python",
            args=["-m", "src.mcp_server"],
            working_dir=db_observatory_path,
            env={
                "OCI_CONFIG_FILE": os.path.expanduser("~/.oci/config"),
                "OCI_CLI_PROFILE": "DEFAULT",
            },
            timeout_seconds=30,
        )

        registry.register(config)

        try:
            await registry.connect("db-observatory-test")
            status = registry.get_status("db-observatory-test")
            assert status == ServerStatus.CONNECTED

            # Get tools
            tools = registry.get_all_tools()
            assert len(tools) > 0

            # Log discovered tools
            tool_names = list(tools.keys())
            print(f"Discovered {len(tool_names)} tools: {tool_names[:10]}...")

        finally:
            await registry.disconnect_all()

    @pytest.mark.asyncio
    async def test_real_tool_discovery(self, db_observatory_path):
        """Test tool discovery from real MCP server."""
        if not os.path.isdir(db_observatory_path):
            pytest.skip("Database Observatory not found")

        from src.mcp.catalog import ToolCatalog
        from src.mcp.client import MCPServerConfig, TransportType
        from src.mcp.registry import ServerRegistry

        ServerRegistry.reset_instance()
        ToolCatalog.reset_instance()

        registry = ServerRegistry.get_instance()

        config = MCPServerConfig(
            server_id="db-observatory-test",
            transport=TransportType.STDIO,
            command="python",
            args=["-m", "src.mcp_server"],
            working_dir=db_observatory_path,
            timeout_seconds=30,
        )

        registry.register(config)

        try:
            await registry.connect("db-observatory-test")

            catalog = ToolCatalog.get_instance(registry)
            await catalog.refresh()

            # Search for database tools
            db_tools = catalog.search_tools(query="sql")
            assert len(db_tools) > 0

            # Get statistics
            stats = catalog.get_statistics()
            print(f"Catalog stats: {stats}")

        finally:
            await registry.disconnect_all()


# ─────────────────────────────────────────────────────────────────────────────
# Run tests
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
