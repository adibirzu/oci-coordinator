"""
Tests for All Agent Types.

Verifies that all agents:
1. Can be instantiated correctly
2. Have valid definitions
3. Can execute skills
4. Can be coordinated together

Run with:
    poetry run pytest tests/test_all_agents.py -v
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from src.mcp.client import ToolCallResult, ToolDefinition


@pytest.fixture
def mock_memory():
    """Create mock memory manager."""
    memory = MagicMock()
    memory.get_session_state = AsyncMock(return_value={})
    memory.set_session_state = AsyncMock()
    memory.get_agent_memory = AsyncMock(return_value=None)
    memory.set_agent_memory = AsyncMock()
    return memory


@pytest.fixture
def mock_catalog():
    """Create mock tool catalog with common tools."""
    from src.mcp.catalog import ToolCatalog

    catalog = MagicMock(spec=ToolCatalog)

    # Define mock tools - using unified naming convention
    mock_tools = {
        # OCI Unified tools
        "oci_database_list_autonomous": ToolDefinition(
            name="oci_database_list_autonomous",
            description="List autonomous databases",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_database_get_autonomous": ToolDefinition(
            name="oci_database_get_autonomous",
            description="Get autonomous database details",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_observability_get_metrics": ToolDefinition(
            name="oci_observability_get_metrics",
            description="Get metrics",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_observability_query_logs": ToolDefinition(
            name="oci_observability_query_logs",
            description="Query logs",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_compute_list_instances": ToolDefinition(
            name="oci_compute_list_instances",
            description="List compute instances",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_security_cloudguard_list_problems": ToolDefinition(
            name="oci_security_cloudguard_list_problems",
            description="List Cloud Guard problems",
            input_schema={},
            server_id="oci-unified",
        ),
        "oci_cost_get_summary": ToolDefinition(
            name="oci_cost_get_summary",
            description="Get cost summary",
            input_schema={},
            server_id="oci-unified",
        ),
        # Database Observatory OPSI tools
        "oci_opsi_get_fleet_summary": ToolDefinition(
            name="oci_opsi_get_fleet_summary",
            description="Get database fleet summary",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_search_databases": ToolDefinition(
            name="oci_opsi_search_databases",
            description="Search databases by name/type",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_get_database": ToolDefinition(
            name="oci_opsi_get_database",
            description="Get cached database details",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_get_performance_summary": ToolDefinition(
            name="oci_opsi_get_performance_summary",
            description="Get performance summary",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_analyze_cpu": ToolDefinition(
            name="oci_opsi_analyze_cpu",
            description="Analyze CPU usage",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_analyze_memory": ToolDefinition(
            name="oci_opsi_analyze_memory",
            description="Analyze memory usage",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_opsi_analyze_io": ToolDefinition(
            name="oci_opsi_analyze_io",
            description="Analyze I/O performance",
            input_schema={},
            server_id="database-observatory",
        ),
        # Database Observatory SQLcl tools
        "oci_database_execute_sql": ToolDefinition(
            name="oci_database_execute_sql",
            description="Execute SQL via SQLcl",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_database_list_connections": ToolDefinition(
            name="oci_database_list_connections",
            description="List SQLcl connections",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_database_get_schema": ToolDefinition(
            name="oci_database_get_schema",
            description="Get schema info",
            input_schema={},
            server_id="database-observatory",
        ),
        # Database Observatory Logan tools
        "oci_logan_execute_query": ToolDefinition(
            name="oci_logan_execute_query",
            description="Execute Logan query",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_logan_list_sources": ToolDefinition(
            name="oci_logan_list_sources",
            description="List log sources",
            input_schema={},
            server_id="database-observatory",
        ),
        "oci_logan_detect_anomalies": ToolDefinition(
            name="oci_logan_detect_anomalies",
            description="Detect log anomalies",
            input_schema={},
            server_id="database-observatory",
        ),
    }

    def get_tool(name):
        return mock_tools.get(name)

    catalog.get_tool = get_tool
    catalog.execute = AsyncMock(
        return_value=ToolCallResult(
            tool_name="mock",
            success=True,
            result={"data": "mock_result"},
            duration_ms=100,
        )
    )

    return catalog


class TestDbTroubleshootAgent:
    """Test Database Troubleshoot Agent."""

    def test_agent_definition(self, mock_memory, mock_catalog):
        """Test agent definition is valid."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()

        assert definition.agent_id == "db-troubleshoot-agent"
        assert definition.role == "db-troubleshoot-agent"
        assert "database-analysis" in definition.capabilities
        assert "rca_workflow" in definition.skills
        assert len(definition.mcp_tools) > 0

    def test_agent_capabilities(self, mock_memory, mock_catalog):
        """Test agent capabilities."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        caps = agent.get_capabilities()
        assert "database-analysis" in caps
        assert "performance-diagnostics" in caps

    def test_agent_skills(self, mock_memory, mock_catalog):
        """Test agent skills including new Database Observatory workflows."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        skills = agent.get_skills()
        # New Database Observatory skills
        assert "db_rca_workflow" in skills
        assert "db_health_check_workflow" in skills
        assert "db_sql_analysis_workflow" in skills
        # Legacy skill for compatibility
        assert "rca_workflow" in skills


class TestLogAnalyticsAgent:
    """Test Log Analytics Agent."""

    def test_agent_definition(self, mock_memory, mock_catalog):
        """Test agent definition is valid."""
        from src.agents.log_analytics.agent import LogAnalyticsAgent

        agent = LogAnalyticsAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()

        assert definition.agent_id == "log-analytics-agent"
        assert definition.role == "log-analytics-agent"
        assert "log-search" in definition.capabilities

    def test_agent_capabilities(self, mock_memory, mock_catalog):
        """Test agent capabilities."""
        from src.agents.log_analytics.agent import LogAnalyticsAgent

        agent = LogAnalyticsAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        caps = agent.get_capabilities()
        assert "log-search" in caps
        assert "pattern-detection" in caps


class TestSecurityAgent:
    """Test Security Agent."""

    def test_agent_definition(self, mock_memory, mock_catalog):
        """Test agent definition is valid."""
        from src.agents.security.agent import SecurityThreatAgent

        agent = SecurityThreatAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()

        assert definition.agent_id == "security-threat-agent"
        assert definition.role == "security-threat-agent"
        assert "threat-detection" in definition.capabilities

    def test_agent_capabilities(self, mock_memory, mock_catalog):
        """Test agent capabilities."""
        from src.agents.security.agent import SecurityThreatAgent

        agent = SecurityThreatAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        caps = agent.get_capabilities()
        assert "threat-detection" in caps
        assert "compliance-monitoring" in caps


class TestFinOpsAgent:
    """Test FinOps Agent."""

    def test_agent_definition(self, mock_memory, mock_catalog):
        """Test agent definition is valid."""
        from src.agents.finops.agent import FinOpsAgent

        agent = FinOpsAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()

        assert definition.agent_id == "finops-agent"
        assert definition.role == "finops-agent"
        assert "cost-analysis" in definition.capabilities

    def test_agent_capabilities(self, mock_memory, mock_catalog):
        """Test agent capabilities."""
        from src.agents.finops.agent import FinOpsAgent

        agent = FinOpsAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        caps = agent.get_capabilities()
        assert "cost-analysis" in caps


class TestInfrastructureAgent:
    """Test Infrastructure Agent."""

    def test_agent_definition(self, mock_memory, mock_catalog):
        """Test agent definition is valid."""
        from src.agents.infrastructure.agent import InfrastructureAgent

        agent = InfrastructureAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()

        assert definition.agent_id == "infrastructure-agent"
        assert definition.role == "infrastructure-agent"
        assert "compute-management" in definition.capabilities
        # Verify MCP server reference
        assert "oci-infrastructure" in definition.mcp_servers

    def test_agent_capabilities(self, mock_memory, mock_catalog):
        """Test agent capabilities."""
        from src.agents.infrastructure.agent import InfrastructureAgent

        agent = InfrastructureAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        caps = agent.get_capabilities()
        assert "compute-management" in caps
        assert "network-analysis" in caps
        assert "security-operations" in caps

    def test_agent_skills(self, mock_memory, mock_catalog):
        """Test agent skills including new infrastructure workflows."""
        from src.agents.infrastructure.agent import InfrastructureAgent

        agent = InfrastructureAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        skills = agent.get_skills()
        # New oci-infrastructure MCP skills
        assert "infra_inventory_workflow" in skills
        assert "infra_instance_management_workflow" in skills
        assert "infra_network_analysis_workflow" in skills
        assert "infra_security_audit_workflow" in skills

    def test_agent_mcp_tools(self, mock_memory, mock_catalog):
        """Test agent has correct MCP tools defined."""
        from src.agents.infrastructure.agent import InfrastructureAgent

        definition = InfrastructureAgent.get_definition()

        # Verify key MCP tools from oci-infrastructure server
        assert "oci_compute_list_instances" in definition.mcp_tools
        assert "oci_list_compartments" in definition.mcp_tools
        assert "oci_network_list_vcns" in definition.mcp_tools
        assert "oci_security_list_cloud_guard_problems" in definition.mcp_tools
        assert len(definition.mcp_tools) >= 20  # At least 20 tools


class TestAgentCatalog:
    """Test Agent Catalog operations."""

    def test_catalog_initialization(self):
        """Test catalog can be initialized."""
        from src.agents.catalog import AgentCatalog

        AgentCatalog.reset_instance()
        catalog = AgentCatalog.get_instance()

        assert catalog is not None

    def test_catalog_auto_discover(self):
        """Test catalog auto-discovery."""
        from src.agents.catalog import AgentCatalog

        AgentCatalog.reset_instance()
        catalog = AgentCatalog.get_instance()

        # Trigger auto-discovery
        catalog.auto_discover()

        # Should have discovered agents
        agents = catalog.list_all()
        assert len(agents) >= 5  # At least our 5 agents

    def test_catalog_get_by_capability(self):
        """Test finding agents by capability."""
        from src.agents.catalog import AgentCatalog

        AgentCatalog.reset_instance()
        catalog = AgentCatalog.get_instance()
        catalog.auto_discover()

        # Find database agents
        db_agents = catalog.get_by_capability("database-analysis")
        assert len(db_agents) >= 1

        # Find security agents
        sec_agents = catalog.get_by_capability("threat-detection")
        assert len(sec_agents) >= 1

    def test_catalog_get_by_skill(self):
        """Test finding agents by skill."""
        from src.agents.catalog import AgentCatalog

        AgentCatalog.reset_instance()
        catalog = AgentCatalog.get_instance()
        catalog.auto_discover()

        # Find agents with RCA workflow
        rca_agents = catalog.get_by_skill("rca_workflow")
        assert len(rca_agents) >= 1

    def test_catalog_list_all_definitions(self):
        """Test listing all agent definitions."""
        from src.agents.catalog import AgentCatalog

        AgentCatalog.reset_instance()
        catalog = AgentCatalog.get_instance()
        catalog.auto_discover()

        agents = catalog.list_all()

        # Verify all expected agents are present
        agent_ids = [a.agent_id for a in agents]
        assert "db-troubleshoot-agent" in agent_ids
        assert "log-analytics-agent" in agent_ids
        assert "security-threat-agent" in agent_ids
        assert "finops-agent" in agent_ids
        assert "infrastructure-agent" in agent_ids


class TestAgentSkillExecution:
    """Test agent skill execution capabilities."""

    @pytest.fixture
    def skill_setup(self):
        """Set up skills registry."""
        from src.agents.skills import SkillRegistry, register_default_skills

        SkillRegistry.reset_instance()
        register_default_skills()

    def test_agent_can_check_skill(self, mock_memory, mock_catalog, skill_setup):
        """Test agent can check if it can execute a skill."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        # Agent should be able to check skill capability
        can_execute, missing = agent.can_execute_skill("rca_workflow")

        # May or may not be able to execute depending on tools
        assert isinstance(can_execute, bool)
        assert isinstance(missing, list)

    @pytest.mark.asyncio
    async def test_agent_skill_execution_framework(
        self, mock_memory, mock_catalog, skill_setup
    ):
        """Test agent skill execution framework."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent
        from src.agents.skills import SkillRegistry, SkillStatus, StepResult

        # Add mock tools that the RCA workflow needs
        def get_tool_expanded(name):
            # Return a tool for any name
            return ToolDefinition(
                name=name,
                description=f"Mock tool: {name}",
                input_schema={},
                server_id="mock",
            )

        mock_catalog.get_tool = get_tool_expanded

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        # Create a simple test skill
        from src.agents.skills import SkillDefinition, SkillStep

        test_skill = SkillDefinition(
            name="test_agent_skill",
            description="Test skill for agent",
            steps=[
                SkillStep(name="step1", description="Step 1"),
            ],
            required_tools=[],
        )

        # Register the skill
        registry = SkillRegistry.get_instance()
        registry.register(test_skill)

        # Execute the skill
        result = await agent.execute_skill(
            "test_agent_skill",
            context={"test": "value"},
        )

        # Should complete (might fail due to no handlers, but framework should work)
        assert result.skill_name == "test_agent_skill"


class TestAgentOutputFormatting:
    """Test agent response formatting."""

    def test_agent_create_response(self, mock_memory, mock_catalog):
        """Test agent can create structured responses."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        response = agent.create_response(
            title="Database Analysis",
            subtitle="Performance Report",
            severity="high",
        )

        assert response.header.title == "Database Analysis"
        assert response.header.subtitle == "Performance Report"

    def test_agent_format_response(self, mock_memory, mock_catalog):
        """Test agent can format responses."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        response = agent.create_response(title="Test Response")
        formatted = agent.format_response(response)

        # Should return formatted string or dict
        assert formatted is not None

    def test_agent_set_output_format(self, mock_memory, mock_catalog):
        """Test agent output format can be changed."""
        from src.agents.database.troubleshoot import DbTroubleshootAgent

        agent = DbTroubleshootAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        # Default format
        assert agent.get_output_format() == "markdown"

        # Change format
        agent.set_output_format("slack")
        assert agent.get_output_format() == "slack"


class TestCoordinatorIntegration:
    """Test coordinator with all agents."""

    @pytest.mark.asyncio
    async def test_coordinator_can_access_all_agents(self, mock_memory, mock_catalog):
        """Test coordinator can access all agent types."""
        from src.agents.catalog import AgentCatalog

        AgentCatalog.reset_instance()
        catalog = AgentCatalog.get_instance()
        catalog.auto_discover()

        # Verify all agents are accessible
        all_agents = catalog.list_all()
        assert len(all_agents) >= 5

        # Verify capability-based lookup works
        # These match actual capabilities defined in agent definitions
        capabilities = [
            "database-analysis",
            "log-search",
            "threat-detection",
            "cost-analysis",
            "compute-management",
        ]

        for cap in capabilities:
            agents = catalog.get_by_capability(cap)
            assert len(agents) >= 1, f"No agent found for capability: {cap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
