"""
Tests for SelectAI Agent and MCP Tools.

Tests:
1. Agent definition and capabilities
2. Intent detection logic
3. MCP tool functions
4. State management

Run with:
    poetry run pytest tests/test_selectai_agent.py -v
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
    """Create mock tool catalog with SelectAI tools."""
    from src.mcp.catalog import ToolCatalog

    catalog = MagicMock(spec=ToolCatalog)

    # Define mock SelectAI tools
    mock_tools = {
        "oci_selectai_generate": ToolDefinition(
            name="oci_selectai_generate",
            description="Execute SelectAI GENERATE for NL2SQL",
            input_schema={},
            server_id="selectai",
        ),
        "oci_selectai_list_profiles": ToolDefinition(
            name="oci_selectai_list_profiles",
            description="List AI profiles",
            input_schema={},
            server_id="selectai",
        ),
        "oci_selectai_get_profile_tables": ToolDefinition(
            name="oci_selectai_get_profile_tables",
            description="Get profile tables",
            input_schema={},
            server_id="selectai",
        ),
        "oci_selectai_run_agent": ToolDefinition(
            name="oci_selectai_run_agent",
            description="Run SelectAI agent",
            input_schema={},
            server_id="selectai",
        ),
        "oci_selectai_ping": ToolDefinition(
            name="oci_selectai_ping",
            description="SelectAI health check",
            input_schema={},
            server_id="selectai",
        ),
        # Fallback tools from database-observatory
        "oci_database_execute_sql": ToolDefinition(
            name="oci_database_execute_sql",
            description="Execute SQL via Database Observatory",
            input_schema={},
            server_id="database-observatory",
        ),
    }

    catalog.get_tool = MagicMock(side_effect=lambda name: mock_tools.get(name))
    catalog.get_all_tools = MagicMock(return_value=list(mock_tools.values()))
    catalog.list_tools = MagicMock(return_value=list(mock_tools.keys()))
    catalog.has_tool = MagicMock(side_effect=lambda name: name in mock_tools)

    async def mock_call_tool(name, args):
        if name == "oci_selectai_generate":
            return ToolCallResult(
                success=True,
                result=json.dumps({
                    "type": "selectai_generate",
                    "action": args.get("action", "runsql"),
                    "response": "SELECT * FROM customers WHERE total > 10000",
                }),
                error=None,
            )
        elif name == "oci_selectai_list_profiles":
            return ToolCallResult(
                success=True,
                result=json.dumps({
                    "type": "selectai_profiles",
                    "profiles": [
                        {"name": "OCI_GENAI", "provider": "oci", "status": "ENABLED"}
                    ],
                }),
                error=None,
            )
        return ToolCallResult(success=True, result="{}", error=None)

    catalog.call_tool = AsyncMock(side_effect=mock_call_tool)

    return catalog


class TestSelectAIAgentDefinition:
    """Test SelectAI Agent definition and capabilities."""

    def test_agent_definition(self, mock_memory, mock_catalog):
        """Test agent definition is valid."""
        from src.agents.selectai.agent import SelectAIAgent

        agent = SelectAIAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()

        assert definition.agent_id == "selectai-agent"
        assert definition.role == "selectai-agent"
        assert "nl2sql" in definition.capabilities
        assert "data-chat" in definition.capabilities
        assert "ai-agent-orchestration" in definition.capabilities

    def test_agent_capabilities(self, mock_memory, mock_catalog):
        """Test agent capabilities list."""
        from src.agents.selectai.agent import SelectAIAgent

        agent = SelectAIAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        caps = agent.get_capabilities()
        assert "nl2sql" in caps
        assert "data-chat" in caps
        assert "text-summarization" in caps
        assert "database-qa" in caps

    def test_agent_mcp_servers(self, mock_memory, mock_catalog):
        """Test agent MCP server references."""
        from src.agents.selectai.agent import SelectAIAgent

        agent = SelectAIAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()
        assert "selectai" in definition.mcp_servers
        assert "database-observatory" in definition.mcp_servers

    def test_agent_skills(self, mock_memory, mock_catalog):
        """Test agent skills list."""
        from src.agents.selectai.agent import SelectAIAgent

        agent = SelectAIAgent(
            memory_manager=mock_memory,
            tool_catalog=mock_catalog,
        )

        definition = agent.get_definition()
        assert "nl2sql_workflow" in definition.skills
        assert "data_exploration" in definition.skills


class TestSelectAIIntentDetection:
    """Test intent detection logic via SelectAIState defaults."""

    def test_default_intent_is_nl2sql(self):
        """Test that default intent is nl2sql."""
        from src.agents.selectai.agent import SelectAIState

        state = SelectAIState(query="show me customers")
        assert state.intent == "nl2sql"

    def test_state_accepts_different_intents(self):
        """Test that state can be created with different intents."""
        from src.agents.selectai.agent import SelectAIState

        # Test chat intent
        chat_state = SelectAIState(query="explain the schema", intent="chat")
        assert chat_state.intent == "chat"

        # Test agent_run intent
        agent_state = SelectAIState(
            query="run the analysis",
            intent="agent_run",
            team_name="SALES_TEAM",
        )
        assert agent_state.intent == "agent_run"
        assert agent_state.team_name == "SALES_TEAM"

    def test_state_preserves_query(self):
        """Test that state preserves the query."""
        from src.agents.selectai.agent import SelectAIState

        query = "SELECT * FROM customers WHERE total > 10000"
        state = SelectAIState(query=query)
        assert state.query == query


class TestSelectAIMCPTools:
    """Test SelectAI MCP tool functions."""

    @pytest.mark.asyncio
    async def test_selectai_generate_logic(self):
        """Test selectai_generate_logic function."""
        from src.mcp.server.tools.selectai import _selectai_generate_logic

        # Mock the ORDS call
        with patch("src.mcp.server.tools.selectai._execute_sql_via_ords") as mock_ords:
            mock_ords.return_value = {
                "items": [
                    {"response": "SELECT * FROM customers WHERE total > 10000"}
                ]
            }

            result = await _selectai_generate_logic(
                prompt="show customers with orders over 10000",
                profile_name="TEST_PROFILE",
                action="showsql",
                ords_base_url="https://test.adb.region.oraclecloudapps.com",
                ords_schema="ADMIN",
                ords_auth_token="test_token",
            )

            result_data = json.loads(result)
            assert result_data["type"] == "selectai_generate"
            assert result_data["action"] == "showsql"
            assert "response" in result_data

    @pytest.mark.asyncio
    async def test_selectai_generate_no_ords_configured(self):
        """Test error when ORDS is not configured."""
        from src.mcp.server.tools.selectai import _selectai_generate_logic

        with patch.dict("os.environ", {"SELECTAI_ORDS_BASE_URL": ""}, clear=False):
            result = await _selectai_generate_logic(
                prompt="show customers",
                action="runsql",
            )

            result_data = json.loads(result)
            assert "error" in result_data
            assert "ORDS not configured" in result_data["error"]

    @pytest.mark.asyncio
    async def test_selectai_list_profiles_logic(self):
        """Test list profiles logic."""
        from src.mcp.server.tools.selectai import _selectai_list_profiles_logic

        with patch("src.mcp.server.tools.selectai._execute_sql_via_ords") as mock_ords:
            mock_ords.return_value = {
                "items": [
                    {
                        "profile_name": "OCI_GENAI",
                        "provider": "oci",
                        "model": "cohere.command-r-plus",
                        "status": "ENABLED",
                    }
                ]
            }

            result = await _selectai_list_profiles_logic(
                ords_base_url="https://test.adb.region.oraclecloudapps.com",
                ords_schema="ADMIN",
                ords_auth_token="test_token",
            )

            result_data = json.loads(result)
            assert result_data["type"] == "selectai_profiles"
            assert len(result_data["profiles"]) == 1
            assert result_data["profiles"][0]["name"] == "OCI_GENAI"

    @pytest.mark.asyncio
    async def test_selectai_run_agent_logic(self):
        """Test run agent logic."""
        from src.mcp.server.tools.selectai import _selectai_run_agent_logic

        with patch("src.mcp.server.tools.selectai._execute_sql_via_ords") as mock_ords:
            mock_ords.return_value = {
                "result": "Analysis complete. Top region: West with $1.2M revenue."
            }

            result = await _selectai_run_agent_logic(
                team_name="SALES_ANALYST_TEAM",
                user_prompt="What were last quarter revenues?",
                ords_base_url="https://test.adb.region.oraclecloudapps.com",
                ords_schema="ADMIN",
                ords_auth_token="test_token",
            )

            result_data = json.loads(result)
            assert result_data["type"] == "selectai_agent"
            assert result_data["team"] == "SALES_ANALYST_TEAM"


class TestSelectAICatalogIntegration:
    """Test SelectAI catalog integration."""

    def test_selectai_domain_in_catalog(self):
        """Test that SelectAI domain is registered in catalog."""
        from src.agents.catalog import DOMAIN_CAPABILITIES

        assert "selectai" in DOMAIN_CAPABILITIES
        assert "nl2sql" in DOMAIN_CAPABILITIES["selectai"]
        assert "data-chat" in DOMAIN_CAPABILITIES["selectai"]

    def test_selectai_priority_in_catalog(self):
        """Test that SelectAI priority is registered."""
        from src.agents.catalog import DOMAIN_PRIORITY

        assert "selectai" in DOMAIN_PRIORITY
        assert "selectai-agent" in DOMAIN_PRIORITY["selectai"]
        assert DOMAIN_PRIORITY["selectai"]["selectai-agent"] == 100

    def test_selectai_mcp_server_in_agent_definition(self):
        """Test that SelectAI agent references selectai MCP server."""
        from src.agents.selectai.agent import SelectAIAgent

        definition = SelectAIAgent.get_definition()
        assert "selectai" in definition.mcp_servers
        assert "database-observatory" in definition.mcp_servers


class TestSelectAIStateManagement:
    """Test SelectAI state dataclass."""

    def test_selectai_state_initialization(self):
        """Test SelectAIState initialization with defaults."""
        from src.agents.selectai.agent import SelectAIState

        state = SelectAIState(
            session_id="test-session",
            query="show me customers",
        )

        assert state.session_id == "test-session"
        assert state.query == "show me customers"
        assert state.intent == "nl2sql"  # Default intent
        assert state.profile_name is None
        assert state.generated_sql is None
        assert state.query_results == []  # Default to empty list
        assert state.error is None
        assert state.reasoning_chain == []

    def test_selectai_state_with_all_fields(self):
        """Test SelectAIState with all fields populated."""
        from src.agents.selectai.agent import SelectAIState

        state = SelectAIState(
            session_id="test-session",
            query="show me customers",
            intent="nl2sql",
            profile_name="OCI_GENAI",
            database_id="ocid1.autonomousdatabase...",
            generated_sql="SELECT * FROM customers",
            query_results=[{"customer_id": 1, "name": "Test"}],
            error=None,
            reasoning_chain=["Detected NL2SQL intent", "Generated SQL"],
        )

        assert state.intent == "nl2sql"
        assert state.profile_name == "OCI_GENAI"
        assert state.generated_sql == "SELECT * FROM customers"
        assert len(state.query_results) == 1
        assert len(state.reasoning_chain) == 2

    def test_selectai_state_action_types(self):
        """Test SelectAIState with different action types."""
        from src.agents.selectai.agent import SelectAIState

        # Test showsql action
        state = SelectAIState(
            query="show me the SQL for customer query",
            action="showsql",
        )
        assert state.action == "showsql"

        # Test chat action
        chat_state = SelectAIState(
            query="explain the schema",
            intent="chat",
            action="chat",
        )
        assert chat_state.action == "chat"

    def test_selectai_state_agent_fields(self):
        """Test SelectAIState agent execution fields."""
        from src.agents.selectai.agent import SelectAIState

        state = SelectAIState(
            query="run quarterly analysis",
            intent="agent_run",
            team_name="QUARTERLY_ANALYSIS_TEAM",
            agent_name="SALES_ANALYST",
            session_id="session-123",
        )

        assert state.team_name == "QUARTERLY_ANALYSIS_TEAM"
        assert state.agent_name == "SALES_ANALYST"
        assert state.session_id == "session-123"
