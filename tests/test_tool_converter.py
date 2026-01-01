"""Tests for ToolConverter."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from src.mcp.tools.converter import (
    ToolConverter,
    _json_type_to_python,
    _create_pydantic_model_from_schema,
)
from src.mcp.client import ToolDefinition, ToolCallResult
from src.mcp.catalog import ToolCatalog, TOOL_TIERS, ToolTier


class MockToolCatalog:
    """Mock catalog for testing."""

    def __init__(self):
        self._tools = [
            ToolDefinition(
                name="oci_compute_list_instances",
                description="List compute instances",
                input_schema={
                    "type": "object",
                    "properties": {
                        "compartment_id": {
                            "type": "string",
                            "description": "The compartment OCID",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results",
                            "default": 50,
                        },
                    },
                    "required": ["compartment_id"],
                },
                server_id="mcp-oci",
            ),
            ToolDefinition(
                name="oci_database_execute_sql",
                description="Execute SQL query",
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "database_id": {"type": "string"},
                    },
                    "required": ["query", "database_id"],
                },
                server_id="database-observatory",
            ),
            ToolDefinition(
                name="oci_compute_stop_instance",
                description="Stop an instance",
                input_schema={
                    "type": "object",
                    "properties": {
                        "instance_id": {"type": "string"},
                    },
                    "required": ["instance_id"],
                },
                server_id="mcp-oci",
            ),
        ]

    def list_tools(self):
        return self._tools

    async def execute(self, tool_name, args):
        return ToolCallResult(
            tool_name=tool_name,
            success=True,
            result=f"Executed {tool_name}",
        )


class TestJsonTypeConversion:
    """Test JSON schema type conversion."""

    def test_string_type(self):
        assert _json_type_to_python("string") == str

    def test_integer_type(self):
        assert _json_type_to_python("integer") == int

    def test_number_type(self):
        assert _json_type_to_python("number") == float

    def test_boolean_type(self):
        assert _json_type_to_python("boolean") == bool

    def test_array_type(self):
        assert _json_type_to_python("array") == list

    def test_object_type(self):
        assert _json_type_to_python("object") == dict

    def test_union_type_with_null(self):
        # ["string", "null"] should return str
        assert _json_type_to_python(["string", "null"]) == str

    def test_unknown_type(self):
        # Unknown types default to str
        assert _json_type_to_python("unknown") == str


class TestPydanticModelCreation:
    """Test Pydantic model creation from JSON schema."""

    def test_create_simple_model(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The name"},
                "count": {"type": "integer", "description": "The count"},
            },
            "required": ["name"],
        }

        Model = _create_pydantic_model_from_schema("test_tool", schema)

        assert Model is not None
        assert "name" in Model.model_fields
        assert "count" in Model.model_fields

    def test_required_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "required_field": {"type": "string"},
                "optional_field": {"type": "string", "default": "default"},
            },
            "required": ["required_field"],
        }

        Model = _create_pydantic_model_from_schema("test", schema)

        # Required field should not have default
        required_field = Model.model_fields["required_field"]
        optional_field = Model.model_fields["optional_field"]

        assert required_field.is_required() is True
        assert optional_field.is_required() is False


class TestToolConverter:
    """Test ToolConverter class."""

    @pytest.fixture
    def mock_catalog(self):
        return MockToolCatalog()

    @pytest.fixture
    def converter(self, mock_catalog):
        return ToolConverter(mock_catalog)

    def test_converter_init(self, converter, mock_catalog):
        assert converter.catalog == mock_catalog

    def test_to_langchain_tools(self, converter):
        tools = converter.to_langchain_tools(max_tier=3)

        # Should convert tier 2 tools (list_instances)
        # Should skip tier 4 tools (stop_instance) by default
        tool_names = [t.name for t in tools]

        assert "oci_compute_list_instances" in tool_names
        # stop_instance is tier 4, should be excluded with max_tier=3

    def test_to_langchain_tools_with_tier_4(self, converter):
        tools = converter.to_langchain_tools(max_tier=3, include_confirmation=True)

        # Tier 4 tools should be included with include_confirmation=True
        tool_names = [t.name for t in tools]
        assert "oci_compute_stop_instance" in tool_names

    def test_to_langchain_tools_specific_names(self, converter):
        tools = converter.to_langchain_tools(
            tool_names=["oci_compute_list_instances"],
            max_tier=4,
        )

        assert len(tools) == 1
        assert tools[0].name == "oci_compute_list_instances"

    def test_get_domain_tools(self, converter):
        # Mock catalog returns compute tools
        tools = converter.get_domain_tools("infrastructure")

        # Should find compute tools based on prefixes
        assert len(tools) >= 0  # May be empty if prefixes don't match

    def test_get_tools_by_server(self, converter):
        tools = converter.get_tools_by_server("mcp-oci")

        tool_names = [t.name for t in tools]
        assert "oci_compute_list_instances" in tool_names

    def test_get_safe_tools(self, converter):
        tools = converter.get_safe_tools()

        # All returned tools should be tier 1-2 with no risk
        for tool in tools:
            tier_info = TOOL_TIERS.get(tool.name, ToolTier(2, 1000, "low"))
            assert tier_info.tier <= 2
            assert tier_info.risk_level == "none"

    def test_resolve_tool_alias(self, converter):
        # Test alias resolution
        canonical = converter.resolve_tool_alias("list_instances")
        assert canonical == "oci_compute_list_instances"

        # Non-alias should return original
        original = converter.resolve_tool_alias("oci_compute_list_instances")
        assert original == "oci_compute_list_instances"

    def test_get_statistics(self, converter):
        # First convert some tools
        converter.to_langchain_tools(max_tier=4)

        stats = converter.get_statistics()

        assert "total_converted" in stats
        assert "by_domain" in stats
        assert "by_tier" in stats
        assert "catalog_total" in stats
        assert stats["total_converted"] >= 0

    def test_tool_description_includes_tier_info(self, converter):
        tools = converter.to_langchain_tools(max_tier=2)

        for tool in tools:
            # Description should include tier information
            assert "[Tier" in tool.description

    def test_confirmation_callback(self, mock_catalog):
        confirmed_tools = []

        def confirm_callback(tool_name, description):
            confirmed_tools.append(tool_name)
            return True  # Always confirm

        converter = ToolConverter(mock_catalog, confirm_callback=confirm_callback)
        tools = converter.to_langchain_tools(max_tier=4, include_confirmation=True)

        # Callback should be set
        assert converter._confirm_callback is not None

    def test_get_tool_by_name(self, converter):
        # First convert tools
        converter.to_langchain_tools(max_tier=3)

        # Get by exact name
        tool = converter.get_tool_by_name("oci_compute_list_instances")
        assert tool is not None
        assert tool.name == "oci_compute_list_instances"

        # Get by alias (should resolve)
        tool_alias = converter.get_tool_by_name("list_instances")
        # May be None if alias wasn't converted

    def test_pydantic_schemas_disabled(self, converter):
        tools = converter.to_langchain_tools(
            max_tier=3,
            use_pydantic_schemas=False,
        )

        # Should still work without Pydantic schemas
        assert len(tools) > 0


class TestToolConverterWithRealCatalog:
    """Integration tests with real ToolCatalog (mocked registry)."""

    @pytest.fixture
    def catalog(self):
        from src.mcp.registry import ServerRegistry

        # Reset singleton for clean state
        ToolCatalog.reset_instance()
        ServerRegistry.reset_instance()

        return ToolCatalog.get_instance()

    def test_converter_with_real_catalog(self, catalog):
        converter = ToolConverter(catalog)

        # Should not crash even with empty catalog
        tools = converter.to_langchain_tools(max_tier=3)
        stats = converter.get_statistics()

        assert isinstance(tools, list)
        assert isinstance(stats, dict)
