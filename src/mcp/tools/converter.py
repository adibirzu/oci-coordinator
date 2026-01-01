"""
Tool Converter for MCP to LangChain Conversion.

Provides advanced tool conversion capabilities with:
- Dynamic Pydantic model generation from JSON schema
- Tool filtering and grouping
- Domain-specific tool selection
- Confirmation handling for high-risk tools
"""

from __future__ import annotations

import asyncio
from typing import Any, Callable

import structlog
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field, create_model

from src.mcp.catalog import (
    DOMAIN_PREFIXES,
    TOOL_ALIASES,
    TOOL_TIERS,
    ToolCatalog,
    ToolTier,
    _tier_key,
)
from src.mcp.client import ToolDefinition

logger = structlog.get_logger(__name__)


def _json_type_to_python(json_type: str | list, default: Any = None) -> type:
    """Convert JSON schema type to Python type."""
    if isinstance(json_type, list):
        # Handle union types like ["string", "null"]
        non_null = [t for t in json_type if t != "null"]
        if non_null:
            return _json_type_to_python(non_null[0], default)
        return str

    type_map = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    return type_map.get(json_type, str)


def _create_pydantic_model_from_schema(
    name: str, schema: dict[str, Any]
) -> type[BaseModel]:
    """
    Create a Pydantic model from a JSON schema.

    Args:
        name: Model name
        schema: JSON schema dict

    Returns:
        Pydantic model class
    """
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))

    field_definitions: dict[str, Any] = {}

    for prop_name, prop_schema in properties.items():
        prop_type = prop_schema.get("type", "string")
        description = prop_schema.get("description", "")
        default = prop_schema.get("default")
        is_required = prop_name in required

        python_type = _json_type_to_python(prop_type, default)

        if is_required:
            field_definitions[prop_name] = (
                python_type,
                Field(description=description),
            )
        else:
            field_definitions[prop_name] = (
                python_type | None,
                Field(default=default, description=description),
            )

    # Create model with sanitized name (no special chars)
    safe_name = "".join(c if c.isalnum() else "_" for c in name).title()
    model = create_model(f"{safe_name}Args", **field_definitions)

    return model


class ToolConverter:
    """
    Converts MCP tools to LangChain StructuredTools.

    Provides advanced conversion with:
    - Dynamic Pydantic schema generation
    - Tool filtering by domain, tier, risk level
    - Confirmation callbacks for high-risk tools
    - Domain-aware tool grouping

    Example:
        converter = ToolConverter(catalog)

        # Get all tools up to tier 3
        tools = converter.to_langchain_tools(max_tier=3)

        # Get only database tools
        db_tools = converter.get_domain_tools("database")

        # Get tools with confirmation callback
        tools = converter.to_langchain_tools(
            confirm_callback=my_confirm_func
        )
    """

    def __init__(
        self,
        catalog: ToolCatalog,
        confirm_callback: Callable[[str, str], bool] | None = None,
    ):
        """
        Initialize the converter.

        Args:
            catalog: ToolCatalog instance with discovered tools
            confirm_callback: Optional callback for confirming high-risk tools
                              Signature: (tool_name, description) -> bool
        """
        self._catalog = catalog
        self._confirm_callback = confirm_callback
        self._logger = logger.bind(component="ToolConverter")
        self._converted_tools: dict[str, StructuredTool] = {}

    @property
    def catalog(self) -> ToolCatalog:
        """Get the underlying tool catalog."""
        return self._catalog

    def to_langchain_tools(
        self,
        tool_names: list[str] | None = None,
        max_tier: int = 3,
        include_confirmation: bool = True,
        use_pydantic_schemas: bool = True,
    ) -> list[StructuredTool]:
        """
        Convert MCP tools to LangChain StructuredTools.

        Args:
            tool_names: Specific tools to convert (None = all)
            max_tier: Maximum tier to include (1-4)
            include_confirmation: Include tier 4 tools with confirmation
            use_pydantic_schemas: Generate Pydantic models for args

        Returns:
            List of LangChain StructuredTool instances
        """
        tools = self._catalog.list_tools()
        langchain_tools = []

        for tool_def in tools:
            name = tool_def.name

            # Skip if not in requested list
            if tool_names and name not in tool_names:
                continue

            # Skip namespaced duplicates
            if ":" in name:
                base_name = name.split(":", 1)[1]
                if base_name in [t.name for t in tools]:
                    continue

            # Check tier
            tier_info = TOOL_TIERS.get(_tier_key(name), ToolTier(2, 1000, "low"))
            if tier_info.tier > max_tier:
                if not (include_confirmation and tier_info.tier == 4):
                    continue

            # Convert tool
            structured_tool = self._convert_single_tool(
                tool_def,
                tier_info,
                use_pydantic_schemas,
            )

            if structured_tool:
                langchain_tools.append(structured_tool)
                self._converted_tools[name] = structured_tool

        self._logger.info(
            "Converted tools to LangChain format",
            total=len(langchain_tools),
            max_tier=max_tier,
        )

        return langchain_tools

    def _convert_single_tool(
        self,
        tool_def: ToolDefinition,
        tier_info: ToolTier,
        use_pydantic_schemas: bool,
    ) -> StructuredTool | None:
        """Convert a single MCP tool to LangChain format."""
        name = tool_def.name
        description = tool_def.description

        # Add tier and risk info to description
        tier_suffix = f" [Tier {tier_info.tier}"
        if tier_info.risk_level != "none":
            tier_suffix += f", Risk: {tier_info.risk_level}"
        if tier_info.requires_confirmation:
            tier_suffix += ", Requires confirmation"
        tier_suffix += "]"

        enhanced_description = f"{description}{tier_suffix}"

        # Build args schema if requested
        args_schema = None
        if use_pydantic_schemas and tool_def.input_schema:
            try:
                args_schema = _create_pydantic_model_from_schema(
                    name, tool_def.input_schema
                )
            except Exception as e:
                self._logger.warning(
                    "Failed to create Pydantic schema",
                    tool=name,
                    error=str(e),
                )

        # Create async execution function
        async def execute_tool(
            tool_name: str = name,
            requires_confirmation: bool = tier_info.requires_confirmation,
            **kwargs: Any,
        ) -> str:
            # Handle confirmation for high-risk tools
            if requires_confirmation and self._confirm_callback:
                confirmed = self._confirm_callback(
                    tool_name,
                    f"Tool '{tool_name}' requires confirmation. Args: {kwargs}",
                )
                if not confirmed:
                    return f"Execution of '{tool_name}' was cancelled by user."

            result = await self._catalog.execute(tool_name, kwargs)
            if result.success:
                return str(result.result)
            return f"Error: {result.error}"

        # Synchronous wrapper for LangChain compatibility
        def sync_execute_tool(
            tool_name: str = name,
            **kwargs: Any,
        ) -> str:
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                # Can't run sync in async context, return placeholder
                return f"Tool '{tool_name}' requires async execution"
            else:
                return asyncio.run(execute_tool(tool_name=tool_name, **kwargs))

        try:
            structured_tool = StructuredTool.from_function(
                func=sync_execute_tool,
                name=name,
                description=enhanced_description,
                args_schema=args_schema,
                coroutine=execute_tool,
            )
            return structured_tool

        except Exception as e:
            self._logger.error(
                "Failed to create StructuredTool",
                tool=name,
                error=str(e),
            )
            return None

    def get_domain_tools(
        self,
        domain: str,
        max_tier: int = 3,
    ) -> list[StructuredTool]:
        """
        Get LangChain tools for a specific domain.

        Args:
            domain: Domain name (database, infrastructure, finops, etc.)
            max_tier: Maximum tier to include

        Returns:
            List of StructuredTools for the domain
        """
        prefixes = DOMAIN_PREFIXES.get(domain.lower(), [])
        if not prefixes:
            self._logger.warning("Unknown domain", domain=domain)
            return []

        # Get matching tool names
        matching_names = []
        for tool_def in self._catalog.list_tools():
            for prefix in prefixes:
                if tool_def.name.startswith(prefix) or tool_def.name == prefix:
                    matching_names.append(tool_def.name)
                    break

        # Convert matching tools
        return self.to_langchain_tools(
            tool_names=matching_names,
            max_tier=max_tier,
        )

    def get_tools_by_server(
        self,
        server_id: str,
        max_tier: int = 3,
    ) -> list[StructuredTool]:
        """
        Get LangChain tools from a specific MCP server.

        Args:
            server_id: MCP server identifier
            max_tier: Maximum tier to include

        Returns:
            List of StructuredTools from the server
        """
        matching_names = []
        for tool_def in self._catalog.list_tools():
            if tool_def.server_id == server_id:
                matching_names.append(tool_def.name)

        return self.to_langchain_tools(
            tool_names=matching_names,
            max_tier=max_tier,
        )

    def get_safe_tools(self) -> list[StructuredTool]:
        """
        Get only safe tools (tier 1-2, no risk).

        Returns:
            List of low-risk StructuredTools
        """
        safe_names = []
        for tool_def in self._catalog.list_tools():
            tier_info = TOOL_TIERS.get(
                _tier_key(tool_def.name),
                ToolTier(2, 1000, "low"),
            )
            if tier_info.tier <= 2 and tier_info.risk_level == "none":
                safe_names.append(tool_def.name)

        return self.to_langchain_tools(tool_names=safe_names, max_tier=2)

    def get_discovery_tools(self) -> list[StructuredTool]:
        """
        Get discovery/search tools only.

        Returns:
            List of discovery StructuredTools
        """
        discovery_names = [
            "oci_ping",
            "oci_list_domains",
            "oci_search_tools",
            "oci_get_capabilities",
            "search_databases",
            "get_fleet_summary",
        ]

        return self.to_langchain_tools(tool_names=discovery_names, max_tier=2)

    def resolve_tool_alias(self, alias: str) -> str:
        """
        Resolve a tool alias to canonical name.

        Args:
            alias: Possibly aliased tool name

        Returns:
            Canonical tool name
        """
        return TOOL_ALIASES.get(alias, alias)

    def get_tool_by_name(self, name: str) -> StructuredTool | None:
        """
        Get a previously converted tool by name.

        Args:
            name: Tool name (or alias)

        Returns:
            StructuredTool or None if not found
        """
        # Try exact match
        if name in self._converted_tools:
            return self._converted_tools[name]

        # Try alias resolution
        canonical = self.resolve_tool_alias(name)
        return self._converted_tools.get(canonical)

    def get_statistics(self) -> dict[str, Any]:
        """Get converter statistics."""
        by_domain: dict[str, int] = {}
        for name in self._converted_tools:
            for domain, prefixes in DOMAIN_PREFIXES.items():
                for prefix in prefixes:
                    if name.startswith(prefix) or name == prefix:
                        by_domain[domain] = by_domain.get(domain, 0) + 1
                        break

        by_tier: dict[int, int] = {1: 0, 2: 0, 3: 0, 4: 0}
        for name in self._converted_tools:
            tier = TOOL_TIERS.get(_tier_key(name), ToolTier(2, 1000, "low")).tier
            by_tier[tier] += 1

        return {
            "total_converted": len(self._converted_tools),
            "by_domain": by_domain,
            "by_tier": by_tier,
            "catalog_total": len(self._catalog.list_tools()),
        }
