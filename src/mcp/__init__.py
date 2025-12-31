"""
MCP (Model Context Protocol) Integration Layer.

This module provides the infrastructure for connecting to and managing
multiple MCP servers, discovering tools, and executing tool calls.

Architecture:
    ServerRegistry -> MCPClient -> Transport (stdio/HTTP/SSE)
                  -> ToolCatalog -> LangChain Integration

Components:
    - MCPClient: High-level client for MCP server communication
    - ServerRegistry: Multi-server connection management
    - ToolCatalog: Unified tool discovery with progressive disclosure
    - MCPConfig: Configuration loader for MCP servers

Quick Setup:
    from src.mcp import quick_setup
    registry, catalog = await quick_setup()
"""

from src.mcp.catalog import TOOL_TIERS, ToolCatalog, ToolTier
from src.mcp.client import (
    MCPClient,
    MCPError,
    MCPServerConfig,
    ResourceDefinition,
    ToolCallResult,
    ToolDefinition,
    TransportType,
)
from src.mcp.config import (
    MCPConfig,
    ServerConfigEntry,
    initialize_mcp_from_config,
    load_mcp_config,
    load_mcp_config_from_env,
    quick_setup,
)
from src.mcp.registry import ServerInfo, ServerRegistry, ServerStatus

__all__ = [
    # Client
    "MCPClient",
    "MCPError",
    "MCPServerConfig",
    "TransportType",
    "ToolDefinition",
    "ResourceDefinition",
    "ToolCallResult",
    # Registry
    "ServerRegistry",
    "ServerStatus",
    "ServerInfo",
    # Catalog
    "ToolCatalog",
    "ToolTier",
    "TOOL_TIERS",
    # Config
    "MCPConfig",
    "ServerConfigEntry",
    "load_mcp_config",
    "load_mcp_config_from_env",
    "initialize_mcp_from_config",
    "quick_setup",
]
