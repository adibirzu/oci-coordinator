"""
MCP Server Configuration Loader.

Loads MCP server configurations from YAML files and environment variables.
Provides a flexible system for users to configure their own MCP servers.

Usage:
    from src.mcp.config import load_mcp_config, MCPConfig

    config = load_mcp_config()
    for server_id, server_config in config.get_enabled_servers().items():
        print(f"Server: {server_id}, Transport: {server_config.transport}")
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import re

import structlog
import yaml

from src.mcp.client import MCPServerConfig, TransportType

logger = structlog.get_logger(__name__)


def expand_bash_vars(value: str) -> str:
    """Expand bash-style environment variables with default values.

    Handles the following patterns:
    - $VAR or ${VAR} - standard variable expansion
    - ${VAR:-default} - use default if VAR is unset or empty
    - ${VAR-default} - use default if VAR is unset (but not if empty)

    Args:
        value: String potentially containing bash variable references

    Returns:
        String with all variables expanded
    """
    if not value or "$" not in value:
        return value

    # Pattern for ${VAR:-default} or ${VAR-default}
    pattern = r"\$\{([^}:]+)(:-?)?([^}]*)?\}"

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1)
        operator = match.group(2)  # :- or - or None
        default = match.group(3) or ""

        env_value = os.environ.get(var_name)

        if operator == ":-":
            # Use default if unset or empty
            return env_value if env_value else default
        elif operator == "-":
            # Use default only if unset
            return env_value if env_value is not None else default
        else:
            # No default, just expand
            return env_value or ""

    # First expand ${VAR:-default} patterns
    result = re.sub(pattern, replace_var, value)

    # Then use standard expandvars for remaining $VAR patterns
    return os.path.expandvars(result)


@dataclass
class ServerConfigEntry:
    """Configuration entry for a single MCP server."""

    server_id: str
    transport: TransportType
    enabled: bool = True
    command: str | None = None
    args: list[str] = field(default_factory=list)
    working_dir: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    url: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    domains: list[str] = field(default_factory=list)
    description: str = ""
    timeout_seconds: int = 30
    retry_attempts: int = 3
    tool_timeouts: dict[str, int] = field(default_factory=dict)

    def to_mcp_config(self) -> MCPServerConfig:
        """Convert to MCPServerConfig for use with MCP client."""
        # Expand environment variables (including bash-style ${VAR:-default}) and ~ in paths
        env = {}
        for key, value in self.env.items():
            env[key] = expand_bash_vars(os.path.expanduser(value))

        command = self.command
        if command:
            command = expand_bash_vars(os.path.expanduser(command))

        working_dir = self.working_dir
        if working_dir:
            working_dir = expand_bash_vars(os.path.expanduser(working_dir))

        return MCPServerConfig(
            server_id=self.server_id,
            transport=self.transport,
            command=command,
            args=self.args,
            env=env,
            working_dir=working_dir,
            url=self.url,
            headers=self.headers,
            timeout_seconds=self.timeout_seconds,
            retry_attempts=self.retry_attempts,
            tool_timeouts=self.tool_timeouts,
        )


@dataclass
class MCPConfig:
    """Complete MCP configuration with servers and groups."""

    servers: dict[str, ServerConfigEntry] = field(default_factory=dict)
    groups: dict[str, list[str]] = field(default_factory=dict)
    defaults: dict[str, Any] = field(default_factory=dict)

    def get_enabled_servers(self) -> dict[str, ServerConfigEntry]:
        """Get all enabled servers."""
        return {
            server_id: config
            for server_id, config in self.servers.items()
            if config.enabled
        }

    def get_servers_for_domain(self, domain: str) -> list[ServerConfigEntry]:
        """Get all enabled servers that provide a specific domain."""
        return [
            config
            for config in self.servers.values()
            if config.enabled and domain in config.domains
        ]

    def get_servers_for_group(self, group: str) -> list[ServerConfigEntry]:
        """Get all enabled servers in a group."""
        server_ids = self.groups.get(group, [])
        return [
            self.servers[server_id]
            for server_id in server_ids
            if server_id in self.servers and self.servers[server_id].enabled
        ]

    def get_server(self, server_id: str) -> ServerConfigEntry | None:
        """Get a specific server configuration."""
        return self.servers.get(server_id)


def load_mcp_config(
    config_path: str | Path | None = None,
) -> MCPConfig:
    """
    Load MCP configuration from YAML file.

    Args:
        config_path: Path to config file. Defaults to config/mcp_servers.yaml

    Returns:
        MCPConfig with loaded server configurations
    """
    if config_path is None:
        # Default to config/mcp_servers.yaml relative to project root
        config_path = Path(__file__).parent.parent.parent / "config" / "mcp_servers.yaml"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        logger.warning(
            "MCP config file not found, using empty configuration",
            path=str(config_path),
        )
        return MCPConfig()

    logger.info("Loading MCP configuration", path=str(config_path))

    with open(config_path) as f:
        data = yaml.safe_load(f)

    if not data:
        return MCPConfig()

    # Parse defaults
    defaults = data.get("defaults", {})

    # Parse servers
    servers: dict[str, ServerConfigEntry] = {}
    for server_id, server_data in data.get("servers", {}).items():
        if server_data is None:
            continue

        # Determine transport type
        transport_str = server_data.get("transport", "stdio")
        try:
            transport = TransportType(transport_str)
        except ValueError:
            logger.warning(
                "Unknown transport type, skipping server",
                server_id=server_id,
                transport=transport_str,
            )
            continue

        # Apply defaults
        timeout = server_data.get("timeout_seconds", defaults.get("timeout_seconds", 30))
        retry = server_data.get("retry_attempts", defaults.get("retry_attempts", 3))
        # Tool-specific timeouts from defaults
        tool_timeouts = defaults.get("tool_timeouts", {})

        servers[server_id] = ServerConfigEntry(
            server_id=server_id,
            transport=transport,
            enabled=server_data.get("enabled", True),
            command=server_data.get("command"),
            args=server_data.get("args", []),
            working_dir=server_data.get("working_dir"),
            env=server_data.get("env", {}),
            url=server_data.get("url"),
            headers=server_data.get("headers", {}),
            domains=server_data.get("domains", []),
            description=server_data.get("description", ""),
            timeout_seconds=timeout,
            retry_attempts=retry,
            tool_timeouts=tool_timeouts,
        )

    logger.info(
        "MCP configuration loaded",
        total_servers=len(servers),
        enabled_servers=len([s for s in servers.values() if s.enabled]),
    )

    # Parse groups
    groups = data.get("groups", {})

    return MCPConfig(
        servers=servers,
        groups=groups,
        defaults=defaults,
    )


def load_mcp_config_from_env() -> MCPConfig:
    """
    Load MCP configuration from environment variables.

    Environment variable format:
    - MCP_SERVER_{NAME}_TRANSPORT: Transport type (stdio, http, sse)
    - MCP_SERVER_{NAME}_COMMAND: Command for stdio transport
    - MCP_SERVER_{NAME}_ARGS: Comma-separated args for stdio
    - MCP_SERVER_{NAME}_URL: URL for http/sse transport
    - MCP_SERVER_{NAME}_ENABLED: true/false

    Returns:
        MCPConfig from environment variables
    """
    servers: dict[str, ServerConfigEntry] = {}

    # Find all MCP_SERVER_* environment variables
    env_prefix = "MCP_SERVER_"
    server_names: set[str] = set()

    for key in os.environ:
        if key.startswith(env_prefix):
            # Extract server name (e.g., MCP_SERVER_DATABASE_TRANSPORT -> DATABASE)
            parts = key[len(env_prefix):].split("_")
            if len(parts) >= 2:
                server_names.add(parts[0])

    for name in server_names:
        prefix = f"{env_prefix}{name}_"

        transport_str = os.getenv(f"{prefix}TRANSPORT", "stdio")
        try:
            transport = TransportType(transport_str)
        except ValueError:
            continue

        enabled = os.getenv(f"{prefix}ENABLED", "true").lower() == "true"
        command = os.getenv(f"{prefix}COMMAND")
        args_str = os.getenv(f"{prefix}ARGS", "")
        args = [a.strip() for a in args_str.split(",") if a.strip()]
        url = os.getenv(f"{prefix}URL")
        domains_str = os.getenv(f"{prefix}DOMAINS", "")
        domains = [d.strip() for d in domains_str.split(",") if d.strip()]

        server_id = name.lower().replace("_", "-")

        servers[server_id] = ServerConfigEntry(
            server_id=server_id,
            transport=transport,
            enabled=enabled,
            command=command,
            args=args,
            url=url,
            domains=domains,
        )

    return MCPConfig(servers=servers)


async def initialize_mcp_from_config(
    config: MCPConfig | None = None,
) -> tuple[ServerRegistry, ToolCatalog]:
    """
    Initialize MCP infrastructure from configuration.

    Args:
        config: MCP configuration (loads default if not provided)

    Returns:
        Tuple of (ServerRegistry, ToolCatalog) initialized and connected
    """
    from src.mcp.catalog import ToolCatalog
    from src.mcp.registry import ServerRegistry

    if config is None:
        config = load_mcp_config()

    registry = ServerRegistry.get_instance()

    # Register all enabled servers
    for server_id, server_config in config.get_enabled_servers().items():
        mcp_config = server_config.to_mcp_config()

        # Handle working directory for stdio transport
        if server_config.working_dir and server_config.transport == TransportType.STDIO:
            # We need to change to the working directory when spawning
            # This is handled in the transport layer
            pass

        registry.register(mcp_config)

    # Connect to all servers
    results = await registry.connect_all()

    connected_count = sum(1 for success in results.values() if success)
    logger.info(
        "MCP servers initialized",
        total=len(results),
        connected=connected_count,
        failed=len(results) - connected_count,
    )

    # Initialize tool catalog
    catalog = ToolCatalog.get_instance(registry)
    await catalog.refresh()

    return registry, catalog


# Convenience function for quick setup
async def quick_setup() -> tuple[ServerRegistry, ToolCatalog]:
    """
    Quick setup for MCP infrastructure using default configuration.

    Returns:
        Tuple of (ServerRegistry, ToolCatalog) ready for use
    """
    return await initialize_mcp_from_config()
