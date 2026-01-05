"""
MCP Server Registry.

Manages connections to multiple MCP servers with different transports.

Features:
- Dynamic server registration at runtime
- Health check loop with auto-reconnection
- Event callbacks for tool updates
- Circuit breaker pattern for failed servers
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import structlog

from src.mcp.client import (
    MCPClient,
    MCPError,
    MCPServerConfig,
    ToolDefinition,
    TransportType,
)

logger = structlog.get_logger(__name__)

# Event types for callbacks
EVENT_SERVER_CONNECTED = "server_connected"
EVENT_SERVER_DISCONNECTED = "server_disconnected"
EVENT_SERVER_ERROR = "server_error"
EVENT_TOOLS_UPDATED = "tools_updated"
EVENT_HEALTH_CHECK = "health_check"


class ServerStatus(str, Enum):
    """MCP server connection status."""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class ServerInfo:
    """Information about a registered MCP server."""

    config: MCPServerConfig
    client: MCPClient | None = None
    status: ServerStatus = ServerStatus.DISCONNECTED
    error_message: str | None = None
    tool_count: int = 0
    resource_count: int = 0
    # Health tracking
    last_health_check: datetime | None = None
    consecutive_failures: int = 0
    last_successful_call: datetime | None = None
    # Circuit breaker
    circuit_open_until: datetime | None = None
    # Metadata
    registered_at: datetime = field(default_factory=datetime.utcnow)
    connected_at: datetime | None = None
    working_dir: str | None = None
    domains: list[str] = field(default_factory=list)

    @property
    def is_circuit_open(self) -> bool:
        """Check if circuit breaker is open (server should not be called)."""
        if self.circuit_open_until is None:
            return False
        return datetime.utcnow() < self.circuit_open_until

    def record_success(self) -> None:
        """Record a successful call."""
        self.last_successful_call = datetime.utcnow()
        self.consecutive_failures = 0
        self.circuit_open_until = None

    def record_failure(self, max_failures: int = 3, backoff_seconds: int = 60) -> None:
        """Record a failed call, potentially opening circuit breaker."""
        self.consecutive_failures += 1
        if self.consecutive_failures >= max_failures:
            self.circuit_open_until = datetime.utcnow() + timedelta(seconds=backoff_seconds)


# Type alias for event callbacks
EventCallback = Callable[[str, str, dict[str, Any]], None]


class ServerRegistry:
    """
    Registry for managing MCP server connections.

    Provides:
    - Multi-server connection management
    - Dynamic server registration at runtime
    - Automatic reconnection with circuit breaker
    - Health monitoring with periodic checks
    - Event callbacks for tool updates
    - Unified tool discovery across servers
    """

    _instance: ServerRegistry | None = None

    def __init__(self):
        self._servers: dict[str, ServerInfo] = {}
        self._logger = logger.bind(component="ServerRegistry")
        self._event_callbacks: list[EventCallback] = []
        self._health_check_task: asyncio.Task | None = None
        self._health_check_interval: int = 30  # seconds
        self._running: bool = False
        self._tool_cache: dict[str, ToolDefinition] = {}
        self._tool_cache_time: datetime | None = None

    @classmethod
    def get_instance(cls) -> ServerRegistry:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    def register(self, config: MCPServerConfig) -> None:
        """
        Register an MCP server configuration.

        Args:
            config: Server configuration
        """
        if config.server_id in self._servers:
            self._logger.warning(
                "Server already registered, updating config",
                server_id=config.server_id,
            )

        self._servers[config.server_id] = ServerInfo(config=config)
        self._logger.info(
            "Server registered",
            server_id=config.server_id,
            transport=config.transport.value,
        )

    def register_from_dict(self, server_id: str, config_dict: dict[str, Any]) -> None:
        """
        Register a server from a configuration dictionary.

        Args:
            server_id: Unique server identifier
            config_dict: Configuration dictionary
        """
        transport_str = config_dict.get("transport", "http")
        transport = TransportType(transport_str)

        config = MCPServerConfig(
            server_id=server_id,
            transport=transport,
            command=config_dict.get("command"),
            args=config_dict.get("args", []),
            env=config_dict.get("env", {}),
            url=config_dict.get("url"),
            headers=config_dict.get("headers", {}),
            timeout_seconds=config_dict.get("timeout_seconds", 30),
        )

        self.register(config)

    def unregister(self, server_id: str) -> None:
        """Unregister and disconnect a server."""
        if server_id in self._servers:
            del self._servers[server_id]
            self._logger.info("Server unregistered", server_id=server_id)

    async def connect(self, server_id: str) -> None:
        """
        Connect to a specific server.

        Args:
            server_id: Server to connect to
        """
        if server_id not in self._servers:
            raise MCPError(-32600, f"Server not registered: {server_id}")

        info = self._servers[server_id]
        info.status = ServerStatus.CONNECTING

        try:
            client = MCPClient(info.config)
            await client.connect()

            info.client = client
            info.status = ServerStatus.CONNECTED
            info.tool_count = len(client.tools)
            info.resource_count = len(client.resources)
            info.error_message = None
            info.connected_at = datetime.utcnow()
            info.consecutive_failures = 0

            self._logger.info(
                "Server connected",
                server_id=server_id,
                tools=info.tool_count,
                resources=info.resource_count,
            )

        except Exception as e:
            info.status = ServerStatus.ERROR
            info.error_message = str(e)
            self._logger.error(
                "Server connection failed",
                server_id=server_id,
                error=str(e),
            )
            raise

    async def _connect_single(self, server_id: str) -> tuple[str, bool]:
        """
        Connect to a single server, returning success status.

        Used internally by connect_all for parallel connection.

        Returns:
            Tuple of (server_id, success_status)
        """
        try:
            await self.connect(server_id)
            return (server_id, True)
        except Exception:
            return (server_id, False)

    async def connect_all(self, parallel: bool = True) -> dict[str, bool]:
        """
        Connect to all registered servers.

        Args:
            parallel: If True (default), connect to all servers concurrently.
                     If False, connect sequentially (legacy behavior).

        Returns:
            Dictionary of server_id -> success status
        """
        if not parallel:
            # Legacy sequential behavior
            results = {}
            for server_id in self._servers:
                try:
                    await self.connect(server_id)
                    results[server_id] = True
                except Exception:
                    results[server_id] = False
            return results

        # Parallel connection - much faster with multiple servers
        import time
        start_time = time.time()

        tasks = [
            self._connect_single(server_id) for server_id in self._servers
        ]

        # Run all connections concurrently
        connection_results = await asyncio.gather(*tasks, return_exceptions=True)

        results = {}
        for result in connection_results:
            if isinstance(result, Exception):
                # Shouldn't happen with _connect_single, but handle gracefully
                self._logger.error("Unexpected connection error", error=str(result))
                continue
            server_id, success = result
            results[server_id] = success

        duration_ms = int((time.time() - start_time) * 1000)
        connected = sum(1 for s in results.values() if s)
        self._logger.info(
            "Parallel server connection complete",
            total=len(results),
            connected=connected,
            failed=len(results) - connected,
            duration_ms=duration_ms,
        )

        return results

    async def disconnect(self, server_id: str) -> None:
        """Disconnect from a specific server."""
        if server_id not in self._servers:
            return

        info = self._servers[server_id]
        if info.client:
            await info.client.disconnect()
            info.client = None

        info.status = ServerStatus.DISCONNECTED
        self._logger.info("Server disconnected", server_id=server_id)

    async def disconnect_all(self) -> None:
        """Disconnect from all servers."""
        for server_id in self._servers:
            await self.disconnect(server_id)

    def get_client(self, server_id: str) -> MCPClient | None:
        """Get the client for a server."""
        info = self._servers.get(server_id)
        return info.client if info else None

    def get_status(self, server_id: str) -> ServerStatus:
        """Get the status of a server."""
        info = self._servers.get(server_id)
        return info.status if info else ServerStatus.DISCONNECTED

    def list_servers(self) -> list[str]:
        """List all registered server IDs."""
        return list(self._servers.keys())

    def list_connected(self) -> list[str]:
        """List connected server IDs."""
        return [
            server_id
            for server_id, info in self._servers.items()
            if info.status == ServerStatus.CONNECTED
        ]

    def get_all_tools(self) -> dict[str, ToolDefinition]:
        """
        Get all tools from all connected servers.

        Returns:
            Dictionary of tool_name -> ToolDefinition
        """
        tools = {}
        for server_id, info in self._servers.items():
            if info.client and info.status == ServerStatus.CONNECTED:
                for name, tool_def in info.client.tools.items():
                    # Namespace tools by server ID
                    namespaced_name = f"{server_id}:{name}"
                    tools[namespaced_name] = tool_def
                    # Also store without namespace for convenience
                    if name not in tools:
                        tools[name] = tool_def
        return tools

    def get_server_info(self, server_id: str) -> dict[str, Any] | None:
        """Get detailed info about a server."""
        info = self._servers.get(server_id)
        if not info:
            return None

        return {
            "server_id": server_id,
            "transport": info.config.transport.value,
            "status": info.status.value,
            "tool_count": info.tool_count,
            "resource_count": info.resource_count,
            "error": info.error_message,
        }

    def get_health_summary(self) -> dict[str, Any]:
        """Get health summary of all servers."""
        connected = 0
        error = 0
        disconnected = 0

        for info in self._servers.values():
            if info.status == ServerStatus.CONNECTED:
                connected += 1
            elif info.status == ServerStatus.ERROR:
                error += 1
            else:
                disconnected += 1

        return {
            "total": len(self._servers),
            "connected": connected,
            "error": error,
            "disconnected": disconnected,
            "servers": {
                server_id: info.status.value for server_id, info in self._servers.items()
            },
        }

    # ========== Dynamic Registration Methods ==========

    async def register_dynamic(
        self,
        server_id: str,
        config_dict: dict[str, Any],
        auto_connect: bool = True,
    ) -> bool:
        """
        Dynamically register and optionally connect a new MCP server at runtime.

        Args:
            server_id: Unique server identifier
            config_dict: Configuration dictionary with transport, command, args, etc.
            auto_connect: Whether to immediately connect to the server

        Returns:
            True if registration and connection succeeded
        """
        self._logger.info(
            "Dynamic server registration",
            server_id=server_id,
            transport=config_dict.get("transport"),
        )

        # Create config
        transport_str = config_dict.get("transport", "http")
        transport = TransportType(transport_str)

        config = MCPServerConfig(
            server_id=server_id,
            transport=transport,
            command=config_dict.get("command"),
            args=config_dict.get("args", []),
            env=config_dict.get("env", {}),
            url=config_dict.get("url"),
            headers=config_dict.get("headers", {}),
            timeout_seconds=config_dict.get("timeout_seconds", 30),
            working_dir=config_dict.get("working_dir"),
        )

        # Register with metadata
        info = ServerInfo(
            config=config,
            working_dir=config_dict.get("working_dir"),
            domains=config_dict.get("domains", []),
        )
        self._servers[server_id] = info

        self._logger.info(
            "Server registered dynamically",
            server_id=server_id,
            transport=transport.value,
            domains=info.domains,
        )

        # Connect if requested
        if auto_connect:
            try:
                await self.connect(server_id)
                self._emit_event(EVENT_SERVER_CONNECTED, server_id, {
                    "tools": info.tool_count,
                    "resources": info.resource_count,
                })
                # Invalidate tool cache
                self._tool_cache_time = None
                self._emit_event(EVENT_TOOLS_UPDATED, server_id, {
                    "action": "server_added",
                    "tool_count": info.tool_count,
                })
                return True
            except Exception as e:
                self._logger.error(
                    "Dynamic server connection failed",
                    server_id=server_id,
                    error=str(e),
                )
                return False

        return True

    async def unregister_dynamic(self, server_id: str) -> bool:
        """
        Dynamically unregister and disconnect an MCP server at runtime.

        Args:
            server_id: Server to unregister

        Returns:
            True if unregistration succeeded
        """
        if server_id not in self._servers:
            return False

        await self.disconnect(server_id)
        del self._servers[server_id]

        self._emit_event(EVENT_SERVER_DISCONNECTED, server_id, {
            "action": "unregistered",
        })

        # Invalidate tool cache
        self._tool_cache_time = None
        self._emit_event(EVENT_TOOLS_UPDATED, server_id, {
            "action": "server_removed",
        })

        self._logger.info("Server unregistered dynamically", server_id=server_id)
        return True

    # ========== Event System ==========

    def on_event(self, callback: EventCallback) -> None:
        """
        Register an event callback.

        Callback signature: callback(event_type, server_id, data)
        """
        self._event_callbacks.append(callback)

    def _emit_event(self, event_type: str, server_id: str, data: dict[str, Any]) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event_type, server_id, data)
            except Exception as e:
                self._logger.warning(
                    "Event callback error",
                    event=event_type,
                    error=str(e),
                )

    # ========== Health Check Loop ==========

    async def start_health_checks(self, interval_seconds: int = 30) -> None:
        """
        Start the background health check loop.

        Args:
            interval_seconds: Interval between health checks
        """
        if self._running:
            return

        self._health_check_interval = interval_seconds
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        self._logger.info(
            "Health check loop started",
            interval=interval_seconds,
        )

    async def stop_health_checks(self) -> None:
        """Stop the background health check loop."""
        self._running = False
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
            self._health_check_task = None
        self._logger.info("Health check loop stopped")

    async def _health_check_loop(self) -> None:
        """Background loop that checks server health and reconnects failed servers."""
        while self._running:
            try:
                await asyncio.sleep(self._health_check_interval)
                await self._run_health_checks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Health check loop error", error=str(e))

    async def _run_health_checks(self) -> None:
        """Run health checks on all servers."""
        for server_id, info in self._servers.items():
            info.last_health_check = datetime.utcnow()

            # Skip if circuit is open
            if info.is_circuit_open:
                self._logger.debug(
                    "Skipping health check (circuit open)",
                    server_id=server_id,
                    until=info.circuit_open_until,
                )
                continue

            # Check if connected server is still responsive
            if info.status == ServerStatus.CONNECTED and info.client:
                try:
                    # Simple health check: try to list tools
                    tools = info.client.tools
                    if len(tools) != info.tool_count:
                        # Tools changed - update and notify
                        old_count = info.tool_count
                        info.tool_count = len(tools)
                        self._tool_cache_time = None
                        self._emit_event(EVENT_TOOLS_UPDATED, server_id, {
                            "action": "tools_changed",
                            "old_count": old_count,
                            "new_count": info.tool_count,
                        })
                    info.record_success()
                except Exception as e:
                    self._logger.warning(
                        "Health check failed",
                        server_id=server_id,
                        error=str(e),
                    )
                    info.record_failure()
                    info.status = ServerStatus.ERROR
                    info.error_message = str(e)
                    self._emit_event(EVENT_SERVER_ERROR, server_id, {
                        "error": str(e),
                    })

            # Try to reconnect disconnected/error servers
            elif info.status in (ServerStatus.DISCONNECTED, ServerStatus.ERROR):
                try:
                    self._logger.info(
                        "Attempting reconnection",
                        server_id=server_id,
                        failures=info.consecutive_failures,
                    )
                    await self.connect(server_id)
                    self._emit_event(EVENT_SERVER_CONNECTED, server_id, {
                        "reconnected": True,
                        "tools": info.tool_count,
                    })
                    self._tool_cache_time = None
                except Exception as e:
                    self._logger.warning(
                        "Reconnection failed",
                        server_id=server_id,
                        error=str(e),
                    )
                    info.record_failure()

        self._emit_event(EVENT_HEALTH_CHECK, "registry", {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": self.get_health_summary(),
        })

    # ========== Tool Discovery with Caching ==========

    def get_all_tools_cached(self, max_age_seconds: int = 60) -> dict[str, ToolDefinition]:
        """
        Get all tools with caching.

        Args:
            max_age_seconds: Maximum age of cache before refresh

        Returns:
            Dictionary of tool_name -> ToolDefinition
        """
        now = datetime.utcnow()

        # Return cache if still valid
        if (
            self._tool_cache_time
            and (now - self._tool_cache_time).total_seconds() < max_age_seconds
        ):
            return self._tool_cache

        # Rebuild cache
        self._tool_cache = self.get_all_tools()
        self._tool_cache_time = now
        return self._tool_cache

    def invalidate_tool_cache(self) -> None:
        """Invalidate the tool cache, forcing refresh on next access."""
        self._tool_cache_time = None

    # ========== Server Discovery by Domain ==========

    def get_servers_by_domain(self, domain: str) -> list[str]:
        """
        Get server IDs that provide tools for a specific domain.

        Args:
            domain: Domain to search for (e.g., 'database', 'cost', 'security')

        Returns:
            List of server IDs
        """
        matching = []
        for server_id, info in self._servers.items():
            if domain in info.domains:
                matching.append(server_id)
        return matching

    def get_best_server_for_domain(self, domain: str) -> str | None:
        """
        Get the best available server for a domain.

        Considers: connection status, circuit breaker, performance.

        Args:
            domain: Domain to search for

        Returns:
            Best server ID or None
        """
        candidates = self.get_servers_by_domain(domain)
        if not candidates:
            return None

        # Filter to connected servers with closed circuits
        available = [
            server_id
            for server_id in candidates
            if (
                self._servers[server_id].status == ServerStatus.CONNECTED
                and not self._servers[server_id].is_circuit_open
            )
        ]

        if not available:
            return None

        # Sort by consecutive failures (prefer servers with fewer failures)
        available.sort(key=lambda s: self._servers[s].consecutive_failures)
        return available[0]

    # ========== Detailed Server Status ==========

    def get_detailed_status(self) -> dict[str, Any]:
        """Get detailed status of all servers including health metrics."""
        servers = {}
        for server_id, info in self._servers.items():
            servers[server_id] = {
                "status": info.status.value,
                "transport": info.config.transport.value,
                "tool_count": info.tool_count,
                "resource_count": info.resource_count,
                "domains": info.domains,
                "error": info.error_message,
                "consecutive_failures": info.consecutive_failures,
                "circuit_open": info.is_circuit_open,
                "last_health_check": info.last_health_check.isoformat() if info.last_health_check else None,
                "last_successful_call": info.last_successful_call.isoformat() if info.last_successful_call else None,
                "connected_at": info.connected_at.isoformat() if info.connected_at else None,
                "registered_at": info.registered_at.isoformat(),
            }

        return {
            "total_servers": len(self._servers),
            "health_check_running": self._running,
            "health_check_interval": self._health_check_interval,
            "tool_cache_valid": self._tool_cache_time is not None,
            "servers": servers,
        }
