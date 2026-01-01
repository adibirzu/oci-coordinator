"""
MCP Connection Manager.

Provides persistent MCP server connections that survive across multiple
channel handler invocations. Uses the shared AsyncRuntime event loop
to maintain stable connections.

This solves the problem where MCP connections were being reset on every
Slack message, causing 2-5s overhead per request.

Usage:
    from src.mcp.connection_manager import MCPConnectionManager

    # Get or initialize connections (fast if already connected)
    manager = await MCPConnectionManager.get_instance()
    catalog = await manager.get_tool_catalog()

    # Execute tool
    result = await catalog.execute("oci_list_compartments", {})
"""

from __future__ import annotations

import asyncio
import threading
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from src.mcp.catalog import ToolCatalog
    from src.mcp.registry import ServerRegistry

logger = structlog.get_logger(__name__)


class MCPConnectionManager:
    """
    Manages persistent MCP server connections.

    Features:
    - Singleton pattern with thread-safe initialization
    - Lazy connection establishment on first use
    - Connection health monitoring with auto-reconnect
    - Event loop affinity tracking
    - Graceful degradation if connections fail
    """

    _instance: MCPConnectionManager | None = None
    _lock = threading.Lock()
    _init_lock = asyncio.Lock()

    def __init__(self):
        self._registry: ServerRegistry | None = None
        self._catalog: ToolCatalog | None = None
        self._initialized = False
        self._initializing = False
        self._last_health_check: datetime | None = None
        self._health_check_interval = timedelta(seconds=60)
        self._event_loop_id: int | None = None
        self._init_time: float = 0
        self._logger = logger.bind(component="MCPConnectionManager")

    @classmethod
    async def get_instance(cls) -> MCPConnectionManager:
        """
        Get the singleton instance, initializing connections if needed.

        Thread-safe and async-safe initialization.
        """
        # Fast path - already initialized
        if cls._instance is not None and cls._instance._initialized:
            # Verify we're in the same event loop
            current_loop_id = id(asyncio.get_event_loop())
            if cls._instance._event_loop_id == current_loop_id:
                # Check if health check is needed
                await cls._instance._maybe_health_check()
                return cls._instance
            else:
                # Event loop changed - need to reinitialize
                cls._instance._logger.warning(
                    "Event loop changed, reinitializing MCP connections",
                    old_loop=cls._instance._event_loop_id,
                    new_loop=current_loop_id,
                )
                cls._instance._initialized = False

        # Slow path - need to initialize
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()

        # Initialize connections (async-safe)
        await cls._instance._ensure_initialized()
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton (for testing)."""
        if cls._instance is not None:
            cls._instance._initialized = False
            cls._instance._registry = None
            cls._instance._catalog = None
        cls._instance = None

    async def _ensure_initialized(self) -> None:
        """Ensure connections are initialized (thread and async safe)."""
        if self._initialized:
            return

        async with self._init_lock:
            # Double-check after acquiring lock
            if self._initialized:
                return

            if self._initializing:
                # Wait for another coroutine to finish initialization
                while self._initializing and not self._initialized:
                    await asyncio.sleep(0.1)
                return

            self._initializing = True
            try:
                await self._initialize_connections()
                self._initialized = True
                self._event_loop_id = id(asyncio.get_event_loop())
            finally:
                self._initializing = False

    async def _initialize_connections(self) -> None:
        """Initialize MCP server connections."""
        from src.mcp.catalog import ToolCatalog
        from src.mcp.config import initialize_mcp_from_config, load_mcp_config
        from src.mcp.registry import ServerRegistry

        self._logger.info("Initializing MCP connections...")
        start_time = time.time()

        try:
            # Reset any existing singletons to ensure clean state
            ServerRegistry.reset_instance()
            ToolCatalog.reset_instance()

            # Load config and initialize
            config = load_mcp_config()
            self._registry, self._catalog = await initialize_mcp_from_config(config)

            self._init_time = time.time() - start_time
            self._last_health_check = datetime.utcnow()

            connected = self._registry.list_connected() if self._registry else []
            tool_count = len(self._catalog.list_tools()) if self._catalog else 0

            self._logger.info(
                "MCP connections initialized",
                duration_s=f"{self._init_time:.2f}",
                connected_servers=connected,
                tool_count=tool_count,
            )

        except Exception as e:
            self._logger.error("Failed to initialize MCP connections", error=str(e))
            raise

    async def _maybe_health_check(self) -> None:
        """Perform health check if interval has passed."""
        if self._last_health_check is None:
            return

        now = datetime.utcnow()
        if now - self._last_health_check < self._health_check_interval:
            return

        await self._health_check()

    async def _health_check(self) -> None:
        """Check connection health and reconnect if needed."""
        if not self._registry:
            return

        self._last_health_check = datetime.utcnow()

        try:
            health = self._registry.get_health_summary()
            if health["error"] > 0 or health["disconnected"] > 0:
                self._logger.warning(
                    "MCP health check found issues, attempting reconnect",
                    connected=health["connected"],
                    error=health["error"],
                    disconnected=health["disconnected"],
                )
                # Attempt to reconnect failed servers
                await self._registry.connect_all()

                # Refresh tool catalog
                if self._catalog:
                    await self._catalog.refresh()

        except Exception as e:
            self._logger.error("Health check failed", error=str(e))

    async def get_tool_catalog(self) -> ToolCatalog | None:
        """
        Get the tool catalog with active MCP connections.

        Returns:
            ToolCatalog instance or None if not available
        """
        await self._ensure_initialized()
        return self._catalog

    async def get_registry(self) -> ServerRegistry | None:
        """
        Get the server registry.

        Returns:
            ServerRegistry instance or None if not available
        """
        await self._ensure_initialized()
        return self._registry

    def get_status(self) -> dict:
        """Get connection manager status."""
        return {
            "initialized": self._initialized,
            "event_loop_id": self._event_loop_id,
            "init_time_s": self._init_time,
            "last_health_check": (
                self._last_health_check.isoformat() if self._last_health_check else None
            ),
            "connected_servers": (
                self._registry.list_connected() if self._registry else []
            ),
            "tool_count": len(self._catalog.list_tools()) if self._catalog else 0,
        }

    async def disconnect_all(self) -> None:
        """Disconnect all MCP servers."""
        if self._registry:
            await self._registry.disconnect_all()
        self._initialized = False
        self._logger.info("All MCP connections disconnected")


# Convenience function for getting tool catalog
async def get_mcp_catalog() -> ToolCatalog | None:
    """
    Get the MCP tool catalog with persistent connections.

    This is the recommended way to access MCP tools from channel handlers.

    Returns:
        ToolCatalog instance or None if not available
    """
    try:
        manager = await MCPConnectionManager.get_instance()
        return await manager.get_tool_catalog()
    except Exception as e:
        logger.error("Failed to get MCP catalog", error=str(e))
        return None
