"""
Dynamic Tool and Skill Manager.

Bridges MCP servers, Tool Catalog, Agent Catalog, and Memory stores
to enable automatic synchronization when new tools or skills are available.

Features:
- Auto-sync tools from MCP servers to agents
- Register runbooks as executable skills
- Update agent capabilities dynamically
- Event-driven updates across the system
- Hot-reload support for new features
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable

import structlog

from src.agents.catalog import AgentCatalog
from src.mcp.catalog import ToolCatalog
from src.mcp.registry import (
    ServerRegistry,
    EVENT_SERVER_CONNECTED,
    EVENT_SERVER_DISCONNECTED,
    EVENT_TOOLS_UPDATED,
)

logger = structlog.get_logger(__name__)


class UpdateEvent(str, Enum):
    """Types of dynamic updates."""

    TOOLS_ADDED = "tools_added"
    TOOLS_REMOVED = "tools_removed"
    SKILLS_REGISTERED = "skills_registered"
    RUNBOOKS_REGISTERED = "runbooks_registered"
    AGENTS_UPDATED = "agents_updated"
    SYNC_COMPLETE = "sync_complete"


@dataclass
class ToolChangeEvent:
    """Event representing a tool change."""

    event_type: UpdateEvent
    timestamp: datetime
    server_id: str | None
    tools_affected: list[str]
    details: dict[str, Any]


@dataclass
class DynamicRegistration:
    """Information about a dynamically registered tool/skill."""

    name: str
    description: str
    source: str  # 'mcp', 'runbook', 'skill', 'custom'
    domains: list[str]
    registered_at: datetime
    handler: Callable | None = None
    schema: dict[str, Any] | None = None


class DynamicToolManager:
    """
    Manages dynamic tool and skill registration across the system.

    Responsibilities:
    - Listen for MCP server tool changes
    - Register runbooks as executable skills
    - Update agent capabilities dynamically
    - Maintain synchronization between catalogs
    - Broadcast updates to interested parties

    Usage:
        manager = DynamicToolManager.get_instance()

        # Register callbacks for updates
        manager.on_update(UpdateEvent.TOOLS_ADDED, my_callback)

        # Start auto-sync
        await manager.start()

        # Register a runbook
        await manager.register_runbook("db-health-check", executor)
    """

    _instance: DynamicToolManager | None = None

    def __init__(
        self,
        server_registry: ServerRegistry | None = None,
        tool_catalog: ToolCatalog | None = None,
        agent_catalog: AgentCatalog | None = None,
    ):
        self._server_registry = server_registry or ServerRegistry.get_instance()
        self._tool_catalog = tool_catalog or ToolCatalog.get_instance()
        self._agent_catalog = agent_catalog or AgentCatalog.get_instance(
            tool_catalog=self._tool_catalog
        )

        self._logger = logger.bind(component="DynamicToolManager")
        self._callbacks: dict[UpdateEvent, list[Callable]] = {}
        self._registrations: dict[str, DynamicRegistration] = {}
        self._running = False
        self._sync_task: asyncio.Task | None = None
        self._last_sync: datetime | None = None
        self._sync_interval = timedelta(seconds=60)

        # Track tool counts for change detection
        self._previous_tool_count = 0

    @classmethod
    def get_instance(
        cls,
        server_registry: ServerRegistry | None = None,
        tool_catalog: ToolCatalog | None = None,
        agent_catalog: AgentCatalog | None = None,
    ) -> DynamicToolManager:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(server_registry, tool_catalog, agent_catalog)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    # ─────────────────────────────────────────────────────────────────────────
    # Lifecycle Management
    # ─────────────────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """
        Start the dynamic manager.

        Registers event listeners and begins auto-sync loop.
        """
        if self._running:
            return

        self._running = True

        # Register for MCP server events
        self._server_registry.on_event(self._handle_server_event)

        # Register for tool catalog events
        self._tool_catalog.on_event("refresh", self._handle_catalog_refresh)
        self._tool_catalog.on_event("tool_registered", self._handle_tool_registered)

        # Do initial sync
        await self.sync_all()

        # Start background sync task
        self._sync_task = asyncio.create_task(self._sync_loop())

        self._logger.info("Dynamic tool manager started")

    async def stop(self) -> None:
        """Stop the dynamic manager."""
        self._running = False

        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
            self._sync_task = None

        self._logger.info("Dynamic tool manager stopped")

    async def _sync_loop(self) -> None:
        """Background loop for periodic synchronization."""
        while self._running:
            try:
                await asyncio.sleep(self._sync_interval.total_seconds())
                await self.sync_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Sync loop error", error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Event Handling
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_server_event(
        self,
        event_type: str,
        server_id: str,
        data: dict[str, Any],
    ) -> None:
        """Handle events from MCP server registry."""
        if event_type == EVENT_SERVER_CONNECTED:
            self._logger.info(
                "Server connected",
                server_id=server_id,
                tools=data.get("tools", 0),
            )
            # Schedule async sync
            asyncio.create_task(self._sync_server_tools(server_id))

        elif event_type == EVENT_SERVER_DISCONNECTED:
            self._logger.info("Server disconnected", server_id=server_id)
            # Remove tools from this server
            asyncio.create_task(self._remove_server_tools(server_id))

        elif event_type == EVENT_TOOLS_UPDATED:
            self._logger.info(
                "Tools updated",
                server_id=server_id,
                action=data.get("action"),
            )
            asyncio.create_task(self._sync_server_tools(server_id))

    async def _handle_catalog_refresh(
        self,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """Handle tool catalog refresh events."""
        new_count = data.get("tool_count", 0)

        if new_count != self._previous_tool_count:
            self._logger.info(
                "Tool catalog changed",
                previous=self._previous_tool_count,
                current=new_count,
            )
            self._previous_tool_count = new_count

            # Update agent catalogs
            await self._update_agent_tools()

            # Emit event
            await self._emit_event(ToolChangeEvent(
                event_type=UpdateEvent.TOOLS_ADDED if new_count > self._previous_tool_count else UpdateEvent.TOOLS_REMOVED,
                timestamp=datetime.utcnow(),
                server_id=None,
                tools_affected=[],
                details={"old_count": self._previous_tool_count, "new_count": new_count},
            ))

    async def _handle_tool_registered(
        self,
        event: str,
        data: dict[str, Any],
    ) -> None:
        """Handle new tool registration."""
        tool_name = data.get("name", "unknown")
        self._logger.info("Tool registered", tool=tool_name)

        # Update agent domain tools
        await self._update_agent_tools()

    # ─────────────────────────────────────────────────────────────────────────
    # Synchronization
    # ─────────────────────────────────────────────────────────────────────────

    async def sync_all(self) -> dict[str, Any]:
        """
        Synchronize all tools, skills, and agent capabilities.

        Returns:
            Summary of synchronization results
        """
        self._last_sync = datetime.utcnow()

        results = {
            "timestamp": self._last_sync.isoformat(),
            "mcp_tools": 0,
            "runbooks": 0,
            "skills": 0,
            "agents_updated": 0,
        }

        try:
            # 1. Refresh tool catalog
            await self._tool_catalog.ensure_fresh(force=True)
            results["mcp_tools"] = len(self._tool_catalog.list_tools())

            # 2. Sync runbooks
            runbook_count = await self._sync_runbooks()
            results["runbooks"] = runbook_count

            # 3. Sync skills from MCP-OCI
            skill_count = await self._sync_skills()
            results["skills"] = skill_count

            # 4. Update agent capabilities
            agents_updated = await self._update_agent_tools()
            results["agents_updated"] = agents_updated

            self._previous_tool_count = results["mcp_tools"]

            self._logger.info("Sync complete", **results)

            # Emit sync complete event
            await self._emit_event(ToolChangeEvent(
                event_type=UpdateEvent.SYNC_COMPLETE,
                timestamp=self._last_sync,
                server_id=None,
                tools_affected=[],
                details=results,
            ))

        except Exception as e:
            self._logger.error("Sync failed", error=str(e))
            results["error"] = str(e)

        return results

    async def _sync_server_tools(self, server_id: str) -> int:
        """Sync tools from a specific server."""
        # Get tools from this server
        all_tools = self._server_registry.get_all_tools()

        server_tools = [
            name for name, tool in all_tools.items()
            if tool.server_id == server_id
        ]

        self._logger.debug(
            "Syncing server tools",
            server_id=server_id,
            tool_count=len(server_tools),
        )

        await self._tool_catalog.ensure_fresh(force=True)
        await self._update_agent_tools()

        return len(server_tools)

    async def _remove_server_tools(self, server_id: str) -> None:
        """Remove tools from a disconnected server."""
        # Get registrations for this server
        to_remove = [
            name for name, reg in self._registrations.items()
            if reg.source == "mcp" and f":{server_id}:" in name
        ]

        for name in to_remove:
            del self._registrations[name]
            self._logger.debug("Removed tool", tool=name, server=server_id)

    async def _sync_runbooks(self) -> int:
        """Sync runbooks from MCP-OCI server."""
        try:
            # Try to import runbooks from MCP-OCI
            # These are loaded dynamically if the module is available
            from importlib import import_module

            runbooks_module = import_module("mcp_server_oci.skills.runbooks")
            list_runbooks = getattr(runbooks_module, "list_runbooks", None)

            if list_runbooks:
                runbooks = list_runbooks()
                for runbook in runbooks:
                    await self.register_runbook_skill(
                        runbook_id=runbook.id,
                        name=runbook.name,
                        description=runbook.description,
                        category=runbook.category,
                    )
                return len(runbooks)

        except ImportError:
            self._logger.debug("Runbooks module not available locally")
        except Exception as e:
            self._logger.warning("Failed to sync runbooks", error=str(e))

        return 0

    async def _sync_skills(self) -> int:
        """Sync skills from MCP servers."""
        # Look for skill tools in the catalog
        skill_count = 0

        for tool in self._tool_catalog.list_tools():
            if "_skill_" in tool.name or tool.name.startswith("oci_skill"):
                skill_count += 1
                # Register as a skill
                self._registrations[tool.name] = DynamicRegistration(
                    name=tool.name,
                    description=tool.description,
                    source="skill",
                    domains=self._detect_skill_domains(tool.name),
                    registered_at=datetime.utcnow(),
                    schema=tool.input_schema,
                )

        return skill_count

    async def _update_agent_tools(self) -> int:
        """Update agent catalog with new tool information."""
        updated_count = self._agent_catalog.sync_mcp_tools(self._tool_catalog)
        if updated_count > 0:
            self._logger.info("Updated agent tool lists", count=updated_count)
        return updated_count

    def _detect_skill_domains(self, skill_name: str) -> list[str]:
        """Detect domains for a skill."""
        domains = []
        skill_lower = skill_name.lower()

        if "database" in skill_lower or "db" in skill_lower:
            domains.append("database")
        if "instance" in skill_lower or "compute" in skill_lower:
            domains.append("infrastructure")
        if "security" in skill_lower or "audit" in skill_lower:
            domains.append("security")
        if "cost" in skill_lower or "finops" in skill_lower:
            domains.append("finops")

        return domains or ["general"]

    # ─────────────────────────────────────────────────────────────────────────
    # Runbook Registration
    # ─────────────────────────────────────────────────────────────────────────

    async def register_runbook_skill(
        self,
        runbook_id: str,
        name: str,
        description: str,
        category: str,
        executor: Callable | None = None,
    ) -> DynamicRegistration:
        """
        Register a runbook as an executable skill.

        Args:
            runbook_id: Unique runbook identifier
            name: Human-readable name
            description: Runbook description
            category: Category (monitoring, troubleshooting, etc.)
            executor: Optional custom executor function

        Returns:
            DynamicRegistration for the skill
        """
        skill_name = f"oci_skill_runbook_{runbook_id.replace('-', '_')}"

        # Determine domains from category
        category_domains = {
            "monitoring": ["database", "observability"],
            "troubleshooting": ["database", "infrastructure"],
            "capacity": ["database", "finops"],
            "security": ["security", "database"],
            "maintenance": ["database", "infrastructure"],
        }

        domains = category_domains.get(category, ["database"])

        registration = DynamicRegistration(
            name=skill_name,
            description=f"[Runbook] {name}: {description}",
            source="runbook",
            domains=domains,
            registered_at=datetime.utcnow(),
            handler=executor,
            schema={
                "type": "object",
                "properties": {
                    "resource_id": {
                        "type": "string",
                        "description": "Target resource OCID (e.g., database ID)",
                    },
                    "session_id": {
                        "type": "string",
                        "description": "Session ID for tracking",
                    },
                    "variables": {
                        "type": "object",
                        "description": "Override variables for runbook execution",
                    },
                },
                "required": ["resource_id"],
            },
        )

        self._registrations[skill_name] = registration

        # Register with tool catalog if executor provided
        if executor:
            self._tool_catalog.register_tool(
                name=skill_name,
                description=registration.description,
                input_schema=registration.schema,
                handler=executor,
                server_id="runbooks",
                tier=3,  # Moderate latency
                risk_level="low",
            )

        self._logger.info(
            "Runbook skill registered",
            skill=skill_name,
            runbook=runbook_id,
            domains=domains,
        )

        # Emit event
        await self._emit_event(ToolChangeEvent(
            event_type=UpdateEvent.RUNBOOKS_REGISTERED,
            timestamp=datetime.utcnow(),
            server_id="runbooks",
            tools_affected=[skill_name],
            details={"runbook_id": runbook_id, "name": name},
        ))

        return registration

    # ─────────────────────────────────────────────────────────────────────────
    # Custom Tool Registration
    # ─────────────────────────────────────────────────────────────────────────

    async def register_custom_tool(
        self,
        name: str,
        description: str,
        handler: Callable,
        input_schema: dict[str, Any],
        domains: list[str] | None = None,
        tier: int = 2,
    ) -> DynamicRegistration:
        """
        Register a custom tool at runtime.

        Args:
            name: Tool name
            description: Tool description
            handler: Async handler function
            input_schema: JSON schema for inputs
            domains: Domains this tool belongs to
            tier: Tool tier (1-4)

        Returns:
            DynamicRegistration
        """
        domains = domains or ["general"]

        registration = DynamicRegistration(
            name=name,
            description=description,
            source="custom",
            domains=domains,
            registered_at=datetime.utcnow(),
            handler=handler,
            schema=input_schema,
        )

        self._registrations[name] = registration

        # Register with tool catalog
        self._tool_catalog.register_tool(
            name=name,
            description=description,
            input_schema=input_schema,
            handler=handler,
            server_id="custom",
            tier=tier,
        )

        self._logger.info(
            "Custom tool registered",
            name=name,
            domains=domains,
        )

        return registration

    async def unregister_tool(self, name: str) -> bool:
        """Unregister a dynamic tool."""
        if name not in self._registrations:
            return False

        reg = self._registrations.pop(name)

        # Remove from tool catalog
        self._tool_catalog.unregister_tool(name)

        self._logger.info("Tool unregistered", name=name)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Event System
    # ─────────────────────────────────────────────────────────────────────────

    def on_update(
        self,
        event_type: UpdateEvent,
        callback: Callable[[ToolChangeEvent], None],
    ) -> None:
        """
        Register a callback for update events.

        Args:
            event_type: Type of event to listen for
            callback: Callback function (sync or async)
        """
        if event_type not in self._callbacks:
            self._callbacks[event_type] = []
        self._callbacks[event_type].append(callback)

    async def _emit_event(self, event: ToolChangeEvent) -> None:
        """Emit an event to registered callbacks."""
        callbacks = self._callbacks.get(event.event_type, [])

        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event)
                else:
                    callback(event)
            except Exception as e:
                self._logger.error(
                    "Event callback error",
                    event=event.event_type.value,
                    error=str(e),
                )

    # ─────────────────────────────────────────────────────────────────────────
    # Status and Statistics
    # ─────────────────────────────────────────────────────────────────────────

    def get_registrations(self) -> dict[str, DynamicRegistration]:
        """Get all dynamic registrations."""
        return dict(self._registrations)

    def get_registrations_by_source(
        self,
        source: str,
    ) -> list[DynamicRegistration]:
        """Get registrations by source type."""
        return [
            reg for reg in self._registrations.values()
            if reg.source == source
        ]

    def get_status(self) -> dict[str, Any]:
        """Get manager status and statistics."""
        by_source = {}
        for reg in self._registrations.values():
            by_source[reg.source] = by_source.get(reg.source, 0) + 1

        by_domain: dict[str, int] = {}
        for reg in self._registrations.values():
            for domain in reg.domains:
                by_domain[domain] = by_domain.get(domain, 0) + 1

        return {
            "running": self._running,
            "last_sync": self._last_sync.isoformat() if self._last_sync else None,
            "total_registrations": len(self._registrations),
            "by_source": by_source,
            "by_domain": by_domain,
            "callback_count": sum(len(c) for c in self._callbacks.values()),
            "tool_catalog_count": len(self._tool_catalog.list_tools()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────

async def initialize_dynamic_manager(
    server_registry: ServerRegistry | None = None,
    tool_catalog: ToolCatalog | None = None,
    agent_catalog: AgentCatalog | None = None,
) -> DynamicToolManager:
    """
    Initialize and start the dynamic tool manager.

    Should be called during application startup.

    Returns:
        Initialized and running DynamicToolManager
    """
    manager = DynamicToolManager.get_instance(
        server_registry=server_registry,
        tool_catalog=tool_catalog,
        agent_catalog=agent_catalog,
    )

    await manager.start()
    return manager
