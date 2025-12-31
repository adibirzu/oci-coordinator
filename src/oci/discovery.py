"""
OCI Resource Discovery Service.

Handles periodic discovery of OCI resources across tenancies with
scheduling support for daily updates and on-demand refresh.

Usage:
    from src.oci.discovery import DiscoveryService

    service = DiscoveryService()
    await service.start()  # Starts background discovery job

    # Or run on-demand
    await service.run_discovery()
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

import structlog

from src.observability import get_tracer
from src.oci.tenancy_manager import TenancyManager

logger = structlog.get_logger(__name__)


class DiscoveryService:
    """
    Manages scheduled OCI resource discovery.

    Features:
    - Automatic discovery on startup
    - Scheduled daily refresh (configurable)
    - On-demand refresh capability
    - Multi-tenancy support
    """

    _instance: "DiscoveryService | None" = None

    def __init__(
        self,
        refresh_interval_hours: int = 24,
        redis_url: str | None = None,
    ):
        self.refresh_interval = timedelta(hours=refresh_interval_hours)
        self.redis_url = redis_url
        self._running = False
        self._task: asyncio.Task | None = None
        self._last_discovery: datetime | None = None
        self._logger = logger.bind(component="DiscoveryService")
        self._tracer = get_tracer("discovery")

    @classmethod
    def get_instance(cls) -> "DiscoveryService":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    async def start(self) -> None:
        """Start the discovery service with background scheduling."""
        if self._running:
            self._logger.warning("Discovery service already running")
            return

        self._running = True
        self._logger.info("Starting discovery service")

        # Run initial discovery
        await self.run_discovery()

        # Start background scheduler
        self._task = asyncio.create_task(self._scheduler_loop())
        self._logger.info(
            "Discovery scheduler started",
            interval_hours=self.refresh_interval.total_seconds() / 3600,
        )

    async def stop(self) -> None:
        """Stop the discovery service."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._logger.info("Discovery service stopped")

    async def _scheduler_loop(self) -> None:
        """Background loop for scheduled discovery."""
        while self._running:
            try:
                # Sleep until next refresh
                await asyncio.sleep(self.refresh_interval.total_seconds())

                if self._running:
                    self._logger.info("Running scheduled discovery")
                    await self.run_discovery()

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Scheduler error", error=str(e))
                # Wait a bit before retrying
                await asyncio.sleep(60)

    async def run_discovery(self) -> dict[str, Any]:
        """
        Run full OCI resource discovery.

        Returns:
            Discovery results summary
        """
        with self._tracer.start_as_current_span("discovery.run") as span:
            self._logger.info("Starting OCI resource discovery")
            start_time = datetime.now()

            results = {
                "started_at": start_time.isoformat(),
                "tenancies": 0,
                "compartments": 0,
                "errors": [],
            }

            try:
                # Get TenancyManager and initialize
                manager = TenancyManager.get_instance()

                # Force re-discovery by clearing state
                manager._initialized = False
                manager._compartments.clear()
                manager._compartments_by_name.clear()

                # Run discovery
                await manager.initialize()

                # Collect results
                tenancies = manager.list_tenancies()
                compartments = await manager.list_compartments()

                results["tenancies"] = len(tenancies)
                results["compartments"] = len(compartments)

                # Log tenancy details
                for tenancy in tenancies:
                    tenancy_compartments = [
                        c for c in compartments if c.tenancy_profile == tenancy.profile_name
                    ]
                    self._logger.info(
                        "Tenancy discovered",
                        profile=tenancy.profile_name,
                        region=tenancy.region,
                        compartments=len(tenancy_compartments),
                    )

                span.set_attribute("discovery.tenancies", results["tenancies"])
                span.set_attribute("discovery.compartments", results["compartments"])

            except Exception as e:
                self._logger.error("Discovery failed", error=str(e))
                results["errors"].append(str(e))
                span.set_attribute("error", True)
                span.set_attribute("error.message", str(e))

            # Update timing
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            results["completed_at"] = end_time.isoformat()
            results["duration_seconds"] = duration
            self._last_discovery = end_time

            span.set_attribute("discovery.duration_seconds", duration)

            self._logger.info(
                "Discovery completed",
                tenancies=results["tenancies"],
                compartments=results["compartments"],
                duration_seconds=duration,
            )

            return results

    async def get_status(self) -> dict[str, Any]:
        """Get discovery service status."""
        manager = TenancyManager.get_instance()

        compartments = []
        if manager._initialized:
            compartments = await manager.list_compartments()

        return {
            "running": self._running,
            "last_discovery": self._last_discovery.isoformat() if self._last_discovery else None,
            "next_discovery": (
                (self._last_discovery + self.refresh_interval).isoformat()
                if self._last_discovery
                else None
            ),
            "refresh_interval_hours": self.refresh_interval.total_seconds() / 3600,
            "cached_compartments": len(compartments),
            "tenancies": len(manager.list_tenancies()),
        }

    async def refresh_if_stale(self, max_age_hours: int = 1) -> bool:
        """
        Refresh discovery if data is stale.

        Args:
            max_age_hours: Maximum age of cached data before refresh

        Returns:
            True if refresh was performed
        """
        if self._last_discovery is None:
            await self.run_discovery()
            return True

        age = datetime.now() - self._last_discovery
        if age > timedelta(hours=max_age_hours):
            self._logger.info("Cache stale, refreshing", age_hours=age.total_seconds() / 3600)
            await self.run_discovery()
            return True

        return False


async def initialize_discovery() -> DiscoveryService:
    """Initialize and start the discovery service."""
    service = DiscoveryService.get_instance()
    await service.start()
    return service
