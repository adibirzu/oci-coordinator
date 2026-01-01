"""
ShowOCI Cache Loader - Populates Redis cache with discovered resources.

This module runs ShowOCI discovery and loads results into the
OCI Resource Cache for fast agent access.
"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import structlog

from src.cache.oci_resource_cache import OCIResourceCache
from src.showoci.runner import ShowOCIConfig, ShowOCIResult, ShowOCIRunner

logger = structlog.get_logger(__name__)


class ShowOCICacheLoader:
    """
    Loads ShowOCI discovery results into Redis cache.

    Manages:
    - Running discovery for multiple tenancies (profiles)
    - Populating cache with discovered resources
    - Scheduling periodic cache refreshes
    - Cache statistics and health

    Example:
        loader = ShowOCICacheLoader(
            redis_url="redis://localhost:6379",
            profiles=["DEFAULT", "prod-tenancy"]
        )
        await loader.run_full_load()

        # Schedule periodic refresh
        await loader.start_scheduler(interval_hours=4)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        profiles: list[str] | None = None,
        use_instance_principal: bool = False,
    ):
        """
        Initialize cache loader.

        Args:
            redis_url: Redis connection URL
            profiles: OCI config profiles to load (default: ["DEFAULT"])
            use_instance_principal: Use instance principal authentication
        """
        self.redis_url = redis_url
        self.profiles = profiles or ["DEFAULT"]
        self.use_instance_principal = use_instance_principal
        self._cache: OCIResourceCache | None = None
        self._logger = logger.bind(component="ShowOCICacheLoader")
        self._scheduler_task: asyncio.Task | None = None
        self._last_load: dict[str, str] = {}

    async def _get_cache(self) -> OCIResourceCache:
        """Get or initialize cache instance."""
        if self._cache is None:
            self._cache = OCIResourceCache(redis_url=self.redis_url)
            await self._cache.initialize()
        return self._cache

    async def load_profile(self, profile: str) -> dict[str, Any]:
        """
        Run discovery and load resources for a single profile.

        Args:
            profile: OCI config profile name

        Returns:
            Load statistics
        """
        self._logger.info("Loading profile", profile=profile)
        start_time = datetime.now(UTC)

        config = ShowOCIConfig(
            profile=profile,
            use_instance_principal=self.use_instance_principal,
            resource_types=["compute", "network", "database"],
        )

        runner = ShowOCIRunner(config=config)
        result = await runner.run_discovery()

        if not result.success:
            self._logger.error("Discovery failed", profile=profile, error=result.error)
            return {
                "profile": profile,
                "success": False,
                "error": result.error,
            }

        # Load into cache
        cache = await self._get_cache()
        stats = await self._load_result_to_cache(cache, result)

        # Update last load timestamp
        self._last_load[profile] = datetime.now(UTC).isoformat()

        duration = (datetime.now(UTC) - start_time).total_seconds()

        self._logger.info(
            "Profile loaded",
            profile=profile,
            duration_seconds=duration,
            stats=stats,
        )

        return {
            "profile": profile,
            "success": True,
            "duration_seconds": duration,
            "resources_cached": stats,
        }

    async def _load_result_to_cache(
        self, cache: OCIResourceCache, result: ShowOCIResult
    ) -> dict[str, int]:
        """Load ShowOCI result into cache."""
        stats: dict[str, int] = {}

        # Cache compartments
        compartments = result.get_compartments()
        if compartments:
            stats["compartments"] = await cache.cache_compartments(compartments)

        # Cache instances by compartment
        instances = result.get_instances()
        instances_by_comp: dict[str, list] = {}
        for inst in instances:
            comp_id = inst.get("compartment_id", "unknown")
            if comp_id not in instances_by_comp:
                instances_by_comp[comp_id] = []
            instances_by_comp[comp_id].append(inst)

        for comp_id, comp_instances in instances_by_comp.items():
            await cache.cache_instances(comp_id, comp_instances)
        stats["instances"] = len(instances)

        # Cache VCNs by compartment
        vcns = result.get_vcns()
        vcns_by_comp: dict[str, list] = {}
        for vcn in vcns:
            comp_id = vcn.get("compartment_id", "unknown")
            if comp_id not in vcns_by_comp:
                vcns_by_comp[comp_id] = []
            vcns_by_comp[comp_id].append(vcn)

        for comp_id, comp_vcns in vcns_by_comp.items():
            await cache.cache_vcns(comp_id, comp_vcns)
        stats["vcns"] = len(vcns)

        # Cache databases by compartment
        databases = result.get_autonomous_databases()
        dbs_by_comp: dict[str, list] = {}
        for db in databases:
            comp_id = db.get("compartment_id", "unknown")
            if comp_id not in dbs_by_comp:
                dbs_by_comp[comp_id] = []
            dbs_by_comp[comp_id].append(db)

        for comp_id, comp_dbs in dbs_by_comp.items():
            await cache.cache_databases(comp_id, comp_dbs)
        stats["databases"] = len(databases)

        # Cache subnets
        subnets = result.get_subnets()
        for subnet in subnets:
            comp_id = subnet.get("compartment_id", "unknown")
            await cache.cache_resources("subnet", comp_id, [subnet])
        stats["subnets"] = len(subnets)

        # Cache block volumes
        volumes = result.get_block_volumes()
        for vol in volumes:
            comp_id = vol.get("compartment_id", "unknown")
            await cache.cache_resources("block_volume", comp_id, [vol])
        stats["block_volumes"] = len(volumes)

        return stats

    async def run_full_load(self) -> dict[str, Any]:
        """
        Run full cache load for all configured profiles.

        Returns:
            Combined statistics for all profiles
        """
        self._logger.info("Starting full cache load", profiles=self.profiles)
        start_time = datetime.now(UTC)

        results = []
        for profile in self.profiles:
            result = await self.load_profile(profile)
            results.append(result)

        duration = (datetime.now(UTC) - start_time).total_seconds()

        # Aggregate stats
        total_resources = {
            "compartments": 0,
            "instances": 0,
            "vcns": 0,
            "databases": 0,
            "subnets": 0,
            "block_volumes": 0,
        }

        successful = 0
        for r in results:
            if r.get("success"):
                successful += 1
                for key, value in r.get("resources_cached", {}).items():
                    total_resources[key] = total_resources.get(key, 0) + value

        summary = {
            "profiles_loaded": successful,
            "profiles_failed": len(self.profiles) - successful,
            "total_duration_seconds": duration,
            "total_resources": total_resources,
            "profile_results": results,
            "timestamp": datetime.now(UTC).isoformat(),
        }

        self._logger.info("Full cache load complete", summary=summary)
        return summary

    async def start_scheduler(self, interval_hours: float = 4.0) -> None:
        """
        Start periodic cache refresh scheduler.

        Args:
            interval_hours: Refresh interval in hours
        """
        if self._scheduler_task and not self._scheduler_task.done():
            self._logger.warning("Scheduler already running")
            return

        async def _scheduler_loop():
            while True:
                try:
                    await asyncio.sleep(interval_hours * 3600)
                    self._logger.info("Running scheduled cache refresh")
                    await self.run_full_load()
                except asyncio.CancelledError:
                    self._logger.info("Scheduler cancelled")
                    break
                except Exception as e:
                    self._logger.error("Scheduled refresh failed", error=str(e))
                    # Continue running despite errors
                    await asyncio.sleep(300)  # Wait 5 min before retry

        self._scheduler_task = asyncio.create_task(_scheduler_loop())
        self._logger.info("Scheduler started", interval_hours=interval_hours)

    async def stop_scheduler(self) -> None:
        """Stop the cache refresh scheduler."""
        if self._scheduler_task:
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
            self._scheduler_task = None
            self._logger.info("Scheduler stopped")

    async def get_cache_status(self) -> dict[str, Any]:
        """Get current cache status and statistics."""
        cache = await self._get_cache()
        stats = await cache.get_discovery_stats()

        return {
            "profiles": self.profiles,
            "last_load": self._last_load,
            "scheduler_running": self._scheduler_task is not None and not self._scheduler_task.done(),
            "cache_stats": stats,
        }

    async def clear_cache(self) -> int:
        """Clear all cached data."""
        cache = await self._get_cache()
        count = await cache.clear_cache()
        self._last_load.clear()
        self._logger.info("Cache cleared", keys_deleted=count)
        return count
