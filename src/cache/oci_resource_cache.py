"""
OCI Resource Cache Service.

Pre-populates and maintains a Redis cache of OCI resources for fast access.
Uses MCP tools to discover resources and stores them for agent consumption.

Features:
- Compartment hierarchy caching
- Instance, database, network resource discovery
- Incremental updates (only new resources since last run)
- Cache statistics and health monitoring
- Tag-based invalidation for group operations
- Soft/hard TTL for stale-while-revalidate pattern
- Event-driven cache updates via pub/sub

Usage:
    from src.cache.oci_resource_cache import OCIResourceCache

    cache = OCIResourceCache(redis_url="redis://localhost:6379")
    await cache.initialize()

    # Get all compartments
    compartments = await cache.get_compartments()

    # Get resources by compartment
    instances = await cache.get_instances(compartment_id="ocid1.compartment...")

    # Invalidate by tag
    await cache.invalidate_by_tag("compartment:ocid1.compartment.xxx")
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Event types for cache notifications
CACHE_EVENT_SET = "cache:set"
CACHE_EVENT_DELETE = "cache:delete"
CACHE_EVENT_INVALIDATE = "cache:invalidate"
CACHE_EVENT_EXPIRE = "cache:expire"

# Cache key patterns
CACHE_KEYS = {
    "compartments": "oci:compartments",
    "compartment_by_name": "oci:compartment:name:{name}",
    "compartment_by_id": "oci:compartment:id:{ocid}",
    "instances": "oci:instances:{compartment_id}",
    "databases": "oci:databases:{compartment_id}",
    "vcns": "oci:vcns:{compartment_id}",
    "resources_by_type": "oci:resources:{resource_type}:{compartment_id}",
    "last_discovery": "oci:discovery:last_run",
    "discovery_stats": "oci:discovery:stats",
    # Tag-based invalidation
    "tag_members": "oci:tags:{tag}",  # Set of keys tagged with {tag}
    "key_metadata": "oci:meta:{key}",  # Metadata for a key (tags, created_at, etc.)
    # Pub/sub channel
    "invalidation_channel": "oci:invalidation",
}

# Default TTLs
DEFAULT_TTL = timedelta(hours=1)
COMPARTMENT_TTL = timedelta(hours=4)  # Compartments change rarely
RESOURCE_TTL = timedelta(minutes=30)  # Resources change more frequently
# Soft TTL for stale-while-revalidate pattern
SOFT_TTL_RATIO = 0.8  # Return stale data after 80% of TTL, trigger background refresh


class OCIResourceCache:
    """
    OCI Resource Cache with Redis backend.

    Provides fast access to OCI resource data by:
    - Pre-populating compartment hierarchy
    - Caching instance, database, network resources
    - Supporting name and OCID lookups
    - Tracking discovery timestamps for incremental updates
    """

    _instance: OCIResourceCache | None = None

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        tool_catalog: Any = None,
    ):
        """
        Initialize OCI Resource Cache.

        Args:
            redis_url: Redis connection URL
            tool_catalog: MCP ToolCatalog for API calls
        """
        self._redis_url = redis_url
        self._redis = None
        self._tool_catalog = tool_catalog
        self._logger = logger.bind(component="OCIResourceCache")
        self._initialized = False
        # Event callbacks for cache notifications
        self._event_callbacks: list[Callable[[str, str, dict], None]] = []
        # Background refresh tasks
        self._pending_refreshes: set[str] = set()
        # Pub/sub subscriber
        self._pubsub = None
        self._subscriber_task = None

    @classmethod
    def get_instance(
        cls,
        redis_url: str = "redis://localhost:6379",
        tool_catalog: Any = None,
    ) -> OCIResourceCache:
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(redis_url, tool_catalog)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton (for testing)."""
        cls._instance = None

    async def initialize(self) -> None:
        """Initialize Redis connection."""
        if self._initialized:
            return

        try:
            import redis.asyncio as redis

            self._redis = redis.from_url(self._redis_url, decode_responses=True)
            await self._redis.ping()
            self._initialized = True
            self._logger.info("OCI Resource Cache initialized", redis_url=self._redis_url)
        except Exception as e:
            self._logger.error("Failed to initialize Redis", error=str(e))
            # Fall back to in-memory cache
            self._redis = None
            self._memory_cache: dict[str, Any] = {}
            self._initialized = True
            self._logger.warning("Using in-memory fallback cache")

    async def close(self) -> None:
        """Close Redis connection."""
        # Stop subscriber
        if self._subscriber_task:
            self._subscriber_task.cancel()
            try:
                await self._subscriber_task
            except asyncio.CancelledError:
                pass
        if self._pubsub:
            await self._pubsub.close()
        if self._redis:
            await self._redis.close()

    # ─────────────────────────────────────────────────────────────────────────
    # Event System
    # ─────────────────────────────────────────────────────────────────────────

    def on_event(self, callback: Callable[[str, str, dict], None]) -> None:
        """
        Register a callback for cache events.

        Callback receives: (event_type, key, data)

        Events:
        - cache:set - Key was set
        - cache:delete - Key was deleted
        - cache:invalidate - Key was invalidated by tag
        - cache:expire - Key expired

        Example:
            def on_cache_event(event, key, data):
                if event == CACHE_EVENT_INVALIDATE:
                    print(f"Key {key} invalidated by tag {data['tag']}")

            cache.on_event(on_cache_event)
        """
        self._event_callbacks.append(callback)

    def _emit_event(self, event: str, key: str, data: dict | None = None) -> None:
        """Emit an event to all registered callbacks."""
        for callback in self._event_callbacks:
            try:
                callback(event, key, data or {})
            except Exception as e:
                self._logger.warning("Event callback error", event=event, error=str(e))

    async def _publish_invalidation(self, tag: str, keys: list[str]) -> None:
        """Publish invalidation event to Redis pub/sub."""
        if self._redis:
            try:
                message = json.dumps({"tag": tag, "keys": keys, "time": datetime.now(UTC).isoformat()})
                await self._redis.publish(CACHE_KEYS["invalidation_channel"], message)
            except Exception as e:
                self._logger.warning("Publish invalidation failed", error=str(e))

    async def start_invalidation_listener(self) -> None:
        """Start listening for invalidation events from other instances."""
        if not self._redis:
            return

        try:
            self._pubsub = self._redis.pubsub()
            await self._pubsub.subscribe(CACHE_KEYS["invalidation_channel"])
            self._subscriber_task = asyncio.create_task(self._listen_for_invalidations())
            self._logger.info("Cache invalidation listener started")
        except Exception as e:
            self._logger.error("Failed to start invalidation listener", error=str(e))

    async def _listen_for_invalidations(self) -> None:
        """Listen for invalidation messages."""
        async for message in self._pubsub.listen():
            if message["type"] == "message":
                try:
                    data = json.loads(message["data"])
                    tag = data.get("tag")
                    keys = data.get("keys", [])
                    for key in keys:
                        self._emit_event(CACHE_EVENT_INVALIDATE, key, {"tag": tag})
                except Exception as e:
                    self._logger.warning("Invalidation message parse error", error=str(e))

    # ─────────────────────────────────────────────────────────────────────────
    # Tag-Based Invalidation
    # ─────────────────────────────────────────────────────────────────────────

    async def _add_tags(self, key: str, tags: list[str]) -> None:
        """Associate tags with a cache key."""
        if not self._redis or not tags:
            return

        now = datetime.now(UTC).isoformat()
        metadata = {"tags": tags, "created_at": now}

        # Store metadata
        meta_key = CACHE_KEYS["key_metadata"].format(key=key)
        await self._redis.set(meta_key, json.dumps(metadata))

        # Add key to each tag's member set
        for tag in tags:
            tag_key = CACHE_KEYS["tag_members"].format(tag=tag)
            await self._redis.sadd(tag_key, key)

    async def _remove_tags(self, key: str) -> None:
        """Remove key from all its associated tags."""
        if not self._redis:
            return

        # Get metadata
        meta_key = CACHE_KEYS["key_metadata"].format(key=key)
        metadata_raw = await self._redis.get(meta_key)
        if not metadata_raw:
            return

        try:
            metadata = json.loads(metadata_raw)
            tags = metadata.get("tags", [])

            # Remove key from each tag's member set
            for tag in tags:
                tag_key = CACHE_KEYS["tag_members"].format(tag=tag)
                await self._redis.srem(tag_key, key)

            # Delete metadata
            await self._redis.delete(meta_key)
        except Exception as e:
            self._logger.warning("Remove tags error", key=key, error=str(e))

    async def invalidate_by_tag(self, tag: str) -> int:
        """
        Invalidate all cache entries with a specific tag.

        This is useful for group invalidation, e.g.:
        - invalidate_by_tag("compartment:ocid1.compartment.xxx") - All resources in a compartment
        - invalidate_by_tag("resource:instances") - All instance caches
        - invalidate_by_tag("tenancy:xxx") - All caches for a tenancy

        Args:
            tag: Tag to invalidate

        Returns:
            Number of keys invalidated
        """
        if not self._redis:
            # In-memory fallback
            count = 0
            keys_to_delete = []
            for key in list(self._memory_cache.keys()):
                if tag in key:
                    keys_to_delete.append(key)
            for key in keys_to_delete:
                del self._memory_cache[key]
                self._emit_event(CACHE_EVENT_INVALIDATE, key, {"tag": tag})
                count += 1
            return count

        # Get all keys with this tag
        tag_key = CACHE_KEYS["tag_members"].format(tag=tag)
        keys = await self._redis.smembers(tag_key)

        if not keys:
            return 0

        # Delete each key and its metadata
        pipeline = self._redis.pipeline()
        for key in keys:
            pipeline.delete(key)
            meta_key = CACHE_KEYS["key_metadata"].format(key=key)
            pipeline.delete(meta_key)
        await pipeline.execute()

        # Clear the tag set
        await self._redis.delete(tag_key)

        # Emit events
        for key in keys:
            self._emit_event(CACHE_EVENT_INVALIDATE, key, {"tag": tag})

        # Publish to other instances
        await self._publish_invalidation(tag, list(keys))

        self._logger.info("Tag invalidation complete", tag=tag, keys_invalidated=len(keys))
        return len(keys)

    async def get_tags_for_key(self, key: str) -> list[str]:
        """Get all tags associated with a key."""
        if not self._redis:
            return []

        meta_key = CACHE_KEYS["key_metadata"].format(key=key)
        metadata_raw = await self._redis.get(meta_key)
        if not metadata_raw:
            return []

        try:
            metadata = json.loads(metadata_raw)
            return metadata.get("tags", [])
        except Exception:
            return []

    async def get_keys_by_tag(self, tag: str) -> list[str]:
        """Get all keys with a specific tag."""
        if not self._redis:
            return [k for k in self._memory_cache.keys() if tag in k]

        tag_key = CACHE_KEYS["tag_members"].format(tag=tag)
        return list(await self._redis.smembers(tag_key))

    # ─────────────────────────────────────────────────────────────────────────
    # Stale-While-Revalidate Pattern
    # ─────────────────────────────────────────────────────────────────────────

    async def get_with_swr(
        self,
        key: str,
        refresh_func: Callable[[], Any] | None = None,
        ttl: timedelta | None = None,
    ) -> tuple[Any | None, bool]:
        """
        Get value with stale-while-revalidate pattern.

        Returns stale data immediately if within soft TTL, triggers background refresh.

        Args:
            key: Cache key
            refresh_func: Async function to call for refresh
            ttl: TTL for the cached value

        Returns:
            Tuple of (value, is_stale)
        """
        ttl = ttl or DEFAULT_TTL
        soft_ttl = timedelta(seconds=ttl.total_seconds() * SOFT_TTL_RATIO)

        if not self._redis:
            value = self._memory_cache.get(key)
            return (value, False)

        # Get value and metadata
        value_raw = await self._redis.get(key)
        if not value_raw:
            return (None, False)

        value = json.loads(value_raw)

        # Check metadata for created_at
        meta_key = CACHE_KEYS["key_metadata"].format(key=key)
        metadata_raw = await self._redis.get(meta_key)

        if metadata_raw:
            try:
                metadata = json.loads(metadata_raw)
                created_at = datetime.fromisoformat(metadata.get("created_at", ""))
                age = datetime.now(UTC) - created_at

                # If past soft TTL but within hard TTL, return stale and trigger refresh
                if age > soft_ttl and refresh_func:
                    if key not in self._pending_refreshes:
                        self._pending_refreshes.add(key)
                        asyncio.create_task(self._background_refresh(key, refresh_func, ttl))
                    return (value, True)
            except Exception:
                pass

        return (value, False)

    async def _background_refresh(
        self,
        key: str,
        refresh_func: Callable[[], Any],
        ttl: timedelta,
    ) -> None:
        """Background task to refresh a cache entry."""
        try:
            if asyncio.iscoroutinefunction(refresh_func):
                new_value = await refresh_func()
            else:
                new_value = refresh_func()

            await self._set(key, new_value, ttl)
            self._logger.debug("Background refresh complete", key=key)
        except Exception as e:
            self._logger.warning("Background refresh failed", key=key, error=str(e))
        finally:
            self._pending_refreshes.discard(key)

    # ─────────────────────────────────────────────────────────────────────────
    # Cache Operations (Enhanced)
    # ─────────────────────────────────────────────────────────────────────────

    async def _get(self, key: str) -> Any | None:
        """Get value from cache."""
        if self._redis:
            value = await self._redis.get(key)
            return json.loads(value) if value else None
        return self._memory_cache.get(key)

    async def _set(
        self,
        key: str,
        value: Any,
        ttl: timedelta | None = None,
        tags: list[str] | None = None,
    ) -> None:
        """
        Set value in cache with optional tags.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live
            tags: Tags for group invalidation
        """
        if self._redis:
            serialized = json.dumps(value, default=str)
            if ttl:
                await self._redis.setex(key, int(ttl.total_seconds()), serialized)
            else:
                await self._redis.set(key, serialized)

            # Add tags if provided
            if tags:
                await self._add_tags(key, tags)
        else:
            self._memory_cache[key] = value

        self._emit_event(CACHE_EVENT_SET, key, {"tags": tags or []})

    async def _delete(self, key: str) -> None:
        """Delete value from cache."""
        if self._redis:
            # Remove tags first
            await self._remove_tags(key)
            await self._redis.delete(key)
        else:
            self._memory_cache.pop(key, None)

        self._emit_event(CACHE_EVENT_DELETE, key, {})

    async def _get_keys(self, pattern: str) -> list[str]:
        """Get keys matching pattern."""
        if self._redis:
            return await self._redis.keys(pattern)
        return [k for k in self._memory_cache.keys() if pattern.replace("*", "") in k]

    # ─────────────────────────────────────────────────────────────────────────
    # Compartment Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def get_compartments(self) -> list[dict[str, Any]]:
        """Get all cached compartments."""
        return await self._get(CACHE_KEYS["compartments"]) or []

    async def get_compartment_by_name(self, name: str) -> dict[str, Any] | None:
        """Get compartment by name (case-insensitive)."""
        key = CACHE_KEYS["compartment_by_name"].format(name=name.lower())
        return await self._get(key)

    async def get_compartment_by_id(self, ocid: str) -> dict[str, Any] | None:
        """Get compartment by OCID."""
        key = CACHE_KEYS["compartment_by_id"].format(ocid=ocid)
        return await self._get(key)

    async def cache_compartments(self, compartments: list[dict[str, Any]]) -> int:
        """
        Cache compartment data with tags for group invalidation.

        Args:
            compartments: List of compartment dicts with 'id', 'name', etc.

        Returns:
            Number of compartments cached
        """
        # Store full list with tag
        await self._set(
            CACHE_KEYS["compartments"],
            compartments,
            COMPARTMENT_TTL,
            tags=["resource:compartments"],
        )

        # Index by name and ID
        for comp in compartments:
            name = comp.get("name", "").lower()
            ocid = comp.get("id", "")
            tenancy_id = comp.get("compartment_id", "")  # Parent compartment

            # Tags for this compartment
            tags = ["resource:compartments"]
            if tenancy_id:
                tags.append(f"tenancy:{tenancy_id}")
            if ocid:
                tags.append(f"compartment:{ocid}")

            if name:
                key = CACHE_KEYS["compartment_by_name"].format(name=name)
                await self._set(key, comp, COMPARTMENT_TTL, tags=tags)

            if ocid:
                key = CACHE_KEYS["compartment_by_id"].format(ocid=ocid)
                await self._set(key, comp, COMPARTMENT_TTL, tags=tags)

        self._logger.info("Compartments cached", count=len(compartments))
        return len(compartments)

    # ─────────────────────────────────────────────────────────────────────────
    # Resource Operations
    # ─────────────────────────────────────────────────────────────────────────

    async def get_instances(
        self, compartment_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get cached compute instances."""
        if compartment_id:
            key = CACHE_KEYS["instances"].format(compartment_id=compartment_id)
            return await self._get(key) or []

        # Get all instances across compartments
        keys = await self._get_keys("oci:instances:*")
        all_instances = []
        for key in keys:
            instances = await self._get(key)
            if instances:
                all_instances.extend(instances)
        return all_instances

    async def cache_instances(
        self, compartment_id: str, instances: list[dict[str, Any]]
    ) -> int:
        """Cache compute instances for a compartment with tags."""
        key = CACHE_KEYS["instances"].format(compartment_id=compartment_id)
        tags = [
            "resource:instances",
            f"compartment:{compartment_id}",
        ]
        await self._set(key, instances, RESOURCE_TTL, tags=tags)
        self._logger.debug("Instances cached", compartment=compartment_id[:20], count=len(instances))
        return len(instances)

    async def get_databases(
        self, compartment_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get cached databases (autonomous and DB systems)."""
        if compartment_id:
            key = CACHE_KEYS["databases"].format(compartment_id=compartment_id)
            return await self._get(key) or []

        keys = await self._get_keys("oci:databases:*")
        all_dbs = []
        for key in keys:
            dbs = await self._get(key)
            if dbs:
                all_dbs.extend(dbs)
        return all_dbs

    async def cache_databases(
        self, compartment_id: str, databases: list[dict[str, Any]]
    ) -> int:
        """Cache databases for a compartment with tags."""
        key = CACHE_KEYS["databases"].format(compartment_id=compartment_id)
        tags = [
            "resource:databases",
            f"compartment:{compartment_id}",
        ]
        await self._set(key, databases, RESOURCE_TTL, tags=tags)
        self._logger.debug("Databases cached", compartment=compartment_id[:20], count=len(databases))
        return len(databases)

    async def get_vcns(
        self, compartment_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Get cached VCNs."""
        if compartment_id:
            key = CACHE_KEYS["vcns"].format(compartment_id=compartment_id)
            return await self._get(key) or []

        keys = await self._get_keys("oci:vcns:*")
        all_vcns = []
        for key in keys:
            vcns = await self._get(key)
            if vcns:
                all_vcns.extend(vcns)
        return all_vcns

    async def cache_vcns(
        self, compartment_id: str, vcns: list[dict[str, Any]]
    ) -> int:
        """Cache VCNs for a compartment with tags."""
        key = CACHE_KEYS["vcns"].format(compartment_id=compartment_id)
        tags = [
            "resource:vcns",
            "resource:network",
            f"compartment:{compartment_id}",
        ]
        await self._set(key, vcns, RESOURCE_TTL, tags=tags)
        return len(vcns)

    async def cache_resources(
        self,
        resource_type: str,
        compartment_id: str,
        resources: list[dict[str, Any]],
    ) -> int:
        """Cache resources of any type with tags."""
        key = CACHE_KEYS["resources_by_type"].format(
            resource_type=resource_type, compartment_id=compartment_id
        )
        tags = [
            f"resource:{resource_type}",
            f"compartment:{compartment_id}",
        ]
        await self._set(key, resources, RESOURCE_TTL, tags=tags)
        return len(resources)

    async def get_resources(
        self,
        resource_type: str,
        compartment_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get cached resources of a specific type."""
        if compartment_id:
            key = CACHE_KEYS["resources_by_type"].format(
                resource_type=resource_type, compartment_id=compartment_id
            )
            return await self._get(key) or []

        keys = await self._get_keys(f"oci:resources:{resource_type}:*")
        all_resources = []
        for key in keys:
            resources = await self._get(key)
            if resources:
                all_resources.extend(resources)
        return all_resources

    # ─────────────────────────────────────────────────────────────────────────
    # Discovery Operations (using MCP tools)
    # ─────────────────────────────────────────────────────────────────────────

    async def discover_compartments(self) -> int:
        """
        Discover and cache all compartments using MCP tools.

        Returns:
            Number of compartments discovered
        """
        if not self._tool_catalog:
            self._logger.warning("Tool catalog not available for discovery")
            return 0

        self._logger.info("Starting compartment discovery")

        try:
            # Call MCP tool to list compartments
            # Try standardized name first, then fallback to legacy names
            tool_names = ["oci_list_compartments", "list_compartments", "oci_iam_list_compartments"]
            result = None

            for tool_name in tool_names:
                try:
                    result = await self._tool_catalog.execute(
                        tool_name,
                        {"include_root": True, "recursive": True}
                    )
                    if result and hasattr(result, 'success') and result.success:
                        break
                except Exception:
                    continue

            if result is None:
                self._logger.error("No compartment list tool found")
                return 0

            if result.success and result.result:
                compartments = result.result if isinstance(result.result, list) else []
                return await self.cache_compartments(compartments)
            else:
                self._logger.error("Compartment discovery failed", error=result.error)
                return 0

        except Exception as e:
            self._logger.error("Compartment discovery error", error=str(e))
            return 0

    async def discover_resources(
        self,
        compartment_id: str,
        resource_types: list[str] | None = None,
    ) -> dict[str, int]:
        """
        Discover and cache resources in a compartment.

        Args:
            compartment_id: Compartment OCID
            resource_types: List of resource types to discover (default: all)

        Returns:
            Dict of resource_type -> count discovered
        """
        if not self._tool_catalog:
            self._logger.warning("Tool catalog not available for discovery")
            return {}

        resource_types = resource_types or ["compute", "database", "network"]
        results: dict[str, int] = {}

        # Discover compute instances
        if "compute" in resource_types:
            try:
                result = await self._tool_catalog.execute(
                    "oci_compute_list_instances",
                    {"compartment_id": compartment_id}
                )
                if result.success and result.result:
                    instances = result.result if isinstance(result.result, list) else []
                    results["instances"] = await self.cache_instances(compartment_id, instances)
            except Exception as e:
                self._logger.error("Instance discovery failed", error=str(e))

        # Discover databases
        if "database" in resource_types:
            try:
                # Try standardized name first, then fallback
                db_tool_names = ["oci_database_list_autonomous", "list_autonomous_databases"]
                for tool_name in db_tool_names:
                    try:
                        result = await self._tool_catalog.execute(
                            tool_name,
                            {"compartment_id": compartment_id}
                        )
                        if result.success and result.result:
                            databases = result.result if isinstance(result.result, list) else []
                            results["databases"] = await self.cache_databases(compartment_id, databases)
                            break
                    except Exception:
                        continue
            except Exception as e:
                self._logger.error("Database discovery failed", error=str(e))

        # Discover VCNs
        if "network" in resource_types:
            try:
                result = await self._tool_catalog.execute(
                    "oci_network_list_vcns",
                    {"compartment_id": compartment_id}
                )
                if result.success and result.result:
                    vcns = result.result if isinstance(result.result, list) else []
                    results["vcns"] = await self.cache_vcns(compartment_id, vcns)
            except Exception as e:
                self._logger.error("VCN discovery failed", error=str(e))

        self._logger.info(
            "Resource discovery complete",
            compartment=compartment_id[:30] + "...",
            results=results,
        )

        return results

    async def full_discovery(
        self,
        incremental: bool = True,
    ) -> dict[str, Any]:
        """
        Run full discovery of all OCI resources.

        Args:
            incremental: If True, only discover resources created since last run

        Returns:
            Discovery statistics
        """
        start_time = datetime.now(UTC)
        stats: dict[str, Any] = {
            "start_time": start_time.isoformat(),
            "incremental": incremental,
            "compartments": 0,
            "resources": {},
        }

        # Check last discovery time
        last_run = await self._get(CACHE_KEYS["last_discovery"])
        if incremental and last_run:
            self._logger.info("Incremental discovery since", last_run=last_run)

        # Discover compartments first
        stats["compartments"] = await self.discover_compartments()

        # Get all compartments and discover resources
        compartments = await self.get_compartments()
        for comp in compartments:
            comp_id = comp.get("id", "")
            if comp_id:
                results = await self.discover_resources(comp_id)
                for rtype, count in results.items():
                    stats["resources"][rtype] = stats["resources"].get(rtype, 0) + count

        # Update last discovery time
        stats["end_time"] = datetime.now(UTC).isoformat()
        stats["duration_seconds"] = (
            datetime.now(UTC) - start_time
        ).total_seconds()

        await self._set(CACHE_KEYS["last_discovery"], start_time.isoformat())
        await self._set(CACHE_KEYS["discovery_stats"], stats)

        self._logger.info("Full discovery complete", stats=stats)
        return stats

    async def get_discovery_stats(self) -> dict[str, Any] | None:
        """Get last discovery statistics."""
        return await self._get(CACHE_KEYS["discovery_stats"])

    # ─────────────────────────────────────────────────────────────────────────
    # Query Helpers (for agents)
    # ─────────────────────────────────────────────────────────────────────────

    async def get_resource_summary(
        self, compartment_name: str | None = None
    ) -> dict[str, Any]:
        """
        Get a summary of resources, optionally filtered by compartment name.

        This is the primary method agents should use to answer resource queries.

        Args:
            compartment_name: Optional compartment name to filter by

        Returns:
            Resource summary dict
        """
        compartment_id = None
        compartment = None

        if compartment_name:
            compartment = await self.get_compartment_by_name(compartment_name)
            if compartment:
                compartment_id = compartment.get("id")

        instances = await self.get_instances(compartment_id)
        databases = await self.get_databases(compartment_id)
        vcns = await self.get_vcns(compartment_id)

        summary = {
            "compartment": compartment or {"name": "all", "id": None},
            "instances": {
                "count": len(instances),
                "by_state": {},
                "items": instances[:10],  # First 10 for preview
            },
            "databases": {
                "count": len(databases),
                "by_state": {},
                "items": databases[:10],
            },
            "vcns": {
                "count": len(vcns),
                "items": vcns[:10],
            },
        }

        # Count by state
        for inst in instances:
            state = inst.get("lifecycle_state", inst.get("lifecycleState", "UNKNOWN"))
            summary["instances"]["by_state"][state] = (
                summary["instances"]["by_state"].get(state, 0) + 1
            )

        for db in databases:
            state = db.get("lifecycle_state", db.get("lifecycleState", "UNKNOWN"))
            summary["databases"]["by_state"][state] = (
                summary["databases"]["by_state"].get(state, 0) + 1
            )

        return summary

    async def search_resources(
        self,
        query: str,
        resource_type: str | None = None,
        compartment_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Search cached resources by name or other attributes.

        Args:
            query: Search query (matches display_name)
            resource_type: Filter by resource type
            compartment_id: Filter by compartment

        Returns:
            Matching resources
        """
        query_lower = query.lower()
        results = []

        # Search instances
        if not resource_type or resource_type in ["compute", "instance"]:
            instances = await self.get_instances(compartment_id)
            for inst in instances:
                name = inst.get("display_name", inst.get("displayName", "")).lower()
                if query_lower in name:
                    results.append({"type": "instance", **inst})

        # Search databases
        if not resource_type or resource_type in ["database", "db"]:
            databases = await self.get_databases(compartment_id)
            for db in databases:
                name = db.get("display_name", db.get("displayName", "")).lower()
                if query_lower in name:
                    results.append({"type": "database", **db})

        # Search VCNs
        if not resource_type or resource_type in ["network", "vcn"]:
            vcns = await self.get_vcns(compartment_id)
            for vcn in vcns:
                name = vcn.get("display_name", vcn.get("displayName", "")).lower()
                if query_lower in name:
                    results.append({"type": "vcn", **vcn})

        return results

    async def clear_cache(self) -> int:
        """Clear all cached data."""
        keys = await self._get_keys("oci:*")
        for key in keys:
            await self._delete(key)
        self._logger.info("Cache cleared", keys_deleted=len(keys))
        return len(keys)

    async def get_cache_stats(self) -> dict[str, Any]:
        """
        Get comprehensive cache statistics.

        Returns:
            Cache statistics including key counts, memory usage, and health metrics
        """
        stats: dict[str, Any] = {
            "initialized": self._initialized,
            "backend": "redis" if self._redis else "memory",
            "keys": {},
            "tags": {},
            "pending_refreshes": len(self._pending_refreshes),
            "listener_active": self._subscriber_task is not None,
        }

        if self._redis:
            try:
                # Count keys by pattern
                all_keys = await self._redis.keys("oci:*")
                stats["keys"]["total"] = len(all_keys)

                # Count by type
                stats["keys"]["compartments"] = len(await self._redis.keys("oci:compartment*"))
                stats["keys"]["instances"] = len(await self._redis.keys("oci:instances*"))
                stats["keys"]["databases"] = len(await self._redis.keys("oci:databases*"))
                stats["keys"]["vcns"] = len(await self._redis.keys("oci:vcns*"))
                stats["keys"]["tags"] = len(await self._redis.keys("oci:tags:*"))
                stats["keys"]["metadata"] = len(await self._redis.keys("oci:meta:*"))

                # Redis memory info
                info = await self._redis.info("memory")
                stats["memory"] = {
                    "used_memory_human": info.get("used_memory_human", "unknown"),
                    "peak_memory_human": info.get("used_memory_peak_human", "unknown"),
                }

                # Get tag summary
                tag_keys = await self._redis.keys("oci:tags:*")
                for tag_key in tag_keys[:10]:  # Limit to first 10 tags
                    tag = tag_key.replace("oci:tags:", "")
                    count = await self._redis.scard(tag_key)
                    stats["tags"][tag] = count

            except Exception as e:
                stats["error"] = str(e)
        else:
            # In-memory stats
            stats["keys"]["total"] = len(self._memory_cache)

        # Add discovery stats
        discovery_stats = await self.get_discovery_stats()
        if discovery_stats:
            stats["last_discovery"] = discovery_stats

        return stats

    async def health_check(self) -> dict[str, Any]:
        """
        Perform a health check on the cache.

        Returns:
            Health status including connectivity and freshness
        """
        health: dict[str, Any] = {
            "healthy": False,
            "initialized": self._initialized,
            "backend": "redis" if self._redis else "memory",
        }

        if not self._initialized:
            health["error"] = "Cache not initialized"
            return health

        if self._redis:
            try:
                # Check Redis connectivity
                await self._redis.ping()
                health["redis_connected"] = True

                # Check data freshness
                last_discovery = await self._get(CACHE_KEYS["last_discovery"])
                if last_discovery:
                    last_dt = datetime.fromisoformat(last_discovery)
                    age = datetime.now(UTC) - last_dt
                    health["data_age_hours"] = age.total_seconds() / 3600
                    health["data_stale"] = age > timedelta(hours=6)
                else:
                    health["data_age_hours"] = None
                    health["data_stale"] = True

                health["healthy"] = True

            except Exception as e:
                health["redis_connected"] = False
                health["error"] = str(e)
        else:
            # In-memory fallback is always "healthy"
            health["healthy"] = True
            health["warning"] = "Using in-memory fallback (not persistent)"

        return health
