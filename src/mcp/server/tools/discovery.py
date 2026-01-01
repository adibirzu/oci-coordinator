"""
OCI Discovery MCP Tools.

Provides tools for resource discovery using ShowOCI-style
comprehensive tenancy scanning and cache management.
"""

from __future__ import annotations

import json
from typing import Any

from opentelemetry import trace

from src.cache.oci_resource_cache import OCIResourceCache
from src.showoci.cache_loader import ShowOCICacheLoader
from src.showoci.runner import ShowOCIConfig, ShowOCIRunner

# Get tracer for discovery tools
_tracer = trace.get_tracer("mcp-oci-discovery")

# Global cache loader instance
_cache_loader: ShowOCICacheLoader | None = None
_resource_cache: OCIResourceCache | None = None


def _get_cache_loader() -> ShowOCICacheLoader:
    """Get or create cache loader instance."""
    global _cache_loader
    if _cache_loader is None:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        profiles = os.getenv("OCI_PROFILES", "DEFAULT").split(",")
        _cache_loader = ShowOCICacheLoader(
            redis_url=redis_url,
            profiles=[p.strip() for p in profiles],
        )
    return _cache_loader


async def _get_resource_cache() -> OCIResourceCache:
    """Get or create resource cache instance."""
    global _resource_cache
    if _resource_cache is None:
        import os
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        _resource_cache = OCIResourceCache(redis_url=redis_url)
        await _resource_cache.initialize()
    return _resource_cache


async def _run_discovery_logic(
    profile: str = "DEFAULT",
    resource_types: list[str] | None = None,
    compartment_id: str | None = None,
    load_to_cache: bool = True,
) -> str:
    """Internal logic for running discovery."""
    config = ShowOCIConfig(
        profile=profile,
        resource_types=resource_types or ["compute", "network", "database"],
        compartment_ocid=compartment_id,
    )

    runner = ShowOCIRunner(config=config)
    result = await runner.run_discovery()

    if not result.success:
        return json.dumps({
            "success": False,
            "error": result.error,
        }, indent=2)

    # Load to cache if requested
    if load_to_cache:
        loader = _get_cache_loader()
        await loader._load_result_to_cache(await _get_resource_cache(), result)

    return json.dumps({
        "success": True,
        "profile": result.profile,
        "duration_seconds": result.duration_seconds,
        "compartments_scanned": result.compartments_scanned,
        "resource_counts": result.resource_counts,
        "timestamp": result.timestamp,
    }, indent=2)


async def _get_cached_resources_logic(
    resource_type: str,
    compartment_name: str | None = None,
    compartment_id: str | None = None,
    format: str = "json",
) -> str:
    """Internal logic for getting cached resources."""
    cache = await _get_resource_cache()

    # Resolve compartment name to ID if needed
    if compartment_name and not compartment_id:
        comp = await cache.get_compartment_by_name(compartment_name)
        if comp:
            compartment_id = comp.get("id")

    # Get resources based on type
    if resource_type == "instances":
        resources = await cache.get_instances(compartment_id)
    elif resource_type == "databases":
        resources = await cache.get_databases(compartment_id)
    elif resource_type == "vcns":
        resources = await cache.get_vcns(compartment_id)
    elif resource_type == "compartments":
        resources = await cache.get_compartments()
    else:
        resources = await cache.get_resources(resource_type, compartment_id)

    if format == "json":
        return json.dumps(resources, indent=2, default=str)

    # Markdown format
    if not resources:
        return f"No {resource_type} found" + (f" in compartment {compartment_name or compartment_id}" if compartment_name or compartment_id else "")

    lines = [f"# {resource_type.title()} ({len(resources)} found)\n"]

    if resource_type == "instances":
        lines.append("| Name | State | Shape | Region |")
        lines.append("| --- | --- | --- | --- |")
        for r in resources[:50]:  # Limit to 50
            lines.append(f"| {r.get('display_name', 'N/A')} | {r.get('lifecycle_state', 'N/A')} | {r.get('shape', 'N/A')} | {r.get('region', 'N/A')} |")
    elif resource_type == "databases":
        lines.append("| Name | State | Workload | CPUs |")
        lines.append("| --- | --- | --- | --- |")
        for r in resources[:50]:
            lines.append(f"| {r.get('display_name', 'N/A')} | {r.get('lifecycle_state', 'N/A')} | {r.get('db_workload', 'N/A')} | {r.get('cpu_core_count', 'N/A')} |")
    elif resource_type == "vcns":
        lines.append("| Name | CIDR | State |")
        lines.append("| --- | --- | --- |")
        for r in resources[:50]:
            lines.append(f"| {r.get('display_name', 'N/A')} | {r.get('cidr_block', 'N/A')} | {r.get('lifecycle_state', 'N/A')} |")
    elif resource_type == "compartments":
        lines.append("| Name | State | ID |")
        lines.append("| --- | --- | --- |")
        for r in resources[:50]:
            lines.append(f"| {r.get('name', 'N/A')} | {r.get('lifecycle_state', 'N/A')} | `{r.get('id', 'N/A')[:50]}...` |")
    # Generic table
    elif resources:
        keys = list(resources[0].keys())[:5]
        lines.append("| " + " | ".join(keys) + " |")
        lines.append("| " + " | ".join(["---"] * len(keys)) + " |")
        for r in resources[:50]:
            values = [str(r.get(k, "N/A"))[:30] for k in keys]
            lines.append("| " + " | ".join(values) + " |")

    if len(resources) > 50:
        lines.append(f"\n*...and {len(resources) - 50} more*")

    return "\n".join(lines)


async def _refresh_cache_logic(
    profiles: list[str] | None = None,
    resource_types: list[str] | None = None,
) -> str:
    """Internal logic for refreshing cache."""
    loader = _get_cache_loader()

    # Override profiles if specified
    if profiles:
        loader.profiles = profiles

    result = await loader.run_full_load()

    return json.dumps(result, indent=2, default=str)


async def _get_resource_summary_logic(
    compartment_name: str | None = None,
) -> str:
    """Internal logic for getting resource summary."""
    cache = await _get_resource_cache()
    summary = await cache.get_resource_summary(compartment_name)

    return json.dumps(summary, indent=2, default=str)


async def _search_resources_logic(
    query: str,
    resource_type: str | None = None,
    compartment_name: str | None = None,
) -> str:
    """Internal logic for searching resources."""
    cache = await _get_resource_cache()

    # Resolve compartment name to ID
    compartment_id = None
    if compartment_name:
        comp = await cache.get_compartment_by_name(compartment_name)
        if comp:
            compartment_id = comp.get("id")

    results = await cache.search_resources(query, resource_type, compartment_id)

    if not results:
        return json.dumps({
            "query": query,
            "results": [],
            "message": f"No resources found matching '{query}'"
        }, indent=2)

    return json.dumps({
        "query": query,
        "count": len(results),
        "results": results[:20],  # Limit to 20
    }, indent=2, default=str)


async def _get_cache_status_logic() -> str:
    """Internal logic for cache status."""
    loader = _get_cache_loader()
    status = await loader.get_cache_status()
    return json.dumps(status, indent=2, default=str)


def register_discovery_tools(mcp: Any) -> None:
    """Register discovery tools with the MCP server."""

    @mcp.tool()
    async def oci_discovery_run(
        profile: str = "DEFAULT",
        resource_types: str = "compute,network,database",
        compartment_id: str | None = None,
        load_to_cache: bool = True,
    ) -> str:
        """Run OCI resource discovery (ShowOCI-style).

        Discovers resources across all compartments and regions,
        optionally loading results into the cache for fast access.

        Args:
            profile: OCI config profile to use
            resource_types: Comma-separated types (compute,network,database,storage,all)
            compartment_id: Optional compartment OCID to limit discovery
            load_to_cache: Whether to cache discovered resources

        Returns:
            Discovery results with resource counts
        """
        types_list = [t.strip() for t in resource_types.split(",")]
        return await _run_discovery_logic(profile, types_list, compartment_id, load_to_cache)

    @mcp.tool()
    async def oci_discovery_get_cached(
        resource_type: str,
        compartment_name: str | None = None,
        compartment_id: str | None = None,
        format: str = "markdown",
    ) -> str:
        """Get cached OCI resources.

        Retrieves resources from the pre-populated cache for fast access.
        Run oci_discovery_run or oci_discovery_refresh first to populate cache.

        Args:
            resource_type: Type of resources (instances, databases, vcns, compartments)
            compartment_name: Filter by compartment name
            compartment_id: Filter by compartment OCID
            format: Output format ('json' or 'markdown')

        Returns:
            Cached resources in specified format
        """
        return await _get_cached_resources_logic(resource_type, compartment_name, compartment_id, format)

    @mcp.tool()
    async def oci_discovery_refresh(
        profiles: str | None = None,
        resource_types: str = "compute,network,database",
    ) -> str:
        """Refresh the OCI resource cache.

        Runs full discovery for all configured profiles and updates
        the cache with fresh data.

        Args:
            profiles: Comma-separated profiles (default: all configured)
            resource_types: Comma-separated types to discover

        Returns:
            Refresh statistics
        """
        profiles_list = [p.strip() for p in profiles.split(",")] if profiles else None
        return await _refresh_cache_logic(profiles_list)

    @mcp.tool()
    async def oci_discovery_summary(
        compartment_name: str | None = None,
    ) -> str:
        """Get a summary of cached OCI resources.

        Provides counts and state breakdowns for all cached resources.

        Args:
            compartment_name: Optional compartment to summarize

        Returns:
            Resource summary with counts by type and state
        """
        return await _get_resource_summary_logic(compartment_name)

    @mcp.tool()
    async def oci_discovery_search(
        query: str,
        resource_type: str | None = None,
        compartment_name: str | None = None,
    ) -> str:
        """Search cached OCI resources by name.

        Searches across all cached resources matching the query string.

        Args:
            query: Search query (matches display_name)
            resource_type: Filter by type (instance, database, vcn)
            compartment_name: Filter by compartment

        Returns:
            Matching resources (max 20)
        """
        return await _search_resources_logic(query, resource_type, compartment_name)

    @mcp.tool()
    async def oci_discovery_cache_status() -> str:
        """Get cache status and statistics.

        Returns information about cache health, last refresh times,
        and configured profiles.

        Returns:
            Cache status and statistics
        """
        return await _get_cache_status_logic()
