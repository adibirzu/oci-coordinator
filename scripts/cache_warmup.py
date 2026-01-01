#!/usr/bin/env python
"""
OCI Resource Cache Warmup Script.

Pre-populates the Redis cache with compartment and resource data for fast lookups.

Usage:
    # Warmup compartments only (fast)
    poetry run python scripts/cache_warmup.py

    # Full warmup including resources (slower)
    poetry run python scripts/cache_warmup.py --full

    # Show cache status
    poetry run python scripts/cache_warmup.py --status

    # Clear cache
    poetry run python scripts/cache_warmup.py --clear
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment
env_file = Path(__file__).parent.parent / ".env.local"
if env_file.exists():
    load_dotenv(env_file)

import structlog

# Configure basic logging
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)


async def warmup_compartments() -> dict:
    """Warmup compartment cache using TenancyManager."""
    from src.oci.tenancy_manager import TenancyManager

    logger.info("Warming up compartment cache...")

    manager = TenancyManager.get_instance()

    # Force re-discovery
    manager._initialized = False
    manager._compartments.clear()
    manager._compartments_by_name.clear()

    await manager.initialize()

    # Get results
    tenancies = manager.list_tenancies()
    compartments = await manager.list_compartments()

    results = {
        "tenancies": len(tenancies),
        "compartments": len(compartments),
        "tenancy_details": [],
    }

    for tenancy in tenancies:
        tenancy_compartments = [
            c for c in compartments if c.tenancy_profile == tenancy.profile_name
        ]
        results["tenancy_details"].append({
            "profile": tenancy.profile_name,
            "region": tenancy.region,
            "compartments": len(tenancy_compartments),
        })

        # Print compartment names for this tenancy
        logger.info(
            f"Tenancy {tenancy.profile_name}",
            region=tenancy.region,
            compartments=len(tenancy_compartments),
        )
        for comp in tenancy_compartments[:10]:
            print(f"  - {comp.name}: {comp.id[:50]}...")
        if len(tenancy_compartments) > 10:
            print(f"  ... and {len(tenancy_compartments) - 10} more")

    return results


async def warmup_resources_showoci(profiles: list[str] | None = None) -> dict:
    """
    Warmup resource cache using ShowOCI discovery.

    This method uses the OCI SDK directly (via ShowOCIRunner) to discover
    resources, not MCP tools. This works standalone without the bot running.
    """
    from src.showoci.cache_loader import ShowOCICacheLoader

    logger.info("Warming up resource cache using ShowOCI...")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
    profiles = profiles or os.getenv("OCI_PROFILES", "DEFAULT").split(",")
    profiles = [p.strip() for p in profiles if p.strip()]

    loader = ShowOCICacheLoader(
        redis_url=redis_url,
        profiles=profiles,
    )

    result = await loader.run_full_load()

    return result


async def warmup_resources_mcp(compartment_ids: list[str] | None = None) -> dict:
    """
    Warmup resource cache using MCP tools.

    Requires the bot to be running with MCP servers connected.
    Use warmup_resources_showoci() for standalone execution.
    """
    from src.cache.oci_resource_cache import OCIResourceCache
    from src.mcp.catalog import ToolCatalog

    logger.info("Warming up resource cache using MCP tools...")

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Get tool catalog if MCP is available
    tool_catalog = None
    try:
        tool_catalog = ToolCatalog.get_instance()
    except Exception:
        logger.warning("Tool catalog not available, skipping MCP-based discovery")

    cache = OCIResourceCache.get_instance(redis_url, tool_catalog)
    await cache.initialize()

    if not tool_catalog:
        logger.warning("Cannot perform resource discovery without MCP tools")
        return {"error": "MCP tools not available - use --showoci flag instead"}

    # Get compartments to discover
    if not compartment_ids:
        from src.oci.tenancy_manager import TenancyManager
        manager = TenancyManager.get_instance()
        if not manager._initialized:
            await manager.initialize()
        compartments = await manager.list_compartments()
        compartment_ids = [c.id for c in compartments[:20]]  # Limit to 20 for demo

    results = {
        "compartments_processed": 0,
        "resources": {},
    }

    for comp_id in compartment_ids:
        try:
            discovered = await cache.discover_resources(comp_id)
            results["compartments_processed"] += 1
            for rtype, count in discovered.items():
                results["resources"][rtype] = results["resources"].get(rtype, 0) + count
            logger.info(f"Discovered resources in {comp_id[:30]}...", **discovered)
        except Exception as e:
            logger.warning(f"Failed to discover resources", compartment=comp_id[:30], error=str(e))

    return results


async def get_cache_status() -> dict:
    """Get current cache status."""
    from src.cache.oci_resource_cache import OCIResourceCache
    from src.oci.tenancy_manager import TenancyManager

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Check TenancyManager
    manager = TenancyManager.get_instance()
    tenancy_status = {
        "initialized": manager._initialized,
        "tenancies": len(manager._tenancies),
        "compartments": len(manager._compartments),
    }

    # Check OCIResourceCache
    cache = OCIResourceCache.get_instance(redis_url)
    await cache.initialize()

    cache_stats = await cache.get_cache_stats()
    health = await cache.health_check()

    return {
        "tenancy_manager": tenancy_status,
        "resource_cache": {
            "health": health,
            "stats": cache_stats,
        },
    }


async def clear_cache() -> dict:
    """Clear all cached data."""
    from src.cache.oci_resource_cache import OCIResourceCache
    from src.oci.tenancy_manager import TenancyManager

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

    # Clear TenancyManager
    manager = TenancyManager.get_instance()
    manager._initialized = False
    manager._compartments.clear()
    manager._compartments_by_name.clear()
    manager._tenancies.clear()

    # Clear OCIResourceCache
    cache = OCIResourceCache.get_instance(redis_url)
    await cache.initialize()
    keys_cleared = await cache.clear_cache()

    return {
        "tenancy_manager": "cleared",
        "resource_cache_keys_cleared": keys_cleared,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="OCI Resource Cache Warmup Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Warmup compartments only (fast)
    poetry run python scripts/cache_warmup.py

    # Full warmup using ShowOCI (recommended for standalone)
    poetry run python scripts/cache_warmup.py --full --showoci

    # Full warmup using MCP tools (requires bot running)
    poetry run python scripts/cache_warmup.py --full --mcp

    # Show cache status
    poetry run python scripts/cache_warmup.py --status

    # Clear cache
    poetry run python scripts/cache_warmup.py --clear
""",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Full warmup including resources (slower)",
    )
    parser.add_argument(
        "--showoci",
        action="store_true",
        help="Use ShowOCI for resource discovery (standalone, recommended)",
    )
    parser.add_argument(
        "--mcp",
        action="store_true",
        help="Use MCP tools for resource discovery (requires bot running)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all cached data",
    )
    parser.add_argument(
        "--compartment",
        type=str,
        help="Specific compartment name or OCID to warm up resources for",
    )
    parser.add_argument(
        "--profiles",
        type=str,
        help="Comma-separated OCI profiles to use (default: from OCI_PROFILES env)",
    )

    args = parser.parse_args()

    if args.status:
        logger.info("Getting cache status...")
        status = await get_cache_status()
        print("\n--- Cache Status ---")
        import json
        print(json.dumps(status, indent=2, default=str))
        return

    if args.clear:
        logger.info("Clearing cache...")
        result = await clear_cache()
        print("\n--- Cache Cleared ---")
        print(f"TenancyManager: {result['tenancy_manager']}")
        print(f"ResourceCache keys cleared: {result['resource_cache_keys_cleared']}")
        return

    # Warmup compartments
    print("\n=== Warming up OCI Resource Cache ===\n")
    comp_result = await warmup_compartments()

    print(f"\n--- Compartment Discovery Complete ---")
    print(f"Tenancies: {comp_result['tenancies']}")
    print(f"Compartments: {comp_result['compartments']}")

    # Full warmup includes resources
    if args.full:
        # Determine discovery method
        use_showoci = args.showoci or not args.mcp  # Default to ShowOCI

        if use_showoci:
            # Use ShowOCI for standalone resource discovery
            profiles = None
            if args.profiles:
                profiles = [p.strip() for p in args.profiles.split(",")]

            res_result = await warmup_resources_showoci(profiles)

            print(f"\n--- ShowOCI Resource Discovery Complete ---")
            print(f"Profiles loaded: {res_result.get('profiles_loaded', 0)}")
            print(f"Profiles failed: {res_result.get('profiles_failed', 0)}")
            print(f"Duration: {res_result.get('total_duration_seconds', 0):.1f}s")
            print(f"Resources cached:")
            for rtype, count in res_result.get('total_resources', {}).items():
                print(f"  - {rtype}: {count}")

        else:
            # Use MCP tools (requires bot running)
            compartment_ids = None
            if args.compartment:
                # Resolve compartment name if needed
                if args.compartment.startswith("ocid1."):
                    compartment_ids = [args.compartment]
                else:
                    from src.oci.tenancy_manager import TenancyManager
                    manager = TenancyManager.get_instance()
                    ocid = await manager.get_compartment_ocid(args.compartment)
                    if ocid:
                        compartment_ids = [ocid]
                    else:
                        logger.error(f"Compartment not found: {args.compartment}")
                        return

            res_result = await warmup_resources_mcp(compartment_ids)
            print(f"\n--- MCP Resource Discovery Complete ---")
            print(f"Compartments processed: {res_result.get('compartments_processed', 0)}")
            print(f"Resources found: {res_result.get('resources', {})}")

    print("\n=== Cache Warmup Complete ===")
    print("The infrastructure agent can now resolve compartment names to OCIDs.")


if __name__ == "__main__":
    asyncio.run(main())
