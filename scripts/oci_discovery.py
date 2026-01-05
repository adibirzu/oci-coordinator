#!/usr/bin/env python3
"""
OCI Multi-Profile Discovery Script.

Pre-warms cache with compartment and resource data for ALL configured OCI profiles.
Can be run manually or scheduled via cron for faster query responses.

Usage:
    # Full discovery (compartments + databases)
    poetry run python scripts/oci_discovery.py

    # Compartments only
    poetry run python scripts/oci_discovery.py --compartments-only

    # Specific profiles
    poetry run python scripts/oci_discovery.py --profiles DEFAULT,emdemo

    # Check cache status
    poetry run python scripts/oci_discovery.py --status

    # Clear cache
    poetry run python scripts/oci_discovery.py --clear-cache
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# Multi-region configuration per profile
# Add additional regions to check for databases beyond the default region
PROFILE_ADDITIONAL_REGIONS: dict[str, list[str]] = {
    "emdemo": ["uk-london-1", "us-ashburn-1"],  # emdemo has DBs in multiple regions
    # Add more profiles as needed:
    # "myprofile": ["us-ashburn-1", "eu-frankfurt-1"],
}


@dataclass
class DiscoveryResult:
    """Result of a discovery operation for one profile."""

    profile: str
    region: str  # Primary region (may include additional regions in databases)
    tenancy_ocid: str
    compartments: int
    databases: int
    opsi_databases: int
    dbmgmt_databases: int  # DB Management managed databases
    discovery_time: str
    duration_seconds: float
    errors: list[str]
    regions_checked: list[str] | None = None  # All regions checked for DBs


@dataclass
class DiscoveryCache:
    """Cached discovery data for a profile."""

    profile: str
    region: str
    tenancy_ocid: str
    discovered_at: str
    compartments: list[dict]
    databases: list[dict]
    opsi_databases: list[dict]
    dbmgmt_databases: list[dict]  # DB Management managed databases
    regions_checked: list[str] | None = None  # All regions checked for DBs


def get_cache_dir() -> Path:
    """Get the cache directory for discovery data."""
    cache_dir = Path.home() / ".oci_coordinator_cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_cache_file(profile: str) -> Path:
    """Get cache file path for a profile."""
    return get_cache_dir() / f"discovery_{profile}.json"


async def discover_compartments(profile: str, config: Any) -> tuple[list[dict], list[str]]:
    """Discover all compartments for a profile."""
    import oci

    errors = []
    compartments = []

    try:
        identity_client = oci.identity.IdentityClient(config)
        tenancy_ocid = config.get("tenancy")

        # Get root compartment
        try:
            root = identity_client.get_compartment(tenancy_ocid).data
            compartments.append({
                "id": root.id,
                "name": root.name or "root",
                "description": "Tenancy root",
                "parent_id": None,
                "path": "root",
                "lifecycle_state": "ACTIVE",
            })
        except Exception as e:
            errors.append(f"Could not get root compartment: {e}")

        # List all sub-compartments
        response = identity_client.list_compartments(
            compartment_id=tenancy_ocid,
            compartment_id_in_subtree=True,
            access_level="ACCESSIBLE",
            lifecycle_state="ACTIVE",
        )

        # Build compartment map for path resolution
        comp_map = {c.id: c for c in response.data}

        for comp in response.data:
            # Build path
            path_parts = [comp.name]
            current_id = comp.compartment_id
            while current_id in comp_map:
                path_parts.insert(0, comp_map[current_id].name)
                current_id = comp_map[current_id].compartment_id

            compartments.append({
                "id": comp.id,
                "name": comp.name,
                "description": comp.description,
                "parent_id": comp.compartment_id,
                "path": "/".join(path_parts),
                "lifecycle_state": comp.lifecycle_state,
            })

    except Exception as e:
        errors.append(f"Compartment discovery failed: {e}")

    return compartments, errors


async def discover_databases(
    profile: str,
    config: Any,
    compartments: list[dict],
    region_override: str | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Discover Autonomous Databases for a profile.

    Args:
        profile: OCI profile name
        config: OCI config dict
        compartments: List of compartments to search
        region_override: Optional region to use instead of config default
    """
    import oci

    errors = []
    databases = []

    try:
        # Create region-specific config if override provided
        if region_override:
            config = dict(config)
            config["region"] = region_override

        db_client = oci.database.DatabaseClient(config)
        region = config.get("region", "unknown")

        for comp in compartments:
            try:
                response = db_client.list_autonomous_databases(
                    compartment_id=comp["id"],
                    lifecycle_state="AVAILABLE",
                )

                for db in response.data:
                    databases.append({
                        "id": db.id,
                        "display_name": db.display_name,
                        "db_name": db.db_name,
                        "compartment_id": comp["id"],
                        "compartment_name": comp["name"],
                        "lifecycle_state": db.lifecycle_state,
                        "db_workload": db.db_workload,
                        "is_free_tier": db.is_free_tier,
                        "cpu_core_count": db.cpu_core_count,
                        "data_storage_size_in_gbs": db.data_storage_size_in_gbs,
                        "region": region,  # Track which region this DB is in
                    })

            except oci.exceptions.ServiceError as e:
                if e.status != 404:  # Ignore not found
                    errors.append(f"DB discovery in {comp['name']} ({region}): {e.message}")
            except Exception as e:
                errors.append(f"DB discovery in {comp['name']} ({region}): {e}")

    except Exception as e:
        errors.append(f"Database client init failed: {e}")

    return databases, errors


async def discover_opsi_databases(
    profile: str,
    config: Any,
    region_override: str | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Discover databases registered in OPS Insights.

    Args:
        profile: OCI profile name
        config: OCI config dict
        region_override: Optional region to use instead of config default
    """
    import oci

    errors = []
    opsi_databases = []

    try:
        # Create region-specific config if override provided
        if region_override:
            config = dict(config)
            config["region"] = region_override

        opsi_client = oci.opsi.OperationsInsightsClient(config)
        region = config.get("region", "unknown")

        # Get all database insights in the tenancy
        response = opsi_client.list_database_insights(
            compartment_id=config.get("tenancy"),
            compartment_id_in_subtree=True,
            status=["ENABLED"],
        )

        # Handle both single item and collection responses
        items = response.data.items if hasattr(response.data, "items") else response.data
        if items is None:
            items = []

        for db in items:
            opsi_databases.append({
                "id": db.id,
                "database_id": getattr(db, "database_id", None),
                "database_name": getattr(db, "database_name", None),
                "database_display_name": getattr(db, "database_display_name", None),
                "compartment_id": db.compartment_id,
                "database_type": db.database_type,
                "entity_source": db.entity_source,
                "status": db.status,
                "region": region,  # Track which region this DB is in
            })

    except oci.exceptions.ServiceError as e:
        if "NotAuthorizedOrNotFound" not in str(e):
            errors.append(f"OPSI discovery ({region}): {e.message}")
    except Exception as e:
        errors.append(f"OPSI client init failed ({region_override or 'default'}): {e}")

    return opsi_databases, errors


async def discover_dbmgmt_databases(
    profile: str,
    config: Any,
    compartments: list[dict],
    region_override: str | None = None,
) -> tuple[list[dict], list[str]]:
    """
    Discover databases registered in DB Management (Managed Databases).

    These are databases that have been enabled for DB Management features
    like performance monitoring, AWR reports, SQL tuning, etc.

    NOTE: Unlike OPSI, DB Management's list_managed_databases API does NOT
    support compartment_id_in_subtree, so we must iterate through all compartments.

    Args:
        profile: OCI profile name
        config: OCI config dict
        compartments: List of compartments to search (from discover_compartments)
        region_override: Optional region to use instead of config default
    """
    import oci

    errors = []
    dbmgmt_databases = []

    try:
        # Create region-specific config if override provided
        if region_override:
            config = dict(config)
            config["region"] = region_override

        dbmgmt_client = oci.database_management.DbManagementClient(config)
        region = config.get("region", "unknown")

        # Search each compartment (DB Management API doesn't support subtree search)
        all_items = []
        for comp in compartments:
            try:
                response = dbmgmt_client.list_managed_databases(
                    compartment_id=comp["id"],
                )

                items = response.data.items if hasattr(response.data, "items") else []
                # Tag each item with compartment info for better context
                for item in items:
                    item._compartment_name = comp["name"]
                    item._compartment_path = comp.get("path", comp["name"])
                all_items.extend(items)

                # Handle pagination within compartment
                while response.has_next_page:
                    response = dbmgmt_client.list_managed_databases(
                        compartment_id=comp["id"],
                        page=response.next_page,
                    )
                    items = response.data.items if hasattr(response.data, "items") else []
                    for item in items:
                        item._compartment_name = comp["name"]
                        item._compartment_path = comp.get("path", comp["name"])
                    all_items.extend(items)

            except oci.exceptions.ServiceError as e:
                if e.status != 404 and "NotAuthorizedOrNotFound" not in str(e):
                    errors.append(f"DB Management in {comp['name']} ({region}): {e.message}")
            except Exception as e:
                errors.append(f"DB Management in {comp['name']} ({region}): {e}")

        for db in all_items:
            dbmgmt_databases.append({
                "id": db.id,
                "name": db.name,
                "database_type": db.database_type,
                "database_sub_type": db.database_sub_type,
                "compartment_id": db.compartment_id,
                "compartment_name": getattr(db, "_compartment_name", None),
                "compartment_path": getattr(db, "_compartment_path", None),
                "is_cluster": getattr(db, "is_cluster", False),
                "parent_container_id": getattr(db, "parent_container_id", None),
                "deployment_type": getattr(db, "deployment_type", None),
                "management_option": getattr(db, "management_option", None),
                "workload_type": getattr(db, "workload_type", None),
                "time_created": str(getattr(db, "time_created", None)),
                "region": region,  # Track which region this DB is in
            })

    except Exception as e:
        errors.append(f"DB Management client init failed ({region_override or 'default'}): {e}")

    return dbmgmt_databases, errors


async def run_discovery_for_profile(
    profile: str,
    include_databases: bool = True,
    include_opsi: bool = True,
    include_dbmgmt: bool = True,
) -> DiscoveryResult:
    """Run full discovery for a single profile."""
    import oci

    start_time = datetime.now()
    errors = []

    print(f"\n{'=' * 60}")
    print(f"Discovering: {profile}")
    print(f"{'=' * 60}")

    # Load OCI config
    try:
        config = oci.config.from_file(profile_name=profile)
        region = config.get("region", "unknown")
        tenancy_ocid = config.get("tenancy", "")
        print(f"  Region: {region}")
        print(f"  Tenancy: {tenancy_ocid[:40]}...")
    except Exception as e:
        return DiscoveryResult(
            profile=profile,
            region="unknown",
            tenancy_ocid="",
            compartments=0,
            databases=0,
            opsi_databases=0,
            dbmgmt_databases=0,
            discovery_time=start_time.isoformat(),
            duration_seconds=0,
            errors=[f"Config load failed: {e}"],
        )

    # Discover compartments (only in default region - compartments are global)
    print("\n  Discovering compartments...")
    compartments, comp_errors = await discover_compartments(profile, config)
    errors.extend(comp_errors)
    print(f"    Found {len(compartments)} compartments")

    # Get all regions to check for databases
    regions_to_check = [region]  # Start with default region
    additional_regions = PROFILE_ADDITIONAL_REGIONS.get(profile, [])
    regions_to_check.extend(additional_regions)

    if additional_regions:
        print(f"\n  Multi-region discovery enabled: {', '.join(regions_to_check)}")

    # Discover databases across all regions
    databases = []
    if include_databases:
        for check_region in regions_to_check:
            is_additional = check_region != region
            region_label = f" ({check_region})" if is_additional else ""
            print(f"  Discovering Autonomous Databases{region_label}...")

            region_override = check_region if is_additional else None
            db_list, db_errors = await discover_databases(
                profile, config, compartments, region_override=region_override
            )
            errors.extend(db_errors)
            databases.extend(db_list)
            print(f"    Found {len(db_list)} databases in {check_region}")

        if len(regions_to_check) > 1:
            print(f"    Total databases across all regions: {len(databases)}")

    # Discover OPSI databases across all regions
    opsi_databases = []
    if include_opsi:
        for check_region in regions_to_check:
            is_additional = check_region != region
            region_label = f" ({check_region})" if is_additional else ""
            print(f"  Discovering OPSI-registered databases{region_label}...")

            region_override = check_region if is_additional else None
            opsi_list, opsi_errors = await discover_opsi_databases(
                profile, config, region_override=region_override
            )
            errors.extend(opsi_errors)
            opsi_databases.extend(opsi_list)
            print(f"    Found {len(opsi_list)} OPSI databases in {check_region}")

        if len(regions_to_check) > 1:
            print(f"    Total OPSI databases across all regions: {len(opsi_databases)}")

    # Discover DB Management (Managed) databases across all regions
    # NOTE: DB Management API doesn't support compartment_id_in_subtree,
    # so we pass compartments list to iterate through them
    dbmgmt_databases = []
    if include_dbmgmt:
        for check_region in regions_to_check:
            is_additional = check_region != region
            region_label = f" ({check_region})" if is_additional else ""
            print(f"  Discovering DB Management databases{region_label}...")
            print(f"    Searching {len(compartments)} compartments...")

            region_override = check_region if is_additional else None
            dbmgmt_list, dbmgmt_errors = await discover_dbmgmt_databases(
                profile, config, compartments, region_override=region_override
            )
            errors.extend(dbmgmt_errors)
            dbmgmt_databases.extend(dbmgmt_list)
            print(f"    Found {len(dbmgmt_list)} managed databases in {check_region}")

        if len(regions_to_check) > 1:
            print(f"    Total managed databases across all regions: {len(dbmgmt_databases)}")

    # Save to cache file
    cache_data = DiscoveryCache(
        profile=profile,
        region=region,
        tenancy_ocid=tenancy_ocid,
        discovered_at=start_time.isoformat(),
        compartments=compartments,
        databases=databases,
        opsi_databases=opsi_databases,
        dbmgmt_databases=dbmgmt_databases,
        regions_checked=regions_to_check,
    )

    cache_file = get_cache_file(profile)
    with open(cache_file, "w") as f:
        json.dump(asdict(cache_data), f, indent=2)
    print(f"\n  Cache saved: {cache_file}")

    # Also save to Redis if available
    try:
        import redis.asyncio as redis_lib
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis_lib.from_url(redis_url)
        await redis_client.ping()

        # Cache compartments
        for comp in compartments:
            key = f"discovery:{profile}:compartment:{comp['id']}"
            await redis_client.setex(key, 86400, json.dumps(comp))  # 24h TTL

            # Name index
            name_key = f"discovery:{profile}:compartment:name:{comp['name'].lower()}"
            await redis_client.setex(name_key, 86400, comp["id"])

        # Cache databases
        for db in databases:
            key = f"discovery:{profile}:database:{db['id']}"
            await redis_client.setex(key, 86400, json.dumps(db))

        # Cache OPSI databases
        for db in opsi_databases:
            key = f"discovery:{profile}:opsi:{db['id']}"
            await redis_client.setex(key, 86400, json.dumps(db))

        # Cache DB Management databases
        for db in dbmgmt_databases:
            key = f"discovery:{profile}:dbmgmt:{db['id']}"
            await redis_client.setex(key, 86400, json.dumps(db))

        # Cache summary
        summary_key = f"discovery:{profile}:summary"
        await redis_client.setex(
            summary_key,
            86400,
            json.dumps({
                "profile": profile,
                "region": region,
                "regions_checked": regions_to_check,
                "compartments": len(compartments),
                "databases": len(databases),
                "opsi_databases": len(opsi_databases),
                "dbmgmt_databases": len(dbmgmt_databases),
                "discovered_at": start_time.isoformat(),
            }),
        )

        await redis_client.close()
        print("  Redis cache updated")

    except Exception as e:
        print(f"  Redis not available: {e}")

    duration = (datetime.now() - start_time).total_seconds()

    if errors:
        print(f"\n  Errors ({len(errors)}):")
        for err in errors[:5]:
            print(f"    - {err}")
        if len(errors) > 5:
            print(f"    ... and {len(errors) - 5} more")

    return DiscoveryResult(
        profile=profile,
        region=region,
        tenancy_ocid=tenancy_ocid,
        compartments=len(compartments),
        databases=len(databases),
        opsi_databases=len(opsi_databases),
        dbmgmt_databases=len(dbmgmt_databases),
        discovery_time=start_time.isoformat(),
        duration_seconds=duration,
        errors=errors,
        regions_checked=regions_to_check,
    )


async def get_cache_status() -> dict[str, Any]:
    """Get status of cached discovery data."""
    cache_dir = get_cache_dir()
    status = {
        "cache_dir": str(cache_dir),
        "profiles": {},
    }

    for cache_file in cache_dir.glob("discovery_*.json"):
        try:
            with open(cache_file) as f:
                data = json.load(f)
                profile = data.get("profile", cache_file.stem.replace("discovery_", ""))
                status["profiles"][profile] = {
                    "discovered_at": data.get("discovered_at"),
                    "compartments": len(data.get("compartments", [])),
                    "databases": len(data.get("databases", [])),
                    "opsi_databases": len(data.get("opsi_databases", [])),
                    "dbmgmt_databases": len(data.get("dbmgmt_databases", [])),
                    "regions_checked": data.get("regions_checked"),
                    "cache_file": str(cache_file),
                }
        except Exception as e:
            status["profiles"][cache_file.stem] = {"error": str(e)}

    return status


async def clear_cache() -> None:
    """Clear all cached discovery data."""
    cache_dir = get_cache_dir()

    # Clear files
    for cache_file in cache_dir.glob("discovery_*.json"):
        cache_file.unlink()
        print(f"Deleted: {cache_file}")

    # Clear Redis
    try:
        import redis.asyncio as redis_lib
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        redis_client = redis_lib.from_url(redis_url)
        await redis_client.ping()

        # Delete discovery keys
        keys = await redis_client.keys("discovery:*")
        if keys:
            await redis_client.delete(*keys)
            print(f"Deleted {len(keys)} Redis keys")

        await redis_client.close()

    except Exception as e:
        print(f"Redis clear failed: {e}")


def get_configured_profiles() -> list[str]:
    """Get list of configured OCI profiles."""
    config_path = Path.home() / ".oci" / "config"
    profiles = []

    if config_path.exists():
        with open(config_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("[") and line.endswith("]"):
                    profiles.append(line[1:-1])

    return profiles


async def main():
    parser = argparse.ArgumentParser(
        description="OCI Multi-Profile Discovery Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  poetry run python scripts/oci_discovery.py
  poetry run python scripts/oci_discovery.py --profiles DEFAULT,emdemo
  poetry run python scripts/oci_discovery.py --compartments-only
  poetry run python scripts/oci_discovery.py --status
  poetry run python scripts/oci_discovery.py --clear-cache
        """,
    )

    parser.add_argument(
        "--profiles",
        type=str,
        help="Comma-separated list of profiles (default: all configured)",
    )
    parser.add_argument(
        "--compartments-only",
        action="store_true",
        help="Only discover compartments (skip databases)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cache status and exit",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached data and exit",
    )

    args = parser.parse_args()

    # Handle status command
    if args.status:
        status = await get_cache_status()
        print(f"\nCache Directory: {status['cache_dir']}")
        print(f"\nCached Profiles:")
        for profile, data in status["profiles"].items():
            if "error" in data:
                print(f"  {profile}: ERROR - {data['error']}")
            else:
                print(f"  {profile}:")
                print(f"    Discovered: {data['discovered_at']}")
                print(f"    Compartments: {data['compartments']}")
                print(f"    Databases: {data['databases']}")
                print(f"    OPSI DBs: {data['opsi_databases']}")
                print(f"    DB Mgmt DBs: {data['dbmgmt_databases']}")
                if data.get("regions_checked"):
                    print(f"    Regions: {', '.join(data['regions_checked'])}")
        return

    # Handle clear cache command
    if args.clear_cache:
        print("Clearing discovery cache...")
        await clear_cache()
        print("Cache cleared.")
        return

    # Get profiles to discover
    if args.profiles:
        profiles = [p.strip() for p in args.profiles.split(",")]
    else:
        profiles = get_configured_profiles()

    if not profiles:
        print("No OCI profiles found in ~/.oci/config")
        return

    print(f"\nOCI Multi-Profile Discovery")
    print(f"Profiles to discover: {', '.join(profiles)}")
    print(f"Include databases: {not args.compartments_only}")

    # Run discovery for each profile
    results = []
    for profile in profiles:
        result = await run_discovery_for_profile(
            profile=profile,
            include_databases=not args.compartments_only,
            include_opsi=not args.compartments_only,
        )
        results.append(result)

    # Summary
    print(f"\n{'=' * 60}")
    print("DISCOVERY SUMMARY")
    print(f"{'=' * 60}")

    total_compartments = 0
    total_databases = 0
    total_opsi = 0
    total_dbmgmt = 0

    for result in results:
        print(f"\n{result.profile} ({result.region}):")
        print(f"  Compartments: {result.compartments}")
        print(f"  Databases: {result.databases}")
        print(f"  OPSI DBs: {result.opsi_databases}")
        print(f"  DB Mgmt DBs: {result.dbmgmt_databases}")
        print(f"  Duration: {result.duration_seconds:.1f}s")
        if result.errors:
            print(f"  Errors: {len(result.errors)}")

        total_compartments += result.compartments
        total_databases += result.databases
        total_opsi += result.opsi_databases
        total_dbmgmt += result.dbmgmt_databases

    print(f"\nTOTAL:")
    print(f"  Profiles: {len(results)}")
    print(f"  Compartments: {total_compartments}")
    print(f"  Databases: {total_databases}")
    print(f"  OPSI DBs: {total_opsi}")
    print(f"  DB Mgmt DBs: {total_dbmgmt}")
    print(f"\nCache location: {get_cache_dir()}")


if __name__ == "__main__":
    asyncio.run(main())
