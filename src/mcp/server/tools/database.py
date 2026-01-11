"""
MCP Tools for OCI Database Management Service.

Provides tools for:
- Listing managed databases (with recursive compartment search)
- Searching managed databases by name
- Getting database performance metrics
- Getting AWR reports
- Executing SQL queries
"""

import asyncio
import json
from datetime import UTC
from typing import Any

from fastmcp import FastMCP

from src.mcp.server.auth import (
    get_database_client,
    get_database_management_client,
)


async def _call_oci(client_method, **kwargs):
    """Run a blocking OCI SDK call in a worker thread."""
    return await asyncio.to_thread(client_method, **kwargs)


async def _get_compartments_recursive(
    identity_client,
    root_compartment_id: str,
) -> list[dict]:
    """Get all compartments recursively from a root compartment."""
    compartments = [{"id": root_compartment_id, "name": "root"}]

    try:
        response = await _call_oci(
            identity_client.list_compartments,
            compartment_id=root_compartment_id,
            compartment_id_in_subtree=True,
            access_level="ACCESSIBLE",
            lifecycle_state="ACTIVE",
        )
        for comp in response.data:
            compartments.append({
                "id": comp.id,
                "name": comp.name,
            })
    except Exception:
        # If we can't list compartments, just use the root
        pass

    return compartments


async def _list_autonomous_databases_logic(
    compartment_id: str | None = None,
    workload_type: str | None = None,
    lifecycle_state: str | None = None,
    limit: int = 50,
    profile: str | None = None,
    region: str | None = None,
    include_subtree: bool = True,
) -> str:
    """Internal logic for listing autonomous databases.

    Uses the OCI Database service API to list autonomous databases.
    Searches child compartments recursively when include_subtree is True.
    """
    import oci

    try:
        client = get_database_client(profile=profile, region=region)
        config = client.base_client.config
        root_comp_id = compartment_id or config.get("tenancy")

        # Get compartments to search
        compartments_to_search = [root_comp_id]
        if include_subtree:
            try:
                identity_client = oci.identity.IdentityClient(config)
                comps_resp = identity_client.list_compartments(
                    compartment_id=root_comp_id,
                    compartment_id_in_subtree=True,
                    access_level="ACCESSIBLE",
                    lifecycle_state="ACTIVE",
                )
                compartments_to_search.extend([c.id for c in comps_resp.data])
            except Exception:
                pass  # Fall back to single compartment

        # Search for databases in all compartments
        databases = []
        compartments_searched = 0
        for comp_id in compartments_to_search:
            if len(databases) >= limit:
                break

            # Build list parameters
            list_params = {
                "compartment_id": comp_id,
                "limit": min(limit - len(databases), 50),
            }

            # Add optional filters
            if lifecycle_state:
                list_params["lifecycle_state"] = lifecycle_state.upper()
            if workload_type:
                # Map common workload type names to OCI API values
                workload_map = {
                    "atp": "OLTP",
                    "adw": "DW",
                    "oltp": "OLTP",
                    "dw": "DW",
                    "apex": "APEX",
                    "ajd": "AJD",
                }
                list_params["db_workload"] = workload_map.get(
                    workload_type.lower(), workload_type.upper()
                )

            # Call OCI API for this compartment
            try:
                response = await _call_oci(
                    client.list_autonomous_databases,
                    **list_params,
                )
                compartments_searched += 1

                # Process databases from this compartment
                for db in response.data:
                    if len(databases) >= limit:
                        break
                    db_info = {
                        "name": db.display_name,
                        "display_name": db.display_name,
                        "database_name": getattr(db, "db_name", None),
                        "id": db.id,
                        "workload_type": db.db_workload,
                        "lifecycle_state": db.lifecycle_state,
                        "cpu_core_count": db.cpu_core_count,
                        "data_storage_size_tb": db.data_storage_size_in_tbs,
                        "is_free_tier": db.is_free_tier,
                        "db_version": db.db_version,
                        "time_created": db.time_created.isoformat() if db.time_created else None,
                        "compartment_id": comp_id,
                    }
                    # Add compute model info if available
                    if hasattr(db, "compute_model"):
                        db_info["compute_model"] = db.compute_model
                    databases.append(db_info)
            except Exception:
                # Skip compartments that fail (permission issues)
                continue

        return json.dumps({
            "type": "autonomous_databases",
            "databases": databases,
            "count": len(databases),
            "compartments_searched": compartments_searched,
            "include_subtree": include_subtree,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "type": "autonomous_databases",
            "error": f"OCI service error: {e.message}",
            "status_code": e.status,
        })
    except Exception as e:
        return json.dumps({
            "type": "autonomous_databases",
            "error": str(e),
        })


async def _list_managed_databases_logic(
    compartment_id: str | None = None,
    include_subtree: bool = False,
    database_type: str | None = None,
    deployment_type: str | None = None,
    limit: int = 50,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Internal logic for listing managed databases."""
    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)
        config = client.base_client.config
        root_compartment = compartment_id or config.get("tenancy")

        # If include_subtree, get all compartments and search each
        compartments_to_search = [{"id": root_compartment, "name": "root"}]

        if include_subtree:
            try:
                identity_client = oci.identity.IdentityClient(config)
                compartments_to_search = await _get_compartments_recursive(
                    identity_client, root_compartment
                )
            except Exception:
                pass  # Fall back to single compartment

        # Fetch databases from all compartments
        all_databases = []
        for comp in compartments_to_search:
            try:
                kwargs: dict[str, Any] = {"compartment_id": comp["id"]}
                if database_type:
                    kwargs["database_type"] = database_type
                if deployment_type:
                    kwargs["deployment_type"] = deployment_type

                response = await _call_oci(client.list_managed_databases, **kwargs)
                items = response.data.items if hasattr(response.data, "items") else []

                for item in items:
                    item._compartment_name = comp["name"]
                all_databases.extend(items)

                # Handle pagination
                while response.has_next_page and len(all_databases) < limit:
                    response = await _call_oci(
                        client.list_managed_databases,
                        **kwargs,
                        page=response.next_page,
                    )
                    items = response.data.items if hasattr(response.data, "items") else []
                    for item in items:
                        item._compartment_name = comp["name"]
                    all_databases.extend(items)

                if len(all_databases) >= limit:
                    break

            except oci.exceptions.ServiceError:
                continue  # Skip inaccessible compartments (404, 403, etc.)
            except Exception:
                continue

        # Format response
        databases = []
        for db in all_databases[:limit]:
            databases.append({
                "id": db.id,
                "name": db.name,
                "database_type": db.database_type,
                "database_sub_type": db.database_sub_type,
                "deployment_type": getattr(db, "deployment_type", None),
                "management_option": getattr(db, "management_option", None),
                "workload_type": getattr(db, "workload_type", None),
                "is_cluster": getattr(db, "is_cluster", False),
                "compartment_id": db.compartment_id,
                "compartment_name": getattr(db, "_compartment_name", None),
                "time_created": str(getattr(db, "time_created", None)),
            })

        return json.dumps({
            "type": "managed_databases",
            "count": len(databases),
            "compartments_searched": len(compartments_to_search),
            "databases": databases,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OCI API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _search_managed_databases_logic(
    name: str,
    compartment_id: str | None = None,
    include_subtree: bool = True,
    limit: int = 10,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Internal logic for searching managed databases by name."""
    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)
        config = client.base_client.config
        root_compartment = compartment_id or config.get("tenancy")

        # Get compartments to search
        compartments_to_search = [{"id": root_compartment, "name": "root"}]
        if include_subtree:
            try:
                identity_client = oci.identity.IdentityClient(config)
                compartments_to_search = await _get_compartments_recursive(
                    identity_client, root_compartment
                )
            except Exception:
                pass

        # Search each compartment
        all_databases = []
        for comp in compartments_to_search:
            try:
                response = await _call_oci(
                    client.list_managed_databases,
                    compartment_id=comp["id"],
                )
                items = response.data.items if hasattr(response.data, "items") else []
                for item in items:
                    item._compartment_name = comp["name"]
                all_databases.extend(items)

                while response.has_next_page:
                    response = await _call_oci(
                        client.list_managed_databases,
                        compartment_id=comp["id"],
                        page=response.next_page,
                    )
                    items = response.data.items if hasattr(response.data, "items") else []
                    for item in items:
                        item._compartment_name = comp["name"]
                    all_databases.extend(items)
            except Exception:
                continue

        # Filter by name (case-insensitive partial match)
        name_lower = name.lower()
        matched = [
            db for db in all_databases
            if name_lower in (db.name or "").lower()
        ]

        # Format response
        databases = []
        for db in matched[:limit]:
            databases.append({
                "id": db.id,
                "name": db.name,
                "database_type": db.database_type,
                "database_sub_type": db.database_sub_type,
                "deployment_type": getattr(db, "deployment_type", None),
                "management_option": getattr(db, "management_option", None),
                "compartment_id": db.compartment_id,
                "compartment_name": getattr(db, "_compartment_name", None),
            })

        return json.dumps({
            "type": "managed_database_search",
            "query": name,
            "count": len(databases),
            "compartments_searched": len(compartments_to_search),
            "databases": databases,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OCI API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_managed_database_logic(
    managed_database_id: str,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get details of a specific managed database."""
    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)
        response = await _call_oci(
            client.get_managed_database,
            managed_database_id=managed_database_id,
        )
        db = response.data

        return json.dumps({
            "type": "managed_database",
            "id": db.id,
            "name": db.name,
            "database_type": db.database_type,
            "database_sub_type": db.database_sub_type,
            "deployment_type": getattr(db, "deployment_type", None),
            "management_option": getattr(db, "management_option", None),
            "workload_type": getattr(db, "workload_type", None),
            "is_cluster": getattr(db, "is_cluster", False),
            "compartment_id": db.compartment_id,
            "db_system_id": getattr(db, "db_system_id", None),
            "storage_system_id": getattr(db, "storage_system_id", None),
            "time_created": str(getattr(db, "time_created", None)),
            "database_version": getattr(db, "database_version", None),
            "database_status": getattr(db, "database_status", None),
            "parent_container_id": getattr(db, "parent_container_id", None),
            "parent_container_name": getattr(db, "parent_container_name", None),
            "parent_container_compartment_id": getattr(db, "parent_container_compartment_id", None),
            "instance_count": getattr(db, "instance_count", None),
            "db_unique_name": getattr(db, "db_unique_name", None),
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OCI API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_awr_db_report_logic(
    managed_database_id: str,
    hours_back: int = 1,
    report_type: str = "AWR",
    report_format: str = "TEXT",
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get AWR/ADDM report for a managed database.

    Uses the Database Management get_awr_db_report API for actual AWR data.
    Supports ADB, External SIDB, and External RAC databases.
    """
    from datetime import datetime, timedelta

    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)

        end_time = datetime.now(UTC)
        begin_time = end_time - timedelta(hours=hours_back)

        try:
            # List AWR DBs to get the awr_db_id and snapshot info
            awr_dbs_response = await _call_oci(
                client.list_awr_dbs,
                managed_database_id=managed_database_id,
            )
            awr_dbs = awr_dbs_response.data.items if hasattr(awr_dbs_response.data, "items") else []

            if not awr_dbs:
                return json.dumps({
                    "error": "No AWR data available for this database",
                    "managed_database_id": managed_database_id,
                    "suggestion": "Ensure AWR is enabled and snapshots are being collected",
                })

            awr_db = awr_dbs[0]
            awr_db_id = awr_db.awr_db_id

            # Get snapshot info from AWR DB summary (works for all DB types)
            first_snapshot_id = getattr(awr_db, "first_snapshot_id", None)
            latest_snapshot_id = getattr(awr_db, "latest_snapshot_id", None)
            snapshot_interval_min = getattr(awr_db, "snapshot_interval_in_min", 60)

            # Calculate how many snapshots back to go based on hours_back
            snapshots_back = max(1, int(hours_back * 60 / snapshot_interval_min))

            # Use the latest snapshots, going back as needed
            if latest_snapshot_id and first_snapshot_id:
                end_snapshot_id = latest_snapshot_id
                # Go back 'snapshots_back' intervals, but not before first_snapshot_id
                begin_snapshot_id = max(
                    first_snapshot_id,
                    latest_snapshot_id - snapshots_back
                )
            else:
                return json.dumps({
                    "error": "AWR snapshot range not available",
                    "managed_database_id": managed_database_id,
                })

            # Get the AWR report
            report_response = await _call_oci(
                client.get_awr_db_report,
                managed_database_id=managed_database_id,
                awr_db_id=awr_db_id,
                begin_sn_id_greater_than_or_equal_to=begin_snapshot_id,
                end_sn_id_less_than_or_equal_to=end_snapshot_id,
                report_type=report_type,
                report_format=oci.database_management.models.AwarReportFormatType.TEXT
                if report_format.upper() == "TEXT"
                else oci.database_management.models.AwarReportFormatType.HTML,
            )

            # Extract report content
            report_content = ""
            if hasattr(report_response.data, "content"):
                report_content = report_response.data.content
            elif hasattr(report_response.data, "report"):
                report_content = report_response.data.report

            # Additional metadata from AWR DB summary
            db_name = getattr(awr_db, "db_name", None)
            db_version = getattr(awr_db, "db_version", None)
            snapshot_count = getattr(awr_db, "snapshot_count", None)

            return json.dumps({
                "type": "awr_report",
                "managed_database_id": managed_database_id,
                "awr_db_id": awr_db_id,
                "db_name": db_name,
                "db_version": db_version,
                "begin_snapshot_id": begin_snapshot_id,
                "end_snapshot_id": end_snapshot_id,
                "snapshots_used": end_snapshot_id - begin_snapshot_id + 1,
                "total_snapshots_available": snapshot_count,
                "snapshot_interval_min": snapshot_interval_min,
                "hours_covered": (end_snapshot_id - begin_snapshot_id) * snapshot_interval_min / 60,
                "report_type": report_type,
                "report_format": report_format,
                "content": report_content[:10000] if report_content else None,
                "truncated": len(report_content) > 10000 if report_content else False,
                "full_content_length": len(report_content) if report_content else 0,
            })

        except oci.exceptions.ServiceError as e:
            return json.dumps({
                "error": f"AWR API error: {e.message}",
                "status": e.status,
                "managed_database_id": managed_database_id,
                "suggestion": "External databases may return errors if AWR Hub is not configured",
            })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OCI API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_top_sql_logic(
    managed_database_id: str,
    hours_back: int = 1,
    limit: int = 10,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get top SQL statements by CPU activity."""
    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)

        response = await _call_oci(
            client.get_top_sql_cpu_activity,
            managed_database_id=managed_database_id,
        )

        activities = []
        if hasattr(response.data, "activities") and response.data.activities:
            for activity in response.data.activities[:limit]:
                activities.append({
                    "sql_id": getattr(activity, "sql_id", None),
                    "activity_percent": getattr(activity, "activity_percent", None),
                    "database_time_pct": getattr(activity, "database_time_pct", None),
                    "cpu_time_pct": getattr(activity, "cpu_time_pct", None),
                    "wait_time_pct": getattr(activity, "wait_time_pct", None),
                    "user_io_time_pct": getattr(activity, "user_io_time_pct", None),
                    "category": getattr(activity, "category", None),
                })

        return json.dumps({
            "type": "top_sql_cpu_activity",
            "managed_database_id": managed_database_id,
            "sql_count": len(activities),
            "activities": activities,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"Top SQL API error: {e.message}",
            "status": e.status,
            "managed_database_id": managed_database_id,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_wait_events_logic(
    managed_database_id: str,
    hours_back: int = 1,
    top_n: int = 10,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get top wait events from AWR data."""
    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)

        # First get AWR DB ID
        awr_dbs_response = await _call_oci(
            client.list_awr_dbs,
            managed_database_id=managed_database_id,
        )
        if not hasattr(awr_dbs_response.data, "items") or not awr_dbs_response.data.items:
            return json.dumps({
                "error": "No AWR data available",
                "managed_database_id": managed_database_id,
            })

        awr_db = awr_dbs_response.data.items[0]
        awr_db_id = awr_db.awr_db_id
        snapshot_interval = getattr(awr_db, "snapshot_interval_in_min", 60)

        # Calculate snapshot range
        snapshots_back = max(1, int(hours_back * 60 / snapshot_interval))
        latest_snap = getattr(awr_db, "latest_snapshot_id", None)
        first_snap = getattr(awr_db, "first_snapshot_id", None)

        if not latest_snap or not first_snap:
            return json.dumps({"error": "No snapshot range available"})

        begin_snap = max(first_snap, latest_snap - snapshots_back)
        end_snap = latest_snap

        # Get top wait events
        response = await _call_oci(
            client.summarize_awr_db_top_wait_events,
            managed_database_id=managed_database_id,
            awr_db_id=awr_db_id,
            begin_sn_id_greater_than_or_equal_to=begin_snap,
            end_sn_id_less_than_or_equal_to=end_snap,
            top_n=top_n,
        )

        wait_events = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                wait_events.append({
                    "name": getattr(item, "name", None),
                    "waits": getattr(item, "waits", None),
                    "time_waited_sec": getattr(item, "time_waited", None),
                    "avg_wait_ms": getattr(item, "average_wait", None),
                    "wait_class": getattr(item, "wait_class", None),
                    "db_time_pct": getattr(item, "db_time_pct", None),
                })

        return json.dumps({
            "type": "top_wait_events",
            "managed_database_id": managed_database_id,
            "awr_db_id": awr_db_id,
            "snapshot_range": f"{begin_snap}-{end_snap}",
            "hours_covered": (end_snap - begin_snap) * snapshot_interval / 60,
            "wait_event_count": len(wait_events),
            "wait_events": wait_events,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"Wait Events API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _list_sql_plan_baselines_logic(
    managed_database_id: str,
    sql_handle: str | None = None,
    plan_name: str | None = None,
    is_enabled: bool | None = None,
    limit: int = 20,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """List SQL Plan Baselines."""
    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)

        kwargs = {"managed_database_id": managed_database_id, "limit": limit}
        if sql_handle:
            kwargs["sql_handle"] = sql_handle
        if plan_name:
            kwargs["plan_name"] = plan_name
        if is_enabled is not None:
            kwargs["is_enabled"] = is_enabled

        response = await _call_oci(client.list_sql_plan_baselines, **kwargs)

        baselines = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                baselines.append({
                    "plan_name": getattr(item, "plan_name", None),
                    "sql_handle": getattr(item, "sql_handle", None),
                    "sql_text": getattr(item, "sql_text", None)[:200] if getattr(item, "sql_text", None) else None,
                    "origin": getattr(item, "origin", None),
                    "enabled": getattr(item, "enabled", None),
                    "accepted": getattr(item, "accepted", None),
                    "fixed": getattr(item, "fixed", None),
                    "reproduced": getattr(item, "reproduced", None),
                    "time_created": str(getattr(item, "time_created", None)),
                    "time_last_executed": str(getattr(item, "time_last_executed", None)),
                    "executions": getattr(item, "executions", None),
                    "elapsed_time_total": getattr(item, "elapsed_time_total", None),
                })

        return json.dumps({
            "type": "sql_plan_baselines",
            "managed_database_id": managed_database_id,
            "baseline_count": len(baselines),
            "baselines": baselines,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"SQL Plan Baselines API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_fleet_health_logic(
    compartment_id: str | None = None,
    include_subtree: bool = True,
    compare_type: str = "WEEK",
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get database fleet health metrics."""
    from datetime import datetime, timedelta

    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        # Calculate baseline and target times based on compare_type
        now = datetime.now(UTC)
        if compare_type == "WEEK":
            baseline_time = now - timedelta(weeks=1)
        elif compare_type == "MONTH":
            baseline_time = now - timedelta(days=30)
        else:
            baseline_time = now - timedelta(weeks=1)

        # Format as ISO 8601 with milliseconds (YYYY-MM-DDThh:mm:ss.SSSZ)
        baseline_str = baseline_time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        target_str = now.strftime("%Y-%m-%dT%H:%M:%S.000Z")

        # Wrap synchronous OCI SDK call in thread pool to prevent blocking event loop
        response = await _call_oci(
            client.get_database_fleet_health_metrics,
            compare_baseline_time=baseline_str,
            compare_target_time=target_str,
            managed_database_group_id=None,
            compartment_id=compartment,
            compare_type=compare_type,
        )

        data = response.data
        fleet_summary = {
            "database_count": getattr(data, "database_count", None),
            "healthy_count": getattr(data, "healthy_count", None),
            "warning_count": getattr(data, "warning_count", None),
            "critical_count": getattr(data, "critical_count", None),
            "unavailable_count": getattr(data, "unavailable_count", None),
        }

        # Get fleet statistics
        statistics = []
        if hasattr(data, "fleet_status_by_category"):
            for stat in data.fleet_status_by_category:
                statistics.append({
                    "category": getattr(stat, "category", None),
                    "total_count": getattr(stat, "total_count", None),
                    "healthy_count": getattr(stat, "healthy_count", None),
                    "warning_count": getattr(stat, "warning_count", None),
                })

        return json.dumps({
            "type": "fleet_health",
            "compartment_id": compartment,
            "compare_type": compare_type,
            "summary": fleet_summary,
            "statistics": statistics,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"Fleet Health API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_awr_sql_report_logic(
    managed_database_id: str,
    sql_id: str,
    hours_back: int = 24,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get detailed AWR SQL report for a specific SQL ID."""
    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)

        # Get AWR DB info
        awr_dbs_response = await _call_oci(
            client.list_awr_dbs,
            managed_database_id=managed_database_id,
        )
        if not hasattr(awr_dbs_response.data, "items") or not awr_dbs_response.data.items:
            return json.dumps({"error": "No AWR data available"})

        awr_db = awr_dbs_response.data.items[0]
        awr_db_id = awr_db.awr_db_id
        snapshot_interval = getattr(awr_db, "snapshot_interval_in_min", 60)

        # Calculate snapshot range
        snapshots_back = max(1, int(hours_back * 60 / snapshot_interval))
        latest_snap = getattr(awr_db, "latest_snapshot_id", None)
        first_snap = getattr(awr_db, "first_snapshot_id", None)

        if not latest_snap or not first_snap:
            return json.dumps({"error": "No snapshot range available"})

        begin_snap = max(first_snap, latest_snap - snapshots_back)
        end_snap = latest_snap

        # Get SQL report
        response = await _call_oci(
            client.get_awr_db_sql_report,
            managed_database_id=managed_database_id,
            awr_db_id=awr_db_id,
            sql_id=sql_id,
            begin_sn_id_greater_than_or_equal_to=begin_snap,
            end_sn_id_less_than_or_equal_to=end_snap,
            report_format=oci.database_management.models.AwarReportFormatType.TEXT,
        )

        report_content = ""
        if hasattr(response.data, "content"):
            report_content = response.data.content

        return json.dumps({
            "type": "awr_sql_report",
            "managed_database_id": managed_database_id,
            "sql_id": sql_id,
            "awr_db_id": awr_db_id,
            "snapshot_range": f"{begin_snap}-{end_snap}",
            "hours_covered": (end_snap - begin_snap) * snapshot_interval / 60,
            "content": report_content[:10000] if report_content else None,
            "truncated": len(report_content) > 10000 if report_content else False,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"AWR SQL Report API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_database_metrics_logic(
    managed_database_id: str,
    metric_names: list[str] | None = None,
    hours_back: int = 1,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get performance metrics for a managed database."""
    from datetime import datetime, timedelta

    import oci

    try:
        client = get_database_management_client(profile=profile, region=region)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=hours_back)

        # Default metrics to retrieve
        if not metric_names:
            metric_names = [
                "CPU_UTILIZATION",
                "STORAGE_UTILIZATION",
                "IO_THROUGHPUT",
                "ACTIVE_SESSIONS",
            ]

        # Get database performance metrics
        try:
            response = await _call_oci(
                client.summarize_awr_db_metrics,
                managed_database_id=managed_database_id,
                name=metric_names,
                begin_time=start_time.isoformat(),
                end_time=end_time.isoformat(),
            )

            metrics_data = []
            if hasattr(response.data, "items"):
                for item in response.data.items:
                    metrics_data.append({
                        "name": getattr(item, "name", None),
                        "timestamp": str(getattr(item, "timestamp", None)),
                        "value": getattr(item, "value", None),
                        "unit": getattr(item, "unit", None),
                    })

            return json.dumps({
                "type": "database_metrics",
                "managed_database_id": managed_database_id,
                "time_range": f"{start_time.isoformat()} to {end_time.isoformat()}",
                "metric_count": len(metrics_data),
                "metrics": metrics_data,
            })

        except oci.exceptions.ServiceError as e:
            # Fall back to basic database info if metrics not available
            return json.dumps({
                "error": f"Metrics API error: {e.message}",
                "status": e.status,
                "suggestion": "This database may not have metrics enabled. Try get_managed_database for basic info.",
            })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OCI API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _parse_tnsnames(tns_admin: str) -> list[str]:
    """Parse tnsnames.ora to extract connection aliases.

    Args:
        tns_admin: Path to TNS_ADMIN directory containing tnsnames.ora

    Returns:
        List of connection alias names
    """
    import os
    import re

    aliases = []
    tns_file = os.path.join(os.path.expanduser(tns_admin), "tnsnames.ora")

    if not os.path.isfile(tns_file):
        return aliases

    try:
        with open(tns_file, 'r') as f:
            content = f.read()

        # Parse TNS aliases - they appear at the start of lines followed by "="
        # Pattern: ALIAS_NAME = (DESCRIPTION = ...)
        pattern = r'^([A-Za-z][A-Za-z0-9_]*)\s*='
        for match in re.finditer(pattern, content, re.MULTILINE):
            alias = match.group(1).strip()
            if alias.upper() not in ('DESCRIPTION', 'ADDRESS', 'ADDRESS_LIST',
                                      'CONNECT_DATA', 'SERVICE_NAME', 'SID',
                                      'SECURITY', 'SSL_SERVER_CERT_DN'):
                aliases.append(alias)
    except Exception:
        pass

    return aliases


async def _get_database_name_for_connection(
    connection_name: str,
    sqlcl_path: str,
    sqlcl_tns_admin: str,
    sqlcl_username: str,
    sqlcl_password: str,
    timeout_seconds: int = 10,
) -> str | None:
    """Query the database name for a SQLcl connection.

    Executes SELECT NAME FROM V$DATABASE to get the actual database name.

    Args:
        connection_name: TNS alias to connect to
        sqlcl_path: Path to SQLcl executable
        sqlcl_tns_admin: TNS_ADMIN directory path
        sqlcl_username: Database username
        sqlcl_password: Database password
        timeout_seconds: Query timeout (default 10s for quick check)

    Returns:
        Database name or None if query fails
    """
    import os
    import subprocess

    if not sqlcl_password:
        return None

    try:
        sqlcl_path_expanded = os.path.expanduser(sqlcl_path)
        tns_admin_expanded = os.path.expanduser(sqlcl_tns_admin)

        # Quick query to get database name
        sql_script = """
SET PAGESIZE 0
SET LINESIZE 200
SET FEEDBACK OFF
SET HEADING OFF
SELECT NAME FROM V$DATABASE;
EXIT;
"""
        conn_string = f"{sqlcl_username}/{sqlcl_password}@{connection_name}"
        env = os.environ.copy()
        env["TNS_ADMIN"] = tns_admin_expanded

        result = await asyncio.to_thread(
            subprocess.run,
            [sqlcl_path_expanded, "-S", conn_string],
            input=sql_script,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
        )

        if result.returncode == 0 and result.stdout.strip():
            # Extract database name from output (handle leading/trailing whitespace)
            lines = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            if lines:
                db_name = lines[0]
                if db_name and not db_name.startswith(('ORA-', 'SP2-', 'Error', 'WARNING')):
                    return db_name

    except Exception:
        pass  # Silently fail - database name is optional

    return None


async def _list_database_connections_logic(include_db_names: bool = True) -> str:
    """Internal logic for listing available database connections.

    Checks environment variables for ATP wallet-based connections,
    SQLcl CLI availability, and parses tnsnames.ora for all available aliases.

    Args:
        include_db_names: If True, query each connection for its database name
    """
    import os

    connections = []

    # Check SQLcl CLI availability first
    sqlcl_path = os.environ.get("SQLCL_PATH", "/usr/bin/sql")
    sqlcl_exists = os.path.isfile(os.path.expanduser(sqlcl_path))

    # Get SQLcl TNS admin directory for connection aliases
    sqlcl_tns_admin = os.environ.get("SQLCL_TNS_ADMIN")
    sqlcl_default_connection = os.environ.get("SQLCL_DB_CONNECTION")
    sqlcl_fallback_connection = os.environ.get("SQLCL_FALLBACK_CONNECTION")
    sqlcl_username = os.environ.get("SQLCL_DB_USERNAME", "ADMIN")
    sqlcl_password = os.environ.get("SQLCL_DB_PASSWORD")

    # Parse tnsnames.ora from SQLcl wallet if available
    sqlcl_aliases = []
    if sqlcl_tns_admin and os.path.isdir(os.path.expanduser(sqlcl_tns_admin)):
        sqlcl_aliases = _parse_tnsnames(sqlcl_tns_admin)

    # Add SQLcl connections from tnsnames.ora
    if sqlcl_exists and sqlcl_aliases:
        for alias in sqlcl_aliases:
            is_default = alias == sqlcl_default_connection
            is_fallback = alias == sqlcl_fallback_connection
            connections.append({
                "name": alias,
                "type": "sqlcl_tns",
                "tns_alias": alias,
                "tns_admin": sqlcl_tns_admin,
                "user": sqlcl_username,
                "status": "available",
                "is_default": is_default,
                "is_fallback": is_fallback,
                "description": f"SQLcl connection via TNS alias{' (default)' if is_default else ''}{' (fallback)' if is_fallback else ''}",
            })
    elif sqlcl_exists:
        # SQLcl available but no tnsnames.ora found - add generic entry
        connections.append({
            "name": "sqlcl",
            "type": "sqlcl_cli",
            "path": sqlcl_path,
            "status": "available",
            "description": "SQLcl CLI (configure SQLCL_TNS_ADMIN for connection aliases)",
        })
    else:
        connections.append({
            "name": "sqlcl",
            "type": "sqlcl_cli",
            "path": sqlcl_path,
            "status": "not_found",
            "description": "SQLcl CLI (not installed at configured path)",
        })

    # Check for ATP wallet-based connection (oracledb) - for programmatic access
    atp_tns = os.environ.get("ATP_TNS_NAME")
    atp_user = os.environ.get("ATP_USER")
    atp_wallet_dir = os.environ.get("ATP_WALLET_DIR")

    if atp_tns and atp_user and atp_wallet_dir:
        wallet_exists = os.path.isdir(os.path.expanduser(atp_wallet_dir))
        # Only add if not already in SQLcl aliases
        if atp_tns not in [c.get("tns_alias") for c in connections]:
            connections.append({
                "name": "atp_wallet",
                "type": "oracledb_wallet",
                "tns_name": atp_tns,
                "user": atp_user,
                "wallet_dir": atp_wallet_dir,
                "status": "configured" if wallet_exists else "wallet_not_found",
                "description": "ATP connection via wallet (oracledb)",
            })

    # Query database names for available connections (in parallel for speed)
    if include_db_names and sqlcl_exists and sqlcl_tns_admin and sqlcl_password:
        # Get connections that can be queried (sqlcl_tns type with available status)
        queryable_connections = [
            c for c in connections
            if c.get("type") == "sqlcl_tns" and c.get("status") == "available"
        ]

        if queryable_connections:
            # Create tasks for parallel execution
            async def get_db_name_for_conn(conn: dict) -> tuple[str, str | None]:
                conn_name = conn.get("tns_alias") or conn.get("name")
                db_name = await _get_database_name_for_connection(
                    connection_name=conn_name,
                    sqlcl_path=sqlcl_path,
                    sqlcl_tns_admin=sqlcl_tns_admin,
                    sqlcl_username=sqlcl_username,
                    sqlcl_password=sqlcl_password,
                    timeout_seconds=10,
                )
                return (conn_name, db_name)

            # Execute all queries in parallel
            try:
                results = await asyncio.gather(
                    *[get_db_name_for_conn(c) for c in queryable_connections],
                    return_exceptions=True,
                )

                # Build a map of connection name -> database name
                db_name_map = {}
                for result in results:
                    if isinstance(result, tuple) and result[1]:
                        db_name_map[result[0]] = result[1]

                # Update connections with database names
                for conn in connections:
                    conn_name = conn.get("tns_alias") or conn.get("name")
                    if conn_name in db_name_map:
                        conn["database_name"] = db_name_map[conn_name]

            except Exception:
                pass  # Don't fail if db name queries fail

    # Default connection preference
    default_connection = None
    # First check for explicitly configured SQLcl default
    if sqlcl_default_connection and any(
        c.get("tns_alias") == sqlcl_default_connection and c["status"] == "available"
        for c in connections
    ):
        default_connection = sqlcl_default_connection
    elif any(c["name"] == "atp_wallet" and c["status"] == "configured" for c in connections):
        default_connection = "atp_wallet"
    elif sqlcl_exists and sqlcl_aliases:
        default_connection = sqlcl_aliases[0]
    elif any(c["name"] == "sqlcl" and c["status"] == "available" for c in connections):
        default_connection = "sqlcl"

    available_count = len([c for c in connections if c["status"] in ("configured", "available")])

    return json.dumps({
        "type": "database_connections",
        "connections": connections,
        "default_connection": default_connection,
        "count": available_count,
        "sqlcl_available": sqlcl_exists,
        "tns_admin": sqlcl_tns_admin if sqlcl_exists else None,
    })


async def _execute_sql_logic(
    sql: str,
    connection_name: str | None = None,
    timeout_seconds: int = 60,
) -> str:
    """Internal logic for SQL execution.

    Supports three methods:
    1. oracledb with wallet (for ATP) - preferred for Autonomous DBs
    2. SQLcl CLI with TNS alias - for connections from tnsnames.ora
    3. SQLcl CLI subprocess - fallback for any Oracle DB
    """
    import os
    import subprocess

    # Sanitize SQL input
    sql = sql.strip()
    if not sql:
        return json.dumps({"error": "Empty SQL statement"})

    # Determine connection method
    atp_tns = os.environ.get("ATP_TNS_NAME")
    atp_user = os.environ.get("ATP_USER")
    atp_password = os.environ.get("ATP_PASSWORD")
    atp_wallet_dir = os.environ.get("ATP_WALLET_DIR")
    sqlcl_path = os.environ.get("SQLCL_PATH", "/usr/bin/sql")
    sqlcl_tns_admin = os.environ.get("SQLCL_TNS_ADMIN")
    sqlcl_username = os.environ.get("SQLCL_DB_USERNAME", "ADMIN")
    sqlcl_password = os.environ.get("SQLCL_DB_PASSWORD")
    sqlcl_default_connection = os.environ.get("SQLCL_DB_CONNECTION")

    use_oracledb = False
    use_sqlcl = False
    use_tns_alias = False
    tns_alias = None

    # Check if connection_name is a TNS alias from tnsnames.ora
    if connection_name and connection_name not in ("atp_wallet", "sqlcl"):
        # Assume it's a TNS alias
        if sqlcl_tns_admin and os.path.isdir(os.path.expanduser(sqlcl_tns_admin)):
            sqlcl_aliases = _parse_tnsnames(sqlcl_tns_admin)
            if connection_name in sqlcl_aliases:
                use_tns_alias = True
                tns_alias = connection_name

    if not use_tns_alias:
        if connection_name == "atp_wallet" or (connection_name is None and atp_tns):
            # Prefer oracledb with wallet if ATP is configured
            if atp_tns and atp_user and atp_password and atp_wallet_dir:
                use_oracledb = True
        elif connection_name == "sqlcl" or connection_name is None:
            if os.path.isfile(os.path.expanduser(sqlcl_path)):
                # Use default SQLcl connection if available
                if sqlcl_default_connection and sqlcl_tns_admin:
                    use_tns_alias = True
                    tns_alias = sqlcl_default_connection
                else:
                    use_sqlcl = True

    if not use_oracledb and not use_sqlcl and not use_tns_alias:
        return json.dumps({
            "error": "No database connection available",
            "suggestion": "Configure ATP_* environment variables for wallet connection or SQLCL_* for CLI",
        })

    # Execute with oracledb (wallet-based)
    if use_oracledb:
        try:
            import oracledb

            # Initialize thick mode for wallet support if not already done
            try:
                oracledb.init_oracle_client()
            except Exception:
                pass  # May already be initialized

            wallet_dir = os.path.expanduser(atp_wallet_dir)

            # Connect using wallet
            connection = await asyncio.to_thread(
                oracledb.connect,
                user=atp_user,
                password=atp_password,
                dsn=atp_tns,
                config_dir=wallet_dir,
                wallet_location=wallet_dir,
                wallet_password=os.environ.get("ATP_WALLET_PASSWORD"),
            )

            try:
                cursor = connection.cursor()

                # Execute the query
                await asyncio.to_thread(cursor.execute, sql)

                # Check if it's a SELECT query
                if sql.strip().upper().startswith("SELECT"):
                    columns = [col[0] for col in cursor.description] if cursor.description else []
                    rows = await asyncio.to_thread(cursor.fetchall)

                    # Convert to list of dicts
                    data = [dict(zip(columns, row)) for row in rows]

                    return json.dumps({
                        "type": "sql_result",
                        "connection": "atp_wallet",
                        "columns": columns,
                        "data": data,
                        "row_count": len(data),
                    })
                else:
                    # DML or DDL
                    row_count = cursor.rowcount
                    await asyncio.to_thread(connection.commit)

                    return json.dumps({
                        "type": "sql_result",
                        "connection": "atp_wallet",
                        "message": f"Statement executed successfully. Rows affected: {row_count}",
                        "row_count": row_count,
                    })

            finally:
                cursor.close()
                connection.close()

        except ImportError:
            # oracledb not installed, fall through to SQLcl
            if os.path.isfile(os.path.expanduser(sqlcl_path)):
                use_sqlcl = True
                use_oracledb = False
            else:
                return json.dumps({
                    "error": "oracledb module not installed and SQLcl not available",
                    "suggestion": "Install oracledb: pip install oracledb",
                })
        except Exception as e:
            return json.dumps({
                "type": "sql_error",
                "connection": "atp_wallet",
                "error": str(e),
            })

    # Execute with SQLcl CLI using TNS alias
    if use_tns_alias:
        try:
            sqlcl_path_expanded = os.path.expanduser(sqlcl_path)

            # Prepare SQL script (add SET commands for clean JSON output)
            sql_script = f"""
SET PAGESIZE 0
SET LINESIZE 32767
SET FEEDBACK OFF
SET HEADING ON
SET SQLFORMAT JSON
{sql}
EXIT;
"""
            # Build connection string using TNS alias
            tns_admin_expanded = os.path.expanduser(sqlcl_tns_admin)
            conn_string = f"{sqlcl_username}/{sqlcl_password}@{tns_alias}"
            env = os.environ.copy()
            env["TNS_ADMIN"] = tns_admin_expanded

            # Execute SQLcl
            result = await asyncio.to_thread(
                subprocess.run,
                [sqlcl_path_expanded, "-S", conn_string],
                input=sql_script,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )

            if result.returncode != 0:
                return json.dumps({
                    "type": "sql_error",
                    "connection": tns_alias,
                    "error": result.stderr or result.stdout,
                    "return_code": result.returncode,
                })

            # Parse SQLcl JSON output
            output = result.stdout.strip()

            try:
                parsed = json.loads(output)
                if isinstance(parsed, dict) and "results" in parsed:
                    results = parsed["results"]
                    if results and isinstance(results, list):
                        first_result = results[0]
                        if "items" in first_result:
                            data = first_result["items"]
                            columns = list(data[0].keys()) if data else []
                            return json.dumps({
                                "type": "sql_result",
                                "connection": tns_alias,
                                "columns": columns,
                                "data": data,
                                "row_count": len(data),
                            })
                return json.dumps({
                    "type": "sql_result",
                    "connection": tns_alias,
                    "data": parsed,
                })
            except json.JSONDecodeError:
                return json.dumps({
                    "type": "sql_result",
                    "connection": tns_alias,
                    "raw_output": output,
                })

        except subprocess.TimeoutExpired:
            return json.dumps({
                "type": "sql_error",
                "connection": tns_alias,
                "error": f"Query timed out after {timeout_seconds} seconds",
            })
        except Exception as e:
            return json.dumps({
                "type": "sql_error",
                "connection": tns_alias,
                "error": str(e),
            })

    # Execute with SQLcl CLI (legacy/fallback)
    if use_sqlcl:
        try:
            # Build SQLcl command
            sqlcl_path_expanded = os.path.expanduser(sqlcl_path)

            # Prepare SQL script (add SET commands for clean JSON output)
            sql_script = f"""
SET PAGESIZE 0
SET LINESIZE 32767
SET FEEDBACK OFF
SET HEADING ON
SET SQLFORMAT JSON
{sql}
EXIT;
"""
            # Build connection string
            if atp_tns and atp_user and atp_password and atp_wallet_dir:
                # Use wallet for ATP
                wallet_dir = os.path.expanduser(atp_wallet_dir)
                conn_string = f"{atp_user}/{atp_password}@{atp_tns}"
                env = os.environ.copy()
                env["TNS_ADMIN"] = wallet_dir
            else:
                # Use environment-based connection or default
                conn_string = os.environ.get("ORACLE_CONN_STRING", "/as sysdba")
                env = os.environ.copy()

            # Execute SQLcl
            result = await asyncio.to_thread(
                subprocess.run,
                [sqlcl_path_expanded, "-S", conn_string],
                input=sql_script,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                env=env,
            )

            if result.returncode != 0:
                return json.dumps({
                    "type": "sql_error",
                    "connection": "sqlcl",
                    "error": result.stderr or result.stdout,
                    "return_code": result.returncode,
                })

            # Parse SQLcl JSON output
            output = result.stdout.strip()

            # Try to parse as JSON
            try:
                # SQLcl JSON format wraps results in {"results":[...]}
                parsed = json.loads(output)
                if isinstance(parsed, dict) and "results" in parsed:
                    results = parsed["results"]
                    if results and isinstance(results, list):
                        first_result = results[0]
                        if "items" in first_result:
                            data = first_result["items"]
                            columns = list(data[0].keys()) if data else []
                            return json.dumps({
                                "type": "sql_result",
                                "connection": "sqlcl",
                                "columns": columns,
                                "data": data,
                                "row_count": len(data),
                            })
                return json.dumps({
                    "type": "sql_result",
                    "connection": "sqlcl",
                    "data": parsed,
                })
            except json.JSONDecodeError:
                # Return raw output if not JSON
                return json.dumps({
                    "type": "sql_result",
                    "connection": "sqlcl",
                    "raw_output": output,
                    "message": "Query executed (raw output)",
                })

        except subprocess.TimeoutExpired:
            return json.dumps({
                "type": "sql_error",
                "connection": "sqlcl",
                "error": f"Query timed out after {timeout_seconds} seconds",
            })
        except Exception as e:
            return json.dumps({
                "type": "sql_error",
                "connection": "sqlcl",
                "error": str(e),
            })

    return json.dumps({"error": "No execution path available"})


def register_database_tools(mcp: FastMCP) -> None:
    """Register database management tools with the MCP server."""

    @mcp.tool()
    async def oci_dbmgmt_list_databases(
        compartment_id: str | None = None,
        include_subtree: bool = True,
        database_type: str | None = None,
        deployment_type: str | None = None,
        limit: int = 50,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """List managed databases registered with DB Management.

        These are databases enabled for DB Management features like
        performance monitoring, AWR reports, SQL tuning, etc.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy root)
            include_subtree: Search all sub-compartments recursively (default True)
            database_type: Filter by type (EXTERNAL_SIDB, EXTERNAL_RAC, CLOUD_SIDB, etc.)
            deployment_type: Filter by deployment (ONPREMISE, BM, VM, EXADATA, etc.)
            limit: Maximum databases to return (default 50)
            profile: OCI profile name
            region: OCI region override

        Returns:
            List of managed databases with details
        """
        return await _list_managed_databases_logic(
            compartment_id, include_subtree, database_type, deployment_type, limit, profile, region
        )

    @mcp.tool()
    async def oci_dbmgmt_search_databases(
        name: str,
        compartment_id: str | None = None,
        include_subtree: bool = True,
        limit: int = 10,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Search managed databases by name across all compartments.

        Performs case-insensitive partial match on database name.
        By default, searches all sub-compartments recursively.

        Args:
            name: Database name to search for (partial match)
            compartment_id: Starting compartment (defaults to tenancy root)
            include_subtree: Search all sub-compartments (default True)
            limit: Maximum results to return (default 10)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Matching managed databases
        """
        return await _search_managed_databases_logic(
            name, compartment_id, include_subtree, limit, profile, region
        )

    @mcp.tool()
    async def oci_dbmgmt_get_database(
        managed_database_id: str,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get detailed information about a managed database.

        Args:
            managed_database_id: OCID of the managed database
            profile: OCI profile name
            region: OCI region override

        Returns:
            Detailed database information including version, status, etc.
        """
        return await _get_managed_database_logic(
            managed_database_id, profile, region
        )

    @mcp.tool()
    async def oci_dbmgmt_get_awr_report(
        managed_database_id: str,
        hours_back: int = 1,
        report_type: str = "AWR",
        report_format: str = "TEXT",
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get AWR (Automatic Workload Repository) report for a managed database.

        AWR reports provide detailed performance analysis including:
        - Top SQL by elapsed time, CPU, I/O
        - Wait events analysis
        - Instance efficiency metrics
        - I/O statistics

        Args:
            managed_database_id: OCID of the managed database
            hours_back: Number of hours to analyze (default 1)
            report_type: AWR or ADDM (default AWR)
            report_format: TEXT or HTML (default TEXT)
            profile: OCI profile name
            region: OCI region override

        Returns:
            AWR report content
        """
        return await _get_awr_db_report_logic(
            managed_database_id=managed_database_id,
            hours_back=hours_back,
            report_type=report_type,
            report_format=report_format,
            profile=profile,
            region=region,
        )

    @mcp.tool()
    async def oci_dbmgmt_get_metrics(
        managed_database_id: str,
        metric_names: str | None = None,
        hours_back: int = 1,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get performance metrics for a managed database.

        Retrieves CPU, storage, I/O, and session metrics.

        Args:
            managed_database_id: OCID of the managed database
            metric_names: Comma-separated metric names (default: CPU,STORAGE,IO,SESSIONS)
            hours_back: Number of hours to retrieve (default 1)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Performance metrics data
        """
        metrics_list = None
        if metric_names:
            metrics_list = [m.strip() for m in metric_names.split(",")]

        return await _get_database_metrics_logic(
            managed_database_id=managed_database_id,
            metric_names=metrics_list,
            hours_back=hours_back,
            profile=profile,
            region=region,
        )

    @mcp.tool()
    async def oci_dbmgmt_get_top_sql(
        managed_database_id: str,
        hours_back: int = 1,
        limit: int = 10,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get top SQL statements by CPU activity for a managed database.

        Identifies resource-intensive queries that may need tuning.

        Args:
            managed_database_id: OCID of the managed database
            hours_back: Time range in hours (default 1)
            limit: Maximum SQL statements to return (default 10)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Top SQL statements with CPU metrics
        """
        return await _get_top_sql_logic(
            managed_database_id, hours_back, limit, profile, region
        )

    @mcp.tool()
    async def oci_dbmgmt_get_wait_events(
        managed_database_id: str,
        hours_back: int = 1,
        top_n: int = 10,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get top wait events for a managed database from AWR data.

        Wait events help identify performance bottlenecks like I/O waits,
        lock contention, network waits, etc.

        Args:
            managed_database_id: OCID of the managed database
            hours_back: Time range in hours (default 1)
            top_n: Number of top wait events to return (default 10)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Top wait events with statistics
        """
        return await _get_wait_events_logic(
            managed_database_id, hours_back, top_n, profile, region
        )

    @mcp.tool()
    async def oci_dbmgmt_list_sql_plan_baselines(
        managed_database_id: str,
        sql_handle: str | None = None,
        plan_name: str | None = None,
        is_enabled: bool | None = None,
        limit: int = 20,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """List SQL Plan Baselines for a managed database.

        SQL Plan Baselines provide plan stability by locking execution plans.

        Args:
            managed_database_id: OCID of the managed database
            sql_handle: Filter by SQL handle
            plan_name: Filter by plan name
            is_enabled: Filter by enabled status
            limit: Maximum baselines to return (default 20)
            profile: OCI profile name
            region: OCI region override

        Returns:
            List of SQL Plan Baselines
        """
        return await _list_sql_plan_baselines_logic(
            managed_database_id, sql_handle, plan_name, is_enabled, limit, profile, region
        )

    @mcp.tool()
    async def oci_dbmgmt_get_fleet_health(
        compartment_id: str | None = None,
        include_subtree: bool = True,
        compare_type: str = "WEEK",
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get database fleet health metrics across compartments.

        Provides health summary including availability, performance,
        and compliance status for all managed databases.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            include_subtree: Include sub-compartments (default True)
            compare_type: Comparison period (HOUR, DAY, WEEK, default WEEK)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Fleet health summary with statistics
        """
        return await _get_fleet_health_logic(
            compartment_id, include_subtree, compare_type, profile, region
        )

    @mcp.tool()
    async def oci_dbmgmt_get_sql_report(
        managed_database_id: str,
        sql_id: str,
        hours_back: int = 24,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get detailed AWR SQL report for a specific SQL ID.

        Provides execution statistics, plan details, and performance
        metrics for a particular SQL statement.

        Args:
            managed_database_id: OCID of the managed database
            sql_id: SQL ID to analyze
            hours_back: Time range in hours (default 24)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Detailed SQL report with execution statistics
        """
        return await _get_awr_sql_report_logic(
            managed_database_id, sql_id, hours_back, profile, region
        )

    @mcp.tool()
    async def oci_db_list_autonomous(
        compartment_id: str | None = None,
        workload_type: str | None = None,
        lifecycle_state: str | None = None,
        limit: int = 50,
        profile: str | None = None,
        region: str | None = None,
        include_subtree: bool = True,
    ) -> str:
        """List Autonomous Databases in a compartment.

        Lists all autonomous databases (ATP, ADW, AJD, APEX) using
        the OCI Database service API. Searches child compartments by default.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy root)
            workload_type: Filter by workload type (ATP, ADW, APEX, AJD, OLTP, DW)
            lifecycle_state: Filter by state (AVAILABLE, PROVISIONING, STOPPED, etc.)
            limit: Maximum databases to return (default 50)
            profile: OCI profile name
            region: OCI region override
            include_subtree: Search child compartments (default True)

        Returns:
            List of autonomous databases with details
        """
        return await _list_autonomous_databases_logic(
            compartment_id, workload_type, lifecycle_state, limit, profile, region, include_subtree
        )

    # ═══════════════════════════════════════════════════════════════════════════
    # SQLcl Execution Tools - For direct SQL execution against Oracle databases
    # ═══════════════════════════════════════════════════════════════════════════

    @mcp.tool()
    async def oci_database_list_connections() -> str:
        """List available database connections configured in the environment.

        Returns connection configurations from environment variables and wallet settings.
        Connections can be used with oci_database_execute_sql.

        Returns:
            List of available database connections with their configuration status
        """
        return await _list_database_connections_logic()

    @mcp.tool()
    async def oci_database_execute_sql(
        sql: str,
        connection_name: str | None = None,
        timeout_seconds: int = 60,
    ) -> str:
        """Execute SQL query against an Oracle database using SQLcl or oracledb.

        Supports two connection methods:
        1. Wallet-based (oracledb): For Autonomous Database with ATP_* env vars
        2. SQLcl CLI: For any Oracle database with SQLCL_PATH configured

        Args:
            sql: SQL statement to execute
            connection_name: Database connection name (from oci_database_list_connections)
                           If not specified, uses default configured connection
            timeout_seconds: Maximum execution time (default 60)

        Returns:
            Query results in JSON format or error message
        """
        return await _execute_sql_logic(sql, connection_name, timeout_seconds)
