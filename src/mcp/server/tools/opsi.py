"""
MCP Tools for OCI Operations Insights (OPSI) Service.

Provides tools for:
- Database Insights and resource statistics
- SQL Insights and performance analysis
- AWR Hub for centralized AWR data
- ADDM findings and recommendations
- Capacity planning and forecasting
"""

import json
from typing import Any

from fastmcp import FastMCP

from src.mcp.server.auth import get_opsi_client


def _compute_insights_summary(insights: list[dict]) -> dict:
    """Pre-compute summary statistics for database insights (BlinkOps pattern: result distillation)."""
    if not insights:
        return {"summary": "No database insights found.", "health_status": "unknown"}

    enabled = sum(1 for i in insights if i.get("status") == "ENABLED")
    disabled = sum(1 for i in insights if i.get("status") == "DISABLED")
    by_type = {}
    for i in insights:
        db_type = i.get("database_type", "unknown")
        by_type[db_type] = by_type.get(db_type, 0) + 1

    health = "healthy" if enabled > 0 and disabled == 0 else "degraded" if disabled > 0 else "unknown"

    return {
        "summary": f"{len(insights)} databases monitored: {enabled} enabled, {disabled} disabled",
        "health_status": health,
        "enabled_count": enabled,
        "disabled_count": disabled,
        "by_type": by_type,
        "next_action": "Check disabled databases" if disabled > 0 else "All databases healthy",
    }


async def _list_database_insights_logic(
    compartment_id: str | None = None,
    include_subtree: bool = True,
    database_type: str | None = None,
    status: str | None = None,
    limit: int = 20,  # Reduced from 50 to minimize context usage
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """List database insights registered with OPSI."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        kwargs: dict[str, Any] = {
            "compartment_id": compartment,
            "compartment_id_in_subtree": include_subtree,
            "limit": limit,
        }
        if database_type:
            kwargs["database_type"] = [database_type]
        if status:
            kwargs["status"] = [status]

        response = client.list_database_insights(**kwargs)

        insights = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                insights.append({
                    "id": getattr(item, "id", None),
                    "database_id": getattr(item, "database_id", None),
                    "database_name": getattr(item, "database_name", None),
                    "database_display_name": getattr(item, "database_display_name", None),
                    "database_type": getattr(item, "database_type", None),
                    "database_version": getattr(item, "database_version", None),
                    "status": getattr(item, "status", None),
                    "lifecycle_state": getattr(item, "lifecycle_state", None),
                    "compartment_id": getattr(item, "compartment_id", None),
                })

        # Pre-computed summary (BlinkOps pattern: result distillation)
        distilled = _compute_insights_summary(insights)

        return json.dumps({
            "type": "database_insights",
            "distilled_summary": distilled,  # LLM reads this first
            "compartment_id": compartment,
            "include_subtree": include_subtree,
            "insight_count": len(insights),
            "insights": insights,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OPSI API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _get_database_insight_logic(
    database_insight_id: str,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get detailed database insight information."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        response = client.get_database_insight(database_insight_id=database_insight_id)

        data = response.data
        return json.dumps({
            "type": "database_insight_detail",
            "id": getattr(data, "id", None),
            "database_id": getattr(data, "database_id", None),
            "database_name": getattr(data, "database_name", None),
            "database_display_name": getattr(data, "database_display_name", None),
            "database_type": getattr(data, "database_type", None),
            "database_version": getattr(data, "database_version", None),
            "status": getattr(data, "status", None),
            "lifecycle_state": getattr(data, "lifecycle_state", None),
            "compartment_id": getattr(data, "compartment_id", None),
            "freeform_tags": getattr(data, "freeform_tags", {}),
            "defined_tags": getattr(data, "defined_tags", {}),
            "time_created": str(getattr(data, "time_created", None)),
            "time_updated": str(getattr(data, "time_updated", None)),
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OPSI API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _summarize_database_resource_statistics_logic(
    compartment_id: str | None = None,
    resource_metric: str = "CPU",
    include_subtree: bool = True,
    percentile: int = 90,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Summarize database resource statistics across the fleet."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        response = client.summarize_database_insight_resource_statistics(
            compartment_id=compartment,
            resource_metric=resource_metric,
            compartment_id_in_subtree=include_subtree,
            percentile=percentile,
        )

        items = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                items.append({
                    "database_display_name": getattr(item, "database_display_name", None),
                    "database_name": getattr(item, "database_name", None),
                    "database_type": getattr(item, "database_type", None),
                    "current_avg": getattr(item, "current", None),
                    "high_bound": getattr(item, "high_bound", None),
                    "low_bound": getattr(item, "low_bound", None),
                    "usage_change_percent": getattr(item, "usage_change_percent", None),
                })

        return json.dumps({
            "type": "database_resource_statistics",
            "compartment_id": compartment,
            "resource_metric": resource_metric,
            "percentile": percentile,
            "database_count": len(items),
            "items": items,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"OPSI Resource Stats API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _summarize_sql_insights_logic(
    compartment_id: str | None = None,
    database_id: str | None = None,
    include_subtree: bool = True,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Summarize SQL insights for databases."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        kwargs: dict[str, Any] = {
            "compartment_id": compartment,
            "compartment_id_in_subtree": include_subtree,
        }
        if database_id:
            kwargs["database_id"] = [database_id]

        response = client.summarize_sql_insights(**kwargs)

        data = response.data
        inventory = getattr(data, "inventory", None)
        thresholds = getattr(data, "thresholds", None)

        return json.dumps({
            "type": "sql_insights_summary",
            "compartment_id": compartment,
            "inventory": {
                "total_sql_count": getattr(inventory, "total_sql_count", None) if inventory else None,
                "total_databases": getattr(inventory, "total_databases", None) if inventory else None,
            } if inventory else None,
            "thresholds": {
                "degrading_sql_count": getattr(thresholds, "degrading_sql_count", None) if thresholds else None,
                "variant_sql_count": getattr(thresholds, "variant_sql_count", None) if thresholds else None,
                "inefficient_sql_count": getattr(thresholds, "inefficient_sql_count", None) if thresholds else None,
                "improving_sql_count": getattr(thresholds, "improving_sql_count", None) if thresholds else None,
            } if thresholds else None,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"SQL Insights API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _compute_sql_statistics_summary(items: list[dict], sort_by: str) -> dict:
    """Pre-compute SQL statistics summary (BlinkOps pattern: result distillation)."""
    if not items:
        return {
            "summary": "No SQL statements found matching criteria.",
            "severity": "none",
            "next_action": "No SQL performance issues detected",
        }

    # Compute totals (deterministic - not LLM)
    total_db_time = sum(i.get("database_time_in_sec") or 0 for i in items)
    total_cpu_time = sum(i.get("cpu_time_in_sec") or 0 for i in items)
    total_io_time = sum(i.get("io_time_in_sec") or 0 for i in items)
    total_executions = sum(i.get("executions_count") or 0 for i in items)

    # Identify problematic SQLs
    degrading = [i for i in items if i.get("category") == "DEGRADING"]
    inefficient = [i for i in items if i.get("category") == "INEFFICIENT"]
    high_variability = [i for i in items if i.get("variability") == "HIGH"]

    # Top offenders
    top_by_time = sorted(items, key=lambda x: -(x.get("database_time_in_sec") or 0))[:3]
    top_offenders = []
    for sql in top_by_time:
        top_offenders.append({
            "sql_id": sql.get("sql_identifier"),
            "database": sql.get("database_display_name"),
            "db_time_sec": round(sql.get("database_time_in_sec") or 0, 2),
            "executions": sql.get("executions_count"),
            "category": sql.get("category"),
        })

    severity = "critical" if degrading or inefficient else "warning" if high_variability else "low"

    return {
        "summary": f"{len(items)} SQL statements analyzed: {total_db_time:.1f}s total DB time across {total_executions} executions",
        "severity": severity,
        "totals": {
            "database_time_sec": round(total_db_time, 2),
            "cpu_time_sec": round(total_cpu_time, 2),
            "io_time_sec": round(total_io_time, 2),
            "executions": total_executions,
        },
        "problem_counts": {
            "degrading": len(degrading),
            "inefficient": len(inefficient),
            "high_variability": len(high_variability),
        },
        "top_offenders": top_offenders,
        "next_action": f"Tune {degrading[0].get('sql_identifier')} (degrading performance)" if degrading else f"Review {inefficient[0].get('sql_identifier')} (inefficient)" if inefficient else "Monitor SQL performance",
    }


async def _summarize_sql_statistics_logic(
    compartment_id: str | None = None,
    database_id: str | None = None,
    sql_identifier: str | None = None,
    sort_by: str = "databaseTimeInSec",
    category: str | None = None,
    limit: int = 10,  # Reduced from 20 to minimize context
    include_subtree: bool = True,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get SQL statistics for performance analysis."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        kwargs: dict[str, Any] = {
            "compartment_id": compartment,
            "compartment_id_in_subtree": include_subtree,
            "sort_by": sort_by,
            "sort_order": "DESC",
            "limit": limit,
        }
        if database_id:
            kwargs["database_id"] = [database_id]
        if sql_identifier:
            kwargs["sql_identifier"] = [sql_identifier]
        if category:
            kwargs["category"] = [category]

        response = client.summarize_sql_statistics(**kwargs)

        items = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                items.append({
                    "sql_identifier": getattr(item, "sql_identifier", None),
                    "sql_text": getattr(item, "sql_text", None)[:150] if getattr(item, "sql_text", None) else None,  # Reduced from 200
                    "database_display_name": getattr(item, "database_display_name", None),
                    "category": getattr(item, "category", None),
                    "executions_count": getattr(item, "executions_count", None),
                    "cpu_time_in_sec": getattr(item, "cpu_time_in_sec", None),
                    "io_time_in_sec": getattr(item, "io_time_in_sec", None),
                    "database_time_in_sec": getattr(item, "database_time_in_sec", None),
                    "inefficient_wait_time_in_sec": getattr(item, "inefficient_wait_time_in_sec", None),
                    "response_time_in_sec": getattr(item, "response_time_in_sec", None),
                    "plan_count": getattr(item, "plan_count", None),
                    "variability": getattr(item, "variability", None),
                })

        # Pre-computed summary (BlinkOps pattern)
        distilled = _compute_sql_statistics_summary(items, sort_by)

        return json.dumps({
            "type": "sql_statistics",
            "distilled_summary": distilled,  # LLM reads this first
            "compartment_id": compartment,
            "sort_by": sort_by,
            "sql_count": len(items),
            "items": items,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"SQL Statistics API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _compute_addm_findings_summary(items: list[dict]) -> dict:
    """Pre-compute ADDM findings summary with severity classification (BlinkOps pattern)."""
    if not items:
        return {
            "summary": "No ADDM findings - database performance is optimal.",
            "severity": "none",
            "critical_count": 0,
            "high_count": 0,
            "medium_count": 0,
            "next_action": "No action required",
        }

    # Classify by impact (deterministic - not LLM)
    critical = [i for i in items if (i.get("impact_overall_percent") or 0) >= 20]
    high = [i for i in items if 10 <= (i.get("impact_overall_percent") or 0) < 20]
    medium = [i for i in items if 5 <= (i.get("impact_overall_percent") or 0) < 10]
    low = [i for i in items if (i.get("impact_overall_percent") or 0) < 5]

    # Aggregate by category
    by_category = {}
    for i in items:
        cat = i.get("category_display_name") or i.get("category_name") or "unknown"
        by_category[cat] = by_category.get(cat, 0) + 1

    top_issues = []
    for i in sorted(items, key=lambda x: -(x.get("impact_overall_percent") or 0))[:3]:
        top_issues.append({
            "name": i.get("name"),
            "impact_percent": i.get("impact_overall_percent"),
            "category": i.get("category_display_name"),
        })

    severity = "critical" if critical else "high" if high else "medium" if medium else "low"
    total_impact = sum(i.get("impact_overall_percent") or 0 for i in items)

    return {
        "summary": f"{len(items)} performance findings: {len(critical)} critical, {len(high)} high, {len(medium)} medium, {len(low)} low impact",
        "severity": severity,
        "total_impact_percent": round(total_impact, 1),
        "critical_count": len(critical),
        "high_count": len(high),
        "medium_count": len(medium),
        "low_count": len(low),
        "by_category": by_category,
        "top_issues": top_issues,
        "next_action": f"Address {critical[0]['name']} ({critical[0].get('impact_overall_percent')}% impact)" if critical else "Review high-impact findings" if high else "Monitor performance",
    }


async def _summarize_addm_findings_logic(
    compartment_id: str | None = None,
    database_id: str | None = None,
    include_subtree: bool = True,
    finding_type: str | None = None,
    limit: int = 15,  # Reduced from 20 to minimize context
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get ADDM findings for performance issues."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        kwargs: dict[str, Any] = {
            "compartment_id": compartment,
            "compartment_id_in_subtree": include_subtree,
            "limit": limit,
        }
        if database_id:
            kwargs["id"] = [database_id]
        if finding_type:
            kwargs["finding_type"] = finding_type

        response = client.summarize_addm_db_findings(**kwargs)

        items = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                items.append({
                    "id": getattr(item, "id", None),
                    "finding_id": getattr(item, "finding_id", None),
                    "category_name": getattr(item, "category_name", None),
                    "category_display_name": getattr(item, "category_display_name", None),
                    "name": getattr(item, "name", None),
                    "message": getattr(item, "message", None),
                    "impact_overall_percent": getattr(item, "impact_overall_percent", None),
                    "impact_max_percent": getattr(item, "impact_max_percent", None),
                    "impact_avg_active_sessions": getattr(item, "impact_avg_active_sessions", None),
                    "frequency_count": getattr(item, "frequency_count", None),
                })

        # Pre-computed summary with severity (BlinkOps pattern)
        distilled = _compute_addm_findings_summary(items)

        return json.dumps({
            "type": "addm_findings",
            "distilled_summary": distilled,  # LLM reads this first
            "compartment_id": compartment,
            "finding_count": len(items),
            "items": items,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"ADDM Findings API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def _compute_addm_recommendations_summary(items: list[dict]) -> dict:
    """Pre-compute ADDM recommendations summary (BlinkOps pattern)."""
    if not items:
        return {
            "summary": "No ADDM recommendations available.",
            "next_action": "No performance tuning recommendations at this time",
        }

    # Calculate total potential benefit (deterministic)
    total_benefit = sum(i.get("overall_benefit_percent") or 0 for i in items)
    requires_restart = [i for i in items if i.get("require_restart")]

    # Group by type
    by_type = {}
    for i in items:
        rec_type = i.get("type") or "unknown"
        by_type[rec_type] = by_type.get(rec_type, 0) + 1

    # Top recommendations by benefit
    top_recs = sorted(items, key=lambda x: -(x.get("overall_benefit_percent") or 0))[:3]
    top_recommendations = []
    for rec in top_recs:
        top_recommendations.append({
            "type": rec.get("type"),
            "benefit_percent": rec.get("overall_benefit_percent"),
            "requires_restart": rec.get("require_restart"),
            "message": (rec.get("message") or "")[:100],
        })

    return {
        "summary": f"{len(items)} recommendations with {total_benefit:.1f}% total potential benefit",
        "total_potential_benefit_percent": round(total_benefit, 1),
        "requires_restart_count": len(requires_restart),
        "by_type": by_type,
        "top_recommendations": top_recommendations,
        "next_action": f"Apply {top_recs[0].get('type')} recommendation for {top_recs[0].get('overall_benefit_percent')}% improvement" if top_recs else "Review recommendations",
    }


async def _summarize_addm_recommendations_logic(
    compartment_id: str | None = None,
    database_id: str | None = None,
    finding_id: str | None = None,
    include_subtree: bool = True,
    limit: int = 10,  # Reduced from 20 to minimize context
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get ADDM recommendations for performance improvements."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        kwargs: dict[str, Any] = {
            "compartment_id": compartment,
            "compartment_id_in_subtree": include_subtree,
            "limit": limit,
        }
        if database_id:
            kwargs["id"] = [database_id]
        if finding_id:
            kwargs["finding_id"] = [finding_id]

        response = client.summarize_addm_db_recommendations(**kwargs)

        items = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                items.append({
                    "id": getattr(item, "id", None),
                    "recommendation_id": getattr(item, "recommendation_id", None),
                    "finding_id": getattr(item, "finding_id", None),
                    "type": getattr(item, "type", None),
                    "message": getattr(item, "message", None),
                    "require_restart": getattr(item, "require_restart", None),
                    "implement_actions": getattr(item, "implement_actions", None),
                    "rationale": (getattr(item, "rationale", None) or "")[:200],  # Truncate rationale
                    "max_benefit_percent": getattr(item, "max_benefit_percent", None),
                    "overall_benefit_percent": getattr(item, "overall_benefit_percent", None),
                    "max_benefit_avg_active_sessions": getattr(item, "max_benefit_avg_active_sessions", None),
                    "frequency_count": getattr(item, "frequency_count", None),
                })

        # Pre-computed summary (BlinkOps pattern)
        distilled = _compute_addm_recommendations_summary(items)

        return json.dumps({
            "type": "addm_recommendations",
            "distilled_summary": distilled,  # LLM reads this first
            "compartment_id": compartment,
            "recommendation_count": len(items),
            "items": items,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"ADDM Recommendations API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _summarize_capacity_trend_logic(
    compartment_id: str | None = None,
    resource_metric: str = "CPU",
    analysis_time_interval: str = "30d",
    include_subtree: bool = True,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get capacity trend analysis for planning."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        response = client.summarize_database_insight_resource_capacity_trend(
            compartment_id=compartment,
            resource_metric=resource_metric,
            analysis_time_interval=analysis_time_interval,
            compartment_id_in_subtree=include_subtree,
        )

        items = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                items.append({
                    "end_timestamp": str(getattr(item, "end_timestamp", None)),
                    "capacity": getattr(item, "capacity", None),
                    "base_capacity": getattr(item, "base_capacity", None),
                    "total_host_capacity": getattr(item, "total_host_capacity", None),
                })

        return json.dumps({
            "type": "capacity_trend",
            "compartment_id": compartment,
            "resource_metric": resource_metric,
            "analysis_interval": analysis_time_interval,
            "data_points": len(items),
            "items": items,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"Capacity Trend API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _summarize_forecast_trend_logic(
    compartment_id: str | None = None,
    resource_metric: str = "CPU",
    forecast_days: int = 30,
    include_subtree: bool = True,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """Get capacity forecast for future planning."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        response = client.summarize_database_insight_resource_forecast_trend(
            compartment_id=compartment,
            resource_metric=resource_metric,
            forecast_days=forecast_days,
            compartment_id_in_subtree=include_subtree,
        )

        data = response.data
        projected_data = []
        historical_data = []

        if hasattr(data, "projected_data") and data.projected_data:
            for item in data.projected_data:
                projected_data.append({
                    "end_timestamp": str(getattr(item, "end_timestamp", None)),
                    "high_value": getattr(item, "high_value", None),
                    "low_value": getattr(item, "low_value", None),
                    "value": getattr(item, "value", None),
                })

        if hasattr(data, "historical_data") and data.historical_data:
            for item in data.historical_data:
                historical_data.append({
                    "end_timestamp": str(getattr(item, "end_timestamp", None)),
                    "usage": getattr(item, "usage", None),
                    "capacity": getattr(item, "capacity", None),
                })

        return json.dumps({
            "type": "capacity_forecast",
            "compartment_id": compartment,
            "resource_metric": resource_metric,
            "forecast_days": forecast_days,
            "time_interval_start": str(getattr(data, "time_interval_start", None)),
            "time_interval_end": str(getattr(data, "time_interval_end", None)),
            "resource_label": getattr(data, "resource_label", None),
            "resource_capacity_unit": getattr(data, "resource_capacity_unit", None),
            "high_utilization_threshold": getattr(data, "high_utilization_threshold", None),
            "low_utilization_threshold": getattr(data, "low_utilization_threshold", None),
            "projected_data_points": len(projected_data),
            "historical_data_points": len(historical_data),
            "projected_data": projected_data[:10],  # Limit output
            "historical_data": historical_data[-10:],  # Last 10 historical points
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"Forecast Trend API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


async def _list_awr_hubs_logic(
    compartment_id: str | None = None,
    include_subtree: bool = True,
    limit: int = 20,
    profile: str | None = None,
    region: str | None = None,
) -> str:
    """List AWR Hubs in the compartment."""
    import oci

    try:
        client = get_opsi_client(profile=profile, region=region)
        config = client.base_client.config
        compartment = compartment_id or config.get("tenancy")

        response = client.list_awr_hubs(
            compartment_id=compartment,
            limit=limit,
        )

        hubs = []
        if hasattr(response.data, "items"):
            for item in response.data.items:
                hubs.append({
                    "id": getattr(item, "id", None),
                    "display_name": getattr(item, "display_name", None),
                    "operations_insights_warehouse_id": getattr(item, "operations_insights_warehouse_id", None),
                    "awr_mailbox_url": getattr(item, "awr_mailbox_url", None),
                    "lifecycle_state": getattr(item, "lifecycle_state", None),
                    "time_created": str(getattr(item, "time_created", None)),
                })

        return json.dumps({
            "type": "awr_hubs",
            "compartment_id": compartment,
            "hub_count": len(hubs),
            "hubs": hubs,
        })

    except oci.exceptions.ServiceError as e:
        return json.dumps({
            "error": f"AWR Hubs API error: {e.message}",
            "status": e.status,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


def register_opsi_tools(mcp: FastMCP) -> None:
    """Register Operations Insights tools with the MCP server."""

    @mcp.tool()
    async def oci_opsi_list_database_insights(
        compartment_id: str | None = None,
        include_subtree: bool = True,
        database_type: str | None = None,
        status: str | None = None,
        limit: int = 50,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """List database insights registered with Operations Insights (OPSI).

        OPSI provides deep performance analytics, capacity planning, and SQL insights.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            include_subtree: Include sub-compartments (default True)
            database_type: Filter by type (ADW-D, ADW-S, ATP-D, ATP-S, etc.)
            status: Filter by status (ENABLED, DISABLED, etc.)
            limit: Maximum insights to return (default 50)
            profile: OCI profile name
            region: OCI region override

        Returns:
            List of database insights with details
        """
        return await _list_database_insights_logic(
            compartment_id, include_subtree, database_type, status, limit, profile, region
        )

    @mcp.tool()
    async def oci_opsi_get_database_insight(
        database_insight_id: str,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get detailed information about a specific database insight.

        Args:
            database_insight_id: OCID of the database insight
            profile: OCI profile name
            region: OCI region override

        Returns:
            Detailed database insight information
        """
        return await _get_database_insight_logic(database_insight_id, profile, region)

    @mcp.tool()
    async def oci_opsi_summarize_resource_stats(
        compartment_id: str | None = None,
        resource_metric: str = "CPU",
        include_subtree: bool = True,
        percentile: int = 90,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Summarize database resource statistics across the fleet.

        Provides resource utilization stats for CPU, STORAGE, IO, or MEMORY.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            resource_metric: Metric to analyze (CPU, STORAGE, IO, MEMORY)
            include_subtree: Include sub-compartments (default True)
            percentile: Percentile for statistics (default 90)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Resource statistics for all databases
        """
        return await _summarize_database_resource_statistics_logic(
            compartment_id, resource_metric, include_subtree, percentile, profile, region
        )

    @mcp.tool()
    async def oci_opsi_summarize_sql_insights(
        compartment_id: str | None = None,
        database_id: str | None = None,
        include_subtree: bool = True,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Summarize SQL insights for performance analysis.

        Provides inventory of SQL statements and identifies
        degrading, variant, and inefficient queries.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            database_id: Filter by specific database
            include_subtree: Include sub-compartments (default True)
            profile: OCI profile name
            region: OCI region override

        Returns:
            SQL insights summary with problem categories
        """
        return await _summarize_sql_insights_logic(
            compartment_id, database_id, include_subtree, profile, region
        )

    @mcp.tool()
    async def oci_opsi_summarize_sql_statistics(
        compartment_id: str | None = None,
        database_id: str | None = None,
        sql_identifier: str | None = None,
        sort_by: str = "databaseTimeInSec",
        category: str | None = None,
        limit: int = 20,
        include_subtree: bool = True,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get detailed SQL statistics for performance analysis.

        Analyze SQL performance by various metrics like CPU time,
        I/O time, database time, etc.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            database_id: Filter by specific database
            sql_identifier: Filter by specific SQL identifier
            sort_by: Sort metric (databaseTimeInSec, cpuTimeInSec, ioTimeInSec, etc.)
            category: Filter by category (DEGRADING, VARIANT, INEFFICIENT, etc.)
            limit: Maximum SQLs to return (default 20)
            include_subtree: Include sub-compartments (default True)
            profile: OCI profile name
            region: OCI region override

        Returns:
            SQL statistics with performance metrics
        """
        return await _summarize_sql_statistics_logic(
            compartment_id, database_id, sql_identifier, sort_by, category, limit, include_subtree, profile, region
        )

    @mcp.tool()
    async def oci_opsi_get_addm_findings(
        compartment_id: str | None = None,
        database_id: str | None = None,
        include_subtree: bool = True,
        finding_type: str | None = None,
        limit: int = 20,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get ADDM (Automatic Database Diagnostic Monitor) findings.

        ADDM identifies performance issues and their impact.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            database_id: Filter by specific database
            include_subtree: Include sub-compartments (default True)
            finding_type: Filter by finding type
            limit: Maximum findings to return (default 20)
            profile: OCI profile name
            region: OCI region override

        Returns:
            ADDM findings with impact analysis
        """
        return await _summarize_addm_findings_logic(
            compartment_id, database_id, include_subtree, finding_type, limit, profile, region
        )

    @mcp.tool()
    async def oci_opsi_get_addm_recommendations(
        compartment_id: str | None = None,
        database_id: str | None = None,
        finding_id: str | None = None,
        include_subtree: bool = True,
        limit: int = 20,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get ADDM recommendations for performance improvements.

        ADDM provides actionable recommendations to fix issues.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            database_id: Filter by specific database
            finding_id: Filter by specific finding
            include_subtree: Include sub-compartments (default True)
            limit: Maximum recommendations to return (default 20)
            profile: OCI profile name
            region: OCI region override

        Returns:
            ADDM recommendations with benefit analysis
        """
        return await _summarize_addm_recommendations_logic(
            compartment_id, database_id, finding_id, include_subtree, limit, profile, region
        )

    @mcp.tool()
    async def oci_opsi_get_capacity_trend(
        compartment_id: str | None = None,
        resource_metric: str = "CPU",
        analysis_time_interval: str = "30d",
        include_subtree: bool = True,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get historical capacity trend for planning.

        Analyze resource utilization trends over time.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            resource_metric: Metric to analyze (CPU, STORAGE, IO, MEMORY)
            analysis_time_interval: Time interval (e.g., 30d, 7d)
            include_subtree: Include sub-compartments (default True)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Historical capacity trend data
        """
        return await _summarize_capacity_trend_logic(
            compartment_id, resource_metric, analysis_time_interval, include_subtree, profile, region
        )

    @mcp.tool()
    async def oci_opsi_get_capacity_forecast(
        compartment_id: str | None = None,
        resource_metric: str = "CPU",
        forecast_days: int = 30,
        include_subtree: bool = True,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """Get capacity forecast for future planning.

        Predict future resource needs based on historical trends.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            resource_metric: Metric to forecast (CPU, STORAGE, IO, MEMORY)
            forecast_days: Days to forecast (default 30)
            include_subtree: Include sub-compartments (default True)
            profile: OCI profile name
            region: OCI region override

        Returns:
            Capacity forecast with projections
        """
        return await _summarize_forecast_trend_logic(
            compartment_id, resource_metric, forecast_days, include_subtree, profile, region
        )

    @mcp.tool()
    async def oci_opsi_list_awr_hubs(
        compartment_id: str | None = None,
        include_subtree: bool = True,
        limit: int = 20,
        profile: str | None = None,
        region: str | None = None,
    ) -> str:
        """List AWR Hubs for centralized AWR data management.

        AWR Hub aggregates AWR data from multiple databases for analysis.

        Args:
            compartment_id: Filter by compartment (defaults to tenancy)
            include_subtree: Include sub-compartments (default True)
            limit: Maximum hubs to return (default 20)
            profile: OCI profile name
            region: OCI region override

        Returns:
            List of AWR Hubs
        """
        return await _list_awr_hubs_logic(
            compartment_id, include_subtree, limit, profile, region
        )
