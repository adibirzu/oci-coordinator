import asyncio
import json
from datetime import UTC, datetime, timedelta
from typing import Any

from opentelemetry import trace

import oci
from src.mcp.server.auth import get_oci_config, get_usage_client

# Get tracer for cost tools
_tracer = trace.get_tracer("mcp-oci-cost")

# Timeout for OCI Usage API (seconds) - Usage API can be slow
# Set to 60s to match server-level timeout and give the API more time
COST_API_TIMEOUT = 60

# Common service name mappings for filtering
SERVICE_FILTER_PATTERNS = {
    "database": ["database", "autonomous", "atp", "adw", "exadata", "mysql", "nosql"],
    "compute": ["compute", "instance", "virtual machine"],
    "storage": ["storage", "block", "object", "file", "archive"],
    "network": ["network", "vcn", "load balancer", "fastconnect", "vpn"],
}


def _parse_date_range(
    start_date: str | None,
    end_date: str | None,
    days: int = 30
) -> tuple[datetime, datetime, int]:
    """Parse date range from various formats.

    Returns (start_time, end_time, actual_days) tuple.
    Raises ValueError if date format is invalid.
    """
    def parse_date(date_str: str) -> datetime:
        """Parse date string in either YYYY-MM-DD or ISO format."""
        if 'T' in date_str or 'Z' in date_str:
            date_str_clean = date_str.replace('Z', '+00:00')
            dt = datetime.fromisoformat(date_str_clean)
            if dt.tzinfo is not None:
                dt = dt.astimezone(UTC)
            else:
                dt = dt.replace(tzinfo=UTC)
            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

    if start_date and end_date:
        start_time = parse_date(start_date)
        end_time = parse_date(end_date)
        end_time = (end_time + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
        )
        actual_days = (end_time - start_time).days
    else:
        end_time = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        start_time = (end_time - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )
        actual_days = days

    # Clamp end_time to today
    now_utc = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    if end_time > now_utc:
        end_time = now_utc
        actual_days = max((end_time - start_time).days, 0)

    return start_time, end_time, actual_days


async def _call_usage_api_with_timeout(client: Any, request: Any) -> Any:
    """Call OCI Usage API with timeout handling."""
    def _call():
        return client.request_summarized_usages(request)

    return await asyncio.wait_for(
        asyncio.get_event_loop().run_in_executor(None, _call),
        timeout=COST_API_TIMEOUT
    )


async def _get_cost_summary_logic(
    compartment_id: str | None = None,
    days: int = 30,
    service_filter: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    profile: str | None = None,
) -> str:
    """Internal logic for cost summary.

    Returns structured JSON data that can be formatted by the presentation layer.

    Args:
        compartment_id: OCID of the compartment (defaults to tenancy root)
        days: Number of days to look back (ignored if start_date/end_date provided)
        service_filter: Optional service name filter (e.g., "Database", "Autonomous")
        start_date: Optional start date in YYYY-MM-DD format (for historical queries)
        end_date: Optional end date in YYYY-MM-DD format (for historical queries)
    """
    with _tracer.start_as_current_span("mcp.cost.get_summary") as span:
        # Get tenancy ID from config if not provided
        config = get_oci_config(profile)
        tenant_id = compartment_id or config.get("tenancy")

        if not tenant_id:
            return json.dumps({
                "type": "cost_summary",
                "error": "No compartment_id provided and tenancy not found in OCI config"
            })

        span.set_attribute("compartment_id", tenant_id)
        span.set_attribute("days", days)

        client = get_usage_client(profile)

        try:
            # OCI Usage API requires dates with hours/minutes/seconds set to 0 in UTC timezone
            # If explicit dates provided, use them; otherwise calculate from days
            if start_date and end_date:
                try:
                    # Handle both YYYY-MM-DD and ISO datetime formats (YYYY-MM-DDTHH:MM:SSZ)
                    def parse_date(date_str: str) -> datetime:
                        """Parse date string in either YYYY-MM-DD or ISO format, returning UTC datetime at midnight."""
                        # Try ISO format first (with time)
                        if 'T' in date_str or 'Z' in date_str:
                            # Remove Z and parse ISO format
                            date_str_clean = date_str.replace('Z', '+00:00')
                            dt = datetime.fromisoformat(date_str_clean)
                            # Convert to UTC if timezone-aware, otherwise assume UTC
                            if dt.tzinfo is not None:
                                dt = dt.astimezone(UTC)
                            else:
                                dt = dt.replace(tzinfo=UTC)
                            # Set to midnight UTC with all fractions zero
                            return dt.replace(hour=0, minute=0, second=0, microsecond=0)
                        else:
                            # Simple YYYY-MM-DD format - create as UTC midnight
                            dt = datetime.strptime(date_str, "%Y-%m-%d")
                            return dt.replace(hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC)

                    start_time = parse_date(start_date)
                    end_time = parse_date(end_date)
                    # For end_date, we want the start of the next day (exclusive end)
                    # OCI Usage API expects exclusive end, so add 1 day and set to midnight
                    # Ensure timezone is preserved when adding days
                    end_time = (end_time + timedelta(days=1)).replace(
                        hour=0, minute=0, second=0, microsecond=0, tzinfo=UTC
                    )
                    # Recalculate days for display
                    days = (end_time - start_time).days
                    span.set_attribute("start_date", start_date)
                    span.set_attribute("end_date", end_date)
                except (ValueError, AttributeError) as e:
                    return json.dumps({
                        "type": "cost_summary",
                        "error": f"Invalid date format. Use YYYY-MM-DD or ISO format (YYYY-MM-DDTHH:MM:SSZ). Error: {e}"
                    })
            else:
                # Use UTC timezone-aware datetime
                end_time = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
                start_time = (end_time - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

            # Clamp end_time to today if caller requested a future end date.
            # OCI Usage API does not return future usage; clamp to prevent empty results.
            now_utc = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
            if end_time > now_utc:
                end_time = now_utc
                days = max((end_time - start_time).days, 0)

            if end_time <= start_time:
                return json.dumps({
                    "type": "cost_summary",
                    "error": "Requested time range has no completed usage yet. Try an earlier end date.",
                })

            # Ensure datetimes are properly formatted for OCI API
            # OCI requires: UTC timezone, hours/minutes/seconds/microseconds = 0
            if start_time.tzinfo is None:
                start_time = start_time.replace(tzinfo=UTC)
            else:
                start_time = start_time.astimezone(UTC)
            start_time = start_time.replace(hour=0, minute=0, second=0, microsecond=0)

            if end_time.tzinfo is None:
                end_time = end_time.replace(tzinfo=UTC)
            else:
                end_time = end_time.astimezone(UTC)
            end_time = end_time.replace(hour=0, minute=0, second=0, microsecond=0)

            request = oci.usage_api.models.RequestSummarizedUsagesDetails(
                tenant_id=tenant_id,
                time_usage_started=start_time,
                time_usage_ended=end_time,
                granularity="MONTHLY",
                query_type="COST",
                group_by=["service"],  # Group by service for breakdown
                is_aggregate_by_time=False
            )

            # Run synchronous OCI API call with timeout
            def _call_usage_api():
                return client.request_summarized_usages(request)

            try:
                response = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, _call_usage_api),
                    timeout=COST_API_TIMEOUT
                )
            except TimeoutError:
                span.set_attribute("error", "timeout")
                return json.dumps({
                    "type": "cost_summary",
                    "error": f"Cost API timed out after {COST_API_TIMEOUT}s. The OCI Usage API may be slow. Try again or check the OCI console."
                })

            usages = response.data.items

            if not usages:
                return json.dumps({
                    "type": "cost_summary",
                    "error": f"No cost data available for the last {days} days."
                })

            # Group by service
            service_costs: dict[str, float] = {}
            currency = usages[0].currency if usages else "USD"

            # Common service name mappings for filtering
            SERVICE_FILTER_PATTERNS = {
                "database": ["database", "autonomous", "atp", "adw", "exadata", "mysql", "nosql"],
                "compute": ["compute", "instance", "virtual machine"],
                "storage": ["storage", "block", "object", "file", "archive"],
                "network": ["network", "vcn", "load balancer", "fastconnect", "vpn"],
            }

            for u in usages:
                service = getattr(u, 'service', 'Other')
                amount = u.computed_amount or 0

                # Apply service filter if provided
                if service_filter:
                    filter_lower = service_filter.lower()
                    service_lower = service.lower()

                    # Check direct match or pattern match
                    matched = filter_lower in service_lower

                    # Check category patterns
                    if not matched and filter_lower in SERVICE_FILTER_PATTERNS:
                        patterns = SERVICE_FILTER_PATTERNS[filter_lower]
                        matched = any(p in service_lower for p in patterns)

                    if not matched:
                        continue  # Skip non-matching services

                service_costs[service] = service_costs.get(service, 0) + amount

            # Sort by cost descending and calculate percentages
            sorted_services = sorted(service_costs.items(), key=lambda x: x[1], reverse=True)
            total = sum(service_costs.values())

            # Build structured response with table-ready data
            services_data = []
            for service, cost in sorted_services[:15]:  # Top 15 services
                if cost > 0.01:  # Only show services with non-trivial cost
                    pct = (cost / total * 100) if total > 0 else 0
                    services_data.append({
                        "service": service,
                        "cost": f"{cost:,.2f} {currency}",
                        "percent": f"{pct:.1f}%",
                    })

            span.set_attribute("total_cost", total)
            span.set_attribute("services_count", len(services_data))
            span.set_attribute("service_filter", service_filter or "none")

            # Build response with filter info if applied
            response_data = {
                "type": "cost_summary",
                "summary": {
                    "total": f"{total:,.2f} {currency}",
                    "period": f"{start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}",
                    "days": days,
                },
                "services": services_data,
            }

            if service_filter:
                response_data["filter"] = service_filter
                response_data["summary"]["filter_applied"] = service_filter

            # Handle case where filter matched nothing
            if service_filter and not services_data:
                response_data["message"] = f"No costs found for services matching '{service_filter}'"

            return json.dumps(response_data)

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({
                "type": "cost_summary",
                "error": f"Error getting cost summary: {e}"
            })

def register_cost_tools(mcp):
    @mcp.tool()
    async def oci_cost_get_summary(
        compartment_id: str | None = None,
        days: int = 30,
        service_filter: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        profile: str | None = None,
        format: str | None = None,  # Accepted for compatibility, but always returns JSON
    ) -> str:
        """Get summarized cost for a compartment or the entire tenancy.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root for full account costs)
            days: Number of days to look back (default 30, ignored if start_date/end_date provided)
            service_filter: Filter by service type (e.g., "database", "compute", "storage", "network")
            start_date: Start date in YYYY-MM-DD format (for historical queries like "November")
            end_date: End date in YYYY-MM-DD format (for historical queries like "November")
            profile: OCI config profile name (defaults to OCI_CLI_PROFILE)
            format: Output format (accepted for compatibility, but always returns JSON)

        Returns:
            JSON with cost summary including total spend and per-service breakdown.
            Note: This API has a 60s timeout. For slow responses, check the OCI console directly.
        """
        return await _get_cost_summary_logic(
            compartment_id, days, service_filter, start_date, end_date, profile
        )

    @mcp.tool()
    async def oci_cost_by_compartment(
        days: int = 30,
        start_date: str | None = None,
        end_date: str | None = None,
        profile: str | None = None,
    ) -> str:
        """Get cost breakdown by compartment.

        Use this to see which compartments are consuming the most resources.

        Args:
            days: Number of days to look back (default 30)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            profile: OCI config profile name

        Returns:
            JSON with cost breakdown by compartment.
        """
        with _tracer.start_as_current_span("mcp.cost.by_compartment") as span:
            config = get_oci_config(profile)
            tenant_id = config.get("tenancy")

            if not tenant_id:
                return json.dumps({
                    "type": "cost_by_compartment",
                    "error": "Tenancy not found in OCI config"
                })

            try:
                start_time, end_time, actual_days = _parse_date_range(
                    start_date, end_date, days
                )

                if end_time <= start_time:
                    return json.dumps({
                        "type": "cost_by_compartment",
                        "error": "Invalid date range"
                    })

                client = get_usage_client(profile)
                request = oci.usage_api.models.RequestSummarizedUsagesDetails(
                    tenant_id=tenant_id,
                    time_usage_started=start_time,
                    time_usage_ended=end_time,
                    granularity="MONTHLY",
                    query_type="COST",
                    group_by=["compartmentId", "compartmentName"],
                    is_aggregate_by_time=False
                )

                response = await _call_usage_api_with_timeout(client, request)
                usages = response.data.items

                if not usages:
                    return json.dumps({
                        "type": "cost_by_compartment",
                        "error": f"No cost data available for the last {actual_days} days."
                    })

                # Aggregate by compartment
                compartment_costs: dict[str, dict] = {}
                currency = usages[0].currency if usages else "USD"

                for u in usages:
                    comp_id = getattr(u, 'compartment_id', 'Unknown')
                    comp_name = getattr(u, 'compartment_name', 'Unknown')
                    amount = u.computed_amount or 0

                    if comp_id not in compartment_costs:
                        compartment_costs[comp_id] = {"name": comp_name, "cost": 0}
                    compartment_costs[comp_id]["cost"] += amount

                # Sort and calculate percentages
                sorted_comps = sorted(
                    compartment_costs.items(),
                    key=lambda x: x[1]["cost"],
                    reverse=True
                )
                total = sum(c["cost"] for c in compartment_costs.values())

                compartments_data = []
                for comp_id, data in sorted_comps[:20]:
                    cost = data["cost"]
                    if cost > 0.01:
                        pct = (cost / total * 100) if total > 0 else 0
                        compartments_data.append({
                            "compartment": data["name"],
                            "cost": f"{cost:,.2f} {currency}",
                            "percent": f"{pct:.1f}%",
                        })

                span.set_attribute("total_cost", total)
                span.set_attribute("compartment_count", len(compartments_data))

                return json.dumps({
                    "type": "cost_by_compartment",
                    "summary": {
                        "total": f"{total:,.2f} {currency}",
                        "period": f"{start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}",
                        "days": actual_days,
                    },
                    "compartments": compartments_data,
                })

            except TimeoutError:
                span.set_attribute("error", "timeout")
                return json.dumps({
                    "type": "cost_by_compartment",
                    "error": f"Cost API timed out after {COST_API_TIMEOUT}s."
                })
            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                return json.dumps({
                    "type": "cost_by_compartment",
                    "error": f"Error getting compartment costs: {e}"
                })

    @mcp.tool()
    async def oci_cost_service_drilldown(
        days: int = 30,
        start_date: str | None = None,
        end_date: str | None = None,
        profile: str | None = None,
    ) -> str:
        """Get detailed cost breakdown by service with resource-level detail.

        Use this for deeper service analysis than the basic cost summary.

        Args:
            days: Number of days to look back (default 30)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            profile: OCI config profile name

        Returns:
            JSON with detailed service cost breakdown.
        """
        with _tracer.start_as_current_span("mcp.cost.service_drilldown") as span:
            config = get_oci_config(profile)
            tenant_id = config.get("tenancy")

            if not tenant_id:
                return json.dumps({
                    "type": "cost_service_drilldown",
                    "error": "Tenancy not found in OCI config"
                })

            try:
                start_time, end_time, actual_days = _parse_date_range(
                    start_date, end_date, days
                )

                if end_time <= start_time:
                    return json.dumps({
                        "type": "cost_service_drilldown",
                        "error": "Invalid date range"
                    })

                client = get_usage_client(profile)
                request = oci.usage_api.models.RequestSummarizedUsagesDetails(
                    tenant_id=tenant_id,
                    time_usage_started=start_time,
                    time_usage_ended=end_time,
                    granularity="MONTHLY",
                    query_type="COST",
                    group_by=["service", "skuName"],
                    is_aggregate_by_time=False
                )

                response = await _call_usage_api_with_timeout(client, request)
                usages = response.data.items

                if not usages:
                    return json.dumps({
                        "type": "cost_service_drilldown",
                        "error": f"No cost data available for the last {actual_days} days."
                    })

                # Aggregate by service with SKU breakdown
                service_data: dict[str, dict] = {}
                currency = usages[0].currency if usages else "USD"

                for u in usages:
                    service = getattr(u, 'service', 'Other')
                    sku = getattr(u, 'sku_name', 'Unknown')
                    amount = u.computed_amount or 0

                    if service not in service_data:
                        service_data[service] = {"total": 0, "skus": {}}
                    service_data[service]["total"] += amount
                    service_data[service]["skus"][sku] = (
                        service_data[service]["skus"].get(sku, 0) + amount
                    )

                # Sort services by total cost
                sorted_services = sorted(
                    service_data.items(),
                    key=lambda x: x[1]["total"],
                    reverse=True
                )
                total = sum(s["total"] for s in service_data.values())

                services_output = []
                for service, data in sorted_services[:10]:
                    if data["total"] > 0.01:
                        pct = (data["total"] / total * 100) if total > 0 else 0
                        top_skus = sorted(
                            data["skus"].items(),
                            key=lambda x: x[1],
                            reverse=True
                        )[:5]
                        services_output.append({
                            "service": service,
                            "cost": f"{data['total']:,.2f} {currency}",
                            "percent": f"{pct:.1f}%",
                            "top_resources": [
                                {"name": sku, "cost": f"{cost:,.2f} {currency}"}
                                for sku, cost in top_skus if cost > 0.01
                            ],
                        })

                span.set_attribute("total_cost", total)
                span.set_attribute("service_count", len(services_output))

                return json.dumps({
                    "type": "cost_service_drilldown",
                    "summary": {
                        "total": f"{total:,.2f} {currency}",
                        "period": f"{start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}",
                        "days": actual_days,
                    },
                    "services": services_output,
                })

            except TimeoutError:
                span.set_attribute("error", "timeout")
                return json.dumps({
                    "type": "cost_service_drilldown",
                    "error": f"Cost API timed out after {COST_API_TIMEOUT}s."
                })
            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                return json.dumps({
                    "type": "cost_service_drilldown",
                    "error": f"Error getting service drilldown: {e}"
                })

    @mcp.tool()
    async def oci_cost_database_drilldown(
        days: int = 30,
        start_date: str | None = None,
        end_date: str | None = None,
        profile: str | None = None,
    ) -> str:
        """Get detailed cost breakdown for database services only.

        Shows costs for Autonomous Database, Database Cloud Service, MySQL,
        NoSQL, and other database-related services.

        Args:
            days: Number of days to look back (default 30)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            profile: OCI config profile name

        Returns:
            JSON with database service cost breakdown.
        """
        # Use the existing summary logic with database filter
        return await _get_cost_summary_logic(
            compartment_id=None,
            days=days,
            service_filter="database",
            start_date=start_date,
            end_date=end_date,
            profile=profile,
        )

    @mcp.tool()
    async def oci_cost_monthly_trend(
        months: int = 6,
        profile: str | None = None,
    ) -> str:
        """Get month-over-month cost trend.

        Shows how costs have changed over the specified number of months.

        Args:
            months: Number of months to look back (default 6)
            profile: OCI config profile name

        Returns:
            JSON with monthly cost trend data.
        """
        with _tracer.start_as_current_span("mcp.cost.monthly_trend") as span:
            config = get_oci_config(profile)
            tenant_id = config.get("tenancy")

            if not tenant_id:
                return json.dumps({
                    "type": "cost_monthly_trend",
                    "error": "Tenancy not found in OCI config"
                })

            try:
                # Calculate date range for N months
                now = datetime.now(UTC)
                end_time = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                start_time = end_time
                for _ in range(months):
                    start_time = (start_time - timedelta(days=1)).replace(day=1)

                client = get_usage_client(profile)
                request = oci.usage_api.models.RequestSummarizedUsagesDetails(
                    tenant_id=tenant_id,
                    time_usage_started=start_time,
                    time_usage_ended=end_time,
                    granularity="MONTHLY",
                    query_type="COST",
                    group_by=["service"],
                    is_aggregate_by_time=True  # Get monthly breakdown
                )

                response = await _call_usage_api_with_timeout(client, request)
                usages = response.data.items

                if not usages:
                    return json.dumps({
                        "type": "cost_monthly_trend",
                        "error": f"No cost data available for the last {months} months."
                    })

                # Aggregate by month
                monthly_costs: dict[str, float] = {}
                currency = usages[0].currency if usages else "USD"

                for u in usages:
                    if u.time_usage_started:
                        month_key = u.time_usage_started.strftime("%Y-%m")
                        amount = u.computed_amount or 0
                        monthly_costs[month_key] = monthly_costs.get(month_key, 0) + amount

                # Sort by month
                sorted_months = sorted(monthly_costs.items())
                total = sum(monthly_costs.values())
                avg = total / len(monthly_costs) if monthly_costs else 0

                # Calculate month-over-month changes
                trend_data = []
                prev_cost = None
                for month, cost in sorted_months:
                    entry = {
                        "month": month,
                        "cost": f"{cost:,.2f} {currency}",
                    }
                    if prev_cost is not None and prev_cost > 0:
                        change = ((cost - prev_cost) / prev_cost) * 100
                        entry["change"] = f"{change:+.1f}%"
                    prev_cost = cost
                    trend_data.append(entry)

                span.set_attribute("total_cost", total)
                span.set_attribute("months", len(trend_data))

                return json.dumps({
                    "type": "cost_monthly_trend",
                    "summary": {
                        "total": f"{total:,.2f} {currency}",
                        "average": f"{avg:,.2f} {currency}/month",
                        "period": f"{start_time.strftime('%Y-%m')} → {end_time.strftime('%Y-%m')}",
                        "months": months,
                    },
                    "trend": trend_data,
                })

            except TimeoutError:
                span.set_attribute("error", "timeout")
                return json.dumps({
                    "type": "cost_monthly_trend",
                    "error": f"Cost API timed out after {COST_API_TIMEOUT}s."
                })
            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                return json.dumps({
                    "type": "cost_monthly_trend",
                    "error": f"Error getting monthly trend: {e}"
                })

    @mcp.tool()
    async def oci_cost_usage_comparison(
        months: str,
        profile: str | None = None,
    ) -> str:
        """Compare costs between specific months.

        Usage: provide month names like "August vs October vs November"
        or "October, November" to compare.

        Args:
            months: Comma or 'vs' separated month names (e.g., "August vs October")
            profile: OCI config profile name

        Returns:
            JSON with comparison data between the specified months.
        """
        with _tracer.start_as_current_span("mcp.cost.comparison") as span:
            config = get_oci_config(profile)
            tenant_id = config.get("tenancy")

            if not tenant_id:
                return json.dumps({
                    "type": "cost_comparison",
                    "error": "Tenancy not found in OCI config"
                })

            try:
                # Parse month names
                month_names = {
                    "january": 1, "jan": 1,
                    "february": 2, "feb": 2,
                    "march": 3, "mar": 3,
                    "april": 4, "apr": 4,
                    "may": 5,
                    "june": 6, "jun": 6,
                    "july": 7, "jul": 7,
                    "august": 8, "aug": 8,
                    "september": 9, "sep": 9, "sept": 9,
                    "october": 10, "oct": 10,
                    "november": 11, "nov": 11,
                    "december": 12, "dec": 12,
                }

                # Parse input months
                input_months = [
                    m.strip().lower()
                    for m in months.replace(" vs ", ",").replace(" and ", ",").split(",")
                ]

                parsed_months = []
                current_year = datetime.now(UTC).year
                for m in input_months:
                    if m in month_names:
                        month_num = month_names[m]
                        # Assume current year, but previous year if month is in the future
                        year = current_year
                        if month_num > datetime.now(UTC).month:
                            year -= 1
                        parsed_months.append((year, month_num, m.capitalize()))

                if len(parsed_months) < 2:
                    return json.dumps({
                        "type": "cost_comparison",
                        "error": f"Need at least 2 months to compare. Got: {months}"
                    })

                # Get cost data for each month
                client = get_usage_client(profile)
                comparison_data = []
                currency = "USD"

                for year, month, name in parsed_months:
                    start_time = datetime(year, month, 1, tzinfo=UTC)
                    if month == 12:
                        end_time = datetime(year + 1, 1, 1, tzinfo=UTC)
                    else:
                        end_time = datetime(year, month + 1, 1, tzinfo=UTC)

                    request = oci.usage_api.models.RequestSummarizedUsagesDetails(
                        tenant_id=tenant_id,
                        time_usage_started=start_time,
                        time_usage_ended=end_time,
                        granularity="MONTHLY",
                        query_type="COST",
                        group_by=["service"],
                        is_aggregate_by_time=False
                    )

                    response = await _call_usage_api_with_timeout(client, request)
                    usages = response.data.items

                    total = sum(u.computed_amount or 0 for u in usages)
                    if usages:
                        currency = usages[0].currency

                    comparison_data.append({
                        "month": f"{name} {year}",
                        "cost": total,
                        "formatted": f"{total:,.2f} {currency}",
                    })

                # Calculate changes
                for i, data in enumerate(comparison_data):
                    if i > 0:
                        prev = comparison_data[i - 1]["cost"]
                        if prev > 0:
                            change = ((data["cost"] - prev) / prev) * 100
                            data["change_from_prev"] = f"{change:+.1f}%"

                # Find min/max
                costs = [d["cost"] for d in comparison_data]
                total_all = sum(costs)

                span.set_attribute("months_compared", len(comparison_data))
                span.set_attribute("total_cost", total_all)

                return json.dumps({
                    "type": "cost_comparison",
                    "summary": {
                        "months_compared": len(comparison_data),
                        "total": f"{total_all:,.2f} {currency}",
                        "highest": comparison_data[costs.index(max(costs))]["month"],
                        "lowest": comparison_data[costs.index(min(costs))]["month"],
                    },
                    "comparison": comparison_data,
                })

            except TimeoutError:
                span.set_attribute("error", "timeout")
                return json.dumps({
                    "type": "cost_comparison",
                    "error": f"Cost API timed out after {COST_API_TIMEOUT}s."
                })
            except Exception as e:
                span.set_attribute("error", str(e))
                span.record_exception(e)
                return json.dumps({
                    "type": "cost_comparison",
                    "error": f"Error comparing costs: {e}"
                })
