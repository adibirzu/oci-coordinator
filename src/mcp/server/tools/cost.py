import asyncio
import json
from datetime import UTC, datetime, timedelta

from opentelemetry import trace

import oci
from src.mcp.server.auth import get_oci_config, get_usage_client

# Get tracer for cost tools
_tracer = trace.get_tracer("mcp-oci-cost")

# Timeout for OCI Usage API (seconds) - Usage API can be slow
# Set to 60s to match server-level timeout and give the API more time
COST_API_TIMEOUT = 60


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
