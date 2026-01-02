import asyncio
import json
from datetime import datetime, timedelta

import oci
from opentelemetry import trace

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
) -> str:
    """Internal logic for cost summary.

    Returns structured JSON data that can be formatted by the presentation layer.

    Args:
        compartment_id: OCID of the compartment (defaults to tenancy root)
        days: Number of days to look back
        service_filter: Optional service name filter (e.g., "Database", "Autonomous")
    """
    with _tracer.start_as_current_span("mcp.cost.get_summary") as span:
        # Get tenancy ID from config if not provided
        config = get_oci_config()
        tenant_id = compartment_id or config.get("tenancy")

        if not tenant_id:
            return json.dumps({
                "type": "cost_summary",
                "error": "No compartment_id provided and tenancy not found in OCI config"
            })

        span.set_attribute("compartment_id", tenant_id)
        span.set_attribute("days", days)

        client = get_usage_client()

        try:
            # OCI Usage API requires dates with hours/minutes/seconds set to 0
            end_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = (end_time - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

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
            except asyncio.TimeoutError:
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
    ) -> str:
        """Get summarized cost for a compartment or the entire tenancy.

        Args:
            compartment_id: OCID of the compartment (defaults to tenancy root for full account costs)
            days: Number of days to look back (default 30)
            service_filter: Filter by service type (e.g., "database", "compute", "storage", "network")

        Returns:
            JSON with cost summary including total spend and per-service breakdown.
            Note: This API has a 30s timeout. For slow responses, check the OCI console directly.
        """
        return await _get_cost_summary_logic(compartment_id, days, service_filter)
