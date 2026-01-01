import json
from datetime import datetime, timedelta

import oci
from opentelemetry import trace

from src.mcp.server.auth import get_usage_client

# Get tracer for cost tools
_tracer = trace.get_tracer("mcp-oci-cost")


async def _get_cost_summary_logic(
    compartment_id: str,
    days: int = 30,
) -> str:
    """Internal logic for cost summary.

    Returns structured JSON data that can be formatted by the presentation layer.
    """
    with _tracer.start_as_current_span("mcp.cost.get_summary") as span:
        span.set_attribute("compartment_id", compartment_id)
        span.set_attribute("days", days)

        client = get_usage_client()

        try:
            # OCI Usage API requires dates with hours/minutes/seconds set to 0
            end_time = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            start_time = (end_time - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

            request = oci.usage_api.models.RequestSummarizedUsagesDetails(
                tenant_id=compartment_id,
                time_usage_started=start_time,
                time_usage_ended=end_time,
                granularity="MONTHLY",
                query_type="COST",
                group_by=["service"],  # Group by service for breakdown
                is_aggregate_by_time=False
            )

            response = client.request_summarized_usages(request)
            usages = response.data.items

            if not usages:
                return json.dumps({
                    "type": "cost_summary",
                    "error": f"No cost data available for the last {days} days."
                })

            # Group by service
            service_costs: dict[str, float] = {}
            currency = usages[0].currency if usages else "USD"

            for u in usages:
                service = getattr(u, 'service', 'Other')
                amount = u.computed_amount or 0
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

            return json.dumps({
                "type": "cost_summary",
                "summary": {
                    "total": f"{total:,.2f} {currency}",
                    "period": f"{start_time.strftime('%Y-%m-%d')} → {end_time.strftime('%Y-%m-%d')}",
                    "days": days,
                },
                "services": services_data,
            })

        except Exception as e:
            span.set_attribute("error", str(e))
            span.record_exception(e)
            return json.dumps({
                "type": "cost_summary",
                "error": f"Error getting cost summary: {e}"
            })

def register_cost_tools(mcp):
    @mcp.tool()
    async def oci_cost_get_summary(compartment_id: str, days: int = 30) -> str:
        """Get summarized cost for a compartment.

        Args:
            compartment_id: OCID of the compartment (or tenancy for full account)
            days: Number of days to look back (default 30)

        Returns:
            JSON with cost summary including total spend and per-service breakdown
        """
        return await _get_cost_summary_logic(compartment_id, days)
